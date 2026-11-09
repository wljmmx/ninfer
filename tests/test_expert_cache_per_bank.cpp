// Per-bank slot pricing: the layout the 35B-A3B MTP artifact needs.
//
// The pool used to charge every bank the widest bank's expert image, so a 41-bank artifact whose last
// bank is 1.73x wider held under 55% of its experts (measured: 5,740 of 10,496, prefill 76 tok/s
// against 5,315 after the fix). This test builds that shape synthetically - 40 narrow banks plus one
// wide one - and asserts what the fix promises:
//
//   * a narrow bank is billed its own width, so the wide bank no longer prices it out;
//   * an address is `slot_offset(id)`, and `slot * slot_stride(bank) + slot_bias(bank)` reproduces it,
//     which is the arithmetic the Device-side placement table performs;
//   * two slots never share a byte (a uniform multiply would place the wide bank over its neighbours);
//   * a grow hands the wide bank its spare id and moves nothing that was already published, and a
//     shrink drops exactly the ids above the edge;
//   * the spare run is not on the free list before it is mapped (putting it there was the crash this
//     test exists to catch).
//
// Real Device, no artifact: the banks are tiny BF16 matrices.
#include "core/expert_cache.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <span>
#include <vector>

namespace {

using namespace ninfer;

int fail(const char* message) {
    std::fprintf(stderr, "per-bank slots: %s\n", message);
    return 1;
}

constexpr std::uint32_t kNarrowBanks = 40;
constexpr std::uint32_t kWideBank    = kNarrowBanks;
constexpr std::uint32_t kExperts     = 2;

WeightGeometry rows(std::uint64_t count) {
    // 256 columns of BF16 is 512 B/row; the wide bank carries three times the rows.
    const std::uint64_t shape[] = {count, 256};
    return weight_geometry(QType::BF16, QuantLayout::Contiguous, shape);
}

} // namespace

int main() {
    DeviceContext device(0);

    const auto narrow = rows(512);   // 256 KiB a matrix, 512 KiB an expert
    const auto wide   = rows(1536);  // 1.5 MiB an expert, three times the narrow one

    std::vector<ExpertBankSpec> specs;
    specs.reserve(kNarrowBanks + 1);
    for (std::uint32_t bank = 0; bank < kNarrowBanks; ++bank) {
        specs.push_back(ExpertBankSpec{{ExpertMatrixSpec{narrow}}, {ExpertMatrixSpec{narrow}},
                                       kExperts, false});
    }
    specs.push_back(ExpertBankSpec{{ExpertMatrixSpec{wide}}, {ExpertMatrixSpec{wide}}, kExperts,
                                   false});

    // 96 MiB over 41 banks is a 2.34 MiB share: four narrow experts or one wide one. The ceiling is
    // twice that, so a grow can double the wide bank's run.
    ExpertCacheConfig config;
    config.capacity_bytes       = 96u << 20;
    config.capacity_limit_bytes = 192u << 20;
    const auto requirements =
        describe_expert_cache(std::span<const ExpertBankSpec>(specs), config);
    if (requirements.banks.size() != specs.size()) { return fail("the plan lost a bank"); }
    if (requirements.banks[kWideBank].entry_bytes <= requirements.banks[0].entry_bytes) {
        return fail("the fixture's wide bank is not wider");
    }

    ExpertCache cache(std::span<const ExpertBankSpec>(specs), requirements, device,
                      std::span<const ExpertBankPair>());

    const auto narrow_capacity = cache.resident_capacity(0);
    const auto wide_capacity   = cache.resident_capacity(kWideBank);
    if (narrow_capacity != kExperts) { return fail("a narrow bank did not get its whole set"); }
    if (wide_capacity != 1) { return fail("the wide bank did not get exactly one slot"); }
    if (cache.resident_capacity() != wide_capacity) {
        return fail("the no-argument capacity is not the conservative (narrowest) one");
    }
    if (cache.resident_experts() != kNarrowBanks * kExperts + 1) {
        return fail("the resident count does not describe the per-bank layout");
    }

    // Address arithmetic and disjointness.
    const auto pool = cache.pool();
    const auto check_addresses = [&]() -> int {
        std::vector<std::uint64_t> seen;
        for (std::uint32_t bank = 0; bank < specs.size(); ++bank) {
            const auto stride = cache.slot_stride(bank);
            const auto bias   = cache.slot_bias(bank);
            for (std::uint32_t expert = 0; expert < kExperts; ++expert) {
                const auto slot = cache.expert_slot(bank, expert);
                if (!slot) { continue; }
                const auto offset = cache.slot_offset(*slot);
                if (offset + stride > pool.bytes) { return fail("a slot starts outside the pool"); }
                if (offset != static_cast<std::uint64_t>(*slot) * stride + bias) {
                    return fail("stride*bias does not reproduce slot_offset");
                }
                for (const auto other : seen) {
                    if (other == offset) { return fail("two slots share a byte offset"); }
                }
                seen.push_back(offset);
            }
        }
        return seen.empty() ? fail("no expert has a slot") : 0;
    };
    if (const auto code = check_addresses(); code != 0) { return code; }

    const auto narrow_stride = cache.slot_stride(0);
    const auto wide_stride   = cache.slot_stride(kWideBank);
    if (wide_stride <= narrow_stride) { return fail("the wide bank is not billed its own stride"); }

    std::uint64_t pinned_offset = 0;
    if (const auto slot = cache.expert_slot(0, 0)) { pinned_offset = cache.slot_offset(*slot); }

    // Grow: the wide bank's second expert needs the spare run, which must only become addressable
    // once the pool has actually mapped it.
    const auto before_grow = cache.capacity_bytes();
    const auto grew = cache.resize(config.capacity_limit_bytes);
    if (grew <= before_grow) { return fail("the grow added no bytes"); }
    if (cache.expert_slot(kWideBank, 1).has_value()) {
        return fail("the wide bank placed its second expert before the growth");
    }
    if (cache.resident_capacity(kWideBank) != kExperts) {
        return fail("the grown pool did not hand the wide bank its spare slot");
    }
    // The placement is the caller's job; the pool only has to be able to hold it.
    if (const auto slot = cache.expert_slot(0, 0)) {
        if (cache.slot_offset(*slot) != pinned_offset) { return fail("a grow moved a live slot"); }
    }
    if (const auto code = check_addresses(); code != 0) { return code; }

    // Shrink below the wide bank's run: its ids sit at the top of the layout, so a target of 30 MiB
    // drops them while the first narrow banks keep theirs, and nothing below the edge moves.
    constexpr std::uint64_t kShrinkBytes = 30u << 20;
    const auto shrunk = cache.resize(kShrinkBytes);
    if (shrunk >= grew) { return fail("the shrink did not reduce the pool"); }
    if (cache.resident_capacity(kWideBank) != 0) {
        return fail("the shrink did not drop the wide bank's run");
    }
    if (!cache.expert_slot(0, 0)) { return fail("the shrink dropped a slot below its edge"); }
    if (const auto slot = cache.expert_slot(0, 0)) {
        if (cache.slot_offset(*slot) != pinned_offset) { return fail("a shrink moved a live slot"); }
    }
    if (const auto code = check_addresses(); code != 0) { return code; }

    // And back up again: the released run is reusable, not leaked or double handed out.
    (void)cache.resize(config.capacity_limit_bytes);
    if (cache.resident_capacity(kWideBank) != kExperts) {
        return fail("a grow after a shrink did not restore the wide bank's run");
    }
    if (const auto code = check_addresses(); code != 0) { return code; }

    std::printf("per-bank slots: PASS (%u narrow banks at %.0f KiB, wide bank at %.0f KiB, "
                "resident %llu of %u)\n",
                kNarrowBanks, static_cast<double>(narrow_stride) / 1024.0,
                static_cast<double>(wide_stride) / 1024.0,
                static_cast<unsigned long long>(cache.resident_experts()),
                kNarrowBanks * kExperts + kExperts);
    return 0;
}
