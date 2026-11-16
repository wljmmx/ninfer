// S2-4: the id->slot kernel against a host linear lookup, bit for bit.
//
// The kernel is what makes the staged expert->slot table useful on the Device, so its contract is
// exact: every id resolves to the word the table holds at that index, and every route the table
// cannot answer becomes the pool's own "no slot" marker (kAbsentSlot) and is counted. The oracle here
// is a plain host scan of the same table - no launcher, no second implementation of the rule.
#include "ninfer/ops/moe.h"

#include "core/expert_cache.h" // kAbsentSlot: the marker a table written by export_slots() carries
#include "ops/op_tester.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// Test-local fixture numbers: this is a synthetic table, not a model geometry, so nothing here is a
// size the engine would read.
constexpr int   kExperts   = 137;
constexpr int   kIds       = 4096;
constexpr unsigned kSeed   = 20261116u;

// The table export_slots() would produce: a slot for most experts (reused, so two experts can share
// one, which is a legal pool state), and kAbsentSlot for a third of them.
std::vector<std::uint32_t> make_table() {
    std::vector<std::uint32_t> table(kExperts, kAbsentSlot);
    std::mt19937 gen(kSeed);
    std::uniform_int_distribution<std::uint32_t> slot(0, static_cast<std::uint32_t>(kExperts / 2));
    for (int expert = 0; expert < kExperts; ++expert) {
        if (expert % 3 == 1) { continue; } // stays kAbsentSlot: the pool holds no slot for it
        table[expert] = slot(gen);
    }
    return table;
}

// A quarter of the ids are outside the bank (negative and past the end), the rest are uniform over the
// table, so both the "no slot" and the "out of range" branches are exercised, not just one.
std::vector<std::int32_t> make_ids() {
    std::vector<std::int32_t> ids(kIds);
    std::mt19937 gen(kSeed + 1u);
    std::uniform_int_distribution<int> pick(0, kIds - 1);
    for (int i = 0; i < kIds; ++i) {
        switch (pick(gen) % 4) {
            case 0: ids[i] = kExperts + 1 + pick(gen) % 1000; break;  // past the end
            case 1: ids[i] = -1 - pick(gen) % 1000; break;            // before the table
            default: ids[i] = pick(gen) % kExperts; break;            // inside the bank
        }
    }
    return ids;
}

// The rule as a host scan: straight from the kernel's contract, not from its code.
std::vector<std::uint32_t> oracle(const std::vector<std::int32_t>& ids,
                                  const std::vector<std::uint32_t>& table, std::uint32_t& misses) {
    std::vector<std::uint32_t> slots(ids.size(), kAbsentSlot);
    misses = 0;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const auto id = ids[i];
        if (id < 0 || static_cast<std::uint32_t>(id) >= table.size()) {
            ++misses;
            continue;
        }
        const auto slot = table[static_cast<std::size_t>(id)];
        slots[i]        = slot;
        if (slot == kAbsentSlot) { ++misses; }
    }
    return slots;
}

int lookup_against_oracle(cudaStream_t stream) {
    const auto table = make_table();
    const auto ids   = make_ids();

    std::uint32_t expected_misses = 0;
    const auto    expected        = oracle(ids, table, expected_misses);
    // A fixture that resolved everything (or nothing) would let a no-op kernel pass, so assert both
    // branches are actually in the sample before trusting a match.
    if (expected_misses == 0 ||
        static_cast<std::size_t>(std::count(expected.begin(), expected.end(), kAbsentSlot)) ==
            expected.size()) {
        std::cerr << "fixture does not exercise both branches\n";
        return 1;
    }

    DeviceBuffer device_ids   = to_device(ids);
    DeviceBuffer device_table = to_device(table);
    DeviceBuffer device_slots(ids.size() * sizeof(std::uint32_t));
    DeviceBuffer device_misses(sizeof(std::uint32_t));
    const std::uint32_t zero = 0;
    device_misses.copy_from_host(&zero, sizeof(zero));

    ops::moe_route_slots(static_cast<const std::int32_t*>(device_ids.p),
                         static_cast<int>(ids.size()),
                         static_cast<const std::uint32_t*>(device_table.p),
                         static_cast<std::uint32_t>(table.size()),
                         static_cast<std::uint32_t*>(device_slots.p),
                         static_cast<std::uint32_t*>(device_misses.p), stream);
    cuda_check_last_launch("moe_route_slots");
    cuda_synchronize(stream);

    int failures = 0;
    const auto got = from_device<std::uint32_t>(device_slots, ids.size());
    if (got != expected) {
        std::size_t first = 0;
        while (first < got.size() && got[first] == expected[first]) { ++first; }
        std::cerr << "id->slot table mismatch at index " << first << ": id=" << ids[first]
                  << " got=" << got[first] << " expected=" << expected[first] << '\n';
        ++failures;
    }
    std::uint32_t got_misses = 0;
    device_misses.copy_to_host(&got_misses, sizeof(got_misses));
    failures += verify_exact("id->slot miss count", std::vector<std::uint32_t>{got_misses},
                             std::vector<std::uint32_t>{expected_misses});
    return failures;
}

// The two degenerate shapes: no ids at all (the launcher must not launch), and a table of one entry
// where every non-zero id is out of range.
int degenerate_shapes(cudaStream_t stream) {
    DeviceBuffer table = to_device(std::vector<std::uint32_t>{kAbsentSlot});
    DeviceBuffer slots(sizeof(std::uint32_t));
    DeviceBuffer misses(sizeof(std::uint32_t));
    const std::uint32_t zero = 0;
    misses.copy_from_host(&zero, sizeof(zero));

    ops::moe_route_slots(nullptr, 0, static_cast<const std::uint32_t*>(table.p), 1,
                         static_cast<std::uint32_t*>(slots.p),
                         static_cast<std::uint32_t*>(misses.p), stream);
    cuda_check_last_launch("moe_route_slots count=0");
    cuda_synchronize(stream);

    const std::vector<std::int32_t> single{0};
    DeviceBuffer                    ids = to_device(single);
    ops::moe_route_slots(static_cast<const std::int32_t*>(ids.p), 1,
                         static_cast<const std::uint32_t*>(table.p), 1,
                         static_cast<std::uint32_t*>(slots.p),
                         static_cast<std::uint32_t*>(misses.p), stream);
    cuda_check_last_launch("moe_route_slots experts=1");
    cuda_synchronize(stream);

    int failures = 0;
    failures += verify_exact("id->slot absent entry",
                             std::vector<std::uint32_t>{from_device<std::uint32_t>(slots, 1)[0]},
                             std::vector<std::uint32_t>{kAbsentSlot});
    failures += verify_exact("id->slot absent entry counted",
                             std::vector<std::uint32_t>{from_device<std::uint32_t>(misses, 1)[0]},
                             std::vector<std::uint32_t>{1u});
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");

    int failures = 0;
    failures += lookup_against_oracle(stream);
    failures += degenerate_shapes(stream);

    cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    std::cout << (failures == 0 ? "OK" : "FAIL") << " route_slots contract\n";
    return failures == 0 ? 0 : 1;
}
