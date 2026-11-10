// port (sm_89): A1 - the ExpertStream in the gate/up/down form.
//
// Qwen3.8-Flash-Next (Qwen4Exp) stores each expert's gate, up and down as three independent objects
// (q4 gate, q4 up, q6 down, RowSplit), so a layer owns three pool runs instead of two views into one
// entry image. The stream used to describe only the fused [2I,H] bank and could not carry this form
// at all. This test builds that shape synthetically - 8 experts, three banks, no artifact - and
// asserts the properties the form promises:
//
//   * ensure() places the routed experts into ALL THREE runs, and the bytes read back out of each
//     run's slot are the bytes that were read for that (bank, expert) - gate/up q4 and down q6,
//     which have different plane sets and different widths;
//   * every run is priced at its own stride: the q6 down run keeps fewer experts for the same byte
//     budget, so `resident_capacity(Down) < resident_capacity(Gate)`, and each offset table entry is
//     `slot_offset(slot)` of its OWN bank (a single stride multiply would land in a neighbour's run);
//   * the masks are per run and follow the placement, including eviction: a later layer's set takes
//     the seats from the previous one, the displaced experts drop to mask 0 and lose their slot;
//   * the fused `bank.gate_up` view stays EMPTY - the operator refuses a request that names both
//     forms, so a leftover pointer here would be a silent request for the wrong row map.
//
// It runs on a real Device but touches no artifact: the file tier is an in-memory reader whose
// per-expert image is a detectable byte pattern.
#include "core/expert_cache.h"
#include "core/weight_bank.h"
#include "models/qwen3_5/execution/expert_stream.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <vector>

namespace {

using namespace ninfer;
using models::qwen3_5::execution::ExpertMatrixBankDesc;
using models::qwen3_5::execution::ExpertMatrixRole;
using models::qwen3_5::execution::ExpertStream;

constexpr std::uint32_t kExperts = 8;
// The artifact's own routed-expert geometry (Qwen3.8-Flash-Next: H=2560, I=640), so the strides and
// the entry images are the ones the real wiring will see.
constexpr std::uint64_t kHidden  = 2560;
constexpr std::uint64_t kInter   = 640;
constexpr std::int32_t kTopK     = 4;

int failures = 0;

void fail(const char* what) {
    std::fprintf(stderr, "expert stream split: %s\n", what);
    ++failures;
}

// Distinguishable byte for (bank, expert, offset inside the entry image). Every run and every
// expert has its own phase, so a slot that carries another expert's bytes cannot compare equal by
// accident - not even for a single plane.
std::byte pattern(std::uint32_t bank, std::uint32_t expert, std::size_t index) {
    return static_cast<std::byte>((bank * 131 + expert * 37 + index * 7 + 11) & 0xFF);
}

std::vector<std::byte> make_image(std::uint32_t bank, std::uint32_t expert, std::size_t bytes) {
    std::vector<std::byte> image(bytes);
    for (std::size_t i = 0; i < bytes; ++i) { image[i] = pattern(bank, expert, i); }
    return image;
}

// The artifact tier, in memory: one image per (bank, expert). An entry's chunk names its bank
// (`file_index`) and the byte range of that bank's block, exactly like the loader's file mapping.
struct MemoryReader final : ExpertBlobReader {
    const std::vector<std::vector<std::byte>>* images = nullptr;

    void read_entry(const ExpertFileEntry& entry, std::span<std::byte> destination) const override {
        const auto& chunk = entry.chunks.front();
        const auto& image = (*images)[chunk.file_index];
        std::memcpy(destination.data(), image.data() + chunk.file_offset,
                    static_cast<std::size_t>(entry.image_bytes));
    }
};

struct RunExpectation {
    std::uint32_t resident = 0;
    std::vector<std::uint32_t> members;
};

} // namespace

int main() {
    try {
        DeviceContext device(0);

        // One expert matrix per bank, in the artifact's own formats: gate/up are q4, down is q6, so
        // the down entry image is wider and the pool prices it separately.
        const std::uint64_t gate_shape[2]  = {kInter, kHidden};
        const std::uint64_t down_shape[2]  = {kHidden, kInter};
        const auto gate_geometry = weight_geometry(QType::Q4_G64_FP16, QuantLayout::RowSplit, gate_shape);
        const auto down_geometry = weight_geometry(QType::Q6_G64_FP16, QuantLayout::RowSplit, down_shape);
        // A cache entry image always carries two matrix slots (the cache's `ExpertBankLayout` has a
        // gate_up and a down plane set). A run that owns a single matrix leaves the second slot as
        // one row of the same column domain: zero rows are not a valid WeightGeometry
        // (weight_element_count refuses a zero dimension), and expert_cache is out of scope for this
        // change. The placeholder costs ~0.5 KiB per expert and nothing addresses it.
        const std::uint64_t gate_placeholder_shape[2] = {1, kHidden};
        const std::uint64_t down_placeholder_shape[2] = {1, kInter};
        const auto gate_placeholder =
            weight_geometry(QType::Q4_G64_FP16, QuantLayout::RowSplit, gate_placeholder_shape);
        const auto down_placeholder =
            weight_geometry(QType::Q6_G64_FP16, QuantLayout::RowSplit, down_placeholder_shape);

        std::vector<ExpertBankSpec> specs{
            {ExpertMatrixSpec{gate_geometry}, ExpertMatrixSpec{gate_placeholder}, kExperts, false},
            {ExpertMatrixSpec{gate_geometry}, ExpertMatrixSpec{gate_placeholder}, kExperts, false},
            {ExpertMatrixSpec{down_geometry}, ExpertMatrixSpec{down_placeholder}, kExperts, false},
        };

        // Three runs, each charged its own stride by the pool (per-bank pricing engages because the
        // q6 stride exceeds the q4 stride by more than a quarter: 1.44 MiB against 0.875 MiB). An
        // equal share of 1.75 MiB gives the gate/up runs two seats each and the wider down run one,
        // which is the eviction this test needs.
        ExpertCacheConfig config;
        config.capacity_bytes = 3u * 1835008u;
        const auto requirements =
            describe_expert_cache(std::span<const ExpertBankSpec>(specs), config);

        std::vector<std::vector<std::byte>> images(specs.size());
        std::vector<ExpertFileBank> files(specs.size());
        for (std::uint32_t bank = 0; bank < specs.size(); ++bank) {
            const auto entry_bytes = static_cast<std::size_t>(requirements.banks[bank].entry_bytes);
            images[bank].assign(entry_bytes * kExperts, std::byte{0});
            for (std::uint32_t expert = 0; expert < kExperts; ++expert) {
                auto image = make_image(bank, expert, entry_bytes);
                std::memcpy(images[bank].data() + static_cast<std::size_t>(expert) * entry_bytes,
                            image.data(), entry_bytes);
                ExpertFileEntry entry;
                entry.image_bytes = entry_bytes;
                entry.chunks.push_back({bank, static_cast<std::uint64_t>(expert) * entry_bytes,
                                        entry_bytes, 0});
                files[bank].experts.push_back(std::move(entry));
            }
        }
        MemoryReader reader;
        reader.images = &images;

        ExpertCache cache(std::span<const ExpertBankSpec>(specs), requirements, device, {},
                          std::span<const ExpertFileBank>(files), &reader);

        // The artifact's bank layouts, described over a parent that spans the whole bank (each
        // expert is its own object, so the parent is only the row domain the layout indexes).
        const std::uint64_t gate_stack_shape[2] = {kExperts * kInter, kHidden};
        const std::uint64_t down_stack_shape[2] = {kExperts * kHidden, kInter};
        const auto gate_stack =
            weight_geometry(QType::Q4_G64_FP16, QuantLayout::RowSplit, gate_stack_shape);
        const auto down_stack =
            weight_geometry(QType::Q6_G64_FP16, QuantLayout::RowSplit, down_stack_shape);
        const auto gate_layout = describe_weight_bank(gate_stack, {0, kExperts, kInter, kInter});
        const auto down_layout = describe_weight_bank(down_stack, {0, kExperts, kHidden, kHidden});

        const auto desc = [](const WeightBankLayout& layout, const WeightGeometry& parent,
                             std::uint64_t first_object) {
            ExpertMatrixBankDesc out;
            out.layout  = layout;
            out.parent  = parent;
            out.objects.resize(kExperts);
            for (std::uint32_t expert = 0; expert < kExperts; ++expert) {
                out.objects[expert] = first_object + expert;
            }
            return out;
        };
        ExpertStream stream(kExperts, desc(gate_layout, gate_geometry, 1000),
                            desc(gate_layout, gate_geometry, 2000),
                            desc(down_layout, down_geometry, 3000));
        if (!stream.split_banks()) { fail("the stream did not take the gate/up/down form"); }
        if (stream.count() != kExperts) { fail("expert count mismatch"); }
        if (stream.split_expert_geometry(ExpertMatrixRole::Gate).bytes != gate_geometry.bytes ||
            stream.split_expert_geometry(ExpertMatrixRole::Down).bytes != down_geometry.bytes) {
            fail("the per-expert geometries were not derived from the bank layouts");
        }
        if (!stream.per_expert_objects()) {
            fail("a one-object-per-expert bank was not recognised as such");
        }
        stream.attach_split(cache, 0, requirements.banks[0], 1, requirements.banks[1], 2,
                            requirements.banks[2]);

        // The kernel-facing form: three named matrices, and NOT a fused gate_up beside them.
        const auto& experts = stream.experts();
        const auto pool = cache.pool();
        const auto* pool_base = static_cast<const std::byte*>(pool.data);
        const auto view_base = [&](const ExpertBankLayout& entry) {
            return pool_base + entry.codes.offset;
        };
        if (experts.gate.codes != view_base(requirements.banks[0]) ||
            experts.up.codes != view_base(requirements.banks[1]) ||
            experts.bank.down.codes != view_base(requirements.banks[2])) {
            fail("the three views do not name their own run's entry image");
        }
        if (experts.gate.offsets == nullptr || experts.up.offsets == nullptr ||
            experts.bank.down.offsets == nullptr) {
            fail("a run has no offset table");
        }
        if (experts.bank.gate_up.codes != nullptr || experts.bank.gate_up.high != nullptr ||
            experts.bank.gate_up.scales != nullptr || experts.bank.gate_up.offsets != nullptr ||
            experts.bank.gate_up.indexed_storage.data != nullptr) {
            fail("the fused gate_up view is set beside the split views (the operator refuses both)");
        }
        if (experts.bank.count != kExperts) {
            fail("the bank descriptor is wrong");
        }

        // Per-bank pricing: the q6 run is wider, so the same byte budget gives it fewer seats.
        const auto gate_capacity = cache.resident_capacity(0);
        const auto up_capacity   = cache.resident_capacity(1);
        const auto down_capacity = cache.resident_capacity(2);
        const auto gate_stride   = cache.slot_stride(0);
        const auto down_stride   = cache.slot_stride(2);
        std::fprintf(stderr,
                     "expert stream split: layout entry=%llu/%llu/%llu stride=%llu/%llu/%llu "
                     "seats=%u/%u/%u\n",
                     static_cast<unsigned long long>(requirements.banks[0].entry_bytes),
                     static_cast<unsigned long long>(requirements.banks[1].entry_bytes),
                     static_cast<unsigned long long>(requirements.banks[2].entry_bytes),
                     static_cast<unsigned long long>(gate_stride),
                     static_cast<unsigned long long>(cache.slot_stride(1)),
                     static_cast<unsigned long long>(down_stride),
                     gate_capacity, up_capacity, down_capacity);
        if (gate_capacity == 0 || down_capacity == 0) { fail("a run has no seat at all"); }
        if (gate_capacity >= kExperts || down_capacity >= kExperts) {
            fail("the fixture needs a pool that cannot hold every expert");
        }
        if (gate_capacity != up_capacity) { fail("the two q4 runs must have the same width"); }
        if (!(down_capacity < gate_capacity)) {
            fail("the wider down run was not priced at its own stride");
        }
        if (!(down_stride > gate_stride)) { fail("the down stride does not exceed the gate stride"); }
        if (stream.split_resident_capacity(ExpertMatrixRole::Gate) != gate_capacity ||
            stream.split_resident_capacity(ExpertMatrixRole::Down) != down_capacity) {
            fail("split_resident_capacity does not report each run's own seats");
        }

        // Reading one run back: the Device tables against the host copies, then the bytes inside
        // every resident slot against the pattern that was read for that expert.
        const auto verify_run = [&](ExpertMatrixRole role, const RunExpectation& expected,
                                    const char* what) {
            const auto index = static_cast<std::size_t>(role);
            const auto bank  = stream.split_bank(role);
            const auto entry = requirements.banks[bank];
            const auto offsets_host = stream.split_offsets_host(role);
            const auto mask_host    = stream.split_mask_host(role);
            if (offsets_host.size() != kExperts || mask_host.size() != kExperts) {
                fail("a run table has the wrong length");
                return;
            }
            std::vector<std::uint64_t> offsets_device(kExperts);
            std::vector<std::int32_t> mask_device(kExperts);
            const auto copy_offsets =
                cudaMemcpy(offsets_device.data(), stream.split_offsets_device(role),
                           kExperts * sizeof(std::uint64_t), cudaMemcpyDeviceToHost);
            const auto copy_mask = cudaMemcpy(mask_device.data(), stream.split_mask_device(role),
                                              kExperts * sizeof(std::int32_t), cudaMemcpyDeviceToHost);
            if (copy_offsets != cudaSuccess || copy_mask != cudaSuccess) {
                fail("a run's Device tables cannot be read back");
                return;
            }
            if (std::memcmp(offsets_device.data(), offsets_host.data(),
                            kExperts * sizeof(std::uint64_t)) != 0 ||
                std::memcmp(mask_device.data(), mask_host.data(), kExperts * sizeof(std::int32_t)) != 0) {
                std::fprintf(stderr, "expert stream split: %s: the Device tables differ from the host copies\n", what);
                ++failures;
            }
            std::uint32_t resident = 0;
            for (std::uint32_t expert = 0; expert < kExperts; ++expert) {
                const auto slot = cache.expert_slot(bank, expert);
                if (mask_host[expert] == 0) {
                    if (slot) { std::fprintf(stderr, "expert stream split: %s: expert %u is masked but holds a slot\n", what, expert); ++failures; }
                    if (offsets_host[expert] != 0) { std::fprintf(stderr, "expert stream split: %s: expert %u keeps an offset while masked\n", what, expert); ++failures; }
                    continue;
                }
                ++resident;
                if (!slot) { std::fprintf(stderr, "expert stream split: %s: expert %u is resident by mask but has no slot\n", what, expert); ++failures; continue; }
                const auto offset = cache.slot_offset(static_cast<std::uint32_t>(*slot));
                if (offsets_host[expert] != offset) {
                    std::fprintf(stderr, "expert stream split: %s: expert %u offset is not its own slot address\n", what, expert);
                    ++failures;
                }
                // Read the slot back out of the pool and compare it with the image that was read
                // for this (bank, expert): the whole entry, planes and padding.
                std::vector<std::byte> actual(static_cast<std::size_t>(entry.entry_bytes));
                const auto status = cudaMemcpy(actual.data(), pool_base + offset, actual.size(),
                                               cudaMemcpyDeviceToHost);
                if (status != cudaSuccess) { fail("a pool slot cannot be read back"); continue; }
                for (std::size_t i = 0; i < actual.size(); ++i) {
                    if (actual[i] != pattern(bank, expert, i)) {
                        std::fprintf(stderr,
                                     "expert stream split: %s: bank %u expert %u byte %zu holds 0x%02x, expected 0x%02x\n",
                                     what, bank, expert, i, int(actual[i]), int(pattern(bank, expert, i)));
                        ++failures;
                        break;
                    }
                }
            }
            if (resident != expected.resident) {
                std::fprintf(stderr, "expert stream split: %s: run %zu holds %u experts, expected %u\n",
                             what, index, resident, expected.resident);
                ++failures;
            }
            // The published count the stream reports must be the same population the mask holds.
            if (stream.split_resident(role) != resident) {
                std::fprintf(stderr,
                             "expert stream split: %s: the published count %u differs from the mask %u\n",
                             what, stream.split_resident(role), resident);
                ++failures;
            }
            for (const auto expert : expected.members) {
                if (mask_host[expert] == 0) {
                    std::fprintf(stderr, "expert stream split: %s: routed expert %u is not resident\n",
                                 what, expert);
                    ++failures;
                }
            }
        };

        // Layer 1: routes 5,6,0,1. The gate/up runs keep the first two (their seats), the down run
        // keeps the first one. Experts 0 and 1 were the cache's seed, so they are displaced and must
        // come back out of the placement.
        const std::vector<std::int32_t> first_routes{5, 6, 0, 1};
        {
            DeviceBuffer ids(first_routes.size() * sizeof(std::int32_t));
            ids.copy_from_host(first_routes.data(), first_routes.size() * sizeof(std::int32_t));
            const bool placed = stream.ensure(ids.p, kTopK, 1, nullptr, nullptr);
            if (!placed) { fail("ensure() refused a layer the pool can hold"); }
        }
        const auto sync_first = cudaDeviceSynchronize();
        if (sync_first != cudaSuccess) { fail("the first layer's uploads did not land"); }
        verify_run(ExpertMatrixRole::Gate, {gate_capacity, {5, 6}}, "layer 1 gate");
        verify_run(ExpertMatrixRole::Up, {up_capacity, {5, 6}}, "layer 1 up");
        verify_run(ExpertMatrixRole::Down, {down_capacity, {5}}, "layer 1 down");
        if (cache.expert_slot(0, 0) || cache.expert_slot(0, 1)) {
            fail("the displaced seed experts kept their slots");
        }
        if (cache.uploads() < gate_capacity + up_capacity + down_capacity) {
            fail("the first layer did not upload every placed expert");
        }
        const auto uploads_after_first = cache.uploads();

        // Layer 2: routes 1,2,3,4. Every expert of layer 1 that is not routed again is displaced,
        // and the two runs have to carry the new contents - which is what proves the placement is
        // per layer and not a one-off seed.
        const std::vector<std::int32_t> second_routes{1, 2, 3, 4};
        {
            DeviceBuffer ids(second_routes.size() * sizeof(std::int32_t));
            ids.copy_from_host(second_routes.data(), second_routes.size() * sizeof(std::int32_t));
            // The host copy this time: the fused path accepts it too, and the three-run path must
            // not depend on where the ids came from.
            const bool placed = stream.ensure(nullptr, kTopK, 1, nullptr, second_routes.data());
            if (!placed) { fail("ensure() refused the second layer"); }
        }
        const auto sync_second = cudaDeviceSynchronize();
        if (sync_second != cudaSuccess) { fail("the second layer's uploads did not land"); }
        verify_run(ExpertMatrixRole::Gate, {gate_capacity, {1, 2}}, "layer 2 gate");
        verify_run(ExpertMatrixRole::Up, {up_capacity, {1, 2}}, "layer 2 up");
        verify_run(ExpertMatrixRole::Down, {down_capacity, {1}}, "layer 2 down");
        if (cache.expert_slot(0, 5) || cache.expert_slot(0, 6)) {
            fail("layer 1's experts kept their slots through layer 2");
        }
        if (cache.uploads() <= uploads_after_first) {
            fail("the second layer moved nothing into the pool");
        }
        // The demotion journal is what the warm tier admits from; on this form it is the gate run's
        // answer (each run prices its own seats, so each run has its own set to demote from).
        const auto demoted = stream.demoted_this_step();
        for (const auto expert : {5, 6}) {
            bool listed = false;
            for (const auto entry : demoted) { listed = listed || entry == expert; }
            if (!listed) {
                std::fprintf(stderr, "expert stream split: demoted expert %d is not journaled\n", expert);
                ++failures;
            }
        }

        // The switches that used to be refused on this form are applied per run now: a tier bank, a
        // GH3 placement word, a prediction region and a fixed seat ring exist once per matrix, so a
        // run that asks for one is served rather than told. What stays pinned here is the wiring:
        // attach_split() accepts them, and the question the engine asks before building a tier
        // answers yes (the three-run form carries one tier bank per run).
        {
            _putenv_s("NINFER_PREFETCH", "1");
            bool accepted = false;
            try {
                ExpertStream other(kExperts, desc(gate_layout, gate_geometry, 4000),
                                   desc(gate_layout, gate_geometry, 5000),
                                   desc(down_layout, down_geometry, 6000));
                other.attach_split(cache, 0, requirements.banks[0], 1, requirements.banks[1], 2,
                                   requirements.banks[2]);
                accepted = true;
            } catch (const std::logic_error&) {
                accepted = false;
            }
            _putenv_s("NINFER_PREFETCH", "");
            if (!accepted) {
                fail("attach_split() refused a switch that is now applied per run");
            }
            if (!stream.host_tier_supported()) {
                fail("host_tier_supported() refused the three-run form, which has a tier per run");
            }
        }

        if (failures != 0) { return 1; }
        std::printf("expert stream split: ok experts=%u gate=%u up=%u down=%u stride=%llu/%llu "
                    "uploads=%llu evictions=%llu entry=%llu/%llu\n",
                    kExperts, gate_capacity, up_capacity, down_capacity,
                    static_cast<unsigned long long>(gate_stride),
                    static_cast<unsigned long long>(down_stride),
                    static_cast<unsigned long long>(cache.uploads()),
                    static_cast<unsigned long long>(cache.evictions()),
                    static_cast<unsigned long long>(requirements.banks[0].entry_bytes),
                    static_cast<unsigned long long>(requirements.banks[2].entry_bytes));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "expert stream split: %s\n", error.what());
        return 1;
    }
}
