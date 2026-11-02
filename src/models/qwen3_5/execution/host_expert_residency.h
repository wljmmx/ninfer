#pragma once

#include "core/expert_cache.h"
#include "core/expert_file.h"
#include "core/weight_bank.h"
#include "ninfer/ops/moe.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <span>
#include <string>
#include <stdexcept>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace ninfer::models::qwen3_5::execution {

// Everything the host tier needs to describe one expert bank: the slot image the loader fills, the
// plane layout the CPU kernels address through, and the geometry those views belong to.
struct HostExpertBankDesc {
    ExpertBankLayout entry;
    WeightBankLayout gate_up_layout;
    WeightBankLayout down_layout;
    WeightGeometry gate_up_geometry;
    WeightGeometry down_geometry;
    // port (sm_89): true when the artifact gives every expert its own object. It has to be carried
    // here: a per-expert artifact may still pack all those objects into one container file, so the
    // chunk's file index cannot tell the two layouts apart. Keying on it made an e2 bank report
    // `one=1` and would have addressed expert e as `base + e * stride` - the wrong image entirely.
    bool per_expert = false;
};

// The warm tier's three budgets, in bytes and kept apart. They used to be one arena with the
// auxiliary ranges carved out of the routed range, so enabling prefetch or the second-chance band
// silently took routed slots away - the pool got smaller exactly when a cache was added. Each range
// now has its own grant: the caller reserves these separately (and routed first), and this class
// only gives each its own slots. A range that is not granted is zero; it never borrows from routed.
struct HostWarmBudget {
    std::uint64_t routed_bytes        = 0;
    std::uint64_t prefetch_bytes      = 0;
    std::uint64_t second_chance_bytes = 0;
};

// The host half of a split expert tier. It reserves one arena up front, commits it slot by slot,
// and holds one uniform slot per expert using the same entry image the Device pool stores. A slot
// is evicted coldest-first when the arena fills, so RAM holds what fits and everything else keeps
// streaming in from the artifact.
//
// The CPU op validates a host-packed bank: planes start at 0, at codes_bytes and at
// codes_bytes + high_bytes, and expert_offsets name each expert's segment. Gate/up and down begin
// at different byte origins inside that segment, so each plane view carries its own table.
class HostExpertResidency final : public HostResidency, public ExpertArenaSource {
public:
    HostExpertResidency(std::span<const HostExpertBankDesc> banks,
                        std::span<const ExpertFileBank> files, const ExpertBlobReader* reader,
                        HostWarmBudget budget)
        : banks_(banks.begin(), banks.end()), files_(files.begin(), files.end()), reader_(reader) {
        std::uint64_t widest = 0;
        for (const auto& bank : banks_) {
            widest = std::max(widest, bank.entry.entry_bytes);
        }
        slot_bytes_ = slot_bytes_for(widest);
        const auto slots_for = [this](std::uint64_t bytes) {
            return static_cast<std::uint32_t>(std::min<std::uint64_t>(
                bytes / slot_bytes_, std::numeric_limits<std::uint32_t>::max()));
        };
        // Routed is the primary range and is granted first by the caller. Prefetch and the
        // second-chance band get their own bytes; they are not subtracted from routed here.
        routed_capacity_ = slots_for(budget.routed_bytes);
        prefetch_slots_  = slots_for(budget.prefetch_bytes);
        second_chance_slots_      = slots_for(budget.second_chance_bytes);
        if (routed_capacity_ == 0) { routed_capacity_ = 1; } // a warm tier with no routed slot is inert
        slot_capacity_ = routed_capacity_ + prefetch_slots_ + second_chance_slots_;
        arena_bytes_   = static_cast<std::uint64_t>(slot_capacity_) * slot_bytes_;
        if (second_chance_slots_ != 0) {
            std::fprintf(stderr,
                         "[warm] second-chance band slots=%u (routed=%u read-ahead=%u) - its own "
                         "budget, not taken from routed\n",
                         second_chance_slots_, routed_capacity_, prefetch_slots_);
            std::fflush(stderr);
        }
        if (std::getenv("NINFER_HOST_WARM") != nullptr) {
            std::fprintf(stderr,
                         "[warm] arena path enabled: CPU half reads the pinned tier, in-place "
                         "artifact addressing disabled\n");
            std::fflush(stderr);
        }
        slot_bank_.assign(slot_capacity_, kNone);
        slot_expert_.assign(slot_capacity_, kNone);
        slot_last_use_.assign(slot_capacity_, 0);
        // port (sm_89): slots whose bytes a Device copy is reading right now. Handing out the pointer in
        // arena_image() is the only moment that becomes true, and uploads_settled() is the only moment
        // it stops - see the note on ExpertArenaSource.
        in_flight_.assign(slot_capacity_, 0);
        // port (sm_89): slots the read-ahead filled and no routed layer has used yet. Per slot, not
        // per expert, because that is the lifetime that matters: whichever expert the slot holds
        // next loses the mark, and only a hit on the same slot can count as usefulness.
        prefetched_.assign(slot_capacity_, 0);
        // Q20 second leg: slots a queued GEMM is addressing. Two generations, see begin_layer().
        device_held_.assign(slot_capacity_, 0);
        device_held_prev_.assign(slot_capacity_, 0);

        state_.resize(banks_.size());
        direct_.assign(banks_.size(), false);
        // Q69: one bit per expert per bank, so a warming pass can tell "already in the page cache"
        // from "must be read". Sized here rather than lazily because load() asks the question on
        // every fill and a size check per fill is not worth the branch.
        warmed_.resize(files_.size());
        for (std::size_t i = 0; i < banks_.size(); ++i) {
            auto& s = state_[i];
            s.slot_of.assign(banks_[i].entry.count, kNone);
            s.gate_up_offsets.assign(banks_[i].entry.count, 0);
            s.down_offsets.assign(banks_[i].entry.count, 0);
            s.gu_codes.assign(banks_[i].entry.count, 0);
            s.gu_high.assign(banks_[i].entry.count, 0);
            s.gu_scales.assign(banks_[i].entry.count, 0);
            s.dn_codes.assign(banks_[i].entry.count, 0);
            s.dn_high.assign(banks_[i].entry.count, 0);
            s.dn_scales.assign(banks_[i].entry.count, 0);

            // port (sm_89): a bank that lives in one object can be addressed where it already is.
            // The artifact parent then holds every expert's rows, so the CPU op addresses expert e
            // by row range and nothing has to be copied. Only the fused layout qualifies: the
            // per-expert layout spreads the bank over one object per expert, which a single parent
            // cannot describe, so it keeps the arena path.
            const auto& bank_files = i < files_.size() ? files_[i] : ExpertFileBank{};
            // A per-expert bank never qualifies, whatever its container looks like: its experts are
            // separate objects, and one parent can only describe one of them.
            const bool one_object =
                !banks_[i].per_expert && !bank_files.experts.empty() &&
                !bank_files.experts.front().chunks.empty() &&
                std::all_of(bank_files.experts.begin(), bank_files.experts.end(),
                            [&](const ExpertFileEntry& entry) {
                                return !entry.chunks.empty() &&
                                       entry.chunks.front().file_index ==
                                           bank_files.experts.front().chunks.front().file_index;
                            });
            const auto gate_up_base =
                one_object && reader_ != nullptr
                    ? reader_->object_base(bank_files.experts.front().chunks.front().file_index)
                    : nullptr;
            const auto down_base =
                one_object && reader_ != nullptr
                    ? reader_->object_base(bank_files.experts.front().chunks.back().file_index)
                    : nullptr;
            if (std::getenv("NINFER_HOST_DEBUG") != nullptr && i < 2) {
                std::fprintf(stderr,
                             "[direct] bank=%zu one=%d per_expert=%d count=%u base=%p/%p\n", i,
                             int(one_object), int(banks_[i].per_expert),
                             banks_[i].entry.count, static_cast<const void*>(gate_up_base),
                             static_cast<const void*>(down_base));
            }
            // A bank that fits in one object can be addressed in place. The gate is the pair of
            // bases: a non-null base means the object was mapped, and the bank's experts sit at
            // expert * expert_stride inside it. Off by default: the addressing is only correct if
            // the bank is plane-major with a stride matching its own plane, so it stays behind a
            // switch until that has been verified against the PPL gate.
            // port (sm_89): in-place addressing is the default now that both layouts resolve. The
            // branch picks itself per bank - a fused bank has one object for the whole bank, a
            // per-expert bank has one object per expert - and each one verifies its own layout
            // before it is used, falling back to the copying path when the check fails.
            // NINFER_HOST_COPY=1 forces the copying path, which is what the split gate compares
            // against.
            static const bool warm_arena = std::getenv("NINFER_HOST_WARM") != nullptr;
            static const bool allow_direct =
                std::getenv("NINFER_HOST_COPY") == nullptr && !managed_even_in_place() &&
                !warm_arena;
            // port (sm_89): a fused bank keeps codes, high and scales in three runs per matrix, far
            // apart inside one object, so one per-expert offset cannot name all three. Each plane
            // gets its own table (WeightBankView::codes_offsets and friends), filled from the chunk
            // table - which is the authority on where a run starts, and is file absolute, the same
            // convention the per-expert plan below uses.
            if (allow_direct && !banks_[i].per_expert && reader_ != nullptr &&
                !bank_files.experts.empty() && banks_[i].entry.count > 0) {
                const auto& entry     = banks_[i].entry;
                const auto& gu_layout = banks_[i].gate_up_layout;
                const auto& dn_layout = banks_[i].down_layout;
                const auto count      = static_cast<std::uint32_t>(entry.count);
                bool usable           = !bank_files.experts.front().chunks.empty();
                const std::uint32_t file_index =
                    usable ? bank_files.experts.front().chunks.front().file_index : 0;
                // widest chunk seen per plane: the layout and the artifact do not agree on it.
                std::uint64_t plane_bytes[6] = {};
                for (std::uint32_t e = 0; e < count && usable; ++e) {
                    const auto& ex = bank_files.experts[e];
                    if (ex.chunks.empty()) {
                        usable = false;
                        break;
                    }
                    for (const auto& chunk : ex.chunks) {
                        if (chunk.file_index != file_index) {
                            usable = false;
                            break;
                        }
                        // Sort by the plane's image range, not by "before the next plane": a q4
                        // gate_up has no high plane, so its high offset equals its scales offset.
                        const struct {
                            std::uint64_t begin, bytes;
                            std::vector<std::uint64_t>* target;
                        } planes[] = {
                            {entry.codes.offset, entry.codes.bytes, &s.gu_codes},
                            {entry.high.offset, entry.high.bytes, &s.gu_high},
                            {entry.scales.offset, entry.scales.bytes, &s.gu_scales},
                            {entry.down_codes.offset, entry.down_codes.bytes, &s.dn_codes},
                            {entry.down_high.offset, entry.down_high.bytes, &s.dn_high},
                            {entry.down_scales.offset, entry.down_scales.bytes, &s.dn_scales},
                        };
                        int index = -1;
                        for (int pi = 0; pi < 6; ++pi) {
                            const auto& plane = planes[pi];
                            if (!plane.bytes) { continue; }
                            if (chunk.image_offset >= plane.begin &&
                                chunk.image_offset < plane.begin + plane.bytes) {
                                index = pi;
                                break;
                            }
                        }
                        if (index < 0) {
                            usable = false;
                            break;
                        }
                        plane_bytes[index] = std::max(plane_bytes[index], chunk.bytes);
                        // Only a plane's first segment names its origin; the rest continue it.
                        if (chunk.image_offset == planes[index].begin) {
                            (*planes[index].target)[e] = chunk.file_offset;
                        }
                    }
                }
                // The backing has to cover what plane() checks: every origin plus that plane's size.
                std::uint64_t extent = 0;
                const auto reach     = [&](const std::vector<std::uint64_t>& table,
                                       std::uint64_t bytes) {
                    if (!bytes) { return; }
                    for (const auto origin : table) { extent = std::max(extent, origin + bytes); }
                };
                reach(s.gu_codes, std::max<std::uint64_t>(gu_layout.codes.expert_bytes, plane_bytes[0]));
                reach(s.gu_high, std::max<std::uint64_t>(gu_layout.high.expert_bytes, plane_bytes[1]));
                reach(s.gu_scales,
                      std::max<std::uint64_t>(gu_layout.scales.expert_bytes, plane_bytes[2]));
                reach(s.dn_codes, std::max<std::uint64_t>(dn_layout.codes.expert_bytes, plane_bytes[3]));
                reach(s.dn_high, std::max<std::uint64_t>(dn_layout.high.expert_bytes, plane_bytes[4]));
                reach(s.dn_scales,
                      std::max<std::uint64_t>(dn_layout.scales.expert_bytes, plane_bytes[5]));
                const auto* mapped = usable ? reader_->object_base(file_index) : nullptr;
                if (mapped != nullptr) {
                    s.gate_up_parent.geometry       = banks_[i].gate_up_geometry;
                    s.gate_up_parent.geometry.bytes = extent;
                    s.gate_up_parent.device         = Device::cpu();
                    s.gate_up_parent.data           = mapped;
                    s.down_parent.geometry          = banks_[i].down_geometry;
                    s.down_parent.geometry.bytes    = extent;
                    s.down_parent.device            = Device::cpu();
                    s.down_parent.data              = mapped;
                    s.gate_up.parent         = &s.gate_up_parent;
                    s.gate_up.layout         = with_zero_planes(gu_layout);
                    s.gate_up.codes_offsets  = s.gu_codes.data();
                    s.gate_up.high_offsets   = s.gu_high.data();
                    s.gate_up.scales_offsets = s.gu_scales.data();
                    s.gate_up.expert_offsets = nullptr;
                    s.down.parent         = &s.down_parent;
                    s.down.layout         = with_zero_planes(dn_layout);
                    s.down.codes_offsets  = s.dn_codes.data();
                    s.down.high_offsets   = s.dn_high.data();
                    s.down.scales_offsets = s.dn_scales.data();
                    s.down.expert_offsets = nullptr;
                    s.view.gate_up        = s.gate_up;
                    s.view.down           = s.down;
                    direct_[i]            = true;
                    if (std::getenv("NINFER_HOST_DEBUG") != nullptr && i < 2) {
                        std::fprintf(stderr,
                                     "[fused-in-place] bank=%zu experts=%u gu_codes=%llu "
                                     "gu_scales=%llu dn_codes=%llu dn_scales=%llu extent=%llu\n",
                                     i, count, (unsigned long long)s.gu_codes.front(),
                                     (unsigned long long)s.gu_scales.front(),
                                     (unsigned long long)s.dn_codes.front(),
                                     (unsigned long long)s.dn_scales.front(),
                                     (unsigned long long)extent);
                    }
                    continue;
                }
            }

            // port (sm_89): a per-expert bank is addressable in place. Its experts are one object
            // each, but those objects live in one container file and the chunk table already says
            // where each image starts, so the per-expert offset table the arena path fills with a
            // slot can name a place in the mapped file instead: no ensure, no arena bytes, no second
            // pass over 2.8 MB per layer.
            //
            // The image is contiguous per matrix, not as a whole: on the 35B artifact an expert's
            // gate_up run sits at 3,383,009,792 and its down run 284 MB later (measured). Each
            // matrix has its own parent and its own offset table, so the two runs are resolved
            // independently - which is exactly why the single-parent form failed before. The offset
            // stored is `run_base + plane_offset_in_image`, the same shape the arena branch stores
            // as `slot * slot_bytes + plane_offset`, so the op cannot tell the two apart.
            if (allow_direct && banks_[i].per_expert && reader_ != nullptr &&
                !bank_files.experts.empty() && !bank_files.experts.front().chunks.empty()) {
                const auto file_index  = bank_files.experts.front().chunks.front().file_index;
                const auto& entry      = banks_[i].entry;
                const auto down_origin = entry.down_codes.offset;
                bool usable            = true;
                std::uint64_t widest_end = 0;
                std::vector<std::uint64_t> gate_up_run(bank_files.experts.size(), 0);
                std::vector<std::uint64_t> down_run(bank_files.experts.size(), 0);
                // The planes are stored plane-major across the bank, so each expert's run base
                // differs (run_base + expert * stride). What has to hold is that, inside one expert
                // and one matrix, every chunk names the same base: that is what makes the pair of
                // pointers the op derives from `base + offset[e]` land on the right bytes.
                for (std::size_t e = 0; e < bank_files.experts.size() && usable; ++e) {
                    const auto& expert_entry = bank_files.experts[e];
                    if (expert_entry.chunks.empty()) { usable = false; break; }
                    bool have_gate_up = false, have_down = false;
                    for (const auto& chunk : expert_entry.chunks) {
                        if (chunk.file_index != file_index) { usable = false; break; }
                        const auto run      = chunk.file_offset - chunk.image_offset;
                        const bool is_down  = chunk.image_offset >= down_origin;
                        auto& base          = is_down ? down_run[e] : gate_up_run[e];
                        auto& seen          = is_down ? have_down : have_gate_up;
                        if (!seen) {
                            base = run;
                            seen = true;
                        } else if (base != run) {
                            usable = false;
                            break;
                        }
                    }
                    if (!usable) { break; }
                    widest_end = std::max<std::uint64_t>(widest_end, gate_up_run[e] + entry.entry_bytes);
                    widest_end = std::max<std::uint64_t>(widest_end, down_run[e] + entry.entry_bytes);
                }
                const auto* mapped = usable ? reader_->object_base(file_index) : nullptr;
                if (mapped != nullptr) {
                    for (std::size_t e = 0; e < bank_files.experts.size(); ++e) {
                        s.gate_up_offsets[e] = gate_up_run[e] + entry.codes.offset;
                        s.down_offsets[e]    = down_run[e] + entry.down_codes.offset;
                    }
                    s.gate_up_parent.geometry       = banks_[i].gate_up_geometry;
                    s.gate_up_parent.geometry.bytes = widest_end;
                    s.gate_up_parent.device         = Device::cpu();
                    s.gate_up_parent.data           = mapped;
                    s.down_parent.geometry          = banks_[i].down_geometry;
                    s.down_parent.geometry.bytes    = widest_end;
                    s.down_parent.device            = Device::cpu();
                    s.down_parent.data              = mapped;
                    s.gate_up.parent         = &s.gate_up_parent;
                    s.gate_up.layout         = with_packed_planes(banks_[i].gate_up_layout);
                    s.gate_up.expert_offsets = s.gate_up_offsets.data();
                    s.down.parent         = &s.down_parent;
                    s.down.layout         = with_packed_planes(banks_[i].down_layout);
                    s.down.expert_offsets = s.down_offsets.data();
                    s.view.gate_up        = s.gate_up;
                    s.view.down           = s.down;
                    direct_[i]            = true;
                    if (std::getenv("NINFER_HOST_DEBUG") != nullptr && i < 2) {
                        std::fprintf(stderr,
                                     "[in-place] bank=%zu file=%u experts=%zu gu_run=%llu "
                                     "dn_run=%llu end=%llu mapped=%p\n",
                                     i, file_index, bank_files.experts.size(),
                                     (unsigned long long)gate_up_run.front(),
                                     (unsigned long long)down_run.front(),
                                     (unsigned long long)widest_end,
                                     static_cast<const void*>(mapped));
                    }
                    continue;
                }
                if (std::getenv("NINFER_HOST_DEBUG") != nullptr && i < 2) {
                    std::fprintf(stderr, "[in-place] bank=%zu usable=%d mapped=null\n", i,
                                 int(usable));
                }
            }

            // The arena, not the artifact parent, is the backing these views address: the offset
            // table runs over the whole slot range, so the span check has to see the arena size.
            s.gate_up_parent.geometry       = banks_[i].gate_up_geometry;
            s.gate_up_parent.geometry.bytes = arena_bytes_;
            s.gate_up_parent.device         = Device::cpu();
            s.down_parent.geometry          = banks_[i].down_geometry;
            s.down_parent.geometry.bytes    = arena_bytes_;
            s.down_parent.device            = Device::cpu();

            s.gate_up.parent         = &s.gate_up_parent;
            s.gate_up.layout         = with_packed_planes(banks_[i].gate_up_layout);
            s.gate_up.expert_offsets = s.gate_up_offsets.data();

            s.down.parent         = &s.down_parent;
            s.down.layout         = with_packed_planes(banks_[i].down_layout);
            s.down.expert_offsets = s.down_offsets.data();

            s.view.gate_up = s.gate_up;
            s.view.down    = s.down;
        }
    }

    // port (sm_89): R3's shape, in one place so the budget that reserves the bytes and the class
    // that uses them cannot disagree.
    static std::uint64_t slot_bytes_for(std::uint64_t entry_bytes) {
        constexpr std::uint64_t kBlock = 128 * 1024;
        return (entry_bytes + kBlock - 1) / kBlock * kBlock;
    }
    // Slots the read-ahead owns. Zero unless the read-ahead is on: an off switch must not move the
    // arena size, or "prefetch off" would stop being the baseline it is compared against.
    static std::uint32_t prefetch_slots_from_env() {
        if (std::getenv("NINFER_PREFETCH") == nullptr) { return 0; }
        if (const char* value = std::getenv("NINFER_PREFETCH_SLOTS");
            value != nullptr && *value != '\0') {
            return static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10));
        }
        return 128;
    }
    // port (sm_89): Q53's second-chance band. Zero by default, because a region that changes where an
    // evicted expert goes changes the split path's read behaviour, and this is wiring: the switch has
    // to be measurable against a run without it.
    static std::uint32_t second_chance_slots_from_env() {
        const char* value = std::getenv("NINFER_WARM_SECOND_CHANCE_SLOTS");
        // v8 renamed the band; keep the old spelling as an input alias so recorded experiments are
        // still runnable, but never print the old name as if it were the current one.
        if (value == nullptr || *value == '\0') { value = std::getenv("NINFER_WARM_COLD_SLOTS"); }
        if (value != nullptr && *value != '\0') {
            return static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10));
        }
        return 0;
    }
    // Bytes those slots add to the arena, so the ledger can be told before the arena exists.
    static std::uint64_t prefetch_bytes_for(std::uint64_t entry_bytes) {
        return std::uint64_t{prefetch_slots_from_env()} * slot_bytes_for(entry_bytes);
    }
    // Q53: the second-chance band asks for its own bytes now. Zero unless NINFER_WARM_SECOND_CHANCE_SLOTS
    // says otherwise; the caller reserves it separately and never subtracts it from routed.
    static std::uint64_t second_chance_bytes_for(std::uint64_t entry_bytes) {
        return std::uint64_t{second_chance_slots_from_env()} * slot_bytes_for(entry_bytes);
    }

    void ensure(ExpertBankId bank, std::span<const std::int32_t> experts,
                const std::int32_t* resident_mask) override {
        // port (sm_89): the slot tables and the arena are mutated here and read by the uploader in
        // arena_image(), and D1 wants the fill on its own thread. One lock covers all three entry
        // points so that is a safe call to make; without it the two threads would race on slot_of.
        std::lock_guard<std::mutex> guard(mutex_);
        fill_locked(bank, experts, resident_mask, /*prefetch=*/false, nullptr);
    }

    // Q53: the ranked fill. `scores` is this bank's decayed route score (one float per expert), the
    // same number the Device-side placement ranks by. The second-chance band's admission uses it to answer
    // "is the expert being evicted hotter than the one that would lose its second-chance slot"; without it
    // the region is the old "eviction is admission" rule, which measured +33% ens / -29% tps.
    //
    // The pointer is borrowed for the duration of the call and is stable for the bank's life (the
    // stream allocates the score vector once). A torn read would at worst mis-rank one admission,
    // never mis-address a slot: the slot tables themselves are only mutated under this lock.
    void ensure_scored(ExpertBankId bank, std::span<const std::int32_t> experts,
                       const float* scores) {
        std::lock_guard<std::mutex> guard(mutex_);
        fill_locked(bank, experts, nullptr, /*prefetch=*/false, scores);
    }

    // Q53: where the admitted second-chance slots currently end, so the panel can say whether the region is
    // full (a full region is the only state in which admission is a comparison). A promoted slot is
    // returned to the region, not leaked: without that the region shrinks by one every promotion.
    [[nodiscard]] std::uint32_t second_chance_used() const noexcept {
        return second_chance_used_ - static_cast<std::uint32_t>(second_chance_free_.size());
    }

    // port (sm_89): the read-ahead's entry point. Same fill, but every slot it claims is marked, so
    // that when the layer later routes to that expert the hit can be attributed to the read-ahead
    // and reported as usefulness. Nothing else changes: a marked slot is an ordinary slot, and the
    // expert is served from it whether the mark was set or not.
    void fill_prefetch(ExpertBankId bank, std::span<const std::int32_t> experts) {
        std::lock_guard<std::mutex> guard(mutex_);
        fill_locked(bank, experts, nullptr, /*prefetch=*/true, nullptr);
    }

    // Routed experts that were served from a slot the read-ahead had filled, since the last call.
    // The consumer (ExpertStream) turns this into bytes and reports it to the queue's usefulness
    // gate; it is a count of experts, not bytes, so the tier stays the one that knows slot_bytes.
    [[nodiscard]] std::uint64_t take_prefetch_hits() {
        std::lock_guard<std::mutex> guard(mutex_);
        const auto taken = prefetch_hits_;
        prefetch_hits_   = 0;
        return taken;
    }

    // Q69: the read-ahead's other target, and the reason it exists.
    //
    // The tier is 930 MiB of arena carved into slots, and a read-ahead that fills them competes with
    // the routes a layer actually needs - measured at 0.20 useful, i.e. four fifths of what it put in
    // the arena was never routed to. The OS already keeps every expert it has read in the page cache,
    // so there is a second, cheaper target: read the entry through the buffered handle into a scratch
    // buffer and let the page cache hold it. No slot is claimed, nothing can be evicted, and the fill
    // that follows a real miss is a memcpy out of RAM instead of a physical read - which is exactly
    // the 8.6 ms -> 0.2 ms this file already measured on the fill path.
    //
    // The scratch buffer is one slot, allocated on first use, so the cost of the mode is 1.8 MiB and
    // not a second arena.
    void warm(ExpertBankId bank, std::span<const std::int32_t> experts) {
        std::lock_guard<std::mutex> guard(mutex_);
        warm_locked(bank, experts);
    }

    void warm_locked(ExpertBankId bank, std::span<const std::int32_t> experts) {
        if (bank.index >= files_.size()) { return; }
        if (warm_scratch_.empty()) {
            warm_scratch_.resize(static_cast<std::size_t>(slot_bytes_));
        }
        auto& bits = warmed_[bank.index];
        if (bits.size() != files_[bank.index].experts.size()) {
            bits.assign(files_[bank.index].experts.size(), 0);
        }
        for (const auto value : experts) {
            if (value < 0 || static_cast<std::size_t>(value) >= bits.size()) { continue; }
            const auto expert = static_cast<std::size_t>(value);
            if (bits[expert] != 0) {
                ++stats_.warm_skips;
                continue;
            }
            const auto& entry = files_[bank.index].experts[expert];
            try {
                reader_->read_entry_cached(
                    entry, {warm_scratch_.data(), static_cast<std::size_t>(slot_bytes_)});
            } catch (const std::exception&) {
                // A read-ahead that could not read is not an error: the routed fill will read the
                // same bytes on demand and report the failure where it matters.
                continue;
            }
            bits[expert] = 1;
            stats_.warmed_bytes += entry.image_bytes;
        }
    }

    // Warmed experts that a routed fill has since loaded, since the last call. Same contract as
    // take_prefetch_hits(): a count of experts, turned into bytes by the consumer.
    [[nodiscard]] std::uint64_t take_warm_hits() {
        std::lock_guard<std::mutex> guard(mutex_);
        const auto taken = warm_hits_;
        warm_hits_       = 0;
        return taken;
    }

    void fill_locked(ExpertBankId bank, std::span<const std::int32_t> experts,
                     const std::int32_t* resident_mask, bool prefetch, const float* scores) {
        // A bank served straight out of the artifact mapping is backed by file pages the OS may
        // evict at any time, so it is a cold tier wearing a warm tier's address. A managed pinned
        // copy is therefore still worth keeping, and NINFER_ARENA_MANAGED asks for one even here:
        // that is what makes the warm tier a tier we control instead of a page cache we do not.
        // port (sm_89): F1 bisect, step 2. A bank addressed in place is built with file-based
        // offsets and a parent pointing at the artifact; turning the managed copy on makes ensure()
        // rewrite both to arena-based ones on the same object. NINFER_ARENA_SKIP_DIRECT restores the
        // old early return for exactly those banks, so if the crash disappears the mixing of the two
        // addressing schemes is the fault, and not the arena fill itself.
        static const bool skip_direct = std::getenv("NINFER_ARENA_SKIP_DIRECT") != nullptr;
        if ((skip_direct || !managed_even_in_place()) && bank.index < direct_.size() &&
            direct_[bank.index]) {
            return;
        }
        auto& s = state_[bank.index];
        // port (sm_89): a call that wants more experts than its pool has slots would evict the
        // slots it just filled, and the offsets handed back would name another expert's bytes. That
        // is silent corruption, not a slowdown, so it fails here instead. A 1-slot arena turned a
        // split PPL of 7.766 into 1131 before this check existed.
        const auto capacity = prefetch ? prefetch_slots_ : routed_capacity_;
        if (prefetch && prefetch_slots_ == 0) { return; } // read-ahead off: no pool to fill
        if (experts.size() > capacity) {
            throw std::runtime_error("host expert arena: one layer needs " +
                                     std::to_string(experts.size()) + " slots but the " +
                                     (prefetch ? "read-ahead pool holds " : "arena holds ") +
                                     std::to_string(capacity) +
                                     "; raise NINFER_EXPERT_HOST_ARENA_MB");
        }
        for (const auto value : experts) {
            if (resident_mask != nullptr && resident_mask[value] == 0) { continue; }
            const auto expert = static_cast<std::uint32_t>(value);
            auto& slot        = s.slot_of[expert];
            if (slot == kNone) {
                ++stats_.misses;
                slot = claim(bank.index, expert, prefetch, scores);
                load(bank.index, expert, slot);
                // The mark lives with the slot, so an eviction clears it: a slot reused by a routed
                // expert is not "a prefetched expert that was used".
                prefetched_[slot] = prefetch ? 1 : 0;
            } else {
                ++stats_.hits;
                if (prefetched_[slot] != 0) {
                    prefetched_[slot] = 0;
                    ++prefetch_hits_;
                }
                // port (sm_89): Q53. A route that lands in the second-chance band is still a hit - the bytes
                // are in RAM - and if the routed range has room it earns its way back: that is the
                // "按概率分区" half, admission by repeated demand instead of by first contact.
                if (second_chance_slots_ != 0 && slot >= routed_capacity_ &&
                    slot < routed_capacity_ + second_chance_slots_) {
                    ++stats_.second_chance_hits;
                    slot = promote_second_chance(bank, expert, slot);
                }
            }
            slot_last_use_[slot] = ++epoch_;
            const auto base      = std::uint64_t{slot} * slot_bytes_;
            s.gate_up_offsets[expert] = base + banks_[bank.index].entry.codes.offset;
            s.down_offsets[expert]    = base + banks_[bank.index].entry.down_codes.offset;
        }
    }

    [[nodiscard]] const ops::CpuSwiGLUExperts& experts(ExpertBankId bank) const {
        return state_[bank.index].view;
    }

    // port (sm_89): the Device uploader takes a pinned copy from here instead of reading the
    // pageable artifact mapping again. Only banks backed by the arena can answer: a bank addressed
    // in place has no arena slot, and answering anyway would hand back bytes nobody wrote.
    const std::byte* arena_image(std::uint32_t bank, std::uint32_t expert,
                                 std::uint64_t& bytes) const override {
        bytes = 0;
        std::lock_guard<std::mutex> guard(mutex_);
        // port (sm_89): F1 bisect. With NINFER_ARENA_NO_READ the tier is still filled and its offsets
        // rewritten, but the uploader is told nothing is there and reads the artifact as before.
        // Crash gone => the fault is on the read side (the pointer, its lifetime, or when the copy
        // lands); crash stays => the fill side. Two runs instead of another guess.
        static const bool no_read = std::getenv("NINFER_ARENA_NO_READ") != nullptr;
        if (no_read) { return nullptr; }
        if (bank >= state_.size()) { return nullptr; }
        if (!managed_even_in_place() && bank < direct_.size() && direct_[bank]) {
            return nullptr;
        }
        const auto& s = state_[bank];
        if (expert >= s.slot_of.size()) { return nullptr; }
        const auto slot = s.slot_of[expert];
        if (slot == kNone) { return nullptr; }
        if (arena_ == nullptr) { return nullptr; }
        if (bank >= files_.size()) { return nullptr; }
        const auto& bank_files = files_[bank];
        if (expert >= bank_files.experts.size()) { return nullptr; }
        const auto image = bank_files.experts[expert].image_bytes;
        if (image == 0 || image > slot_bytes_) { return nullptr; }
        bytes = image;
        // Handing out the pointer is when the copy can start: from here until uploads_settled() this
        // slot must not be evicted.
        in_flight_[slot] = 1;
        return arena_ + std::uint64_t{slot} * slot_bytes_;
    }

    // port (sm_89): the copies the uploader handed out have landed (called where the staging events
    // are waited). Clearing them all is safe by construction: settle means none is reading any more.
    void uploads_settled() override {
        std::lock_guard<std::mutex> guard(mutex_);
        std::fill(in_flight_.begin(), in_flight_.end(), std::uint8_t{0});
    }
    [[nodiscard]] std::uint64_t slot_bytes() const noexcept { return slot_bytes_; }
    [[nodiscard]] std::uint32_t slot_capacity() const noexcept { return slot_capacity_; }
    // port (sm_89): the tier as the Device sees it, or null when the driver would not map the arena
    // (pageable fallback). This is Q20's second addressing leg: a route the pool cannot hold gets an
    // address here instead of an upload, so the layer itself needs no host round trip. Callers must
    // treat the bytes at an offset as valid only for a slot `claim()` currently names - the arena
    // moves slots, the address space does not.
    [[nodiscard]] const std::byte* arena_device_base() const noexcept { return arena_device_; }

    // port (sm_89): Q20 second leg, host side. Three things the Device publisher needs, all read
    // under the same lock the fill path takes:
    //
    //   plane_offsets  - where each plane starts inside a slot, so the Device bases are
    //                    arena_device_base() + offset and the per-expert table is just slot*bytes
    //   peek           - which slot an expert occupies RIGHT NOW, without filling it and without
    //                    marking it in flight. -1 means "not in RAM", and a caller that reads it
    //                    anyway would get another expert's bytes
    //   hold/release   - keep a slot from being evicted while the Device is reading it. The layer
    //                    reads asynchronously, so a claim() in the next layer can otherwise move the
    //                    bytes out from under a GEMM that has not run yet.
    [[nodiscard]] bool plane_offsets(std::uint32_t bank, std::uint64_t gate_up[3],
                                     std::uint64_t down[3], std::uint64_t& gate_up_high_bytes,
                                     std::uint64_t& down_high_bytes) const {
        std::lock_guard<std::mutex> guard(mutex_);
        if (arena_device_ == nullptr || bank >= banks_.size()) { return false; }
        const auto& e      = banks_[bank].entry;
        gate_up[0]         = e.codes.offset;
        gate_up[1]         = e.high.offset;
        gate_up[2]         = e.scales.offset;
        down[0]            = e.down_codes.offset;
        down[1]            = e.down_high.offset;
        down[2]            = e.down_scales.offset;
        gate_up_high_bytes = e.high.bytes;
        down_high_bytes    = e.down_high.bytes;
        return true;
    }
    [[nodiscard]] std::int64_t peek_slot(std::uint32_t bank, std::uint32_t expert) const {
        std::lock_guard<std::mutex> guard(mutex_);
        if (bank >= state_.size() || expert >= state_[bank].slot_of.size()) { return -1; }
        const auto slot = state_[bank].slot_of[expert];
        return slot == kNone ? -1 : static_cast<std::int64_t>(slot);
    }
    void hold_for_device(std::uint32_t bank, std::uint32_t expert) {
        std::lock_guard<std::mutex> guard(mutex_);
        if (bank >= state_.size() || expert >= state_[bank].slot_of.size()) { return; }
        const auto slot = state_[bank].slot_of[expert];
        if (slot != kNone && slot < device_held_.size()) { device_held_[slot] = 1; }
    }
    // port (sm_89): GH3. The publisher needs "where does the tier hold each expert" as one answer,
    // not 256 of them. peek_slot() per expert took the lock 256 times and could observe the tier
    // before one load and after the next, so a table built that way was never guaranteed to have
    // been true at any single instant. This returns one consistent snapshot instead, and the publisher
    // builds its placement word from it.
    //
    // kAbsentSlot means "not in RAM". A caller that names a slot in a Device table must keep the
    // entry in `held` for as long as that table is live: the bytes at a slot are only that expert's
    // until somebody else claims it.
    static constexpr std::uint32_t kAbsentSlot = 0xffffffffu;
    void snapshot_slots(std::uint32_t bank, std::span<std::uint32_t> slots) const {
        std::lock_guard<std::mutex> guard(mutex_);
        std::fill(slots.begin(), slots.end(), kAbsentSlot);
        if (bank >= state_.size()) { return; }
        const auto& state = state_[bank];
        const auto count  = std::min(slots.size(), state.slot_of.size());
        for (std::size_t expert = 0; expert < count; ++expert) {
            slots[expert] = state.slot_of[expert];
        }
    }
    // Hold a whole batch of slots under one lock. Used with the snapshot above: exactly the entries a
    // published table names are held, so the protection ends when the table does.
    void hold_slots(std::span<const std::uint32_t> slots) {
        std::lock_guard<std::mutex> guard(mutex_);
        for (const auto slot : slots) {
            if (slot < device_held_.size()) { device_held_[slot] = 1; }
        }
    }
    // port (sm_89): two generations, not one. The Device reads asynchronously, and the next layer's
    // fill happens before the previous layer's GEMM has necessarily finished, so a single set cleared
    // at layer start would release exactly the slots that are still being read. Rotating means a slot
    // is protected for the layer that named it and the one after, and released only once a later
    // layer has waited on the stream.
    void begin_layer() {
        std::lock_guard<std::mutex> guard(mutex_);
        std::swap(device_held_, device_held_prev_);
        std::fill(device_held_.begin(), device_held_.end(), std::uint8_t{0});
    }
    void release_device_holds() {
        std::lock_guard<std::mutex> guard(mutex_);
        std::fill(device_held_.begin(), device_held_.end(), std::uint8_t{0});
    }
    // The two pools inside the arena (see claim()).
    [[nodiscard]] std::uint32_t routed_capacity() const noexcept { return routed_capacity_; }
    [[nodiscard]] std::uint32_t prefetch_slots() const noexcept { return prefetch_slots_; }
    [[nodiscard]] std::uint32_t bank_count() const noexcept {
        return static_cast<std::uint32_t>(state_.size());
    }
    // Q20 static candidates: bumped whenever any slot->expert mapping moves. A Device-side candidate
    // table is only valid for the generation it was copied at, so the ring path checks this before
    // it names a host slot.
    [[nodiscard]] std::uint64_t slot_generation() const noexcept { return slot_generation_; }

    // port (sm_89): what the arena actually did. A split run is only cheap when `miss` stops
    // growing: a miss is an artifact read, a hit is a byte range that was already in RAM. Reported
    // from the split timing line so "experts are not resident, so there is no cache" is a number
    // rather than an inference.
    struct Stats {
        std::uint64_t hits      = 0;
        std::uint64_t misses    = 0;
        std::uint64_t evictions = 0;
        // Q69: what the page-cache target did. `warmed_bytes` is what the read-ahead pulled through
        // the buffered handle without claiming an arena slot; `memcpy_hits` is how many of those
        // experts were later actually loaded, i.e. how much of the warming paid. Both are counters,
        // not judgements: the queue's usefulness gate decides.
        std::uint64_t warmed_bytes = 0;
        std::uint64_t memcpy_hits  = 0;
        std::uint64_t warm_skips   = 0; // asked to warm an expert that is already warm
        // Q53: what the second-chance band did. `second_chance_admissions` are evictions that became a copy instead
        // of a drop, `second_chance_hits` are routes answered from it, `second_chance_promotions` are the ones that
        // earned their way back into the routed range.
        std::uint64_t second_chance_admissions = 0;
        std::uint64_t second_chance_declines   = 0; // evictions the ranking refused to keep
        std::uint64_t second_chance_hits       = 0;
        std::uint64_t second_chance_promotions = 0;
    };
    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }

private:
    static constexpr std::uint32_t kNone = 0xffffffffu;

    // Rewrite the plane offsets into the packed form the CPU op validates for host banks.
    static WeightBankLayout with_packed_planes(const WeightBankLayout& layout) {
        WeightBankLayout out = layout;
        out.codes.offset     = 0;
        out.high.offset      = out.codes.expert_bytes;
        out.scales.offset    = out.codes.expert_bytes + out.high.expert_bytes;
        return out;
    }

    static WeightBankLayout with_zero_planes(const WeightBankLayout& layout) {
        WeightBankLayout out = layout;
        out.codes.offset     = 0;
        out.high.offset      = 0;
        out.scales.offset    = 0;
        return out;
    }

    struct State {
        std::vector<std::uint32_t> slot_of;
        std::vector<std::uint64_t> gate_up_offsets;
        std::vector<std::uint64_t> down_offsets;
        std::vector<std::uint64_t> gu_codes, gu_high, gu_scales;
        std::vector<std::uint64_t> dn_codes, dn_high, dn_scales;
        WeightParent gate_up_parent;
        WeightParent down_parent;
        WeightBankView gate_up;
        WeightBankView down;
        ops::CpuSwiGLUExperts view;
    };

    std::byte* arena() {
        if (arena_ == nullptr) {
            // port (sm_89): the arena is the source of every Device upload and the memory the CPU
            // kernels stream through, so it is page-locked when the driver allows it. Two measured
            // reasons: a batch of pinned slices submits 3.5-16x cheaper than the same slices from
            // pageable memory, and a pageable working set gets trimmed, which cost the split path
            // 0.43 ms per expert against 0.041 ms on warm pages. Pageable reservation stays as the
            // fallback, because a locked allocation can fail on a busy box.
            void* locked = nullptr;
            // port (sm_89): mapped as well as locked. Q20 needs the Device to address this tier
            // directly - that is what makes a route the pool cannot hold resolvable inside the layer
            // without a host upload, and therefore what `capt_free` is a measurement of. Mapping does
            // not change the upload path: `arena_image()` still hands the host address to the
            // uploader, and the Device address is only read by callers that ask for it.
            if (cudaHostAlloc(&locked, arena_bytes_, cudaHostAllocMapped) == cudaSuccess) {
                arena_  = static_cast<std::byte*>(locked);
                pinned_ = true;
                void* device = nullptr;
                if (cudaHostGetDevicePointer(&device, locked, 0) == cudaSuccess) {
                    arena_device_ = static_cast<std::byte*>(device);
                }
                // One line, once: whether the Device can address this tier at all is the premise
                // Q20 rests on, and a silent fallback to "not mapped" would turn the second
                // addressing leg into a host round trip that no counter would blame.
                cudaPointerAttributes attributes{};
                const auto known = arena_device_ != nullptr &&
                                   cudaPointerGetAttributes(&attributes, arena_device_) == cudaSuccess;
                std::fprintf(stderr,
                             "[host] arena mapped bytes=%llu host=%p dev=%p device_accessible=%d\n",
                             static_cast<unsigned long long>(arena_bytes_),
                             static_cast<const void*>(arena_), static_cast<const void*>(arena_device_),
                             known ? 1 : 0);
                std::fflush(stderr);
                return arena_;
            }
            if (cudaHostAlloc(&locked, arena_bytes_, cudaHostAllocDefault) == cudaSuccess) {
                arena_  = static_cast<std::byte*>(locked);
                pinned_ = true;
                return arena_;
            }
#if defined(_WIN32)
            arena_ = static_cast<std::byte*>(
                VirtualAlloc(nullptr, arena_bytes_, MEM_RESERVE, PAGE_NOACCESS));
            if (arena_ == nullptr) {
                throw std::runtime_error("host expert arena: reserve failed");
            }
#endif
        }
        return arena_;
    }

    void commit(std::uint32_t slot) {
        if (pinned_) { return; } // a locked arena is fully committed at allocation
#if defined(_WIN32)
        if (VirtualAlloc(arena_ + std::uint64_t{slot} * slot_bytes_, slot_bytes_, MEM_COMMIT,
                         PAGE_READWRITE) == nullptr) {
            throw std::runtime_error("host expert arena: commit failed");
        }
#else
        (void)slot;
#endif
    }

    // port (sm_89): three ranges, one arena - routed [0, routed_capacity_), second-chance
    // [routed, second_chance_end), read-ahead [second_chance_end, slot_capacity_). Each range has
    // its own budget, so nothing can evict what another range just filled and no auxiliary range can
    // take a routed slot.
    std::uint32_t claim(std::uint32_t bank, std::uint32_t expert, bool prefetch,
                        const float* scores) {
        const std::uint32_t second_chance_end = routed_capacity_ + second_chance_slots_;
        const std::uint32_t begin    = prefetch ? second_chance_end : 0;
        const std::uint32_t end      = prefetch ? slot_capacity_ : routed_capacity_;
        std::uint32_t& used       = prefetch ? used_prefetch_ : used_;
        if (used < end - begin) {
            const auto slot    = begin + used++;
            slot_bank_[slot]   = bank;
            slot_expert_[slot] = expert;
            ++slot_generation_;
            return slot;
        }
        std::uint32_t victim  = kNone;
        std::uint64_t coldest = ~std::uint64_t{0};
        for (std::uint32_t s = begin; s < end; ++s) {
            // A slot a copy is reading is not a candidate, whatever its age. Evicting it would hand
            // the Device another expert's bytes with no error anywhere - the crash this protection
            // exists to prevent (a read-ahead issued after wait_for_progress() freed the slot an
            // in-flight copy was reading).
            if (in_flight_[s] != 0) { continue; }
            // A slot the Device is reading is not a candidate either, and for the same reason: the
            // GEMM runs asynchronously, so claim() in the next layer can free a slot whose bytes are
            // still being consumed. Released once per layer, after the stream has been waited on.
            if (device_held_[s] != 0 || device_held_prev_[s] != 0) { continue; }
            if (slot_last_use_[s] < coldest) {
                coldest = slot_last_use_[s];
                victim  = s;
            }
        }
        if (victim == kNone) {
            throw std::runtime_error("host expert arena: every slot is in flight");
        }
        if (coldest == 0) { throw std::runtime_error("host expert arena: nothing to evict"); }
        ++stats_.evictions;
        // port (sm_89): Q53. The victim's bytes are still the expert that owned them, and that
        // expert was hot a moment ago - so it moves to the warm tier's second-chance band (a memcpy
        // inside the same pinned arena) instead of leaving the arena. The band has its own budget;
        // it never takes a routed slot. When the band is off this is the old behaviour exactly: the
        // mapping is cleared and the slot is handed to the newcomer.
        if (!demote_second_chance(victim, scores)) {
            state_[slot_bank_[victim]].slot_of[slot_expert_[victim]] = kNone;
        }
        prefetched_[victim]  = 0; // the slot now holds another expert, so the mark dies with it
        slot_bank_[victim]   = bank;
        slot_expert_[victim] = expert;
        ++slot_generation_;
        return victim;
    }

    // port (sm_89): Q53, the two movements of the second-chance band. `demote_second_chance` runs inside
    // claim(), on a slot that has already been chosen as a victim; `promote_second_chance` runs on a band hit
    // while the routed range still has room. Both are copies inside the arena - no artifact read,
    // no upload - and neither changes the band's budget: a promoted slot goes to `second_chance_free_`.
    //
    // Q53 second take: admission is a comparison, not "eviction means second chance". Measured with the
    // previous rule, the band was a net loss (+33% ens, -29% tps): it kept every evicted expert,
    // including the ones the ranking had just decided were not worth a routed slot, and the copy
    // displaced an expert that had been asked for more recently. The region now admits an evicted
    // expert only when its decayed score beats the coldest score already in the region; otherwise
    // the eviction is a drop, exactly as it was before the region existed. With `scores == nullptr`
    // (a caller that does not rank) the old mechanical rule is the fallback.
    bool demote_second_chance(std::uint32_t slot, const float* scores) {
        if (second_chance_slots_ == 0 || slot >= routed_capacity_) { return false; }
        const auto bank   = slot_bank_[slot];
        const auto expert = slot_expert_[slot];
        if (bank == kNone || expert == kNone) { return false; }
        const auto incoming = scores != nullptr && expert < banks_[bank].entry.count
                                  ? scores[expert]
                                  : 0.0F;
        std::uint32_t target = kNone;
        if (!second_chance_free_.empty()) {
            target = second_chance_free_.back();
            second_chance_free_.pop_back();
        } else if (second_chance_used_ < second_chance_slots_) {
            target = routed_capacity_ + second_chance_used_++;
        } else {
            float coldest = std::numeric_limits<float>::max();
            for (std::uint32_t candidate = routed_capacity_;
                 candidate < routed_capacity_ + second_chance_slots_; ++candidate) {
                if (slot_bank_[candidate] == kNone || in_flight_[candidate] != 0 ||
                    device_held_[candidate] != 0 || device_held_prev_[candidate] != 0) {
                    continue;
                }
                const auto other_bank   = slot_bank_[candidate];
                const auto other_expert = slot_expert_[candidate];
                const auto value = other_bank < banks_.size() &&
                                           other_expert < banks_[other_bank].entry.count
                                       ? score_of(other_bank, other_expert)
                                       : 0.0F;
                if (value < coldest) {
                    coldest = value;
                    target  = candidate;
                }
            }
            if (target == kNone) { return false; }
            // Strictly hotter. A tie is not evidence that the newcomer is worth displacing what is
            // already there, and the score is decayed, so an unused expert sinks on its own.
            if (scores != nullptr && incoming <= coldest) {
                ++stats_.second_chance_declines;
                return false;
            }
        }
        // Whatever the target held stops being addressable: that slot is about to hold another
        // expert, and a stale mapping would hand a route the wrong bytes - the failure this tier has
        // already paid for once.
        if (const auto previous = slot_bank_[target]; previous != kNone && previous < state_.size()) {
            auto& other = state_[previous].slot_of[slot_expert_[target]];
            if (other == target) { other = kNone; }
        }
        auto* base = arena();
        std::memcpy(base + std::uint64_t{target} * slot_bytes_,
                    base + std::uint64_t{slot} * slot_bytes_,
                    static_cast<std::size_t>(slot_bytes_));
        slot_bank_[target]     = bank;
        slot_expert_[target]   = expert;
        slot_last_use_[target] = slot_last_use_[slot];
        prefetched_[target]    = 0;
        state_[bank].slot_of[expert] = target;
        ++slot_generation_;
        ++stats_.second_chance_admissions;
        return true;
    }

    // The score of an expert whose bank is not the one being filled. The second-chance band is shared, so
    // an admission may compare this bank's evictee against another bank's occupant; the score
    // vectors are per bank and owned by the streams, which keep them alive for the process.
    [[nodiscard]] float score_of(std::uint32_t bank, std::uint32_t expert) const noexcept {
        constexpr std::uint32_t kMaxBanks = 64;
        if (bank >= kMaxBanks || score_bank_[bank] == nullptr) { return 0.0F; }
        if (expert >= score_count_[bank]) { return 0.0F; }
        return score_bank_[bank][expert];
    }
public:
    // Each stream registers its own score vector once, at attach. Borrowed, not owned.
    void attach_scores(std::uint32_t bank, const float* scores, std::uint32_t count) {
        constexpr std::uint32_t kMaxBanks = 64;
        if (bank >= kMaxBanks) { return; }
        std::lock_guard<std::mutex> guard(mutex_);
        score_bank_[bank]  = scores;
        score_count_[bank] = count;
    }
private:

    std::uint32_t promote_second_chance(ExpertBankId bank, std::uint32_t expert, std::uint32_t from) {
        if (slot_bank_.empty() || used_ >= routed_capacity_) { return from; }
        const auto target = used_++;
        auto* base        = arena();
        std::memcpy(base + std::uint64_t{target} * slot_bytes_,
                    base + std::uint64_t{from} * slot_bytes_,
                    static_cast<std::size_t>(slot_bytes_));
        slot_bank_[target]     = bank.index;
        slot_expert_[target]   = expert;
        slot_last_use_[target] = slot_last_use_[from];
        prefetched_[target]    = 0;
        slot_bank_[from]       = kNone;
        slot_expert_[from]     = kNone;
        second_chance_free_.push_back(from);
        state_[bank.index].slot_of[expert] = target;
        ++slot_generation_;
        ++stats_.second_chance_promotions;
        return target;
    }

    void load(std::uint32_t bank, std::uint32_t expert, std::uint32_t slot) {
        auto* base = arena();
        commit(slot);
        auto* destination = base + std::uint64_t{slot} * slot_bytes_;
        const auto& entry = files_[bank].experts[expert];
        if (std::getenv("NINFER_HOST_DEBUG") && expert < 2) {
            std::fprintf(stderr, "[host-chunk] bank=%u expert=%u n=%zu", bank, expert,
                         entry.chunks.size());
            for (std::size_t i = 0; i < entry.chunks.size() && i < 6; ++i) {
                std::fprintf(stderr, " [%zu:file=%u off=%llu bytes=%llu image=%llu]", i,
                             entry.chunks[i].file_index,
                             static_cast<unsigned long long>(entry.chunks[i].file_offset),
                             static_cast<unsigned long long>(entry.chunks[i].bytes),
                             static_cast<unsigned long long>(entry.chunks[i].image_offset));
            }
            std::fprintf(stderr, "\n");
        }
        // The Device path uploads through spans() and only falls back to read_entry; use the same
        // proven route so both tiers build identical slot images.
        std::vector<ExpertBlobSpan> slices;
        if (reader_->spans(entry, slices) && !slices.empty()) {
            for (const auto& slice : slices) {
                if (slice.image_offset > slot_bytes_ ||
                    slice.bytes > slot_bytes_ - slice.image_offset) {
                    throw std::runtime_error("host expert slice lies outside its slot");
                }
                std::memcpy(destination + slice.image_offset, slice.source,
                            static_cast<std::size_t>(slice.bytes));
            }
        } else {
            // port (sm_89): the arena is the first-level cache and the OS page cache is the second.
            // Reading around the cache made every miss a physical read (8.6 ms per 1.8 MiB slot),
            // which was the whole cost of a split decode step; through the cache the same miss is a
            // memcpy out of RAM (0.2 ms). Measured 14 ms per token of ensure against 560-830 ms.
            reader_->read_entry_cached(entry, {destination, static_cast<std::size_t>(slot_bytes_)});
        }
        // Q69: this fill is what the read-ahead was for. Either path above reads through the page
        // cache (spans() is an mmap memcpy, read_entry_cached is a buffered read), so an expert the
        // warming pass already touched costs no physical read here. Counted once: the bit is cleared
        // so a second fill of the same expert is not credited twice.
        if (bank < warmed_.size() && expert < warmed_[bank].size() && warmed_[bank][expert] != 0) {
            warmed_[bank][expert] = 0;
            // Two counters, one event: the cumulative one is what the panel prints, the pending one
            // is what the consumer drains into the usefulness gate. Keeping them apart is what lets
            // the panel show the total after the drain has zeroed the delta.
            ++stats_.memcpy_hits;
            ++warm_hits_;
        }
        // One arena base for every bank; the per-plane tables carry the slot origin.
        state_[bank].gate_up_parent.data = base;
        state_[bank].down_parent.data    = base;
        if (std::getenv("NINFER_HOST_VERIFY")) {
            static std::uint32_t verified = 0;
            if (verified++ < 4) {
                // One-shot check of the slot image: the same slices read back must match what was
                // copied. A mismatch means the description and the reader disagree about offsets,
                // which is invisible until a miss actually runs on the CPU.
                std::vector<ExpertBlobSpan> again;
                std::size_t compared = 0, different = 0;
                if (reader_->spans(entry, again)) {
                    for (const auto& slice : again) {
                        const auto* written = destination + slice.image_offset;
                        const auto* expected_source = static_cast<const std::byte*>(slice.source);
                        for (std::uint64_t i = 0; i < slice.bytes; ++i) {
                            ++compared;
                            if (written[i] != expected_source[i]) { ++different; }
                        }
                    }
                }
                std::fprintf(
                    stderr, "[host-verify] bank=%u expert=%u chunks=%zu slices=%zu diff=%zu\n", bank,
                    expert, entry.chunks.size(), again.size(),
                    static_cast<std::size_t>(different));
            }
        }
    }

    std::vector<HostExpertBankDesc> banks_;
    std::vector<ExpertFileBank> files_;
    const ExpertBlobReader* reader_ = nullptr;
    std::vector<State> state_;
    std::uint64_t slot_bytes_    = 0;
    std::uint64_t arena_bytes_   = 0;
    std::byte* arena_            = nullptr;
    // Device-visible alias of `arena_`, same bytes (cudaHostGetDevicePointer). Null on the pageable
    // fallback; see arena_device_base().
    std::byte* arena_device_     = nullptr;
    bool pinned_                 = false;
    std::uint32_t slot_capacity_ = 0;
    // Slots [0, routed_capacity_) belong to the layers, [routed_capacity_, slot_capacity_) to the
    // read-ahead. `used_` counts the first range's bump allocation, `used_prefetch_` the second's.
    std::uint32_t used_            = 0;
    std::uint32_t routed_capacity_ = 0;
    std::uint32_t prefetch_slots_  = 0;
    std::uint32_t used_prefetch_   = 0;
    // Q53: the second-chance band sits between the routed range and the read-ahead's, and `second_chance_used_` is
    // how many of its slots have ever been filled - the region is a fixed-size set, and admission is
    // only a comparison once it is full.
    std::uint32_t second_chance_slots_ = 0;
    std::uint32_t second_chance_used_  = 0;
    std::uint64_t slot_generation_     = 0;
    // Slots a promotion handed back. Reused before the bump pointer so a promotion does not shrink
    // the region by one slot forever.
    std::vector<std::uint32_t> second_chance_free_;
    // Q53: each bank's decayed route scores, registered by its stream. Borrowed pointers; see
    // score_of(). kMaxBanks is the same bound the bank ids already carry.
    std::array<const float*, 64> score_bank_{};
    std::array<std::uint32_t, 64> score_count_{};
    std::vector<bool> direct_; // bank addressed in place, no arena copy
    // When set, an in-place bank still gets a managed pinned copy. Off by default so a run that
    // never asks for the warm tier behaves exactly as before.
    //
    // NINFER_HOST_TIER implies this: asking for the warm tier means asking for a tier we control
    // rather than a page cache we do not. Without the pairing the consumer call was inert - every
    // bank on this artifact resolves in place, so ensure() returned before touching the arena and the
    // counter read hits=0 misses=0 - which is exactly how "the tier is never filled" survived a
    // switch that looked connected.
    static bool managed_even_in_place() {
        // port (sm_89): 2026-10-31 - NOT bound to NINFER_HOST_TIER any more.
        //
        // Binding them made the switch reach an arena-backed path on an artifact whose every bank is
        // addressed in place, and a decode request with the switch on then failed to return (no
        // exception, connection reset, process alive). Prefill was fine, which is why it looked
        // verified: perplexity is a prefill path.
        //
        // Until that failure is diagnosed, the switch stays inert on this artifact - a direct_ bank
        // returns before touching the arena - rather than reachable and broken. The managed copy is
        // opt-in again through NINFER_ARENA_MANAGED, which is what it was before, and the decode
        // failure is recorded with it.
        static const bool value = std::getenv("NINFER_ARENA_MANAGED") != nullptr;
        return value;
    }
    std::vector<std::uint32_t> slot_bank_, slot_expert_;
    std::vector<std::uint64_t> slot_last_use_;
    // mutable: arena_image() is const (it is the uploader's view) but handing out a pointer is what
    // marks the slot in flight.
    mutable std::vector<std::uint8_t> in_flight_;
    // Slots the Device is addressing this layer (Q20 second leg). Kept apart from in_flight_: that
    // one covers a copy the uploader is reading, this one covers a GEMM that has been queued but may
    // not have run. Both make a slot un-evictable; they end at different times.
    mutable std::vector<std::uint8_t> device_held_;
    mutable std::vector<std::uint8_t> device_held_prev_;
    // Slots the read-ahead filled and a routed layer has not consumed yet (D4's numerator).
    std::vector<std::uint8_t> prefetched_;
    std::uint64_t prefetch_hits_ = 0;
    // Q69: experts whose bytes the read-ahead pulled through the page cache, per bank, and the one
    // scratch slot that pass reads into. Guarded by the same mutex as the slot tables.
    std::vector<std::vector<std::uint8_t>> warmed_;
    std::vector<std::byte> warm_scratch_;
    std::uint64_t warm_hits_ = 0;
    // Serialises ensure() with the uploader's arena_image()/uploads_settled(): the first mutates slot
    // tables, the others read them, and D1 runs the fill on its own thread.
    mutable std::mutex mutex_;
    std::uint64_t epoch_ = 0;
    Stats stats_{};
};

} // namespace ninfer::models::qwen3_5::execution



