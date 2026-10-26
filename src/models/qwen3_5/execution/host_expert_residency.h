#pragma once

#include "core/expert_cache.h"
#include "core/expert_file.h"
#include "core/weight_bank.h"
#include "ninfer/ops/moe.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
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
                        std::uint64_t arena_bytes)
        : banks_(banks.begin(), banks.end()), files_(files.begin(), files.end()), reader_(reader),
          arena_bytes_(arena_bytes) {
        std::uint64_t widest = 0;
        for (const auto& bank : banks_) {
            widest = std::max(widest, bank.entry.entry_bytes);
        }
        constexpr std::uint64_t kBlock = 128 * 1024;
        slot_bytes_    = (widest + kBlock - 1) / kBlock * kBlock;
        arena_bytes_   = std::max(arena_bytes_, slot_bytes_);
        slot_capacity_ = static_cast<std::uint32_t>(arena_bytes_ / slot_bytes_);
        slot_bank_.assign(slot_capacity_, kNone);
        slot_expert_.assign(slot_capacity_, kNone);
        slot_last_use_.assign(slot_capacity_, 0);

        state_.resize(banks_.size());
        direct_.assign(banks_.size(), false);
        for (std::size_t i = 0; i < banks_.size(); ++i) {
            auto& s = state_[i];
            s.slot_of.assign(banks_[i].entry.count, kNone);
            s.gate_up_offsets.assign(banks_[i].entry.count, 0);
            s.down_offsets.assign(banks_[i].entry.count, 0);

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
            static const bool allow_direct = std::getenv("NINFER_HOST_COPY") == nullptr;
            // port (sm_89): the fused in-place branch is off. A fused bank keeps codes, high and
            // scales in three plane runs inside one object, and the addressing below assumes each
            // expert sits at expert * expert_stride inside a plane whose offset is already the
            // object-relative one. On the 35B fused artifact that is not what the stored layout
            // says (gate_up geometry 285,212,672 = 256 x 1 MiB codes + 256 x 64 KiB high, but the
            // plane offsets do not land there), so the branch read the wrong plane and the split
            // PPL came out as 159,099 against 7.776858 on the copying path. Keep the copying path
            // until the plane offsets are resolved against the object; NINFER_HOST_FUSED_DIRECT=1
            // re-enables the branch for whoever finishes that check.
            static const bool allow_fused_direct =
                std::getenv("NINFER_HOST_FUSED_DIRECT") != nullptr;
            if (allow_fused_direct && allow_direct && gate_up_base != nullptr &&
                down_base != nullptr && banks_[i].entry.count > 0) {
                if (std::getenv("NINFER_HOST_DEBUG") != nullptr && i < 2) {
                    const auto& gl = banks_[i].gate_up_layout;
                    const auto& dl = banks_[i].down_layout;
                    std::fprintf(stderr,
                                 "[direct-layout] gu off=%llu stride=%llu plane=%llu | dn off=%llu "
                                 "stride=%llu plane=%llu | entry=%llu bytes=%llu/%llu geom=%llu/%llu\n",
                                 (unsigned long long)gl.codes.offset,
                                 (unsigned long long)gl.codes.expert_stride,
                                 (unsigned long long)gl.codes.expert_bytes,
                                 (unsigned long long)dl.codes.offset,
                                 (unsigned long long)dl.codes.expert_stride,
                                 (unsigned long long)dl.codes.expert_bytes,
                                 (unsigned long long)banks_[i].entry.entry_bytes,
                                 (unsigned long long)banks_[i].entry.codes.offset,
                                 (unsigned long long)banks_[i].entry.down_codes.offset,
                                 (unsigned long long)banks_[i].gate_up_geometry.bytes,
                                 (unsigned long long)banks_[i].down_geometry.bytes);
                }
                direct_[i] = true;
                s.gate_up_parent.geometry = banks_[i].gate_up_geometry;
                s.gate_up_parent.device = Device::cpu();
                s.gate_up_parent.data   = gate_up_base;
                s.down_parent.geometry = banks_[i].down_geometry;
                s.down_parent.device = Device::cpu();
                s.down_parent.data   = down_base;
                s.gate_up.parent = &s.gate_up_parent;
                s.gate_up.layout = banks_[i].gate_up_layout; // plane offsets stay as stored
                s.gate_up.expert_offsets = nullptr;         // expert * stride
                s.down.parent = &s.down_parent;
                s.down.layout = banks_[i].down_layout;
                s.down.expert_offsets = nullptr;
                s.view.gate_up = s.gate_up;
                s.view.down    = s.down;
                continue;
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

    void ensure(ExpertBankId bank, std::span<const std::int32_t> experts,
                const std::int32_t* resident_mask) override {
        // Addressed in place: the weights are already where the kernels read them.
        if (bank.index < direct_.size() && direct_[bank.index]) { return; }
        auto& s = state_[bank.index];
        // port (sm_89): a call that wants more experts than the arena has slots would evict the
        // slots it just filled, and the offsets handed back would name another expert's bytes. That
        // is silent corruption, not a slowdown, so it fails here instead. A 1-slot arena turned a
        // split PPL of 7.766 into 1131 before this check existed.
        if (experts.size() > slot_capacity_) {
            throw std::runtime_error("host expert arena: one layer needs " +
                                     std::to_string(experts.size()) + " slots but the arena holds " +
                                     std::to_string(slot_capacity_) +
                                     "; raise NINFER_EXPERT_HOST_ARENA_MB");
        }
        for (const auto value : experts) {
            if (resident_mask != nullptr && resident_mask[value] == 0) { continue; }
            const auto expert = static_cast<std::uint32_t>(value);
            auto& slot        = s.slot_of[expert];
            if (slot == kNone) {
                ++stats_.misses;
                slot = claim(bank.index, expert);
                load(bank.index, expert, slot);
            } else {
                ++stats_.hits;
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
        if (bank >= state_.size()) { return nullptr; }
        if (bank < direct_.size() && direct_[bank]) { return nullptr; }
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
        return arena_ + std::uint64_t{slot} * slot_bytes_;
    }
    [[nodiscard]] std::uint64_t slot_bytes() const noexcept { return slot_bytes_; }
    [[nodiscard]] std::uint32_t slot_capacity() const noexcept { return slot_capacity_; }

    // port (sm_89): what the arena actually did. A split run is only cheap when `miss` stops
    // growing: a miss is an artifact read, a hit is a byte range that was already in RAM. Reported
    // from the split timing line so "experts are not resident, so there is no cache" is a number
    // rather than an inference.
    struct Stats {
        std::uint64_t hits      = 0;
        std::uint64_t misses    = 0;
        std::uint64_t evictions = 0;
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

    struct State {
        std::vector<std::uint32_t> slot_of;
        std::vector<std::uint64_t> gate_up_offsets;
        std::vector<std::uint64_t> down_offsets;
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

    std::uint32_t claim(std::uint32_t bank, std::uint32_t expert) {
        if (used_ < slot_capacity_) {
            const auto slot    = used_++;
            slot_bank_[slot]   = bank;
            slot_expert_[slot] = expert;
            return slot;
        }
        std::uint32_t victim  = 0;
        std::uint64_t coldest = ~std::uint64_t{0};
        for (std::uint32_t s = 0; s < slot_capacity_; ++s) {
            if (slot_last_use_[s] < coldest) {
                coldest = slot_last_use_[s];
                victim  = s;
            }
        }
        if (coldest == 0) { throw std::runtime_error("host expert arena: nothing to evict"); }
        ++stats_.evictions;
        state_[slot_bank_[victim]].slot_of[slot_expert_[victim]] = kNone;
        slot_bank_[victim]   = bank;
        slot_expert_[victim] = expert;
        return victim;
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
    bool pinned_                 = false;
    std::uint32_t slot_capacity_ = 0;
    std::uint32_t used_          = 0;
    std::vector<bool> direct_; // bank addressed in place, no arena copy
    std::vector<std::uint32_t> slot_bank_, slot_expert_;
    std::vector<std::uint64_t> slot_last_use_;
    std::uint64_t epoch_ = 0;
    Stats stats_{};
};

} // namespace ninfer::models::qwen3_5::execution



