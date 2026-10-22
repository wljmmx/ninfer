#pragma once

#include "core/expert_cache.h"
#include "core/expert_file.h"
#include "core/weight_bank.h"
#include "ninfer/ops/moe.h"

#include <algorithm>
#include <cstdint>
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
};

// The host half of a split expert tier. It reserves one arena up front, commits it slot by slot,
// and holds one uniform slot per expert using the same entry image the Device pool stores. A slot
// is evicted coldest-first when the arena fills, so RAM holds what fits and everything else keeps
// streaming in from the artifact.
class HostExpertResidency final : public HostResidency {
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
        slot_bytes_   = (widest + kBlock - 1) / kBlock * kBlock;
        arena_bytes_  = std::max(arena_bytes_, slot_bytes_);
        slot_capacity_ = static_cast<std::uint32_t>(arena_bytes_ / slot_bytes_);
        slot_bank_.assign(slot_capacity_, kNone);
        slot_expert_.assign(slot_capacity_, kNone);
        slot_last_use_.assign(slot_capacity_, 0);

        state_.resize(banks_.size());
        for (std::size_t i = 0; i < banks_.size(); ++i) {
            auto& s = state_[i];
            s.slot_of.assign(banks_[i].entry.count, kNone);
            s.offsets.assign(banks_[i].entry.count, 0);

            // The CPU views address a slot, not the whole parent: plane offsets come from the
            // entry image, so the same bytes the Device pool holds line up here unchanged.
            s.gate_up_parent.geometry = banks_[i].gate_up_geometry;
            s.gate_up_parent.device   = Device::cpu();
            s.down_parent.geometry    = banks_[i].down_geometry;
            s.down_parent.device      = Device::cpu();

            s.gate_up.parent = &s.gate_up_parent;
            s.gate_up.layout = banks_[i].gate_up_layout;
            s.gate_up.layout.codes.offset  = banks_[i].entry.codes.offset;
            s.gate_up.layout.high.offset   = banks_[i].entry.high.offset;
            s.gate_up.layout.scales.offset = banks_[i].entry.scales.offset;
            s.gate_up.expert_offsets       = s.offsets.data();

            s.down.parent = &s.down_parent;
            s.down.layout = banks_[i].down_layout;
            s.down.layout.codes.offset  = banks_[i].entry.down_codes.offset;
            s.down.layout.high.offset   = banks_[i].entry.down_high.offset;
            s.down.layout.scales.offset = banks_[i].entry.down_scales.offset;
            s.down.expert_offsets       = s.offsets.data();

            s.view.gate_up = s.gate_up;
            s.view.down    = s.down;
        }
    }

    void ensure(ExpertBankId bank, std::span<const std::int32_t> experts,
                const std::int32_t* resident_mask) override {
        auto& s = state_[bank.index];
        for (const auto value : experts) {
            if (resident_mask != nullptr && resident_mask[value] == 0) { continue; }
            const auto expert = static_cast<std::uint32_t>(value);
            auto& slot        = s.slot_of[expert];
            if (slot == kNone) {
                slot = claim(bank.index, expert);
                load(bank.index, expert, slot);
            }
            slot_last_use_[slot] = ++epoch_;
            s.offsets[expert]    = std::uint64_t{slot} * slot_bytes_;
        }
    }

    [[nodiscard]] const ops::CpuSwiGLUExperts& experts(ExpertBankId bank) const {
        return state_[bank.index].view;
    }
    [[nodiscard]] std::uint64_t slot_bytes() const noexcept { return slot_bytes_; }
    [[nodiscard]] std::uint32_t slot_capacity() const noexcept { return slot_capacity_; }

private:
    static constexpr std::uint32_t kNone = 0xffffffffu;

    struct State {
        std::vector<std::uint32_t> slot_of;
        std::vector<std::uint64_t> offsets;
        WeightParent gate_up_parent;
        WeightParent down_parent;
        WeightBankView gate_up;
        WeightBankView down;
        ops::CpuSwiGLUExperts view;
    };

    std::byte* arena() {
        if (arena_ == nullptr) {
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
#if defined(_WIN32)
        if (VirtualAlloc(arena_ + std::uint64_t{slot} * slot_bytes_, slot_bytes_, MEM_COMMIT,
                         PAGE_READWRITE) == nullptr) {
            throw std::runtime_error("host expert arena: commit failed");
        }
#endif
    }

    std::uint32_t claim(std::uint32_t bank, std::uint32_t expert) {
        if (used_ < slot_capacity_) {
            const auto slot   = used_++;
            slot_bank_[slot]  = bank;
            slot_expert_[slot] = expert;
            return slot;
        }
        std::uint32_t victim = 0;
        std::uint64_t coldest = ~std::uint64_t{0};
        for (std::uint32_t s = 0; s < slot_capacity_; ++s) {
            if (slot_last_use_[s] < coldest) {
                coldest = slot_last_use_[s];
                victim  = s;
            }
        }
        if (coldest == 0) { throw std::runtime_error("host expert arena: nothing to evict"); }
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
        reader_->read_entry(entry, {destination, static_cast<std::size_t>(slot_bytes_)});
        // One arena base for every bank; a slot offset is what distinguishes them.
        state_[bank].gate_up_parent.data = base;
        state_[bank].down_parent.data    = base;
    }

    std::vector<HostExpertBankDesc> banks_;
    std::vector<ExpertFileBank> files_;
    const ExpertBlobReader* reader_ = nullptr;
    std::vector<State> state_;
    std::uint64_t slot_bytes_   = 0;
    std::uint64_t arena_bytes_  = 0;
    std::byte* arena_           = nullptr;
    std::uint32_t slot_capacity_ = 0;
    std::uint32_t used_          = 0;
    std::vector<std::uint32_t> slot_bank_, slot_expert_;
    std::vector<std::uint64_t> slot_last_use_;
    std::uint64_t epoch_ = 0;
};

} // namespace ninfer::models::qwen3_5::execution

