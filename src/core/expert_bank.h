#pragma once

#include "core/weight_view.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer {

enum class ExpertPlaneKind : std::uint8_t { Codes, High, Scales };

struct ExpertPlaneSpan {
    const std::byte* data = nullptr;
    std::size_t bytes     = 0;
    Device device;
};

// port (sm_89): one indexed view over a bank of experts.
//
// master stores one 2D parent per expert (experts/e/gate, experts/e/up, ...); the bench line
// stores a single rank-3 parent plus a per-expert layout. Both shapes are expressed here so
// the expert cache can be written once and used by either backing.
struct ExpertBankView {
    // One 2D parent per expert (master shape).
    const WeightParent* const* parents = nullptr;
    std::uint32_t count                = 0;

    [[nodiscard]] bool valid() const noexcept { return parents != nullptr && count != 0; }

    [[nodiscard]] const WeightParent& parent_of(std::uint32_t expert) const {
        if (!parents || expert >= count) {
            throw std::out_of_range("expert bank index is outside its domain");
        }
        return *parents[expert];
    }

    // Byte range of one stored plane for one expert. Each expert owns a complete 2D parent,
    // so the plane offsets come straight from its geometry.
    [[nodiscard]] ExpertPlaneSpan plane(ExpertPlaneKind kind, std::uint32_t expert) const {
        const auto& parent = parent_of(expert);
        const auto& g      = parent.geometry;
        switch (kind) {
        case ExpertPlaneKind::Codes:
            return {parent.data, static_cast<std::size_t>(g.code_bytes),
                    parent.device};
        case ExpertPlaneKind::High:
            return {g.high_bytes ? parent.data + g.high_offset : nullptr,
                    static_cast<std::size_t>(g.high_bytes), parent.device};
        case ExpertPlaneKind::Scales:
            return {g.scale_bytes ? parent.data + g.scale_offset : nullptr,
                    static_cast<std::size_t>(g.scale_bytes), parent.device};
        }
        return {};
    }
};

} // namespace ninfer
