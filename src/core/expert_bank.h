#pragma once

#include "core/weight_view.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer {

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
};

} // namespace ninfer
