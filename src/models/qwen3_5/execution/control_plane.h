#pragma once

// port (sm_89), T131: the control plane's pure half - the three deployment modes as named
// bundles of knobs the controller and the expert stream ALREADY own, plus the freeze family's
// registered range. Nothing here executes anything: the engine worker is the only thread that
// may move a mode or a freeze policy (the controller and the stream it steers live on it), and
// EngineCore::apply_pending_control is the single place that hands these values over. Keeping
// the mapping in a header with no engine dependency is what makes it testable without a GPU
// (tests/models/qwen3_5/test_control_plane.cpp) and what keeps the HTTP layer from inventing a
// second copy of the mode semantics.
//
// Why a bundle and not just the mode: v8 12.1 says the mode is a deployment choice the
// controller optimizes for, and the two knobs it needs beyond its own rules are the expert
// freeze policy's. The mode therefore carries WHICH freeze policy it wants:
//
//   auto          the measured freeze criterion stays in force at the process's configured
//                 warm-up (NINFER_EXPERT_FREEZE, default 4). All three measured criteria live.
//   stable        the same freeze criterion (v8 12.4: stable KEEPS it), plus the controller's
//                 own only-lower concurrency and disabled pool dial.
//   performance   the freeze placement is disabled (freeze_after = 0): a frozen set stops the
//                 placement from following the router, which is the throughput lever this mode
//                 exists to spend. The controller's own 5% revert allowance is its business.
//
// The freeze policy is only an actuator where a freeze mechanism is armed: the measured rule
// (NINFER_FREEZE_RULE / the old NINFER_FREEZE_ON_NO_MISS name) and the host-freeze mode
// (NINFER_HOST_FREEZE) are what decide, and `freeze_after`/`reselect_after`/`thaw_on_request`
// are their knobs (expert_stream.h's FreezePolicy). A deployment with neither armed reads the
// policy back and applies it to the next freeze it takes - it does not start freezing by
// itself, and the control state says so (`freeze_armed`).

#include "core/controller_mode.h"

#include <cstdint>
#include <string>
#include <string_view>

namespace ninfer::models::qwen3_5::execution {

// The rows param_bus.h registers: freeze_after [0..32] (0 = never freeze), freeze_reselect
// [0..256] (0 = never re-select), freeze_thaw_on_request {0,1}. The control plane refuses a
// value outside those bounds with the same reason the bus's own range check gives, so a caller
// sees one contract whether it writes through HTTP or through the bus.
inline constexpr std::uint32_t kFreezeAfterMax    = 32;
inline constexpr std::uint32_t kFreezeReselectMax = 256;

enum class FreezeChoice : std::uint8_t {
    KeepInitial = 0, // the process's configured policy (freeze_after from NINFER_EXPERT_FREEZE)
    Disabled    = 1, // freeze_after = 0: the placement always follows the router
};

struct ModeBundle {
    ninfer::ControllerMode mode   = ninfer::ControllerMode::Auto;
    FreezeChoice           freeze = FreezeChoice::KeepInitial;
    const char*            name   = "auto";
};

// ASCII case-insensitive, exactly "auto" / "stable" / "performance". A typo is refused rather
// than read as auto - the same rule core/controller_mode.h applies to the startup channel.
[[nodiscard]] inline bool parse_mode_bundle(std::string_view text, ModeBundle& out) {
    ninfer::ControllerMode parsed = ninfer::ControllerMode::Auto;
    if (!ninfer::parse_controller_mode(text, parsed)) { return false; }
    out.mode = parsed;
    out.name = ninfer::controller_mode_name(parsed);
    out.freeze = parsed == ninfer::ControllerMode::Performance ? FreezeChoice::Disabled
                                                              : FreezeChoice::KeepInitial;
    return true;
}

// Range check for an explicit freeze policy write; `reason` is filled only on refusal.
[[nodiscard]] inline bool freeze_policy_in_range(std::uint32_t freeze_after, std::uint32_t reselect,
                                                std::string* reason = nullptr) {
    if (freeze_after > kFreezeAfterMax) {
        if (reason != nullptr) {
            *reason = "freeze_after must be in [0," + std::to_string(kFreezeAfterMax) +
                      "] (0 = never freeze)";
        }
        return false;
    }
    if (reselect > kFreezeReselectMax) {
        if (reason != nullptr) {
            *reason = "reselect_after must be in [0," + std::to_string(kFreezeReselectMax) +
                      "] (0 = never re-select)";
        }
        return false;
    }
    return true;
}

} // namespace ninfer::models::qwen3_5::execution
