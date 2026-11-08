#pragma once

// v8 section 12 (runtime parameter bus), first phase: the controller mode.
//
// Three deployment modes name WHO the controller is optimizing for, not which rules it may read
// (v8 12.4 "lambda follows the SLO mode"):
//
//   auto         three measured criteria live, 3% revert protection, balanced lambda.
//   stable       the freeze criterion is kept; the concurrency actuator only lowers (static<80
//                goes serial immediately, never rises back); the score's latency weight is 4x.
//   performance  the concurrency rise is allowed to cost up to 5% before it is reverted, and the
//                prefetch actuator re-explores after its park instead of waiting for usefulness to
//                recover from below.
//
// The environment/CLI flag is the INITIAL VALUE channel only (v8 12.1: after startup the owner is
// the controller). set_controller_mode_initial() therefore refuses to run once any thread has read
// the mode: a mode that could be changed under a running engine is a different feature.

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace ninfer {

enum class ControllerMode : std::uint8_t {
    Auto = 0,
    Stable = 1,
    Performance = 2,
};

[[nodiscard]] inline constexpr const char* controller_mode_name(ControllerMode mode) {
    switch (mode) {
    case ControllerMode::Stable: return "stable";
    case ControllerMode::Performance: return "performance";
    case ControllerMode::Auto: break;
    }
    return "auto";
}

// Parses "auto" / "stable" / "performance" (ASCII case-insensitive). Anything else is refused so a
// typo on a flag that steers a live controller cannot silently mean "auto".
[[nodiscard]] inline bool parse_controller_mode(std::string_view text, ControllerMode& out) {
    auto equals = [](std::string_view value, std::string_view lower) {
        if (value.size() != lower.size()) { return false; }
        for (std::size_t i = 0; i < value.size(); ++i) {
            const char c = value[i] | 0x20;
            if (c != lower[i]) { return false; }
        }
        return true;
    };
    if (equals(text, "auto")) {
        out = ControllerMode::Auto;
        return true;
    }
    if (equals(text, "stable")) {
        out = ControllerMode::Stable;
        return true;
    }
    if (equals(text, "performance")) {
        out = ControllerMode::Performance;
        return true;
    }
    return false;
}

namespace detail {

inline std::atomic<ControllerMode>& controller_mode_cell() {
    static std::atomic<ControllerMode> value{ControllerMode::Auto};
    return value;
}

// Whether controller_mode() has ever been read - the point after which the initial value is fixed.
inline std::atomic<bool>& controller_mode_read() {
    static std::atomic<bool> read{false};
    return read;
}

} // namespace detail

[[nodiscard]] inline ControllerMode controller_mode() {
    if (!detail::controller_mode_read().exchange(true)) {
        // First reader initializes from the environment: NINFER_CONTROLLER_MODE is the initial
        // value only, exactly like every other NINFER_* knob this phase.
        if (const char* text = std::getenv("NINFER_CONTROLLER_MODE");
            text != nullptr && *text != '\0') {
            ControllerMode parsed = ControllerMode::Auto;
            if (parse_controller_mode(text, parsed)) {
                detail::controller_mode_cell().store(parsed);
                std::fprintf(stderr, "[ctl] controller mode: %s (initial from "
                                     "NINFER_CONTROLLER_MODE)\n",
                             controller_mode_name(parsed));
                std::fflush(stderr);
            } else {
                std::fprintf(stderr,
                             "[ctl] NINFER_CONTROLLER_MODE='%s' is not "
                             "auto|stable|performance; keeping auto\n",
                             text);
                std::fflush(stderr);
            }
        }
    }
    return detail::controller_mode_cell().load();
}

// Applies a serve/CLI flag as the mode's initial value. Call before any engine exists (no worker
// thread may have read the mode yet). Returns false - with a printed reason - when the value is
// invalid or the mode has already been read; both are refusals, not silent clamps.
[[nodiscard]] inline bool set_controller_mode_initial(std::string_view text) {
    ControllerMode parsed = ControllerMode::Auto;
    if (!parse_controller_mode(text, parsed)) {
        std::fprintf(stderr,
                     "[ctl] controller mode '%.*s' is not auto|stable|performance; ignored\n",
                     static_cast<int>(text.size()), text.data());
        std::fflush(stderr);
        return false;
    }
    if (detail::controller_mode_read().exchange(true)) {
        std::fprintf(stderr,
                     "[ctl] controller mode already initialized; '%.*s' ignored "
                     "(the flag is an initial value only)\n",
                     static_cast<int>(text.size()), text.data());
        std::fflush(stderr);
        return false;
    }
    detail::controller_mode_cell().store(parsed);
    std::fprintf(stderr, "[ctl] controller mode: %s (initial from the command line)\n",
                 controller_mode_name(parsed));
    std::fflush(stderr);
    return true;
}

// T131: the RUNTIME owner's write. v8 12.1 says the environment/flag is the initial value and
// after startup the controller owns the mode - this is that ownership expressed as one write,
// made by the engine worker when the control plane delivers a mode (EngineCore's parked-request
// scheduling point). It marks the mode as read (so a late --controller-mode cannot overwrite it)
// and stores the value in the same cell every reader already consults - one value, one home.
inline void set_controller_mode_runtime(ControllerMode mode) {
    detail::controller_mode_read().store(true);
    detail::controller_mode_cell().store(mode);
}

} // namespace ninfer
