#include "models/qwen3_5/execution/control_plane.h"

#include <cstdint>
#include <iostream>
#include <string>

// T131: the control plane's pure half. The three deployment modes are a named bundle of knobs
// the controller and the expert stream already own, so the mapping - and the refusal of a typo -
// are testable without a GPU, a model or an engine. The engine-side half (park/apply/wait and
// the HTTP routes) is exercised by the serve tests and the live acceptance; this file pins the
// contract the console relies on: a mode name means exactly one freeze policy, and an
// out-of-range freeze value is refused with a reason.

namespace {

using namespace ninfer::models::qwen3_5::execution;

int failures = 0;

void require(bool condition, const char* message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

} // namespace

int main() {
    // ---- the three modes map to one bundle each ----
    {
        ModeBundle bundle;
        require(parse_mode_bundle("auto", bundle) &&
                    bundle.mode == ninfer::ControllerMode::Auto,
                "auto did not parse to the auto controller mode");
        require(bundle.freeze == FreezeChoice::KeepInitial,
                "auto must keep the configured freeze criterion (v8 12.2: all criteria live)");

        require(parse_mode_bundle("STABLE", bundle) &&
                    bundle.mode == ninfer::ControllerMode::Stable,
                "stable did not parse case-insensitively to the stable controller mode");
        require(bundle.freeze == FreezeChoice::KeepInitial,
                "stable must keep the freeze criterion (v8 12.4)");

        require(parse_mode_bundle("performance", bundle) &&
                    bundle.mode == ninfer::ControllerMode::Performance,
                "performance did not parse to the performance controller mode");
        require(bundle.freeze == FreezeChoice::Disabled,
                "performance must spend the freeze placement (freeze_after = 0)");
        require(std::string(bundle.name) == "performance",
                "the bundle did not carry the canonical mode name");

        // A typo is refused instead of read as auto - the same rule the startup channel applies.
        require(!parse_mode_bundle("aggressive", bundle), "a typo was accepted as a mode");
        require(!parse_mode_bundle("", bundle), "an empty mode was accepted");
        require(!parse_mode_bundle("auto ", bundle), "a mode with trailing space was accepted");
    }

    // ---- the freeze family's ranges are the bus rows' own ----
    {
        std::string reason;
        require(freeze_policy_in_range(0, 0, &reason), "freeze_after=0 (never freeze) was refused");
        require(reason.empty(), "a successful range check filled a refusal reason");
        require(freeze_policy_in_range(kFreezeAfterMax, kFreezeReselectMax),
                "the inclusive upper bounds were refused");
        require(!freeze_policy_in_range(kFreezeAfterMax + 1, 0, &reason),
                "freeze_after above the registered maximum was accepted");
        require(reason.find("freeze_after") != std::string::npos,
                "the refusal did not name the offending field");
        reason.clear();
        require(!freeze_policy_in_range(0, kFreezeReselectMax + 1, &reason),
                "reselect_after above the registered maximum was accepted");
        require(reason.find("reselect_after") != std::string::npos,
                "the refusal did not name the offending field");
    }

    // ---- a mode's freeze half resolves against the process's configured policy ----
    // The engine worker stores the configured policy at startup and "keep initial" restores it;
    // what matters here is that only performance asks for an explicit zero.
    {
        ModeBundle perf;
        require(parse_mode_bundle("performance", perf) &&
                    perf.freeze == FreezeChoice::Disabled,
                "performance did not ask for the disabled placement");
        ModeBundle stable;
        require(parse_mode_bundle("stable", stable) &&
                    stable.freeze == FreezeChoice::KeepInitial,
                "stable asked for an explicit value instead of the configured policy");
    }

    // ---- the runtime owner's write lands in the process-wide cell ----
    // v8 12.1: the flag is an initial value; after startup the controller owns the mode. The
    // control plane's write must move the same cell every panel reads, and a late initial value
    // must not be able to overwrite it.
    {
        ninfer::set_controller_mode_runtime(ninfer::ControllerMode::Stable);
        require(ninfer::controller_mode() == ninfer::ControllerMode::Stable,
                "the runtime mode write did not reach the process-wide cell");
        require(!ninfer::set_controller_mode_initial("auto"),
                "a late initial value overwrote the runtime mode");
        require(ninfer::controller_mode() == ninfer::ControllerMode::Stable,
                "the refused initial value changed the runtime mode");
    }

    if (failures == 0) {
        std::cout << "control plane: all checks passed\n";
        return 0;
    }
    std::cerr << failures << " check(s) failed\n";
    return 1;
}
