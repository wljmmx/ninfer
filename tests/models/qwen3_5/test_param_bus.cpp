#include "core/controller_mode.h"
#include "models/qwen3_5/execution/expert_path_control.h"
#include "models/qwen3_5/execution/param_bus.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

// v8 section 12, first phase: the parameter bus, the serial rule (9.3) and the three deployment
// modes are pure host-side logic - no GPU, no model - so the whole controller surface is testable
// without one. The environment is the INITIAL-value channel, so every NINFER_* knob this file
// cares about is pinned at the very start of main, before any bus or controller is constructed.

namespace {

using namespace ninfer::models::qwen3_5::execution;

int failures = 0;

void require(bool condition, const char* message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

#ifdef _WIN32
void set_env(const char* name, const char* value) { (void)_putenv_s(name, value); }
#else
void set_env(const char* name, const char* value) { (void)setenv(name, value, 1); }
#endif

// One serial/concurrent transition per test: a fresh controller over the shared thread-local bus,
// with a recording executor standing in for the engine.
struct Harness {
    ninfer::models::qwen3_5::execution::PolicyController controller;
    std::uint32_t applied = 0;
    std::uint32_t refused  = 0;

    explicit Harness(std::uint32_t max_concurrency, ninfer::ControllerMode mode,
                     double static_pct, double tps, std::uint32_t lanes) {
        controller.mode = mode;
        controller.signals.static_pct = static_pct;
        controller.signals.decode_tps = tps;
        controller.observe_lanes(lanes);
        param_bus().attach_applier(
            "concurrency", max_concurrency, 1.0, static_cast<double>(max_concurrency),
            [this](double value) {
                const auto lanes = static_cast<std::uint32_t>(value);
                if (lanes < 1U || lanes > 8U) {
                    ++refused;
                    return false;
                }
                applied = lanes;
                return true;
            });
    }

    // One layer: the bus ticks first (the same order ffn.cpp uses), then the controller steps.
    void layer() {
        param_bus().tick();
        controller.step(false, false, false, false, /*concurrency_wired=*/true);
    }
};

} // namespace

int main() {
    // Pin the initial-value channel before anything reads it.
    set_env("NINFER_CONTROLLER_MODE", "performance");
    set_env("NINFER_EXPERT_CACHE_MB", "4096");
    set_env("NINFER_BUDGET_RAM_MB", "930");
    set_env("NINFER_EXPERT_EPOCH", "64");
    set_env("NINFER_EXPERT_FREEZE", "4");
    set_env("NINFER_PREFETCH_PER_LAYER", "8");

    // ---- controller_mode: the environment is the initial value, read once ----
    {
        ninfer::ControllerMode parsed = ninfer::ControllerMode::Auto;
        require(ninfer::parse_controller_mode("stable", parsed) &&
                    parsed == ninfer::ControllerMode::Stable,
                "parse_controller_mode rejected a valid mode");
        require(ninfer::parse_controller_mode("PERFORMANCE", parsed) &&
                    parsed == ninfer::ControllerMode::Performance,
                "parse_controller_mode is not case-insensitive");
        require(!ninfer::parse_controller_mode("aggressive", parsed),
                "parse_controller_mode accepted a typo as auto");

        PolicyController env_controller;
        require(env_controller.mode == ninfer::ControllerMode::Performance,
                "the controller did not read NINFER_CONTROLLER_MODE as its initial mode");
        // The mode was read above; a later "initial" application must be refused - a mode that
        // can be changed under a running engine is a different feature.
        require(!ninfer::set_controller_mode_initial("stable"),
                "set_controller_mode_initial succeeded after the mode was already read");
        require(ninfer::controller_mode() == ninfer::ControllerMode::Performance,
                "a refused initial application still changed the mode");
    }

    // ---- ParamBus: registration, refusal reasons, ranges, cooldown ----
    {
        ParamBus bus; // fresh instance: reads the (pinned) environment once, here
        require(bus.get("expert_pool_mb") == 4096.0,
                "expert_pool_mb did not take NINFER_EXPERT_CACHE_MB as its initial value");
        require(bus.get("warm_arena_mb") == 930.0,
                "warm_arena_mb did not take NINFER_BUDGET_RAM_MB as its initial value");
        require(bus.get("kv_pages") < 0.0,
                "kv_pages should be source-missing (negative), not a made-up number");
        require(bus.get("no_such_param") != bus.get("no_such_param"), // NaN != NaN
                "an unknown parameter must read as NaN, not 0");

        // L1: every write is refused this phase, reason "load acceptance pending" (v8 13.5-5).
        require(bus.set("expert_pool_mb", 8192.0, /*from_controller=*/true) ==
                    ParamReject::LoadAccept,
                "L1 expert_pool_mb write was not refused as load-acceptance-pending");
        require(bus.set("kv_pages", 1.0, true) == ParamReject::LoadAccept,
                "L1 kv_pages write was not refused");
        // Authority: L2/L3 belong to the controller; anyone else is refused (越权).
        require(bus.set("prefetch_slots", 8.0, /*from_controller=*/false) == ParamReject::TierLocked,
                "a non-controller caller could write an L2 parameter");
        // No accepted executor wired for prefetch_slots yet.
        require(bus.set("prefetch_slots", 8.0, true) == ParamReject::NoExecutor,
                "writing a parameter without an accepted executor was not refused");
        require(bus.set("nobody", 1.0, true) == ParamReject::Unknown,
                "an unknown parameter was not refused");

        // Ranges are hard bounds, checked before anything else moves.
        bus.attach_applier("concurrency", 4.0, 1.0, 4.0, [](double) { return true; });
        require(bus.set("concurrency", 0.0, true) == ParamReject::Range,
                "set below lo was not refused as out of range");
        require(bus.set("concurrency", 5.0, true) == ParamReject::Range,
                "set above hi was not refused as out of range");
        require(bus.set("concurrency", 4.0, true) == ParamReject::Ok,
                "re-asserting the current value must be a no-op success");
        require(bus.set("concurrency", 1.0, true) == ParamReject::Ok,
                "a valid in-range write was refused");
        require(bus.get("concurrency") == 1.0, "the value did not move to what was set");
        require(bus.set("concurrency", 4.0, true) == ParamReject::Cooldown,
                "a second change inside one window was not refused as cooldown");
        for (int i = 0; i < 32; ++i) { bus.tick(); }
        require(bus.set("concurrency", 4.0, true) == ParamReject::Ok,
                "the cooldown did not expire after one full window of ticks");

        // L0 geometry is never writable, by anyone.
        bus.register_spec({.name = "slot_bytes",
                          .initial = 1,
                          .lo = 0,
                          .hi = 1,
                          .tier = ParamTier::L0,
                          .cost_ms = 0,
                          .observe = "n/a",
                          .rollback = 1,
                          .source = "test"});
        require(bus.set("slot_bytes", 1.0, true) == ParamReject::TierLocked,
                "an L0 geometry write was not refused");

        // record/settle: the ledger keeps (name, from, to, tps_before, tps_after).
        bus.record("concurrency", 1.0, 4.0, 50.0, -1.0);
        require(bus.ledger().back().tps_after < 0.0,
                "a fresh ledger entry must not pretend to know tps_after");
        bus.settle_pending(55.0);
        require(bus.ledger().back().tps_after == 55.0,
                "settle_pending did not fill the newest open entry");
    }

    // ---- want_serial: the three states of the 80% rule (v8 9.3) ----
    {
        PolicyController c;
        c.mode = ninfer::ControllerMode::Auto;
        c.signals.static_pct = 42.8; // 8 GiB coverage: concurrent measured -39%
        c.observe_lanes(4);
        c.concurrency.enabled = false;
        require(c.want_serial(), "static<80 with lanes>1 must want serial");
        c.observe_lanes(1);
        require(!c.want_serial(),
                "static<80 with one lane is already serial-shaped; the rule must hold");
        c.concurrency.enabled = true;
        require(c.want_serial(),
                "serial in force with one lane must hold (nothing to serialize, nothing to rise)");
        c.signals.static_pct = 84.3; // 17 GiB coverage: concurrent measured +13%
        require(!c.want_serial(), "static>=80 with serial in force must allow the rise");
        c.mode = ninfer::ControllerMode::Stable;
        require(c.want_serial(), "stable mode must never rise, even at static>=80");
        c.mode = ninfer::ControllerMode::Performance;
        require(!c.want_serial(), "performance mode rises at static>=80 like auto");
        c.signals.static_pct = -1.0;
        require(c.want_serial() == c.concurrency.enabled,
                "an unmeasured static must hold the current state");
    }

    // ---- the concurrency actuator: 32 agreeing layers, then a live flip through the bus ----
    {
        Harness h(4, ninfer::ControllerMode::Auto, 42.8, 100.0, 4);
        for (int i = 0; i < 31; ++i) { h.layer(); }
        require(!h.controller.concurrency.enabled,
                "31 agreeing layers must not flip yet (the 32-layer agreement)");
        h.layer();
        require(h.controller.concurrency.enabled,
                "the 32nd agreeing layer did not flip to serial");
        require(h.applied == 1U, "the serial flip did not reach the engine executor");
        require(param_bus().get("concurrency") == 1.0,
                "the bus does not hold the value the executor was given");
        require(param_bus().has_applier("concurrency"), "the executor attachment was lost");
        // Shadow: without the executor the same decision must move nothing.
        Harness shadow(4, ninfer::ControllerMode::Auto, 42.8, 100.0, 4);
        PolicyController& c = shadow.controller;
        c.concurrency.wired = false;
        for (int i = 0; i < 40; ++i) {
            param_bus().tick();
            c.step(false, false, false, false, /*concurrency_wired=*/false);
        }
        require(!c.concurrency.enabled, "a shadow flip must not change the in-force state");
        require(shadow.applied == 0U, "a shadow flip must not touch the executor");
    }

    // ---- the revert guardrail: auto reverts at 3%, performance keeps to 5% ----
    {
        // auto: serial in force at baseline 100; a 4% degradation is reverted (threshold 3%)
        // once the settle window has let the tps EWMA turn over, and the refused probe parks
        // with the exponential backoff.
        Harness h(4, ninfer::ControllerMode::Auto, 42.8, 100.0, 4);
        for (int i = 0; i < 32; ++i) { h.layer(); } // flip to serial at layer 32, baseline 100
        require(h.controller.concurrency.enabled, "setup: the serial flip did not happen");
        h.controller.signals.decode_tps = 96.0; // 4% below the flip-time baseline
        h.controller.observe_lanes(1);          // serial in force: the multi-lane traffic drained
        for (int i = 0; i < 64; ++i) { h.layer(); }
        require(!h.controller.concurrency.enabled,
                "auto did not revert a serial that cost 4% (threshold 3%)");
        require(h.controller.concurrency.parked_layers >= 2U * 40U * 32U,
                "the refused probe did not back off beyond the base park");
        require(h.applied == 4U, "the revert did not restore the configured maximum");

        // performance: the same 4% degradation stays (threshold 5%); serial holds unparked.
        Harness p(4, ninfer::ControllerMode::Performance, 42.8, 100.0, 4);
        for (int i = 0; i < 32; ++i) { p.layer(); }
        require(p.controller.concurrency.enabled, "setup: the performance flip did not happen");
        p.controller.signals.decode_tps = 96.0;
        p.controller.observe_lanes(1);
        for (int i = 0; i < 64; ++i) { p.layer(); }
        require(p.controller.concurrency.enabled,
                "performance undid a serial inside its 5% allowance");
        require(p.controller.concurrency.parked_layers == 0,
                "a holding serial should not park the actuator");
        require(p.applied == 1U, "the performance serial did not hold its state");
    }

    // ---- the rise: static>=80 re-opens multi-lane admission after a serial ----
    {
        Harness h(4, ninfer::ControllerMode::Auto, 42.8, 100.0, 4);
        for (int i = 0; i < 32; ++i) { h.layer(); } // serial in force, applied==1
        require(h.applied == 1U, "setup: the serial flip did not happen");
        h.controller.signals.static_pct = 90.0;
        h.controller.signals.decode_tps = 110.0; // serial is healthy: the rise must not be punished
        h.controller.observe_lanes(1);
        for (int i = 0; i < 32; ++i) { h.layer(); }
        require(!h.controller.concurrency.enabled, "static>=80 did not rise back to concurrent");
        require(h.applied == 4U, "the rise did not restore the configured maximum");
        require(h.controller.concurrency.parked_layers == 0, "a healthy rise should not park");
    }

    // ---- the phase gate: prefill layers neither argue for serial nor against it ----
    {
        Harness h(4, ninfer::ControllerMode::Auto, 42.8, 100.0, 4);
        h.controller.observe_prefill(true);
        for (int i = 0; i < 200; ++i) { h.layer(); }
        require(!h.controller.concurrency.enabled,
                "a prefill window flipped the serial rule (its tps is not a decode measurement)");
        require(h.applied == 0U, "a prefill window touched the executor");
        h.controller.observe_prefill(false); // decode traffic arrives
        for (int i = 0; i < 32; ++i) { h.layer(); }
        require(h.controller.concurrency.enabled,
                "decode layers with static<80 and lanes>1 did not flip to serial");
        require(h.applied == 1U, "the decode-window flip did not reach the executor");
    }

    // ---- the stale-verdict rail: a serial with no multi-lane traffic re-opens admission ----
    {
        Harness h(4, ninfer::ControllerMode::Auto, 42.8, 100.0, 4);
        for (int i = 0; i < 32; ++i) { h.layer(); } // serial in force, baseline 100
        require(h.applied == 1U, "setup: serial did not engage");
        // Simulate the false-confirm that gated a whole queue in the first live A/B: a
        // cold-read-depressed baseline makes every window confirm serial, and the queue starves.
        h.controller.signals.decode_tps = 98.0; // inside the 3% keep band: the monitor confirms
        h.controller.observe_lanes(1);
        for (int i = 0; i < 1023; ++i) { h.layer(); }
        require(h.controller.concurrency.enabled,
                "setup: a confirming serial must hold until the rail fires");
        h.layer();
        require(!h.controller.concurrency.enabled,
                "1024 layers without multi-lane traffic did not re-open admission");
        require(h.applied == 4U, "the rail's rise did not restore the configured maximum");
        // The rule may re-drop once there is multi-lane traffic to judge again.
        h.controller.observe_lanes(4);
        for (int i = 0; i < 32; ++i) { h.layer(); }
        require(h.controller.concurrency.enabled,
                "multi-lane traffic with static<80 did not re-propose serial");
        require(h.applied == 1U, "the re-proposed serial did not reach the executor");
    }

    // ---- stable: 静态<80 立即串行, and no rise ever ----
    {
        Harness h(4, ninfer::ControllerMode::Stable, 42.8, 100.0, 4);
        h.layer(); // ONE layer: stable lowers immediately, no 32-layer ramp
        require(h.controller.concurrency.enabled,
                "stable mode did not serialize on the first agreeing layer");
        require(h.applied == 1U, "stable's immediate drop did not reach the executor");
        h.controller.signals.static_pct = 95.0;
        h.controller.observe_lanes(1);
        h.controller.signals.decode_tps = 50.0; // even a bad serial is kept: stable chose latency
        for (int i = 0; i < 200; ++i) { h.layer(); }
        require(h.controller.concurrency.enabled,
                "stable mode rose (or reverted) against its only-lower semantics");
        require(h.applied == 1U, "stable mode did not hold the serial state");
    }

    // ---- freeze live marking: the real stream state, not a shadow decision ----
    {
        PolicyController c;
        c.mode = ninfer::ControllerMode::Auto;
        c.observe_freeze_state(true);
        param_bus().tick();
        c.step(false, false, false, /*freeze_wired=*/true, false);
        require(c.freeze_set.enabled && c.freeze_set.wired,
                "an armed freeze did not mark the actuator live with the stream's real state");
        require(param_bus().get("freeze_on") == 1.0,
                "the bus did not mirror the live freeze state");
        c.observe_freeze_state(false);
        param_bus().tick();
        c.step(false, false, false, true, false);
        require(!c.freeze_set.enabled, "a thawed stream did not clear the live freeze state");
        // Without the mechanism armed the actuator stays the shadow view of the same criterion.
        PolicyController s;
        s.mode = ninfer::ControllerMode::Auto;
        param_bus().tick();
        s.step(false, false, false, /*freeze_wired=*/false, false);
        require(!s.freeze_set.wired, "an unarmed freeze reported itself as live");
    }

    // ---- the mode line inputs: the score is tps - lambda1*latency - lambda2*miss_cost ----
    {
        PolicyController c;
        c.mode = ninfer::ControllerMode::Auto;
        c.signals.decode_tps = 100.0;
        c.signals.wall_ms = 0.15; // 6 ms/token via the layer-wall proxy (no p99 collection point)
        c.signals.miss_per_layer = 0.5;
        c.signals.upload_us_per_expert = 500.0; // 10 ms/token of miss cost
        const double expected = 100.0 - c.lambda_p99() * 6.0 - c.lambda_miss() * 10.0;
        require(c.score() == expected, "the score is not tps - lat - miss terms");
        const double auto_p99 = c.lambda_p99();
        const double auto_miss = c.lambda_miss();
        c.mode = ninfer::ControllerMode::Stable;
        require(c.lambda_p99() > auto_p99,
                "stable must weight latency higher than auto");
        c.mode = ninfer::ControllerMode::Performance;
        require(c.lambda_p99() < auto_p99 && c.lambda_miss() >= auto_miss,
                "performance must nearly zero latency and not lower the miss weight");
    }

    if (failures == 0) {
        std::cout << "param bus + policy controller: all checks passed\n";
        return 0;
    }
    std::cerr << failures << " check(s) failed\n";
    return 1;
}
