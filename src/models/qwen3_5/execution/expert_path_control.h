#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// port (sm_89): the runtime policy controller - the one place that decides how a running instance
// serves experts, whether it prefetches, whether it may capture graphs, whether it keeps a frozen
// set, and (v8 9.3) how many lanes the scheduler may fill. Every decision is taken from
// measurement; none is keyed on a pool size or a GiB figure.
//
// Why: the same 12 GiB pool serves a converged set at 0.05 misses a layer and a cold one at 2.2.
// A size number describes two different machines, so a threshold on it is a guess wearing a
// constant's clothes.
//
// The plan already names what this drives (v6 section 7, runtime console):
//
//   actuator     plan id   changes                                  guardrail (v6 section 6)
//   -----------  --------  ---------------------------------------  ---------------------------
//   expert path  E5/O1     host upload vs device-side serving      cost model, see below
//   prefetch     D1-D5     queue on/off, window budget              usefulness < 10% -> off
//   layer graph  GH3/GH4   capture whole layers or not              capturable < 90% -> off
//   frozen set   B1-B5     freeze the placement set or not          no_miss < 99%    -> off
//   concurrency  v8 9.3    serial vs multi-lane admission          static < 80% -> serial
//
// Each actuator follows the same shape: observe, compare against a rule, require a run of agreeing
// layers, then act - recording the throughput at the moment of acting, so a change that costs more
// than it buys is reverted and that actuator is parked for a while.
//
// Until an actuator is wired into the kernels it reports "(shadow)": it prints what it would do and
// why, and changes nothing.
//
// v8 section 12, first phase: the three deployment modes. "mode" is WHO the controller optimizes
// for, not which rules it may read (v8 12.4: the lambda follows the SLO mode):
//
//   auto         three measured criteria live, 3% revert protection, balanced lambda.
//   stable       the freeze criterion is kept; the concurrency actuator only lowers - static<80
//                goes serial immediately (no 32-layer ramp) and never rises back, so a bad serial
//                is not un-done either: the operator chose latency over throughput.
//   performance  the concurrency rise may cost up to 5% before it is reverted, and the prefetch
//                actuator re-explores once its park expires instead of idling at a usefulness of
//                zero forever.
//
// The mode is a deployment choice read once at startup (core/controller_mode.h); NINFER_CONTROLLER_
// MODE and the serve/CLI --controller-mode flag are its initial-value channels only.

#include "core/controller_mode.h"
#include "models/qwen3_5/execution/param_bus.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>

namespace ninfer::models::qwen3_5::execution {

struct PolicySignals {
    double fetch_us             = 0.0; // host waiting for this layer's routes
    double upload_us_per_expert = 0.0; // host-to-device travel per expert
    double read_us_per_expert   = 0.0; // device reading host RAM, from the measured link
    double miss_per_layer       = 0.0; // experts the pool had to move in
    double ens_ms               = 0.0;
    double gpu_ms               = 0.0;
    double wall_ms              = 0.0;
    // Coverage, from GH1. Negative means "not measured on this path"; never zero, which would read
    // as a measurement of failure rather than an absence of one.
    double static_pct     = -1.0;
    double no_miss_pct    = -1.0;
    double capturable_pct = -1.0;
    // Q20: the execution half of the capture question, next to the capacity half above.
    // capt_free = the layer asked the host for nothing; capt_ready = that AND it was fully covered,
    // which is the state a whole-layer graph can actually be captured in. `graph` is driven by
    // capt_ready, not by capturable - see want_graph().
    double capt_free_pct  = -1.0;
    double capt_ready_pct = -1.0;
    // Warm-tier decision (1y.7 step 3f): what a cold read costs, what a hit saves, and what making
    // the copy costs. Break-even is a ratio between the last two, measured - not a RAM size.
    double cold_read_us_per_expert = 0.0;
    double mirror_fill_us_per_expert = 0.0;
    double mirror_hit_ratio          = -1.0;
    double prefetch_usefulness = -1.0; // D4 gate input
    double decode_tps          = -1.0; // the outcome every actuator is judged by
    // 13.4: occupied share of the paged KV edge, 0..1, -1 = not measured this window.
    double kv_occupancy        = -1.0;
    // v8 9.3: the serial rule is judged on the lanes actually in service, not the configured
    // ceiling - serializing one lane is a no-op. Fed by the engine worker (the thread that owns
    // the slots) once per scheduling pass.
    std::uint32_t lanes_busy = 0;
    // v8 9.2 row 3: when the freeze mechanism is armed, the ExpertStream owns the real freeze
    // decision (32 clean layers, keep only if static rose or tps held); the controller mirrors
    // that state here so the panel can print it instead of a shadow decision.
    double freeze_active = -1.0; // < 0: no observation yet; 0/1: the stream's frozen state
    // v8 9.3, learned from the first live A/B: the serial rule is a DECODE rule. During a prefill
    // window one layer pass carries a whole chunk (not one token), so the layer-interval tps
    // reads 0.4-9 tok/s there and lanes count a phase whose admission serializing is a no-op
    // (prefill is one lane at a time by construction). The engine worker feeds this once per
    // dispatch; the controller neither flips nor verifies serial on a prefill layer.
    bool in_prefill = false;
};

struct PolicyActuator {
    bool enabled  = false; // value in force
    bool wired    = false; // false: shadow, the switch does not exist yet
    std::uint32_t agree = 0;
    double baseline_tps = -1.0;
    std::uint32_t parked_layers = 0;
    std::uint32_t settle_layers = 0; // concurrency: layers until the post-change check may run
    const char* reason = "n/a";
};

class PolicyController {
public:
    PolicySignals signals;
    PolicyActuator expert_path;
    PolicyActuator prefetch;
    PolicyActuator graph;
    PolicyActuator warm_mirror; // 1y.7: keep managed copies in the pinned tier
    PolicyActuator freeze_set;
    // v8 9.3: enabled == true means SERIAL admission is in force (the engine fills one lane).
    PolicyActuator concurrency;
    ninfer::ControllerMode mode = ninfer::controller_mode();
    double link_gbps    = 0.0;
    std::uint64_t layers = 0;

    void observe_fetch(double micros) { signals.fetch_us += (micros - signals.fetch_us) / 16.0; }

    void observe_upload(double micros, std::uint32_t experts) {
        if (experts == 0) { return; }
        const double per = micros / static_cast<double>(experts);
        signals.upload_us_per_expert += (per - signals.upload_us_per_expert) / 16.0;
    }

    void observe_misses(std::uint32_t experts) {
        signals.miss_per_layer += (static_cast<double>(experts) - signals.miss_per_layer) / 16.0;
    }

    void observe_layer(double ens_ms, double gpu_ms, double wall_ms) {
        signals.ens_ms += (ens_ms - signals.ens_ms) / 16.0;
        signals.gpu_ms += (gpu_ms - signals.gpu_ms) / 16.0;
        signals.wall_ms += (wall_ms - signals.wall_ms) / 16.0;
    }

    void observe_coverage(double static_pct, double no_miss_pct, double capturable_pct,
                          double capt_free_pct = -1.0, double capt_ready_pct = -1.0) {
        signals.static_pct     = static_pct;
        signals.no_miss_pct    = no_miss_pct;
        signals.capturable_pct = capturable_pct;
        signals.capt_free_pct  = capt_free_pct;
        signals.capt_ready_pct = capt_ready_pct;
    }

    void observe_decode(double tps) { signals.decode_tps = tps; }

    // v8 9.3: the engine's live lane count, one observation per scheduling pass.
    void observe_lanes(std::uint32_t busy) { signals.lanes_busy = busy; }

    // 13.4: how much of the paged KV edge the last window actually used, 0..1 (-1 = unmeasured).
    // The KV slow dial's only structural input: an edge that is never approached is device memory
    // the expert pool could use, and an edge that saturates is the opposite statement. Fed by the
    // engine worker once per window from the program's own physical usage, never estimated here.
    void observe_kv_pool(double occupancy) { signals.kv_occupancy = occupancy; }

    // v8 9.2 row 3: the real frozen state of the stream currently being served.
    void observe_freeze_state(bool frozen) { signals.freeze_active = frozen ? 1.0 : 0.0; }

    // v8 9.3: the engine marks whether the layer about to run is prefill or decode.
    void observe_prefill(bool active) { signals.in_prefill = active; }

    // One 32 MiB host-to-device copy, timed once: the link the Device would read host experts over.
    // Replaced by a per-layer measurement once that path exists.
    void probe_link(std::uint64_t slot_bytes) {
        constexpr std::size_t kBytes = 32U << 20;
        void* host                   = nullptr;
        void* device                 = nullptr;
        if (cudaHostAlloc(&host, kBytes, cudaHostAllocMapped) != cudaSuccess) { return; }
        if (cudaMalloc(&device, kBytes) != cudaSuccess) {
            (void)cudaFreeHost(host);
            return;
        }
        cudaEvent_t begin = nullptr;
        cudaEvent_t end   = nullptr;
        if (cudaEventCreate(&begin) != cudaSuccess || cudaEventCreate(&end) != cudaSuccess) {
            (void)cudaFree(device);
            (void)cudaFreeHost(host);
            return;
        }
        float best = 1e9F;
        for (int round = 0; round < 3; ++round) {
            (void)cudaEventRecord(begin);
            (void)cudaMemcpyAsync(device, host, kBytes, cudaMemcpyHostToDevice);
            (void)cudaEventRecord(end);
            (void)cudaEventSynchronize(end);
            float ms = 0.0F;
            (void)cudaEventElapsedTime(&ms, begin, end);
            if (ms < best) { best = ms; }
        }
        if (best > 0.0F) {
            link_gbps = static_cast<double>(kBytes) / (static_cast<double>(best) * 1e-3) / 1e9;
            signals.read_us_per_expert =
                static_cast<double>(slot_bytes) / (link_gbps * 1e9) * 1e6;
        }
        (void)cudaEventDestroy(begin);
        (void)cudaEventDestroy(end);
        (void)cudaFree(device);
        (void)cudaFreeHost(host);
    }

    [[nodiscard]] double host_cost_us() const {
        return signals.fetch_us + signals.miss_per_layer * signals.upload_us_per_expert;
    }
    [[nodiscard]] double device_cost_us() const {
        return signals.miss_per_layer * signals.read_us_per_expert;
    }
    [[nodiscard]] bool want_device_path() const {
        if (signals.read_us_per_expert <= 0.0) { return false; }
        return device_cost_us() < host_cost_us() * 0.8;
    }
    // Prefetch: the queue no longer decides this for itself (it used to close its own gate at a
    // written 10%). Two things take its place, and neither is a constant:
    //   1. exploration while nothing has been measured - a rule cannot judge a mechanism that has
    //      never run, and the revert loop below bounds what exploring costs;
    //   2. once bytes have been measured, "did anything the read-ahead fetched ever get used".
    //      Whether that is *worth* the I/O is not decided by a ratio here: drive() records the
    //      throughput the switch was flipped at and parks the actuator if the next window is 3%
    //      slower, which is the only measurement specific to this machine.
    //
    // Mode semantics (v8 12.2, prefetch row): stable defaults the read-ahead ON and closes it as
    // soon as the measured usefulness sits under D4's threshold - 0.10 is not a new constant, it
    // is the gate the plan itself measured with (Strata's read-ahead at 18 GiB, the same number
    // the queue's min_usefulness still reports); auto/performance hold the exploration form and
    // only a usefulness of exactly zero (nothing fetched was ever used) argues for off.
    // NINFER_CONTROLLER_PREFETCH_OFF is the pinned-arm opt-out (the same diagnostic pattern as
    // NINFER_CONTROLLER_SERIAL_OFF): the rule holds its current state forever, so an A/B can
    // compare a controller-driven read-ahead against a fixed one.
    [[nodiscard]] bool want_prefetch() const {
        static const bool pinned_off = std::getenv("NINFER_CONTROLLER_PREFETCH_OFF") != nullptr;
        if (pinned_off) { return prefetch.enabled; } // the operator pinned the switch
        if (signals.prefetch_usefulness < 0.0) { return true; } // nothing measured yet: explore
        if (mode == ninfer::ControllerMode::Stable) {
            return signals.prefetch_usefulness >= 0.10; // stable: 低有用率即关 (D4's threshold)
        }
        return signals.prefetch_usefulness > 0.0;
    }
    // Q20 / F3: this used to gate on `capturable` alone, which is a capacity statement. On an 8 GiB
    // pool capturable is 0% by construction, so the gate could never open there even on a layer that
    // needed no host at all - and on a larger pool it could open on a layer that still fetched and
    // published, which is worse. The gate is now the AND of the capacity statement and the measured
    // execution statement (capt_ready), which is the condition a captured whole layer really needs.
    //
    // Q30: the 90/95 percentages were a guess and have been replaced by the measurement they were
    // standing in for. At 19.4 GiB (the only tier that can capture today), two interleaved rounds:
    // graph on 43.6 / 43.7 tok/s, graph off 41.5 / 41.6 tok/s - the graph is worth +5.0%. The
    // condition is now structural rather than a rate: a captured whole layer replays a table, so
    // EVERY layer has to be host-free and static, not 90% of them. 100% is the capture's validity
    // requirement; anything less means a replayed layer would read a table it was not captured with.
    //
    // port (sm_89), phase two - why this actuator stays SHADOW: the +5% above is the WHOLE-STEP
    // decode graph, and that is a construction-time decision with no runtime flip path:
    //   * ProgramImpl::use_cuda_graph is a const member, frozen by the planner in
    //     model_instance.cpp's graph_ok (options.use_cuda_graph = graph_ok), so the object that
    //     would receive a set() has no writable field to receive it;
    //   * the capture body is chosen once at construction (graphs.cpp builds its plan under the
    //     use_cuda_graph branch); "runtime on" would mean capturing a step body that was not
    //     built to be captured;
    //   * the split path's host fetch waits on cudaStreamSynchronize inside the step
    //     (ffn.cpp, the line the NINFER_GRAPH_BUDGET=1 construction crash lands on), and a
    //     synchronize inside a capture is cudaErrorStreamCaptureUnsupported - so "runtime on"
    //     on any tier that still splits is structurally impossible, and on the only tier that
    //     does not (19.4 GiB) the graph is already on by construction.
    // The graph rule, decided by this controller and consumed by the construction decision:
    //   * measured capture state: capture when every layer is capturable AND fully covered -
    //     a structural fact, not a profit threshold;
    //   * nothing measured yet (construction asks before the first layer runs): ON. The old rule
    //     defaulted an unproven graph to OFF; the measured benefit of the whole-step graph is
    //     small, but it is not negative, and a capture that the guard below skips costs nothing -
    //     so the default is now on and only a measured structural failure (a layer that needs a
    //     host fetch inside the step) turns it off.
    [[nodiscard]] bool want_graph() const {
        if (signals.capt_ready_pct >= 0.0) {
            return signals.capt_ready_pct >= 100.0 && signals.static_pct >= 100.0;
        }
        if (signals.capturable_pct >= 0.0) {
            return signals.capturable_pct >= 100.0 && signals.static_pct >= 100.0;
        }
        return true;
    }
    // Q11/Q12: the 99% threshold is gone here too. Freezing is a state - the pool answered every
    // route in the window - and the keep/thaw decision is made by ExpertStream from the measured
    // static/tps after 32 layers. This actuator is the shadow view of that decision, so a cumulative
    // no_miss below 100% means "not a freeze candidate"; it never invents a percentage.
    [[nodiscard]] bool want_freeze() const { return signals.no_miss_pct >= 100.0; }

    // A managed warm copy pays a fill per admission and saves a read per hit, so it wins when
    // hit_ratio * cold_read > fill. Both sides are measured; the margin keeps it from flapping.
    [[nodiscard]] bool want_warm_mirror() const {
        if (signals.mirror_hit_ratio <= 0.0 || signals.mirror_fill_us_per_expert <= 0.0) {
            return false;
        }
        return signals.mirror_hit_ratio * signals.cold_read_us_per_expert >
               signals.mirror_fill_us_per_expert * 1.2;
    }

    // v8 9.3 - the serial rule. Measured on this box (same harness, only --max-concurrency
    // changes): 8 GiB pool (42.8% static) concurrent -39%; 12 GiB (58.2%) a wash; 17 GiB (84.3%)
    // concurrent +13%. The flip point sits between the wash and the win at 80% static - not a
    // percentage pulled from a table, the boundary of the region where the measurement says
    // multi-lane serving pays.
    //
    //   static < 80  && lanes > 1   -> serial (multi-lane admission is buying misses)
    //   static >= 80 && serial      -> rise is allowed (drive_concurrency re-verifies tps)
    //   anything else               -> hold the current state; a no-op flip is churn
    //
    // NINFER_CONTROLLER_SERIAL_OFF is the opt-out for a pinned-concurrency A/B arm (the same
    // diagnostic pattern as NINFER_STATIC_FAST_OFF): the rule holds its current state forever,
    // which is exactly "the operator chose the lane count". It is not a mode - a mode changes
    // what the controller optimizes for, this removes one actuator from the experiment.
    [[nodiscard]] bool want_serial() const {
        static const bool off = std::getenv("NINFER_CONTROLLER_SERIAL_OFF") != nullptr;
        if (off) { return concurrency.enabled; }
        if (signals.static_pct < 0.0) { return concurrency.enabled; } // unmeasured: hold
        if (signals.static_pct < 80.0) {
            if (signals.lanes_busy > 1) { return true; }
            return concurrency.enabled; // already serial-shaped: hold
        }
        // static >= 80: multi-lane serving can pay.
        if (mode == ninfer::ControllerMode::Stable) {
            return concurrency.enabled; // stable only lowers; a rise is refused outright
        }
        if (concurrency.enabled) {
            // Serial in force. Lanes > 1 means in-flight lanes are still draining - hold until
            // the state we are judging is the state we are in.
            return signals.lanes_busy > 1 ? true : false;
        }
        return false; // concurrent in force, healthy: keep it
    }

    // v8 12.4: score = tps - lambda1*p99_latency - lambda2*miss*miss_cost. No per-request p99
    // collection point exists yet, so the latency term uses the layer-wall EWMA x 40 as the
    // measured proxy (and 1000/tps when even that is absent, i.e. the timing switch is off); the
    // panel prints which proxy is in force. The lambdas are first-phase anchors, printable and
    // mode-scaled rather than tuned: stable quadruples the latency weight, performance nearly
    // zeroes it and doubles the miss weight. The decisions this phase come from the three measured
    // criteria; the score is the common denominator the later cost model (v8 12.5 step 3) needs.
    [[nodiscard]] double lambda_p99() const {
        switch (mode) {
        case ninfer::ControllerMode::Stable: return 2.0;
        case ninfer::ControllerMode::Performance: return 0.1;
        case ninfer::ControllerMode::Auto: break;
        }
        return 0.5;
    }
    [[nodiscard]] double lambda_miss() const {
        return mode == ninfer::ControllerMode::Performance ? 2.0 : 1.0;
    }
    [[nodiscard]] double latency_ms_per_token(const char** proxy = nullptr) const {
        if (signals.wall_ms > 0.0) {
            if (proxy != nullptr) { *proxy = "layer-wall x40"; }
            return signals.wall_ms * 40.0;
        }
        if (proxy != nullptr) { *proxy = "1000/tps (no layer timing)"; }
        return signals.decode_tps > 0.0 ? 1000.0 / signals.decode_tps : 0.0;
    }
    [[nodiscard]] double miss_cost_ms_per_token() const {
        return signals.miss_per_layer * signals.upload_us_per_expert * 40.0 / 1000.0;
    }
    [[nodiscard]] double score() const {
        if (signals.decode_tps <= 0.0) { return 0.0; }
        return signals.decode_tps - lambda_p99() * latency_ms_per_token() -
               lambda_miss() * miss_cost_ms_per_token();
    }

    // v8 12.5 step 4 - the L1 slow dial for the expert pool. One 512 MiB step at most per period
    // (60 s in auto, 30 s in performance, never in stable), judged by the window score (v8 12.4
    // step 2) with the same 3% tps guardrail drive() uses; only a direction that measured a
    // payoff is re-tried, and a direction that measured worse is reverted and backed off
    // exponentially. The application is deliberately indirect: bus.set parks the target with
    // the engine worker's executor, and the worker applies it through Engine::resize_expert_
    // pool's idle contract at its own scheduling point - a refusal there is logged and backs
    // the dial off, never bypassed. The dial re-reads the bus each window, so a refused or
    // clamped resize (the worker notes the real capacity back into the bus) is what the next
    // verdict actually sees.
    //
    // MEASURED CEILING (2026-11-06, e2, initial 8192 MiB, 4-lane traffic): ExpertCache::resize
    // is elastic only within the address reservation made at CONSTRUCTION (slots_max = the
    // construction capacity, expert_cache.cpp), so a step above the construction capacity is
    // clamped to it - and the slot alignment can even come out marginally BELOW the starting
    // capacity (8023 -> 8021 MiB), which the verdict then correctly reads as worse and reverts.
    // Upward crawling beyond the construction value is therefore structurally unreachable;
    // the dial's real headroom is [lo, slots_max], and a deployment that wants the pool to
    // reach a higher tier must reserve that tier at construction (the environment stays the
    // initial value channel, v8 12.1). The full chain WAS exercised live: park 8535 -> idle
    // window apply (clamped to 8021) -> ledger tps 117.9 -> 103.5 (-12%, past the 3% rail) ->
    // revert + exponential park. Under SATURATED traffic the idle window never arrives, so a
    // parked target simply waits - the dial does not treat "not yet" as a refusal (measured:
    // 88.8 s of back-to-back 2-lane traffic, zero applies, no bypass).
    struct SlowPoolDial {
        static constexpr double kStepMb = 512.0; // v8 12.3: 1024-19456, step 512
        double from_mb       = -1.0;   // capacity before the in-flight step
        double score_before  = 0.0;   // window score when the step was taken
        double tps_before    = -1.0;   // and the tps, for the shared 3% guardrail
        bool in_flight       = false;  // a step the bus accepted, verdict pending
        bool last_up         = false;
        bool up_paid         = false;  // a step up measured a better window (v8 12.5: only a
                                        // direction that measured a payoff moves again)
        bool stepped_once    = false;
        std::uint32_t park_windows    = 0;
        std::uint32_t refused_streak  = 0; // consecutive steps the idle contract refused
        std::uint32_t falsified       = 0; // consecutive steps that measured worse
        std::chrono::steady_clock::time_point last_step{};
        // The dial's own epoch: the FIRST step waits out the observation period too, so a
        // cold-start window (a warming miss EWMA, an artifact still paging in) cannot be the
        // baseline a step is judged on.
        std::chrono::steady_clock::time_point started_at{};
        bool started = false;
        // Silent-return diagnostics (printed on the [ctl] pool line): the dial's guards return
        // without a trace, so a window that never steps needs its blocker named.
        std::uint64_t blocked_prefill = 0;
        std::uint64_t blocked_tps = 0;
        std::uint64_t blocked_miss = 0;
        std::uint64_t blocked_period = 0;
    } pool_dial;

    // 13.4 - the KV half of the L1 slow dial. Same lifecycle as the pool dial (one step per
    // period, one decision window between verdicts, the 3% tps rail, exponential park after a
    // falsified step, disabled in stable), with the direction inverted: the pool dial can only
    // grow its own reservation, the KV edge can only be spent down and won back. The rule is
    // structural rather than a tuned constant: an edge the last window never approached (occupancy
    // under a half) is device memory the expert pool could be serving from, so one step is spent
    // to find out; the 3% rail buys it back the moment the measurement says the step cost more
    // than it saved. Because the physical backing is never returned, growing back is exact
    // (DeviceKVPagePool::grow_capacity), which is what makes the revert a real experiment rather
    // than a one-way door.
    struct SlowKvDial {
        static constexpr double kOccupancyPct = 50.0; // "the edge is not being used"
        double from_pages    = -1.0; // edge before the in-flight step
        double score_before  = 0.0;
        double tps_before    = -1.0;
        bool in_flight       = false;
        bool last_down       = false;
        bool down_paid       = false; // a shrink measured a better window
        std::uint32_t park_windows   = 0;
        std::uint32_t refused_streak = 0;
        std::uint32_t falsified      = 0;
        std::chrono::steady_clock::time_point last_step{};
        bool started   = false;
        std::uint64_t blocked_prefill = 0;
        std::uint64_t blocked_tps     = 0;
        std::uint64_t blocked_period  = 0;
        std::uint64_t blocked_occupancy = 0;
        std::uint64_t steps = 0;      // applied steps this process
        std::uint64_t reverts = 0;    // steps undone by the rail
    } kv_dial;

    // Test seam: the dial's period is wall-clock, so the no-GPU tests drive it themselves.
    std::chrono::steady_clock::time_point (*clock_now)() = &std::chrono::steady_clock::now;

    void slow_pool_step() {
        auto& bus  = param_bus();
        auto& dial = pool_dial;
        if (!bus.has_applier("expert_pool_mb")) { return; } // no engine worker on this thread
        if (mode == ninfer::ControllerMode::Stable) {
            return; // stable: the dial is disabled, the pool keeps its initial value
        }
        if (signals.decode_tps <= 0.0) {
            ++dial.blocked_tps;
            return; // nothing measurable to judge by
        }
        // The same lesson the first live concurrency A/B taught (2026-11-05): a prefill layer's
        // interval carries a whole chunk, so its tps reads 0.4-9 tok/s and its miss count is a
        // cold-read artifact number. A step taken - or judged - inside a prefill window would
        // freeze a garbage baseline that any later, healthy window "improves on", so the dial
        // neither steps nor verifies there.
        if (signals.in_prefill) {
            ++dial.blocked_prefill;
            return;
        }
        if (dial.park_windows > 0) {
            --dial.park_windows;
            return;
        }
        const double now_mb = bus.get("expert_pool_mb");
        if (!(now_mb > 0.0)) { return; }
        const auto now    = clock_now();
        const auto period = mode == ninfer::ControllerMode::Performance ? std::chrono::seconds{30}
                                                                      : std::chrono::seconds{60};
        // The observation clock starts at the dial's first window; the first step waits out one
        // full period on it (the cold-start rule above).
        if (!dial.started) {
            dial.started     = true;
            dial.started_at  = now;
            dial.last_step   = now;
            dial.stepped_once = true; // gates the first step behind the same period check
            return;
        }
        if (now - dial.last_step < period) {
            ++dial.blocked_period;
            return;
        }

        if (dial.in_flight) {
            // Verdict on the in-flight step. The engine's idle contract may have refused it
            // after the bus accepted it: the worker notes the pre-step capacity back into the
            // bus, which is how the dial sees a refusal without sharing a lock with the engine.
            if (now_mb == dial.from_mb) {
                dial.in_flight    = false;
                ++dial.refused_streak;
                dial.park_windows = 4U << std::min(dial.refused_streak - 1U, 3U);
                std::fprintf(stderr,
                             "[ctl] pool dial %.0f MiB refused by the idle contract (streak %u, "
                             "parking %u windows) mode=%s\n",
                             dial.from_mb + SlowPoolDial::kStepMb, dial.refused_streak,
                             dial.park_windows, ninfer::controller_mode_name(mode));
                std::fflush(stderr);
                return;
            }
            dial.in_flight     = false;
            const double score_now = score();
            const bool worse = score_now < dial.score_before ||
                               (dial.tps_before > 0.0 &&
                                signals.decode_tps < dial.tps_before * 0.97);
            if (worse) {
                // Undo the step: the same 3% rule drive() uses, judged on the score the plan
                // names (v8 12.4). The direction measured worse, so it loses its "paid" mark
                // and the next probe backs off exponentially.
                const ParamReject reject =
                    bus.set("expert_pool_mb", dial.from_mb, /*from_controller=*/true);
                if (reject == ParamReject::Ok) {
                    bus.record("expert_pool_mb", now_mb, dial.from_mb, signals.decode_tps, -1.0);
                }
                if (dial.last_up) { dial.up_paid = false; }
                ++dial.falsified;
                dial.park_windows = 4U << std::min(dial.falsified - 1U, 3U);
                dial.last_step    = now;
                std::fprintf(stderr,
                             "[ctl] pool dial step back %.0f -> %.0f MiB (score %.1f -> %.1f, "
                             "tps %.1f -> %.1f; park %u windows) mode=%s\n",
                             now_mb, dial.from_mb, dial.score_before, score_now, dial.tps_before,
                             signals.decode_tps, dial.park_windows,
                             ninfer::controller_mode_name(mode));
                std::fflush(stderr);
                return;
            }
            // The step paid: this direction has now measured a payoff (v8 12.5).
            dial.up_paid       = dial.up_paid || dial.last_up;
            dial.falsified     = 0;
            dial.refused_streak = 0;
        }

        // Direction: the only pool-related cost the score can see is misses (v8 12.4 step 3's
        // "would moving back be better"), and the dial only moves UP on its own - a step down
        // exists as the revert above, which is where "both directions" get their measurement.
        // No miss cost means the pool already answers everything: turning the dial can only
        // introduce misses, so it holds.
        if (miss_cost_ms_per_token() <= 0.0) {
            ++dial.blocked_miss;
            return;
        }
        const double hi     = bus.hi_of("expert_pool_mb");
        const double target = now_mb + SlowPoolDial::kStepMb;
        if (target > hi) { return; } // the hard range edge (v8 12.1)
        const ParamReject reject = bus.set("expert_pool_mb", target, /*from_controller=*/true);
        if (reject != ParamReject::Ok) {
            // The bus owns the ladder (range/cooldown/executor); a refusal parks nothing - the
            // next window re-evaluates. Cooldown cannot fire at a 60 s cadence; this catch is
            // for the executor being detached mid-run (engine teardown).
            std::fprintf(stderr, "[ctl] pool dial refused by the bus: %s\n",
                         param_reject_text(reject));
            std::fflush(stderr);
            return;
        }
        bus.record("expert_pool_mb", now_mb, target, signals.decode_tps, -1.0);
        dial.from_mb      = now_mb;
        dial.score_before = score();
        dial.tps_before   = signals.decode_tps;
        dial.in_flight    = true;
        dial.last_up     = true;
        dial.stepped_once = true;
        dial.last_step    = now;
        std::fprintf(stderr,
                     "[ctl] pool dial up %.0f -> %.0f MiB (score %.1f, miss %.2f/layer, "
                     "miss_cost %.3f ms/tok) mode=%s\n",
                     now_mb, target, dial.score_before, signals.miss_per_layer,
                     miss_cost_ms_per_token(), ninfer::controller_mode_name(mode));
        std::fflush(stderr);
    }

    // The KV dial's period: v8 12.3's L1 cadence (60 s auto, 30 s performance), overridable so a
    // bounded acceptance can watch a real step land without waiting two minutes of wall clock.
    // Read once per process - the same rule the bus applies to its initial values.
    static std::chrono::seconds kv_dial_period() {
        static const std::chrono::seconds period = [] {
            if (const char* text = std::getenv("NINFER_KV_DIAL_PERIOD_MS"); text != nullptr) {
                const long long ms = std::strtoll(text, nullptr, 10);
                if (ms > 0) { return std::chrono::seconds{ms / 1000 + (ms % 1000 ? 1 : 0)}; }
            }
            return std::chrono::seconds{60};
        }();
        return period;
    }

    void slow_kv_step() {
        auto& bus  = param_bus();
        auto& dial = kv_dial;
        if (!bus.has_applier("kv_pages")) { return; }
        if (mode == ninfer::ControllerMode::Stable) { return; } // stable: KV keeps its edge
        if (signals.decode_tps <= 0.0) {
            ++dial.blocked_tps;
            return;
        }
        if (signals.in_prefill) {
            ++dial.blocked_prefill;
            return;
        }
        if (dial.park_windows > 0) {
            --dial.park_windows;
            return;
        }
        const double now_pages = bus.get("kv_pages");
        if (!(now_pages > 0.0)) { return; }
        const auto now = clock_now();
        auto period    = kv_dial_period();
        if (mode == ninfer::ControllerMode::Performance) { period = period / 2; }
        if (!dial.started) {
            dial.started   = true;
            dial.last_step = now;
            return; // the first step waits out one full observation period
        }
        if (now - dial.last_step < period) {
            ++dial.blocked_period;
            return;
        }
        if (dial.in_flight) {
            dial.in_flight = false;
            // The engine's idle contract may have refused the parked step: the worker notes the
            // pre-step edge back into the bus, which is how the dial sees a refusal without
            // sharing a lock with the engine.
            if (now_pages == dial.from_pages) {
                ++dial.refused_streak;
                dial.park_windows = 4U << std::min(dial.refused_streak - 1U, 3U);
                std::fprintf(stderr,
                             "[ctl] kv dial %.0f pages refused by the idle contract (streak %u, "
                             "parking %u windows) mode=%s\n",
                             dial.from_pages, dial.refused_streak, dial.park_windows,
                             ninfer::controller_mode_name(mode));
                std::fflush(stderr);
                return;
            }
            const double score_now = score();
            const bool worse = score_now < dial.score_before ||
                               (dial.tps_before > 0.0 &&
                                signals.decode_tps < dial.tps_before * 0.97);
            if (worse) {
                // Undo it: the edge goes back to where the step found it. Unlike the pool's
                // revert this one is exact, because the backing was never unmapped.
                const ParamReject reject =
                    bus.set("kv_pages", dial.from_pages, /*from_controller=*/true);
                if (reject == ParamReject::Ok) {
                    bus.record("kv_pages", now_pages, dial.from_pages, signals.decode_tps, -1.0);
                    ++dial.reverts;
                } else {
                    std::fprintf(stderr, "[ctl] kv dial revert refused by the bus: %s\n",
                                 param_reject_text(reject));
                    std::fflush(stderr);
                }
                dial.down_paid = false;
                ++dial.falsified;
                dial.park_windows = 4U << std::min(dial.falsified - 1U, 3U);
                dial.last_step    = now;
                std::fprintf(stderr,
                             "[ctl] kv dial step back %.0f -> %.0f pages (score %.1f -> %.1f, "
                             "tps %.1f -> %.1f; park %u windows) mode=%s\n",
                             now_pages, dial.from_pages, dial.score_before, score_now,
                             dial.tps_before, signals.decode_tps, dial.park_windows,
                             ninfer::controller_mode_name(mode));
                std::fflush(stderr);
                return;
            }
            dial.down_paid      = true;
            dial.falsified      = 0;
            dial.refused_streak = 0;
        }
        // Direction: spend the edge down while the measured occupancy says nobody is using it.
        // An unmeasured occupancy (-1) or one above the half means the edge is working: hold.
        if (signals.kv_occupancy < 0.0 ||
            signals.kv_occupancy * 100.0 >= SlowKvDial::kOccupancyPct) {
            ++dial.blocked_occupancy;
            return;
        }
        const double floor_pages = bus.lo_of("kv_pages");
        const double step        = std::max(1.0, std::floor(bus.hi_of("kv_pages") / 8.0));
        const double target      = std::max(floor_pages, now_pages - step);
        if (!(target < now_pages)) { return; } // already at the floor
        const ParamReject reject = bus.set("kv_pages", target, /*from_controller=*/true);
        if (reject != ParamReject::Ok) {
            std::fprintf(stderr, "[ctl] kv dial refused by the bus: %s\n",
                         param_reject_text(reject));
            std::fflush(stderr);
            return;
        }
        bus.record("kv_pages", now_pages, target, signals.decode_tps, -1.0);
        dial.from_pages   = now_pages;
        dial.score_before = score();
        dial.tps_before   = signals.decode_tps;
        dial.in_flight    = true;
        dial.last_down    = true;
        dial.last_step    = now;
        ++dial.steps;
        std::fprintf(stderr,
                     "[ctl] kv dial down %.0f -> %.0f page(s) (occupancy %.0f%%, score %.1f) "
                     "mode=%s\n",
                     now_pages, target, signals.kv_occupancy * 100.0, dial.score_before,
                     ninfer::controller_mode_name(mode));
        std::fflush(stderr);
    }

    void step(bool device_path_ready, bool prefetch_wired, bool graph_wired, bool freeze_wired,
              bool concurrency_wired = false) {
        ++layers;
        auto& bus = param_bus();
        // performance mode keeps exploring the read-ahead (v8 9.2, prefetch row): once the park
        // from the last revert has expired and the measured usefulness is still dead, mark it
        // unmeasured again so want_prefetch() explores instead of idling at a ratio of zero -
        // a gate that can never re-open after one bad window is not a policy.
        if (mode == ninfer::ControllerMode::Performance && !prefetch.enabled &&
            prefetch.parked_layers == 0 && signals.prefetch_usefulness >= 0.0) {
            if (++pf_idle_layers_ >= 8U * 32U) {
                signals.prefetch_usefulness = -1.0;
                pf_idle_layers_             = 0;
            }
        } else {
            pf_idle_layers_ = 0;
        }
        drive(expert_path, want_device_path(), device_path_ready, "path", "cost model");
        // v8 12.2/12.4 step 4: the prefetch flip is bus-mediated - ParamBus::set owns the ladder
        // (range/cooldown/executor) and the ledger entry, so the flip is printable and refusable
        // like every other controller write instead of a direct call into the queue.
        drive(prefetch, want_prefetch(), prefetch_wired, "pf", "usefulness measured",
              /*bus_name=*/"prefetch_on");
        drive(graph, want_graph(), graph_wired, "graph", "capt_ready==100 && static==100");
        drive(warm_mirror, want_warm_mirror(), true, "warm", "hit_ratio*cold>fill");
        // Freeze (v8 9.2 row 3): when the mechanism is armed the stream owns the decision, so the
        // actuator mirrors the real state and the panel marks it live; the controller's own
        // criterion stays in the "would:" line so the two remain comparable. When nothing is armed
        // the actuator stays the shadow view of the same state criterion.
        if (freeze_wired) {
            freeze_set.enabled = signals.freeze_active > 0.0;
            freeze_set.wired   = true;
            freeze_set.reason  = "live: stream freeze rule";
            freeze_set.agree   = 0;
        } else {
            drive(freeze_set, want_freeze(), false, "freeze",
                  "32 clean layers; keep if static or tps holds");
        }
        drive_concurrency(concurrency_wired);
        // Bus accounting: publish what the live mechanisms actually hold, and settle the last
        // ledger entry's tps at the next window boundary (v8 12.4 step 6). freeze_on mirrors
        // the armed SWITCH (the parameter); the frozen state is a signal and prints as
        // freeze_real. prefetch_on needs NO mirror: the queue's gate is applied synchronously by
        // its bus executor, so the bus value is the mechanism's state by construction - a mirror
        // here would only give a not-yet-aligned actuator a second write path.
        bus.note_value("freeze_on", freeze_wired ? 1.0 : 0.0);
        if (layers % 32 == 0) {
            bus.settle_pending(signals.decode_tps);
            // v8 12.5 step 4: the L1 slow dial runs on the same window beat, once per window,
            // rate-limited by its own wall-clock period (60 s auto / 30 s performance).
            slow_pool_step();
            // 13.4: the KV half, same beat and same rail, inverted direction.
            slow_kv_step();
        }
    }

    void report(bool force = false) {
        if (!force && layers % 40 != 0) { return; }
        char s[8];
        char n[8];
        char c[8];
        char f[8];
        char r[8];
        fmt_pct(s, sizeof(s), signals.static_pct);
        fmt_pct(n, sizeof(n), signals.no_miss_pct);
        fmt_pct(c, sizeof(c), signals.capturable_pct);
        fmt_pct(f, sizeof(f), signals.capt_free_pct);
        fmt_pct(r, sizeof(r), signals.capt_ready_pct);
        std::fprintf(stderr,
                     "[ctl] fetch=%.0fus upload=%.0fus/exp read=%.0fus/exp miss=%.2f | "
                     "wall=%.3fms gpu=%.3fms ens=%.3fms | cov static=%s no_miss=%s capt=%s "
                     "capt_free=%s capt_ready=%s | "
                     "path=%s(%s) pf=%s(%s) graph=%s(%s) serial=%s(%s) freeze=%s(%s) | tps=%.1f\n",
                     signals.fetch_us, signals.upload_us_per_expert, signals.read_us_per_expert,
                     signals.miss_per_layer, signals.wall_ms, signals.gpu_ms, signals.ens_ms, s, n,
                     c, f, r,
                     on_off(expert_path.enabled), expert_path.wired ? "live" : "shadow",
                     on_off(prefetch.enabled), prefetch.wired ? "live" : "shadow",
                     on_off(graph.enabled), graph.wired ? "live" : "shadow",
                     on_off(concurrency.enabled), concurrency.wired ? "live" : "shadow",
                     on_off(freeze_set.enabled), freeze_set.wired ? "live" : "shadow",
                     signals.decode_tps < 0.0 ? 0.0 : signals.decode_tps);
        std::fprintf(stderr,
                     "[ctl] would: path=%s (host=%.0fus vs device=%.0fus) pf=%s graph=%s freeze=%s "
                     "serial=%s\n",
                     want_device_path() ? "device" : "host", host_cost_us(), device_cost_us(),
                     want_prefetch() ? "on" : "off", want_graph() ? "on" : "off",
                     want_freeze() ? "on" : "off", want_serial() ? "serial" : "concurrent");
        std::fprintf(stderr,
                     "[ctl] warm: cold=%.0fus fill=%.0fus hit=%.2f -> mirror=%s (%.2f vs %.2f)\n",
                     signals.cold_read_us_per_expert, signals.mirror_fill_us_per_expert,
                     signals.mirror_hit_ratio, want_warm_mirror() ? "on" : "off",
                     signals.mirror_hit_ratio * signals.cold_read_us_per_expert,
                     signals.mirror_fill_us_per_expert * 1.2);
        // v8 12.4 + the mode semantics: one line per window with the deployment mode, the score
        // and its weights, the serial state with the lanes it was judged on, and the real freeze
        // state next to the actuator's own view.
        const char* proxy = "";
        const double lat  = latency_ms_per_token(&proxy);
        std::fprintf(stderr,
                     "[ctl] mode=%s score=%.1f = tps %.1f - %.2f*%.2fms(lat, %s) - %.2f*%.3fms | "
                     "serial=%s(%s, %s) lanes=%u freeze_real=%s\n",
                     ninfer::controller_mode_name(mode), score(),
                     signals.decode_tps < 0.0 ? 0.0 : signals.decode_tps, lambda_p99(), lat, proxy,
                     lambda_miss(), miss_cost_ms_per_token(),
                     on_off(concurrency.enabled), concurrency.wired ? "live" : "shadow",
                     concurrency.reason != nullptr ? concurrency.reason : "n/a",
                     signals.lanes_busy,
                     signals.freeze_active < 0.0
                         ? "n/a"
                         : (signals.freeze_active > 0.0 ? "frozen" : "streaming"));
        // v8 12.5 step 4: the slow dial's own line - what it holds, what it is waiting for, and
        // how many windows the idle contract has cost it. Printed every panel pass regardless of
        // mode so a stable run says "disabled" instead of saying nothing.
        {
            auto& bus  = param_bus();
            auto& dial = pool_dial;
            const bool attached = bus.has_applier("expert_pool_mb");
            std::fprintf(stderr,
                         "[ctl] pool: capacity=%g MiB dial=%s in_flight=%d up_paid=%d "
                         "park=%u refused=%u falsified=%u | miss_cost=%.3f ms/tok score=%.1f "
                         "| windows: prefill=%llu tps=%llu period=%llu miss=%llu\n",
                         bus.get("expert_pool_mb"),
                         !attached                      ? "no-executor"
                         : mode == ninfer::ControllerMode::Stable ? "stable:disabled"
                         : dial.in_flight ? "verdict-pending"
                                          : "idle",
                         int(dial.in_flight), int(dial.up_paid), dial.park_windows,
                         dial.refused_streak, dial.falsified, miss_cost_ms_per_token(), score(),
                         static_cast<unsigned long long>(dial.blocked_prefill),
                         static_cast<unsigned long long>(dial.blocked_tps),
                         static_cast<unsigned long long>(dial.blocked_period),
                         static_cast<unsigned long long>(dial.blocked_miss));
        }
        // 13.4: the KV dial's own line, same discipline: what the edge is, what the dial is
        // waiting for, and every window that did not step names its blocker.
        {
            auto& bus  = param_bus();
            auto& dial = kv_dial;
            const bool attached = bus.has_applier("kv_pages");
            std::fprintf(stderr,
                         "[ctl] kv: pages=%g occupancy=%.0f%% dial=%s in_flight=%d down_paid=%d "
                         "park=%u refused=%u falsified=%u steps=%llu reverts=%llu | score=%.1f "
                         "| windows: prefill=%llu tps=%llu period=%llu occupancy=%llu\n",
                         bus.get("kv_pages"),
                         signals.kv_occupancy < 0.0 ? -1.0 : signals.kv_occupancy * 100.0,
                         !attached                      ? "no-executor"
                         : mode == ninfer::ControllerMode::Stable ? "stable:disabled"
                         : dial.in_flight ? "verdict-pending"
                                          : "idle",
                         int(dial.in_flight), int(dial.down_paid), dial.park_windows,
                         dial.refused_streak, dial.falsified,
                         static_cast<unsigned long long>(dial.steps),
                         static_cast<unsigned long long>(dial.reverts), score(),
                         static_cast<unsigned long long>(dial.blocked_prefill),
                         static_cast<unsigned long long>(dial.blocked_tps),
                         static_cast<unsigned long long>(dial.blocked_period),
                         static_cast<unsigned long long>(dial.blocked_occupancy));
        }
        // P2 (cpu in-place path): the CPU-leg decision observation - verdict counts, hit rates
        // and the cost model's last two sides, shadow until an actuator lands (the runtime legs
        // the verdicts name are later batches; NINFER_CPU_PATH_POLICY defaults them off).
        cpu_path_ledger().report();
        std::fflush(stderr);
    }
private:
    std::uint32_t pf_idle_layers_    = 0;
    std::uint32_t drop_revert_streak_ = 0; // consecutive refused serial probes (backoff input)
    std::uint32_t serial_idle_layers_ = 0; // serial layers since the last multi-lane observation

    static const char* on_off(bool value) { return value ? "on" : "off"; }

    static void fmt_pct(char* out, std::size_t size, double value) {
        if (value < 0.0) {
            std::snprintf(out, size, "n/a");
        } else {
            std::snprintf(out, size, "%.0f%%", value);
        }
    }

    static void print_flip(const char* name, bool was, bool now, const char* reason, double tps,
                           ninfer::ControllerMode mode_name) {
        std::fprintf(stderr, "[ctl] flip %s %s->%s (%s) tps=%.1f mode=%s\n", name,
                     was ? "on" : "off", now ? "on" : "off", reason,
                     tps < 0.0 ? 0.0 : tps, ninfer::controller_mode_name(mode_name));
        std::fflush(stderr);
    }

    // One actuator's lifecycle: agree for a layer stack, then act; remember the throughput the
    // change was made at; if the next window comes back more than 3% slower, revert and park it.
    // When bus_name is non-null the flips go through ParamBus::set (v8 12.2/12.4 step 4) - the
    // bus owns the ladder and the ledger, a refusal leaves the in-force state untouched and the
    // next window re-accumulates the agreement run.
    void drive(PolicyActuator& actuator, bool want, bool wired_now, const char* name,
               const char* reason, const char* bus_name = nullptr) {
        if (actuator.parked_layers > 0) {
            --actuator.parked_layers;
            actuator.reason = "parked";
            return;
        }
        if (want == actuator.enabled) {
            actuator.agree = 0;
            return;
        }
        if (++actuator.agree < 32) { return; }
        const bool was = actuator.enabled;
        if (actuator.baseline_tps > 0.0 && signals.decode_tps > 0.0 &&
            signals.decode_tps < actuator.baseline_tps * 0.97) {
            if (bus_name != nullptr && actuator.wired) {
                // Revert through the bus. The bus cooldown usually still holds from the flip
                // being reverted (32 layers, and the tps EWMA needs ~64 to turn over anyway);
                // the refusal keeps the old baseline so the next agreement run re-judges the
                // same degradation against the same number instead of a moved goalpost.
                if (flip_through_bus(actuator, name, bus_name, !actuator.enabled,
                                     "reverted: degraded since the last flip")) {
                    actuator.baseline_tps  = signals.decode_tps;
                    actuator.parked_layers = 40U * 32U;
                    actuator.reason        = "reverted";
                }
                return;
            }
            actuator.enabled       = !actuator.enabled;
            actuator.agree         = 0;
            actuator.baseline_tps  = signals.decode_tps;
            actuator.parked_layers = 40U * 32U;
            actuator.reason        = "reverted";
            if (actuator.wired && actuator.enabled != was) {
                print_flip(name, was, actuator.enabled, "reverted: degraded since the last flip",
                           signals.decode_tps, mode);
            }
            return;
        }
        if (bus_name != nullptr && wired_now) {
            if (flip_through_bus(actuator, name, bus_name, want, reason)) {
                actuator.baseline_tps = signals.decode_tps;
                actuator.reason        = reason;
            }
            actuator.wired = wired_now;
            return;
        }
        actuator.enabled      = want;
        actuator.agree        = 0;
        actuator.baseline_tps = signals.decode_tps;
        actuator.reason       = reason;
        if (!wired_now) {
            // Shadow: record the decision, but leave the switch where it is.
            actuator.enabled = !want;
            actuator.reason  = "shadow";
        } else if (actuator.enabled != was) {
            print_flip(name, was, actuator.enabled, reason, signals.decode_tps, mode);
        }
        actuator.wired = wired_now;
    }

    // One flip, bus-mediated: ParamBus::set owns the range check, the cooldown and the executor;
    // the applied change lands in the ledger with the flip-time throughput and the panel gets
    // the flip with its reason. A refusal returns false with the in-force state untouched (the
    // caller decides what to do with its agreement run).
    bool flip_through_bus(PolicyActuator& actuator, const char* name, const char* bus_name,
                          bool target, const char* reason) {
        auto& bus         = param_bus();
        const bool was    = actuator.enabled;
        const double from = bus.get(bus_name);
        const double to   = target ? 1.0 : 0.0;
        if (from == to) {
            // The mechanism already holds the target (e.g. the bus mirror caught up first):
            // sync the actuator to reality, print nothing, ledger nothing.
            actuator.enabled = target;
            actuator.agree   = 0;
            return true;
        }
        const ParamReject reject = bus.set(bus_name, to, /*from_controller=*/true);
        if (reject != ParamReject::Ok) {
            actuator.reason = param_reject_text(reject);
            actuator.agree  = 0;
            std::fprintf(stderr, "[ctl] %s flip refused by the bus: %s\n", name,
                         param_reject_text(reject));
            std::fflush(stderr);
            return false;
        }
        actuator.enabled = target;
        actuator.agree   = 0;
        bus.record(bus_name, from, to, signals.decode_tps, -1.0);
        print_flip(name, was, target, reason, signals.decode_tps, mode);
        return true;
    }

    // The concurrency actuator's own lifecycle. It keeps the shared shape (32-agreeing-layer
    // flips, flip-time tps baseline, revert-and-park on degradation, shadow/live marking) but
    // two things differ from the generic drive(), both learned from the first live A/B
    // (2026-11-05, 8 GiB pool, 24-request burst, this box, wall 114.5s vs the pinned 94.4s):
    //
    //   1. flips and verification run on DECODE layers only. The first cut fired during prefill
    //      windows, where the layer-interval tps reads 0.4-9 tok/s (a prefill layer pass carries
    //      a whole chunk, not one token), so every flip was made and then judged on numbers
    //      describing a different phase of work.
    //   2. a refused drop backs off exponentially. The 80% static rule was measured under one
    //      miss path (host-direct, v8 9.3); on the upload path concurrent decode amortizes
    //      uploads across lanes and serial genuinely loses. The guardrail reverts each probe;
    //      the backoff turns "the machine says no" from a 17% tax into a vanishing one while
    //      still re-probing when enough tokens have passed for the conditions to have changed.
    void drive_concurrency(bool wired) {
        PolicyActuator& actuator = concurrency;
        if (actuator.parked_layers > 0) {
            if (--actuator.parked_layers == 0) {
                actuator.baseline_tps = -1.0; // park over: nothing to verify against anymore
            }
            actuator.reason = "parked";
            return;
        }
        if (actuator.settle_layers > 0) { --actuator.settle_layers; }
        const double keep = mode == ninfer::ControllerMode::Performance ? 0.95 : 0.97;
        // Post-change verification (v8 12.4 step 5): once per window, after the tps EWMA has
        // turned over (64 layers at its 1/16 smoothing), compare the state we are in against
        // the throughput it was entered at. Stable skips it: its revert would be a rise.
        if (mode != ninfer::ControllerMode::Stable && actuator.baseline_tps > 0.0 &&
            signals.decode_tps > 0.0 && actuator.settle_layers == 0 && layers % 32 == 0 &&
            signals.lanes_busy > 0 && !signals.in_prefill) {
            if (signals.decode_tps < actuator.baseline_tps * keep) {
                const bool undone_drop = actuator.enabled; // a serial the machine refused
                flip_serial(!actuator.enabled, wired,
                            "reverted: tps below the flip-time baseline");
                if (undone_drop) {
                    // Each refused probe says the rule's premise does not hold for this traffic;
                    // a re-probe should cost less than the last one (capped at 32x the base).
                    ++drop_revert_streak_;
                    actuator.parked_layers = (40U * 32U) << std::min(drop_revert_streak_, 5U);
                } else {
                    // A refused rise: the serial state was right after all; base park, fresh tab.
                    actuator.parked_layers = 40U * 32U;
                    drop_revert_streak_    = 0;
                }
                return;
            }
            drop_revert_streak_ = 0; // the state is holding its own: future refusals start fresh
        }
        // Decisions (and their agreement runs) are made only where the measurement is a decode
        // measurement; a prefill window neither argues for serial nor against it.
        if (signals.in_prefill) {
            actuator.agree = 0;
            return;
        }
        // The stale-verdict rail: serial's purpose is to serve MULTI-lane traffic better. A
        // serial that has seen no multi-lane layer for a long stretch is holding an admission
        // gate closed on a verdict whose baseline cannot have been healthy - the first live A/B
        // produced exactly this (a cold-read-depressed concurrent baseline made single-stream
        // serial look better, and it then gated a 24-request queue for the whole run). Rise,
        // un-parked: the rule re-drops within 32 layers if the traffic still argues for it,
        // and the next probe is measured against a fresh baseline.
        if (actuator.enabled && signals.lanes_busy <= 1) {
            if (++serial_idle_layers_ >= 1024U) {
                flip_serial(false, wired, "re-verify: multi-lane traffic absent for 1024 layers");
                return;
            }
        } else {
            serial_idle_layers_ = 0;
        }
        const bool want = want_serial();
        if (want == actuator.enabled) {
            actuator.agree = 0;
            return;
        }
        if (mode == ninfer::ControllerMode::Stable && !want && actuator.enabled) {
            // stable: a rise is refused outright (only-lower semantics)
            actuator.agree = 0;
            return;
        }
        // stable lowers on the first agreeing layer - "静态<80 立即串行" - because a stable
        // deployment asked for the conservative state, not for evidence about it.
        const std::uint32_t ramp = mode == ninfer::ControllerMode::Stable ? 1U : 32U;
        if (++actuator.agree < ramp) { return; }
        flip_serial(want, wired, want ? "static<80 with lanes>1" : "static>=80, rise re-verified");
    }

    // Applies one serial/concurrent transition through the bus. Shadow runs record the decision
    // and move nothing; live runs go through ParamBus::set (which owns the range check, the
    // cooldown and the executor) and land in the ledger with the flip-time throughput.
    void flip_serial(bool serial_now, bool wired, const char* reason) {
        PolicyActuator& actuator = concurrency;
        const bool was           = actuator.enabled;
        actuator.enabled        = serial_now;
        actuator.agree          = 0;
        // 64 layers: the tps EWMA smooths at 1/16, so ~48 layers are 95% of the turnover - the
        // verification must not judge the change on a number still describing the old state.
        actuator.settle_layers  = 64;
        actuator.baseline_tps   = signals.decode_tps > 0.0 ? signals.decode_tps : -1.0;
        serial_idle_layers_     = 0;
        actuator.reason         = reason;
        if (!wired) {
            actuator.enabled = was;
            actuator.reason  = "shadow";
            actuator.wired   = false;
            return;
        }
        auto& bus = param_bus();
        const double from = bus.get("concurrency");
        const double to   = serial_now ? 1.0 : bus.rollback_of("concurrency");
        const ParamReject reject = bus.set("concurrency", to, /*from_controller=*/true);
        if (reject != ParamReject::Ok) {
            // The bus is the authority on ranges, cooldowns and accepted executors; a refusal
            // means the flip does not happen, and the panel gets the reason.
            actuator.enabled = was;
            actuator.reason  = param_reject_text(reject);
            actuator.wired   = false;
            std::fprintf(stderr, "[ctl] serial flip refused by the bus: %s\n",
                         param_reject_text(reject));
            std::fflush(stderr);
            return;
        }
        actuator.wired = true;
        bus.record("concurrency", from, to, signals.decode_tps, -1.0);
        std::fprintf(stderr,
                     "[ctl] flip serial %s->%s: concurrency %g -> %g (%s) tps=%.1f mode=%s\n",
                     was ? "on" : "off", serial_now ? "on" : "off", from, to, reason,
                     signals.decode_tps < 0.0 ? 0.0 : signals.decode_tps,
                     ninfer::controller_mode_name(mode));
        std::fflush(stderr);
    }
};

inline PolicyController& policy_controller() {
    static thread_local PolicyController state;
    return state;
}

} // namespace ninfer::models::qwen3_5::execution
