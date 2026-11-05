#pragma once

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
    [[nodiscard]] bool want_prefetch() const {
        if (signals.prefetch_usefulness < 0.0) { return true; } // nothing measured yet: explore
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
    [[nodiscard]] bool want_graph() const {
        if (signals.capt_ready_pct >= 0.0) {
            return signals.capt_ready_pct >= 100.0 && signals.static_pct >= 100.0;
        }
        return signals.capturable_pct >= 100.0 && signals.static_pct >= 100.0;
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
        drive(prefetch, want_prefetch(), prefetch_wired, "pf", "usefulness measured");
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
        // ledger entry's tps at the next window boundary (v8 12.4 step 6). freeze_on mirrors the
        // armed SWITCH (the parameter); the frozen state is a signal and prints as freeze_real.
        bus.note_value("prefetch_on", prefetch.enabled ? 1.0 : 0.0);
        bus.note_value("freeze_on", freeze_wired ? 1.0 : 0.0);
        if (layers % 32 == 0) { bus.settle_pending(signals.decode_tps); }
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
    void drive(PolicyActuator& actuator, bool want, bool wired_now, const char* name,
               const char* reason) {
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
