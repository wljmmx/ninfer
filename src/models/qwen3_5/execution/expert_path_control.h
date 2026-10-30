#pragma once

// port (sm_89): the runtime policy controller - the one place that decides how a running instance
// serves experts, whether it prefetches, whether it may capture graphs, and whether it keeps a
// frozen set. Every decision is taken from measurement; none is keyed on a pool size or a GiB figure.
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
//
// Each actuator follows the same shape: observe, compare against a rule, require a run of agreeing
// layers, then act - recording the throughput at the moment of acting, so a change that costs more
// than it buys is reverted and that actuator is parked for a while.
//
// Until an actuator is wired into the kernels it reports "(shadow)": it prints what it would do and
// why, and changes nothing.

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdint>

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
    double prefetch_usefulness = -1.0; // D4 gate input
    double decode_tps          = -1.0; // the outcome every actuator is judged by
};

struct PolicyActuator {
    bool enabled  = false; // value in force
    bool wired    = false; // false: shadow, the switch does not exist yet
    std::uint32_t agree = 0;
    double baseline_tps = -1.0;
    std::uint32_t parked_layers = 0;
    const char* reason = "n/a";
};

class PolicyController {
public:
    PolicySignals signals;
    PolicyActuator expert_path;
    PolicyActuator prefetch;
    PolicyActuator graph;
    PolicyActuator freeze_set;
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

    void observe_coverage(double static_pct, double no_miss_pct, double capturable_pct) {
        signals.static_pct     = static_pct;
        signals.no_miss_pct    = no_miss_pct;
        signals.capturable_pct = capturable_pct;
    }

    void observe_decode(double tps) { signals.decode_tps = tps; }

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
    [[nodiscard]] bool want_prefetch() const { return signals.prefetch_usefulness >= 0.10; }
    [[nodiscard]] bool want_graph() const {
        return signals.capturable_pct >= 90.0 && signals.static_pct >= 95.0;
    }
    [[nodiscard]] bool want_freeze() const { return signals.no_miss_pct >= 99.0; }

    void step(bool device_path_ready, bool prefetch_wired, bool graph_wired, bool freeze_wired) {
        ++layers;
        drive(expert_path, want_device_path(), device_path_ready, "cost model");
        drive(prefetch, want_prefetch(), prefetch_wired, "usefulness>=10%");
        drive(graph, want_graph(), graph_wired, "capturable>=90% && static>=95%");
        drive(freeze_set, want_freeze(), freeze_wired, "no_miss>=99%");
    }

    void report(bool force = false) {
        if (!force && layers % 40 != 0) { return; }
        char s[8];
        char n[8];
        char c[8];
        fmt_pct(s, sizeof(s), signals.static_pct);
        fmt_pct(n, sizeof(n), signals.no_miss_pct);
        fmt_pct(c, sizeof(c), signals.capturable_pct);
        std::fprintf(stderr,
                     "[ctl] fetch=%.0fus upload=%.0fus/exp read=%.0fus/exp miss=%.2f | "
                     "wall=%.3fms gpu=%.3fms ens=%.3fms | cov static=%s no_miss=%s capt=%s | "
                     "path=%s(%s) pf=%s(%s) graph=%s(%s) freeze=%s(%s) | tps=%.1f\n",
                     signals.fetch_us, signals.upload_us_per_expert, signals.read_us_per_expert,
                     signals.miss_per_layer, signals.wall_ms, signals.gpu_ms, signals.ens_ms, s, n,
                     c, on_off(expert_path.enabled), expert_path.wired ? "live" : "shadow",
                     on_off(prefetch.enabled), prefetch.wired ? "live" : "shadow",
                     on_off(graph.enabled), graph.wired ? "live" : "shadow",
                     on_off(freeze_set.enabled), freeze_set.wired ? "live" : "shadow",
                     signals.decode_tps < 0.0 ? 0.0 : signals.decode_tps);
        std::fprintf(stderr,
                     "[ctl] would: path=%s (host=%.0fus vs device=%.0fus) pf=%s graph=%s freeze=%s\n",
                     want_device_path() ? "device" : "host", host_cost_us(), device_cost_us(),
                     want_prefetch() ? "on" : "off", want_graph() ? "on" : "off",
                     want_freeze() ? "on" : "off");
        std::fflush(stderr);
    }

private:
    static const char* on_off(bool value) { return value ? "on" : "off"; }

    static void fmt_pct(char* out, std::size_t size, double value) {
        if (value < 0.0) {
            std::snprintf(out, size, "n/a");
        } else {
            std::snprintf(out, size, "%.0f%%", value);
        }
    }

    // One actuator's lifecycle: agree for a layer stack, then act; remember the throughput the
    // change was made at; if the next window comes back more than 3% slower, revert and park it.
    void drive(PolicyActuator& actuator, bool want, bool wired_now, const char* reason) {
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
        if (actuator.baseline_tps > 0.0 && signals.decode_tps > 0.0 &&
            signals.decode_tps < actuator.baseline_tps * 0.97) {
            actuator.enabled       = !actuator.enabled;
            actuator.agree         = 0;
            actuator.baseline_tps  = signals.decode_tps;
            actuator.parked_layers = 40U * 32U;
            actuator.reason        = "reverted";
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
        }
        actuator.wired = wired_now;
    }
};

inline PolicyController& policy_controller() {
    static thread_local PolicyController state;
    return state;
}

} // namespace ninfer::models::qwen3_5::execution
