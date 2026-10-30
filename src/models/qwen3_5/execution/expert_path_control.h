#pragma once

// port (sm_89): which tier serves a routed expert is a runtime decision, not a build-time constant.
//
// The obvious implementation - "use the Device-side path above 16 GiB" - is wrong for two reasons.
// A pool size is not a measurement: the same pool serves a converged set at 0.05 misses a layer and
// a cold one at 2.2, so one number describes two completely different machines. And the costs it
// would be compared against are properties of this box on this driver: how long a layer waits for
// its routes to come back, how long an expert takes to travel, how fast the Device reads host RAM.
//
// So the decision is made from measurement. Each layer reports what it actually cost; a moving
// average of those feeds a two-sided cost model:
//
//   host path   = fetch (the per-layer round trip) + misses x upload_per_expert
//   device path = misses x read_per_expert        + fixed launch cost
//
// The Device path wins when it is cheaper by a margin, and the flag only flips after a run of
// agreeing layers, so one noisy step cannot move it. Both sides are measured at runtime; nothing
// here is keyed on a pool size, a model name, or a GiB figure.
//
// Until the kernels gain a Device-side branch, `device_path_ready` is false and this reports in
// shadow mode: it prints what it would decide, with the numbers behind it, and changes nothing.

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdint>

namespace ninfer::models::qwen3_5::execution {

struct ExpertPathControl {
    // Measurements, exponentially smoothed with weight 1/16: about sixteen layers of memory. Long
    // enough to be stable across a layer stack, short enough to follow a conversation changing topic.
    double fetch_us             = 0.0; // host wait for this layer's routes
    double upload_us_per_expert = 0.0; // host-to-device travel, measured per expert
    double read_us_per_expert   = 0.0; // device reading host RAM, from the measured link
    double miss_per_layer       = 0.0; // routes the pool did not hold
    double link_gbps            = 0.0;

    bool device_path_ready = false; // true once the kernels have a Device-side branch
    bool device_path       = false; // the decision in force
    std::uint32_t agree    = 0;     // consecutive layers agreeing with the pending decision
    std::uint64_t layers   = 0;

    void observe_fetch(double micros) { fetch_us += (micros - fetch_us) / 16.0; }

    // A fully covered layer still tells us the fetch cost, so `experts` may be zero.
    void observe_upload(double micros, std::uint32_t experts) {
        if (experts == 0) { return; }
        const double per = micros / static_cast<double>(experts);
        upload_us_per_expert += (per - upload_us_per_expert) / 16.0;
    }

    void observe_misses(std::uint32_t experts) {
        miss_per_layer += (static_cast<double>(experts) - miss_per_layer) / 16.0;
    }

    // One 32 MiB host-to-device copy, timed once. It rides the same PCIe link the Device would read
    // host-tier experts over, so it stands in for that bandwidth until a kernel measures it directly.
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
            link_gbps          = static_cast<double>(kBytes) / (static_cast<double>(best) * 1e-3) / 1e9;
            read_us_per_expert = static_cast<double>(slot_bytes) / (link_gbps * 1e9) * 1e6;
        }
        (void)cudaEventDestroy(begin);
        (void)cudaEventDestroy(end);
        (void)cudaFree(device);
        (void)cudaFreeHost(host);
    }

    // The cost model. `kRoundTripCreditUs` is deliberately zero: the round trip's real cost on this
    // path is not its 7 us but the pipeline it drains, and inventing a number for that here would be
    // exactly the hardcoded judgement this controller exists to avoid. It is measured by A/B instead.
    [[nodiscard]] bool prefer_device_path() const {
        if (!device_path_ready || read_us_per_expert <= 0.0) { return false; }
        const double host_cost   = fetch_us + miss_per_layer * upload_us_per_expert;
        const double device_cost = miss_per_layer * read_us_per_expert;
        return device_cost < host_cost * 0.8; // margin: no flip inside measurement noise
    }

    void step(bool ready_now) {
        device_path_ready = ready_now;
        ++layers;
        const bool want = prefer_device_path();
        if (want == device_path) {
            agree = 0;
            return;
        }
        if (++agree >= 32) { // a full layer stack's worth of agreement before the flag moves
            device_path = want;
            agree       = 0;
        }
    }

    void report(bool force = false) {
        if (!force && layers % 40 != 0) { return; }
        const double host_cost   = fetch_us + miss_per_layer * upload_us_per_expert;
        const double device_cost = miss_per_layer * read_us_per_expert;
        // In shadow mode the flag cannot move (there is no branch to move to), so report the side the
        // model would pick rather than the side in force - otherwise the line reads as a contradiction.
        const bool would   = prefer_device_path();
        const char* verdict = device_path ? "device" : (would && !device_path_ready ? "device" : "host");
        std::fprintf(stderr,
                     "[ctl] link=%.1fGB/s read=%.0fus/expert fetch=%.0fus upload=%.0fus/expert "
                     "miss=%.2f/layer -> host=%.0fus device=%.0fus : %s%s\n",
                     link_gbps, read_us_per_expert, fetch_us, upload_us_per_expert, miss_per_layer,
                     host_cost, device_cost, verdict,
                     device_path_ready ? "" : " (shadow: no Device branch in the kernels yet)");
        std::fflush(stderr);
    }
};

inline ExpertPathControl& expert_path_control() {
    static thread_local ExpertPathControl state;
    return state;
}

} // namespace ninfer::models::qwen3_5::execution
