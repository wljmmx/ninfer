#pragma once

// V10-82: measure the two Host -> Device legs on THIS machine, on the payloads the engine will
// actually send, so the capacity bet can price A and B instead of ranking them by preference.
//
// Why a probe and not a constant: the payloads are the run's own (one KV page, one expert slot),
// and the two legs do not share a rate even on one card - a 1.4 MiB page copy and a 1.25 MiB slot
// copy differ by their copy count and their source buffer, and a rate taken from a 4 GiB bulk
// transfer would not be the rate a per-page fetch sees. Numbers from another machine are the one
// thing this file cannot honestly supply, so it measures and reports them with the shape they came
// from.
//
// It is deliberately cheap and it is only run where the answer is used (a run with the page-in leg
// on): `kTransferTrialBytes` per trial, three trials, the best one kept. Nothing here allocates
// Device memory that outlives the call, and a failed probe leaves both rates at zero - an unpriced
// leg is the bet's documented fallback to the priority rule, never a zero-cost claim.

#include "core/transfer.h"
#include "core/transfer_cost.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace ninfer::core {

// One trial sends at least this much, so a small payload is measured as a stream and not through
// its own launch latency - a single 1.4 MiB copy is 0.1% of a millisecond's worth of event noise.
inline constexpr std::uint64_t kTransferTrialBytes = 64ULL << 20;
inline constexpr int kTransferTrialRounds          = 3;
inline constexpr int kTransferWarmupCopies         = 4;

struct TransferRateProbe {
    TransferRates rates{};
    std::uint32_t page_copies = 0; // copies in one page-in trial (the shape the rate came from)
    std::uint32_t slot_copies = 0;
    // The largest Host allocation the probe pinned. The two legs are measured one after the other,
    // so this is the footprint, not their sum - it is printed so the probe's own cost is visible.
    std::uint64_t pinned_bytes = 0;

    [[nodiscard]] bool measured() const noexcept { return rates.measured(); }
};

namespace detail {

// One payload, one best-of-rounds rate. Returns zero when anything along the way fails: a probe
// that cannot run must not fail the run it was asked to price.
[[nodiscard]] inline double measure_h2d_copy_rate(std::uint64_t bytes, std::uint32_t copies,
                                                  cudaStream_t stream) noexcept {
    if (bytes == 0 || copies == 0) { return 0.0; }
    void* host   = nullptr;
    void* device = nullptr;
    if (cudaHostAlloc(&host, static_cast<std::size_t>(bytes), cudaHostAllocDefault) !=
        cudaSuccess) {
        return 0.0;
    }
    if (cudaMalloc(&device, static_cast<std::size_t>(bytes)) != cudaSuccess) {
        (void)cudaFreeHost(host);
        return 0.0;
    }
    // Touch the source so the copy does not read a first-touch fault pattern that the engine's own
    // (already written) buffers would not have.
    for (std::uint64_t i = 0; i < bytes; i += 4096) {
        static_cast<unsigned char*>(host)[i] = static_cast<unsigned char>(i);
    }
    cudaEvent_t begin = nullptr;
    cudaEvent_t end   = nullptr;
    double best_ms    = 0.0;
    if (cudaEventCreate(&begin) == cudaSuccess && cudaEventCreate(&end) == cudaSuccess) {
        for (int warm = 0; warm < kTransferWarmupCopies; ++warm) {
            (void)copy_bytes_status(device, host, static_cast<std::size_t>(bytes),
                                    TransferDirection::HostToDevice, stream);
        }
        (void)cudaStreamSynchronize(stream);
        for (int round = 0; round < kTransferTrialRounds; ++round) {
            (void)cudaEventRecord(begin, stream);
            for (std::uint32_t copy = 0; copy < copies; ++copy) {
                (void)copy_bytes_status(device, host, static_cast<std::size_t>(bytes),
                                        TransferDirection::HostToDevice, stream);
            }
            (void)cudaEventRecord(end, stream);
            if (cudaEventSynchronize(end) != cudaSuccess) { break; }
            float elapsed = 0.0F;
            if (cudaEventElapsedTime(&elapsed, begin, end) != cudaSuccess) { break; }
            const double ms = static_cast<double>(elapsed);
            if (ms > 0.0 && (best_ms == 0.0 || ms < best_ms)) { best_ms = ms; }
        }
        (void)cudaEventDestroy(begin);
        (void)cudaEventDestroy(end);
    }
    (void)cudaFree(device);
    (void)cudaFreeHost(host);
    if (best_ms <= 0.0) { return 0.0; }
    const double moved = static_cast<double>(bytes) * static_cast<double>(copies);
    return moved / (best_ms / 1000.0);
}

[[nodiscard]] inline std::uint32_t copies_for_trial(std::uint64_t bytes) noexcept {
    if (bytes == 0) { return 0; }
    const std::uint64_t wanted = (kTransferTrialBytes + bytes - 1) / bytes;
    return static_cast<std::uint32_t>(std::clamp<std::uint64_t>(wanted, 4, 4096));
}

} // namespace detail

// Both legs, on the two payload shapes the run will send. Zero rates mean "not measured".
[[nodiscard]] inline TransferRateProbe measure_transfer_rates(std::uint64_t page_bytes,
                                                              std::uint64_t slot_bytes,
                                                              cudaStream_t stream) noexcept {
    TransferRateProbe probe;
    probe.page_copies = detail::copies_for_trial(page_bytes);
    probe.slot_copies = detail::copies_for_trial(slot_bytes);
    probe.rates.page_in_bytes_per_s =
        detail::measure_h2d_copy_rate(page_bytes, probe.page_copies, stream);
    probe.rates.expert_bytes_per_s =
        detail::measure_h2d_copy_rate(slot_bytes, probe.slot_copies, stream);
    probe.pinned_bytes = std::max<std::uint64_t>(page_bytes, slot_bytes);
    return probe;
}

} // namespace ninfer::core
