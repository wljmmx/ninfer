#pragma once

// V10-82: measure the Host -> Device legs on THIS machine, on the payloads the engine will actually
// send, so the capacity bet can price A and B instead of ranking them by preference.
// V10-83: and measure the leg that feeds them - the artifact read.
//
// Why a probe and not a constant: the payloads are the run's own (one KV page, one expert slot),
// and the legs do not share a rate even on one card - a 1.4 MiB page copy and a 1.25 MiB slot copy
// differ by their copy count and their source buffer, and a rate taken from a 4 GiB bulk transfer
// would not be the rate a per-page fetch sees. Numbers from another machine are the one thing this
// file cannot honestly supply, so it measures and reports them with the shape they came from.
//
// V10-83: the expert leg also reads the artifact before it copies, and that read is priced now
// (core/transfer_cost.h prices the two in series). It is measured through the SAME reader and the
// SAME mode the run uses - `DiskReadSource` is a callback, because core cannot include the artifact
// layer, and the caller (runtime/engine/model_instance.cpp) supplies the engine's own Reader - at
// the expert payload size, over a few fresh regions of the file rather than one hot one.
//
// It is deliberately cheap and only run where the answer is used (a run with the page-in leg on):
// `kTransferTrialBytes` per trial plus `kDiskTrialBytes` per disk round. Nothing here allocates
// Device memory that outlives the call, and a failed probe leaves the rate it could not measure at
// zero - an unpriced leg is the bet's documented fallback, never a zero-cost claim.

#include "core/transfer.h"
#include "core/transfer_cost.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <span>
#include <vector>

namespace ninfer::core {

// One trial sends at least this much, so a small payload is measured as a stream and not through
// its own launch latency - a single 1.4 MiB copy is 0.1% of a millisecond's worth of event noise.
inline constexpr std::uint64_t kTransferTrialBytes = 64ULL << 20;
inline constexpr int kTransferTrialRounds          = 3;
inline constexpr int kTransferWarmupCopies         = 4;
// The disk leg reads fewer bytes per trial: it is the slow leg, and the bet needs its RATE, not its
// duration. 24 MiB per round over the artifact's own regions is a few milliseconds on this volume.
inline constexpr std::uint64_t kDiskTrialBytes = 24ULL << 20;
inline constexpr int kDiskTrialRounds          = 3;

// A file read the probe can drive without knowing what a file is. `read` must fill the whole span
// or report a short count, which the probe treats as a failure; the caller wires it to whatever the
// engine streams experts from, in the mode it picked for that volume.
struct DiskReadSource {
    void* context = nullptr;
    std::size_t (*read)(void* context, std::uint64_t offset, std::span<std::byte> destination) =
        nullptr;
    std::uint64_t bytes = 0;      // the object's length
    const char*   mode  = "none"; // printed: "buffered", "direct", or whatever the caller names
};

struct TransferRateProbe {
    TransferRates rates{};
    std::uint32_t page_copies = 0; // copies in one page-in trial (the shape the rate came from)
    std::uint32_t slot_copies = 0;
    // The disk leg's shape, so the number can be checked against the read it claims to describe.
    // Zero copies / a null mode mean the leg was not measured, which is the `cost.measured()`
    // fallback and not a claim that reading is free.
    std::uint32_t disk_copies       = 0;
    std::uint64_t disk_payload_bytes = 0;
    std::uint64_t disk_span_bytes    = 0; // bytes actually read, all rounds included
    const char*   disk_mode          = "none";
    // The largest Host allocation the probe pinned. The legs are measured one after the other, so
    // this is the footprint, not their sum - it is printed so the probe's own cost is visible.
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

// V10-83: the artifact leg, at one expert payload. Each round reads a different part of the file,
// so the answer is a cold read of that payload and not a replay of a warm one - the shape a
// streaming layer sees. Best of the rounds, since a scheduling hiccup or a cold metadata lookup in
// one of them is not the volume's rate. Every failure path leaves the rate at zero, and the shape
// it did measure is handed back so a caller can print what the number came from.
[[nodiscard]] inline double measure_disk_read_rate(const DiskReadSource& source,
                                                   std::uint64_t payload_bytes,
                                                   std::uint32_t& copies_out,
                                                   std::uint64_t& span_out) noexcept {
    copies_out = 0;
    span_out   = 0;
    if (source.read == nullptr || payload_bytes == 0 || source.bytes < payload_bytes) {
        return 0.0;
    }
    const std::uint64_t wanted = kDiskTrialBytes / payload_bytes;
    const std::uint32_t copies =
        static_cast<std::uint32_t>(std::clamp<std::uint64_t>(wanted, 2, 64));
    const std::uint64_t run_bytes = payload_bytes * copies;
    if (run_bytes > source.bytes) { return 0.0; }
    // Aligned generously: the artifact's own payloads are 4 KiB aligned, and a direct read demands
    // it. `vector` gives no alignment guarantee, so the pointer is rounded up inside the block.
    constexpr std::uint64_t kAlign = 4096;
    std::vector<std::byte> buffer;
    try {
        buffer.resize(static_cast<std::size_t>(run_bytes + kAlign));
    } catch (...) {
        return 0.0;
    }
    const auto base = reinterpret_cast<std::uintptr_t>(buffer.data());
    auto* aligned   = reinterpret_cast<std::byte*>((base + kAlign - 1) / kAlign * kAlign);
    double best_ms = 0.0;
    for (int round = 0; round < kDiskTrialRounds; ++round) {
        // A fresh region per round: thirds of the object, kept whole.
        const std::uint64_t region = source.bytes / static_cast<std::uint64_t>(kDiskTrialRounds);
        const std::uint64_t slack  = region > run_bytes ? (region - run_bytes) / 2 : 0;
        std::uint64_t start = region * static_cast<std::uint64_t>(round) + slack;
        start = start / kAlign * kAlign;
        if (start + run_bytes > source.bytes) { start = source.bytes - run_bytes; }
        const auto begin = std::chrono::steady_clock::now();
        bool complete    = true;
        for (std::uint32_t copy = 0; copy < copies; ++copy) {
            const std::uint64_t offset = start + static_cast<std::uint64_t>(copy) * payload_bytes;
            if (source.read(source.context, offset,
                            std::span<std::byte>(aligned, static_cast<std::size_t>(
                                                              payload_bytes))) !=
                payload_bytes) {
                complete = false;
                break;
            }
        }
        const auto end = std::chrono::steady_clock::now();
        if (!complete) { continue; }
        const double ms = std::chrono::duration<double, std::milli>(end - begin).count();
        if (ms > 0.0 && (best_ms == 0.0 || ms < best_ms)) {
            best_ms  = ms;
            span_out = run_bytes;
        }
    }
    if (best_ms <= 0.0) { return 0.0; }
    copies_out = copies;
    return static_cast<double>(span_out) / (best_ms / 1000.0);
}


// Both legs, on the two payload shapes the run will send, plus the artifact read when the caller
// can name one. Zero rates mean "not measured".
[[nodiscard]] inline TransferRateProbe measure_transfer_rates(
    std::uint64_t page_bytes, std::uint64_t slot_bytes, cudaStream_t stream,
    const DiskReadSource* disk = nullptr) noexcept {
    TransferRateProbe probe;
    probe.page_copies = detail::copies_for_trial(page_bytes);
    probe.slot_copies = detail::copies_for_trial(slot_bytes);
    probe.rates.page_in_bytes_per_s =
        detail::measure_h2d_copy_rate(page_bytes, probe.page_copies, stream);
    probe.rates.expert_bytes_per_s =
        detail::measure_h2d_copy_rate(slot_bytes, probe.slot_copies, stream);
    probe.pinned_bytes = std::max<std::uint64_t>(page_bytes, slot_bytes);
    if (disk != nullptr) {
        probe.disk_payload_bytes = slot_bytes;
        probe.disk_mode          = disk->mode;
        probe.rates.disk_bytes_per_s = measure_disk_read_rate(
            *disk, slot_bytes, probe.disk_copies, probe.disk_span_bytes);
    }
    return probe;
}

} // namespace ninfer::core
