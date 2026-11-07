#pragma once

// port (sm_89): pure cost model for the host partition partial (partial.h). No runtime
// wiring lives here and nothing in the tree calls it yet — this header is the DECISION
// INPUT the tiering controller consumes later, when it starts choosing which KV splits run
// as CPU partials over host-resident pages instead of GPU grouped instances. The rates
// below are CALIBRATED on this machine (RTX 4090 host / i9-14900K / Win11, AVX2, the
// scalar mirror kernel single-threaded, 2026-11-07); the estimate is an upper bound on the
// host side of the controller's copy-vs-compute choice, never a promise.
//
// Arithmetic-shape mirror of the grouped instance the partial reproduces (grouped_mma.cuh):
// a split covering `keys` tokens against a `width`-token query tile of `query_heads` heads
// performs, per kv-head group of rows:
//   QK^T   : keys * width * query_heads * 256 int8 MACs   (the int8 MMA lines 786-813)
//   PV     : keys * width * query_heads * 256 fp16 MACs   (the fp16 MMA lines 1040-1058)
//   exp    : width * query_heads * keys exp() calls plus 2 * tiles per row for the rescale
//   rescale: width * query_heads * 256 * tiles FP32 multiplies (alpha, lines 1033-1038)
// Memory reads are page-plane traffic only (Q and the partial buffer are negligible):
//   per key and kv head: K plane (128 B int4 / 256 B int8) + V plane (128 B int4) + 2 FP16
//   G64 scale planes (8 B) — the same bytes the GPU instance streams from the same pages.
//
// Calibration (measurement discipline: min of 200 timed calls, 96 MiB cache flush before
// every timed call, physical-page rotation through a store far past L3, single thread;
// K/V planes filled constant so every online-softmax p survives FP16 rounding and the PV
// stage runs its full counted work — real peaked-score calls take the bits==0 skip and run
// cheaper, which makes these rates an upper bound). Points: D256H24Kv4 and D256H16Kv2,
// rk4v4 (packed-int4 K) and rk8v4 (int8 K), keys 2048/8192/32768 — per-key cost is linear
// across that whole range (9.15 -> 9.24 us/key for H24Kv4 packed), so the linear form holds
// and 8192 anchors the fit. Stage attribution via a scratch phase-timed copy of the
// kernel: the PV+rescale phase pins kRkPartialHostF16MacRate to within 0.4% on all four
// configs; the QK+decode phase varies 1.45-1.98e9 with the per-row decode amortization
// (kv-head decode cost spread over 24 vs 16 rows, int4 unpack vs int8 sign-extend), which
// a single constant cannot track — the geometric mean is used and the residuals below are
// the price. kRkPartialHostByteRate is the geometric mean of measured cold-page stream
// passes over the same addresses (1.8-1.9 GB/s int4 K rows, 3.0-3.3 GB/s int8 K rows);
// the byte term is 4-6% of a call at these geometries, so its codec-blindness costs little.
// Model error against the measured minima at the four calibration configs (keys=8192):
// -6.8% (H24Kv4 packed) / +0.1% (H24Kv4 int8K) / +0.6% (H16Kv2 packed) / +6.0% (H16Kv2
// int8K); the model is codec-blind in the MAC terms, so it under-estimates the heaviest
// decode geometry (H24 packed) and over-estimates the cheapest config (H16 int8K).
// Controller decisions should tolerate a ~7% band, or recalibrate per geometry when wired.

#include <cstdint>

namespace ninfer::ops::detail::softmax_cpu {

// CALIBRATED rates (i9-14900K AVX2, scalar partial.h, min/200, 96 MiB flush, cold-page
// rotation — see the header comment for the fit and its residuals). Controller decision
// input only: the engine never reads these until the tiering controller lands.
inline constexpr double kRkPartialHostInt8MacRate = 1.70e9;  // int8 MACs per second
inline constexpr double kRkPartialHostF16MacRate   = 1.40e9;  // fp16 MACs per second
inline constexpr double kRkPartialHostByteRate     = 2.38e9;  // page bytes per second

struct RkPartialOpCounts {
    double qk_macs      = 0.0;  // int8 MACs (exact integer dots + FP32 group scales)
    double pv_macs      = 0.0;  // fp16 MACs (FP16 p times FP16 V, FP32 accumulate)
    double exp_calls    = 0.0;  // host expf invocations
    double rescale_muls = 0.0;  // FP32 alpha rescales of the accumulator
    double kv_bytes     = 0.0;  // K/V/scale plane bytes read
};

// tokens x heads x d operation count for one split's partial. `keys` is the split's token
// count (range.end - range.begin), `width`/`query_heads`/`kv_heads` the query tile geometry,
// `key_tile` the online-softmax granularity (the mirrored instance's kKeyRows), `packed_k`
// true for rk4v4's int4 K plane, false for rk8v4's int8 K plane.
inline RkPartialOpCounts rk_partial_count_ops(int keys, int width, int query_heads, int kv_heads,
                                              int key_tile, bool packed_k) noexcept {
    RkPartialOpCounts out;
    const double rows  = static_cast<double>(width) * query_heads;
    const double tile  = key_tile > 0 ? key_tile : 1;
    const double tiles = keys > 0 ? (static_cast<double>(keys) + tile - 1.0) / tile : 0.0;
    out.qk_macs        = static_cast<double>(keys) * rows * 256.0;
    out.pv_macs        = static_cast<double>(keys) * rows * 256.0;
    out.exp_calls      = static_cast<double>(keys) * rows + 2.0 * rows * tiles;
    out.rescale_muls   = rows * 256.0 * tiles;
    out.kv_bytes       = static_cast<double>(keys) * kv_heads *
                         ((packed_k ? 128.0 : 256.0) + 128.0 + 8.0);
    return out;
}

// Wall-clock estimate from the counts. Returns 0.0 while any rate is <= 0, so a zeroed
// rate can never masquerade as a measurement — callers treat 0 as "estimate unavailable",
// never as "free".
inline double rk_partial_cost_seconds(const RkPartialOpCounts& ops) noexcept {
    if (kRkPartialHostInt8MacRate <= 0.0 || kRkPartialHostF16MacRate <= 0.0 ||
        kRkPartialHostByteRate <= 0.0) {
        return 0.0;
    }
    return ops.qk_macs / kRkPartialHostInt8MacRate + ops.pv_macs / kRkPartialHostF16MacRate +
           ops.exp_calls / kRkPartialHostInt8MacRate +
           ops.rescale_muls / kRkPartialHostF16MacRate + ops.kv_bytes / kRkPartialHostByteRate;
}

} // namespace ninfer::ops::detail::softmax_cpu
