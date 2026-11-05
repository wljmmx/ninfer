#pragma once

// port (sm_89): pure cost model for the host partition partial (partial.h). No runtime
// wiring lives here and nothing in the tree calls it yet — this header is the DECISION
// INPUT the tiering controller consumes later, when it starts choosing which KV splits run
// as CPU partials over host-resident pages instead of GPU grouped instances. Every rate
// constant below is a PLACEHOLDER: the controller must calibrate them on this machine
// (RTX 4090 / i9-14900K / Win11) with the measured partial batch before wiring any
// decision, exactly like the GPU side calibrates its schedules. Until then the seconds
// estimate returns 0, which callers must treat as "estimate unavailable", never as "free".
//
// Arithmetic-shape mirror of the grouped instance the partial reproduces (grouped_mma.cuh):
// a split covering `keys` tokens against a `width`-token query tile of `query_heads` heads
// performs, per kv-head group of rows:
//   QK^T   : keys * width * query_heads * 256 int8 MACs   (the int8 MMA lines 786-813)
//   PV     : keys * width * query_heads * 256 fp16 MACs    (the fp16 MMA lines 1040-1058)
//   exp    : width * query_heads * keys exp() calls plus 2 * tiles per row for the rescale
//   rescale: width * query_heads * 256 * tiles FP32 multiplies (alpha, lines 1033-1038)
// Memory reads are page-plane traffic only (Q and the partial buffer are negligible):
//   per key and kv head: K plane (128 B int4 / 256 B int8) + V plane (128 B int4) + 2 FP16
//   G64 scale planes (8 B) — the same bytes the GPU instance streams from the same pages.

#include <cstdint>

namespace ninfer::ops::detail::softmax_cpu {

// PLACEHOLDER rates — zero means "uncalibrated". Calibration recipe (later batch): run the
// partial over N keys x fixed geometry on this host, repeat >= 200 with clock medians
// following the house measurement discipline, then set the three rates from the fit.
inline constexpr double kRkPartialHostInt8MacRate = 0.0;  // int8 MACs per second
inline constexpr double kRkPartialHostF16MacRate   = 0.0;  // fp16 MACs per second
inline constexpr double kRkPartialHostByteRate     = 0.0;  // page bytes per second

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

// Wall-clock estimate from the counts. Returns 0.0 while any rate is uncalibrated, so the
// controller can never mistake a placeholder for a measurement.
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
