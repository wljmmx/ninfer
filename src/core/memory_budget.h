#pragma once

// port (sm_89): Q50 - the device budget as a decision, in one place.
//
// The split between the KV arena and the expert pool used to be a constant: give KV what it asks,
// keep a gigabyte of safety, and hand the pool the rest. That is a rule of thumb wearing a policy's
// clothes - it says nothing about what the next gigabyte buys, which is the only question a budget
// can answer. Measured on this box (8 segments x 128 tokens, C = 1, e2, epoch 64, host leg on):
//
//   pool   slots/bank  coverage  miss/token  host%   capt_free  tok/s
//   4 GiB  50          43.8%     4.49        67.0%   3.4%       13.7
//   6 GiB  81          45.5%     4.36        63.0%   7.9%       16.6
//   8 GiB  106         45.2%     4.39        59.8%   11.6%      20.4
//   12 GiB 163         65.1%     2.79        32.0%   34.5%      30.7
//
// The shape is not a line. 4 -> 8 GiB barely buys coverage (43.8 -> 45.2%) and buys no misses at all
// (4.49 -> 4.39); it buys SPEED, by moving routes off the PCIe leg into video memory (+6.7 tok/s).
// 8 -> 12 GiB buys coverage (45.2 -> 65.1%, -1.60 miss/token, +10.3 tok/s). So the marginal value of
// a gigabyte depends on where you already are, which is exactly what this header makes visible.
//
// Q50, second half: the table above is the initial curve, not the answer. A run measures its own
// (pool bytes, misses per token, tokens per second) triple and every observation replaces the
// nearest 512 MiB bucket, so the next decision is made from this machine's numbers. The baked-in
// points stay as the prior until the run has measured its own.
//
// This is still wiring: the default split is the old rule byte for byte, and only
// NINFER_BUDGET_SELECT=1 asks the curve to choose the point. What is new is that the choice is a
// function of measured value, printed next to the point it picked, rather than a size copied from a
// comment.

#include "ninfer/types.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace ninfer::core {

// One measured point of the pool's value curve. `pool_bytes` is what the pool got; the three numbers
// are what the same run measured at that size.
struct PoolValuePoint {
    std::uint64_t pool_bytes = 0;
    double misses_per_token   = 0.0;
    double tokens_per_second  = 0.0;
};

inline constexpr PoolValuePoint kMeasuredPoolValue[] = {
    {4ULL << 30, 4.49, 13.7},
    {6ULL << 30, 4.36, 16.6},
    {8ULL << 30, 4.39, 20.4},
    {12ULL << 30, 2.79, 30.7},
};

inline constexpr std::size_t kPoolCurvePoints = 16;

// The live curve. One instance per process: the split is decided once at startup, and every
// observation after that is evidence for the next decision (the runtime re-split belongs to the
// controller; this is the memory it will read). Writes are single-threaded today - the panel that
// observes is printed from the engine thread - and the read at startup happens before any layer, so
// the array needs no lock. Stated so a later caller does not assume a safety that is not here.
struct PoolValueCurve {
    std::array<PoolValuePoint, kPoolCurvePoints> points{};
    std::size_t count = 0;

    PoolValueCurve() {
        for (const auto& point : kMeasuredPoolValue) { add(point); }
    }

    // 512 MiB buckets: two observations of the same operating point replace each other instead of
    // accumulating noise, and a 4 -> 12 GiB sweep still has room for every point.
    void add(PoolValuePoint point) {
        if (point.pool_bytes == 0 || point.tokens_per_second <= 0.0) { return; }
        constexpr std::uint64_t kBucket = 512ULL << 20;
        for (std::size_t i = 0; i < count; ++i) {
            const auto low  = points[i].pool_bytes / kBucket;
            const auto high = point.pool_bytes / kBucket;
            if (low == high) {
                points[i] = point;
                return;
            }
        }
        if (count == kPoolCurvePoints) { return; }
        points[count++] = point;
        for (std::size_t i = count; i > 1; --i) {
            if (points[i - 1].pool_bytes >= points[i - 2].pool_bytes) { break; }
            const auto swap = points[i - 1];
            points[i - 1]     = points[i - 2];
            points[i - 2]     = swap;
        }
    }

    [[nodiscard]] double marginal_misses_per_gib(std::uint64_t pool_bytes) const {
        for (std::size_t i = 0; i + 1 < count; ++i) {
            const auto& low  = points[i];
            const auto& high = points[i + 1];
            if (pool_bytes < low.pool_bytes || pool_bytes >= high.pool_bytes) { continue; }
            const auto span =
                static_cast<double>(high.pool_bytes - low.pool_bytes) / static_cast<double>(1ULL << 30);
            return (low.misses_per_token - high.misses_per_token) / span;
        }
        return 0.0;
    }

    [[nodiscard]] double marginal_tps_per_gib(std::uint64_t pool_bytes) const {
        for (std::size_t i = 0; i + 1 < count; ++i) {
            const auto& low  = points[i];
            const auto& high = points[i + 1];
            if (pool_bytes < low.pool_bytes || pool_bytes >= high.pool_bytes) { continue; }
            const auto span =
                static_cast<double>(high.pool_bytes - low.pool_bytes) / static_cast<double>(1ULL << 30);
            return (high.tokens_per_second - low.tokens_per_second) / span;
        }
        return 0.0;
    }

    // The nearest measured point at or below `bytes`, or 0 when the curve starts above it.
    [[nodiscard]] std::uint64_t floor_point(std::uint64_t bytes) const {
        std::uint64_t best = 0;
        for (std::size_t i = 0; i < count; ++i) {
            if (points[i].pool_bytes <= bytes) { best = points[i].pool_bytes; }
        }
        return best;
    }
};

inline PoolValueCurve& pool_value_curve() {
    static PoolValueCurve curve;
    return curve;
}

// Q50's observation entry point. Called from the split path's own panel, so the curve is built from
// the run that just measured it; `tokens_per_second` may be negative (not measured yet) and is then
// ignored.
inline void observe_pool_value(std::uint64_t pool_bytes, double misses_per_token,
                               double tokens_per_second) {
    if (misses_per_token < 0.0 || tokens_per_second <= 0.0) { return; }
    pool_value_curve().add(PoolValuePoint{pool_bytes, misses_per_token, tokens_per_second});
}

// What the next gigabyte of pool is worth at this size: misses avoided per token, from the segment
// the size falls in. Zero outside the measured range - "not measured" is not "no value".
inline double marginal_misses_per_gib(std::uint64_t pool_bytes) {
    return pool_value_curve().marginal_misses_per_gib(pool_bytes);
}

struct DeviceSplit {
    std::size_t pool_bytes   = 0;
    std::size_t safety_bytes = 0;
    std::size_t kv_bytes     = 0;
    // Which knob produced the split. Printed, because "the pool is this big" is only useful next to
    // "and here is what decided it".
    const char* decided_by = "kv-first";
    // Q50's selection, when it is enabled: the measured point the decision was taken at, and whether
    // the pool was rounded down onto it (a partial gigabyte above a measured point is not itself a
    // measurement, so a run that wants repeatable points stays on the point).
    std::uint64_t selected_point = 0;
    bool rounded_to_point        = false;
};

namespace detail {
inline std::uint64_t env_mib(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') { return 0; }
    return static_cast<std::uint64_t>(std::strtoull(value, nullptr, 10)) << 20;
}
} // namespace detail

// Q50: choose the point from the curve. The default split reproduces the old rule, so this only runs
// when NINFER_BUDGET_SELECT is set; what it adds is the decision the plan asked for - the pool is a
// point whose value has been measured, not "whatever was left".
//
// The rule: take as much as the device can spare, except that a size less than one gigabyte above a
// measured point is rounded down onto that point. A partial gigabyte buys the value of its segment
// (already measured) but makes the run incomparable with the point itself, and the whole point of a
// curve is that runs stay comparable. At or beyond the largest measured point nothing is held back:
// "not measured" is not "worth nothing", and the pool is the only consumer left once KV and the
// safety floor are paid.
inline std::uint64_t select_pool_bytes(std::uint64_t available_bytes, DeviceSplit& split) {
    if (std::getenv("NINFER_BUDGET_SELECT") == nullptr) { return available_bytes; }
    const auto point = pool_value_curve().floor_point(available_bytes);
    if (point != 0 && available_bytes - point < (1ULL << 30)) {
        split.selected_point   = point;
        split.rounded_to_point = true;
        return point;
    }
    split.selected_point = available_bytes;
    return available_bytes;
}

// The split, in one place. Default reproduces the rule that was here before, to the byte:
//   pool = free - (kv + startup planning headroom), floored at zero.
// The headroom is the startup planning allowance: Graph capture and weight staging may consume it,
// and nothing requires it to stay free after initialization. It is configurable
// (--startup-planning-headroom-mib) and is NOT the KV sizing headroom of --kv-capacity auto, which
// was already deducted when the KV reservation above was resolved.
// The overrides pin either side instead, and the other side takes what is left; a pinned pair that
// does not fit is reported and the pool gives way, because KV is touched by every step of every
// request and the pool only by the routed top-k.
inline DeviceSplit
plan_device_split(std::size_t free_bytes, std::size_t kv_need_bytes,
                  std::size_t startup_headroom_bytes = kDefaultStartupPlanningHeadroomBytes) {
    DeviceSplit split{};
    split.safety_bytes = startup_headroom_bytes;
    split.kv_bytes     = kv_need_bytes;

    const auto pinned_pool = static_cast<std::size_t>(detail::env_mib("NINFER_BUDGET_EXPERT_MB"));
    const auto pinned_kv   = static_cast<std::size_t>(detail::env_mib("NINFER_BUDGET_KV_MB"));
    if (pinned_kv != 0) {
        split.kv_bytes   = pinned_kv;
        split.decided_by = "NINFER_BUDGET_KV_MB";
    }
    if (pinned_pool != 0) {
        split.pool_bytes = pinned_pool;
        split.decided_by = pinned_kv != 0 ? "NINFER_BUDGET_EXPERT_MB+KV_MB"
                                          : "NINFER_BUDGET_EXPERT_MB";
        // The pin is an intent; the device is a fact. Report the shortfall here rather than letting
        // the first allocation fail somewhere that cannot explain itself.
        const auto promised = split.kv_bytes + split.pool_bytes + split.safety_bytes;
        if (promised > free_bytes) {
            const auto over = promised - free_bytes;
            split.pool_bytes = split.pool_bytes > over ? split.pool_bytes - over : 0;
            std::fprintf(stderr,
                         "[budget] pinned split asks %llu MiB more than the device has; the pool "
                         "gives way (now %llu MiB)\n",
                         static_cast<unsigned long long>(over >> 20),
                         static_cast<unsigned long long>(split.pool_bytes >> 20));
            std::fflush(stderr);
        }
        return split;
    }
    const auto pledged = split.kv_bytes + split.safety_bytes;
    const auto available = free_bytes > pledged ? free_bytes - pledged : 0;
    split.pool_bytes   = select_pool_bytes(available, split);
    split.decided_by   = split.rounded_to_point ? "curve point" : "curve: affordable";
    return split;
}

inline void report_device_split(const DeviceSplit& split, std::size_t free_bytes) {
    constexpr double kMib = 1024.0 * 1024.0;
    const auto misses    = marginal_misses_per_gib(split.pool_bytes);
    const auto tps       = pool_value_curve().marginal_tps_per_gib(split.pool_bytes);
    // Zero here means "this size is outside every measured segment", not "a gigabyte is worthless".
    // Printing 0.00 for both would turn an absence of evidence into a claim.
    char marginal[96] = {};
    if (misses == 0.0 && tps == 0.0) {
        std::snprintf(marginal, sizeof(marginal), "next GiB unmeasured at this size");
    } else {
        std::snprintf(marginal, sizeof(marginal), "next GiB of pool worth %.2f miss/token, %.2f tok/s",
                      misses, tps);
    }
    std::fprintf(stderr,
                 "[budget][split] free=%.0f MiB kv=%.0f MiB pool=%.0f MiB safety=%.0f MiB by=%s | "
                 "%s\n",
                 static_cast<double>(free_bytes) / kMib, static_cast<double>(split.kv_bytes) / kMib,
                 static_cast<double>(split.pool_bytes) / kMib,
                 static_cast<double>(split.safety_bytes) / kMib, split.decided_by, marginal);
    if (split.rounded_to_point) {
        std::fprintf(stderr,
                     "[budget][split] rounded %.0f MiB down onto the measured %.0f MiB point: a "
                     "partial gigabyte above a point is not itself a measurement\n",
                     static_cast<double>(free_bytes - split.kv_bytes - split.safety_bytes) / kMib,
                     static_cast<double>(split.selected_point) / kMib);
    }
    std::fflush(stderr);
}

} // namespace ninfer::core
