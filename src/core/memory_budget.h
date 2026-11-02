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
// This is wiring, not tuning: the default path is the old rule, reproduced number for number, and
// the new part is that the decision is printed with its marginal value and can be overridden
// (NINFER_BUDGET_EXPERT_MB / NINFER_BUDGET_KV_MB). Choosing the split from the table is Q50's next
// step and belongs to the controller, not to a constant baked in here.

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace ninfer::core {

// One measured point of the pool's value curve. `pool_bytes` is what the pool got; the two numbers
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

// What the next gigabyte of pool is worth at this size: misses avoided per token, from the segment
// the size falls in. Zero outside the measured range - "not measured" is not "no value".
inline double marginal_misses_per_gib(std::uint64_t pool_bytes) {
    constexpr std::size_t count = sizeof(kMeasuredPoolValue) / sizeof(kMeasuredPoolValue[0]);
    for (std::size_t i = 0; i + 1 < count; ++i) {
        const auto& low  = kMeasuredPoolValue[i];
        const auto& high = kMeasuredPoolValue[i + 1];
        if (pool_bytes < low.pool_bytes || pool_bytes >= high.pool_bytes) { continue; }
        const auto span = static_cast<double>(high.pool_bytes - low.pool_bytes);
        return (low.misses_per_token - high.misses_per_token) / (span / static_cast<double>(1ULL << 30));
    }
    return 0.0;
}

struct DeviceSplit {
    std::size_t pool_bytes   = 0;
    std::size_t safety_bytes = 0;
    std::size_t kv_bytes     = 0;
    // Which knob produced the split. Printed, because "the pool is this big" is only useful next to
    // "and here is what decided it".
    const char* decided_by = "kv-first";
};

namespace detail {
inline std::uint64_t env_mib(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') { return 0; }
    return static_cast<std::uint64_t>(std::strtoull(value, nullptr, 10)) << 20;
}
} // namespace detail

// The split, in one place. Default reproduces the rule that was here before, to the byte:
//   pool = free - (kv + 1 GiB safety), floored at zero.
// The overrides pin either side instead, and the other side takes what is left; a pinned pair that
// does not fit is reported and the pool gives way, because KV is touched by every step of every
// request and the pool only by the routed top-k.
inline DeviceSplit plan_device_split(std::size_t free_bytes, std::size_t kv_need_bytes) {
    constexpr std::size_t kPoolSafetyBytes = 1ULL << 30;
    DeviceSplit split{};
    split.safety_bytes = kPoolSafetyBytes;
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
    split.pool_bytes   = free_bytes > pledged ? free_bytes - pledged : 0;
    return split;
}

inline void report_device_split(const DeviceSplit& split, std::size_t free_bytes) {
    constexpr double kMib = 1024.0 * 1024.0;
    std::fprintf(stderr,
                 "[budget][split] free=%.0f MiB kv=%.0f MiB pool=%.0f MiB safety=%.0f MiB by=%s | "
                 "next GiB of pool worth %.2f miss/token\n",
                 static_cast<double>(free_bytes) / kMib, static_cast<double>(split.kv_bytes) / kMib,
                 static_cast<double>(split.pool_bytes) / kMib,
                 static_cast<double>(split.safety_bytes) / kMib, split.decided_by,
                 marginal_misses_per_gib(split.pool_bytes));
    std::fflush(stderr);
}

} // namespace ninfer::core
