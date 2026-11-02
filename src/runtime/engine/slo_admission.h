#pragma once

// 13.4: SLO admission.
//
// The engine already refuses work when a hard queue or memory limit is hit. What it did not have is
// a *latency* target: every request was admitted until the queue physically filled, and a user
// learned the service was slow by waiting. This class turns the measured rates into an expected
// service time and refuses the request before it is queued when that time is already over budget.
//
// The model is deliberately small and explicit:
//
//   TTFT  = (queue ahead + this prompt) / prefill rate + admission overhead
//   total = TTFT + output tokens / decode rate
//
// Rates are EWMAs of the engine's own counters, sampled here. A cold process starts from the
// configured initial rates and converges; an idle process keeps its last rates, because a change of
// load should not be remembered as a change of speed. The decision is opt-in: NINFER_SLO_TTFT_MS
// and/or NINFER_SLO_TOTAL_MS must be set.

#include "ninfer/types.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <string>

namespace ninfer::runtime {

struct SloLimits {
    bool enabled          = false;
    double ttft_ms        = 0.0; // <= 0 means "not constrained"
    double total_ms       = 0.0; // <= 0 means "not constrained"
    std::uint32_t max_waiting = 0; // 0 means "not constrained"
    double initial_prefill_tokens_per_second = 800.0;
    double initial_decode_tokens_per_second  = 30.0;
    // Cost of admitting a request at all (binding, source selection, first submission). Measured
    // request logs put the admission path in the low milliseconds; this is a constant until the
    // controller has a per-phase estimate.
    double admission_overhead_ms = 2.0;
};

[[nodiscard]] inline SloLimits slo_limits_from_env() {
    SloLimits limits;
    const auto read_double = [](const char* name, double fallback) {
        if (const char* value = std::getenv(name); value != nullptr && *value != '\0') {
            char* end = nullptr;
            const double parsed = std::strtod(value, &end);
            if (end != value && std::isfinite(parsed) && parsed > 0.0) { return parsed; }
        }
        return fallback;
    };
    limits.ttft_ms  = read_double("NINFER_SLO_TTFT_MS", 0.0);
    limits.total_ms = read_double("NINFER_SLO_TOTAL_MS", 0.0);
    if (const char* value = std::getenv("NINFER_SLO_MAX_WAITING");
        value != nullptr && *value != '\0') {
        const long parsed = std::strtol(value, nullptr, 10);
        if (parsed > 0) { limits.max_waiting = static_cast<std::uint32_t>(parsed); }
    }
    limits.initial_prefill_tokens_per_second =
        read_double("NINFER_SLO_PREFILL_TOK_S", limits.initial_prefill_tokens_per_second);
    limits.initial_decode_tokens_per_second =
        read_double("NINFER_SLO_DECODE_TOK_S", limits.initial_decode_tokens_per_second);
    limits.admission_overhead_ms =
        read_double("NINFER_SLO_ADMISSION_MS", limits.admission_overhead_ms);
    limits.enabled = limits.ttft_ms > 0.0 || limits.total_ms > 0.0 || limits.max_waiting != 0;
    return limits;
}

class SloAdmission {
public:
    using Clock = std::chrono::steady_clock;

    explicit SloAdmission(SloLimits limits) : limits_(limits) {}

    [[nodiscard]] static SloAdmission from_env() { return SloAdmission(slo_limits_from_env()); }

    [[nodiscard]] bool enabled() const noexcept { return limits_.enabled; }
    [[nodiscard]] const SloLimits& limits() const noexcept { return limits_; }

    // Returns a reason when the request must be refused, nullopt when it may be queued.
    [[nodiscard]] std::optional<std::string>
    check(std::uint32_t prompt_tokens, std::uint32_t output_tokens, const RuntimeStats& stats) {
        if (!limits_.enabled) { return std::nullopt; }
        const auto now = Clock::now();
        std::lock_guard lock(mutex_);
        observe_locked(now, stats);
        if (prompt_tokens != 0) {
            average_prompt_tokens_ = average_prompt_tokens_ == 0.0
                                         ? static_cast<double>(prompt_tokens)
                                         : average_prompt_tokens_ * 0.9 + prompt_tokens * 0.1;
        }
        if (limits_.max_waiting != 0 && stats.waiting_requests >= limits_.max_waiting) {
            return "queue is at its SLO depth (" + std::to_string(stats.waiting_requests) + " of " +
                   std::to_string(limits_.max_waiting) + " waiting)";
        }
        const double prefill_rate = std::max(1.0, prefill_tokens_per_second_);
        const double decode_rate  = std::max(1.0, decode_tokens_per_second_);
        const double queue_ahead =
            static_cast<double>(stats.waiting_requests) * std::max(1.0, average_prompt_tokens_);
        const double ttft_ms =
            (queue_ahead + static_cast<double>(prompt_tokens)) * 1000.0 / prefill_rate +
            static_cast<double>(stats.waiting_requests) * limits_.admission_overhead_ms;
        const double total_ms =
            ttft_ms + static_cast<double>(std::max<std::uint32_t>(output_tokens, 1)) * 1000.0 /
                          decode_rate;
        if (limits_.ttft_ms > 0.0 && ttft_ms > limits_.ttft_ms) {
            return "expected TTFT " + format_ms(ttft_ms) + " ms exceeds " +
                   format_ms(limits_.ttft_ms) + " ms (prefill " + format_rate(prefill_rate) +
                   " tok/s, " + std::to_string(stats.waiting_requests) + " waiting)";
        }
        if (limits_.total_ms > 0.0 && total_ms > limits_.total_ms) {
            return "expected completion " + format_ms(total_ms) + " ms exceeds " +
                   format_ms(limits_.total_ms) + " ms (decode " + format_rate(decode_rate) +
                   " tok/s, " + std::to_string(output_tokens) + " output token(s))";
        }
        ++admitted_;
        return std::nullopt;
    }

    [[nodiscard]] double prefill_tokens_per_second() const {
        std::lock_guard lock(mutex_);
        return prefill_tokens_per_second_;
    }
    [[nodiscard]] double decode_tokens_per_second() const {
        std::lock_guard lock(mutex_);
        return decode_tokens_per_second_;
    }
    [[nodiscard]] std::uint64_t admitted() const {
        std::lock_guard lock(mutex_);
        return admitted_;
    }
    [[nodiscard]] std::uint64_t refused() const {
        std::lock_guard lock(mutex_);
        return refused_;
    }
    void note_refusal() {
        std::lock_guard lock(mutex_);
        ++refused_;
    }

private:
    void observe_locked(Clock::time_point now, const RuntimeStats& stats) {
        if (!sampled_) {
            last_sample_ = now;
            last_computed_prefill_ = stats.computed_prefill_tokens;
            last_committed_decode_ = stats.committed_decode_tokens;
            sampled_               = true;
            return;
        }
        const double seconds = std::chrono::duration<double>(now - last_sample_).count();
        if (seconds < 0.25) { return; }
        const std::uint64_t prefill = stats.computed_prefill_tokens >= last_computed_prefill_
                                          ? stats.computed_prefill_tokens - last_computed_prefill_
                                          : 0;
        const std::uint64_t decode = stats.committed_decode_tokens >= last_committed_decode_
                                         ? stats.committed_decode_tokens - last_committed_decode_
                                         : 0;
        if (prefill != 0) {
            const double rate = static_cast<double>(prefill) / seconds;
            prefill_tokens_per_second_ = prefill_tokens_per_second_ * 0.8 + rate * 0.2;
        }
        if (decode != 0) {
            const double rate = static_cast<double>(decode) / seconds;
            decode_tokens_per_second_ = decode_tokens_per_second_ * 0.8 + rate * 0.2;
        }
        last_sample_           = now;
        last_computed_prefill_ = stats.computed_prefill_tokens;
        last_committed_decode_ = stats.committed_decode_tokens;
    }

    static std::string format_ms(double value) {
        char buffer[32];
        std::snprintf(buffer, sizeof(buffer), "%.0f", value);
        return buffer;
    }
    static std::string format_rate(double value) {
        char buffer[32];
        std::snprintf(buffer, sizeof(buffer), "%.0f", value);
        return buffer;
    }

    SloLimits limits_;
    mutable std::mutex mutex_;
    bool sampled_                    = false;
    Clock::time_point last_sample_{};
    std::uint64_t last_computed_prefill_ = 0;
    std::uint64_t last_committed_decode_ = 0;
    double prefill_tokens_per_second_    = limits_.initial_prefill_tokens_per_second;
    double decode_tokens_per_second_     = limits_.initial_decode_tokens_per_second;
    double average_prompt_tokens_        = 0.0;
    std::uint64_t admitted_              = 0;
    std::uint64_t refused_               = 0;
};

} // namespace ninfer::runtime
