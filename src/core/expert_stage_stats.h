#pragma once

// V10-85: what a streaming layer's fill actually costs, timed where it blocks.
//
// V10-83 priced the per-layer round trip at 0.5735 ms and said what that number was: a regression
// intercept (22.94 ms/step over 40 layers), not a measurement of the path it named. That path is
// measurable, and it is two legs in series - the layer reads its missing experts out of the
// artifact into the Host tier on its own thread (host_expert_residency.h, `fill_locked`), then
// submits the copies and drains them (core/expert_cache.cpp, `prepare()` and
// `wait_for_progress()`) before its GEMM can start. The two legs are timed separately because they
// differ by 3-5x on this machine (a buffered artifact read against a 23 GB/s link), and a single
// number could not say which one a run is paying.
//
// A third number is what a price uses, and it is not either leg: the whole FFN wall of a layer that
// staged a copy (models/qwen3_5/execution/ffn.cpp, `LayerScope`, whose clock the timeline already
// took). On 35B e2 the route read-back and the CPU half are the larger part of that wall, so a
// price built from the upload turn alone came out 4x too low - measured, not assumed: the same
// run's turn averaged 0.116 ms/layer against a 0.612 ms layer wall (report §5).
//
// The counters are process-wide, like kv_tier_stats() beside them and for the same reason: the
// places that update them share no other object, and a diagnostic must not change the layout of the
// engine's main state. One instance per process reads as "since the start": a consumer compares
// snapshots, and a SECOND Engine in the same process sees the first one's measurement - which is
// what lets a later plan price a measured fill instead of the constant (core/capacity_bet.h takes
// the number through core/transfer_cost.h).
//
// Nothing here allocates, logs, or touches CUDA.

#include <chrono>
#include <cstdint>

namespace ninfer::core {

struct ExpertStageStats {
    // (a) The Host tier's fill: the artifact read (a page-cache memcpy or a buffered read), on the
    // layer's thread and under the tier's lock. Zero while a run serves everything from the pool.
    double read_seconds = 0.0;
    // (b) Host -> Device: the upload turn's bookkeeping plus the drain the layer blocks on. Covers
    // the copy AND everything progress() does to submit it - those are one wait to the layer.
    double copy_seconds = 0.0;
    // (c) The whole FFN wall of a layer that staged a copy, summed over such layers. This is the
    // priced one.
    double stage_seconds = 0.0;
    // Drains that actually staged a copy. A resident layer calls the same wait and stages nothing,
    // so this is the field that separates "streamed" from "resident" without a second switch.
    std::uint64_t layers = 0;
    // Fills that ran, i.e. the read leg's own denominator: a layer can fill the tier without the
    // pool missing anything (the CPU half then serves those routes), so the two counts differ.
    std::uint64_t read_calls = 0;

    [[nodiscard]] double ms_per_layer() const noexcept {
        if (layers == 0) { return 0.0; }
        return stage_seconds * 1000.0 / static_cast<double>(layers);
    }

    [[nodiscard]] double read_ms_per_layer() const noexcept {
        return layers == 0 ? 0.0 : read_seconds * 1000.0 / static_cast<double>(layers);
    }

    [[nodiscard]] double copy_ms_per_layer() const noexcept {
        return layers == 0 ? 0.0 : copy_seconds * 1000.0 / static_cast<double>(layers);
    }
};

inline ExpertStageStats& expert_stage_stats() noexcept {
    static ExpertStageStats stats;
    return stats;
}

// Milliseconds one streaming layer costs, or 0 when nothing has been measured. Zero layers is "no
// layer staged a copy" - and never a measurement of a free fill: the caller (core/transfer_cost.h)
// reads a zero as "no counter" and falls back to its constant, printing which of the two it used.
[[nodiscard]] inline double expert_fill_ms_per_layer(const ExpertStageStats& stats) noexcept {
    return stats.ms_per_layer();
}

[[nodiscard]] inline double expert_fill_ms_per_layer() noexcept {
    return expert_stage_stats().ms_per_layer();
}

} // namespace ninfer::core
