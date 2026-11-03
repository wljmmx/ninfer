#pragma once

// port (sm_89): S0's first step - the shape classes in one place, so "how wide is a batch" has a
// single answer instead of being re-derived at every call site.
//
// The plan's rule (1b, ????): a kernel's grid may only change with the class, and the class set
// must be small and closed. ragged batches are absorbed by rounding a width UP to its class and
// padding the extra lanes with zero-weight scratch rows, never by letting an arbitrary width reach
// the kernels.
//
// The decode classes are the lane counts a batch may take; the prefill classes are the chunk widths
// the engine actually asks for. 8192 is deliberately absent: it measured 28% slower than 4096 at the
// same prompt (1x.4j), and a class that loses is a class that should not exist.
//
// Nothing here decides memory: the per-class workspace cap is measured at startup and printed as
// "[shape] class width=... bytes=..." (286f6eae). This header only fixes which widths those are.

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::models::qwen3_5::execution {

inline constexpr std::int32_t kDecodeShapeClasses[] = {1, 2, 4, 8};
// Q67/S1: the prefill ladder starts at 16, not at 128. A batch of queued prompts has to be padded up
// to a class, and the old set - {1024, 2048, 4096} - turned four 26-token prompts into a 1024-row
// pass, which the measurements say is slower than running them one by one (5.1 s against 4.9 s).
// 16 is the floor because an sm_89 MMA tile is 16 rows (m16n8k16): a narrower class would still
// issue a 16-row tile, so 8 and below buy nothing. 1 is deliberately absent - the single-request path
// must stay bit-identical, and that is decode_class_for(1)'s job, not this ladder's.
inline constexpr std::int32_t kPrefillShapeClasses[] = {16,  32,  64,   128,  256,
                                                        512, 1024, 2048, 4096};

// Smallest class that can hold `width`, or `width` itself when it is already a class. A width above
// the largest class is returned unchanged: the caller has to either split it or raise the class set,
// and silently clamping it would be the "sizes are not quotas" mistake in a new place.
constexpr std::int32_t decode_class_for(std::int32_t width) {
    for (const auto candidate : kDecodeShapeClasses) {
        if (width <= candidate) { return candidate; }
    }
    return width;
}

constexpr std::int32_t prefill_class_for(std::int32_t width) {
    for (const auto candidate : kPrefillShapeClasses) {
        if (width <= candidate) { return candidate; }
    }
    return width;
}

// A single lane must stay a class of its own: the whole point of keeping batches out of the k=1 path
// is that the untuned single-request path is bit-identical to what shipped.
static_assert(decode_class_for(1) == 1);
static_assert(decode_class_for(3) == 4);
static_assert(prefill_class_for(1024) == 1024);
static_assert(prefill_class_for(1025) == 2048);
static_assert(prefill_class_for(5000) == 5000);
// The small end of the ladder, pinned one boundary at a time: these are the widths a merged prefill
// will actually ask for, and an off-by-one here is a 2x pass for nothing.
static_assert(prefill_class_for(1) == 16);
static_assert(prefill_class_for(16) == 16);
static_assert(prefill_class_for(17) == 32);
static_assert(prefill_class_for(64) == 64);
static_assert(prefill_class_for(65) == 128);
static_assert(prefill_class_for(104) == 128);
static_assert(prefill_class_for(256) == 256);
static_assert(prefill_class_for(500) == 512);

// Q67/S1: is a merged prefill actually cheaper than running the queued prompts one by one? A
// padding ratio cannot answer that, because it ignores what a pass costs before any of it is
// useful: about a second of state setup and expert fill, then ~12 ms per row at small widths
// (measured on the 4090: 26 tokens = 1232 ms, 74 = 2245, 138 = 2596; the marginal drops to ~2 ms
// per row above ~490, which makes this model conservative where it matters).
//
// Two constants, both measured, decide it by comparing predicted costs. At 4x26 tokens merging wins
// 2.2x; at 2x600 it loses by 1.5x and the caller runs them separately, exactly as it should.
struct PrefillCostModel {
    double fixed_ms     = 1200.0;
    double per_token_ms = 12.0;
};

struct PrefillBatchFit {
    std::int32_t width = 0;  // class width to run at; 0 means "do not merge"
    std::size_t lanes  = 0;  // how many of the leading lanes that width covers
    double merged_ms   = 0.0;
    double separate_ms = 0.0;
};

// Takes the longest prefix of `remaining` whose sum still fits the largest class, and picks the
// prefix/class pair that costs least. One lane is never a merge: that is the single-request path,
// which stays bit-identical by taking its exact width.
constexpr PrefillBatchFit prefill_batch_fit(std::span<const std::uint32_t> remaining,
                                            const PrefillCostModel model = {}) {
    constexpr std::int32_t largest = kPrefillShapeClasses[std::size(kPrefillShapeClasses) - 1];
    PrefillBatchFit best{};
    std::uint64_t sum      = 0;
    double separate        = 0.0;
    for (std::size_t i = 0; i < remaining.size(); ++i) {
        const std::uint64_t next = sum + remaining[i];
        if (next > static_cast<std::uint64_t>(largest)) { break; }
        sum = next;
        separate += model.fixed_ms + static_cast<double>(remaining[i]) * model.per_token_ms;
        if (i == 0) { continue; }
        const std::int32_t width = prefill_class_for(static_cast<std::int32_t>(sum));
        const double merged      = model.fixed_ms + static_cast<double>(width) * model.per_token_ms;
        if (merged >= separate) { continue; }
        best.width       = width;
        best.lanes       = i + 1;
        best.merged_ms   = merged;
        best.separate_ms = separate;
    }
    return best;
}

} // namespace ninfer::models::qwen3_5::execution
