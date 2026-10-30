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

#include <cstdint>

namespace ninfer::models::qwen3_5::execution {

inline constexpr std::int32_t kDecodeShapeClasses[] = {1, 2, 4, 8};
inline constexpr std::int32_t kPrefillShapeClasses[] = {1024, 2048, 4096};

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

} // namespace ninfer::models::qwen3_5::execution
