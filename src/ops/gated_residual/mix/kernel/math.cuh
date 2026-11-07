#pragma once

#include "ops/common/math.cuh"

namespace ninfer::ops::detail::hc::mix {
// Fast tanh is qualified for the BF16 mixing output and its private low-rank operands.
// Public FP32 injection uses the more accurate expf sigmoid, outside this helper.
__device__ __forceinline__ float sigmoid_gate(float x) { return sigmoid_approx(x); }
} // namespace ninfer::ops::detail::hc::mix
