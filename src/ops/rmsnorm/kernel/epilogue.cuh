#pragma once
#include "ops/rmsnorm/types.h"
#include "ops/common/math.cuh"

namespace ninfer::ops::detail::rmsnorm {
template <Epilogue Kind>
__device__ __forceinline__ float apply_epilogue(float x, float inv, float weight, float z) {
    if constexpr (Kind == Epilogue::Offset) { weight += 1.0f; }
    float value = x * inv * weight;
    if constexpr (Kind == Epilogue::SiluGated) { value *= silu(z); }
    if constexpr (Kind == Epilogue::SigmoidGated) { value *= sigmoid(z); }
    return value;
}

} // namespace ninfer::ops::detail::rmsnorm
