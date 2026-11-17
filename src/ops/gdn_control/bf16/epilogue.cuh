#pragma once
#include "ops/common/numeric/math.cuh"
#include "ops/common/numeric/convert.cuh"

namespace ninfer::ops::detail::gdn_control {
__device__ __forceinline__ float decay(float value, float a_log, float bias) {
    return -expf(a_log) * softplus(value + bias);
}
} // namespace ninfer::ops::detail::gdn_control
