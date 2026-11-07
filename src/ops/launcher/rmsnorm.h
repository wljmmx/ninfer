#pragma once

// ninfer::ops::detail - private launch prototype for rmsnorm.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

enum class RmsEpilogue { Plain, Offset, SiluGated, SigmoidGated };

void rmsnorm_launch(const Tensor& x, const Tensor& weight, float eps, RmsEpilogue epilogue,
                    const Tensor* z, Tensor& out, std::int32_t multiprocessor_count,
                    cudaStream_t stream);

} // namespace ninfer::ops::detail
