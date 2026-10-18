#pragma once

#include "ninfer/ops/rope.h"

#include "core/device.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void rope_dispatch(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
                   CudaExecutionView execution, const RopeScaling& scaling);

void rope_dispatch(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
                   CudaExecutionView execution, const RopeScaling& scaling);

} // namespace ninfer::ops::detail
