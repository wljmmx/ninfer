#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void scaled_add_launch(const Tensor& source, const Tensor& coefficient, Tensor& destination,
                       cudaStream_t stream);

} // namespace ninfer::ops::detail
