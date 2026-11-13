#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void q6_linear_swiglu_mma_c8_launch(const Tensor& x, const Weight& w, Tensor& out,
                                    cudaStream_t stream);
void q6_linear_swiglu_mma_c48_launch(const Tensor& x, const Weight& w, Tensor& out,
                                     cudaStream_t stream);
void q6_linear_swiglu_mma_c128_launch(const Tensor& x, const Weight& w, Tensor& out,
                                      cudaStream_t stream);

} // namespace ninfer::ops::detail
