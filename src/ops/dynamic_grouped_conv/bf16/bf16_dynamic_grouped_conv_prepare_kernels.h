#pragma once
#include "core/weight.h"
#include "core/tensor.h"
#include <cuda_runtime.h>

namespace ninfer::ops::detail {
struct DynamicConvPrepareRoute {
    int hidden_rows; // channel count of the residual (2560 or 5120): the reduction's K extent
    int rows;        // tile rows per CTA
    int columns;     // tile columns per CTA
    int split_k;
};

void bf16_dynamic_grouped_conv_prepare_partial_launch(DynamicConvPrepareRoute route,
                                                      const Tensor& input, const Weight& weight,
                                                      float* partial, cudaStream_t stream);
void bf16_dynamic_grouped_conv_prepare_reduce_launch(DynamicConvPrepareRoute route,
                                                     const Tensor& base, const float* partial,
                                                     Tensor& prepared, Tensor& finish,
                                                     cudaStream_t stream);
} // namespace ninfer::ops::detail
