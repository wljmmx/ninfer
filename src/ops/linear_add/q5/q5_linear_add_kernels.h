#pragma once
#include "core/weight.h"
#include "core/tensor.h"
#include <cuda_runtime.h>

namespace ninfer::ops::detail {
void q5_linear_add_split2_exact_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_sliced_r16_t8_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_sliced_r16_t16_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_sliced_r16_t24_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_sliced_r32_t32_w4_s2_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_sliced_r32_t24_pairwise_launch(const Tensor&, const Weight&, Tensor&,
                                                  cudaStream_t);
void q5_linear_add_sliced_r32_t32_w4_s1_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_sliced_r32_t32_w2_s2_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_sliced_r32_t64_w2_s1_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_mma_r32_t32_k128_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_mma_r32_t128_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_mma_r64_t128_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);

// sm_89 (RTX 4090) v2-tuned decode routes (ported from the v1.2.0 reference).
void q5_linear_add_gemv_residual_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_split2_exact_sm89_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_mma_r64_c16_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_mma_r64_c24_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_mma_r64_c32_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_mma_r64_c64_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_mma_r64_c128_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
} // namespace ninfer::ops::detail
