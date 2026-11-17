#pragma once

// ninfer::ops::detail - private launch prototype for logits_topk.

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void logits_topk_launch(const Tensor& logits, std::int32_t valid_rows, std::int32_t k, Tensor& ids,
                        Tensor& logprobs, Tensor& log_normalizer, cudaStream_t stream);

} // namespace ninfer::ops::detail
