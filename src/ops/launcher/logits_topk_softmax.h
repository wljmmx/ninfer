#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Launcher for logits_topk_softmax: per column of logits [V, T], extract
// top-K (id, softmax_probability) pairs and the argmax token ID.
// Outputs are column-major: element(k, t) at offset t * column_stride + k.
// Pass column_stride=0 for the natural contiguous stride (top_k).
// id_map is a DEVICE pointer to V int32 values (indexed→global token
// ID mapping), or null for identity.
void logits_topk_softmax_launch(const Tensor& logits, Tensor& top_ids, Tensor& top_probs,
                                Tensor& argmax_ids, const std::int32_t* id_map,
                                std::int32_t top_k, std::int32_t column_stride,
                                cudaStream_t stream);

} // namespace ninfer::ops::detail
