#pragma once

#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Launcher for apply_sampling_mask: copies BF16 [V, T] logits, writing -inf for every row the
// column's bitmap forbids. A null bitmap makes it an exact device copy.
// `column_stride_words` is the distance between two columns' bitmaps. Zero means "packed", i.e.
// mask.stride; a speculative bitmap laid out per (batch row, draft position) passes the whole
// window so column b lands on its own row's column 0.
void apply_sampling_mask(const void* source, void* destination, std::int32_t rows,
                         std::int32_t columns, SamplingMask mask,
                         std::int32_t column_stride_words, cudaStream_t stream);

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
