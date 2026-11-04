#pragma once

#include "core/paged_kv_cache.h"
#include "ninfer/ops/softmax_attention.h"

namespace ninfer::ops::detail {

// Dedicated path for a batch wider than the verify width: B in 2..8 with W > 16. The grouped path is
// built for narrow batches and the tiled path is explicitly single-row, so this shape has its own
// kernel (batch_wide_mma.cuh) instead of widening either one. Split-K partials are not used: one CTA
// owns a query tile and walks the whole key range, which keeps the correctness argument small. Add
// split-K here later if the measurement asks for it; nothing outside this file depends on it.
void bf16_kv_batch_wide_attention(const Tensor& q, const Tensor& positions, const Tensor& valid_columns,
                                  const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
                                  CausalAttentionExecutionEnvelope envelope, Tensor& out,
                                  DeviceExecutionView execution, Tensor* page_score = nullptr);

} // namespace ninfer::ops::detail
