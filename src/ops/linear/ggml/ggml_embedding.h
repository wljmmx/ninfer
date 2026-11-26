#pragma once

#include "core/weight.h"
#include "ninfer/ops/embedding.h"

namespace ninfer::ops::detail::ggml {

// y[t] = table[ids[t]] for a `ggml_block_v1` table: the selected rows are decoded from their stored
// blocks straight into the BF16 output, with no dequantised copy of the table.
void launch_embedding_gather(const Tensor& ids, const Weight& table, Tensor& out,
                             cudaStream_t stream);

} // namespace ninfer::ops::detail::ggml
