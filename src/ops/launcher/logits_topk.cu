// Implements: include/ninfer/ops/logits_topk.h
// Match: wrapper-validated contiguous tensors and a valid vocabulary row count.
// Algorithm assumptions: one independent CTA per column; no global workspace.
#include "ops/launcher/logits_topk.h"

#include "core/device.h"
#include "ops/kernel/logits_topk.cuh"

namespace ninfer::ops::detail {

void logits_topk_launch(const Tensor& logits, std::int32_t valid_rows, std::int32_t k, Tensor& ids,
                        Tensor& logprobs, Tensor& log_normalizer, cudaStream_t stream) {
    const auto columns = static_cast<unsigned int>(logits.ne[1]);
    logits_topk_kernel<kLogitsTopkBlock><<<columns, kLogitsTopkBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), k, static_cast<std::int32_t*>(ids.data),
        static_cast<float*>(logprobs.data), static_cast<float*>(log_normalizer.data), valid_rows,
        logits.ne[0]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
