#include "ops/launcher/logits_topk_softmax.h"

#include "core/device.h"
#include "ops/kernel/logits_topk_softmax.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

void logits_topk_softmax_launch(const Tensor& logits, Tensor& top_ids, Tensor& top_probs,
                                Tensor& argmax_ids, const std::int32_t* id_map,
                                std::int32_t top_k, std::int32_t column_stride,
                                cudaStream_t stream) {
    if (logits.dtype != DType::BF16) {
        throw std::invalid_argument("logits_topk_softmax: logits must be BF16");
    }
    if (top_ids.dtype != DType::I32 || top_probs.dtype != DType::FP32 ||
        argmax_ids.dtype != DType::I32) {
        throw std::invalid_argument("logits_topk_softmax: output dtypes must be I32/FP32/I32");
    }
    if (top_k <= 0 || top_k > kLogitsTopkMax) {
        throw std::invalid_argument("logits_topk_softmax: top_k must be in [1,32]");
    }

    const std::int32_t physical_rows = static_cast<std::int32_t>(logits.ne[0]);
    const std::int32_t columns       = static_cast<std::int32_t>(logits.ne[1]);
    const std::int32_t stride = column_stride > 0 ? column_stride : top_k;
    // Capacity check only for the contiguous case (column_stride=0); strided
    // writes are caller-managed (the base pointer may point into a larger
    // tensor's step slice, so the remaining capacity is not simply stride*columns).
    if (column_stride <= 0) {
        if (top_ids.ne[0] < stride * columns) {
            throw std::invalid_argument(
                "logits_topk_softmax: top_ids allocation too small for stride");
        }
        if (top_probs.ne[0] < stride * columns) {
            throw std::invalid_argument(
                "logits_topk_softmax: top_probs allocation too small for stride");
        }
    }
    if (top_ids.data == nullptr || top_probs.data == nullptr) {
        throw std::invalid_argument("logits_topk_softmax: output data must be non-null");
    }
    if (argmax_ids.ne[0] < columns) {
        throw std::invalid_argument("logits_topk_softmax: argmax_ids shape mismatch");
    }

    logits_topk_softmax_kernel<<<static_cast<unsigned>(columns), kLogitsTopkBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data),
        static_cast<std::int32_t*>(top_ids.data), static_cast<float*>(top_probs.data),
        static_cast<std::int32_t*>(argmax_ids.data), id_map, physical_rows, physical_rows, top_k,
        stride);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
