#include "ops/launcher/logits_topk_softmax.h"

#include "core/device.h"
#include "ops/kernel/logits_topk_softmax.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// The per-thread top-K storage must be a compile-time size or it lands in local memory.
// Instantiate the exact K the caller asks for and dispatch on it.
template <int K>
void launch_fixed(const __nv_bfloat16* logits, std::int32_t* top_ids, float* top_probs,
                  std::int32_t* argmax_ids, const std::int32_t* id_map, std::int32_t columns,
                  std::int32_t physical_rows, std::int32_t stride, cudaStream_t stream) {
    logits_topk_softmax_kernel<K><<<static_cast<unsigned>(columns), kLogitsTopkBlock, 0, stream>>>(
        logits, top_ids, top_probs, argmax_ids, id_map, physical_rows, physical_rows, stride);
}

} // namespace

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

    const auto* src  = static_cast<const __nv_bfloat16*>(logits.data);
    auto* ids        = static_cast<std::int32_t*>(top_ids.data);
    auto* probs      = static_cast<float*>(top_probs.data);
    auto* argmax     = static_cast<std::int32_t*>(argmax_ids.data);

#define NINFER_LOGITS_TOPK_CASE(K)                                                              \
    case K:                                                                                     \
        launch_fixed<K>(src, ids, probs, argmax, id_map, columns, physical_rows, stride, stream); \
        break;

    switch (top_k) {
        NINFER_LOGITS_TOPK_CASE(1)
        NINFER_LOGITS_TOPK_CASE(2)
        NINFER_LOGITS_TOPK_CASE(3)
        NINFER_LOGITS_TOPK_CASE(4)
        NINFER_LOGITS_TOPK_CASE(5)
        NINFER_LOGITS_TOPK_CASE(6)
        NINFER_LOGITS_TOPK_CASE(7)
        NINFER_LOGITS_TOPK_CASE(8)
        NINFER_LOGITS_TOPK_CASE(9)
        NINFER_LOGITS_TOPK_CASE(10)
        NINFER_LOGITS_TOPK_CASE(11)
        NINFER_LOGITS_TOPK_CASE(12)
        NINFER_LOGITS_TOPK_CASE(13)
        NINFER_LOGITS_TOPK_CASE(14)
        NINFER_LOGITS_TOPK_CASE(15)
        NINFER_LOGITS_TOPK_CASE(16)
        NINFER_LOGITS_TOPK_CASE(17)
        NINFER_LOGITS_TOPK_CASE(18)
        NINFER_LOGITS_TOPK_CASE(19)
        NINFER_LOGITS_TOPK_CASE(20)
        NINFER_LOGITS_TOPK_CASE(21)
        NINFER_LOGITS_TOPK_CASE(22)
        NINFER_LOGITS_TOPK_CASE(23)
        NINFER_LOGITS_TOPK_CASE(24)
        NINFER_LOGITS_TOPK_CASE(25)
        NINFER_LOGITS_TOPK_CASE(26)
        NINFER_LOGITS_TOPK_CASE(27)
        NINFER_LOGITS_TOPK_CASE(28)
        NINFER_LOGITS_TOPK_CASE(29)
        NINFER_LOGITS_TOPK_CASE(30)
        NINFER_LOGITS_TOPK_CASE(31)
        NINFER_LOGITS_TOPK_CASE(32)
    default:
        throw std::invalid_argument("logits_topk_softmax: top_k must be in [1,32]");
    }
#undef NINFER_LOGITS_TOPK_CASE

    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
