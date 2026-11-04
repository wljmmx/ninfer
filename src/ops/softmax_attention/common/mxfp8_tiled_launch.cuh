#pragma once

#include "core/device.h"
#include "ops/softmax_attention/common/mxfp8_tiled_mma.cuh"
#include "ops/softmax_attention/common/mxfp8_tiled_plan.h"
#include <cstdlib>
#include <stdexcept>

namespace ninfer::ops::detail {

// Same three probes as the INT8 and BF16 paths; this header is shared by fp8 and k8v4, so both read
// the same environment knobs.
inline std::int32_t kv_page_score_row_span() {
    static const std::int32_t span = [] {
        const char* value = std::getenv("NINFER_KV_PAGE_SCORE_ROW_SPAN");
        if (value == nullptr) {
            if (std::getenv("NINFER_KV_EVICT_PROFILE") != nullptr ||
                std::getenv("NINFER_KV_EVICT_KEEP_RATIO") != nullptr) {
                return std::int32_t{8};
            }
            return std::int32_t{0};
        }
        const long parsed = std::strtol(value, nullptr, 10);
        return parsed > 0 ? static_cast<std::int32_t>(parsed) : std::int32_t{0};
    }();
    return span;
}

inline float kv_page_score_tolerance() {
    static const float tolerance = [] {
        const char* value = std::getenv("NINFER_KV_PAGE_SCORE_TOL");
        if (value == nullptr) { return 0.0f; }
        const float parsed = std::strtof(value, nullptr);
        return parsed > 0.0f ? parsed : 0.0f;
    }();
    return tolerance;
}

inline std::int32_t kv_page_score_mass() {
    static const std::int32_t on = [] {
        const char* value = std::getenv("NINFER_KV_PAGE_SCORE_MASS");
        if (value == nullptr) { return std::int32_t{0}; }
        const long parsed = std::strtol(value, nullptr, 10);
        return parsed > 0 ? std::int32_t{1} : std::int32_t{0};
    }();
    return on;
}

template <class G, class S, class Values, class View>
void launch_mxfp8_kv_tiled_mma(const CausalAttentionOperands& p, View cache,
                               CausalKvPartition partition, CausalPartialView partial,
                               Tensor* page_score, cudaStream_t stream) {
    validate_quantized_causal_operands<G>(p, cache);
    if (p.batch != 1 || partition.target < 1 || partition.target > kMxfp8TiledMaxSplits ||
        partition.capacity != partition.active(p.visible_capacity) || !partial.acc ||
        !partial.maximum || !partial.sum)
        throw std::invalid_argument("MXFP8 tiled attention: invalid batch or partial storage");
    const auto invoke = [&]<class Metadata>(Metadata metadata) {
        constexpr auto kernel    = mxfp8_kv_tiled_mma_kernel<G, S, Values, Metadata>;
        static const auto status = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, S::kSharedBytes);
        CUDA_CHECK(status);
        const dim3 grid(div_up(p.width, S::kQueryRows), G::QHeads, partition.capacity);
        unsigned* const override_scores =
            page_score != nullptr ? static_cast<unsigned*>(page_score->data) : nullptr;
        kernel<<<grid, S::kThreads, S::kSharedBytes, stream>>>(
            p.q, cache.keys, cache.values, cache.key_scales, cache.value_scales, metadata,
            p.positions, p.scale, p.width, partition, partial, override_scores,
            cache.page_scores_recent, kv_page_score_row_span(), kv_page_score_tolerance(),
            kv_page_score_mass());
        CUDA_CHECK(cudaGetLastError());
    };
    if (!cache.table_rows)
        invoke(PagedKVDirectMetadata{cache.tables, cache.page_scores, cache.page_scores_peak,
                                     cache.page_scores_recent});
    else if (cache.valid_columns)
        invoke(PagedKVBatchMetadata<true>{cache.tables, cache.valid_columns, cache.table_rows,
                                          cache.table_stride, cache.page_scores,
                                          cache.page_scores_peak, cache.page_scores_recent});
    else
        invoke(PagedKVBatchMetadata<false>{cache.tables, nullptr, cache.table_rows,
                                           cache.table_stride, cache.page_scores,
                                           cache.page_scores_peak, cache.page_scores_recent});
}

} // namespace ninfer::ops::detail
