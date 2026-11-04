// Second scoring owner for the INT8 KV attention: the wide-batch kernel (B>1, W>16) runs in its own
// translation unit so cudafe++ never expands both it and the tiled kernel, while the page-score planes
// still travel through the metadata and therefore reach the same counters. Declaring the owner here is
// the explicit registration the P6 2c guard asks for; nothing about the dump changes.
#define NINFER_INT8_KV_SCORING_OWNER

#include "ops/softmax_attention/dense/causal_cache/int8/batch_wide_mma.cuh"
#include "ops/softmax_attention/dense/causal_cache/int8/instances.h"

#include "core/device.h"
#include "ops/softmax_attention/common/causal_operands.h"

#include <cstdlib>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// Same three probes as template_launch.cuh. Kept inline here so the wide-batch TU does not have to
// include the tiled header (which owns the single-TU guard and the grouped launches).
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

template <class G, class S>
void launch_wide(const CausalAttentionOperands& p, Int8KvReadView cache, Tensor* page_score,
                 cudaStream_t stream) {
    validate_quantized_causal_operands<G>(p, cache);
    const auto invoke = [&]<class Metadata>(Metadata metadata) {
        constexpr auto kernel = int8_kv_batch_wide_mma_kernel<G, S, Metadata>;
        const auto status     = cudaFuncSetAttribute(kernel,
                                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                     Int8KvTiledSmemBytes<S, false>::kBytes);
        CUDA_CHECK(status);
        const dim3 grid(div_up(p.width, S::kQueryRows), G::QHeads, p.batch);
        unsigned* const override_scores =
            page_score != nullptr ? static_cast<unsigned*>(page_score->data) : nullptr;
        kernel<<<grid, S::kThreads, Int8KvTiledSmemBytes<S, false>::kBytes, stream>>>(
            p.q, cache.keys, cache.values, cache.key_scales, cache.value_scales, metadata,
            p.positions, p.scale, p.out, override_scores, p.width, cache.page_scores_recent,
            kv_page_score_row_span(), kv_page_score_tolerance(), kv_page_score_mass());
        CUDA_CHECK(cudaGetLastError());
    };
    if (cache.valid_columns)
        invoke(PagedKVBatchMetadata<true>{cache.tables, cache.valid_columns, cache.table_rows,
                                          cache.table_stride, cache.page_scores,
                                          cache.page_scores_peak, cache.page_scores_recent});
    else
        invoke(PagedKVBatchMetadata<false>{cache.tables, nullptr, cache.table_rows,
                                           cache.table_stride, cache.page_scores,
                                           cache.page_scores_peak, cache.page_scores_recent});
}

} // namespace

void int8_kv_batch_wide_attention(const CausalAttentionOperands& p, Int8KvReadView cache,
                                  Tensor* page_score, cudaStream_t stream) {
    if (p.query_heads == 24)
        launch_wide<CausalD256H24Kv4, Int8KvTiledInstance>(p, cache, page_score, stream);
    else
        launch_wide<CausalD256H16Kv2, Int8KvTiledInstance>(p, cache, page_score, stream);
}

} // namespace ninfer::ops::detail

