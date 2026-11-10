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

// The rk layouts reuse this kernel unchanged: only the cache read-back codec differs, exactly as in
// rk_tiled. PackedV also halves the V slot, so it has to reach the smem request as well.
template <class G, class S, bool PackedV = false, bool PackedK = false, bool E8Root = false,
          bool V8Root = false, bool E8Lattice = false>
void launch_wide(const CausalAttentionOperands& p, Int8KvReadView cache, Tensor* page_score,
                 cudaStream_t stream) {
    validate_quantized_causal_operands<G>(p, cache);
    const auto invoke = [&]<class Metadata>(Metadata metadata) {
        constexpr auto kernel = int8_kv_batch_wide_mma_kernel<G, S, Metadata, PackedV, PackedK,
                                                              E8Root, V8Root, E8Lattice>;
        constexpr int kSmem   = Int8KvTiledSmemBytes<S, PackedV>::kBytes;
        const auto status =
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
        CUDA_CHECK(status);
        const dim3 grid(div_up(p.width, S::kQueryRows), G::QHeads, p.batch);
        unsigned* const override_scores =
            page_score != nullptr ? static_cast<unsigned*>(page_score->data) : nullptr;
        kernel<<<grid, S::kThreads, kSmem, stream>>>(
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

// The geometry is part of the kernel template, and the cache view is the only operand that knows
// the KV head count: 24 query heads run either four (Qwen3.6 attention) or two (Qwen4Exp's indexed
// layers) KV heads, and both pairings need their own instance.
template <bool PackedV = false, bool PackedK = false, bool E8Root = false, bool V8Root = false,
          bool E8Lattice = false>
void launch_wide_by_geometry(const CausalAttentionOperands& p, Int8KvReadView cache,
                             Tensor* page_score, cudaStream_t stream) {
    if (p.query_heads == 24 && cache.kv_heads == 2)
        launch_wide<CausalD256H24Kv2, Int8KvTiledInstance, PackedV, PackedK, E8Root, V8Root,
                    E8Lattice>(p, cache, page_score, stream);
    else if (p.query_heads == 24)
        launch_wide<CausalD256H24Kv4, Int8KvTiledInstance, PackedV, PackedK, E8Root, V8Root,
                    E8Lattice>(p, cache, page_score, stream);
    else
        launch_wide<CausalD256H16Kv2, Int8KvTiledInstance, PackedV, PackedK, E8Root, V8Root,
                    E8Lattice>(p, cache, page_score, stream);
}

} // namespace

void int8_kv_batch_wide_attention(const CausalAttentionOperands& p, Int8KvReadView cache,
                                  Tensor* page_score, cudaStream_t stream) {
    launch_wide_by_geometry(p, cache, page_score, stream);
}

// rk8v4: 8-bit K + int4 V (PackedV only).
void rk8v4_kv_batch_wide_attention(const CausalAttentionOperands& p, Int8KvReadView cache,
                                   Tensor* page_score, cudaStream_t stream) {
    launch_wide_by_geometry<true>(p, cache, page_score, stream);
}

// rk4v4: int4 K + int4 V.
void rk4v4_kv_batch_wide_attention(const CausalAttentionOperands& p, Int8KvReadView cache,
                                   Tensor* page_score, cudaStream_t stream) {
    launch_wide_by_geometry<true, true>(p, cache, page_score, stream);
}

// rk4v4-e8: int4 K carried as E8 lattice + residual (E8Lattice) + int4 V.
void rk4v4e8_kv_batch_wide_attention(const CausalAttentionOperands& p, Int8KvReadView cache,
                                     Tensor* page_score, cudaStream_t stream) {
    launch_wide_by_geometry<true, true, false, false, true>(p, cache, page_score, stream);
}

// rk4v2-e8: int4 K + 2-bit E8 cylinder V (V8Root), both under the E8 lattice carrier.
void rk4v2e8_kv_batch_wide_attention(const CausalAttentionOperands& p, Int8KvReadView cache,
                                     Tensor* page_score, cudaStream_t stream) {
    launch_wide_by_geometry<false, true, false, true, true>(p, cache, page_score, stream);
}

// rk2v4-e8: 2-bit E8 cylinder K (E8Root) + int4 V.
void rk2v4e8_kv_batch_wide_attention(const CausalAttentionOperands& p, Int8KvReadView cache,
                                     Tensor* page_score, cudaStream_t stream) {
    launch_wide_by_geometry<true, false, true>(p, cache, page_score, stream);
}

} // namespace ninfer::ops::detail

