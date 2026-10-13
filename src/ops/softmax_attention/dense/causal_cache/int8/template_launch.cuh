#pragma once

// P6 2c C3: this header can launch the INT8 KV attention that writes the page-score counters, so
// the scoring path is kept to a single translation unit. Only int8/launch.cu defines
// NINFER_INT8_KV_SCORING_OWNER; any new include from another TU fails to compile instead of silently
// adding a second scoring launch point that the counters cannot see (appendices CJ.1/C3).
#ifndef NINFER_INT8_KV_SCORING_OWNER
#error "int8 KV scoring kernels: include via int8/template_launch.cuh, from int8/launch.cu (P6 2c C3)"
#endif

#include "core/device.h"
#include "ops/softmax_attention/dense/causal_cache/int8/grouped_mma.cuh"
#include "ops/softmax_attention/dense/causal_cache/int8/tiled_mma.cuh"
#include "ops/softmax_attention/common/causal_merge.cuh"
#include <cstdlib>
#include <stdexcept>

namespace ninfer::ops::detail {

// P6 2c B4 (appendix DH): how many of a call's newest query rows may vote on a page's score. Read once;
// 0 - the default - lets every row vote, which is the behaviour every dump in .temp/kv was produced
// with, so the default path stays bit-identical.
inline std::int32_t kv_page_score_row_span() {
    static const std::int32_t span = [] {
        const char* value = std::getenv("NINFER_KV_PAGE_SCORE_ROW_SPAN");
        if (value == nullptr) { return std::int32_t{0}; }
        const long parsed = std::strtol(value, nullptr, 10);
        return parsed > 0 ? static_cast<std::int32_t>(parsed) : std::int32_t{0};
    }();
    return span;
}

// P6 2c (appendix EE.6): tolerance on the "is this page the row's best" test. A tile also counts when its
// maximum is within `tolerance` of the row's running maximum, turning the hard argmax count into a soft
// competition count - a cheap stand-in for the page's attention mass (mass sums exp(logit - max), so a
// page whose logits sit near the maximum on many rows accumulates far more of it than a page that wins
// outright exactly once). 0 - the default - is the exact comparison, so the default path stays
// bit-identical to every archived run.
inline float kv_page_score_tolerance() {
    static const float tolerance = [] {
        const char* value = std::getenv("NINFER_KV_PAGE_SCORE_TOL");
        if (value == nullptr) { return 0.0f; }
        const float parsed = std::strtof(value, nullptr);
        return parsed > 0.0f ? parsed : 0.0f;
    }();
    return tolerance;
}

// P6 2c (appendix EH): probe for the attention-mass statistic. When set - and a row span is active -
// the two kernels write the page's fixed-point attention mass into the newest-rows plane INSTEAD of the
// hit count, so the existing dump exposes it as the `r` table and the ranking hypothesis can be tested
// before a fourth plane is wired in. 0 - the default - leaves both kernels byte-identical.
inline std::int32_t kv_page_score_mass() {
    static const std::int32_t on = [] {
        const char* value = std::getenv("NINFER_KV_PAGE_SCORE_MASS");
        if (value == nullptr) { return std::int32_t{0}; }
        const long parsed = std::strtol(value, nullptr, 10);
        return parsed > 0 ? std::int32_t{1} : std::int32_t{0};
    }();
    return on;
}

template <class G, class S, bool MultiBatch, bool Masked, bool Writable, class Input,
          bool ParallelQueries = false, bool PackedV = false, bool PackedK = false,
          bool E8Lattice = false, bool E8Root = false, bool V8Root = false>
void launch_int8_kv_grouped_mma(const CausalAttentionOperands& p, Int8KvCacheView<Writable> cache,
                                Input input, CausalKvPartition partition, CausalPartialView partial,
                                cudaStream_t stream) {
    static_assert(Writable == Input::writes_cache);
    validate_quantized_causal_operands<G>(p, cache);
    if ((!ParallelQueries && p.width != S::kTokenTile) || MultiBatch != (p.batch > 1) ||
        Masked != (cache.valid_columns != nullptr) || partition.capacity < 1 ||
        partition.target > CausalKvPartition::kMaxSplits || partition.target < 1 ||
        partition.key_shift < 6 || partition.key_shift > 12 ||
        partition.capacity != partition.active(p.visible_capacity) || !partial.acc ||
        !partial.maximum || !partial.sum)
        throw std::invalid_argument("INT8 grouped attention: invalid schedule/partials");
    if constexpr (Input::writes_cache)
        if (!input.k || !input.v) throw std::invalid_argument("INT8 append requires K/V");
    constexpr auto kernel =
        int8_kv_grouped_mma_kernel<G, S, MultiBatch, Masked, Input, ParallelQueries, PackedV,
                                   PackedK, E8Lattice, E8Root, V8Root>;
    constexpr int bytes = S::kDynamicArena ? S::kArenaBytes : 0;
    if constexpr (S::kDynamicArena) {
        static const auto status =
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        CUDA_CHECK(status);
    }
    const dim3 grid(G::KVHeads * (ParallelQueries ? div_up(p.width, S::kTokenTile) : 1),
                    partition.capacity, p.batch);
    kernel<<<grid, S::kThreads, bytes, stream>>>(
        p.q, input, p.positions, cache.keys, cache.values, cache.key_scales, cache.value_scales,
        cache.tables, cache.valid_columns, cache.table_rows, cache.table_stride, p.width,
        p.visible_capacity, partition, p.scale, partial.acc, partial.maximum, partial.sum,
        cache.page_scores, cache.page_scores_peak, cache.page_scores_recent,
        kv_page_score_row_span(), kv_page_score_tolerance(), kv_page_score_mass());
    CUDA_CHECK(cudaGetLastError());
}

template <class G, class S, bool PackedV = false, bool PackedK = false, bool E8Root = false,
          bool V8Root = false, bool E8Lattice = false>
void launch_int8_kv_tiled_mma(const CausalAttentionOperands& p, Int8KvReadView cache,
                              Tensor* page_score, cudaStream_t stream) {
    validate_quantized_causal_operands<G>(p, cache);
    if (p.batch != 1)
        throw std::invalid_argument("INT8 tiled attention requires a complete single query row");
    const auto invoke = [&]<class Metadata>(Metadata metadata) {
        constexpr auto kernel = int8_kv_tiled_mma_kernel<G, S, Metadata, PackedV, PackedK, E8Root,
                                                         V8Root, E8Lattice>;
        static const auto status = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            Int8KvTiledSmemBytes<S, PackedV>::kBytes);
        CUDA_CHECK(status);
        const dim3 grid(div_up(p.width, S::kQueryRows), G::QHeads);
        // P6 2c: the op-level argument is an already-resolved counter row and overrides the view;
        // the view's buffer is a base that the metadata resolves per row (appendix CA.3).
        unsigned* const override_scores =
            page_score != nullptr ? static_cast<unsigned*>(page_score->data) : nullptr;
        kernel<<<grid, S::kThreads, Int8KvTiledSmemBytes<S, PackedV>::kBytes, stream>>>(
            p.q, cache.keys, cache.values, cache.key_scales, cache.value_scales, metadata,
            p.positions, p.scale, p.out, override_scores, p.width, cache.page_scores_recent,
            kv_page_score_row_span(), kv_page_score_tolerance(), kv_page_score_mass());
        CUDA_CHECK(cudaGetLastError());
    };
    // The tiled path records hit counts only, so it resolves `page_scores`, the count plane. The
    // peak plane is bound as well: a metadata should be a complete description of the row rather
    // than a record of "the planes this particular caller needed", which is exactly how a stale call
    // site would hide (appendices DD/DE).
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
