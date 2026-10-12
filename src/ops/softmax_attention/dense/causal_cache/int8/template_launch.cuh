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
#include <stdexcept>

namespace ninfer::ops::detail {

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
        cache.page_scores);
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
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, S::kSharedBytes);
        CUDA_CHECK(status);
        const dim3 grid(div_up(p.width, S::kQueryRows), G::QHeads);
        // P6 2c: the op-level argument is an already-resolved counter row and overrides the view;
        // the view's buffer is a base that the metadata resolves per row (appendix CA.3).
        unsigned* const override_scores =
            page_score != nullptr ? static_cast<unsigned*>(page_score->data) : nullptr;
        kernel<<<grid, S::kThreads, S::kSharedBytes, stream>>>(
            p.q, cache.keys, cache.values, cache.key_scales, cache.value_scales, metadata,
            p.positions, p.scale, p.out, override_scores, p.width);
        CUDA_CHECK(cudaGetLastError());
    };
    if (!cache.table_rows)
        invoke(PagedKVDirectMetadata{cache.tables, cache.page_scores});
    else if (cache.valid_columns)
        invoke(PagedKVBatchMetadata<true>{cache.tables, cache.valid_columns, cache.table_rows,
                                          cache.table_stride, cache.page_scores});
    else
        invoke(PagedKVBatchMetadata<false>{cache.tables, nullptr, cache.table_rows,
                                           cache.table_stride, cache.page_scores});
}

} // namespace ninfer::ops::detail
