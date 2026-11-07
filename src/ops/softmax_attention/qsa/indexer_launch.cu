#include "ninfer/ops/qsa.h"
#include "ops/softmax_attention/qsa/indexer_score.cuh"
#include "ops/softmax_attention/qsa/indexer_select.cuh"

namespace ninfer::ops::detail::qsa {
namespace {
template <int QueryTile, bool VectorCache>
void launch_score(IndexerArguments a, DeviceExecutionView execution) {
    constexpr int threads = QueryTile > 16 ? QueryTile * 8 : 128;
    constexpr int bytes   = (QueryTile * 4 + kIndexerKeyTile) * 128 * sizeof(__nv_bfloat16);
    constexpr auto kernel = indexer_score_mma_kernel<QueryTile, VectorCache>;
    if constexpr (bytes >= 48 * 1024) {
        static const auto status =
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        CUDA_CHECK(status);
    }
    const int tasks =
        a.batch * ((a.width + QueryTile - 1) / QueryTile) * (a.columns / kIndexerKeyTile);
    const int waves   = a.width <= 2 || a.width > 32 ? 4 : 2;
    const int workers = std::max(1, std::min(waves * execution.multiprocessor_count, tasks));
    kernel<<<workers, threads, bytes, execution.stream>>>(a);
}

template <bool VectorCache>
void launch_score(IndexerArguments a, DeviceExecutionView execution) {
    if (a.width <= 4)
        launch_score<4, VectorCache>(a, execution);
    else if (a.width <= 8)
        launch_score<8, VectorCache>(a, execution);
    else if (a.width <= 16)
        launch_score<16, VectorCache>(a, execution);
    else
        launch_score<32, VectorCache>(a, execution);
}
} // namespace

void launch_indexer(const Tensor& query, const QsaIndexCacheView& cache, const QsaBatchView& batch,
                    IndexerPlan plan, const IndexerWorkspace& scratch, QsaBlockSelection& out,
                    DeviceExecutionView execution) {
    const auto i32 = [](const Tensor& t) { return static_cast<const std::int32_t*>(t.data); };
    IndexerArguments a{static_cast<const __nv_bfloat16*>(query.data),
                       static_cast<const __nv_bfloat16*>(cache.keys.data),
                       i32(batch.positions),
                       i32(batch.valid_columns),
                       i32(batch.table_rows),
                       i32(cache.block_tables),
                       cache.keys.nb[1] / 2,
                       cache.keys.nb[3] / 2,
                       cache.block_tables.nb[1] / 4,
                       static_cast<float*>(scratch.scores.data),
                       static_cast<std::uint64_t*>(scratch.candidates.data),
                       static_cast<std::int32_t*>(out.block_ids.data),
                       static_cast<std::int32_t*>(out.block_counts.data),
                       query.ne[2],
                       query.ne[3],
                       0,
                       plan.window_tokens,
                       plan.columns,
                       plan.partitions};
    const bool vector_cache = reinterpret_cast<std::uintptr_t>(cache.keys.data) % 16 == 0 &&
                              cache.keys.nb[1] % 16 == 0 && cache.keys.nb[3] % 16 == 0;
    for (std::int64_t begin = 0; begin < a.tokens; begin += plan.window_tokens) {
        a.begin                  = int(begin);
        a.width                  = std::min(plan.window_tokens, a.tokens - a.begin);
        const int waves          = a.width > 32 ? 4 : 2;
        const int select_workers = std::max(
            1, std::min(waves * execution.multiprocessor_count, a.width * a.batch * a.partitions));
        // Every capture has the same three stages per window, including all-select and inert
        // replays. Actual per-row context affects only Device work maps and masked reads.
        if (vector_cache)
            launch_score<true>(a, execution);
        else
            launch_score<false>(a, execution);
        indexer_partition_kernel<kIndexerPartition>
            <<<select_workers, 256, 0, execution.stream>>>(a);
        indexer_finish_kernel<kIndexerPartition>
            <<<a.width * a.batch, 256, 0, execution.stream>>>(a);
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail::qsa
