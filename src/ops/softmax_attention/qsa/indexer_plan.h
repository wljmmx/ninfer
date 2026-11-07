#pragma once

#include "core/layout.h"
#include "ninfer/ops/qsa.h"

#include <algorithm>

namespace ninfer::ops::detail::qsa {

inline constexpr int kIndexerMaxVisibleTokens = 262144;
inline constexpr int kIndexerMaxBlocks        = kIndexerMaxVisibleTokens / 4;
inline constexpr int kIndexerWindowQueries    = 64;
inline constexpr int kIndexerPartition        = 4096;
inline constexpr int kIndexerSelected         = 512;
inline constexpr int kIndexerKeyTile          = 64;

struct IndexerPlan {
    int window_tokens, columns, partitions;
};

inline IndexerPlan indexer_plan(int tokens, int batch, std::uint32_t visible_tokens) {
    const int blocks = visible_tokens / 4;
    return {std::min(tokens, kIndexerWindowQueries / batch),
            (blocks + kIndexerKeyTile - 1) / kIndexerKeyTile * kIndexerKeyTile,
            (blocks + kIndexerPartition - 1) / kIndexerPartition};
}

struct IndexerWorkspace {
    Tensor scores, candidates;
};

template <class Allocator>
IndexerWorkspace allocate_indexer_workspace(Allocator& allocator, IndexerPlan plan, int batch) {
    IndexerWorkspace out;
    if (plan.columns <= kIndexerSelected) return out;
    const int rows = plan.window_tokens * batch;
    out.scores     = allocator.alloc(DType::FP32, {plan.columns, rows}, 256);
    out.candidates = allocator.alloc(DType::I64, {kIndexerSelected, plan.partitions, rows}, 256);
    return out;
}

void launch_indexer(const Tensor& query, const QsaIndexCacheView& cache, const QsaBatchView& batch,
                    IndexerPlan plan, const IndexerWorkspace& scratch, QsaBlockSelection& out,
                    DeviceExecutionView execution);

} // namespace ninfer::ops::detail::qsa
