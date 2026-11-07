#pragma once

#include "core/layout.h"
#include "ninfer/ops/qsa.h"

#include <algorithm>

namespace ninfer::ops::detail::qsa {

inline constexpr int kSparseGqaDirectQueries = 256;
enum class SparseGqaInstance { Mma, Simt, FullPrefix };

struct SparseGqaPlan {
    int key_rows, splits;
    SparseGqaInstance instance;
    bool partial;
};

inline SparseGqaPlan sparse_gqa_plan(int tokens, int batch, QsaSparseGqaExecutionEnvelope envelope,
                                     int sms) {
    const auto queries = std::int64_t(tokens) * batch;
    const auto visible = envelope.max_visible_tokens;
    if (queries >= kSparseGqaDirectQueries)
        return visible <= 2051 ? SparseGqaPlan{64, 1, SparseGqaInstance::FullPrefix, false}
                               : SparseGqaPlan{32, 1, SparseGqaInstance::Mma, false};
    // Budget parallel partitions independently of history. The envelope only bounds the
    // selected support and admits a short-context arithmetic schedule.
    const auto target =
        std::max<std::int64_t>(1, (std::int64_t(sms) * 3 / 4 + queries - 1) / queries);
    int splits = 1;
    while (splits < target && splits < 32) splits *= 2;
    const bool simt = visible <= 64;
    if (simt) splits = std::max(1, splits / 12);
    const int tiles = std::max(1, (int(std::min(visible, 2051u)) + 31) / 32);
    return {32, std::min(splits, tiles), simt ? SparseGqaInstance::Simt : SparseGqaInstance::Mma,
            true};
}

struct SparseGqaWorkspace {
    Tensor numerator, maximum, sum;
};

template <class Allocator>
SparseGqaWorkspace allocate_sparse_gqa_workspace(Allocator& allocator, int tokens, int batch,
                                                 SparseGqaPlan plan) {
    if (!plan.partial) return {};
    return {allocator.alloc(DType::FP32, {256, 24, tokens, batch * plan.splits}, 256),
            allocator.alloc(DType::FP32, {24, tokens, batch * plan.splits}, 256),
            allocator.alloc(DType::FP32, {24, tokens, batch * plan.splits}, 256)};
}

void launch_sparse_gqa(const Tensor& query, const Tensor& gate, const PagedKVBatchLayerView& cache,
                       const QsaBatchView& batch, const QsaBlockSelection& selected,
                       SparseGqaPlan plan, const SparseGqaWorkspace& scratch, Tensor& out,
                       DeviceExecutionView execution);

} // namespace ninfer::ops::detail::qsa
