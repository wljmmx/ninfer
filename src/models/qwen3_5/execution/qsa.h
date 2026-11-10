#pragma once

#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/qsa.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::models::qwen3_5::execution {

// Qwen4Exp QSA layer geometry, in the units the ops take: 24 query heads over 2 KV heads of 256,
// four index heads of 128, and one compressed index key per four tokens. The index cache is addressed
// in the main cache's page space - sixteen four-token blocks to one 64-token page - so a QSA layer
// needs no table of its own.
struct QsaGeometry {
    std::int32_t head_dim       = 0;  // 256, main cache feature extent
    std::int32_t kv_heads       = 0;  // 2
    std::int32_t index_head_dim = 0;  // 128
    std::int32_t block_tokens   = 0;  // 4
    std::int32_t page_tokens    = 0;  // 64, kPagedKVPageSize
    std::int32_t tail_columns   = 0;  // 3, the incomplete block's visible prefix
    std::int32_t layers         = 0;  // indexed layer count
};

// Everything the transient side depends on, taken from the controller's resolved dial values rather
// than compiled-in caps: `min_tokens`/`max_tokens` are the prefill chunk interval the plan resolved
// (or a decode width) and `batch_size` is its concurrency.
struct QsaPlanInputs {
    std::int32_t min_tokens  = 0;
    std::int32_t max_tokens  = 0;
    std::int32_t batch_size  = 0;
    std::int32_t query_heads = 0;  // 24
    std::int32_t kv_heads    = 0;  // 2
    KvCacheStorage storage      = KvCacheStorage::BFloat16;
    ops::QsaIndexerExecutionEnvelope indexer{};
    ops::QsaSparseGqaExecutionEnvelope gqa{};
    DeviceExecutionView execution{};
};

// Four-token blocks that share one 64-token main-cache page.
[[nodiscard]] std::int32_t qsa_index_blocks_per_page(const QsaGeometry& geometry);

// Persistent index plane: BF16 [index_head_dim, blocks_per_page, 1, pages] for every indexed layer,
// page-major, exactly the layout `QsaIndexCacheView.keys` names.
[[nodiscard]] std::size_t qsa_index_page_group_bytes(const QsaGeometry& geometry);
[[nodiscard]] std::size_t qsa_index_pool_bytes(const QsaGeometry& geometry,
                                               std::uint32_t physical_pages);

// Persistent tail state: per layer the three raw index-key columns and the block's first RoPE
// coordinates, over `state_slots` independently selectable slots (QsaTailStateView).
[[nodiscard]] std::size_t qsa_tail_state_bytes(const QsaGeometry& geometry, std::int32_t state_slots);

// Transient scratch of the two stateful ops, delegated to their own capacity queries so a plan can
// never drift from what execution asks for.
[[nodiscard]] std::size_t qsa_indexer_workspace_bytes(const QsaPlanInputs& inputs);
[[nodiscard]] std::size_t qsa_sparse_gqa_workspace_bytes(const QsaPlanInputs& inputs);

} // namespace ninfer::models::qwen3_5::execution
