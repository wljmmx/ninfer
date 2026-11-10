#include "models/qwen3_5/execution/qsa.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {

std::int32_t qsa_index_blocks_per_page(const QsaGeometry& geometry) {
    if (geometry.page_tokens <= 0 || geometry.block_tokens <= 0 ||
        geometry.page_tokens % geometry.block_tokens != 0) {
        throw std::invalid_argument(
            "qsa plan: the index block size must divide the main-cache page");
    }
    return geometry.page_tokens / geometry.block_tokens;
}

std::size_t qsa_index_page_group_bytes(const QsaGeometry& geometry) {
    if (geometry.index_head_dim <= 0 || geometry.layers <= 0) {
        throw std::invalid_argument("qsa plan: invalid index plane geometry");
    }
    // [head_dim, blocks, 1, pages] BF16: one feature-contiguous block per four tokens, one plane per
    // indexed layer, every layer in one page group so a page id names the same row everywhere.
    const auto blocks = static_cast<std::size_t>(qsa_index_blocks_per_page(geometry));
    return static_cast<std::size_t>(geometry.index_head_dim) * blocks *
           static_cast<std::size_t>(geometry.layers) * sizeof(std::uint16_t);
}

std::size_t qsa_index_pool_bytes(const QsaGeometry& geometry, std::uint32_t physical_pages) {
    if (physical_pages == 0) {
        throw std::invalid_argument("qsa plan: the index pool needs at least one page");
    }
    return qsa_index_page_group_bytes(geometry) * static_cast<std::size_t>(physical_pages);
}

std::size_t qsa_tail_state_bytes(const QsaGeometry& geometry, std::int32_t state_slots) {
    if (geometry.index_head_dim <= 0 || geometry.tail_columns <= 0 || geometry.layers <= 0 ||
        state_slots <= 0) {
        throw std::invalid_argument("qsa plan: invalid tail state geometry");
    }
    const std::size_t slots  = static_cast<std::size_t>(state_slots);
    const std::size_t layers = static_cast<std::size_t>(geometry.layers);
    const std::size_t keys   = static_cast<std::size_t>(geometry.index_head_dim) *
                             static_cast<std::size_t>(geometry.tail_columns) * slots *
                             sizeof(std::uint16_t);
    const std::size_t first_rope = static_cast<std::size_t>(geometry.tail_columns) * slots *
                                   sizeof(std::int32_t);
    return layers * (keys + first_rope);
}

std::size_t qsa_indexer_workspace_bytes(const QsaPlanInputs& inputs) {
    if (inputs.query_heads <= 0 || inputs.kv_heads <= 0) {
        throw std::invalid_argument("qsa plan: the indexer needs the layer's head geometry");
    }
    return ops::qsa_indexer_workspace_capacity_bytes(inputs.min_tokens, inputs.max_tokens,
                                                     inputs.batch_size, inputs.indexer,
                                                     inputs.execution);
}

std::size_t qsa_sparse_gqa_workspace_bytes(const QsaPlanInputs& inputs) {
    if (inputs.query_heads <= 0 || inputs.kv_heads <= 0) {
        throw std::invalid_argument("qsa plan: sparse GQA needs the layer's head geometry");
    }
    return ops::qsa_sparse_gqa_workspace_capacity_bytes(inputs.storage, inputs.query_heads,
                                                        inputs.kv_heads, inputs.min_tokens,
                                                        inputs.max_tokens, inputs.batch_size,
                                                        inputs.gqa, inputs.execution);
}

} // namespace ninfer::models::qwen3_5::execution
