#include "models/qwen4_exp/execution/qsa.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen4_exp::execution {
namespace {

// This header is also used by the planner before any execution object exists, so it reads the
// config's own integers instead of borrowing the execution dimension wrapper.
std::int32_t positive_dimension(std::uint64_t value, const char* label) {
    if (value == 0 || value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument(std::string("qsa plan: invalid ") + label);
    }
    return static_cast<std::int32_t>(value);
}

} // namespace

QsaGeometry qsa_geometry(const qwen3_5::TextConfig& config) {
    QsaGeometry geometry;
    if (config.indexed_attention_layers == 0) { return geometry; }
    if (!config.indexer || !config.attention) {
        throw std::invalid_argument("qsa plan: indexed layers need an indexer and attention config");
    }
    geometry.layers         = positive_dimension(config.indexed_attention_layers, "layer count");
    geometry.head_dim       = positive_dimension(config.attention->head_dim, "head extent");
    geometry.kv_heads       = positive_dimension(config.attention->num_key_value_heads, "KV heads");
    geometry.index_head_dim = positive_dimension(config.indexer->head_dim, "index head extent");
    geometry.block_tokens   = positive_dimension(config.indexer->compress_ratio, "block size");
    geometry.page_tokens    = static_cast<std::int32_t>(kPagedKVPageSize);
    // The raw index-key tail holds the unfinished block's already-projected columns: a block of
    // `block_tokens` tokens has `block_tokens - 1` of them still unwritten when the next call
    // starts inside it.
    geometry.tail_columns = geometry.block_tokens - 1;
    if (geometry.tail_columns <= 0 ||
        geometry.page_tokens % geometry.block_tokens != 0) {
        throw std::invalid_argument("qsa plan: the index block size is not usable");
    }
    return geometry;
}

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

QsaPlanInputs qsa_plan_inputs(const qwen3_5::TextConfig& config, std::int32_t first,
                              std::int32_t last, std::int32_t batch, KvCacheStorage storage,
                              std::uint32_t visible_tokens, DeviceExecutionView execution) {
    if (!config.indexer) { throw std::invalid_argument("qsa plan: the layer has no indexer"); }
    QsaPlanInputs inputs;
    inputs.min_tokens  = first;
    inputs.max_tokens  = last;
    inputs.batch_size  = batch;
    inputs.query_heads = positive_dimension(config.attention->num_attention_heads, "query heads");
    inputs.kv_heads    = positive_dimension(config.attention->num_key_value_heads, "KV heads");
    inputs.storage     = storage;
    // Both stateful Ops take the same host-side bound: the deepest position any live query can
    // name. It is a resource promise over the batch maximum, never a mask, and it fixes the launch
    // topology and the scratch size - hence one number for plan and execution.
    inputs.indexer.max_visible_tokens = visible_tokens;
    inputs.gqa.max_visible_tokens     = visible_tokens;
    inputs.execution                  = execution;
    return inputs;
}

} // namespace ninfer::models::qwen4_exp::execution
