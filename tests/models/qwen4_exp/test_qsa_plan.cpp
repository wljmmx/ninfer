#include "models/qwen4_exp/execution/qsa.h"
#include "models/qwen3_5/config.h"

#include <cstdint>
#include <exception>
#include <iostream>
#include <string>

using namespace ninfer;
using namespace ninfer::models::qwen3_5;
using namespace ninfer::models::qwen4_exp::execution;

namespace {

int failures = 0;

void check(bool condition, const std::string& label) {
    if (!condition) {
        std::cerr << "qsa plan: " << label << " failed\n";
        ++failures;
    }
}

template <class Fn>
void check_throws(Fn&& fn, const std::string& label) {
    try {
        fn();
        std::cerr << "qsa plan: " << label << " did not throw\n";
        ++failures;
    } catch (const std::exception&) {
    }
}

QsaGeometry qwen4exp_geometry() {
    return QsaGeometry{
        .head_dim       = 256,
        .kv_heads       = 2,
        .index_head_dim = 128,
        .block_tokens   = 4,
        .page_tokens    = kPagedKVPageSize,
        .tail_columns   = 3,
        .layers         = 12,
    };
}

} // namespace

int main() {
    using namespace ninfer;
    const QsaGeometry geometry = qwen4exp_geometry();

    // One compressed index key per four tokens, sixteen of them in a 64-token main page: the index
    // plane needs no table of its own.
    check(qsa_index_blocks_per_page(geometry) == 16, "blocks per page");

    // BF16 [128, 16, 1, pages] for each of the twelve indexed layers, in one page group.
    const std::size_t page_group = qsa_index_page_group_bytes(geometry);
    check(page_group == std::size_t{128} * 16 * 12 * sizeof(std::uint16_t), "page group bytes");
    check(qsa_index_pool_bytes(geometry, 32) == page_group * 32, "pool bytes");
    check(qsa_index_pool_bytes(geometry, 33) - qsa_index_pool_bytes(geometry, 32) == page_group,
          "pool growth is exactly one page group");

    // Tail state: three raw index-key columns plus their first-block RoPE coordinates per slot.
    const std::int32_t slots = 8;
    const std::size_t tail   = qsa_tail_state_bytes(geometry, slots);
    check(tail == std::size_t{12} * (std::size_t{128} * 3 * slots * sizeof(std::uint16_t) +
                                     3 * std::size_t{slots} * sizeof(std::int32_t)),
          "tail bytes");

    // The geometry is derived from the artifact's own config, so a plan can never outlive the
    // numbers it was sized for.
    TextConfig config;
    config.indexed_attention_layers = 12;
    config.attention                = AttentionConfig{24, 2, 256};
    config.indexer                  = IndexerConfig{4, 1, 128, 4, 2048};
    const QsaGeometry declared      = qsa_geometry(config);
    check(declared.layers == 12 && declared.head_dim == 256 && declared.kv_heads == 2 &&
              declared.index_head_dim == 128 && declared.block_tokens == 4 &&
              declared.page_tokens == kPagedKVPageSize && declared.tail_columns == 3,
          "declared geometry");
    check(qsa_index_pool_bytes(declared, 32) == qsa_index_pool_bytes(geometry, 32),
          "declared geometry prices the same index pool");
    check(qsa_tail_state_bytes(declared, slots) == tail,
          "declared geometry prices the same tail state");

    // The transient side is delegated, so the plan cannot drift from the ops' own queries. The
    // interval is what a controller dial resolves: a prefill chunk or a decode width, not a cap.
    QsaPlanInputs inputs{
        .min_tokens  = 1,
        .max_tokens  = 1024,
        .batch_size  = 1,
        .query_heads = 24,
        .kv_heads    = 2,
        .storage     = KvCacheStorage::BFloat16,
        .indexer     = ops::QsaIndexerExecutionEnvelope{.max_visible_tokens = 2048},
        .gqa         = ops::QsaSparseGqaExecutionEnvelope{.max_visible_tokens = 2048},
        .execution   = DeviceExecutionView{nullptr, 128, Device{}},
    };
    const std::size_t indexer = qsa_indexer_workspace_bytes(inputs);
    const std::size_t gqa     = qsa_sparse_gqa_workspace_bytes(inputs);
    check(indexer == ops::qsa_indexer_workspace_capacity_bytes(1, 1024, 1, inputs.indexer,
                                                               inputs.execution),
          "indexer workspace matches the op query");
    check(gqa == ops::qsa_sparse_gqa_workspace_capacity_bytes(
                      KvCacheStorage::BFloat16, 24, 2, 1, 1024, 1, inputs.gqa, inputs.execution),
          "sparse GQA workspace matches the op query");
    // ... and the config-driven form the planner actually calls resolves to the same two numbers.
    check(qsa_indexer_workspace_bytes(qsa_plan_inputs(config, 1, 1024, 1,
                                                      KvCacheStorage::BFloat16, 2048,
                                                      inputs.execution)) == indexer,
          "config-driven indexer workspace");
    check(qsa_sparse_gqa_workspace_bytes(qsa_plan_inputs(config, 1, 1024, 1,
                                                        KvCacheStorage::BFloat16, 2048,
                                                        inputs.execution)) == gqa,
          "config-driven sparse GQA workspace");

    // Wider intervals and larger batches must not shrink: the dial only moves the interval.
    QsaPlanInputs wider = inputs;
    wider.max_tokens    = 2048;
    check(qsa_indexer_workspace_bytes(wider) >= indexer, "indexer workspace is monotone in tokens");
    check(qsa_sparse_gqa_workspace_bytes(wider) >= gqa, "GQA workspace is monotone in tokens");
    QsaPlanInputs batched = inputs;
    batched.batch_size    = 4;
    batched.min_tokens    = 1024;
    check(qsa_indexer_workspace_bytes(batched) >= indexer, "indexer workspace is monotone in batch");

    // Degenerate geometry is refused rather than planned as zero.
    QsaGeometry bad = geometry;
    bad.block_tokens = 5;
    check_throws([&] { (void)qsa_index_blocks_per_page(bad); }, "block size must divide the page");
    check_throws([&] { (void)qsa_index_pool_bytes(geometry, 0); }, "zero-page index pool");
    check_throws([&] { (void)qsa_tail_state_bytes(geometry, 0); }, "zero-slot tail state");
    check_throws([&] { (void)qsa_indexer_workspace_bytes(QsaPlanInputs{}); },
                 "indexer without head geometry");
    config.indexed_attention_layers = 0;
    check(qsa_geometry(config).layers == 0, "a model without indexed layers plans no QSA state");

    std::cout << (failures == 0 ? "OK" : "FAIL") << " qsa_plan (page group " << page_group
              << " B, pool@32 " << qsa_index_pool_bytes(geometry, 32) << " B, tail@" << slots << " "
              << tail << " B, indexer " << indexer << " B, gqa " << gqa << " B)\n";
    return failures == 0 ? 0 : 1;
}
