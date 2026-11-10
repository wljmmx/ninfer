#include "ninfer/ops/qsa.h"
#include "ops/softmax_attention/qsa/sparse_gqa_plan.h"
#include "ops/softmax_attention/qsa/storage.h"
#include "ops/softmax_attention/qsa/validation.h"

#include <cstdlib>
#include <cstring>

namespace ninfer::ops {
namespace {
constexpr detail::qsa::ViewValidation v{"qsa_sparse_gqa"};

// The low-bit cache routes are ported but kept off by default: only the BF16 profile has been
// through the bit-exact gate on this port. NINFER_QSA_SPARSE_GQA_QUANT_KV=1 opts in.
bool quantized_kv_enabled() {
    const char* value = std::getenv("NINFER_QSA_SPARSE_GQA_QUANT_KV");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

void profile(KvCacheStorage storage, int query_heads, int kv_heads, int lo, int hi, int batch,
             QsaSparseGqaExecutionEnvelope envelope, DeviceExecutionView execution) {
    v.require((query_heads == 24 && kv_heads == 2) || (query_heads == 12 && kv_heads == 1),
              "query/KV heads");
    v.require(envelope.max_visible_tokens <= 262144, "visible-token capacity");
    // The implemented set is shared with prepare; the low-bit profiles additionally sit behind
    // NINFER_QSA_SPARSE_GQA_QUANT_KV until each one has been through its own gate on this port.
    if (!detail::qsa::qsa_storage_supported(storage)) { detail::qsa::throw_unsupported_storage("qsa_sparse_gqa", storage); }
    const bool quantized = storage != KvCacheStorage::BFloat16;
    v.require(!quantized || quantized_kv_enabled(), "cache storage (NINFER_QSA_SPARSE_GQA_QUANT_KV)");
    v.require(lo > 0 && hi >= lo && batch >= 1 && batch <= 8, "T/B interval");
    v.require(execution.multiprocessor_count > 0, "SM count");
}
} // namespace

std::size_t qsa_sparse_gqa_workspace_capacity_bytes(KvCacheStorage storage,
                                                    std::int32_t query_heads, std::int32_t kv_heads,
                                                    std::int32_t lo, std::int32_t hi,
                                                    std::int32_t batch,
                                                    QsaSparseGqaExecutionEnvelope envelope,
                                                    DeviceExecutionView execution) {
    profile(storage, query_heads, kv_heads, lo, hi, batch, envelope, execution);
    std::size_t maximum = 0;
    // Above this bound every extent uses direct output. Below it, allocation is not monotonic
    // because the split count decreases with query parallelism.
    const int end =
        std::min(hi, (detail::qsa::sparse_gqa_direct_queries(kv_heads) - 1) / batch);
    for (int t = lo; t <= end; ++t) {
        WorkspaceLayoutBuilder layout;
        detail::qsa::allocate_sparse_gqa_workspace(
            layout, t, batch, query_heads,
            detail::qsa::sparse_gqa_plan(t, batch, kv_heads, envelope,
                                         execution.multiprocessor_count));
        maximum = std::max(maximum, layout.peak_bytes(1));
    }
    return maximum;
}

void qsa_sparse_gqa(const Tensor& query, const Tensor& gate, const PagedKVBatchLayerView& cache,
                    const QsaBatchView& batch, const QsaBlockSelection& selected,
                    QsaSparseGqaExecutionEnvelope envelope, WorkspaceArena& workspace, Tensor& out,
                    DeviceExecutionView execution) {
    const int tokens = query.ne[2], rows = query.ne[3], pages = cache.k_pages.ne[3];
    const int query_heads = query.ne[1], kv_heads = cache.num_kv_heads;
    profile(cache.storage, query_heads, kv_heads, tokens, tokens, rows, envelope, execution);
    v.dense(query, DType::BF16, {256, query_heads, tokens, rows}, 16, "query");
    v.dense(gate, DType::BF16, {256, query_heads, tokens, rows}, 16, "gate");
    v.dense(out, DType::BF16, {256, query_heads, tokens, rows}, 16, "output");
    v.require(cache.head_dim == 256, "cache geometry");
    const auto layout = paged_kv_storage_layout(cache.storage, 256);
    const auto plane  = [&](const Tensor& data, const Tensor& scales, PagedKVVectorLayout format) {
        v.plane(data, format.data_dtype, {format.data_leading_extent, 64, kv_heads, pages},
                 "cache data");
        if (format.has_scale())
            v.plane(scales, format.scale_dtype, {format.scale_leading_extent, 64, kv_heads, pages},
                     "cache scales");
        else
            v.require(!scales.data, "absent scales");
    };
    plane(cache.k_pages, cache.k_scale_pages, layout.key);
    plane(cache.v_pages, cache.v_scale_pages, layout.value);
    v.plane(cache.block_tables, DType::I32,
            {cache.block_tables.ne[0], cache.block_tables.ne[1], 1, 1}, "block tables");
    v.dense(batch.positions, DType::I32, {tokens, rows, 1, 1}, 4, "positions");
    v.metadata(batch.valid_columns, rows, "valid columns");
    v.metadata(batch.table_rows, rows, "table rows");
    v.dense(selected.block_ids, DType::I32, {512, tokens, rows, 1}, 4, "block ids");
    v.dense(selected.block_counts, DType::I32, {tokens, rows, 1, 1}, 4, "block counts");
    const Tensor* inputs[]{&query,
                           &gate,
                           &cache.k_pages,
                           &cache.v_pages,
                           &cache.k_scale_pages,
                           &cache.v_scale_pages,
                           &cache.block_tables,
                           &batch.positions,
                           &batch.valid_columns,
                           &batch.table_rows,
                           &selected.block_ids,
                           &selected.block_counts};
    for (const auto* input : inputs)
        if (input->data) v.require(!v.overlap(out, *input), "output/input overlap");
    const auto plan = detail::qsa::sparse_gqa_plan(tokens, rows, kv_heads, envelope,
                                                   execution.multiprocessor_count);
    auto scope         = workspace.scope();
    const auto scratch = detail::qsa::allocate_sparse_gqa_workspace(workspace, tokens, rows,
                                                                    query_heads, plan);
    for (const auto* work : {&scratch.numerator, &scratch.maximum, &scratch.sum}) {
        if (!work->data) continue;
        v.require(!v.overlap(*work, out), "workspace/output overlap");
        for (const auto* input : inputs)
            if (input->data) v.require(!v.overlap(*work, *input), "workspace/input overlap");
    }
    detail::qsa::launch_sparse_gqa(query, gate, cache, batch, selected, plan, scratch, out,
                                   execution);
}
} // namespace ninfer::ops
