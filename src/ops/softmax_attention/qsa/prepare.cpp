#include "ninfer/ops/qsa.h"
#include "ops/softmax_attention/qsa/launch.h"
#include "ops/softmax_attention/qsa/validation.h"

#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>

namespace ninfer::ops {
namespace {

constexpr detail::qsa::ViewValidation v{"qsa_prepare"};

void dense(const Tensor& t, DType dtype, std::array<int, 4> shape, int alignment,
           const char* label) {
    v.require(t.dtype == dtype && t.is_contiguous() && v.aligned(t.data, alignment), label);
    for (int d = 0; d < 4; ++d) v.require(t.ne[d] == shape[d] && shape[d] > 0, label);
}

void plane(const Tensor& t, DType dtype, std::array<int, 4> shape, const char* label) {
    const std::int64_t element = dtype_size(dtype);
    v.require(t.dtype == dtype && t.nb[0] == element && v.aligned(t.data, element), label);
    std::uint64_t span = element;
    for (int d = 0; d < 4; ++d) {
        v.require(t.ne[d] == shape[d] && shape[d] > 0, label);
        if (shape[d] == 1) continue;
        v.require(t.nb[d] >= static_cast<std::int64_t>(span) && t.nb[d] % element == 0, label);
        const std::uint64_t stride = t.nb[d], count = shape[d] - 1;
        v.require(stride <= (std::numeric_limits<std::int64_t>::max() - span) / count, label);
        span += count * stride;
    }
    v.require(span <= std::numeric_limits<std::uintptr_t>::max() -
                          reinterpret_cast<std::uintptr_t>(t.data),
              label);
}

bool identical(const Tensor& a, const Tensor& b) {
    if (a.data != b.data || a.dtype != b.dtype) return false;
    for (int d = 0; d < 4; ++d)
        if (a.ne[d] != b.ne[d] || a.nb[d] != b.nb[d]) return false;
    return true;
}

bool overlap(const Tensor& a, const Tensor& b) {
    const auto x  = reinterpret_cast<std::uintptr_t>(a.data),
               y  = reinterpret_cast<std::uintptr_t>(b.data);
    const auto nx = a.bytes(), ny = b.bytes();
    v.require(nx <= std::numeric_limits<std::uintptr_t>::max() - x &&
                  ny <= std::numeric_limits<std::uintptr_t>::max() - y,
              "address range");
    return x < y + ny && y < x + nx;
}

} // namespace

void qsa_prepare(const QsaProjectionView& p, const QsaPrepareParameters& parameters,
                 const QsaBatchView& batch, const Tensor& rope_positions,
                 const QsaTailStateView& before, QsaCacheView cache, QsaRawRecordWriteView records,
                 Tensor& query, Tensor& index_query, DeviceExecutionView execution) {
    const int tokens = p.query.ne[2], rows = p.query.ne[3], axes = rope_positions.ne[1];
    v.require(tokens > 0 && rows >= 1 && rows <= 8 && (axes == 1 || axes == 3), "T/B/A");
    v.require(execution.multiprocessor_count > 0, "SM count");
    v.require(parameters.rms_epsilon > 0 && std::isfinite(parameters.rms_epsilon) &&
                  parameters.rope_theta > 0 && std::isfinite(parameters.rope_theta),
              "epsilon/theta");
    dense(p.query, DType::BF16, {256, 24, tokens, rows}, 16, "query projection");
    dense(p.key, DType::BF16, {256, 2, tokens, rows}, 16, "key projection");
    dense(p.value, DType::BF16, {256, 2, tokens, rows}, 16, "value projection");
    dense(p.index_query, DType::BF16, {128, 4, tokens, rows}, 16, "index-query projection");
    dense(p.index_key, DType::BF16, {128, tokens, rows, 1}, 16, "index-key projection");
    dense(query, DType::BF16, {256, 24, tokens, rows}, 16, "query output");
    dense(index_query, DType::BF16, {128, 4, tokens, rows}, 16, "index-query output");
    dense(parameters.query_norm, DType::BF16, {256, 1, 1, 1}, 2, "query norm");
    dense(parameters.key_norm, DType::BF16, {256, 1, 1, 1}, 2, "key norm");
    dense(parameters.index_query_norm, DType::BF16, {128, 1, 1, 1}, 2, "index-query norm");
    dense(parameters.index_key_norm, DType::BF16, {128, 1, 1, 1}, 2, "index-key norm");
    dense(batch.positions, DType::I32, {tokens, rows, 1, 1}, 4, "positions");
    dense(rope_positions, DType::I32, {tokens, axes, rows, 1}, 4, "RoPE positions");
    v.metadata(batch.valid_columns, rows, "valid columns");
    v.metadata(batch.table_rows, rows, "table rows");
    v.metadata(records.offsets, rows, "record offsets");
    v.state(before, rows);
    const int capacity = records.raw_keys.ne[1];
    v.columns(records.raw_keys, DType::BF16, 128, capacity, rows, "raw records");
    v.columns(records.rope_positions, DType::I32, 3, capacity, rows, "coordinate records");

    const auto& main = cache.main;
    const int pages  = main.k_pages.ne[3];
    v.require(main.storage == KvCacheStorage::BFloat16 && main.head_dim == 256 &&
                  main.num_kv_heads == 2 && main.k_scale_pages.data == nullptr &&
                  main.v_scale_pages.data == nullptr,
              "BF16 cache geometry/storage");
    plane(main.k_pages, DType::BF16, {256, 64, 2, pages}, "K pages");
    plane(main.v_pages, DType::FP16, {256, 64, 2, pages}, "V pages");
    plane(cache.index.keys, DType::BF16, {128, 16, 1, pages}, "index pages");
    plane(main.block_tables, DType::I32, {main.block_tables.ne[0], main.block_tables.ne[1], 1, 1},
          "block tables");
    v.require(identical(main.block_tables, cache.index.block_tables), "shared block tables");

    v.require(!overlap(query, index_query), "output overlap");
    for (const Tensor* out : {&query, &index_query}) {
        for (const Tensor* in : {&p.query, &p.key, &p.value, &p.index_query, &p.index_key}) {
            const bool allowed =
                (out == &query && in == &p.query) || (out == &index_query && in == &p.index_query);
            v.require((allowed && identical(*out, *in)) || !overlap(*out, *in), "projection alias");
        }
        for (const Tensor* in : std::initializer_list<const Tensor*>{
                 &parameters.query_norm, &parameters.key_norm, &parameters.index_query_norm,
                 &parameters.index_key_norm, &batch.positions, &batch.valid_columns,
                 &batch.table_rows, &rope_positions, &before.slots, &records.offsets})
            v.require(!overlap(*out, *in), "output/parameter overlap");
    }
    detail::qsa::launch_prepare(p, parameters, batch, rope_positions, before, cache, records, query,
                                index_query, execution);
}

} // namespace ninfer::ops
