// This family's indexed mixer: the same Q/K/V/gate projection the dense mixer runs plus the two
// indexer matrices, then prepare (the only writer of both caches), the indexer's block selection over
// four-token KV blocks, and the sparse GQA read that consumes it. Its diagnostics live here too -
// the two QSA gates compare bytes from two runs of the same prefix, so the only usable evidence is a
// file dump plus a stream hash. A Qwen3.5 artifact has no indexed layer and never enters this file.
#include "models/qwen3_5/execution/text.h"

#include "models/qwen3_5/execution/attention.h"
#include "models/qwen3_5/execution/diagnostics.h"
#include "models/qwen3_5/execution/workspace.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/qsa.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/w4a8.h"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::execution {
namespace {

void project(const Tensor& x, const LinearParameters& parameters, Tensor& out,
             WorkspaceArena& workspace, cudaStream_t stream) {
    ops::linear(x, parameters.weight, out, parameters.policy, workspace, stream);
}

// `columns` is the pass's column extent, or zero for a plane that keeps one file per layer.
// NINFER_QSA_SYNC: settle after each QSA Op so the first faulty submission names itself instead of
// surfacing at whatever kernel syncs next.
bool qsa_sync_probe() {
    static const bool enabled = std::getenv("NINFER_QSA_SYNC") != nullptr;
    return enabled;
}

void qsa_mark(cudaStream_t stream, const char* step, int layer) {
    if (!qsa_sync_probe()) { return; }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::fprintf(stderr, "[qsa] step ok layer=%d %s\n", layer, step);
    std::fflush(stderr);
}

// NINFER_QSA_META: read the pass's device metadata back once per layer call. This is the one place
// the wiring reads device state, and it exists only to attribute a fault to a bad selector.
void qsa_meta_probe(const char* stage, int layer, const Tensor& positions, const Tensor& columns,
                    const Tensor& base, const Tensor& offsets, const Tensor& source,
                    const Tensor& destination, std::int32_t batch, cudaStream_t stream) {
    if (std::getenv("NINFER_QSA_META") == nullptr) { return; }
    std::array<std::int32_t, 8> meta{};
    const auto read = [&](const Tensor& t, std::size_t slot) {
        if (t.data == nullptr || batch <= 0) { return; }
        CUDA_CHECK(cudaMemcpyAsync(&meta[slot], t.data, sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost, stream));
    };
    read(positions, 0);
    read(columns, 1);
    read(base, 2);
    read(offsets, 3);
    read(source, 4);
    read(destination, 5);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::fprintf(stderr,
                 "[qsa] meta %s layer=%d pos=%d cols=%d base=%d off=%d src=%d dst=%d\n", stage,
                 layer, meta[0], meta[1], meta[2], meta[3], meta[4], meta[5]);
    std::fflush(stderr);
}



// NINFER_QSA_DENSE_READ: keep the read side dense (S2's own stage) - prepare still owns the cache,
// and the attention reads it back through the read-only cached Op instead of the indexer. Only a
// single-row traversal has a read-only dense form, so the batch paths stay on the sparse arm.
bool qsa_dense_read() {
    static const bool dense = std::getenv("NINFER_QSA_DENSE_READ") != nullptr;
    return dense;
}
} // namespace

ops::QsaTailStateView TextContext::qsa_tail_view(int index, const Tensor& slots) const {
    if (batch_text_kv_ == nullptr || !batch_text_kv_->has_qsa()) {
        throw std::logic_error("QSA tail state needs an indexed text cache");
    }
    const qwen3_5::QsaStateSlot layer =
        batch_text_kv_->qsa_tail_layer(static_cast<std::uint32_t>(index));
    return {layer.raw_keys, layer.first_rope, slots};
}

void TextContext::qsa_mix(const BlockParameters& w, Tensor& x, int fidx, Phase, const Tensor* normalized) {
    const auto& p = std::get<AttentionParameters>(w.mixer);
    if (!p.indexer) { throw std::logic_error("indexed mixer has no indexer weights"); }
    if (batch_text_kv_ == nullptr || !batch_text_kv_->has_qsa()) {
        throw std::logic_error("indexed mixer has no QSA cache planes");
    }
    if (active_qsa_columns_ == nullptr || active_qsa_base_positions_ == nullptr ||
        active_qsa_offsets_ == nullptr) {
        throw std::logic_error("QSA pass metadata is not bound");
    }
    const auto& indexer = *p.indexer;
    cudaStream_t s      = ctx_.stream;
    const int T         = x.ne[1];
    // The same row/width/slot model every other mixer uses: a merged pass is one rectangle of
    // `width x batch` columns, a decode is that rectangle at width one, a single-lane chunk is
    // width T by one row.
    const std::int32_t batch = active_sequence_batch_ != 0 ? active_sequence_batch_ : 1;
    const std::int32_t width = active_sequence_batch_ != 0 ? active_sequence_width_ : T;
    if (width <= 0 || width * batch != T) {
        throw std::logic_error("QSA sequence batch binding does not match aggregate columns");
    }

    const auto head_dim  = dimension(config_.attention->head_dim);
    const auto query_hd  = dimension(config_.attention->num_attention_heads);
    const auto kv_heads  = dimension(config_.attention->num_key_value_heads);
    const auto query_w   = dimension(config_.attention->query_width());
    const auto key_w     = dimension(config_.attention->key_width());
    if (!config_.indexer) { throw std::logic_error("indexed mixer has no indexer config"); }
    const auto index_hd  = dimension(config_.indexer->head_dim);
    const DeviceExecutionView execution = ctx_.execution_view();

    // Positions, RoPE coordinates and the live-column count come from the pass, never from a host
    // read of device state: this is what keeps a captured traversal replayable at another length.
    const Tensor& cache_positions =
        active_cache_positions_ != nullptr ? *active_cache_positions_ : io_.pos;
    const Tensor& rope_source =
        active_rope_positions_ != nullptr ? *active_rope_positions_ : io_.rope_pos;
    Tensor positions = active_sequence_batch_ != 0 ? cache_positions : cache_positions.view({T, 1});
    Tensor rope;
    if (active_sequence_batch_ != 0) {
        rope = width == 1 ? rope_source.view({1, 1, batch})
                          : rope_source.view({width, 1, batch});
    } else {
        rope = rope_source.ne[1] == 3 ? rope_source.view({T, 3, 1})
                                      : rope_source.view({T, 1, 1});
    }
    const Tensor& valid_columns = *active_qsa_columns_;
    const Tensor& kv_table_rows =
        active_kv_table_rows_ != nullptr ? *active_kv_table_rows_ : io_.text_kv_table_row;
    const Tensor& source_slots =
        active_linear_state_source_slots_ != nullptr ? *active_linear_state_source_slots_
                                                     : *active_qsa_source_slots_;
    const Tensor& destination_slots =
        active_linear_state_destination_slots_ != nullptr ? *active_linear_state_destination_slots_
                                                          : *active_qsa_destination_slots_;
    if (valid_columns.ne[0] != batch || source_slots.ne[0] != batch ||
        destination_slots.ne[0] != batch || kv_table_rows.ne[0] < batch) {
        throw std::logic_error("QSA pass metadata does not match the bound batch");
    }

    const auto projection = workspace::text_attention_projection(work_, config_, T);
    auto qsa              = workspace::qsa_roots(work_, config_, T);
    Tensor h              = normalized != nullptr ? *normalized : projection.hidden;
    if (normalized == nullptr) ops::rmsnorm(x, w.input_norm, config_.rms_norm_eps, true, h, s);

    Tensor q_flat    = projection.query.view({query_w, T});
    Tensor gate_flat = projection.gate.view({query_w, T});
    Tensor k_flat    = projection.key.view({key_w, T});
    Tensor v_flat    = projection.value.view({key_w, T});
    attention_projection(h, p, q_flat, gate_flat, k_flat, v_flat, work_, s);
    // The artifact keeps the indexer's two matrices in their own parents, so each is one Linear
    // segment instead of a merged parent no Op overload could take.
    project(h, indexer.query, qsa.index_query, work_, s);
    project(h, indexer.key, qsa.index_key, work_, s);
    nan_probe_tensor("qsa-proj-q", q_flat, 100 + std::size_t(fidx), s);

    ops::QsaProjectionView projected{
        q_flat.view({head_dim, query_hd, T, batch}),
        k_flat.view({head_dim, kv_heads, T, batch}),
        v_flat.view({head_dim, kv_heads, T, batch}),
        qsa.index_query.view({index_hd, 4, T, batch}),
        qsa.index_key.view({index_hd, T, batch}),
    };
    const ops::QsaBatchView batch_view{positions, valid_columns, kv_table_rows};
    // Two norms of 256 for the mixer's heads and two of 128 for the indexer's, all unit-offset.
    const ops::QsaPrepareParameters prepare_parameters{
        p.query_norm, p.key_norm, indexer.query_norm, indexer.key_norm,
        config_.rms_norm_eps, config_.rope_parameters->rope_theta,
    };
    // One cache view for both planes: the index plane is addressed through the main cache's own
    // block table, which prepare itself requires to be the identical tensor.
    PagedKVBatchLayerView main_cache = batch_text_kv_->batch_layer_view(static_cast<std::uint32_t>(fidx));
    const ops::QsaIndexCacheView index_cache{
        batch_text_kv_->qsa_index_layer(static_cast<std::uint32_t>(fidx)),
        main_cache.block_tables,
    };
    const ops::QsaCacheView cache{main_cache, index_cache};
    const ops::QsaRawRecordWriteView records{
        qsa.record_keys.view({index_hd, T, batch}),
        qsa.record_rope.view({3, T, batch}),
        *active_qsa_offsets_,
    };
    const ops::QsaTailStateView before = qsa_tail_view(fidx, source_slots);

    // `query` and `index_query` exactly alias their projections, which prepare admits: it is the
    // normalization and rotation of that same storage, not a second buffer.
    Tensor prepared_query      = projected.query;
    Tensor prepared_index_query = projected.index_query;
    ops::qsa_prepare(projected, prepare_parameters, batch_view, rope, before, cache, records,
                     prepared_query, prepared_index_query, execution);
    qsa_meta_probe("after-prepare", fidx, positions, valid_columns, *active_qsa_base_positions_,
                   *active_qsa_offsets_, source_slots, destination_slots, batch, s);
    qsa_mark(s, "prepare", fidx);
    // Same names the dense arm writes: its post-norm/post-RoPE Q and its pre-append V.
    qsa_dump_tensor("NINFER_QSA_LAYER_DUMP", "query",
                    prepared_query.view({head_dim, query_hd, T}), fidx, T, s);
    qsa_dump_tensor("NINFER_QSA_LAYER_DUMP", "value_in",
                    projected.value.view({head_dim, kv_heads, T}), fidx, T, s);

    // The envelope is the resolved capacity, which is also what the startup plan priced: a host
    // launch promise, fixed across every replay, never a mask. It is taken from the cache the Ops
    // actually address: a decode pass binds its tables through `batch_text_kv_` and leaves `kv_`
    // empty, and a zero envelope would let the Ops size nothing and then score live blocks.
    const auto visible =
        std::min<std::uint32_t>(batch_text_kv_->max_context(), 262144U);
    if (visible == 0) { throw std::logic_error("QSA envelope needs a resolved cache capacity"); }
    Tensor attention   = qsa.attention.view({head_dim, query_hd, T, batch});
    // The read-only dense Op takes one execution row's own table, which only a single-lane pass
    // carries; a batch traversal binds its tables through `batch_text_kv_` instead, so it stays on
    // the sparse arm even under the switch.
    const bool dense_read = qsa_dense_read() && batch == 1 && kv_.valid();
    if (qsa_dense_read() && !dense_read) {
        static const bool told = [] {
            std::fprintf(stderr,
                         "[qsa] NINFER_QSA_DENSE_READ has no read-only row for this pass; the "
                         "layer runs the sparse arm\n");
            std::fflush(stderr);
            return true;
        }();
        (void)told;
    }
    if (dense_read) {
        // S2's own stage: prepare owns the cache, and the dense Op reads that cache back without
        // appending anything (`prepared_query` is already the normalized, rotated query).
        Tensor dense_out = attention.view({head_dim, query_hd, T});
        ops::causal_softmax_attention_cached(
            prepared_query.view({head_dim, query_hd, T}), positions.view({T}),
            {head_dim, query_hd, kv_heads},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim))),
            kv_.layer_view(static_cast<std::uint32_t>(fidx)),
            ops::CausalAttentionExecutionEnvelope{visible, visible}, work_, dense_out, execution);
        ops::sigmoid_mul(gate_flat.view({head_dim, query_hd, T}), dense_out, s);
    } else {
        const ops::QsaIndexerExecutionEnvelope indexer_envelope{visible};
        const ops::QsaSparseGqaExecutionEnvelope gqa_envelope{visible};
        ops::QsaBlockSelection selected{
            qsa.block_ids.view({512, T, batch}),
            qsa.block_counts.view({T, batch}),
        };
        ops::qsa_indexer(prepared_index_query, index_cache, batch_view, indexer_envelope, work_,
                         selected, execution);
        qsa_mark(s, "indexer", fidx);
        // The sparse Op applies the output gate itself, so its output is already the gated
        // attention result the dense arm computes with sigmoid_mul.
        ops::qsa_sparse_gqa(prepared_query, gate_flat.view({head_dim, query_hd, T, batch}),
                            main_cache, batch_view, selected, gqa_envelope, work_, attention,
                            execution);
        qsa_mark(s, "sparse_gqa", fidx);
    }

    ops::QsaRawRecordReadView read_records{
        records.raw_keys, records.rope_positions, *active_qsa_base_positions_, valid_columns};
    ops::qsa_tail_extract(before, read_records, valid_columns,
                          qsa_tail_view(fidx, destination_slots), execution);
    qsa_mark(s, "tail_extract", fidx);

    qsa_dump_tensor("NINFER_QSA_KV_DUMP", "main_key", main_cache.k_pages, fidx, 0, s);
    qsa_dump_tensor("NINFER_QSA_KV_DUMP", "main_value", main_cache.v_pages, fidx, 0, s);
    qsa_dump_tensor("NINFER_QSA_KV_DUMP", "main_key_scale", main_cache.k_scale_pages, fidx, 0, s);
    qsa_dump_tensor("NINFER_QSA_KV_DUMP", "main_value_scale", main_cache.v_scale_pages, fidx, 0, s);
    qsa_dump_tensor("NINFER_QSA_KV_DUMP", "index_key", index_cache.keys, fidx, 0, s);
    // The coverage check: after this call the written index blocks and the raw-key tail must tile
    // 0..p with no hole. The tail plane is dumped whole so the check can address the row's slot.
    qsa_dump_tensor("NINFER_QSA_KV_DUMP", "tail_keys",
                    qsa_tail_view(fidx, destination_slots).raw_keys, fidx, 0, s);
    qsa_dump_tensor("NINFER_QSA_LAYER_DUMP", "attention", attention, fidx, T, s);

    {
        Tensor attn_out = qsa.attention.view({query_w, T});
        nan_probe_tensor("qsa-attn", attn_out, 100 + std::size_t(fidx), s);
        if (normalized != nullptr) {
            // The four-stream correction owns the residual, so the projection is a pure write.
            ops::linear(attn_out, p.output.weight, x, p.output.policy, work_, s);
        } else if (ops::w4a8_supported(p.output.weight, T)) {
            ops::w4a8_linear_add(attn_out, p.output.weight, x, work_, s);
        } else {
            ops::linear_add(attn_out, p.output.weight, x, p.output.policy, work_, s);
        }
    }
}


} // namespace ninfer::models::qwen3_5::execution
