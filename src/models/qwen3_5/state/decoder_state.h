#pragma once

#include "core/layout.h"
#include "core/paged_kv_cache.h"
#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace ninfer::models::qwen3_5 {

inline constexpr std::int32_t kKvInt8QuantGroup = 64;
inline constexpr std::int32_t kKvFp8QuantGroup  = 256;

// Qwen4Exp's sparse index plane and raw-index-key tail state. Both are addressed in the main
// cache's page space: one index block holds four tokens, sixteen blocks fill one 64-token page, so
// a logical page id names the same physical row in `block_table` for the main cache and for every
// index plane. There is deliberately no second id space and no table of its own.
struct QsaCacheSpec {
    std::uint32_t indexed_layers       = 0;  // the layers that switch to QSA
    std::int32_t index_head_dim        = 0;  // 128
    std::int32_t index_blocks_per_page = 0;  // 16 = page_tokens / 4
    std::int32_t tail_columns          = 0;  // 3, the unfinished block's visible prefix
    std::int32_t state_slots           = 0;  // pool slots, the Linear Attention slot space
};

struct QsaCacheLayout {
    QsaCacheSpec spec;
    std::vector<TensorRegion> index;      // BF16 [128,16,1,physical_pages], one per indexed layer
    std::vector<TensorRegion> tail_keys;  // BF16 [128,3,state_slots], one per indexed layer
    std::vector<TensorRegion> tail_rope;  // I32 [3,state_slots], one per indexed layer
    // One live column per decode row: a width-1 traversal has exactly one valid column, and the
    // QSA Ops take that count as metadata. Filled with ones once, so no per-call host upload can
    // land inside a captured graph.
    TensorRegion decode_columns;  // I32 [kMaximumConcurrency]

    [[nodiscard]] std::size_t payload_bytes() const noexcept;
};

struct DecoderStateSpec {
    std::uint32_t full_attention_layers     = 0;
    std::uint32_t mtp_layers                = 0;
    std::uint32_t capacity                  = 0;
    std::int32_t kv_heads                   = 0;
    std::int32_t attention_head_dim         = 0;
    KvCacheStorage kv_storage               = KvCacheStorage::BFloat16;
    bool enable_mtp                         = false;
    std::int32_t kv_table_rows              = 1;
    std::uint32_t text_physical_page_groups = 0;
    std::uint32_t mtp_physical_page_groups  = 0;
    // Absent on every model line that has no indexed mixer, and on the MTP cache.
    std::optional<QsaCacheSpec> qsa;
};

struct PagedKVCacheLayout {
    DeviceKVPagePoolLayout pages;
    KVExecutionTableLayout execution_tables;
    std::uint32_t layers      = 0;
    std::uint32_t max_context = 0;
    std::int32_t kv_heads     = 0;
    PagedKVStorageLayout layer_storage;
    std::optional<QsaCacheLayout> qsa;

    [[nodiscard]] std::size_t payload_bytes() const noexcept {
        return pages.payload_bytes() + (qsa ? qsa->payload_bytes() : 0);
    }
};

// The three planes one indexed layer owns: the compressed index keys, and the tail state at one
// slot. `QsaStateSlot` is a pair of tensors because `QsaTailStateView` takes them apart.
struct QsaStateSlot {
    Tensor raw_keys;   // BF16 [128,3,slots]
    Tensor first_rope; // I32 [3,slots]
};

class PagedKVCache;

class PagedKVCacheView {
public:
    PagedKVCacheView() noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return cache_ != nullptr; }

    [[nodiscard]] std::uint32_t max_context() const noexcept;
    [[nodiscard]] PagedKVLayerView layer_view(std::uint32_t layer) const;

private:
    friend class PagedKVCache;
    PagedKVCacheView(const PagedKVCache& cache, Tensor block_table, Tensor page_score,
                     Tensor page_score_peak, Tensor page_score_recent) noexcept;

    const PagedKVCache* cache_ = nullptr;
    Tensor block_table_;
    Tensor page_score_;
    Tensor page_score_peak_;
    Tensor page_score_recent_;
};

class PagedKVCache {
public:
    PagedKVCache(DeviceSpan backing, const PagedKVCacheLayout& layout);

    PagedKVCache(const PagedKVCache&)            = delete;
    PagedKVCache& operator=(const PagedKVCache&) = delete;
    PagedKVCache(PagedKVCache&&)                 = delete;
    PagedKVCache& operator=(PagedKVCache&&)      = delete;

    [[nodiscard]] std::uint32_t max_context() const noexcept { return max_context_; }

    [[nodiscard]] std::uint32_t layers() const noexcept { return layers_; }

    [[nodiscard]] KvCacheStorage storage() const noexcept { return layer_storage_.storage; }

    [[nodiscard]] DeviceKVPagePool& page_pool() noexcept { return pages_; }

    [[nodiscard]] const DeviceKVPagePool& page_pool() const noexcept { return pages_; }

    [[nodiscard]] KVExecutionTablePool& execution_tables() noexcept { return execution_tables_; }

    [[nodiscard]] const KVExecutionTablePool& execution_tables() const noexcept {
        return execution_tables_;
    }

    [[nodiscard]] PagedKVCacheView execution_view(const KVExecutionRowLease& row) const;

    [[nodiscard]] PagedKVBatchLayerView batch_layer_view(std::uint32_t layer) const;

    // QSA: absent unless the artifact declares an indexer. `index_layer` is the layer's ordinal
    // among the full-attention layers, which is the same compact index `batch_layer_view` takes.
    [[nodiscard]] bool has_qsa() const noexcept { return qsa_.has_value(); }
    [[nodiscard]] Tensor qsa_index_layer(std::uint32_t layer) const;
    [[nodiscard]] QsaStateSlot qsa_tail_layer(std::uint32_t layer) const;
    // I32 [batch] of ones, the width-1 traversal's live-column count.
    [[nodiscard]] Tensor qsa_decode_columns(std::int32_t batch) const;
    // The physical page count the index plane and the main cache share.
    [[nodiscard]] std::uint32_t qsa_pages() const noexcept;

private:
    friend class PagedKVCacheView;
    [[nodiscard]] PagedKVLayerView layer_view(std::uint32_t layer, Tensor block_table,
                                              Tensor page_score = Tensor{},
                                              Tensor page_score_peak = Tensor{},
                                              Tensor page_score_recent = Tensor{}) const;

    DeviceKVPagePool pages_;
    KVExecutionTablePool execution_tables_;
    std::uint32_t layers_      = 0;
    std::uint32_t max_context_ = 0;
    std::int32_t kv_heads_     = 0;
    PagedKVStorageLayout layer_storage_;
    std::optional<QsaCacheLayout> qsa_;
    std::vector<Tensor> qsa_index_;      // bound per indexed layer
    std::vector<Tensor> qsa_tail_keys_;  // bound per indexed layer
    std::vector<Tensor> qsa_tail_rope_;  // bound per indexed layer
    Tensor qsa_decode_columns_;
};

struct DecoderStateLayout {
    PagedKVCacheLayout text_kv;
    std::optional<PagedKVCacheLayout> mtp_kv;

    [[nodiscard]] std::size_t kv_payload_bytes() const noexcept;
};

[[nodiscard]] DecoderStateLayout plan_decoder_state(LayoutBuilder& builder,
                                                    const DecoderStateSpec& spec);

struct DecoderState {
    PagedKVCache text_kv;
    std::optional<PagedKVCache> mtp_kv;

    DecoderState(DeviceSpan backing, const DecoderStateLayout& layout);

    [[nodiscard]] PagedKVCache* mtp_cache() noexcept;
    [[nodiscard]] const PagedKVCache* mtp_cache() const noexcept;
};

} // namespace ninfer::models::qwen3_5
