#pragma once
#include "models/qwen3_5/program/internal.h"


#include "core/arena.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/qsa.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/sparse_moe.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/program/round_buffers.h"
#include "models/qwen3_5/execution/workspace.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::models::qwen4_exp {
// The model-side PLE owner (table window, gate, convolution, history). Declared here so this
// header stays free of the disk layer it borrows from.
class PleTextStage;
} // namespace ninfer::models::qwen4_exp

namespace ninfer::models::qwen3_5::execution {

using Phase = qwen3_5::TextPhase;

enum class GdnStateAction : std::uint8_t {
    UpdateInPlace,
    RecordForReplay,
};

struct NullTap {
    static constexpr bool enabled = false;
};

struct PrefillChunkResult {
    std::uint32_t processed_tokens = 0;
    bool finalized                 = false;
    runtime::ExecutionTiming timing;
};

struct DFlashFeatureSink {
    static constexpr bool enabled = true;
    using PrefillConsumer         = std::function<void(const Tensor&, const Tensor&, bool)>;

    Tensor* features                  = nullptr;
    Tensor* positions                 = nullptr;
    Tensor* batch_features            = nullptr;
    const Tensor* batch_lanes         = nullptr;
    const Tensor* batch_valid_columns = nullptr;
    std::int32_t batch_width          = 0;
    std::int32_t batch_size           = 0;
    std::span<const std::uint32_t> layers;
    PrefillConsumer consume_prefill;
    std::uint32_t captured_mask = 0;
    std::int32_t active_tokens  = 0;

    void begin(const Tensor& value);
    void capture_layer(int layer, const Tensor& value, cudaStream_t stream);
    void capture_positions(const Tensor& source, cudaStream_t stream);
    void consume_prefill_chunk(std::int32_t tokens, bool rewrite_checkpoint);
};

class VisionPrefillSession;

class TextContext {
public:
    TextContext(DeviceContext& ctx, const execution::Parameters& weights, WorkspaceArena& work,
                qwen3_5::PagedKVCacheView kv, LinearAttentionStatePool& state,
                qwen3_5::RoundState& io, Tensor& prefill_hidden, std::uint32_t prefill_chunk,
                std::uint32_t text_kv_base,
                qwen3_5::PagedKVCacheView mtp_kv           = qwen3_5::PagedKVCacheView(),
                const qwen3_5::PagedKVCache* batch_text_kv = nullptr,
                const qwen3_5::PagedKVCache* batch_mtp_kv  = nullptr);
    ~TextContext();

    TextContext(const TextContext&)            = delete;
    TextContext& operator=(const TextContext&) = delete;

    void set_proposal_head(const LinearParameters* weight, const std::int32_t* ids,
                           int count) noexcept {
        proposal_head_     = weight;
        proposal_head_ids_ = ids;
        proposal_head_n_   = count;
    }

    void set_rope_delta(std::int32_t delta) noexcept { rope_delta_ = delta; }

    void set_sampling(const ops::SamplingConfig* config) noexcept { sampling_config_ = config; }

    void set_prefill_gpu_timer(CudaEventTimer* timer) noexcept { prefill_gpu_timer_ = timer; }

    void set_prefill_split_frontier(std::int64_t position) noexcept {
        prefill_split_frontier_ = position;
    }

    void set_rewrite_checkpoint_hidden_output(Tensor* output) noexcept {
        rewrite_checkpoint_hidden_output_ = output;
    }

    void set_mtp_proposal_extent(std::uint32_t extent) noexcept { mtp_proposal_extent_ = extent; }

    void set_linear_state_slots(std::int32_t source_slot, std::int32_t destination_slot);
    void set_gdn_state_action(GdnStateAction action, const GdnReplayRecords* replay_records);

    [[nodiscard]] const LinearParameters* proposal_head() const noexcept { return proposal_head_; }

    [[nodiscard]] const std::int32_t* proposal_head_ids() const noexcept {
        return proposal_head_ids_;
    }

    [[nodiscard]] int proposal_head_n() const noexcept { return proposal_head_n_; }

    [[nodiscard]] PrefillChunkResult prefill_chunk(std::span<const int> full_ids,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   bool finalize_at_end);
    [[nodiscard]] PrefillChunkResult prefill_chunk(std::span<const int> full_ids,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   bool finalize_at_end, DFlashFeatureSink& sink);
    [[nodiscard]] PrefillChunkResult
    prefill_chunk(const qwen3_5::PreparedPromptData& input, std::uint32_t begin,
                  std::uint32_t nominal_length, VisionPrefillSession& vision, bool finalize_at_end);
    [[nodiscard]] PrefillChunkResult prefill_chunk(const qwen3_5::PreparedPromptData& input,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   VisionPrefillSession& vision,
                                                   bool finalize_at_end, DFlashFeatureSink& sink);
    void ordinary_decode_batch(const Tensor& ids, const Tensor& cache_positions,
                               const Tensor& rope_positions, const Tensor& kv_table_rows,
                               const Tensor& linear_state_source_slots,
                               const Tensor& linear_state_destination_slots,
                               ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                               Tensor& logits);
    void target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                             const Tensor& rope_positions, const Tensor& valid_columns,
                             const Tensor& kv_table_rows, const Tensor& linear_state_source_slots,
                             ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                             Tensor& logits, Tensor& target_tokens);
    void target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                             const Tensor& rope_positions, const Tensor& valid_columns,
                             const Tensor& kv_table_rows, const Tensor& linear_state_source_slots,
                             ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                             Tensor& logits, Tensor& target_tokens, DFlashFeatureSink& sink);
    // Q67/S1: one merged prefill pass. Same per-row bindings the verify path uses - per-row KV table
    // rows, per-row positions, per-row valid columns, per-row state slots - which is what keeps the
    // rows block-diagonal without a mask: row i only ever reads KV row i. Host spans are copied in
    // here so the caller only builds plain arrays; `logits` is [vocab, batch] and `tokens` is
    // [batch], one column per row, the row's last valid token, because prefill only needs that one.
    // The logits tensor this needs is allocated inside, after the workspace reset, and only the
    // resulting tokens come back: a caller-owned [vocab, batch] tensor would be freed by the reset.
    void prefill_batch(std::span<const std::int32_t> ids, std::span<const std::int32_t> positions,
                       std::span<const std::int32_t> rope_positions,
                       std::span<const std::int32_t> valid_columns,
                       std::span<const std::int32_t> kv_table_rows,
                       std::span<const std::int32_t> state_source_slots,
                       std::span<const std::int32_t> state_destination_slots, std::int32_t width,
                       std::int32_t batch, ops::CausalAttentionExecutionEnvelope envelope,
                       std::span<std::int32_t> next_tokens,
                       std::span<Tensor* const> tail_hidden_rows);
    void mtp_forward_decode_batch(const Tensor& ids, const Tensor& hidden,
                                  const Tensor& cache_positions, const Tensor& rope_positions,
                                  const Tensor& valid_columns, const Tensor& kv_table_rows,
                                  ops::CausalAttentionExecutionEnvelope envelope,
                                  Tensor& mtp_hidden);
    void mtp_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens);
    // `mask` carries one constraint bitmap per proposal column. When set, the top-k candidates are
    // taken from a masked copy of the logits so a constrained request never proposes a token the
    // verifier is bound to reject.
    void mtp_propose_topk(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens,
                          Tensor* candidate_ids, Tensor* proposal_q, std::int32_t top_k,
                          ops::SamplingMask mask = {});
    // `mask_column` selects the draft-position column of a per-position constraint bitmap, and
    // `mask_window_words` is the distance between two batch rows' windows. The bitmap is laid out
    // per (row, position), so a single draft row is addressed by stepping one window; without it a
    // multi-row batch would mask row b with row b-1's later column.
    void mtp_propose_topk_strided(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens,
                                  Tensor& candidate_ids, Tensor& proposal_q, std::int32_t top_k,
                                  std::int32_t column_stride, ops::SamplingMask mask = {},
                                  std::int32_t mask_column = 0,
                                  std::int32_t mask_window_words = 0);
    void mtp_forward_batch(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                           ops::CausalAttentionExecutionEnvelope envelope, Tensor& mtp_hidden,
                           int logits_column, Tensor* logits, Tensor* draft_token,
                           const Tensor* explicit_rope_positions = nullptr,
                           const Tensor* input_embeddings        = nullptr);
    void mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                             const Tensor& position, ops::CausalAttentionExecutionEnvelope envelope,
                             Tensor& mtp_hidden, Tensor& logits, Tensor& draft_token,
                             ops::SamplingMask mask = {});
private:
    [[nodiscard]] bool mtp_enabled() const noexcept {
        return mtp_kv_.valid() || batch_mtp_kv_ != nullptr;
    }

    // A mixer reads its activation from `normalized` when the block's four-stream correction has
    // already produced it, and leaves the residual to that correction. Otherwise it normalizes the
    // residual in `x` itself and adds its output back into it.
    void attn_mix(const BlockParameters& weights, Tensor& x, int index, Phase phase,
                  const Tensor* normalized = nullptr);
    void gdn_mix(const BlockParameters& weights, Tensor& x, int index, Phase phase,
                 const Tensor* normalized = nullptr);
    // Qwen4Exp's indexed mixer: the same Q/K/V/gate projection the dense mixer runs plus the two
    // indexer matrices, then prepare (the only writer of both caches), the indexer's block
    // selection, and the sparse GQA read.
    void qsa_mix(const BlockParameters& weights, Tensor& x, int index, Phase phase,
                 const Tensor* normalized = nullptr);
    [[nodiscard]] ops::QsaTailStateView qsa_tail_view(int index, const Tensor& slots) const;
    void mlp_tail(const BlockParameters& weights, Tensor& x, Phase phase,
                  const ops::SparseMoeHints& hints, int layer,
                  const Tensor* normalized = nullptr);
    [[nodiscard]] ops::SparseMoeHints next_projection_hints(int layer) const;

    // Qwen4Exp's four-stream residual: the embedding is expanded once, every block folds its
    // normalized view to a single stream, and the correction splits the product back out.
    [[nodiscard]] bool hc_enabled() const noexcept { return final_hc_ != nullptr; }
    void hc_stem(Tensor& embedding, std::int32_t tokens);
    void hc_mix_into(const HcParameters& weights, std::int32_t tokens);
    void hc_combine(const Tensor& product, const Tensor& norm_weight, std::int32_t tokens);
    // The last hidden the head reads: the final correction's fold, or the plain final norm.
    void final_hidden(Tensor& x, Tensor& out, std::int32_t tokens);
    void run_layers(Tensor& x, Phase phase);
    template <class Tap>
    void run_layers(Tensor& x, Phase phase, Tap& tap);
    template <class Tap>
    void target_verify_batch_impl(const Tensor& ids, const Tensor& cache_positions,
                                  const Tensor& rope_positions, const Tensor& valid_columns,
                                  const Tensor& kv_table_rows,
                                  const Tensor& linear_state_source_slots,
                                  ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                                  Tensor& logits, Tensor& target_tokens, Tap& tap);

    void mtp_forward_stem(const Tensor& ids, const Tensor& hidden, const Tensor* input_embeddings,
                          Tensor& x, Tensor& ah);    void mtp_forward_tail(Tensor& x, const Tensor& ah, const Tensor& positions,
                          const Tensor& rope_positions,
                          ops::CausalAttentionExecutionEnvelope envelope, Tensor& mtp_hidden);
    void mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                          const Tensor& rope_positions,
                          ops::CausalAttentionExecutionEnvelope envelope, Tensor& mtp_hidden,
                          const Tensor* input_embeddings);
    void mtp_prefill_chunk(const Tensor& ids, const Tensor& hidden, const Tensor* input_embeddings,
                           const Tensor& positions, const Tensor& rope_positions,
                           ops::CausalAttentionExecutionEnvelope envelope, bool final_chunk,
                           Tensor* final_hidden, Tensor* logits, Tensor* draft_token);
    // `mask` carries one constraint bitmap per proposal position (stride apart). When it is null the
    // proposal is a plain argmax; when it is set the proposal is drawn greedily through the sampling
    // op, which excludes forbidden vocabulary rows, so a constrained request does not propose
    // tokens the verifier must always reject.
    void proposal_argmax(const Tensor& hidden, Tensor& logits, Tensor& proposal_tokens,
                         ops::SamplingMask mask = {});

    struct MultimodalPrefill {
        std::span<const int> token_ids;
        std::span<const std::int32_t> positions;
        VisionPrefillSession* vision = nullptr;
        std::uint32_t begin          = 0;
        std::int32_t rope_delta      = 0;
    };

    struct TextPrefill {
        std::span<const int> token_ids;
        std::uint32_t begin = 0;
    };

    template <class Tap>
    [[nodiscard]] PrefillChunkResult
    prefill_impl(std::span<const int> ids, const TextPrefill* text_prefill,
                 const MultimodalPrefill* multimodal, Tap& tap, bool finalize_at_end);
    DeviceContext& ctx_;
    const Parameters& parameters_;
    const TextConfig& config_;
    WorkspaceArena& work_;
    qwen3_5::PagedKVCacheView kv_;
    qwen3_5::PagedKVCacheView mtp_kv_;
    const qwen3_5::PagedKVCache* batch_text_kv_ = nullptr;
    const qwen3_5::PagedKVCache* batch_mtp_kv_  = nullptr;
    LinearAttentionStatePool& state_;
    qwen3_5::RoundState& io_;
    Tensor& prefill_hidden_;
    // Borrowed from Program; every measured chunk retires before this pair is reused.
    CudaEventTimer* prefill_gpu_timer_ = nullptr;
    std::uint32_t prefill_chunk_;
    std::uint32_t text_kv_base_;
    const Tensor* active_cache_positions_                                          = nullptr;
    const Tensor* active_rope_positions_                                           = nullptr;
    const Tensor* active_kv_table_rows_                                            = nullptr;
    const Tensor* active_linear_state_source_slots_                                = nullptr;
    const Tensor* active_linear_state_destination_slots_                           = nullptr;
    const Tensor* active_valid_columns_                                            = nullptr;
    // QSA per-pass metadata: the row's live columns, the record base (`positions[0,b]`) and the
    // record offset this call writes from. Bound by every pass entry point beside the GDN slots,
    // because a QSA layer takes them from the pass rather than reading device state on the host.
    const Tensor* active_qsa_columns_                                              = nullptr;
    const Tensor* active_qsa_base_positions_                                       = nullptr;
    const Tensor* active_qsa_offsets_                                              = nullptr;
    // A single-lane pass publishes its two host slot integers here, so every pass hands the QSA
    // tail state its slots the same way it hands the GDN state its own.
    const Tensor* active_qsa_source_slots_                                         = nullptr;
    const Tensor* active_qsa_destination_slots_                                    = nullptr;
    const Tensor* active_backend_kv_table_rows_                                    = nullptr;
    const ops::CausalAttentionExecutionEnvelope* active_causal_attention_envelope_ = nullptr;
    std::int32_t active_sequence_batch_                                            = 0;
    std::int32_t active_sequence_width_                                            = 0;
    std::int32_t rope_delta_                                                       = 0;
    std::int32_t linear_state_source_slot_                                         = 0;
    std::int32_t linear_state_destination_slot_                                     = 0;
    // Host mirrors of the batch bindings, for the merged prefill pass's per-row host loop: the GDN
    // conv split slices each row's columns and state slots on the host (the pass is never graph
    // captured), which the device selector tensors alone cannot express.
    std::span<const std::int32_t> active_linear_state_source_slots_host_{};
    std::span<const std::int32_t> active_linear_state_destination_slots_host_{};
    std::span<const std::int32_t> active_valid_columns_host_{};
    GdnStateAction gdn_state_action_          = GdnStateAction::UpdateInPlace;
    const GdnReplayRecords* replay_records_   = nullptr;
    std::int64_t prefill_split_frontier_      = -1;
    Tensor* rewrite_checkpoint_hidden_output_ = nullptr;
    std::uint32_t mtp_proposal_extent_        = 0;

    const Weight* embed_                        = nullptr;
    const Tensor* final_norm_                   = nullptr;
    const HcParameters* final_hc_               = nullptr;
    // Qwen4Exp's n-gram embedding, when the model carries it. The stage owns the table window and
    // one runtime per PLE block; `ple_window_`/`ple_tokens_` are the pass's open window, which is
    // what a PLE block reads its embedding from. A PLE block that runs with no window is refused
    // in run_layers instead of quietly skipping the layer.
    std::unique_ptr<qwen4_exp::PleTextStage> ple_;
    std::int32_t ple_tokens_ = 0;
    bool ple_window_         = false;
    // The live four-stream state of the current call. Allocated by run_layers from the same recipe
    // the workspace plan uses, so the two reserve identical bytes.
    workspace::TextHcRoots hc_{};
    const LinearParameters* lm_head_            = nullptr;
    const LinearParameters* proposal_head_      = nullptr;
    const std::int32_t* proposal_head_ids_      = nullptr;
    int proposal_head_n_                        = 0;
    const ops::SamplingConfig* sampling_config_ = nullptr;
    const MtpParameters* mtp_                   = nullptr;
};

} // namespace ninfer::models::qwen3_5::execution


