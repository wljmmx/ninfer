#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "models/qwen4_exp/ple/gpu.h"
#include "models/qwen4_exp/ple/prepare.h"
#include "models/qwen4_exp/ple/table.h"
#include "ninfer/ops/linear.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::qwen4_exp {

// The PLE Op contract fixes this geometry (include/ninfer/ops/ple.h): 2560 features per stream,
// four hyper-connection streams, nine convolution history columns and a four-tap kernel.
inline constexpr std::uint32_t kPleWidth      = 2560;
inline constexpr std::uint32_t kPleStreams    = 4;
inline constexpr std::uint32_t kPleChannels   = kPleWidth * kPleStreams;
inline constexpr std::uint32_t kPleHistory    = 9;
inline constexpr std::uint32_t kPleKernelTaps = 4;
// One projection, because the artifact keeps both row blocks in one [12800,2560] parent.
inline constexpr std::uint32_t kPleProjectionRows = kPleChannels + kPleWidth;

// The layer's numerical parameters, already in their execution form. `projection` carries the
// four streams' key rows first and the shared value rows after them, so `ple/key` and `ple/value`
// are two views of one Linear; the norms are the artifact's unit-offset gains (1 + stored value).
struct PleLayerWeights {
    Weight projection;
    ops::LinearPolicy policy = ops::LinearPolicy::A16Only;
    Tensor norm_key;         // BF16 [2560,4]
    Tensor norm_query;       // BF16 [2560,4]
    Tensor norm_conv;        // BF16 [2560,4]
    Tensor convolution;      // BF16 [10240,4], taps fastest
    float rms_epsilon = 1e-6F;
};

// One PLE layer of a Qwen4Exp block: the n-gram embedding arrives already gathered on the Device
// (PleGpuPreparation), and this owner turns it into the layer's residual contribution:
//
//   key/value = projection . E                        (one Linear, 12800 rows)
//   gated, records = ple_gate(key, value, residual)    (per-stream gate, normalized U recorded)
//   residual, normalized = ple_conv_add_norm(...)      (three-tap convolution folded in)
//
// The convolution reads the nine preceding normalized columns of a per-sequence slot pool, so a
// step is two calls: `apply` writes the outputs, `commit` advances each row's history by the
// columns the model accepted. Nothing here allocates per call, reads the Host, or touches CUDA
// graphs: PLE is host-driven (the table is on disk), so a captured graph cannot contain it.
//
// Shapes follow the Op contract exactly. `residual`, `gated` and `normalized` are BF16
// [H,C,T,B]; the projection parent is BF16 [H*C+H, T*B]. Batch rows share tokens but own a
// history slot each. This owner's buffers are sized once at construction from (max_tokens,
// max_batch, slots); a step that exceeds them is refused instead of truncated.
class PleLayerRuntime {
public:
    PleLayerRuntime(const PleLayerWeights&, std::uint32_t max_tokens, std::uint32_t max_batch,
                    std::uint32_t slots);
    ~PleLayerRuntime();
    PleLayerRuntime(const PleLayerRuntime&)            = delete;
    PleLayerRuntime& operator=(const PleLayerRuntime&) = delete;

    // residual := residual + gated + SiLU(conv(history, records)); normalized := Norm(residual).
    // `embedding` is BF16 [H,tokens,batch]; `valid_columns` and `slots` are I32 [batch]. The norm
    // weights belong to the block that consumes `normalized` (the next block's input norm, or the
    // final norm after the last block), so the caller passes them rather than this owner.
    void apply(const Tensor& embedding, Tensor& residual, Tensor& normalized,
               const Tensor& next_norm_weight, float next_epsilon, const Tensor& valid_columns,
               const Tensor& slots, CudaExecutionView execution);

    // Advances each row's nine-column history by its accepted prefix of this step's records.
    // `accepted_columns` is I32 [batch]; `valid_columns` bounds it.
    void commit(const Tensor& accepted_columns, const Tensor& valid_columns, const Tensor& slots,
                CudaExecutionView execution);

    // Intermediates, in the form the Op contract publishes them. The dump paths and the numeric
    // gates read these words instead of re-deriving them, which is the only way a comparison can
    // consume the declared BF16 seams.
    [[nodiscard]] const Tensor& projected() const noexcept { return projection_; }
    [[nodiscard]] const Tensor& gated() const noexcept { return gated_; }
    [[nodiscard]] const Tensor& records() const noexcept { return records_; }
    // The nine-column pool. Writable because a sequence start zeroes it and a checkpoint copies it
    // in or out; the model never scribbles on it between a step and its commit.
    [[nodiscard]] const Tensor& history() const noexcept { return history_; }
    [[nodiscard]] Tensor& history() noexcept { return history_; }

    [[nodiscard]] std::uint32_t max_tokens() const noexcept { return max_tokens_; }
    [[nodiscard]] std::uint32_t max_batch() const noexcept { return max_batch_; }
    [[nodiscard]] std::uint32_t slots() const noexcept { return slots_; }
    [[nodiscard]] std::size_t device_bytes() const noexcept;

private:
    void ensure_step(std::uint32_t tokens, std::uint32_t batch);
    void require_inputs(const Tensor& embedding, const Tensor& residual, const Tensor& normalized,
                        const Tensor& valid_columns, const Tensor& slots, std::uint32_t tokens,
                        std::uint32_t batch) const;

    PleLayerWeights weights_;
    std::uint32_t max_tokens_;
    std::uint32_t max_batch_;
    std::uint32_t slots_;
    // The shape the step scratch currently holds; zero until the first apply.
    std::uint32_t tokens_ = 0, batch_ = 0;
    std::size_t workspace_bytes_ = 0;
    // Persistent: the nine-column history pool and the records offsets. Never reallocated.
    std::unique_ptr<DeviceArena> state_;
    // Step-shaped scratch: reallocated when a step asks for a wider shape.
    std::unique_ptr<DeviceArena> step_;
    std::unique_ptr<DeviceArena> workspace_;
    Tensor projection_; // BF16 [H*C+H, tokens*batch]
    Tensor gated_;      // BF16 [H,C,tokens,batch]
    Tensor records_;    // BF16 [H*C,tokens,batch]
    Tensor history_;    // BF16 [H*C,9,slots]
    Tensor offsets_;    // I32 [max_batch], the records column every row starts at (always 0)
};


// What the artifact declares for one PLE layer's n-gram table. The reader (`PleTableSource`)
// decodes exactly one encoding family, so the load path states the declared facts here and this
// answers with the encoding's own name when it cannot be served. Nothing in this tree silently
// skips a PLE layer: an unsupported table is a construction failure with the shapes in it.
struct PleTableRequirement {
    QType format        = QType::BF16;
    QuantLayout layout  = QuantLayout::Contiguous;
    std::uint64_t rows  = 0;
    std::uint64_t width = 0;
    std::string parameter; // the logical binding name, so a refusal names what to fix
};

// True when `PleRowSource`/`decode_nf4_row` can serve this table: one NF4 codebook against an
// E4M3 scale byte per 16-wide group, rows of 160 words in 4096-byte pages. Logs the declared
// encoding (and its row count) either way, then throws
// "<parameter>: 表编码 <format>/<layout>（<rows> 行 × <width>）不受支持，可选：转码 / 加解码器"
// when it cannot. `stderr` rather than an exception string alone, because a refusal at load time
// is the only chance to say which encoding the artifact actually carries.
void require_supported_ple_table(const PleTableRequirement& requirement);

// The encoding `PleTableSource` serves, written once so the loader check and the reader cannot
// drift apart.
[[nodiscard]] constexpr bool supported_ple_table(QType format, QuantLayout layout,
                                                 std::uint64_t width) noexcept {
    return format == QType::NF4_G16_E4M3FN && layout == QuantLayout::RowPagedG16_4K &&
           width == kPleRowElements;
}

// One column window of a text pass, as the PLE stage consumes it. `ids` is token-fastest
// ([tokens*batch]) and `slots` names the history slot of every row: a row's slot must identify
// its sequence, because the nine-column convolution history and the two-token hash tail live
// there. The stage remembers each slot's tail from the window it last saw, so a caller only
// states where a sequence starts.
struct PleStepWindow {
    std::span<const std::int32_t> ids;
    std::span<const std::int32_t> slots;
    std::span<const std::uint32_t> valid_columns;
    std::int32_t tokens = 0;
};

// One text model's PLE: the n-gram table and its Host/Device preparation, plus one
// PleLayerRuntime per text block that carries PLE. This is the model-side owner of everything
// between "the pass knows its token ids" and "the block's four-stream residual is rewritten":
//
//   begin  -> hash, read, upload one window (the table is on disk, so the Host waits here)
//   apply  -> per PLE block, residual += contribution and normalized := Norm(residual)
//   commit -> per PLE block, advance each row's nine-column history by its accepted prefix
//   end    -> retire the window once the consumer stream has passed its last reader
//
// Every buffer is sized at construction; a step wider than the capacity is refused, not
// truncated. There is no per-call allocation and no Host read of Device state.
class PleTextStage {
public:
    // `layers` is indexed by text block and holds an empty entry where that block has no PLE.
    // `slots` is the history pool depth, i.e. the number of sequences that may be resident.
    PleTextStage(const PleTableSource& table, std::int32_t eos, const DeviceContext& context,
                 std::vector<std::optional<PleLayerWeights>> layers, std::uint32_t max_tokens,
                 std::uint32_t max_batch, std::uint32_t slots, PlePrepareOptions options);
    ~PleTextStage();
    PleTextStage(const PleTextStage&)            = delete;
    PleTextStage& operator=(const PleTextStage&) = delete;

    // Opens one window. A row's `valid_columns[b]` in [0,tokens] bounds its live columns; every
    // row of a window is prepared even when it rejects all of them (`PlePrepareService` requires
    // at least one live column per sealed window, so an all-zero window is refused here).
    void begin(const PleStepWindow& window, CudaExecutionView execution);
    // residual and normalized are BF16 [H,C,T,B] with this window's T and B; `next_norm` is the
    // norm of the block that consumes `normalized`.
    void apply(std::uint32_t layer, Tensor& residual, Tensor& normalized, const Tensor& next_norm,
               float epsilon, CudaExecutionView execution);
    // Advances each row's history by the window's accepted columns. The two text entries this
    // tree drives (prefill and ordinary decode) accept every live column, so the window's own
    // valid columns are the accepted prefix; a speculative step whose acceptance is only known
    // after the verifier runs must pass them explicitly.
    void commit(std::uint32_t layer, CudaExecutionView execution);
    void commit(std::uint32_t layer, const Tensor& accepted, CudaExecutionView execution);
    // Retires the open window. Safe to call with no window open.
    void end(CudaExecutionView execution);

    [[nodiscard]] bool open() const noexcept { return ticket_.has_value(); }

    [[nodiscard]] bool has_layer(std::uint32_t layer) const noexcept {
        return layer < layers_.size() && layers_[layer] != nullptr;
    }

    // The open window's Device E, [H,T,B]; empty until begin.
    [[nodiscard]] const Tensor& embedding() const noexcept { return embedding_; }

    // One PLE block's last gated value, recorded normalized columns and nine-column pool. The
    // numeric gates and the dump paths read these words instead of re-deriving them.
    [[nodiscard]] const Tensor& gated(std::uint32_t layer) const;
    [[nodiscard]] const Tensor& records(std::uint32_t layer) const;
    [[nodiscard]] const Tensor& history(std::uint32_t layer) const;

    [[nodiscard]] std::uint32_t max_tokens() const noexcept { return max_tokens_; }

    [[nodiscard]] std::uint32_t max_batch() const noexcept { return max_batch_; }

    [[nodiscard]] std::size_t device_bytes() const noexcept;

private:
    void retire();
    void require_window(const PleStepWindow& window) const;
    // The window metadata as the Ops' I32 [B] views over the stage's own storage.
    [[nodiscard]] Tensor live_columns() const;
    [[nodiscard]] Tensor slot_rows() const;
    [[nodiscard]] Tensor accepted_columns() const;

    std::int32_t eos_;
    std::uint32_t max_tokens_;
    std::uint32_t max_batch_;
    std::uint32_t slots_;
    std::vector<std::unique_ptr<PleLayerRuntime>> layers_;
    std::unique_ptr<PleGpuPreparation> gpu_;
    // The window that is open right now, and the one its consumer has released but whose slot the
    // worker may not have reclaimed yet.
    std::optional<PleTicket> ticket_;
    std::optional<PleTicket> retired_;
    Tensor embedding_;         // the open window's Device E, [H,T,B]
    DeviceBuffer live_mask_;   // I32 [max_batch], the open window's live columns
    DeviceBuffer slot_index_;  // I32 [max_batch], the open window's history slots
    DeviceBuffer accept_mask_; // I32 [max_batch], the commit prefix (the valid columns by default)
    Device device_;
    // Per-slot hash context: the two tokens preceding each slot's next window, and whether the
    // slot has seen a window at all (a fresh sequence hashes with EOS history).
    std::vector<std::array<std::int32_t, 2>> tail_;
    std::vector<bool> started_;
    std::int32_t tokens_ = 0, batch_ = 0;
};


} // namespace ninfer::models::qwen4_exp