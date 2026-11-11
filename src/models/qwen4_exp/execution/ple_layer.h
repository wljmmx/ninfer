#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include <cstddef>
#include <cstdint>
#include <memory>

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

} // namespace ninfer::models::qwen4_exp
