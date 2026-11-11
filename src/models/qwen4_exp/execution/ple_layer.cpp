#include "models/qwen4_exp/execution/ple_layer.h"

#include "ninfer/ops/ple.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen4_exp {
namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(std::string("PLE layer: ") + message);
}

// Shape/type check for one BF16 operand the Op contract names exactly.
void bf16_tensor(const Tensor& t, std::initializer_list<std::int32_t> shape, const char* label) {
    const Tensor expected = Tensor::metadata(DType::BF16, shape);
    require(t.data != nullptr && t.dtype == DType::BF16 && t.is_contiguous(), label);
    for (int d = 0; d < 4; ++d) require(t.ne[d] == expected.ne[d], label);
}

void i32_batch(const Tensor& t, const char* label) {
    require(t.data != nullptr && t.dtype == DType::I32 && t.is_contiguous() && t.ne[1] == 1 &&
                t.ne[2] == 1 && t.ne[3] == 1 && t.ne[0] > 0,
            label);
}

// Bytes the step-shaped scratch needs for one (tokens, batch) step.
std::size_t step_bytes(std::uint32_t tokens, std::uint32_t batch) {
    const std::size_t columns = std::size_t(tokens) * batch;
    return (std::size_t(kPleProjectionRows) + 2 * std::size_t(kPleChannels)) * columns *
           sizeof(std::uint16_t);
}

} // namespace

PleLayerRuntime::PleLayerRuntime(const PleLayerWeights& weights, std::uint32_t max_tokens,
                                 std::uint32_t max_batch, std::uint32_t slots)
    : weights_(weights),
      max_tokens_(max_tokens),
      max_batch_(max_batch),
      slots_(slots),
      tokens_(0),
      batch_(0) {
    require(max_tokens >= 1 && max_batch >= 1 && max_batch <= 8 && slots >= 1,
            "capacities must be positive (batch at most 8)");
    // The Op contract is fixed-2560/4; a config that says otherwise is refused here rather than
    // multiplying by a matrix the Ops will reject three calls later.
    require(weights_.projection.n == std::int32_t(kPleProjectionRows) &&
                weights_.projection.k == std::int32_t(kPleWidth),
            "the projection must be the artifact's [H*C+H, H] parent");
    bf16_tensor(weights_.norm_key, {std::int32_t(kPleWidth), std::int32_t(kPleStreams)},
                "norm_key must be BF16 [2560,4]");
    bf16_tensor(weights_.norm_query, {std::int32_t(kPleWidth), std::int32_t(kPleStreams)},
                "norm_query must be BF16 [2560,4]");
    bf16_tensor(weights_.norm_conv, {std::int32_t(kPleWidth), std::int32_t(kPleStreams)},
                "norm_conv must be BF16 [2560,4]");
    bf16_tensor(weights_.convolution, {std::int32_t(kPleChannels), std::int32_t(kPleKernelTaps)},
                "convolution must be BF16 [10240,4]");
    require(std::isfinite(weights_.rms_epsilon) && weights_.rms_epsilon > 0,
            "the gate epsilon must be positive and finite");

    // The history pool outlives every step, so it owns its own arena and is never reallocated.
    const std::size_t state = std::size_t(kPleChannels) * kPleHistory * slots_ *
                                  sizeof(std::uint16_t) +
                              std::size_t(max_batch_) * sizeof(std::int32_t);
    state_    = std::make_unique<DeviceArena>(state);
    history_  = state_->alloc(DType::BF16, {std::int32_t(kPleChannels),
                                            std::int32_t(kPleHistory), std::int32_t(slots_)});
    offsets_  = state_->alloc(DType::I32, {std::int32_t(max_batch_)});
    // A pool nobody seeded is a defined all-+0 history, not whatever the allocator returned.
    CUDA_CHECK(cudaMemset(history_.data, 0, history_.bytes()));
    CUDA_CHECK(cudaMemset(offsets_.data, 0, offsets_.bytes()));

    // The Linear's transient arena, sized for the widest step this owner admits. A BF16 route
    // needs none, but the arena is still created so the call site has one form.
    workspace_bytes_ = ops::linear_workspace_capacity_bytes(
        weights_.projection.qtype, weights_.projection.n, weights_.projection.k, weights_.policy, 1,
        std::int32_t(std::size_t(max_tokens_) * max_batch_));
    workspace_ = std::make_unique<DeviceArena>(std::max<std::size_t>(workspace_bytes_, 256));
}

PleLayerRuntime::~PleLayerRuntime() = default;

void PleLayerRuntime::ensure_step(std::uint32_t tokens, std::uint32_t batch) {
    require(tokens <= max_tokens_, "token count exceeds the constructor's capacity");
    require(batch <= max_batch_, "batch exceeds the constructor's capacity");
    if (step_ && tokens == tokens_ && batch == batch_) return;
    // The gate and convolution insist on a contiguous [H,C,T,B] block, so the step scratch is
    // allocated at the step's own shape rather than as a slice of the capacity shape. Releasing it
    // is safe: the history pool is separate and no earlier step keeps a view into this arena.
    step_.reset();
    step_ = std::make_unique<DeviceArena>(step_bytes(tokens, batch));
    const std::int32_t columns = std::int32_t(std::size_t(tokens) * batch);
    projection_ = step_->alloc(DType::BF16,
                               {std::int32_t(kPleProjectionRows), columns});
    gated_      = step_->alloc(DType::BF16, {std::int32_t(kPleWidth), std::int32_t(kPleStreams),
                                             std::int32_t(tokens), std::int32_t(batch)});
    records_    = step_->alloc(DType::BF16, {std::int32_t(kPleChannels), std::int32_t(tokens),
                                             std::int32_t(batch)});
    tokens_     = tokens;
    batch_      = batch;
}

void PleLayerRuntime::require_inputs(const Tensor& embedding, const Tensor& residual,
                                     const Tensor& normalized, const Tensor& valid_columns,
                                     const Tensor& slots, std::uint32_t tokens,
                                     std::uint32_t batch) const {
    i32_batch(valid_columns, "valid_columns must be a contiguous I32 [B]");
    i32_batch(slots, "slots must be a contiguous I32 [B]");
    require(std::uint32_t(valid_columns.ne[0]) == batch, "valid_columns must name one row per batch");
    require(std::uint32_t(slots.ne[0]) == batch, "slots must name one row per batch");
    require(residual.data != nullptr && residual.dtype == DType::BF16 && residual.is_contiguous() &&
                residual.numel() == std::int64_t(kPleChannels) * tokens * batch,
            "residual must be contiguous BF16 [H,C,T,B]");
    require(embedding.data != nullptr && embedding.dtype == DType::BF16 &&
                embedding.is_contiguous() &&
                embedding.numel() == std::int64_t(kPleWidth) * tokens * batch,
            "embedding must be contiguous BF16 [H,T,B]");
    require(normalized.data != nullptr && normalized.dtype == DType::BF16 &&
                normalized.is_contiguous() && normalized.numel() == residual.numel(),
            "normalized must be contiguous BF16 [H,C,T,B]");
}

void PleLayerRuntime::apply(const Tensor& embedding, Tensor& residual, Tensor& normalized,
                            const Tensor& next_norm_weight, float next_epsilon,
                            const Tensor& valid_columns, const Tensor& slots,
                            CudaExecutionView execution) {
    bf16_tensor(next_norm_weight, {std::int32_t(kPleWidth), std::int32_t(kPleStreams)},
                "next_norm_weight must be BF16 [2560,4]");
    require(std::isfinite(next_epsilon) && next_epsilon > 0,
            "the next block's epsilon must be positive and finite");
    i32_batch(valid_columns, "valid_columns must be a contiguous I32 [B]");
    const std::int64_t batch = valid_columns.ne[0];
    require(batch >= 1 && batch <= std::int64_t(max_batch_),
            "batch exceeds the constructor's capacity");
    require(residual.data != nullptr && residual.dtype == DType::BF16 && residual.is_contiguous(),
            "residual must be contiguous BF16 [H,C,T,B]");
    const std::int64_t cells = std::int64_t(kPleChannels) * batch;
    require(residual.numel() % cells == 0, "residual is not a whole number of [H,C,T] columns");
    const std::int64_t tokens = residual.numel() / cells;
    require(tokens >= 1 && tokens <= std::int64_t(max_tokens_),
            "token count exceeds the constructor's capacity");
    require_inputs(embedding, residual, normalized, valid_columns, slots, std::uint32_t(tokens),
                   std::uint32_t(batch));
    ensure_step(std::uint32_t(tokens), std::uint32_t(batch));

    const std::int32_t width   = std::int32_t(kPleWidth);
    const std::int32_t streams = std::int32_t(kPleStreams);
    const std::int32_t columns = std::int32_t(std::size_t(tokens) * batch);

    Tensor residual4   = residual.view({width, streams, std::int32_t(tokens), std::int32_t(batch)});
    Tensor normalized4 = normalized.view({width, streams, std::int32_t(tokens),
                                          std::int32_t(batch)});
    Tensor input       = embedding.view({width, columns});
    Tensor projected   = projection_.view({std::int32_t(kPleProjectionRows), columns});
    ops::linear(input, weights_.projection, projected, weights_.policy, *workspace_,
                execution.stream);

    // `projection` is [key rows (H per stream); value rows], so the two Op views are the two
    // slices of one contiguous parent that the Op contract documents as an example. Their column
    // stride is the parent's row count, not H*C: both blocks of a column sit side by side.
    Tensor key(projection_.data, projection_.device, DType::BF16,
               {width, streams, std::int32_t(tokens), std::int32_t(batch)});
    key.nb[2] = std::int64_t(kPleProjectionRows) * 2;
    key.nb[3] = std::int64_t(kPleProjectionRows) * 2 * tokens;
    Tensor value(static_cast<std::uint16_t*>(projection_.data) + std::size_t(kPleChannels),
                 projection_.device, DType::BF16,
                 {width, std::int32_t(tokens), std::int32_t(batch)});
    value.nb[1] = std::int64_t(kPleProjectionRows) * 2;
    value.nb[2] = std::int64_t(kPleProjectionRows) * 2 * tokens;

    Tensor record_offsets(offsets_.data, offsets_.device, DType::I32, {std::int32_t(batch)});

    const ops::PleBatchView batch_view{valid_columns};
    const ops::PleGateParameters gate_parameters{weights_.norm_key, weights_.norm_query,
                                                 weights_.norm_conv, weights_.rms_epsilon};
    ops::ple_gate({key, value}, residual4, gate_parameters, batch_view,
                  {records_, record_offsets}, gated_, execution);
    ops::ple_conv_add_norm(residual4, gated_, weights_.convolution, batch_view, {history_, slots},
                           {records_, record_offsets}, next_norm_weight, next_epsilon, residual4,
                           normalized4, execution);
}

void PleLayerRuntime::commit(const Tensor& accepted_columns, const Tensor& valid_columns,
                             const Tensor& slots, CudaExecutionView execution) {
    i32_batch(accepted_columns, "accepted_columns must be a contiguous I32 [B]");
    i32_batch(valid_columns, "valid_columns must be a contiguous I32 [B]");
    i32_batch(slots, "slots must be a contiguous I32 [B]");
    require(accepted_columns.ne[0] == valid_columns.ne[0] && slots.ne[0] == valid_columns.ne[0],
            "commit needs one row per batch entry");
    require(!step_.get() || std::uint32_t(valid_columns.ne[0]) == batch_,
            "commit must follow the apply of the same batch");
    ops::ple_state_extract({history_, slots}, {records_, valid_columns}, accepted_columns,
                           {history_, slots}, execution);
}

std::size_t PleLayerRuntime::device_bytes() const noexcept {
    return (state_ ? state_->capacity() : 0) + (step_ ? step_->capacity() : 0) +
           (workspace_ ? workspace_->capacity() : 0);
}

} // namespace ninfer::models::qwen4_exp
