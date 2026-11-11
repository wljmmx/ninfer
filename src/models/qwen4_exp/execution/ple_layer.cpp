#include "models/qwen4_exp/execution/ple_layer.h"

#include "ninfer/ops/ple.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>

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

void PleLayerRuntime::reset_slot(std::int32_t slot, CudaExecutionView execution) {
    require(slot >= 0 && std::uint32_t(slot) < slots_, "the slot is outside the history pool");
    // The pool is [D,9,S], so one slot's nine columns are one contiguous run of nb[2] bytes. Zeroing
    // it is the "a fresh sequence starts from an all-+0 history" rule, applied when the identity is
    // reused rather than when the pool is built.
    auto* base = static_cast<std::uint8_t*>(history_.data) + std::size_t(slot) * history_.nb[2];
    CUDA_CHECK(cudaMemsetAsync(base, 0, std::size_t(history_.nb[2]), execution.stream));
}


std::size_t PleLayerRuntime::device_bytes() const noexcept {
    return (state_ ? state_->capacity() : 0) + (step_ ? step_->capacity() : 0) +
           (workspace_ ? workspace_->capacity() : 0);
}


namespace {

const char* qtype_name(QType type) noexcept {
    switch (type) {
    case QType::BF16:
        return "bf16";
    case QType::FP32:
        return "fp32";
    case QType::INT32:
        return "int32";
    case QType::Q4_G64_FP16:
        return "q4_g64_fp16";
    case QType::Q5_G64_FP16:
        return "q5_g64_fp16";
    case QType::Q6_G64_FP16:
        return "q6_g64_fp16";
    case QType::Q8_G32_FP16:
        return "q8_g32_fp16";
    case QType::NF4_G16_E4M3FN:
        return "nf4_g16_e4m3fn";
    case QType::NVFP4:
        return "nvfp4";
    case QType::FP8_E4M3FN_ROW_BF16:
        return "fp8_e4m3fn_row_bf16";
    case QType::FP8_E4M3FN_BLOCK128X128_BF16:
        return "fp8_e4m3fn_block128x128_bf16";
    case QType::FP8_E4M3FN_BLOCK64X64_BF16:
        return "fp8_e4m3fn_block64x64_bf16";
    }
    return "unknown";
}

const char* layout_name(QuantLayout layout) noexcept {
    switch (layout) {
    case QuantLayout::Contiguous:
        return "contiguous_le_v1";
    case QuantLayout::RowScale:
        return "row_scale_v1";
    case QuantLayout::BlockScaleK16M128x4:
        return "block_scale_k16_m128x4_v1";
    case QuantLayout::RowSplit:
        return "row_split_k128_v1";
    case QuantLayout::RowPagedG16_4K:
        return "row_paged_g16_4k_v1";
    case QuantLayout::BatchedBlockScaleK16M128x4:
        return "batched_block_scale_k16_m128x4_v1";
    case QuantLayout::BlockScaleM128K128:
        return "block_scale_m128_k128_v1";
    case QuantLayout::BlockScaleM64K64:
        return "block_scale_m64_k64_v1";
    }
    return "unknown";
}

// Bounded Host wait for one window. The table lives on disk, so a PLE step cannot avoid it; the
// bound is what keeps a stalled read from becoming a hang.
constexpr auto kWindowDeadline = std::chrono::seconds(120);

} // namespace

void require_supported_ple_table(const PleTableRequirement& requirement) {
    const bool supported =
        supported_ple_table(requirement.format, requirement.layout, requirement.width);
    std::fprintf(stderr, "[ple] table %s: encoding %s/%s, %llu rows x %llu, %s\n",
                 requirement.parameter.c_str(), qtype_name(requirement.format),
                 layout_name(requirement.layout), static_cast<unsigned long long>(requirement.rows),
                 static_cast<unsigned long long>(requirement.width),
                 supported ? "supported" : "unsupported");
    std::fflush(stderr);
    if (supported) return;
    throw std::invalid_argument(
        requirement.parameter + ": 表编码 " + qtype_name(requirement.format) + "/" +
        layout_name(requirement.layout) + "（" + std::to_string(requirement.rows) + " 行 × " +
        std::to_string(requirement.width) + "，每行 " + std::to_string(kPleRowElements) +
        " 个特征）不受支持，可选：转码 / 加解码器（读取器服务 nf4_g16_e4m3fn / "
        "row_paged_g16_4k_v1，每行 90 B；以及 q4_g64_fp16 / row_split_k128_v1，每行 "
        "128 B 码平面 + 8 B fp16 尺度平面）");
}

PleTextStage::PleTextStage(const PleTableSource& table, std::int32_t eos,
                           const DeviceContext& context,
                           std::vector<std::optional<PleLayerWeights>> layers,
                           std::uint32_t max_tokens, std::uint32_t max_batch, std::uint32_t slots,
                           PlePrepareOptions options)
    : eos_(eos), max_tokens_(max_tokens), max_batch_(max_batch), slots_(slots),
      max_positions_(options.max_positions),
      tail_(slots, std::array<std::int32_t, 2>{eos, eos}), started_(slots, false) {
    if (eos < 0) throw std::invalid_argument("PLE stage: the hash's EOS token id is unknown");
    if (!max_tokens || !max_batch || max_batch > 8 || !slots)
        throw std::invalid_argument("PLE stage: invalid step capacity");
    // The window capacity is the number of prepared positions, i.e. the largest T*B one window may
    // ask for - not max(T)*max(B). A pass that only ever runs one wide row (prefill) therefore costs
    // one row's worth of staging, and a batch pass is bounded by the same number of positions.
    if (!options.max_positions || options.max_positions < max_tokens || options.windows < 2)
        throw std::invalid_argument(
            "PLE stage: the window capacity must cover the widest step with two windows");
    layers_.reserve(layers.size());
    for (auto& layer : layers) {
        if (!layer) {
            layers_.push_back(nullptr);
            continue;
        }
        layers_.push_back(std::make_unique<PleLayerRuntime>(*layer, max_tokens, max_batch, slots));
    }
    device_                    = Device::cuda(context.device);
    gpu_                       = std::make_unique<PleGpuPreparation>(table, eos, context, options);
    const std::size_t metadata = std::size_t(max_batch) * sizeof(std::int32_t);
    live_mask_                 = DeviceBuffer(metadata);
    slot_index_                = DeviceBuffer(metadata);
    accept_mask_               = DeviceBuffer(metadata);
}

PleTextStage::~PleTextStage() = default;

Tensor PleTextStage::live_columns() const {
    return Tensor(live_mask_.p, device_, DType::I32, {batch_});
}

Tensor PleTextStage::slot_rows() const {
    return Tensor(slot_index_.p, device_, DType::I32, {batch_});
}

Tensor PleTextStage::accepted_columns() const {
    return Tensor(accept_mask_.p, device_, DType::I32, {batch_});
}

const Tensor& PleTextStage::gated(std::uint32_t layer) const {
    if (!has_layer(layer)) throw std::invalid_argument("PLE stage: this block has no PLE layer");
    return layers_[layer]->gated();
}

const Tensor& PleTextStage::records(std::uint32_t layer) const {
    if (!has_layer(layer)) throw std::invalid_argument("PLE stage: this block has no PLE layer");
    return layers_[layer]->records();
}

const Tensor& PleTextStage::history(std::uint32_t layer) const {
    if (!has_layer(layer)) throw std::invalid_argument("PLE stage: this block has no PLE layer");
    return layers_[layer]->history();
}

void PleTextStage::require_window(const PleStepWindow& window) const {
    require(window.tokens >= 1 && std::uint32_t(window.tokens) <= max_tokens_,
            "the window is wider than the stage's capacity");
    require(!window.slots.empty() && window.slots.size() <= max_batch_ &&
                window.valid_columns.size() == window.slots.size() &&
                window.ids.size() == std::size_t(window.tokens) * window.slots.size(),
            "the window's ids, slots and valid columns disagree");
    require(std::uint64_t(window.tokens) * window.slots.size() <= max_positions_,
            "the window asks for more prepared positions than the stage owns");
    for (const auto slot : window.slots)
        require(slot >= 0 && std::uint32_t(slot) < slots_, "a slot is outside the history pool");
    for (const auto valid : window.valid_columns)
        require(valid <= std::uint32_t(window.tokens), "a live-column count exceeds the window");
}

void PleTextStage::begin(const PleStepWindow& window, CudaExecutionView execution) {
    require(!ticket_.has_value(), "a PLE window is already open");
    require_window(window);
    retire();
    // Where a window's Host time goes, in the order it is spent: the hash and enqueue `prepare`
    // does, the seal, the wait for the worker to publish, and the upload the consumer starts from.
    // The worker's own split (hash / read wait / decode) travels with the reader's stats.
    static const bool timing = std::getenv("NINFER_PLE_TIMING") != nullptr;
    const auto window_begin = std::chrono::steady_clock::now();
    const auto mark         = [&] {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                        window_begin)
            .count();
    };
    const auto batch = std::uint32_t(window.slots.size());
    // A window whose rows all reject their columns has nothing to hash; the reader would refuse
    // it, so it is refused here with the reason instead of three calls later.
    const bool any_live = std::any_of(window.valid_columns.begin(), window.valid_columns.end(),
                                      [](std::uint32_t valid) { return valid != 0; });
    if (!any_live) { throw std::invalid_argument("PLE stage: every row of the window is empty"); }

    std::vector<PleInputRow> rows(batch);
    // The window's ids and live columns are kept on the Host: the hash tail advances by the accepted
    // prefix of them at commit time, and a verify rectangle's ids include candidates nobody accepted.
    window_ids_.assign(window.ids.begin(), window.ids.end());
    window_live_.assign(window.valid_columns.begin(), window.valid_columns.end());
    window_slots_.assign(window.slots.begin(), window.slots.end());
    for (std::uint32_t b = 0; b < batch; ++b) {
        const auto slot  = std::uint32_t(window.slots[b]);
        const auto& tail = started_[slot] ? tail_[slot] : std::array<std::int32_t, 2>{eos_, eos_};
        rows[b]          = PleInputRow{PleSequenceIdentity{.sequence = slot,
                                                           .epoch    = started_[slot] ? 1U : 0U,
                                                           .frontier = std::uint64_t(window.tokens),
                                                           .raw_tail = {tail[0], tail[1]}},
                                       window.ids.subspan(std::size_t(b) * window.tokens,
                                                          std::size_t(window.valid_columns[b]))};
    }
    auto ticket = gpu_->host().prepare(std::uint32_t(window.tokens), rows);
    if (!ticket) {
        // The retired window's worker reclamation is asynchronous; one step of slack is the
        // documented capacity, so a second refusal is a real capacity error.
        for (int attempt = 0; attempt < 2000 && !ticket; ++attempt) {
            std::this_thread::sleep_for(std::chrono::microseconds(500));
            retire();
            ticket = gpu_->host().prepare(std::uint32_t(window.tokens), rows);
        }
        if (!ticket) throw std::runtime_error("PLE stage: no free preparation window");
    }
    ticket_ = ticket;
    tokens_ = window.tokens;
    batch_  = static_cast<std::int32_t>(batch);
    gpu_->host().seal(*ticket_, window.valid_columns);
    const double prepared_ms = mark();

    const auto deadline = std::chrono::steady_clock::now() + kWindowDeadline;
    for (;;) {
        const auto state = gpu_->host().poll(*ticket_);
        if (state.status == PlePrepareStatus::Failed) {
            const auto error = state.error;
            ticket_.reset();
            if (error) { std::rethrow_exception(error); }
            throw std::runtime_error("PLE stage: the window failed without a reason");
        }
        if (state.status == PlePrepareStatus::HostReady) break;
        if (state.status == PlePrepareStatus::Cancelled)
            throw std::runtime_error("PLE stage: the window was cancelled");
        if (std::chrono::steady_clock::now() >= deadline) {
            gpu_->host().cancel(*ticket_);
            ticket_.reset();
            throw std::runtime_error("PLE stage: the window did not reach the Host in time");
        }
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    const double ready_ms = mark();

    live_mask_.copy_from_host(window.valid_columns.data(), window.valid_columns.size() * 4);
    accept_mask_.copy_from_host(window.valid_columns.data(), window.valid_columns.size() * 4);
    slot_index_.copy_from_host(window.slots.data(), window.slots.size() * 4);
    gpu_->upload(*ticket_, execution);
    embedding_ = gpu_->embedding(*ticket_, execution);
    if (timing) {
        const auto stats = gpu_->host().stats();
        const auto count = [](std::uint64_t now, std::uint64_t before) {
            return static_cast<unsigned long long>(now - before);
        };
        std::fprintf(stderr,
                     "[ple-t] window tokens=%d batch=%u: prepare=%.3f ms seal+wait=%.3f ms "
                     "upload=%.3f ms | rows=%llu pages=%llu bytes=%llu hits=%llu shared=%llu/%llu "
                     "| hash=%.3f ms wait=%.3f ms decode=%.3f ms | inflight=%llu\n",
                     window.tokens, batch, prepared_ms, ready_ms - prepared_ms, mark() - ready_ms,
                     count(stats.rows, stats_seen_.rows), count(stats.pages, stats_seen_.pages),
                     count(stats.read_bytes, stats_seen_.read_bytes),
                     count(stats.cache_hits, stats_seen_.cache_hits),
                     count(stats.shared_rows, stats_seen_.shared_rows),
                     count(stats.shared_pages, stats_seen_.shared_pages),
                     double(count(stats.hash_ns, stats_seen_.hash_ns)) / 1e6,
                     double(count(stats.wait_ns, stats_seen_.wait_ns)) / 1e6,
                     double(count(stats.decode_ns, stats_seen_.decode_ns)) / 1e6,
                     static_cast<unsigned long long>(stats.peak_inflight));
        std::fflush(stderr);
        stats_seen_ = stats;
    }
    // One line per window, once: what the n-gram stage gathered and for how many rows. It is the
    // evidence that a pass really drove PLE (an id, its slot and its live columns), and it is the
    // only Host read the stage adds beyond the ids the hash needs.
    static std::atomic<bool> announced{false};
    if (!announced.exchange(true)) {
        std::fprintf(stderr,
                     "[ple] window: tokens=%d batch=%u first id=%d slot=%d live=%u, table rows=%llu\n",
                     window.tokens, batch, window.ids[0], window.slots[0], window.valid_columns[0],
                     static_cast<unsigned long long>(gpu_->host().stats().rows));
        std::fflush(stderr);
    }
    // The hash tail of the next window of each slot is the end of the prefix this window commits,
    // which is why `advance_tail` (not `begin`) moves it: the window's later columns are candidates
    // a verifier may reject.
}

void PleTextStage::advance_tail(std::span<const std::uint32_t> accepted) {
    if (!ticket_.has_value()) { throw std::logic_error("advance_tail has no open window"); }
    const auto batch = std::uint32_t(batch_);
    if (accepted.empty()) { accepted = window_live(); }
    require(accepted.size() == batch, "advance_tail needs one accepted count per row");
    for (std::uint32_t b = 0; b < batch; ++b) {
        const auto slot = std::uint32_t(window_slots_[b]);
        const auto live = window_live_[b];
        const auto count = std::min(accepted[b], live);
        if (count == 0) { continue; }
        const auto* ids = window_ids_.data() + std::size_t(b) * tokens_;
        auto& tail      = tail_[slot];
        // The hash rolls `tail = {tail[1], ids[t]}` per token, so the tail after a committed prefix
        // is the prefix's last two ids - or the previous window's newest id plus the prefix's first.
        tail = count >= 2 ? std::array<std::int32_t, 2>{ids[count - 2], ids[count - 1]}
                          : std::array<std::int32_t, 2>{tail[1], ids[0]};
        started_[slot] = true;
    }
}

void PleTextStage::apply(std::uint32_t layer, Tensor& residual, Tensor& normalized,
                         const Tensor& next_norm, float epsilon, CudaExecutionView execution) {
    require(ticket_.has_value(), "apply has no open window");
    require(has_layer(layer), "this block has no PLE layer");
    require(residual.ne[2] == tokens_ && residual.ne[3] == batch_,
            "the residual does not carry the open window's shape");
    layers_[layer]->apply(embedding_, residual, normalized, next_norm, epsilon, live_columns(),
                          slot_rows(), execution);
}

void PleTextStage::commit(std::uint32_t layer, CudaExecutionView execution) {
    commit(layer, accepted_columns(), execution);
}

void PleTextStage::commit(std::uint32_t layer, const Tensor& accepted,
                          CudaExecutionView execution) {
    require(ticket_.has_value(), "commit has no open window");
    require(has_layer(layer), "this block has no PLE layer");
    layers_[layer]->commit(accepted, live_columns(), slot_rows(), execution);
}

void PleTextStage::reset_slot(std::int32_t slot, CudaExecutionView execution) {
    require(!ticket_.has_value(), "a slot cannot be reset under an open window");
    require(slot >= 0 && std::uint32_t(slot) < slots_, "the slot is outside the history pool");
    for (auto& layer : layers_)
        if (layer) layer->reset_slot(slot, execution);
    // A fresh sequence hashes with EOS history, exactly as a pool nobody has written yet does.
    tail_[std::size_t(slot)]    = {eos_, eos_};
    started_[std::size_t(slot)] = false;
}

void PleTextStage::end(CudaExecutionView execution) {
    if (!ticket_) return;
    gpu_->record_consumed(*ticket_, execution);
    retired_ = ticket_;
    ticket_.reset();
}

void PleTextStage::retire() {
    if (!retired_) return;
    if (gpu_->try_release(*retired_)) retired_.reset();
}

std::size_t PleTextStage::device_bytes() const noexcept {
    std::size_t bytes = live_mask_.bytes + slot_index_.bytes + accept_mask_.bytes;
    for (const auto& layer : layers_) bytes += layer ? layer->device_bytes() : 0;
    if (gpu_) {
        const auto memory = gpu_->memory();
        bytes += memory.device_bytes;
    }
    return bytes;
}


} // namespace ninfer::models::qwen4_exp