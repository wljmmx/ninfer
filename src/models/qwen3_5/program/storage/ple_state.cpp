#include "models/qwen3_5/program/storage/ple_state.h"

#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen4_exp/execution/ple_layer.h"
#include "ninfer/types.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::detail {
namespace {

std::vector<std::optional<qwen4_exp::PleLayerWeights>>
ple_layers(const execution::Parameters& parameters) {
    std::vector<std::optional<qwen4_exp::PleLayerWeights>> layers;
    layers.reserve(parameters.text.layers.size());
    for (const auto& block : parameters.text.layers) {
        if (!block.ple) {
            layers.push_back(std::nullopt);
            continue;
        }
        qwen4_exp::PleLayerWeights weights;
        weights.projection  = block.ple->projection.weight;
        weights.policy      = block.ple->projection.policy;
        weights.norm_key    = block.ple->norm_key;
        weights.norm_query  = block.ple->norm_query;
        weights.norm_conv   = block.ple->norm_conv;
        weights.convolution = block.ple->convolution;
        weights.rms_epsilon = parameters.model.config().text.rms_norm_eps;
        layers.push_back(std::move(weights));
    }
    return layers;
}

} // namespace

std::unique_ptr<qwen4_exp::PleTextStage>
make_ple_stage(const execution::Parameters& parameters, const DeviceContext& context,
               std::uint32_t prefill_chunk) {
    if (!parameters.model.config().text.ple) { return nullptr; }
    // A config that declares PLE layers gets a stage or an error: running the blocks without their
    // n-gram embeddings would be a different model.
    if (!parameters.text.ple_table) {
        throw std::invalid_argument(
            "text: the config declares ple_layer_ids but no table source was provided; PLE needs "
            "PleTableSource, which only the load path can build from a Reader");
    }
    if (parameters.text.ple_eos < 0) {
        throw std::invalid_argument(
            "text: PLE requires the request's EOS id for the n-gram hash; the Program owns it");
    }
    qwen4_exp::PlePrepareOptions options;
    // One window per pass, two so a step's reads overlap the next step's commit. The staging holds
    // `prefill_chunk` prepared positions: one wide prefill row, or a batch rectangle of at most that
    // many columns, and the stage refuses a window wider than that.
    options.max_positions = prefill_chunk;
    options.windows       = 2;
    const auto built_at = std::chrono::steady_clock::now();
    auto stage = std::make_unique<qwen4_exp::PleTextStage>(
        *parameters.text.ple_table, parameters.text.ple_eos, context, ple_layers(parameters),
        prefill_chunk, static_cast<std::uint32_t>(kMaximumConcurrency),
        static_cast<std::uint32_t>(kMaximumConcurrency), options);
    if (std::getenv("NINFER_PLE_TIMING") != nullptr) {
        const auto ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - built_at)
                            .count();
        std::fprintf(stderr, "[ple-t] stage built in %.1f ms (io_depth=%u, %zu bytes device)\n", ms,
                     options.io_depth, stage->device_bytes());
        std::fflush(stderr);
    }
    return stage;
}

PleProgramState::PleProgramState(std::unique_ptr<qwen4_exp::PleTextStage> stage) noexcept
    : stage_(std::move(stage)) {}

PleProgramState::~PleProgramState() = default;

void PleProgramState::window_opened(std::int32_t tokens, bool deferred) noexcept {
    open_     = true;
    deferred_ = deferred;
    tokens_   = tokens;
}

void PleProgramState::window_closed() noexcept {
    open_     = false;
    deferred_ = false;
    tokens_   = 0;
}

bool PleProgramState::commit_verify(const Tensor& accepted, std::size_t layers,
                                    CudaExecutionView execution) {
    if (!stage_ || !open_ || !deferred_) { return false; }
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        if (stage_->has_layer(layer)) stage_->commit(layer, accepted, execution);
    }
    // The hash is a Host computation, so the tail it continues from needs the accepted prefix on the
    // Host. The counts arrive from the verifier's Device tensor; the round already pays one readback
    // for its licensed tokens, and this one is [batch] ints behind the same stream.
    const auto batch = static_cast<std::size_t>(stage_->window_batch());
    std::array<std::uint32_t, kMaximumConcurrency> host{};
    CUDA_CHECK(cudaMemcpyAsync(host.data(), accepted.data, batch * sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, execution.stream));
    CUDA_CHECK(cudaStreamSynchronize(execution.stream));
    stage_->advance_tail(std::span<const std::uint32_t>(host.data(), batch));
    stage_->end(execution);
    window_closed();
    return true;
}

void PleProgramState::close_window(CudaExecutionView execution) {
    if (!stage_ || !open_) { return; }
    // Every PLE block was committed by run_layers as it applied, so only the hash tail is left: it
    // moves by the same live columns the pool did.
    stage_->advance_tail({});
    stage_->end(execution);
    window_closed();
}

void PleProgramState::abort_window(CudaExecutionView execution) noexcept {
    if (!stage_ || !open_) { return; }
    try {
        stage_->end(execution);
    } catch (...) {}
    window_closed();
}

void PleProgramState::reset_lane(std::int32_t lane, CudaExecutionView execution) {
    if (!stage_ || lane < 0 || std::uint32_t(lane) >= stage_->max_batch()) { return; }
    stage_->reset_slot(lane, execution);
}

} // namespace ninfer::models::qwen3_5::detail
