#include "models/qwen3_5/program/graph_execution.h"
#include "models/qwen3_5/execution/linear.h"
#include <cmath>
#include <string>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>
#include "models/qwen3_5/execution/verify_probe.h"
#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/program/execution_context.h"
#include "models/qwen3_5/execution/workspace.h"

#include "core/nvtx.h"
#include "core/transfer.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/dynamic_grouped_conv.h"
#include "ninfer/ops/context_kv_materialize.h"
#include "ninfer/ops/rmsnorm_rope.h"
#include "ninfer/ops/rmsnorm_pack_tail.h"
#include "ninfer/ops/linear_topk.h"
#include "ninfer/ops/candidate_selector.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/prepare_masked_block.h"
#include "ninfer/ops/prepare_ragged_prefix.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/sliding_window_attention.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/speculative_round.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5::execution {
namespace {

using detail::DFlashPersistentState;

// NINFER_DFLASH_TIMING=1 splits one DFlash context append into CUDA-event intervals and prints them
// once per call. The profile the append needs is "where does a prefill chunk's draft half go", and
// the answer has to be per segment: a single nvtx range around the call cannot separate a slow GEMM
// from a slow KV append. Default off, and then every method returns before touching CUDA.
class AppendTiming {
public:
    enum Category : int {
        kFeatureProject = 0,
        kContextNorm,
        kContextMaterialize,
        kLayerLinearPair,
        kKeyNorm,
        kRope,
        kKvAppend,
        kCategoryCount,
    };

    static bool enabled() {
        static const bool on = std::getenv("NINFER_DFLASH_TIMING") != nullptr;
        return on;
    }

    explicit AppendTiming(cudaStream_t stream) : stream_(stream) {
        if (!enabled()) { return; }
        // The decode path calls the same append inside a captured graph. Creating or recording a
        // timing event there invalidates the capture, so the probe stands down on a capturing
        // stream instead of corrupting the graph it is trying to measure from the outside.
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &status) == cudaSuccess &&
            status != cudaStreamCaptureStatusNone) {
            return;
        }
        for (cudaEvent_t& event : events_) {
            if (cudaEventCreate(&event) != cudaSuccess) { return; }
        }
        created_ = true;
        mark(kCategoryCount);
    }

    AppendTiming(const AppendTiming&)            = delete;
    AppendTiming& operator=(const AppendTiming&) = delete;

    ~AppendTiming() {
        if (!created_) { return; }
        for (cudaEvent_t event : events_) { cudaEventDestroy(event); }
    }

    void mark(Category category) {
        if (!created_ || count_ >= static_cast<int>(events_.size())) { return; }
        if (cudaEventRecord(events_[static_cast<std::size_t>(count_)], stream_) != cudaSuccess) {
            return;
        }
        if (count_ > 0) { interval_category_[static_cast<std::size_t>(count_)] = category; }
        ++count_;
    }

    void report(const char* label, int width, int batch) {
        if (!created_ || count_ < 2) { return; }
        cudaEventSynchronize(events_[static_cast<std::size_t>(count_ - 1)]);
        std::array<double, kCategoryCount> total{};
        double span = 0.0;
        for (int i = 0; i < count_ - 1; ++i) {
            float ms = 0.0F;
            if (cudaEventElapsedTime(&ms, events_[static_cast<std::size_t>(i)],
                                     events_[static_cast<std::size_t>(i + 1)]) != cudaSuccess) {
                continue;
            }
            const int category = interval_category_[static_cast<std::size_t>(i + 1)];
            if (category >= 0 && category < kCategoryCount) {
                total[static_cast<std::size_t>(category)] += static_cast<double>(ms);
            }
            span += static_cast<double>(ms);
        }
        std::fprintf(stderr,
                     "[df] %s width=%d batch=%d intervals=%d total=%.2f ms | feature=%.2f norm=%.2f "
                     "materialize=%.2f kv_proj=%.2f knorm=%.2f rope=%.2f kv_append=%.2f\n",
                     label, width, batch, count_ - 1, span, total[kFeatureProject],
                     total[kContextNorm], total[kContextMaterialize], total[kLayerLinearPair],
                     total[kKeyNorm], total[kRope], total[kKvAppend]);
        std::fflush(stderr);
    }

private:
    static constexpr std::size_t kMaxMarks = 128;

    cudaStream_t stream_ = nullptr;
    bool created_        = false;
    int count_           = 0;
    std::array<cudaEvent_t, kMaxMarks> events_{};
    std::array<int, kMaxMarks> interval_category_{};
};

void require_dflash_state(const PrefillContext& state) {
    if (state.dflash == nullptr || !state.execution.parameters.draft.has_value()) {
        throw std::logic_error("masked draft schedule requires its weights and state");
    }
}

DFlashPersistentState& dflash_state(PrefillContext& state) {
    require_dflash_state(state);
    return *state.dflash;
}

DFlashPersistentState& dflash_state(DFlashBatchContext& state) { return state.dflash; }

DFlashPersistentState& dflash_state(DFlashAppendContext& state) { return state.dflash; }

DFlashFeatureSink prefill_feature_sink_impl(PrefillContext& state,
                                            DFlashFeatureSink::PrefillConsumer consume_prefill) {
    {
        require_dflash_state(state);
        const auto& target = state.execution.parameters.model.config().text;
        const auto& config = *state.execution.parameters.model.config().draft;
        return DFlashFeatureSink{
            .features        = &dflash_state(state).prefill_features,
            .positions       = &dflash_state(state).prefill_positions,
            .layers          = std::span<const std::uint32_t>(config.target_layer_ids),
            .consume_prefill = std::move(consume_prefill),
        };
    }
}

DFlashFeatureSink batch_feature_sink_impl(DFlashBatchContext& state, const Tensor& lanes,
                                          const Tensor& valid_columns, std::int32_t width,
                                          std::int32_t batch_size) {
    {
        const auto& target = state.execution.parameters.model.config().text;
        const auto& config = *state.execution.parameters.model.config().draft;
        return DFlashFeatureSink{
            .batch_features      = &dflash_state(state).pending_features,
            .batch_lanes         = &lanes,
            .batch_valid_columns = &valid_columns,
            .batch_width         = width,
            .batch_size          = batch_size,
            .layers              = std::span<const std::uint32_t>(config.target_layer_ids),
        };
    }
}

template <class Context>
void append_context_impl(Context& state, const Tensor& features, const Tensor& positions,
                         const Tensor& commit_counts, const Tensor& lanes, const Tensor& table_rows,
                         ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    {
        const auto& target         = state.execution.parameters.model.config().text;
        const auto& config         = *state.execution.parameters.model.config().draft;
        const std::int32_t width   = features.ne[1];
        const std::int32_t batch   = features.ne[2];
        const std::int32_t columns = width * batch;
        nvtx::ScopedRange append_range(nvtx::Name::DFlashContextAppend, nvtx::Category::DFlash,
                                       static_cast<std::uint64_t>(columns));
        if (width <= 0 || batch <= 0 || features.dtype != DType::BF16 ||
            features.ne[0] !=
                dimension(target.hidden_size * std::uint64_t(config.target_layer_ids.size())) ||
            features.ne[3] != 1 || positions.dtype != DType::I32 || positions.ne[0] != width ||
            positions.ne[1] != batch || commit_counts.dtype != DType::I32 ||
            commit_counts.ne[0] != batch || lanes.dtype != DType::I32 || lanes.ne[0] != batch ||
            table_rows.dtype != DType::I32 || table_rows.ne[0] != batch) {
            throw std::invalid_argument("DFlash context append inputs are invalid");
        }
        const bool replace_local_window =
            batch == 1 && width > dimension(config.sliding_window.value_or(0));
        if (replace_local_window && (envelope.min_count != static_cast<std::uint32_t>(width) ||
                                     envelope.max_count != static_cast<std::uint32_t>(width))) {
            throw std::invalid_argument(
                "DFlash oversized local append requires an exact full-prefix commit");
        }
        const int local_offset =
            replace_local_window ? width - dimension(config.sliding_window.value_or(0)) : 0;
        const int local_width =
            replace_local_window ? dimension(config.sliding_window.value_or(0)) : width;
        const ops::KVCacheAppendPrefixExecutionEnvelope local_envelope{
            replace_local_window
                ? static_cast<std::uint32_t>(dimension(config.sliding_window.value_or(0)))
                : envelope.min_count,
            replace_local_window
                ? static_cast<std::uint32_t>(dimension(config.sliding_window.value_or(0)))
                : envelope.max_count,
        };
        Tensor local_counts = commit_counts;
        if (replace_local_window) {
            if (!state.execution.io.dflash_prefill) {
                throw std::logic_error("DFlash prefill count storage is unavailable");
            }
            local_counts = state.execution.io.dflash_prefill->local_append_count;
            ops::set_i32_scalar(local_counts, dimension(config.sliding_window.value_or(0)),
                                state.execution.device.stream);
        }

        const int projected_width = config.dflash2.has_value() ? local_width : width;
        const Tensor input        = config.dflash2.has_value() && replace_local_window
                                        ? features.slice(1, local_offset, local_width)
                                        : features;
        AppendTiming timing(state.execution.device.stream);
        const auto roots          = workspace::dflash_context(state.execution.work, target, config,
                                                              projected_width * batch);
        Tensor projected          = roots.projected;
        Tensor context            = roots.normalized;
        project(input.view(
                    {dimension(target.hidden_size * std::uint64_t(config.target_layer_ids.size())),
                     projected_width * batch}),
                state.execution.parameters.draft->feature_projection, projected,
                state.execution.work, state.execution.device.stream);
        timing.mark(AppendTiming::kFeatureProject);
        ops::rmsnorm(projected, state.execution.parameters.draft->context_norm, config.rms_norm_eps,
                     false, context, state.execution.device.stream);
        timing.mark(AppendTiming::kContextNorm);

        if (config.dflash2) {
            std::array<ops::ContextKVMaterializeLayerView, ops::kContextKVMaterializeLayers> layers;
            // The fused path is a geometry contract, not a model identity: five fully local
            // layers, the D128/H8 context head, a Q8 kernel projection at the target width, and
            // the eps the fused norm uses. The RoPE base is an argument, so drafts whose config
            // carries another base still take this path.
            const auto& head = state.execution.parameters.draft->layers.front().context_key.weight;
            if (config.num_hidden_layers != layers.size() ||
                config.local_layer_count() != layers.size() || config.rms_norm_eps != 1e-6F ||
                dimension(config.attention.key_width()) != 1024 ||
                dimension(config.attention.num_key_value_heads) != 8 ||
                dimension(config.attention.head_dim) != 128 ||
                head.qtype != QType::Q8_G32_FP16 || head.k != dimension(target.hidden_size)) {
                const std::string text =
                    std::string("masked draft context: config is outside the fused materialization"
                                " contract (layers=") +
                    std::to_string(dimension(config.num_hidden_layers)) +
                    " local=" + std::to_string(dimension(config.local_layer_count())) +
                    " fused=" + std::to_string(layers.size()) +
                    " eps=" + std::to_string(static_cast<double>(config.rms_norm_eps)) +
                    " theta=" + std::to_string(static_cast<double>(config.rope_theta)) +
                    " hidden=" + std::to_string(dimension(target.hidden_size)) +
                    " kv_size=" + std::to_string(dimension(config.attention.key_width())) +
                    " kv_heads=" + std::to_string(dimension(config.attention.num_key_value_heads)) +
                    " head_dim=" + std::to_string(dimension(config.attention.head_dim)) +
                    " context_key_qtype=" + std::to_string(static_cast<int>(head.qtype)) +
                    " context_key_k=" + std::to_string(head.k);
                throw std::invalid_argument(text);
            }
            for (int layer = 0; layer < dimension(config.num_hidden_layers); ++layer) {
                const auto& weights = state.execution.parameters.draft->layers[layer];
                layers[layer]       = {weights.context_key.weight, weights.context_value.weight,
                                       weights.key_norm, dflash_state(state).local_layer(layer)};
            }
            const Tensor local_positions =
                replace_local_window ? positions.slice(0, local_offset, local_width) : positions;
            ops::context_kv_materialize(
                context.view({dimension(target.hidden_size), local_width, batch}), local_positions,
                local_counts, lanes, layers, {local_envelope.min_count, local_envelope.max_count},
                config.rope_theta, state.execution.work, state.execution.device.stream);
            timing.mark(AppendTiming::kContextMaterialize);
        } else {
            for (int layer = 0; layer < dimension(config.num_hidden_layers); ++layer) {
                auto layer_scope = state.execution.work.scope();
                const auto& weight =
                    state.execution.parameters.draft->layers.at(static_cast<std::size_t>(layer));
                const bool local_layer =
                    config.layer_types[layer] == DraftAttentionKind::SlidingAttention;
                const int layer_width   = local_layer ? local_width : width;
                const int layer_columns = layer_width * batch;
                Tensor layer_context    = local_layer && replace_local_window
                                              ? context.slice(1, local_offset, local_width)
                                              : context;
                Tensor layer_positions  = local_layer && replace_local_window
                                              ? positions.slice(0, local_offset, local_width)
                                              : positions;
                auto layer_roots = workspace::dflash_context_layer(state.execution.work, target,
                                                                   config, layer_columns);
                Tensor key_raw   = layer_roots.key_raw.view(
                    {dimension(config.attention.head_dim),
                       dimension(config.attention.num_key_value_heads), layer_columns});
                Tensor value = layer_roots.value.view(
                    {dimension(config.attention.head_dim),
                     dimension(config.attention.num_key_value_heads), layer_columns});
                Tensor key_flat =
                    key_raw.view({dimension(config.attention.key_width()), layer_columns});
                Tensor value_flat =
                    value.view({dimension(config.attention.key_width()), layer_columns});
                ops::linear_pair(layer_context, weight.context_key.weight,
                                 weight.context_value.weight, key_flat, value_flat,
                                 state.execution.device.stream);
                timing.mark(AppendTiming::kLayerLinearPair);
                Tensor key = layer_roots.key.view({dimension(config.attention.head_dim),
                                                   dimension(config.attention.num_key_value_heads),
                                                   layer_columns});
                ops::rmsnorm(key_raw, weight.key_norm, config.rms_norm_eps, false, key,
                             state.execution.device.stream);
                timing.mark(AppendTiming::kKeyNorm);
                ops::rope(layer_positions.view({layer_columns}),
                          dimension(config.attention.head_dim), config.rope_theta, key,
                          state.execution.device.execution_view());
                timing.mark(AppendTiming::kRope);
                Tensor key_batch =
                    key.view({dimension(config.attention.head_dim),
                              dimension(config.attention.num_key_value_heads), layer_width, batch});
                Tensor value_batch    = value.view({dimension(config.attention.head_dim),
                                                    dimension(config.attention.num_key_value_heads),
                                                    layer_width, batch});
                Tensor position_batch = layer_positions.view({layer_width, batch});
                if (local_layer) {
                    ops::kv_cache_append_prefix(
                        key_batch, value_batch, position_batch, local_counts, lanes, local_envelope,
                        dflash_state(state).local_layer(config.compact_layer_index(layer)),
                        state.execution.device.stream);
                } else {
                    ops::kv_cache_append_prefix(
                        key_batch, value_batch, position_batch, commit_counts, table_rows, envelope,
                        dflash_state(state).full_batch_layer(config.compact_layer_index(layer)),
                        state.execution.device.stream);
                }
                timing.mark(AppendTiming::kKvAppend);
            }
        }
        timing.report("append", width, batch);
    }
}

// NINFER_DFLASH2_BLOCK_PROBE: what a block plane actually holds after the last pass captured it.
// One line per plane column - its absolute position and the first four components of kv head 0 -
// so a width dimension that was written once, or a column that carries another column's row, reads
// as a number. Off unless the caller saw the switch.
void probe_block_plane(const Tensor& plane, const Tensor& positions, int width) {
    const int head_dim = plane.ne[0];
    const int kv_heads = plane.ne[1];
    const std::int64_t column_elements = static_cast<std::int64_t>(head_dim) * kv_heads;
    std::vector<std::uint16_t> host(static_cast<std::size_t>(column_elements) * width);
    std::vector<std::int32_t> host_positions(static_cast<std::size_t>(width));
    // Probe only: the pass loop queues the plane writes on the model stream, and pageable host
    // memory is not ordered against that stream, so the readback waits for the device first.
    cudaDeviceSynchronize();
    // Probe only: pageable host memory, so the synchronous forms are the ones that are defined here.
    const cudaError_t plane_status =
        cudaMemcpy2D(host.data(), static_cast<std::size_t>(column_elements) * 2, plane.data,
                     static_cast<std::size_t>(plane.nb[2]),
                     static_cast<std::size_t>(column_elements) * 2, static_cast<std::size_t>(width),
                     cudaMemcpyDeviceToHost);
    const cudaError_t position_status =
        cudaMemcpy(host_positions.data(), positions.data, static_cast<std::size_t>(width) * sizeof(std::int32_t),
                   cudaMemcpyDeviceToHost);
    if (plane_status != cudaSuccess || position_status != cudaSuccess) {
        std::fprintf(stderr, "[blk-probe] plane readback failed: %s / %s\n",
                     cudaGetErrorName(plane_status), cudaGetErrorName(position_status));
        return;
    }
    const auto to_float = [](std::uint16_t bits) {
        const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
        float value;
        std::memcpy(&value, &wide, sizeof(value));
        return value;
    };
    for (int col = 0; col < width; ++col) {
        const std::uint16_t* base = host.data() + col * column_elements;
        std::fprintf(stderr, "[blk-probe] plane col=%d pos=%d k=[%.5f %.5f %.5f %.5f]\n", col,
                     host_positions[static_cast<std::size_t>(col)], to_float(base[0]),
                     to_float(base[1]), to_float(base[2]), to_float(base[3]));
    }
    std::fflush(stderr);
}

// NINFER_DFLASH2_ROPE_PROBE: what the block hands RoPE and what comes back, column by column. The
// block's whole per-column difference rests on `positions`, so its shape and values are printed
// before the call and the rotated q/k after it: a positions tensor with one value per request (or a
// whole block sharing pos0) would leave every column's RoPE identical, which is indistinguishable
// from a head problem once only the proposals are read.
void probe_rope_columns(const Tensor& positions, const Tensor& query, const Tensor& key, int width,
                        int batch, int layer) {
    const auto bits_to_float = [](std::uint16_t bits) {
        const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
        float value;
        std::memcpy(&value, &wide, sizeof(value));
        return value;
    };
    std::vector<std::int32_t> host_positions(static_cast<std::size_t>(positions.ne[0]) *
                                             static_cast<std::size_t>(positions.ne[1]));
    cudaDeviceSynchronize();
    if (cudaMemcpy(host_positions.data(), positions.data,
                   host_positions.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr, "[rope-probe] positions readback failed\n");
        return;
    }
    std::fprintf(stderr, "[rope-probe] layer=%d positions shape=[%d,%d,%d,%d] nb=[%lld,%lld] vals=[",
                 layer, positions.ne[0], positions.ne[1], positions.ne[2], positions.ne[3],
                 static_cast<long long>(positions.nb[0]), static_cast<long long>(positions.nb[1]));
    for (int col = 0; col < positions.ne[0]; ++col) {
        std::fprintf(stderr, "%s%d", col == 0 ? "" : ",", host_positions[static_cast<std::size_t>(col)]);
    }
    std::fprintf(stderr, "]\n");

    const auto print_columns = [&](const char* name, const Tensor& tensor) {
        const std::int64_t column_elements = tensor.nb[2] / static_cast<std::int64_t>(sizeof(std::uint16_t));
        std::vector<std::uint16_t> host(static_cast<std::size_t>(column_elements) * width);
        if (cudaMemcpy(host.data(), tensor.data, host.size() * sizeof(std::uint16_t),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::fprintf(stderr, "[rope-probe] %s readback failed\n", name);
            return;
        }
        for (int col = 0; col < width; ++col) {
            const std::uint16_t* base = host.data() + col * column_elements;
            double sum = 0.0;
            for (std::int64_t i = 0; i < column_elements; ++i) {
                const double v = bits_to_float(base[i]);
                sum += v * v;
            }
            std::fprintf(stderr,
                         "[rope-probe] layer=%d %s col=%d rms=%.5f first=[%.5f %.5f %.5f %.5f]\n", layer,
                         name, col, std::sqrt(sum / static_cast<double>(column_elements)),
                         bits_to_float(base[0]), bits_to_float(base[1]), bits_to_float(base[2]),
                         bits_to_float(base[3]));
        }
    };
    print_columns("q", query);
    print_columns("k", key);
    std::fflush(stderr);
}

// NINFER_DFLASH2_CONV_PROBE: the two-tap dynamic convolution, in and out, column by column. The
// Op's contract is (base[h,0,s] + d[0,s])*n[h,i] + I(i>0)*(base[h,1,s] + d[1,s])*n[h,i-1], so the
// printed pair answers which tap each column actually consumes: column 0 can only consume its own,
// column 1 the anchor's, and every later column the masked row before it.
void probe_conv_taps(const Tensor& residual, const Tensor& prepared, const Tensor& base_kernel,
                     int projection_rows, int projection_columns, int width, int layer) {
    const auto bits_to_float = [](std::uint16_t bits) {
        const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
        float value;
        std::memcpy(&value, &wide, sizeof(value));
        return value;
    };
    const auto print_columns = [&](const char* name, const Tensor& tensor) {
        const std::int64_t column_elements =
            tensor.nb[1] / static_cast<std::int64_t>(sizeof(std::uint16_t));
        std::vector<std::uint16_t> host(static_cast<std::size_t>(column_elements) * width);
        if (cudaMemcpy(host.data(), tensor.data, host.size() * sizeof(std::uint16_t),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::fprintf(stderr, "[conv-probe] %s readback failed\n", name);
            return;
        }
        for (int col = 0; col < width; ++col) {
            const std::uint16_t* base = host.data() + col * column_elements;
            double sum                = 0.0;
            for (std::int64_t i = 0; i < column_elements; ++i) {
                const double v = bits_to_float(base[i]);
                sum += v * v;
            }
            std::fprintf(stderr,
                         "[conv-probe] layer=%d %s col=%d rms=%.5f first=[%.5f %.5f %.5f %.5f]\n",
                         layer, name, col, std::sqrt(sum / static_cast<double>(column_elements)),
                         bits_to_float(base[0]), bits_to_float(base[1]), bits_to_float(base[2]),
                         bits_to_float(base[3]));
        }
    };
    std::vector<std::uint16_t> host_base(static_cast<std::size_t>(base_kernel.numel()));
    cudaDeviceSynchronize();
    if (cudaMemcpy(host_base.data(), base_kernel.data, host_base.size() * sizeof(std::uint16_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "[conv-probe] base kernel readback failed\n");
        return;
    }
    std::fprintf(stderr,
                 "[conv-probe] layer=%d base_kernel shape=[%d,%d,%d] proj=[%d,%d] "
                 "h0 tap0/1 side0=[%.5f %.5f] side1=[%.5f %.5f]\n",
                 layer, base_kernel.ne[0], base_kernel.ne[1], base_kernel.ne[2], projection_rows,
                 projection_columns, bits_to_float(host_base[0]), bits_to_float(host_base[1]),
                 bits_to_float(host_base[2]), bits_to_float(host_base[3]));
    print_columns("in", residual);
    print_columns("out", prepared);
    std::fflush(stderr);
}

void prepare_dynamic_branch(ExecutionCore& execution, const Tensor& residual, const Tensor& norm,
                            float eps, const DynamicConvParameters& weights,
                            workspace::DFlash2BranchRoots& branch) {
    auto scope      = execution.work.scope();
    const int width = residual.ne[1], batch = residual.ne[2];
    WorkspaceArena scratch(execution.work.alloc_bytes(
        ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(
            residual.ne[0], width, width, batch, batch)));
    ops::rmsnorm_dynamic_grouped_conv_prepare(
        residual, norm, eps, weights.base_kernel, weights.kernel_projection.weight, branch.prepared,
        branch.finish_delta, scratch, execution.device.stream);
}

void finish_dynamic_branch(ExecutionCore& execution, const Tensor& input,
                           const LinearParameters& projection, const DynamicConvParameters& weights,
                           const Tensor& finish_delta, Tensor& residual) {
    auto scope      = execution.work.scope();
    const int width = input.ne[1], batch = input.ne[2];
    WorkspaceArena scratch(
        execution.work.alloc_bytes(ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
            residual.ne[0], input.ne[0], width, width, batch, batch)));
    ops::linear_dynamic_grouped_conv_add(input, projection.weight, weights.base_kernel,
                                         finish_delta, residual, scratch, execution.device.stream);
}

void propose_dflash2_batch(DFlashBatchContext& state, qwen3_5::DFlashDecodeState& frame, int batch,
                           int k, DFlashEnvelopes envelopes) {
    if (state.execution.parameters.model.config().draft->dflash2) {
        const auto& target = state.execution.parameters.model.config().text;
        const auto& config = *state.execution.parameters.model.config().draft;
        if (config.rms_norm_eps != 1e-6F) {
            throw std::invalid_argument(
                "DFlash2: fused norm/RoPE requires epsilon=1e-6");
        }

        const int width                = k + 1;
        const int columns              = width * batch;
        const int mask_columns         = k * batch;
        auto& work                     = state.execution.work;
        const cudaStream_t stream      = state.execution.device.stream;
        const DraftParameters& weights = *state.execution.parameters.draft;
        Tensor anchors                 = frame.anchors.slice(0, 0, batch);
        Tensor frontiers               = frame.execution_frontiers.slice(0, 0, batch);
        Tensor valid_columns           = frame.proposal_valid_columns.slice(0, 0, batch);
        Tensor state_slots             = frame.state_destination_slots.slice(0, 0, batch);
        Tensor ids                     = frame.proposal_ids.slice(1, 0, batch);
        Tensor positions               = frame.proposal_positions.slice(1, 0, batch);
        work.reset();
        ops::prepare_masked_block(anchors, frontiers, valid_columns,
                                  dimension(config.mask_token_id), ids, positions, stream);
        Tensor residual = work.alloc(DType::BF16, {dimension(target.hidden_size), width, batch});
        Tensor flat_residual = residual.view({dimension(target.hidden_size), columns});
        // NINFER_DFLASH2_BLOCK_PASSES > 1 only: per-layer K/V planes so a later pass can read the
        // rows an earlier one fixed. The reference block reads the ring alone, so the default path
        // allocates nothing here; see the register entry for why the replay is kept at all.
        const std::int32_t stack_runs = [&] {
            if (std::getenv("NINFER_DFLASH2_BLOCK_LEGACY") != nullptr) { return 1; }
            if (const char* widened = std::getenv("NINFER_DFLASH2_BLOCK_PASSES")) {
                return std::clamp(std::atoi(widened), 1, width);
            }
            return 1;
        }();
        const bool stage_planes = stack_runs > 1;
        std::vector<Tensor> block_keys;
        std::vector<Tensor> block_values;
        if (stage_planes) {
            block_keys.reserve(weights.layers.size());
            block_values.reserve(weights.layers.size());
            for (std::size_t layer_index = 0; layer_index < weights.layers.size(); ++layer_index) {
                block_keys.push_back(work.alloc(DType::BF16,
                                                {dimension(config.attention.head_dim),
                                                 dimension(config.attention.num_key_value_heads),
                                                 width, batch}));
                block_values.push_back(work.alloc(DType::BF16,
                                                  {dimension(config.attention.head_dim),
                                                   dimension(config.attention.num_key_value_heads),
                                                   width, batch}));
            }
        }

        // NINFER_DFLASH2_BLOCK_LEGACY drops the bound, which is what this tree did before it
        // existed; kept only so the A/B can name both sides from one binary.
        const bool causal = std::getenv("NINFER_DFLASH2_BLOCK_LEGACY") == nullptr;
        const auto run_stack = [&](std::int32_t live_block_rows) {
            ops::embedding(ids.view({columns}), state.execution.parameters.text.token_embedding,
                           flat_residual, stream);
            for (std::size_t layer_index = 0; layer_index < weights.layers.size(); ++layer_index) {
                const auto& layer = weights.layers[layer_index];
                nvtx::ScopedRange layer_range(nvtx::Name::DFlashLayer, nvtx::Category::DFlash,
                                              layer_index);
                {
                    auto scope  = work.scope();
                    auto branch = workspace::dflash2_branch(work, target, config, width, batch);
                    prepare_dynamic_branch(state.execution, residual, layer.input_norm,
                                           config.rms_norm_eps, *layer.attention_conv, branch);
                    {
                        static const bool conv_probe =
                            std::getenv("NINFER_DFLASH2_ROW_PROBE") != nullptr;
                        if (conv_probe && layer_index == 0) {
                            verify_column_probe("dconv-prepared", branch.prepared,
                                                dimension(target.hidden_size),
                                                static_cast<std::int32_t>(branch.prepared.nb[1]),
                                                width, 0, state.execution.device.stream);
                        }
                    }
                    if (std::getenv("NINFER_DFLASH2_CONV_PROBE") != nullptr && layer_index == 0) {
                        probe_conv_taps(residual, branch.prepared, layer.attention_conv->base_kernel,
                                        layer.attention_conv->kernel_projection.weight.n,
                                        layer.attention_conv->kernel_projection.weight.k, width,
                                        static_cast<int>(layer_index));
                    }
                    Tensor query = work.alloc(
                        DType::BF16, {dimension(config.attention.head_dim),
                                      dimension(config.attention.num_attention_heads), width, batch});
                    Tensor key = work.alloc(
                        DType::BF16, {dimension(config.attention.head_dim),
                                      dimension(config.attention.num_key_value_heads), width, batch});
                    Tensor value = work.alloc(
                        DType::BF16, {dimension(config.attention.head_dim),
                                      dimension(config.attention.num_key_value_heads), width, batch});
                    Tensor query_flat =
                        query.view({dimension(config.attention.query_width()), columns});
                    Tensor key_flat = key.view({dimension(config.attention.key_width()), columns});
                    Tensor value_flat =
                        value.view({dimension(config.attention.key_width()), columns});
                    ops::attn_input_proj(
                        branch.prepared.view({dimension(target.hidden_size), columns}),
                        layer.query_key_value.weight, query_flat, key_flat, value_flat, stream);
                    ops::rmsnorm_rope(positions, layer.query_norm, layer.key_norm, query, key,
                                      config.rope_theta, stream);
                    if (std::getenv("NINFER_DFLASH2_ROPE_PROBE") != nullptr && layer_index == 0) {
                        probe_rope_columns(positions, query, key, width, batch,
                                           static_cast<int>(layer_index));
                    }
                    Tensor attention = work.alloc(
                        DType::BF16, {dimension(config.attention.head_dim),
                                      dimension(config.attention.num_attention_heads), width, batch});
                    ops::sliding_window_attention(
                        query, key, value, positions, valid_columns, state_slots,
                        {dimension(config.attention.head_dim),
                         dimension(config.attention.num_attention_heads),
                         dimension(config.attention.num_key_value_heads)},
                        dimension(config.sliding_window.value_or(0)),
                        static_cast<float>(1.0 / std::sqrt(double(config.attention.head_dim))),
                        state.dflash.local_layer(static_cast<std::uint32_t>(layer_index)),
                        envelopes.local, work, attention, stream,
                        // DFlash2 only: the block's rows are [anchor, mask, mask, ...], and the
                        // reference block reads the committed ring alone - no block row sees the
                        // anchor or its siblings. The planes below exist only for the diagnostic
                        // replay (NINFER_DFLASH2_BLOCK_PASSES > 1); the v1 path keeps the default.
                        /*block_causal=*/causal,
                        live_block_rows > 0 ? &block_keys[layer_index] : nullptr,
                        live_block_rows > 0 ? &block_values[layer_index] : nullptr,
                        live_block_rows);
                    finish_dynamic_branch(
                        state.execution,
                        attention.view({dimension(config.attention.query_width()), width, batch}),
                        layer.output, *layer.attention_conv, branch.finish_delta, residual);
                    if (stage_planes) {
                        // V10-12: every plain cudaMemcpyAsync in this file names its direction here.
                        // The blocking cudaMemcpy/cudaMemcpy2D probe reads further down branch on the
                        // result and stay as they are (no stream, different call).
                        core::copy_bytes(block_keys[layer_index].data, key.data, key.bytes(),
                                         core::TransferDirection::DeviceToDevice, stream);
                        core::copy_bytes(block_values[layer_index].data, value.data, value.bytes(),
                                         core::TransferDirection::DeviceToDevice, stream);
                    }
                }
                {
                    auto scope  = work.scope();
                    auto branch = workspace::dflash2_branch(work, target, config, width, batch);
                    prepare_dynamic_branch(state.execution, residual, layer.post_attention_norm,
                                           config.rms_norm_eps, *layer.mlp_conv, branch);
                    Tensor intermediate =
                        work.alloc(DType::BF16, {dimension(config.intermediate_size), width, batch});
                    Tensor intermediate_flat =
                        intermediate.view({dimension(config.intermediate_size), columns});
                    project_swiglu(branch.prepared.view({dimension(target.hidden_size), columns}),
                                   layer.mlp.gate_up, intermediate_flat, work, stream);
                    finish_dynamic_branch(state.execution, intermediate, layer.mlp.down,
                                          *layer.mlp_conv, branch.finish_delta, residual);
                }
            }
        };
        // Two passes: the first replays the block with committed context alone in view and leaves
        // its rows in the planes; the second admits them causally, so column j sees columns 0..j-1
        // as the values the pass before computed for them.
        for (std::int32_t pass = 0; pass < stack_runs; ++pass) { run_stack(pass); }
        if (std::getenv("NINFER_DFLASH2_BLOCK_PROBE") != nullptr && !block_keys.empty()) {
            probe_block_plane(block_keys.front(), positions, width);
        }
        Tensor hidden = work.alloc(DType::BF16, {dimension(target.hidden_size), mask_columns});
        ops::rmsnorm_pack_tail(residual, weights.final_norm, hidden, stream);
        // NINFER_DFLASH2_ROW_PROBE: what each masked row hands the head, per column. A head that is
        // only valid for the first row would show one column with an ordinary magnitude and the
        // rest collapsing, which is the same signature an input-side error would have.
        {
            static const bool row_probe = std::getenv("NINFER_DFLASH2_ROW_PROBE") != nullptr;
            if (row_probe) {
                static const bool told = [&] {
                    std::fprintf(stderr,
                                 "[row-probe] mask_token_id=%d selector_top_k=%d selector_rank=%d "
                                 "rows=%d\n",
                                 dimension(config.mask_token_id),
                                 dimension(config.dflash2->selector_top_k),
                                 dimension(config.dflash2->selector_rank), int(mask_columns));
                    std::fflush(stderr);
                    return true;
                }();
                (void)told;
                // Which ring each layer addresses: the append path compacts the layer index by
                // kind, this attention call does not, so a draft with mixed layer kinds would have
                // the two disagree. Print both for every layer once.
                static const bool told_layers = [&] {
                    std::fprintf(stderr, "[row-probe] layers=%zu ring_map=",
                                 size_t(dimension(config.num_hidden_layers)));
                    for (std::int32_t l = 0; l < dimension(config.num_hidden_layers); ++l) {
                        std::fprintf(stderr, "%d:%u-", l,
                                     config.compact_layer_index(static_cast<std::uint32_t>(l)));
                    }
                    std::fputc('\n', stderr);
                    std::fflush(stderr);
                    return true;
                }();
                (void)told_layers;
                verify_column_probe("draft-head-in", hidden, dimension(target.hidden_size),
                                    static_cast<std::int32_t>(hidden.nb[1]), mask_columns, 0, stream);
            }
        }
        // The head, and the selector walk that turns its top-k into one token per column. Each run
        // owns its own scratch so the block can be replayed after the walk without the two peaks
        // overlapping in the arena.
        const auto run_head = [&] {
            auto scope        = work.scope();
            Tensor candidates = frame.candidate_ids.slice(2, 0, batch);
            Tensor ids_flat =
                candidates.view({dimension(config.dflash2->selector_top_k), mask_columns});
            Tensor scores =
                work.alloc(DType::FP32, {dimension(config.dflash2->selector_top_k), mask_columns});
            if (state.execution.proposal_head == ProposalHead::Full) {
                ops::linear_topk(
                    hidden, state.execution.parameters.draft->output_head.weight,
                    dimension(state.execution.parameters.model.resources().public_token_count),
                    ids_flat, scores, work, stream);
            } else {
                const auto& head = *state.execution.parameters.proposal;
                if (head.token_ids) {
                    ops::linear_topk(hidden, head.head.weight, *head.token_ids, ids_flat, scores,
                                     work, stream);
                } else {
                    ops::linear_topk(
                        hidden, head.head.weight,
                        dimension(state.execution.parameters.model.resources().public_token_count),
                        ids_flat, scores, work, stream);
                }
            }
            Tensor projected =
                work.alloc(DType::BF16, {dimension(config.dflash2->selector_rank), mask_columns});
            project(hidden, weights.selector->hidden_projection, projected, work, stream);
            Tensor drafts     = frame.draft_tokens.slice(1, 0, batch);
            Tensor proposal_q = frame.proposal_q.slice(2, 0, batch);
            ops::candidate_selector_path(
                candidates, scores.view({dimension(config.dflash2->selector_top_k), k, batch}),
                projected.view({dimension(config.dflash2->selector_rank), k, batch}), anchors,
                weights.selector->predecessor_codebook, weights.selector->successor_codebook,
                frontiers, frame.sampling, drafts, proposal_q, work, stream);
            // NINFER_DRAFT_CONF / --draft-conf: the first column whose pick is below the floor ends
            // the verified prefix, so the target never spends a column on a suffix the draft itself
            // does not believe. This score tensor is the head's own top sixteen - the one place a
            // per-column confidence exists here - and zero (the shipped default) runs nothing.
            if (state.execution.draft_conf > 0.0F) {
                Tensor extents = frame.proposal_extents.slice(0, 0, batch);
                ops::speculative_draft_conf_extent(
                    scores.view({dimension(config.dflash2->selector_top_k), k, batch}), extents,
                    static_cast<std::int32_t>(k), state.execution.draft_conf, stream);
            }
            // NINFER_DFLASH2_HEAD_AB=<path>: the same draft hidden row (column 0) through the target's
            // own LM head as well, so "the two heads disagree on one hidden" and "the draft hidden
            // ranks badly whatever head reads it" stop looking alike. One line per round:
            // step, q8 argmax, q5 argmax, top-16 intersection, q8 unary of its own top-1.
            if (const char* ab = std::getenv("NINFER_DFLASH2_HEAD_AB")) {
                static std::FILE* file = std::fopen(ab, "wb");
                if (file != nullptr) {
                    static std::atomic<std::uint64_t> head_steps{0};
                    const std::uint64_t step   = head_steps.fetch_add(1);
                    const auto vocab_rows      = dimension(
                        state.execution.parameters.model.resources().public_token_count);
                    Tensor column = hidden.slice(1, 0, 1);
                    Tensor ab_ids =
                        work.alloc(DType::I32, {dimension(config.dflash2->selector_top_k), 1});
                    Tensor ab_scores =
                        work.alloc(DType::FP32, {dimension(config.dflash2->selector_top_k), 1});
                    ops::linear_topk(column, state.execution.parameters.text.output_head.weight,
                                     vocab_rows, ab_ids, ab_scores, work, stream);
                    std::vector<std::int32_t> q5(16), q8(16);
                    std::vector<float> q5_scores(16);
                    core::copy_bytes(q8.data(), candidates.slice(1, 0, 1).data,
                                     q8.size() * sizeof(std::int32_t),
                                     core::TransferDirection::DeviceToHost, stream);
                    core::copy_bytes(q5.data(), ab_ids.data, q5.size() * sizeof(std::int32_t),
                                     core::TransferDirection::DeviceToHost, stream);
                    core::copy_bytes(q5_scores.data(), ab_scores.data,
                                     q5_scores.size() * sizeof(float),
                                     core::TransferDirection::DeviceToHost, stream);
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    int overlap = 0;
                    for (std::int32_t id : q8) {
                        overlap += std::find(q5.begin(), q5.end(), id) != q5.end() ? 1 : 0;
                    }
                    std::fprintf(file, "%llu,%d,%d,%d,%.6f\n",
                                 static_cast<unsigned long long>(step), q8.front(), q5.front(),
                                 overlap, double(q5_scores.front()));
                    std::fflush(file);
                }
            }
        };
        run_head();
        work.reset();
    }
}

void propose_batch_impl(DFlashBatchContext& state, qwen3_5::DFlashDecodeState& frame,
                        std::int32_t batch_size, std::uint32_t k, DFlashEnvelopes envelopes) {
    if (state.execution.parameters.model.config().draft->dflash2) {
        nvtx::ScopedRange proposal_range(nvtx::Name::DFlashProposal, nvtx::Category::DFlash,
                                         static_cast<std::uint64_t>(k + 1U) * batch_size);
        propose_dflash2_batch(state, frame, batch_size, k, envelopes);
    } else {
        const auto& target         = state.execution.parameters.model.config().text;
        const auto& config         = *state.execution.parameters.model.config().draft;
        const std::int32_t width   = static_cast<std::int32_t>(k) + 1;
        const std::int32_t columns = width * batch_size;
        nvtx::ScopedRange proposal_range(nvtx::Name::DFlashProposal, nvtx::Category::DFlash,
                                         static_cast<std::uint64_t>(columns));
        Tensor anchors            = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers          = frame.execution_frontiers.slice(0, 0, batch_size);
        Tensor valid_columns      = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor state_destinations = frame.state_destination_slots.slice(0, 0, batch_size);
        Tensor full_rows          = frame.dflash_kv_table_rows.slice(0, 0, batch_size);
        Tensor ids                = frame.proposal_ids.slice(1, 0, batch_size);
        Tensor positions          = frame.proposal_positions.slice(1, 0, batch_size);
        Tensor drafts             = frame.draft_tokens.slice(1, 0, batch_size);

        state.execution.work.reset();
        ops::prepare_masked_block(anchors, frontiers, valid_columns,
                                  dimension(config.mask_token_id), ids, positions,
                                  state.execution.device.stream);
        Tensor residual =
            state.execution.work.alloc(DType::BF16, {dimension(target.hidden_size), columns});
        ops::embedding(ids.view({columns}), state.execution.parameters.text.token_embedding,
                       residual, state.execution.device.stream);

        for (int layer = 0; layer < dimension(config.num_hidden_layers); ++layer) {
            nvtx::ScopedRange layer_range(nvtx::Name::DFlashLayer, nvtx::Category::DFlash,
                                          static_cast<std::uint64_t>(layer));
            const auto& weight =
                state.execution.parameters.draft->layers.at(static_cast<std::size_t>(layer));
            {
                nvtx::ScopedRange attention_range(nvtx::Name::DFlashAttention,
                                                  nvtx::Category::Attention,
                                                  static_cast<std::uint64_t>(layer));
                auto attention_scope = state.execution.work.scope();
                auto roots =
                    workspace::dflash_attention(state.execution.work, target, config, columns);
                ops::rmsnorm(residual, weight.input_norm, config.rms_norm_eps, false, roots.hidden,
                             state.execution.device.stream);
                Tensor query_raw = roots.query_raw.view(
                    {dimension(config.attention.head_dim),
                     dimension(config.attention.num_attention_heads), columns});
                Tensor key_raw =
                    roots.key_raw.view({dimension(config.attention.head_dim),
                                        dimension(config.attention.num_key_value_heads), columns});
                Tensor value =
                    roots.value.view({dimension(config.attention.head_dim),
                                      dimension(config.attention.num_key_value_heads), columns});
                Tensor query_flat =
                    query_raw.view({dimension(config.attention.query_width()), columns});
                Tensor key_flat = key_raw.view({dimension(config.attention.key_width()), columns});
                Tensor value_flat = value.view({dimension(config.attention.key_width()), columns});
                ops::attn_input_proj(roots.hidden, weight.query_key_value.weight, query_flat,
                                     key_flat, value_flat, state.execution.device.stream);
                Tensor query =
                    roots.query.view({dimension(config.attention.head_dim),
                                      dimension(config.attention.num_attention_heads), columns});
                Tensor key =
                    roots.key.view({dimension(config.attention.head_dim),
                                    dimension(config.attention.num_key_value_heads), columns});
                ops::rmsnorm(query_raw, weight.query_norm, config.rms_norm_eps, false, query,
                             state.execution.device.stream);
                ops::rmsnorm(key_raw, weight.key_norm, config.rms_norm_eps, false, key,
                             state.execution.device.stream);
                ops::rope(positions.view({columns}), dimension(config.attention.head_dim),
                          config.rope_theta, query, key, state.execution.device.execution_view());
                Tensor query_batch = query.view({dimension(config.attention.head_dim),
                                                 dimension(config.attention.num_attention_heads),
                                                 width, batch_size});
                Tensor key_batch =
                    key.view({dimension(config.attention.head_dim),
                              dimension(config.attention.num_key_value_heads), width, batch_size});
                Tensor value_batch     = value.view({dimension(config.attention.head_dim),
                                                     dimension(config.attention.num_key_value_heads),
                                                     width, batch_size});
                Tensor attention_batch = roots.attention.view(
                    {dimension(config.attention.head_dim),
                     dimension(config.attention.num_attention_heads), width, batch_size});
                if (config.layer_types[layer] == DraftAttentionKind::SlidingAttention) {
                    ops::sliding_window_attention(
                        query_batch, key_batch, value_batch, positions, valid_columns,
                        state_destinations,
                        {dimension(config.attention.head_dim),
                         dimension(config.attention.num_attention_heads),
                         dimension(config.attention.num_key_value_heads)},
                        dimension(config.sliding_window.value_or(0)),
                        static_cast<float>(1.0 / std::sqrt(double(config.attention.head_dim))),
                        dflash_state(state).local_layer(config.compact_layer_index(layer)),
                        envelopes.local, state.execution.work, attention_batch,
                        state.execution.device.stream);
                } else {
                    ops::context_softmax_attention(
                        query_batch, key_batch, value_batch, frontiers, valid_columns, full_rows,
                        {dimension(config.attention.head_dim),
                         dimension(config.attention.num_attention_heads),
                         dimension(config.attention.num_key_value_heads)},
                        static_cast<float>(1.0 / std::sqrt(double(config.attention.head_dim))),
                        dflash_state(state).full_batch_layer(config.compact_layer_index(layer)),
                        envelopes.full, state.execution.work, attention_batch,
                        state.execution.device.stream);
                }
                project_add(
                    roots.attention.view({dimension(config.attention.query_width()), columns}),
                    weight.output, residual, state.execution.work, state.execution.device.stream);
            }
            {
                nvtx::ScopedRange mlp_range(nvtx::Name::DFlashMlp, nvtx::Category::PostMixer,
                                            static_cast<std::uint64_t>(layer));
                auto mlp_scope = state.execution.work.scope();
                auto roots = workspace::dflash_mlp(state.execution.work, target, config, columns);
                ops::rmsnorm(residual, weight.post_attention_norm, config.rms_norm_eps, false,
                             roots.hidden, state.execution.device.stream);
                project_swiglu(roots.hidden, weight.mlp.gate_up, roots.intermediate,
                               state.execution.work, state.execution.device.stream);
                project_add(roots.intermediate, weight.mlp.down, residual, state.execution.work,
                            state.execution.device.stream);
            }
        }

        Tensor packed =
            state.execution.work.alloc(DType::BF16, {dimension(target.hidden_size),
                                                     static_cast<std::int32_t>(k) * batch_size});
        const std::size_t element_bytes = dtype_size(DType::BF16);
        const std::size_t row_bytes     = static_cast<std::size_t>(dimension(target.hidden_size)) *
                                      static_cast<std::size_t>(k) * element_bytes;
        const std::size_t source_pitch =
            static_cast<std::size_t>(dimension(target.hidden_size)) * width * element_bytes;
        const auto* source =
            static_cast<const std::byte*>(residual.data) +
            static_cast<std::size_t>(dimension(target.hidden_size)) * element_bytes;
        CUDA_CHECK(cudaMemcpy2DAsync(packed.data, row_bytes, source, source_pitch, row_bytes,
                                     static_cast<std::size_t>(batch_size), cudaMemcpyDeviceToDevice,
                                     state.execution.device.stream));
        Tensor proposal_hidden =
            state.execution.work.alloc(DType::BF16, {dimension(target.hidden_size),
                                                     static_cast<std::int32_t>(k) * batch_size});
        ops::rmsnorm(packed, state.execution.parameters.draft->final_norm, config.rms_norm_eps,
                     false, proposal_hidden, state.execution.device.stream);
        Tensor flat_drafts = drafts.view({static_cast<std::int32_t>(k) * batch_size});
        if (state.execution.proposal_head == ProposalHead::Full) {
            Tensor logits = state.execution.work.alloc(
                DType::BF16,
                {dimension(target.vocab_size), static_cast<std::int32_t>(k) * batch_size});
            project(proposal_hidden, state.execution.parameters.draft->output_head, logits,
                    state.execution.work, state.execution.device.stream);
            ops::argmax(logits, flat_drafts,
                        dimension(state.execution.parameters.model.resources().public_token_count),
                        state.execution.device.stream);
        } else {
            if (!state.execution.parameters.proposal.has_value()) {
                throw std::logic_error("optimized DFlash proposal head is unavailable");
            }
            const auto& proposal = *state.execution.parameters.proposal;
            Tensor logits        = state.execution.work.alloc(
                DType::BF16, {dimension(proposal.rows), static_cast<std::int32_t>(k) * batch_size});
            project(proposal_hidden, proposal.head, logits, state.execution.work,
                    state.execution.device.stream);
            ops::argmax(
                logits, flat_drafts,
                proposal.token_ids
                    ? dimension(proposal.rows)
                    : dimension(state.execution.parameters.model.resources().public_token_count),
                state.execution.device.stream);
            if (proposal.token_ids)
                ops::proposal_remap_token_ids(
                    flat_drafts, static_cast<const std::int32_t*>(proposal.token_ids->data),
                    dimension(proposal.rows), state.execution.device.stream);
        }
        state.execution.work.reset();
    }
}

auto dflash_decode_batch_body(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                              DFlashEnvelopes envelopes,
                              ops::CausalAttentionExecutionEnvelope target_envelope,
                              SpeculativePhase phase, bool capturing) {
    return [&state, batch_size, k, envelopes, target_envelope, phase, capturing] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency) ||
            k == 0 || k > kDFlashDecodeMaximumDrafts) {
            throw std::logic_error("DFlash decode batch state is incomplete");
        }
        // NINFER_DFLASH_PHASE_TIMER=<path>: the round split into proposal / verify / accept, in GPU
        // milliseconds, one CSV line per round. The tree question is whether a verified row costs the
        // same when its state comes from a sibling fork as when it comes from the row before it, and
        // that cannot be read out of a round total that also carries the draft. Off by default, and
        // eager only: an event record inside a captured round is illegal, so a captured round runs
        // exactly as before.
        struct PhaseTimer {
            cudaEvent_t begin = nullptr, split = nullptr, end = nullptr;
            cudaEvent_t accept_begin = nullptr, accept_end = nullptr;
            std::FILE* file     = nullptr;
            std::uint64_t round = 0;
            bool active         = false;

            PhaseTimer() {
                const char* path = std::getenv("NINFER_DFLASH_PHASE_TIMER");
                if (path == nullptr) return;
                file = std::fopen(path, "wb");
                if (file == nullptr) return;
                if (cudaEventCreate(&begin) != cudaSuccess || cudaEventCreate(&split) != cudaSuccess ||
                    cudaEventCreate(&end) != cudaSuccess ||
                    cudaEventCreate(&accept_begin) != cudaSuccess ||
                    cudaEventCreate(&accept_end) != cudaSuccess) {
                    CUDA_CHECK(cudaGetLastError());
                    return;
                }
                active = true;
            }
        };
        static PhaseTimer timer;
        const bool timing = timer.active && !capturing;

        state.execution.work.reset();
        qwen3_5::DFlashDecodeState& frame = state.frame;
        const std::int32_t width          = static_cast<std::int32_t>(k) + 1;
        if (phase == SpeculativePhase::Forward) {
            core::copy_bytes(frame.ingress.data, &state.host_ingress,
                             sizeof(qwen3_5::DFlashDecodeIngress),
                             core::TransferDirection::HostToDevice,
                             state.execution.device.stream);
        }

        Tensor anchors            = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers          = frame.execution_frontiers.slice(0, 0, batch_size);
        Tensor context_starts     = frame.context_frontiers.slice(0, 0, batch_size);
        Tensor extents            = frame.proposal_extents.slice(0, 0, batch_size);
        Tensor valid_columns      = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor target_rope        = frame.target_rope_positions.slice(1, 0, batch_size);
        Tensor text_rows          = frame.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor dflash_rows        = frame.dflash_kv_table_rows.slice(0, 0, batch_size);
        Tensor active_lanes       = frame.active_lanes.slice(0, 0, batch_size);
        Tensor state_sources      = frame.state_source_slots.slice(0, 0, batch_size);
        Tensor state_destinations = frame.state_destination_slots.slice(0, 0, batch_size);
        Tensor append_positions   = frame.append_positions.slice(1, 0, batch_size);
        Tensor append_counts      = frame.append_counts.slice(0, 0, batch_size);
        Tensor drafts             = frame.draft_tokens.slice(1, 0, batch_size);
        Tensor verify_ids         = frame.verify_ids.slice(1, 0, batch_size);
        Tensor target_positions   = frame.verify_positions.slice(1, 0, batch_size);
        Tensor target_tokens      = frame.target_argmax.slice(1, 0, batch_size);
        Tensor target_logits      = frame.target_logits.slice(2, 0, batch_size);
        Tensor target_hidden      = frame.target_hidden.slice(2, 0, batch_size);
        Tensor selected_hidden    = frame.target_continuation_hidden.slice(1, 0, batch_size);
        Tensor licensed_tokens    = frame.licensed_tokens.slice(1, 0, batch_size);
        Tensor licensed_counts    = frame.licensed_counts.slice(0, 0, batch_size);
        Tensor accepted           = frame.accepted_drafts.slice(0, 0, batch_size);

        static bool told_lanes[kMaximumConcurrency + 1] = {};
        if (timing && phase == SpeculativePhase::Forward && !told_lanes[batch_size]) {
            told_lanes[batch_size] = true;
            // Deliverable of the probe: which state image and which KV table row each lane's whole
            // rectangle runs on. One pair per LANE, not per column - every column of the rectangle
            // shares its lane's source and destination, which is what makes the columns a chain
            // inside gated_delta_net_batch and not siblings of one parent.
            for (std::int32_t row = 0; row < batch_size; ++row) {
                std::fprintf(stderr,
                             "[phase-timer] lane=%d/%d width=%d k=%u state_source=%d state_dest=%d "
                             "text_kv_row=%d valid_columns=%d extent=%d anchor=%d\n",
                             row, batch_size, width, k, state.host_ingress.state_source_slots[row],
                             state.host_ingress.state_destination_slots[row],
                             state.host_ingress.text_kv_table_rows[row],
                             state.host_ingress.target_valid_columns[row],
                             state.host_ingress.proposal_extents[row],
                             state.host_ingress.anchors[row]);
            }
            std::fflush(stderr);
        }
        if (timing && phase == SpeculativePhase::Forward) {
            CUDA_CHECK(cudaEventRecord(timer.begin, state.execution.device.stream));
        }

        if (phase == SpeculativePhase::Forward) {
            state.execution.work.reset();
            Tensor compact_features = state.execution.work.alloc(
                DType::BF16,
                {dimension(state.execution.parameters.draft->feature_projection.weight.k), width,
                 batch_size});
            ops::prepare_ragged_prefix(
                dflash_state(state).pending_features, active_lanes, context_starts, frontiers,
                compact_features, append_positions, append_counts, state.execution.device.stream);
            append_context_impl(state, compact_features, append_positions, append_counts,
                                state_destinations, dflash_rows, envelopes.append);

            propose_batch_impl(state, frame, batch_size, k, envelopes);
            const auto draft_count = static_cast<std::size_t>(k) * batch_size;
            if (state.host_drafts.size() < draft_count) {
                throw std::logic_error("DFlash host draft buffer is too small");
            }
            core::copy_bytes(state.host_drafts.data(), drafts.data, draft_count * sizeof(TokenId),
                             core::TransferDirection::DeviceToHost,
                             state.execution.device.stream);
            if (capturing) {
                state.drafts_ready.record_external(state.execution.device.stream);
            } else {
                state.drafts_ready.record(state.execution.device.stream);
            }
            ops::speculative_prepare_verify_inputs(anchors, drafts, frontiers, extents, verify_ids,
                                                   target_positions, state.execution.device.stream);
            if (timing) CUDA_CHECK(cudaEventRecord(timer.split, state.execution.device.stream));
        }


        TextContext card(state.execution.device, state.execution.parameters, state.execution.work,
                         {}, state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache, nullptr, state.execution.ple);
        DFlashFeatureSink sink =
            batch_feature_sink_impl(state, active_lanes, valid_columns, width, batch_size);
        {
            nvtx::ScopedRange target_range(nvtx::Name::DecodeDFlashTarget, nvtx::Category::DFlash,
                                           static_cast<std::uint64_t>(width) * batch_size);
            TargetVerifyFrameView verify{
                .ids                     = verify_ids,
                .cache_positions         = target_positions,
                .rope_positions          = target_rope,
                .valid_columns           = valid_columns,
                .kv_table_rows           = text_rows,
                .state_source_slots      = state_sources,
                .state_destination_slots = state_destinations,
                .target_hidden           = target_hidden,
                .target_logits           = target_logits,
                .target_tokens           = target_tokens,
                .drafts                  = drafts,
                .current_extents         = extents,
                .candidate_ids           = frame.candidate_ids.data
                                               ? frame.candidate_ids.slice(2, 0, batch_size)
                                               : Tensor{},
                .proposal_q =
                    frame.proposal_q.data ? frame.proposal_q.slice(2, 0, batch_size) : Tensor{},
                .frontiers       = frontiers,
                .anchors         = anchors,
                .licensed_tokens = licensed_tokens,
                .licensed_counts = licensed_counts,
                .accepted_drafts = accepted,
                .selected_hidden = selected_hidden,
                .ple_slots = std::span<const std::int32_t>(state.host_ingress.ple_slots.data(),
                                                          static_cast<std::size_t>(batch_size)),
                .replay_records  = state.execution.replay_records,
                .sampling        = frame.sampling,
                .feature_sink    = &sink,
            };
            if (phase == SpeculativePhase::Forward) {
                if (std::getenv("NINFER_DFLASH_TRACE") != nullptr) {
                    std::fprintf(stderr, "[dfl] verify phase forward batch=%d k=%d lanes=%d,%d\n",
                                 batch_size, k, state.host_ingress.ple_slots[0],
                                 state.host_ingress.ple_slots[1]);
                    std::fflush(stderr);
                }
                target_verify_forward(state.execution, card, verify, target_envelope);
                if (timing) CUDA_CHECK(cudaEventRecord(timer.end, state.execution.device.stream));
                if (std::getenv("NINFER_DFLASH_TRACE") != nullptr) {
                    std::fprintf(stderr, "[dfl] verify forward done\n");
                    std::fflush(stderr);
                }
                return;
            }
            if (std::getenv("NINFER_DFLASH_TRACE") != nullptr) {
                std::fprintf(stderr, "[dfl] accept phase\n");
                std::fflush(stderr);
            }
            if (timing) CUDA_CHECK(cudaEventRecord(timer.accept_begin, state.execution.device.stream));
            target_accept(state.execution, state.continuation_hidden_store, verify);
            if (timing) {
                CUDA_CHECK(cudaEventRecord(timer.accept_end, state.execution.device.stream));
                CUDA_CHECK(cudaEventSynchronize(timer.accept_end));
                float draft_ms = 0.0F, verify_ms = 0.0F, accept_ms = 0.0F;
                // begin/split/end belong to this round's forward pass, which is upstream of this
                // accept on the same stream, so all three are already complete here.
                CUDA_CHECK(cudaEventElapsedTime(&draft_ms, timer.begin, timer.split));
                CUDA_CHECK(cudaEventElapsedTime(&verify_ms, timer.split, timer.end));
                CUDA_CHECK(cudaEventElapsedTime(&accept_ms, timer.accept_begin, timer.accept_end));
                std::fprintf(timer.file, "%llu,%u,%d,%d,%.3f,%.3f,%.3f\n",
                             static_cast<unsigned long long>(timer.round), k, batch_size,
                             static_cast<std::int32_t>(width), double(draft_ms), double(verify_ms),
                             double(accept_ms));
                std::fflush(timer.file);
                ++timer.round;
            }
            if (std::getenv("NINFER_DFLASH_TRACE") != nullptr) {
                std::fprintf(stderr, "[dfl] accept done\n");
                std::fflush(stderr);
            }
        }
        // NINFER_DFLASH2_DUMP=<path>: one line per (round, draft column) with what the draft
        // proposed at that column and what the target's own logits at the same position say, so
        // "the draft is wrong", "the selector is wrong" and "column j is worse than column 0" can be
        // told apart without guessing. Column i of the verify rectangle predicts the token draft i
        // has to guess, so the comparison at column i is exactly the acceptance test for that
        // column. Only a pass that is not being captured may read these back (the copies need the
        // stream to drain), so the switch is eager-only: a captured round runs exactly as before.
        if (!capturing && frame.candidate_ids.data != nullptr) {
            if (const char* dump = std::getenv("NINFER_DFLASH2_DUMP")) {
                constexpr std::int32_t kCandidates = 16;
                constexpr std::int32_t kTop        = 16;
                cudaStream_t s                     = state.execution.device.stream;
                static std::FILE* file = std::fopen(dump, "wb");
                if (file != nullptr) {
                    static std::atomic<std::uint64_t> steps{0};
                    const std::uint64_t step = steps.fetch_add(1);
                    // NINFER_DFLASH2_EDGE_DUMP=<path>: the selector's own scores, which no other dump
                    // carries - ids alone cannot price a branch, because "the right token was rank 3 in
                    // this column" says nothing about how many tokens a tree would have to verify to
                    // keep it. The k=7 window takes the lattice route, so all sixteen predecessor rows
                    // exist and a tree can be priced offline instead of guessed. One line per (column,
                    // predecessor rank): step, column, rank, target argmax, 16 ids, 16 scores. Empty
                    // when the probe is off or the round ran a short direct-route window.
                    const char* edge_path       = std::getenv("NINFER_DFLASH2_EDGE_DUMP");
                    static std::FILE* edge_file = edge_path != nullptr ? std::fopen(edge_path, "wb")
                                                                       : nullptr;
                    std::int32_t probe_columns  = 0;
                    std::int32_t probe_batch    = 0;
                    const float* probe_scores =
                        edge_file != nullptr
                            ? ops::candidate_selector_edge_probe(probe_columns, probe_batch)
                            : nullptr;
                    const auto vocab         = dimension(
                        state.execution.parameters.model.config().text.vocab_size);
                    const std::size_t words = static_cast<std::size_t>(vocab);
                    std::int32_t anchor     = 0;
                    core::copy_bytes(&anchor, frame.anchors.data, sizeof(anchor),
                                     core::TransferDirection::DeviceToHost, s);
                    // The target publishes BF16 logits; widen them here instead of reading the bit
                    // patterns as floats.
                    std::vector<std::uint16_t> raw(words);
                    std::vector<float> logits(words);
                    std::vector<std::int32_t> order(words);
                    std::vector<std::int32_t> candidates(static_cast<std::size_t>(kCandidates));
                    for (std::int32_t column = 0; column <= static_cast<std::int32_t>(k); ++column) {
                        // The last rectangle column has no proposal of its own (candidate_ids holds
                        // one per draft), but its target distribution is still the only way to see
                        // what a short extent does to a column the anchor does not border.
                        const bool has_candidates = column < static_cast<std::int32_t>(k);
                        const Tensor candidate_column = has_candidates
                                                            ? frame.candidate_ids.slice(1, column, 1)
                                                                  .slice(2, 0, 1)
                                                                  .view({kCandidates})
                                                            : Tensor{};
                        const Tensor target_column = frame.target_logits.slice(1, column, 1)
                                                         .slice(2, 0, 1)
                                                         .view({vocab});
                        std::int32_t selected = 0;
                        if (has_candidates) {
                            core::copy_bytes(candidates.data(), candidate_column.data,
                                             candidates.size() * sizeof(std::int32_t),
                                             core::TransferDirection::DeviceToHost, s);
                        } else {
                            std::fill(candidates.begin(), candidates.end(), -1);
                        }
                        // The selector's own pick for this column (not just the first candidate):
                        // selector_hit is pick == target argmax, topk_recall is argmax inside the
                        // column's sixteen candidates. They separate "the edge score ranks the right
                        // token badly" from "the right token was never a candidate".
                        core::copy_bytes(raw.data(), target_column.data,
                                         raw.size() * sizeof(std::uint16_t),
                                         core::TransferDirection::DeviceToHost, s);
                        if (column == 0) {
                            const Tensor draft_step =
                                frame.draft_tokens.slice(1, 0, 1).slice(0, 0, 1);
                            core::copy_bytes(&selected, draft_step.data, sizeof(selected),
                                             core::TransferDirection::DeviceToHost, s);
                        }
                        CUDA_CHECK(cudaStreamSynchronize(s));
                        for (std::size_t i = 0; i < words; ++i) {
                            const std::uint32_t bits = static_cast<std::uint32_t>(raw[i]) << 16U;
                            std::memcpy(&logits[i], &bits, sizeof(float));
                        }
                        std::iota(order.begin(), order.end(), 0);
                        std::nth_element(order.begin(), order.begin() + kTop, order.end(),
                                         [&](std::int32_t lhs, std::int32_t rhs) {
                                             return logits[static_cast<std::size_t>(lhs)] >
                                                    logits[static_cast<std::size_t>(rhs)];
                                         });
                        // `order` stays full length: shrinking it to kTop made every column after the
                        // first sort tokens 0..15 only, so argmax, top-16 and overlap described a
                        // sixteen-token vocabulary from column 1 on. The top-k copy is what is read.
                        std::vector<std::int32_t> top(order.begin(), order.begin() + kTop);
                        std::sort(top.begin(), top.end(),
                                  [&](std::int32_t lhs, std::int32_t rhs) {
                                      return logits[static_cast<std::size_t>(lhs)] >
                                             logits[static_cast<std::size_t>(rhs)];
                                  });
                        const std::int32_t argmax = top.front();
                        int overlap               = -1;
                        int hit                   = -1;
                        if (has_candidates) {
                            overlap = 0;
                            for (std::int32_t id : candidates) {
                                overlap += std::find(top.begin(), top.end(), id) != top.end()
                                               ? 1
                                               : 0;
                            }
                            hit = std::find(top.begin(), top.end(), candidates.front()) != top.end()
                                      ? 1
                                      : 0;
                        }
                        // What the target was actually fed in this column: the proposal it should
                        // carry, or something else (mask id, zero, the anchor again).
                        std::int32_t fed = 0;
                        core::copy_bytes(&fed, frame.verify_ids.slice(0, column, 1).slice(1, 0, 1).data,
                                         sizeof(fed), core::TransferDirection::DeviceToHost, s);
                        CUDA_CHECK(cudaStreamSynchronize(s));
                        std::fprintf(file, "%llu,%d,%d,%d,%d,%d,%d,%d,%d",
                                     static_cast<unsigned long long>(step), column, anchor,
                                     candidates.front(), selected, argmax, hit, overlap, fed);
                        for (std::int32_t id : candidates) { std::fprintf(file, ",%d", id); }
                        for (std::int32_t id : top) { std::fprintf(file, ",%d", id); }
                        // Where this column thinks it is, and what the target actually wrote there:
                        // the first four words of the column's hidden plus its norm, and how peaked
                        // the column's logits are. A column whose prediction is merely wrong still
                        // has a peaked distribution; a column whose write never landed reads flat.
                        std::int32_t cache_position = 0;
                        std::int32_t rope_position  = 0;
                        core::copy_bytes(
                            &cache_position,
                            frame.verify_positions.slice(0, column, 1).slice(1, 0, 1).data,
                            sizeof(cache_position), core::TransferDirection::DeviceToHost, s);
                        core::copy_bytes(
                            &rope_position,
                            frame.target_rope_positions.slice(0, column, 1).slice(1, 0, 1).data,
                            sizeof(rope_position), core::TransferDirection::DeviceToHost, s);
                        constexpr std::int32_t kHeadWords = 4;
                        std::vector<std::uint16_t> head(static_cast<std::size_t>(kHeadWords));
                        const auto* hidden_base = static_cast<const char*>(frame.target_hidden.data) +
                                                  static_cast<std::size_t>(column) *
                                                      frame.target_hidden.nb[1];
                        core::copy_bytes(head.data(), hidden_base,
                                         head.size() * sizeof(std::uint16_t),
                                         core::TransferDirection::DeviceToHost, s);
                        const auto head_value = [](std::uint16_t bits16) {
                            const std::uint32_t bits = static_cast<std::uint32_t>(bits16) << 16U;
                            float value              = 0.0F;
                            std::memcpy(&value, &bits, sizeof(float));
                            return value;
                        };
                        CUDA_CHECK(cudaStreamSynchronize(s));
                        double logit_mean = 0.0;
                        double logit_var  = 0.0;
                        for (const float value : logits) { logit_mean += double(value); }
                        logit_mean /= double(words);
                        for (const float value : logits) {
                            const double d = double(value) - logit_mean;
                            logit_var += d * d;
                        }
                        logit_var /= double(words);
                        const float top1 = logits[static_cast<std::size_t>(argmax)];
                        const float top2 = logits[static_cast<std::size_t>(top[1])];
                        std::fprintf(file, ",%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
                                     cache_position, rope_position, head_value(head[0]),
                                     head_value(head[1]), head_value(head[2]), head_value(head[3]),
                                     std::sqrt(logit_var), double(top1 - top2), double(top1));
                        std::int32_t pick = -1;
                        if (has_candidates) {
                            core::copy_bytes(&pick, frame.draft_tokens.slice(0, column, 1)
                                                       .slice(1, 0, 1)
                                                       .data,
                                             sizeof(pick), core::TransferDirection::DeviceToHost, s);
                        }
                        CUDA_CHECK(cudaStreamSynchronize(s));
                        const int selector_hit = has_candidates && pick == argmax ? 1 : 0;
                        const int topk_recall  = has_candidates &&
                                                        std::find(candidates.begin(), candidates.end(),
                                                                  argmax) != candidates.end()
                                                    ? 1
                                                    : 0;
                        // Where the draft's own top candidate lands in the target's FULL distribution,
                        // and its logit distance from the target's argmax. Rank 2-3 means the two heads
                        // are close and the argmax flips; a deep rank means the distributions disagree.
                        int unary_rank   = 0;
                        double logit_gap = 0.0;
                        if (has_candidates) {
                            const float unary_top = logits[static_cast<std::size_t>(candidates.front())];
                            for (const float value : logits) { unary_rank += value > unary_top ? 1 : 0; }
                            ++unary_rank;
                            logit_gap = double(logits[static_cast<std::size_t>(argmax)]) -
                                        double(unary_top);
                        }
                        std::fprintf(file, ",%d,%d,%d,%.6f", selector_hit, topk_recall, unary_rank,
                                     logit_gap);
                        if (probe_scores != nullptr && has_candidates && probe_batch == 1 &&
                            probe_columns == static_cast<std::int32_t>(k)) {
                            std::vector<float> edge(
                                static_cast<std::size_t>(kCandidates) * kCandidates);
                            core::copy_bytes(
                                edge.data(),
                                probe_scores + static_cast<std::size_t>(column) * kCandidates *
                                                   kCandidates,
                                edge.size() * sizeof(float), core::TransferDirection::DeviceToHost,
                                s);
                            CUDA_CHECK(cudaStreamSynchronize(s));
                            for (std::int32_t rank = 0; rank < kCandidates; ++rank) {
                                std::fprintf(edge_file, "%llu,%d,%d,%d",
                                             static_cast<unsigned long long>(step), column, rank,
                                             argmax);
                                for (std::int32_t id : candidates) {
                                    std::fprintf(edge_file, ",%d", id);
                                }
                                for (std::int32_t c = 0; c < kCandidates; ++c) {
                                    std::fprintf(
                                        edge_file, ",%.6f",
                                        double(edge[static_cast<std::size_t>(rank) * kCandidates +
                                                    c]));
                                }
                                std::fputc('\n', edge_file);
                            }
                            std::fflush(edge_file);
                        }
                        std::fputc('\n', file);
                    }
                    std::fflush(file);
                }
            }
        }
        core::copy_bytes(&state.host_egress, frame.egress.data,
                         sizeof(qwen3_5::DFlashDecodeEgress),
                         core::TransferDirection::DeviceToHost, state.execution.device.stream);
    };
}

} // namespace

DFlashFeatureSink dflash_feature_sink(PrefillContext& state,
                                      DFlashFeatureSink::PrefillConsumer consume_prefill) {
    return prefill_feature_sink_impl(state, std::move(consume_prefill));
}

void dflash_append_context(DFlashAppendContext& state, const Tensor& features,
                           const Tensor& positions, const Tensor& commit_counts,
                           const Tensor& lanes, const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    append_context_impl(state, features, positions, commit_counts, lanes, table_rows, envelope);
}

void dflash_append_context(PrefillContext& state, const Tensor& features, const Tensor& positions,
                           const Tensor& commit_counts, const Tensor& lanes,
                           const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    append_context_impl(state, features, positions, commit_counts, lanes, table_rows, envelope);
}

void capture_dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size,
                                 std::uint32_t k, DFlashEnvelopes envelopes,
                                 ops::CausalAttentionExecutionEnvelope target_envelope,
                                 DecodeGraphDefinition& definition, SpeculativePhase phase) {
    auto body =
        dflash_decode_batch_body(state, batch_size, k, envelopes, target_envelope, phase, true);
    capture_graph(state, definition, body);
}

void dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                         DFlashEnvelopes envelopes,
                         ops::CausalAttentionExecutionEnvelope target_envelope,
                         DecodeGraphExecutable* executable, SpeculativePhase phase) {
    auto body =
        dflash_decode_batch_body(state, batch_size, k, envelopes, target_envelope, phase, false);
    run_prepared(state, executable, body);
}

} // namespace ninfer::models::qwen3_5::execution
