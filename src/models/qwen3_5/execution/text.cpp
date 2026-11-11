#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/execution/text.h"
#include "models/qwen3_5/execution/attention.h"
#include "models/qwen3_5/execution/gdn.h"
#include "models/qwen3_5/execution/ffn.h"
#include "models/qwen3_5/execution/expert_stream.h"
#include "models/qwen3_5/execution/mtp.h"
#include "models/qwen3_5/execution/qsa.h"
#include "models/qwen3_5/execution/shape_class.h"
#include "models/qwen3_5/execution/workspace.h"

#include "core/nvtx.h"
#include "models/qwen3_5/execution/visual_scatter.h"
#include "models/qwen3_5/execution/vision.h"
#include "models/qwen3_5/program/vision_control.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/act_quant_a8.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/gdn_control.h"
#include "ninfer/ops/gated_residual.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/w4a8.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/mtp_pack.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/qsa.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/sparse_moe.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/softmax_attention.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::execution {
namespace {

// An indexed layer runs `qsa_mix` unless NINFER_QSA_FORCE_FULL asks for the dense mixer, which is
// the reference arm the byte and output gates compare against. Saying so once keeps a reference run
// from being mistaken for the real mixer.
void note_dense_indexed_arm() {
    static const bool told = [] {
        std::fprintf(stderr,
                     "[qsa] NINFER_QSA_FORCE_FULL: the indexed layers run the dense mixer (the "
                     "reference arm)\n");
        std::fflush(stderr);
        return true;
    }();
    (void)told;
}

void project(const Tensor& x, const LinearParameters& parameters, Tensor& out,
             WorkspaceArena& workspace, cudaStream_t stream) {
    ops::linear(x, parameters.weight, out, parameters.policy, workspace, stream);
}

// Diagnosis: the two QSA gates compare bytes produced by two different runs of the same prefix, so
// the only usable evidence is a file dump plus a stream hash. NINFER_QSA_KV_DUMP=dir writes the
// main K plane and the index plane prepare wrote; NINFER_QSA_LAYER_DUMP=dir writes the indexed
// layer's attention output as the dense arm and the sparse arm produce it. Both dump one layer,
// NINFER_QSA_DUMP_LAYER (default 3, the artifact's first indexed layer).
// NINFER_QSA_DUMP_LAYER selects one indexed layer by its compact index, or every layer with -1.
bool qsa_dump_wants_layer(int layer) {
    const char* value = std::getenv("NINFER_QSA_DUMP_LAYER");
    if (value == nullptr || *value == '\0') { return layer == 0; }
    const long selected = std::strtol(value, nullptr, 10);
    return selected < 0 || static_cast<int>(selected) == layer;
}

std::uint64_t fnv1a64(const void* data, std::size_t bytes) {
    const auto* bytes_in = static_cast<const std::uint8_t*>(data);
    std::uint64_t hash   = 1469598103934665603ULL;
    for (std::size_t i = 0; i < bytes; ++i) {
        hash ^= bytes_in[i];
        hash *= 1099511628211ULL;
    }
    return hash;
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

void qsa_dump_tensor(const char* variable, const char* name, const Tensor& value, int layer,
                     std::int32_t columns, cudaStream_t stream) {
    const char* directory = std::getenv(variable);
    if (directory == nullptr || *directory == '\0' || !qsa_dump_wants_layer(layer)) { return; }
    // A BF16 cache has no scale planes: an absent tensor is simply not dumped.
    if (value.data == nullptr) { return; }
    if (!value.is_contiguous()) {
        throw std::logic_error("QSA dump needs a contiguous device tensor");
    }
    std::vector<std::uint8_t> host(value.bytes());
    CUDA_CHECK(cudaMemcpyAsync(host.data(), value.data, host.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::string path(directory);
    path += "\\";
    path += name;
    path += "_layer";
    path += std::to_string(layer);
    // The column extent names the call: a prefill chunk and a decode step write different files, so
    // a comparison can pick the pass it means instead of racing the last write.
    if (columns > 0) {
        path += "_T";
        path += std::to_string(columns);
    }
    path += ".bin";
    if (std::FILE* file = std::fopen(path.c_str(), "wb")) {
        (void)std::fwrite(host.data(), 1, host.size(), file);
        (void)std::fclose(file);
    }
    std::fprintf(stderr, "[qsa] dump %s layer=%d bytes=%zu fnv=%016llx path=%s\n", name, layer,
                 host.size(), static_cast<unsigned long long>(fnv1a64(host.data(), host.size())),
                 path.c_str());
    std::fflush(stderr);
}

// NINFER_QSA_FORCE_FULL: the indexed layers run the dense mixer instead - its own KV append
// included - which is the reference arm of both the byte gate and the long-context A/B. Read once,
// because the startup plan follows the same switch.
bool qsa_dense_arm() {
    static const bool dense = std::getenv("NINFER_QSA_FORCE_FULL") != nullptr;
    return dense;
}

// NINFER_QSA_DENSE_READ: keep the read side dense (S2's own stage) - prepare still owns the cache,
// and the attention reads it back through the read-only cached Op instead of the indexer. Only a
// single-row traversal has a read-only dense form, so the batch paths stay on the sparse arm.
bool qsa_dense_read() {
    static const bool dense = std::getenv("NINFER_QSA_DENSE_READ") != nullptr;
    return dense;
}

void copy_first_row(const Tensor& source, Tensor& destination, cudaStream_t stream) {
    if (source.dtype != DType::I32 || source.nb[1] % 4 != 0 ||
        destination.ne[0] != source.ne[1] || !destination.is_contiguous()) {
        throw std::invalid_argument("QSA record base copy does not match the pass metadata");
    }
    CUDA_CHECK(cudaMemcpy2DAsync(destination.data, sizeof(std::int32_t), source.data,
                                 static_cast<std::size_t>(source.nb[1]), sizeof(std::int32_t),
                                 static_cast<std::size_t>(source.ne[1]),
                                 cudaMemcpyDeviceToDevice, stream));
}

void copy_i32(const std::int32_t* source, Tensor& destination, cudaStream_t stream) {
    if (source == nullptr || destination.dtype != DType::I32 || !destination.is_contiguous() ||
        destination.data == nullptr) {
        throw std::invalid_argument("copy_i32: invalid host source or I32 destination");
    }
    CUDA_CHECK(cudaMemcpyAsync(destination.data, source, destination.bytes(),
                               cudaMemcpyHostToDevice, stream));
}

void zero_i32(Tensor& destination, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(destination.data, 0, destination.bytes(), stream));
}

// One pass's QSA metadata, allocated from the same arena the layer workspace uses and bound for the
// whole pass. A pass with no indexed layer allocates nothing.
struct QsaPassBinding {
    workspace::QsaPassBindings storage;
    Tensor columns, base, offsets;
    Tensor source_slot, destination_slot;
    bool active = false;
};

void begin_qsa_pass(WorkspaceArena& work, const TextConfig& config, std::int32_t batch,
                    QsaPassBinding& out) {
    if (config.indexed_attention_layers == 0) { return; }
    out.storage          = workspace::qsa_pass_bindings(work);
    out.active           = true;
    out.columns          = out.storage.columns.slice(0, 0, batch);
    out.base             = out.storage.base_positions.slice(0, 0, batch);
    out.offsets          = out.storage.offsets.slice(0, 0, batch);
    out.source_slot      = out.storage.source_slot;
    out.destination_slot = out.storage.destination_slot;
}

void require_tensor_shape(const Tensor& t, DType dtype, std::initializer_list<std::int32_t> shape,
                          const char* label) {
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    int i = 0;
    for (const std::int32_t dim : shape) {
        if (t.ne[i] != dim) { throw std::invalid_argument(std::string(label) + " shape mismatch"); }
        ++i;
    }
    for (; i < 4; ++i) {
        if (t.ne[i] != 1) { throw std::invalid_argument(std::string(label) + " shape mismatch"); }
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

void require_tensor_window(const Tensor& t, DType dtype, std::int32_t rows, std::int32_t cols,
                           const char* label) {
    if (cols <= 0) { throw std::invalid_argument(std::string(label) + " cols must be positive"); }
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    if (t.ne[0] != rows || t.ne[1] < cols || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(label) + " shape mismatch");
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

Tensor matrix_window(Tensor& t, std::int32_t cols) {
    if (cols <= 0) { throw std::invalid_argument("matrix_window cols must be positive"); }
    if (t.ne[1] < cols || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument("matrix_window shape mismatch");
    }
    return t.slice(1, 0, cols);
}

class ScopedPositions {
public:
    ScopedPositions(const Tensor*& slot, const Tensor& positions) : slot_(slot) {
        slot_ = &positions;
    }

    ScopedPositions(const ScopedPositions&)            = delete;
    ScopedPositions& operator=(const ScopedPositions&) = delete;

    ~ScopedPositions() { slot_ = nullptr; }

private:
    const Tensor*& slot_;
};

class ScopedEnvelope {
public:
    ScopedEnvelope(const ops::CausalAttentionExecutionEnvelope*& slot,
                   const ops::CausalAttentionExecutionEnvelope& envelope)
        : slot_(slot) {
        slot_ = &envelope;
    }

    ScopedEnvelope(const ScopedEnvelope&)            = delete;
    ScopedEnvelope& operator=(const ScopedEnvelope&) = delete;

    ~ScopedEnvelope() { slot_ = nullptr; }

private:
    const ops::CausalAttentionExecutionEnvelope*& slot_;
};

template <class T>
class ScopedValue {
public:
    ScopedValue(T& slot, T value) : slot_(slot), previous_(slot) { slot_ = value; }

    ScopedValue(const ScopedValue&)            = delete;
    ScopedValue& operator=(const ScopedValue&) = delete;

    ~ScopedValue() { slot_ = previous_; }

private:
    T& slot_;
    T previous_;
};

// NINFER_PREFILL_PHASES splits one prefill chunk into the phases a draft tap can perturb: the
// embedding, the layer stack (where the feature copy runs), the final norm and sample, the MTP
// block, and the draft consumer. It syncs once per phase, which is enough to perturb a run that is
// already close to its memory ceiling, so it is a separate switch from NINFER_DFLASH_TIMING:
// NINFER_PREFILL_TIMING reports the same chunk total with no sync at all. Default off.
class ChunkProbe {
public:
    enum Phase : int {
        kEmbed = 0,
        kLayers,
        kFinal,
        kMtp,
        kDraftSink,
        kPhaseCount,
    };

    explicit ChunkProbe(cudaStream_t stream) : stream_(stream) {
        if (std::getenv("NINFER_PREFILL_PHASES") == nullptr) { return; }
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &status) == cudaSuccess &&
            status != cudaStreamCaptureStatusNone) {
            return;
        }
        for (cudaEvent_t& event : events_) {
            if (cudaEventCreate(&event) != cudaSuccess) { return; }
        }
        created_ = true;
        mark(kPhaseCount);
    }

    ChunkProbe(const ChunkProbe&)            = delete;
    ChunkProbe& operator=(const ChunkProbe&) = delete;

    ~ChunkProbe() {
        if (!created_) { return; }
        for (cudaEvent_t event : events_) { cudaEventDestroy(event); }
    }

    void mark(Phase phase) {
        if (!created_ || count_ >= static_cast<int>(events_.size())) { return; }
        if (cudaEventRecord(events_[static_cast<std::size_t>(count_)], stream_) != cudaSuccess) {
            return;
        }
        if (count_ > 0) { interval_phase_[static_cast<std::size_t>(count_)] = phase; }
        ++count_;
    }

    void report(std::int32_t tokens) {
        if (!created_ || count_ < 2) { return; }
        cudaEventSynchronize(events_[static_cast<std::size_t>(count_ - 1)]);
        std::array<double, kPhaseCount> sum{};
        double span = 0.0;
        for (int i = 0; i < count_ - 1; ++i) {
            float ms = 0.0F;
            if (cudaEventElapsedTime(&ms, events_[static_cast<std::size_t>(i)],
                                     events_[static_cast<std::size_t>(i + 1)]) != cudaSuccess) {
                continue;
            }
            const int phase = interval_phase_[static_cast<std::size_t>(i + 1)];
            if (phase >= 0 && phase < kPhaseCount) { sum[static_cast<std::size_t>(phase)] += ms; }
            span += static_cast<double>(ms);
        }
        std::fprintf(stderr,
                     "[df] target tokens=%d total=%.2f ms | embed=%.2f layers=%.2f final=%.2f "
                     "mtp=%.2f draft=%.2f\n",
                     tokens, span, sum[kEmbed], sum[kLayers], sum[kFinal], sum[kMtp],
                     sum[kDraftSink]);
        std::fflush(stderr);
    }

private:
    static constexpr std::size_t kMaxMarks = 8;

    cudaStream_t stream_ = nullptr;
    bool created_        = false;
    int count_           = 0;
    std::array<cudaEvent_t, kMaxMarks> events_{};
    std::array<int, kMaxMarks> interval_phase_{};
};

// NINFER_DFLASH_TIMING: the target-side feature copies are the other half of a draft's prefill
// cost - one 2D copy per tap layer per chunk. They are timed as one bracket around all of them
// (begin / end recorded once each) so the measurement adds a single sync per chunk instead of one
// per copy.
struct FeatureCopyTiming {
    cudaEvent_t begin_event = nullptr;
    cudaEvent_t end_event   = nullptr;
    int copies              = 0;
    bool active             = false;

    static bool enabled() {
        static const bool on = std::getenv("NINFER_DFLASH_TIMING") != nullptr;
        return on;
    }

    void start(cudaStream_t stream) {
        if (!enabled() || active) { return; }
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &status) == cudaSuccess &&
            status != cudaStreamCaptureStatusNone) {
            return;
        }
        if (begin_event == nullptr &&
            (cudaEventCreate(&begin_event) != cudaSuccess ||
             cudaEventCreate(&end_event) != cudaSuccess)) {
            return;
        }
        if (cudaEventRecord(begin_event, stream) == cudaSuccess) { active = true; }
    }

    void stop(cudaStream_t stream) {
        if (!active) { return; }
        if (cudaEventRecord(end_event, stream) == cudaSuccess) { ++copies; }
        active = false;
    }

    void report(const char* label) {
        if (!enabled() || begin_event == nullptr || copies == 0) { return; }
        cudaEventSynchronize(end_event);
        float ms = 0.0F;
        if (cudaEventElapsedTime(&ms, begin_event, end_event) == cudaSuccess) {
            std::fprintf(stderr, "[df] feature %s taps=%d copy_span=%.2f ms\n", label, copies,
                         static_cast<double>(ms));
            std::fflush(stderr);
        }
        copies = 0;
    }
};

FeatureCopyTiming& feature_copy_timing() {
    static thread_local FeatureCopyTiming timing;
    return timing;
}

} // namespace

void DFlashFeatureSink::begin(const Tensor& value) {
    const bool prefill = features != nullptr && positions != nullptr && batch_features == nullptr;
    const bool batch   = batch_features != nullptr && batch_lanes != nullptr &&
                       batch_valid_columns != nullptr && batch_width > 0 && batch_size > 0;
    if ((!prefill && !batch) || layers.empty()) {
        throw std::logic_error("DFlash feature sink is incomplete");
    }
    captured_mask = 0;
    active_tokens = batch ? batch_width * batch_size : value.ne[1];
    if (value.ne[1] != active_tokens) {
        throw std::logic_error("DFlash batch feature source has an invalid width");
    }
}

void DFlashFeatureSink::capture_layer(int layer, const Tensor& value, cudaStream_t stream) {
    const auto it = std::find(layers.begin(), layers.end(), layer);
    if (it == layers.end()) { return; }
    const std::size_t index = static_cast<std::size_t>(it - layers.begin());
    Tensor* destination     = batch_features != nullptr ? batch_features : features;
    if (layers.size() > 32 || active_tokens <= 0 || value.dtype != DType::BF16 ||
        destination == nullptr ||
        value.ne[0] * static_cast<std::int32_t>(layers.size()) != destination->ne[0] ||
        value.ne[1] != active_tokens) {
        throw std::logic_error("DFlash feature capture shape is invalid");
    }
    if (batch_features != nullptr) {
        Tensor source = value.view({value.ne[0], batch_width, batch_size});
        Tensor target =
            batch_features->slice(0, static_cast<std::int32_t>(index) * value.ne[0], value.ne[0]);
        ops::scatter_bf16_batch(source, *batch_lanes, *batch_valid_columns, target, stream);
        captured_mask |= 1U << index;
        return;
    }
    if (active_tokens > features->ne[1]) {
        throw std::logic_error("DFlash prefill feature capture exceeds its buffer");
    }
    const std::size_t element_bytes = dtype_size(DType::BF16);
    const std::size_t width_bytes   = static_cast<std::size_t>(value.ne[0]) * element_bytes;
    const std::size_t source_pitch  = static_cast<std::size_t>(value.nb[1]);
    const std::size_t target_pitch  = static_cast<std::size_t>(features->nb[1]);
    auto* target                    = static_cast<std::byte*>(features->data) + index * width_bytes;
    feature_copy_timing().start(stream);
    CUDA_CHECK(cudaMemcpy2DAsync(target, target_pitch, value.data, source_pitch, width_bytes,
                                 static_cast<std::size_t>(active_tokens), cudaMemcpyDeviceToDevice,
                                 stream));
    feature_copy_timing().stop(stream);
    captured_mask |= 1U << index;
}

void DFlashFeatureSink::capture_positions(const Tensor& source, cudaStream_t stream) {
    const std::uint32_t complete_mask = layers.size() == 32 ? ~0U : ((1U << layers.size()) - 1U);
    if (captured_mask != complete_mask) {
        throw std::logic_error("DFlash target call did not publish every feature layer");
    }
    if (batch_features != nullptr) {
        if (source.dtype != DType::I32 || source.ne[0] != batch_width ||
            source.ne[1] != batch_size) {
            throw std::logic_error("DFlash batch feature positions are invalid");
        }
        return;
    }
    if (active_tokens <= 0 || source.dtype != DType::I32 || source.ne[0] != active_tokens ||
        positions == nullptr || active_tokens > positions->ne[0]) {
        throw std::logic_error("DFlash feature positions are invalid");
    }
    CUDA_CHECK(cudaMemcpyAsync(positions->data, source.data,
                               static_cast<std::size_t>(active_tokens) * sizeof(std::int32_t),
                               cudaMemcpyDeviceToDevice, stream));
}

void DFlashFeatureSink::consume_prefill_chunk(std::int32_t tokens, bool rewrite_checkpoint) {
    if (!consume_prefill || tokens != active_tokens) {
        throw std::logic_error("DFlash prefill feature consumer is unavailable");
    }
    feature_copy_timing().report("prefill");
    Tensor feature_window  = features->slice(1, 0, tokens);
    Tensor position_window = positions->slice(0, 0, tokens);
    consume_prefill(feature_window, position_window, rewrite_checkpoint);
}

TextContext::TextContext(DeviceContext& ctx, const execution::Parameters& weights,
                         WorkspaceArena& work, qwen3_5::PagedKVCacheView kv,
                         LinearAttentionStatePool& state, qwen3_5::RoundState& io,
                         Tensor& prefill_hidden, std::uint32_t prefill_chunk,
                         std::uint32_t text_kv_base, qwen3_5::PagedKVCacheView mtp_kv,
                         const qwen3_5::PagedKVCache* batch_text_kv,
                         const qwen3_5::PagedKVCache* batch_mtp_kv)
    : ctx_(ctx), parameters_(weights), config_(weights.model.config().text), work_(work), kv_(kv),
      mtp_kv_(mtp_kv), state_(state), io_(io), prefill_hidden_(prefill_hidden),
      prefill_chunk_(prefill_chunk), text_kv_base_(text_kv_base), batch_text_kv_(batch_text_kv),
      batch_mtp_kv_(batch_mtp_kv) {
    if (prefill_chunk_ == 0 ||
        prefill_chunk_ > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("TextContext effective prefill chunk must fit positive int32");
    }
    if (mtp_enabled() && !io_.mtp_decode && !io_.mtp) {
        throw std::invalid_argument("MTP TextContext requires MTP round state");
    }
    set_linear_state_slots(0, 0);
    embed_      = &parameters_.text.token_embedding;
    final_norm_ = &parameters_.text.final_norm;
    final_hc_   = parameters_.text.final_hc ? &*parameters_.text.final_hc : nullptr;
    lm_head_    = &parameters_.text.output_head;
    mtp_        = parameters_.mtp ? &*parameters_.mtp : nullptr;
    if (mtp_enabled() && mtp_ == nullptr) {
        throw std::invalid_argument("MTP state requires selected MTP parameters");
    }
    if (parameters_.proposal) {
        const auto& p = *parameters_.proposal;
        set_proposal_head(
            &p.head, p.token_ids ? static_cast<const std::int32_t*>(p.token_ids->data) : nullptr,
            dimension(p.rows));
    }
}

TextContext::~TextContext() = default;

void TextContext::set_linear_state_slots(std::int32_t source_slot, std::int32_t destination_slot) {
    if (source_slot < 0 || source_slot >= state_.slot_count() || destination_slot < 0 ||
        destination_slot >= state_.slot_count()) {
        throw std::invalid_argument("TextContext Linear Attention slots are invalid");
    }
    linear_state_source_slot_      = source_slot;
    linear_state_destination_slot_ = destination_slot;
}

void TextContext::set_gdn_state_action(GdnStateAction action,
                                       const GdnReplayRecords* replay_records) {
    if ((action == GdnStateAction::RecordForReplay) != (replay_records != nullptr)) {
        throw std::invalid_argument("TextContext GDN state action has inconsistent records");
    }
    gdn_state_action_ = action;
    replay_records_   = replay_records;
}

void TextContext::mtp_forward_stem(const Tensor& ids, const Tensor& hidden,
                                   const Tensor* input_embeddings, Tensor& x, Tensor& ah) {
    cudaStream_t s     = ctx_.stream;
    const int T        = ids.ne[0] * ids.ne[1];
    Tensor flat_ids    = ids.view({T});
    Tensor flat_hidden = hidden.view({dimension(config_.hidden_size), T});

    auto roots = workspace::mtp_stem(work_, config_, T, input_embeddings == nullptr);
    Tensor emb;
    if (input_embeddings != nullptr) {
        if (input_embeddings->dtype != DType::BF16 ||
            input_embeddings->ne[0] != dimension(config_.hidden_size) ||
            input_embeddings->numel() !=
                static_cast<std::int64_t>(dimension(config_.hidden_size)) * T ||
            !input_embeddings->is_contiguous() || input_embeddings->data == nullptr) {
            throw std::invalid_argument("MTP input embeddings shape mismatch");
        }
        emb = input_embeddings->view({dimension(config_.hidden_size), T});
    } else {
        emb = roots.embedding;
        ops::embedding(flat_ids, *embed_, emb, s);
    }

    Tensor e = roots.normalized_embedding;
    Tensor h = roots.normalized_hidden;
    ops::rmsnorm(emb, mtp_->embedding_norm, config_.rms_norm_eps, true, e, s);
    ops::rmsnorm(flat_hidden, mtp_->hidden_norm, config_.rms_norm_eps, true, h, s);

    Tensor fc_in = roots.packed_input;
    ops::mtp_pack_fc_input(e, h, fc_in, s);

    x = roots.residual;
    project(fc_in, mtp_->input_projection, x, work_, s);

    ah = roots.attention_hidden;
    ops::rmsnorm(x, mtp_->input_norm, config_.rms_norm_eps, true, ah, s);
}

void TextContext::mtp_forward_tail(Tensor& x, const Tensor& ah, const Tensor& positions,
                                   const Tensor& rope_positions,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   Tensor& mtp_hidden) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    const auto projection = workspace::mtp_attention_projection(work_, config_, T);
    Tensor q              = projection.query.view({dimension(config_.attention->head_dim),
                                                   dimension(config_.attention->num_attention_heads), T});
    Tensor k              = projection.key.view({dimension(config_.attention->head_dim),
                                                 dimension(config_.attention->num_key_value_heads), T});
    Tensor gate           = projection.gate.view({dimension(config_.attention->head_dim),
                                                  dimension(config_.attention->num_attention_heads), T});
    Tensor v              = projection.value.view({dimension(config_.attention->head_dim),
                                                   dimension(config_.attention->num_key_value_heads), T});
    Tensor q_flat         = q.view({dimension(config_.attention->query_width()), T});
    Tensor gate_flat      = gate.view({dimension(config_.attention->query_width()), T});
    Tensor k_flat         = k.view({dimension(config_.attention->key_width()), T});
    Tensor v_flat         = v.view({dimension(config_.attention->key_width()), T});
    mtp_projection(ah, mtp_->projection, *config_.attention, q_flat, gate_flat, k_flat, v_flat,
                   work_, s);

    const auto results = workspace::mtp_attention_results(work_, config_, T);
    Tensor qn =
        results.normalized_query.view({dimension(config_.attention->head_dim),
                                       dimension(config_.attention->num_attention_heads), T});
    Tensor kn = results.normalized_key.view({dimension(config_.attention->head_dim),
                                             dimension(config_.attention->num_key_value_heads), T});
    ops::rmsnorm(q, mtp_->query_norm, config_.rms_norm_eps, true, qn, s);
    ops::rmsnorm(k, mtp_->key_norm, config_.rms_norm_eps, true, kn, s);
    Tensor rope_for_op = active_sequence_batch_ != 0 ? rope_positions.view({T}) : rope_positions;
    text_rope(rope_for_op, *config_.rope_parameters, qn, kn, ctx_.execution_view());

    Tensor a = results.attention.view({dimension(config_.attention->head_dim),
                                       dimension(config_.attention->num_attention_heads), T});
    if (active_sequence_batch_ != 0) {
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T ||
            active_backend_kv_table_rows_ == nullptr || active_valid_columns_ == nullptr) {
            throw std::logic_error("MTP sequence batch binding is incomplete");
        }
        Tensor q_batch        = qn.view({dimension(config_.attention->head_dim),
                                         dimension(config_.attention->num_attention_heads), width,
                                         active_sequence_batch_});
        Tensor k_batch        = kn.view({dimension(config_.attention->head_dim),
                                         dimension(config_.attention->num_key_value_heads), width,
                                         active_sequence_batch_});
        Tensor v_batch        = v.view({dimension(config_.attention->head_dim),
                                        dimension(config_.attention->num_key_value_heads), width,
                                        active_sequence_batch_});
        Tensor a_batch        = a.view({dimension(config_.attention->head_dim),
                                        dimension(config_.attention->num_attention_heads), width,
                                        active_sequence_batch_});
        Tensor position_batch = positions.view({width, active_sequence_batch_});
        ops::causal_softmax_attention(
            q_batch, k_batch, v_batch, position_batch, *active_valid_columns_,
            *active_backend_kv_table_rows_,
            {dimension(config_.attention->head_dim),
             dimension(config_.attention->num_attention_heads),
             dimension(config_.attention->num_key_value_heads)},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(config_.attention->head_dim))),
            batch_mtp_kv_->batch_layer_view(0), envelope, work_, a_batch, ctx_.execution_view());
    } else {
        ops::causal_softmax_attention(
            qn, kn, v, positions, Tensor{}, io_.backend_kv_table_row,
            {dimension(config_.attention->head_dim),
             dimension(config_.attention->num_attention_heads),
             dimension(config_.attention->num_key_value_heads)},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(config_.attention->head_dim))),
            batch_mtp_kv_->batch_layer_view(0), envelope, work_, a, ctx_.execution_view());
    }
    ops::sigmoid_mul(gate, a, s);

    const auto post = workspace::mtp_post_attention(work_, config_, T);
    Tensor o        = post.output;
    project(a.view({dimension(config_.attention->query_width()), T}), mtp_->output, o, work_, s);
    ops::residual_add(o, x, s);

    Tensor mh = post.post_mixer_hidden;
    ops::rmsnorm(x, mtp_->post_attention_norm, config_.rms_norm_eps, true, mh, s);

    {
        auto post_mixer_scope = work_.scope();
        ffn(mh, mtp_->ffn, x, {}, work_, ctx_.execution_view(), true);
    }

    Tensor flat_mtp_hidden = mtp_hidden.view({dimension(config_.hidden_size), T});
    ops::rmsnorm(x, mtp_->final_norm, config_.rms_norm_eps, true, flat_mtp_hidden, s);
}

void TextContext::mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                                   const Tensor& rope_positions,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   Tensor& mtp_hidden, const Tensor* input_embeddings) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    nvtx::ScopedRange forward_range(nvtx::Name::MtpForward, nvtx::Category::Mtp,
                                    static_cast<std::uint64_t>(ids.numel()));
    auto scratch_scope = work_.scope();
    Tensor x;
    Tensor ah;
    mtp_forward_stem(ids, hidden, input_embeddings, x, ah);
    mtp_forward_tail(x, ah, positions, rope_positions, envelope, mtp_hidden);
}

void TextContext::mtp_prefill_chunk(const Tensor& ids, const Tensor& hidden,
                                    const Tensor* input_embeddings, const Tensor& positions,
                                    const Tensor& rope_positions,
                                    ops::CausalAttentionExecutionEnvelope envelope,
                                    bool final_chunk, Tensor* final_hidden, Tensor* logits,
                                    Tensor* draft_token) {
    if (!mtp_kv_.valid()) { throw std::runtime_error("MTP prefill is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP prefill chunk T must be in [1,prefill_chunk]");
    }
    nvtx::ScopedRange mtp_prefill_range(nvtx::Name::PrefillMtpChunk, nvtx::Category::Mtp,
                                        static_cast<std::uint64_t>(T));
    require_tensor_shape(ids, DType::I32, {T}, "MTP prefill ids");
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), T},
                         "MTP prefill hidden");
    require_tensor_shape(positions, DType::I32, {T}, "MTP prefill positions");
    if (rope_positions.dtype != DType::I32 || rope_positions.ne[0] != T ||
        (rope_positions.ne[1] != 1 && rope_positions.ne[1] != 3) || rope_positions.ne[2] != 1 ||
        rope_positions.ne[3] != 1 || !rope_positions.is_contiguous() ||
        rope_positions.data == nullptr) {
        throw std::invalid_argument("MTP prefill rope positions must be [T] or [T,3]");
    }
    if (final_chunk) {
        if (final_hidden == nullptr || logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP final prefill outputs are required");
        }
        require_tensor_shape(*final_hidden, DType::BF16, {dimension(config_.hidden_size), 1},
                             "MTP final prefill hidden");
        require_tensor_shape(*logits, DType::BF16, {dimension(config_.vocab_size), 1},
                             "MTP final prefill logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP final prefill draft token");
    }

    cudaStream_t s     = ctx_.stream;
    auto scratch_scope = work_.scope();
    Tensor x_last;
    Tensor ah_last;
    if (final_chunk) {
        x_last  = work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
        ah_last = work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
    }

    {
        auto bulk_scope = work_.scope();
        Tensor x;
        Tensor ah;
        mtp_forward_stem(ids, hidden, input_embeddings, x, ah);

        Tensor k_flat = work_.alloc(DType::BF16, {dimension(config_.attention->key_width()), T});
        Tensor v_flat = work_.alloc(DType::BF16, {dimension(config_.attention->key_width()), T});
        mtp_kv_projection(ah, mtp_->projection, *config_.attention, k_flat, v_flat, work_, s);
        Tensor k = k_flat.view({dimension(config_.attention->head_dim),
                                dimension(config_.attention->num_key_value_heads), T});
        Tensor v = v_flat.view({dimension(config_.attention->head_dim),
                                dimension(config_.attention->num_key_value_heads), T});
        Tensor kn =
            work_.alloc(DType::BF16, {dimension(config_.attention->head_dim),
                                      dimension(config_.attention->num_key_value_heads), T});
        ops::rmsnorm(k, mtp_->key_norm, config_.rms_norm_eps, true, kn, s);
        text_rope(rope_positions, *config_.rope_parameters, kn, ctx_.execution_view());
        ops::kv_cache_append(kn, v, positions, mtp_kv_.layer_view(0), s);

        if (final_chunk) {
            const std::size_t column_bytes =
                static_cast<std::size_t>(dimension(config_.hidden_size)) * dtype_size(DType::BF16);
            const auto* x_src = static_cast<const unsigned char*>(x.data) +
                                static_cast<std::size_t>(T - 1) * column_bytes;
            const auto* ah_src = static_cast<const unsigned char*>(ah.data) +
                                 static_cast<std::size_t>(T - 1) * column_bytes;
            // port (sm_89): the two slices are independent and share a stream, so one batch
            // submission replaces two. Both are sub-megabyte, which is where the submission cost
            // dominates the transfer.
            void* const       destinations[2] = {x_last.data, ah_last.data};
            const void* const sources[2]      = {x_src, ah_src};
            const std::size_t sizes[2]        = {column_bytes, column_bytes};
            cudaMemcpyAttributes attributes{};
            attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
            attributes.flags          = 0;
            std::size_t attribute_index = 0;
            const auto submitted        = cudaMemcpyBatchAsync(destinations, sources, sizes, 2,
                                                               &attributes, &attribute_index, 1, s);
            if (submitted != cudaSuccess) {
                CUDA_CHECK(
                    cudaMemcpyAsync(x_last.data, x_src, column_bytes, cudaMemcpyDeviceToDevice, s));
                CUDA_CHECK(
                    cudaMemcpyAsync(ah_last.data, ah_src, column_bytes, cudaMemcpyDeviceToDevice, s));
            }
        }
    }

    if (final_chunk) {
        Tensor q_flat = work_.alloc(DType::BF16, {dimension(config_.attention->query_width()), 1});
        Tensor gate_flat =
            work_.alloc(DType::BF16, {dimension(config_.attention->query_width()), 1});
        mtp_query_gate_projection(ah_last, mtp_->projection, *config_.attention, q_flat, gate_flat,
                                  work_, s);
        Tensor q    = q_flat.view({dimension(config_.attention->head_dim),
                                   dimension(config_.attention->num_attention_heads), 1});
        Tensor gate = gate_flat.view({dimension(config_.attention->head_dim),
                                      dimension(config_.attention->num_attention_heads), 1});
        Tensor qn =
            work_.alloc(DType::BF16, {dimension(config_.attention->head_dim),
                                      dimension(config_.attention->num_attention_heads), 1});
        ops::rmsnorm(q, mtp_->query_norm, config_.rms_norm_eps, true, qn, s);
        Tensor last_position = positions.slice(0, T - 1, 1);
        Tensor last_rope_position;
        if (rope_positions.ne[1] == 1) {
            last_rope_position = rope_positions.slice(0, T - 1, 1);
        } else {
            last_rope_position = work_.alloc(DType::I32, {1, 3});
            // port (sm_89): the three axes are one column of the [3, T] position tensor, so this is
            // a single pitched copy instead of three four-byte submissions.
            const auto* axes_base =
                static_cast<const std::int32_t*>(rope_positions.data) + (T - 1);
            CUDA_CHECK(cudaMemcpy2DAsync(last_rope_position.data, sizeof(std::int32_t), axes_base,
                                         static_cast<std::size_t>(T) * sizeof(std::int32_t),
                                         sizeof(std::int32_t), 3, cudaMemcpyDeviceToDevice, s));
        }
        text_rope(last_rope_position, *config_.rope_parameters, qn, ctx_.execution_view());

        Tensor a = work_.alloc(DType::BF16, {dimension(config_.attention->head_dim),
                                             dimension(config_.attention->num_attention_heads), 1});
        ops::causal_softmax_attention_cached(
            qn, last_position,
            {dimension(config_.attention->head_dim),
             dimension(config_.attention->num_attention_heads),
             dimension(config_.attention->num_key_value_heads)},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(config_.attention->head_dim))),
            mtp_kv_.layer_view(0), envelope, work_, a, ctx_.execution_view());
    ops::sigmoid_mul(gate, a, s);

        Tensor o = work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
        project(a.view({dimension(config_.attention->query_width()), 1}), mtp_->output, o, work_,
                s);
        ops::residual_add(o, x_last, s);

        Tensor mh = work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
        ops::rmsnorm(x_last, mtp_->post_attention_norm, config_.rms_norm_eps, true, mh, s);
        {
            auto post_mixer_scope = work_.scope();
            ffn(mh, mtp_->ffn, x_last, {}, work_, ctx_.execution_view(), true);
        }
        ops::rmsnorm(x_last, mtp_->final_norm, config_.rms_norm_eps, true, *final_hidden, s);
        proposal_argmax(*final_hidden, *logits, *draft_token);
    }
}

void TextContext::proposal_argmax(const Tensor& hidden, Tensor& logits, Tensor& proposal_tokens,
                                  ops::SamplingMask mask) {
    auto proposal_scope = work_.scope();
    const int T         = hidden.ne[1];
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), T},
                         "proposal hidden");
    require_tensor_shape(proposal_tokens, DType::I32, {T}, "proposal tokens");
    require_tensor_window(logits, DType::BF16, dimension(config_.vocab_size), T, "proposal logits");
    nvtx::ScopedRange proposal_range(nvtx::Name::MtpProposal, nvtx::Category::Mtp,
                                     static_cast<std::uint64_t>(T));
    const auto token_domain = dimension(parameters_.model.resources().public_token_count);
    if (mask.words != nullptr) {
        // Constrained request: the proposal must respect the same bitmap the verifier applies, or
        // every draft is rejected and the speculative round only costs time. The sampling op
        // excludes forbidden rows before its greedy argmax, and it is fed one config per proposal
        // position so each position uses its own bitmap. The remapped proposal head is not used
        // here because its ids live in a reduced domain and the bitmap is over the public
        // vocabulary.
        Tensor output_logits = matrix_window(logits, T);
        project(hidden, *lm_head_, output_logits, work_, ctx_.stream);
        std::vector<ops::SamplingConfig> configs(static_cast<std::size_t>(T));
        for (int t = 0; t < T; ++t) {
            configs[static_cast<std::size_t>(t)].temperature = 0.0f; // greedy
            configs[static_cast<std::size_t>(t)].mask =
                ops::SamplingMask{mask.words + static_cast<std::ptrdiff_t>(t) * mask.stride,
                                  mask.stride};
        }
        const auto config_bytes = sizeof(ops::SamplingConfig) * static_cast<std::size_t>(T);
        auto config_span        = work_.alloc_bytes(config_bytes);
        CUDA_CHECK(cudaMemcpyAsync(config_span.data, configs.data(), config_bytes,
                                   cudaMemcpyHostToDevice, ctx_.stream));
        std::vector<std::int32_t> positions(static_cast<std::size_t>(T));
        for (int t = 0; t < T; ++t) { positions[static_cast<std::size_t>(t)] = t; }
        Tensor position_tensor = work_.alloc(DType::I32, {T});
        CUDA_CHECK(cudaMemcpyAsync(position_tensor.data, positions.data(),
                                   sizeof(std::int32_t) * static_cast<std::size_t>(T),
                                   cudaMemcpyHostToDevice, ctx_.stream));
        ops::sample(output_logits, proposal_tokens, token_domain,
                    static_cast<const ops::SamplingConfig*>(config_span.data), position_tensor,
                    ops::kSamplePurposeDecode, work_, ctx_.stream);
        return;
    }
    if (proposal_head_ != nullptr) {
        Tensor proposal_logits = work_.alloc(DType::BF16, {proposal_head_n_, T});
        project(hidden, *proposal_head_, proposal_logits, work_, ctx_.stream);
        ops::argmax(proposal_logits, proposal_tokens,
                    proposal_head_ids_
                        ? proposal_head_n_
                        : dimension(parameters_.model.resources().public_token_count),
                    ctx_.stream);
        if (proposal_head_ids_ != nullptr) {
            ops::proposal_remap_token_ids(proposal_tokens, proposal_head_ids_, proposal_head_n_,
                                          ctx_.stream);
        }
    } else {
        Tensor output_logits = matrix_window(logits, T);
        project(hidden, mtp_->output_head, output_logits, work_, ctx_.stream);
        ops::argmax(output_logits, proposal_tokens,
                    dimension(parameters_.model.resources().public_token_count), ctx_.stream);
    }
}

void TextContext::mtp_forward_batch(const Tensor& ids, const Tensor& hidden,
                                    const Tensor& positions,
                                    ops::CausalAttentionExecutionEnvelope envelope,
                                    Tensor& mtp_hidden, int logits_column, Tensor* logits,
                                    Tensor* draft_token, const Tensor* explicit_rope_positions,
                                    const Tensor* input_embeddings) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP batch T must be in [1,prefill_chunk]");
    }
    require_tensor_shape(ids, DType::I32, {T}, "MTP ids");
    require_tensor_shape(positions, DType::I32, {T}, "MTP positions");
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), T}, "MTP hidden");
    require_tensor_shape(mtp_hidden, DType::BF16, {dimension(config_.hidden_size), T},
                         "MTP output hidden");
    if (logits_column >= T) { throw std::invalid_argument("MTP logits column out of range"); }
    if (logits_column >= 0) {
        if (logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP logits and draft_token outputs are required");
        }
        require_tensor_shape(*logits, DType::BF16, {dimension(config_.vocab_size), 1},
                             "MTP logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP draft token");
    }

    auto position_scope = work_.scope();
    Tensor generated_rope_positions;
    const Tensor* rope_positions = explicit_rope_positions;
    if (rope_positions == nullptr) {
        generated_rope_positions = work_.alloc(DType::I32, {T});
        ops::offset_i32_positions(positions, io_.rope_delta, generated_rope_positions, ctx_.stream);
        rope_positions = &generated_rope_positions;
    } else if (rope_positions->dtype != DType::I32 || rope_positions->ne[0] != T ||
               (rope_positions->ne[1] != 1 && rope_positions->ne[1] != 3) ||
               rope_positions->ne[2] != 1 || rope_positions->ne[3] != 1 ||
               !rope_positions->is_contiguous() || rope_positions->data == nullptr) {
        throw std::invalid_argument("MTP explicit rope positions must be [T] or [T,3]");
    }
    mtp_forward_core(ids, hidden, positions, *rope_positions, envelope, mtp_hidden,
                     input_embeddings);

    if (logits_column >= 0) {
        auto logits_scope = work_.scope();
        Tensor col        = mtp_hidden.slice(1, logits_column, 1);
        proposal_argmax(col, *logits, *draft_token);
    }
}

void TextContext::mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                                      const Tensor& position,
                                      ops::CausalAttentionExecutionEnvelope envelope,
                                      Tensor& mtp_hidden, Tensor& logits, Tensor& draft_token,
                                      ops::SamplingMask mask) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    require_tensor_shape(token, DType::I32, {1}, "MTP AR token");
    require_tensor_shape(position, DType::I32, {1}, "MTP AR position");
    require_tensor_shape(previous_hidden, DType::BF16, {dimension(config_.hidden_size), 1},
                         "MTP AR previous hidden");
    require_tensor_shape(mtp_hidden, DType::BF16, {dimension(config_.hidden_size), 1},
                         "MTP AR output hidden");
    require_tensor_shape(logits, DType::BF16, {dimension(config_.vocab_size), 1}, "MTP AR logits");
    require_tensor_shape(draft_token, DType::I32, {1}, "MTP AR draft token");

    auto position_scope  = work_.scope();
    Tensor rope_position = work_.alloc(DType::I32, {1});
    ops::offset_i32_positions(position, io_.rope_delta, rope_position, ctx_.stream);
    mtp_forward_core(token, previous_hidden, position, rope_position, envelope, mtp_hidden,
                     nullptr);
    auto logits_scope = work_.scope();
    proposal_argmax(mtp_hidden, logits, draft_token, mask);
}

void TextContext::ordinary_decode_batch(const Tensor& ids, const Tensor& cache_positions,
                                        const Tensor& rope_positions, const Tensor& kv_table_rows,
                                        const Tensor& linear_state_source_slots,
                                        const Tensor& linear_state_destination_slots,
                                        ops::CausalAttentionExecutionEnvelope envelope,
                                        Tensor& hidden, Tensor& logits) {
    const std::int32_t batch = ids.ne[0];
    if (batch <= 0 || batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("ordinary decode batch size must be in [1,8]");
    }
    require_tensor_shape(ids, DType::I32, {batch}, "ordinary decode ids");
    require_tensor_shape(cache_positions, DType::I32, {batch}, "ordinary decode cache positions");
    require_tensor_shape(rope_positions, DType::I32, {batch}, "ordinary decode RoPE positions");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "ordinary decode KV rows");
    require_tensor_shape(linear_state_source_slots, DType::I32, {batch},
                         "ordinary decode Linear Attention source slots");
    require_tensor_shape(linear_state_destination_slots, DType::I32, {batch},
                         "ordinary decode Linear Attention destination slots");
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), batch},
                         "ordinary decode hidden");
    require_tensor_shape(logits, DType::BF16, {dimension(config_.vocab_size), batch},
                         "ordinary decode logits");

    cudaStream_t stream = ctx_.stream;
    work_.reset();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions);
        ScopedEnvelope envelope_binding(active_causal_attention_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_rows);
        ScopedValue<const Tensor*> source_binding(active_linear_state_source_slots_,
                                                  &linear_state_source_slots);
        ScopedValue<const Tensor*> destination_binding(active_linear_state_destination_slots_,
                                                       &linear_state_destination_slots);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, 1);

        Tensor x = work_.alloc(DType::BF16, {dimension(config_.hidden_size), batch});
        ops::embedding(ids, *embed_, x, stream);
        // A decode column is one token, so the pass's live-column count is the pool's row of ones
        // and the record base is the row's own position; both stay device tensors, which keeps the
        // captured traversal replayable without a host read.
        QsaPassBinding qsa;
        begin_qsa_pass(work_, config_, batch, qsa);
        if (qsa.active) {
            const Tensor ones = batch_text_kv_->qsa_decode_columns(batch);
            CUDA_CHECK(cudaMemcpyAsync(qsa.columns.data, ones.data, qsa.columns.bytes(),
                                       cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(qsa.base.data, cache_positions.data, qsa.base.bytes(),
                                       cudaMemcpyDeviceToDevice, stream));
            zero_i32(qsa.offsets, stream);
        }
        ScopedValue<const Tensor*> qsa_columns_binding(active_qsa_columns_,
                                                       qsa.active ? &qsa.columns : nullptr);
        ScopedValue<const Tensor*> qsa_base_binding(active_qsa_base_positions_,
                                                    qsa.active ? &qsa.base : nullptr);
        ScopedValue<const Tensor*> qsa_offsets_binding(active_qsa_offsets_,
                                                       qsa.active ? &qsa.offsets : nullptr);
        NullTap tap;
        run_layers(x, Phase::Verify, tap);
        final_hidden(x, hidden, batch);
        // Q67: zero-weight scratch rows for the row-independent output projection. The class width
        // is the compiled shape; the extra rows read zero hidden and their logits are never sampled.
        // This is deliberately the first absorption site, not the whole batch: attention and the
        // recurrent state still run at the real width until the batch container owns scratch KV and
        // state rows (S1). Off by default, and real rows are copied, so values are unchanged.
        static const bool pad_class = [] {
            const char* value = std::getenv("NINFER_PAD_CLASS");
            return value != nullptr && *value != '\0' && std::strtol(value, nullptr, 10) != 0;
        }();
        const std::int32_t width = pad_class ? decode_class_for(batch) : batch;
        if (width > batch) {
            Tensor padded_hidden = work_.alloc(DType::BF16, {dimension(config_.hidden_size), width});
            CUDA_CHECK(cudaMemsetAsync(padded_hidden.data, 0, padded_hidden.bytes(), stream));
            CUDA_CHECK(cudaMemcpy2DAsync(padded_hidden.data, padded_hidden.nb[1], hidden.data,
                                         hidden.nb[1], hidden.bytes() / static_cast<std::size_t>(batch),
                                         static_cast<std::size_t>(batch), cudaMemcpyDeviceToDevice,
                                         stream));
            Tensor padded_logits = work_.alloc(DType::BF16, {dimension(config_.vocab_size), width});
            project(padded_hidden, *lm_head_, padded_logits, work_, stream);
            CUDA_CHECK(cudaMemcpy2DAsync(logits.data, logits.nb[1], padded_logits.data,
                                         padded_logits.nb[1],
                                         logits.bytes() / static_cast<std::size_t>(batch),
                                         static_cast<std::size_t>(batch), cudaMemcpyDeviceToDevice,
                                         stream));
        } else {
            project(hidden, *lm_head_, logits, work_, stream);
        }
    }
    work_.reset();
}

template <class Tap>
void TextContext::target_verify_batch_impl(const Tensor& ids, const Tensor& cache_positions,
                                           const Tensor& rope_positions,
                                           const Tensor& valid_columns, const Tensor& kv_table_rows,
                                           const Tensor& linear_state_source_slots,
                                           ops::CausalAttentionExecutionEnvelope envelope,
                                           Tensor& hidden, Tensor& logits, Tensor& target_tokens,
                                           Tap& tap) {
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kDFlashDecodeMaximumWidth) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("target verify batch shape is outside the supported domain");
    }
    const std::int32_t columns = width * batch;
    require_tensor_shape(ids, DType::I32, {width, batch}, "target verify batch ids");
    require_tensor_shape(cache_positions, DType::I32, {width, batch},
                         "target verify batch cache positions");
    require_tensor_shape(rope_positions, DType::I32, {width, batch},
                         "target verify batch RoPE positions");
    require_tensor_shape(valid_columns, DType::I32, {batch}, "target verify batch valid columns");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "target verify batch KV rows");
    require_tensor_shape(linear_state_source_slots, DType::I32, {batch},
                         "target verify batch Linear Attention slots");
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), width, batch},
                         "target verify batch hidden");
    require_tensor_shape(logits, DType::BF16, {dimension(config_.vocab_size), width, batch},
                         "target verify batch logits");
    require_tensor_shape(target_tokens, DType::I32, {width, batch}, "target verify batch tokens");

    cudaStream_t stream = ctx_.stream;
    work_.reset();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions);
        ScopedEnvelope envelope_binding(active_causal_attention_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_rows);
        ScopedValue<const Tensor*> state_binding(active_linear_state_source_slots_,
                                                 &linear_state_source_slots);
        ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);

        Tensor x        = work_.alloc(DType::BF16, {dimension(config_.hidden_size), columns});
        Tensor flat_ids = ids.view({columns});
        ops::embedding(flat_ids, *embed_, x, stream);
        // The verify rectangle's metadata is the pass's own: its live columns are `valid_columns`,
        // its record base is each row's first position, and the record offset is zero.
        QsaPassBinding qsa;
        begin_qsa_pass(work_, config_, batch, qsa);
        if (qsa.active) {
            copy_first_row(cache_positions, qsa.base, stream);
            zero_i32(qsa.offsets, stream);
        }
        ScopedValue<const Tensor*> qsa_columns_binding(
            active_qsa_columns_, qsa.active ? &valid_columns : nullptr);
        ScopedValue<const Tensor*> qsa_base_binding(active_qsa_base_positions_,
                                                    qsa.active ? &qsa.base : nullptr);
        ScopedValue<const Tensor*> qsa_offsets_binding(active_qsa_offsets_,
                                                       qsa.active ? &qsa.offsets : nullptr);
        if constexpr (Tap::enabled) { tap.begin(x); }
        run_layers(x, Phase::Verify, tap);
        if constexpr (requires { tap.capture_positions(cache_positions, stream); }) {
            tap.capture_positions(cache_positions, stream);
        }
        Tensor flat_hidden = hidden.view({dimension(config_.hidden_size), columns});
        Tensor flat_logits = logits.view({dimension(config_.vocab_size), columns});
        Tensor flat_tokens = target_tokens.view({columns});
        final_hidden(x, flat_hidden, columns);
        project(flat_hidden, *lm_head_, flat_logits, work_, stream);
        ops::argmax(flat_logits, flat_tokens,
                    dimension(parameters_.model.resources().public_token_count), stream);
    }
    work_.reset();
}

void TextContext::target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                                      const Tensor& rope_positions, const Tensor& valid_columns,
                                      const Tensor& kv_table_rows,
                                      const Tensor& linear_state_source_slots,
                                      ops::CausalAttentionExecutionEnvelope envelope,
                                      Tensor& hidden, Tensor& logits, Tensor& target_tokens) {
    NullTap tap;
    target_verify_batch_impl(ids, cache_positions, rope_positions, valid_columns, kv_table_rows,
                             linear_state_source_slots, envelope, hidden, logits, target_tokens,
                             tap);
}

void TextContext::target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                                      const Tensor& rope_positions, const Tensor& valid_columns,
                                      const Tensor& kv_table_rows,
                                      const Tensor& linear_state_source_slots,
                                      ops::CausalAttentionExecutionEnvelope envelope,
                                      Tensor& hidden, Tensor& logits, Tensor& target_tokens,
                                      DFlashFeatureSink& sink) {
    target_verify_batch_impl(ids, cache_positions, rope_positions, valid_columns, kv_table_rows,
                             linear_state_source_slots, envelope, hidden, logits, target_tokens,
                             sink);
}

void TextContext::prefill_batch(std::span<const std::int32_t> ids,
                                std::span<const std::int32_t> positions,
                                std::span<const std::int32_t> rope_positions,
                                std::span<const std::int32_t> valid_columns,
                                std::span<const std::int32_t> kv_table_rows,
                                std::span<const std::int32_t> state_source_slots,
                                std::span<const std::int32_t> state_destination_slots,
                                std::int32_t width, std::int32_t batch,
                                ops::CausalAttentionExecutionEnvelope envelope,
                                std::span<std::int32_t> next_tokens,
                                std::span<Tensor* const> tail_hidden_rows) {
    if (width <= 0 || batch <= 0 || batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("prefill batch shape is outside the supported domain");
    }
    const std::int32_t columns = width * batch;
    if (static_cast<std::uint32_t>(columns) > prefill_chunk_) {
        throw std::invalid_argument("prefill batch exceeds the chunk the workspace was planned for");
    }
    const std::size_t lane_count = static_cast<std::size_t>(batch);
    if (ids.size() != static_cast<std::size_t>(columns) || positions.size() != ids.size() ||
        rope_positions.size() != ids.size() || valid_columns.size() != lane_count ||
        kv_table_rows.size() != lane_count || state_source_slots.size() != lane_count ||
        state_destination_slots.size() != lane_count) {
        throw std::invalid_argument("prefill batch host spans do not match the pass shape");
    }
    if (next_tokens.size() != lane_count || tail_hidden_rows.size() != lane_count) {
        throw std::invalid_argument("prefill batch token output does not match the row count");
    }

    cudaStream_t stream = ctx_.stream;
    work_.reset();
    Tensor d_ids       = work_.alloc(DType::I32, {width, batch});
    Tensor d_positions = work_.alloc(DType::I32, {width, batch});
    Tensor d_rope      = work_.alloc(DType::I32, {width, batch});
    Tensor d_valid     = work_.alloc(DType::I32, {batch});
    Tensor d_kv_rows   = work_.alloc(DType::I32, {batch});
    Tensor d_src       = work_.alloc(DType::I32, {batch});
    Tensor d_dst       = work_.alloc(DType::I32, {batch});
    copy_i32(ids.data(), d_ids, stream);
    copy_i32(positions.data(), d_positions, stream);
    copy_i32(rope_positions.data(), d_rope, stream);
    copy_i32(valid_columns.data(), d_valid, stream);
    copy_i32(kv_table_rows.data(), d_kv_rows, stream);
    copy_i32(state_source_slots.data(), d_src, stream);
    copy_i32(state_destination_slots.data(), d_dst, stream);

    ScopedPositions cache_binding(active_cache_positions_, d_positions);
    ScopedPositions rope_binding(active_rope_positions_, d_rope);
    ScopedEnvelope envelope_binding(active_causal_attention_envelope_, envelope);
    ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &d_kv_rows);
    ScopedValue<const Tensor*> state_binding(active_linear_state_source_slots_, &d_src);
    ScopedValue<const Tensor*> dest_binding(active_linear_state_destination_slots_, &d_dst);
    ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &d_valid);
    ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
    ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);
    // The GDN conv split below slices each row's columns and state slots on the host, so the pass
    // publishes its own host spans next to the device copies. The layers are not graph captured,
    // and the spans point at the caller's storage, which outlives this call.
    ScopedValue<std::span<const std::int32_t>> source_host_binding(
        active_linear_state_source_slots_host_, state_source_slots);
    ScopedValue<std::span<const std::int32_t>> destination_host_binding(
        active_linear_state_destination_slots_host_, state_destination_slots);
    ScopedValue<std::span<const std::int32_t>> valid_host_binding(active_valid_columns_host_,
                                                                  valid_columns);

    Tensor x = work_.alloc(DType::BF16, {dimension(config_.hidden_size), columns});
    ops::embedding(d_ids.view({columns}), *embed_, x, stream);
    // The merged rectangle hands every QSA layer the same per-row metadata the GDN path uses: the
    // row's live prefix, its record base and an identity record offset.
    QsaPassBinding qsa;
    begin_qsa_pass(work_, config_, batch, qsa);
    if (qsa.active) {
        copy_first_row(d_positions, qsa.base, stream);
        zero_i32(qsa.offsets, stream);
    }
    ScopedValue<const Tensor*> qsa_columns_binding(active_qsa_columns_,
                                                   qsa.active ? &d_valid : nullptr);
    ScopedValue<const Tensor*> qsa_base_binding(active_qsa_base_positions_,
                                                qsa.active ? &qsa.base : nullptr);
    ScopedValue<const Tensor*> qsa_offsets_binding(active_qsa_offsets_,
                                                   qsa.active ? &qsa.offsets : nullptr);
    if (prefill_gpu_timer_ != nullptr) { prefill_gpu_timer_->start(); }
    run_layers(x, Phase::Verify);
    // Prefill only needs the token that follows each row's last prompt token, so the projection runs
    // over one column per row instead of the whole rectangle: a [vocab, width, batch] logits tensor
    // at width 1024 would be hundreds of megabytes for one useful column. With four residual streams
    // the final correction folds the rectangle first: its fold is the head's input, so the tail
    // columns are read from it and no separate final norm runs.
    Tensor sampled;
    if (hc_enabled()) {
        sampled = work_.alloc(DType::BF16, {dimension(config_.hidden_size), columns});
        final_hidden(x, sampled, columns);
    }
    const Tensor& source = hc_enabled() ? sampled : x;
    Tensor last   = work_.alloc(DType::BF16, {dimension(config_.hidden_size), batch});
    Tensor normed = hc_enabled() ? last
                                 : work_.alloc(DType::BF16,
                                               {dimension(config_.hidden_size), batch});
    for (std::int32_t row = 0; row < batch; ++row) {
        const std::int32_t column = valid_columns[static_cast<std::size_t>(row)] - 1;
        if (column < 0 || column >= width) {
            throw std::invalid_argument("prefill batch row has no valid column");
        }
        // The batch rectangle is filled row-major (prefill.cpp: at = b*width + t), so x is the flat
        // [hidden, width*batch] tensor and row b's token at width index t lives in flat column
        // b*width + t. The fill, every batched kernel the pass feeds (attention positions, the GDN
        // conv column walk, the delta net's record mapping) and this tail-column read share the
        // one helper - a private layout here would be the same bug the width-major fill was.
        CUDA_CHECK(cudaMemcpyAsync(
            static_cast<char*>(last.data) + static_cast<std::size_t>(row) * last.nb[1],
            static_cast<const char*>(source.data) +
                static_cast<std::size_t>(
                    models::qwen3_5::execution::prefill_batch_flat_column(column, row, width)) *
                source.nb[1],
            static_cast<std::size_t>(dimension(config_.hidden_size)) * sizeof(std::uint16_t),
            cudaMemcpyDeviceToDevice, stream));
    }
    if (!hc_enabled())
        ops::rmsnorm(last, *final_norm_, config_.rms_norm_eps, true, normed, stream);
    // The single-lane path projects each chunk's last hidden column alone (prefill_impl: matrix
    // window at T=1, then project). A [vocab, batch] projection is a different column count and
    // therefore a different route, so the merged pass samples each row at T=1 too: the rows'
    // logits then match the single-lane arm's by the same construction every other per-column op
    // here relies on. Greedy output only - the pass carries no per-row sampling state.
    Tensor tokens = work_.alloc(DType::I32, {batch});
    Tensor logits = work_.alloc(DType::BF16, {dimension(config_.vocab_size), 1});
    for (std::int32_t row = 0; row < batch; ++row) {
        project(normed.slice(1, row, 1), *lm_head_, logits, work_, stream);
        Tensor row_token(static_cast<char*>(tokens.data) +
                             static_cast<std::size_t>(row) * tokens.nb[0],
                         DType::I32, {1});
        ops::argmax(logits, row_token,
                    dimension(parameters_.model.resources().public_token_count), stream);
    }
    // Each row's own tail hidden: the next decode step reads it, so it has to land in that lane's
    // sequence rather than in a shared scratch tensor.
    for (std::int32_t row = 0; row < batch; ++row) {
        Tensor* target = tail_hidden_rows[static_cast<std::size_t>(row)];
        CUDA_CHECK(cudaMemcpyAsync(target->data,
                                   static_cast<const char*>(last.data) +
                                       static_cast<std::size_t>(row) * last.nb[1],
                                   target->bytes(), cudaMemcpyDeviceToDevice, stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(next_tokens.data(), tokens.data,
                               lane_count * sizeof(std::int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (prefill_gpu_timer_ != nullptr) { prefill_gpu_timer_->record_stop(); }
    work_.reset();
}

void TextContext::mtp_forward_decode_batch(const Tensor& ids, const Tensor& hidden,
                                           const Tensor& cache_positions,
                                           const Tensor& rope_positions,
                                           const Tensor& valid_columns, const Tensor& kv_table_rows,
                                           ops::CausalAttentionExecutionEnvelope envelope,
                                           Tensor& mtp_hidden) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kMaximumMtpDraftTokens + 1) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("MTP decode batch shape is outside the supported domain");
    }
    require_tensor_shape(ids, DType::I32, {width, batch}, "MTP decode batch ids");
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), width, batch},
                         "MTP decode batch target hidden");
    require_tensor_shape(cache_positions, DType::I32, {width, batch},
                         "MTP decode batch cache positions");
    require_tensor_shape(rope_positions, DType::I32, {width, batch},
                         "MTP decode batch RoPE positions");
    require_tensor_shape(valid_columns, DType::I32, {batch}, "MTP decode batch valid columns");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "MTP decode batch KV rows");
    require_tensor_shape(mtp_hidden, DType::BF16, {dimension(config_.hidden_size), width, batch},
                         "MTP decode batch hidden");

    ScopedValue<const Tensor*> backend_binding(active_backend_kv_table_rows_, &kv_table_rows);
    ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns);
    ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
    ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);
    mtp_forward_core(ids, hidden, cache_positions, rope_positions, envelope, mtp_hidden, nullptr);
}

void TextContext::mtp_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens) {
    const std::int32_t batch = hidden.ne[1];
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), batch},
                         "MTP proposal batch hidden");
    require_tensor_shape(logits, DType::BF16, {dimension(config_.vocab_size), batch},
                         "MTP proposal batch logits");
    require_tensor_shape(draft_tokens, DType::I32, {batch}, "MTP proposal batch tokens");
    proposal_argmax(hidden, logits, draft_tokens);
}

void TextContext::mtp_propose_topk(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens,
                                   Tensor* candidate_ids, Tensor* proposal_q,
                                   std::int32_t top_k, ops::SamplingMask mask) {
    const std::int32_t batch = hidden.ne[1];
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), batch},
                         "MTP propose-topk hidden");
    if (candidate_ids == nullptr || proposal_q == nullptr) {
        throw std::logic_error("mtp_propose_topk requires candidate_ids and proposal_q outputs");
    }
    require_tensor_shape(*candidate_ids, DType::I32, {top_k, batch}, "MTP topk candidate ids");
    require_tensor_shape(*proposal_q, DType::FP32, {top_k, batch}, "MTP topk proposal q");

    if (mask.words != nullptr) {
        // Constrained: take the candidates from a masked copy of the full-vocabulary logits. The
        // reduced proposal head is skipped because its ids live in another domain than the bitmap.
        auto proposal_scope  = work_.scope();
        const auto domain    = dimension(config_.vocab_size);
        Tensor output_logits = work_.alloc(DType::BF16, {domain, batch});
        project(hidden, *lm_head_, output_logits, work_, ctx_.stream);
        Tensor masked = work_.alloc(DType::BF16, {domain, batch});
        ops::apply_sampling_mask(output_logits, masked, mask, ctx_.stream);
        ops::logits_topk_softmax(masked, *candidate_ids, *proposal_q, draft_tokens, nullptr, top_k,
                                 0, ctx_.stream);
        return;
    }

    auto proposal_scope = work_.scope();
    nvtx::ScopedRange proposal_range(nvtx::Name::MtpProposal, nvtx::Category::Mtp,
                                     static_cast<std::uint64_t>(batch));

    if (proposal_head_ != nullptr) {
        Tensor proposal_logits = work_.alloc(DType::BF16, {proposal_head_n_, batch});
        project(hidden, *proposal_head_, proposal_logits, work_, ctx_.stream);
        ops::logits_topk_softmax(proposal_logits, *candidate_ids, *proposal_q, draft_tokens,
                                 proposal_head_ids_, top_k, 0, ctx_.stream);
    } else {
        Tensor output_logits = work_.alloc(DType::BF16,
                                           {dimension(config_.vocab_size), batch});
        project(hidden, mtp_->output_head, output_logits, work_, ctx_.stream);
        ops::logits_topk_softmax(output_logits, *candidate_ids, *proposal_q, draft_tokens,
                                 nullptr, top_k, 0, ctx_.stream);
    }
}

void TextContext::mtp_propose_topk_strided(const Tensor& hidden, Tensor& logits,
                                           Tensor& draft_tokens, Tensor& candidate_ids,
                                           Tensor& proposal_q, std::int32_t top_k,
                                           std::int32_t column_stride, ops::SamplingMask mask,
                                           std::int32_t mask_column,
                                           std::int32_t mask_window_words) {
    const std::int32_t batch = hidden.ne[1];
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), batch},
                         "MTP propose-topk hidden");
    require_tensor_shape(draft_tokens, DType::I32, {batch}, "MTP topk draft tokens");
    if (mask.words != nullptr && mask_column > 0) {
        // Move inside the row: every batch row then sits one window further along.
        mask.words = mask.words + static_cast<std::ptrdiff_t>(mask_column) * mask.stride;
    }

    auto proposal_scope = work_.scope();
    nvtx::ScopedRange proposal_range(nvtx::Name::MtpProposal, nvtx::Category::Mtp,
                                     static_cast<std::uint64_t>(batch));

    if (mask.words != nullptr) {
        if (std::getenv("NINFER_MTP_DEBUG") != nullptr) {
            static bool reported = false;
            if (!reported) {
                reported = true;
                std::fprintf(stderr, "[mtp-mask] strided reached with bitmap stride=%d\n",
                             mask.stride);
            }
        }
        Tensor output_logits =
            work_.alloc(DType::BF16, {dimension(config_.vocab_size), batch});
        project(hidden, *lm_head_, output_logits, work_, ctx_.stream);
        Tensor masked = work_.alloc(DType::BF16, {dimension(config_.vocab_size), batch});
        ops::apply_sampling_mask(output_logits, masked, mask, ctx_.stream, mask_window_words);
        ops::logits_topk_softmax(masked, candidate_ids, proposal_q, draft_tokens, nullptr, top_k,
                                 column_stride, ctx_.stream);
        return;
    }

    if (proposal_head_ != nullptr) {
        Tensor proposal_logits = work_.alloc(DType::BF16, {proposal_head_n_, batch});
        project(hidden, *proposal_head_, proposal_logits, work_, ctx_.stream);
        ops::logits_topk_softmax(proposal_logits, candidate_ids, proposal_q, draft_tokens,
                                 proposal_head_ids_, top_k, column_stride, ctx_.stream);
    } else {
        Tensor output_logits =
            work_.alloc(DType::BF16, {dimension(config_.vocab_size), batch});
        project(hidden, mtp_->output_head, output_logits, work_, ctx_.stream);
        ops::logits_topk_softmax(output_logits, candidate_ids, proposal_q, draft_tokens, nullptr,
                                 top_k, column_stride, ctx_.stream);
    }
}

void TextContext::hc_stem(Tensor& embedding, std::int32_t tokens) {
    const auto& hc = config_.hc.value();
    // The stream width the artifact states, so a plan built from a different one is refused here
    // rather than by a silent shape mismatch three ops later.
    if (hc.count != 4 || hc.stream_width(config_.hidden_size) != 4 * config_.hidden_size) {
        throw std::invalid_argument("Qwen4Exp hyper-connection requires four residual streams of "
                                    "the hidden width");
    }
    const auto roots = workspace::text_hc_roots(work_, config_, tokens);
    hc_              = roots;
    const auto width = dimension(config_.hidden_size), streams = dimension(hc.count);
    Tensor weight    = parameters_.text.layers.front().input_norm.view({width, streams});
    Tensor residual  = hc_.residual.view({width, streams, tokens});
    Tensor normed    = hc_.normalized.view({width, streams, tokens});
    // The residual starts as the embedding, stream-replicated and stream-normed exactly once.
    ops::hc_expand_norm(embedding, weight, config_.rms_norm_eps, residual, normed,
                        ctx_.execution_view());
}

namespace {
void nan_probe_tensor(const char* tag, const Tensor& value, std::size_t layer, cudaStream_t stream) {
    if (std::getenv("NINFER_NAN_PROBE") == nullptr || value.data == nullptr) { return; }
    if (value.dtype != DType::BF16) { return; }
    const auto rows = value.ne[0], cols = value.ne[1];
    std::vector<std::uint16_t> host(static_cast<std::size_t>(rows) * cols);
    if (cudaMemcpyAsync(host.data(), value.data, host.size() * sizeof(std::uint16_t),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        return;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) { return; }
    std::size_t bad = 0, elements = 0;
    int first = -1;
    std::vector<std::size_t> row_hits(static_cast<std::size_t>(rows), 0);
    std::vector<std::size_t> col_hits(static_cast<std::size_t>(cols), 0);
    for (int c = 0; c < cols; ++c) {
        std::size_t token_bad = 0;
        for (int r = 0; r < rows; ++r) {
            const auto bits = host[static_cast<std::size_t>(c) * rows + r];
            if ((bits & 0x7f80U) == 0x7f80U && (bits & 0x7fU) != 0U) {
                ++token_bad;
                ++elements;
                ++row_hits[static_cast<std::size_t>(r)];
            }
        }
        if (token_bad != 0) {
            ++bad;
            col_hits[static_cast<std::size_t>(c)] = token_bad;
            if (first < 0) { first = c; }
        }
    }
    std::size_t bad_rows = 0, first_bad_row = static_cast<std::size_t>(-1);
    for (std::size_t r = 0; r < row_hits.size(); ++r) {
        if (row_hits[r] != 0) {
            ++bad_rows;
            if (first_bad_row == static_cast<std::size_t>(-1)) { first_bad_row = r; }
        }
    }
    std::fprintf(stderr,
                 "[nan-probe] layer=%zu %s columns=%d bad_columns=%zu first=%d elements=%zu "
                 "bad_rows=%zu first_row=%lld\n",
                 layer, tag, cols, bad, first, elements, bad_rows,
                 static_cast<long long>(first_bad_row));
    if (std::getenv("NINFER_NAN_PROBE") != nullptr &&
        std::string(std::getenv("NINFER_NAN_PROBE")) == "2") {
        std::fprintf(stderr, "[nan-probe]   %s per-column:", tag);
        for (int c = 0; c < cols; ++c) {
            std::fprintf(stderr, " %d:%zu", c, col_hits[static_cast<std::size_t>(c)]);
        }
        std::fprintf(stderr, "\n");
    }
    if (std::getenv("NINFER_NAN_PROBE") != nullptr &&
        std::string(std::getenv("NINFER_NAN_PROBE")) == "3") {
        std::fprintf(stderr, "[nan-probe]   %s sentinel-per-column (0x8080):", tag);
        for (int c = 0; c < cols; ++c) {
            std::size_t hits = 0;
            for (int r = 0; r < rows; ++r) {
                if (host[static_cast<std::size_t>(c) * rows + r] == 0x8080U) { ++hits; }
            }
            std::fprintf(stderr, " %d:%zu", c, hits);
        }
        std::fprintf(stderr, "\n");
    }
    std::fflush(stderr);
}
} // namespace

void TextContext::hc_mix_into(const HcParameters& weights, std::int32_t tokens) {
    const auto normalized = hc_.normalized.view({dimension(config_.hidden_size),
                                                 dimension(config_.hc->count), tokens});
    // Produces the block's single-stream input and the four coefficients its correction applies.
    nan_probe_tensor("hc-mix-pre", hc_.input, 300, ctx_.stream);
    nan_probe_tensor("hc-normed", hc_.normalized, 400, ctx_.stream);
    ops::hc_mix(normalized, weights.block_form(), work_, hc_.input, hc_.injection,
                ctx_.execution_view());
    nan_probe_tensor("hc-mix-out", hc_.input, 200, ctx_.stream);
}

void TextContext::hc_combine(const Tensor& product, const Tensor& norm_weight,
                             std::int32_t tokens) {
    const auto width   = dimension(config_.hidden_size);
    const auto streams = dimension(config_.hc->count);
    Tensor residual    = hc_.residual.view({width, streams, tokens});
    Tensor normed      = hc_.normalized.view({width, streams, tokens});
    const auto weight  = norm_weight.view({width, streams});
    // residual += injection * product, then the grouped norm of the result for the next block.
    // The residual aliases itself in place, which the Op admits exactly.
    ops::hc_combine_norm(residual, product, hc_.injection, weight, config_.rms_norm_eps, residual,
                         normed, ctx_.execution_view());
}

void TextContext::final_hidden(Tensor& x, Tensor& out, std::int32_t tokens) {
    if (!hc_enabled()) {
        ops::rmsnorm(x, *final_norm_, config_.rms_norm_eps, true, out, ctx_.stream);
        return;
    }
    const auto& weights = *final_hc_;
    ops::hc_mix(hc_.normalized.view({dimension(config_.hidden_size),
                                     dimension(config_.hc->count), tokens}),
                weights.final_form(), work_, out, ctx_.execution_view());
}

void TextContext::attn_mix(const BlockParameters& w, Tensor& x, int fidx, Phase ph,
                           const Tensor* normalized) {
    const auto& p  = std::get<AttentionParameters>(w.mixer);
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];
    if (active_causal_attention_envelope_ == nullptr) {
        throw std::logic_error("Text GQA execution envelope is not set");
    }

    const auto projection = workspace::text_attention_projection(work_, config_, T);
    Tensor h              = normalized != nullptr ? *normalized : projection.hidden;
    if (normalized == nullptr) ops::rmsnorm(x, w.input_norm, config_.rms_norm_eps, true, h, s);

    Tensor q         = projection.query.view({dimension(config_.attention->head_dim),
                                              dimension(config_.attention->num_attention_heads), T});
    Tensor gate      = projection.gate.view({dimension(config_.attention->head_dim),
                                             dimension(config_.attention->num_attention_heads), T});
    Tensor k         = projection.key.view({dimension(config_.attention->head_dim),
                                            dimension(config_.attention->num_key_value_heads), T});
    Tensor v         = projection.value.view({dimension(config_.attention->head_dim),
                                              dimension(config_.attention->num_key_value_heads), T});
    Tensor q_flat    = q.view({dimension(config_.attention->query_width()), T});
    Tensor gate_flat = gate.view({dimension(config_.attention->query_width()), T});
    Tensor k_flat    = k.view({dimension(config_.attention->key_width()), T});
    Tensor v_flat    = v.view({dimension(config_.attention->key_width()), T});
    attention_projection(h, p, q_flat, gate_flat, k_flat, v_flat, work_, s);
    nan_probe_tensor("attn-proj-q", q_flat, 100 + std::size_t(fidx), s);
    nan_probe_tensor("attn-proj-k", k_flat, 100 + std::size_t(fidx), s);
    nan_probe_tensor("attn-proj-v", v_flat, 100 + std::size_t(fidx), s);

    const auto results = workspace::text_attention_results(work_, config_, T);
    Tensor qn =
        results.normalized_query.view({dimension(config_.attention->head_dim),
                                       dimension(config_.attention->num_attention_heads), T});
    Tensor kn = results.normalized_key.view({dimension(config_.attention->head_dim),
                                             dimension(config_.attention->num_key_value_heads), T});
    ops::rmsnorm(q, p.query_norm, config_.rms_norm_eps, true, qn, s);
    ops::rmsnorm(k, p.key_norm, config_.rms_norm_eps, true, kn, s);
    nan_probe_tensor("attn-qn", qn.view({dimension(config_.attention->query_width()), T}),
                     100 + std::size_t(fidx), s);
    const Tensor& cache_positions =
        active_cache_positions_ != nullptr ? *active_cache_positions_ : io_.pos;
    const Tensor& rope_positions =
        active_rope_positions_ != nullptr ? *active_rope_positions_ : io_.rope_pos;
    Tensor rope_for_op = active_sequence_batch_ != 0 ? rope_positions.view({T}) : rope_positions;
    text_rope(rope_for_op, *config_.rope_parameters, qn, kn, ctx_.execution_view());
    // Diagnosis: the normalized-and-rotated Q/K/V the dense arm feeds its append, so the same
    // tensors can be compared against prepare's own output instead of being inferred through
    // attention.
    qsa_dump_tensor("NINFER_QSA_LAYER_DUMP", "query", qn, fidx, T, s);
    qsa_dump_tensor("NINFER_QSA_LAYER_DUMP", "key_in", kn, fidx, T, s);
    qsa_dump_tensor("NINFER_QSA_LAYER_DUMP", "value_in", v, fidx, T, s);

    Tensor a = results.attention.view({dimension(config_.attention->head_dim),
                                       dimension(config_.attention->num_attention_heads), T});
    const Tensor& kv_table_rows =
        active_kv_table_rows_ != nullptr ? *active_kv_table_rows_ : io_.text_kv_table_row;
    if (active_sequence_batch_ != 0) {
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T) {
            throw std::logic_error("Text sequence batch binding does not match aggregate columns");
        }
        Tensor q_batch        = qn.view({dimension(config_.attention->head_dim),
                                         dimension(config_.attention->num_attention_heads), width,
                                         active_sequence_batch_});
        Tensor k_batch        = kn.view({dimension(config_.attention->head_dim),
                                         dimension(config_.attention->num_key_value_heads), width,
                                         active_sequence_batch_});
        Tensor v_batch        = v.view({dimension(config_.attention->head_dim),
                                        dimension(config_.attention->num_key_value_heads), width,
                                        active_sequence_batch_});
        Tensor a_batch        = a.view({dimension(config_.attention->head_dim),
                                        dimension(config_.attention->num_attention_heads), width,
                                        active_sequence_batch_});
        Tensor position_batch = cache_positions.view({width, active_sequence_batch_});
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        ops::causal_softmax_attention(
            q_batch, k_batch, v_batch, position_batch, valid, kv_table_rows,
            {dimension(config_.attention->head_dim),
             dimension(config_.attention->num_attention_heads),
             dimension(config_.attention->num_key_value_heads)},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(config_.attention->head_dim))),
            batch_text_kv_->batch_layer_view(fidx), *active_causal_attention_envelope_, work_,
            a_batch, ctx_.execution_view());
    } else {
        ops::causal_softmax_attention(
            qn, kn, v, cache_positions, Tensor{}, kv_table_rows,
            {dimension(config_.attention->head_dim),
             dimension(config_.attention->num_attention_heads),
             dimension(config_.attention->num_key_value_heads)},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(config_.attention->head_dim))),
            batch_text_kv_->batch_layer_view(fidx), *active_causal_attention_envelope_, work_, a,
            ctx_.execution_view());
    }
    ops::sigmoid_mul(gate, a, s);
    // The same file the sparse arm writes: the two arms' attention outputs are the S3 gate.
    qsa_dump_tensor("NINFER_QSA_LAYER_DUMP", "attention",
                    a.view({dimension(config_.attention->head_dim),
                            dimension(config_.attention->num_attention_heads), T}),
                    fidx, T, s);
    // This arm appends through the dense Op, so its K plane is the reference the byte gate compares
    // prepare's write against.
    qsa_dump_tensor("NINFER_QSA_KV_DUMP", "main_key",
                    batch_text_kv_->batch_layer_view(static_cast<std::uint32_t>(fidx)).k_pages, fidx,
                    0, s);
    qsa_dump_tensor("NINFER_QSA_KV_DUMP", "main_value",
                    batch_text_kv_->batch_layer_view(static_cast<std::uint32_t>(fidx)).v_pages, fidx,
                    0, s);

    {
        Tensor attn_out = a.view({dimension(config_.attention->query_width()), T});
        nan_probe_tensor("attn-out", attn_out, 100 + std::size_t(fidx), s);
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

void TextContext::gdn_mix(const BlockParameters& w, Tensor& x, int gidx, Phase ph,
                          const Tensor* normalized) {
    const auto& p  = std::get<GdnParameters>(w.mixer);
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    const auto control = workspace::gdn_control(work_, config_, T);
    Tensor h           = normalized != nullptr ? *normalized : control.hidden;
    Tensor g           = control.g;
    Tensor beta        = control.beta;
    if (normalized != nullptr) {
        // The four-stream correction already produced the grouped norm, so only the two control
        // projections remain.
        ops::gdn_control_proj(h, p.control, p.a_log, p.dt_bias, {g, beta}, work_,
                              ctx_.execution_view());
    } else {
        gdn_norm_control(x, w.input_norm, config_.rms_norm_eps, p, h, g, beta, work_,
                         ctx_.execution_view());
    }

    nan_probe_tensor("gdn-h-in", h, gidx, s);
    const auto projection = workspace::gdn_projection(work_, config_, T);    Tensor z  = projection.output_gate.view({dimension(config_.gdn->linear_value_head_dim),
                                             dimension(config_.gdn->linear_num_value_heads), T});
    Tensor qc = projection.query;
    Tensor kc = projection.key;
    Tensor vc = projection.value;
    if (ph == Phase::Verify) {
        if (active_sequence_batch_ == 0 || active_linear_state_source_slots_ == nullptr) {
            throw std::logic_error(
                "Verify GDN requires an explicit sequence batch and state slots");
        }
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T) {
            throw std::logic_error("GDN sequence batch binding does not match aggregate columns");
        }
        if (gdn_state_action_ == GdnStateAction::UpdateInPlace &&
            active_linear_state_destination_slots_ == nullptr) {
            throw std::logic_error(
                "In-place batched GDN update requires bound destination state slots");
        }
        Tensor projection_input =
            h.view({dimension(config_.hidden_size), width, active_sequence_batch_});
        Tensor query_output =
            qc.view({dimension(config_.gdn->key_width()), width, active_sequence_batch_});
        Tensor key_output =
            kc.view({dimension(config_.gdn->key_width()), width, active_sequence_batch_});
        Tensor value_output =
            vc.view({dimension(config_.gdn->value_width()), width, active_sequence_batch_});
        Tensor gate_output =
            z.view({dimension(config_.gdn->value_width()), width, active_sequence_batch_});
        Tensor conv_states = state_.layer_view(static_cast<std::uint32_t>(gidx)).conv;
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            if (replay_records_ == nullptr) {
                throw std::logic_error("Replay-record GDN has no record storage");
            }
            GdnReplayRecordLayer records = replay_records_->layer(gidx, active_sequence_batch_);
            gdn_projection_record(projection_input, p, *config_.gdn, conv_states, valid,
                                  *active_linear_state_source_slots_, records.conv, query_output,
                                  key_output, value_output, gate_output, work_, s);
        } else if (width > 1) {
            // The merged prefill rectangle (UpdateInPlace, W>1; only prefill_batch binds this).
            // It runs the single-lane prefill chunk's own conv form per row: the aggregate
            // projection at the rectangle's column count, then the SAME causal_conv1d_silu_split op
            // on each row's slice with that row's state slots. The conv outputs and the
            // destination conv state are therefore the single-lane results by construction - no
            // snapshot column scratch is borrowed from the state pool, so the pass needs no slot
            // spans, and the state lands in the row's destination slot where the fork settle
            // expects it.
            if (active_valid_columns_host_.empty() ||
                active_linear_state_source_slots_host_.empty() ||
                active_linear_state_destination_slots_host_.empty()) {
                throw std::logic_error(
                    "merged prefill GDN requires the pass's host state bindings");
            }
            Tensor qkv = workspace::gdn_prefill_conv(work_, config_, T);
            Tensor z_flat = z.view({dimension(config_.gdn->value_width()), T});
            gdn_projection(h, p, qkv, z_flat, work_, s);
            // Column-window helper: the split op takes writable Tensor& operands and Tensor::slice
            // returns a temporary, so each row's window is built explicitly from the base pointer.
            const auto row_window = [](Tensor& source, std::int32_t first, std::int32_t columns) {
                return Tensor(
                    static_cast<char*>(source.data) +
                        static_cast<std::size_t>(first) * source.nb[1],
                    source.dtype, {source.ne[0], columns});
            };
            for (std::int32_t row = 0; row < active_sequence_batch_; ++row) {
                const std::int32_t count = active_valid_columns_host_[static_cast<std::size_t>(row)];
                const std::int32_t base  = row * width;
                Tensor conv_state_in =
                    state_.conv_slot(static_cast<std::uint32_t>(gidx),
                                     active_linear_state_source_slots_host_[static_cast<std::size_t>(row)]);
                Tensor conv_state_out = state_.conv_slot(
                    static_cast<std::uint32_t>(gidx),
                    active_linear_state_destination_slots_host_[static_cast<std::size_t>(row)]);
                Tensor query_out   = row_window(qc, base, count);
                Tensor key_out     = row_window(kc, base, count);
                Tensor value_out   = row_window(vc, base, count);
                Tensor projected_in = row_window(qkv, base, count);
                ops::causal_conv1d_silu_split(projected_in, p.convolution, conv_state_in,
                                              conv_state_out, query_out, key_out, value_out, s);
            }
        } else {
            gdn_projection_snapshot(projection_input, p, *config_.gdn, conv_states, valid,
                                    *active_linear_state_source_slots_,
                                    *active_linear_state_destination_slots_, query_output,
                                    key_output, value_output, gate_output, work_, s);
        }
    } else {
        Tensor qkv    = workspace::gdn_prefill_conv(work_, config_, T);
        Tensor z_flat = z.view({dimension(config_.gdn->value_width()), T});
        // Diagnosis: a finite sentinel in the projection's destinations shows whether a token column
        // the probe reports as NaN was computed or simply never written.
        if (std::getenv("NINFER_NAN_PROBE_FILL") != nullptr) {
            CUDA_CHECK(cudaMemsetAsync(qkv.data, 0x80, qkv.bytes(), s));
        }
        gdn_projection(h, p, qkv, z_flat, work_, s);
        Tensor conv_state_in =
            state_.conv_slot(static_cast<std::uint32_t>(gidx), linear_state_source_slot_);
        Tensor conv_state_out =
            state_.conv_slot(static_cast<std::uint32_t>(gidx), linear_state_destination_slot_);
        nan_probe_tensor("gdn-qkv-in", qkv, gidx, s);
        nan_probe_tensor("gdn-conv-in-state", conv_state_in, gidx, s);
        ops::causal_conv1d_silu_split(qkv, p.convolution, conv_state_in, conv_state_out, qc, kc, vc,
                                      s);
        nan_probe_tensor("gdn-kc", kc, gidx, s);
    }
    nan_probe_tensor("gdn-qc", qc, gidx, s);
    nan_probe_tensor("gdn-vc", vc, gidx, s);
    nan_probe_tensor("gdn-gate", z.view({dimension(config_.gdn->value_width()), T}), gidx, s);

    Tensor q_recurrent = qc.view({dimension(config_.gdn->linear_key_head_dim),
                                  dimension(config_.gdn->linear_num_key_heads), T});
    Tensor k_recurrent = kc.view({dimension(config_.gdn->linear_key_head_dim),
                                  dimension(config_.gdn->linear_num_key_heads), T});

    Tensor vv = vc.view({dimension(config_.gdn->linear_value_head_dim),
                         dimension(config_.gdn->linear_num_value_heads), T});
    Tensor o  = workspace::gdn_recurrent_output(work_, config_, T)
                   .view({dimension(config_.gdn->linear_value_head_dim),
                          dimension(config_.gdn->linear_num_value_heads), T});
    if (ph == Phase::Verify) {
        Tensor recurrent_states  = state_.layer_view(static_cast<std::uint32_t>(gidx)).recurrent;
        const std::int32_t width = active_sequence_width_;
        Tensor q_batch           = q_recurrent.view({dimension(config_.gdn->linear_key_head_dim),
                                                     dimension(config_.gdn->linear_num_key_heads), width,
                                                     active_sequence_batch_});
        Tensor k_batch           = k_recurrent.view({dimension(config_.gdn->linear_key_head_dim),
                                                     dimension(config_.gdn->linear_num_key_heads), width,
                                                     active_sequence_batch_});
        Tensor v_batch           = vv.view({dimension(config_.gdn->linear_value_head_dim),
                                            dimension(config_.gdn->linear_num_value_heads), width,
                                            active_sequence_batch_});
        Tensor g_batch =
            g.view({dimension(config_.gdn->linear_num_value_heads), width, active_sequence_batch_});
        Tensor beta_batch = beta.view(
            {dimension(config_.gdn->linear_num_value_heads), width, active_sequence_batch_});
        Tensor out_batch =
            o.view({dimension(config_.gdn->linear_value_head_dim),
                    dimension(config_.gdn->linear_num_value_heads), width, active_sequence_batch_});
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            GdnReplayRecordLayer records = replay_records_->layer(gidx, active_sequence_batch_);
            ops::gated_delta_net_replay_record(
                q_batch, k_batch, v_batch, g_batch, beta_batch,
                static_cast<float>(
                    1.0 / std::sqrt(static_cast<double>(config_.gdn->linear_key_head_dim))),
                recurrent_states, valid, *active_linear_state_source_slots_, records.key,
                records.value, records.gate, out_batch, s);
        } else if (width > 1) {
            // The merged rectangle's recurrence runs each row through the same kernel its
            // single-lane counterpart runs (chunked above the minimum, sequential below), with
            // the row's own state slots and live prefix - the entry whose op-level contract is
            // per-row bit-identity with gated_delta_net.
            ops::gated_delta_net_batch(
                q_batch, k_batch, v_batch, g_batch, beta_batch,
                static_cast<float>(
                    1.0 / std::sqrt(static_cast<double>(config_.gdn->linear_key_head_dim))),
                /*normalize_qk=*/true, work_, recurrent_states, valid,
                *active_linear_state_source_slots_, *active_linear_state_destination_slots_,
                out_batch, ctx_.execution_view());
        } else {
            ops::gated_delta_net_batch_update(
                q_batch, k_batch, v_batch, g_batch, beta_batch,
                static_cast<float>(
                    1.0 / std::sqrt(static_cast<double>(config_.gdn->linear_key_head_dim))),
                /*normalize_qk=*/true, recurrent_states, *active_linear_state_source_slots_,
                *active_linear_state_destination_slots_, out_batch, s);
        }
    } else {
        Tensor recurrent_state_in =
            state_.recurrent_slot(static_cast<std::uint32_t>(gidx), linear_state_source_slot_);
        Tensor recurrent_state_out =
            state_.recurrent_slot(static_cast<std::uint32_t>(gidx), linear_state_destination_slot_);
        ops::gated_delta_net(q_recurrent, k_recurrent, vv, g, beta,
                             static_cast<float>(1.0 / std::sqrt(static_cast<double>(
                                                          config_.gdn->linear_key_head_dim))),
                             /*normalize_qk=*/true, work_, recurrent_state_in, recurrent_state_out,
                             o, ctx_.execution_view());
    }

    // Diagnosis (NINFER_NAN_PROBE only): re-read the conv outputs after the recurrence, so a NaN
    // that only shows up here is an overwrite by the scan's own scratch rather than a conv result.
    nan_probe_tensor("gdn-qc-post", qc, gidx, s);
    nan_probe_tensor("gdn-vc-post", vc, gidx, s);

    Tensor on = workspace::gdn_normalized_output(work_, config_, T)
                    .view({dimension(config_.gdn->linear_value_head_dim),
                           dimension(config_.gdn->linear_num_value_heads), T});
    ops::silu_gated_rmsnorm(o, p.norm, z, config_.rms_norm_eps, on, ctx_.execution_view());
    nan_probe_tensor("gdn-recurrent-o",
                     o.view({dimension(config_.gdn->value_width()), T}), gidx, s);
    nan_probe_tensor("gdn-normed-on",
                     on.view({dimension(config_.gdn->value_width()), T}), gidx, s);

    {
        Tensor gdn_out = on.view({dimension(config_.gdn->value_width()), T});
        if (normalized != nullptr) {
            ops::linear(gdn_out, p.output.weight, x, p.output.policy, work_, s);
        } else if (ops::w4a8_supported(p.output.weight, T)) {
            ops::w4a8_linear_add(gdn_out, p.output.weight, x, work_, s);
        } else {
            ops::linear_add(gdn_out, p.output.weight, x, p.output.policy, work_, s);
        }
    }
}

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

ops::SparseMoeHints TextContext::next_projection_hints(int layer) const {
    const auto next = static_cast<std::size_t>(layer) + 1;
    return next < parameters_.text.layers.size() ? parameters_.text.layers[next].projection_prefetch
                                                 : ops::SparseMoeHints{};
}

void TextContext::mlp_tail(const BlockParameters& weights, Tensor& x, Phase,
                           const ops::SparseMoeHints& hints, int layer,
                           const Tensor* normalized) {
    Tensor h = normalized != nullptr ? *normalized
                                     : workspace::post_mixer_hidden(work_, config_, x.ne[1]);
    if (normalized == nullptr)
        ops::rmsnorm(x, weights.post_attention_norm, config_.rms_norm_eps, true, h, ctx_.stream);
    // [P1 probe] optional per-token int8 activation round-trip (NINFER_ACT_A8)
    ops::act_quant_a8_roundtrip_bf16(h.data, h.ne[0], h.ne[1], ctx_.stream);
    // Q68b: hand the NEXT layer's routed block in, so the prediction computed after this layer's
    // residual is the one that layer will consume. Layer indices here are the block ordinal, which is
    // also the expert bank's index, so `+1` names both.
    const auto next = static_cast<std::size_t>(layer) + 1;
    const RoutedMoeParameters* predict_next = nullptr;
    if (next < parameters_.text.layers.size()) {
        const auto& ffn_next = parameters_.text.layers[next].ffn;
        if (const auto* moe_next = std::get_if<RoutedMoeParameters>(&ffn_next)) {
            predict_next = moe_next;
        }
        if (layer == 0 && std::getenv("NINFER_EXPERT_TIMING") != nullptr) {
            std::fprintf(stderr, "[pred] probe layer=0 this.variant=%zu next.variant=%zu stream=%d\n",
                         weights.ffn.index(), ffn_next.index(),
                         predict_next != nullptr && predict_next->stream != nullptr ? 1 : 0);
            std::fflush(stderr);
        }
    }
    ffn(h, weights.ffn, x, hints, work_, ctx_.execution_view(), false,
        // Q68b probe: score this layer's router against the residual stream the previous layer left
        // behind, then record this layer's own for the next one. Both calls are no-ops unless
        // NINFER_LAYER_PROBE=2.
        router_probe_input(ctx_.execution_view().device()), predict_next);
    router_probe_save(x);
    // Q30/P6: the prediction is handed to the next layer's region right here, between the layers, so
    // the copy is in flight before that layer's kernels are queued. Doing it in the next layer's
    // ensure() is too late: by then its routes are known and the "prediction" is the demand set.
    if (predict_next != nullptr && predict_next->stream) {
        predict_next->stream->stage_predicted(ctx_.stream);
    }
}

template <class Tap>
void TextContext::run_layers(Tensor& x, Phase ph, Tap& tap) {
    const bool prefill = ph == Phase::Prefill;
    const auto nan_probe = [&](const char* tag, const Tensor& value, std::size_t layer) {
        if (std::getenv("NINFER_NAN_PROBE") == nullptr || value.data == nullptr) { return; }
        const auto rows = value.ne[0], cols = value.ne[1];
        std::vector<std::uint16_t> host(static_cast<std::size_t>(rows) * cols);
        CUDA_CHECK(cudaMemcpyAsync(host.data(), value.data, host.size() * sizeof(std::uint16_t),
                                   cudaMemcpyDeviceToHost, ctx_.stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx_.stream));
        std::size_t bad = 0;
        int first = -1;
        for (int c = 0; c < cols; ++c) {
            std::size_t token_bad = 0;
            for (int r = 0; r < rows; ++r) {
                const auto bits = host[static_cast<std::size_t>(c) * rows + r];
                if ((bits & 0x7f80U) == 0x7f80U && (bits & 0x7fU) != 0U) { ++token_bad; }
            }
            if (token_bad != 0) {
                ++bad;
                if (first < 0) { first = c; }
            }
        }
        std::fprintf(stderr, "[nan-probe] layer=%zu %s columns=%d bad_columns=%zu first=%d\n", layer,
                     tag, cols, bad, first);
        std::fflush(stderr);
    };
    // Qwen4Exp: the embedding becomes four residual streams once, and every block reads a folded
    // single-stream view of them. The tensors are roots of the call, so they are allocated here and
    // live to its end; the plan reserves them from the same recipe in the same order.
    if (hc_enabled()) hc_stem(x, x.ne[1]);
    for (std::size_t layer = 0; layer < parameters_.text.layers.size(); ++layer) {
        const auto& block = parameters_.text.layers[layer];
        // The config owns the dispatch: an indexed layer is a full-attention mixer whose softmax
        // runs over the indexer's selected KV blocks (see `qsa_mix`).
        const auto kind     = config_.layer_types[layer];
        const bool full     = attention_mixer(kind);
        const bool indexed  = kind == MixerKind::IndexedAttention;
        const auto compact  = dimension(config_.compact_layer_indices[layer]);
        const bool hc       = hc_enabled();
        const int tokens    = x.ne[1];
        // With four streams the mixer and the FFN write their own product and the correction folds
        // it in; otherwise both add straight into the residual `x`.
        Tensor& product     = hc ? hc_.output : x;
        const Tensor* folded = nullptr;
        nvtx::ScopedRange layer_range(
            full ? (prefill ? nvtx::Name::PrefillLayerFull : nvtx::Name::VerifyLayerFull)
                 : (prefill ? nvtx::Name::PrefillLayerGdn : nvtx::Name::VerifyLayerGdn),
            full ? nvtx::Category::Attention : nvtx::Category::Gdn, layer);
        try {
            {
                nvtx::ScopedRange mixer_range(
                    full ? (prefill ? nvtx::Name::PrefillAttention : nvtx::Name::VerifyAttention)
                         : (prefill ? nvtx::Name::PrefillGdn : nvtx::Name::VerifyGdn),
                    full ? nvtx::Category::Attention : nvtx::Category::Gdn, layer);
                auto scope = work_.scope();
                const bool trace_stage =
                    std::getenv("NINFER_ALLOC_DEBUG") != nullptr && layer == 0;
                if (trace_stage) {
                    std::fprintf(stderr, "[stage] layer %zu mixer start: used %zu\n", layer,
                                 work_.used());
                    std::fflush(stderr);
                }
                if (full) {
                    if (hc) {
                        hc_mix_into(*block.input_hc, tokens);
                        folded = &hc_.input;
                    }
                    // An indexed layer writes its own KV through prepare and reads it back through
                    // the indexer's selection; NINFER_QSA_FORCE_FULL keeps the old dense mixer as
                    // the reference arm of both QSA gates.
                    if (indexed && !qsa_dense_arm()) {
                        qsa_mix(block, product, compact, ph, folded);
                    } else {
                        if (indexed) { note_dense_indexed_arm(); }
                        attn_mix(block, product, compact, ph, folded);
                    }
                } else {
                    if (hc) {
                        hc_mix_into(*block.input_hc, tokens);
                        folded = &hc_.input;
                    }
                    gdn_mix(block, product, compact, ph, folded);
                }
                if (hc) {
                    nan_probe("pre-combine-residual", hc_.residual, layer);
                    nan_probe("pre-combine-product", hc_.output, layer);
                    hc_combine(product, block.post_attention_norm, tokens);
                    nan_probe("post-combine-residual", hc_.residual, layer);
                    nan_probe("post-combine-normed", hc_.normalized, layer);
                    nan_probe("post-mixer-combine", hc_.input, layer);
                    hc_mix_into(*block.post_attention_hc, tokens);
                    folded = &hc_.input;
                }
                nan_probe("post-mixer", hc ? hc_.input : x, layer);
                if (trace_stage) {
                    std::fprintf(stderr, "[stage] layer %zu mixer end: used %zu\n", layer,
                                 work_.used());
                    std::fflush(stderr);
                }
                if (std::getenv("NINFER_ALLOC_DEBUG") != nullptr &&
                    work_.used() * 10 >= work_.capacity() * 9) {
                    std::fprintf(stderr, "[stage] layer %zu %s mixer: used %zu of %zu\n", layer,
                                 full ? "full" : "gdn", work_.used(), work_.capacity());
                    std::fflush(stderr);
                }
            }
            {
                nvtx::ScopedRange range(prefill ? nvtx::Name::PrefillPostMixer
                                                : nvtx::Name::VerifyPostMixer,
                                        nvtx::Category::PostMixer, layer);
                auto scope = work_.scope();
                const bool trace_stage =
                    std::getenv("NINFER_ALLOC_DEBUG") != nullptr && layer == 0;
                if (trace_stage) {
                    std::fprintf(stderr, "[stage] layer %zu post-mixer start: used %zu\n", layer,
                                 work_.used());
                    std::fflush(stderr);
                }
                if (hc) {
                    // The FFN family adds into its residual operand, so the product starts empty; a
                    // write form would save this fill and is the next step for this path.
                    CUDA_CHECK(cudaMemsetAsync(product.data, 0, product.bytes(), ctx_.stream));
                }
                mlp_tail(block, product, ph, next_projection_hints(static_cast<int>(layer)),
                         static_cast<int>(layer), folded);
                nan_probe("post-mlp", hc ? hc_.input : x, layer);
                if (trace_stage) {
                    std::fprintf(stderr, "[stage] layer %zu post-mixer end: used %zu\n", layer,
                                 work_.used());
                    std::fflush(stderr);
                }
                if (std::getenv("NINFER_ALLOC_DEBUG") != nullptr &&
                    work_.used() * 10 >= work_.capacity() * 9) {
                    std::fprintf(stderr, "[stage] layer %zu post-mixer: used %zu of %zu\n", layer,
                                 work_.used(), work_.capacity());
                    std::fflush(stderr);
                }
            }
            if (hc) {
                // The block that consumes the normalized view supplies the norm: the next block's
                // attention correction, or the final one after the last block.
                const auto next = layer + 1;
                hc_combine(product,
                           next < parameters_.text.layers.size()
                               ? parameters_.text.layers[next].input_norm
                               : *final_norm_,
                           tokens);
                nan_probe("pre-next-input", hc_.input, layer);
            }
            if constexpr (Tap::enabled) {
                tap.capture_layer(static_cast<int>(layer), hc ? hc_.input : x, ctx_.stream);
            }
        } catch (const std::exception& error) {
            // The arena state at the failure is the only thing that says whether the plan sized the
            // workspace for this width or whether the request outgrew it. Printed before the rethrow
            // so the numbers survive the single-line error the caller reports.
            if (std::getenv("NINFER_ALLOC_DEBUG") != nullptr) {
                std::fprintf(stderr,
                             "[alloc] layer %zu %s failed at columns=%d: arena peak %zu of %zu\n",
                             layer, prefill ? "prefill" : "verify", x.ne[1], work_.peak_used(),
                             work_.capacity());
                std::fflush(stderr);
            }
            throw std::runtime_error("text/layers/" + std::to_string(layer) +
                                     (prefill ? " prefill" : " verify") +
                                     " columns=" + std::to_string(x.ne[1]) + ": " + error.what());
        }
    }
}

void TextContext::run_layers(Tensor& x, Phase ph) {
    NullTap tap;
    run_layers(x, ph, tap);
}

template <class Tap>
PrefillChunkResult
TextContext::prefill_impl(std::span<const int> ids, const TextPrefill* text_prefill,
                          const MultimodalPrefill* multimodal, Tap& tap, bool finalize_at_end) {
    runtime::ExecutionTimingRecorder timing;
    if (ids.empty()) { throw std::invalid_argument("TextContext::prefill requires tokens"); }
    if (ids.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill token count exceeds int32");
    }
    cudaStream_t s           = ctx_.stream;
    const int T              = static_cast<int>(ids.size());
    const int chunk          = static_cast<int>(prefill_chunk_);
    const std::uint32_t base = text_kv_base_;

    if (text_prefill != nullptr) {
        if (multimodal != nullptr || base != text_prefill->begin ||
            text_prefill->token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
            throw std::invalid_argument("text prefill chunk does not match its full prompt");
        }
    }
    if (multimodal != nullptr) {
        if (base != multimodal->begin ||
            multimodal->token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
            throw std::invalid_argument("multimodal prefill suffix does not match its cache base");
        }
        if (multimodal->positions.size() != 3 * multimodal->token_ids.size()) {
            throw std::invalid_argument("multimodal positions must have shape [3,T]");
        }
        if (multimodal->vision == nullptr) {
            throw std::invalid_argument("multimodal prefill requires a Vision session");
        }
        rope_delta_ = multimodal->rope_delta;
    } else if (text_kv_base_ == 0) {
        rope_delta_ = 0;
    }
    ops::set_i32_scalar(io_.rope_delta, rope_delta_, s);

    // Prefix-append prefill continues an existing cache: positions are absolute (start at the
    // resident length) and KV/GDN state is not reset. For a reset prefill base == 0.
    if (static_cast<std::uint64_t>(base) + static_cast<std::uint64_t>(T) >
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill absolute position exceeds int32");
    }
    const int base_i = static_cast<int>(base);

    const std::int64_t base64    = static_cast<std::int64_t>(base);
    const std::int64_t split_abs = prefill_split_frontier_;
    const bool has_split = split_abs > base64 && split_abs <= base64 + static_cast<std::int64_t>(T);
    const int split_rel  = has_split ? static_cast<int>(split_abs - base64) : -1;
    const bool prepare_mtp_prompt = mtp_enabled() && io_.mtp.has_value();
    if (prepare_mtp_prompt &&
        mtp_proposal_extent_ > static_cast<std::uint32_t>(io_.mtp->draft_tokens.ne[0])) {
        throw std::logic_error("MTP proposal extent exceeds the configured draft window");
    }
    int t0 = 0;
    for (; t0 < T;) {
        int len = std::min(chunk, T - t0);
        if (split_rel > 0 && t0 < split_rel && t0 + len > split_rel) { len = split_rel - t0; }
        work_.reset();

        VisionChunk vision_chunk;
        const std::uint32_t prompt_t0 = base + static_cast<std::uint32_t>(t0);
        if (multimodal != nullptr) {
            if (multimodal->vision == nullptr) {
                throw std::logic_error("multimodal prefill has no Vision session");
            }
            vision_chunk =
                multimodal->vision->prepare_chunk(prompt_t0, static_cast<std::uint32_t>(len));
            len = vision_chunk.length;
        }
        // Vision has its own interval. The Program-owned pair measures this text chunk,
        // including MTP alignment or the draft feature consumer submitted below.
        if (prefill_gpu_timer_) { prefill_gpu_timer_->start(); }
        const bool is_last = finalize_at_end && (t0 + len == T);
        nvtx::ScopedRange chunk_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                      static_cast<std::uint64_t>(len));
        ChunkProbe chunk_probe(s);

        {
            std::vector<std::int32_t> local_scatter_indices;
            std::int32_t visual_begin = 0;
            if (vision_chunk.control != nullptr) {
                const auto scatter =
                    std::span<const std::int32_t>(vision_chunk.control->scatter_indices);
                const auto begin = std::lower_bound(scatter.begin(), scatter.end(), prompt_t0);
                const auto end   = std::lower_bound(begin, scatter.end(), prompt_t0 + len);
                const auto count = static_cast<std::int32_t>(end - begin);
                visual_begin     = static_cast<std::int32_t>(begin - scatter.begin());
                local_scatter_indices.resize(static_cast<std::size_t>(count));
                for (std::int32_t i = 0; i < count; ++i) {
                    local_scatter_indices[static_cast<std::size_t>(i)] =
                        begin[i] - static_cast<std::int32_t>(prompt_t0);
                }
            }

            const std::int32_t rope_axes = multimodal != nullptr ? 3 : (rope_delta_ != 0 ? 1 : 0);
            const auto roots             = workspace::text_prefill_roots(
                work_, config_, len, rope_axes,
                static_cast<std::int32_t>(local_scatter_indices.size()));
            Tensor ids_device = roots.ids;
            copy_i32(ids.data() + t0, ids_device, s);

            Tensor positions = roots.positions;
            ops::fill_i32_positions(positions, base_i + t0, s);

            Tensor rope_positions = positions;
            std::vector<std::int32_t> rope_positions_host;
            if (multimodal != nullptr) {
                rope_positions = roots.rope_positions;
                rope_positions_host.resize(static_cast<std::size_t>(3) * len);
                const std::size_t prompt_tokens = multimodal->token_ids.size();
                for (int axis = 0; axis < 3; ++axis) {
                    const auto* src = multimodal->positions.data() +
                                      static_cast<std::size_t>(axis) * prompt_tokens + prompt_t0;
                    std::copy_n(src, len,
                                rope_positions_host.data() + static_cast<std::size_t>(axis) * len);
                }
                copy_i32(rope_positions_host.data(), rope_positions, s);
            } else if (rope_delta_ != 0) {
                rope_positions = roots.rope_positions;
                ops::offset_i32_positions(positions, io_.rope_delta, rope_positions, s);
            }
            ScopedPositions scoped_cache(active_cache_positions_, positions);
            ScopedPositions scoped_rope(active_rope_positions_, rope_positions);
            const auto visible = static_cast<std::uint32_t>(base_i + t0 + len);
            const ops::CausalAttentionExecutionEnvelope chunk_envelope{visible, visible};
            ScopedEnvelope scoped_envelope(active_causal_attention_envelope_, chunk_envelope);

            // This chunk is one row of `len` tokens: every column is live, the record base is the
            // chunk's first absolute position and the record offset is zero. A chunk is never graph
            // captured, so these three are the pass's host values.
            QsaPassBinding qsa;
            begin_qsa_pass(work_, config_, 1, qsa);
            if (qsa.active) {
                std::int32_t columns_host = len;
                std::int32_t base_host    = base_i + t0;
                std::int32_t offset_host  = 0;
                copy_i32(&columns_host, qsa.columns, s);
                copy_i32(&base_host, qsa.base, s);
                copy_i32(&offset_host, qsa.offsets, s);
                copy_i32(&linear_state_source_slot_, qsa.source_slot, s);
                copy_i32(&linear_state_destination_slot_, qsa.destination_slot, s);
            }
            ScopedValue<const Tensor*> qsa_columns_binding(active_qsa_columns_,
                                                           qsa.active ? &qsa.columns : nullptr);
            ScopedValue<const Tensor*> qsa_base_binding(active_qsa_base_positions_,
                                                        qsa.active ? &qsa.base : nullptr);
            ScopedValue<const Tensor*> qsa_offsets_binding(active_qsa_offsets_,
                                                           qsa.active ? &qsa.offsets : nullptr);
            ScopedValue<const Tensor*> qsa_source_binding(active_qsa_source_slots_,
                                                          qsa.active ? &qsa.source_slot : nullptr);
            ScopedValue<const Tensor*> qsa_destination_binding(
                active_qsa_destination_slots_, qsa.active ? &qsa.destination_slot : nullptr);

            Tensor x = roots.residual;
            ops::embedding(ids_device, *embed_, x, s);
            if (!local_scatter_indices.empty()) {
                Tensor indices_device = roots.scatter_indices;
                copy_i32(local_scatter_indices.data(), indices_device, s);
                Tensor embeddings = vision_chunk.embeddings.slice(
                    1, visual_begin, static_cast<std::int32_t>(local_scatter_indices.size()));
                ops::scatter(embeddings, indices_device, x, s);
            }
            chunk_probe.mark(ChunkProbe::kEmbed);
            if constexpr (Tap::enabled) { tap.begin(x); }
            run_layers(x, Phase::Prefill, tap);
            if constexpr (requires { tap.capture_positions(positions, s); }) {
                tap.capture_positions(positions, s);
            }
            chunk_probe.mark(ChunkProbe::kLayers);

            Tensor xf = prefill_hidden_.data != nullptr
                            ? matrix_window(prefill_hidden_, len)
                            : work_.alloc(DType::BF16, {dimension(config_.hidden_size), len});
            final_hidden(x, xf, len);

            if (is_last) {
                Tensor last_xf = xf.slice(1, len - 1, 1);
                Tensor logits  = matrix_window(io_.logits, 1);
                project(last_xf, *lm_head_, logits, work_, s);
                // Set io_.pos to the bonus token's absolute position (base + T) before picking so
                // the sampler RNG is keyed by it (prefill purpose keeps it distinct from the first
                // decode step, which reuses the same io_.pos).
                ops::set_i32_scalar(io_.pos, base_i + T, s);
                ops::set_i32_scalar(io_.rope_pos, base_i + T + rope_delta_, s);
                if (sampling_config_ != nullptr) {
                    ops::sample(logits, io_.token,
                                dimension(parameters_.model.resources().public_token_count),
                                sampling_config_, io_.pos, ops::kSamplePurposePrefill, work_, s);
                } else {
                    ops::argmax(logits, io_.token,
                                dimension(parameters_.model.resources().public_token_count), s);
                }
            }

            chunk_probe.mark(ChunkProbe::kFinal);

            if (prepare_mtp_prompt) {
                const std::uint32_t alignment_tokens =
                    multimodal != nullptr ? static_cast<std::uint32_t>(multimodal->token_ids.size())
                    : text_prefill != nullptr
                        ? static_cast<std::uint32_t>(text_prefill->token_ids.size())
                        : static_cast<std::uint32_t>(T);
                const std::uint32_t alignment_begin =
                    multimodal != nullptr || text_prefill != nullptr
                        ? prompt_t0
                        : static_cast<std::uint32_t>(t0);
                const qwen3_5::MtpAlignmentWindow mtp_window = qwen3_5::plan_mtp_alignment_window(
                    alignment_tokens, alignment_begin, static_cast<std::uint32_t>(len));
                const std::span<const int> alignment_ids =
                    multimodal != nullptr     ? multimodal->token_ids
                    : text_prefill != nullptr ? text_prefill->token_ids
                                              : ids;
                const int prompt_columns =
                    len - static_cast<int>(mtp_window.final_column_uses_generated_token);
                Tensor mtp_ids = work_.alloc(DType::I32, {len});
                if (prompt_columns != 0) {
                    Tensor prompt_mtp_ids = mtp_ids.slice(0, 0, prompt_columns);
                    copy_i32(alignment_ids.data() + mtp_window.shifted_embedding_begin,
                             prompt_mtp_ids, s);
                }
                if (mtp_window.final_column_uses_generated_token) {
                    Tensor generated_mtp_id = mtp_ids.slice(0, len - 1, 1);
                    CUDA_CHECK(cudaMemcpyAsync(generated_mtp_id.data, io_.token.data,
                                               sizeof(std::int32_t), cudaMemcpyDeviceToDevice, s));
                }

                Tensor mtp_input_embeddings;
                const Tensor* mtp_input_embeddings_ptr = nullptr;
                if (multimodal != nullptr) {
                    mtp_input_embeddings =
                        work_.alloc(DType::BF16, {dimension(config_.hidden_size), len});
                    ops::embedding(mtp_ids, *embed_, mtp_input_embeddings, s);
                    if (vision_chunk.control != nullptr) {
                        const qwen3_5::MtpVisualOverlap overlap = qwen3_5::shifted_visual_overlap(
                            vision_chunk.control->scatter_indices, alignment_tokens, mtp_window);
                        if (!overlap.empty()) {
                            Tensor shifted_indices = workspace::visual_scatter_indices(
                                work_, static_cast<std::int32_t>(overlap.size()));
                            qwen3_5::detail::scatter_shifted_visual_embeddings(
                                mtp_input_embeddings, vision_chunk.embeddings, overlap,
                                shifted_indices, s);
                        }
                    }
                    mtp_input_embeddings_ptr = &mtp_input_embeddings;
                }
                if (is_last && mtp_proposal_extent_ != 0) {
                    Tensor logits = matrix_window(io_.logits, 1);
                    Tensor draft0 = io_.mtp->draft_tokens.slice(0, 0, 1);
                    mtp_prefill_chunk(mtp_ids, xf, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, true, &io_.mtp->ar_hidden,
                                      &logits, &draft0);

                    Tensor ar_position = io_.mtp->position.slice(0, 0, 1);
                    ops::set_i32_scalar(ar_position, base_i + T, s);
                    for (int i = 1; i < static_cast<int>(mtp_proposal_extent_); ++i) {
                        Tensor prev_token = io_.mtp->draft_tokens.slice(0, i - 1, 1);
                        Tensor next_token = io_.mtp->draft_tokens.slice(0, i, 1);
                        Tensor next_hidden =
                            work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
                        const auto ar_visible = static_cast<std::uint32_t>(base_i + T + i);
                        const ops::CausalAttentionExecutionEnvelope ar_envelope{ar_visible,
                                                                                ar_visible};
                        mtp_forward_ar_step(prev_token, io_.mtp->ar_hidden, ar_position,
                                            ar_envelope, next_hidden, logits, next_token);
                        CUDA_CHECK(cudaMemcpyAsync(io_.mtp->ar_hidden.data, next_hidden.data,
                                                   io_.mtp->ar_hidden.bytes(),
                                                   cudaMemcpyDeviceToDevice, s));
                        ops::increment_i32_scalar(ar_position, s);
                    }
                } else {
                    mtp_prefill_chunk(mtp_ids, xf, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, false, nullptr, nullptr,
                                      nullptr);
                }
            }
            chunk_probe.mark(ChunkProbe::kMtp);

            if (split_rel > 0 && t0 + len == split_rel &&
                rewrite_checkpoint_hidden_output_ != nullptr) {
                require_tensor_shape(*rewrite_checkpoint_hidden_output_, DType::BF16,
                                     {dimension(config_.hidden_size), 1},
                                     "rewrite checkpoint hidden output");
                const Tensor checkpoint_hidden = xf.slice(1, len - 1, 1);
                CUDA_CHECK(cudaMemcpyAsync(rewrite_checkpoint_hidden_output_->data,
                                           checkpoint_hidden.data, checkpoint_hidden.bytes(),
                                           cudaMemcpyDeviceToDevice, s));
            }
        }

        if constexpr (requires { tap.consume_prefill_chunk(len, false); }) {
            work_.reset();
            tap.consume_prefill_chunk(len, split_rel > 0 && t0 + len == split_rel);
        }
        chunk_probe.mark(ChunkProbe::kDraftSink);
        chunk_probe.report(len);

        t0 += len;
        break;
    }

    prefill_split_frontier_ = -1;

    if (prefill_gpu_timer_) { prefill_gpu_timer_->record_stop(); }
    timing.begin_wait();
    ctx_.synchronize();
    timing.end_wait();
    if (prefill_gpu_timer_) {
        timing.include({.gpu_elapsed_ns = static_cast<std::uint64_t>(
                            static_cast<double>(prefill_gpu_timer_->elapsed_ms()) * 1.0e6 + 0.5)});
    }
    work_.reset();
    return PrefillChunkResult{.processed_tokens = static_cast<std::uint32_t>(t0),
                              .finalized        = finalize_at_end && t0 == T,
                              .timing           = timing.finish()};
}

PrefillChunkResult TextContext::prefill_chunk(std::span<const int> full_ids, std::uint32_t begin,
                                              std::uint32_t nominal_length, bool finalize_at_end) {
    if (begin >= full_ids.size() || nominal_length == 0 ||
        nominal_length > full_ids.size() - begin) {
        throw std::invalid_argument("text prefill chunk is outside the prompt");
    }
    const TextPrefill text_prefill{full_ids, begin};
    NullTap tap;
    return prefill_impl(full_ids.subspan(begin, nominal_length), &text_prefill, nullptr, tap,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(std::span<const int> full_ids, std::uint32_t begin,
                                              std::uint32_t nominal_length, bool finalize_at_end,
                                              DFlashFeatureSink& sink) {
    if (begin >= full_ids.size() || nominal_length == 0 ||
        nominal_length > full_ids.size() - begin) {
        throw std::invalid_argument("text prefill chunk is outside the prompt");
    }
    const TextPrefill text_prefill{full_ids, begin};
    return prefill_impl(full_ids.subspan(begin, nominal_length), &text_prefill, nullptr, sink,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(const qwen3_5::PreparedPromptData& input,
                                              std::uint32_t begin, std::uint32_t nominal_length,
                                              VisionPrefillSession& vision, bool finalize_at_end) {
    if (begin >= input.token_ids.size() || nominal_length == 0 ||
        nominal_length > input.token_ids.size() - begin) {
        throw std::invalid_argument("multimodal prefill chunk is outside the prompt");
    }
    const std::span<const int> tokens(input.token_ids);
    const MultimodalPrefill multimodal{tokens, input.positions, &vision, begin, input.rope_delta};
    NullTap tap;
    return prefill_impl(tokens.subspan(begin, nominal_length), nullptr, &multimodal, tap,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(const qwen3_5::PreparedPromptData& input,
                                              std::uint32_t begin, std::uint32_t nominal_length,
                                              VisionPrefillSession& vision, bool finalize_at_end,
                                              DFlashFeatureSink& sink) {
    if (begin >= input.token_ids.size() || nominal_length == 0 ||
        nominal_length > input.token_ids.size() - begin) {
        throw std::invalid_argument("multimodal prefill chunk is outside the prompt");
    }
    const std::span<const int> tokens(input.token_ids);
    const MultimodalPrefill multimodal{tokens, input.positions, &vision, begin, input.rope_delta};
    return prefill_impl(tokens.subspan(begin, nominal_length), nullptr, &multimodal, sink,
                        finalize_at_end);
}

} // namespace ninfer::models::qwen3_5::execution
