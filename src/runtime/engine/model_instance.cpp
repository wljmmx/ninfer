#include "runtime/engine/model_instance.h"
#include "artifact/expert_files.h"
#include "artifact/expert_reader.h"
#include "artifact/reader.h"
#include "artifact/formats.h"
#include "core/expert_cache.h"
#include "core/startup.h"
#include "models/qwen3_5/execution/expert_stream.h"
#include "models/qwen3_5/execution/host_expert_reader.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/measurement.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <set>
#include <stdexcept>
#include <utility>

namespace ninfer::runtime {
namespace {
using Clock = std::chrono::steady_clock;

using models::qwen3_5::execution::ExpertStream;
using models::qwen3_5::execution::RoutedMoeParameters;

// port (sm_89): collect the routed banks that were declared as streamed.
void collect_streams(const models::qwen3_5::execution::FfnParameters& ffn,
                     std::vector<ExpertStream*>& out) {
    if (const auto* moe = std::get_if<RoutedMoeParameters>(&ffn); moe != nullptr && moe->stream) {
        out.push_back(moe->stream.get());
    }
}

void attach_expert_cache(ModelInstance& instance, const EngineOptions& options,
                         DeviceContext& device) {
    std::vector<ExpertStream*> streams;
    for (const auto& layer : instance.parameters.text.layers) { collect_streams(layer.ffn, streams); }
    if (instance.parameters.mtp) {
        collect_streams(instance.parameters.mtp->ffn, streams);
    }
    if (streams.empty()) {
        // A dense model has no routed bank; the switch then simply does nothing.
        return;
    }
    // port (sm_89): the streamed tier reads either the artifact or a retained Host replica.
    // port (sm_89): the cache keeps a raw pointer, so the source must outlive it - it lives on
    // the instance, never on this stack frame.
    if (options.expert_host_copy) {
        instance.expert_host_source =
            std::make_unique<models::qwen3_5::execution::HostExpertReader>(*instance.model);
    } else {
        instance.expert_source = std::make_unique<artifact::ArtifactExpertReader>(options.artifact_path);
    }
    const auto* source = options.expert_host_copy
                             ? static_cast<const ExpertBlobReader*>(instance.expert_host_source.get())
                             : static_cast<const ExpertBlobReader*>(instance.expert_source.get());

    std::vector<ExpertBankSpec> specs;
    specs.reserve(streams.size());
    for (const auto* stream : streams) {
        specs.push_back({{stream->gate_up_expert_geometry()}, {stream->down_expert_geometry()},
                         stream->count(), false});
    }
    const auto probe = describe_expert_cache(specs, {.capacity_bytes = 256});
    // port (sm_89): the pool holds a whole number of slots, one per expert regardless of bank,
    // because an evicted slot is handed to whichever bank needs it next.
    const auto entry    = probe.max_entry_bytes;
    const auto capacity = options.expert_cache_bytes / entry * entry;
    if (capacity == 0) {
        throw std::invalid_argument("expert cache holds less than one expert entry (" +
                                    std::to_string(entry) + " bytes)");
    }
    if (capacity % 256 != 0) {
        throw std::invalid_argument("expert cache capacity must be 256-byte aligned");
    }
    const auto requirements = describe_expert_cache(specs, {.capacity_bytes = capacity});

    std::vector<ExpertFileBank> files;
    files.reserve(streams.size());
    // port (sm_89): the bank description must match the reader that was created above. A host
    // tier has no artifact reader at all, so asking it for one dereferences a null pointer.
    const bool host_tier = instance.expert_host_source != nullptr;
    for (std::size_t i = 0; i < streams.size(); ++i) {
        const auto* stream = streams[i];
        const auto gate_up = artifact::ObjectHandle{stream->gate_up_object()};
        const auto down    = artifact::ObjectHandle{stream->down_object()};
        const auto gu_rows = static_cast<std::uint32_t>(stream->gate_up_layout().region.rows);
        const auto dn_rows = static_cast<std::uint32_t>(stream->down_layout().region.rows);
        files.push_back(host_tier
                            ? artifact::describe_fused_expert_host(
                                  gate_up, down, stream->gate_up_parent(), stream->down_parent(),
                                  stream->count(), gu_rows, dn_rows, requirements.banks[i])
                            : artifact::describe_fused_expert_files(
                                  instance.expert_source->reader(), gate_up, down,
                                  stream->gate_up_parent(), stream->down_parent(), stream->count(),
                                  gu_rows, dn_rows, requirements.banks[i]));
    }
    instance.expert_cache = std::make_unique<ExpertCache>(
        specs, requirements, device, std::span<const ExpertBankPair>{}, files,
        source);
    for (std::size_t i = 0; i < streams.size(); ++i) {
        streams[i]->attach(*instance.expert_cache, static_cast<std::uint32_t>(i),
                           requirements.banks[i]);
    }
    // port (sm_89): one line that makes the expert tier observable without a debugger.
    std::fprintf(stderr, "[experts] banks=%zu slot=%llu windows=%llu pool=%lluMiB host=%d\n",
                 streams.size(),
                 static_cast<unsigned long long>(instance.expert_cache->slot_bytes()),
                 static_cast<unsigned long long>(instance.expert_cache->windows()),
                 static_cast<unsigned long long>(instance.expert_cache->capacity_bytes() >> 20),
                 int(instance.expert_host_source != nullptr));
}

void validate_options(const EngineOptions& options) {
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    if (options.artifact_path.extension() != ".ninfer") {
        throw std::invalid_argument("NInfer accepts only .ninfer artifacts");
    }
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit:
        if (options.kv_capacity.explicit_tokens == 0) {
            throw std::invalid_argument("Engine explicit kv_capacity must be nonzero");
        }
        if (options.kv_capacity.automatic_headroom_bytes != 0) {
            throw std::invalid_argument(
                "Engine explicit kv_capacity must not carry automatic headroom");
        }
        break;
    case KvCapacityMode::Automatic:
        if (options.kv_capacity.explicit_tokens != 0) {
            throw std::invalid_argument(
                "Engine automatic kv_capacity must not carry explicit tokens");
        }
        break;
    default:
        throw std::invalid_argument("Engine kv_capacity mode is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0 || options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine pending request capacity and timeout must be nonzero");
    }
    if (options.enable_vision && options.media_live_bytes == 0) {
        throw std::invalid_argument(
            "Engine media_live_bytes must be nonzero when Vision is enabled");
    }
    if (options.media_preprocess_threads > 64) {
        throw std::invalid_argument("Engine media_preprocess_threads must be in [0,64]");
    }
}

std::size_t current_free_device_bytes(bool wddm_evictable_budget = false,
                                     std::size_t weights_bytes = 0) {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
#if defined(_WIN32)
    // Allow WDDM to evict background apps down to a 512 MiB non-evictable DWM display floor.
    constexpr std::size_t kMinDwmHeadroom = 512ULL * 1024ULL * 1024ULL;
    if (wddm_evictable_budget && total_bytes > weights_bytes + kMinDwmHeadroom) {
        const std::size_t evictable_free = total_bytes - weights_bytes - kMinDwmHeadroom;
        return std::max(free_bytes, evictable_free);
    }
#endif
    return free_bytes;
}

} // namespace

EngineOptions normalize_engine_options(EngineOptions options) {
    // port (sm_89): a streamed expert bank uploads and synchronizes inside each layer, which a
    // captured graph cannot contain. The environment names the pool for command-line tools.
    if (const char* host = std::getenv("NINFER_EXPERT_HOST"); host != nullptr && *host && *host != 0x30) {
        options.expert_host_copy = true;
    }
    if (options.expert_cache_bytes == 0) {
        if (const char* value = std::getenv("NINFER_EXPERT_CACHE_MB"); value != nullptr && *value) {
            const auto mib = std::strtoull(value, nullptr, 10);
            options.expert_cache_bytes = static_cast<std::size_t>(mib) << 20;
        }
    }
    if (options.expert_cache_bytes != 0) { options.use_cuda_graph = false; }
    switch (options.purpose) {
    case EnginePurpose::Generation:
        break;
    case EnginePurpose::CausalScoring:
        options.max_concurrency      = 1;
        options.max_pending_requests = 1;
        options.prefill_chunk        = 1024;
        options.kv_capacity          = KvCapacityPolicy::explicit_capacity(options.max_context);
        options.speculative          = {};
        options.enable_vision        = false;
        options.use_cuda_graph       = false;
        // appendix FF (M3 3.2b): everything here is forced for CausalScoring except the Host KV
        // capacity, which a caller may legitimately ask for (that is how the demotion path is
        // exercised). Zero - what every existing caller leaves it at - keeps the old behaviour.
        // The field is read while the temporary is built, so no local (and no default: jump) is needed.
        options.context_cache = ContextCacheOptions{
            .enabled             = false,
            .device_state_slots  = 0,
            .host_capacity_bytes = options.context_cache.host_capacity_bytes.value_or(0)};
        break;
    default:
        throw std::invalid_argument("Engine purpose is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }

    ContextCacheOptions& cache      = options.context_cache;
    const std::uint32_t concurrency = options.max_concurrency;
    cache.device_state_slots        = cache.device_state_slots.value_or(concurrency);
    const std::uint64_t total_device_state_slots =
        static_cast<std::uint64_t>(concurrency) + *cache.device_state_slots;
    if (total_device_state_slots > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context cache Device state capacity exceeds uint32");
    }
    return options;
}

ModelInstance::ModelInstance(std::unique_ptr<models::qwen3_5::Model> source,
                             const EngineOptions& options)
    : model(std::move(source)), parameters(*model),
      frontend(models::qwen3_5::make_frontend(
          model->resources(), {.chat_template_path       = options.chat_template_path,
                               .grammar_cache_bytes      = options.grammar_cache_bytes,
                               .architecture             = model->config().text.architecture,
                               .vision_enabled           = options.enable_vision,
                               .max_context              = options.max_context,
                               .media_cache_bytes        = options.media_cache_bytes,
                               .media_live_bytes         = options.media_live_bytes,
                               .media_preprocess_threads = options.media_preprocess_threads})),
      capacity(options.max_context) {}

ModelInstance::~ModelInstance() = default;

ConstructedModel construct_model(EngineOptions& options, DeviceContext& device) {
    validate_options(options);
    const auto start = Clock::now();
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    artifact::Reader reader(options.artifact_path);
    inspect.complete();
    StartupPhaseScope binding(options.startup_observer, StartupPhase::TargetPlan);
    auto plan = models::qwen3_5::plan_load(reader, models::load_options(options));
    binding.complete();
    auto model =
        models::qwen3_5::materialize_model(std::move(plan), device, &options.startup_observer);
    device.synchronize();
    StartupPhaseScope frontend(options.startup_observer, StartupPhase::FrontendInitialize);
    auto instance = std::make_unique<ModelInstance>(std::move(model), options);
    frontend.complete();
    if (options.expert_cache_bytes != 0) {
        attach_expert_cache(*instance, options, device);
        device.synchronize();
    }
    StartupPhaseScope planning(options.startup_observer, StartupPhase::TargetFinalize);
    const auto signature = models::qwen3_5::prefill_signature(*instance->model);
    auto context_cost    = resolve_context_machine_cost(
        {.hardware_class =
                context_cost_hardware_class(device.props.name, device.props.major, device.props.minor),
            .prefill_signature = signature},
        options.context_cost.preset_path);
    auto planner    = models::qwen3_5::make_sequence_planner(instance->parameters, device, options);
    const auto capacity_curve = planner.capacity_curve();
    auto resolution = resolve_kv_capacity(options.kv_capacity, planner.capacity_curve(),
                                          current_free_device_bytes(options.wddm_evictable_budget));
    auto sequence   = std::move(planner).finalize(resolution.main_page_groups);
    if (sequence.device_reservation_bytes() != resolution.runtime_reservation_bytes ||
        sequence.kv_capacity() != resolution.resolved_tokens) {
        throw std::logic_error("resolved KV capacity does not match the finalized Program plan");
    }
    options.context_cache.host_capacity_bytes = sequence.host_capacity_bytes();
    // appendix IK: decide the eviction contract here - the only place where the resolved arena,
    // the host budget, the per-lane requirement and the KV dtype are all known. The environment
    // still wins when it names the contract, so every recorded experiment is reproducible.
    {
        const bool env_named = models::qwen3_5::kv_eviction_window_from_environment_enabled();
        const std::uint32_t page_tokens = capacity_curve.main_page_tokens;
        const std::uint32_t per_lane_pages =
            (page_tokens != 0 && options.max_context != 0)
                ? (options.max_context + page_tokens - 1U) / page_tokens
                : 0U;
        const std::uint32_t lanes = per_lane_pages != 0
                                        ? resolution.main_page_groups / per_lane_pages
                                        : 0U;
        const std::uint64_t retained_percent =
            (per_lane_pages != 0 && options.max_concurrency != 0)
                ? 100ULL * (resolution.main_page_groups / options.max_concurrency) /
                      per_lane_pages
                : 0ULL;
        const std::uint64_t host_needed =
            static_cast<std::uint64_t>(per_lane_pages) * options.max_concurrency *
            capacity_curve.bytes_per_additional_main_page_group;
        std::uint32_t reasons = 0;
        if (lanes < options.max_concurrency) { reasons |= 1U; }
        // The <2% band starts at 58% retention (appendices GV/GW/GX).
        if (retained_percent >= 58ULL) { reasons |= 2U; }
        if (sequence.host_capacity_bytes() >= host_needed) { reasons |= 4U; }
        if (models::qwen3_5::kv_absent_pages_supported_for(options.kv_cache)) { reasons |= 8U; }
        const bool supported = (reasons & 8U) != 0;
        const bool wanted = env_named || options.evict_contract == EvictionContract::On ||
                            (options.evict_contract == EvictionContract::Auto &&
                             (reasons & 7U) == 7U);
        const bool effective = wanted && supported;
        if (effective && !env_named) {
            models::qwen3_5::install_kv_eviction_window(options.evict_sink_pages,
                                                        options.evict_window_pages);
        }
        resolution.evict_contract_effective = effective ? 1U : 0U;
        resolution.evict_contract_reasons   = reasons;
    }
    instance->kv_capacity_resolution          = resolution;
    planning.complete();
    StartupPhaseScope program(options.startup_observer, StartupPhase::ProgramInitialize);
    instance->program = models::qwen3_5::create_program(instance->parameters, std::move(sequence),
                                                        device, options.startup_observer);
    device.synchronize();
    program.complete();
    instance->kv_capacity_resolution.available_after_startup_bytes = current_free_device_bytes(options.wddm_evictable_budget);
    const auto& stats = instance->model->storage_stats();
    LoadSummary summary;
    summary.architecture = models::architecture_name(instance->model->config().text.architecture);
    summary.model_name   = instance->model->info().name;
    summary.prefill_signature = signature;
    std::set<std::string> formats;
    for (const auto& weight : instance->model->weight_data()) {
        for (const auto& part : weight.view.parts) {
            formats.emplace(artifact::format_name(part.parent->geometry.format));
        }
    }
    summary.weight_formats.assign(formats.begin(), formats.end());
    summary.load_seconds         = std::chrono::duration<double>(Clock::now() - start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.read_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.device_object_count  = stats.device_object_count;
    summary.host_object_count    = stats.host_object_count;
    summary.context_cost         = std::move(context_cost.summary);
    summary.kv_main_page_groups                = resolution.main_page_groups;
    summary.kv_max_main_page_groups            = resolution.maximum_main_page_groups;
    summary.kv_resolved_tokens                 = resolution.resolved_tokens;
    summary.kv_runtime_reservation_bytes       = resolution.runtime_reservation_bytes;
    summary.kv_planned_slack_bytes             = resolution.planned_slack_bytes;
    summary.kv_available_after_weights_bytes   = resolution.available_after_weights_bytes;
    summary.kv_available_after_startup_bytes =
        instance->kv_capacity_resolution.available_after_startup_bytes;
    summary.kv_bytes_per_additional_page_group = resolution.bytes_per_additional_main_page_group;
    return {std::move(instance), std::move(summary), std::move(context_cost.model)};
}

} // namespace ninfer::runtime
