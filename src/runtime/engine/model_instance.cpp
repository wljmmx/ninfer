#include "runtime/engine/model_instance.h"
#include "artifact/expert_files.h"
#include "artifact/expert_reader.h"
#include "artifact/reader.h"
#include "artifact/formats.h"
#include "core/expert_cache.h"
#include "core/startup.h"
#include "models/qwen3_5/execution/expert_stream.h"
#include "models/qwen3_5/execution/host_expert_reader.h"
#include "models/qwen3_5/execution/host_expert_residency.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/measurement.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <set>
#include <stdexcept>
#include <utility>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace ninfer::runtime {
namespace {

std::uint64_t host_expert_arena_bytes(std::uint64_t want_bytes);
using Clock = std::chrono::steady_clock;

// port (sm_89): a sentinel that asks the engine to size the expert pool from free device memory.
constexpr std::size_t kExpertCacheAuto = std::numeric_limits<std::size_t>::max();

using models::qwen3_5::execution::ExpertStream;
using models::qwen3_5::execution::RoutedMoeParameters;

// port (sm_89): collect the routed banks that were declared as streamed.
void collect_streams(const models::qwen3_5::execution::FfnParameters& ffn,
                     std::vector<ExpertStream*>& out) {
    if (const auto* moe = std::get_if<RoutedMoeParameters>(&ffn); moe != nullptr && moe->stream) {
        out.push_back(moe->stream.get());
    }
}

// Set NINFER_EXPERT_DEBUG to trace how far expert-tier construction gets before a startup fault.
// The line is flushed so the last stage survives an access violation.
void expert_debug(const char* stage, std::uint64_t first = 0, std::uint64_t second = 0) {
    if (std::getenv("NINFER_EXPERT_DEBUG") == nullptr) { return; }
    std::fprintf(stderr, "[expert-debug] %s %llu %llu\n", stage,
                 static_cast<unsigned long long>(first), static_cast<unsigned long long>(second));
    std::fflush(stderr);
}

// Where the routed experts ended up. Reported once at startup so a run can be read without a
// debugger, and reused by the placement policy once it starts tuning at runtime.
struct ExpertPlacement {
    std::uint64_t total_experts = 0;
    std::uint64_t gpu_experts   = 0;
    std::uint64_t host_experts  = 0;
    std::uint64_t file_experts  = 0;
    std::uint64_t slot_bytes    = 0;
};

// Placement policy: split the free device budget between the KV arena and the expert pool.
// The arena is satisfied first because every decode step touches the whole active context, while
// the pool only serves the routed top-k; the pool then takes what is left. P12.4 replaces this
// with a measured cost model - keeping the split in one place makes that swap local.
struct DeviceSplit {
    std::size_t pool_bytes   = 0;
    std::size_t safety_bytes = 0;
};

[[nodiscard]] DeviceSplit split_device_budget(std::size_t free_bytes,
                                              std::size_t kv_reservation_bytes) {
    constexpr std::size_t kPoolSafetyBytes = 1ULL << 30;
    const auto pledged = kv_reservation_bytes + kPoolSafetyBytes;
    return {free_bytes > pledged ? free_bytes - pledged : std::size_t{0}, kPoolSafetyBytes};
}

[[nodiscard]] bool kv_capacity_shortage(const std::invalid_argument& error) noexcept;

ExpertPlacement attach_expert_cache(ModelInstance& instance, const EngineOptions& options,
                                    DeviceContext& device, std::size_t kv_reserve_bytes) {
    ExpertPlacement placement;
    std::vector<ExpertStream*> streams;
    for (const auto& layer : instance.parameters.text.layers) { collect_streams(layer.ffn, streams); }
    if (instance.parameters.mtp) {
        collect_streams(instance.parameters.mtp->ffn, streams);
    }
    if (streams.empty()) {
        // A dense model has no routed bank; the switch then simply does nothing.
        return {};
    }
    expert_debug("streams", streams.size());
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
    const auto entry = probe.max_entry_bytes;
    auto requested   = options.expert_cache_bytes;
    if (requested == kExpertCacheAuto) {
        // The KV arena is planned first and already knows its exact reservation, so the pool takes
        // everything above it instead of guessing a fixed 6 GiB: on a 256K/k8v4 run that guess
        // left 3.8 GiB idle while 28% of the experts still streamed from the host.
        std::size_t free_bytes = 0, total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess || free_bytes == 0) {
            throw std::runtime_error("expert cache auto sizing cannot read device memory");
        }
        // Graph capture and the staging buffers still need room after the pool takes its share.
        const auto split = split_device_budget(free_bytes, kv_reserve_bytes);
        requested        = split.pool_bytes;
    }
    // Whole windows only: a partial window is dead space and would make the slot count odd.
    std::uint64_t widest = 0;
    for (const auto& spec : specs) { widest = std::max<std::uint64_t>(widest, spec.count); }
    auto capacity = requested / entry * entry;
    // A pool that cannot hold a whole layer keeps whatever slots fit and the host tier covers the
    // rest of the layer; rounding to zero here would refuse a pool that is perfectly usable.
    if (widest != 0 && capacity / entry >= widest) {
        capacity = capacity / (entry * widest) * (entry * widest);
    }
    if (capacity == 0) {
        throw std::invalid_argument("expert cache holds less than one expert window (" +
                                    std::to_string(entry * widest) + " bytes)");
    }
    expert_debug("capacity", capacity, entry);
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
    expert_debug("files", files.size(), requirements.banks.size());
    instance.expert_cache = std::make_unique<ExpertCache>(
        specs, requirements, device, std::span<const ExpertBankPair>{}, files,
        source);
    for (std::size_t i = 0; i < streams.size(); ++i) {
        streams[i]->attach(*instance.expert_cache, static_cast<std::uint32_t>(i),
                           requirements.banks[i]);
    }
    expert_debug("attached", instance.expert_cache->capacity_bytes(),
                 instance.expert_cache->windows());
    // port (sm_89): one line that makes the expert tier observable without a debugger.
    // How much the host tier could usefully hold: every expert the Device pool does not keep
    // resident. The arena then takes the smaller of that and a quarter of free RAM, so RAM is used
    // up to what the model needs and only the remainder falls through to the artifact.
    std::uint64_t host_want = 0;
    for (const auto& bank : requirements.banks) {
        if (!bank.resident) {
            host_want += static_cast<std::uint64_t>(bank.count) * instance.expert_cache->slot_bytes();
        }
    }
    std::fprintf(stderr, "[experts] banks=%zu slot=%llu windows=%llu pool=%lluMiB host=%d arena=%lluMiB want=%lluMiB\n",
                 streams.size(),
                 static_cast<unsigned long long>(instance.expert_cache->slot_bytes()),
                 static_cast<unsigned long long>(instance.expert_cache->windows()),
                 static_cast<unsigned long long>(instance.expert_cache->capacity_bytes() >> 20),
                 int(instance.expert_host_source != nullptr),
                 static_cast<unsigned long long>(host_expert_arena_bytes(host_want) >> 20),
                 static_cast<unsigned long long>(host_want >> 20));
    // A pool that cannot hold a layer hands the remainder to the host tier: one arena, the same
    // uniform slot images the Device pool uses, filled on demand.
    bool short_pool = false;
    for (const auto& bank : requirements.banks) {
        if (!bank.resident && instance.expert_cache->resident_capacity() < bank.count) {
            short_pool = true;
        }
    }
    if (short_pool && !streams.empty()) {
        std::vector<models::qwen3_5::execution::HostExpertBankDesc> descs;
        descs.reserve(streams.size());
        for (std::size_t i = 0; i < streams.size(); ++i) {
            descs.push_back({requirements.banks[i], streams[i]->gate_up_layout(),
                             streams[i]->down_layout(), streams[i]->gate_up_parent(),
                             streams[i]->down_parent()});
        }
        auto residency = std::make_unique<models::qwen3_5::execution::HostExpertResidency>(
            descs, files, source, host_want);
        for (std::size_t i = 0; i < streams.size(); ++i) {
            streams[i]->attach_host(*residency);
        }
        std::fprintf(stderr, "[experts] host tier slots=%u slot_bytes=%llu\n",
                     residency->slot_capacity(),
                     static_cast<unsigned long long>(residency->slot_bytes()));
        placement.host_experts =
            static_cast<std::uint64_t>(residency->slot_capacity());
        instance.expert_host_residency = std::move(residency);
    }
    placement.slot_bytes = instance.expert_cache->slot_bytes();
    placement.gpu_experts =
        placement.slot_bytes != 0 ? instance.expert_cache->capacity_bytes() / placement.slot_bytes
                                  : 0;
    for (const auto& bank : requirements.banks) { placement.total_experts += bank.count; }
    // Whatever the pool does not keep is served by the cache's source: the retained Host replica
    // when one exists, the artifact otherwise. The residency arena above is an extra cache in
    // front of that source, not a separate tier.
    const auto streamed = placement.total_experts > placement.gpu_experts
                              ? placement.total_experts - placement.gpu_experts
                              : 0;
    placement.host_experts = instance.expert_host_source != nullptr ? streamed : 0;
    placement.file_experts = instance.expert_host_source != nullptr ? 0 : streamed;
    return placement;
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

std::uint64_t current_free_host_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status) != 0) { return status.ullAvailPhys; }
#endif
    return 0;
}

// Host-side arena for experts the Device pool cannot hold. It takes what the caller says the host
// tier could usefully hold, capped by a share of free RAM so one process cannot claim the box.
// Anything past that streams in from the artifact: RAM first, SSD only when it does not fit.
// NINFER_EXPERT_HOST_ARENA_MB overrides the result when a fixed budget is wanted.
std::uint64_t host_expert_arena_bytes(std::uint64_t want_bytes) {
    const auto share = [&] {
        const auto free_bytes = current_free_host_bytes();
        return free_bytes == 0 ? want_bytes : free_bytes / 4;
    }();
    std::uint64_t budget = std::min<std::uint64_t>(want_bytes, share);
    if (const char* value = std::getenv("NINFER_EXPERT_HOST_ARENA_MB"); value != nullptr && *value) {
        const auto mib = std::strtoull(value, nullptr, 10);
        // An explicit budget replaces the RAM share rather than tightening it, so a run can ask
        // for the whole model when the machine has room.
        if (mib != 0) { budget = std::min<std::uint64_t>(want_bytes, mib << 20); }
    }
    return budget;
}

std::size_t current_free_device_bytes(bool wddm_evictable_budget = false,
                                     std::size_t weights_bytes = 0) {    std::size_t free_bytes  = 0;
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
            if (std::strcmp(value, "auto") == 0) {
                options.expert_cache_bytes = kExpertCacheAuto;
            } else {
                const auto mib = std::strtoull(value, nullptr, 10);
                options.expert_cache_bytes = static_cast<std::size_t>(mib) << 20;
            }
        }
    }
    if (options.expert_cache_bytes != 0) { options.use_cuda_graph = false; }
    // port (sm_89): a streamed expert tier makes every request wait for its window to load, so
    // the queue drains at a fraction of the resident rate. Scale the pending deadline with the
    // number of requests instead of failing them at the resident-speed timeout.
    if (options.expert_cache_bytes != 0 && options.max_concurrency > 1) {
        options.pending_timeout_ms *= options.max_concurrency;
    }
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

std::uint64_t ModelInstance::resize_expert_pool(std::uint64_t target_bytes) {
    if (!expert_cache) { return 0; }
    const auto before = expert_cache->capacity_bytes();
    const auto after  = expert_cache->resize(target_bytes);
    if (const char* debug = std::getenv("NINFER_EXPERT_DEBUG"); debug != nullptr && *debug) {
        std::size_t free_bytes = 0, total_bytes = 0;
        (void)cudaMemGetInfo(&free_bytes, &total_bytes);
        std::fprintf(stderr,
                     "[expert-pool] capacity %.1f -> %.1f MiB windows=%llu free=%.1f MiB\n",
                     static_cast<double>(before) / 1048576.0,
                     static_cast<double>(after) / 1048576.0,
                     static_cast<unsigned long long>(expert_cache->windows()),
                     static_cast<double>(free_bytes) / 1048576.0);
    }
    return after;
}

std::uint64_t ModelInstance::expert_pool_capacity() const noexcept {
    return expert_cache ? expert_cache->capacity_bytes() : 0;
}

namespace {

ConstructedModel construct_model_once(EngineOptions& options, DeviceContext& device) {
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
    StartupPhaseScope planning(options.startup_observer, StartupPhase::TargetFinalize);
    const auto signature = models::qwen3_5::prefill_signature(*instance->model);
    auto context_cost    = resolve_context_machine_cost(
        {.hardware_class =
                context_cost_hardware_class(device.props.name, device.props.major, device.props.minor),
            .prefill_signature = signature},
        options.context_cost.preset_path);
    auto planner    = models::qwen3_5::make_sequence_planner(instance->parameters, device, options);
    const auto capacity_curve = planner.capacity_curve();
    auto resolution =
        [&]() -> KvCapacityResolution {
        const auto weights_bytes = instance->model->storage_stats().device_capacity_bytes;
        const auto conservative = current_free_device_bytes(
            options.wddm_evictable_budget, weights_bytes);
        try {
            return resolve_kv_capacity(options.kv_capacity, planner.capacity_curve(),
                                       conservative);
        } catch (const std::invalid_argument& error) {
            // The instantaneous free count is not what the driver can actually give back. When
            // the plan misses it, fall back once to the WDDM evictable budget (total minus the
            // weights this process must keep, minus a display floor). That starts the run, but
            // the pages it forces the driver to evict do cost decode throughput, so it is a
            // second attempt rather than the default.
            if (!kv_capacity_shortage(error) ||
                std::getenv("NINFER_NO_WDDM_EVICTABLE") != nullptr) {
                throw;
            }
            const auto optimistic = current_free_device_bytes(true, weights_bytes);
            if (optimistic <= conservative) { throw; }
            std::fprintf(stderr,
                         "[budget] device memory is tight (%zu bytes free); planning against the "
                         "WDDM evictable budget (%zu bytes) - paging may slow decode\n",
                         conservative, optimistic);
            std::fflush(stderr);
            return resolve_kv_capacity(options.kv_capacity, planner.capacity_curve(), optimistic);
        }
    }();
    // The KV plan is settled before the expert pool is sized, so the pool can take everything the
    // arena does not need instead of holding back a fixed amount.
    if (options.expert_cache_bytes != 0) {
        const auto placement = attach_expert_cache(*instance, options, device,
                                                   resolution.runtime_reservation_bytes);
        device.synchronize();
        if (placement.total_experts != 0) {
            constexpr double kMib = 1024.0 * 1024.0;
            std::fprintf(stderr,
                         "[budget] experts: gpu=%llu (%.0f MiB) host=%llu (%.0f MiB) "
                         "file=%llu of %llu, slot %.0f KiB; kv: %u pages, %.0f MiB reserved\n",
                         static_cast<unsigned long long>(placement.gpu_experts),
                         static_cast<double>(placement.gpu_experts * placement.slot_bytes) / kMib,
                         static_cast<unsigned long long>(placement.host_experts),
                         static_cast<double>(placement.host_experts * placement.slot_bytes) / kMib,
                         static_cast<unsigned long long>(placement.file_experts),
                         static_cast<unsigned long long>(placement.total_experts),
                         static_cast<double>(placement.slot_bytes) / 1024.0,
                         resolution.main_page_groups,
                         static_cast<double>(resolution.runtime_reservation_bytes) / kMib);
            std::fflush(stderr);
        }
    }
    auto sequence = std::move(planner).finalize(resolution.main_page_groups);
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

// A capacity plan that does not fit the resident weights is not a broken configuration: the same
// artifact runs with the routed experts on the host. The device allocations come straight from
// cudaMalloc (core/arena.cu), so unwinding the first attempt returns every byte to the driver and
// the retry plans against the full budget again.
[[nodiscard]] bool kv_capacity_shortage(const std::invalid_argument& error) noexcept {
    const char* what = error.what();
    return what != nullptr && std::strstr(what, "Engine runtime reservation requires") != nullptr;
}

} // namespace

ConstructedModel construct_model(EngineOptions& options, DeviceContext& device) {
    try {
        return construct_model_once(options, device);
    } catch (const std::invalid_argument& error) {
        if (options.expert_cache_bytes != 0 || options.expert_host_copy ||
            !kv_capacity_shortage(error) || std::getenv("NINFER_NO_EXPERT_FALLBACK") != nullptr) {
            throw;
        }
        // The routed experts can live in host RAM or be read from the artifact on demand. RAM is
        // the faster tier (~15 against ~9 tok/s here), but the retained replica needs the whole
        // expert payload resident, so the file tier takes over when RAM cannot hold it. The
        // payload is about three quarters of the artifact (the dense part is the rest), and a
        // quarter of the free RAM stays with the rest of the system.
        std::error_code size_error;
        const auto artifact_bytes   = std::filesystem::file_size(options.artifact_path, size_error);
        const auto free_host_bytes  = current_free_host_bytes();
        const auto expert_estimate  = artifact_bytes / 4 * 3;
        bool ram_fits = !size_error && artifact_bytes != 0 &&
                        (free_host_bytes == 0 || free_host_bytes / 4 * 3 >= expert_estimate);
        if (const char* forced = std::getenv("NINFER_EXPERT_FALLBACK"); forced != nullptr && *forced) {
            ram_fits = std::strcmp(forced, "file") != 0;
        }
        EngineOptions fallback      = options;
        fallback.expert_host_copy   = ram_fits;
        fallback.expert_cache_bytes = kExpertCacheAuto;
        // The routed banks upload and synchronize inside each layer, which a captured graph
        // cannot contain (normalize_engine_options sets the same constraint for the switch).
        fallback.use_cuda_graph     = false;
        std::fprintf(stderr,
                     "[fallback] resident weights leave no room for the requested KV capacity "
                     "(%s); retrying with the routed experts in %s - decode will be slower\n",
                     error.what(), ram_fits ? "host RAM" : "the artifact, no RAM replica fits");
        std::fflush(stderr);
        auto constructed = construct_model_once(fallback, device);
        // The engine reads the flags back when it runs, so it has to see the mode that is
        // actually in force.
        options = fallback;
        return constructed;
    }
}

} // namespace ninfer::runtime
