#include "runtime/engine/model_instance.h"
#include "artifact/expert_files.h"
#include "artifact/expert_reader.h"
#include "artifact/reader.h"
#include "artifact/formats.h"
#include "core/expert_cache.h"
#include "core/memory_budget.h"
#include "core/startup.h"
#include "runtime/engine/context_cache/joint_planning.h"
#include "models/qwen3_5/execution/expert_prefetch.h"
#include "models/qwen3_5/execution/expert_path_control.h"
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
std::uint64_t readahead_bytes(std::span<const ExpertBankLayout> banks);
std::uint64_t second_chance_budget_bytes(std::span<const ExpertBankLayout> banks);
using Clock = std::chrono::steady_clock;

// port (sm_89): a sentinel that asks the engine to size the expert pool from free device memory.
constexpr std::size_t kExpertCacheAuto = std::numeric_limits<std::size_t>::max();

using models::qwen3_5::execution::ExpertStream;
using models::qwen3_5::execution::RoutedMoeParameters;

// port (sm_89): collect the routed banks that were declared as streamed.
void collect_streams(const models::qwen3_5::execution::FfnParameters& ffn,
                     std::vector<ExpertStream*>& out, std::uint32_t& routed_top_k) {
    if (const auto* moe = std::get_if<RoutedMoeParameters>(&ffn); moe != nullptr && moe->stream) {
        out.push_back(moe->stream.get());
        // How many experts one token routes inside one bank. The host budget below is a working-set
        // bound, and the routed count is what makes it one.
        routed_top_k = std::max(routed_top_k,
                                static_cast<std::uint32_t>(moe->profile.routed.top_k));
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
// the pool only serves the routed top-k; the pool then takes what is left.
//
// port (sm_89): Q50 - the rule moved to core/memory_budget.h, where the same decision carries the
// measured value of a gigabyte and two overrides. It still reproduces this rule byte for byte by
// default; what is new is that the split is printed with what the next GiB is worth, and that a
// pinned split says so when it does not fit.

[[nodiscard]] bool kv_capacity_shortage(const std::invalid_argument& error) noexcept;

ExpertPlacement attach_expert_cache(ModelInstance& instance, const EngineOptions& options,
                                    DeviceContext& device, std::size_t kv_reserve_bytes) {
    ExpertPlacement placement;
    std::vector<ExpertStream*> streams;
    std::uint32_t routed_top_k = 0;
    for (const auto& layer : instance.parameters.text.layers) {
        collect_streams(layer.ffn, streams, routed_top_k);
    }
    if (instance.parameters.mtp) {
        collect_streams(instance.parameters.mtp->ffn, streams, routed_top_k);
    }
    if (streams.empty()) {
        // A dense model has no routed bank; the switch then simply does nothing.
        return {};
    }
    // T131: keep the bank list for the control plane's freeze policy (all banks, not just the
    // one that happens to print). The pointers are the parameters' own shared objects, which
    // outlive the instance's streams (Parameters is a member of it).
    instance.expert_streams = streams;
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
        // Graph capture and the staging buffers still need room after the pool takes its share;
        // that allowance is the configurable startup planning headroom (not the KV sizing
        // headroom, which resolve_kv_capacity already left free when the arena was planned).
        const auto split =
            core::plan_device_split(free_bytes, kv_reserve_bytes,
                                    options.startup_planning_headroom_bytes);
        core::report_device_split(split, free_bytes);
        requested = split.pool_bytes;
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
        // port (sm_89): an artifact that gives every expert its own object is described expert by
        // expert, which turns the six plane ranges of each expert into one contiguous copy per
        // matrix. A fused artifact repeats one object for the whole bank and keeps the row-range
        // mapping, so both forms stay readable.
        if (stream->per_expert_objects()) {
            std::vector<artifact::ObjectHandle> gate_up_objects;
            std::vector<artifact::ObjectHandle> down_objects;
            gate_up_objects.reserve(stream->count());
            down_objects.reserve(stream->count());
            for (const auto object : stream->gate_up_objects()) {
                gate_up_objects.push_back(artifact::ObjectHandle{object});
            }
            for (const auto object : stream->down_objects()) {
                down_objects.push_back(artifact::ObjectHandle{object});
            }
            files.push_back(host_tier
                                ? artifact::describe_expert_objects(
                                      gate_up_objects, down_objects,
                                      stream->gate_up_expert_geometry(),
                                      stream->down_expert_geometry(), requirements.banks[i])
                                : artifact::describe_expert_files(
                                      instance.expert_source->reader(), gate_up_objects,
                                      down_objects, stream->gate_up_expert_geometry(),
                                      stream->down_expert_geometry(), requirements.banks[i]));
            continue;
        }
        // port (sm_89): the host tier takes the same plan as the device tier. describe_fused_expert_host
        // skipped the object's own offset and the reader's segment walk, so its chunks carried
        // object-relative offsets while every consumer (spans, read_entry) reads them as file
        // absolute - which is why the fused artifact only scored correctly through the copying path
        // that happened to re-read by image offset. The per-expert plan was never affected.
        files.push_back(artifact::describe_fused_expert_files(
            instance.expert_source->reader(), gate_up, down, stream->gate_up_parent(),
            stream->down_parent(), stream->count(), gu_rows, dn_rows, requirements.banks[i]));
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
    // Size the host tier from what one pass actually reads. The CPU half of a split layer computes
    // every routed expert the pool does not keep, and a prefill of many tokens routes (nearly) every
    // expert of the layer, so one layer can touch the whole non-resident part of a bank - not just
    // the top-k of a decode step. Two layers of that is the floor the arena needs: one call must
    // never evict its own slots, and a second layer of headroom keeps the next call from throwing
    // away what the previous one still points at.
    //
    // Everything colder than that is served by the artifact, and on a machine where the model fits
    // in RAM the OS page cache absorbs it - which is why the arena no longer has to be sized to the
    // model. Measured: 600 MiB of arena and 4 GiB of arena give the same 14 ms of ensure per token
    // once the reads go through the cache.
    //
    // The routed top-k is printed beside it because it is the decode-side working set: at one token
    // the CPU reads top_k - kept per layer, which is why the same arena is mostly idle in decode and
    // full in prefill.
    std::uint64_t host_want = 0;
    std::uint64_t host_kept = 0;
    std::uint64_t host_layer = 0;
    std::uint64_t widest_count = 0;
    std::uint64_t streamed_banks = 0;
    for (const auto& bank : requirements.banks) {
        if (bank.resident) { continue; }
        ++streamed_banks;
        widest_count = std::max<std::uint64_t>(widest_count, bank.count);
        const auto kept = std::min<std::uint64_t>(
            instance.expert_cache->resident_capacity(), static_cast<std::uint64_t>(routed_top_k));
        host_kept += kept;
        host_layer = std::max<std::uint64_t>(host_layer, std::uint64_t{bank.count} - kept);
    }
    constexpr std::uint64_t kLayerHeadroom = 2;
    host_want = host_layer * kLayerHeadroom * instance.expert_cache->slot_bytes();
    // P1: the per-layer frozen set is a placement, so its capacity is a formula rather than a
    // working-set estimate. S is the complement of what the pool holds; D is whatever is left of
    // that complement, with at least one decode step's routes as headroom. S+D covers every expert
    // the Device does not keep, which is the only state in which a layer has no CPU route.
    const bool freeze_mode = [] {
        const char* value = std::getenv("NINFER_HOST_FREEZE");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    std::uint64_t freeze_s = 0, freeze_d = 0;
    if (freeze_mode) {
        const auto device = std::min<std::uint64_t>(
            instance.expert_cache->resident_capacity(), widest_count);
        const auto complement = widest_count > device ? widest_count - device : 0;
        if (const char* value = std::getenv("NINFER_WARM_STATIC_SLOTS");
            value != nullptr && *value != '\0') {
            freeze_s = std::min<std::uint64_t>(std::strtoull(value, nullptr, 10), complement);
        } else {
            freeze_s = complement;
        }
        freeze_d = std::max<std::uint64_t>(8, complement - freeze_s);
        if (streamed_banks != 0) {
            host_want = (freeze_s + freeze_d) * streamed_banks *
                        instance.expert_cache->slot_bytes();
        }
        // The freeze route wants the read-ahead on: its D is deliberately smaller than the
        // complement, so a predicted expert that arrives before its route is a fill the layer does
        // not wait for. Only the worker switch is set here; the SIZE is left to the tier
        // (NINFER_PREFETCH_SLOTS would also carve the Device prediction region, which this mode does
        // not want), so the tier's own default applies.
        if (std::getenv("NINFER_PREFETCH") == nullptr) {
            _putenv_s("NINFER_PREFETCH", "1");
        }
    }
    // port (sm_89): 1y.7 step two - the warm tier stops taking a share of free RAM on its own and
    // asks the ledger, which is the one place that can see every tier at once. The number it grants
    // is computed exactly as before (working-set estimate, RAM share, explicit override), so this
    // step is behaviour-neutral; what changes is that the amount is now accounted and printed
    // beside the other two tiers instead of being decided in isolation.
    // 13.4/Q53: the warm tier is three budgets, not one arena with auxiliaries carved out of it.
    // Routed is granted first; prefetch and the second-chance band get what is left. Enabling either
    // auxiliary therefore cannot shrink the routed working set - the failure mode that made "more
    // cache" mean "less warm tier".
    const auto routed_bytes = [&] {
        // The freeze formula is the budget: bypass the working-set clamp, which would cut a
        // complement-sized tier (about 12 GiB here) down to 4 GiB and silently turn the static set
        // back into a churning one. An explicit override still wins, so an experiment can ask for
        // less and watch S get clamped instead.
        if (freeze_mode && std::getenv("NINFER_EXPERT_HOST_ARENA_MB") == nullptr) {
            return host_want;
        }
        return host_expert_arena_bytes(host_want);
    }();
    const auto prefetch_bytes      = readahead_bytes(requirements.banks);
    const auto second_chance_ask = second_chance_budget_bytes(requirements.banks);
    const auto host_ask            = routed_bytes + prefetch_bytes + second_chance_ask;
    // port (sm_89): Q51 - the two levels of budget meet the three-tier ledger here, and a refusal
    // is a number instead of silence.
    //
    // Before: the caps were the asks (a ledger that cannot fail is a ledger that says nothing), the
    // KV part was a hardcoded 260 MiB rather than what the planner reserved, and both device
    // reservations were `(void)`. Now the Device cap covers the three consumers that actually hold
    // device memory - the pool, the KV pages and the safety the split reserved - each is admitted
    // separately, and a refusal names the owner, what it asked for and what is left. The totals are
    // unchanged, so this is accounting that can say "no", not a new allocation policy.
    // The third consumer is the startup planning allowance itself, so the cap carries the
    // configured value: a lowered allowance must tighten the ledger too, not just the pool.
    const std::uint64_t startup_headroom = options.startup_planning_headroom_bytes;
    const std::uint64_t device_cap =
        instance.expert_cache->capacity_bytes() + kv_reserve_bytes + startup_headroom;
    // T156 (upstream 049d5fb4): the three device accounts - expert pool, KV pages and the
    // transient workspace (the startup allowance carries it) - are quoted together. The
    // planning model lives in context_cache/joint_planning.h; the model layer still owns the
    // StateImage stride, so `state` reports zero here rather than an invented number. When
    // NINFER_CONTEXT_JOINT_PLAN=1 the check is printed; a feasible=false means the three
    // accounts cannot fit the cap and the ledger admission below will name the owner.
    if (const char* joint = std::getenv("NINFER_CONTEXT_JOINT_PLAN");
        joint != nullptr && *joint == '1') {
        const auto budget = context_cache::plan_joint_capacity(
            device_cap,
            {.kv_bytes        = kv_reserve_bytes,
             .state_bytes     = 0,
             .workspace_bytes = startup_headroom},
            0);
        std::fprintf(stderr,
                     "[joint] cap=%llu MiB kv=%llu MiB state=0(layout not exposed) "
                     "workspace(headroom)=%llu MiB feasible=%d\n",
                     static_cast<unsigned long long>(device_cap >> 20),
                     static_cast<unsigned long long>(budget.kv_bytes >> 20),
                     static_cast<unsigned long long>(budget.workspace_bytes >> 20),
                     budget.feasible ? 1 : 0);
        std::fflush(stderr);
    }
    instance.ledger.set_cap(core::Tier::Device, device_cap);
    const auto admit = [&instance](core::Tier tier, const char* owner, std::uint64_t bytes) {
        if (instance.ledger.reserve(tier, owner, bytes)) { return true; }
        std::fprintf(stderr,
                     "[ledger] REJECTED owner=%s asked=%llu MiB cap=%llu MiB used=%llu MiB - the "
                     "caller has to lower its ask\n",
                     owner, static_cast<unsigned long long>(bytes >> 20),
                     static_cast<unsigned long long>(instance.ledger.cap(tier) >> 20),
                     static_cast<unsigned long long>(instance.ledger.used(tier) >> 20));
        std::fflush(stderr);
        return false;
    };
    (void)admit(core::Tier::Device, "expert-pool", instance.expert_cache->capacity_bytes());
    (void)admit(core::Tier::Device, "kv-pages", kv_reserve_bytes);
    (void)admit(core::Tier::Device, "staging", startup_headroom);
    // The warm tier's ceiling is the budget it just computed, not "a quarter of free RAM": the
    // partition is what the scheduler grants, and a module may not widen it on its own. The override
    // exists so a caller can grant less - and then the arena is what actually shrinks, rather than
    // the ledger being told a smaller number than the arena takes.
    std::uint64_t host_cap = host_ask;
    if (const char* value = std::getenv("NINFER_BUDGET_RAM_MB"); value != nullptr && *value != '\0') {
        const auto pinned = static_cast<std::uint64_t>(std::strtoull(value, nullptr, 10)) << 20;
        if (pinned != 0 && pinned < host_cap) { host_cap = pinned; }
    }
    instance.ledger.set_cap(core::Tier::Host, host_cap);
    // Routed first, then prefetch, then the second-chance band. A tight cap drops the auxiliaries
    // before it touches the working set; it never takes a routed slot to pay for a cache.
    const std::uint64_t grant_routed = std::min(routed_bytes, host_cap);
    std::uint64_t remaining          = host_cap - grant_routed;
    const std::uint64_t grant_prefetch = std::min(prefetch_bytes, remaining);
    remaining -= grant_prefetch;
    const std::uint64_t grant_second_chance = std::min(second_chance_ask, remaining);
    const std::uint64_t host_arena_bytes =
        grant_routed + grant_prefetch + grant_second_chance;
    if (host_arena_bytes != host_ask) {
        std::fprintf(stderr,
                     "[ledger] warm lowered %llu -> %llu MiB (routed %llu, prefetch %llu, "
                     "second-chance %llu)\n",
                     static_cast<unsigned long long>(host_ask >> 20),
                     static_cast<unsigned long long>(host_arena_bytes >> 20),
                     static_cast<unsigned long long>(grant_routed >> 20),
                     static_cast<unsigned long long>(grant_prefetch >> 20),
                     static_cast<unsigned long long>(grant_second_chance >> 20));
        std::fflush(stderr);
    }
    (void)admit(core::Tier::Host, "expert-warm", grant_routed);
    if (grant_prefetch != 0) { (void)admit(core::Tier::Host, "expert-prefetch", grant_prefetch); }
    if (grant_second_chance != 0) {
        (void)admit(core::Tier::Host, "expert-second-chance", grant_second_chance);
    }
    std::fprintf(stderr,
                 "[experts] banks=%zu slot=%llu windows=%llu pool=%lluMiB host=%d arena=%lluMiB "
                 "(routed=%llu prefetch=%llu second-chance=%llu) want=%lluMiB top_k=%u kept=%llu "
                 "layer=%llu\n",
                 streams.size(),
                 static_cast<unsigned long long>(instance.expert_cache->slot_bytes()),
                 static_cast<unsigned long long>(instance.expert_cache->windows()),
                 static_cast<unsigned long long>(instance.expert_cache->capacity_bytes() >> 20),
                 int(instance.expert_host_source != nullptr),
                 static_cast<unsigned long long>(host_arena_bytes >> 20),
                 static_cast<unsigned long long>(grant_routed >> 20),
                 static_cast<unsigned long long>(grant_prefetch >> 20),
                 static_cast<unsigned long long>(grant_second_chance >> 20),
                 static_cast<unsigned long long>(host_want >> 20), routed_top_k,
                 static_cast<unsigned long long>(host_kept),
                 static_cast<unsigned long long>(host_layer));
    if (freeze_mode) {
        const auto pool_keeps = std::min<std::uint64_t>(
            instance.expert_cache->resident_capacity(), widest_count);
        std::fprintf(stderr,
                     "[freeze] budget: per bank s=%llu d=%llu (%llu slots; the pool keeps %llu, "
                     "the complement is %llu), c=%llu p=%llu; banks=%llu\n",
                     static_cast<unsigned long long>(freeze_s),
                     static_cast<unsigned long long>(freeze_d),
                     static_cast<unsigned long long>(freeze_s + freeze_d),
                     static_cast<unsigned long long>(pool_keeps),
                     static_cast<unsigned long long>(widest_count - pool_keeps),
                     static_cast<unsigned long long>(grant_second_chance /
                                                     std::max<std::uint64_t>(
                                                         1, instance.expert_cache->slot_bytes()) /
                                                     std::max<std::uint64_t>(1, streamed_banks)),
                     static_cast<unsigned long long>(grant_prefetch /
                                                     std::max<std::uint64_t>(
                                                         1, instance.expert_cache->slot_bytes()) /
                                                     std::max<std::uint64_t>(1, streamed_banks)),
                     static_cast<unsigned long long>(streamed_banks));
        std::fflush(stderr);
    }
    // port (sm_89): one line that says what each tier holds, from the ledger rather than from three
    // modules that cannot see each other (1y.7 step two: accounting only, allocations unchanged).
    {
        const std::uint64_t payload = host_layer == 0
                                          ? 0
                                          : static_cast<std::uint64_t>(requirements.banks.front().count) *
                                                static_cast<std::uint64_t>(requirements.banks.size()) *
                                                instance.expert_cache->slot_bytes();
        instance.ledger.report(payload);
    }
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
            // port (sm_89): the host arena describes a whole bank, but a per-expert artifact gives
            // the weight a parent of exactly one expert's rows. Scale that geometry back up to the
            // bank so the CPU op's bank-validation sees the same domain either way; a fused
            // artifact already has this shape and the call is a no-op there.
            const auto bank_geometry = [](const WeightGeometry& expert, std::uint32_t count) {
                const std::uint64_t shape[2] = {
                    static_cast<std::uint64_t>(count) * expert.shape.front(), expert.shape.back()};
                return weight_geometry(expert.format, expert.layout, shape);
            };
            descs.push_back({requirements.banks[i], streams[i]->gate_up_layout(),
                             streams[i]->down_layout(),
                             bank_geometry(streams[i]->gate_up_expert_geometry(),
                                           streams[i]->count()),
                             bank_geometry(streams[i]->down_expert_geometry(),
                                           streams[i]->count()),
                             streams[i]->per_expert_objects()});
        }
        auto residency = std::make_shared<models::qwen3_5::execution::HostExpertResidency>(
            descs, files, source,
            models::qwen3_5::execution::HostWarmBudget{
                .routed_bytes = grant_routed,
                .prefetch_bytes = grant_prefetch,
                .second_chance_bytes = grant_second_chance});
        for (std::size_t i = 0; i < streams.size(); ++i) {
            streams[i]->attach_host(residency);
        }
        // port (sm_89): the read-ahead tracks this residency from here on, so the worker exists
        // exactly as long as the tier does. Only when the switch is on: a run that never prefetches
        // starts no thread and has none to join. NINFER_PREFETCH=0 attaches the queue but starts
        // the gate CLOSED - the environment is the initial value only (v8 12.1), and the runtime
        // owner of the gate is the controller; an A/B arm that wants it pinned off pairs this with
        // NINFER_CONTROLLER_PREFETCH_OFF.
        if (std::getenv("NINFER_PREFETCH") != nullptr) {
            models::qwen3_5::execution::ExpertPrefetchQueue::instance().retain();
            instance.expert_prefetch_retained = true;
            const char* initial = std::getenv("NINFER_PREFETCH");
            if (initial != nullptr && *initial != '\0' && std::strcmp(initial, "0") == 0) {
                models::qwen3_5::execution::ExpertPrefetchQueue::instance().set_enabled(false);
                std::fprintf(stderr, "[experts] read-ahead attached, gate off (NINFER_PREFETCH=0)\n");
            }
        }
        std::fprintf(stderr, "[experts] host tier slots=%u (%u routed + %u read-ahead) slot_bytes=%llu\n",
                     residency->slot_capacity(), residency->routed_capacity(),
                     residency->prefetch_slots(),
                     static_cast<unsigned long long>(residency->slot_bytes()));
        // port (sm_89): the Device uploader may read a miss out of this pinned tier instead of the
        // pageable artifact mapping. Only experts the host tier actually keeps are answered; the rest
        // still stream from the artifact.
        instance.expert_cache->attach_arena(residency.get());
        if (freeze_mode) { residency->materialize_arena(); }
        if (freeze_mode && residency->arena_device_base() == nullptr) {
            // The freeze route's whole point is a placement the Device can name. A fallback to
            // pageable memory would still run, but every "host route" would be an upload, no table
            // could be captured, and no counter would say so. Fail where it is visible.
            throw std::runtime_error(
                "NINFER_HOST_FREEZE requires a device-accessible pinned arena; the mapped "
                "allocation was refused (see [host] arena mapped ... device_accessible=0)");
        }
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
    // port (sm_89): these two numbers describe different things and used to overwrite each other.
    // host_experts is the RAM cache in front of the source (set where the residency is built);
    // file_experts is how many experts of the model the pool has no slot for. Reporting the second
    // as the first made a healthy 4 GiB host tier read as `host_claim=0 (arena is smaller)`.
    placement.file_experts = streamed;
    if (instance.expert_host_residency) {
        placement.host_experts =
            static_cast<std::uint64_t>(instance.expert_host_residency->slot_capacity());
    }
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

// port (sm_89): R3's bytes, asked for before the arena exists so the ledger is told the truth
// rather than discovering the pool afterwards. Zero unless the read-ahead is on - an off switch
// must not move the arena size, or "prefetch off" stops being the baseline it is compared against.
std::uint64_t readahead_bytes(std::span<const ExpertBankLayout> banks) {
    std::uint64_t widest = 0;
    for (const auto& bank : banks) { widest = std::max(widest, bank.entry_bytes); }
    return models::qwen3_5::execution::HostExpertResidency::prefetch_bytes_for(widest);
}

std::uint64_t second_chance_budget_bytes(std::span<const ExpertBankLayout> banks) {
    std::uint64_t widest = 0;
    for (const auto& bank : banks) { widest = std::max(widest, bank.entry_bytes); }
    return models::qwen3_5::execution::HostExpertResidency::second_chance_bytes_for(widest);
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
    // port (sm_89): a working-set bound, not just a RAM-share bound. A split layer touches a few
    // experts per layer and the same ones come back within the next tokens, so a few GiB hold every
    // byte the CPU half reads. Sizing the arena to a quarter of RAM instead (14.5 GiB here) spreads
    // that working set over four times the pages, and the split path then measures 0.43 ms per
    // expert against 0.041 ms for the same kernel on a warm one - 10.6 tok/s against 28.7 on the
    // same pool. Anything colder than the working set belongs to the artifact, not to more arena.
    constexpr std::uint64_t kWorkingSetCap = 4ull << 30;
    std::uint64_t budget = std::min<std::uint64_t>({want_bytes, share, kWorkingSetCap});
    if (const char* value = std::getenv("NINFER_EXPERT_HOST_ARENA_MB"); value != nullptr && *value) {
        const auto mib = std::strtoull(value, nullptr, 10);
        // An explicit budget replaces the RAM share rather than tightening it, so a run can ask for
        // the whole model when the machine has room. It has to replace the working-set estimate too:
        // clamping to `want_bytes` made every request above the estimate (~937 MiB on this model) a
        // silent no-op, which is why earlier sweeps measured 600 MiB and 4 GiB as identical and
        // concluded the arena size did not matter.
        if (mib != 0) { budget = mib << 20; }
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
    // port (sm_89): whether a streamed expert tier can keep its captured graph depends on the pool
    // layout, which is not known until attach_expert_cache() has sized it. construct_model_once()
    // clears the flag only when the pool has to stream between layers.
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

ModelInstance::~ModelInstance() {
    // port (sm_89): detach before any member is destroyed. release() stops and joins the read-ahead
    // worker, so no fill can be running while the arena, its reader or the CUDA context go away.
    // The weak reference in each job is the safety net; this is the ordering that makes the net
    // unnecessary on the normal path.
    if (expert_prefetch_retained) {
        models::qwen3_5::execution::ExpertPrefetchQueue::instance().release();
        expert_prefetch_retained = false;
    }
}

std::uint64_t ModelInstance::resize_expert_pool(std::uint64_t target_bytes) {
    if (!expert_cache) { return 0; }
    if (expert_pool_pinned) {
        // A captured graph froze the window offsets; unmapping a window it reads is a
        // use-after-free. Dropping the graph is P12.4 work, so refuse instead of corrupting.
        throw std::invalid_argument(
            "expert pool cannot resize once a decode graph captured its offsets");
    }
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

std::uint32_t ModelInstance::resize_kv_capacity(std::uint32_t target_pages) {
    if (!program) { return 0; }
    const auto after = program->resize_kv_capacity(target_pages);
    if (const char* debug = std::getenv("NINFER_KV_DEBUG"); debug != nullptr && *debug) {
        std::fprintf(stderr, "[kv-pool] capacity -> %u page(s) (target %u)\n", after,
                     target_pages);
        std::fflush(stderr);
    }
    return after;
}

// T131: the control plane's freeze half. One policy for every bank, because the placement of a
// bank is per-bank: applying it to bank 0 alone would leave the other layers on their startup
// values and the console would show a policy only one fortieth of the model obeys.
void ModelInstance::set_expert_freeze_policy(std::uint32_t freeze_after,
                                             std::uint32_t reselect_after,
                                             bool thaw_on_request) {
    models::qwen3_5::execution::ExpertStream::FreezePolicy policy;
    policy.freeze_after    = freeze_after;
    policy.reselect_after  = reselect_after;
    policy.thaw_on_request = thaw_on_request;
    for (auto* stream : expert_streams) {
        if (stream != nullptr) { stream->set_freeze_policy(policy); }
    }
    if (!expert_streams.empty()) {
        std::fprintf(stderr,
                     "[ctl] freeze policy: freeze_after=%u reselect_after=%u thaw_on_request=%d "
                     "banks=%zu\n",
                     freeze_after, reselect_after, int(thaw_on_request), expert_streams.size());
        std::fflush(stderr);
    }
}

bool ModelInstance::expert_freeze_armed() const noexcept {
    return std::any_of(expert_streams.begin(), expert_streams.end(),
                       [](const models::qwen3_5::execution::ExpertStream* stream) {
                           return stream != nullptr && stream->freeze_armed();
                       });
}

bool ModelInstance::expert_frozen() const noexcept {
    return std::any_of(expert_streams.begin(), expert_streams.end(),
                       [](const models::qwen3_5::execution::ExpertStream* stream) {
                           return stream != nullptr && stream->frozen();
                       });
}

namespace {

ConstructedModel construct_model_once(EngineOptions& options, DeviceContext& device) {
    validate_options(options);
    // port (sm_89): the pool layout decides whether a captured decode graph is valid, so the
    // caller's intent is kept here and applied after attach_expert_cache().
    const bool graph_requested = options.use_cuda_graph;
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
    // The same instant the plan was priced against: after the weights, before the expert pool and
    // the KV/state pools exist. Comparing this against free_after_startup_bytes is what tells
    // "the device changed under us" apart from "we spent it ourselves".
    resolution.available_after_startup_bytes = current_free_device_bytes(
        options.wddm_evictable_budget, instance->model->storage_stats().device_capacity_bytes);
    // port (sm_89): request admission and output clamping both run against max_context, so the
    // arena must never resolve below it. The capacity curve guarantees this structurally - its
    // minimum is max(page_count(max_context), max_concurrency) and the degradation path in
    // kv_capacity.cpp never drops below that minimum - but this is the one place where both
    // numbers meet, and the invariant is too load-bearing to leave implicit.
    if (resolution.resolved_tokens < options.max_context) {
        throw std::logic_error(
            "resolved KV capacity of " + std::to_string(resolution.resolved_tokens) +
            " tokens cannot serve the configured max_context of " +
            std::to_string(options.max_context) + " tokens");
    }
    // The KV plan is settled before the expert pool is sized, so the pool can take everything the
    // arena does not need instead of holding back a fixed amount.
    if (options.expert_cache_bytes != 0) {
        const auto placement = attach_expert_cache(*instance, options, device,
                                                   resolution.runtime_reservation_bytes);
        // One window per bank means no layer ever evicts another, so the resident set is fixed and
        // the decode graph may be captured. Uploading the whole set first keeps the capture from
        // recording an upload that a later step would then repeat with the wrong experts.
        const bool every_layer = instance->expert_cache->can_hold_every_layer();
        if (every_layer) { instance->expert_cache->preload_all(); }
        // P6/step 4 + the default-on rule: the construction decision asks the controller's rule
        // (PolicyController::want_graph) instead of hard-coding a capacity or profit threshold.
        // Before the first layer has run nothing is measured, and the rule now answers "on" there:
        // the whole-step graph's measured benefit is small but not negative, and a capture the
        // guard below skips costs nothing. Only a measured structural failure (a layer that needs
        // a host fetch inside the step) turns it off. NINFER_GRAPH_BUDGET no longer gates
        // anything; it is accepted for compatibility.
        bool graph_ok = graph_requested &&
                        models::qwen3_5::execution::policy_controller().want_graph() &&
                        std::getenv("NINFER_EXPERT_POOL_CYCLE") == nullptr;
        // port (sm_89), capture guard: a pool that cannot hold every layer still streams, and
        // on this artifact that means some layer WILL take the split path, whose host fetch
        // waits on cudaStreamSynchronize (ffn.cpp) - an operation a capture forbids. The budget
        // used to open the door anyway and the NINFER_GRAPH_BUDGET=1 construction died inside
        // the warm body of the capture as CUDA_CHECK abort (cudaErrorStreamCaptureUnsupported,
        // ffn.cpp:355), before the pool was ever pinned. Now the guard skips the capture
        // instead: a streamed layout keeps the ordinary path, and the budget stays paid for
        // the one tier that can actually capture (full residency). every_layer is the
        // construction-time form of "no layer needs a host fetch": with one window per bank
        // and the whole set preloaded, no route can miss.
        if (graph_ok && !every_layer) {
            std::fprintf(stderr,
                         "[graph] skipped: layer needs host fetch (pool cannot hold every layer; "
                         "the split path's stream sync is illegal inside a capture)\n");
            std::fflush(stderr);
            graph_ok = false;
        }
        // Only a captured graph freezes the offsets, so only then is the pool off-limits to resize.
        instance->expert_pool_pinned = graph_ok;
        options.use_cuda_graph = graph_ok;
        // The plan was built before the pool existed, so its graph decision has to be revisited
        // now that the resident set is known.
        planner.set_use_cuda_graph(graph_ok);
        device.synchronize();
        if (placement.total_experts != 0) {
            constexpr double kMib = 1024.0 * 1024.0;
            std::fprintf(stderr,
                         "[budget] experts: gpu=%llu (%.0f MiB) host_cache=%llu slots (%.0f MiB) "
                         "unslotted=%llu of %llu, slot %.0f KiB; kv: %u pages, %.0f MiB reserved\n",
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
    // port (sm_89) / upstream 19367582: the graph term of the plan is still the static topology
    // allowance, and the capture itself is what turns it into a real allocation. Read the Device
    // around that step so the retained growth is measured, and treat a Device allocation failure
    // as one recoverable sizing miss instead of a dead startup.
    const auto before_program = device_memory_snapshot(device.device);
    try {
        instance->program = models::qwen3_5::create_program(instance->parameters,
                                                           std::move(sequence), device,
                                                           options.startup_observer);
    } catch (const DeviceAllocationError& error) {
        // One recovery: keep the explicit capacity and every execution feature, and give back the
        // automatic part of the KV arena. An explicit capacity is never rewritten here.
        if (options.kv_capacity.mode != KvCapacityMode::Automatic) { throw; }
        auto recovery_planner =
            models::qwen3_5::make_sequence_planner(instance->parameters, device, options);
        const auto& recovery_curve            = recovery_planner.capacity_curve();
        const std::uint32_t floor_pages       = recovery_curve.minimum_main_page_groups;
        const std::size_t floor_reservation   = recovery_curve.reservation_bytes(floor_pages);
        if (floor_pages >= resolution.main_page_groups) { throw; }
        const std::size_t planning_available =
            floor_reservation + options.kv_capacity.automatic_headroom_bytes;
        auto recovered = resolve_kv_capacity(options.kv_capacity, recovery_curve,
                                            planning_available);
        recovered.available_after_weights_bytes = resolution.available_after_weights_bytes;
        recovered.available_after_startup_bytes = resolution.available_after_startup_bytes;
        recovered.evict_contract_effective      = resolution.evict_contract_effective;
        recovered.evict_contract_reasons        = resolution.evict_contract_reasons;
        std::fprintf(stderr,
                     "[budget] Program allocation failed (%s); retrying once with the minimum KV "
                     "capacity (%u -> %u page groups)\n",
                     error.what(), resolution.main_page_groups, recovered.main_page_groups);
        std::fflush(stderr);
        auto recovered_plan = std::move(recovery_planner).finalize(recovered.main_page_groups);
        options.context_cache.host_capacity_bytes = recovered_plan.host_capacity_bytes();
        instance->kv_capacity_resolution          = recovered;
        instance->program = models::qwen3_5::create_program(instance->parameters,
                                                            std::move(recovered_plan), device,
                                                            options.startup_observer);
    }
    device.synchronize();
    program.complete();
    // T156 (upstream 049d5fb4): the state account, now with the layout in hand. Before the
    // Program exists the recurrent state has no identity (its stride is a property of the built
    // graph), which is why attach_expert_cache() could only report it as zero. Here the exact
    // union is quoted: one StateImage per device state slot (the context cache normalizes the
    // slot count to the concurrency when the option is unset, so the same rule is applied).
    if (const char* joint = std::getenv("NINFER_CONTEXT_JOINT_PLAN");
        joint != nullptr && *joint == '1') {
        const auto& geometry = instance->program->checkpoint_geometry();
        // The context cache normalizes the extra-slot count to the concurrency when the option is
        // unset (configuration.cpp prints the same expression), so the state account uses it too.
        const std::uint64_t slots = std::max<std::uint64_t>(
            1, options.context_cache.device_state_slots.value_or(options.max_concurrency));
        const std::uint64_t state_bytes = geometry.state_image_bytes * slots;
        const std::uint64_t kv_bytes    = instance->kv_capacity_resolution.runtime_reservation_bytes;
        // A dense model has no routed pool at all (attach_expert_cache returns before creating one),
        // so its account is zero rather than a dereference of an absent cache.
        const std::uint64_t pool_cap    = instance->expert_cache != nullptr
                                              ? instance->expert_cache->capacity_bytes()
                                              : 0;
        const std::uint64_t device_cap  = pool_cap + kv_bytes +
                                          options.startup_planning_headroom_bytes;
        const auto exact = context_cache::plan_joint_capacity(
            device_cap,
            {.kv_bytes        = kv_bytes,
             .state_bytes     = state_bytes,
             .workspace_bytes = options.startup_planning_headroom_bytes},
            0);
        std::fprintf(stderr,
                     "[joint] exact: cap=%llu MiB kv=%llu MiB state=%llu MiB (%llu slot(s) x "
                     "%llu B) workspace(headroom)=%llu MiB feasible=%d\n",
                     static_cast<unsigned long long>(device_cap >> 20),
                     static_cast<unsigned long long>(exact.kv_bytes >> 20),
                     static_cast<unsigned long long>(exact.state_bytes >> 20),
                     static_cast<unsigned long long>(slots),
                     static_cast<unsigned long long>(geometry.state_image_bytes),
                     static_cast<unsigned long long>(exact.workspace_bytes >> 20),
                     exact.feasible ? 1 : 0);
        std::fflush(stderr);
    }
    {
        // Everything this Program owns is explicit except what the capture and the driver keep;
        // the difference against the reservation is the measured startup retainer.
        const auto after_program = device_memory_snapshot(device.device);
        instance->startup_retained_bytes =
            device_memory_growth(before_program, after_program) -
            static_cast<std::int64_t>(instance->kv_capacity_resolution.runtime_reservation_bytes);
        if (const auto summary = instance->program->memory_summary(); summary.cuda_graph) {
            // The estimate is a per-topology constant; this line says whether it held. Printed,
            // not enforced: the arena above has already been placed from the plan.
            if (summary.cuda_graph->preparation_increment_bytes >
                static_cast<std::int64_t>(summary.cuda_graph_allowance_bytes)) {
                std::fprintf(stderr,
                             "[graph] capture exceeded the planned allowance by %.1f MiB; "
                             "keep the arena and record the measured cost\n",
                             static_cast<double>(summary.cuda_graph->preparation_increment_bytes -
                                                 static_cast<std::int64_t>(
                                                     summary.cuda_graph_allowance_bytes)) /
                                 1048576.0);
                std::fflush(stderr);
            }
        }
    }
    instance->kv_capacity_resolution.free_after_startup_bytes = current_free_device_bytes(options.wddm_evictable_budget);
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

// port (sm_89): an artifact that gives every expert its own object cannot be loaded as resident
// weights - a bank-wide row region runs past a parent that holds one expert. The loader says so,
// and the same artifact runs fine as a streamed tier, so treat it as a request to stream instead of
// a broken artifact. Only when the caller asked for resident weights (pool = 0): with a pool the
// message means what it says.
[[nodiscard]] bool resident_moe_needs_streaming(const std::invalid_argument& error) noexcept {
    const char* what = error.what();
    return what != nullptr &&
           std::strstr(what, "weight bank exceeds its parent or native matrix domain") != nullptr;
}

} // namespace

ConstructedModel construct_model(EngineOptions& options, DeviceContext& device) {
    try {
        return construct_model_once(options, device);
    } catch (const std::invalid_argument& error) {
        // A per-expert artifact cannot be loaded as resident weights. Ask for the streamed tier and
        // retry before anything else: the request was "no pool", not "this artifact is broken".
        if (options.expert_cache_bytes == 0 && !options.expert_host_copy &&
            resident_moe_needs_streaming(error) &&
            std::getenv("NINFER_NO_EXPERT_FALLBACK") == nullptr) {
            EngineOptions streamed      = options;
            streamed.expert_cache_bytes = kExpertCacheAuto;
            std::fprintf(stderr,
                         "[fallback] %s; retrying with the routed experts streamed from the "
                         "artifact\n", error.what());
            std::fflush(stderr);
            auto constructed = construct_model_once(streamed, device);
            options          = streamed;
            return constructed;
        }
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
