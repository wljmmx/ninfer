#include "runtime/engine/model_instance.h"
#include "artifact/expert_files.h"
#include "artifact/expert_reader.h"
#include "artifact/reader.h"
#include "artifact/formats.h"
#include "core/capacity_bet.h"
#include "core/expert_cache.h"
#include "core/memory_budget.h"
#include "core/startup.h"
#include "runtime/engine/context_capacity.h"
#include "models/qwen3_5/execution/cpu_path_policy.h"
#include "models/qwen3_5/execution/expert_prefetch.h"
#include "models/qwen3_5/execution/expert_path_control.h"
#include "models/qwen3_5/execution/expert_stream.h"
#include "models/qwen3_5/execution/host_expert_reader.h"
#include "models/qwen3_5/execution/host_expert_residency.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/measurement.h"
#include "models/qwen4_exp/execution/qsa.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <set>
#include <stdexcept>
#include <string>
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
std::uint64_t current_free_host_bytes();
models::qwen3_5::execution::HostArenaBlock allocate_host_arena(std::uint64_t want_bytes,
                                                               bool          exact);
std::uint64_t current_commit_headroom_bytes();
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

// P7.1 (v10 8.4, V10-55): a refusal that can be re-solved. Thrown by attach_expert_cache() when the
// pool would hold less than one slot AND the controller's bet says the KV leg can give way.
// construct_model_once() catches it, prices the arena again against the released bytes, and attaches
// a second time; a second refusal is an ordinary invalid_argument - the startup fault this replaces.
struct CapacityRebindRequest {
    std::uint64_t        kv_release_bytes = 0;
    core::CapacityBetPlan plan{};
};

// One builder for the bet, so the tight-run report and the rebind attempt cannot describe the same
// Device with two different requests. `free_bytes` is the free count the split used, so the bet's
// budget and plan_device_split()'s budget are the same number.
[[nodiscard]] core::CapacityBetRequest capacity_bet_request(std::size_t free_bytes,
                                                           std::size_t kv_reserve_bytes,
                                                           std::size_t headroom_bytes,
                                                           std::uint64_t slot_bytes,
                                                           std::uint64_t window_slots,
                                                           std::uint64_t full_slots,
                                                           std::size_t target_tokens) {
    core::CapacityBetRequest request;
    request.device_budget_bytes    = free_bytes;
    request.startup_headroom_bytes = headroom_bytes;
    request.expert_slot_bytes      = slot_bytes;
    request.expert_window_slots    = window_slots;
    // The ask is the whole routed set (every bank's window): "experts fully resident" in the bet's
    // own words. Naming only one window here would make a card that is a gigabyte short of the set
    // answer "both asks fit" - the exact run V10-55 is about.
    request.expert_full_slots = full_slots;
    request.kv_target_tokens  = target_tokens;
    // The KV arena was priced for exactly this context, so the price per token is the resolved
    // reservation divided by the tokens it serves: a derivation of the number the arena already
    // charged, not a second estimate of it.
    request.kv_bytes_per_token = target_tokens != 0 ? kv_reserve_bytes / target_tokens : 0;
    request.mode               = controller_mode();
    // The controller's own observations, with its "not measured" sentinels preserved: policy signals
    // use -1 (and 0 for the latency proxy before a layer has run), and a zero reaching the decision
    // would read as a measurement of a perfect run.
    auto& controller           = models::qwen3_5::execution::policy_controller();
    request.signals.decode_tps = controller.signals.decode_tps;
    request.signals.miss_rate  = controller.signals.no_miss_pct >= 0.0
                                     ? 1.0 - controller.signals.no_miss_pct / 100.0
                                     : -1.0;
    const double latency          = controller.latency_ms_per_token();
    request.signals.lambda_p99_ms = latency > 0.0 ? latency : -1.0;
    return request;
}

// One line for the plan, whichever way the bet went: the letter, what the two consumers asked for,
// the reason and the two sizes it names. `achieved_pool_bytes` is what the split actually handed
// the pool, so a target that could not be reached is visible instead of being printed as if it had
// been. The ask is printed because the whole decision is a comparison, and a reader who cannot see
// both sides cannot check the answer.
void report_capacity_bet(const char* what, const core::CapacityBetRequest& request,
                         const core::CapacityBetPlan& plan, std::uint64_t achieved_pool_bytes) {
    std::fprintf(stderr,
                 "[plan] capacity %s: choice=%s(%s) asked=%llu+%llu MiB reason=%s "
                 "expert_pool_mb=%llu kv_pages_mb=%llu achieved_pool_mb=%llu\n",
                 what, core::capacity_bet_name(plan.choice), core::capacity_bet_label(plan.choice),
                 static_cast<unsigned long long>(request.expert_ask_bytes() >> 20),
                 static_cast<unsigned long long>(request.kv_ask_bytes() >> 20), plan.reason,
                 static_cast<unsigned long long>(plan.expert_pool_bytes >> 20),
                 static_cast<unsigned long long>(plan.kv_pages_bytes >> 20),
                 static_cast<unsigned long long>(achieved_pool_bytes >> 20));
    std::fflush(stderr);
}

ExpertPlacement attach_expert_cache(ModelInstance& instance, const EngineOptions& options,
                                    DeviceContext& device, std::size_t kv_reserve_bytes,
                                    std::size_t kv_floor_bytes, bool rebind_allowed) {
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

    // port (sm_89): the gate/up/down form owns three cache runs per layer - one per matrix - while
    // the fused form owns one. The cache's entry image always carries two matrix slots (a gate_up
    // and a down), so a run that owns one matrix leaves the spare slot as a single row of the same
    // column axis: about 0.5 KiB per expert that nothing ever addresses.
    //
    // `runs[i]` records where stream i's runs start, so the attach and the file plans below index
    // exactly the way the cache does.
    struct StreamRuns {
        bool split          = false;
        std::uint32_t first = 0;
    };
    std::vector<StreamRuns> runs(streams.size());
    const auto spare_matrix = [](const WeightGeometry& expert) {
        const std::uint64_t shape[2] = {1, expert.shape.empty() ? 0 : expert.shape.back()};
        return weight_geometry(expert.format, expert.layout, shape);
    };
    std::vector<ExpertBankSpec> specs;
    specs.reserve(streams.size() * 3);
    for (std::size_t i = 0; i < streams.size(); ++i) {
        const auto* stream = streams[i];
        runs[i].first      = static_cast<std::uint32_t>(specs.size());
        if (!stream->split_banks()) {
            specs.push_back({{stream->gate_up_expert_geometry()}, {stream->down_expert_geometry()},
                             stream->count(), false});
            continue;
        }
        runs[i].split = true;
        for (const auto role : {models::qwen3_5::execution::ExpertMatrixRole::Gate,
                                models::qwen3_5::execution::ExpertMatrixRole::Up,
                                models::qwen3_5::execution::ExpertMatrixRole::Down}) {
            const auto& expert = stream->split_expert_geometry(role);
            specs.push_back({ExpertMatrixSpec{expert}, ExpertMatrixSpec{spare_matrix(expert)},
                             stream->count(), false});
        }
    }
    const auto probe = describe_expert_cache(specs, {.capacity_bytes = 256});
    // port (sm_89): the pool holds a whole number of slots, one per expert regardless of bank,
    // because an evicted slot is handed to whichever bank needs it next.
    const auto entry = probe.max_entry_bytes;
    auto requested   = options.expert_cache_bytes;
    // The KV arena is planned first and already knows its exact reservation, so the pool takes
    // everything above it instead of guessing a fixed 6 GiB: on a 256K/k8v4 run that guess left
    // 3.8 GiB idle while 28% of the experts still streamed from the host.
    std::size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess || free_bytes == 0) {
        throw std::runtime_error("expert cache sizing cannot read device memory");
    }
    // Graph capture and the staging buffers still need room after the pool takes its share; that
    // allowance is the configurable startup planning headroom (not the KV sizing headroom, which
    // resolve_kv_capacity already left free when the arena was planned).
    const auto split = core::plan_device_split(free_bytes, kv_reserve_bytes,
                                               options.startup_planning_headroom_bytes);
    // Whether the SPLIT sized this pool, captured before `requested` is overwritten below: the bet
    // may only move Device bytes for a pool the split chose itself. A named budget is never
    // second-guessed - the caller asked for that number, and a tight one is its own decision.
    const bool pool_from_split = requested == kExpertCacheAuto;
    if (pool_from_split) {
        core::report_device_split(split, free_bytes);
        requested = split.pool_bytes;
    }
    // The VMM reservation is the growth ceiling. Reserve the auto-sized maximum even when the
    // caller pinned a smaller live pool: the L1 slow dial's upward steps then map segments from the
    // driver instead of being clamped at the construction size (slots_max == slots_live was the
    // measured ceiling that made every grow a no-op). A caller asking for more than auto gets its
    // own number as the ceiling.
    const std::size_t pool_limit = std::max<std::size_t>(requested, split.pool_bytes);
    // Whole windows only: a partial window is dead space and would make the slot count odd.
    // `widest` is the FLOOR (one bank's window) and `routed_slots` is the ASK (every bank's window,
    // i.e. "experts fully resident"); the two are different questions and are kept apart.
    std::uint64_t widest       = 0;
    std::uint64_t routed_slots = 0;
    for (const auto& spec : specs) {
        widest = std::max<std::uint64_t>(widest, spec.count);
        routed_slots += spec.count;
    }
    auto capacity = requested / entry * entry;
    // A pool that cannot hold a whole layer keeps whatever slots fit and the host tier covers the
    // rest of the layer; rounding to zero here would refuse a pool that is perfectly usable.
    if (widest != 0 && capacity / entry >= widest) {
        capacity = capacity / (entry * widest) * (entry * widest);
    }
    // P7.1/P7.2 (V10-55): the capacity bet is the controller's decision, and it is taken on every
    // run that creates a pool - not only on the ones that would have refused. The line is printed
    // either way, because "no bet was taken" is information: it says the whole ask fits.
    //
    // Two tight states reach this point, and only one of them is a fault:
    //   * the pool holds some slots but not a whole window - legal today (the Host tier serves the
    //     rest of the layer), and the run only needs the decision written down;
    //   * the pool holds less than one slot - the pool cannot exist at all, which is the startup
    //     fault this decision replaces. If the controller's bet moves Device bytes from the KV leg
    //     to the pool, the caller is asked to re-price the arena once and attach again; only a
    //     second refusal is an error.
    const auto request =
        capacity_bet_request(free_bytes, kv_reserve_bytes, options.startup_planning_headroom_bytes,
                             entry, widest, routed_slots, options.max_context);
    const auto bet   = core::plan_capacity_bet(request);
    const bool tight = widest != 0 && capacity / entry < widest;
    if (tight && capacity != 0) {
        report_capacity_bet("tight", request, bet, capacity);
    } else if (!tight) {
        report_capacity_bet("plan", request, bet, capacity);
    } else {
        std::fprintf(
            stderr,
            "[plan] capacity REJECTED owner=expert_pool asked=%llu MiB cap=%llu MiB "
            "(free=%llu MiB kv=%llu MiB startup=%llu MiB)\n",
            static_cast<unsigned long long>((entry * widest) >> 20),
            static_cast<unsigned long long>(capacity >> 20),
            static_cast<unsigned long long>(free_bytes >> 20),
            static_cast<unsigned long long>(kv_reserve_bytes >> 20),
            static_cast<unsigned long long>(options.startup_planning_headroom_bytes >> 20));
        std::fflush(stderr);
    }
    // The bet is not advisory: when it says the pool should hold more Device memory than the split
    // gave it, and the KV leg still has bytes above its own target to give, the caller is asked to
    // price the arena again. One attempt only (rebind_allowed), and only for a pool the split sized
    // itself - a pinned pool is the caller's number. A re-plan that cannot serve max_context is
    // refused by the caller's own check, so this cannot trade the configured context for the pool.
    const std::uint64_t release =
        bet.kv_pages_bytes < kv_reserve_bytes
            ? static_cast<std::uint64_t>(kv_reserve_bytes) - bet.kv_pages_bytes
            : 0;
    // One expert slot is the smallest release that can seat another expert: a KV figure a few bytes
    // below the reservation is alignment noise, and re-pricing the arena for it would be a re-plan
    // that buys nothing. The second bound is the arena's own contract: a plan below its floor
    // cannot serve max_context, so asking the caller for one is asking for a refusal that costs a
    // second attach. A tiered KV leg names a floor of zero - pages may leave the Device there.
    const bool moves_bytes = bet.feasible && bet.expert_pool_bytes > capacity && release >= entry &&
                             bet.kv_pages_bytes >= kv_floor_bytes &&
                             (pool_from_split || capacity == 0);
    if (rebind_allowed && moves_bytes) {
        std::fprintf(stderr,
                     "[plan] capacity: choice=%s(%s) moves %llu MiB from the KV leg to the pool - "
                     "re-solving once\n",
                     core::capacity_bet_name(bet.choice), core::capacity_bet_label(bet.choice),
                     static_cast<unsigned long long>(release >> 20));
        std::fflush(stderr);
        report_capacity_bet("rebound", request, bet, bet.expert_pool_bytes);
        throw CapacityRebindRequest{release, bet};
    }
    if (capacity == 0) {
        // No bet frees a slot: say which one was taken and let the check below fail loud.
        report_capacity_bet("no rebound", request, bet, 0);
        throw std::invalid_argument("expert cache holds less than one expert window (" +
                                    std::to_string(entry * widest) +
                                    " bytes) and the capacity re-solve freed none");
    }
    // The ceiling is aligned like the live capacity so slots_max is always a whole slot count.
    const auto capacity_limit = std::max<std::uint64_t>(
        capacity, static_cast<std::uint64_t>(pool_limit) / entry * entry);
    expert_debug("capacity", capacity, entry);
    auto requirements =
        describe_expert_cache(specs, {.capacity_bytes = capacity,
                                      .capacity_limit_bytes = capacity_limit});
    std::vector<ExpertFileBank> files;
    files.reserve(streams.size() * 3);
    // port (sm_89): the bank description must match the reader that was created above. A host
    // tier has no artifact reader at all, so asking it for one dereferences a null pointer.
    const bool host_tier = instance.expert_host_source != nullptr;
    for (std::size_t i = 0; i < streams.size(); ++i) {
        const auto* stream = streams[i];
        if (runs[i].split) {
            // One file plan per matrix: the entry image carries only that matrix, and the artifact
            // stores its experts either as row blocks of one object or as one object each.
            for (std::size_t role = 0; role < 3; ++role) {
                const auto matrix_role =
                    static_cast<models::qwen3_5::execution::ExpertMatrixRole>(role);
                std::vector<artifact::ObjectHandle> objects;
                for (const auto object : stream->split_objects(matrix_role)) {
                    objects.push_back(artifact::ObjectHandle{object});
                }
                // A host tier has no artifact reader (`instance.expert_source` is null), so the
                // single-matrix plan has to be built from the object bytes the replica holds.
                if (const char* debug = std::getenv("NINFER_EXPERT_DEBUG"); debug != nullptr) {
                    const auto& shape = stream->split_expert_geometry(matrix_role).shape;
                    std::fprintf(stderr,
                                 "[expert-files] stream=%zu role=%zu rows=%u count=%u objects=%llu,%llu,%llu shape=%llu,%llu",
                                 i, role,
                                 static_cast<unsigned>(
                                     stream->split_layout(matrix_role).region.rows),
                                 stream->count(),
                                 objects.empty() ? 0ULL : objects[0].index,
                                 objects.size() < 2 ? 0ULL : objects[1].index,
                                 objects.size() < 3 ? 0ULL : objects[2].index,
                                 shape.empty() ? 0ULL : shape[0],
                                 shape.size() < 2 ? 0ULL : shape[1]);
                    std::fprintf(stderr, "\n");
                }
                files.push_back(
                    host_tier
                        ? artifact::describe_matrix_expert_objects(
                              objects, stream->split_expert_geometry(matrix_role), stream->count(),
                              static_cast<std::uint32_t>(
                                  stream->split_layout(matrix_role).region.rows),
                              requirements.banks[runs[i].first + role])
                        : artifact::describe_matrix_expert_files(
                              instance.expert_source->reader(), objects,
                              stream->split_expert_geometry(matrix_role), stream->count(),
                              static_cast<std::uint32_t>(
                                  stream->split_layout(matrix_role).region.rows),
                              requirements.banks[runs[i].first + role]));
            }
            continue;
        }
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
                                      stream->down_expert_geometry(),
                                      requirements.banks[runs[i].first])
                                : artifact::describe_expert_files(
                                      instance.expert_source->reader(), gate_up_objects,
                                      down_objects, stream->gate_up_expert_geometry(),
                                      stream->down_expert_geometry(),
                                      requirements.banks[runs[i].first]));
            continue;
        }
        // port (sm_89): the host tier takes the same plan as the device tier. describe_fused_expert_host
        // skipped the object's own offset and the reader's segment walk, so its chunks carried
        // object-relative offsets while every consumer (spans, read_entry) reads them as file
        // absolute - which is why the fused artifact only scored correctly through the copying path
        // that happened to re-read by image offset. The per-expert plan was never affected.
        files.push_back(host_tier
                            ? artifact::describe_fused_expert_host(
                                  gate_up, down, stream->gate_up_parent(), stream->down_parent(),
                                  stream->count(), gu_rows, dn_rows,
                                  requirements.banks[runs[i].first])
                            : artifact::describe_fused_expert_files(
                                  instance.expert_source->reader(), gate_up, down,
                                  stream->gate_up_parent(), stream->down_parent(), stream->count(),
                                  gu_rows, dn_rows, requirements.banks[runs[i].first]));
    }
    expert_debug("files", files.size(), requirements.banks.size());
    // The prediction region's budget: a DECISION (C4), taken here because the region is carved at
    // construction and the numbers it needs are the model's own geometry - the routing width, the
    // number of layers, the runs a layer has and the pool's window per bank. The size is
    // decision 5's answer, not a constant and not a default: `decide_prefetch_slots()` scales the
    // prior by the cold share and the measured rate, and returns 0 when nothing stages on this path.
    //
    // Why 0 by default, stated once more because the carve is not free: the region is taken from the
    // pool's own slots (core/expert_cache.cpp pops them off each bank's free list and reserves
    // `prefetch + ring` per bank), so an armed region shrinks the routed working set by
    // banks*slots*entry_bytes - 20 x 144 x 1.25 MB = 3.6 GiB, thirty percent of the pool, measured
    // 2026-11-16 at 1.54 tok/s against 8.28 without. A region with no prediction source has nothing to
    // stage, so `staging_reachable` is false unless THIS run asked for a source (vram / ring / signal),
    // and decision 5 then answers 0 - which is a result, not a switch.
    std::uint32_t region_slots = 0;
    {
        using namespace models::qwen3_5::execution;
        PrefetchSignals signals;
        signals.top_k       = static_cast<int>(routed_top_k);
        signals.runs_per_layer =
            streams.empty() ? 1 : static_cast<int>(requirements.banks.size() / streams.size());
        signals.pool_slots_per_bank = entry != 0 ? static_cast<int>(capacity / entry) : 0;
        signals.layers              = static_cast<int>(streams.size());
        const auto asked = [](const char* name) {
            const char* value = std::getenv(name);
            return value != nullptr && *value != '\0' && value[0] != '0';
        };
        signals.staging_reachable = asked("NINFER_PREFETCH_MODE") || asked("NINFER_PRED_RING") ||
                                    asked("NINFER_PREFETCH_SIGNAL");
        note_prefetch_signals(signals);
        const auto sizing = decide_prefetch_slots();
        // P1.2/P2.2: the region is zero when the plan asked for nothing, and that is the only reason
        // it can be zero now. `core::device_prefetch_consumer_present()` used to be the rule that held
        // the line at zero because the Device form placed nothing; P2.2 gave the region a real
        // consumer (`ops::moe_prefetch_select` + the pool's adoption), so the predicate is true and
        // the carve follows decision 5 like any other line - which means "nothing asked to stage" is
        // still zero, twice over: `staging_reachable` gates the ask, and the size is the model's
        // geometry times the measured rate.
        region_slots = core::device_prefetch_consumer_present()
                           ? static_cast<std::uint32_t>(sizing.slots_per_bank)
                           : 0;
        // NINFER_PREFETCH_SLOTS was the escape hatch that used to carve a region with no consumer
        // (which then read as "the policy chose nothing"). With a consumer present it names the
        // region's size, clamped by the CONTROLLER's own ceiling (prefetch_slot_ceiling) so an
        // override cannot name slots the pool does not have - and so the override and the policy
        // cannot drift apart about what that ceiling is. It also still sizes the host read-ahead
        // band (host_expert_residency.h), which is why the line says both.
        if (const char* slots = std::getenv("NINFER_PREFETCH_SLOTS");
            slots != nullptr && *slots != '\0' && std::strtoul(slots, nullptr, 10) != 0) {
            const auto forced = prefetch_slots_forced();
            if (core::device_prefetch_consumer_present() && forced > 0) {
                const auto ceiling =
                    models::qwen3_5::execution::prefetch_slot_ceiling(signals.pool_slots_per_bank);
                region_slots = static_cast<std::uint32_t>(std::min(forced, ceiling));
            }
            std::fprintf(stderr,
                         "[ctl] prefetch: NINFER_PREFETCH_SLOTS=%s -> region=%u slot(s)/bank "
                         "(clamped to a quarter of a bank's window; a region index names one of the "
                         "pool's own slots), and the host read-ahead band\n",
                         slots, region_slots);
            std::fflush(stderr);
        }
        std::fprintf(stderr,
                     "[ctl] prefetch: region=%u slot(s)/bank reason=%d (top_k=%d pool=%u/bank "
                     "runs=%d layers=%d staging=%d) - carved from the routed set, not extra memory\n",
                     region_slots, sizing.reason, signals.top_k, signals.pool_slots_per_bank,
                     signals.runs_per_layer, signals.layers, int(signals.staging_reachable));
        std::fflush(stderr);
    }
    requirements.config.prefetch_slots_per_bank = region_slots;
    instance.expert_cache = std::make_unique<ExpertCache>(
        specs, requirements, device, std::span<const ExpertBankPair>{}, files,
        source);
    for (std::size_t i = 0; i < streams.size(); ++i) {
        const auto first = runs[i].first;
        if (!runs[i].split) {
            streams[i]->attach(*instance.expert_cache, first, requirements.banks[first]);
            continue;
        }
        // port (sm_89): one cache run per matrix. `bank_` becomes the gate run, which is the
        // layer's identity; the three tables are reachable through the stream's split accessors.
        streams[i]->attach_split(*instance.expert_cache, first, requirements.banks[first], first + 1,
                                 requirements.banks[first + 1], first + 2,
                                 requirements.banks[first + 2]);
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
    std::uint64_t complement_slots = 0;
    std::uint64_t layer_slots = 0;
    for (std::size_t index = 0; index < requirements.banks.size(); ++index) {
        const auto& bank = requirements.banks[index];
        if (bank.resident) { continue; }
        ++streamed_banks;
        widest_count = std::max<std::uint64_t>(widest_count, bank.count);
        // Each bank is charged only the seats the pool gives it. Per-bank pricing means one bank may
        // be much wider than the rest, so the pool-wide number understates the narrower banks' share
        // and would size the host tier for a tier that is not actually short.
        const auto seats =
            instance.expert_cache->resident_capacity(static_cast<std::uint32_t>(index));
        const auto kept = std::min<std::uint64_t>(seats, static_cast<std::uint64_t>(routed_top_k));
        host_kept += kept;
        // The tier's real complement: what the pool has no seat for at all. `kept` above is the
        // decode working set (top_k), which is the right number for "is the pool warm enough" but
        // not for "how much RAM would hold the rest" - the pool keeps seats beyond top_k.
        const auto missing = bank.count > seats ? static_cast<std::uint64_t>(bank.count) - seats : 0;
        layer_slots = std::max(layer_slots, missing);
        complement_slots += missing;
        host_layer = std::max<std::uint64_t>(host_layer, std::uint64_t{bank.count} - kept);
    }
    // port (sm_89): the warm budget is the model's own complement, not a two-layer guess.
    //
    // The old formula was `host_layer * 2 * slot_bytes`, which says "keep two layers of headroom
    // so a fill cannot evict itself". That is the right size for a model the Device mostly holds
    // (35B: complement 0, so the tier is not allocated at all), and the wrong size for one it does
    // not: Flash-Next's complement is 55200 of 73728 experts (67.5 GiB), and two layers buy a 2/48
    // hit window - measured 5.84 tok/s prefill, 3.16 tok/s decode, with 91 GiB of the RAM partition
    // unused. What the tier needs is what would hold the complement; what it may take is bounded by
    // RAM in host_expert_arena_bytes(). The layer floor stays so one prefill fill always fits (a
    // fill larger than the stripe throws, see HostExpertResidency::fill_locked).
    const auto banks_per_layer =
        streams.empty() ? std::uint64_t{1}
                        : std::max<std::uint64_t>(
                              1, static_cast<std::uint64_t>(requirements.banks.size()) /
                                     static_cast<std::uint64_t>(streams.size()));
    const auto slot_bytes = instance.expert_cache->slot_bytes();
    const auto layer_bytes    = layer_slots * banks_per_layer * slot_bytes;
    const auto complement_bytes = complement_slots * slot_bytes;
    host_want = std::max(layer_bytes, complement_bytes);
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
    const auto routed_ask = [&] {
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
    // P0.1: allocate first, size second. A model whose Device pool holds every layer never builds a
    // host tier, so it must not hold a mapping it will not use - hence the same `short_pool` test
    // the attach below uses, asked before the block exists rather than after.
    bool short_pool = false;
    for (std::size_t index = 0; index < requirements.banks.size(); ++index) {
        const auto& bank = requirements.banks[index];
        if (bank.resident) { continue; }
        if (instance.expert_cache->resident_capacity(static_cast<std::uint32_t>(index)) <
            bank.count) {
            short_pool = true;
        }
    }
    const bool host_tier_wanted = short_pool && !streams.empty();
    const std::uint64_t host_ask = routed_ask + prefetch_bytes + second_chance_ask;
    models::qwen3_5::execution::HostArenaBlock host_block;
    if (host_tier_wanted) {
        // An explicitly named size is asked for as named (P0.3: a refusal there is the caller's
        // error, not something to step around). Everything else steps down until it fits.
        const char* explicit_mb = std::getenv("NINFER_EXPERT_HOST_ARENA_MB");
        const bool  exact_size  = explicit_mb != nullptr && *explicit_mb != '\0' &&
                                 std::strtoull(explicit_mb, nullptr, 10) != 0;
        host_block = allocate_host_arena(host_ask, exact_size);
    }
    // The block is the arena, so the line below quotes what the allocator granted - not what was
    // asked for. `asked` is on the [host] line above beside the refusal count.
    const std::uint64_t arena_held = host_block.bytes != 0 ? host_block.bytes : host_ask;
    // The host tier has TWO consumers that size themselves independently: this warm arena, and the
    // KV pages a tiered context keeps in RAM (`context_cache.host_capacity_bytes`, which the context
    // planner owns and this ledger does not admit - the Host tier here holds only the three expert
    // owners). They must not each take "half of free" and then discover the overlap at run time, so
    // the two ask-sizes and the RAM they are asking against are printed together. Nothing is
    // enforced at this point on purpose: the KV figure is a planning input that is only final once a
    // sequence exists, and inventing a reduction from a number that is about to change would be
    // worse than showing the overlap. A controller that divides the host budget needs exactly this
    // line to divide it against.
    {
        const std::uint64_t free_now   = current_free_host_bytes();
        const std::uint64_t kv_host    = options.context_cache.host_capacity_bytes.value_or(0);
        const std::uint64_t reserve_mib = static_cast<std::uint64_t>(8ull << 30);
        const std::uint64_t reserve_live = free_now / 8 > reserve_mib ? free_now / 8 : reserve_mib;
        const std::uint64_t total_host = arena_held + kv_host;
        std::fprintf(stderr,
                     "[ctl] host split: free=%llu MiB | kv-host=%llu MiB | expert-arena=%llu MiB "
                     "(reserve %llu MiB) | combined=%llu MiB%s\n",
                     static_cast<unsigned long long>(free_now >> 20),
                     static_cast<unsigned long long>(kv_host >> 20),
                     static_cast<unsigned long long>(arena_held >> 20),
                     static_cast<unsigned long long>(reserve_live >> 20),
                     static_cast<unsigned long long>(total_host >> 20),
                     (free_now != 0 && total_host > free_now) ? " - OVERCOMMITS FREE RAM" : "");
        std::fflush(stderr);
    }
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
    // T156 (V10-01): the three device accounts are sized in the capacity path
    // (`resolve_context_capacity`), where the sequence planner's StateImage stride is available
    // before any allocation. Nothing is re-quoted here: a plan printed beside an allocation it did
    // not shape is exactly how this item's `state=0(layout not exposed)` line survived.
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
    (void)admit; // every admission below is read out of the plan, not written here
    // The warm tier's ceiling is the budget it just computed, not "a quarter of free RAM": the
    // partition is what the scheduler grants, and a module may not widen it on its own. The override
    // exists so a caller can grant less - and then the arena is what actually shrinks, rather than
    // the ledger being told a smaller number than the arena takes. The allocated block is the last
    // word: a grant larger than the bytes that exist would name slots the tier cannot fill.
    std::uint64_t host_cap = host_ask;
    if (const char* value = std::getenv("NINFER_BUDGET_RAM_MB"); value != nullptr && *value != '\0') {
        const auto pinned = static_cast<std::uint64_t>(std::strtoull(value, nullptr, 10)) << 20;
        if (pinned != 0 && pinned < host_cap) { host_cap = pinned; }
    }
    if (host_block.bytes != 0 && host_block.bytes < host_cap) { host_cap = host_block.bytes; }
    instance.ledger.set_cap(core::Tier::Host, host_cap);
    // Routed first, then prefetch, then the second-chance band. A tight cap drops the auxiliaries
    // before it touches the working set; it never takes a routed slot to pay for a cache.
    const std::uint64_t grant_routed = std::min(routed_ask, host_cap);
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
    // ---- the plan, then the ledger as its projection -------------------------------------------
    //
    // Every tier is divided here, once, among a closed list of consumers (core/tier_plan.h). The
    // ledger below is told these numbers and no others, so "who took the bytes" and "what was the
    // plan" cannot disagree. Three of the device figures are slices rather than claims (the
    // prediction region and the SLRU band are carved out of the pool; the startup allowance is what
    // the cap carries), and the KV host figure is a planning input the context planner claims for
    // itself - the table shows it without claiming it, which is what that split has always been.
    auto& plan = instance.tier_plan;
    plan.set_cap(core::Tier::Device, device_cap);
    plan.set_cap(core::Tier::Host, host_cap);
    const std::uint64_t region_bytes =
        static_cast<std::uint64_t>(instance.expert_cache->prefetch_slots_per_bank()) *
        instance.expert_cache->bank_count() * instance.expert_cache->slot_bytes();
    plan.set(core::Budget::DeviceExpertPool, instance.expert_cache->capacity_bytes());
    plan.set(core::Budget::DeviceKvPages, kv_reserve_bytes);
    plan.set(core::Budget::DevicePrefetch, region_bytes);
    // No bytes: SLRU chooses which half of a slot's life an expert is in, it does not hold memory of
    // its own (expert_stream.cpp, the 50/50 branch). A number here would be invented.
    plan.set(core::Budget::DeviceSlruBand, 0);
    plan.set(core::Budget::DeviceStartup, startup_headroom);
    plan.set(core::Budget::HostExpertArena, grant_routed);
    plan.set(core::Budget::HostReadahead, grant_prefetch);
    plan.set(core::Budget::HostSecondChance, grant_second_chance);
    plan.set(core::Budget::HostKvHost, options.context_cache.host_capacity_bytes.value_or(0));
    // SSD keeps no experts: the cold range is the complement of what the two warm tiers held, so it
    // is a result. The KV spill has no planning cap either - it is read back to be judged (re-read
    // count) - but the spec asks for a triple, and the file that exists does have a capacity, so all
    // three facts are recorded instead of one zero.
    plan.set(core::Budget::DiskKvSpill, 0);
    {
        core::KvSpillView spill;
        // The only thing that creates this tier: program_impl.cpp attaches a spill file when
        // NINFER_KV_SPILL_MB is set, and nothing else does. `enabled` is that switch, read here from
        // the same variable rather than inferred from a counter that would be zero before the run.
        const char* mb  = std::getenv("NINFER_KV_SPILL_MB");
        const auto  mib = (mb != nullptr && *mb != '\0') ? std::strtoull(mb, nullptr, 10) : 0ull;
        spill.enabled   = mib != 0;
        // cap: the only capacity that exists is the file's extent count, which the file's own attach
        // line reports. It is not a planning knob (a cap would only decide which pages get read cold
        // twice), so nothing is divided here; with no file there is nothing to cap.
        spill.cap = mib != 0 ? std::to_string(mib) + " MiB (file extents, not a plan claim)" : "none";
        // reread: there is no per-page re-read counter in the tree. kv_tier_stats() counts
        // spilled/fetched pages over the Host leg, which is a different question - so "none" here
        // rather than an invented 0 that would read as "measured, and it is fine".
        spill.reread = "none";
        plan.set_kv_spill(spill);
    }
    for (const auto id : {core::Budget::DeviceExpertPool, core::Budget::DeviceKvPages,
                          core::Budget::DeviceStartup, core::Budget::HostExpertArena,
                          core::Budget::HostReadahead, core::Budget::HostSecondChance}) {
        const auto& row = core::TierPlan::slot(id);
        if (plan.bytes_of(id) != 0) { (void)admit(row.tier, row.name, plan.bytes_of(id)); }
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
    // short_pool was decided before the arena was allocated: the same test, asked where the block is
    // requested rather than only here, so a fully resident pool never holds a mapping it cannot use.
    // This bank's own seats, not the narrowest bank's: with per-bank pricing a wide bank can fit
    // entirely while the sibling that set the pool-wide number does not.
    // port (sm_89): the host tier backs both forms. The fused form describes one bank per layer; the
    // gate/up/down form describes THREE - one per matrix - because each run's pool set, tier slot
    // table and GH3 placement word are its own. A split run's entry image carries that one matrix in
    // the cache's gate_up planes and a one-row placeholder in the down planes (the loader's
    // ExpertMatrixSpec above), so the tier is told exactly what the pool was told.
    const bool any_split = std::any_of(runs.begin(), runs.end(),
                                       [](const StreamRuns& entry) { return entry.split; });
    if (host_tier_wanted) {
        std::vector<models::qwen3_5::execution::HostExpertBankDesc> descs;
        descs.reserve(streams.size() * 3);
        // port (sm_89): the host arena describes a whole bank, but a per-expert artifact gives
        // the weight a parent of exactly one expert's rows. Scale that geometry back up to
        // the bank so the CPU op's bank-validation sees the same domain either way; a fused
        // artifact already has this shape and the call is a no-op there.
        const auto bank_geometry = [](const WeightGeometry& expert, std::uint32_t count) {
            const std::uint64_t shape[2] = {
                static_cast<std::uint64_t>(count) * expert.shape.front(), expert.shape.back()};
            return weight_geometry(expert.format, expert.layout, shape);
        };
        for (std::size_t i = 0; i < streams.size(); ++i) {
            if (runs[i].split) {
                for (std::size_t role = 0; role < 3; ++role) {
                    const auto matrix_role =
                        static_cast<models::qwen3_5::execution::ExpertMatrixRole>(role);
                    const auto geometry = bank_geometry(
                        streams[i]->split_expert_geometry(matrix_role), streams[i]->count());
                    descs.push_back(
                        {requirements.banks[runs[i].first + role],
                         streams[i]->split_layout(matrix_role), streams[i]->split_layout(matrix_role),
                         geometry, geometry, streams[i]->per_expert_objects()});
                }
                continue;
            }
            descs.push_back({requirements.banks[runs[i].first], streams[i]->gate_up_layout(),
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
                .second_chance_bytes = grant_second_chance},
            host_block);
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
        std::fprintf(stderr,
                     "[experts] host tier slots=%u (%u routed + %u read-ahead) slot_bytes=%llu "
                     "banks=%zu form=%s\n",
                     residency->slot_capacity(), residency->routed_capacity(),
                     residency->prefetch_slots(),
                     static_cast<unsigned long long>(residency->slot_bytes()), descs.size(),
                     any_split ? "gate/up/down (one tier bank per run)" : "fused");
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
    placement.gpu_experts = instance.expert_cache->resident_experts();
    for (const auto& bank : requirements.banks) { placement.total_experts += bank.count; }
    // One step routes up to `routed_top_k` experts per lane inside each bank, so a pool holding that
    // many slots per bank runs a step without streaming anything in. Expressed in bytes, this is the
    // number a caller can size a tier by (Engine::expert_step_bytes) - the model's activation rather
    // than a constant - and the one that says whether a run should be attempted at all.
    {
        const std::uint64_t lanes = std::max<std::uint32_t>(1, options.max_concurrency);
        std::uint64_t step_slots  = 0;
        for (const auto& bank : requirements.banks) {
            step_slots += std::min<std::uint64_t>(bank.count, lanes * routed_top_k);
        }
        instance.expert_step_bytes_ = step_slots * placement.slot_bytes;
    }
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
    // P1.4: the cold range is an OBSERVATION - what neither warm tier holds - and all three terms
    // come from this placement, the same source the `[budget] experts: ... unslotted=NN of MM` line
    // prints, not a second calculation. That is why the plan is reported here and not where it was
    // decided: its last line needs an answer only the attach knows.
    plan.set_cold_range(placement.total_experts, placement.gpu_experts, placement.host_experts);
    plan.report();
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

// What a pinned (and mapped) block can actually be backed by. Pinned memory is committed to the page
// file, and on WDDM the Device allocations are charged to the same budget: measured on this box with
// the 17.5 GiB pool held, `GlobalMemoryStatusEx` reported avail commit 23.3 GiB before the arena and
// the 22.4 GiB default arena succeeded. Physical RAM alone is not this number, which is why the
// arena's share of free RAM overshoots on a machine that also holds a 22 GiB expert pool.
std::uint64_t current_commit_headroom_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status) != 0) {
        // Keep a gigabyte for what the process still has to allocate after the tier (staging, the
        // run's own host buffers). Without it the request lands on the edge, and the edge is not a
        // smaller tier - see allocate_host_arena.
        constexpr std::uint64_t kReserve = 1ull << 30;
        return status.ullAvailPageFile > kReserve ? status.ullAvailPageFile - kReserve : 0;
    }
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

// The host tier's block, allocated before its size is decided.
//
// The old shape decided a byte budget first (an estimate, a share of free RAM and a fixed `21 GiB`
// ceiling) and allocated it lazily at first use. That orders the two facts backwards: only the
// allocator knows how much of this box can be mapped right now, and on this machine the answer
// moves (22527 MiB refused / 21504 accepted in the morning; 20941 refused in the evening, 18432 and
// 16384 accepted). A probe is not a substitute - the same bytes were accepted by a probe and then
// refused by the real request milliseconds later, and the probe cost 3.7 s to map and unmap 22 GiB.
//
// So: ask for the candidate, but never for more than the commit headroom can back, and size the tier
// from the bytes that were granted. The property is fixed by the call - locked (the CPU half streams
// through it) and mapped (the Device addresses it directly) - because that is the only form whose
// absence this tier can be judged on: a pageable block still runs, and silently turns every
// Device-side route into an upload. There is no fallback below this point.
//
// Why the clamp instead of "fail, then step down one GiB and try again": a refused mapped allocation
// is TERMINAL in this process, so the step-down only ever produced a fake second chance. Measured
// (standalone probe on this box, RTX 4090 / Win11): with 21.5 GiB of Device memory held, avail commit
// was 23.3 GiB; asking 44 GiB failed, and the failed request left commit used=91.1 GiB of a
// 91.1 GiB limit - avail 0 - after which every later request in that process (40, 32, 24, 20, 16,
// 12 GiB) failed with `resource already mapped`. In the engine the same pattern printed 42 attempts
// and no success. So the size is chosen from the budget that actually backs pinned memory before
// the first call, not discovered by breaking the process.
//
// `exact` is the explicit-override form (NINFER_EXPERT_HOST_ARENA_MB): a size the caller named is
// asked for as named, with no clamp - otherwise an impossible experiment would silently run at a
// smaller arena and the numbers would not be comparable. A refusal there is the caller's error.
models::qwen3_5::execution::HostArenaBlock allocate_host_arena(std::uint64_t want_bytes,
                                                               bool          exact) {
    models::qwen3_5::execution::HostArenaBlock block;
    if (want_bytes == 0) { return block; }
    const std::uint64_t asked    = want_bytes;
    const std::uint64_t headroom = current_commit_headroom_bytes();
    const std::uint64_t len      = (!exact && headroom != 0)
                                       ? std::min<std::uint64_t>(asked, headroom)
                                       : asked;
    if (len == 0) {
        std::fprintf(stderr,
                     "[host] arena allocation failed: asked %llu MiB, but no commit headroom is left "
                     "(%llu bytes available, a reserved gigabyte is not subtracted here) - the host "
                     "expert tier cannot exist\n",
                     static_cast<unsigned long long>(asked >> 20),
                     static_cast<unsigned long long>(headroom));
        std::fflush(stderr);
        throw std::runtime_error("expert host arena: no commit headroom for " +
                                 std::to_string(asked >> 20) + " MiB");
    }
    void*      locked = nullptr;
    const auto status =
        cudaHostAlloc(&locked, static_cast<std::size_t>(len), cudaHostAllocMapped);
    if (status == cudaSuccess) {
        void*      device = nullptr;
        const auto alias  = cudaHostGetDevicePointer(&device, locked, 0);
        if (alias != cudaSuccess) {
            // The mapping was granted but the alias was not: the bytes are real, the Device just
            // cannot name them. Clear the error so it cannot surface at the next launch and report
            // the state on the same line as the size.
            (void)cudaGetLastError();
            device = nullptr;
        }
        block.host   = locked;
        block.device = device;
        block.bytes  = len;
        std::fprintf(stderr,
                     "[host] arena allocated %llu MiB (asked %llu, commit headroom %llu) "
                     "device_accessible=%d - the tier is sized from this\n",
                     static_cast<unsigned long long>(len >> 20),
                     static_cast<unsigned long long>(asked >> 20),
                     static_cast<unsigned long long>(headroom >> 20), device != nullptr ? 1 : 0);
        std::fflush(stderr);
        return block;
    }
    (void)cudaGetLastError();
    // Fail at the allocation point. The alternative is what this replaces: a pageable stand-in that
    // still runs, keeps every Device-side route an upload, and surfaces at the first attention
    // launch as `cudaErrorLaunchOutOfResources` at qsa/prepare_launch.cu:333 - a sizing failure that
    // reads as a kernel bug. The first line names the arena, the bytes asked for and the driver's own
    // reason; the throw stops construction before any of that can be misattributed. No smaller
    // attempt follows: see the note above - a refusal has already spent the process's commit.
    std::fprintf(stderr,
                 "[host] arena allocation failed: asked %llu MiB (committed ask %llu MiB, headroom "
                 "%llu MiB) (%s) - the host expert tier cannot exist, there is no pageable stand-in "
                 "for it, and no smaller retry after a refusal\n",
                 static_cast<unsigned long long>(asked >> 20),
                 static_cast<unsigned long long>(len >> 20),
                 static_cast<unsigned long long>(headroom >> 20), cudaGetErrorString(status));
    std::fflush(stderr);
    throw std::runtime_error("expert host arena: cudaHostAlloc(Mapped) refused " +
                             std::to_string(len >> 20) + " MiB of the " +
                             std::to_string(asked >> 20) + " MiB asked for (" +
                             cudaGetErrorString(status) + ")");
}

// Host-side arena for experts the Device pool cannot hold. It takes what the caller says the host
// tier could usefully hold, capped by a share of free RAM so one process cannot claim the box.
// Anything past that streams in from the artifact: RAM first, SSD only when it does not fit.
// NINFER_EXPERT_HOST_ARENA_MB overrides the result when a fixed budget is wanted. The result is a
// CANDIDATE now: allocate_host_arena() above is what turns it into bytes the tier can address.
std::uint64_t host_expert_arena_bytes(std::uint64_t want_bytes) {
    const auto share = [&] {
        const auto free_bytes = current_free_host_bytes();
        if (free_bytes == 0) { return want_bytes; }
        // Leave the box room to breathe before taking a share. "Half of free" does not reserve any:
        // the arena is MAPPED (and on the fallback path PINNED), so the mapping itself needs headroom
        // on top of the arena it is mapping. Measured on this box, 2026-11-16: with ~52 GiB free the
        // arena came out at 26 GiB, `[host] mapped allocation of 26073 MiB failed: out of memory -
        // trying a registered fallback` followed, the registered fallback succeeded, and the run then
        // died at the first attention launch (`qsa/prepare_launch.cu:333`,
        // cudaErrorLaunchOutOfResources) - a sizing failure that reads exactly like a kernel bug, and
        // that a reserve of this kind prevents.
        //
        // The reserve is max(8 GiB, 1/8 of free): the constant covers a desktop that grows while the
        // model loads (a browser tab is a gigabyte), the fraction scales with the box. The share is
        // still a half of what REMAINS, so the arena does not grow into the reserve.
        const std::uint64_t reserve =
            std::max<std::uint64_t>(8ull << 30, free_bytes / 8);
        const std::uint64_t usable = free_bytes > reserve ? free_bytes - reserve : 0;
        return usable / 2;
    }();
    // There used to be a fixed `21 GiB` ceiling here. It is gone: a constant that has to track a
    // refusal cliff which moves within one day (22527 MiB in the morning, 20941 MiB at night) is a
    // guess with a date on it, and it was the reason "how much the tier may hold" and "how much
    // could actually be allocated" disagreed. The candidate below is an ASK; allocate_host_arena()
    // discovers the answer and the tier is sized from it.
    // port (sm_89): the 4 GiB working-set cap is gone, and the RAM share is now half instead of a
    // quarter.
    //
    // That cap was measured on the 35B split path, whose complement is a few GiB: there, a larger
    // arena spread the same working set over more pages and cost 0.43 ms per expert against
    // 0.041 ms - "more arena" really was worse. It does not transfer to an artifact whose
    // complement is 67.5 GiB: capping at 4 GiB keeps 63 GiB of experts on the artifact and turns
    // every CPU-half fill into a physical read (measured: 55200 unslotted experts, 1004 slots of
    // tier, 3.16 tok/s decode). The cap is therefore a function of the ask: a small complement is
    // still bounded by it, a large one is bounded by RAM instead.
    constexpr std::uint64_t kSmallWorkingSetCap = 4ull << 30;
    std::uint64_t budget = std::min<std::uint64_t>(want_bytes, share);
    if (want_bytes <= kSmallWorkingSetCap) {
        budget = std::min<std::uint64_t>({want_bytes, share, kSmallWorkingSetCap});
    }
    if (const char* value = std::getenv("NINFER_EXPERT_HOST_ARENA_MB"); value != nullptr && *value) {
        const auto mib = std::strtoull(value, nullptr, 10);
        // An explicit budget replaces the RAM share rather than tightening it, so a run can ask for
        // the whole model when the machine has room. It has to replace the working-set estimate too:
        // clamping to `want_bytes` made every request above the estimate (~937 MiB on this model) a
        // silent no-op, which is why earlier sweeps measured 600 MiB and 4 GiB as identical and
        // concluded the arena size did not matter.
        if (mib != 0) { budget = mib << 20; }
        return budget;
    }
    return budget;
}

// port (sm_89): is the KV residency model in force for this construction? It is OPT-IN, and only
// for an Explicit capacity:
//   * automatic already spends the leftover Device budget on pages, so it has nothing to tier;
//   * tiering the curve minimum means the Device pool is deliberately smaller than max_context, and
//     requests past it are refused with a diagnostic instead of served. That is the right trade for
//     a 256K plan on a 24 GB card, and the wrong default for everyone else - without the switch every
//     existing run keeps the fully resident plan (and the exact numbers) it had.
// NINFER_KV_NO_HOST_RESIDENCY is the A/B arm: it wins over the request, which is also how the
// historical "minimum Engine runtime reservation requires ..." failure is reproduced on demand.
bool kv_host_residency_requested(const EngineOptions& options) {
    if (options.kv_capacity.mode != KvCapacityMode::Explicit) { return false; }
    if (const char* off = std::getenv("NINFER_KV_NO_HOST_RESIDENCY");
        off != nullptr && *off != '\0' && std::strcmp(off, "0") != 0) {
        return false;
    }
    const char* on = std::getenv("NINFER_KV_HOST_RESIDENCY");
    return on != nullptr && *on != '\0' && std::strcmp(on, "0") != 0;
}

// Device memory a residency plan keeps free. It is priced from free-after-weights, and the graph
// capture plus the driver's own storage are paid from that same budget afterwards - measured on this
// card the capture alone adds ~54 MiB. A plan with zero slack dies inside the capture instead of
// starting, so the reservation holds this much back (overridable for a sweep).
//
// port (sm_89): 512 MiB was the first value and it is too small on this card. This is not a capture
// allowance: it is the room the process must still have AFTER the plan is priced, and everything the
// desktop, the driver or a later allocation takes comes out of it. When it runs out, WDDM pages this
// process's own allocations - the KV pool among them.
//
// Measured on the 256K dense arm (RTX 4090, 19 GiB of resident weights):
//   * 512 MiB -> 1,610-1,633 resident page groups (39.3-39.9%), free_after_startup 0.61-0.65 GiB
//   * 2 GiB   ->   910-932 resident page groups (22.2-22.8%), free_after_startup 2.23-2.29 GiB
// The extra margin is what absorbs memory the process does not control. With a second process holding
// Device memory (the "the desktop took it after startup" case) the 512 MiB arm is consistently slower:
// 16K-token prefill 1,998 tok/s against 2,392 at 2 GiB held by that process, and 457 against 788 tok/s
// at 12 GiB held. Hence 2 GiB as the default; NINFER_KV_RESIDENCY_SLACK_MB still overrides it.
std::size_t kv_residency_slack_bytes() {
    constexpr std::size_t kDefault = 2048ULL << 20;
    const char* value              = std::getenv("NINFER_KV_RESIDENCY_SLACK_MB");
    if (value == nullptr || *value == '\0') { return kDefault; }
    const auto mib = std::strtoull(value, nullptr, 10);
    return static_cast<std::size_t>(mib) << 20;
}

// The residency contract is read from the environment while the decoder layout is planned
// (state/decoder_state.cpp owns that gate), so the plan has to declare the window it will install
// before the layout is built rather than after.
void set_environment_value(const char* name, const std::string& value) {
#if defined(_WIN32)
    (void)_putenv_s(name, value.c_str());
#else
    (void)setenv(name, value.c_str(), 1);
#endif
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
                             const EngineOptions& options,
                             models::qwen3_5::execution::PleTableBinding ple)
    : model(std::move(source)), parameters(*model, std::move(ple)),
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

// port (sm_89): the KV storage the run will actually use.
//
// The perplexity app defaults to fp8 because the low-bit routes are what it measures, and a Qwen4Exp
// artifact runs its indexed layers through qsa_sparse_gqa, whose low-bit profiles are still behind
// NINFER_QSA_SPARSE_GQA_QUANT_KV on this port. The default therefore used to end at
//   "error: qsa_sparse_gqa: invalid cache storage (NINFER_QSA_SPARSE_GQA_QUANT_KV)"
// after the run had already printed the storage it was going to use. BF16 is the one profile this
// port has qualified, so that is what an inadmissible request becomes - printed with the reason, and
// only for a model that has QSA layers (no QSA layer, no QSA storage constraint). The knob is still
// the way to ask for a low-bit run, so an operator who sets it is not overruled by this.
void resolve_kv_storage(EngineOptions& options, const models::qwen3_5::TextConfig& text) {
    if (text.indexed_attention_layers == 0) { return; }
    if (models::qwen4_exp::execution::qsa_kv_storage_admitted(options.kv_cache)) { return; }
    const auto requested = options.kv_cache;
    options.kv_cache     = KvCacheStorage::BFloat16;
    std::fprintf(stderr,
                 "[kv] this artifact's indexed layers do not admit '%s' on this port (only bf16 is "
                 "enabled; NINFER_QSA_SPARSE_GQA_QUANT_KV=1 opts into the ported low-bit routes); "
                 "using bf16 for this run\n",
                 models::qwen4_exp::execution::qsa_kv_storage_name(requested));
    std::fflush(stderr);
}

// port (sm_89): whether a budget for the routed experts was NAMED. Unset is the controller's
// decision; "0" is a named budget (it asks for the weights resident), and so is any MiB value or
// "auto" - normalize_engine_options has already turned those into expert_cache_bytes.
[[nodiscard]] bool expert_budget_unnamed(const EngineOptions& options) noexcept {
    if (options.expert_cache_bytes != 0) { return false; }
    const char* value = std::getenv("NINFER_EXPERT_CACHE_MB");
    return value == nullptr || *value == '\0';
}

// The resident weight plan is priced before a byte is allocated (materializer.cpp), so "will it fit"
// is answerable here instead of by a cudaMalloc failure halfway through the upload. The startup
// allowance is the caller's own figure; the KV arena is NOT subtracted because it is planned from
// what is left AFTER the weights, and a plan that leaves too little for it is the existing
// "Engine runtime reservation requires ..." fallback's case, not this one.
[[nodiscard]] bool resident_plan_fits(std::uint64_t plan_bytes, std::size_t free_bytes,
                                      std::size_t headroom_bytes) noexcept {
    return plan_bytes <= free_bytes && free_bytes - plan_bytes >= headroom_bytes;
}

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
    resolve_kv_storage(options, plan.config().text);
    // port (sm_89): a resident MoE plan that cannot fit the card is a decision, not a crash. This
    // artifact is 115 GiB with 72.4 GiB of Device-resident weights against a 24 GiB card, and the
    // old default (no pool, experts resident) died in cudaMalloc while "loading weights". Re-plan
    // with the experts streamed; the pool and the Host arena are then sized by
    // attach_expert_cache()/plan_device_split() from the Device left after the KV arena, and the
    // measured 3.24 GiB resident plan is what loads. A named budget is never second-guessed.
    if (expert_budget_unnamed(options)) {
        // port (sm_89) P6: a repacked artifact keeps ONE OBJECT PER EXPERT
        // (`weight/expert/<layer>/<expert>/gate_up`), so no parent holds a whole bank and the FFN's
        // resident form has no view to read: parameters.cpp builds an ExpertStream for every layer
        // of such an artifact whatever the load plan decided. If the resident plan fits, no pool is
        // created and layer 0 dies with "expert stream is not attached" - which is what
        // qwen3_6_35b_a3b_e2 did on 2026-11-16 (19.6 GiB resident, 2.19 GiB of slack, eleven lines
        // of log with no [experts] line to say why). The stream and the pool are one decision, so it
        // is taken here, before anything is materialized: stream the experts and let the pool be
        // sized from what the KV arena leaves. A named budget is never second-guessed.
        if (plan.experts_per_object()) {
            const auto to_gib = [](std::uint64_t bytes) {
                return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
            };
            const auto resident         = plan.materialization().device_capacity_bytes;
            EngineOptions streamed      = options;
            streamed.expert_cache_bytes = kExpertCacheAuto;
            auto streamed_plan = models::qwen3_5::plan_load(reader, models::load_options(streamed));
            const auto streamed_bytes = streamed_plan.materialization().device_capacity_bytes;
            std::fprintf(stderr,
                         "[budget] experts: this artifact keeps one object per expert, so no parent "
                         "holds a resident bank view; the run streams them (%.1f GiB resident -> "
                         "%.1f GiB) and sizes the pool from what the KV arena leaves\n",
                         to_gib(resident), to_gib(streamed_bytes));
            std::fflush(stderr);
            plan    = std::move(streamed_plan);
            options = streamed;
        }
        // Query the free count rather than the WDDM evictable budget: this is a "cannot possibly
        // fit" gate, and overstating the Device would put the decision back where it was - in a
        // failed allocation.
        const auto free_bytes = current_free_device_bytes();
        const auto resident   = plan.materialization().device_capacity_bytes;
        if (!resident_plan_fits(resident, free_bytes, options.startup_planning_headroom_bytes)) {
            EngineOptions streamed      = options;
            streamed.expert_cache_bytes = kExpertCacheAuto;
            auto streamed_plan = models::qwen3_5::plan_load(reader, models::load_options(streamed));
            const auto streamed_bytes = streamed_plan.materialization().device_capacity_bytes;
            const auto to_gib         = [](std::uint64_t bytes) {
                return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
            };
            if (resident_plan_fits(streamed_bytes, free_bytes,
                                   options.startup_planning_headroom_bytes)) {
                std::fprintf(stderr,
                             "[budget] resident weights need %.1f GiB of Device memory, %.1f GiB "
                             "is free (%.1f GiB is held back for startup); streaming the routed "
                             "experts keeps %.1f GiB resident and sizes the pool from what is left\n",
                             to_gib(resident), to_gib(free_bytes),
                             to_gib(options.startup_planning_headroom_bytes), to_gib(streamed_bytes));
                std::fflush(stderr);
                plan    = std::move(streamed_plan);
                options = streamed;
            } else {
                // Moving the experts out did not help (a dense artifact has none to move), so the
                // run is left exactly as it was asked for: no silent downgrade, and the failure
                // that follows is the one the caller's request earns.
                std::fprintf(stderr,
                             "[budget] this artifact does not fit this Device: resident weights "
                             "need %.1f GiB, streaming the routed experts still needs %.1f GiB, "
                             "%.1f GiB is free\n",
                             to_gib(resident), to_gib(streamed_bytes), to_gib(free_bytes));
                std::fflush(stderr);
            }
        }
    }
    // The PLE table source is opened by the plan (it owns its read handles), so it survives both
    // the Reader above and this function: the Parameters it is installed in live as long as the
    // model. Keeping the shared_ptr here is only what carries it across materialization.
    const auto ple_table = plan.ple_table();
    const auto ple_eos   = plan.ple_eos();
    binding.complete();
    auto model =
        models::qwen3_5::materialize_model(std::move(plan), device, &options.startup_observer);
    device.synchronize();
    StartupPhaseScope frontend(options.startup_observer, StartupPhase::FrontendInitialize);
    // Qwen4Exp's n-gram embedding reads its table from disk, so its owner belongs to the
    // Parameters rather than to any pass: every text context this model builds hands the same
    // source to its PLE stage, which is what keeps the nine-column history and the hash tail
    // consistent across requests. The source owns its own read handles, so it outlives the Reader
    // this function opened.
    auto instance = std::make_unique<ModelInstance>(
        std::move(model), options,
        models::qwen3_5::execution::PleTableBinding{.source = ple_table, .eos = ple_eos});
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
    // P7.1 (V10-55): the KV plan is a named function rather than a lambda inside a lambda, because
    // the capacity re-solve below has to price the arena a SECOND time - against the budget the
    // expert pool just won - and it must use the same curve, the same residency strides and the same
    // slack as the first attempt. Two pricing paths would make the two plans incomparable.
    const auto weights_bytes = instance->model->storage_stats().device_capacity_bytes;
    const auto conservative =
        current_free_device_bytes(options.wddm_evictable_budget, weights_bytes);
    // port (sm_89): the residency split measured from the real layout. A model with no
    // full-attention pages reports a zero payload stride and keeps the fully resident plan.
    KvResidencyStrides strides = planner.residency_strides();
    strides.tier_minimum_residency = true;
    const bool tiered_mode = kv_host_residency_requested(options) &&
                             strides.payload_bytes_per_resident_page_group != 0;
    const std::size_t residency_slack = tiered_mode ? kv_residency_slack_bytes() : 0;
    // V10-01 (T156): the StateImage geometry already exists before the Program does - it is the
    // planner's `PersistentLayout::state_images` - so the context pool's three ledgers are sized
    // together here instead of being quoted after the allocation they did not shape. The anchor
    // `capacity_curve` prices is the full minimum plan, which already contains the execution lanes'
    // StateImages and every extra slot `--device-state-slots` names (planning/startup.cpp builds
    // `state_image_slots = max_concurrency + extra`) plus its own workspace. Those slots are
    // therefore the joint model's lane set, not a second charge, and the solver prices only slots
    // beyond them. Nothing in this tree supplies a checkpoint interval, so the checkpoint target
    // stays at the anchor's slot count; a real interval would also have to rebuild the sequence
    // layout with the slots it wants, because that layout is what allocates them.
    const std::size_t state_image_bytes    = planner.state_image_bytes();
    const std::uint32_t anchor_state_slots = static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(options.max_concurrency) +
        options.context_cache.device_state_slots.value_or(options.max_concurrency));
    const ContextCapacityCurve joint_capacity_curve{
        .kv                              = capacity_curve,
        .minimum_state_slots             = anchor_state_slots,
        .bytes_per_additional_state_slot = state_image_bytes,
        .workspace_bytes                 = 0,
        .control_bytes_per_page_group    = 0,
        .control_bytes_per_state_slot    = 0,
    };
    // The anchor above already holds the configured extra slots, so the joint cache must not ask
    // for them again. That also keeps the checkpoint target off (it needs a non-null slot count
    // and a nonzero interval, neither of which this path has).
    ContextCacheOptions joint_cache = options.context_cache;
    joint_cache.device_state_slots.reset();
    // Only automatic sizing is joint: a pinned capacity is the caller's number, and a model with
    // no StateImage stride has nothing to price on the state account.
    const bool joint_sizing =
        options.kv_capacity.mode == KvCapacityMode::Automatic && state_image_bytes != 0;

    const auto plan_kv_at = [&](std::size_t budget) -> KvCapacityResolution {
        if (!tiered_mode) {
            if (joint_sizing) {
                return resolve_context_capacity(options.kv_capacity, joint_cache,
                                                joint_capacity_curve, budget,
                                                /*checkpoint_interval=*/0);
            }
            return resolve_kv_capacity(options.kv_capacity, capacity_curve, budget);
        }
        const std::size_t usable = budget > residency_slack ? budget - residency_slack : 0;
        const auto tiered        = resolve_kv_capacity_with_residency(
            options.kv_capacity, capacity_curve, strides, usable);
        KvCapacityResolution out;
        out.mode                     = options.kv_capacity.mode;
        out.main_page_groups         = tiered.logical_page_groups;
        out.maximum_main_page_groups = capacity_curve.maximum_main_page_groups;
        out.resolved_tokens = capacity_curve.resolved_tokens(tiered.logical_page_groups);
        out.logical_page_groups  = tiered.logical_page_groups;
        out.resident_page_groups = tiered.resident_page_groups;
        out.device_kv_payload_bytes =
            strides.payload_bytes_per_resident_page_group *
            static_cast<std::size_t>(tiered.resident_page_groups);
        out.host_kv_payload_bytes = tiered.host_payload_bytes;
        out.kv_host_residency_active =
            tiered.residency_active && tiered.resident_page_groups != tiered.logical_page_groups;
        out.minimum_runtime_reservation_bytes = capacity_curve.minimum_device_reservation_bytes;
        out.bytes_per_additional_main_page_group =
            capacity_curve.bytes_per_additional_main_page_group;
        out.runtime_reservation_bytes     = tiered.device_reservation_bytes;
        out.available_after_weights_bytes = usable;
        out.automatic_headroom_bytes      = options.kv_capacity.automatic_headroom_bytes;
        out.planned_slack_bytes           = usable - tiered.device_reservation_bytes;
        out.residency_slack_bytes         = residency_slack;
        return out;
    };
    auto resolution = [&]() -> KvCapacityResolution {
        // A tiered plan is priced against what the driver can give back WITHOUT taking pages away
        // from somebody else. The WDDM evictable budget below deliberately plans into memory the
        // driver then has to reclaim, and reclaiming it from this process means paging the KV pool -
        // measured on the 256K dense arm as 18 tok/s prefill against 2,826. So the retry is for the
        // fully resident plan only; a tiered plan that does not fit in the honest budget is a plan
        // that would page.
        if (tiered_mode) { return plan_kv_at(conservative); }
        try {
            return plan_kv_at(conservative);
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
            return plan_kv_at(optimistic);
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
        // P7.1 (V10-55): the pool gets one chance to be re-solved instead of refused. The request
        // carries the Device bytes the controller's bet moves out of the KV leg; the arena is
        // priced again against the smaller budget, and only then is the pool attached. Every other
        // refusal (a dense artifact, a re-plan that claims more or cannot serve max_context) keeps
        // the first plan and attaches again with the caller's own numbers.
        ExpertPlacement placement;
        // A KV leg that must stay on the Device can only give up the bytes above its own floor; a
        // tiered leg moves pages to the Host, so it names no floor here. Read per call, because a
        // successful re-solve replaces `resolution`.
        const auto kv_floor = [&]() -> std::size_t {
            return tiered_mode ? 0 : resolution.minimum_runtime_reservation_bytes;
        };
        try {
            placement = attach_expert_cache(*instance, options, device,
                                            resolution.runtime_reservation_bytes, kv_floor(),
                                            /*rebind_allowed=*/true);
        } catch (const CapacityRebindRequest& rebind) {
            const auto released = rebind.kv_release_bytes;
            const auto budget   = conservative > released ? conservative - released : 0;
            bool       rebound  = false;
            if (budget != 0 && budget < conservative) {
                try {
                    const auto replanned = plan_kv_at(budget);
                    // A re-plan may only give bytes back, and it may not trade the configured
                    // context for them: both are the arena's contract, not the bet's.
                    if (replanned.resolved_tokens >= options.max_context &&
                        replanned.runtime_reservation_bytes < resolution.runtime_reservation_bytes) {
                        resolution = replanned;
                        rebound    = true;
                    }
                } catch (const std::invalid_argument& error) {
                    std::fprintf(stderr,
                                 "[plan] capacity: the KV leg cannot give up %llu MiB (%s)\n",
                                 static_cast<unsigned long long>(released >> 20), error.what());
                    std::fflush(stderr);
                }
            }
            if (rebound) {
                resolution.available_after_startup_bytes = current_free_device_bytes(
                    options.wddm_evictable_budget,
                    instance->model->storage_stats().device_capacity_bytes);
                std::fprintf(stderr,
                             "[plan] capacity: KV pages lowered to %llu MiB (%u of %u page(s) "
                             "resident) - the pool takes the %llu MiB the bet released\n",
                             static_cast<unsigned long long>(
                                 resolution.runtime_reservation_bytes >> 20),
                             resolution.resident_page_groups, resolution.main_page_groups,
                             static_cast<unsigned long long>(released >> 20));
                std::fflush(stderr);
            } else {
                std::fprintf(stderr,
                             "[plan] capacity: the re-solve kept the KV plan; the pool keeps the "
                             "bytes the split already gave it\n");
                std::fflush(stderr);
            }
            // The second attach never re-throws the bet: one re-solve is the contract, and a pool
            // that still does not fit after it is the fault the caller asked for.
            placement = attach_expert_cache(*instance, options, device,
                                            resolution.runtime_reservation_bytes, kv_floor(),
                                            /*rebind_allowed=*/false);
        }
        // One window per bank means no layer ever evicts another, so the resident set is fixed and
        // the decode graph may be captured. Uploading the whole set first keeps the capture from
        // recording an upload that a later step would then repeat with the wrong experts.
        //
        // A DENSE artifact never gets a pool: attach_expert_cache() returns before creating one
        // when the model has no routed bank, so naming a budget for it is a no-op, not an error -
        // the same invariant the pool-capacity read below already relies on. Reading through the
        // pointer unconditionally is what turned "NINFER_EXPERT_CACHE_MB set on a dense model"
        // into a null dereference (V10-68, measured with the 27B and any non-zero value).
        const bool every_layer =
            instance->expert_cache != nullptr && instance->expert_cache->can_hold_every_layer();
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
    // port (sm_89): a captured decode graph cannot contain a Host fetch - the copy is followed by a
    // cudaStreamSynchronize, which a capture forbids - so a tiered plan must not freeze one. The
    // graph allowance stays in the plan (it is what the curve priced) but no capture is requested,
    // exactly as the streamed expert tier does for the same reason.
    if (resolution.kv_host_residency_active && options.use_cuda_graph) {
        std::fprintf(stderr,
                     "[graph] skipped: KV residency may need a Host fetch inside the step (the "
                     "fetch's stream sync is illegal inside a capture)\n");
        std::fflush(stderr);
        options.use_cuda_graph = false;
        planner.set_use_cuda_graph(false);
    }
    // port (sm_89): a residency plan addresses EVERY logical page - the full-attention layers read
    // the whole history - so nothing may be masked. The window therefore keeps everything: sink 0
    // and a span as long as the logical address space, which makes eff_window == live on every
    // round. What is left of the absent-page machinery is then a pure fetch path: a page that is not
    // resident is copied back before the step reads it, and the append path refuses to grow past
    // what the pool can hold. The decoder layout refuses a Device pool below the logical capacity
    // unless a retention contract is in force, and that gate reads the environment at layout time
    // (state/decoder_state.cpp), so the contract has to be declared before the plan is finalized.
    if (resolution.kv_host_residency_active) {
        set_environment_value("NINFER_KV_EVICT_WINDOW",
                              std::to_string(resolution.logical_page_groups));
        models::qwen3_5::install_kv_eviction_window(0, resolution.logical_page_groups);
        constexpr double kMib = 1024.0 * 1024.0;
        // port (sm_89): say which slack produced the split. 512 MiB is a legal override and the arm
        // that reproduces the paging collapse, so the log has to name the value in force rather than
        // leave the reader to assume the default.
        std::fprintf(stderr,
                     "[kv-residency] slack=%.0f MiB (%s) usable=%.0f MiB logical=%u page(s) "
                     "resident=%u page(s) (%.1f%%) device_kv=%.1f MiB host_kv=%.1f MiB\n",
                     static_cast<double>(resolution.residency_slack_bytes) / kMib,
                     std::getenv("NINFER_KV_RESIDENCY_SLACK_MB") != nullptr
                         ? "NINFER_KV_RESIDENCY_SLACK_MB"
                         : "default",
                     static_cast<double>(resolution.available_after_weights_bytes) / kMib,
                     resolution.logical_page_groups, resolution.resident_page_groups,
                     100.0 * static_cast<double>(resolution.resident_page_groups) /
                         static_cast<double>(resolution.logical_page_groups),
                     static_cast<double>(resolution.device_kv_payload_bytes) / kMib,
                     static_cast<double>(resolution.host_kv_payload_bytes) / kMib);
        std::fflush(stderr);
    }
    auto sequence = std::move(planner).finalize(resolution.main_page_groups,
                                                resolution.resident_page_groups,
                                                resolution.host_kv_payload_bytes);
    // port (sm_89): the layout's own KV payload, not the stride estimate - the draft cache adds one
    // page of its own, which a per-page stride cannot see. This is the Device side of the two-tier
    // account the ledger and the summaries report.
    resolution.device_kv_payload_bytes = sequence.kv_payload_bytes();
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
        // Same three ledgers as the first attempt: the recovery layouts the same artifact with the
        // same slot count, so only the KV half of the curve moves.
        ContextCapacityCurve recovery_joint = joint_capacity_curve;
        recovery_joint.kv                   = recovery_curve;
        auto recovered = resolve_kv_capacity(options.kv_capacity, recovery_curve,
                                            planning_available);
        if (joint_sizing) {
            recovered = resolve_context_capacity(options.kv_capacity, joint_cache, recovery_joint,
                                                 planning_available,
                                                 /*checkpoint_interval=*/0);
        }
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
    // T156 (V10-01): the sizing decision, reported where it was taken. The three ledgers are the
    // ones the automatic path priced against one budget: page groups, the StateImages the anchor
    // holds, and the workspace inside that anchor. It replaces a line that quoted a plan after the
    // allocation it never shaped, with the state account hard-coded to zero.
    if (const char* joint = std::getenv("NINFER_CONTEXT_JOINT_PLAN");
        joint != nullptr && *joint == '1') {
        const std::uint64_t state_bytes =
            static_cast<std::uint64_t>(state_image_bytes) * anchor_state_slots;
        constexpr std::uint64_t kMib = 1024ULL * 1024ULL;
        std::fprintf(stderr,
                     "[joint] sizing: auto=%d joint=%d pages=%u/%u reservation=%llu MiB | "
                     "state=%llu MiB (%u slot(s) x %llu B, inside the anchor) | slack=%llu MiB\n",
                     options.kv_capacity.mode == KvCapacityMode::Automatic ? 1 : 0,
                     joint_sizing ? 1 : 0, resolution.main_page_groups,
                     resolution.maximum_main_page_groups,
                     static_cast<unsigned long long>(resolution.runtime_reservation_bytes / kMib),
                     static_cast<unsigned long long>(state_bytes / kMib), anchor_state_slots,
                     static_cast<unsigned long long>(state_image_bytes),
                     static_cast<unsigned long long>(resolution.planned_slack_bytes / kMib));
        std::fflush(stderr);
    }
    // T156 (upstream 049d5fb4): the verification, now that the layout is in hand. The stride the
    // sizing used came from the same host layout this accessor reads (StateImageStore::image_bytes
    // returns `device_->host_layout().image_bytes`), so the two must agree; the state bytes are
    // part of the reservation above, not a second charge on top of it.
    if (const char* joint = std::getenv("NINFER_CONTEXT_JOINT_PLAN");
        joint != nullptr && *joint == '1') {
        const auto& geometry = instance->program->checkpoint_geometry();
        // One StateImage per device slot: the lanes plus the extra slots the option names (the
        // option is normalized to max_concurrency when unset).
        const std::uint64_t lanes = options.max_concurrency;
        const std::uint64_t slots =
            lanes + options.context_cache.device_state_slots.value_or(lanes);
        constexpr std::uint64_t kMib = 1024ULL * 1024ULL;
        std::fprintf(stderr,
                     "[joint] exact: reservation=%llu MiB | state=%llu MiB (%llu slot(s) x %llu B) "
                     "| stride planned=%llu B match=%d\n",
                     static_cast<unsigned long long>(
                         instance->kv_capacity_resolution.runtime_reservation_bytes / kMib),
                     static_cast<unsigned long long>(geometry.state_image_bytes * slots / kMib),
                     static_cast<unsigned long long>(slots),
                     static_cast<unsigned long long>(geometry.state_image_bytes),
                     static_cast<unsigned long long>(state_image_bytes),
                     geometry.state_image_bytes == state_image_bytes ? 1 : 0);
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
    // port (sm_89): the two KV tiers, itemised in the same ledger the expert tiers use. Device holds
    // the resident page groups; the Host tier holds the payload of every logical page that is not
    // resident. Reported only for a tiered plan - a fully resident engine has one tier and the old
    // silence, so every recorded run keeps its log shape.
    if (resolution.kv_host_residency_active) {
        const auto runtime = instance->program->memory_summary();
        instance->ledger.set_cap(core::Tier::Device, resolution.runtime_reservation_bytes);
        (void)instance->ledger.reserve(core::Tier::Device, "kv-device",
                                       resolution.device_kv_payload_bytes);
        instance->ledger.set_cap(core::Tier::Host, runtime.host_context_capacity_bytes);
        (void)instance->ledger.reserve(core::Tier::Host, "kv-host",
                                       resolution.host_kv_payload_bytes);
        instance->ledger.report(resolution.device_kv_payload_bytes +
                                resolution.host_kv_payload_bytes);
    }
    const auto& stats = instance->model->storage_stats();
    LoadSummary summary;
    summary.architecture = models::architecture_name(instance->model->config().text.architecture);
    summary.model_name   = instance->model->info().name;
    // The memory summary's rotary facts: the window every pool below was sized against. Reported
    // here (with the rest of the startup ledger) because the reported capacity - not the trained
    // one - is what max_context is checked against.
    if (const auto& rope = instance->model->config().text.rope_parameters) {
        const auto capacity =
            rope->scaled_position_capacity(instance->model->config().text.max_position_embeddings);
        std::fprintf(stderr, "[mem] rope=%s factor=%g original_max=%u capacity=%llu\n",
                     rope->applied_rope_type(),
                     static_cast<double>(rope->yarn ? rope->yarn_factor : 1.0f),
                     rope->yarn ? rope->yarn_original_max_position_embeddings
                                : instance->model->config().text.max_position_embeddings,
                     static_cast<unsigned long long>(capacity));
        std::fflush(stderr);
    }
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
    summary.kv_logical_page_groups             = resolution.logical_page_groups != 0
                                                     ? resolution.logical_page_groups
                                                     : resolution.main_page_groups;
    summary.kv_resident_page_groups            = resolution.resident_page_groups != 0
                                                     ? resolution.resident_page_groups
                                                     : resolution.main_page_groups;
    summary.kv_device_payload_bytes            = resolution.device_kv_payload_bytes;
    summary.kv_host_payload_bytes              = resolution.host_kv_payload_bytes;
    summary.kv_residency_slack_bytes           = resolution.residency_slack_bytes;
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
        // The graph decision stays with the controller and with the capture guard, even here: a
        // tier that cannot hold every layer fails the guard inside construct_model_once() and says
        // why ("layer needs host fetch"), while a tier that can hold every layer is capturable and
        // keeps the caller's default-on graph. Forcing the flag off here made the fallback silently
        // diverge from every other construction, with nothing in the log to explain it.
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
