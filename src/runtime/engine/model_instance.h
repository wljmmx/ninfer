#pragma once

#include "models/qwen3_5/model.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/runtime_types.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "core/expert_file.h"
#include "core/memory_ledger.h"
#include "runtime/engine/kv_capacity.h"

#include <memory>
#include <vector>

namespace ninfer::artifact {
class ArtifactExpertReader;
}
namespace ninfer {
class ExpertCache;
}

namespace ninfer::models::qwen3_5::execution {
class ExpertStream;
class HostExpertResidency;
}

namespace ninfer::runtime {

[[nodiscard]] EngineOptions normalize_engine_options(EngineOptions options);

struct ModelInstance {
    using ModelContract = models::qwen3_5::RuntimeTypes;

    std::unique_ptr<models::qwen3_5::Model> model;
    const models::qwen3_5::execution::Parameters parameters;
    models::qwen3_5::Frontend frontend;
    KvCapacityResolution kv_capacity_resolution;
    // port (sm_89) / upstream 19367582: measured growth of the final Program backing beyond its
    // explicit reservation. Driver/module storage and delayed loading are signed facts, not part
    // of the plan; reported so the plan's constants can be checked against reality.
    std::int64_t startup_retained_bytes = 0;
    const std::uint32_t capacity;
    std::unique_ptr<models::qwen3_5::Program> program;
    // port (sm_89): routed experts stream from the artifact into this shared pool.
    std::unique_ptr<artifact::ArtifactExpertReader> expert_source;
    std::unique_ptr<ExpertBlobReader> expert_host_source;
    // Host half of a split expert tier; only created when the Device pool cannot hold a layer.
    //
    // Shared, because a read-ahead job outlives the call that issued it: the queue holds it weakly
    // and takes a strong reference only for the length of one fill. Declared before the cache and
    // the pool so that it is destroyed after both - the cache holds a non-owning ExpertArenaSource*
    // to it, and the reader it reads through is declared above and therefore outlives it.
    std::shared_ptr<models::qwen3_5::execution::HostExpertResidency> expert_host_residency;
    std::unique_ptr<ExpertCache> expert_cache;
    // port (sm_89): true when the whole resident set fits at once, so a captured decode graph may
    // already point into it and the pool must not change size underneath that graph.
    bool expert_pool_pinned = false;
    // port (sm_89): true when this instance attached to the process-wide read-ahead queue, so the
    // destructor knows whether it has to detach. The detach is what joins the worker: doing it in
    // the destructor body rather than in a static destructor keeps the join while the pinned arena
    // and the CUDA context are still alive.
    bool expert_prefetch_retained = false;
    // T131: every routed bank this instance streams, in layer order (the same list
    // attach_expert_cache built to size the pool). The control plane's freeze policy has to reach
    // ALL of them - a policy applied to bank 0 only would leave the other 39 banks on their
    // startup values - and the engine worker is the thread that writes them (the control-plane
    // contract in include/ninfer/engine.h).
    std::vector<models::qwen3_5::execution::ExpertStream*> expert_streams;
    // port (sm_89): the one place that knows what each tier holds (1y.7). Step two only registers
    // and prints; the allocations still use their own estimates.
    core::MemoryLedger ledger;

    ModelInstance(std::unique_ptr<models::qwen3_5::Model> model, const EngineOptions& options);
    ~ModelInstance();
    ModelInstance(const ModelInstance&)            = delete;
    ModelInstance& operator=(const ModelInstance&) = delete;

    // port (sm_89): grows or shrinks the elastic expert pool while the model is loaded. The caller
    // must be at a step boundary; returns the capacity in effect afterwards, or zero when the model
    // has no expert pool. Experts of a dropped window stream in again on their next use.
    std::uint64_t resize_expert_pool(std::uint64_t target_bytes);
    [[nodiscard]] std::uint64_t expert_pool_capacity() const noexcept;
    // 13.4: the KV half of the same contract. Pages are the unit (64 tokens each); the call
    // validates ownership first and returns the capacity actually in effect.
    std::uint32_t resize_kv_capacity(std::uint32_t target_pages);

    // T131: apply one freeze policy to every routed bank. Engine worker thread only (the streams
    // are not synchronized); a dense model - no expert_streams - is a no-op, not an error.
    void set_expert_freeze_policy(std::uint32_t freeze_after, std::uint32_t reselect_after,
                                  bool thaw_on_request);
    // Any bank has a freeze mechanism armed; any bank is currently frozen. Same thread rule.
    [[nodiscard]] bool expert_freeze_armed() const noexcept;
    [[nodiscard]] bool expert_frozen() const noexcept;
    // How many routed banks this model streams (0 = dense). The control plane reports it so a
    // dense deployment says "no expert banks" instead of implying a frozen placement exists.
    [[nodiscard]] std::size_t expert_bank_count() const noexcept {
        return expert_streams.size();
    }
};

struct ConstructedModel {
    std::unique_ptr<ModelInstance> instance;
    LoadSummary load;
    ContextMachineCostModel context_cost;
};

[[nodiscard]] ConstructedModel construct_model(EngineOptions& options, DeviceContext& device);

} // namespace ninfer::runtime
