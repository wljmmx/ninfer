#pragma once

#include "models/qwen3_5/model.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/runtime_types.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "core/expert_file.h"
#include "runtime/engine/kv_capacity.h"

#include <memory>

namespace ninfer::artifact {
class ArtifactExpertReader;
}
namespace ninfer {
class ExpertCache;
}

namespace ninfer::models::qwen3_5::execution { class HostExpertResidency; }

namespace ninfer::runtime {

[[nodiscard]] EngineOptions normalize_engine_options(EngineOptions options);

struct ModelInstance {
    using ModelContract = models::qwen3_5::RuntimeTypes;

    std::unique_ptr<models::qwen3_5::Model> model;
    const models::qwen3_5::execution::Parameters parameters;
    models::qwen3_5::Frontend frontend;
    KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;
    std::unique_ptr<models::qwen3_5::Program> program;
    // port (sm_89): routed experts stream from the artifact into this shared pool.
    std::unique_ptr<artifact::ArtifactExpertReader> expert_source;
    std::unique_ptr<ExpertCache> expert_cache;
    std::unique_ptr<ExpertBlobReader> expert_host_source;
    // Host half of a split expert tier; only created when the Device pool cannot hold a layer.
    std::unique_ptr<models::qwen3_5::execution::HostExpertResidency> expert_host_residency;
    // port (sm_89): true when the whole resident set fits at once, so a captured decode graph may
    // already point into it and the pool must not change size underneath that graph.
    bool expert_pool_pinned = false;

    ModelInstance(std::unique_ptr<models::qwen3_5::Model> model, const EngineOptions& options);
    ~ModelInstance();
    ModelInstance(const ModelInstance&)            = delete;
    ModelInstance& operator=(const ModelInstance&) = delete;

    // port (sm_89): grows or shrinks the elastic expert pool while the model is loaded. The caller
    // must be at a step boundary; returns the capacity in effect afterwards, or zero when the model
    // has no expert pool. Experts of a dropped window stream in again on their next use.
    std::uint64_t resize_expert_pool(std::uint64_t target_bytes);
    [[nodiscard]] std::uint64_t expert_pool_capacity() const noexcept;
};

struct ConstructedModel {
    std::unique_ptr<ModelInstance> instance;
    LoadSummary load;
    ContextMachineCostModel context_cost;
};

[[nodiscard]] ConstructedModel construct_model(EngineOptions& options, DeviceContext& device);

} // namespace ninfer::runtime
