#pragma once

#include "artifact/framing.h"
#include "artifact/materializer.h"
#include "models/qwen3_5/config.h"
#include "models/qwen3_5/expert_priority.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/weights.h"
#include "ninfer/ops/weight_input.h"

#include <memory>
#include <span>
#include <string>

namespace ninfer::models::qwen3_5 {

struct InstanceInfo {
    std::string name;
    std::string metadata_json;
    std::string provenance_json;
    artifact::ArtifactId artifact_id{};
};

class LoadPlan;

class Model {
public:
    ~Model();
    Model(const Model&)            = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&)                 = delete;
    Model& operator=(Model&&)      = delete;

    [[nodiscard]] const Config& config() const noexcept { return config_; }

    [[nodiscard]] const LoadOptions& options() const noexcept { return options_; }

    [[nodiscard]] const ModelWeights& weights() const noexcept { return weights_; }

    [[nodiscard]] const BoundWeight& weight(WeightId id) const { return bound_.at(id.index); }

    [[nodiscard]] ops::WeightInput input(WeightUseId id) const;
    [[nodiscard]] ops::WeightInput input(WeightId id) const;

    [[nodiscard]] std::span<const BoundWeight> weight_data() const noexcept { return bound_; }

    // port (sm_89): payload of a Host-resident object, used as the expert cache source.
    [[nodiscard]] std::span<const std::byte> host_bytes(artifact::ObjectHandle handle) const {
        return backing_.host_bytes(handle);
    }

    [[nodiscard]] const FrontendResources& resources() const noexcept { return resources_; }

    [[nodiscard]] const InstanceInfo& info() const noexcept { return info_; }

    // V10-14: the model's cold-start expert prior for one text block, as the calibration states it
    // (a permutation of every logical expert id, highest cache priority first). Empty when neither
    // the artifact nor an override carries one - the caller then installs nothing, and the expert
    // stream keeps the id order it used before this existed.
    [[nodiscard]] std::span<const std::uint32_t> expert_priority_row(std::size_t layer) const noexcept {
        return expert_priority_.row(layer);
    }
    [[nodiscard]] const ExpertPriorityTable& expert_priority_table() const noexcept {
        return expert_priority_;
    }

    [[nodiscard]] const artifact::MaterializationStats& storage_stats() const noexcept {
        return backing_.stats();
    }

private:
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, DeviceContext&,
                                                    const StartupObserver*);
    Model(Config config, LoadOptions options, ModelWeights weights, std::vector<BoundWeight> bound,
          FrontendResources resources, InstanceInfo info, ExpertPriorityTable expert_priority,
          artifact::MaterializedArtifact backing);

    // Destroy all borrowers before backing. The caller keeps DeviceContext alive through cleanup.
    artifact::MaterializedArtifact backing_;
    Config config_;
    LoadOptions options_;
    ModelWeights weights_;
    std::vector<BoundWeight> bound_;
    FrontendResources resources_;
    InstanceInfo info_;
    // V10-14: resolved once at load (the artifact's table, overlaid by an explicit override) and read
    // per layer where the expert stream is built. Never mutated afterwards.
    ExpertPriorityTable expert_priority_;
};

} // namespace ninfer::models::qwen3_5
