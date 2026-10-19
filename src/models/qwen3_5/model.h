#pragma once

#include "artifact/materializer.h"
#include "models/common/moe/weights.h"
#include "models/qwen3_5/config.h"
#include "models/common/qwen/frontend/resources.h"
#include "models/qwen3_5/weights.h"
#include "ninfer/ops/weight_input.h"

#include <memory>
#include <span>
#include <string>

namespace ninfer::models::qwen3_5 {

struct InstanceInfo {
    std::string name;
};

class ResolvedLoadPlan;

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

    [[nodiscard]] std::span<const ExpertBankSource> expert_banks() const noexcept { return banks_; }

    // port (sm_89): empty unless the file tier is enabled; the reader outlives every cache use.
    [[nodiscard]] ExpertFileTier expert_file_tier() const noexcept {
        return file_reader_ ? ExpertFileTier{file_reader_.get(), file_banks_} : ExpertFileTier{};
    }

    [[nodiscard]] bool file_tier_enabled() const noexcept { return file_reader_ != nullptr; }

    [[nodiscard]] const qwen::FrontendResources& resources() const noexcept { return resources_; }

    [[nodiscard]] const InstanceInfo& info() const noexcept { return info_; }

    [[nodiscard]] const artifact::MaterializationStats& storage_stats() const noexcept {
        return backing_.stats();
    }

private:
    friend std::unique_ptr<Model> materialize_model(ResolvedLoadPlan&&, DeviceContext&,
                                                    const StartupObserver*);
    Model(Config config, LoadOptions options, ModelWeights weights, std::vector<BoundWeight> bound,
          std::vector<ExpertBankSource> banks, qwen::FrontendResources resources, InstanceInfo info,
          artifact::MaterializedArtifact backing, std::vector<ExpertFileBank> file_banks = {},
          std::unique_ptr<moe::ExpertFileReader> file_reader = {});

    // Destroy all borrowers before backing. The caller keeps DeviceContext alive through cleanup.
    artifact::MaterializedArtifact backing_;
    Config config_;
    LoadOptions options_;
    ModelWeights weights_;
    std::vector<BoundWeight> bound_;
    std::vector<ExpertBankSource> banks_;
    std::vector<ExpertFileBank> file_banks_;
    std::unique_ptr<moe::ExpertFileReader> file_reader_;
    qwen::FrontendResources resources_;
    InstanceInfo info_;
};

} // namespace ninfer::models::qwen3_5
