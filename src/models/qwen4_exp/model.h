#pragma once

#include "artifact/materializer.h"
#include "models/common/moe/weights.h"
#include "models/common/qwen/frontend/resources.h"
#include "models/qwen4_exp/config.h"
#include "models/qwen4_exp/weights.h"
#include "models/qwen4_exp/ple/disk/table.h"

namespace ninfer::models::qwen4_exp {
class ResolvedLoadPlan;

struct InstanceInfo {
    std::string name;
};

class Model {
public:
    Model(const Model&)            = delete;
    Model& operator=(const Model&) = delete;

    [[nodiscard]] const Config& config() const { return config_; }

    [[nodiscard]] const LoadOptions& options() const { return options_; }

    [[nodiscard]] const ModelWeights& weights() const { return weights_; }

    [[nodiscard]] const BoundWeight& weight(WeightId id) const { return bound_.at(id.index); }

    [[nodiscard]] ops::WeightInput input(WeightId id) const { return weight_input(bound_, id); }

    [[nodiscard]] std::span<const BoundWeight> weight_data() const { return bound_; }

    [[nodiscard]] artifact::MaterializedArtifact& backing() noexcept { return backing_; }
    [[nodiscard]] std::span<const ExpertBankSource> expert_banks() const { return banks_; }

    // port (sm_89): expert uploads read from the artifact when the file tier is on.
    [[nodiscard]] ExpertFileTier expert_file_tier() const noexcept {
        return file_reader_ ? ExpertFileTier{file_reader_.get(), file_banks_} : ExpertFileTier{};
    }

    [[nodiscard]] const qwen::FrontendResources& resources() const { return resources_; }

    [[nodiscard]] const ple::disk::Table& ple_table() const { return *ple_; }

    [[nodiscard]] const InstanceInfo& info() const { return info_; }

    [[nodiscard]] const artifact::MaterializationStats& storage_stats() const {
        return backing_.stats();
    }

private:
    friend std::unique_ptr<Model> materialize_model(ResolvedLoadPlan&&, DeviceContext&,
                                                    const StartupObserver*,
                                                    artifact::MaterializedArtifact*);
    Model(Config, LoadOptions, ModelWeights, std::vector<BoundWeight>,
          std::vector<ExpertBankSource>, qwen::FrontendResources, std::string,
          artifact::MaterializedArtifact, std::unique_ptr<ple::disk::Table>,
          std::vector<ExpertFileBank> = {}, std::unique_ptr<moe::ExpertFileReader> = {});

    artifact::MaterializedArtifact backing_;
    std::unique_ptr<ple::disk::Table> ple_;
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
} // namespace ninfer::models::qwen4_exp



