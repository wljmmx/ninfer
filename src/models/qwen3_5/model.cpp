#include "models/qwen3_5/model.h"

#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5 {

Model::Model(Config config, LoadOptions options, ModelWeights weights,
             std::vector<BoundWeight> bound, std::vector<ExpertBankSource> banks,
             qwen::FrontendResources resources, InstanceInfo info,
             artifact::MaterializedArtifact backing, std::vector<ExpertFileBank> file_banks,
             std::unique_ptr<moe::ExpertFileReader> file_reader)
    : backing_(std::move(backing)), config_(std::move(config)), options_(options),
      weights_(std::move(weights)), bound_(std::move(bound)), banks_(std::move(banks)),
      file_banks_(std::move(file_banks)), file_reader_(std::move(file_reader)),
      resources_(std::move(resources)), info_(std::move(info)) {}

Model::~Model() = default;

ops::WeightInput Model::input(WeightUseId id) const { return weight_input(bound_, id); }

ops::WeightInput Model::input(WeightId id) const { return weight_input(bound_, id); }

} // namespace ninfer::models::qwen3_5
