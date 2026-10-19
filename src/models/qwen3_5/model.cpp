#include "models/qwen3_5/model.h"

#include "models/common/moe/expert_host.h"

#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5 {

Model::Model(Config config, LoadOptions options, ModelWeights weights,
             std::vector<BoundWeight> bound, std::vector<ExpertBankSource> banks,
             qwen::FrontendResources resources, InstanceInfo info,
             artifact::MaterializedArtifact backing, std::vector<ExpertFileBank> file_banks,
             std::unique_ptr<moe::ExpertFileReader> file_reader,
             std::unique_ptr<moe::ExpertHostTier> host_tier)
    : backing_(std::move(backing)), config_(std::move(config)), options_(options),
      weights_(std::move(weights)), bound_(std::move(bound)), banks_(std::move(banks)),
      file_banks_(std::move(file_banks)), file_reader_(std::move(file_reader)),
      host_tier_(std::move(host_tier)), resources_(std::move(resources)), info_(std::move(info)) {}

Model::~Model() {
    if (host_tier_) {
        std::fprintf(stderr,
                     "expert host tier: loads=%llu hits=%llu evictions=%llu committed=%.2f GiB "
                     "peak=%.2f GiB quota=%s\n",
                     static_cast<unsigned long long>(host_tier_->loads()),
                     static_cast<unsigned long long>(host_tier_->hits()),
                     static_cast<unsigned long long>(host_tier_->evictions()),
                     static_cast<double>(host_tier_->committed_bytes()) / (1024.0 * 1024 * 1024),
                     static_cast<double>(host_tier_->peak_committed_bytes()) / (1024.0 * 1024 * 1024),
                     host_tier_->quota_bytes() ? "limited" : "none");
    }
}

HostResidency* Model::host_residency() const noexcept { return host_tier_.get(); }

ops::WeightInput Model::input(WeightUseId id) const { return weight_input(bound_, id); }

ops::WeightInput Model::input(WeightId id) const { return weight_input(bound_, id); }

} // namespace ninfer::models::qwen3_5
