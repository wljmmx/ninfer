#pragma once

#include "artifact/binder.h"
#include "models/qwen3_5/config.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/weights.h"

#include <map>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::loading {

[[nodiscard]] FrontendResources bind_resources(artifact::Binder& binder, const Config& config);

struct PendingWeight {
    // The artifact binding that carries the value. It differs from reference.name once an alias or
    // an expert bank selected it, and it is the key a Use is looked up under.
    std::string binding_name;
    artifact::ParameterReference reference;
    std::vector<WeightUse> uses;
    std::vector<std::string> source_objects;
};

class Bindings {
public:
    explicit Bindings(artifact::Binder& binder) : binder(binder) {}

    [[nodiscard]] WeightId parameter(std::string name, artifact::Shape shape,
                                     std::vector<std::string> inputs   = {},
                                     std::optional<QType> exact_format = {},
                                     artifact::Residency residency    = artifact::Residency::Device);
    [[nodiscard]] WeightId direct(std::string name, artifact::Shape shape,
                                  QType format = QType::BF16);

    // Qwen4Exp stores some parameters under its own binding names. `aliased` keeps `name` as the
    // logical parameter and reads its value from `binding` instead; `banked` additionally takes
    // the row block a per-expert parameter occupies inside a packed expert bank.
    [[nodiscard]] WeightId aliased(std::string name, std::string binding, artifact::Shape shape,
                                   std::vector<std::string> inputs   = {},
                                   std::optional<QType> exact_format = {},
                                   artifact::Residency residency     = artifact::Residency::Device);
    [[nodiscard]] WeightId aliased_direct(std::string name, std::string binding,
                                          artifact::Shape shape, QType format = QType::BF16);
    [[nodiscard]] WeightId banked(std::string name, std::string binding, artifact::Shape shape,
                                  std::uint64_t first_row, std::uint64_t rows,
                                  std::vector<std::string> inputs = {},
                                  artifact::Residency residency   = artifact::Residency::Device);

    // Qwen4Exp spells the activation each projection consumes in its own vocabulary: the mixer
    // reads `attention_input` where the engine says `mixer_input`, and the MoE reads `moe_input`
    // where it says `ffn_input`. Registering the pair keeps the logical name as the engine's
    // vocabulary; `resolve_use` accepts either spelling and refuses a Use that matches both or
    // neither, naming the logical parameter.
    void alias_input(std::string logical, std::string artifact);

    [[nodiscard]] const PendingWeight& at(WeightId id) const { return weights.at(id.index); }

    [[nodiscard]] WeightUseId use(WeightId id, std::string_view input) const;

    artifact::Binder& binder;
    std::vector<PendingWeight> weights;
    // port (sm_89): routed experts are declared without materializing them.
    bool stream_experts = false;
    bool expert_host_copy = false;

private:
    struct Alias {
        std::string binding;
        std::uint64_t first_row = 0;
        std::uint64_t rows      = 0;
    };

    [[nodiscard]] WeightId bind(std::string name, artifact::Shape shape,
                                std::vector<std::string> inputs, std::optional<QType> exact_format,
                                artifact::Residency residency, std::optional<Alias> alias);

    // The Use under the logical parameter's own name, or under its registered artifact spelling.
    [[nodiscard]] const artifact::Use& resolve_use(const std::string& parameter,
                                                   std::string_view binding,
                                                   std::string_view input) const;

    std::map<std::string, WeightId, std::less<>> parameters_;
    std::map<std::string, std::string, std::less<>> input_aliases_;
};

[[nodiscard]] AttentionWeights bind_attention(Bindings& bindings, const TextConfig& config,
                                              const std::string& prefix, bool indexed);
[[nodiscard]] DenseWeights bind_dense(Bindings& bindings, std::uint64_t hidden,
                                      std::uint64_t intermediate, const std::string& prefix,
                                      bool draft = false);
[[nodiscard]] BlockWeights bind_block(Bindings& bindings, const TextConfig& config,
                                      const std::string& prefix, MixerKind mixer);
[[nodiscard]] TextWeights bind_text(Bindings& bindings, const TextConfig& config,
                                    const LoadOptions& options);
[[nodiscard]] VisionWeights bind_vision(Bindings& bindings, const VisionConfig& config,
                                        const TextConfig& target);
[[nodiscard]] MtpWeights bind_mtp(Bindings& bindings, const TextConfig& config,
                                  const TextWeights& target);
[[nodiscard]] DraftWeights bind_draft(Bindings& bindings, const DraftConfig& config,
                                      const TextConfig& target, const TextWeights& weights,
                                      const std::string& component);
void bind_dflash2(Bindings& bindings, DraftWeights& weights, const DraftConfig& config,
                  const TextConfig& target);
[[nodiscard]] ProposalWeights bind_proposal(Bindings& bindings, const artifact::Proposal& proposal,
                                            const TextConfig& target, const LoadOptions& options,
                                            std::uint32_t public_tokens);
[[nodiscard]] std::vector<BoundWeight>
resolve_weights(std::vector<PendingWeight>&& pending,
                const artifact::MaterializedArtifact& materialized);

} // namespace ninfer::models::qwen3_5::loading
