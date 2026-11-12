#include "models/qwen4_exp/config.h"

#include "artifact/schema.h"

#include <stdexcept>
#include <string>
#include <variant>

namespace ninfer::models::qwen4_exp {

using artifact::ArtifactError;

Identity identity() noexcept {
    return Identity{.mtp_architecture     = "Qwen4ExpMTP",
                    .moe                  = true,
                    .banked_experts       = true,
                    .aliased_activations  = true,
                    .four_stream_residual = true,
                    .output_gate          = "sigmoid",
                    .complete             = &complete};
}

void complete(qwen3_5::TextConfig& config) {
    // The residual is four streams: the embedding is expanded once, every block folds its grouped
    // normalized view to a single stream, and the correction splits the product back out. A config
    // without that structure is not this model's text block.
    if (!config.hc) {
        throw ArtifactError("Qwen4Exp text config: hc_count/hc_lowrank are required "
                            "(the residual is four streams)");
    }
    if (config.hc->count != 4) {
        throw ArtifactError("Qwen4Exp text config: hc_count must be four, got " +
                            std::to_string(config.hc->count));
    }
    if (config.hc->lowrank == 0) {
        throw ArtifactError("Qwen4Exp text config: hc_lowrank must be positive");
    }
    // The routed experts are this family's FFN.
    if (!std::holds_alternative<qwen3_5::MoeConfig>(config.ffn)) {
        throw ArtifactError("Qwen4Exp text config: the FFN must be a sparse MoE");
    }
}

qwen3_5::Config parse_config(const artifact::Directory& directory, const LoadOptions& options) {
    auto config = qwen3_5::parse_config(directory, options);
    if (config.text.architecture != Architecture::Qwen4Exp) {
        throw ArtifactError("Qwen4Exp loader was given a text config declaring " +
                            std::string(architecture_name(config.text.architecture)));
    }
    return config;
}

} // namespace ninfer::models::qwen4_exp
