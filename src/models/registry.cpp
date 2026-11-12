#include "models/registry.h"
#include "models/load_rules.h"
#include "models/qwen3_5/config.h"
#include "models/qwen4_exp/config.h"
#include "models/qwen4_exp/load/text.h"

#include <stdexcept>
#include <string>

namespace ninfer::models {

Architecture resolve_architecture(std::string_view architecture, std::string_view model_type) {
    if (architecture == "Qwen3_5ForCausalLM" && model_type == "qwen3_5_text") {
        return Architecture::Qwen3_5;
    }
    if (architecture == "Qwen3_5MoeForCausalLM" && model_type == "qwen3_5_moe_text") {
        return Architecture::Qwen3_5Moe;
    }
    // The generation-only and multimodal entry names share one text artifact.
    if ((architecture == "Qwen4ExpForCausalLM" ||
         architecture == "Qwen4ExpForConditionalGeneration") &&
        model_type == "qwen4_exp_text") {
        return Architecture::Qwen4Exp;
    }
    throw std::invalid_argument("unsupported architecture/config pair " +
                                std::string(architecture) + "/" + std::string(model_type));
}

std::string_view architecture_name(Architecture architecture) noexcept {
    switch (architecture) {
    case Architecture::Qwen3_5:
        return "Qwen3_5ForCausalLM";
    case Architecture::Qwen3_5Moe:
        return "Qwen3_5MoeForCausalLM";
    case Architecture::Qwen4Exp:
        return "Qwen4ExpForCausalLM";
    }
    return {};
}

Identity identity(Architecture architecture) noexcept {
    switch (architecture) {
    case Architecture::Qwen3_5:
    case Architecture::Qwen3_5Moe:
        return qwen3_5::identity(architecture);
    case Architecture::Qwen4Exp:
        return qwen4_exp::identity();
    }
    return {};
}

const TextBindingRules* text_binding_rules(Architecture architecture) noexcept {
    switch (architecture) {
    case Architecture::Qwen3_5:
    case Architecture::Qwen3_5Moe:
        return nullptr;
    case Architecture::Qwen4Exp:
        return &qwen4_exp::loading::binding_rules();
    }
    return nullptr;
}

} // namespace ninfer::models
