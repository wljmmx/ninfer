#include "models/text_core.h"

#include "models/config_json.h"

#include <string>
#include <vector>

namespace ninfer::models::text_core {
namespace {

using artifact::ArtifactError;
using artifact::Json;
using config_json::dimension;
using config_json::flag;
using config_json::name;
using config_json::positive_float;

// Every key the neutral reader may consume. A family appends the optional keys it owns, so the
// key census still rejects anything no one declares while staying free of family vocabulary.
std::vector<std::string_view> optional_keys(std::initializer_list<std::string_view> family) {
    std::vector<std::string_view> keys{
        "num_attention_heads", "num_key_value_heads", "head_dim", "rope_parameters",
        "linear_num_key_heads", "linear_key_head_dim", "linear_num_value_heads",
        "linear_value_head_dim", "linear_conv_kernel_dim", "intermediate_size", "num_experts",
        "num_experts_per_tok", "moe_intermediate_size", "shared_expert_intermediate_size",
        "attention_bias", "mamba_ssm_dtype", "norm_topk_prob",
        // Tolerated without a consumer: hidden_act is the only activation this engine implements,
        // and the sampler takes its stop IDs from generation_config.json.
        "hidden_act", "eos_token_id"};
    keys.insert(keys.end(), family.begin(), family.end());
    return keys;
}

} // namespace

qwen3_5::TextConfig read(const Json& value, Architecture architecture, bool mtp,
                         std::initializer_list<std::string_view> family_keys) {
    const std::vector<std::string_view> required{
        "architectures", "model_type", "hidden_size", "vocab_size", "num_hidden_layers",
        "max_position_embeddings", "tie_word_embeddings", "rms_norm_eps", "layer_types"};
    const auto optional = optional_keys(family_keys);
    config_json::require_members(value, required, optional, "text config");

    qwen3_5::TextConfig out;
    out.architecture = resolve_architecture(
        config_json::architecture(value), artifact::require_id(value.at("model_type"), "model_type"));
    out.hidden_size             = dimension(value, "hidden_size");
    out.vocab_size              = dimension(value, "vocab_size");
    out.num_hidden_layers       = dimension(value, "num_hidden_layers");
    out.max_position_embeddings = dimension(value, "max_position_embeddings");
    out.rms_norm_eps            = positive_float(value, "rms_norm_eps");
    if (!value.at("tie_word_embeddings").is_boolean()) {
        throw ArtifactError("tie_word_embeddings must be boolean");
    }
    out.tie_word_embeddings = value.at("tie_word_embeddings").get<bool>();

    const auto& layers = value.at("layer_types");
    if (!layers.is_array() || layers.size() != out.num_hidden_layers) {
        throw ArtifactError("layer_types must cover every text block");
    }
    for (const auto& layer : layers) {
        if (layer == "full_attention") {
            out.layer_types.push_back(qwen3_5::MixerKind::FullAttention);
            out.compact_layer_indices.push_back(out.full_attention_layers++);
        } else if (layer == "indexed_attention") {
            out.layer_types.push_back(qwen3_5::MixerKind::IndexedAttention);
            out.compact_layer_indices.push_back(out.full_attention_layers++);
            ++out.indexed_attention_layers;
        } else if (layer == "linear_attention") {
            out.layer_types.push_back(qwen3_5::MixerKind::LinearAttention);
            out.compact_layer_indices.push_back(out.linear_attention_layers++);
        } else {
            throw ArtifactError("unknown text layer_type");
        }
    }

    if (out.full_attention_layers || mtp || value.contains("head_dim") ||
        value.contains("rope_parameters") || value.contains("num_attention_heads") ||
        value.contains("num_key_value_heads")) {
        out.attention       = config_json::attention(value);
        out.rope_parameters = config_json::rope(value.at("rope_parameters"), out.attention->head_dim);
    }
    if (out.linear_attention_layers || value.contains("linear_num_key_heads") ||
        value.contains("linear_key_head_dim") || value.contains("linear_num_value_heads") ||
        value.contains("linear_value_head_dim") || value.contains("linear_conv_kernel_dim")) {
        qwen3_5::GdnConfig gdn{
            dimension(value, "linear_num_key_heads"), dimension(value, "linear_key_head_dim"),
            dimension(value, "linear_num_value_heads"), dimension(value, "linear_value_head_dim"),
            dimension(value, "linear_conv_kernel_dim")};
        if (gdn.linear_num_value_heads % gdn.linear_num_key_heads) {
            throw ArtifactError("GDN value heads must be divisible by key heads");
        }
        (void)gdn.conv_channels();
        out.gdn = gdn;
    }

    out.attention_bias  = flag(value, "attention_bias", false);
    out.norm_topk_prob  = flag(value, "norm_topk_prob", false);
    out.mamba_ssm_dtype = name(value, "mamba_ssm_dtype");

    if (identity(out.architecture).moe) {
        if (value.contains("intermediate_size")) {
            throw ArtifactError("MoE config carries Dense FFN width");
        }
        qwen3_5::MoeConfig moe{dimension(value, "num_experts"),
                               dimension(value, "num_experts_per_tok"),
                               dimension(value, "moe_intermediate_size"),
                               dimension(value, "shared_expert_intermediate_size")};
        if (moe.num_experts_per_tok > moe.num_experts) {
            throw ArtifactError("selected experts exceed expert count");
        }
        out.ffn = moe;
    } else {
        for (const auto* key : {"num_experts", "num_experts_per_tok", "moe_intermediate_size",
                                "shared_expert_intermediate_size"}) {
            if (value.contains(key)) { throw ArtifactError("Dense config carries MoE fields"); }
        }
        out.ffn = qwen3_5::DenseConfig{dimension(value, "intermediate_size")};
    }
    return out;
}

} // namespace ninfer::models::text_core
