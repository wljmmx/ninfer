#include "models/qwen3_5/config.h"
#include "models/qwen3_5/text_config.h"
#include "models/qwen4_exp/text_config.h"

#include "artifact/schema.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <set>
#include <string>

namespace ninfer::models::qwen3_5 {
namespace {

using artifact::ArtifactError;
using artifact::Json;
using artifact::require_members;

std::uint32_t integer(const Json& value, std::string_view label, bool positive = true) {
    const auto n = artifact::require_u64(value, label, positive);
    if (n > std::numeric_limits<std::uint32_t>::max()) {
        throw ArtifactError(std::string(label) + ": config dimension exceeds u32");
    }
    return static_cast<std::uint32_t>(n);
}

std::uint32_t dimension(const Json& config, const char* name) {
    return integer(config.at(name), name);
}

float positive_float(const Json& config, const char* name) {
    const auto& value = config.at(name);
    if (!value.is_number()) { throw ArtifactError(std::string(name) + " must be a real value"); }
    const float result = value.get<float>();
    if (!std::isfinite(result) || result <= 0) {
        throw ArtifactError(std::string(name) + " must be positive finite FP32");
    }
    return result;
}

bool flag(const Json& config, const char* name, bool fallback) {
    if (!config.contains(name)) { return fallback; }
    const auto& value = config.at(name);
    if (!value.is_boolean()) { throw ArtifactError(std::string(name) + " must be boolean"); }
    return value.get<bool>();
}

std::string name(const Json& config, const char* key) {
    return config.contains(key) ? artifact::require_id(config.at(key), key) : std::string{};
}

std::string architecture(const Json& config) {
    const auto& names = config.at("architectures");
    if (!names.is_array() || names.size() != 1) {
        throw ArtifactError("architectures must name one implementation");
    }
    return artifact::require_id(names[0], "architecture");
}

const artifact::Component& companion(const artifact::Directory& directory, std::string_view name) {
    const auto& component = directory.component(name);
    if (component.target != "text") {
        throw ArtifactError(std::string(name) + ": target must be text");
    }
    return component;
}

// The artifact names the activation its linear attention gates with (the checkpoint's gate name, falling
// back to `hidden_act`). A family states the one its artifacts declare; an artifact that names
// another one belongs to a different mathematics.

AttentionConfig attention(const Json& value) {
    AttentionConfig out{dimension(value, "num_attention_heads"),
                        dimension(value, "num_key_value_heads"), dimension(value, "head_dim")};
    if (out.num_attention_heads % out.num_key_value_heads) {
        throw ArtifactError("attention query heads must be divisible by KV heads");
    }
    return out;
}

constexpr double kYarnPi = 3.14159265358979323846;

enum class RopeScalingMode { Native, Yarn };

// NINFER_ROPE_SCALING selects whether the artifact's YaRN table is applied. `native` (the default)
// runs the trained theta^(-2*i/rotary_dim) frequencies and the trained position capacity even when
// the file carries a yarn table; `yarn` applies the table and extends the capacity by its factor.
// The same weights serve both windows, so the choice belongs to the run, not to the file.
RopeScalingMode rope_scaling_mode() {
    static const RopeScalingMode mode = [] {
        const char* value = std::getenv("NINFER_ROPE_SCALING");
        if (value == nullptr || *value == '\0') { return RopeScalingMode::Native; }
        const std::string name{value};
        if (name == "native") { return RopeScalingMode::Native; }
        if (name == "yarn") { return RopeScalingMode::Yarn; }
        throw ArtifactError("NINFER_ROPE_SCALING must be native or yarn, got '" + name + "'");
    }();
    return mode;
}

// YaRN (Peng et al. 2023) per-pair inverse frequencies: a linear ramp between the trained
// extrapolated frequency and the 1/factor interpolated one, bounded by the two correction
// dimensions derived from the trained position capacity.
void compute_yarn_frequencies(RopeConfig& out, double beta_fast, double beta_slow) {
    const std::uint32_t pairs = out.rotary_dim / 2;
    if (pairs == 0 || pairs > out.yarn_inv_frequency.size()) {
        throw ArtifactError("rotary width exceeds the YaRN frequency table");
    }
    const double base    = out.rope_theta;
    const double max_pos = out.yarn_original_max_position_embeddings;
    const double factor  = out.yarn_factor;
    const auto correction = [&](double rotations) {
        return (static_cast<double>(out.rotary_dim) *
                std::log(max_pos / (rotations * 2.0 * kYarnPi))) /
               (2.0 * std::log(base));
    };
    const double low_raw  = std::floor(correction(beta_fast));
    const double high_raw = std::ceil(correction(beta_slow));
    const double low      = std::max(low_raw, 0.0);
    const double high =
        std::min(high_raw, static_cast<double>(out.rotary_dim) - 1.0);
    for (std::uint32_t i = 0; i < pairs; ++i) {
        const double extrapolated =
            1.0 / std::pow(base, 2.0 * static_cast<double>(i) / static_cast<double>(out.rotary_dim));
        const double interpolated = extrapolated / factor;
        double ramp               = 0.0;
        if (high > low) {
            ramp = (static_cast<double>(i) - low) / (high - low);
            ramp = std::min(std::max(ramp, 0.0), 1.0);
        }
        out.yarn_inv_frequency[i] =
            static_cast<float>(interpolated * ramp + extrapolated * (1.0 - ramp));
    }
    out.yarn_pair_count = pairs;
}

RopeConfig rope(const Json& value, std::uint32_t head_dim) {
    require_members(value, {"rope_theta", "partial_rotary_factor", "mrope_section"},
                    {"rope_type", "factor", "original_max_position_embeddings", "beta_fast",
                     "beta_slow", "attention_factor", "mrope_interleaved"},
                    "text RoPE");
    RopeConfig out;
    out.rope_theta            = positive_float(value, "rope_theta");
    out.partial_rotary_factor = positive_float(value, "partial_rotary_factor");
    if (out.partial_rotary_factor > 1) { throw ArtifactError("partial_rotary_factor exceeds one"); }
    out.rotary_dim = static_cast<std::uint32_t>(double(head_dim) * out.partial_rotary_factor);
    if (!out.rotary_dim || out.rotary_dim % 2) {
        throw ArtifactError("rotary dimension must be positive and even");
    }
    const auto& sections = value.at("mrope_section");
    if (!sections.is_array() || sections.size() != 3) {
        throw ArtifactError("MRoPE requires three sections");
    }
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < 3; ++i) {
        out.mrope_section[i] = integer(sections[i], "MRoPE section", false);
        sum += out.mrope_section[i];
    }
    if (sum != out.rotary_dim / 2) {
        throw ArtifactError("MRoPE sections differ from the rotary width");
    }
    std::array<std::uint32_t, 3> counts{};
    for (std::uint32_t pair = 0; pair < out.rotary_dim / 2; ++pair) {
        const std::uint8_t axis = pair % 3 == 1 && pair < 3ULL * out.mrope_section[1]   ? 1
                                  : pair % 3 == 2 && pair < 3ULL * out.mrope_section[2] ? 2
                                                                                        : 0;
        out.pair_axes.push_back(axis);
        ++counts[axis];
    }
    if (counts != out.mrope_section) {
        throw ArtifactError("MRoPE sections exceed interleaved axis ranges");
    }
    out.mrope_interleaved = flag(value, "mrope_interleaved", true);
    if (!out.mrope_interleaved) { throw ArtifactError("non-interleaved MRoPE is not implemented"); }
    // The rope subtree is read whatever it declares, so the startup line can report it. Whether the
    // table is USED is decided below by NINFER_ROPE_SCALING: a default artifact and a yarn artifact
    // are the same weights, and the extended window is a deployment choice.
    const std::string declared =
        value.contains("rope_type") ? artifact::require_id(value.at("rope_type"), "rope_type")
                                    : std::string{"default"};
    if (declared != "yarn" && declared != "default") {
        throw ArtifactError("unsupported rope_type " + declared);
    }
    out.declared_rope_type = declared;
    out.yarn_declared      = declared == "yarn" || value.contains("factor") ||
                        value.contains("original_max_position_embeddings");
    if (value.contains("factor")) { out.yarn_declared_factor = positive_float(value, "factor"); }
    if (value.contains("original_max_position_embeddings")) {
        out.yarn_declared_original_max_position_embeddings =
            integer(value.at("original_max_position_embeddings"),
                    "original_max_position_embeddings");
    }
    if (rope_scaling_mode() != RopeScalingMode::Yarn) { return out; }
    if (!value.contains("factor") || !value.contains("original_max_position_embeddings")) {
        // The served window is a deployment choice, and a checkpoint exported without
        // `rope_scaling` still has the trained window to scale from. Let the run name the two
        // numbers it wants instead of requiring them to travel inside the file.
        const char* factor_env   = std::getenv("NINFER_ROPE_YARN_FACTOR");
        const char* original_env = std::getenv("NINFER_ROPE_YARN_ORIGINAL_MAX");
        if (factor_env == nullptr || *factor_env == '\0' || original_env == nullptr ||
            *original_env == '\0') {
            throw ArtifactError(
                "NINFER_ROPE_SCALING=yarn needs factor and original_max_position_embeddings, "
                "either in the artifact's rope_parameters or as NINFER_ROPE_YARN_FACTOR / "
                "NINFER_ROPE_YARN_ORIGINAL_MAX");
        }
        const double factor = std::strtod(factor_env, nullptr);
        const std::uint64_t original = std::strtoull(original_env, nullptr, 10);
        if (!(factor > 1.0) || original == 0) {
            throw ArtifactError("NINFER_ROPE_YARN_FACTOR must exceed 1 and "
                                "NINFER_ROPE_YARN_ORIGINAL_MAX must be positive");
        }
        out.yarn_declared_factor = static_cast<float>(factor);
        out.yarn_declared_original_max_position_embeddings = original;
    }
    out.yarn                                  = true;
    out.yarn_factor                           = out.yarn_declared_factor;
    out.yarn_original_max_position_embeddings =
        out.yarn_declared_original_max_position_embeddings;
    const double beta_fast =
        value.contains("beta_fast") ? positive_float(value, "beta_fast") : 32.0;
    const double beta_slow =
        value.contains("beta_slow") ? positive_float(value, "beta_slow") : 1.0;
    out.yarn_attention_factor =
        value.contains("attention_factor")
            ? positive_float(value, "attention_factor")
            : static_cast<float>(0.1 * std::log(static_cast<double>(out.yarn_factor)) + 1.0);
    compute_yarn_frequencies(out, beta_fast, beta_slow);
    return out;
}


VisionConfig vision(const Json& value) {
    require_members(value,
                    {"model_type", "depth", "hidden_size", "intermediate_size", "num_heads",
                     "patch_size", "temporal_patch_size", "spatial_merge_size",
                     "num_position_embeddings"},
                    {}, "vision config");
    // `qwen4_exp_vision` is the name the Qwen4Exp converter writes for the same tower
    // mathematics: an mmproj carries its own geometry, so the tower is family-independent and one
    // parser serves every family.
    if (value.at("model_type") != "qwen3_5_vision" &&
        value.at("model_type") != "qwen3_5_moe_vision" &&
        value.at("model_type") != "qwen4_exp_vision") {
        throw ArtifactError("unknown Vision model_type");
    }
    VisionConfig out;
    out.depth                   = dimension(value, "depth");
    out.hidden_size             = dimension(value, "hidden_size");
    out.intermediate_size       = dimension(value, "intermediate_size");
    out.num_heads               = dimension(value, "num_heads");
    out.patch_size              = dimension(value, "patch_size");
    out.temporal_patch_size     = dimension(value, "temporal_patch_size");
    out.spatial_merge_size      = dimension(value, "spatial_merge_size");
    out.num_position_embeddings = dimension(value, "num_position_embeddings");
    out.position_grid_side =
        static_cast<std::uint32_t>(std::sqrt(double(out.num_position_embeddings)));
    if (out.hidden_size % out.num_heads || (out.hidden_size / out.num_heads) % 4 ||
        std::uint64_t(out.position_grid_side) * out.position_grid_side !=
            out.num_position_embeddings) {
        throw ArtifactError("invalid Vision head or square position geometry");
    }
    (void)out.patch_width();
    (void)out.merger_width();
    return out;
}

DraftConfig draft(const Json& value, const TextConfig& target, bool dflash2) {
    require_members(value,
                    {"architectures", "model_type", "intermediate_size", "num_attention_heads",
                     "num_key_value_heads", "head_dim", "num_hidden_layers",
                     "max_position_embeddings", "rms_norm_eps", "rope_parameters", "layer_types",
                     "dflash_config"},
                    {"sliding_window"}, "draft config");
    if (architecture(value) != (dflash2 ? "DFlash2DraftModel" : "DFlashDraftModel") ||
        value.at("model_type") != "qwen3") {
        throw ArtifactError("draft architecture/model_type mismatch");
    }
    DraftConfig out;
    out.attention = attention(value);
    if (out.attention.head_dim % 2) { throw ArtifactError("draft head dimension must be even"); }
    out.intermediate_size       = dimension(value, "intermediate_size");
    out.num_hidden_layers       = dimension(value, "num_hidden_layers");
    out.max_position_embeddings = dimension(value, "max_position_embeddings");
    out.rms_norm_eps            = positive_float(value, "rms_norm_eps");
    require_members(value.at("rope_parameters"), {"rope_theta"}, {}, "draft RoPE");
    out.rope_theta     = positive_float(value.at("rope_parameters"), "rope_theta");
    const auto& layers = value.at("layer_types");
    if (!layers.is_array() || layers.size() != out.num_hidden_layers) {
        throw ArtifactError("draft layer_types length mismatch");
    }
    bool sliding = false;
    for (const auto& layer : layers) {
        if (layer == "full_attention") {
            out.layer_types.push_back(DraftAttentionKind::FullAttention);
        } else if (layer == "sliding_attention") {
            out.layer_types.push_back(DraftAttentionKind::SlidingAttention);
            sliding = true;
        } else {
            throw ArtifactError("unknown draft layer_type");
        }
    }
    if (sliding || value.contains("sliding_window")) {
        out.sliding_window = dimension(value, "sliding_window");
    }
    const auto& config = value.at("dflash_config");
    if (dflash2) {
        require_members(config,
                        {"target_layer_ids", "mask_token_id", "conv_kernel_size", "conv_group_size",
                         "selector_rank", "selector_top_k"},
                        {}, "DFlash2 config");
    } else {
        require_members(config, {"target_layer_ids", "mask_token_id"}, {}, "DFlash config");
    }
    const auto& taps = config.at("target_layer_ids");
    if (!taps.is_array() || taps.empty()) {
        throw ArtifactError("target_layer_ids must be nonempty");
    }
    std::set<std::uint32_t> unique;
    for (const auto& tap : taps) {
        const auto index = integer(tap, "target layer index", false);
        if (index >= target.num_hidden_layers || !unique.insert(index).second) {
            throw ArtifactError("invalid target layer index");
        }
        out.target_layer_ids.push_back(index);
    }
    out.mask_token_id = integer(config.at("mask_token_id"), "mask token", false);
    if (out.mask_token_id >= target.vocab_size) {
        throw ArtifactError("mask token exceeds target embedding rows");
    }
    if (dflash2) {
        DFlash2Config extra{
            dimension(config, "conv_kernel_size"), dimension(config, "conv_group_size"),
            dimension(config, "selector_rank"), dimension(config, "selector_top_k")};
        if (target.hidden_size % extra.conv_group_size) {
            throw ArtifactError("DFlash2 conv group does not divide hidden width");
        }
        out.dflash2 = extra;
    }
    return out;
}

} // namespace

std::uint64_t GdnConfig::conv_channels() const {
    return artifact::checked_add(artifact::checked_mul(2, key_width(), "GDN key width"),
                                 value_width(), "GDN convolution channels");
}

std::uint64_t HcConfig::stream_width(std::uint32_t hidden) const {
    return artifact::checked_mul(count, hidden, "HC stream width");
}

std::uint64_t PleConfig::table_rows() const {
    return artifact::checked_mul(shard_shape[0], split_parts, "PLE table rows");
}

std::uint64_t VisionConfig::patch_width() const {
    return artifact::checked_mul(artifact::checked_mul(3, temporal_patch_size, "patch channels"),
                                 std::uint64_t(patch_size) * patch_size, "patch width");
}

std::uint64_t VisionConfig::merger_width() const {
    return artifact::checked_mul(std::uint64_t(spatial_merge_size) * spatial_merge_size,
                                 hidden_size, "merger width");
}

Config parse_config(const artifact::Directory& directory, const LoadOptions& options) {
    try {
        if (options.purpose != EnginePurpose::Generation &&
            options.purpose != EnginePurpose::CausalScoring) {
            throw ArtifactError("unknown loading purpose");
        }
        if (options.purpose == EnginePurpose::CausalScoring &&
            (options.vision || options.speculative != SpeculativeBackend::None)) {
            throw ArtifactError("CausalScoring loads the Text backbone only");
        }
        if (options.speculative != SpeculativeBackend::None &&
            options.speculative_component().empty()) {
            throw ArtifactError("unknown speculative backend");
        }
        if (options.proposal_head != ProposalHead::Full &&
            options.proposal_head != ProposalHead::Optimized) {
            throw ArtifactError("unknown proposal head selection");
        }
        Config out;
        out.mtp  = options.speculative == SpeculativeBackend::Mtp;
        const auto& text_json = directory.component("text").config;
        const auto text_architecture = resolve_architecture(
            artifact::require_id(text_json.at("architectures").at(0), "architectures"),
            artifact::require_id(text_json.at("model_type"), "model_type"));
        out.text = text_architecture == Architecture::Qwen4Exp
                       ? qwen4_exp::parse_text_config(text_architecture, text_json, out.mtp)
                       : qwen3_5::parse_text_config(text_architecture, text_json, out.mtp);
        const auto family = ::ninfer::models::identity(out.text.architecture);
        if (family.complete != nullptr) { family.complete(out.text); }
        if (options.vision) { out.vision = vision(companion(directory, "vision").config); }
        if (out.mtp) {
            const auto& config = companion(directory, "mtp").config;
            // The draft component states its own architecture and the source's structural hints; the
            // block itself is described by the target's text config, so those hints are accepted
            // without a consumer rather than silently dropped.
            require_members(config, {"architectures"},
                            {"model_type", "num_hidden_layers", "layer_types", "hybrid",
                             "rope_theta", "mtp_use_hidden_state_from_layer",
                             "mtp_use_dedicated_embeddings"},
                            "MTP config");
            // The draft head must be the same mathematics as the target: the family states the
            // architecture name its MTP companion declares.
            const std::string expected{family.mtp_architecture};
            if (architecture(config) != expected) {
                throw ArtifactError("MTP architecture differs from target mathematics: expected " +
                                    expected);
            }
        }
        if (options.speculative == SpeculativeBackend::DFlash ||
            options.speculative == SpeculativeBackend::DFlash2) {
            out.draft = draft(companion(directory, options.speculative_component()).config,
                              out.text, options.speculative == SpeculativeBackend::DFlash2);
        }
        return out;
    } catch (const std::exception& error) {
        throw ArtifactError(std::string("Qwen3.5 config: ") + error.what());
    }
}

Identity identity(Architecture architecture) noexcept {
    const bool moe = architecture == Architecture::Qwen3_5Moe;
    return Identity{.mtp_architecture = moe ? "Qwen3_5MoeMTP" : "Qwen3_5MTP",
                    .moe             = moe,
                    .output_gate     = "silu"};
}

} // namespace ninfer::models::qwen3_5
