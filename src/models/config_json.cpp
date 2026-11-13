#include "models/config_json.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>
namespace ninfer::models::config_json {

using qwen3_5::AttentionConfig;
using qwen3_5::RopeConfig;

namespace {

using artifact::ArtifactError;

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

} // namespace

std::uint32_t integer(const Json& value, std::string_view label, bool positive) {
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

void require_members(const Json& value, std::span<const std::string_view> required,
                     std::span<const std::string_view> optional, std::string_view label) {
    if (!value.is_object()) { throw ArtifactError(std::string(label) + " must be an object"); }
    for (const auto key : required) {
        if (!value.contains(key)) {
            throw ArtifactError(std::string(label) + ": missing " + std::string(key));
        }
    }
    for (const auto& [key, unused] : value.items()) {
        if (std::find(required.begin(), required.end(), key) == required.end() &&
            std::find(optional.begin(), optional.end(), key) == optional.end()) {
            throw ArtifactError(std::string(label) + ": unknown member " + key);
        }
    }
}

void refuse_keys(const Json& config, std::initializer_list<std::string_view> keys,
                 std::string_view architecture) {
    for (const auto key : keys) {
        if (config.contains(key)) {
            throw ArtifactError("member '" + std::string(key) + "' does not belong to " +
                                std::string(architecture));
        }
    }
}

AttentionConfig attention(const Json& value) {
    AttentionConfig out{dimension(value, "num_attention_heads"),
                        dimension(value, "num_key_value_heads"), dimension(value, "head_dim")};
    if (out.num_attention_heads % out.num_key_value_heads) {
        throw ArtifactError("attention query heads must be divisible by KV heads");
    }
    return out;
}

RopeConfig rope(const Json& value, std::uint32_t head_dim) {
    artifact::require_members(value, {"rope_theta", "partial_rotary_factor", "mrope_section"},
                              {"rope_type", "factor", "original_max_position_embeddings",
                               "beta_fast", "beta_slow", "attention_factor", "mrope_interleaved"},
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

} // namespace ninfer::models::config_json
