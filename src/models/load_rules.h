#pragma once

#include "models/qwen3_5/weights.h"

#include <cstdint>
#include <string>

namespace ninfer::models::qwen3_5 {
struct TextConfig;
struct TextWeights;
struct BlockWeights;
struct MtpWeights;
namespace loading {
class Bindings;
} // namespace loading
} // namespace ninfer::models::qwen3_5

namespace ninfer::models {

enum class Architecture;

// The parts of a text artifact's binding vocabulary that belong to one family. A family spells and
// shapes some parameters its own way - a norm that covers every residual stream group, the correction
// pair folded around it, an n-gram embedding on named blocks, a draft head's own final correction -
// and the shared binder must not learn those spellings. It asks the family instead, through the table
// `text_binding_rules` returns for the architecture it is loading.
//
// Every hook is called where the shared binder would otherwise bind the same parameter, so the plan's
// order is unchanged: a family that supplies no table binds exactly the Qwen3.5 form.
struct TextBindingRules {
    // The block's two norms, when the family folds each into the correction that consumes it.
    // Called where the shared binder would bind `input_norm` and `post_attention_norm`.
    void (*block_norms)(qwen3_5::loading::Bindings& bindings, const qwen3_5::TextConfig& config,
                        const std::string& prefix, qwen3_5::BlockWeights& out) = nullptr;

    // The two low-rank corrections around those norms, written back by the residual.
    void (*block_corrections)(qwen3_5::loading::Bindings& bindings,
                              const qwen3_5::TextConfig& config, const std::string& prefix,
                              qwen3_5::BlockWeights& out) = nullptr;

    // The final correction the head reads, and the norm it folds.
    void (*final_correction)(qwen3_5::loading::Bindings& bindings,
                             const qwen3_5::TextConfig& config, qwen3_5::TextWeights& out) = nullptr;

    // The n-gram embedding a block the config names carries.
    void (*block_ple)(qwen3_5::loading::Bindings& bindings, const qwen3_5::TextConfig& config,
                      const std::string& prefix, qwen3_5::BlockWeights& out) = nullptr;

    // The draft head's norm, when the family keeps it inside the final correction.
    void (*mtp_norm)(qwen3_5::loading::Bindings& bindings, const qwen3_5::TextConfig& config,
                     qwen3_5::MtpWeights& out) = nullptr;

    // The draft head's final correction: a low-rank pair with no injection stream.
    void (*mtp_final_correction)(qwen3_5::loading::Bindings& bindings,
                                 const qwen3_5::TextConfig& config, qwen3_5::MtpWeights& out) = nullptr;
};

// The family's table, or null for a family that binds the shared vocabulary as it is.
[[nodiscard]] const TextBindingRules* text_binding_rules(Architecture architecture) noexcept;

} // namespace ninfer::models
