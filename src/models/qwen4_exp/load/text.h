#pragma once

#include "models/load_rules.h"
#include "models/qwen3_5/load/bindings.h"

#include <string>

namespace ninfer::models::qwen4_exp::loading {

// This family's low-rank correction around one grouped stream norm. `inject` writes the streams back,
// which the final correction does without: it feeds the head instead.
[[nodiscard]] qwen3_5::HcWeights bind_hc(qwen3_5::loading::Bindings& bindings,
                                         const qwen3_5::TextConfig& config,
                                         const std::string& prefix, bool inject);

// The table the shared binder asks for when the artifact belongs to this family. It is one static
// record: the spellings are the artifact's, not a checkpoint's.
[[nodiscard]] const TextBindingRules& binding_rules() noexcept;

} // namespace ninfer::models::qwen4_exp::loading
