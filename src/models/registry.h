#pragma once

#include <string_view>

namespace ninfer::models {

enum class Architecture { Qwen3_5, Qwen3_5Moe, Qwen4Exp };

[[nodiscard]] Architecture resolve_architecture(std::string_view architecture,
                                                std::string_view model_type);
[[nodiscard]] std::string_view architecture_name(Architecture architecture) noexcept;

// Qwen4Exp reuses the Qwen3.5 MoE text mathematics; only its binding names and the extra
// stream/indexer/embedding structure differ.
[[nodiscard]] constexpr bool moe_text(Architecture architecture) noexcept {
    return architecture == Architecture::Qwen3_5Moe || architecture == Architecture::Qwen4Exp;
}

} // namespace ninfer::models
