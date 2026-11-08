#pragma once

#include "text/grammar.h"

#include <memory>
#include <string_view>

namespace ninfer::models::qwen3_5::frontend {

// Companion state for the static tool grammar: recognizes only the canonical outer framing and
// rejects duplicate top-level parameter names, including inside one candidate token.
std::unique_ptr<text::GrammarStateConstraint>
make_tool_name_constraint(std::shared_ptr<const text::GrammarVocabulary> vocabulary,
                          std::string_view reasoning_close, bool combined_body);

} // namespace ninfer::models::qwen3_5::frontend
