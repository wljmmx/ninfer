#pragma once

#include "artifact/schema.h"
#include "models/qwen3_5/config.h"
#include "models/registry.h"

#include <initializer_list>
#include <string_view>

// The text keys every family declares: the block geometry, its layer mixers, the attention and
// RoPE subtrees, GDN, and the FFN. They are read into the engine's text record
// (qwen3_5::TextConfig, shared because the execution side is shared) here, once.
//
// What a family adds to that set is not this reader's business: `family_keys` names the optional
// keys the calling architecture declares, and each family passes its own list from its own
// `text_config.cpp`. Nothing here knows what a residual-stream count or a per-layer embedding is.
namespace ninfer::models::text_core {

[[nodiscard]] qwen3_5::TextConfig read(const artifact::Json& value, Architecture architecture,
                                       bool mtp,
                                       std::initializer_list<std::string_view> family_keys);

} // namespace ninfer::models::text_core
