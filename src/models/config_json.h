#pragma once

#include "artifact/schema.h"
#include "models/qwen3_5/config.h"

#include <cstdint>
#include <initializer_list>
#include <span>
#include <string>
#include <string_view>

// Reading an architecture config out of its JSON is the same work whichever family wrote it: a
// dimension is a positive integer that fits u32, a flag is a boolean, `architectures` names exactly
// one implementation, and the attention and RoPE subtrees carry the mechanics this engine
// implements. Those primitives live here so that each family's text key table and its constraints
// can stay in its own `text_config.*`: what is shared is how a number is read, not which numbers a
// block declares.
namespace ninfer::models::config_json {

using artifact::Json;

[[nodiscard]] std::uint32_t integer(const Json& value, std::string_view label, bool positive = true);
[[nodiscard]] std::uint32_t dimension(const Json& config, const char* name);
[[nodiscard]] float positive_float(const Json& config, const char* name);
[[nodiscard]] bool flag(const Json& config, const char* name, bool fallback);
[[nodiscard]] std::string name(const Json& config, const char* key);
[[nodiscard]] std::string architecture(const Json& config);

// The same key-set rule as artifact::require_members, for a list the caller assembles at run time:
// a family appends the optional keys it adds to the ones every family declares.
void require_members(const Json& value, std::span<const std::string_view> required,
                     std::span<const std::string_view> optional, std::string_view label);

// The keys another architecture declares. A config that carries one of them is not this
// architecture, and reading it with these constraints would run another block's mathematics.
void refuse_keys(const Json& config, std::initializer_list<std::string_view> keys,
                 std::string_view architecture);

[[nodiscard]] qwen3_5::AttentionConfig attention(const Json& value);

// The RoPE subtree: partial rotary width, interleaved MRoPE sections, and - when the run asks for
// it through NINFER_ROPE_SCALING=yarn - the YaRN table the file carries.
[[nodiscard]] qwen3_5::RopeConfig rope(const Json& value, std::uint32_t head_dim);

} // namespace ninfer::models::config_json
