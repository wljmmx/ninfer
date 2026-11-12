#pragma once

#include "models/qwen3_5/config.h"
#include "models/registry.h"

#include <string_view>

namespace ninfer::models::qwen4_exp {

// The Qwen4Exp row of the registry's identity table: four residual streams mixed through a
// rank-`hc_lowrank` pair, one bank per routed-expert matrix, the artifact's own activation names,
// and a sigmoid linear-attention output gate.
[[nodiscard]] Identity identity() noexcept;

// The family's structural requirements, run on the parsed text config by the shared parser. A text
// config that declares Qwen4Exp's architecture but not its structure is refused here rather than by
// a shape mismatch inside execution.
void complete(qwen3_5::TextConfig& config);

// A Qwen4Exp text artifact's config: the shared parse (which already runs this family's identity and
// requirements) plus the check that the artifact really is this family.
[[nodiscard]] qwen3_5::Config parse_config(const artifact::Directory& directory,
                                           const LoadOptions& options);

} // namespace ninfer::models::qwen4_exp
