#pragma once

#include "artifact/schema.h"
#include "models/qwen3_5/config.h"
#include "models/registry.h"

namespace ninfer::models::qwen4_exp {

// A Qwen4Exp text component's keys: the shared block geometry plus the structure this architecture
// adds - the residual as `hc_count` streams mixed through a rank-`hc_lowrank` pair, the sparse
// indexer behind its indexed layers, the per-layer n-gram embedding bank, the sigmoid output gate
// and the media sentinel ids.
//
// The constraints that make a config this model's block live here too: a text config that declares
// Qwen4Exp's architecture but not its four-stream residual is refused at parse time rather than by
// a shape mismatch inside execution.
[[nodiscard]] qwen3_5::TextConfig parse_text_config(Architecture architecture,
                                                    const artifact::Json& value, bool mtp);

} // namespace ninfer::models::qwen4_exp
