#pragma once

#include "artifact/schema.h"
#include "models/qwen3_5/config.h"
#include "models/registry.h"

namespace ninfer::models::qwen3_5 {

// A Qwen3.5 / Qwen3.5MoE text component's keys, read into the shared text record. The keys this
// parser accepts are the ones these two architectures declare: the shared block geometry plus
// nothing. A text config carrying another architecture's structure (Qwen4Exp's four residual
// streams, sparse indexer, per-layer n-gram embeddings, sigmoid output gate, media sentinel ids) is
// refused by name instead of being read as if that structure were simply absent.
[[nodiscard]] TextConfig parse_text_config(Architecture architecture, const artifact::Json& value,
                                           bool mtp);

} // namespace ninfer::models::qwen3_5
