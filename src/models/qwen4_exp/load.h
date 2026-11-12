#pragma once

#include "models/qwen3_5/load.h"
#include "models/qwen3_5/weights.h"
#include "models/qwen4_exp/ple/table.h"

#include <cstdint>
#include <span>

namespace ninfer::artifact {
class Binder;
class Reader;
} // namespace ninfer::artifact

namespace ninfer::models::qwen3_5::loading {
struct PendingWeight;
} // namespace ninfer::models::qwen3_5::loading

namespace ninfer::models::qwen4_exp::loading {

// The PLE n-gram table's disk source, built while the artifact is still open. It owns its own read
// handles, so it outlives the Reader that described it; the load plan keeps it until the Parameters
// it belongs to are built.
//
// One table serves every PLE block - the stage gathers each block's embedding from the same n-gram
// bank - so a model whose `ple_layer_ids` name blocks over different objects is refused instead of
// silently running two of them against one bank. `pending` supplies the bound object ranges, and the
// hash controls are the artifact's `text/ple_hash_controls` resource.
[[nodiscard]] std::shared_ptr<const PleTableSource>
open_ple_table(const artifact::Reader& reader, artifact::Binder& binder,
               std::span<const qwen3_5::loading::PendingWeight> pending,
               const qwen3_5::TextWeights& weights, const qwen3_5::TextConfig& config);

// The n-gram hash substitutes this id for tokens before a sequence's start, so it is the token the
// artifact declares as end-of-sequence. -1 when the artifact does not declare one, which the text
// context refuses by name rather than hashing with a made-up id.
[[nodiscard]] std::int32_t ple_eos_token(const artifact::Directory& directory);

} // namespace ninfer::models::qwen4_exp::loading

namespace ninfer::models::qwen4_exp {

// The Qwen4Exp entry points. Loading is the shared text loader driven by this family's identity
// (four streams, banked experts, the artifact's own activation names, the MTP companion's name); the
// runtime picks the identity from the artifact, so these are the explicit handles a caller that
// already knows the family uses.
using qwen3_5::LoadPlan;
using qwen3_5::Model;
using qwen3_5::load_model;
using qwen3_5::materialize_model;

} // namespace ninfer::models::qwen4_exp
