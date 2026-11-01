#pragma once

// Q54: the adapter between what the engine exports and what the third tier stores.
//
// The engine already knows how to take a checkpoint apart (export_checkpoint / export_state) and how
// to put one back together (import_checkpoint_current_layout / import_state). The tier knows how to
// hold bytes on disk. Neither knows about the other, and this is the only place that has to: one
// function turns a pair of exports into a stored blob, one turns a stored blob back into a pair.
//
// Deliberately not a ProgramImpl member. Keeping it a free function on the tier means it can be
// exercised with a synthetic export, without a model, a device or a request.

#include "core/kv_tier.h"
#include "models/qwen3_5/program/program.h"

namespace ninfer::models::qwen3_5 {

// Stores the pair under `key`. False when the payload cannot be built (a KV run whose pages do not
// add up) or when the backend refuses the write. Never throws: a tier that is down must not take a
// request with it.
[[nodiscard]] bool tier_store_export(core::KvTier& tier, const core::KvBlockKey& key,
                                     const CheckpointExport& kv,
                                     const CheckpointStateExport& state);

// Reads the pair back. False on a miss, a decode the format refuses, or a backend failure; the
// outputs are only written on success, so a caller cannot act on half a checkpoint.
[[nodiscard]] bool tier_load_export(core::KvTier& tier, const core::KvBlockKey& key,
                                    CheckpointExport& kv, CheckpointStateExport& state);

// Store, read back, compare both halves. The tier is the one component that can be wrong in a way
// nothing downstream notices - a blob that decodes cleanly and carries another frontier's KV - so
// the round trip is a function the caller can run before it trusts a backend.
[[nodiscard]] bool tier_round_trip_ok(core::KvTier& tier, const core::KvBlockKey& key,
                                      const CheckpointExport& kv,
                                      const CheckpointStateExport& state);

} // namespace ninfer::models::qwen3_5
