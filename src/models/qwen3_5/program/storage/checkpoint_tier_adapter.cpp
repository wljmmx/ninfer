#include "models/qwen3_5/program/storage/checkpoint_tier_adapter.h"

#include "core/tier_checkpoint_payload.h"
#include "models/qwen3_5/program/program_impl.h"

#include <cstdio>
#include <vector>

namespace ninfer::models::qwen3_5 {
namespace {

core::TierCheckpointPayload to_payload(const CheckpointExport& kv,
                                       const CheckpointStateExport& state) {
    core::TierCheckpointPayload out;
    out.frontier         = kv.frontier;
    out.page_count       = kv.page_count;
    out.page_bytes       = static_cast<std::uint64_t>(kv.page_bytes);
    out.kv               = kv.kv_bytes;
    out.image_bytes      = state.image_bytes;
    out.valid_tokens     = state.valid_tokens;
    out.layers           = state.layers;
    out.conv_channels    = state.conv_channels;
    out.value_heads      = state.value_heads;
    out.value_head_dim   = state.value_head_dim;
    out.key_head_dim     = state.key_head_dim;
    out.hidden           = state.hidden;
    out.has_dflash_local = state.has_dflash_local;
    out.state            = state.bytes;
    return out;
}

void from_payload(const core::TierCheckpointPayload& payload, CheckpointExport& kv,
                  CheckpointStateExport& state) {
    kv.frontier   = payload.frontier;
    kv.page_count = payload.page_count;
    kv.page_bytes = static_cast<std::size_t>(payload.page_bytes);
    kv.kv_bytes   = payload.kv;
    // Not carried across processes: epochs name pages of the pool that exported the run, and a
    // reader's pages are its own. The importer addresses by page index and re-derives them.
    kv.epochs.clear();

    state.image_bytes       = payload.image_bytes;
    state.valid_tokens      = payload.valid_tokens;
    state.layers            = payload.layers;
    state.conv_channels     = payload.conv_channels;
    state.value_heads       = payload.value_heads;
    state.value_head_dim    = payload.value_head_dim;
    state.key_head_dim      = payload.key_head_dim;
    state.hidden            = payload.hidden;
    state.has_dflash_local  = payload.has_dflash_local;
    // A blob is always a materialised image; the flag only ever described the exporting process.
    state.materialised_host_copy = true;
    state.bytes                  = payload.state;
}

} // namespace

bool tier_store_export(core::KvTier& tier, const core::KvBlockKey& key, const CheckpointExport& kv,
                       const CheckpointStateExport& state) {
    const auto blob = core::encode_tier_checkpoint(to_payload(kv, state));
    if (blob.empty()) { return false; }
    return tier.store(key, blob);
}

bool tier_load_export(core::KvTier& tier, const core::KvBlockKey& key, CheckpointExport& kv,
                      CheckpointStateExport& state) {
    const auto hit = tier.lookup(key);
    if (hit.state == core::KvTierState::miss || hit.bytes == 0) { return false; }
    std::vector<std::byte> buffer(hit.bytes, std::byte{0});
    if (!tier.retrieve(key, buffer)) { return false; }
    const auto payload = core::decode_tier_checkpoint(buffer);
    if (!payload) { return false; }
    from_payload(*payload, kv, state);
    return true;
}

bool tier_round_trip_ok(core::KvTier& tier, const core::KvBlockKey& key, const CheckpointExport& kv,
                        const CheckpointStateExport& state) {
    if (!tier_store_export(tier, key, kv, state)) { return false; }
    CheckpointExport back;
    CheckpointStateExport back_state;
    if (!tier_load_export(tier, key, back, back_state)) { return false; }
    return back.frontier == kv.frontier && back.page_count == kv.page_count &&
           back.page_bytes == kv.page_bytes && back.kv_bytes == kv.kv_bytes &&
           back_state.image_bytes == state.image_bytes &&
           back_state.valid_tokens == state.valid_tokens && back_state.layers == state.layers &&
           back_state.hidden == state.hidden && back_state.bytes == state.bytes;
}

} // namespace ninfer::models::qwen3_5

namespace ninfer::models::qwen3_5::detail {

bool ProgramImpl::tier_selftest() {
    if (!kv_tier) { return false; }
    // A synthetic pair, so the check exercises the adapter and the backend without a model, a
    // device or a request. It answers exactly one question: can this directory hold a checkpoint
    // and give it back unchanged. A backend that fails here would fail every real store too, and
    // the failure would otherwise be attributed to the checkpoint path.
    CheckpointExport kv;
    kv.frontier   = 1;
    kv.page_count = 1;
    kv.page_bytes = 64;
    kv.kv_bytes.assign(64, std::byte{0x5a});
    CheckpointStateExport state;
    state.image_bytes  = 32;
    state.valid_tokens = 1;
    state.layers       = 1;
    state.hidden       = 8;
    state.bytes.assign(32, std::byte{0x3c});
    const core::KvBlockKey key{0x5e1f7e5712345678ULL, 1};
    const bool ok = tier_round_trip_ok(*kv_tier, key, kv, state);
    std::fprintf(stderr, "[kvt] tier selftest %s\n", ok ? "OK" : "FAIL");
    std::fflush(stderr);
    return ok;
}

} // namespace ninfer::models::qwen3_5::detail