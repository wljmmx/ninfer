#pragma once

// Q54: the payload the third tier carries, as bytes.
//
// `export_checkpoint()` already produces the KV run in logical page order, and `export_state()`
// produces the state image with the shape it was cut from. Neither knows about the tier, and the
// tier knows nothing about handles, page tables or state slots. This is the seam between them: one
// blob that holds both halves, self-describing, so a reader in another process - the whole point of
// a disk tier - can rebuild the same pair of exports.
//
// The shape travels with the state, not beside it. A state image restored into a different geometry
// is silent corruption: the bytes land in slots that mean something else, and the only symptom is
// text that is subtly wrong. Carrying the shape lets the importer refuse instead.
//
// L6-b: the state header also carries the checkpoint's real token length. The KV half travels in
// whole pages, so `page_count * page_tokens` is an upper bound on what the run covers, never the
// length itself -- the last page is usually part filled. A reader that only had the bound would
// either lose that tail (today) or, once the tail is padded to a whole page, claim coverage for
// tokens that were never prefilled. The state is the half that was captured after those tokens, so
// the number belongs here rather than beside it.

#include "core/kv_checkpoint_blob.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace ninfer::core {

// One checkpoint as the tier stores it: the KV run plus the recurrent state it was captured with.
// Field names mirror CheckpointExport / CheckpointStateExport so the adapter is a copy, not a
// translation, and a field added on either side shows up as a compile error rather than as a zero.
struct TierCheckpointPayload {
    std::uint32_t frontier   = 0;
    std::uint32_t page_count = 0;
    std::uint64_t page_bytes = 0;
    std::vector<std::byte> kv;

    std::uint64_t image_bytes    = 0;
    std::uint32_t layers         = 0;
    std::int32_t conv_channels   = 0;
    std::int32_t value_heads     = 0;
    std::int32_t value_head_dim  = 0;
    std::int32_t key_head_dim    = 0;
    std::int32_t hidden          = 0;
    bool has_dflash_local        = false;
    // Tokens the checkpoint really covers (L6-b). The page run is rounded up, so this is the only
    // field that says where the real prefix ends.
    std::uint32_t valid_tokens   = 0;
    std::vector<std::byte> state;
};

// 8 (image_bytes) + 8 * 4 (layers, five shape ints, the dflash flag, valid_tokens).
inline constexpr std::size_t kStateHeaderBytes = 8 + 4 * 8;

[[nodiscard]] inline std::vector<std::byte>
encode_tier_checkpoint(const TierCheckpointPayload& payload) {
    if (payload.page_bytes != 0 &&
        payload.page_bytes * payload.page_count != payload.kv.size()) {
        return {};
    }
    std::vector<std::byte> state;
    state.reserve(kStateHeaderBytes + payload.state.size());
    detail::put_u64(state, payload.image_bytes);
    detail::put_u32(state, payload.layers);
    detail::put_u32(state, static_cast<std::uint32_t>(payload.conv_channels));
    detail::put_u32(state, static_cast<std::uint32_t>(payload.value_heads));
    detail::put_u32(state, static_cast<std::uint32_t>(payload.value_head_dim));
    detail::put_u32(state, static_cast<std::uint32_t>(payload.key_head_dim));
    detail::put_u32(state, static_cast<std::uint32_t>(payload.hidden));
    detail::put_u32(state, payload.has_dflash_local ? 1U : 0U);
    detail::put_u32(state, payload.valid_tokens);
    state.insert(state.end(), payload.state.begin(), payload.state.end());

    const CheckpointSpan spans[2] = {
        CheckpointSpan{CheckpointSpanTag::KvPages,
                       std::span<const std::byte>(payload.kv.data(), payload.kv.size()),
                       static_cast<std::uint32_t>(payload.page_bytes), payload.page_count},
        CheckpointSpan{CheckpointSpanTag::StateSlots,
                       std::span<const std::byte>(state.data(), state.size()), 0, 0},
    };
    return encode_checkpoint(payload.frontier, std::span<const CheckpointSpan>(spans, 2));
}

[[nodiscard]] inline std::optional<TierCheckpointPayload>
decode_tier_checkpoint(std::span<const std::byte> blob) {
    const auto view = decode_checkpoint(blob);
    if (!view || view->spans.size() != 2) { return std::nullopt; }
    const CheckpointSpan* kv    = nullptr;
    const CheckpointSpan* state = nullptr;
    for (const auto& span : view->spans) {
        if (span.tag == CheckpointSpanTag::KvPages) { kv = &span; }
        if (span.tag == CheckpointSpanTag::StateSlots) { state = &span; }
    }
    // Both halves or nothing: a checkpoint with KV and no state would skip prefill and compute the
    // GDN layers from an empty state, which reads as a wrong answer rather than as a miss.
    if (kv == nullptr || state == nullptr) { return std::nullopt; }
    if (kv->page_bytes == 0 || kv->pages == 0) { return std::nullopt; }
    if (state->bytes.size() < kStateHeaderBytes) { return std::nullopt; }

    TierCheckpointPayload out;
    out.frontier   = view->frontier;
    out.page_count = kv->pages;
    out.page_bytes = kv->page_bytes;
    out.kv.assign(kv->bytes.begin(), kv->bytes.end());

    const auto* header    = state->bytes.data();
    out.image_bytes       = detail::get_u64(header);
    out.layers            = detail::get_u32(header + 8);
    out.conv_channels     = static_cast<std::int32_t>(detail::get_u32(header + 12));
    out.value_heads       = static_cast<std::int32_t>(detail::get_u32(header + 16));
    out.value_head_dim    = static_cast<std::int32_t>(detail::get_u32(header + 20));
    out.key_head_dim      = static_cast<std::int32_t>(detail::get_u32(header + 24));
    out.hidden            = static_cast<std::int32_t>(detail::get_u32(header + 28));
    out.has_dflash_local  = detail::get_u32(header + 32) != 0;
    out.valid_tokens      = detail::get_u32(header + 36);
    out.state.assign(state->bytes.begin() + static_cast<std::ptrdiff_t>(kStateHeaderBytes),
                     state->bytes.end());
    // The header's size and the bytes it arrived with have to agree: an image that claims 12 MiB and
    // carries 11 would be restored with its last layer read from whatever was in the slot.
    if (out.image_bytes != out.state.size()) { return std::nullopt; }
    // Length and page run have to close. A zero length is not a length, and a state captured after
    // more tokens than the KV run covers would let a restore resume over a tail the pages do not
    // hold. Only the upper side is checked: the run may be padded past the real length (L7), and
    // that is exactly the case this field exists to describe.
    if (out.valid_tokens == 0 || out.valid_tokens > out.frontier) { return std::nullopt; }
    return out;
}

} // namespace ninfer::core
