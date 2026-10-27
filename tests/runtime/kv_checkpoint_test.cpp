// P1-b-2d (part 1): the checkpoint payload that the tier has to carry.
//
// The interesting cases are the failure ones: a blob is accepted only when every span sits exactly
// where the table says, covers the whole payload, and the digest still matches. Anything else must
// be rejected, because a partially decoded checkpoint would be handed to the engine as valid state.

#include "core/kv_checkpoint_blob.h"

#include <algorithm>
#include <cstddef>
#include <span>
#include <cstdio>
#include <vector>

int main() {
    int failures = 0;
    const auto expect = [&](bool condition, const char* what) {
        std::printf("%s %s\n", condition ? "ok  " : "FAIL", what);
        if (!condition) { ++failures; }
    };

    using ninfer::core::CheckpointSpan;
    using ninfer::core::CheckpointSpanTag;
    using ninfer::core::decode_checkpoint;
    using ninfer::core::encode_checkpoint;
    using ninfer::core::checkpoint_key;

    std::vector<std::byte> kv(4096), state(1024);
    for (std::size_t i = 0; i < kv.size(); ++i) { kv[i] = static_cast<std::byte>((i * 7) & 0xFF); }
    for (std::size_t i = 0; i < state.size(); ++i) { state[i] = static_cast<std::byte>((i * 13) & 0xFF); }

    const CheckpointSpan spans[]{
        {CheckpointSpanTag::KvPages, kv},
        {CheckpointSpanTag::StateSlots, state},
    };

    const auto blob = encode_checkpoint(803, spans);
    expect(blob.size() == ninfer::core::kCheckpointHeaderBytes + 2 * ninfer::core::kCheckpointSpanBytes +
                              kv.size() + state.size(),
           "the blob is header + span table + payload");

    const auto view = decode_checkpoint(blob);
    expect(view.has_value(), "a well-formed blob decodes");
    if (view) {
        expect(view->frontier == 803, "the frontier survives");
        expect(view->spans.size() == 2, "both spans are present");
        expect(view->spans[0].first == CheckpointSpanTag::KvPages &&
                   std::equal(view->spans[0].second.begin(), view->spans[0].second.end(), kv.begin()),
               "the KV span comes back byte for byte");
        expect(view->spans[1].first == CheckpointSpanTag::StateSlots &&
                   std::equal(view->spans[1].second.begin(), view->spans[1].second.end(), state.begin()),
               "the state span comes back byte for byte");
    }

    expect(!checkpoint_key(0).empty() && checkpoint_key(0).starts_with("ninfer.ckpt."),
           "checkpoint keys live in their own namespace");
    expect(checkpoint_key(1) != checkpoint_key(2), "different digests are different keys");

    {
        auto truncated = blob;
        truncated.pop_back();
        expect(!decode_checkpoint(truncated).has_value(), "a truncated blob is rejected");
    }
    {
        auto corrupt = blob;
        corrupt[ninfer::core::kCheckpointHeaderBytes + 2 * ninfer::core::kCheckpointSpanBytes + 10] ^=
            std::byte{0xFF};
        expect(!decode_checkpoint(corrupt).has_value(), "a flipped payload byte breaks the digest");
    }
    {
        auto bad_magic = blob;
        bad_magic[0] = std::byte{'X'};
        expect(!decode_checkpoint(bad_magic).has_value(), "a foreign blob is not mistaken for ours");
    }
    {
        // Move the second span's offset so the spans overlap: the table would still "sum" to the
        // payload length, so only the order check catches it.
        auto overlap = blob;
        overlap[ninfer::core::kCheckpointHeaderBytes + ninfer::core::kCheckpointSpanBytes + 8] =
            std::byte{0x01};
        expect(!decode_checkpoint(overlap).has_value(), "overlapping spans are rejected");
    }
    {
        const std::span<const CheckpointSpan> empty{};
        const auto empty_blob = encode_checkpoint(0, empty);
        const auto empty_view = decode_checkpoint(empty_blob);
        expect(empty_view.has_value() && empty_view->spans.empty(), "a checkpoint with no spans is still well formed");
    }

    std::printf("%s\n", failures == 0 ? "ALL OK" : "FAILURES");
    return failures == 0 ? 0 : 1;
}


