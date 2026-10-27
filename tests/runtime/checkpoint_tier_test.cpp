// P1-b-2d (part 2): checkpoint blobs through the KV tier, keyed by content digest.
// Runs against NINFER_RESP_PORT when set (77 = skipped otherwise).

#include "core/checkpoint_tier.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main() {
    int failures = 0;
    const auto expect = [&](bool condition, const char* what) {
        std::printf("%s %s\n", condition ? "ok  " : "FAIL", what);
        if (!condition) { ++failures; }
    };

    using ninfer::core::CheckpointSpan;
    using ninfer::core::CheckpointSpanTag;
    using ninfer::core::CheckpointTier;

    const char* port_text = std::getenv("NINFER_RESP_PORT");
    if (port_text == nullptr || *port_text == '\0') {
        std::printf("SKIP no server: set NINFER_RESP_PORT to run the checkpoint tier test\n");
        return failures == 0 ? 77 : 1;
    }
    const char* host_text = std::getenv("NINFER_RESP_HOST");
    const std::string host = (host_text != nullptr && *host_text != '\0') ? host_text : "127.0.0.1";

    CheckpointTier tier(host, static_cast<std::uint16_t>(std::atoi(port_text)));
    if (!tier.connect()) {
        std::printf("FAIL connect: %s\n", tier.error().c_str());
        return 1;
    }

    std::vector<std::byte> kv(256 * 1024), state(64 * 1024);
    for (std::size_t i = 0; i < kv.size(); ++i) { kv[i] = static_cast<std::byte>((i * 3) & 0xFF); }
    for (std::size_t i = 0; i < state.size(); ++i) { state[i] = static_cast<std::byte>((i * 11) & 0xFF); }
    const CheckpointSpan spans[]{
        {CheckpointSpanTag::KvPages, kv},
        {CheckpointSpanTag::StateSlots, state},
    };

    const auto digest = tier.store(803, spans);
    expect(digest.has_value(), "a 320 KiB checkpoint is stored under its digest");
    if (!digest) { return 1; }
    expect(tier.exists(*digest), "EXISTS sees the checkpoint");
    const auto encoded_size = ninfer::core::encode_checkpoint(803, spans).size();
    if (tier.bytes_written() != encoded_size) {
        std::printf("     wrote %llu, encoder says %zu\n",
                    static_cast<unsigned long long>(tier.bytes_written()), encoded_size);
    }
    expect(tier.bytes_written() == encoded_size, "the transport wrote exactly one blob");

    std::vector<std::byte> buffer;
    const auto loaded = tier.load(*digest, buffer);
    expect(loaded.has_value(), "the checkpoint reads back and decodes");
    if (loaded) {
        expect(loaded->frontier == 803, "the frontier survives the round trip");
        expect(loaded->spans.size() == 2, "both spans come back");
        expect(std::equal(loaded->spans[0].second.begin(), loaded->spans[0].second.end(), kv.begin()) &&
                   std::equal(loaded->spans[1].second.begin(), loaded->spans[1].second.end(), state.begin()),
               "KV and state bytes are identical");
    }

    // Same content, different frontier: a different name, because a checkpoint is only reusable at
    // the frontier it was captured at.
    const auto other = tier.store(1606, spans);
    expect(other.has_value() && *other != *digest, "the frontier is part of the identity");

    // Same content and frontier: the same key, and storing twice is harmless.
    const auto again = tier.store(803, spans);
    expect(again.has_value() && *again == *digest, "identical content maps to the same key");

    // A value that no longer decodes must not be handed back as a checkpoint.
    {
        ninfer::core::RespClient writer;
        if (!writer.connect_to(host, static_cast<std::uint16_t>(std::atoi(port_text)))) {
            std::printf("FAIL writer connect\n");
            return 1;
        }
        std::vector<std::byte> garbage(64, std::byte{0x7A});
        (void)writer.set(ninfer::core::checkpoint_key(0xABCDEF), garbage);
    }
    std::vector<std::byte> junk;
    expect(!tier.load(0xABCDEF, junk).has_value(), "a corrupted value is rejected, not decoded");

    std::vector<std::byte> missing;
    expect(!tier.load(0x12345678, missing).has_value(), "an unknown digest misses");

    {
        ninfer::core::RespClient cleanup;
        if (cleanup.connect_to(host, static_cast<std::uint16_t>(std::atoi(port_text)))) {
            (void)cleanup.del(ninfer::core::checkpoint_key(*digest));
            (void)cleanup.del(ninfer::core::checkpoint_key(*other));
            (void)cleanup.del(ninfer::core::checkpoint_key(0xABCDEF));
        }
    }

    std::printf("%s\n", failures == 0 ? "ALL OK" : "FAILURES");
    return failures == 0 ? 0 : 1;
}


