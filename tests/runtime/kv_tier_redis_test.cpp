// P1-b: the LMCache key format, the consecutive-prefix rule, and a KV tier that speaks both.
// The key/prefix checks always run; the Redis part runs when NINFER_RESP_PORT is set (77 = skipped).

#include "core/lmcache_key.h"
#include "core/prefix_index.h"
#include "core/redis_kv_tier.h"

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

    using ninfer::core::LmcacheKeyParts;
    using ninfer::core::lmcache_key_string;

    {
        const LmcacheKeyParts parts{.model = "qwen3.6-35b-a3b", .world_size = 1, .worker_id = 0,
                                    .chunk_hash = 0xDEADBEEF, .dtype = "bfloat16"};
        expect(lmcache_key_string(parts) == "qwen3.6-35b-a3b@1@0@deadbeef@bfloat16",
               "key matches CacheEngineKey.to_string (lowercase hex, no padding)");
    }
    {
        const LmcacheKeyParts parts{.model = "m", .world_size = 2, .worker_id = 3,
                                    .chunk_hash = 0, .dtype = "half"};
        expect(lmcache_key_string(parts) == "m@2@3@0@half", "zero hash prints as 0, not padded");
    }
    {
        const LmcacheKeyParts parts{.model = "m", .chunk_hash = 0x1F, .dtype = "float"};
        expect(lmcache_key_string(parts) == "m@1@0@1f@float", "a single hex digit stays one digit");
    }

    {
        ninfer::core::PrefixIndex index;
        const std::uint64_t a = 11, b = 22, c = 33, d = 44;
        index.record(a);
        index.record(b);
        index.record(c);
        index.record(d);
        const std::uint64_t full[]{a, b, c, d};
        expect(index.consecutive_match(full) == 4, "four stored chunks are four reusable chunks");

        const std::uint64_t longer[]{a, b, c, d, 99};
        expect(index.consecutive_match(longer) == 4, "the match stops at the first chunk we lack");

        index.forget(c);
        expect(index.consecutive_match(full) == 2, "a hole in the middle stops the match there");
        expect(index.consecutive_match(longer) == 2, "and it does not skip over the hole");

        index.record(c, false);
        expect(index.consecutive_match(full) == 2, "a chunk still being written is not reusable");
        index.record(c, true);
        expect(index.consecutive_match(full) == 4, "once written, it counts again");
    }

    const char* port_text = std::getenv("NINFER_RESP_PORT");
    if (port_text == nullptr || *port_text == '\0') {
        std::printf("SKIP no server: set NINFER_RESP_PORT to run the tier round trip\n");
        return failures == 0 ? 77 : 1;
    }
    const char* host_text = std::getenv("NINFER_RESP_HOST");
    const std::string host = (host_text != nullptr && *host_text != '\0') ? host_text : "127.0.0.1";
    const auto port = static_cast<std::uint16_t>(std::atoi(port_text));

    ninfer::core::RedisKvTier::Config config;
    config.host                 = host;
    config.port                 = port;
    config.identity.model       = "ninfer-tier-test";
    config.identity.world_size  = 1;
    config.identity.worker_id   = 0;
    config.identity.dtype       = "bfloat16";
    config.chunk_tokens         = 16;
    config.payload_bytes        = 4096;

    ninfer::core::RedisKvTier tier(config);
    if (!tier.connect()) {
        std::printf("FAIL connect: %s\n", tier.error().c_str());
        return 1;
    }

    const ninfer::core::KvBlockKey key{.hash = 0xABCDEF, .tokens = 16};
    std::vector<std::byte> payload(config.payload_bytes);
    for (std::size_t i = 0; i < payload.size(); ++i) {
        payload[i] = static_cast<std::byte>((i * 13u + 5u) & 0xFFu);
    }

    expect(tier.lookup(key).state == ninfer::core::KvTierState::miss, "an unknown chunk misses");
    expect(tier.store(key, payload), "store writes the block under the LMCache key");
    const auto hit = tier.lookup(key);
    expect(hit.state == ninfer::core::KvTierState::resident && hit.bytes == config.payload_bytes,
           "the stored chunk is resident with the configured size");
    std::vector<std::byte> out(config.payload_bytes);
    expect(tier.retrieve(key, out) && out == payload, "retrieve returns the same bytes");

    std::vector<std::byte> short_buffer(config.payload_bytes / 2);
    expect(!tier.retrieve(key, short_buffer), "a short buffer is refused, not truncated");

    const ninfer::core::KvBlockKey wrong_shape{.hash = 0xABCDEF, .tokens = 8};
    expect(!tier.store(wrong_shape, payload), "a chunk of a different size is refused");
    expect(tier.rejected_shape() >= 1, "and the refusal is counted");

    // Another writer (LMCache itself, in the real deployment) puts a block there; prefetch is what
    // discovers it, and from then on lookup answers without IO.
    const ninfer::core::KvBlockKey foreign{.hash = 0x1234, .tokens = 16};
    {
        ninfer::core::RespClient writer;
        if (!writer.connect_to(host, port)) { std::printf("FAIL writer connect\n"); return 1; }
        const auto written = writer.set(tier.key_string(foreign), payload);
        expect(written.has_value(), "an external writer stores a block under the same key");
    }
    expect(tier.lookup(foreign).state == ninfer::core::KvTierState::miss,
           "lookup alone does not do IO, so it does not see it yet");
    expect(tier.prefetch(foreign), "prefetch finds it and remembers it");
    expect(tier.lookup(foreign).state == ninfer::core::KvTierState::resident,
           "after prefetch the lookup answers from the directory");

    {
        ninfer::core::RespClient cleanup;
        if (cleanup.connect_to(host, port)) {
            (void)cleanup.del(tier.key_string(key));
            (void)cleanup.del(tier.key_string(foreign));
        }
    }

    std::printf("%s\n", failures == 0 ? "ALL OK" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
