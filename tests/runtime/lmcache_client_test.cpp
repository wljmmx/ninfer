// Exercise the real client: one round trip through a leased connection, then the same work on a pool
// in parallel.
//
// What this is checking, beyond "it works": that a store carrying every KV chunk plus one state
// tensor addressed to the trailing chunk comes back byte for byte, and that N connections actually
// run at the same time -- the failure mode when they do not is silent (everything still passes, just
// serially), so the timing is printed rather than asserted.

#include "core/lmcache_client.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

using ninfer::core::lmcache::CacheKey;
using ninfer::core::lmcache::ClientOptions;
using ninfer::core::lmcache::Connection;
using ninfer::core::lmcache::GroupLayout;
using ninfer::core::lmcache::Pool;
using ninfer::core::lmcache::RegisterContext;
using ninfer::core::lmcache::RestorePayload;

std::int64_t g_instance = 9000;

std::string make_payload(std::size_t size, unsigned char seed) {
    std::string out(size, '\0');
    for (std::size_t i = 0; i < size; ++i) { out[i] = static_cast<char>(seed + (i % 251)); }
    return out;
}

RegisterContext make_registration(std::int64_t layers, std::int64_t chunk_tokens,
                                  std::int64_t kv_hidden, std::int64_t state_layers,
                                  std::int64_t state_hidden) {
    RegisterContext context;
    context.instance_id        = g_instance;
    context.model_name         = "client-test";
    context.world_size         = 1;
    context.block_size         = chunk_tokens;
    context.num_layers         = layers;
    context.hidden_dim_size    = kv_hidden;
    context.dtype_str          = "bfloat16";
    context.use_mla            = false;
    context.num_physical_slots = chunk_tokens;

    GroupLayout kv;
    kv.num_layers       = layers;
    kv.hidden_dim_size  = kv_hidden;
    kv.dtype_str        = "bfloat16";
    kv.tokens_per_block = chunk_tokens;
    kv.kv_size          = 2;
    kv.window_chunks    = 0;  // full attention: every chunk, from the first
    context.group_layouts.push_back(kv);

    GroupLayout state;
    state.num_layers       = state_layers;
    state.hidden_dim_size  = state_hidden;
    state.dtype_str        = "float32";
    state.tokens_per_block = chunk_tokens;
    state.kv_size          = 1;
    // One token per object: the state is an opaque image, not per-token data. The shape the server
    // derives is [layers, window_tokens, hidden], so this keeps it independent of chunk size.
    state.window_tokens = 1;
    // Without a one-chunk window the server expects this group to cover every chunk, and a retrieve
    // then misses no matter what was stored.
    state.window_chunks = 1;
    state.recurrent     = true;
    context.group_layouts.push_back(state);
    return context;
}

// The cache identity is the token sequence, so each key needs its own tokens; reusing a sequence
// would hit whatever an earlier run stored under it.
CacheKey make_key(int index, int chunks, std::int64_t chunk_tokens) {
    CacheKey key;
    key.model_name = "client-test";
    key.world_size = 1;
    key.worker_id  = 0;
    const std::int64_t base = (g_instance % 1000) * 1000 + index * 100000;
    std::vector<std::int64_t> tokens(static_cast<std::size_t>(chunks * chunk_tokens));
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = base + static_cast<std::int64_t>((i * 13 + 5) % 1000);
    }
    key.token_bytes = ninfer::core::lmcache::pack_tokens(tokens);
    key.start       = 0;
    key.end         = chunks * chunk_tokens;
    // Session state on the server is keyed by request_id and its rolling hash is not reset when the
    // token buffer is replaced, so this has to be unique per request, not per process.
    key.request_id = "req-" + std::to_string(g_instance) + "-" + std::to_string(index);
    return key;
}

struct Case {
    int chunks{5};
    std::int64_t layers{40};
    std::int64_t chunk_tokens{128};
    std::int64_t kv_hidden{128};
    // The state image is an opaque blob, not a per-layer structure, so its shape is only a way to
    // declare a byte count. Defaulting it to one "layer" lets a caller pin an exact image size
    // without having to make it divide by the KV layer count.
    std::int64_t state_layers{1};
    std::int64_t state_hidden{1600};
};

bool round_trip(Connection& connection, const Case& c, int index) {
    const std::size_t kv_bytes = static_cast<std::size_t>(2 * c.layers * c.chunk_tokens * c.kv_hidden) *
                                 sizeof(std::uint16_t);
    const std::size_t state_bytes =
        static_cast<std::size_t>(c.layers * c.state_hidden) * sizeof(std::uint32_t);

    std::vector<std::string> kv(static_cast<std::size_t>(c.chunks));
    for (int i = 0; i < c.chunks; ++i) {
        kv[static_cast<std::size_t>(i)] = make_payload(kv_bytes, static_cast<unsigned char>(0x10 + i));
    }
    const std::string state = make_payload(state_bytes, 0x77);

    const std::vector<std::int64_t> kv_shape{2, c.layers, c.chunk_tokens, c.kv_hidden};
    const std::vector<std::int64_t> state_shape{c.state_layers, 1, c.state_hidden};

    const CacheKey key = make_key(index, c.chunks, c.chunk_tokens);
    if (!connection.store(key, kv, kv_shape, state, state_shape)) {
        std::printf("FAIL store %d: %s\n", index, connection.error().c_str());
        return false;
    }
    if (!connection.has(key)) {
        std::printf("FAIL lookup %d: %s\n", index, connection.error().c_str());
        return false;
    }
    RestorePayload got;
    if (!connection.fetch(key, &got)) {
        std::printf("FAIL fetch %d: %s\n", index, connection.error().c_str());
        return false;
    }
    if (got.kv_chunks.size() != kv.size()) {
        std::printf("FAIL chunk count %d: %zu vs %zu\n", index, got.kv_chunks.size(), kv.size());
        return false;
    }
    for (std::size_t i = 0; i < kv.size(); ++i) {
        if (got.kv_chunks[i] != kv[i]) {
            std::printf("FAIL chunk %zu of %d differs (%zu vs %zu B)\n", i, index,
                        got.kv_chunks[i].size(), kv[i].size());
            return false;
        }
    }
    if (got.state != state) {
        std::printf("FAIL state of %d differs (%zu vs %zu B); server shape:", index,
                    got.state.size(), state.size());
        for (const auto dimension : got.state_shape) {
            std::printf(" %lld", static_cast<long long>(dimension));
        }
        std::printf("\n");
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    ClientOptions options;
    Case c;
    int connections = 4;

    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        const auto next = [&](std::int64_t& target) {
            if (i + 1 < argc) { target = std::strtoll(argv[++i], nullptr, 10); }
        };
        if (flag == "--server" && i + 1 < argc) {
            options.endpoint = argv[++i];
        } else if (flag == "--instance") {
            next(g_instance);
        } else if (flag == "--connections") {
            std::int64_t value = connections;
            next(value);
            connections = static_cast<int>(value);
        } else if (flag == "--chunks") {
            std::int64_t value = c.chunks;
            next(value);
            c.chunks = static_cast<int>(value);
        } else if (flag == "--layers") {
            next(c.layers);
        } else if (flag == "--chunk-tokens") {
            next(c.chunk_tokens);
        } else if (flag == "--kv-hidden") {
            next(c.kv_hidden);
        } else if (flag == "--state-hidden") {
            next(c.state_hidden);
        } else if (flag == "--state-layers") {
            next(c.state_layers);
        } else if (flag == "--help") {
            std::printf("usage: lmcache_client_test [--server URL] [--instance N] [--connections N]\n"
                        "                           [--chunks N] [--layers N] [--chunk-tokens N]\n"
                        "                           [--kv-hidden N] [--state-hidden N]\n");
            return 0;
        } else {
            std::printf("unknown argument: %s\n", argv[i]);
            return 2;
        }
    }

    const std::size_t kv_bytes = static_cast<std::size_t>(2 * c.layers * c.chunk_tokens * c.kv_hidden) *
                                 sizeof(std::uint16_t);
    const std::size_t state_bytes =
        static_cast<std::size_t>(c.state_layers * c.state_hidden) * sizeof(std::uint32_t);
    std::printf("geometry: %d chunk(s) x %zu B KV + %zu B state = %.1f MB per store, %d connection(s)\n",
                c.chunks, kv_bytes, state_bytes,
                (static_cast<double>(kv_bytes) * c.chunks + static_cast<double>(state_bytes)) / 1048576.0,
                connections);

    Pool pool;
    if (!pool.open(options, static_cast<std::size_t>(connections))) {
        std::printf("FAIL open: %s\n", pool.error().c_str());
        return 1;
    }
    const RegisterContext registration =
        make_registration(c.layers, c.chunk_tokens, c.kv_hidden, c.state_layers, c.state_hidden);
    if (const char* dump = std::getenv("LMCACHE_DUMP_REG")) {
        const std::string payload = ninfer::core::lmcache::encode_register_context(registration);
        std::ofstream out(dump, std::ios::binary);
        out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
        std::printf("dumped registration payload (%zu B) to %s\n", payload.size(), dump);
    }
    {
        const GroupLayout& kv = registration.group_layouts[0];
        const GroupLayout& state = registration.group_layouts[1];
        std::printf("register: kv [2,%lld,%lld,%lld] -> %zu B, state [%lld,1,%lld] -> %zu B\n",
                    static_cast<long long>(kv.num_layers),
                    static_cast<long long>(c.chunk_tokens),
                    static_cast<long long>(kv.hidden_dim_size),
                    static_cast<std::size_t>(2 * kv.num_layers * c.chunk_tokens * kv.hidden_dim_size) * 2,
                    static_cast<long long>(state.num_layers),
                    static_cast<long long>(state.hidden_dim_size),
                    static_cast<std::size_t>(state.num_layers * state.hidden_dim_size) * 4);
    }
    if (!pool.register_context(registration)) {
        std::printf("FAIL register: %s\n", pool.error().c_str());
        return 1;
    }
    std::printf("registered as instance %lld\n", static_cast<long long>(g_instance));

    int failures = 0;
    {
        auto lease = pool.acquire();
        if (!lease) {
            std::printf("FAIL acquire\n");
            return 1;
        }
        if (!round_trip(*lease, c, 0)) { ++failures; }
    }
    std::printf("%s single round trip\n", failures == 0 ? "ok  " : "FAIL");

    // Serial baseline, then the same work in parallel. Both do identical I/O; if the pool is not
    // actually concurrent the two durations come out the same.
    const auto run_batch = [&](bool parallel) {
        const auto start = std::chrono::steady_clock::now();
        std::vector<std::thread> threads;
        // Bytes, not vector<bool>: each thread writes its own slot, and a packed bit array would put
        // several of those slots in one machine word.
        std::vector<char> results(static_cast<std::size_t>(connections), 1);
        for (int i = 0; i < connections; ++i) {
            const auto body = [&, i]() {
                auto lease = pool.acquire();
                if (!lease) {
                    results[static_cast<std::size_t>(i)] = 0;
                    return;
                }
                results[static_cast<std::size_t>(i)] = round_trip(*lease, c, 1 + i) ? 1 : 0;
            };
            if (parallel) {
                threads.emplace_back(body);
            } else {
                body();
            }
        }
        for (auto& thread : threads) { thread.join(); }
        const auto end = std::chrono::steady_clock::now();
        for (const char ok : results) {
            if (ok == 0) { ++failures; }
        }
        return std::chrono::duration<double, std::milli>(end - start).count();
    };

    const double serial_ms = run_batch(false);
    const double parallel_ms = run_batch(true);
    std::printf("ok   %d round trips: serial %.1f ms, parallel %.1f ms (%.2fx)\n", connections,
                serial_ms, parallel_ms, serial_ms / parallel_ms);

    pool.close();
    std::printf("\n%d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
