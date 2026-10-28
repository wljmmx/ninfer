// Check the bridge against LMCache with TWO different models' geometries.
//
// The point of two profiles is that nothing here may be a constant: layer count, head width, K/V
// plane count and the state dtype all differ between them, so a split that only works for one set of
// numbers fails on the other. Synthetic bytes stand in for the real exports; what is being tested is
// that a flat run of pages and one blob become the shapes the server accepts and come back intact.

#include "core/lmcache_bridge.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <span>
#include <string>
#include <vector>

namespace {

using ninfer::core::lmcache::Bridge;
using ninfer::core::lmcache::BridgeConfig;
using ninfer::core::lmcache::BridgeKey;
using ninfer::core::lmcache::ClientOptions;
using ninfer::core::lmcache::GroupSpec;

std::string make_payload(std::size_t size, unsigned char seed) {
    std::string out(size, '\0');
    for (std::size_t i = 0; i < size; ++i) { out[i] = static_cast<char>(seed + (i % 251)); }
    return out;
}

// Profile A is synthetic geometry shaped like the model on this box (separate K/V, 64-token pages,
// float32 state image). The page size is a PROFILE CONSTANT, not a measurement: on this box the live
// layout reports 1,310,720 B for bf16, but the point of the test is the byte plumbing, and a profile
// that hard-codes the engine's number would stop testing the arithmetic the moment the model changes.
BridgeConfig profile_a() {
    BridgeConfig config;
    config.model_name      = "bridge-a";
    config.page_tokens     = 64;
    config.pages_per_chunk = 2;
    config.page_bytes      = 1310720;
    config.kv.dtype        = "bfloat16";
    config.kv.shape        = {2, 40, 128, 128};
    config.kv.kv_size      = 2;
    config.kv.window_chunks = 0;
    config.state.dtype      = "float32";
    config.state.shape      = {1, 1, 16000000};
    config.state.kv_size    = 1;
    config.state.window_chunks = 1;
    config.state.recurrent  = true;
    return config;
}

// Profile B is deliberately unlike A: 28 layers, a single fused plane (MLA-style), 576 wide, and a
// bf16 state image. Every one of those is a different number, which is the point.
BridgeConfig profile_b() {
    BridgeConfig config;
    config.model_name      = "bridge-b";
    config.page_tokens     = 64;
    config.pages_per_chunk = 2;
    config.page_bytes      = 2064384;  // 28 x 64 x 576 x 2
    config.kv.dtype        = "bfloat16";
    config.kv.shape        = {28, 128, 576};
    config.kv.kv_size      = 1;
    config.kv.window_chunks = 0;
    config.state.dtype      = "bfloat16";
    config.state.shape      = {8, 1, 1048576};
    config.state.kv_size    = 1;
    config.state.window_chunks = 1;
    config.state.recurrent  = true;
    return config;
}

// Profile C: rk4v4, the symmetric rotated-int4 format. Per plane per layer per token the layout is
// {U8 data 128 B, FP16 scales 4 elements = 8 B} = 136 B, not a plain 4-bit halving. LMCache has no
// such dtype, so the object travels as an opaque uint8 view whose shape still has to describe the
// bytes. The numbers below are SYNTHETIC and do not match this box: the live rk4v4 page here is
// 348,160 B (20 layer-kv_head groups x 2 planes x 64 tokens x 136 B), which this profile does not
// reproduce. Treat a pass here as evidence about the chunking, never as evidence about the format.
BridgeConfig profile_rk4v4() {
    BridgeConfig config;
    config.model_name      = "bridge-rk4v4";
    config.page_tokens     = 64;
    config.pages_per_chunk = 2;
    config.page_bytes      = 675840;
    config.kv.dtype        = "uint8";
    config.kv.shape        = {2, 40, 128, 132};
    config.kv.kv_size      = 2;
    config.state.dtype     = "bfloat16";
    config.state.shape     = {8, 1, 1048576};
    config.state.kv_size   = 1;
    config.state.window_chunks = 1;
    config.state.recurrent = true;
    return config;
}

// Profile D: rk4v2-e8, where the K and V planes are UNEQUAL, so a two-plane shape cannot describe
// them at all and the object has to go as one plane with the planes laid end to end. Synthetic, like
// profile C: the extents are hand-written, not read from the engine's layout.
BridgeConfig profile_rk4v2() {
    BridgeConfig config;
    config.model_name      = "bridge-rk4v2";
    config.page_tokens     = 64;
    config.pages_per_chunk = 2;
    config.page_bytes      = 430080;
    config.kv.dtype        = "uint8";
    config.kv.shape        = {40, 128, 168};
    config.kv.kv_size      = 1;
    config.state.dtype     = "bfloat16";
    config.state.shape     = {8, 1, 1048576};
    config.state.kv_size   = 1;
    config.state.window_chunks = 1;
    config.state.recurrent = true;
    return config;
}

BridgeKey make_key(std::int64_t instance, std::size_t pages, std::int64_t page_tokens) {
    BridgeKey key;
    key.tokens.reserve(pages * static_cast<std::size_t>(page_tokens));
    for (std::size_t i = 0; i < pages * static_cast<std::size_t>(page_tokens); ++i) {
        key.tokens.push_back(
            static_cast<std::int64_t>((instance % 1000) * 1000 + (i * 7 + 11) % 1000));
    }
    key.request_id = "bridge-" + std::to_string(instance);
    return key;
}

// What happens when two different KV formats share a token sequence. The cache key is the tokens
// alone -- no dtype, no geometry -- so this is expected to HIT, and the hit is silent: the payload
// comes back with the other format's byte count. This is why the fingerprint has to travel in
// cache_salt; the transport itself offers no protection.
void collision_check(const ClientOptions& options, std::int64_t instance, std::size_t pages) {
    const BridgeConfig a = profile_a();
    BridgeConfig rk = profile_rk4v4();
    // The realistic case: one model, whose KV format was changed between runs. model_name is part of
    // the object key, so leaving it different would hide the very collision being tested.
    rk.model_name = a.model_name;
    Bridge writer;
    Bridge reader;
    if (!writer.open(options, 1, a, instance)) {
        std::printf("FAIL collision setup (a): %s\n", writer.error().c_str());
        return;
    }
    if (!reader.open(options, 1, rk, instance + 500)) {
        std::printf("FAIL collision setup (rk): %s\n", reader.error().c_str());
        return;
    }
    const std::string kv = make_payload(pages * a.page_bytes, 0x31);
    const std::string state = make_payload(a.state.object_bytes(), 0x41);
    const auto as_bytes = [](const std::string& text) {
        return std::span<const std::byte>(reinterpret_cast<const std::byte*>(text.data()),
                                          text.size());
    };
    const BridgeKey key = make_key(instance, pages, a.page_tokens);
    if (!writer.publish(key, as_bytes(kv), as_bytes(state))) {
        std::printf("FAIL collision publish: %s\n", writer.error().c_str());
        return;
    }

    // Same tokens, different format, different expected size.
    std::vector<std::byte> got_kv;
    std::vector<std::byte> got_state;
    const bool hit = reader.restore(key, &got_kv, &got_state);
    const std::size_t expected = pages * rk.page_bytes;
    std::printf("[collision] rk4v4 lookup on a bf16 entry: hit=%s, got %zu B, expected %zu B\n",
                hit ? "yes" : "no", got_kv.size(), expected);
    if (hit && got_kv.size() != expected) {
        std::printf("[collision] CONFIRMED: a format-blind hit returns the other format's bytes "
                    "(ratio %.2fx)\n",
                    static_cast<double>(got_kv.size()) / static_cast<double>(expected));
    } else if (!hit) {
        std::printf("[collision] no hit -- the key did include the geometry this time\n");
    }

    // Now the defence: the same lookup with the format named in cache_salt. If this still hit, the
    // salt is not doing its job and a format change would silently serve the old bytes.
    BridgeConfig salted = rk;
    salted.cache_salt = "rk4v4";
    Bridge guarded;
    if (!guarded.open(options, 1, salted, instance + 501)) {
        std::printf("FAIL collision setup (salted): %s\n", guarded.error().c_str());
        return;
    }
    std::vector<std::byte> guarded_kv;
    std::vector<std::byte> guarded_state;
    const bool guarded_hit = guarded.restore(key, &guarded_kv, &guarded_state);
    std::printf("[collision] with the format in cache_salt: hit=%s%s\n", guarded_hit ? "yes" : "no",
                guarded_hit ? " -- THE SALT IS NOT ISOLATING" : " (isolated, as intended)");
}


// Profile E: the same rk4v4 bytes, but expressed the only way that generalises -- one plane, one
// token, N elements. The engine's Host page layout is plane-major and carries four planes for a
// quantized format (K data, K scales, V data, V scales); the server can express two at most, so any
// shape naming planes is a shape that fits one format only. This states the byte count instead.
BridgeConfig profile_exact_rk4v4() {
    BridgeConfig config;
    config.model_name      = "bridge-rk4v4-exact";
    config.page_tokens     = 64;
    config.pages_per_chunk = 2;
    config.page_bytes      = 675840;
    config.kv              = GroupSpec::exact("uint8", 2 * config.page_bytes);
    config.state           = GroupSpec::exact("bfloat16", 16777216);
    config.state.window_chunks = 1;
    config.state.recurrent     = true;
    return config;
}

bool run(const ClientOptions& options, const BridgeConfig& config, std::int64_t instance,
         std::size_t pages) {
    Bridge bridge;
    if (!bridge.open(options, 1, config, instance)) {
        std::printf("FAIL [%s] open: %s\n", config.model_name.c_str(), bridge.error().c_str());
        return false;
    }
    const std::size_t kv_bytes = pages * config.page_bytes;
    const std::string kv = make_payload(kv_bytes, 0x21);
    const std::string state = make_payload(config.state.object_bytes(), 0x63);
    const auto as_bytes = [](const std::string& text) {
        return std::span<const std::byte>(reinterpret_cast<const std::byte*>(text.data()),
                                          text.size());
    };
    std::printf("[%s] %zu page(s) = %zu B KV in %zu chunk(s) + %zu B state = %.1f MB\n",
                config.model_name.c_str(), pages, kv_bytes, bridge.chunks_for(as_bytes(kv)),
                config.state.object_bytes(),
                (static_cast<double>(kv_bytes) + static_cast<double>(config.state.object_bytes())) /
                    1048576.0);

    BridgeKey key = make_key(instance, pages, config.page_tokens);

    const auto start = std::chrono::steady_clock::now();
    if (!bridge.publish(key, as_bytes(kv), as_bytes(state))) {
        std::printf("FAIL [%s] publish: %s\n", config.model_name.c_str(), bridge.error().c_str());
        return false;
    }
    const auto stored = std::chrono::steady_clock::now();
    std::vector<std::byte> back_kv;
    std::vector<std::byte> back_state;
    if (!bridge.restore(key, &back_kv, &back_state)) {
        std::printf("FAIL [%s] restore: %s\n", config.model_name.c_str(), bridge.error().c_str());
        return false;
    }
    const auto end = std::chrono::steady_clock::now();

    bool ok = true;
    if (back_kv.size() != kv.size() || std::memcmp(back_kv.data(), kv.data(), kv.size()) != 0) {
        std::printf("FAIL [%s] KV differs (%zu vs %zu B)\n", config.model_name.c_str(),
                    back_kv.size(), kv.size());
        ok = false;
    }
    if (back_state.size() != state.size() ||
        std::memcmp(back_state.data(), state.data(), state.size()) != 0) {
        std::printf("FAIL [%s] state differs (%zu vs %zu B)\n", config.model_name.c_str(),
                    back_state.size(), state.size());
        ok = false;
    }
    const auto ms = [](auto from, auto to) {
        return std::chrono::duration<double, std::milli>(to - from).count();
    };
    if (ok) {
        std::printf("ok   [%s] reassembled: publish %.1f ms, restore %.1f ms\n",
                    config.model_name.c_str(), ms(start, stored), ms(stored, end));
    }
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    ClientOptions options;
    std::int64_t instance = 9600;
    std::size_t pages = 10;

    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--server" && i + 1 < argc) {
            options.endpoint = argv[++i];
        } else if (flag == "--instance" && i + 1 < argc) {
            instance = std::strtoll(argv[++i], nullptr, 10);
        } else if (flag == "--pages" && i + 1 < argc) {
            pages = static_cast<std::size_t>(std::strtoll(argv[++i], nullptr, 10));
        }
    }

    int failures = 0;
    if (!run(options, profile_a(), instance, pages)) { ++failures; }
    if (!run(options, profile_b(), instance + 1, pages)) { ++failures; }
    if (!run(options, profile_rk4v4(), instance + 2, pages)) { ++failures; }
    if (!run(options, profile_rk4v2(), instance + 3, pages)) { ++failures; }
    if (!run(options, profile_exact_rk4v4(), instance + 4, pages)) { ++failures; }
    // Far from the run() range: a smaller offset collided with the next invocation's base instance,
    // and the collision then showed up as "same size, different bytes" in an unrelated profile.
    collision_check(options, instance + 500, pages);

    std::printf("\n%d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
