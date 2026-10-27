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

// Profile A is the model on this box: 40 layers, separate K/V, 128 wide, 64-token pages, and a
// float32 state image of 64 M B.
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

    BridgeKey key;
    key.tokens.reserve(pages * static_cast<std::size_t>(config.page_tokens));
    for (std::size_t i = 0; i < pages * static_cast<std::size_t>(config.page_tokens); ++i) {
        key.tokens.push_back(static_cast<std::int64_t>((instance % 1000) * 1000 + (i * 7 + 11) % 1000));
    }
    key.request_id = "bridge-" + std::to_string(instance);

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

    std::printf("\n%d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
