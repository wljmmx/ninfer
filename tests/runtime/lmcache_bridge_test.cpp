// Check the bridge's split and reassembly against LMCache, using the real geometry (page 1,310,720 B
// and a 64 MB image) with synthetic bytes.
//
// The engine's half is already covered elsewhere; what is new here is that a flat run of pages and a
// single blob turn into the object shapes the server accepts, and come back as the same flat run.

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

std::string make_payload(std::size_t size, unsigned char seed) {
    std::string out(size, '\0');
    for (std::size_t i = 0; i < size; ++i) { out[i] = static_cast<char>(seed + (i % 251)); }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    ClientOptions options;
    std::int64_t instance = 9500;
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

    BridgeConfig config;
    config.model_name        = "bridge-test";
    config.chunk_tokens      = 128;
    config.page_bytes        = 1310720;   // the measured page stride
    config.state_image_bytes = 64000000;  // the measured image size, rounded to 4 bytes

    Bridge bridge;
    if (!bridge.open(options, 1, config, instance)) {
        std::printf("FAIL open: %s\n", bridge.error().c_str());
        return 1;
    }
    const std::size_t kv_bytes = pages * config.page_bytes;
    const std::string kv = make_payload(kv_bytes, 0x21);
    const std::string state = make_payload(config.state_image_bytes, 0x63);

    const auto as_bytes = [](const std::string& text) {
        return std::span<const std::byte>(reinterpret_cast<const std::byte*>(text.data()),
                                          text.size());
    };

    std::printf("geometry: %zu page(s) = %zu B KV in %zu chunk(s), + %zu B state = %.1f MB\n", pages,
                kv_bytes, bridge.chunks_for(as_bytes(kv)), config.state_image_bytes,
                (static_cast<double>(kv_bytes) + static_cast<double>(config.state_image_bytes)) /
                    1048576.0);

    BridgeKey key;
    key.tokens.reserve(pages * 64);
    for (std::size_t i = 0; i < pages * 64; ++i) {
        key.tokens.push_back(static_cast<std::int64_t>((instance % 1000) * 1000 + (i * 7 + 11) % 1000));
    }
    key.request_id = "bridge-" + std::to_string(instance);

    const auto start = std::chrono::steady_clock::now();
    if (!bridge.publish(key, as_bytes(kv), as_bytes(state))) {
        std::printf("FAIL publish: %s\n", bridge.error().c_str());
        return 1;
    }
    const auto stored = std::chrono::steady_clock::now();
    std::vector<std::byte> back_kv;
    std::vector<std::byte> back_state;
    if (!bridge.restore(key, &back_kv, &back_state)) {
        std::printf("FAIL restore: %s\n", bridge.error().c_str());
        return 1;
    }
    const auto end = std::chrono::steady_clock::now();

    int failures = 0;
    const auto ms = [](auto from, auto to) {
        return std::chrono::duration<double, std::milli>(to - from).count();
    };
    if (back_kv.size() != kv.size()) {
        std::printf("FAIL KV size %zu vs %zu\n", back_kv.size(), kv.size());
        ++failures;
    } else if (std::memcmp(back_kv.data(), kv.data(), kv.size()) != 0) {
        std::printf("FAIL KV bytes differ\n");
        ++failures;
    }
    if (back_state.size() != state.size()) {
        std::printf("FAIL state size %zu vs %zu\n", back_state.size(), state.size());
        ++failures;
    } else if (std::memcmp(back_state.data(), state.data(), state.size()) != 0) {
        std::printf("FAIL state bytes differ\n");
        ++failures;
    }

    std::printf("ok   %zu page(s) reassembled: publish %.1f ms, restore %.1f ms\n", pages,
                ms(start, stored), ms(stored, end));
    std::printf("\n%d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
