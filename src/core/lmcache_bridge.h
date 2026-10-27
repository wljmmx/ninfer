// The seam between a checkpoint export and LMCache.
//
// The engine exports KV as a flat run of pages (64 tokens each) and the state image as one blob; the
// cache wants chunks of `chunk_tokens` and one object per group. This translates between the two and
// owns nothing else -- deliberately: no engine types appear in the interface, so the splitting and
// reassembly can be tested without a model.
//
// Two shapes are fixed by what the server validates, not by taste (v6 1n):
//   * KV: one tensor per chunk, [2, layers, chunk_tokens, hidden] -- the leading 2 is the K/V plane
//     count. Registering layers=1 keeps the byte count exact when the model's layer count does not
//     matter to the transport.
//   * state: ONE tensor addressed to the trailing chunk, declared [1, 1, bytes/4] with a one-chunk
//     window. The image has no token dimension, so any other decomposition is padding.
#pragma once

#include "core/lmcache_client.h"
#include "core/lmcache_wire.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace ninfer::core::lmcache {

struct BridgeConfig {
    std::string model_name;
    // Tokens per cache chunk. The engine page is 64 tokens, so 128 is two pages; the value only has
    // to be a whole multiple of the page or the split would cut a page in half.
    std::int64_t chunk_tokens{128};
    // Per-page and per-image byte counts, taken from the live layout at startup.
    std::size_t page_bytes{0};
    std::size_t state_image_bytes{0};
};

// The cache identity: the token sequence alone. Reusing a sequence hits an entry an earlier run
// stored, and a shape mismatch there reads as a wrong byte count rather than an error, so callers
// must not reuse one either.
struct BridgeKey {
    std::vector<std::int64_t> tokens;
    std::string request_id;
};

class Bridge {
public:
    Bridge() = default;
    Bridge(const Bridge&) = delete;
    Bridge& operator=(const Bridge&) = delete;

    [[nodiscard]] const std::string& error() const { return error_; }
    [[nodiscard]] std::int64_t instance_id() const { return instance_id_; }
    [[nodiscard]] const BridgeConfig& config() const { return config_; }

    bool open(const ClientOptions& options, int connections, const BridgeConfig& config,
              std::int64_t instance_id) {
        config_      = config;
        instance_id_ = instance_id;
        if (config_.page_bytes == 0 || config_.state_image_bytes == 0) {
            error_ = "geometry is not up yet (page_bytes or state_image_bytes is zero)";
            return false;
        }
        if (config_.chunk_tokens <= 0 || config_.chunk_tokens % kPageTokens != 0) {
            error_ = "chunk_tokens must be a whole number of 64-token pages";
            return false;
        }
        // The registrations decompose a byte count into a shape; a remainder means the transport
        // would reject every payload, so refuse here where the reason is visible.
        if (config_.page_bytes % (2 * static_cast<std::size_t>(config_.chunk_tokens)) != 0) {
            error_ = "page_bytes does not decompose into (2, 1, chunk_tokens, hidden)";
            return false;
        }
        if (config_.state_image_bytes % 4 != 0) {
            error_ = "state image is not a whole number of float32 elements";
            return false;
        }
        if (!pool_.open(options, static_cast<std::size_t>(connections))) {
            error_ = pool_.error();
            return false;
        }
        if (!pool_.register_context(registration())) {
            error_ = pool_.error();
            return false;
        }
        return true;
    }

    void close() { pool_.close(); }

    // One turn: every KV chunk, then the state image at the trailing chunk.
    [[nodiscard]] std::size_t chunks_for(std::span<const std::byte> kv) const {
        const std::size_t pages = kv.size() / config_.page_bytes;
        const std::size_t per_chunk = static_cast<std::size_t>(config_.chunk_tokens / kPageTokens);
        return per_chunk == 0 ? 0 : pages / per_chunk;
    }

    bool publish(const BridgeKey& key, std::span<const std::byte> kv, std::span<const std::byte> state) {
        if (!pool_.size() || instance_id_ == 0) {
            error_ = "bridge is not open";
            return false;
        }
        const std::size_t per_chunk_pages =
            static_cast<std::size_t>(config_.chunk_tokens / kPageTokens);
        if (kv.empty() || kv.size() % config_.page_bytes != 0 ||
            kv.size() / config_.page_bytes % per_chunk_pages != 0) {
            error_ = "KV bytes are not a whole number of chunks";
            return false;
        }
        if (state.size() != config_.state_image_bytes) {
            error_ = "state image size does not match the registered layout";
            return false;
        }

        const std::size_t chunk_bytes = per_chunk_pages * config_.page_bytes;
        std::vector<std::string> chunks;
        chunks.reserve(kv.size() / chunk_bytes);
        for (std::size_t offset = 0; offset < kv.size(); offset += chunk_bytes) {
            chunks.emplace_back(reinterpret_cast<const char*>(kv.data() + offset), chunk_bytes);
        }
        const std::string payload(reinterpret_cast<const char*>(state.data()), state.size());

        auto lease = pool_.acquire();
        if (!lease) {
            error_ = "no connection available";
            return false;
        }
        const bool ok = lease->store(make_key(key), chunks, kv_shape(), payload, state_shape());
        if (!ok) { error_ = lease->error(); }
        return ok;
    }

    bool has(const BridgeKey& key) {
        auto lease = pool_.acquire();
        if (!lease) {
            error_ = "no connection available";
            return false;
        }
        const bool ok = lease->has(make_key(key));
        if (!ok) { error_ = lease->error(); }
        return ok;
    }

    // Rebuilds the flat page run the engine's import expects.
    bool restore(const BridgeKey& key, std::vector<std::byte>* kv, std::vector<std::byte>* state) {
        auto lease = pool_.acquire();
        if (!lease) {
            error_ = "no connection available";
            return false;
        }
        RestorePayload got;
        const CacheKey cache_key = make_key(key);
        if (!lease->has(cache_key) || !lease->fetch(cache_key, &got)) {
            error_ = lease->error().empty() ? "not cached" : lease->error();
            return false;
        }
        if (kv != nullptr) {
            kv->clear();
            for (const std::string& chunk : got.kv_chunks) {
                const auto* bytes = reinterpret_cast<const std::byte*>(chunk.data());
                kv->insert(kv->end(), bytes, bytes + chunk.size());
            }
        }
        if (state != nullptr) {
            state->assign(reinterpret_cast<const std::byte*>(got.state.data()),
                          reinterpret_cast<const std::byte*>(got.state.data()) + got.state.size());
        }
        return true;
    }

private:
    static constexpr std::int64_t kPageTokens = 64;

    [[nodiscard]] RegisterContext registration() const {
        RegisterContext context;
        context.instance_id        = instance_id_;
        context.model_name         = config_.model_name;
        context.world_size         = 1;
        context.block_size         = config_.chunk_tokens;
        context.num_layers         = 1;
        context.hidden_dim_size    = kv_hidden();
        context.dtype_str          = "bfloat16";
        context.use_mla            = false;
        context.num_physical_slots = config_.chunk_tokens;

        GroupLayout kv;
        kv.num_layers       = 1;
        kv.hidden_dim_size  = kv_hidden();
        kv.dtype_str        = "bfloat16";
        kv.tokens_per_block = config_.chunk_tokens;
        kv.kv_size          = 2;
        kv.window_chunks    = 0;  // full attention: every chunk from the first
        context.group_layouts.push_back(kv);

        GroupLayout state;
        state.num_layers       = 1;
        state.hidden_dim_size  = static_cast<std::int64_t>(config_.state_image_bytes / 4);
        state.dtype_str        = "float32";
        state.tokens_per_block = config_.chunk_tokens;
        state.kv_size          = 1;
        state.window_tokens    = 1;  // the image is one object, not per-token data
        state.window_chunks    = 1;  // and it answers for the trailing chunk only
        state.recurrent        = true;
        context.group_layouts.push_back(state);
        return context;
    }

    // 2 (K/V planes) x 1 layer x chunk_tokens x hidden x 2 bytes = one chunk.
    [[nodiscard]] std::int64_t kv_hidden() const {
        return static_cast<std::int64_t>(config_.page_bytes /
                                         (2 * static_cast<std::size_t>(config_.chunk_tokens)));
    }

    [[nodiscard]] std::vector<std::int64_t> kv_shape() const {
        return {2, 1, config_.chunk_tokens, kv_hidden()};
    }

    [[nodiscard]] std::vector<std::int64_t> state_shape() const {
        return {1, 1, static_cast<std::int64_t>(config_.state_image_bytes / 4)};
    }

    [[nodiscard]] CacheKey make_key(const BridgeKey& key) const {
        CacheKey out;
        out.model_name  = config_.model_name;
        out.world_size  = 1;
        out.worker_id   = 0;
        out.token_bytes = pack_tokens(key.tokens);
        out.start       = 0;
        out.end         = static_cast<std::int64_t>(key.tokens.size());
        out.request_id  = key.request_id;
        return out;
    }

    Pool pool_;
    BridgeConfig config_{};
    std::int64_t instance_id_{0};
    std::string error_;
};

}  // namespace ninfer::core::lmcache
