#pragma once

// The third KV level (R6b) backed by the Redis instance that LMCache's RESP connector also uses.
//
// Contract note: KvTier::lookup is documented as "never blocks on IO". A Redis EXISTS is a round
// trip, so lookup answers from an in-memory directory instead; the round trip lives in prefetch(),
// which is the stage that is allowed to start work early. store() populates the directory, and
// prefetch() populates it from a hit, so a second request with the same prefix pays no lookup IO.
//
// The key comes from lmcache_key.h (the real format) and the value is the raw block, which is what
// RESPConnector._put/_get put on the wire. No header, no compression, no envelope: adding one here
// would make our blocks unreadable by LMCache and vice versa.

#include "core/kv_tier.h"
#include "core/lmcache_key.h"
#include "core/resp.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::core {

class RedisKvTier final : public KvTier {
public:
    struct Config {
        std::string host     = "127.0.0.1";
        std::uint16_t port   = 6390;
        LmcacheKeyParts identity{};
        std::uint32_t chunk_tokens  = 0;  // must match the writer's chunking
        std::size_t payload_bytes   = 0;  // fixed per (model, chunk, dtype)
        int timeout_ms              = 2000;
    };

    explicit RedisKvTier(Config config) : config_(std::move(config)) {}

    [[nodiscard]] bool connect() {
        return client_.connect_to(config_.host, config_.port, config_.timeout_ms);
    }
    [[nodiscard]] const std::string& error() const { return client_.error(); }

    [[nodiscard]] KvTierLookup lookup(const KvBlockKey& key) const override {
        if (!acceptable(key)) { return {}; }
        const std::lock_guard lock(directory_mutex_);
        if (directory_.count(key.hash) == 0) {
            ++misses_;
            return {};
        }
        ++hits_;
        return KvTierLookup{KvTierState::resident, config_.payload_bytes};
    }

    bool prefetch(const KvBlockKey& key) override {
        if (!acceptable(key)) { return false; }
        {
            const std::lock_guard lock(directory_mutex_);
            if (directory_.count(key.hash) != 0) { return true; }
        }
        const auto present = client_.exists(key_string(key));
        if (!present.has_value() || !*present) { return false; }
        remember(key.hash);
        return true;
    }

    bool retrieve(const KvBlockKey& key, std::span<std::byte> destination) override {
        if (!acceptable(key)) { return false; }
        if (destination.size() != config_.payload_bytes) {
            // A partial copy would hand the caller a truncated block that looks valid.
            return false;
        }
        const auto read = client_.get(key_string(key), destination);
        if (!read.has_value()) { return false; }
        bytes_out_ += *read;
        return true;
    }

    bool store(const KvBlockKey& key, std::span<const std::byte> source) override {
        if (!acceptable(key)) { return false; }
        if (source.size() != config_.payload_bytes) { return false; }
        const auto written = client_.set(key_string(key), source);
        if (!written.has_value()) { return false; }
        bytes_in_ += *written;
        remember(key.hash);
        return true;
    }

    // A block is only addressable when the caller's chunking and payload size are the ones this
    // tier was configured with; otherwise the same key would name bytes of a different shape.
    [[nodiscard]] bool acceptable(const KvBlockKey& key) const {
        if (config_.payload_bytes == 0) { return false; }
        if (config_.chunk_tokens != 0 && key.tokens != config_.chunk_tokens) {
            ++rejected_shape_;
            return false;
        }
        return true;
    }

    [[nodiscard]] std::string key_string(const KvBlockKey& key) const {
        LmcacheKeyParts parts = config_.identity;
        parts.chunk_hash      = key.hash;
        return lmcache_key_string(parts);
    }

    [[nodiscard]] std::uint64_t hits() const { return hits_.load(); }
    [[nodiscard]] std::uint64_t misses() const { return misses_.load(); }
    [[nodiscard]] std::uint64_t bytes_in() const { return bytes_in_.load(); }
    [[nodiscard]] std::uint64_t bytes_out() const { return bytes_out_.load(); }
    [[nodiscard]] std::uint64_t rejected_shape() const { return rejected_shape_.load(); }

private:
    void remember(std::uint64_t hash) {
        const std::lock_guard lock(directory_mutex_);
        directory_.insert(hash);
    }

    Config config_;
    mutable RespClient client_;
    mutable std::mutex directory_mutex_;
    mutable std::set<std::uint64_t> directory_;
    mutable std::atomic<std::uint64_t> hits_{0};
    mutable std::atomic<std::uint64_t> misses_{0};
    std::atomic<std::uint64_t> bytes_in_{0};
    std::atomic<std::uint64_t> bytes_out_{0};
    mutable std::atomic<std::uint64_t> rejected_shape_{0};
};

} // namespace ninfer::core
