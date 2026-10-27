#pragma once

// A KV tier the engine can hand blocks to, shaped after LMCache's protocol: a lookup that answers
// "do you have this prefix", a prefetch that starts the fetch early, a retrieve that waits, and a
// store that writes back. The three stages are separate on purpose - a hit that cannot be fetched
// without stalling the step is worse than a miss that was never asked for.
//
// Nothing here is wired into the model yet. This is the seam the KV work (plan section 44, F2/F3)
// and the prefetcher (D1) both attach to; it is header-only so it costs the build nothing until a
// backend and a caller exist.

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <map>
#include <vector>

namespace ninfer::core {

// A prefix is identified by a rolling hash over its token ids plus its length, so two requests that
// share a prefix name the same block. The length is part of the key: a hash collision between
// different lengths would otherwise hand back a block of the wrong size.
struct KvBlockKey {
    std::uint64_t hash   = 0;
    std::uint32_t tokens = 0;

    friend bool operator==(const KvBlockKey&, const KvBlockKey&) = default;
    friend bool operator<(const KvBlockKey& a, const KvBlockKey& b) noexcept {
        return a.hash < b.hash || (a.hash == b.hash && a.tokens < b.tokens);
    }
};

[[nodiscard]] inline std::uint64_t kv_hash_tokens(std::span<const TokenId> tokens,
                                                  std::uint64_t seed = 1469598103934665603ULL) noexcept {
    std::uint64_t hash = seed;
    for (const auto token : tokens) {
        hash ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(token));
        hash *= 1099511628211ULL;
        hash ^= hash >> 29;
    }
    return hash;
}

[[nodiscard]] inline KvBlockKey kv_block_key(std::span<const TokenId> prefix) noexcept {
    return KvBlockKey{kv_hash_tokens(prefix), static_cast<std::uint32_t>(prefix.size())};
}

enum class KvTierState { miss, fetching, resident };

// What a backend reports for a prefix. `bytes` is the size of the payload a retrieve would fill.
struct KvTierLookup {
    KvTierState state = KvTierState::miss;
    std::size_t bytes = 0;
};

class KvTier {
public:
    virtual ~KvTier() = default;

    // Never blocks on IO: answers from the directory of what this tier holds or is fetching.
    [[nodiscard]] virtual KvTierLookup lookup(const KvBlockKey& key) const = 0;
    // Starts the fetch if this tier holds the block and it is not resident yet. Returns false when
    // the tier does not hold it at all.
    virtual bool prefetch(const KvBlockKey& key) = 0;
    // Waits for the payload and copies it out. Returns false when the tier cannot serve it.
    virtual bool retrieve(const KvBlockKey& key, std::span<std::byte> destination) = 0;
    // Writes a payload back so a later request can hit it.
    virtual bool store(const KvBlockKey& key, std::span<const std::byte> source) = 0;
};

// A local-disk tier: one file per block, named by the hash and the token count, with the directory
// kept in memory. This is the third level of plan section 44 (R6b) in its simplest useful form -
// enough to make the lookup/prefetch/retrieve split real and measurable before any remote tier.
class DiskKvTier final : public KvTier {
public:
    explicit DiskKvTier(std::filesystem::path directory) : directory_(std::move(directory)) {
        std::error_code ec;
        std::filesystem::create_directories(directory_, ec);
        std::lock_guard lock(mutex_);
        if (!std::filesystem::exists(directory_, ec)) { return; }
        for (const auto& entry : std::filesystem::directory_iterator(directory_, ec)) {
            if (!entry.is_regular_file(ec)) { continue; }
            const auto name = entry.path().filename().string();
            const auto dot  = name.find('.');
            if (dot == std::string::npos) { continue; }
            const auto hash   = std::strtoull(name.c_str(), nullptr, 16);
            const auto tokens = static_cast<std::uint32_t>(std::strtoul(name.c_str() + dot + 1, nullptr, 10));
            index_[KvBlockKey{hash, tokens}] = entry.path();
        }
    }

    [[nodiscard]] KvTierLookup lookup(const KvBlockKey& key) const override {
        std::lock_guard lock(mutex_);
        const auto it = index_.find(key);
        if (it == index_.end()) { return {}; }
        std::error_code ec;
        const auto bytes = std::filesystem::file_size(it->second, ec);
        if (ec) { return {}; }
        return KvTierLookup{KvTierState::resident, static_cast<std::size_t>(bytes)};
    }

    bool prefetch(const KvBlockKey& key) override {
        // A local file needs no staging: the first read is the prefetch, and it lands in the page
        // cache the retrieve then hits. Kept as a distinct call so a remote tier can do real work.
        return lookup(key).state == KvTierState::resident;
    }

    bool retrieve(const KvBlockKey& key, std::span<std::byte> destination) override {
        std::filesystem::path path;
        {
            std::lock_guard lock(mutex_);
            const auto it = index_.find(key);
            if (it == index_.end()) { return false; }
            path = it->second;
        }
        std::FILE* file = std::fopen(path.string().c_str(), "rb");
        if (file == nullptr) { return false; }
        const auto read = std::fread(destination.data(), 1, destination.size(), file);
        std::fclose(file);
        return read == destination.size();
    }

    bool store(const KvBlockKey& key, std::span<const std::byte> source) override {
        const auto name = block_name(key);
        const auto path = directory_ / name;
        std::FILE* file = std::fopen(path.string().c_str(), "wb");
        if (file == nullptr) { return false; }
        const auto written = std::fwrite(source.data(), 1, source.size(), file);
        std::fclose(file);
        if (written != source.size()) { return false; }
        std::lock_guard lock(mutex_);
        index_[key] = path;
        return true;
    }

private:
    static std::string block_name(const KvBlockKey& key) {
        char buffer[32];
        std::snprintf(buffer, sizeof buffer, "%016llx.%u", static_cast<unsigned long long>(key.hash),
                      key.tokens);
        return buffer;
    }

    std::filesystem::path directory_;
    mutable std::mutex mutex_;
    std::map<KvBlockKey, std::filesystem::path> index_;
};

} // namespace ninfer::core

