#pragma once

// Q54: the cache's key is a whole prompt, so a request that extends a stored prefix computes a
// different key and cannot find the shorter entry. The server can only answer a lookup that names
// the tokens it already holds; the missing half is knowing *which* prefix lengths are worth asking
// about.
//
// This registry is that index. It is keyed by (frontier, digest of tokens[0..frontier)), and a
// request looks up its own prefixes from longest to shortest. The frontier is the checkpoint's real
// coverage, so an entry restored at 388 is findable by a 398-token prompt even though the two
// prompts are different keys.
//
// The digest is a two-word FNV chain, computed over the prefix only. It is an index, not a
// correctness check: a hit is still verified by the backend's own token/frontier validation, so a
// collision costs one wasted lookup, never wrong KV.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::core::lmcache {

struct PrefixDigest {
    std::uint64_t a = 0;
    std::uint64_t b = 0;

    [[nodiscard]] friend bool operator==(const PrefixDigest&, const PrefixDigest&) noexcept =
        default;
    [[nodiscard]] friend bool operator<(const PrefixDigest& left,
                                        const PrefixDigest& right) noexcept {
        return left.a != right.a ? left.a < right.a : left.b < right.b;
    }
};

[[nodiscard]] inline PrefixDigest prefix_digest_step(PrefixDigest digest, std::int64_t token) noexcept {
    const auto value = static_cast<std::uint64_t>(token);
    digest.a ^= value + 0x9e3779b97f4a7c15ULL + (digest.a << 6U) + (digest.a >> 2U);
    digest.b ^= value * 0x9e3779b97f4a7c15ULL + (digest.b << 5U) + (digest.b >> 3U);
    digest.b = (digest.b << 17U) | (digest.b >> 47U);
    return digest;
}

[[nodiscard]] inline PrefixDigest prefix_digest(std::span<const std::int64_t> tokens,
                                                std::size_t frontier) noexcept {
    PrefixDigest digest{0xcbf29ce484222325ULL, 0x84222325cbf29ce4ULL};
    const std::size_t count = std::min(frontier, tokens.size());
    for (std::size_t index = 0; index < count; ++index) {
        digest = prefix_digest_step(digest, tokens[index]);
    }
    return digest;
}

class PrefixRegistry {
public:
    // Records that the backend can serve this prefix. Called for a store and for an import: the
    // second is Q54's "the run that came from the cache is not captured" - its bytes are here, so
    // the next request in this process must be able to find them without another round trip.
    void record(std::span<const std::int64_t> tokens, std::size_t frontier) {
        if (frontier == 0 || frontier > tokens.size()) { return; }
        const auto digest = prefix_digest(tokens, frontier);
        const std::lock_guard lock(mutex_);
        frontiers_[digest] = frontier;
    }

    void forget(std::span<const std::int64_t> tokens, std::size_t frontier) {
        if (frontier == 0 || frontier > tokens.size()) { return; }
        const auto digest = prefix_digest(tokens, frontier);
        const std::lock_guard lock(mutex_);
        frontiers_.erase(digest);
    }

    // Longest recorded frontier that is a prefix of `tokens`, or nullopt. O(tokens) once, then a
    // lookup per recorded frontier on the way down - the number of recorded prefixes is small.
    [[nodiscard]] std::optional<std::size_t>
    longest_prefix(std::span<const std::int64_t> tokens) const {
        const auto found = candidates(tokens, 1);
        return found.empty() ? std::nullopt : std::optional<std::size_t>(found.front());
    }

    // Matching frontiers, longest first, at most `limit`. The caller probes in this order; the
    // first one the backend serves wins. Duplicate frontiers collapse.
    [[nodiscard]] std::vector<std::size_t> candidates(std::span<const std::int64_t> tokens,
                                                      std::size_t limit) const {
        std::vector<std::size_t> result;
        if (tokens.empty() || limit == 0) { return result; }
        std::vector<std::pair<std::size_t, PrefixDigest>> recorded;
        {
            const std::lock_guard lock(mutex_);
            recorded.reserve(frontiers_.size());
            for (const auto& [digest, frontier] : frontiers_) {
                if (frontier <= tokens.size()) { recorded.emplace_back(frontier, digest); }
            }
        }
        if (recorded.empty()) { return result; }
        std::sort(recorded.begin(), recorded.end(),
                  [](const auto& left, const auto& right) { return left.first < right.first; });
        // One pass over the longest candidate: compute the digest at each recorded frontier. The
        // records are walked in ascending frontier order because the digest itself is a prefix
        // chain; the public result is reversed to "longest first" below.
        PrefixDigest digest{0xcbf29ce484222325ULL, 0x84222325cbf29ce4ULL};
        std::size_t next = 0;
        for (std::size_t position = 1; position <= recorded.back().first; ++position) {
            digest = prefix_digest_step(digest, tokens[position - 1]);
            while (next < recorded.size() && recorded[next].first == position) {
                if (recorded[next].second == digest) {
                    if (result.empty() || result.back() != position) { result.push_back(position); }
                }
                ++next;
            }
        }
        std::reverse(result.begin(), result.end());
        if (result.size() > limit) { result.resize(limit); }
        return result;
    }

    [[nodiscard]] std::size_t size() const {
        const std::lock_guard lock(mutex_);
        return frontiers_.size();
    }

    void clear() {
        const std::lock_guard lock(mutex_);
        frontiers_.clear();
    }

private:
    mutable std::mutex mutex_;
    std::map<PrefixDigest, std::size_t> frontiers_;
};

} // namespace ninfer::core::lmcache
