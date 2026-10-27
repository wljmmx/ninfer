#pragma once

// "How much of this request's prefix does the tier already hold", which is the question a KV reuse
// path actually asks before it skips prefill.
//
// It answers with LMCache's semantics, not with a union: batched_async_contains walks the chunk keys
// in order and stops at the first one that is missing, so a request whose chunks are
// [hit, hit, miss, hit] reuses two chunks, not three. A union count would over-report and make the
// engine skip a prefill segment it never had.
//
// Hash values are supplied by the caller. The chaining rule belongs to the producer (vLLM hashes
// block N over block N-1 plus the token ids); this index deliberately does not guess it.

#include <cstdint>
#include <mutex>
#include <set>
#include <span>

namespace ninfer::core {

class PrefixIndex {
public:
    // Records that a chunk is stored. `complete` is false while a chunk is being written: a
    // half-written block must not count as reusable.
    void record(std::uint64_t chunk_hash, bool complete = true) {
        const std::lock_guard lock(mutex_);
        if (complete) {
            complete_.insert(chunk_hash);
        } else {
            complete_.erase(chunk_hash);
            pending_.insert(chunk_hash);
        }
    }

    void forget(std::uint64_t chunk_hash) {
        const std::lock_guard lock(mutex_);
        complete_.erase(chunk_hash);
        pending_.erase(chunk_hash);
    }

    // Number of leading chunks the tier can serve. Stops at the first hole, like LMCache.
    [[nodiscard]] std::size_t consecutive_match(std::span<const std::uint64_t> chunks) const {
        const std::lock_guard lock(mutex_);
        std::size_t count = 0;
        for (const auto hash : chunks) {
            if (complete_.count(hash) == 0) { break; }
            ++count;
        }
        return count;
    }

    [[nodiscard]] bool holds(std::uint64_t chunk_hash) const {
        const std::lock_guard lock(mutex_);
        return complete_.count(chunk_hash) != 0;
    }

    [[nodiscard]] std::size_t size() const {
        const std::lock_guard lock(mutex_);
        return complete_.size();
    }

    void clear() {
        const std::lock_guard lock(mutex_);
        complete_.clear();
        pending_.clear();
    }

private:
    mutable std::mutex mutex_;
    std::set<std::uint64_t> complete_;
    std::set<std::uint64_t> pending_;
};

} // namespace ninfer::core
