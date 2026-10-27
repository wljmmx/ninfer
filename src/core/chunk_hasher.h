#pragma once

// Chunk hashing for prefix reuse, plus the "how much can I skip" helper.
//
// Two honest constraints, stated up front:
//
//  1. This hash is OURS, not vLLM's. vLLM derives a block hash with sha256_cbor over
//     (parent hash, token ids, extras) and LMCache keys are built from it. A cache written by a
//     vLLM+LMCache deployment therefore will NOT be hit by this function. What it does support is
//     the case we can measure today: our own engine writing and re-reading its own prefixes, which
//     is exactly what the P1 DoD ("repeat prefix TTFT -30%") measures. Swapping in the vLLM hash
//     later means replacing chunk_hash() and nothing else.
//
//  2. The chain matters. Block N's hash includes block N-1's hash, so two requests that share a
//     prefix but differ afterwards still agree on the shared blocks, and a request whose block 2
//     differs cannot collide with another request's block 2 that continues differently.
//
// Chunking is fixed-token (LMCache's `chunk_size`), not tokenizer-dependent: tokens, not bytes.

#include "core/prefix_index.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::core {

// FNV-1a over the token ids, seeded with the parent hash, then a final avalanche step so that a
// one-token difference in the tail changes the whole word instead of a few bits.
[[nodiscard]] inline std::uint64_t chunk_hash(std::uint64_t parent,
                                              std::span<const std::uint32_t> tokens) noexcept {
    std::uint64_t hash = parent ^ 1469598103934665603ULL;
    for (const auto token : tokens) {
        hash ^= static_cast<std::uint64_t>(token);
        hash *= 1099511628211ULL;
    }
    hash ^= hash >> 33;
    hash *= 0xff51afd7ed558ccdULL;
    hash ^= hash >> 33;
    return hash;
}

// Hashes a token sequence into chunk hashes. A partial tail is not emitted: the next request's tail
// will not match it, so storing it would only create keys that never hit.
[[nodiscard]] inline std::vector<std::uint64_t> chunk_hashes(std::span<const std::uint32_t> tokens,
                                                             std::uint32_t chunk_tokens) {
    std::vector<std::uint64_t> hashes;
    if (chunk_tokens == 0) { return hashes; }
    hashes.reserve(tokens.size() / chunk_tokens);
    std::uint64_t parent = 0;
    for (std::size_t begin = 0; begin + chunk_tokens <= tokens.size(); begin += chunk_tokens) {
        parent = chunk_hash(parent, tokens.subspan(begin, chunk_tokens));
        hashes.push_back(parent);
    }
    return hashes;
}

// How many leading tokens of this prompt the tier can serve, i.e. how much prefill to skip.
[[nodiscard]] inline std::size_t reusable_prefix_tokens(std::span<const std::uint32_t> prompt,
                                                        std::uint32_t chunk_tokens,
                                                        const PrefixIndex& index) {
    if (chunk_tokens == 0) { return 0; }
    return index.consecutive_match(chunk_hashes(prompt, chunk_tokens)) * chunk_tokens;
}

} // namespace ninfer::core
