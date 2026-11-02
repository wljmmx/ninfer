#pragma once

// port (sm_89): Q64 2d-2b-4 - the checkpoint payload layout, as arithmetic instead of two copies of
// the same reasoning.
//
// Publishing (generation_service.cpp store()) and restoring (restore()) both have to answer the same
// question - how many whole chunks the key addresses, what the real token length is, how much of the
// tail rides in the state object - and they used to answer it twice, in two places, with the same
// constants spelled differently. That is how 384 and 467 could both be "the coverage" in one build.
//
// These two functions are that arithmetic and nothing else: no bridge, no logging, no I/O. The store
// side calls plan_payload() before it publishes, the restore side calls plan_restore() before it
// commits pages, and a unit test can drive both without a server.
//
// The invariants both keep:
//   - a key addresses whole chunks only (the tail chunk is not addressable, it is carried);
//   - the tail the state object can carry is at most one chunk, page aligned;
//   - the real length never exceeds the request, and never exceeds what the payload can back.
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace ninfer::core::lmcache {

// What the store side is about to publish.
struct PayloadPlan {
    std::size_t payload_chunks = 0; // whole chunks the captured KV actually has
    std::size_t chunks         = 0; // ... of which the key addresses
    std::size_t complete       = 0; // bytes of those chunks (what the KV group carries)
    std::size_t covered        = 0; // tokens those chunks cover
    std::size_t tail_begin     = 0; // byte offset of the first byte past `complete`
    std::size_t tail_capacity  = 0; // tail bytes the state object can carry (<= one chunk)
    std::size_t tail_tokens    = 0; // tokens those tail bytes actually hold
    std::uint32_t valid_tokens = 0; // the real length, clamped to what can be backed
};

inline PayloadPlan plan_payload(std::size_t kv_bytes, std::uint32_t reported_valid,
                                std::uint32_t frontier, std::size_t prompt_tokens,
                                std::size_t chunk_bytes, std::size_t chunk_tokens,
                                std::size_t page_bytes, std::size_t page_tokens) {
    PayloadPlan plan{};
    if (chunk_bytes == 0 || chunk_tokens == 0 || page_bytes == 0 || page_tokens == 0) { return plan; }
    plan.payload_chunks = kv_bytes / chunk_bytes;
    const std::size_t addressable = prompt_tokens / chunk_tokens;
    plan.chunks                   = std::min(plan.payload_chunks, addressable);
    if (plan.chunks == 0) { return plan; }
    plan.complete   = plan.chunks * chunk_bytes;
    plan.covered    = plan.chunks * chunk_tokens;
    plan.tail_begin = plan.complete;
    const std::size_t tail_have = kv_bytes > plan.tail_begin ? kv_bytes - plan.tail_begin : 0;
    plan.tail_capacity          = std::min(tail_have, chunk_bytes);
    // Only whole pages of tail tokens are worth claiming: a reader cuts the tail by pages, so a
    // partial page would be a length it cannot honour.
    plan.tail_tokens = plan.tail_capacity / page_bytes * page_tokens;
    std::uint32_t valid = reported_valid != 0 ? reported_valid : frontier;
    plan.valid_tokens   = static_cast<std::uint32_t>(
        std::min({static_cast<std::size_t>(valid), prompt_tokens, plan.covered + plan.tail_tokens}));
    return plan;
}

// What the restore side may commit.
struct RestorePlan {
    std::size_t chunks     = 0; // whole chunks the stored KV has (0 => the payload is not usable)
    std::size_t frontier   = 0; // tokens the sequence may treat as present
    std::size_t tail_pages = 0; // pages spliced back from the state object
};

inline RestorePlan plan_restore(std::size_t kv_bytes, std::uint32_t valid_tokens,
                                std::size_t tail_bytes, std::size_t chunk_bytes,
                                std::size_t chunk_tokens, std::size_t page_bytes,
                                std::size_t page_tokens) {
    RestorePlan plan{};
    if (chunk_bytes == 0 || chunk_tokens == 0 || page_bytes == 0 || page_tokens == 0) { return plan; }
    plan.chunks   = kv_bytes / chunk_bytes;
    plan.frontier = plan.chunks * chunk_tokens;
    if (valid_tokens == 0 || valid_tokens <= plan.frontier) { return plan; }
    const auto tail_tokens = static_cast<std::size_t>(valid_tokens) - plan.frontier;
    const auto pages       = (tail_tokens + page_tokens - 1) / page_tokens;
    if (pages * page_bytes > tail_bytes) {
        // The header promises more than the object can carry. Refusing the whole payload is the only
        // honest answer: committing it would mark padding as KV.
        return RestorePlan{};
    }
    plan.tail_pages = pages;
    plan.frontier   = valid_tokens;
    return plan;
}

} // namespace ninfer::core::lmcache
