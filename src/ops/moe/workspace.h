#pragma once

#include "core/arena.h"
#include "core/layout.h"
#include "ninfer/ops/moe.h"

#include <algorithm>
#include <span>

namespace ninfer::ops::detail {

// port (sm_89, upstream 7ea6cb6e): workspace capacities are measured from a 256-byte-aligned local
// origin. Every entry therefore suballocates its scratch as one aligned block of the caller's arena
// and restores the caller's cursor when it returns, instead of treating whatever offset the caller
// sits at as the origin. A caller composing several operators accounts for the leading gap itself.
// Zero bytes allocates nothing: the returned span still names the arena's device so the operand
// checks read the same answer as a real allocation.
inline MemorySpan aligned_scratch(MemoryArena& arena, std::size_t bytes) {
    return bytes ? arena.alloc_bytes(bytes) : MemorySpan{nullptr, 0, arena.device()};
}

template <class Function>
std::size_t interval_capacity(int first, int last, std::span<const int> boundaries,
                              int slice_tokens, Function&& capacity) {
    first             = std::min(first, slice_tokens);
    last              = std::min(last, slice_tokens);
    std::size_t bytes = std::max(capacity(first), capacity(last));
    for (const int boundary : boundaries) {
        for (const int tokens : {boundary - 1, boundary}) {
            if (tokens >= first && tokens <= last) { bytes = std::max(bytes, capacity(tokens)); }
        }
    }
    return bytes;
}

template <class Arena>
ExpertRoutes allocate_routes(Arena& arena, int top_k, int tokens) {
    return {arena.alloc(DType::I32, {top_k, tokens}), arena.alloc(DType::FP32, {top_k, tokens})};
}

} // namespace ninfer::ops::detail
