#pragma once

#include "ops/softmax_attention/dense/causal_cache/int8/schedule.cuh"

namespace ninfer::ops::detail {

template <class G, int Tokens>
struct Int8KvGroupedInstance {
    // One tile may hold at most 64 query rows, i.e. 64 / GroupSize tokens: a GQA-6 or GQA-8
    // pairing fits the eight-token tile unchanged, while Qwen4Exp's indexed layers (24 query heads
    // over 2 KV heads, GQA-12) fit only five. Deriving the tile here instead of lowering the shared
    // plan constant keeps every existing instance on exactly the tiling it was qualified with.
    static constexpr int kMaxTokens = 64 / G::GroupSize;
    static constexpr int kTokens    = Tokens < kMaxTokens ? Tokens : kMaxTokens;
    static_assert(kTokens > 0 && kTokens * G::GroupSize <= 64);
    static_assert(Tokens * G::GroupSize > 64 || Tokens == kTokens);
    static constexpr int kRowTiles  = (kTokens * G::GroupSize + 15) / 16;
    static_assert(kRowTiles >= 1 && kRowTiles <= 4);
    static constexpr bool kWideKeys = G::GroupSize == 8 && kRowTiles >= 3;
    using Schedule                  = Int8KvGroupedMmaSchedule<kTokens,
                                              kWideKeys        ? 4 * kRowTiles
                                                               : kRowTiles <= 2 ? 8
                                                                                : 2 * kRowTiles,
                                              kWideKeys ? 64 : 32, kWideKeys ? 1 : 2>;
    using Merge                     = Int8KvMergeSchedule<G::QHeads == 24 ? 256 : 64>;
};

using Int8KvTiledInstance = Int8KvTiledMmaSchedule<64, 64, 128>;

} // namespace ninfer::ops::detail
