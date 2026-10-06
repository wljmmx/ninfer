#pragma once

// NVFP4 tensor core paths (kind::mxf4nvf4 MMA, TMA, setmaxnreg warp specialization
// and the E2M1 hardware codecs) require Blackwell sm_120a. This translation unit is
// compiled out on other architectures; dispatch rejects NVFP4 weights there.
#if defined(NINFER_ENABLE_NVFP4)
#include "ops/softmax_attention/dense/causal_cache/nvfp4/schedule.cuh"

namespace ninfer::ops::detail {
template <class G, int Tokens>
struct Nvfp4KvGroupedInstance {
    static_assert(Tokens > 0 && Tokens * G::GroupSize <= 64);
    static constexpr int kRowTiles = (Tokens * G::GroupSize + 15) / 16;
    using Schedule =
        Nvfp4KvGroupedMmaSchedule<Tokens, kRowTiles <= 2 ? 8 : 4 * kRowTiles,
                                  kRowTiles <= 2 ? 32 : 64, kRowTiles <= 2 ? 2 : 1, true>;
    using Merge = Nvfp4KvMergeSchedule;
};

using Nvfp4KvTiledInstance = Nvfp4KvTiledMmaSchedule<64, 12>;
} // namespace ninfer::ops::detail

#endif // NINFER_ENABLE_NVFP4
