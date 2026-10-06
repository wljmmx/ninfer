#pragma once

// K8V4 stores its value plane as NVFP4 group16 and therefore shares the Blackwell-only
// E2M1 hardware codec (quantize pack and scaled decode). This translation unit is
// compiled out on non-Blackwell architectures; dispatch rejects K8V4 KV there.
#if defined(NINFER_ENABLE_NVFP4)
#include "ops/softmax_attention/dense/causal_cache/k8v4/schedule.cuh"

namespace ninfer::ops::detail {

template <class G, int Tokens>
struct K8V4KvGroupedInstance {
    static_assert(Tokens > 0 && Tokens * G::GroupSize <= 64);
    static constexpr int kRowTiles = (Tokens * G::GroupSize + 15) / 16;
    using Schedule                 = K8V4KvGroupedMmaSchedule<Tokens,
                                              kRowTiles == 4   ? 16
                                                              : kRowTiles == 3 ? 12
                                                                               : 8,
                                              Tokens == 1 ? 32 : 64, Tokens == 1 ? 2 : 1>;
    using Merge                    = K8V4KvMergeSchedule;
};

using K8V4KvTiledInstance = K8V4KvTiledMmaSchedule<kMxfp8TiledQueryRows, 64>;

} // namespace ninfer::ops::detail

#endif // NINFER_ENABLE_NVFP4
