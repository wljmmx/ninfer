#pragma once

// NVFP4 tensor core paths (kind::mxf4nvf4 MMA, TMA, setmaxnreg warp specialization
// and the E2M1 hardware codecs) require Blackwell sm_120a. This translation unit is
// compiled out on other architectures; dispatch rejects NVFP4 weights there.
#if defined(NINFER_ENABLE_NVFP4)
#include "ninfer/ops/softmax_attention.h"
#include "ops/softmax_attention/common/causal_partition.h"

namespace ninfer::ops::detail {

enum class Nvfp4KvFamily { Grouped, ParallelGrouped, Tiled };

struct Nvfp4KvCausalPlan {
    static constexpr int kTokenTile = 8;
    Nvfp4KvFamily family;
    int query_heads, width, batch, query_tile;
    CausalAttentionExecutionEnvelope envelope;
    CausalKvPartition partition;
};

Nvfp4KvCausalPlan make_nvfp4_kv_causal_plan(int heads, int width, int batch,
                                            CausalAttentionExecutionEnvelope envelope,
                                            int multiprocessor_count);
std::size_t nvfp4_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                     CausalAttentionExecutionEnvelope envelope,
                                     int multiprocessor_count);

} // namespace ninfer::ops::detail

#endif // NINFER_ENABLE_NVFP4
