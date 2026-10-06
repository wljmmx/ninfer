#pragma once

// K8V4 stores its value plane as NVFP4 group16 and therefore shares the Blackwell-only
// E2M1 hardware codec (quantize pack and scaled decode). This translation unit is
// compiled out on non-Blackwell architectures; dispatch rejects K8V4 KV there.
#if defined(NINFER_ENABLE_NVFP4)
#include "ninfer/ops/softmax_attention.h"
#include "ops/softmax_attention/common/causal_partition.h"

namespace ninfer::ops::detail {

enum class K8V4KvFamily { Grouped, ParallelGrouped, Tiled };

struct K8V4KvCausalPlan {
    static constexpr int kTokenTile = 8;
    K8V4KvFamily family;
    int query_heads, width, batch, query_tile;
    CausalAttentionExecutionEnvelope envelope;
    CausalKvPartition partition;
};

K8V4KvCausalPlan make_k8v4_kv_causal_plan(int heads, int width, int batch,
                                          CausalAttentionExecutionEnvelope envelope,
                                          int multiprocessor_count);
std::size_t k8v4_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                    CausalAttentionExecutionEnvelope envelope,
                                    int multiprocessor_count);

} // namespace ninfer::ops::detail

#endif // NINFER_ENABLE_NVFP4
