#pragma once

// K8V4 stores its value plane as NVFP4 group16; the codec runs through official CUDA 13
// conversion intrinsics on every architecture, so this path is available on Ada (sm_89)
// as well as Blackwell.
#if defined(NINFER_ENABLE_K8V4)
#include "ops/softmax_attention/common/causal_partition.h"
#include "ops/softmax_attention/dense/causal_cache/k8v4/operands.h"

namespace ninfer::ops::detail {
void k8v4_kv_tiled_attention(const CausalAttentionOperands&, K8V4KvReadView, CausalKvPartition,
                             WorkspaceArena&, cudaStream_t);
} // namespace ninfer::ops::detail

#endif // NINFER_ENABLE_K8V4
