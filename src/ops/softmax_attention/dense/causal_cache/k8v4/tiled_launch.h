#pragma once

// K8V4 stores its value plane as NVFP4 group16 and therefore shares the Blackwell-only
// E2M1 hardware codec (quantize pack and scaled decode). This translation unit is
// compiled out on non-Blackwell architectures; dispatch rejects K8V4 KV there.
#if defined(NINFER_ENABLE_NVFP4)
#include "ops/softmax_attention/common/causal_partition.h"
#include "ops/softmax_attention/dense/causal_cache/k8v4/operands.h"

namespace ninfer::ops::detail {
void k8v4_kv_tiled_attention(const CausalAttentionOperands&, K8V4KvReadView, CausalKvPartition,
                             WorkspaceArena&, cudaStream_t);
} // namespace ninfer::ops::detail

#endif // NINFER_ENABLE_NVFP4
