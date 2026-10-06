#pragma once

// K8V4 stores its value plane as NVFP4 group16; the codec runs through official CUDA 13
// conversion intrinsics on every architecture, so this path is available on Ada (sm_89)
// as well as Blackwell.
#if defined(NINFER_ENABLE_K8V4)
#include "ninfer/ops/softmax_attention.h"

namespace ninfer::ops::detail {

void k8v4_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& positions, const Tensor& valid, const Tensor& rows,
                              float scale, PagedKVBatchLayerView cache,
                              CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                              Tensor& out, DeviceExecutionView execution);

void k8v4_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                              const PagedKVLayerView& cache,
                              CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                              Tensor& out, DeviceExecutionView execution);

} // namespace ninfer::ops::detail

#endif // NINFER_ENABLE_K8V4
