#pragma once

// K8V4 stores its value plane as NVFP4 group16 and therefore shares the Blackwell-only
// E2M1 hardware codec (quantize pack and scaled decode). This translation unit is
// compiled out on non-Blackwell architectures; dispatch rejects K8V4 KV there.
#if defined(NINFER_ENABLE_NVFP4)
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

#endif // NINFER_ENABLE_NVFP4
