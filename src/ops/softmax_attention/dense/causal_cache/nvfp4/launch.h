#pragma once

// NVFP4 tensor core paths (kind::mxf4nvf4 MMA, TMA, setmaxnreg warp specialization
// and the E2M1 hardware codecs) require Blackwell sm_120a. This translation unit is
// compiled out on other architectures; dispatch rejects NVFP4 weights there.
#if defined(NINFER_ENABLE_NVFP4)
#include "ninfer/ops/softmax_attention.h"

namespace ninfer::ops::detail {

void nvfp4_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                               const Tensor& positions, const Tensor& valid, const Tensor& rows,
                               float scale, PagedKVBatchLayerView cache,
                               CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                               Tensor& out, DeviceExecutionView execution);

void nvfp4_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                               const PagedKVLayerView& cache,
                               CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                               Tensor& out, DeviceExecutionView execution);

} // namespace ninfer::ops::detail

#endif // NINFER_ENABLE_NVFP4
