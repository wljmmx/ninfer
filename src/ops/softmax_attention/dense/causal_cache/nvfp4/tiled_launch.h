#pragma once

// NVFP4 tensor core paths (kind::mxf4nvf4 MMA, TMA, setmaxnreg warp specialization
// and the E2M1 hardware codecs) require Blackwell sm_120a. This translation unit is
// compiled out on other architectures; dispatch rejects NVFP4 weights there.
#if defined(NINFER_ENABLE_NVFP4)
#include "ops/softmax_attention/dense/causal_cache/nvfp4/operands.h"

namespace ninfer::ops::detail {
void nvfp4_kv_tiled_attention(const CausalAttentionOperands&, Nvfp4KvReadView, cudaStream_t);
} // namespace ninfer::ops::detail

#endif // NINFER_ENABLE_NVFP4
