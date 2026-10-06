#pragma once

// NVFP4 tensor core paths (kind::mxf4nvf4 MMA, TMA, setmaxnreg warp specialization
// and the E2M1 hardware codecs) require Blackwell sm_120a. This translation unit is
// compiled out on other architectures; dispatch rejects NVFP4 weights there.
#if defined(NINFER_ENABLE_NVFP4)
#include "ops/linear/nvfp4/nvfp4_operands.h"

namespace ninfer::ops::detail {
void launch_nvfp4_a4_tma_linear_add(const Nvfp4A4Operands& p, __nv_bfloat16* residual,
                                    cudaStream_t stream);
} // namespace ninfer::ops::detail

#endif // NINFER_ENABLE_NVFP4
