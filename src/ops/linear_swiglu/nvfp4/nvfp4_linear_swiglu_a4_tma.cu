// NVFP4 tensor core paths (kind::mxf4nvf4 MMA, TMA, setmaxnreg warp specialization
// and the E2M1 hardware codecs) require Blackwell sm_120a. This translation unit is
// compiled out on other architectures; dispatch rejects NVFP4 weights there.
#if defined(NINFER_ENABLE_NVFP4)
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4_tma_launch.h"
#include "ops/linear/nvfp4/nvfp4_a4_tma.cuh"
#include "ops/linear_swiglu/token_major_mma_epilogue.cuh"

namespace ninfer::ops::detail {
void launch_nvfp4_linear_swiglu_a4_tma(const Nvfp4A4Operands& p, __nv_bfloat16* output,
                                       cudaStream_t stream) {
    using S = Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<256, 3, 1>, 5120>;
    launch_nvfp4_a4_tma_mma<S>(p, LinearBf16Output{output, p.rows / 2},
                               SwiGluTokenMajorMmaEpilogue{}, stream, SwiGluTokenMajorMmaRows<S>{});
}
} // namespace ninfer::ops::detail

#endif // NINFER_ENABLE_NVFP4
