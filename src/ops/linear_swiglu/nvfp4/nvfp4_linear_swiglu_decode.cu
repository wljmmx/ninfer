// NVFP4 tensor core paths (kind::mxf4nvf4 MMA, TMA, setmaxnreg warp specialization
// and the E2M1 hardware codecs) require Blackwell sm_120a. This translation unit is
// compiled out on other architectures; dispatch rejects NVFP4 weights there.
#if defined(NINFER_ENABLE_NVFP4)
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_epilogue.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"

namespace ninfer::ops::detail {
void nvfp4_linear_swiglu_decode_launch(const Tensor& x, const Weight& w, Tensor& y,
                                       cudaStream_t stream) {
    using S = Nvfp4ScheduleInstance<
        Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 2>,
        5120>;
    launch_nvfp4_a16_gemv<S>(nvfp4_a16_operands(x, w),
                             LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n / 2},
                             Nvfp4SwiGluEpilogue{}, stream, Nvfp4SwiGluRows<1>{});
}
} // namespace ninfer::ops::detail

#endif // NINFER_ENABLE_NVFP4
