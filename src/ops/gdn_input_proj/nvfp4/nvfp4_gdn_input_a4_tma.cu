// NVFP4 tensor core paths (kind::mxf4nvf4 MMA, TMA, setmaxnreg warp specialization
// and the E2M1 hardware codecs) require Blackwell sm_120a. This translation unit is
// compiled out on other architectures; dispatch rejects NVFP4 weights there.
#if defined(NINFER_ENABLE_NVFP4)
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_a4_tma_launch.h"
#include "ops/linear/nvfp4/nvfp4_a4_tma.cuh"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"

namespace ninfer::ops::detail {
void launch_nvfp4_a4_tma_gdn(const Nvfp4A4Operands& p, __nv_bfloat16* qkv, __nv_bfloat16* z,
                             cudaStream_t stream) {
    launch_nvfp4_a4_tma_mma<Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<256, 3, 1>, 5120>>(
        p, Nvfp4GdnInputOutput{qkv, z}, LinearIdentityEpilogue{}, stream);
}
} // namespace ninfer::ops::detail

#endif // NINFER_ENABLE_NVFP4
