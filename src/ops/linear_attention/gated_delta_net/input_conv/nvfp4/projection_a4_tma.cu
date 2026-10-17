#include "ops/linear_attention/gated_delta_net/input_conv/nvfp4/tma_launch.h"
#if NINFER_CUDA_HAS_TMA
#include "ops/linear/kernels/nvfp4/nvfp4_a4_tma.cuh"
#include "ops/linear/common/output.cuh"

namespace ninfer::ops::detail {
void launch_nvfp4_a4_tma_gdn(const Nvfp4A4Operands& p, __nv_bfloat16* qkv, __nv_bfloat16* z,
                             cudaStream_t stream) {
    launch_nvfp4_a4_tma_mma<Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<256, 3, 1>, 5120>>(
        p, LinearBf16SegmentedOutput<10240, 6144>{qkv, z}, LinearIdentityEpilogue{}, stream);
}
} // namespace ninfer::ops::detail

#else
// port (sm_89): this implementation needs TMA, setmaxnreg and mbarrier (sm_90+). The same
// shape has a non-TMA implementation, and the dispatch sites test NINFER_CUDA_HAS_TMA, so this
// entry point exists only to keep the symbol defined for the shape's tables.
#include <stdexcept>

namespace ninfer::ops::detail {
void launch_nvfp4_a4_tma_gdn(const Nvfp4A4Operands&, __nv_bfloat16*, __nv_bfloat16*, cudaStream_t) {
    throw std::logic_error("launch_nvfp4_a4_tma_gdn requires TMA (sm_90+), which this build cannot execute");
}
} // namespace ninfer::ops::detail
#endif
