#include "ops/linear/swiglu/nvfp4/tma_launch.h"
#if NINFER_CUDA_HAS_TMA
#include "ops/linear/kernels/nvfp4/nvfp4_a4_tma.cuh"
#include "ops/linear/swiglu/token_major_mma_epilogue.cuh"

namespace ninfer::ops::detail {
void launch_nvfp4_swiglu_a4_tma(const Nvfp4A4Operands& p, __nv_bfloat16* output,
                                cudaStream_t stream) {
    using S = Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<256, 3, 1>, 5120>;
    launch_nvfp4_a4_tma_mma<S>(p, LinearBf16Output{output, p.rows / 2},
                               SwiGluTokenMajorMmaEpilogue{}, stream, SwiGluTokenMajorMmaRows<S>{});
}
} // namespace ninfer::ops::detail

#else
// port (sm_89): this implementation needs TMA, setmaxnreg and mbarrier (sm_90+). The same
// shape has a non-TMA implementation, and the dispatch sites test NINFER_CUDA_HAS_TMA, so this
// entry point exists only to keep the symbol defined for the shape's tables.
#include <stdexcept>

namespace ninfer::ops::detail {
void launch_nvfp4_swiglu_a4_tma(const Nvfp4A4Operands&, __nv_bfloat16*, cudaStream_t) {
    throw std::logic_error("launch_nvfp4_swiglu_a4_tma requires TMA (sm_90+), which this build cannot execute");
}
} // namespace ninfer::ops::detail
#endif
