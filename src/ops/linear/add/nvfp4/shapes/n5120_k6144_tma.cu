#include "ops/linear/add/nvfp4/dispatch.h"
#if NINFER_CUDA_HAS_TMA
#include "ops/linear/kernels/nvfp4/nvfp4_a4_tma.cuh"

namespace ninfer::ops::detail {
void launch_nvfp4_add_n5120_k6144_tma(const Tensor& x, const Weight& w, Tensor& out,
                                      Nvfp4A4Workspace scratch, cudaStream_t stream) {
    using S    = Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<256, 3, 1>, 6144>;
    auto* data = static_cast<__nv_bfloat16*>(out.data);
    launch_nvfp4_a4_tma_mma<S>(nvfp4_a4_operands(w, scratch, x.ne[1], Nvfp4ScaleLayout::Tiled256),
                               LinearBf16Output{data, w.n}, LinearResidualAddEpilogue{{data, w.n}},
                               stream);
}
} // namespace ninfer::ops::detail

#else
// port (sm_89): this implementation needs TMA, setmaxnreg and mbarrier (sm_90+). The same
// shape has a non-TMA implementation, and the dispatch sites test NINFER_CUDA_HAS_TMA, so this
// entry point exists only to keep the symbol defined for the shape's tables.
#include <stdexcept>

namespace ninfer::ops::detail {
void launch_nvfp4_add_n5120_k6144_tma(const Tensor&, const Weight&, Tensor&, Nvfp4A4Workspace, cudaStream_t) {
    throw std::logic_error("launch_nvfp4_add_n5120_k6144_tma requires TMA (sm_90+), which this build cannot execute");
}
} // namespace ninfer::ops::detail
#endif
