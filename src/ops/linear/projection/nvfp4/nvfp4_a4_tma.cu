#include "ops/linear/projection/nvfp4/nvfp4_a4_tma_launch.h"
#if NINFER_CUDA_HAS_TMA
#include "ops/linear/kernels/nvfp4/nvfp4_a4_tma.cuh"

namespace ninfer::ops::detail {
namespace {
template <int K, CUtensorMapL2promotion Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE>
void launch(const Nvfp4A4Operands& p, __nv_bfloat16* output, cudaStream_t stream) {
    const LinearBf16Output y{output, p.rows};
    if (p.scale_layout == Nvfp4ScaleLayout::Tiled128)
        launch_nvfp4_a4_tma_mma<
            Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<128, 4, 1, Promotion>, K>>(
            p, y, LinearIdentityEpilogue{}, stream);
    else
        launch_nvfp4_a4_tma_mma<
            Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<256, 3, 1, Promotion>, K>>(
            p, y, LinearIdentityEpilogue{}, stream);
}
} // namespace

void launch_nvfp4_a4_tma_linear(Nvfp4GeometryId problem, const Nvfp4A4Operands& p,
                                __nv_bfloat16* output, cudaStream_t stream) {
    switch (problem) {
    case Nvfp4GeometryId::N14336K5120:
        launch<5120>(p, output, stream);
        return;
    case Nvfp4GeometryId::N16384K5120:
        launch<5120>(p, output, stream);
        return;
    case Nvfp4GeometryId::N34816K5120:
        launch<5120, CU_TENSOR_MAP_L2_PROMOTION_L2_128B>(p, output, stream);
        return;
    case Nvfp4GeometryId::N5120K6144:
        launch<6144>(p, output, stream);
        return;
    case Nvfp4GeometryId::N5120K17408:
        launch<17408>(p, output, stream);
        return;
    }
    throw std::invalid_argument("NVFP4 TMA linear: unsupported geometry");
}
} // namespace ninfer::ops::detail

#else
// port (sm_89): this implementation needs TMA, setmaxnreg and mbarrier (sm_90+). The same
// shape has a non-TMA implementation, and the dispatch sites test NINFER_CUDA_HAS_TMA, so this
// entry point exists only to keep the symbol defined for the shape's tables.
#include <stdexcept>

namespace ninfer::ops::detail {
void launch_nvfp4_a4_tma_linear(Nvfp4GeometryId, const Nvfp4A4Operands&, __nv_bfloat16*, cudaStream_t) {
    throw std::logic_error("launch_nvfp4_a4_tma_linear requires TMA (sm_90+), which this build cannot execute");
}
} // namespace ninfer::ops::detail
#endif
