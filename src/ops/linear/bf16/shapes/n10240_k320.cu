#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using S0 = Bf16KTailSchedule<Bf16ScheduleInstance<
    Bf16A16GemvSchedule<2, 1, 1, 16, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                        Bf16PhaseOrder::Sequential, 1, 1, 4, 2>,
    320>>;
using S1 = Bf16A16MmaSchedule<32, 8, 64, 16, 8, 3, 2, Cache::cg, Cache::cg,
                              Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
using S2 = Bf16A16MmaSchedule<64, 16, 64, 16, 8, 3, 2, Cache::cg, Cache::cg,
                              Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
using S3 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 64, 16, 16, 3, 2>, 320>;
using S4 = Bf16A16MmaSchedule<64, 64, 64, 16, 16, 3, 1, Cache::cg, Cache::cg,
                              Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
using S5 =
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 2, 3, Bf16MmaRaster::Grouped, 8>,
                         320>;

// Ada fallbacks (docs/maintainer/sm89-fallback-table.md): the cp.async Xor64
// equivalents of the TMA instances above - tile geometry, stages, raster and
// outer wrappers preserved (only the producer staging changes).
using F3 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<64, 32, 64, 16, 16, 3, 2, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast,
                       Bf16MmaSwizzle::Xor64, 1>,
                         320>;
using F5 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<64, 64, 64, 32, 16, 2, 3, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::Grouped,
                       Bf16MmaSwizzle::Xor64, 8>,
                         320>;
} // namespace

Bf16Launch select_bf16_n10240_k320(std::int32_t tokens) {
    if (tokens <= 1) return launch_bf16_gemv<S0>;
    if (tokens <= 16) return launch_bf16_mma<S1>;
    if (tokens <= 48) return launch_bf16_mma<S2>;
#if defined(NINFER_ENABLE_TMA)
    if (tokens <= 64) return launch_bf16_tma_mma<S3>;
    if (tokens <= 128) return launch_bf16_mma<S4>;
    return launch_bf16_tma_mma<S5>;
#else
    // Ada fallback: no TMA on sm_89; run the ladder above on the recorded cp.async
    // Xor64 equivalents of each TMA instance (docs/maintainer/sm89-fallback-table.md).
    if (tokens <= 64) return launch_bf16_mma<F3>;
    if (tokens <= 128) return launch_bf16_mma<S4>;
    return launch_bf16_mma<F5>;
#endif
}
} // namespace ninfer::ops::detail
