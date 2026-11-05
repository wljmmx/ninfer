#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using S0 = Bf16ScheduleInstance<
    Bf16A16GemvSchedule<2, 1, 1, 16, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                        Bf16PhaseOrder::Sequential, 1, 1, 4, 2>,
    2560>;
using S1 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 5, 128, 2>, 2560>;
using S2 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<32, 24, 4, 64, 2>, 2560>;
using S3 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 32, 4, 64, 2>, 2560>;
using S4 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 32, 128, 32, 16, 2, 1>, 2560>;
using S5 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 48, 128, 32, 16, 2, 1>, 2560>;
using S6 = Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>;

// Ada fallbacks (docs/maintainer/sm89-fallback-table.md): the cp.async Xor64
// equivalents of the TMA instances above - tile geometry, stages, raster and
// outer wrappers preserved (only the producer staging changes).
using F4 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<128, 32, 128, 32, 16, 2, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast,
                       Bf16MmaSwizzle::Xor64, 1>,
                         2560>;
using F5 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<128, 48, 128, 32, 16, 2, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast,
                       Bf16MmaSwizzle::Xor64, 1>,
                         2560>;
using F6 = Bf16A16MmaSchedule<128, 128, 64, 64, 32, 3, 1, Cache::cg, Cache::cg,
                              Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast,
                              Bf16MmaSwizzle::Xor64, 1>;
} // namespace

Bf16Launch select_bf16_n248320_k2560(std::int32_t tokens) {
    if (tokens <= 1) return launch_bf16_gemv<S0>;
    if (tokens <= 8) return launch_bf16_sliced_k_mma<S1>;
    if (tokens <= 16) return launch_bf16_sliced_k_mma<S2>;
    if (tokens <= 32) return launch_bf16_sliced_k_mma<S3>;
#if defined(NINFER_ENABLE_TMA)
    if (tokens <= 64) return launch_bf16_tma_mma<S4>;
    if (tokens <= 96) return launch_bf16_tma_mma<S5>;
    return launch_bf16_tma_mma<S6>;
#else
    // Ada fallback: no TMA on sm_89; run the ladder above on the recorded cp.async
    // Xor64 equivalents of each TMA instance (docs/maintainer/sm89-fallback-table.md).
    if (tokens <= 64) return launch_bf16_mma<F4>;
    if (tokens <= 96) return launch_bf16_mma<F5>;
    return launch_bf16_mma<F6>;
#endif
}
} // namespace ninfer::ops::detail
