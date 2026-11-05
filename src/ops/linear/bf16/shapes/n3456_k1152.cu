#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
using S0 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 6, 64, 2>, 1152>;
using S1 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<32, 16, 128, 16, 8, 2, 2, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    1152>;
using S2 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 64, 16, 16, 3, 2>, 1152>;
using S3 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 3, 2>, 1152>;
using S4 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<96, 64, 64, 48, 32, 2, 2>, 1152>;
using S5 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<96, 128, 64, 32, 32, 3, 1>, 1152>;
using S6 = Bf16ScheduleInstance<
    Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2, 2, Bf16MmaRaster::TokenFast, 1, 6>, 1152>;

// Ada fallbacks (docs/maintainer/sm89-fallback-table.md): the cp.async Xor64
// equivalents of the TMA instances above - tile geometry, stages, raster and
// outer wrappers preserved (only the producer staging changes).
using F2 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<64, 32, 64, 16, 16, 3, 2, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast,
                       Bf16MmaSwizzle::Xor64, 1>,
                         1152>;
using F3 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<64, 64, 64, 32, 16, 3, 2, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast,
                       Bf16MmaSwizzle::Xor64, 1>,
                         1152>;
using F4 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<96, 64, 64, 48, 32, 2, 2, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast,
                       Bf16MmaSwizzle::Xor64, 1>,
                         1152>;
using F5 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<96, 128, 64, 32, 32, 3, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast,
                       Bf16MmaSwizzle::Xor64, 1>,
                         1152>;
using F6 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<64, 128, 64, 32, 32, 2, 2, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast,
                       Bf16MmaSwizzle::Xor64, 1>,
                         1152>;
} // namespace

Bf16Launch select_bf16_n3456_k1152(std::int32_t tokens) {
    if (tokens > 131072 || tokens % 4 != 0)
        throw std::invalid_argument("bf16 linear: P must be a multiple of 4 in [4,131072]");
    if (tokens <= 4) return launch_bf16_sliced_k_mma<S0>;
    if (tokens <= 16) return launch_bf16_mma<S1>;
#if defined(NINFER_ENABLE_TMA)
    if (tokens <= 96) return launch_bf16_tma_mma<S2>;
    if (tokens <= 128) return launch_bf16_tma_mma<S3>;
    if (tokens <= 256) return launch_bf16_tma_mma<S4>;
    if (tokens <= 512) return launch_bf16_tma_mma<S5>;
    if (tokens <= 1024) return launch_bf16_tma_mma<S6>;
    return launch_bf16_tma_mma<S4>;
#else
    // Ada fallback: no TMA on sm_89; run the ladder above on the recorded cp.async
    // Xor64 equivalents of each TMA instance (docs/maintainer/sm89-fallback-table.md).
    if (tokens <= 96) return launch_bf16_mma<F2>;
    if (tokens <= 128) return launch_bf16_mma<F3>;
    if (tokens <= 256) return launch_bf16_mma<F4>;
    if (tokens <= 512) return launch_bf16_mma<F5>;
    if (tokens <= 1024) return launch_bf16_mma<F6>;
    return launch_bf16_mma<F4>;
#endif
}
} // namespace ninfer::ops::detail
