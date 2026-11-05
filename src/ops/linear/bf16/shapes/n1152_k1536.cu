#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
using S0 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 4, 64, 2>, 1536>;
using S1 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<32, 8, 4, 64, 2>, 1536>;
using S2 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<32, 8, 3, 128, 2>, 1536>;
using S3 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<32, 16, 256, 16, 8, 3, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    1536>;
using S4 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<32, 32, 256, 16, 16, 3, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    1536>;
using S5 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 64, 16, 16, 3, 2>, 1536>;
using S6 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 3, 2>, 1536>;
using S7 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 3, 1>, 1536>;
using S8 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>, 1536>;

// Ada fallbacks (docs/maintainer/sm89-fallback-table.md): the cp.async Xor64
// equivalents of the TMA instances above - tile geometry, stages, raster and
// outer wrappers preserved (only the producer staging changes).
using F5 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<64, 32, 64, 16, 16, 3, 2, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast,
                       Bf16MmaSwizzle::Xor64, 1>,
                         1536>;
using F6 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<64, 64, 64, 32, 16, 3, 2, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast,
                       Bf16MmaSwizzle::Xor64, 1>,
                         1536>;
using F7 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<64, 128, 64, 32, 32, 3, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast,
                       Bf16MmaSwizzle::Xor64, 1>,
                         1536>;
using F8 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<128, 128, 64, 64, 32, 3, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast,
                       Bf16MmaSwizzle::Xor64, 1>,
                         1536>;
} // namespace

Bf16Launch select_bf16_n1152_k1536(std::int32_t tokens) {
    if (tokens > 131072 || tokens % 4 != 0)
        throw std::invalid_argument("bf16 linear: P must be a multiple of 4 in [4,131072]");
    if (tokens <= 8) return launch_bf16_sliced_k_mma<S0>;
    if (tokens <= 16) return launch_bf16_sliced_k_mma<S1>;
    if (tokens <= 32) return launch_bf16_sliced_k_mma<S2>;
    if (tokens <= 64) return launch_bf16_mma<S3>;
    if (tokens <= 128) return launch_bf16_mma<S4>;
#if defined(NINFER_ENABLE_TMA)
    if (tokens <= 256) return launch_bf16_tma_mma<S5>;
    if (tokens <= 512) return launch_bf16_tma_mma<S6>;
    if (tokens <= 1024) return launch_bf16_tma_mma<S7>;
    if (tokens <= 2048) return launch_bf16_tma_mma<S8>;
    return launch_bf16_tma_mma<S6>;
#else
    // Ada fallback: no TMA on sm_89; run the ladder above on the recorded cp.async
    // Xor64 equivalents of each TMA instance (docs/maintainer/sm89-fallback-table.md).
    if (tokens <= 256) return launch_bf16_mma<F5>;
    if (tokens <= 512) return launch_bf16_mma<F6>;
    if (tokens <= 1024) return launch_bf16_mma<F7>;
    if (tokens <= 2048) return launch_bf16_mma<F8>;
    return launch_bf16_mma<F6>;
#endif
}
} // namespace ninfer::ops::detail
