#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
using S0 = Bf16ScheduleInstance<
    Bf16A16GemvSchedule<1, 1, 1, 16, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                        Bf16PhaseOrder::Sequential, 1, 1, 4, 2>,
    4608>;
using S1 = Bf16KTailSchedule<Bf16ScheduleInstance<
    Bf16A16SimtSchedule<8, 1, 2, 8, 4, 2, Bf16SimtActivationAccess::DirectStream,
                        Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 4, 1, 2, 2>,
    4608>>;
using S2 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 4, 64, 2>, 4608>;
using S3 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 4, 128, 2>, 4608>;
using S4 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<32, 8, 4, 64, 2>, 4608>;
using S5 = Bf16A16MmaSchedule<32, 16, 256, 16, 8, 3, 1, Cache::cg, Cache::cg,
                              Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
using S6 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 128, 16, 16, 3, 1>, 4608>;
using S7 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<96, 32, 128, 16, 16, 3, 1>, 4608>;
using S8 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 128, 32, 16, 3, 1>, 4608>;
using S9 = Bf16ScheduleInstance<
    Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 4, 1, Bf16MmaRaster::Grouped, 4>, 4608>;
using S10 = Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>;
using S11 = Bf16ScheduleInstance<
    Bf16A16TmaMmaSchedule<96, 64, 64, 48, 32, 2, 2, Bf16MmaRaster::TokenFast, 1, 9>, 4608>;

// Ada fallbacks (docs/maintainer/sm89-fallback-table.md): the cp.async Xor64
// equivalents of the TMA instances above - tile geometry, stages, raster and
// outer wrappers preserved (only the producer staging changes).
using F6 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<64, 32, 128, 16, 16, 3, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast,
                       Bf16MmaSwizzle::Xor64, 1>,
                         4608>;
using F7 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<96, 32, 128, 16, 16, 3, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast,
                       Bf16MmaSwizzle::Xor64, 1>,
                         4608>;
using F8 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<64, 64, 128, 32, 16, 3, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast,
                       Bf16MmaSwizzle::Xor64, 1>,
                         4608>;
using F9 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<128, 64, 64, 32, 32, 4, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::Grouped,
                       Bf16MmaSwizzle::Xor64, 4>,
                         4608>;
using F10 = Bf16A16MmaSchedule<128, 128, 64, 64, 32, 3, 1, Cache::cg, Cache::cg,
                               Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast,
                               Bf16MmaSwizzle::Xor64, 1>;
using F11 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<96, 64, 64, 48, 32, 2, 2, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast,
                       Bf16MmaSwizzle::Xor64, 1>,
                         4608>;
} // namespace

Bf16Launch select_bf16_n4608_k4608(std::int32_t tokens) {
    if (tokens > 32768) throw std::invalid_argument("bf16 linear: V must be in [1,32768]");
    if (tokens <= 1) return launch_bf16_gemv<S0>;
    if (tokens <= 2) return launch_bf16_simt<S1>;
    if (tokens <= 3) return launch_bf16_sliced_k_mma<S2>;
    if (tokens <= 4) return launch_bf16_sliced_k_mma<S3>;
    if (tokens <= 16) return launch_bf16_sliced_k_mma<S4>;
    if (tokens <= 32) return launch_bf16_mma<S5>;
#if defined(NINFER_ENABLE_TMA)
    if (tokens <= 64) return launch_bf16_tma_mma<S6>;
    if (tokens <= 96) return launch_bf16_tma_mma<S7>;
    if (tokens <= 128) return launch_bf16_tma_mma<S8>;
    if (tokens <= 256) return launch_bf16_tma_mma<S9>;
    if (tokens <= 512) return launch_bf16_tma_mma<S10>;
    return launch_bf16_tma_mma<S11>;
#else
    // Ada fallback: no TMA on sm_89; run the ladder above on the recorded cp.async
    // Xor64 equivalents of each TMA instance (docs/maintainer/sm89-fallback-table.md).
    if (tokens <= 64) return launch_bf16_mma<F6>;
    if (tokens <= 96) return launch_bf16_mma<F7>;
    if (tokens <= 128) return launch_bf16_mma<F8>;
    if (tokens <= 256) return launch_bf16_mma<F9>;
    if (tokens <= 512) return launch_bf16_mma<F10>;
    return launch_bf16_mma<F11>;
#endif
}
} // namespace ninfer::ops::detail
