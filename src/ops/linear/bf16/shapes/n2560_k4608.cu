#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
using S0 = Bf16ScheduleInstance<
    Bf16A16GemvSchedule<8, 1, 1, 16, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Streaming,
                        Bf16PhaseOrder::Sequential, 1, 1, 4, 2>,
    4608>;
using S1 = Bf16ScheduleInstance<
    Bf16A16SimtSchedule<1, 1, 1, 8, 4, 2, Bf16SimtActivationAccess::WarpPacked,
                        Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 4, 1, 2, 2>,
    4608>;
using S2 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 4, 128, 3>, 4608>;
using S3 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<64, 8, 4, 64, 2>, 4608>;
using S4 = Bf16RowTailSchedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<48, 32, 192, 16, 16, 3, 1>, 4608>>;
using S5 = Bf16ScheduleInstance<
    Bf16A16TmaMmaSchedule<64, 32, 128, 16, 16, 3, 1, Bf16MmaRaster::TokenFast, 1, 18>, 4608>;
using S6 = Bf16RowTailSchedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<48, 64, 128, 16, 16, 3, 1>, 4608>>;
using S7 =
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 4, 1, Bf16MmaRaster::Grouped, 4>,
                         4608>;
using S8 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 3, 1>, 4608>;
using S9 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>, 4608>;

// Ada fallbacks (docs/maintainer/sm89-fallback-table.md): the cp.async Xor64
// equivalents of the TMA instances above - tile geometry, stages, raster and
// outer wrappers preserved (only the producer staging changes).
using F4 = Bf16RowTailSchedule<Bf16ScheduleInstance<
    Bf16A16MmaSchedule<48, 32, 192, 16, 16, 3, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast,
                       Bf16MmaSwizzle::Xor64, 1>,
                         4608>>;
using F5 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<64, 32, 128, 16, 16, 3, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast,
                       Bf16MmaSwizzle::Xor64, 1>,
                         4608>;
using F6 = Bf16RowTailSchedule<Bf16ScheduleInstance<
    Bf16A16MmaSchedule<48, 64, 128, 16, 16, 3, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast,
                       Bf16MmaSwizzle::Xor64, 1>,
                         4608>>;
using F7 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<64, 64, 64, 32, 16, 4, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::Grouped,
                       Bf16MmaSwizzle::Xor64, 4>,
                         4608>;
using F8 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<128, 64, 64, 32, 32, 3, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast,
                       Bf16MmaSwizzle::Xor64, 1>,
                         4608>;
using F9 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<128, 128, 64, 64, 32, 3, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast,
                       Bf16MmaSwizzle::Xor64, 1>,
                         4608>;
} // namespace

Bf16Launch select_bf16_n2560_k4608(std::int32_t tokens) {
    if (tokens > 32768) throw std::invalid_argument("bf16 linear: V must be in [1,32768]");
    if (tokens <= 1) return launch_bf16_gemv<S0>;
    if (tokens <= 2) return launch_bf16_simt<S1>;
    if (tokens <= 8) return launch_bf16_sliced_k_mma<S2>;
    if (tokens <= 24) return launch_bf16_sliced_k_mma<S3>;
#if defined(NINFER_ENABLE_TMA)
    if (tokens <= 64) return launch_bf16_tma_mma<S4>;
    if (tokens <= 96) return launch_bf16_tma_mma<S5>;
    if (tokens <= 128) return launch_bf16_tma_mma<S6>;
    if (tokens <= 256) return launch_bf16_tma_mma<S7>;
    if (tokens <= 512) return launch_bf16_tma_mma<S8>;
    return launch_bf16_tma_mma<S9>;
#else
    // Ada fallback: no TMA on sm_89; run the ladder above on the recorded cp.async
    // Xor64 equivalents of each TMA instance (docs/maintainer/sm89-fallback-table.md).
    if (tokens <= 64) return launch_bf16_mma<F4>;
    if (tokens <= 96) return launch_bf16_mma<F5>;
    if (tokens <= 128) return launch_bf16_mma<F6>;
    if (tokens <= 256) return launch_bf16_mma<F7>;
    if (tokens <= 512) return launch_bf16_mma<F8>;
    return launch_bf16_mma<F9>;
#endif
}
} // namespace ninfer::ops::detail
