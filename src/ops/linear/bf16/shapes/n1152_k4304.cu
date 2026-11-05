#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
using S0 =
    Bf16KTailSchedule<Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 6, 128, 2>, 4304>>;
using S1 =
    Bf16KTailSchedule<Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 16, 8, 64, 2>, 4304>>;
using S2 =
    Bf16KTailSchedule<Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<32, 16, 8, 64, 2>, 4304>>;
using S3 = Bf16KTailSchedule<
    Bf16ScheduleInstance<Bf16A16MmaSchedule<32, 16, 256, 16, 8, 3, 1, Cache::cg, Cache::cg,
                                            Bf16MmaFragmentPipeline::PingPong,
                                            Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>,
                         4304>>;
using S4 = Bf16KTailSchedule<
    Bf16ScheduleInstance<Bf16A16MmaSchedule<32, 32, 256, 16, 8, 3, 1, Cache::cg, Cache::cg,
                                            Bf16MmaFragmentPipeline::PingPong,
                                            Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>,
                         4304>>;
using S5 = Bf16KTailSchedule<Bf16ScheduleInstance<
    Bf16A16TmaMmaSchedule<32, 32, 64, 16, 8, 4, 1, Bf16MmaRaster::TokenFast, 1, 1>, 4304>>;
using S6 = Bf16KTailSchedule<
    Bf16A16TmaMmaSchedule<64, 40, 64, 16, 8, 5, 1, Bf16MmaRaster::TokenFast, 1, 1>>;
using S7 = Bf16KTailSchedule<Bf16ScheduleInstance<
    Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 4, 1, Bf16MmaRaster::RowFast, 1, 1>, 4304>>;
using S8 = Bf16KTailSchedule<Bf16ScheduleInstance<
    Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 3, 1, Bf16MmaRaster::Grouped, 4, 1>, 4304>>;
using S9 = Bf16KTailSchedule<
    Bf16A16TmaMmaSchedule<128, 128, 64, 32, 32, 3, 1, Bf16MmaRaster::TokenFast, 1, 1>>;
using S10 = Bf16KTailSchedule<Bf16ScheduleInstance<
    Bf16A16TmaMmaSchedule<128, 112, 64, 32, 16, 2, 1, Bf16MmaRaster::TokenFast, 1, 1>, 4304>>;

// Ada fallbacks (docs/maintainer/sm89-fallback-table.md): the cp.async Xor64
// equivalents of the TMA instances above - tile geometry, stages, raster and
// outer wrappers preserved (only the producer staging changes).
using F5 = Bf16KTailSchedule<Bf16ScheduleInstance<
    Bf16A16MmaSchedule<32, 32, 64, 16, 8, 4, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast,
                       Bf16MmaSwizzle::Xor64, 1>,
                         4304>>;
using F6 = Bf16KTailSchedule<
    Bf16A16MmaSchedule<64, 40, 64, 16, 8, 5, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast,
                       Bf16MmaSwizzle::Xor64, 1>>;
using F9 = Bf16KTailSchedule<
    Bf16A16MmaSchedule<128, 128, 64, 32, 32, 3, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast,
                       Bf16MmaSwizzle::Xor64, 1>>;
using F10 = Bf16KTailSchedule<Bf16ScheduleInstance<
    Bf16A16MmaSchedule<128, 112, 64, 32, 16, 2, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast,
                       Bf16MmaSwizzle::Xor64, 1>,
                         4304>>;
} // namespace

Bf16Launch select_bf16_n1152_k4304(std::int32_t tokens) {
    if (tokens > 131072 || tokens % 4 != 0)
        throw std::invalid_argument("bf16 linear: P must be a multiple of 4 in [4,131072]");
    if (tokens <= 16) return launch_bf16_sliced_k_mma<S0>;
    if (tokens <= 32) return launch_bf16_sliced_k_mma<S1>;
    if (tokens <= 40) return launch_bf16_sliced_k_mma<S2>;
    if (tokens <= 64) return launch_bf16_mma<S3>;
    if (tokens <= 116) return launch_bf16_mma<S4>;
#if defined(NINFER_ENABLE_TMA)
    if (tokens <= 128) return launch_bf16_tma_mma<S5>;
    if (tokens <= 256) return launch_bf16_tma_mma<S6>;
    if (tokens <= 512) return launch_bf16_tma_mma<S7>;
    if (tokens <= 1024) return launch_bf16_tma_mma<S8>;
    if (tokens <= 2048) return launch_bf16_tma_mma<S9>;
    return launch_bf16_tma_mma<S10>;
#else
    // Ada fallback: no TMA on sm_89; run the ladder above on the recorded cp.async
    // Xor64 equivalents of each TMA instance (docs/maintainer/sm89-fallback-table.md).
    // Measured on sm_89: the 64x64 / 64x128 tiles of the (256, 1024] rungs regressed
    // (+23%/+14%), and the 128x128 F9 rung wave-quantizes at exactly 2048 tokens
    // (144 blocks -> 2 unbalanced waves), so those spans stay on this shape's own
    // K-tail mma schedule S4; the remaining rungs keep the wired fallbacks.
    if (tokens <= 128) return launch_bf16_mma<F5>;
    if (tokens <= 256) return launch_bf16_mma<F6>;
    if (tokens <= 1024) return launch_bf16_mma<S4>;
    if (tokens <= 1792) return launch_bf16_mma<F9>;
    if (tokens <= 2048) return launch_bf16_mma<S4>;
    return launch_bf16_mma<F10>;
#endif
}
} // namespace ninfer::ops::detail
