#pragma once
#include "ops/linear/bf16/bf16_schedule.cuh"

namespace ninfer::ops::detail {

using Bf16A16MmaR32T32K256S3 =
    Bf16A16MmaSchedule<32, 32, 256, 16, 8, 3, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
using Bf16A16MmaR32T32K192S2 =
    Bf16A16MmaSchedule<32, 32, 192, 16, 8, 2, 2, Cache::cg, Cache::ca,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;

using Bf16A16MmaR32T32K128S3 =
    Bf16A16MmaSchedule<32, 32, 128, 16, 16, 3, 2, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
using Bf16A16MmaR64T32K64S3 =
    Bf16A16MmaSchedule<64, 32, 64, 32, 16, 3, 2, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
using Bf16A16SlicedR32T16W4  = Bf16A16SlicedKMmaSchedule<32, 16, 4>;
using Bf16A16TmaR64T32K64S3  = Bf16A16TmaMmaSchedule<64, 32, 64, 32, 16, 3>;
using Bf16A16TmaR64T64K64S3  = Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 3>;
using Bf16A16TmaR64T64K128S2 = Bf16A16TmaMmaSchedule<64, 64, 128, 32, 32, 2>;
using Bf16A16TmaR64T128K64S2 = Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2>;

// sm_89 companions of the TMA instances above (same row/tile geometry, stages and raster;
// only the producer staging changes, Tma128 swizzle -> cp.async Xor64). The TMA aliases
// select large-token spans inside #if NINFER_ENABLE_TMA; these serve the same spans in the
// #else branches of n14336_k5120.cu / n256_k5120.cu / n5120_k6144.cu.
using Bf16AdaR64T32K64S3  = Bf16A16MmaSchedule<64, 32, 64, 32, 16, 3, 1, Cache::cg,
                                                Cache::cg, Bf16MmaFragmentPipeline::PingPong,
                                                Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64>;
using Bf16AdaR64T64K64S3  = Bf16A16MmaSchedule<64, 64, 64, 32, 32, 3, 1, Cache::cg,
                                                Cache::cg, Bf16MmaFragmentPipeline::PingPong,
                                                Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64>;
using Bf16AdaR64T64K128S2 = Bf16A16MmaSchedule<64, 64, 128, 32, 32, 2, 1, Cache::cg,
                                                Cache::cg, Bf16MmaFragmentPipeline::PingPong,
                                                Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64>;
using Bf16AdaR64T128K64S2 = Bf16A16MmaSchedule<64, 128, 64, 32, 32, 2, 1, Cache::cg,
                                                Cache::cg, Bf16MmaFragmentPipeline::PingPong,
                                                Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64>;

} // namespace ninfer::ops::detail
