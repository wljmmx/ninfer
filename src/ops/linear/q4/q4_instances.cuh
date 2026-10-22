#pragma once
#include "ops/linear/q4/q4_schedule.cuh"

namespace ninfer::ops::detail::q4_instances {
using GemvR4W1 =
    Q4A16GemvSchedule<4, 1, 16, 1, Q4GemvActivationAccess::Direct, Q4GemvLaneMapping::PackedWord8,
                      Q4GemvDecodeMode::Fp16Mantissa, Q4GemvCodeTransfer::AsyncVector16,
                      Q4GemvScaleAccess::SharedPair32, Cache::ca, 0, 1>;
using GemvR1W8K5120 =
    Q4A16GemvSchedule<1, 8, 16, 1, Q4GemvActivationAccess::Direct, Q4GemvLaneMapping::PackedByte2,
                      Q4GemvDecodeMode::ScalarInteger, Q4GemvCodeTransfer::SyncVector16,
                      Q4GemvScaleAccess::Scalar16Shuffle, Cache::ca, 80, 1>;

using MmaR32T64K64Wr16Wt32S2A2B2 =
    Q4A16MmaSchedule<32, 64, 64, 16, 32, 2, 2, Q4MmaFragmentPipeline::Serial, Cache::cg, Cache::cg,
                     Q4ScaleLoad::Pair32>;
using GemvR1W8K6144 =
    Q4A16GemvSchedule<1, 8, 16, 1, Q4GemvActivationAccess::Direct, Q4GemvLaneMapping::PackedByte2,
                      Q4GemvDecodeMode::ScalarInteger, Q4GemvCodeTransfer::SyncVector16,
                      Q4GemvScaleAccess::Scalar16Shuffle, Cache::ca, 6144 / 64, 1>;
using MmaR32T32K64Wr16Wt16S3A3B2 =
    Q4A16MmaSchedule<32, 32, 64, 16, 16, 3, 2, Q4MmaFragmentPipeline::Serial, Cache::cg, Cache::cg,
                     Q4ScaleLoad::Pair32>;
using MmaR32T64K64Wr16Wt32S3A3B2 =
    Q4A16MmaSchedule<32, 64, 64, 16, 32, 3, 2, Q4MmaFragmentPipeline::Serial, Cache::cg, Cache::cg,
                     Q4ScaleLoad::Pair32>;

// Every 64-token tile runs the 128-wide K tile: the barrier count per unit of K halves,
// which is worth 40% once the weights stream from HBM and 5% while they stay in L2. The
// instance lives with the other K128 schedules below (MmaR64T64K128S2A1).

using SlicedK5120T16 = Q4A16SlicedKMmaSchedule<16, 16, 8, 1, Cache::cg, Cache::ca, 6, 5120, 16>;
using SlicedK5120T24 = Q4A16SlicedKMmaSchedule<16, 24, 8, 1, Cache::cg, Cache::ca, 6, 5120, 24>;
using SlicedK2048T4  = Q4A16SlicedKMmaSchedule<16, 8, 8, 1, Cache::cg, Cache::ca, 6, 2048, 4>;

using SlicedK5120T4 = Q4A16SlicedKMmaSchedule<16, 8, 8, 1, Cache::cg, Cache::ca, 6, 5120, 4>;

// The 48-token tile is narrow enough that folding the group scale into the tile's MMA result,
// rather than into every weight, is a measured win (-5.5% warm L2, level cold); wider tiles keep
// the decode fold because their accumulator would cost more than the decode there.
using MmaR64T48K64Wr16Wt16S2A2B2 =
    Q4A16MmaSchedule<64, 48, 64, 16, 16, 2, 2, Q4MmaFragmentPipeline::Serial, Cache::cg, Cache::cg,
                     Q4ScaleLoad::Pair32, 2, Q4ScaleApply::PostMma>;

using MmaR64T72K64Wr32Wt24S2A2B2 =
    Q4A16MmaSchedule<64, 72, 64, 32, 24, 2, 2, Q4MmaFragmentPipeline::Serial, Cache::cg, Cache::cg,
                     Q4ScaleLoad::Pair32>;
using MmaR64T80K64Wr16Wt40S2A2B1 =
    Q4A16MmaSchedule<64, 80, 64, 16, 40, 2, 1, Q4MmaFragmentPipeline::Serial, Cache::cg, Cache::cg,
                     Q4ScaleLoad::Pair32>;
// 96 tokens is the widest tile that still fits the 48 KiB static budget with a 128-wide K
// tile, and that only at 32 rows: -13% once the weights stream, -5% while they sit in L2.
using MmaR32T96K128Wr16Wt16S2A1B2 =
    Q4A16MmaSchedule<32, 96, 128, 16, 16, 2, 2, Q4MmaFragmentPipeline::Serial, Cache::cg,
                     Cache::cg, Q4ScaleLoad::Pair32, 1>;
using MmaR64T112K64Wr32Wt16S2A2B1 =
    Q4A16MmaSchedule<64, 112, 64, 32, 16, 2, 1, Q4MmaFragmentPipeline::Serial, Cache::cg, Cache::cg,
                     Q4ScaleLoad::Pair32>;
using MmaR64T112K64Wr16Wt112S2A2B1 =
    Q4A16MmaSchedule<64, 112, 64, 16, 112, 2, 1, Q4MmaFragmentPipeline::Serial, Cache::cg,
                     Cache::cg, Q4ScaleLoad::Pair32>;
using MmaR64T120K64Wr32Wt24S2A2B1 =
    Q4A16MmaSchedule<64, 120, 64, 32, 24, 2, 1, Q4MmaFragmentPipeline::Serial, Cache::cg, Cache::cg,
                     Q4ScaleLoad::Pair32>;
using MmaR64T120K64Wr16Wt120S2A2B1 =
    Q4A16MmaSchedule<64, 120, 64, 16, 120, 2, 1, Q4MmaFragmentPipeline::Serial, Cache::cg,
                     Cache::cg, Q4ScaleLoad::Pair32>;
// A 128-wide K tile does not fit beside a 128-token activation stage, so this tile splits
// the block into eight warps instead of four: two more warps per SM schedule the same work,
// worth -7% streaming and -3% in L2.
using MmaR64T128K64Wr32Wt32S2A2B1 =
    Q4A16MmaSchedule<64, 128, 64, 32, 32, 2, 1, Q4MmaFragmentPipeline::Serial, Cache::cg, Cache::cg,
                     Q4ScaleLoad::Pair32>;

using SlicedR16T8W4S2      = Q4A16SlicedKMmaSchedule<16, 8, 4, 2, Cache::cg, Cache::ca, 2>;
using SlicedR16T8Capacity4 = Q4A16SlicedKMmaSchedule<16, 8, 4, 2, Cache::cg, Cache::ca, 2, 0, 4>;
using SlicedR16T16W4S2     = Q4A16SlicedKMmaSchedule<16, 16, 4, 2, Cache::cg, Cache::ca, 2>;
using SlicedR16T16W2S2     = Q4A16SlicedKMmaSchedule<16, 16, 2, 2, Cache::cg, Cache::ca, 2>;
using SlicedR16T32W4S2     = Q4A16SlicedKMmaSchedule<16, 32, 4, 2, Cache::cg, Cache::ca, 2>;
using SlicedR32T8W4S2      = Q4A16SlicedKMmaSchedule<32, 8, 4, 2, Cache::cg, Cache::ca, 2>;
using SlicedR32T16W4S2     = Q4A16SlicedKMmaSchedule<32, 16, 4, 2, Cache::cg, Cache::ca, 2>;
using SlicedR32T16W4S1     = Q4A16SlicedKMmaSchedule<32, 16, 4, 1, Cache::cg, Cache::ca, 2>;
using SlicedR32T32W4S1     = Q4A16SlicedKMmaSchedule<32, 32, 4, 1, Cache::cg, Cache::ca, 2>;
using SlicedR32T32W4S2     = Q4A16SlicedKMmaSchedule<32, 32, 4, 2, Cache::cg, Cache::ca, 2>;
using SlicedR32T32W2S2     = Q4A16SlicedKMmaSchedule<32, 32, 2, 2, Cache::cg, Cache::ca, 2>;
using SlicedR32T64W2S1     = Q4A16SlicedKMmaSchedule<32, 64, 2, 1, Cache::cg, Cache::ca, 2>;
using MmaR32T32K128S2A2 = Q4A16MmaSchedule<32, 32, 128, 16, 16, 2, 2, Q4MmaFragmentPipeline::Serial,
                                           Cache::cg, Cache::cg, Q4ScaleLoad::Pair32, 2>;
using MmaR64T64K128S2A1 = Q4A16MmaSchedule<64, 64, 128, 16, 32, 2, 2, Q4MmaFragmentPipeline::Serial,
                                           Cache::cg, Cache::cg, Q4ScaleLoad::Pair32, 1>;
using MmaR32T128K64S2A2 = Q4A16MmaSchedule<32, 128, 64, 32, 32, 2, 2, Q4MmaFragmentPipeline::Serial,
                                           Cache::cg, Cache::cg, Q4ScaleLoad::Pair32, 2>;
using SimtR4T4W2G8S2    = Q4A16SimtSchedule<4, 4, 2, 8, 2, Cache::ca, 1>;
using SimtR4T1W2G8S2    = Q4A16SimtSchedule<4, 1, 2, 8, 2, Cache::ca, 1>;
} // namespace ninfer::ops::detail::q4_instances
