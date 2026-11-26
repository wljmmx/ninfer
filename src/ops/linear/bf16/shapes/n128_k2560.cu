#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_launch.cuh"
#include "ops/linear/bf16/bf16_shapes.h"

namespace ninfer::ops::detail {

// QSA's indexer key projection: the hidden width (2560) onto the 128-row index space. The ladder is
// n256_k2560's (same K); 128 rows is a decode-sized projection, so the tile choice is not what limits
// the layer.
Bf16Launch select_bf16_n128_k2560(std::int32_t tokens) {
    if (tokens <= 76)
        return launch_bf16_sliced_k_mma<
            Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 8>, 2560>>;
    if (tokens <= 160)
        return launch_bf16_sliced_k_mma<
            Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 16, 8>, 2560>>;
    if (tokens <= 640) return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR32T32K256S3, 2560>>;
    if (tokens <= 1024) return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR32T32K128S3, 2560>>;
#if defined(NINFER_ENABLE_TMA)
    if (tokens <= 1280) return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T32K64S3, 2560>>;
    return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T64K128S2, 2560>>;
#else
    // Ada fallback: no TMA on sm_89; run the cp.async companions of the TMA instances.
    if (tokens <= 1280) return launch_bf16_mma<Bf16ScheduleInstance<Bf16AdaR64T32K64S3, 2560>>;
    return launch_bf16_mma<Bf16ScheduleInstance<Bf16AdaR64T64K128S2, 2560>>;
#endif
}

} // namespace ninfer::ops::detail
