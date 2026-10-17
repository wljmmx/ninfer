#include "ops/linear/kernels/bf16/bf16_instances.cuh"
#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
Bf16Launch select_bf16_n256_k5120(std::int32_t tokens) {
    if (tokens <= 76)
        return launch_bf16_sliced_k_mma<
            Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 16>, 5120>>;
    if (tokens <= 160)
        return launch_bf16_sliced_k_mma<
            Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 16, 8>, 5120>>;
    if (tokens <= 640) return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR32T32K256S3, 5120>>;
    if (tokens <= 1024) return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR32T32K128S3, 5120>>;
    if (tokens <= 1280)
        #if NINFER_CUDA_HAS_TMA
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T32K64S3, 5120>>;
        #else
        // port (sm_89): TMA needs sm_90; this selector's largest non-TMA schedule
        // takes the same branch until the Ada schedule is measured.
        return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR32T32K128S3, 5120>>;
        #endif
    #if NINFER_CUDA_HAS_TMA
    return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T64K128S2, 5120>>;
    #else
    // port (sm_89): TMA needs sm_90; this selector's largest non-TMA schedule
    // takes the same branch until the Ada schedule is measured.
    return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR32T32K128S3, 5120>>;
    #endif
}
} // namespace ninfer::ops::detail
