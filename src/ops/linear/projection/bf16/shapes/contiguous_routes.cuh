#pragma once

#include "ops/linear/projection/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {

struct Bf16ContiguousRoutes {
    template <class S>
    static Bf16Launch gemv() {
        return launch_bf16_gemv<S>;
    }

    template <class S>
    static Bf16Launch sliced() {
        return launch_bf16_sliced_k_mma<S>;
    }

    template <class S>
    static Bf16Launch mma() {
        return launch_bf16_mma<S>;
    }

    template <class S>
    static Bf16Launch tma() {
        #if NINFER_CUDA_HAS_TMA
        return launch_bf16_tma_mma<S>;
        #else
        // port (sm_89): TMA needs sm_90; this selector's largest non-TMA schedule
        // takes the same branch until the Ada schedule is measured.
        return launch_bf16_mma<S>;
        #endif
    }
};

} // namespace ninfer::ops::detail
