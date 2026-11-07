#pragma once

#include "ops/linear/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {

// Contiguous-output routes for one measured shape ladder. A shape header states the
// contraction winner per token interval once; this binder turns the chosen schedule back
// into a plain `Linear` launcher so the same ladder serves callers that write segmented
// outputs (QSA input projection) as well.
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

#if defined(NINFER_ENABLE_TMA)
    template <class S>
    static Bf16Launch tma() {
        return launch_bf16_tma_mma<S>;
    }
#endif
};

} // namespace ninfer::ops::detail
