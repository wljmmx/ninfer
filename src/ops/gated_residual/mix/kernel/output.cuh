#pragma once

#include "ops/gated_residual/mix/operands.h"
#include "ops/gated_residual/mix/kernel/math.cuh"

namespace ninfer::ops::detail::hc::mix {
template <Kind Form>
struct DownOutput {
    __nv_bfloat16* activation;
    float* injection;

    __device__ __forceinline__ void store(int row, int token, float dot) const {
        if constexpr (Form == Kind::Block) {
            if (row >= kRank) {
                injection[std::int64_t(token) * kStreams + row - kRank] =
                    2.F * sigmoid(dot * 0.25F);
                return;
            }
        }
        const float a                                 = dot * 0.25F;
        activation[std::int64_t(token) * kRank + row] = __float2bfloat16_rn(a * sigmoid_gate(a));
    }
};

// The separate-weight form launches the low-rank rows and the injection rows as two matrices, so
// each has its own single-purpose store: the fused row branch above never fires there.
struct RankActivationOutput {
    __nv_bfloat16* activation;

    __device__ __forceinline__ void store(int row, int token, float dot) const {
        const float a                                 = dot * 0.25F;
        activation[std::int64_t(token) * kRank + row] = __float2bfloat16_rn(a * sigmoid_gate(a));
    }
};

struct InjectionOutput {
    float* injection;

    __device__ __forceinline__ void store(int row, int token, float dot) const {
        injection[std::int64_t(token) * kStreams + row] = 2.F * sigmoid(dot * 0.25F);
    }
};

} // namespace ninfer::ops::detail::hc::mix
