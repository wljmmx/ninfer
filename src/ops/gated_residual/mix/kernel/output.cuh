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

} // namespace ninfer::ops::detail::hc::mix
