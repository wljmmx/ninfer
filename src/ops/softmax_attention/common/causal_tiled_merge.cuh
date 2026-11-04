#pragma once

#include "core/device.h"
#include "ops/softmax_attention/common/causal_epilogue.cuh"
#include "ops/softmax_attention/common/causal_operands.h"
#include "ops/softmax_attention/common/causal_partition.h"
#include <math_constants.h>

namespace ninfer::ops::detail {

// Prefill has many output rows and at most eight KV partitions. A warp owns
// the complete D256 row, including K8V4's inverse rotation in FP32.
// MultiBatch makes one launch cover B requests: the row it derives from blockIdx is a flat
// (token, batch) column, so positions, the partial planes and out all move by the batch block
// exactly as they do in causal_merge.cuh. `flat_column` is token + width * batch.
template <class G, bool InverseRotation, bool MultiBatch>
__launch_bounds__(256) __global__
    void causal_tiled_merge_kernel(CausalPartialView partial, const std::int32_t* positions,
                                   const std::int32_t* valid_columns, int width,
                                   CausalKvPartition partition, __nv_bfloat16* out) {
    const int lane        = threadIdx.x & 31;
    const int row         = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int head        = row % G::QHeads;
    const int flat_column = row / G::QHeads;
    int batch             = 0;
    int token             = flat_column;
    if constexpr (MultiBatch) {
        batch = flat_column / width;
        token = flat_column - batch * width;
    }
    if (token >= width) return;
    if (valid_columns && token >= valid_columns[batch]) {
#pragma unroll
        for (int d = lane; d < 256; d += 32)
            causal_store_output(out + causal_q_index<G>(head, d, flat_column), 0.0F);
        return;
    }
    if constexpr (MultiBatch) {
        positions += static_cast<std::int64_t>(batch) * width;
        const std::int64_t stat_offset =
            static_cast<std::int64_t>(batch) * G::QHeads * width * partition.capacity;
        partial.acc += static_cast<std::int64_t>(batch) * G::kHeadDim * G::QHeads * width *
                       partition.capacity;
        partial.maximum += stat_offset;
        partial.sum += stat_offset;
    }
    const int splits        = partition.active(positions[width - 1] + 1);
    const auto stat         = causal_stat_index<G>(head, token, lane, width);
    const float m           = lane < splits ? partial.maximum[stat] : -CUDART_INF_F;
    const float l           = lane < splits ? partial.sum[stat] : 0.0F;
    const float maximum     = warp_max(m);
    const float weight      = l > 0.0F ? expf(m - maximum) : 0.0F;
    const float denominator = warp_sum(l * weight);
    float values[8]{};
    for (int split = 0; split < splits; ++split) {
        const float w     = __shfl_sync(0xffffffffU, weight, split);
        const auto offset = causal_partial_index<G>(head, lane, token, split, width);
#pragma unroll
        for (int i = 0; i < 8; ++i) values[i] += partial.acc[offset + i * 32] * w;
    }
    const float inverse = denominator > 0.0F ? __frcp_rn(denominator) : 0.0F;
#pragma unroll
    for (float& value : values) value *= inverse;
    if constexpr (InverseRotation) normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
    for (int i = 0; i < 8; ++i)
        causal_store_output(out + causal_q_index<G>(head, lane + i * 32, flat_column), values[i]);
}

template <class G, bool InverseRotation>
void launch_causal_tiled_merge(const CausalAttentionOperands& p, const std::int32_t* valid_columns,
                               CausalKvPartition partition, CausalPartialView partial,
                               cudaStream_t stream) {
    const int columns = p.width * p.batch;
    if (p.batch == 1) {
        causal_tiled_merge_kernel<G, InverseRotation, false>
            <<<div_up(columns * G::QHeads, 8), 256, 0, stream>>>(
                partial, p.positions, valid_columns, p.width, partition, p.out);
    } else {
        causal_tiled_merge_kernel<G, InverseRotation, true>
            <<<div_up(columns * G::QHeads, 8), 256, 0, stream>>>(
                partial, p.positions, valid_columns, p.width, partition, p.out);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
