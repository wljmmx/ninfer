#pragma once

// Implements: include/ninfer/ops/logits_topk.h
// Match: contiguous BF16 [physical_rows,C], I32 [C,k], FP32 [C,k] and FP32 [C].
// Algorithm assumptions: one 256-thread CTA per column. Rank 0 also evaluates the column
// normalizer with the same two-pass evaluation target_logprobs uses; ranks 1..k-1 are k-1 further
// scans that each take the largest packed (value,id) key strictly below the previous one.

#include "ops/common/cuda/reduce.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kLogitsTopkBlock = 256;

template <int BlockSize>
__launch_bounds__(BlockSize) __global__
    void logits_topk_kernel(const __nv_bfloat16* logits, std::int32_t k, std::int32_t* ids,
                            float* logprobs, float* log_normalizer, std::int32_t valid_rows,
                            std::int32_t physical_rows) {
    const std::int32_t column = static_cast<std::int32_t>(blockIdx.x);
    const __nv_bfloat16* source =
        logits + static_cast<std::int64_t>(column) * static_cast<std::int64_t>(physical_rows);

    __shared__ std::uint64_t warp_keys[BlockSize / common::kWarpSize];
    __shared__ std::uint64_t best_key;
    __shared__ float warp_sums[BlockSize / common::kWarpSize];
    __shared__ float column_log_sum;

    std::uint64_t bound = ~0ull; // key 0 is below every finite (value,id) pair
    float column_max    = 0.0f;
    for (std::int32_t rank = 0; rank < k; ++rank) {
        std::uint64_t local = 0;
        for (std::int32_t row = static_cast<std::int32_t>(threadIdx.x); row < valid_rows;
             row += BlockSize) {
            const std::uint64_t key =
                common::score_id_order_key(__bfloat162float(source[row]), row);
            if (key < bound && key > local) { local = key; }
        }
        const std::uint64_t reduced = common::block_reduce_max_key(local, warp_keys);
        if (threadIdx.x == 0) { best_key = reduced; }
        __syncthreads();
        const std::uint64_t key = best_key;

        if (rank == 0) {
            // Same maximum and same reduction shape as target_logprobs: the column maximum comes
            // from the key of rank 0, the sum uses one thread per strided row.
            column_max      = common::score_from_order_key(key);
            float local_sum = 0.0f;
            for (std::int32_t row = static_cast<std::int32_t>(threadIdx.x); row < valid_rows;
                 row += BlockSize) {
                local_sum += expf(__bfloat162float(source[row]) - column_max);
            }
            const float sum = common::block_reduce_sum<BlockSize>(local_sum, warp_sums);
            if (threadIdx.x == 0) {
                const float log_sum    = logf(sum);
                log_normalizer[column] = column_max + log_sum;
                column_log_sum         = log_sum;
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            ids[static_cast<std::int64_t>(column) * k + rank] = common::id_from_order_key(key);
            logprobs[static_cast<std::int64_t>(column) * k + rank] =
                common::score_from_order_key(key) - column_max - column_log_sum;
        }
        bound = key;
        __syncthreads();
    }
}

} // namespace ninfer::ops
