#pragma once

// Logits top-K + softmax kernel for MTP rejection sampling.
// Implements: include/ninfer/ops/speculative_round.h (logits_topk_softmax)
//
// Per column t of logits [V, T]:
//   sum_exp = Σ_v exp(logits[v, t])
//   top-K = K largest (value, index) pairs, ties broken by lower index
//   top_probs[i, t] = exp(top_value[i]) / sum_exp
//   top_ids[i, t] = id_map ? id_map[top_index[i]] : top_index[i]
//   argmax_ids[t] = top_ids[0, t]
//
// Algorithm: single-pass per-thread bounded insertion (each thread keeps its
// local top-K in registers), then a shared-memory merge selects the block
// top-K. For K=16, V=248320, blockDim=256: per-thread scan is V/256 ≈ 970
// elements (1 comparison each for most), merge is 256 threads × K entries
// iteratively reduced. The old approach scanned V K times (16×970 per thread);
// this does 970 + merge overhead — a ~16x reduction.
//
// Output layout: column-major with caller-supplied column_stride.
// Pass column_stride=0 for the natural contiguous stride (top_k).
// id_map is a DEVICE pointer to V int32 values, or null for identity.

#include <cuda_bf16.h>
#include <cstdint>
#include <climits>
#include <math_constants.h>

namespace ninfer::ops {

inline constexpr int kLogitsTopkBlock = 256;
inline constexpr int kLogitsTopkMax   = 32;

__device__ __forceinline__ bool topk_better(float value, std::int32_t index, float best_value,
                                            std::int32_t best_index) {
    return value > best_value || (value == best_value && index < best_index);
}

// Insert (value, index) into a descending-sorted register array of size K.
// Precondition: value > arr[K-1] (caller checks).
__device__ __forceinline__ void topk_insert(float* val, std::int32_t* idx, int k, float value,
                                            std::int32_t index) {
    int pos = k - 1;
    for (int i = k - 2; i >= 0; --i) {
        if (value <= val[i]) { break; }
        val[i + 1] = val[i];
        idx[i + 1] = idx[i];
        pos        = i;
    }
    val[pos] = value;
    idx[pos] = index;
}

__launch_bounds__(kLogitsTopkBlock) __global__ void logits_topk_softmax_kernel(
    const __nv_bfloat16* __restrict__ logits, std::int32_t* __restrict__ top_ids,
    float* __restrict__ top_probs, std::int32_t* __restrict__ argmax_ids,
    const std::int32_t* __restrict__ id_map, std::int32_t valid_rows,
    std::int32_t physical_rows, std::int32_t top_k, std::int32_t column_stride) {
    const std::int32_t t    = static_cast<std::int32_t>(blockIdx.x);
    const std::int64_t base = static_cast<std::int64_t>(t) * physical_rows;
    const __nv_bfloat16* col = logits + base;

    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // --- Phase 1: per-thread scan (sum_exp + local top-K in registers) ---
    float sum_exp = 0.0f;

    // Per-thread bounded top-K in registers (initialized to -inf)
    float lval[kLogitsTopkMax];
    std::int32_t lidx[kLogitsTopkMax];
#pragma unroll
    for (int i = 0; i < kLogitsTopkMax; ++i) {
        lval[i] = -CUDART_INF_F;
        lidx[i] = INT32_MAX;
    }

    for (std::int32_t v = tid; v < valid_rows; v += kLogitsTopkBlock) {
        const float val = __bfloat162float(col[v]);
        sum_exp += __expf(val);
        if (val > lval[top_k - 1]) {
            topk_insert(lval, lidx, top_k, val, v);
        }
    }

    // Warp-level sum_exp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_exp += __shfl_down_sync(0xffffffffu, sum_exp, offset);
    }

    // Block-level sum_exp
    __shared__ float s_warp_sum[kLogitsTopkBlock / 32];
    __shared__ float s_sum_exp;
    if (lane == 0) { s_warp_sum[warp] = sum_exp; }
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (int w = 0; w < kLogitsTopkBlock / 32; ++w) { total += s_warp_sum[w]; }
        s_sum_exp = total;
    }
    __syncthreads();

    // --- Phase 2: block merge (iterative selection from per-thread candidates) ---
    // Each thread's top-K is already in registers. We iteratively select the
    // block-wide top-K by having each thread find its best unselected entry,
    // doing a warp + block reduction, and marking the winner.
    __shared__ float s_top_val[kLogitsTopkMax];
    __shared__ std::int32_t s_top_idx[kLogitsTopkMax];
    __shared__ float s_warp_val[kLogitsTopkBlock / 32];
    __shared__ std::int32_t s_warp_idx[kLogitsTopkBlock / 32];

    for (int k = 0; k < top_k; ++k) {
        // Each thread finds its best among its local top-K entries not yet selected
        float best_val        = -CUDART_INF_F;
        std::int32_t best_idx = INT32_MAX;
        for (int i = 0; i < top_k; ++i) {
            const float v  = lval[i];
            const std::int32_t ix = lidx[i];
            bool selected = false;
            for (int s = 0; s < k; ++s) {
                if (s_top_idx[s] == ix) { selected = true; break; }
            }
            if (!selected && topk_better(v, ix, best_val, best_idx)) {
                best_val = v;
                best_idx = ix;
            }
        }

        // Warp reduce
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, best_val, offset);
            const std::int32_t oi = __shfl_down_sync(0xffffffffu, best_idx, offset);
            if (ov > best_val || (ov == best_val && oi < best_idx)) {
                best_val = ov;
                best_idx = oi;
            }
        }
        if (lane == 0) {
            s_warp_val[warp] = best_val;
            s_warp_idx[warp] = best_idx;
        }
        __syncthreads();

        // Block reduce (thread 0)
        if (tid == 0) {
            float bv        = -CUDART_INF_F;
            std::int32_t bi = INT32_MAX;
            for (int w = 0; w < kLogitsTopkBlock / 32; ++w) {
                if (topk_better(s_warp_val[w], s_warp_idx[w], bv, bi)) {
                    bv = s_warp_val[w];
                    bi = s_warp_idx[w];
                }
            }
            s_top_val[k] = bv;
            s_top_idx[k] = bi;
        }
        __syncthreads();
    }

    // --- Phase 3: write results (column-major with stride) ---
    if (tid == 0) {
        const float inv_sum = s_sum_exp > 0.0f ? 1.0f / s_sum_exp : 0.0f;
        const std::int32_t stride = column_stride > 0 ? column_stride : top_k;
        for (int k = 0; k < top_k; ++k) {
            const std::int32_t global_id = id_map != nullptr ? id_map[s_top_idx[k]] : s_top_idx[k];
            top_ids[static_cast<std::int64_t>(t) * stride + k] = global_id;
            top_probs[static_cast<std::int64_t>(t) * stride + k] =
                __expf(s_top_val[k]) * inv_sum;
        }
        argmax_ids[t] = id_map != nullptr ? id_map[s_top_idx[0]] : s_top_idx[0];
    }
}

} // namespace ninfer::ops
