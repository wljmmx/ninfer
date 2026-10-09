#pragma once

// Logits top-K + softmax kernel for MTP rejection sampling.
// Implements: include/ninfer/ops/speculative_round.h (logits_topk_softmax)
//
// Per column t of logits [V, T]:
//   sum_exp = sum_v exp(logits[v, t])
//   top-K = K largest (value, index) pairs, ties broken by lower index
//   top_probs[i, t] = exp(top_value[i]) / sum_exp
//   top_ids[i, t] = id_map ? id_map[top_index[i]] : top_index[i]
//   argmax_ids[t] = top_ids[0, t]
//
// K is a compile-time template parameter. The per-thread top-K register arrays are
// indexed by K and by the insertion position, so a runtime top_k would force them into
// local memory and dominate the scan; with K known the arrays stay in registers.
//
// Algorithm: single-pass per-thread bounded insertion (each thread keeps its local
// top-K in registers), then a shared-memory merge selects the block top-K. Each thread
// still visits the identical index sequence (tid, tid+Block, tid+2*Block, ...), so the
// result is bit-identical to the previous runtime-K implementation.
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

// Number of independent column loads issued back-to-back before the dependent
// expf/insert work, so the scan does not serialise on one load latency at a time.
inline constexpr int kLogitsTopkScanBatch = 8;

__device__ __forceinline__ bool topk_better(float value, std::int32_t index, float best_value,
                                            std::int32_t best_index) {
    return value > best_value || (value == best_value && index < best_index);
}

// Insert (value, index) into a descending-sorted register array of size K.
// Precondition: value > arr[K-1] (caller checks).
template <int K>
__device__ __forceinline__ void topk_insert(float* val, std::int32_t* idx, float value,
                                            std::int32_t index) {
    int pos = K - 1;
    for (int i = K - 2; i >= 0; --i) {
        if (value <= val[i]) { break; }
        val[i + 1] = val[i];
        idx[i + 1] = idx[i];
        pos        = i;
    }
    val[pos] = value;
    idx[pos] = index;
}

template <int K>
__launch_bounds__(kLogitsTopkBlock) __global__ void logits_topk_softmax_kernel(
    const __nv_bfloat16* __restrict__ logits, std::int32_t* __restrict__ top_ids,
    float* __restrict__ top_probs, std::int32_t* __restrict__ argmax_ids,
    const std::int32_t* __restrict__ id_map, std::int32_t valid_rows,
    std::int32_t physical_rows, std::int32_t column_stride) {
    static_assert(K >= 1 && K <= kLogitsTopkMax, "logits top-K template bound");
    const std::int32_t t     = static_cast<std::int32_t>(blockIdx.x);
    const std::int64_t base  = static_cast<std::int64_t>(t) * physical_rows;
    const __nv_bfloat16* col = logits + base;

    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // --- Phase 1: per-thread scan (sum_exp + local top-K in registers) ---
    float sum_exp = 0.0f;

    float lval[K];
    std::int32_t lidx[K];
#pragma unroll
    for (int i = 0; i < K; ++i) {
        lval[i] = -CUDART_INF_F;
        lidx[i] = INT32_MAX;
    }

    std::int32_t v = tid;
    for (; v + (kLogitsTopkScanBatch - 1) * kLogitsTopkBlock < valid_rows;
         v += kLogitsTopkScanBatch * kLogitsTopkBlock) {
        float w[kLogitsTopkScanBatch];
#pragma unroll
        for (int j = 0; j < kLogitsTopkScanBatch; ++j) {
            w[j] = __bfloat162float(col[v + j * kLogitsTopkBlock]);
        }
#pragma unroll
        for (int j = 0; j < kLogitsTopkScanBatch; ++j) {
            const float val = w[j];
            sum_exp += __expf(val);
            if (val > lval[K - 1]) {
                topk_insert<K>(lval, lidx, val, v + j * kLogitsTopkBlock);
            }
        }
    }
    for (; v < valid_rows; v += kLogitsTopkBlock) {
        const float val = __bfloat162float(col[v]);
        sum_exp += __expf(val);
        if (val > lval[K - 1]) {
            topk_insert<K>(lval, lidx, val, v);
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

    // --- Phase 2: block merge ---
    // Iteratively select the block-wide best among each thread's not-yet-used entries.
    // A per-thread "used" flag replaces the previous re-scan over every already-selected
    // slot (O(K^2) shared reads per round); the selection order and tie-breaking are
    // unchanged.
    __shared__ float s_top_val[kLogitsTopkMax];
    __shared__ std::int32_t s_top_idx[kLogitsTopkMax];
    __shared__ float s_warp_val[kLogitsTopkBlock / 32];
    __shared__ std::int32_t s_warp_idx[kLogitsTopkBlock / 32];

    bool used[K];
#pragma unroll
    for (int i = 0; i < K; ++i) { used[i] = false; }

    for (int k = 0; k < K; ++k) {
        float best_val        = -CUDART_INF_F;
        std::int32_t best_idx = INT32_MAX;
#pragma unroll
        for (int i = 0; i < K; ++i) {
            if (used[i]) { continue; }
            if (topk_better(lval[i], lidx[i], best_val, best_idx)) {
                best_val = lval[i];
                best_idx = lidx[i];
            }
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            const float ov        = __shfl_down_sync(0xffffffffu, best_val, offset);
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

        const std::int32_t winner = s_top_idx[k];
#pragma unroll
        for (int i = 0; i < K; ++i) {
            if (!used[i] && lidx[i] == winner) { used[i] = true; }
        }
    }

    // --- Phase 3: write results (column-major with stride) ---
    if (tid == 0) {
        const float inv_sum = s_sum_exp > 0.0f ? 1.0f / s_sum_exp : 0.0f;
        const std::int32_t stride = column_stride > 0 ? column_stride : K;
        for (int k = 0; k < K; ++k) {
            const std::int32_t global_id = id_map != nullptr ? id_map[s_top_idx[k]] : s_top_idx[k];
            top_ids[static_cast<std::int64_t>(t) * stride + k] = global_id;
            top_probs[static_cast<std::int64_t>(t) * stride + k] =
                __expf(s_top_val[k]) * inv_sum;
        }
        argmax_ids[t] = id_map != nullptr ? id_map[s_top_idx[0]] : s_top_idx[0];
    }
}

} // namespace ninfer::ops
