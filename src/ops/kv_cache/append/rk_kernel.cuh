#pragma once

// PackedV append kernel for rank-compressed KV layouts (rk8v4, rk4v4, rk4v4-e8, rk2v4-e8).
// K and V both receive the fixed FP32 D256 Hadamard rotation. V is quantized to int4
// (symmetric [-7,7], absmax/7 FP16 scale) and packed two adjacent dims per byte at
// stride 128 (head_dim/2). K is either int8 (rk8v4) or int4 (rk4v4/rk4v4-e8).

#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/append/geometry.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/kv_cache/rk4_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

template <typename Geometry, bool PackedK = false>
__device__ __forceinline__ void kv_cache_append_rk_row(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    std::int8_t* __restrict__ cache_k, std::int8_t* __restrict__ cache_v,
    __half* __restrict__ scale_k, __half* __restrict__ scale_v,
    int token, int kv_head, int physical_page, int page_offset, int lane,
    float* __restrict__ warp_scratch) {
    constexpr unsigned FullMask = 0xffffffffU;
    constexpr int D = 256;
    constexpr int Groups = D / 64;

    // --- K plane ---
    float k_vals[8];
    float k_absmax = 0.0F;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        k_vals[r] = __bfloat162float(k[static_cast<std::int64_t>(d) +
            static_cast<std::int64_t>(D) * (static_cast<std::int64_t>(kv_head) +
            static_cast<std::int64_t>(Geometry::KVHeads) * token)]);
    }
    normalized_hadamard_d256_inplace(k_vals, lane);
#pragma unroll
    for (float val : k_vals) k_absmax = fmaxf(k_absmax, fabsf(val));
    k_absmax = warp_max(k_absmax, FullMask);

    if constexpr (PackedK) {
        // int4 K: absmax/7 scale, pack two adjacent dims per byte at stride 128.
        __half k_scale = rk4_absmax_to_scale_h(k_absmax);
        float k_inv = __half2float(k_scale);
        k_inv = k_inv > 0.0f ? 1.0f / k_inv : 0.0f;
        // Use shuffle to get adjacent lane's value for packing.
        // After Hadamard, lane k holds dims [k, k+32, k+64, k+96, k+128, k+160, k+192, k+224].
        // For int4 packing we need adjacent dims (0,1), (2,3), ... per byte.
        // Lane 2k holds dim 2k, lane 2k+1 holds dim 2k+1 → shuffle to pack.
        // Each "half" (0-31, 32-63) within each group is packed by even lanes.
        for (int grp = 0; grp < Groups; ++grp) {
            const int base = grp * 64;
            for (int half = 0; half < 2; ++half) {
                const int dim_in_half = lane;  // 0..31
                const int abs_dim = base + half * 32 + dim_in_half;
                // The Hadamard value for this dim is in k_vals[half] (since
                // d = lane + 32*half, so k_vals[half] corresponds to dim abs_dim).
                float val = k_vals[half];
                float neighbor = __shfl_xor_sync(FullMask, val, 1);
                if ((dim_in_half & 1) == 0) {
                    auto c0 = rk4_quant_code(val, k_inv);
                    auto c1 = rk4_quant_code(neighbor, k_inv);
                    const int byte_idx = (half * 32 + dim_in_half) >> 1;
                    const std::int64_t off = paged_kv_element_offset<128, Geometry::KVHeads>(
                        physical_page, kv_head, page_offset, byte_idx);
                    cache_k[off] = static_cast<std::int8_t>(rk4_pack(c0, c1));
                }
            }
        }
        if (lane < Groups) {
            scale_k[paged_kv_element_offset<4, Geometry::KVHeads>(
                physical_page, kv_head, page_offset, lane)] = k_scale;
        }
    } else {
        // int8 K: same as the standard int8 codec (stride 256).
        const auto k_quant = kv_cache_int8_quant_params(k_absmax);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            cache_k[paged_kv_element_offset<D, Geometry::KVHeads>(
                physical_page, kv_head, page_offset, d)] =
                kv_cache_int8_quant_code(k_vals[r], k_quant.inverse_scale);
        }
        // int8 K has 4 groups, each lane 0 writes the group-0 scale; all 4 groups
        // need scales. Actually int8 K scale is per-group too: lane < Groups writes.
        if (lane < Groups) {
            // Each group's scale is the same (warp_max is broadcast), but the int8
            // codec computes absmax per 64-dim group. Since warp_max gives the global
            // max, and int8_group64 uses per-group absmax, we need per-group absmax.
            // For simplicity in the append kernel, use the global absmax for all groups
            // (same as the attention kernel does for Q).
            scale_k[paged_kv_element_offset<4, Geometry::KVHeads>(
                physical_page, kv_head, page_offset, lane)] = k_quant.scale;
        }
    }

    // --- V plane (int4 packed, shared by all rk variants) ---
    float v_vals[8];
    float v_absmax = 0.0F;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        v_vals[r] = __bfloat162float(v[static_cast<std::int64_t>(d) +
            static_cast<std::int64_t>(D) * (static_cast<std::int64_t>(kv_head) +
            static_cast<std::int64_t>(Geometry::KVHeads) * token)]);
    }
    normalized_hadamard_d256_inplace(v_vals, lane);
#pragma unroll
    for (float val : v_vals) v_absmax = fmaxf(v_absmax, fabsf(val));
    v_absmax = warp_max(v_absmax, FullMask);

    __half v_scale = rk4_absmax_to_scale_h(v_absmax);
    float v_inv = __half2float(v_scale);
    v_inv = v_inv > 0.0f ? 1.0f / v_inv : 0.0f;

    // Pack int4 V: two adjacent dims per byte at stride 128.
    for (int grp = 0; grp < Groups; ++grp) {
        for (int half = 0; half < 2; ++half) {
            const int dim_in_half = lane;
            float val = v_vals[half];
            float neighbor = __shfl_xor_sync(FullMask, val, 1);
            if ((dim_in_half & 1) == 0) {
                auto c0 = rk4_quant_code(val, v_inv);
                auto c1 = rk4_quant_code(neighbor, v_inv);
                const int byte_idx = (grp * 64 + half * 32 + dim_in_half) >> 1;
                const std::int64_t off = paged_kv_element_offset<128, Geometry::KVHeads>(
                    physical_page, kv_head, page_offset, byte_idx);
                cache_v[off] = static_cast<std::int8_t>(rk4_pack(c0, c1));
            }
        }
    }
    if (lane < Groups) {
        scale_v[paged_kv_element_offset<4, Geometry::KVHeads>(
            physical_page, kv_head, page_offset, lane)] = v_scale;
    }
}

template <typename Geometry, typename Metadata, bool PackedK = false, bool MultiBatch = false>
__launch_bounds__(256) __global__
    void kv_cache_append_full_rk_kernel(const __nv_bfloat16* __restrict__ k,
                                         const __nv_bfloat16* __restrict__ v,
                                         const std::int32_t* __restrict__ positions,
                                         Metadata metadata,
                                         std::int8_t* __restrict__ cache_k,
                                         std::int8_t* __restrict__ cache_v,
                                         __half* __restrict__ scale_k,
                                         __half* __restrict__ scale_v,
                                         std::int32_t tokens) {
    const int warp_id = threadIdx.x / 32;
    const int lane     = threadIdx.x & 31;
    const int warps    = blockDim.x / 32;

    // Shared memory for warp scratch (256 floats per warp).
    extern __shared__ float warp_scratch_all[];
    float* warp_scratch = warp_scratch_all + warp_id * 256;

    for (int tile = warp_id; tile < tokens * Geometry::KVHeads; tile += warps) {
        const int token  = tile / Geometry::KVHeads;
        const int kv_head = tile - token * Geometry::KVHeads;
        const int batch  = MultiBatch ? blockIdx.z : 0;
        const int pos    = MultiBatch
            ? positions[token + batch * tokens]
            : positions[token];
        if (pos < 0) continue;
        const std::int32_t* block_table = metadata.block_table();
        const int physical_page = block_table[pos >> 6];
        const int page_offset   = pos & 63;

        kv_cache_append_rk_row<Geometry, PackedK>(
            k + (MultiBatch ? static_cast<std::int64_t>(batch) * Geometry::KVHeads * 256 * tokens : 0),
            v + (MultiBatch ? static_cast<std::int64_t>(batch) * Geometry::KVHeads * 256 * tokens : 0),
            cache_k, cache_v, scale_k, scale_v,
            token, kv_head, physical_page, page_offset, lane, warp_scratch);
    }
}

} // namespace ninfer::ops
