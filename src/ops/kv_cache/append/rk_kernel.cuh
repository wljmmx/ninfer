#pragma once

// PackedV append kernel for rank-compressed KV layouts (rk8v4, rk4v4, rk4v4-e8, rk2v4-e8).
// K and V both receive the fixed FP32 D256 Hadamard rotation. V is quantized to int4
// (symmetric [-7,7], absmax/7 FP16 scale) and packed two adjacent dims per byte at
// stride 128 (head_dim/2). K is int8 (rk8v4), int4 RTN (rk4v4), int4 with E8 lattice
// projection (rk4v4-e8), or the 2-bit E8 cylinder (rk2v4-e8, stride 64).

#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kernel/e8_lattice.cuh"
#include "ops/kernel/e8_root_codec.cuh"
#include "ops/kv_cache/append/geometry.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/kv_cache/rk4_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

template <typename Geometry, bool PackedK = false, bool E8Lattice = false, bool E8Root = false>
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
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        k_vals[r] = __bfloat162float(k[static_cast<std::int64_t>(d) +
            static_cast<std::int64_t>(D) * (static_cast<std::int64_t>(kv_head) +
            static_cast<std::int64_t>(Geometry::KVHeads) * token)]);
    }
    normalized_hadamard_d256_inplace(k_vals, lane);

        if constexpr (E8Root) {
        // rk2v4-e8 K: 2-bit E8 cylinder codes, one (root, rad_axis) byte pair per
        // 8D subspace — 16 bytes per G64 group, 64 bytes per token (stride 64).
        // Scale convention follows the reference cylinder codec: FP16-RNE(group
        // absmax / 7). Per-group absmax (like the fused append) keeps the radius
        // index centred in the log-radius table. k_vals[2g] holds dims g*64+lane
        // and k_vals[2g+1] dims g*64+32+lane, so the 8-lane subgroup (lane>>3)
        // carries one consecutive 8-dim subspace: sub = half*4 + (lane>>3).
        // Vectorized store: every lane encodes BOTH halves into registers; each
        // leader lane (lane&7)==0 gathers the NEXT subgroup's byte pair via one
        // shuffle and writes two uint32 words per group covering subspaces
        // {s, s+1} and {s+4, s+5} — 16 uint32 stores per token instead of 64
        // byte stores.
        for (int grp = 0; grp < Groups; ++grp) {
            float g_abs = fmaxf(fabsf(k_vals[2 * grp]), fabsf(k_vals[2 * grp + 1]));
            g_abs       = warp_max(g_abs, FullMask);
            const __half k_scale = rk4_absmax_to_scale_h(g_abs);
            const float ks       = __half2float(k_scale);
            uint8_t root0, rad0, root1, rad1;
            e8_encode_cylinder_8d_warp(k_vals[2 * grp], ks, root0, rad0, lane);
            e8_encode_cylinder_8d_warp(k_vals[2 * grp + 1], ks, root1, rad1, lane);
            // Byte pair per subspace: (root, rad_axis) at offset sub*2 within the
            // group's 16-byte block. half=0 → sub = lane>>3, half=1 → sub+4.
            const std::uint32_t pair0 = root0 | (rad0 << 8);
            const std::uint32_t pair1 = root1 | (rad1 << 8);
            const std::uint32_t pair0_next = __shfl_xor_sync(FullMask, pair0, 8);
            const std::uint32_t pair1_next = __shfl_xor_sync(FullMask, pair1, 8);
            // Writers are lanes 0 and 16 (subgroups 0 and 2): the XOR-8 gather
            // pairs each writer with the NEXT subgroup (0→1, 2→3), so one writer
            // covers subspaces {s, s+1} and {s+4, s+5}. Odd subgroup leaders must
            // NOT write — their XOR partner is the PREVIOUS subgroup.
            if ((lane & 15) == 0) {
                const int s0 = lane >> 3;                      // 0 or 2
                const int bo0 = (grp * 8 + s0) * 2;            // subspaces {s0, s0+1}
                const int bo1 = (grp * 8 + s0 + 4) * 2;       // subspaces {s0+4, s0+5}
                const std::int64_t off0 = paged_kv_element_offset<64, Geometry::KVHeads>(
                    physical_page, kv_head, page_offset, bo0);
                *reinterpret_cast<std::uint32_t*>(&cache_k[off0]) =
                    pair0 | (pair0_next << 16);
                const std::int64_t off1 = paged_kv_element_offset<64, Geometry::KVHeads>(
                    physical_page, kv_head, page_offset, bo1);
                *reinterpret_cast<std::uint32_t*>(&cache_k[off1]) =
                    pair1 | (pair1_next << 16);
            }
            if (lane == 0) {
                scale_k[paged_kv_element_offset<4, Geometry::KVHeads>(
                    physical_page, kv_head, page_offset, grp)] = k_scale;
            }
        }
    } else if constexpr (PackedK) {
        // int4 K: PER-GROUP absmax/7 FP16 scale, matching the fused append and the
        // reference rk4v4/rk4v4-e8 codecs — a single global scale wastes 1-2 bits
        // of the [-7,7] grid whenever a group's absmax sits below the row maximum.
        // k_vals[r] holds dim lane + 32*r, so group g's two halves live in
        // k_vals[2g] and k_vals[2g+1].
        for (int grp = 0; grp < Groups; ++grp) {
            float g_abs = fmaxf(fabsf(k_vals[2 * grp]), fabsf(k_vals[2 * grp + 1]));
            g_abs       = warp_max(g_abs, FullMask);
            const __half k_scale = rk4_absmax_to_scale_h(g_abs);
            float k_inv = __half2float(k_scale);
            k_inv = k_inv > 0.0f ? 1.0f / k_inv : 0.0f;
            float half_vals[2] = {k_vals[2 * grp], k_vals[2 * grp + 1]};
            if constexpr (E8Lattice) {
                // E8 projection in the SCALED space (units of THIS group's k_scale),
                // matching the reference rk4v4-e8 codec; the half-integral coset
                // collapses in the int4 round below.
                half_vals[0] *= k_inv;
                half_vals[1] *= k_inv;
                e8_project_8d_warp(half_vals[0], half_vals[1], lane);
                k_inv = 1.0f;
            }
            // Vectorized int4 K pack: every lane quantizes its own dimension, even
            // lanes build the packed byte, then writer lanes (multiples of 8) gather
            // four consecutive bytes with three shuffles and store one uint32 —
            // 32 uint32 stores per token-head instead of 128 byte stores. The byte
            // offsets stay 4-aligned ((grp*64+half*32+lane)>>1 with lane=8m).
            for (int half = 0; half < 2; ++half) {
                const int c_self = rk4_quant_code(half_vals[half], k_inv);
                const int c_nbr  = __shfl_xor_sync(FullMask, c_self, 1);
                std::uint32_t my_byte = 0;
                if ((lane & 1) == 0) {
                    my_byte = rk4_pack(static_cast<std::int8_t>(c_self),
                                       static_cast<std::int8_t>(c_nbr));
                }
                const std::uint32_t b1 = __shfl_xor_sync(FullMask, my_byte, 2);
                const std::uint32_t b2 = __shfl_xor_sync(FullMask, my_byte, 4);
                const std::uint32_t b3 = __shfl_xor_sync(FullMask, my_byte, 6);
                if ((lane & 7) == 0) {
                    const int byte_idx = (grp * 64 + half * 32 + lane) >> 1;
                    const std::int64_t off = paged_kv_element_offset<128, Geometry::KVHeads>(
                        physical_page, kv_head, page_offset, byte_idx);
                    *reinterpret_cast<std::uint32_t*>(&cache_k[off]) =
                        my_byte | (b1 << 8) | (b2 << 16) | (b3 << 24);
                }
            }
            if (lane == 0) {
                scale_k[paged_kv_element_offset<4, Geometry::KVHeads>(
                    physical_page, kv_head, page_offset, grp)] = k_scale;
            }
        }
    } else {
        // int8 K: PER-GROUP absmax/127 FP16 scale, identical to the standard
        // int8-group64 codec. Group g's dims live in k_vals[2g] (lane+32*2g) and
        // k_vals[2g+1] (lane+32*(2g+1)); each group quantizes against its own scale.
        for (int grp = 0; grp < Groups; ++grp) {
            float g_abs = fmaxf(fabsf(k_vals[2 * grp]), fabsf(k_vals[2 * grp + 1]));
            g_abs       = warp_max(g_abs, FullMask);
            const auto k_quant = kv_cache_int8_quant_params(g_abs);
            cache_k[paged_kv_element_offset<D, Geometry::KVHeads>(
                physical_page, kv_head, page_offset, lane + 32 * (2 * grp))] =
                kv_cache_int8_quant_code(k_vals[2 * grp], k_quant.inverse_scale);
            cache_k[paged_kv_element_offset<D, Geometry::KVHeads>(
                physical_page, kv_head, page_offset, lane + 32 * (2 * grp + 1))] =
                kv_cache_int8_quant_code(k_vals[2 * grp + 1], k_quant.inverse_scale);
            if (lane == 0) {
                scale_k[paged_kv_element_offset<4, Geometry::KVHeads>(
                    physical_page, kv_head, page_offset, grp)] = k_quant.scale;
            }
        }
    }

    // --- V plane (int4, shared by all rk variants) ---
    // NOTE: V is stored in the ORIGINAL (unrotated) coordinate frame, matching the
    // fused append path and the int8-family merge (InverseRotation=false): the PV
    // MMA output goes straight to the residual stream, so the cache V must be raw.
    // Rotating V here corrupted every prefill-appended token (rk8v4 "2020" garbage).
    float v_vals[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        v_vals[r] = __bfloat162float(v[static_cast<std::int64_t>(d) +
            static_cast<std::int64_t>(D) * (static_cast<std::int64_t>(kv_head) +
            static_cast<std::int64_t>(Geometry::KVHeads) * token)]);
    }

    // PER-GROUP absmax/7 FP16 scale (matches the fused append): each 64-dim group
    // quantizes against its own scale. Vectorized int4 V pack (same writer-gather
    // pattern as the K pack): every lane quantizes its own dim, even lanes build
    // the packed byte, writer lanes (multiples of 8) gather four bytes and store
    // one uint32.
    for (int grp = 0; grp < Groups; ++grp) {
        float g_abs = fmaxf(fabsf(v_vals[2 * grp]), fabsf(v_vals[2 * grp + 1]));
        g_abs       = warp_max(g_abs, FullMask);
        const __half v_scale = rk4_absmax_to_scale_h(g_abs);
        float v_inv = __half2float(v_scale);
        v_inv = v_inv > 0.0f ? 1.0f / v_inv : 0.0f;
        for (int half = 0; half < 2; ++half) {
            const int c_self = rk4_quant_code(v_vals[2 * grp + half], v_inv);
            const int c_nbr  = __shfl_xor_sync(FullMask, c_self, 1);
            std::uint32_t my_byte = 0;
            if ((lane & 1) == 0) {
                my_byte = rk4_pack(static_cast<std::int8_t>(c_self),
                                   static_cast<std::int8_t>(c_nbr));
            }
            const std::uint32_t b1 = __shfl_xor_sync(FullMask, my_byte, 2);
            const std::uint32_t b2 = __shfl_xor_sync(FullMask, my_byte, 4);
            const std::uint32_t b3 = __shfl_xor_sync(FullMask, my_byte, 6);
            if ((lane & 7) == 0) {
                const int byte_idx = (grp * 64 + half * 32 + lane) >> 1;
                const std::int64_t off = paged_kv_element_offset<128, Geometry::KVHeads>(
                    physical_page, kv_head, page_offset, byte_idx);
                *reinterpret_cast<std::uint32_t*>(&cache_v[off]) =
                    my_byte | (b1 << 8) | (b2 << 16) | (b3 << 24);
            }
        }
        if (lane == 0) {
            scale_v[paged_kv_element_offset<4, Geometry::KVHeads>(
                physical_page, kv_head, page_offset, grp)] = v_scale;
        }
    }
}

template <typename Geometry, typename Metadata, bool PackedK = false, bool E8Lattice = false,
          bool E8Root = false, bool MultiBatch = false>
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

    // Grid-stride over (token, kv_head) rows: the launch sizes the grid as
    // div_up(tokens * KVHeads, warps), so every (block, warp) pair owns exactly
    // one row. The base offset must include blockIdx.x * warps — without it every
    // block restarts the same tile sequence and the whole grid duplicates one
    // block's work (512x on a 1024-token prefill chunk).
    const int total_rows = tokens * Geometry::KVHeads;
    const int row_base   = static_cast<int>(blockIdx.x) * warps;
    for (int tile = row_base + warp_id; tile < total_rows; tile += gridDim.x * warps) {
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

        kv_cache_append_rk_row<Geometry, PackedK, E8Lattice, E8Root>(
            k + (MultiBatch ? static_cast<std::int64_t>(batch) * Geometry::KVHeads * 256 * tokens : 0),
            v + (MultiBatch ? static_cast<std::int64_t>(batch) * Geometry::KVHeads * 256 * tokens : 0),
            cache_k, cache_v, scale_k, scale_v,
            token, kv_head, physical_page, page_offset, lane, warp_scratch);
    }
}

} // namespace ninfer::ops
