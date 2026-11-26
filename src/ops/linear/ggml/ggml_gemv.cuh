#pragma once

// The GGML encodings' quantised matvec: one warp per output row, one lane per 32-value group.
//
// A weight row is a run of encoded blocks; a lane owns whole 32-value groups, so the decode never
// materialises a block or an element array - it accumulates `dot(decode(group), x)` in registers and
// the warp reduces at the end. That is the decomposition strata uses for the same encodings
// (src/kernels/cuda/dequant_bf16.cu's `group32`, and llama.cpp's `vec_dot_*` in vecdotq.cuh), which
// keeps this kernel on a path the reference implementations already validated at scale.
//
// Invariants this file relies on:
//   * K is a multiple of 32, and every block's element count is a multiple of 32 (true for all the
//     types carried: 256-value super-blocks, Q2_0's 64, IQ4_NL's 32);
//   * a row's blocks are contiguous in GGUF order (n_in contiguous), which is how the container
//     stores them;
//   * the activation is BF16 [K, T] with x[k + t*D] the value for column t. Every activation value is
//     converted exactly once per row it takes part in - a lane's 32 values are its own group, not a
//     copy of another lane's.
//
// Numerics: the accumulation is binary32 in group order, like src/core/ggml_dequant.h followed by a
// float dot, so tests/ops/test_ggml_linear.cpp bounds it against that reference.

#include "core/ggml_types.h"
#include "ops/linear/ggml/ggml_block_decode.cuh"

#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops::detail::ggml {

__device__ __forceinline__ float bf(const __nv_bfloat16* x, int i) noexcept {
    return __bfloat162float(x[i]);
}

// One 32-value group of one encoded block, dotted against the group's 32 activation values.
template <int TYPE>
__device__ __forceinline__ float dot32(const std::uint8_t* block, int group_in_block,
                                       const __nv_bfloat16* x) {
    float acc = 0.0f;
    if constexpr (TYPE == 0) { // Q4_K
        const float d              = half_at(block);
        const float mn             = half_at(block + 2);
        const std::uint8_t* scales = block + 4;
        const int chunk            = group_in_block / 2;
        const bool high            = group_in_block % 2 != 0;
        int sc = 0, m = 0;
        scale_min_k4(2 * chunk + (high ? 1 : 0), scales, sc, m);
        const float dg        = d * float(sc), mg = mn * float(m);
        const std::uint8_t* q = block + 16 + 32 * chunk;
#pragma unroll
        for (int l = 0; l < 32; ++l) {
            const int code = high ? (q[l] >> 4) : (q[l] & 0x0f);
            acc            = fmaf(dg * float(code) - mg, bf(x, l), acc);
        }
    } else if constexpr (TYPE == 1) { // Q5_K
        const float d              = half_at(block);
        const float mn             = half_at(block + 2);
        const std::uint8_t* scales = block + 4;
        const std::uint8_t* qh     = block + 16;
        const int chunk            = group_in_block / 2;
        const bool high            = group_in_block % 2 != 0;
        int sc = 0, m = 0;
        scale_min_k4(2 * chunk + (high ? 1 : 0), scales, sc, m);
        const float dg         = d * float(sc), mg = mn * float(m);
        const std::uint8_t* ql = block + 48 + 32 * chunk;
        const int bit          = (1 << (2 * chunk)) << (high ? 1 : 0);
#pragma unroll
        for (int l = 0; l < 32; ++l) {
            const int code = (high ? (ql[l] >> 4) : (ql[l] & 0x0f)) + ((qh[l] & bit) ? 16 : 0);
            acc            = fmaf(dg * float(code) - mg, bf(x, l), acc);
        }
    } else if constexpr (TYPE == 2) { // Q6_K
        const float d          = half_at(block + 208);
        const int half_index   = group_in_block / 4;
        const int sub          = group_in_block % 4;
        const std::uint8_t* ql = block + 64 * half_index;
        const std::uint8_t* qh = block + 128 + 32 * half_index;
        const std::uint8_t* sc = block + 192 + 8 * half_index;
#pragma unroll
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            int code;
            float scale;
            if (sub == 0) {
                code  = int((ql[l + 0] & 0x0f) | (((qh[l] >> 0) & 3) << 4)) - 32;
                scale = d * float(std::int8_t(sc[is + 0]));
            } else if (sub == 1) {
                code  = int((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4)) - 32;
                scale = d * float(std::int8_t(sc[is + 2]));
            } else if (sub == 2) {
                code  = int((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                scale = d * float(std::int8_t(sc[is + 4]));
            } else {
                code  = int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                scale = d * float(std::int8_t(sc[is + 6]));
            }
            acc = fmaf(scale * float(code), bf(x, l), acc);
        }
    } else if constexpr (TYPE == 3) { // Q2_0
        const float d = half_at(block);
        const int e0  = (group_in_block % 2) * 32;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            const int e    = e0 + j;
            const int code = (block[2 + e / 4] >> ((e % 4) * 2)) & 3;
            acc            = fmaf((float(code) - 1.0f) * d, bf(x, j), acc);
        }
    } else if constexpr (TYPE == 4) { // IQ4_NL - one 32-value group per block
        const float d = half_at(block);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            acc = fmaf(d * float(kIq4nlValues[block[2 + j] & 0x0f]), bf(x, j), acc);
            acc = fmaf(d * float(kIq4nlValues[block[2 + j] >> 4]), bf(x, j + 16), acc);
        }
    } else if constexpr (TYPE == 5) { // IQ4_XS
        const float d            = half_at(block);
        const std::uint32_t high = load_u16(block + 2);
        const std::uint8_t* low  = block + 4;
        const int ls = int((low[group_in_block / 2] >> (4 * (group_in_block % 2))) & 0x0f) |
                       int(((high >> (2 * group_in_block)) & 3u) << 4);
        const float dl         = d * float(ls - 32);
        const std::uint8_t* qs = block + 8 + 16 * group_in_block;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            acc = fmaf(dl * float(kIq4nlValues[qs[j] & 0x0f]), bf(x, j), acc);
            acc = fmaf(dl * float(kIq4nlValues[qs[j] >> 4]), bf(x, j + 16), acc);
        }
    } else if constexpr (TYPE == 6) { // IQ2_XXS
        const float d            = half_at(block);
        const std::uint8_t* aux8 = block + 2 + 8 * group_in_block;
        const std::uint32_t aux1 = load_u32(aux8 + 4);
        const float db           = d * (0.5f + float(aux1 >> 28)) * 0.25f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const std::uint64_t entry = kIq2xxsGrid[aux8[l]];
            const std::uint8_t signs  = kIq2xsSigns[(aux1 >> (7 * l)) & 127u];
#pragma unroll
            for (int j = 0; j < 8; ++j)
                acc = fmaf(db * grid8(entry, j) * sign_of(signs, j), bf(x, 8 * l + j), acc);
        }
    } else if constexpr (TYPE == 7) { // IQ2_XS
        const float d              = half_at(block);
        const std::uint8_t* qs     = block + 2 + 8 * group_in_block;
        const std::uint8_t* scales = block + 66 + group_in_block;
        const float db[2] = {d * (0.5f + float(scales[0] & 0x0f)) * 0.25f,
                             d * (0.5f + float(scales[0] >> 4)) * 0.25f};
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const std::uint32_t w    = load_u16(qs + 2 * l);
            const std::uint64_t e    = kIq2xsGrid[w & 511u];
            const std::uint8_t signs = kIq2xsSigns[w >> 9];
#pragma unroll
            for (int j = 0; j < 8; ++j)
                acc = fmaf(db[l / 2] * grid8(e, j) * sign_of(signs, j), bf(x, 8 * l + j), acc);
        }
    } else if constexpr (TYPE == 8) { // IQ2_S
        const float d              = half_at(block);
        const std::uint8_t* qs     = block + 2 + 4 * group_in_block;
        const std::uint8_t* qh     = block + 66 + group_in_block;
        const std::uint8_t* signs0 = block + 34 + 4 * group_in_block;
        const std::uint8_t* scales = block + 74 + group_in_block;
        const float db[2] = {d * (0.5f + float(scales[0] & 0x0f)) * 0.25f,
                             d * (0.5f + float(scales[0] >> 4)) * 0.25f};
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const std::uint32_t index =
                std::uint32_t(qs[l]) | (std::uint32_t(qh[0] << (8 - 2 * l)) & 0x300u);
            const std::uint64_t e = kIq2sGrid[index];
#pragma unroll
            for (int j = 0; j < 8; ++j)
                acc = fmaf(db[l / 2] * grid8(e, j) * sign_of(signs0[l], j), bf(x, 8 * l + j), acc);
        }
    } else if constexpr (TYPE == 9) { // IQ3_XXS
        const float d                        = half_at(block);
        const std::uint8_t* q                = block + 2 + 8 * group_in_block;
        const std::uint8_t* scales_and_signs = block + 66 + 4 * group_in_block;
        const std::uint32_t aux              = load_u32(scales_and_signs);
        const float db                       = d * (0.5f + float(aux >> 28)) * 0.5f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const std::uint8_t signs = kIq2xsSigns[(aux >> (7 * l)) & 127u];
            const std::uint32_t g1   = kIq3xxsGrid[q[2 * l + 0]];
            const std::uint32_t g2   = kIq3xxsGrid[q[2 * l + 1]];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc = fmaf(db * grid4(g1, j) * sign_of(signs, j), bf(x, 8 * l + j), acc);
                acc = fmaf(db * grid4(g2, j) * sign_of(signs, j + 4), bf(x, 8 * l + j + 4), acc);
            }
        }
    } else if constexpr (TYPE == 10) { // IQ3_S
        const float d              = half_at(block);
        const int pair             = group_in_block / 2;
        const int half             = group_in_block % 2;
        const std::uint8_t* q      = block + 2 + 8 * group_in_block;
        const std::uint8_t* signs  = block + 74 + 4 * group_in_block;
        const std::uint8_t* h      = block + 66 + group_in_block;
        const std::uint8_t* scales = block + 106 + pair;
        const float db =
            d * float(1 + 2 * int(half == 0 ? (scales[0] & 0x0f) : (scales[0] >> 4)));
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const std::uint32_t i1 =
                std::uint32_t(q[2 * l + 0]) | (std::uint32_t(h[0] << (8 - 2 * l)) & 256u);
            const std::uint32_t i2 =
                std::uint32_t(q[2 * l + 1]) | (std::uint32_t(h[0] << (7 - 2 * l)) & 256u);
            const std::uint32_t g1 = kIq3sGrid[i1];
            const std::uint32_t g2 = kIq3sGrid[i2];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc = fmaf(db * grid4(g1, j) * sign_of(signs[l], j), bf(x, 8 * l + j), acc);
                acc = fmaf(db * grid4(g2, j) * sign_of(signs[l], j + 4), bf(x, 8 * l + j + 4),
                           acc);
            }
        }
    }
    return acc;
}

// One output row of one column, reduced across the warp. `weight` is the row's first encoded block.
template <int TYPE>
__device__ __forceinline__ float gemv_row(const std::uint8_t* weight, const __nv_bfloat16* x, int k,
                                          int groups_per_block, int type_bytes) {
    const int lane           = threadIdx.x & 31;
    const int groups_per_row = k / 32;
    float acc                = 0.0f;
    for (int group = lane; group < groups_per_row; group += 32) {
        const std::uint8_t* block_ptr =
            weight + std::size_t(group / groups_per_block) * std::size_t(type_bytes);
        acc += dot32<TYPE>(block_ptr, group % groups_per_block, x + std::size_t(group) * 32);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    return acc;
}

} // namespace ninfer::ops::detail::ggml
