#pragma once

// Grouped int8 gate_up for the decode shape: one warp per 16 output rows, one token per expert.
//
// Naive on purpose - no shared memory, no ldmatrix, nibble-at-a-time decode. It exists so the
// grouping and the scaling can be verified before any performance work, in the same order the
// single-expert chain was verified. Once the numbers match the bf16 route, the staging and
// ldmatrix structure from grouped_gate_up.cuh can be layered on top without touching the arithmetic.
//
// Shapes are compile-time because the MoE profiles are: this instantiates for the 2048/512/256/8
// expert that the artifact actually uses.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail::moe_cuda::groupwise_int {

__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One nibble in [0,15] -> the signed value the quantiser meant: (n ^ 8) - 8, range [-8, 7].
__device__ __forceinline__ int i8_nib(const std::uint8_t* row_codes, int k) {
    const int byte = row_codes[k >> 1];
    const int n    = (k & 1) ? (byte >> 4) : (byte & 0x0f);
    return (n ^ 8) - 8;
}

__device__ __forceinline__ unsigned i8_pack4(const std::uint8_t* row_codes, int k) {
    unsigned w = 0;
    for (int i = 0; i < 4; ++i) {
        w |= (unsigned)(std::uint8_t)i8_nib(row_codes, k + i) << (8 * i);
    }
    return w;
}

// act_i8:   int8 [tokens][k]        quantised activations
// act_scale: float [tokens][k/64]   one scale per 64-wide k group
// codes:    uint8 [experts][rows][k/2]   packed 4-bit weights, expert-major
// wscales:  __half [experts][rows][k/64] per-(row, group) weight scale
// offsets:  int [experts + 1]       token range of each expert
// order:    int [total]             which token row each routed slot came from
// out:      bf16 [total][rows]      gate/up result, scaled
//
// Row order inside `out` matches the bf16 kernel: the first half of the rows is `gate`, the
// second half `up`, both indexed by the routed slot.
template <int K, int Rows>
__global__ void grouped_gate_up_i8(const std::int8_t* __restrict__ act_i8,
                                   const float* __restrict__ act_scale,
                                   const std::uint8_t* __restrict__ codes,
                                   const __half* __restrict__ wscales,
                                   const int* __restrict__ offsets, const int* __restrict__ order,
                                   int experts, __nv_bfloat16* __restrict__ out) {
    constexpr int kG = K / 64;
    const int t      = threadIdx.x;   // one warp
    const int expert = blockIdx.y;
    const int row0   = blockIdx.x * 16;
    if (expert >= experts) { return; }

    const int begin = offsets[expert];
    const int count = offsets[expert + 1] - begin;
    if (count <= 0) { return; }
    // Decode shape: one token per expert. Anything wider belongs to the prefill path.
    const int tok       = order[begin];
    const auto* a8      = act_i8 + (std::int64_t)tok * K;
    const auto* asc     = act_scale + (std::int64_t)tok * kG;
    const auto* w_base  = codes + (std::int64_t)expert * Rows * (K / 2);
    const auto* s_base  = wscales + (std::int64_t)expert * Rows * kG;
    auto* dst           = out + (std::int64_t)begin * Rows;

    const int r  = t / 4;
    const int c4 = (t % 4) * 4;
    if (row0 + r >= Rows) { return; }

    float accf[4] = {0.f, 0.f, 0.f, 0.f};
    for (int g = 0; g < kG; ++g) {
        int acc[4] = {0, 0, 0, 0};
        for (int sub = 0; sub < 2; ++sub) {
            const int k0     = g * 64 + sub * 32;
            const auto* c_lo = w_base + (std::int64_t)(row0 + r) * (K / 2);
            const auto* c_hi = w_base + (std::int64_t)(row0 + r + 8) * (K / 2);
            const unsigned a0 = i8_pack4(c_lo, k0 + c4);
            const unsigned a1 = i8_pack4(c_hi, k0 + c4);
            const unsigned a2 = i8_pack4(c_lo, k0 + c4 + 16);
            const unsigned a3 = i8_pack4(c_hi, k0 + c4 + 16);

            unsigned b0 = 0, b1 = 0;
            for (int i = 0; i < 4; ++i) {
                b0 |= (unsigned)(std::uint8_t)a8[k0 + c4 + i] << (8 * i);
                b1 |= (unsigned)(std::uint8_t)a8[k0 + c4 + 16 + i] << (8 * i);
            }
            mma_s8(acc[0], acc[1], acc[2], acc[3], a0, a1, a2, a3, b0, b1);
        }
        // c0/c1 belong to row r, c2/c3 to row r+8: the two rows carry different weight scales.
        const float s_lo = asc[g] * __half2float(s_base[(std::int64_t)(row0 + r) * kG + g]);
        const float s_hi = asc[g] * __half2float(s_base[(std::int64_t)(row0 + r + 8) * kG + g]);
        accf[0] += (float)acc[0] * s_lo;
        accf[1] += (float)acc[1] * s_lo;
        accf[2] += (float)acc[2] * s_hi;
        accf[3] += (float)acc[3] * s_hi;
    }
    if ((t % 4) == 0) {
        dst[(std::int64_t)(row0 + r) * 1 + 0]           = __float2bfloat16_rn(accf[0]);
        dst[(std::int64_t)(row0 + r + 8) * 1 + 0]       = __float2bfloat16_rn(accf[2]);
    }
}

} // namespace ninfer::ops::detail::moe_cuda::groupwise_int
