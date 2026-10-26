#pragma once

// int8 gate_up for the prefill tile: 16 output rows by 8 token columns per mma, one warp per 8
// columns, four warps per block, 32 token columns per job.
//
// The decode kernel this replaces assumed one token per expert, which the product never produces:
// the grouped path starts at 47 tokens and always tiles its columns at 32 or 64. So the shape that
// matters is the one where the mma's n dimension is full, and there the int8 m16n8k32 form actually
// spends its doubled k throughput.
//
// Two things this has to get right that the decode form did not:
//   * gate and up rows are computed together so the swiglu product can be formed in registers; the
//     bf16 route writes exactly one activation plane per routed slot, not two
//   * the store is [slot][row], the layout the down kernel reads, with the slot being the routed
//     slot index rather than a token index
//
// Naive on purpose: weights are decoded nibble-at-a-time straight out of global memory and the
// activations are read per mma. Correctness first, then staging.
#include "ops/moe/kernels/cuda/groupwise_int/expert.cuh"
#include "ops/common/numeric/math.cuh"

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

// Four consecutive bytes of one token's quantised activation.
__device__ __forceinline__ unsigned a8_pack4(const std::int8_t* row, int k) {
    return (unsigned)(std::uint8_t)row[k] | ((unsigned)(std::uint8_t)row[k + 1] << 8) |
           ((unsigned)(std::uint8_t)row[k + 2] << 16) | ((unsigned)(std::uint8_t)row[k + 3] << 24);
}

// act_i8:    int8 [tokens][k]        quantised activations, one row per token
// act_scale: float [tokens][k/64]    one scale per 64-wide k group
// codes:     uint8 [rows][k/2]       packed 4-bit weights of the whole bank
// wscales:   __half [rows][k/64]     per-(row, group) weight scale of the whole bank
// weights:   Access (Packed / Strided / Indexed) - resolve(expert) yields that expert's codes,
//            scales and row offset
// offsets:   int [experts + 1]       routed-slot range of each expert
// order:     int [total]             which token row each routed slot came from
// out:       bf16 [slots][intermediate]  silu(gate) * up, one plane, slot-major
//
// Cols is the job's token-column tile width, the same 32/64 split the shape policies use: one warp
// covers eight columns, so the block is Cols/8 warps. The body below does not care which it is.
template <int K, int Rows, int Cols, class Access>
__global__ __launch_bounds__(Cols * 4) void grouped_gate_up_i8_tile(
    const std::int8_t* __restrict__ act_i8, const float* __restrict__ act_scale, Access weights,
    const int* __restrict__ offsets, const int* __restrict__ order,
    const int* __restrict__ job_experts, const int* __restrict__ job_columns,
    const GroupedInfo* __restrict__ info, int stride, __nv_bfloat16* __restrict__ out) {
    constexpr int kGateRows = Rows / 2;   // rows of gate (and of up)
    constexpr int kG        = K / 64;
    constexpr int kCols     = Cols;       // token columns a job covers

    if (info->direct) { return; }
    const int job = blockIdx.y;
    if (job >= info->jobs) { return; }

    const int expert = job_experts[job];
    const int begin  = offsets[expert];
    const int count  = offsets[expert + 1] - begin;
    const int column_base = job_columns[job];
    const int cols        = min(count - column_base, kCols);
    if (cols <= 0) { return; }

    const int t    = threadIdx.x;          // 0..127
    const int warp = t >> 5;               // token sub-tile: 8 columns each
    const int lane = t & 31;
    const int r    = lane >> 2;            // mma row inside the 16-row tile
    const int c0   = warp * 8 + 2 * (lane & 3);  // local column pair held by this thread's c0/c1
    const int c1   = c0 + 1;
    const int bc   = warp * 8 + r;         // column this thread's B fragment carries
    const int br   = (lane & 3) * 4;       // k offset inside a 32-wide chunk

    const int r0 = blockIdx.x * 16;        // gate row tile
    if (r0 >= kGateRows) { return; }

    const auto matrix  = weights.template resolve<Rows>(expert);
    const auto* codes  = matrix.codes;
    const auto* wsc    = reinterpret_cast<const __half*>(matrix.scales);
    const int rb       = matrix.row;

    // Rows are clamped onto a live column and then masked at the store, so the fragments never read
    // past the job's slot range.
    const auto* g_lo = codes + (std::int64_t)(rb + r0 + r) * (K / 2);
    const auto* g_hi = codes + (std::int64_t)(rb + r0 + r + 8) * (K / 2);
    const auto* u_lo = codes + (std::int64_t)(rb + kGateRows + r0 + r) * (K / 2);
    const auto* u_hi = codes + (std::int64_t)(rb + kGateRows + r0 + r + 8) * (K / 2);

    const int slot_b = begin + column_base + (bc < cols ? bc : 0);
    const auto* a_b  = act_i8 + (std::int64_t)order[slot_b] * K;
    const int tok0   = order[begin + column_base + (c0 < cols ? c0 : 0)];
    const int tok1   = order[begin + column_base + (c1 < cols ? c1 : 0)];
    const auto* asc0 = act_scale + (std::int64_t)tok0 * kG;
    const auto* asc1 = act_scale + (std::int64_t)tok1 * kG;

    float gate[4] = {0.f, 0.f, 0.f, 0.f};
    float up[4]   = {0.f, 0.f, 0.f, 0.f};

    for (int g = 0; g < kG; ++g) {
        int acc_g[4] = {0, 0, 0, 0};
        int acc_u[4] = {0, 0, 0, 0};
#pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
            const int k0 = g * 64 + sub * 32;
            const unsigned ag0 = i8_pack4(g_lo, k0 + br);
            const unsigned ag1 = i8_pack4(g_hi, k0 + br);
            const unsigned ag2 = i8_pack4(g_lo, k0 + br + 16);
            const unsigned ag3 = i8_pack4(g_hi, k0 + br + 16);
            const unsigned au0 = i8_pack4(u_lo, k0 + br);
            const unsigned au1 = i8_pack4(u_hi, k0 + br);
            const unsigned au2 = i8_pack4(u_lo, k0 + br + 16);
            const unsigned au3 = i8_pack4(u_hi, k0 + br + 16);
            const unsigned b0  = a8_pack4(a_b, k0 + br);
            const unsigned b1  = a8_pack4(a_b, k0 + br + 16);
            mma_s8(acc_g[0], acc_g[1], acc_g[2], acc_g[3], ag0, ag1, ag2, ag3, b0, b1);
            mma_s8(acc_u[0], acc_u[1], acc_u[2], acc_u[3], au0, au1, au2, au3, b0, b1);
        }

        // c0/c1 are row r at the two column-pair positions; c2/c3 are row r + 8. Each column carries
        // its own activation scale because each column is its own token. gate and up rows have
        // different weight scales because they are different rows of the bank.
        const float sg_lo = __half2float(wsc[(std::int64_t)(r0 + r) * kG + g]);
        const float sg_hi = __half2float(wsc[(std::int64_t)(r0 + r + 8) * kG + g]);
        const float su_lo = __half2float(wsc[(std::int64_t)(kGateRows + r0 + r) * kG + g]);
        const float su_hi = __half2float(wsc[(std::int64_t)(kGateRows + r0 + r + 8) * kG + g]);
        const float a0 = asc0[g], a1 = asc1[g];

        gate[0] += (float)acc_g[0] * sg_lo * a0;
        gate[1] += (float)acc_g[1] * sg_lo * a1;
        gate[2] += (float)acc_g[2] * sg_hi * a0;
        gate[3] += (float)acc_g[3] * sg_hi * a1;
        up[0] += (float)acc_u[0] * su_lo * a0;
        up[1] += (float)acc_u[1] * su_lo * a1;
        up[2] += (float)acc_u[2] * su_hi * a0;
        up[3] += (float)acc_u[3] * su_hi * a1;
    }

    const int row0 = r0 + r;
    const int row1 = row0 + 8;
    if (c0 < cols) {
        const std::int64_t s = std::int64_t(begin + column_base + c0) * stride;
        out[s + row0] = __float2bfloat16_rn(common::silu(gate[0]) * up[0]);
        out[s + row1] = __float2bfloat16_rn(common::silu(gate[2]) * up[2]);
    }
    if (c1 < cols) {
        const std::int64_t s = std::int64_t(begin + column_base + c1) * stride;
        out[s + row0] = __float2bfloat16_rn(common::silu(gate[1]) * up[1]);
        out[s + row1] = __float2bfloat16_rn(common::silu(gate[3]) * up[3]);
    }
}

} // namespace ninfer::ops::detail::moe_cuda::groupwise_int

