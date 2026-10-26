#pragma once

// Shared-memory staging for the grouped int8 gate_up, without ldmatrix.
//
// The naive kernel reads both operands straight from global memory inside the mma loop, so the int8
// mma's 2x k-throughput is spent waiting on loads. This version stages one 64-wide k tile at a time
// through shared memory - weights as packed 4-bit, decoded in place to int8 - and runs the mma out
// of shared. The fragment layout is unchanged; only where the bytes come from changes.
//
// Kept as a separate kernel rather than replacing the naive one: the two must produce the same
// numbers, and having both lets the test assert exactly that.
#include "ops/moe/kernels/cuda/groupwise_int/grouped_gate_up_i8.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail::moe_cuda::groupwise_int {
namespace {

constexpr int kSmemBK        = 64;                          // k tile staged per iteration
constexpr int kSmemRows      = 16;                          // one mma m-tile
constexpr int kSmemCodeBytes = kSmemRows * (kSmemBK / 2);   // packed 4-bit for the tile

// Four consecutive int8 out of an unpacked tile.
__device__ __forceinline__ unsigned pack4_i8(const std::int8_t* p) {
    return (unsigned)(std::uint8_t)p[0] | ((unsigned)(std::uint8_t)p[1] << 8) |
           ((unsigned)(std::uint8_t)p[2] << 16) | ((unsigned)(std::uint8_t)p[3] << 24);
}

} // namespace

template <int K, int Rows, class Access>
__global__ void grouped_gate_up_i8_smem(const std::int8_t* __restrict__ act_i8,
                                        const float* __restrict__ act_scale, Access weights,
                                        const int* __restrict__ offsets,
                                        const int* __restrict__ order, int experts, int stride,
                                        __nv_bfloat16* __restrict__ out) {
    constexpr int kG     = K / 64;
    constexpr int kTiles = K / kSmemBK;

    __shared__ __align__(16) std::uint8_t Cr[kSmemCodeBytes];
    __shared__ __align__(16) std::int8_t As[kSmemRows * kSmemBK];
    __shared__ __align__(16) std::int8_t Bs[kSmemBK];

    const int t      = threadIdx.x;
    const int expert = blockIdx.y;
    const int row0   = blockIdx.x * kSmemRows;
    if (expert >= experts) { return; }

    const int begin = offsets[expert];
    const int count = offsets[expert + 1] - begin;
    if (count <= 0) { return; }

    const int tok      = order[begin];
    const auto* a8     = act_i8 + (std::int64_t)tok * K;
    const auto* asc    = act_scale + (std::int64_t)tok * kG;
    const auto matrix  = weights.template resolve<Rows>(expert);
    const auto* w_base = matrix.codes;
    const auto* s_base = reinterpret_cast<const __half*>(matrix.scales);
    const int rb       = matrix.row;

    const int r  = t / 4;
    const int c4 = (t % 4) * 4;

    float accf[4] = {0.f, 0.f, 0.f, 0.f};

    for (int kt = 0; kt < kTiles; ++kt) {
        const int k0 = kt * kSmemBK;

        {
            const int row  = t / 2;
            const int half = t % 2;
            const auto* src =
                w_base + (std::int64_t)(rb + row0 + row) * (K / 2) + k0 / 2 + half * 16;
            auto* dst = reinterpret_cast<uint4*>(&Cr[row * (kSmemBK / 2) + half * 16]);
            *dst      = *reinterpret_cast<const uint4*>(src);
        }
        Bs[t * 2]     = a8[k0 + t * 2];
        Bs[t * 2 + 1] = a8[k0 + t * 2 + 1];

        __syncwarp();

        for (int i = t; i < kSmemRows * kSmemBK; i += 32) {
            const int row = i / kSmemBK;
            const int k   = i - row * kSmemBK;
            As[i]         = static_cast<std::int8_t>(i8_nib(&Cr[row * (kSmemBK / 2)], k));
        }

        __syncwarp();

        int acc[4] = {0, 0, 0, 0};
#pragma unroll
        for (int sub = 0; sub < kSmemBK / 32; ++sub) {
            const int kk      = sub * 32;
            const unsigned a0 = pack4_i8(&As[r * kSmemBK + kk + c4]);
            const unsigned a1 = pack4_i8(&As[(r + 8) * kSmemBK + kk + c4]);
            const unsigned a2 = pack4_i8(&As[r * kSmemBK + kk + c4 + 16]);
            const unsigned a3 = pack4_i8(&As[(r + 8) * kSmemBK + kk + c4 + 16]);
            const unsigned b0 = pack4_i8(&Bs[kk + c4]);
            const unsigned b1 = pack4_i8(&Bs[kk + c4 + 16]);
            mma_s8(acc[0], acc[1], acc[2], acc[3], a0, a1, a2, a3, b0, b1);
        }

        const float s_lo = asc[kt] * __half2float(s_base[(std::int64_t)(row0 + r) * kG + kt]);
        const float s_hi = asc[kt] * __half2float(s_base[(std::int64_t)(row0 + r + 8) * kG + kt]);
        accf[0] += (float)acc[0] * s_lo;
        accf[1] += (float)acc[1] * s_lo;
        accf[2] += (float)acc[2] * s_hi;
        accf[3] += (float)acc[3] * s_hi;

        __syncwarp();
    }

    if ((t % 4) == 0) {
        out[(std::int64_t)(row0 + r) * stride + begin]     = __float2bfloat16_rn(accf[0]);
        out[(std::int64_t)(row0 + r + 8) * stride + begin] = __float2bfloat16_rn(accf[2]);
    }
}

} // namespace ninfer::ops::detail::moe_cuda::groupwise_int
