#include "ninfer/ops/w4a8.h"

#include "core/device.h"
#include "ops/common/math.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstdlib>

namespace ninfer::ops {
namespace {

constexpr int kBM = 128, kBN = 128, kBK = 64, kThreads = 512;
constexpr int kMT = 2, kNT = 4;
constexpr int kNU16 = kBK / 16;
constexpr int kStagesDequant = 3; // pipeline depth of the separate-dequant path
constexpr int kStagesQ5      = 2; // pipeline depth of the inline q5-decode path
constexpr int kCodeBytes     = 32; // Q5 group: 32 packed nibble bytes
constexpr int kHighBytes     = 8;  // Q5 group: 8 high-bit bytes

__device__ __forceinline__ int swz(int c16, int row) { return c16 ^ (row & (kNU16 - 1)); }
__device__ __forceinline__ int phys_off(int r, int c) { return swz(c / 16, r) * 16 + (c % 16); }

__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                       unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldm_x2(unsigned& r0, unsigned& r1, unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];\n"
                 : "=r"(r0), "=r"(r1) : "r"(a));
}
__device__ __forceinline__ unsigned su(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cpa16(void* d, const void* s) {
    asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n" ::"r"(su(d)), "l"(s));
}
__device__ __forceinline__ void cpa8(void* d, const void* s) {
    asm volatile("cp.async.ca.shared.global [%0],[%1],8;\n" ::"r"(su(d)), "l"(s));
}
__device__ __forceinline__ void cpc() { asm volatile("cp.async.commit_group;\n"); }
template <int G> __device__ __forceinline__ void cpw() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(G));
}

// q5_g64 packed (4-bit nibbles + 1 high bit per element) -> int8. Values -16..15 fit int8 exactly.
// One thread decodes a 16-element quarter of one group (8 packed code bytes + 2 high-bit
// bytes -> one 16-byte vector store), so the reads and writes are vectorized and no
// per-element 64-bit division is needed (the row comes from blockIdx.y).
__global__ void dequant_q5_i8(const std::uint8_t* __restrict__ codes,
                              const std::uint8_t* __restrict__ high, std::int8_t* __restrict__ out,
                              int groups, int k) {
    const int task = blockIdx.x * blockDim.x + threadIdx.x;
    if (task >= groups * 4) return;
    const long row = blockIdx.y;
    const int group = task >> 2;
    const int q     = task & 3;
    const std::uint8_t* cw = codes + (row * groups + group) * 32;
    const std::uint8_t* hw = high + (row * groups + group) * 8;
    const uint2 c2 = *reinterpret_cast<const uint2*>(cw + q * 8);
    const std::uint16_t h2 = *reinterpret_cast<const std::uint16_t*>(hw + q * 2);
    const std::uint8_t* cb = reinterpret_cast<const std::uint8_t*>(&c2);
    const std::uint8_t* hb = reinterpret_cast<const std::uint8_t*>(&h2);
    alignas(16) std::int8_t vals[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int half = j >> 1;
        const std::uint8_t code = cb[half];
        const std::uint8_t hi   = hb[half >> 2];
        const int sh            = (half & 3) * 2;
        int v = (j & 1) ? ((code >> 4) | (((hi >> (sh + 1)) & 1) << 4))
                        : ((code & 0x0F) | (((hi >> sh) & 1) << 4));
        if (v >= 16) v -= 32;
        vals[j] = (std::int8_t)v;
    }
    *reinterpret_cast<int4*>(out + row * k + (long)group * 64 + q * 16) =
        *reinterpret_cast<const int4*>(vals);
}

__global__ void quant_a8(const __nv_bfloat16* __restrict__ x, std::int8_t* __restrict__ out,
                         float* __restrict__ scale, int k, int tokens) {
    const int row = blockIdx.x;
    if (row >= tokens) return;
    const __nv_bfloat16* xr = x + (long)row * k;
    float amax = 0.0f;
    for (int i = threadIdx.x; i < k; i += blockDim.x) amax = fmaxf(amax, fabsf(__bfloat162float(xr[i])));
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    __shared__ float warp_max[32];
    __shared__ float s;
    if ((threadIdx.x & 31) == 0) warp_max[threadIdx.x >> 5] = amax;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = 0.0f;
        const int warps = (blockDim.x + 31) >> 5;
        for (int i = 0; i < warps; ++i) m = fmaxf(m, warp_max[i]);
        s = m > 0.0f ? m / 127.0f : 0.0f;
        scale[row] = s;
    }
    __syncthreads();
    const float inv = s > 0.0f ? 1.0f / s : 0.0f;
    std::int8_t* orow = out + (long)row * k;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        int q = (int)lrintf(__bfloat162float(xr[i]) * inv);
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
        orow[i] = (std::int8_t)q;
    }
}

// ---------------------------------------------------------------------------
// Path A: weights pre-dequantized to an int8 matrix by dequant_q5_i8.
// ---------------------------------------------------------------------------
template <int kN, int kK>
__global__ __launch_bounds__(kThreads) void w4a8_gemm_residual_kernel(
    const std::int8_t* __restrict__ A, const std::int8_t* __restrict__ W,
    const float* __restrict__ asc, const __half* __restrict__ wsc, __nv_bfloat16* __restrict__ out,
    int ld) {
    constexpr int kGroups = kK / 64, kTilesK = kK / kBK, kGPK = 64 / kBK;
    __shared__ __align__(16) std::int8_t As[kStagesDequant][kBM][kBK];
    __shared__ __align__(16) std::int8_t Ws[kStagesDequant][kBN][kBK];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wt_m = warp / 4, wt_n = warp % 4;
    const int m0 = blockIdx.x * kBM, n0 = blockIdx.y * kBN;

    auto load = [&](int st, int kt) {
        const int r = (tid * 16) / kBK, c = (tid * 16) % kBK;
        const int off = phys_off(r, c);
        cpa16(&As[st][r][off], A + (long)(m0 + r) * kK + kt * kBK + c);
        cpa16(&Ws[st][r][off], W + (long)(n0 + r) * kK + kt * kBK + c);
        cpc();
    };
    // Single-barrier software pipeline. The load for tile kt+kPrefetch lands in
    // buffer (kt+kPrefetch)%kStages == (kt-1)%kStages and is issued after the
    // barrier but before the compute of buffer kt, so the one full-block barrier
    // per k-tile already orders both the staged data and the previous tile's reads.
    constexpr int kPrefetch = kStagesDequant - 1;
#pragma unroll
    for (int s = 0; s < kPrefetch; ++s) load(s, s);

    float accf[kMT][kNT][4];
    int acci[kMT][kNT][4];
#pragma unroll
    for (int a = 0; a < kMT; ++a)
        for (int b = 0; b < kNT; ++b)
            for (int r = 0; r < 4; ++r) { accf[a][b][r] = 0.0f; acci[a][b][r] = 0; }

#pragma unroll 1
    for (int kt = 0; kt < kTilesK; ++kt) {
        const int st = kt % kStagesDequant;
        if (kt + kPrefetch <= kTilesK) cpw<kPrefetch - 1>(); else cpw<0>();
        __syncthreads();
        if (kt + kPrefetch < kTilesK) load((kt + kPrefetch) % kStagesDequant, kt + kPrefetch);
#pragma unroll
        for (int kk = 0; kk < kBK / 32; ++kk) {
            unsigned bf[kNT][2];
            // One ldmatrix.x4 fetches the B fragments of two adjacent 8-wide n tiles:
            // r0/r2 -> tile np*2, r1/r3 -> tile np*2+1 (row selects the n half,
            // lane>>4 selects the 16-byte k chunk).
#pragma unroll
            for (int np = 0; np < kNT / 2; ++np) {
                const int cbase = wt_n * 32 + np * 16;
                const int brow  = cbase + (lane & 7) + (((lane >> 3) & 1) << 3);
                unsigned r0, r1, r2, r3;
                ldm_x4(r0, r1, r2, r3, su(&Ws[st][brow][swz(kk * 2 + (lane >> 4), brow) * 16]));
                bf[np * 2][0]     = r0;
                bf[np * 2][1]     = r2;
                bf[np * 2 + 1][0] = r1;
                bf[np * 2 + 1][1] = r3;
            }
#pragma unroll
            for (int mt = 0; mt < kMT; ++mt) {
                unsigned a0, a1, a2, a3;
                const int row = wt_m * 32 + mt * 16 + (lane & 15);
                ldm_x4(a0, a1, a2, a3, su(&As[st][row][swz(kk * 2 + (lane >> 4), row) * 16]));
#pragma unroll
                for (int nt = 0; nt < kNT; ++nt)
                    mma_s8(acci[mt][nt][0], acci[mt][nt][1], acci[mt][nt][2], acci[mt][nt][3], a0,
                           a1, a2, a3, bf[nt][0], bf[nt][1]);
            }
        }
        if ((kt % kGPK) == (kGPK - 1)) {
            const int g = kt / kGPK;
#pragma unroll
            for (int mt = 0; mt < kMT; ++mt) {
                const int row = m0 + wt_m * 32 + mt * 16 + (lane >> 2);
                const float as0 = asc[row];
                const float as1 = asc[row + 8];
#pragma unroll
                for (int nt = 0; nt < kNT; ++nt) {
                    const int col = n0 + wt_n * 32 + nt * 8 + (lane & 3) * 2;
                    const float w0 = __half2float(wsc[(long)col * kGroups + g]);
                    const float w1 = __half2float(wsc[(long)(col + 1) * kGroups + g]);
                    accf[mt][nt][0] += (float)acci[mt][nt][0] * as0 * w0;
                    accf[mt][nt][1] += (float)acci[mt][nt][1] * as0 * w1;
                    accf[mt][nt][2] += (float)acci[mt][nt][2] * as1 * w0;
                    accf[mt][nt][3] += (float)acci[mt][nt][3] * as1 * w1;
                    acci[mt][nt][0] = acci[mt][nt][1] = acci[mt][nt][2] = acci[mt][nt][3] = 0;
                }
            }
        }
    }
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt)
        for (int nt = 0; nt < kNT; ++nt) {
            const int row = m0 + wt_m * 32 + mt * 16 + (lane >> 2);
            const int col = n0 + wt_n * 32 + nt * 8 + (lane & 3) * 2;
            __nv_bfloat16* d0 = out + (long)row * ld + col;
            __nv_bfloat16* d1 = out + (long)(row + 8) * ld + col;
            d0[0] = __float2bfloat16_rn(__bfloat162float(d0[0]) + accf[mt][nt][0]);
            d0[1] = __float2bfloat16_rn(__bfloat162float(d0[1]) + accf[mt][nt][1]);
            d1[0] = __float2bfloat16_rn(__bfloat162float(d1[0]) + accf[mt][nt][2]);
            d1[1] = __float2bfloat16_rn(__bfloat162float(d1[1]) + accf[mt][nt][3]);
        }
}

// ---------------------------------------------------------------------------
// Path A-narrow: identical math and an identical 32x32 warp tile, but the block
// tile is halved to 128x64 with 8 warps so __launch_bounds__(256, 2) fits the
// 64K register file (110 * 256 * 2 = 56320 <= 65536) and the SM runs TWO
// independent blocks. The other block keeps issuing while this one sits in its
// per-k-tile barrier - the barrier hiding a single 128x128 block (16 warps
// rendezvous per k-tile, 1 block/SM) cannot get. Warp tile, fragment shape,
// LDSM/HMMA ratio and per-thread register demand are unchanged, so this
// isolates the barrier cost.
// ---------------------------------------------------------------------------
constexpr int kBMN = 128, kBNN = 64, kThreadsN = 256;
constexpr int kMTN = 2, kNTN = 4; // warp tile 32 rows x 32 cols (as in path A)
constexpr int kStagesN = 4;

template <int kN, int kK>
__global__ __launch_bounds__(kThreadsN, 2) void w4a8_gemm_residual_narrow_kernel(
    const std::int8_t* __restrict__ A, const std::int8_t* __restrict__ W,
    const float* __restrict__ asc, const __half* __restrict__ wsc, __nv_bfloat16* __restrict__ out,
    int ld) {
    constexpr int kGroups = kK / 64, kTilesK = kK / kBK, kGPK = 64 / kBK;
    __shared__ __align__(16) std::int8_t As[kStagesN][kBMN][kBK];
    __shared__ __align__(16) std::int8_t Ws[kStagesN][kBNN][kBK];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wt_m = warp / 2, wt_n = warp % 2;
    const int m0 = blockIdx.x * kBMN, n0 = blockIdx.y * kBNN;

    auto load = [&](int st, int kt) {
#pragma unroll
        for (int j = 0; j < (kBMN * (kBK / 16)) / kThreadsN; ++j) {
            const int item = tid + j * kThreadsN;
            const int r = item >> 2, c16 = item & 3;
            cpa16(&As[st][r][(c16 ^ (r & 3)) * 16], A + (long)(m0 + r) * kK + kt * kBK + c16 * 16);
        }
        {
            const int item = tid; // kBNN * (kBK/16) == kThreadsN
            const int r = item >> 2, c16 = item & 3;
            cpa16(&Ws[st][r][(c16 ^ (r & 3)) * 16], W + (long)(n0 + r) * kK + kt * kBK + c16 * 16);
        }
        cpc();
    };

    constexpr int kPrefetch = kStagesN - 1;
#pragma unroll
    for (int s = 0; s < kPrefetch; ++s) load(s, s);

    float accf[kMTN][kNTN][4];
    int acci[kMTN][kNTN][4];
#pragma unroll
    for (int a = 0; a < kMTN; ++a)
        for (int b = 0; b < kNTN; ++b)
            for (int r = 0; r < 4; ++r) { accf[a][b][r] = 0.0f; acci[a][b][r] = 0; }

#pragma unroll 1
    for (int kt = 0; kt < kTilesK; ++kt) {
        const int st = kt % kStagesN;
        if (kt + kPrefetch <= kTilesK) cpw<kPrefetch - 1>(); else cpw<0>();
        __syncthreads();
        if (kt + kPrefetch < kTilesK) load((kt + kPrefetch) % kStagesN, kt + kPrefetch);
#pragma unroll
        for (int kk = 0; kk < kBK / 32; ++kk) {
            unsigned bf[kNTN][2];
#pragma unroll
            for (int np = 0; np < kNTN / 2; ++np) {
                const int cbase = wt_n * 32 + np * 16;
                const int brow  = cbase + (lane & 7) + (((lane >> 3) & 1) << 3);
                unsigned r0, r1, r2, r3;
                ldm_x4(r0, r1, r2, r3, su(&Ws[st][brow][swz(kk * 2 + (lane >> 4), brow) * 16]));
                bf[np * 2][0]     = r0;
                bf[np * 2][1]     = r2;
                bf[np * 2 + 1][0] = r1;
                bf[np * 2 + 1][1] = r3;
            }
#pragma unroll
            for (int mt = 0; mt < kMTN; ++mt) {
                unsigned a0, a1, a2, a3;
                const int row = wt_m * 32 + mt * 16 + (lane & 15);
                ldm_x4(a0, a1, a2, a3, su(&As[st][row][swz(kk * 2 + (lane >> 4), row) * 16]));
#pragma unroll
                for (int nt = 0; nt < kNTN; ++nt)
                    mma_s8(acci[mt][nt][0], acci[mt][nt][1], acci[mt][nt][2], acci[mt][nt][3], a0,
                           a1, a2, a3, bf[nt][0], bf[nt][1]);
            }
        }
        if ((kt % kGPK) == (kGPK - 1)) {
            const int g = kt / kGPK;
#pragma unroll
            for (int mt = 0; mt < kMTN; ++mt) {
                const int row = m0 + wt_m * 32 + mt * 16 + (lane >> 2);
                const float as0 = asc[row];
                const float as1 = asc[row + 8];
#pragma unroll
                for (int nt = 0; nt < kNTN; ++nt) {
                    const int col = n0 + wt_n * 32 + nt * 8 + (lane & 3) * 2;
                    const float w0 = __half2float(wsc[(long)col * kGroups + g]);
                    const float w1 = __half2float(wsc[(long)(col + 1) * kGroups + g]);
                    accf[mt][nt][0] += (float)acci[mt][nt][0] * as0 * w0;
                    accf[mt][nt][1] += (float)acci[mt][nt][1] * as0 * w1;
                    accf[mt][nt][2] += (float)acci[mt][nt][2] * as1 * w0;
                    accf[mt][nt][3] += (float)acci[mt][nt][3] * as1 * w1;
                    acci[mt][nt][0] = acci[mt][nt][1] = acci[mt][nt][2] = acci[mt][nt][3] = 0;
                }
            }
        }
    }
#pragma unroll
    for (int mt = 0; mt < kMTN; ++mt)
        for (int nt = 0; nt < kNTN; ++nt) {
            const int row = m0 + wt_m * 32 + mt * 16 + (lane >> 2);
            const int col = n0 + wt_n * 32 + nt * 8 + (lane & 3) * 2;
            __nv_bfloat16* d0 = out + (long)row * ld + col;
            __nv_bfloat16* d1 = out + (long)(row + 8) * ld + col;
            d0[0] = __float2bfloat16_rn(__bfloat162float(d0[0]) + accf[mt][nt][0]);
            d0[1] = __float2bfloat16_rn(__bfloat162float(d0[1]) + accf[mt][nt][1]);
            d1[0] = __float2bfloat16_rn(__bfloat162float(d1[0]) + accf[mt][nt][2]);
            d1[1] = __float2bfloat16_rn(__bfloat162float(d1[1]) + accf[mt][nt][3]);
        }
}

// ---------------------------------------------------------------------------
// Path B: weights streamed as packed q5 and decoded to int8 inside the GEMM.
// The packed codes/high bits are staged with cp.async (5-bit/element read from
// DRAM instead of 8-bit), decoded in registers, and written to a swizzled int8
// tile for the m16n8k32 s8 MMA. No separate dequantization pass is required.
// ---------------------------------------------------------------------------
struct W4A8Q5Smem {
    alignas(16) std::int8_t As[kStagesQ5][kBM][kBK];
    alignas(16) std::int8_t Ws[kStagesQ5][kBN][kBK];
    alignas(16) std::uint8_t Cw[kStagesQ5][kBN][kCodeBytes];
    alignas(16) std::uint8_t Hw[kStagesQ5][kBN][kHighBytes];
};

template <int kN, int kK>
__global__ __launch_bounds__(kThreads) void w4a8_q5_gemm_residual_kernel(
    const std::int8_t* __restrict__ A, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const float* __restrict__ asc,
    const __half* __restrict__ wsc, __nv_bfloat16* __restrict__ out, int ld) {
    constexpr int kGroups = kK / 64, kTilesK = kK / kBK, kGPK = 64 / kBK;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    W4A8Q5Smem& sm = *reinterpret_cast<W4A8Q5Smem*>(smem_raw);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wt_m = warp / 4, wt_n = warp % 4;
    const int m0 = blockIdx.x * kBM, n0 = blockIdx.y * kBN;

    auto load = [&](int st, int kt) {
        const int r = (tid * 16) / kBK, c = (tid * 16) % kBK;
        cpa16(&sm.As[st][r][phys_off(r, c)], A + (long)(m0 + r) * kK + kt * kBK + c);
        if (tid < kBN * 2) {
            const int wr = tid >> 1, wh = tid & 1;
            cpa16(&sm.Cw[st][wr][wh * 16],
                  codes + (long)(n0 + wr) * kGroups * kCodeBytes + kt * kCodeBytes + wh * 16);
        }
        if (tid < kBN) {
            cpa8(&sm.Hw[st][tid][0],
                 high + (long)(n0 + tid) * kGroups * kHighBytes + kt * kHighBytes);
        }
        cpc();
    };

    // One thread owns four 16-element quarters of one weight row: 8 packed code
    // bytes + 2 high-bit bytes -> 16 int8, stored 16-byte contiguous (swizzled).
    auto decode_w = [&](int st) {
        const int r = tid >> 2, q = tid & 3;
        const uint2 c2 = *reinterpret_cast<const uint2*>(&sm.Cw[st][r][q * 8]);
        const std::uint16_t h2 = *reinterpret_cast<const std::uint16_t*>(&sm.Hw[st][r][q * 2]);
        const std::uint8_t* cb = reinterpret_cast<const std::uint8_t*>(&c2);
        const std::uint8_t* hb = reinterpret_cast<const std::uint8_t*>(&h2);
        alignas(16) std::int8_t vals[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int half = j >> 1;
            const std::uint8_t code = cb[half];
            const std::uint8_t hi   = hb[half >> 2];
            const int sh            = (half & 3) * 2;
            int v = (j & 1) ? ((code >> 4) | (((hi >> (sh + 1)) & 1) << 4))
                            : ((code & 0x0F) | (((hi >> sh) & 1) << 4));
            if (v >= 16) v -= 32;
            vals[j] = (std::int8_t)v;
        }
        *reinterpret_cast<int4*>(&sm.Ws[st][r][(q ^ (r & 3)) * 16]) =
            *reinterpret_cast<const int4*>(vals);
    };
#pragma unroll
    for (int s = 0; s < kStagesQ5; ++s) load(s, s);

    float accf[kMT][kNT][4];
    int acci[kMT][kNT][4];
#pragma unroll
    for (int a = 0; a < kMT; ++a)
        for (int b = 0; b < kNT; ++b)
            for (int r = 0; r < 4; ++r) { accf[a][b][r] = 0.0f; acci[a][b][r] = 0; }

#pragma unroll 1
    for (int kt = 0; kt < kTilesK; ++kt) {
        const int st = kt % kStagesQ5;
        if (kt + kStagesQ5 <= kTilesK) cpw<kStagesQ5 - 1>(); else cpw<0>();
        __syncthreads();
        decode_w(st);
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < kBK / 32; ++kk) {
            unsigned bf[kNT][2];
#pragma unroll
            for (int nt = 0; nt < kNT; ++nt) {
                const int cn = wt_n * 32 + nt * 8 + (lane & 7);
                ldm_x2(bf[nt][0], bf[nt][1], su(&sm.Ws[st][cn][swz(kk * 2 + ((lane >> 3) & 1), cn) * 16]));
            }
#pragma unroll
            for (int mt = 0; mt < kMT; ++mt) {
                unsigned a0, a1, a2, a3;
                const int row = wt_m * 32 + mt * 16 + (lane & 15);
                ldm_x4(a0, a1, a2, a3, su(&sm.As[st][row][swz(kk * 2 + (lane >> 4), row) * 16]));
#pragma unroll
                for (int nt = 0; nt < kNT; ++nt)
                    mma_s8(acci[mt][nt][0], acci[mt][nt][1], acci[mt][nt][2], acci[mt][nt][3], a0,
                           a1, a2, a3, bf[nt][0], bf[nt][1]);
            }
        }
        __syncthreads();
        if (kt + kStagesQ5 < kTilesK) load(st, kt + kStagesQ5);
        if ((kt % kGPK) == (kGPK - 1)) {
            const int g = kt / kGPK;
#pragma unroll
            for (int mt = 0; mt < kMT; ++mt) {
                const int row = m0 + wt_m * 32 + mt * 16 + (lane >> 2);
                const float as0 = asc[row];
                const float as1 = asc[row + 8];
#pragma unroll
                for (int nt = 0; nt < kNT; ++nt) {
                    const int col = n0 + wt_n * 32 + nt * 8 + (lane & 3) * 2;
                    const float w0 = __half2float(wsc[(long)col * kGroups + g]);
                    const float w1 = __half2float(wsc[(long)(col + 1) * kGroups + g]);
                    accf[mt][nt][0] += (float)acci[mt][nt][0] * as0 * w0;
                    accf[mt][nt][1] += (float)acci[mt][nt][1] * as0 * w1;
                    accf[mt][nt][2] += (float)acci[mt][nt][2] * as1 * w0;
                    accf[mt][nt][3] += (float)acci[mt][nt][3] * as1 * w1;
                    acci[mt][nt][0] = acci[mt][nt][1] = acci[mt][nt][2] = acci[mt][nt][3] = 0;
                }
            }
        }
    }
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt)
        for (int nt = 0; nt < kNT; ++nt) {
            const int row = m0 + wt_m * 32 + mt * 16 + (lane >> 2);
            const int col = n0 + wt_n * 32 + nt * 8 + (lane & 3) * 2;
            __nv_bfloat16* d0 = out + (long)row * ld + col;
            __nv_bfloat16* d1 = out + (long)(row + 8) * ld + col;
            d0[0] = __float2bfloat16_rn(__bfloat162float(d0[0]) + accf[mt][nt][0]);
            d0[1] = __float2bfloat16_rn(__bfloat162float(d0[1]) + accf[mt][nt][1]);
            d1[0] = __float2bfloat16_rn(__bfloat162float(d1[0]) + accf[mt][nt][2]);
            d1[1] = __float2bfloat16_rn(__bfloat162float(d1[1]) + accf[mt][nt][3]);
        }
}

// ---------------------------------------------------------------------------
// Route B: W4A8 for the folded gate/up projection (SwiGLU).
// The dequantized weight is written in the same folded row order the bf16 gate/up
// kernel uses: a block of 32 parents stores its 32 gate rows followed by the 32
// matching up rows, so the warp's m16 tiles pair gate and up inside one
// accumulator array (acc[mi] gate, acc[mi + MT/2] up) and the epilogue is a local
// silu(gate) * up. int8 m16n8k32 mma with per-(original row, group) weight scales
// and per-token activation scales, folded into fp32 once per 64-K group.
//
// kParents = 17408, kK = 5120, tokens % kSGBN == 0.
// ---------------------------------------------------------------------------
constexpr int kSGBM      = 64, kSGBN = 128, kSGBK = 64;
constexpr int kSGThreads = 256;              // 8 warps: 2 row x 4 token
constexpr int kSGMT      = 2, kSGNT = 4;     // warp tile 32 rows x 32 tokens
constexpr int kSGStages  = 4;
constexpr int kSGPM      = kSGBM / 2;        // 32 parents per block

// q4 (int4 nibble + per-64-group fp16 scale) -> int8, written in folded row order.
__global__ void dequant_q4_i8_folded(const std::uint8_t* __restrict__ codes,
                                     std::int8_t* __restrict__ out, int groups, int k,
                                     int parents) {
    const int task = blockIdx.x * blockDim.x + threadIdx.x;
    if (task >= groups * 4) return;
    const int R     = blockIdx.y; // folded row
    const int local = R & 63;
    const int b     = R >> 6;
    const int grow  = (local < kSGPM) ? (b * kSGPM + local)
                                      : (parents + b * kSGPM + (local - kSGPM));
    const int group = task >> 2, q = task & 3;
    const std::uint8_t* cw = codes + ((long)grow * groups + group) * 32;
    const uint2 c2 = *reinterpret_cast<const uint2*>(cw + q * 8);
    const std::uint8_t* cb = reinterpret_cast<const std::uint8_t*>(&c2);
    alignas(16) std::int8_t vals[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const std::uint8_t byte = cb[j >> 1];
        const int nib = (j & 1) ? (int)(byte >> 4) : (int)(byte & 0x0F);
        vals[j] = (std::int8_t)((nib ^ 0x08) - 0x08);
    }
    *reinterpret_cast<int4*>(out + (long)R * k + (long)group * 64 + q * 16) =
        *reinterpret_cast<const int4*>(vals);
}

template <int kParents, int kK>
__global__ __launch_bounds__(kSGThreads, 2) void w4a8_swiglu_kernel(
    const std::int8_t* __restrict__ Wfold, const std::int8_t* __restrict__ Bq,
    const float* __restrict__ asc, const __half* __restrict__ wsc, __nv_bfloat16* __restrict__ out,
    int tokens) {
    constexpr int kGroups = kK / 64, kTilesK = kK / 64, kGPK = 64 / kSGBK;
    __shared__ __align__(16) std::int8_t As[kSGStages][kSGBM][kSGBK];
    __shared__ __align__(16) std::int8_t Bs[kSGStages][kSGBN][kSGBK];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp / 4, wn = warp % 4;
    const int half_tile = wm * (kSGMT / 2);
    const int gid = lane >> 2, lid = lane & 3;
    const int m0 = blockIdx.y * kSGPM; // parent base
    const int t0 = blockIdx.x * kSGBN; // token base

    auto grow = [&](int L) { return (L < kSGPM) ? (m0 + L) : (kParents + m0 + L - kSGPM); };

    auto load = [&](int st, int kt) {
        {
            const int item = tid; // kSGBM * (kSGBK/16) == kSGThreads
            const int r = item >> 2, c16 = item & 3;
            cpa16(&As[st][r][(c16 ^ (r & 3)) * 16],
                  Wfold + (long)(blockIdx.y * kSGBM + r) * kK + kt * kSGBK + c16 * 16);
        }
#pragma unroll
        for (int j = 0; j < (kSGBN * (kSGBK / 16)) / kSGThreads; ++j) {
            const int item = tid + j * kSGThreads;
            const int r = item >> 2, c16 = item & 3;
            cpa16(&Bs[st][r][(c16 ^ (r & 3)) * 16],
                  Bq + (long)(t0 + r) * kK + kt * kSGBK + c16 * 16);
        }
        cpc();
    };

    constexpr int kPrefetch = kSGStages - 1;
#pragma unroll
    for (int s = 0; s < kPrefetch; ++s) load(s, s);

    float accf[kSGMT][kSGNT][4];
    int acci[kSGMT][kSGNT][4];
#pragma unroll
    for (int a = 0; a < kSGMT; ++a)
        for (int b = 0; b < kSGNT; ++b)
            for (int r = 0; r < 4; ++r) { accf[a][b][r] = 0.0f; acci[a][b][r] = 0; }

#pragma unroll 1
    for (int kt = 0; kt < kTilesK; ++kt) {
        const int st = kt % kSGStages;
        if (kt + kPrefetch <= kTilesK) cpw<kPrefetch - 1>(); else cpw<0>();
        __syncthreads();
        if (kt + kPrefetch < kTilesK) load((kt + kPrefetch) % kSGStages, kt + kPrefetch);
#pragma unroll
        for (int kk = 0; kk < kSGBK / 32; ++kk) {
            unsigned bf[kSGNT][2];
#pragma unroll
            for (int nt = 0; nt < kSGNT; ++nt) {
                const int cn = wn * 32 + nt * 8 + (lane & 7);
                ldm_x2(bf[nt][0], bf[nt][1],
                       su(&Bs[st][cn][swz(kk * 2 + ((lane >> 3) & 1), cn) * 16]));
            }
#pragma unroll
            for (int mi = 0; mi < kSGMT; ++mi) {
                const int arow = (mi >= kSGMT / 2 ? kSGPM : 0) +
                                 (half_tile + (mi % (kSGMT / 2))) * 16 + (lane & 15);
                unsigned a0, a1, a2, a3;
                ldm_x4(a0, a1, a2, a3, su(&As[st][arow][swz(kk * 2 + (lane >> 4), arow) * 16]));
#pragma unroll
                for (int nt = 0; nt < kSGNT; ++nt)
                    mma_s8(acci[mi][nt][0], acci[mi][nt][1], acci[mi][nt][2], acci[mi][nt][3], a0,
                           a1, a2, a3, bf[nt][0], bf[nt][1]);
            }
        }
        if ((kt % kGPK) == (kGPK - 1)) {
            const int g = kt / kGPK;
#pragma unroll
            for (int mi = 0; mi < kSGMT; ++mi) {
                const int base =
                    (mi >= kSGMT / 2 ? kSGPM : 0) + (half_tile + (mi % (kSGMT / 2))) * 16;
                const float wg0 = __half2float(wsc[(long)grow(base + gid) * kGroups + g]);
                const float wg1 = __half2float(wsc[(long)grow(base + gid + 8) * kGroups + g]);
#pragma unroll
                for (int nt = 0; nt < kSGNT; ++nt) {
                    const int tk = t0 + wn * 32 + nt * 8 + 2 * lid;
                    const float av0 = asc[tk];
                    const float av1 = asc[tk + 1];
                    accf[mi][nt][0] += (float)acci[mi][nt][0] * wg0 * av0;
                    accf[mi][nt][1] += (float)acci[mi][nt][1] * wg0 * av1;
                    accf[mi][nt][2] += (float)acci[mi][nt][2] * wg1 * av0;
                    accf[mi][nt][3] += (float)acci[mi][nt][3] * wg1 * av1;
                    acci[mi][nt][0] = acci[mi][nt][1] = acci[mi][nt][2] = acci[mi][nt][3] = 0;
                }
            }
        }
    }
#pragma unroll
    for (int mi = 0; mi < kSGMT / 2; ++mi) {
        const int r0 = m0 + (half_tile + mi) * 16 + gid;
        const int r1 = r0 + 8;
#pragma unroll
        for (int nt = 0; nt < kSGNT; ++nt) {
            const int c0 = t0 + wn * 32 + nt * 8 + 2 * lid;
            const int c1 = c0 + 1;
            auto store = [&](int col, int row, float gv, float uv) {
                out[(long)col * kParents + row] = __float2bfloat16_rn(silu(gv) * uv);
            };
            store(c0, r0, accf[mi][nt][0], accf[mi + kSGMT / 2][nt][0]);
            store(c1, r0, accf[mi][nt][1], accf[mi + kSGMT / 2][nt][1]);
            store(c0, r1, accf[mi][nt][2], accf[mi + kSGMT / 2][nt][2]);
            store(c1, r1, accf[mi][nt][3], accf[mi + kSGMT / 2][nt][3]);
        }
    }
}

struct W4A8Cache {
    std::int8_t* w_i8 = nullptr;   // dequantized int8 weights (path A only)
    long w_elems      = 0;
    const void* w_key = nullptr;
    std::int8_t* a_i8 = nullptr;
    float* a_scale    = nullptr;
    long a_cap        = 0;
    int a_tokens      = 0;         // a_scale is sized for this many tokens
};
W4A8Cache& cache() {
    static W4A8Cache c;
    return c;
}

// The separate-dequantization path is the default: at prefill chunk sizes it
// measures slightly faster than the fused q5 decoder, because the fused decode
// is repeated once per M tile. NINFER_W4A8_Q5INLINE selects the fused kernel.
bool use_dequant_path() {
    static const bool inline_q5 = [] {
        const char* v = std::getenv("NINFER_W4A8_Q5INLINE");
        return v != nullptr && v[0] != '\0';
    }();
    return !inline_q5;
}

// The 128x64 / 8-warp / 2-blocks-per-SM variant is the default: measured ~3-5%
// faster on the down GEMM than the 128x128 single-block path A (the second
// independent block hides part of the per-k-tile barrier) with bit-identical
// numerics. NINFER_W4A8_WIDE restores the 128x128 path for A/B and rollback.
bool use_narrow_kernel() {
    static const bool wide = [] {
        const char* v = std::getenv("NINFER_W4A8_WIDE");
        return v != nullptr && v[0] != '\0';
    }();
    return !wide;
}

template <int kN, int kK>
void run(const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream) {
    const int tokens   = x.ne[1];
    W4A8Cache& c       = cache();
    const long a_elems = (long)tokens * kK;
    if (c.a_cap < a_elems || c.a_tokens < tokens) {
        if (c.a_i8) cudaFree(c.a_i8);
        if (c.a_scale) cudaFree(c.a_scale);
        if (cudaMalloc(&c.a_i8, (size_t)a_elems) != cudaSuccess) {
            c.a_i8 = nullptr; c.a_cap = 0; c.a_tokens = 0; return;
        }
        cudaMalloc(&c.a_scale, (size_t)tokens * sizeof(float));
        c.a_cap = a_elems;
        c.a_tokens = tokens;
    }
    quant_a8<<<tokens, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data), c.a_i8,
                                         c.a_scale, kK, tokens);
    CUDA_CHECK(cudaGetLastError());
    const dim3 grid(tokens / kBM, kN / kBN);
    const auto* wsc    = static_cast<const __half*>(w.scales);
    auto* out          = static_cast<__nv_bfloat16*>(residual.data);
    const int ld       = (int)(residual.nb[1] / sizeof(__nv_bfloat16));

    if (use_dequant_path()) {
        const long w_elems = (long)kN * kK;
        if (c.w_elems < w_elems) {
            if (c.w_i8) cudaFree(c.w_i8);
            if (cudaMalloc(&c.w_i8, (size_t)w_elems) != cudaSuccess) { c.w_i8 = nullptr; c.w_elems = 0; return; }
            c.w_elems = w_elems;
            c.w_key   = nullptr;
        }
        if (c.w_key != w.qdata) {
            const int dq_groups = kK / 64;
            const dim3 dq_grid((unsigned)((dq_groups * 4 + 255) / 256), (unsigned)kN);
            dequant_q5_i8<<<dq_grid, 256, 0, stream>>>(
                static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.qhigh),
                c.w_i8, dq_groups, kK);
            c.w_key = w.qdata;
        }
        if (use_narrow_kernel()) {
            const dim3 grid_n(tokens / kBMN, kN / kBNN);
            w4a8_gemm_residual_narrow_kernel<kN, kK><<<grid_n, kThreadsN, 0, stream>>>(
                c.a_i8, c.w_i8, c.a_scale, wsc, out, ld);
        } else {
            w4a8_gemm_residual_kernel<kN, kK><<<grid, kThreads, 0, stream>>>(
                c.a_i8, c.w_i8, c.a_scale, wsc, out, ld);
        }
    } else {
        constexpr int kSmem = (int)sizeof(W4A8Q5Smem);
        static bool configured = false;
        if (!configured) {
            cudaFuncSetAttribute(w4a8_q5_gemm_residual_kernel<kN, kK>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
            configured = true;
        }
        w4a8_q5_gemm_residual_kernel<kN, kK><<<grid, kThreads, kSmem, stream>>>(
            c.a_i8, static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh), c.a_scale, wsc, out, ld);
    }
    CUDA_CHECK(cudaGetLastError());
}

struct W4A8SgCache {
    std::int8_t* w_i8 = nullptr; // folded int8 gate/up weight
    long w_elems      = 0;
    const void* w_key = nullptr;
};
W4A8SgCache& sg_cache() {
    static W4A8SgCache c;
    return c;
}

template <int kParents, int kK>
void run_swiglu(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const int tokens   = x.ne[1];
    W4A8Cache& c       = cache();
    const long a_elems = (long)tokens * kK;
    if (c.a_cap < a_elems || c.a_tokens < tokens) {
        if (c.a_i8) cudaFree(c.a_i8);
        if (c.a_scale) cudaFree(c.a_scale);
        if (cudaMalloc(&c.a_i8, (size_t)a_elems) != cudaSuccess) {
            c.a_i8 = nullptr; c.a_cap = 0; c.a_tokens = 0; return;
        }
        cudaMalloc(&c.a_scale, (size_t)tokens * sizeof(float));
        c.a_cap    = a_elems;
        c.a_tokens = tokens;
    }
    quant_a8<<<tokens, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data), c.a_i8,
                                         c.a_scale, kK, tokens);
    CUDA_CHECK(cudaGetLastError());

    W4A8SgCache& sg    = sg_cache();
    const long w_elems = (long)(2 * kParents) * kK;
    if (sg.w_elems < w_elems) {
        if (sg.w_i8) cudaFree(sg.w_i8);
        if (cudaMalloc(&sg.w_i8, (size_t)w_elems) != cudaSuccess) { sg.w_i8 = nullptr; sg.w_elems = 0; return; }
        sg.w_elems = w_elems;
        sg.w_key   = nullptr;
    }
    if (sg.w_key != w.qdata) {
        const int groups = kK / 64;
        const dim3 dq((unsigned)((groups * 4 + 255) / 256), (unsigned)(2 * kParents));
        dequant_q4_i8_folded<<<dq, 256, 0, stream>>>(static_cast<const std::uint8_t*>(w.qdata),
                                                     sg.w_i8, groups, kK, kParents);
        sg.w_key = w.qdata;
    }
    const dim3 grid(tokens / kSGBN, (unsigned)(kParents / kSGPM));
    w4a8_swiglu_kernel<kParents, kK><<<grid, kSGThreads, 0, stream>>>(
        sg.w_i8, c.a_i8, c.a_scale, static_cast<const __half*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

bool w4a8_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("NINFER_W4A8");
        return v != nullptr && v[0] != '\0';
    }();
    return on;
}

bool w4a8_supported(const Weight& weight, std::int32_t tokens) {
    if (!w4a8_enabled()) return false;
    if (tokens <= 0 || (tokens % kBM) != 0) return false;
    if (weight.n != 5120) return false;
    if (weight.padded_shape[1] != weight.k) return false;
    // FFN down [5120,17408] and the output projections [5120,6144]
    // (attention o_proj and GDN output proj) are the registered profiles.
    return weight.k == 17408 || weight.k == 6144;
}

void w4a8_linear_add(const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream) {
    if (w.k == 6144) {
        run<5120, 6144>(x, w, residual, stream);
        return;
    }
    run<5120, 17408>(x, w, residual, stream);
}

bool w4a8_swiglu_supported(const Weight& weight, std::int32_t tokens) {
    if (!w4a8_enabled()) return false;
    if (tokens <= 0 || (tokens % kSGBN) != 0) return false;
    return weight.qtype == QType::Q4_G64_FP16 && weight.group == 64 && weight.n == 34816 &&
           weight.k == 5120 && weight.padded_shape[1] == 5120;
}

void w4a8_swiglu(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    run_swiglu<17408, 5120>(x, w, out, stream);
}

} // namespace ninfer::ops
