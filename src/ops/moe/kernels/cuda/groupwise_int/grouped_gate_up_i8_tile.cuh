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

// --- fragment assembly through shared memory -------------------------------------------------
// The per-thread form has to build every fragment by hand: eight A words and two B words per mma,
// each from four byte loads and shifts. ldmatrix moves a whole 8x8 block per instruction and returns
// exactly the registers the mma wants, so the tile is staged as int8 and read back with two x4 loads
// (gate and up) plus one x2 (activations) per k chunk.

__device__ __forceinline__ void cp_async_16(void* dst, const void* src) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                      const void* p) {
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a));
}
__device__ __forceinline__ void ldm_x2(unsigned& r0, unsigned& r1, const void* p) {
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(a));
}

// A staged row is 64 bytes, so the eight rows of an ldmatrix block would all start on the same bank
// group and read four-way conflicted. Xor-ing the 16-byte chunk index with the row's low bits spreads
// them; writers and readers must use the same mapping, which the bitwise test checks.
__device__ __forceinline__ int swz(const int row, const int kbyte) {
    return row * 64 + ((((kbyte >> 4) ^ (row & 3)) & 3) << 4) + (kbyte & 15);
}

// Double-buffered tile: the next 64-wide k group is copied in with cp.async while the current one is
// being multiplied, its weights are decoded into a second activation buffer before they are needed,
// and RowTiles 16-row groups share one staging pass. One row group per block made the staging and the
// two barriers per k group cover only 32 rows; four make them cover 128, which is the shape the bf16
// route uses.
template <int K, int Rows, int Cols, int RowTiles, class Access>
__global__ __launch_bounds__(Cols * 4) void grouped_gate_up_i8_tile_ldm(
    const std::int8_t* __restrict__ act_i8, const float* __restrict__ act_scale, Access weights,
    const int* __restrict__ offsets, const int* __restrict__ order,
    const int* __restrict__ job_experts, const int* __restrict__ job_columns,
    const GroupedInfo* __restrict__ info, int stride, __nv_bfloat16* __restrict__ out) {
    constexpr int kGateRows = Rows / 2;
    constexpr int kG        = K / 64;
    constexpr int kThreads  = Cols * 4;
    constexpr int kHalf     = 16;
    constexpr int kGateSpan = kHalf * RowTiles;   // gate rows this block covers
    constexpr int kTileRows = 2 * kGateSpan;      // gate rows first, then up rows
    constexpr int kDecodeIt = (kTileRows * 8 + kThreads - 1) / kThreads;

    if (info->direct) { return; }
    const int job = blockIdx.y;
    if (job >= info->jobs) { return; }

    __shared__ __align__(16) std::int8_t As[2][kTileRows][64];
    __shared__ __align__(16) std::int8_t Bs[2][Cols][64];
    __shared__ __align__(16) std::uint8_t Cr[2][kTileRows][kHalf * 2];

    const int expert = job_experts[job];
    const int begin  = offsets[expert];
    const int count  = offsets[expert + 1] - begin;
    const int column_base = job_columns[job];
    const int cols        = min(count - column_base, Cols);
    if (cols <= 0) { return; }

    const int t    = threadIdx.x;
    const int warp = t >> 5;
    const int lane = t & 31;
    const int r    = lane >> 2;
    const int c0   = warp * 8 + 2 * (lane & 3);
    const int c1   = c0 + 1;
    const int col0 = warp * 8;

    const int r0 = blockIdx.x * kGateSpan;
    if (r0 >= kGateRows) { return; }

    const auto matrix = weights.template resolve<Rows>(expert);
    const auto* codes = matrix.codes;
    const auto* wsc   = reinterpret_cast<const __half*>(matrix.scales);
    const int rb      = matrix.row;

    const int tok0   = order[begin + column_base + (c0 < cols ? c0 : 0)];
    const int tok1   = order[begin + column_base + (c1 < cols ? c1 : 0)];
    const auto* asc0 = act_scale + std::int64_t(tok0) * kG;
    const auto* asc1 = act_scale + std::int64_t(tok1) * kG;

    const auto stage = [&](int g, int buf) {
        // One weight row per thread only holds while the block covers kTileRows x 2; with 32 columns
        // and four row groups the block is 128 threads for 256 copies, so this loops.
#pragma unroll
        for (int i = 0; i < (kTileRows * 2 + kThreads - 1) / kThreads; ++i) {
            const int e = t + i * kThreads;
            if (e >= kTileRows * 2) { break; }
            const int row   = e >> 1;
            const int half  = e & 1;
            const int grow  = row < kGateSpan ? (r0 + row) : (kGateRows + r0 + row - kGateSpan);
            const auto* src = codes + std::int64_t(rb + grow) * (K / 2) + g * 32 + half * 16;
            cp_async_16(&Cr[buf][row][half * 16], src);
        }
        {
            const int col   = t >> 2;
            const int quad  = t & 3;
            const int slot  = begin + column_base + (col < cols ? col : 0);
            const auto* src = act_i8 + std::int64_t(order[slot]) * K + g * 64 + quad * 16;
            cp_async_16(reinterpret_cast<std::int8_t*>(&Bs[buf][0][0]) + swz(col, quad * 16), src);
        }
        cp_async_commit();
    };
    // Eight nibbles at a time: one 32-bit read of the packed row, per-byte xor/subtract to turn each
    // nibble into (n ^ 8) - 8, then two byte permutes to interleave the low and high nibbles into the
    // k order the tile stores.
    const auto decode = [&](int buf) {
#pragma unroll
        for (int i = 0; i < kDecodeIt; ++i) {
            const int e = t + i * kThreads;
            if (e >= kTileRows * 8) { break; }
            const int row = e >> 3;
            const int k0  = (e & 7) * 8;
            const std::uint32_t w =
                *reinterpret_cast<const std::uint32_t*>(&Cr[buf][row][k0 >> 1]);
            const std::uint32_t lo =
                __vsubss4(static_cast<int>((w & 0x0F0F0F0Fu) ^ 0x08080808u), 0x08080808);
            const std::uint32_t hi =
                __vsubss4(static_cast<int>(((w >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u), 0x08080808);
            auto* dst = reinterpret_cast<std::int8_t*>(&As[buf][0][0]);
            *reinterpret_cast<std::uint32_t*>(dst + swz(row, k0)) = __byte_perm(lo, hi, 0x5140);
            *reinterpret_cast<std::uint32_t*>(dst + swz(row, k0 + 4)) = __byte_perm(lo, hi, 0x7362);
        }
    };

    float gate[RowTiles][4];
    float up[RowTiles][4];
#pragma unroll
    for (int i = 0; i < RowTiles; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            gate[i][j] = 0.f;
            up[i][j]   = 0.f;
        }
    }

    // The four 8x8 blocks of the x4 load, and the two of the x2 load, are addressed by lane group.
    const int a_mat = lane >> 3;                      // 0..3
    const int a_row = ((a_mat & 1) ? 8 : 0) + (lane & 7);
    const int a_kt  = (a_mat >> 1) ? 16 : 0;
    const int b_mat = (lane >> 3) & 1;
    const int b_row = lane & 7;

    stage(0, 0);
    cp_async_wait<0>();
    __syncthreads();
    decode(0);
    __syncthreads();

    for (int g = 0; g < kG; ++g) {
        const int cur = g & 1;
        const int nxt = cur ^ 1;
        if (g + 1 < kG) { stage(g + 1, nxt); }

#pragma unroll
        for (int t4 = 0; t4 < RowTiles; ++t4) {
            int acc_g[4] = {0, 0, 0, 0};
            int acc_u[4] = {0, 0, 0, 0};
#pragma unroll
            for (int sub = 0; sub < 2; ++sub) {
                const int kk = sub * 32;
                unsigned ga0, ga1, ga2, ga3, ua0, ua1, ua2, ua3, bb0, bb1;
                const auto* as_base = reinterpret_cast<const std::int8_t*>(&As[cur][0][0]);
                ldm_x4(ga0, ga1, ga2, ga3, as_base + swz(t4 * kHalf + a_row, kk + a_kt));
                ldm_x4(ua0, ua1, ua2, ua3,
                       as_base + swz(kGateSpan + t4 * kHalf + a_row, kk + a_kt));
                ldm_x2(bb0, bb1,
                       reinterpret_cast<const std::int8_t*>(&Bs[cur][0][0]) +
                           swz(col0 + b_row, kk + b_mat * 16));
                mma_s8(acc_g[0], acc_g[1], acc_g[2], acc_g[3], ga0, ga1, ga2, ga3, bb0, bb1);
                mma_s8(acc_u[0], acc_u[1], acc_u[2], acc_u[3], ua0, ua1, ua2, ua3, bb0, bb1);
            }
            const int gr = r0 + t4 * kHalf + r;
            const float sg_lo = __half2float(wsc[(std::int64_t)gr * kG + g]);
            const float sg_hi = __half2float(wsc[(std::int64_t)(gr + 8) * kG + g]);
            const float su_lo = __half2float(wsc[(std::int64_t)(kGateRows + gr) * kG + g]);
            const float su_hi = __half2float(wsc[(std::int64_t)(kGateRows + gr + 8) * kG + g]);
            const float a0 = asc0[g], a1 = asc1[g];
            gate[t4][0] += (float)acc_g[0] * sg_lo * a0;
            gate[t4][1] += (float)acc_g[1] * sg_lo * a1;
            gate[t4][2] += (float)acc_g[2] * sg_hi * a0;
            gate[t4][3] += (float)acc_g[3] * sg_hi * a1;
            up[t4][0] += (float)acc_u[0] * su_lo * a0;
            up[t4][1] += (float)acc_u[1] * su_lo * a1;
            up[t4][2] += (float)acc_u[2] * su_hi * a0;
            up[t4][3] += (float)acc_u[3] * su_hi * a1;
        }

        if (g + 1 < kG) {
            cp_async_wait<0>();
            __syncthreads();
            decode(nxt);
            __syncthreads();
        }
    }

#pragma unroll
    for (int t4 = 0; t4 < RowTiles; ++t4) {
        const int row0 = r0 + t4 * kHalf + r;
        const int row1 = row0 + 8;
        if (c0 < cols) {
            const std::int64_t s = std::int64_t(begin + column_base + c0) * stride;
            out[s + row0] = __float2bfloat16_rn(common::silu(gate[t4][0]) * up[t4][0]);
            out[s + row1] = __float2bfloat16_rn(common::silu(gate[t4][2]) * up[t4][2]);
        }
        if (c1 < cols) {
            const std::int64_t s = std::int64_t(begin + column_base + c1) * stride;
            out[s + row0] = __float2bfloat16_rn(common::silu(gate[t4][1]) * up[t4][1]);
            out[s + row1] = __float2bfloat16_rn(common::silu(gate[t4][3]) * up[t4][3]);
        }
    }
}

} // namespace ninfer::ops::detail::moe_cuda::groupwise_int
