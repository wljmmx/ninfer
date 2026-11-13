#pragma once

// Folded gate/up GEMM for the width-5 and width-6 row-split formats.
//
// The Q4 kernel (`q4_linear_swiglu_gemm_mma.cuh`) pairs a 32-row gate slice with its 32 matching
// up rows inside one BM=64 weight tile, so one tensor-core contraction produces both projections
// and the warp's two accumulator halves pair directly in the SiLU epilogue. Q5 and Q6 keep that
// exact layout and differ only in how a code is decoded: the fourth bit is replaced by one (Q5) or
// two (Q6) bits that live in a separate per-row high-bit plane. This file is that kernel with the
// plane staged alongside the codes and scales.
//
// `Atom` is `Q5MmaDecodeAtom` or `Q6MmaDecodeAtom`; both take
// `(staged_codes, staged_high, scale_ptr, staged_group_index, lane)` and return the bf16 pair.
// `HB` is the plane's bytes per group per row (8 for Q5, 16 for Q6).

#include "ops/common/math.cuh"
#include "ops/common/rowsplit_mma.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <class Cfg, class Atom, int HB, bool FullTiles>
__global__
__launch_bounds__(Cfg::THREADS, Cfg::MIN_BLOCKS) void folded_swiglu_high_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ out, std::int32_t intermediate, std::int32_t k, std::int32_t t,
    std::int32_t padded_k) {
    constexpr int BM   = Cfg::BM;
    constexpr int BN   = Cfg::BN;
    constexpr int BK   = Cfg::BK;
    constexpr int WN   = Cfg::WN;
    constexpr int MT   = Cfg::MT;
    constexpr int NT   = Cfg::NT;
    constexpr int KSUB = BK / 16;
    constexpr int S    = Cfg::STAGES;
    constexpr int SB   = Cfg::SCALE_BYTES;
    constexpr int PM   = BM / 2;
    static_assert(BK == 64, "folded gate/up kernel requires one group per K tile");
    static_assert(BM % 2 == 0 && MT % 2 == 0,
                  "folded gate/up mapping needs an even number of 16-row MMA tiles per warp");
    static_assert(Cfg::WARPS_M * (MT / 2) * 16 == BM / 2,
                  "folded gate/up mapping must tile each half of the block exactly once");

    __shared__ __align__(16) __nv_bfloat16 As[BM * BK];
    __shared__ __align__(16) __nv_bfloat16 Bs[S][BN * BK];
    __shared__ __align__(16) std::uint8_t Cr[S][BM * 32];
    __shared__ __align__(16) std::uint8_t Hr[S][BM * HB];
    __shared__ __align__(16) std::uint8_t Sr[S][BM * SB];

    const int kg   = padded_k >> 6;
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wn   = warp % Cfg::WARPS_N;
    const int wm   = warp / Cfg::WARPS_N;
    const int half_tile = wm * (MT / 2);
    const int gid       = lane >> 2;
    const int lid       = lane & 3;
    const int m0        = static_cast<int>(blockIdx.y) * PM;
    const int t0        = static_cast<int>(blockIdx.x) * BN;

    float acc[MT][NT][4];
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
#pragma unroll
            for (int c = 0; c < 4; ++c) { acc[mi][ni][c] = 0.0f; }
        }
    }

    const int NKT      = padded_k / BK;
    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    auto stage_load_x = [&](int stage, int kt) {
        const int k0 = kt * BK;
#pragma unroll 1
        for (int c = tid; c < BN * (BK / 8); c += Cfg::THREADS) {
            const int tl       = c / (BK / 8);
            const int kg8      = c - tl * (BK / 8);
            const int kl       = kg8 * 8;
            const int col      = t0 + tl;
            const int kk       = k0 + kl;
            __nv_bfloat16* dst = &Bs[stage][tl * BK + gemm_swz64(tl, kl)];
            if constexpr (FullTiles) {
                gemm_cp_async<16, Cfg>(dst, &x[static_cast<std::int64_t>(col) * k + kk]);
            } else if (col < t && kk + 8 <= k) {
                gemm_cp_async<16, Cfg>(dst, &x[static_cast<std::int64_t>(col) * k + kk]);
            } else {
                store_vec(dst, make_int4(0, 0, 0, 0));
            }
        }
    };

    auto global_row = [&](int row) { return m0 + (row & (PM - 1)) + (row / PM) * intermediate; };

    auto stage_load_quant = [&](int stage, int kt) {
        const int g = (kt * BK) >> 6;
#pragma unroll 1
        for (int c = tid; c < BM * 2; c += Cfg::THREADS) {
            const int row  = c >> 1;
            const int half = c & 1;
            const int grow = global_row(row);
            auto* dst      = &Cr[stage][row * 32 + half * 16];
            if constexpr (FullTiles) {
                const std::int64_t gi = static_cast<std::int64_t>(grow) * kg + g;
                gemm_cp_async<16, Cfg>(dst, &codes[gi * 32 + half * 16]);
            } else if (m0 + (row & (PM - 1)) < intermediate) {
                const std::int64_t gi = static_cast<std::int64_t>(grow) * kg + g;
                gemm_cp_async<16, Cfg>(dst, &codes[gi * 32 + half * 16]);
            } else {
                store_vec(dst, make_int4(0, 0, 0, 0));
            }
        }
#pragma unroll 1
        for (int row = tid; row < BM; row += Cfg::THREADS) {
            const int grow   = global_row(row);
            auto* dst        = &Hr[stage][row * HB];
            auto* dst_scale  = &Sr[stage][row * SB];
            const bool valid = FullTiles || (m0 + (row & (PM - 1)) < intermediate);
            if (valid) {
                const std::int64_t gi = static_cast<std::int64_t>(grow) * kg + g;
                gemm_cp_async<HB, Cfg>(dst, &high[gi * HB]);
            } else {
                store_vec(dst, make_int4(0, 0, 0, 0));
            }
            if (valid) {
                const int aligned_g           = g & ~1;
                const std::int64_t gi         = static_cast<std::int64_t>(grow) * kg + g;
                const std::int64_t aligned_gi = static_cast<std::int64_t>(grow) * kg + aligned_g;
                if (aligned_g + 1 < kg) {
                    gemm_cp_async<4, Cfg>(dst_scale, &scales[aligned_gi * 2]);
                } else {
                    *reinterpret_cast<std::uint16_t*>(dst_scale) =
                        *reinterpret_cast<const std::uint16_t*>(&scales[gi * 2]);
                    *reinterpret_cast<std::uint16_t*>(dst_scale + 2) = 0;
                }
            } else {
                *reinterpret_cast<std::uint32_t*>(dst_scale) = 0;
            }
        }
    };

    auto dequant_to_As = [&](int stage, int kt) {
        const int scale_off = ((kt * BK >> 6) & 1) * 2;
        for (int row = warp; row < BM; row += Cfg::WARPS) {
            // The atom indexes both its code plane and its high-bit plane by the staged group
            // index, so both take the stage base; only the scale pointer is pre-offset.
            const __nv_bfloat162 w = Atom::decode_pair(
                Cr[stage], Hr[stage], &Sr[stage][row * SB + scale_off], row, lane);
            const int sc = gemm_swz64(row, 2 * lane);
            store_vec(&As[row * BK + sc], w);
        }
    };

#pragma unroll
    for (int s = 0; s < S; ++s) {
        if (s < NKT) { stage_load_x(s, s); stage_load_quant(s, s); }
        ninfer::ops::cp_commit();
    }

    for (int it = 0; it < NKT; ++it) {
        const int stage = it % S;
        ninfer::ops::cp_wait<S - 1>();
        __syncthreads();
        dequant_to_As(stage, it);
        __syncthreads();

        unsigned af[MT][4];
        unsigned bf[NT][2];
#pragma unroll
        for (int ki = 0; ki < KSUB; ++ki) {
            const int ks = ki * 16;
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int arow =
                    (mi >= MT / 2 ? PM : 0) + (half_tile + mi % (MT / 2)) * 16 + a_rowoff;
                ldmatrix_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                            smem_addr(&As[arow * BK + gemm_swz64(arow, ks + a_coloff)]));
            }
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int brow = wn * WN + ni * 8 + b_rin;
                ldmatrix_x2(bf[ni][0], bf[ni][1],
                            smem_addr(&Bs[stage][brow * BK + gemm_swz64(brow, ks + b_koff)]));
            }
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    mma_bf16(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                             af[mi][0], af[mi][1], af[mi][2], af[mi][3], bf[ni][0], bf[ni][1]);
                }
            }
        }

        __syncthreads();
        const int next = it + S;
        if (next < NKT) { stage_load_x(stage, next); stage_load_quant(stage, next); }
        ninfer::ops::cp_commit();
    }

#pragma unroll
    for (int mi = 0; mi < MT / 2; ++mi) {
        const int r0 = m0 + (half_tile + mi) * 16 + gid;
        const int r1 = r0 + 8;
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const int cc0 = t0 + wn * WN + ni * 8 + 2 * lid;
            const int cc1 = cc0 + 1;
            auto store    = [&](int col, int row, float gv, float uv) {
                out[static_cast<std::int64_t>(col) * intermediate + row] =
                    __float2bfloat16_rn(silu(gv) * uv);
            };
            if constexpr (FullTiles) {
                store(cc0, r0, acc[mi][ni][0], acc[mi + MT / 2][ni][0]);
                store(cc1, r0, acc[mi][ni][1], acc[mi + MT / 2][ni][1]);
                store(cc0, r1, acc[mi][ni][2], acc[mi + MT / 2][ni][2]);
                store(cc1, r1, acc[mi][ni][3], acc[mi + MT / 2][ni][3]);
            } else {
                if (r0 < intermediate) {
                    if (cc0 < t) { store(cc0, r0, acc[mi][ni][0], acc[mi + MT / 2][ni][0]); }
                    if (cc1 < t) { store(cc1, r0, acc[mi][ni][1], acc[mi + MT / 2][ni][1]); }
                }
                if (r1 < intermediate) {
                    if (cc0 < t) { store(cc0, r1, acc[mi][ni][2], acc[mi + MT / 2][ni][2]); }
                    if (cc1 < t) { store(cc1, r1, acc[mi][ni][3], acc[mi + MT / 2][ni][3]); }
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
