#include "core/weight.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "core/device.h" // CUDA_CHECK
#include "ops/linear/q4/q4_sliced_k_launch.cuh"
#include "ops/linear/q4/q4_small_t_mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>
#include <array>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kN                 = 34816;
constexpr int kK                 = 5120;
constexpr int kIntermediate      = kN / 2;
constexpr int kGroupK            = 64;
constexpr int kGroups            = kK / kGroupK;
constexpr int kBytesPerGroup     = 32;
constexpr int kVecBytes          = 16;
constexpr int kGroupsPerWarpTile = 16;
constexpr int kVecsPerWarpTile   = kGroupsPerWarpTile * kBytesPerGroup / kVecBytes;
constexpr int kWarpsPerBlock     = 4;
constexpr int kBlockThreads      = kWarpsPerBlock * 32;
constexpr int kPairsPerBlock     = kWarpsPerBlock;
constexpr int kXVecs             = kK / 8; // x as uint4 (8 bf16 each)
constexpr int kTiles             = kGroups / kGroupsPerWarpTile;
static_assert(kIntermediate % kPairsPerBlock == 0);
static_assert(kBytesPerGroup == 2 * kVecBytes);
static_assert(kGroups % kGroupsPerWarpTile == 0);
static_assert(kVecsPerWarpTile == 32);

struct Q4SwiGluSmallTRows {
    static constexpr int kOutputRowsPerCta = 8;

    __host__ __device__ int output_rows(int rows) const { return rows / 2; }

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + (local_row & 7) + (local_row >= 8 ? kIntermediate : 0);
    }
};

struct Q4SwiGluSmallTEpilogue {
    template <class Output>
    __device__ __forceinline__ void store_fragment(const Output& output, int row, int token,
                                                   float4 projected, int rows,
                                                   int token_end) const {
        if (row >= rows / 2) return;
        if (token < token_end) output.store(row, token, silu(projected.x) * projected.z);
        if (token + 1 < token_end) output.store(row, token + 1, silu(projected.y) * projected.w);
    }

    // Interface for q4_small_t_mma_kernel's else-branch epilogue dispatch:
    // store<ActiveCols>(int row, int col0, float4 projected) where projected =
    // (gate[col0], gate[col0+1], up[col0], up[col0+1]). The SiLU-gated product
    // is written to out at (row, col0) / (row, col0+1).
    __nv_bfloat16* out_ptr;
    std::int32_t out_ld_val;

    __device__ __forceinline__ Q4SwiGluSmallTEpilogue make(__nv_bfloat16* out, std::int32_t out_ld) const {
        Q4SwiGluSmallTEpilogue e;
        e.out_ptr = out;
        e.out_ld_val = out_ld;
        return e;
    }

    template <int ActiveCols>
    __device__ __forceinline__ void store(int row, int col0, float4 projected) const {
        const float g0 = silu(projected.x);
        const float g1 = silu(projected.y);
        if (col0 < ActiveCols) {
            out_ptr[static_cast<std::int64_t>(col0) * out_ld_val + row] =
                __float2bfloat16_rn(g0 * projected.z);
        }
        if (col0 + 1 < ActiveCols) {
            out_ptr[static_cast<std::int64_t>(col0 + 1) * out_ld_val + row] =
                __float2bfloat16_rn(g1 * projected.w);
        }
    }
};

using SmallTLauncher = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <int ActiveCols>
void launch_small_t_active(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    // Wider tiles are shared-memory limited before six CTAs can be resident.
    // Match launch bounds to that limit instead of forcing register spills.
    constexpr int kMinBlocks = ActiveCols <= 8    ? 6
                               : ActiveCols <= 16 ? 4
                               : ActiveCols <= 24 ? 3
                                                  : 2;
    using Schedule =
        Q4A16SlicedKMmaSchedule<16, (ActiveCols + 7) / 8 * 8, 8, 1, Cache::cg, Cache::ca,
                                kMinBlocks, kK, ActiveCols, Q4SlicedKReduction::Pairwise>;
    launch_q4_a16_sliced_k_mma<Schedule>(
        q4_linear_operands(x, w),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), kIntermediate},
        Q4SwiGluSmallTEpilogue{}, stream, Q4SwiGluSmallTRows{});
}

template <std::size_t... Offsets>
constexpr auto make_small_t_launchers(std::index_sequence<Offsets...>) {
    return std::array<SmallTLauncher, sizeof...(Offsets)>{
        &launch_small_t_active<8 * (1 + static_cast<int>(Offsets))>...};
}

constexpr auto kSmallTLaunchers = make_small_t_launchers(std::make_index_sequence<4>{});

// sm_89 (RTX 4090) v2-tuned small_t_exact: one exact-ActiveCols instance per T
// (T=2..32), using q4_small_t_mma_kernel (RowSplit path, already ported). The
// kernel's else-branch epilogue dispatch calls epilogue.store<ActiveCols>(row,
// col0, float4) — the Q4SwiGluSmallTEpilogue above now provides that method,
// applying silu(gate) * up and writing to the output.
struct Q4SwiGluSmallTGeometry {
    static constexpr int kInputRows    = kK;
    static constexpr int kGroupsPerRow = kK / 64;
};

template <int ActiveCols>
void launch_small_t_exact(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    constexpr int TileCols =
        ActiveCols <= 8 ? 8 : (ActiveCols <= 16 ? 16 : (ActiveCols <= 24 ? 24 : 32));
    constexpr int kBlocks = kIntermediate / Q4SwiGluSmallTRows::kOutputRowsPerCta;
    const auto out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    Q4SwiGluSmallTEpilogue epilogue;
    epilogue.out_ptr = static_cast<__nv_bfloat16*>(out.data);
    epilogue.out_ld_val = out_ld;

    q4_small_t_mma_kernel<Q4SwiGluSmallTGeometry, TileCols, ActiveCols, Q4SwiGluSmallTEpilogue,
                          Q4SwiGluSmallTRows>
        <<<kBlocks, Q4DraftSmallTSchedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data),
            epilogue, Q4SwiGluSmallTRows{});
    CUDA_CHECK(cudaGetLastError());
}

template <std::size_t... Offsets>
constexpr auto make_small_t_exact_launchers(std::index_sequence<Offsets...>) {
    return std::array<SmallTLauncher, sizeof...(Offsets)>{
        &launch_small_t_exact<2 + static_cast<int>(Offsets)>...};
}

constexpr auto kSmallTExactLaunchers = make_small_t_exact_launchers(std::make_index_sequence<31>{});

__device__ __forceinline__ void q4_issue_pair_tile(uint4 (*__restrict__ s_code)[kVecsPerWarpTile],
                                                   uint4 (*__restrict__ s_scale)[2],
                                                   const std::uint8_t* __restrict__ gate_code_row,
                                                   const std::uint8_t* __restrict__ gate_scale_row,
                                                   const std::uint8_t* __restrict__ up_code_row,
                                                   const std::uint8_t* __restrict__ up_scale_row,
                                                   int tile, int lane) {
    const int g0 = tile * kGroupsPerWarpTile;
    pipe_copy<16>(&s_code[0][lane],
                  reinterpret_cast<const uint4*>(gate_code_row + g0 * kBytesPerGroup) + lane);
    pipe_copy<16>(&s_code[1][lane],
                  reinterpret_cast<const uint4*>(up_code_row + g0 * kBytesPerGroup) + lane);
    if (lane < 2) {
        pipe_copy<16>(&s_scale[0][lane],
                      reinterpret_cast<const uint4*>(gate_scale_row + g0 * 2) + lane);
        pipe_copy<16>(&s_scale[1][lane],
                      reinterpret_cast<const uint4*>(up_scale_row + g0 * 2) + lane);
    }
    pipe_commit();
}

__global__ void q4_linear_swiglu_gemv_pair_kernel(const __nv_bfloat16* __restrict__ x,
                                                  const std::uint8_t* __restrict__ codes,
                                                  const std::uint8_t* __restrict__ scales,
                                                  __nv_bfloat16* __restrict__ out) {
    constexpr int kStages   = 3;
    constexpr int kPrefetch = kStages - 1;
    __shared__ __align__(16) __nv_bfloat16 x_sh[kK];
    __shared__ uint4 code_tile[kWarpsPerBlock][kStages][2][kVecsPerWarpTile];
    __shared__ uint4 scale_tile[kWarpsPerBlock][kStages][2][2];

    auto* x_sh_v    = reinterpret_cast<uint4*>(x_sh);
    const auto* x_g = reinterpret_cast<const uint4*>(x);
    for (int i = static_cast<int>(threadIdx.x); i < kXVecs; i += static_cast<int>(blockDim.x)) {
        x_sh_v[i] = x_g[i];
    }
    __syncthreads();

    const int lane    = static_cast<int>(threadIdx.x) & 31;
    const int warp    = static_cast<int>(threadIdx.x) >> 5;
    const int out_row = static_cast<int>(blockIdx.x) * kPairsPerBlock + warp;

    const std::uint8_t* gate_code_row =
        codes + static_cast<std::int64_t>(out_row) * kGroups * kBytesPerGroup;
    const std::uint8_t* gate_scale_row = scales + static_cast<std::int64_t>(out_row) * kGroups * 2;
    const std::uint8_t* up_code_row =
        codes + static_cast<std::int64_t>(out_row + kIntermediate) * kGroups * kBytesPerGroup;
    const std::uint8_t* up_scale_row =
        scales + static_cast<std::int64_t>(out_row + kIntermediate) * kGroups * 2;
    const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x_sh);

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
        if (p < kTiles) {
            q4_issue_pair_tile(code_tile[warp][p], scale_tile[warp][p], gate_code_row,
                               gate_scale_row, up_code_row, up_scale_row, p, lane);
        } else {
            pipe_commit();
        }
    }

#pragma unroll 1
    for (int tile = 0; tile < kTiles; ++tile) {
        const int fetch = tile + kPrefetch;
        if (fetch < kTiles) {
            const int buf = fetch % kStages;
            q4_issue_pair_tile(code_tile[warp][buf], scale_tile[warp][buf], gate_code_row,
                               gate_scale_row, up_code_row, up_scale_row, fetch, lane);
        } else {
            pipe_commit();
        }
        pipe_wait<kPrefetch>();
        __syncwarp();

        const int buf           = tile % kStages;
        const auto* gate_codes  = reinterpret_cast<const std::uint8_t*>(code_tile[warp][buf][0]);
        const auto* up_codes    = reinterpret_cast<const std::uint8_t*>(code_tile[warp][buf][1]);
        const auto* gate_scales = reinterpret_cast<const std::uint16_t*>(scale_tile[warp][buf][0]);
        const auto* up_scales   = reinterpret_cast<const std::uint16_t*>(scale_tile[warp][buf][1]);
#pragma unroll
        for (int tile_group = 0; tile_group < kGroupsPerWarpTile; ++tile_group) {
            const float gate_scale =
                __half2float(__ushort_as_half(static_cast<std::uint16_t>(gate_scales[tile_group])));
            const float up_scale =
                __half2float(__ushort_as_half(static_cast<std::uint16_t>(up_scales[tile_group])));

            const int gate_packed =
                static_cast<int>(gate_codes[tile_group * kBytesPerGroup + lane]);
            const int gate_q0   = sign_extend<4>(gate_packed & 0x0f);
            const int gate_q1   = sign_extend<4>(gate_packed >> 4);
            const int up_packed = static_cast<int>(up_codes[tile_group * kBytesPerGroup + lane]);
            const int up_q0     = sign_extend<4>(up_packed & 0x0f);
            const int up_q1     = sign_extend<4>(up_packed >> 4);
            const int k0        = (tile * kGroupsPerWarpTile + tile_group) * kGroupK + lane * 2;
            const float2 xv     = __bfloat1622float2(x2[k0 >> 1]);
            gate_acc            = fmaf(static_cast<float>(gate_q0) * gate_scale, xv.x, gate_acc);
            gate_acc            = fmaf(static_cast<float>(gate_q1) * gate_scale, xv.y, gate_acc);
            up_acc              = fmaf(static_cast<float>(up_q0) * up_scale, xv.x, up_acc);
            up_acc              = fmaf(static_cast<float>(up_q1) * up_scale, xv.y, up_acc);
        }
        __syncwarp();
    }

    gate_acc = warp_reduce_sum(gate_acc);
    up_acc   = warp_reduce_sum(up_acc);
    if (lane == 0) { out[out_row] = __float2bfloat16(silu(gate_acc) * up_acc); }
}

} // namespace

void q4_linear_swiglu_gemv_pair_launch(const Tensor& x, const Weight& w, Tensor& out,
                                       cudaStream_t stream) {
    if (w.n != kN || w.k != kK || w.padded_shape[1] != kK) {
        throw std::invalid_argument("q4 linear_swiglu GEMV requires weight [34816,5120]");
    }
    const int grid = kIntermediate / kPairsPerBlock;
    q4_linear_swiglu_gemv_pair_kernel<<<grid, kBlockThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void q4_linear_swiglu_small_t_tiled_launch(const Tensor& x, const Weight& w, Tensor& out,
                                           cudaStream_t stream) {
    if (x.ne[1] < 2 || x.ne[1] > 32) {
        throw std::invalid_argument("Q4 LinearSwiGLU exact small-T requires T=2..32");
    }
    kSmallTLaunchers[static_cast<std::size_t>((x.ne[1] - 1) / 8)](x, w, out, stream);
}

void q4_linear_swiglu_small_t_exact_launch(const Tensor& x, const Weight& w, Tensor& out,
                                           cudaStream_t stream) {
    if (x.ne[1] < 2 || x.ne[1] > 32) {
        throw std::invalid_argument("Q4 LinearSwiGLU exact small-T requires T=2..32");
    }
    kSmallTExactLaunchers[static_cast<std::size_t>(x.ne[1] - 2)](x, w, out, stream);
}

} // namespace ninfer::ops::detail
