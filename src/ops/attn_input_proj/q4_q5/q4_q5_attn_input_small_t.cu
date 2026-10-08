#include "ops/linear/q5/q5_instances.cuh"
#include "ops/linear/q4/q4_instances.cuh"
#include "core/weight.h"
#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q4/q4_sliced_k_launch.cuh"
#include "ops/linear/q4/q4_simt_launch.cuh"
#include "ops/linear/q4/q4_gemv_launch.cuh"
#include "ops/linear/q5/q5_simt_launch.cuh"
#include "ops/linear/q5/q5_gemv_launch.cuh"
#include "ops/linear/q4/q4_small_t_mma.cuh"
#include "ops/linear/q5/q5_small_t_mma.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kSplitRow = 6144;
constexpr std::int32_t kHidden   = 5120;

// ---------------------------------------------------------------------------
// sm89 decode path: reuse the proven q4/q5 small-T MMA kernels (warp-per-K-slice,
// double-buffered cp.async, HMMA tensor-core compute) with a SPLIT epilogue that
// preserves v3's fused Q/K and gate/V single-pass output. The v3 SIMT kernels
// (q4_a16_simt, q5_a16_direct_simt) don't use tensor cores and ran ~10% slower
// per MTP decode round on the RTX 4090.
// ---------------------------------------------------------------------------

template <int N, int ActiveCols>
void launch_q4_attn_sm89(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
                          cudaStream_t stream) {
    constexpr int TileCols  = ActiveCols <= 8 ? 8 : 16;
    using Geometry          = Q4SmallTGeometry<N, kHidden>;
    // 8-row CTA: 960 CTAs for N=7680 (7.5 waves on 128 SMs) vs 480 for 16-row.
    constexpr int kBlocks   = (N + 7) / 8;

    Q4SmallTSplitEpilogue<kSplitRow> epilogue;
    epilogue.out     = static_cast<__nv_bfloat16*>(q.data);
    epilogue.out_tail = static_cast<__nv_bfloat16*>(key.data);
    epilogue.out_ld  = static_cast<std::int32_t>(q.nb[1] / sizeof(__nv_bfloat16));
    epilogue.tail_ld = static_cast<std::int32_t>(key.nb[1] / sizeof(__nv_bfloat16));

    q4_small_t_mma_kernel<Geometry, TileCols, ActiveCols, Q4SmallTSplitEpilogue<kSplitRow>,
                          Q4SmallTMmaIdentityRows8Row>
        <<<kBlocks, Q4DraftSmallTSchedule8Row::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales),
            static_cast<__nv_bfloat16*>(q.data),
            epilogue, Q4SmallTMmaIdentityRows8Row{});
    CUDA_CHECK(cudaGetLastError());
}

template <int N>
void launch_q4_attn_sm89_dispatch(int t, const Tensor& x, const Weight& weight, Tensor& q,
                                   Tensor& key, cudaStream_t stream) {
    switch (t) {
    case 1:  launch_q4_attn_sm89<N, 1>(x, weight, q, key, stream);  return;
    case 2:  launch_q4_attn_sm89<N, 2>(x, weight, q, key, stream);  return;
    case 3:  launch_q4_attn_sm89<N, 3>(x, weight, q, key, stream);  return;
    case 4:  launch_q4_attn_sm89<N, 4>(x, weight, q, key, stream);  return;
    case 5:  launch_q4_attn_sm89<N, 5>(x, weight, q, key, stream);  return;
    case 6:  launch_q4_attn_sm89<N, 6>(x, weight, q, key, stream);  return;
    case 7:  launch_q4_attn_sm89<N, 7>(x, weight, q, key, stream);  return;
    case 8:  launch_q4_attn_sm89<N, 8>(x, weight, q, key, stream);  return;
    case 9:  launch_q4_attn_sm89<N, 9>(x, weight, q, key, stream);  return;
    case 10: launch_q4_attn_sm89<N, 10>(x, weight, q, key, stream); return;
    case 11: launch_q4_attn_sm89<N, 11>(x, weight, q, key, stream); return;
    case 12: launch_q4_attn_sm89<N, 12>(x, weight, q, key, stream); return;
    default: throw std::invalid_argument("sm89 q4 attn: T out of [1,12]");
    }
}

template <int N, int ActiveCols>
void launch_q5_attn_sm89(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                          cudaStream_t stream) {
    constexpr int TileCols  = ActiveCols <= 8 ? 8 : 16;
    using Geometry          = Q5SmallTGeometry<N, kHidden>;
    constexpr int kBlocks   = (N + 15) / 16;

    Q5SmallTSplitEpilogue<kSplitRow> epilogue;
    epilogue.out_tail = static_cast<__nv_bfloat16*>(value.data);
    epilogue.tail_ld  = static_cast<std::int32_t>(value.nb[1] / sizeof(__nv_bfloat16));

    q5_small_t_mma_kernel<Geometry, TileCols, ActiveCols, Q5SmallTSplitEpilogue<kSplitRow>,
                          Q5SmallTMmaIdentityRows, 1>
        <<<kBlocks, Q5SmallTSchedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.qhigh),
            static_cast<const std::uint8_t*>(weight.scales),
            static_cast<__nv_bfloat16*>(gate.data),
            static_cast<std::int32_t>(x.nb[1] / sizeof(__nv_bfloat16)),
            static_cast<std::int32_t>(gate.nb[1] / sizeof(__nv_bfloat16)),
            epilogue, Q5SmallTMmaIdentityRows{});
    CUDA_CHECK(cudaGetLastError());
}

template <int N>
void launch_q5_attn_sm89_dispatch(int t, const Tensor& x, const Weight& weight, Tensor& gate,
                                   Tensor& value, cudaStream_t stream) {
    switch (t) {
    case 1:  launch_q5_attn_sm89<N, 1>(x, weight, gate, value, stream);  return;
    case 2:  launch_q5_attn_sm89<N, 2>(x, weight, gate, value, stream);  return;
    case 3:  launch_q5_attn_sm89<N, 3>(x, weight, gate, value, stream);  return;
    case 4:  launch_q5_attn_sm89<N, 4>(x, weight, gate, value, stream);  return;
    case 5:  launch_q5_attn_sm89<N, 5>(x, weight, gate, value, stream);  return;
    case 6:  launch_q5_attn_sm89<N, 6>(x, weight, gate, value, stream);  return;
    case 7:  launch_q5_attn_sm89<N, 7>(x, weight, gate, value, stream);  return;
    case 8:  launch_q5_attn_sm89<N, 8>(x, weight, gate, value, stream);  return;
    case 9:  launch_q5_attn_sm89<N, 9>(x, weight, gate, value, stream);  return;
    case 10: launch_q5_attn_sm89<N, 10>(x, weight, gate, value, stream); return;
    case 11: launch_q5_attn_sm89<N, 11>(x, weight, gate, value, stream); return;
    case 12: launch_q5_attn_sm89<N, 12>(x, weight, gate, value, stream); return;
    default: throw std::invalid_argument("sm89 q5 attn: T out of [1,12]");
    }
}

using Q4AttnSimtR8T4Schedule = Q4A16SimtSchedule<8, 4, 1, 16, 2, Cache::ca, 1>;

void launch_q4_gemv(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
                    cudaStream_t stream) {
    launch_q4_a16_gemv<q4_instances::GemvR1W8K5120>(
        q4_linear_operands(x, weight),
        LinearBf16SplitOutput2<kSplitRow>{
            {static_cast<__nv_bfloat16*>(q.data),
             static_cast<std::int64_t>(q.nb[1] / sizeof(__nv_bfloat16)), 0},
            {static_cast<__nv_bfloat16*>(key.data),
             static_cast<std::int64_t>(key.nb[1] / sizeof(__nv_bfloat16)), 0}},
        LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void launch_q4_simt(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
                    cudaStream_t stream) {
    launch_q4_a16_simt<Schedule>(
        q4_linear_operands(x, weight),
        LinearBf16SplitOutput2<kSplitRow>{
            {static_cast<__nv_bfloat16*>(q.data),
             static_cast<std::int64_t>(q.nb[1] / sizeof(__nv_bfloat16)), 0},
            {static_cast<__nv_bfloat16*>(key.data),
             static_cast<std::int64_t>(key.nb[1] / sizeof(__nv_bfloat16)), 0}},
        LinearIdentityEpilogue{}, stream);
}

template <std::int32_t Capacity>
void launch_q4_sliced_exact(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
                            cudaStream_t stream) {
    using Schedule = Q4A16SlicedKMmaSchedule<16, (Capacity + 7) / 8 * 8, 8, 1, Cache::cg, Cache::ca,
                                             6, kHidden, Capacity>;
    launch_q4_a16_sliced_k_mma<Schedule>(
        q4_linear_operands(x, weight),
        LinearBf16SplitOutput2<kSplitRow>{
            {static_cast<__nv_bfloat16*>(q.data),
             static_cast<std::int64_t>(q.nb[1] / sizeof(__nv_bfloat16)), 0},
            {static_cast<__nv_bfloat16*>(key.data),
             static_cast<std::int64_t>(key.nb[1] / sizeof(__nv_bfloat16)), 0}},
        LinearIdentityEpilogue{}, stream);
}

void launch_q4_sliced_band(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
                           cudaStream_t stream) {
    if (weight.padded_shape[1] != kHidden) {
        throw std::invalid_argument("attention Q4 K-split requires padded K == hidden");
    }
    switch (x.ne[1]) {
    case 7:
        launch_q4_sliced_exact<7>(x, weight, q, key, stream);
        return;
    case 8:
        launch_q4_sliced_exact<8>(x, weight, q, key, stream);
        return;
    case 9:
        launch_q4_sliced_exact<9>(x, weight, q, key, stream);
        return;
    case 10:
        launch_q4_sliced_exact<10>(x, weight, q, key, stream);
        return;
    case 11:
        launch_q4_sliced_exact<11>(x, weight, q, key, stream);
        return;
    case 12:
        launch_q4_sliced_exact<12>(x, weight, q, key, stream);
        return;
    default:
        throw std::invalid_argument("attention Q4 K-split band covers T in [7,12]");
    }
}

void launch_q4(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key, cudaStream_t stream) {
    const int t = x.ne[1];
    // sm89 8-row CTA: 960 CTAs for N=7680 (7.5 waves) vs v3 SIMT 960 CTAs (7.5
    // waves). The 8-row small-T MMA uses tensor cores (HMMA) which the v3 SIMT
    // does not; the Q4SmallTSplitEpilogue preserves the fused Q/K output.
    if (t >= 1 && t <= 12 && weight.padded_shape[1] == kHidden && (weight.n % 8) == 0) {
        switch (weight.n) {
        case 7680:
            launch_q4_attn_sm89_dispatch<7680>(t, x, weight, q, key, stream);
            return;
        default:
            break;
        }
    }
    switch (x.ne[1]) {
    case 1:
        launch_q4_gemv(x, weight, q, key, stream);
        return;
    case 7:
    case 8:
    case 9:
    case 10:
    case 11:
    case 12:
        launch_q4_sliced_band(x, weight, q, key, stream);
        return;
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
        launch_q4_simt<Q4AttnSimtR8T4Schedule>(x, weight, q, key, stream);
        return;
    default:
        throw std::invalid_argument("attention Q4 split-output requires T in [1,12]");
    }
}

auto q5_projection_output(Tensor& first, Tensor& second) {
    return LinearBf16SplitOutput2<kSplitRow>{{static_cast<__nv_bfloat16*>(first.data),
                                              std::int64_t(first.nb[1] / sizeof(__nv_bfloat16)), 0},
                                             {static_cast<__nv_bfloat16*>(second.data),
                                              std::int64_t(second.nb[1] / sizeof(__nv_bfloat16)),
                                              0}};
}

void launch_q5_gemv(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                    cudaStream_t stream) {
    launch_q5_a16_gemv<q5_instances::GemvR16W1G16S2XK5120>(q5_linear_operands(x, weight),
                                                           q5_projection_output(gate, value),
                                                           LinearIdentityEpilogue{}, stream);
}

template <int Cols>
void launch_q5_split4(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                      cudaStream_t stream) {
    using Schedule = Q5A16DirectSimtSchedule<1, Cols, 4, 4, 10, kHidden, true>;
    launch_q5_a16_direct_simt<Schedule>(q5_linear_operands(x, weight),
                                        q5_projection_output(gate, value), LinearIdentityEpilogue{},
                                        stream);
}

void launch_q5_split4_exact(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                            cudaStream_t stream) {
    switch (x.ne[1]) {
    case 2:
        launch_q5_split4<2>(x, weight, gate, value, stream);
        return;
    case 3:
        launch_q5_split4<3>(x, weight, gate, value, stream);
        return;
    case 4:
        launch_q5_split4<4>(x, weight, gate, value, stream);
        return;
    case 5:
        launch_q5_split4<5>(x, weight, gate, value, stream);
        return;
    case 6:
        launch_q5_split4<6>(x, weight, gate, value, stream);
        return;
    case 7:
        launch_q5_split4<7>(x, weight, gate, value, stream);
        return;
    case 8:
        launch_q5_split4<8>(x, weight, gate, value, stream);
        return;
    case 9:
        launch_q5_split4<9>(x, weight, gate, value, stream);
        return;
    default:
        throw std::invalid_argument("attention Q5 split4 requires T in [2,9]");
    }
}

template <int ColsPerTile>
void launch_q5_simt(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                    cudaStream_t stream) {
    using Schedule = Q5A16SimtSchedule<8, ColsPerTile, 1, 16, 2, Cache::ca, 1>;
    launch_q5_a16_simt<Schedule>(q5_linear_operands(x, weight), q5_projection_output(gate, value),
                                 LinearIdentityEpilogue{}, stream);
}

void launch_q5(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
               cudaStream_t stream) {
    // sm89 small-T MMA measured SLOWER than v3 split4/SIMT for N=7680 (same
    // tail-wave issue as q4). Falls through to v3 routing.
    const int t = x.ne[1];
    if (t == 1) {
        launch_q5_gemv(x, weight, gate, value, stream);
        return;
    }
    if (t <= 9) {
        launch_q5_split4_exact(x, weight, gate, value, stream);
        return;
    }
    if (t <= 12) {
        launch_q5_simt<4>(x, weight, gate, value, stream);
        return;
    }
    throw std::invalid_argument("attention Q5 split-output requires T in [1,12]");
}

} // namespace

void q4_q5_attn_input_small_t_launch(const Tensor& x, const Weight& query_key_weight,
                                     const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_q4(x, query_key_weight, q, k, stream);
    launch_q5(x, gate_value_weight, gate, v, stream);
}

} // namespace ninfer::ops::detail
