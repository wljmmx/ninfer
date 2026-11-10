#include "ops/gated_residual/mix/launch_split.h"
#include "ops/gated_residual/mix/kernel/output.cuh"
#include "ops/gated_residual/mix/kernel/up_q5_mix.cuh"
#include "ops/linear/q5/q5_operands.h"
#include "ops/linear/q5/q5_schedule.cuh"
#include "ops/linear/q5/q5_simt_launch.cuh"
#include "ops/common/token_slices.h"

namespace ninfer::ops::detail::hc::mix {
namespace {
// sm_89: one warp per row walks the whole 160-group K of the down matrix, mirroring the shipped
// Q5 projection instances; the injection matrix is four rows and takes one warp each.
using DownQ5Schedule   = Q5A16SimtSchedule<8, 8, 1, 16, 2, Cache::ca, 1>;
using InjectQ5Schedule = Q5A16SimtSchedule<4, 8, 1, 16, 2, Cache::ca, 1>;

Q5LinearOperands operands(const void* activation, const Weight& weight, const Q5Planes& planes,
                          std::int32_t columns) {
    return {static_cast<const __nv_bfloat16*>(activation),
            static_cast<const std::uint8_t*>(planes.codes),
            static_cast<const std::uint8_t*>(planes.high),
            static_cast<const std::uint8_t*>(planes.scales),
            weight.n,
            weight.k,
            columns,
            planes.padded_k};
}

template <int Features, int Tokens>
void folded_up_q5(SeparateOperands o, Q5Planes up, Scratch scratch, cudaStream_t stream) {
    using S               = FoldedUpSchedule<Features, Tokens>;
    constexpr auto kernel = up_q5_mix_kernel<Features, Tokens>;
    const int bytes       = bf16_prepare_shared<S::kSharedBytes, kernel>();
    for_each_token_slice(o.columns, Tokens, [&](int offset, int count) {
        const unsigned blocks = (kWidth / Features) * div_up(count, Tokens);
        kernel<<<blocks, S::kThreads, bytes, stream>>>(
            o, up, static_cast<const __nv_bfloat16*>(scratch.activation), offset);
        CUDA_CHECK(cudaGetLastError());
    });
}
} // namespace

void launch_separate(SeparateWeights weights, SeparateOperands o, Scratch scratch,
                     DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    const auto down           = q5_planes(weights.down, "down", kRank, kResidualWidth);
    launch_q5_a16_simt<DownQ5Schedule>(
        operands(o.normalized, weights.down, down, o.columns),
        RankActivationOutput{static_cast<__nv_bfloat16*>(scratch.activation)},
        LinearIdentityEpilogue{}, stream);
    if (weights.inject != nullptr) {
        const auto inject = q5_planes(*weights.inject, "inject", kStreams, kResidualWidth);
        launch_q5_a16_simt<InjectQ5Schedule>(
            operands(o.normalized, *weights.inject, inject, o.columns),
            InjectionOutput{o.injection}, LinearIdentityEpilogue{}, stream);
    }
    const auto up = q5_planes(weights.up, "up", kResidualWidth, kRank);
    if (o.columns <= 32)
        folded_up_q5<16, 16>(o, up, scratch, stream);
    else
        folded_up_q5<32, 32>(o, up, scratch, stream);
}
} // namespace ninfer::ops::detail::hc::mix
