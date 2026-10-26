#include "ops/moe/kernels/cuda/groupwise_int/grouped_gate_up.cuh"
#include "ops/moe/kernels/cuda/groupwise_int/grouped_gate_up_i8_tile.cuh"
#include "ops/moe/kernels/cuda/groupwise_int/quantize_a8.cuh"
#include "ops/moe/kernels/cuda/groupwise_int/grouped_down.cuh"

#include <algorithm>

namespace ninfer::ops::detail::moe_cuda::groupwise_int {
namespace {

template <class Shape, class Output>
__global__ void reduce_assignments(const __nv_bfloat16* values, const int* inverse,
                                   const float* coefficients, Output output,
                                   const GroupedInfo* info) {
    if (info->direct) { return; }
    __shared__ int columns[Shape::top_k];
    __shared__ float weights[Shape::top_k];
    const int token = blockIdx.x, tid = threadIdx.x;
    if (tid < Shape::top_k) {
        columns[tid] = inverse[token * Shape::top_k + tid];
        weights[tid] = coefficients[token * Shape::top_k + tid];
    }
    __syncthreads();
    constexpr int Items = Shape::hidden / 256;
    float sum[Items]    = {};
#pragma unroll
    for (int route = 0; route < Shape::top_k; ++route) {
        const int column = columns[route];
        if (column >= 0) {
            common::Bf16x8Bits packed;
            packed.raw = common::load_vec<uint4>(values + std::int64_t(column) * Shape::hidden +
                                                 tid * Items);
#pragma unroll
            for (int pair = 0; pair < Items / 2; ++pair) {
                const float2 decoded = __bfloat1622float2(packed.pair[pair]);
                sum[pair * 2] += weights[route] * decoded.x;
                sum[pair * 2 + 1] += weights[route] * decoded.y;
            }
        }
    }
#pragma unroll
    for (int item = 0; item < Items; ++item) {
        const auto index = std::int64_t(token) * Shape::hidden + tid * Items + item;
        output.store(index, output.initial(index) + sum[item]);
    }
}

template <class Function>
void with_column_tile(int columns, Function&& launch) {
    switch (columns) {
    case 32:
        launch.template operator()<32>();
        break;
    case 64:
        launch.template operator()<64>();
        break;
    default:
        throw std::logic_error("invalid Q8 tile");
    }
}

void shared_gate_up(const Tensor& x, const ScaledSwiGLUTerm& shared, const ExpertStorage& storage,
                    const GroupedInfo* info, int columns, cudaStream_t stream) {
    using S          = SwiGLUH2048I512E256K8;
    auto* activation = static_cast<__nv_bfloat16*>(storage.shared_activation);
    with_column_tile(columns, [&]<int N> {
        grouped_q8_gate_up<S, PackedExpertMatrix, false, N>
            <<<dim3(S::intermediate / 32, (x.ne[1] + N - 1) / N), N * 4, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data), nullptr, nullptr, nullptr, nullptr,
                matrix(shared.weights.gate_up.weight), activation,
                activation + S::intermediate * x.ne[1], x.ne[1], info);
    });
    CUDA_CHECK(cudaGetLastError());
}

void shared_down(const Tensor& x, const ScaledSwiGLUTerm& shared, Tensor& destination,
                 MoeEpilogue epilogue, const ExpertStorage& storage, const GroupedInfo* info,
                 int columns, cudaStream_t stream) {
    using S                = SwiGLUH2048I512E256K8;
    const auto* activation = static_cast<const __nv_bfloat16*>(storage.shared_activation);
    with_output(destination, epilogue, [&](auto output) {
        with_column_tile(columns, [&]<int N> {
            grouped_q8_down<S, PackedExpertMatrix, decltype(output), false, N>
                <<<dim3(S::hidden / 64, (x.ne[1] + N - 1) / N), N * 4, 0, stream>>>(
                    activation, activation + S::intermediate * x.ne[1], nullptr, nullptr, nullptr,
                    matrix(shared.weights.down.weight), nullptr, storage.routed_sum,
                    static_cast<const float*>(shared.coefficients.data), output, x.ne[1], info);
        });
        CUDA_CHECK(cudaGetLastError());
    });
}
} // namespace

void launch_grouped_swiglu(const Tensor& x, const RoutedSwiGLUProfile& profile,
                           const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                           const ScaledSwiGLUTerm* shared, MoeEpilogue epilogue,
                           Tensor& destination, const GroupedPlan& plan,
                           const ExpertStorage& storage, CudaExecutionView execution) {
    using S               = SwiGLUH2048I512E256K8;
    const auto stream     = execution.stream;
    const auto& group     = storage.groups;
    const auto* input     = static_cast<const __nv_bfloat16*>(x.data);
    auto* activation      = static_cast<__nv_bfloat16*>(storage.grouped_activation);
    auto* values          = static_cast<__nv_bfloat16*>(storage.grouped_io);
    const int limit       = execution.multiprocessor_count * plan.queued_ctas_per_sm;
    const int jobs        = S::top_k * x.ne[1] / plan.columns + S::experts;
    const int gate_blocks = std::min(jobs * (S::intermediate / 32), limit);
    const int down_blocks = std::min(jobs * (S::hidden / 64), limit);
    with_expert_addresses(experts, profile, [&](auto gate_up, auto down) {
        using Access = decltype(gate_up);
        // Temporary probe: how the prefill MoE time splits between the two halves. Synchronising per
        // call perturbs the absolute numbers but keeps the ratio honest, which is all it is for.
        static const bool split_timing = std::getenv("NINFER_MOE_SPLIT_TIMING") != nullptr;
        static cudaEvent_t s_ev[3]     = {nullptr, nullptr, nullptr};
        static double s_gu = 0.0, s_dn = 0.0;
        static long s_calls = 0;
        if (split_timing && s_ev[0] == nullptr) {
            for (auto& ev : s_ev) { (void)cudaEventCreate(&ev); }
        }
        if (split_timing) { (void)cudaEventRecord(s_ev[0], stream); }
        // port (sm_89): int8 mma route for both prefill tiles (32 columns below 768 tokens, 64 at or
        // above). Off by default; the split PPL gate has to pass against bf16 before it can default.
        static const bool int8_mma = [] {
            const char* v = std::getenv("NINFER_MOE_INT8_MMA");
            return v != nullptr && *v != '\0' && std::strtol(v, nullptr, 10) != 0;
        }();
        if (int8_mma && plan.columns <= 64 && profile.ffn.gate_up_type == QType::Q4_G64_FP16) {
            launch_quantize_a8(input, storage.activation_i8, storage.activation_scale, S::hidden,
                               x.ne[1], stream);
            CUDA_CHECK(cudaGetLastError());
            const dim3 grid(S::intermediate / 16, jobs);
            if (plan.columns == 64) {
                grouped_gate_up_i8_tile<S::hidden, 2 * S::intermediate, 64, Access>
                    <<<grid, 256, 0, stream>>>(storage.activation_i8, storage.activation_scale,
                                               gate_up, group.offsets, group.tokens,
                                               group.job_experts, group.job_columns, group.info,
                                               S::intermediate, activation);
            } else {
                grouped_gate_up_i8_tile<S::hidden, 2 * S::intermediate, 32, Access>
                    <<<grid, 128, 0, stream>>>(storage.activation_i8, storage.activation_scale,
                                               gate_up, group.offsets, group.tokens,
                                               group.job_experts, group.job_columns, group.info,
                                               S::intermediate, activation);
            }
            CUDA_CHECK(cudaGetLastError());
        } else if (profile.ffn.gate_up_type == QType::Q4_G64_FP16) {
            if (plan.columns == 64) {
                grouped_q4_gate_up<S, Access, 8, 64, 2, GateUpDensity::Any>
                    <<<gate_blocks, 256, 0, stream>>>(input, group.tokens, group.offsets,
                                                      group.job_experts, group.job_columns,
                                                      group.info, gate_up, activation);
            } else {
                grouped_q4_gate_up<S, Access, 4, 32, 2, GateUpDensity::Packed>
                    <<<gate_blocks, 128, 0, stream>>>(input, group.tokens, group.offsets,
                                                      group.job_experts, group.job_columns,
                                                      group.info, gate_up, activation);
                CUDA_CHECK(cudaGetLastError());
                grouped_q4_gate_up<S, Access, 4, 32, 6, GateUpDensity::Spread>
                    <<<gate_blocks, 128, 0, stream>>>(input, group.tokens, group.offsets,
                                                      group.job_experts, group.job_columns,
                                                      group.info, gate_up, activation);
            }
        } else {
            with_column_tile(plan.columns, [&]<int N> {
                grouped_q8_gate_up<S, Access, true, N><<<gate_blocks, N * 4, 0, stream>>>(
                    input, group.tokens, group.offsets, group.job_experts, group.job_columns,
                    gate_up, activation, activation + S::intermediate * S::top_k * x.ne[1], x.ne[1],
                    group.info);
            });
        }
        CUDA_CHECK(cudaGetLastError());
        if (split_timing) { (void)cudaEventRecord(s_ev[1], stream); }
        if (shared) {
            shared_gate_up(x, *shared, storage, group.info, plan.shared_columns, stream);
        }
        const auto integer_down = [&]<class Codec>() {
            if (plan.columns == 64) {
                grouped_integer_down<S, Codec, Access, 8, 64>
                    <<<down_blocks, 256, 0, stream>>>(activation, group.offsets, group.job_experts,
                                                      group.job_columns, group.info, down, values);
            } else {
                grouped_integer_down<S, Codec, Access, 4, 32>
                    <<<down_blocks, 128, 0, stream>>>(activation, group.offsets, group.job_experts,
                                                      group.job_columns, group.info, down, values);
            }
        };
        switch (profile.ffn.down_type) {
        case QType::Q5_G64_FP16:
            integer_down.template operator()<Q5DownMma>();
            break;
        case QType::Q6_G64_FP16:
            integer_down.template operator()<Q6DownMma>();
            break;
        default:
            with_column_tile(plan.columns, [&]<int N> {
                grouped_q8_down<S, Access, FloatContribution, true, N>
                    <<<down_blocks, N * 4, 0, stream>>>(
                        activation, activation + S::intermediate * S::top_k * x.ne[1],
                        group.offsets, group.job_experts, group.job_columns, down, values, nullptr,
                        nullptr, FloatContribution{}, x.ne[1], group.info);
            });
            break;
        }
        CUDA_CHECK(cudaGetLastError());
        if (split_timing) {
            (void)cudaEventRecord(s_ev[2], stream);
            (void)cudaEventSynchronize(s_ev[2]);
            float gu = 0.0F, dn = 0.0F;
            (void)cudaEventElapsedTime(&gu, s_ev[0], s_ev[1]);
            (void)cudaEventElapsedTime(&dn, s_ev[1], s_ev[2]);
            s_gu += gu;
            s_dn += dn;
            if (++s_calls % 40 == 0) {
                std::fprintf(stderr, "[split] columns=%d gate_up %.3f ms  down %.3f ms (per call)\n",
                             plan.columns, s_gu / double(s_calls), s_dn / double(s_calls));
            }
        }
    });
    const auto reduce = [&](auto output) {
        reduce_assignments<S><<<x.ne[1], 256, 0, stream>>>(
            values, group.inverse, static_cast<const float*>(routes.coefficients.data), output,
            group.info);
        CUDA_CHECK(cudaGetLastError());
    };
    if (shared) {
        reduce(FloatContribution{storage.routed_sum});
        shared_down(x, *shared, destination, epilogue, storage, group.info, plan.shared_columns,
                    stream);
    } else {
        with_output(destination, epilogue, reduce);
    }
}

void launch_shared_swiglu(const Tensor& x, const ScaledSwiGLUTerm& shared, MoeEpilogue epilogue,
                          Tensor& destination, const ExpertStorage& storage, const SharedPlan& plan,
                          CudaExecutionView execution) {
    if (plan.columns) {
        shared_gate_up(x, shared, storage, nullptr, plan.columns, execution.stream);
        shared_down(x, shared, destination, epilogue, storage, nullptr, plan.columns,
                    execution.stream);
    } else {
        launch_token_swiglu(x, {}, {}, {}, {ExpertSubsetKind::None, {}}, &shared, epilogue,
                            destination, plan.token, storage, {}, execution);
    }
}

} // namespace ninfer::ops::detail::moe_cuda::groupwise_int
