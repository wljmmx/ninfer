// The GGML-encoded routed SwiGLU: one CTA per token, the three expert banks read in their stored
// blocks. This is the correctness-first core of the expert path - the router, the subset masks and the
// shared-expert term stay outside it - so its shape is chosen for being checkable rather than for
// throughput: the CTA walks its token's routes serially (no atomics, no route-order dependence) and
// each thread owns whole output rows for the down pass.
//
// Numerics: gate and up are accumulated in binary32 and their product is rounded to BF16 before the
// down pass, which is the operand the native family hands the down matrix too. `out` is BF16 [H, T];
// tests/ops/test_ggml_experts.cu bounds it against a host reference built on core/ggml_dequant.h
// (itself pinned to gguf-py).

#include "ops/moe/ggml/ggml_experts.h"

#include "core/device.h"
#include "core/ggml_types.h"
#include "ops/linear/ggml/ggml_gemv.cuh"

#include <cuda_bf16.h>
#include <stdexcept>

namespace ninfer::ops::detail::ggml {
namespace {

int type_index(QType format) {
    switch (format) {
    case QType::GGML_Q4_K: return 0;
    case QType::GGML_Q5_K: return 1;
    case QType::GGML_Q6_K: return 2;
    case QType::GGML_Q2_0: return 3;
    case QType::GGML_IQ4_NL: return 4;
    case QType::GGML_IQ4_XS: return 5;
    case QType::GGML_IQ2_XXS: return 6;
    case QType::GGML_IQ2_XS: return 7;
    case QType::GGML_IQ2_S: return 8;
    case QType::GGML_IQ3_XXS: return 9;
    case QType::GGML_IQ3_S: return 10;
    default: throw std::invalid_argument("routed swiglu: not a GGML block encoding");
    }
}

__device__ __forceinline__ float row_dot(int type, const std::uint8_t* row,
                                         const __nv_bfloat16* x, int k, int groups_per_block,
                                         int type_bytes) {
    switch (type) {
    case 0: return lane_dot<0>(row, x, k, groups_per_block, type_bytes);
    case 1: return lane_dot<1>(row, x, k, groups_per_block, type_bytes);
    case 2: return lane_dot<2>(row, x, k, groups_per_block, type_bytes);
    case 3: return lane_dot<3>(row, x, k, groups_per_block, type_bytes);
    case 4: return lane_dot<4>(row, x, k, groups_per_block, type_bytes);
    case 5: return lane_dot<5>(row, x, k, groups_per_block, type_bytes);
    case 6: return lane_dot<6>(row, x, k, groups_per_block, type_bytes);
    case 7: return lane_dot<7>(row, x, k, groups_per_block, type_bytes);
    case 8: return lane_dot<8>(row, x, k, groups_per_block, type_bytes);
    case 9: return lane_dot<9>(row, x, k, groups_per_block, type_bytes);
    case 10: return lane_dot<10>(row, x, k, groups_per_block, type_bytes);
    }
    return 0.0f;
}

struct Bank {
    const std::uint8_t* data;
    int row_bytes;
    int groups_per_block;
    int type_bytes;
    int type;
};

__device__ __forceinline__ const std::uint8_t* row_at(const Bank& bank, int row) {
    return bank.data + std::size_t(row) * std::size_t(bank.row_bytes);
}

__global__ void routed_swiglu_kernel(const __nv_bfloat16* x, int hidden, int intermediate,
                                     Bank gate, Bank up, Bank down, const std::int32_t* ids,
                                     const float* coefficients, int top_k, int tokens,
                                     int experts, __nv_bfloat16* out, float* scratch) {
    // Scratch layout: top_k * intermediate BF16 products, then top_k floats of coefficients.
    __nv_bfloat16* product = reinterpret_cast<__nv_bfloat16*>(scratch);
    float* weights =
        reinterpret_cast<float*>(product + std::size_t(top_k) * std::size_t(intermediate));

    const int token         = blockIdx.x;
    const int thread        = threadIdx.x;
    const int threads       = blockDim.x;
    const __nv_bfloat16* xt = x + std::size_t(token) * std::size_t(hidden);

    for (int route = 0; route < top_k; ++route) {
        const int expert = ids[route * tokens + token];
        weights[route]   = coefficients[route * tokens + token];
        const bool valid = expert >= 0 && expert < experts;
        __nv_bfloat16* dst = product + std::size_t(route) * std::size_t(intermediate);
        for (int i = thread; i < intermediate; i += threads) {
            if (!valid) {
                // Out-of-domain route: contribute nothing rather than reading outside a bank.
                dst[i] = __float2bfloat16(0.0f);
                continue;
            }
            const float g =
                row_dot(gate.type, row_at(gate, expert * intermediate + i), xt, hidden,
                        gate.groups_per_block, gate.type_bytes);
            const float u = row_dot(up.type, row_at(up, expert * intermediate + i), xt, hidden,
                                    up.groups_per_block, up.type_bytes);
            dst[i] = __float2bfloat16((g / (1.0f + __expf(-g))) * u);
        }
        __syncthreads();
    }

    for (int d = thread; d < hidden; d += threads) {
        float acc = 0.0f;
        for (int route = 0; route < top_k; ++route) {
            const int expert = ids[route * tokens + token];
            if (expert < 0 || expert >= experts) { continue; }
            acc += weights[route] *
                   row_dot(down.type, row_at(down, expert * hidden + d),
                           product + std::size_t(route) * std::size_t(intermediate), intermediate,
                           down.groups_per_block, down.type_bytes);
        }
        out[std::size_t(token) * std::size_t(hidden) + d] = __float2bfloat16(acc);
    }
}

Bank describe(const Weight& weight, int width) {
    const auto geometry = core::ggml_block_geometry(weight.qtype);
    if (width <= 0 || weight.k != width) {
        throw std::invalid_argument("routed swiglu: bank width disagrees with the profile");
    }
    if (std::int64_t(width) % std::int64_t(geometry.block) != 0 || geometry.block % 32 != 0) {
        throw std::invalid_argument("routed swiglu: bank width is not whole GGML blocks");
    }
    return Bank{static_cast<const std::uint8_t*>(weight.qdata),
                int(weight.k / int(geometry.block) * int(geometry.type_bytes)),
                int(geometry.block) / 32, int(geometry.type_bytes), type_index(weight.qtype)};
}

} // namespace

std::size_t routed_swiglu_workspace_bytes(int top_k, int intermediate) {
    if (top_k <= 0 || intermediate <= 0) { return 0; }
    const std::size_t product = std::size_t(top_k) * std::size_t(intermediate) * 2;
    return ((product + 255) / 256) * 256 + std::size_t(top_k) * sizeof(float) + 256;
}

void launch_routed_swiglu(const Tensor& x, const ExpertBanks& banks, int experts, int top_k,
                          const Tensor& ids, const Tensor& coefficients, Tensor& out,
                          WorkspaceArena& workspace, cudaStream_t stream) {
    if (banks.gate == nullptr || banks.up == nullptr || banks.down == nullptr) {
        throw std::invalid_argument("routed swiglu: a bank is missing");
    }
    const int hidden       = banks.gate->k;
    const int intermediate = banks.gate->n / experts;
    const int tokens       = x.ne[1];
    if (hidden <= 0 || intermediate <= 0 || tokens <= 0 || top_k <= 0 || experts <= 0) {
        throw std::invalid_argument("routed swiglu: empty geometry");
    }
    if (banks.gate->n != experts * intermediate || banks.up->n != banks.gate->n ||
        banks.up->k != hidden || banks.down->n != experts * hidden ||
        banks.down->k != intermediate) {
        throw std::invalid_argument("routed swiglu: the three banks are not one expert set");
    }
    if (banks.gate->qdata == nullptr || banks.up->qdata == nullptr ||
        banks.down->qdata == nullptr) {
        throw std::invalid_argument("routed swiglu: a bank has no payload");
    }

    const Bank gate = describe(*banks.gate, hidden);
    const Bank up   = describe(*banks.up, hidden);
    const Bank down = describe(*banks.down, intermediate);

    const auto scratch = workspace.alloc_bytes(routed_swiglu_workspace_bytes(top_k, intermediate));
    const int threads  = 128;
    routed_swiglu_kernel<<<tokens, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), hidden, intermediate, gate, up, down,
        static_cast<const std::int32_t*>(ids.data), static_cast<const float*>(coefficients.data),
        top_k, tokens, experts, static_cast<__nv_bfloat16*>(out.data),
        static_cast<float*>(scratch.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail::ggml
