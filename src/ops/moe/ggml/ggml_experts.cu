// The GGML-encoded routed SwiGLU: one CTA per token, the expert banks read in their stored blocks.
//
// Shape of the work, per token: the routes are walked serially (no atomics, no route-order
// dependence), each route's gate/up rows are decoded in place, the SiLU product is rounded to BF16,
// and the down pass accumulates the coefficient-weighted result. The shared expert, when the layer has
// one, appends its own term inside the same CTA.
//
// Numerics: gate/up accumulate in binary32; the product is rounded to BF16 before the down pass, which
// is the operand the native family hands the down matrix too. `out` is BF16 [H, T].
// tests/ops/test_ggml_experts.cu bounds it against a host reference built on core/ggml_dequant.h
// (itself pinned to gguf-py).

#include "ops/moe/ggml/ggml_experts.h"

#include "core/device.h"
#include "core/ggml_types.h"
#include "ops/linear/ggml/ggml_gemv.cuh"

#include <cuda_bf16.h>
#include <stdexcept>
#include <string>

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

// The first row of expert `expert` in this bank, on whichever leg serves it.
__device__ __forceinline__ const std::uint8_t* expert_base(const ExpertBank& bank, int expert,
                                                           int64_t& row_bytes_out) {
    row_bytes_out = bank.row_bytes;
    if (bank.host_flags != nullptr && bank.host_flags[expert] != 0) {
        return bank.host_codes + (bank.host_offsets != nullptr
                                      ? bank.host_offsets[expert]
                                      : std::uint64_t(expert) * bank.expert_stride);
    }
    return bank.codes + (bank.offsets != nullptr ? bank.offsets[expert]
                                                 : std::uint64_t(expert) * bank.expert_stride);
}

__device__ __forceinline__ bool participates(ExpertSubsetKind kind, const std::int32_t* mask,
                                             int expert) noexcept {
    switch (kind) {
    case ExpertSubsetKind::All: return true;
    case ExpertSubsetKind::None: return false;
    case ExpertSubsetKind::MaskOnes: return mask[expert] == 1;
    case ExpertSubsetKind::MaskZeros: return mask[expert] == 0;
    }
    return false;
}

// SiLU(g) * u, rounded exactly as the down matrix's operand is rounded on the native family.
__device__ __forceinline__ __nv_bfloat16 swiglu(float g, float u) {
    return __float2bfloat16((g / (1.0f + __expf(-g))) * u);
}

__global__ void routed_swiglu_kernel(const __nv_bfloat16* x, int hidden, int intermediate,
                                     ExpertBanks banks, const std::int32_t* ids,
                                     const float* coefficients, int top_k, int tokens, int experts,
                                     ExpertSubsetKind subset_kind, const std::int32_t* mask,
                                     MoeEpilogue mode, SharedExpert shared, __nv_bfloat16* out,
                                     float* scratch) {
    // Scratch layout: top_k * intermediate BF16 products, then top_k coefficients, then (when the
    // layer has a shared expert) intermediate products and one coefficient for it.
    const int slots        = top_k + (shared.gate_up != nullptr ? 1 : 0);
    __nv_bfloat16* product = reinterpret_cast<__nv_bfloat16*>(scratch);
    float* weights = reinterpret_cast<float*>(product + std::size_t(slots) * std::size_t(intermediate));

    const int token         = blockIdx.x;
    const int thread        = threadIdx.x;
    const int threads       = blockDim.x;
    const __nv_bfloat16* xt = x + std::size_t(token) * std::size_t(hidden);

    // One pass over `slots` rows of products: the token's routes, then the shared expert.
    for (int slot = 0; slot < slots; ++slot) {
        __nv_bfloat16* dst = product + std::size_t(slot) * std::size_t(intermediate);
        if (slot == top_k) {
            const float coef      = shared.coefficients->data != nullptr
                                        ? static_cast<const float*>(shared.coefficients->data)[token]
                                        : 0.0f;
            weights[slot]         = coef;
            int64_t gate_row      = 0;
            const std::uint8_t* gu = expert_base(*shared.gate_up, 0, gate_row);
            const std::uint8_t* up = gu + std::uint64_t(intermediate) * std::uint64_t(gate_row);
            for (int i = thread; i < intermediate; i += threads) {
                const float g = row_dot(shared.gate_up->type,
                                        gu + std::size_t(i) * std::size_t(gate_row), xt, hidden,
                                        shared.gate_up->groups_per_block, shared.gate_up->type_bytes);
                const float u = row_dot(shared.gate_up->type,
                                        up + std::size_t(i) * std::size_t(gate_row), xt, hidden,
                                        shared.gate_up->groups_per_block, shared.gate_up->type_bytes);
                dst[i]        = swiglu(g, u);
            }
            __syncthreads();
            continue;
        }
        const int expert   = ids[token * top_k + slot];
        const float coef   = coefficients[token * top_k + slot];
        weights[slot]      = coef;
        const bool valid   = expert >= 0 && expert < experts &&
                           participates(subset_kind, mask, expert);
        for (int i = thread; i < intermediate; i += threads) {
            if (!valid) {
                dst[i] = __float2bfloat16(0.0f);
                continue;
            }
            int64_t gate_row      = 0;
            int64_t up_row        = 0;
            const std::uint8_t* g = expert_base(banks.gate, expert, gate_row);
            const std::uint8_t* u = expert_base(banks.up, expert, up_row);
            const float gv = row_dot(banks.gate.type, g + std::size_t(i) * std::size_t(gate_row), xt,
                                     hidden, banks.gate.groups_per_block, banks.gate.type_bytes);
            const float uv = row_dot(banks.up.type, u + std::size_t(i) * std::size_t(up_row), xt,
                                     hidden, banks.up.groups_per_block, banks.up.type_bytes);
            dst[i]         = swiglu(gv, uv);
        }
        __syncthreads();
    }

    for (int d = thread; d < hidden; d += threads) {
        float acc = 0.0f;
        for (int slot = 0; slot < slots; ++slot) {
            const __nv_bfloat16* src = product + std::size_t(slot) * std::size_t(intermediate);
            if (slot == top_k) {
                int64_t down_row      = 0;
                const std::uint8_t* r = expert_base(*shared.down, 0, down_row);
                acc += weights[slot] * row_dot(shared.down->type,
                                               r + std::size_t(d) * std::size_t(down_row), src,
                                               intermediate, shared.down->groups_per_block,
                                               shared.down->type_bytes);
                continue;
            }
            const int expert = ids[token * top_k + slot];
            if (expert < 0 || expert >= experts ||
                !participates(subset_kind, mask, expert)) {
                continue;
            }
            int64_t down_row      = 0;
            const std::uint8_t* r = expert_base(banks.down, expert, down_row);
            acc += weights[slot] * row_dot(banks.down.type,
                                           r + std::size_t(d) * std::size_t(down_row), src,
                                           intermediate, banks.down.groups_per_block,
                                           banks.down.type_bytes);
        }
        const std::size_t index = std::size_t(token) * std::size_t(hidden) + d;
        out[index] = mode == MoeEpilogue::Write
                         ? __float2bfloat16(acc)
                         : __float2bfloat16(__bfloat162float(out[index]) + acc);
    }
}

} // namespace

ExpertBank bank_from(const ExpertMatrixDeviceView& view, int k, QType qtype) {
    const auto geometry = core::ggml_block_geometry(qtype);
    if (k <= 0 || view.codes == nullptr) {
        throw std::invalid_argument("routed swiglu: an expert bank has no width or no payload");
    }
    if (std::int64_t(k) % std::int64_t(geometry.block) != 0 || geometry.block % 32 != 0) {
        throw std::invalid_argument("routed swiglu: a bank's width is not whole GGML blocks");
    }
    ExpertBank bank{};
    bank.codes            = reinterpret_cast<const std::uint8_t*>(view.codes);
    bank.host_codes       = reinterpret_cast<const std::uint8_t*>(view.host_codes);
    bank.offsets          = view.offsets;
    bank.host_offsets     = view.host_offsets;
    bank.host_flags       = view.host_flags;
    bank.expert_stride    = view.code_stride;
    bank.groups_per_block = std::int32_t(geometry.block) / 32;
    bank.type_bytes       = std::int32_t(geometry.type_bytes);
    bank.type             = type_index(qtype);
    // One row of k values, in this encoding.
    bank.row_bytes = std::int32_t(std::int64_t(k) / std::int64_t(geometry.block) *
                                  std::int64_t(geometry.type_bytes));
    return bank;
}

std::size_t routed_swiglu_workspace_bytes(int top_k, int intermediate) {
    if (top_k <= 0 || intermediate <= 0) { return 0; }
    // top_k + 1 slots: the routes, plus the shared expert's product when a layer has one.
    const std::size_t product = std::size_t(top_k + 1) * std::size_t(intermediate) * 2;
    return ((product + 255) / 256) * 256 + std::size_t(top_k + 1) * sizeof(float) + 256;
}

void launch_routed_swiglu(const Tensor& x, const ExpertBanks& banks, int hidden, int intermediate,
                          int experts, int top_k, const ExpertSubset& subset, MoeEpilogue mode,
                          const Tensor& ids, const Tensor& coefficients, const SharedExpert* shared,
                          Tensor& out, MemoryArena& workspace, cudaStream_t stream) {
    if (experts <= 0 || top_k <= 0 || hidden <= 0 || intermediate <= 0) {
        throw std::invalid_argument("routed swiglu: empty profile");
    }
    if (banks.gate.codes == nullptr || banks.up.codes == nullptr || banks.down.codes == nullptr) {
        throw std::invalid_argument("routed swiglu: a bank has no payload");
    }
    const int tokens = x.ne[1];
    if (tokens <= 0 || x.ne[0] != hidden) {
        throw std::invalid_argument("routed swiglu: the activation does not match the profile");
    }

    SharedExpert prepared{};
    if (shared != nullptr) {
        if (shared->gate_up == nullptr || shared->down == nullptr ||
            shared->coefficients == nullptr) {
            throw std::invalid_argument("routed swiglu: an incomplete shared term");
        }
        prepared = *shared;
    }

    // The pricer asks for `peak_bytes(1)`, so the request has to be the same shape: an extra
    // alignment pad here would exceed the arena the plan reserved. The helper already pads its own
    // 256-byte boundaries, and the arena base is aligned by the caller.
    const auto scratch = workspace.alloc_bytes(routed_swiglu_workspace_bytes(top_k, intermediate), 1);
    const int threads  = 128;
    routed_swiglu_kernel<<<tokens, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), hidden, intermediate, banks,
        static_cast<const std::int32_t*>(ids.data), static_cast<const float*>(coefficients.data),
        top_k, tokens, experts, subset.kind,
        subset.mask.data != nullptr ? static_cast<const std::int32_t*>(subset.mask.data) : nullptr,
        mode, prepared, static_cast<__nv_bfloat16*>(out.data), static_cast<float*>(scratch.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail::ggml
