#include "models/qwen3_5/execution/ffn.h"

#include "models/qwen3_5/execution/expert_stream.h"

#include "core/layout.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/act_quant_a8.h"
#include <cstdio>
#include <cuda_bf16.h>
#include <algorithm>
#include <vector>
#include "ninfer/ops/w4a8.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/silu_mul.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {
namespace {

// port (sm_89): the streamed routed path runs routing, a host round trip and the expert call as
// three steps, so its scratch is the route tensors plus the larger of the two Op capacities.
std::size_t stream_route_bytes(const RoutedMoeParameters& moe, std::int32_t first,
                               std::int32_t last) {
    const auto& routed = moe.profile.routed;
    return ops::topk_softmax_route_workspace_capacity_bytes(
        {routed.ffn.hidden, routed.experts, routed.top_k}, first, last);
}

std::size_t stream_expert_bytes(const RoutedMoeParameters& moe, std::int32_t first,
                                std::int32_t last) {
    return ops::routed_swiglu_workspace_capacity_bytes(
        moe.profile.routed, moe.profile.shared, ops::ExpertSubsetKind::All, DType::BF16,
        ops::MoeEpilogue::AddResidual, first, last);
}

std::size_t stream_route_tensor_bytes(const RoutedMoeParameters& moe, std::int32_t last) {
    const auto top_k  = static_cast<std::size_t>(moe.profile.routed.top_k);
    const auto tokens = static_cast<std::size_t>(last);
    return top_k * tokens * sizeof(std::int32_t) + top_k * tokens * sizeof(float) +
           tokens * sizeof(float) + 512;
}

void streamed_ffn(const Tensor& hidden, const RoutedMoeParameters& moe, Tensor& residual,
                  WorkspaceArena& workspace, DeviceExecutionView execution) {
    const auto columns       = hidden.ne[1];
    const auto& routed       = moe.profile.routed;
    auto ids                 = workspace.alloc(DType::I32, {routed.top_k, columns});
    auto coefficients        = workspace.alloc(DType::FP32, {routed.top_k, columns});
    auto shared_coefficients = workspace.alloc(DType::FP32, {columns});
    const auto scratch = workspace.alloc_bytes(std::max(stream_route_bytes(moe, columns, columns),
                                                        stream_expert_bytes(moe, columns, columns)));
    const ops::RouteProfile route_profile{routed.ffn.hidden, routed.experts, routed.top_k};
    ops::ExpertRoutes routes{ids, coefficients};
    try {
        MemoryArena arena(MemorySpan{scratch.data, scratch.bytes, execution.device()});
        ops::topk_softmax_route(hidden, route_profile, moe.control, routes, shared_coefficients,
                                arena, execution);
    } catch (const std::exception& error) {
        throw std::invalid_argument(std::string("expert stream route: ") + error.what());
    }
    try {
        moe.stream->ensure(ids.data, routed.top_k, columns, execution.stream);
    } catch (const std::exception& error) {
        throw std::invalid_argument(std::string("expert stream upload: ") + error.what());
    }
    try {
        // streamed bank reproduces for this artifact.
        MemoryArena arena(MemorySpan{scratch.data, scratch.bytes, execution.device()});
        // port (sm_89): routed and shared run in one combined plan - measured closest to the
        // resident path (7.094349 vs 7.096182 for two additive passes).
        ops::routed_swiglu(hidden, routed, moe.stream->experts(), routes, {},
                           ops::ScaledSwiGLUTerm{moe.shared, shared_coefficients},
                           ops::MoeEpilogue::AddResidual, residual, arena, execution);
    } catch (const std::exception& error) {
        throw std::invalid_argument(std::string("expert stream run: ") + error.what());
    }
}

} // namespace

std::size_t ffn_workspace_bytes(const FfnParameters& parameters, std::int32_t first,
                                std::int32_t last, bool mtp) {
    if (first <= 0 || last < first) { throw std::invalid_argument("FFN: invalid column interval"); }
    if (const auto* moe = std::get_if<ops::SparseMoeWeights>(&parameters)) {
        return ops::sparse_moe_workspace_capacity_bytes(moe->routed_gate_up.qtype,
                                                        moe->routed_down.qtype, first, last);
    }
    if (const auto* moe = std::get_if<RoutedMoeParameters>(&parameters)) {
        if (moe->stream) {
            return stream_route_tensor_bytes(*moe, last) +
                   std::max(stream_route_bytes(*moe, first, last),
                            stream_expert_bytes(*moe, first, last));
        }
        return ops::topk_softmax_swiglu_moe_workspace_capacity_bytes(
            moe->profile, ops::MoeEpilogue::AddResidual, first, last);
    }
    const auto& p    = std::get<DenseParameters>(parameters);
    const auto& gu   = p.gate_up.weight;
    const auto& down = p.down.weight;
    WorkspaceLayoutBuilder layout;
    if (mtp) {
        (void)layout.alloc(DType::BF16, {gu.n, last});
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(
                gu.qtype, gu.n, gu.k, p.gate_up.policy, first, last));
        }
        (void)layout.alloc(DType::BF16, {gu.n / 2, last});
        (void)layout.alloc(DType::BF16, {down.n, last});
        (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(down.qtype, down.n, down.k,
                                                                      p.down.policy, first, last));
    } else {
        (void)layout.alloc(DType::BF16, {gu.n / 2, last});
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(std::max(ops::linear_swiglu_workspace_capacity_bytes(gu.qtype, gu.n, gu.k,
                                                                   p.gate_up.policy, first, last),
                               ops::w4a8_swiglu_workspace_capacity_bytes(gu.n, gu.k, first, last)));
        }
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(std::max(ops::linear_add_workspace_capacity_bytes(down.qtype, down.n, down.k,
                                                                     p.down.policy, first, last),
                               ops::w4a8_linear_add_workspace_capacity_bytes(down.n, down.k, first,
                                                                            last)));
        }
    }
    return layout.peak_bytes(1);
}

void ffn(const Tensor& hidden, const FfnParameters& parameters, Tensor& residual,
         const ops::SparseMoeHints& hints, WorkspaceArena& workspace, DeviceExecutionView execution,
         bool mtp) {
    const auto stream  = execution.stream;
    auto scope         = workspace.scope();
    const auto columns = hidden.ne[1];
    if (const auto* moe = std::get_if<ops::SparseMoeWeights>(&parameters)) {
        const auto storage =
            workspace.alloc_bytes(ffn_workspace_bytes(parameters, columns, columns));
        WorkspaceArena scratch(storage);
        ops::sparse_moe(hidden, *moe, ops::SparseMoeEpilogue::AddResidual, residual, hints, scratch,
                        execution);
        return;
    }
    if (const auto* moe = std::get_if<RoutedMoeParameters>(&parameters)) {
        // [probe] W4A8 quantises the routed A operand to int8 per token. Round-trip the MoE input
        // through that scale so the perplexity cost can be measured before any kernel changes.
        ops::act_quant_a8_roundtrip_bf16(hidden.data, hidden.ne[0], columns, stream);
        if (moe->stream) {
            streamed_ffn(hidden, *moe, residual, workspace, execution);
            return;
        }
        const auto storage =
            workspace.alloc_bytes(ffn_workspace_bytes(parameters, columns, columns));
        MemoryArena scratch(MemorySpan{storage.data, storage.bytes, execution.device()});
        ops::topk_softmax_swiglu_moe(hidden, moe->profile,
                                     ops::MoeWeights{moe->control, moe->routed, moe->shared},
                                     ops::MoeEpilogue::AddResidual, residual, scratch, execution);
        return;
    }
    const auto& p    = std::get<DenseParameters>(parameters);
    const auto& gu   = p.gate_up.weight;
    const auto& down = p.down.weight;
    if (mtp) {
        Tensor gate_up = workspace.alloc(DType::BF16, {gu.n, columns});
        {
            auto call = workspace.scope();
            ops::linear(hidden, gu, gate_up, p.gate_up.policy, workspace, stream);
        }
        Tensor activation = workspace.alloc(DType::BF16, {gu.n / 2, columns});
        ops::silu_mul(gate_up.slice(0, 0, gu.n / 2), gate_up.slice(0, gu.n / 2, gu.n / 2),
                      activation, stream);
        Tensor delta = workspace.alloc(DType::BF16, {down.n, columns});
        ops::linear(activation, down, delta, p.down.policy, workspace, stream);
        ops::residual_add(delta, residual, stream);
        return;
    }
    Tensor activation = workspace.alloc(DType::BF16, {gu.n / 2, columns});
    {
        auto call = workspace.scope();
        if (ops::w4a8_swiglu_supported(gu, hidden.ne[1])) {
            ops::w4a8_swiglu(hidden, gu, activation, workspace, stream);
        } else {
            ops::linear_swiglu(hidden, gu, activation, p.gate_up.policy, workspace, stream);
        }
    }
    // [P1 probe] A8 round-trip on the down-projection activation
    ops::act_quant_a8_roundtrip_bf16(activation.data, activation.ne[0], activation.ne[1], stream);
    // [P1] W4A8 route for the down projection when enabled (NINFER_W4A8)
    if (ops::w4a8_supported(down, activation.ne[1])) {
        ops::w4a8_linear_add(activation, down, residual, workspace, stream);
        return;
    }
    ops::linear_add(activation, down, residual, p.down.policy, workspace, stream);
}

} // namespace ninfer::models::qwen3_5::execution
