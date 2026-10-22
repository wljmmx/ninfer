#include "models/qwen3_5/execution/ffn.h"

#include "models/qwen3_5/execution/expert_stream.h"
#include "models/qwen3_5/execution/host_expert_residency.h"
#include "core/cpu/executor.h"
#include "core/cpu/system.h"
#include <cstring>

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

// Whole-layer host fallback, used when the Device pool cannot hold the layer's expert set at all.
// The routed term is computed on the CPU over the host tier's experts and added to the residual;
// the CPU form deliberately excludes residual addition, so that happens here.
void host_routed_ffn(const Tensor& hidden, const RoutedMoeParameters& moe,
                     const ops::ExpertRoutes& routes, Tensor& residual, WorkspaceArena& workspace,
                     DeviceExecutionView execution) {
    auto* host = moe.stream->host();
    if (host == nullptr) { throw std::runtime_error("host routed ffn: no host tier"); }
    const auto cols   = hidden.ne[0];
    const auto tokens = hidden.ne[1];
    const auto top_k  = moe.profile.routed.top_k;
    const auto stream = execution.stream;

    static CpuTopology topology;
    static CpuExecutor executor(topology, plan_cpu_team(topology.info(), 8));
    const auto cpu = executor.execution_view();

    std::vector<std::uint16_t> host_x(static_cast<std::size_t>(cols) * tokens);
    std::vector<std::int32_t> host_ids(static_cast<std::size_t>(top_k) * tokens);
    std::vector<float> host_coefficients(static_cast<std::size_t>(top_k) * tokens);
    std::vector<float> host_out(static_cast<std::size_t>(cols) * tokens);

    CUDA_CHECK(cudaMemcpyAsync(host_x.data(), hidden.data, host_x.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(host_ids.data(), routes.ids.data,
                               host_ids.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(host_coefficients.data(), routes.coefficients.data,
                               host_coefficients.size() * sizeof(float), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    Tensor x_cpu(host_x.data(), Device::cpu(), DType::BF16, {cols, tokens});
    Tensor out_cpu(host_out.data(), Device::cpu(), DType::FP32, {cols, tokens});
    ops::ExpertRoutes routes_cpu{Tensor(host_ids.data(), Device::cpu(), DType::I32, {top_k, tokens}),
                            Tensor(host_coefficients.data(), Device::cpu(), DType::FP32, {top_k, tokens})};

    // Publish every expert of this bank into the host arena before the CPU kernels read it.
    std::vector<std::int32_t> all(static_cast<std::size_t>(moe.stream->count()));
    for (std::size_t i = 0; i < all.size(); ++i) { all[i] = static_cast<std::int32_t>(i); }
    host->ensure(ExpertBankId{moe.stream->bank_index()}, all, nullptr);

    const auto bytes = ops::routed_swiglu_workspace_capacity_bytes(
        moe.profile.routed, ops::ExpertSubsetKind::All, tokens, tokens);
    // The CPU kernels want a 256-byte aligned cursor; a plain vector only guarantees 16.
    std::vector<std::byte> scratch(bytes + 256);
    const auto base    = reinterpret_cast<std::uintptr_t>(scratch.data());
    const auto aligned = (256 - (base & 255)) & 255;
    MemoryArena arena(
        MemorySpan{scratch.data() + aligned, bytes, Device::cpu()});
    ops::routed_swiglu(x_cpu, moe.profile.routed, host->experts(ExpertBankId{moe.stream->bank_index()}), routes_cpu,
                       ops::ExpertSubset{ops::ExpertSubsetKind::All, {}}, out_cpu, arena, cpu);
    if (std::getenv("NINFER_HOST_DEBUG")) {
        std::fprintf(stderr, "[host-x] bank=%u x0..3=%.6f %.6f %.6f %.6f\n",
                     moe.stream->bank_index(),
                     __bfloat162float(__ushort_as_bfloat16(host_x[0])),
                     __bfloat162float(__ushort_as_bfloat16(host_x[1])),
                     __bfloat162float(__ushort_as_bfloat16(host_x[2])),
                     __bfloat162float(__ushort_as_bfloat16(host_x[3])));
    }
    if (std::getenv("NINFER_HOST_DEBUG")) {
        const auto& gu = host->experts(ExpertBankId{moe.stream->bank_index()}).gate_up;
        const auto probe = gu.plane(gu.layout.codes, static_cast<std::uint32_t>(host_ids[0]));
        std::fprintf(stderr, "[host] bank=%u base=%p plane=%p off=%llu delta=%lld first4=%02x %02x %02x %02x\n",
                     moe.stream->bank_index(), static_cast<const void*>(gu.parent->data),
                     probe.data,
                     static_cast<unsigned long long>(gu.expert_offsets[host_ids[0]]),
                     static_cast<long long>(
                         reinterpret_cast<std::intptr_t>(probe.data) -
                         reinterpret_cast<std::intptr_t>(gu.parent->data)),
                     probe.data ? static_cast<const std::uint8_t*>(probe.data)[0] : 0,
                     probe.data ? static_cast<const std::uint8_t*>(probe.data)[1] : 0,
                     probe.data ? static_cast<const std::uint8_t*>(probe.data)[2] : 0,
                     probe.data ? static_cast<const std::uint8_t*>(probe.data)[3] : 0);
    }

    auto delta = workspace.alloc(DType::BF16, {cols, tokens});
    std::vector<std::uint16_t> packed(host_out.size());
    for (std::size_t i = 0; i < host_out.size(); ++i) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &host_out[i], sizeof(bits));
        // Round to nearest even, the same rule __float2bfloat16_rn applies; truncating biases
        // every layer downward and compounds over the stack.
        const std::uint32_t lsb = (bits >> 16) & 1u;
        bits += 0x7fffu + lsb;
        packed[i] = static_cast<std::uint16_t>(bits >> 16);
    }
    CUDA_CHECK(cudaMemcpyAsync(delta.data, packed.data(), packed.size() * sizeof(std::uint16_t),
                               cudaMemcpyHostToDevice, stream));
    ops::residual_add(delta, residual, stream);
    if (std::getenv("NINFER_HOST_DEBUG")) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        float probe[4] = {};
        CUDA_CHECK(cudaMemcpy(probe, residual.data, sizeof(probe), cudaMemcpyDeviceToHost));
        std::fprintf(stderr, "[host-residual] bank=%u r0..3=%.6f %.6f %.6f %.6f\n",
                     moe.stream->bank_index(), probe[0], probe[1], probe[2], probe[3]);
    }
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
        if (!moe.stream->ensure(ids.data, routed.top_k, columns, execution.stream)) {
            // The pool cannot hold this layer: the shared expert still runs on the device (subset
            // None touches only the shared term), and the routed term moves to the host tier.
            MemoryArena shared_arena(MemorySpan{scratch.data, scratch.bytes, execution.device()});
            ops::routed_swiglu(hidden, routed, moe.stream->experts(), routes,
                               ops::ExpertSubset{ops::ExpertSubsetKind::None, {}},
                               ops::ScaledSwiGLUTerm{moe.shared, shared_coefficients},
                               ops::MoeEpilogue::AddResidual, residual, shared_arena, execution);
            host_routed_ffn(hidden, moe, routes, residual, workspace, execution);
            return;
        }
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
    if (std::getenv("NINFER_HOST_DEBUG")) {
        std::uint16_t host_hidden[4] = {};
        CUDA_CHECK(cudaMemcpy(host_hidden, hidden.data, sizeof(host_hidden),
                              cudaMemcpyDeviceToHost));
        std::fprintf(stderr, "[hidden] bank=%u x0..3=%.6f %.6f %.6f %.6f\n",
                     moe.stream->bank_index(),
                     __bfloat162float(__ushort_as_bfloat16(host_hidden[0])),
                     __bfloat162float(__ushort_as_bfloat16(host_hidden[1])),
                     __bfloat162float(__ushort_as_bfloat16(host_hidden[2])),
                     __bfloat162float(__ushort_as_bfloat16(host_hidden[3])));
        float probe[4] = {};
        CUDA_CHECK(cudaMemcpy(probe, residual.data, sizeof(probe), cudaMemcpyDeviceToHost));
        std::fprintf(stderr, "[residual] bank=%u r0..3=%.6f %.6f %.6f %.6f\n",
                     moe.stream->bank_index(), probe[0], probe[1], probe[2], probe[3]);
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
