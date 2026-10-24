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

#include <thread>

#include <stdexcept>
#include <cstdlib>
#include <optional>

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
// the CPU form deliberately excludes residual addition, so that happens here. The shared term
// arrives as FP32 from the device, and the three contributions are summed before a single BF16
// rounding, which is what the device epilogue does on the resident path.
// Everything the host half needs that the Device kernel does not produce. It is copied before that
// kernel is launched, so the CPU work can overlap with it instead of queueing behind it.
struct HostRoutedFetch {
    std::vector<std::uint16_t> x;
    std::vector<std::uint16_t> residual;
    std::vector<std::int32_t> ids;
    std::vector<float> coefficients;
};

HostRoutedFetch fetch_host_routed(const Tensor& hidden, const ops::ExpertRoutes& routes,
                                  const Tensor& residual, cudaStream_t stream) {
    const auto cols   = hidden.ne[0];
    const auto tokens = hidden.ne[1];
    const auto top_k  = routes.ids.ne[0];
    HostRoutedFetch fetch;
    fetch.x.resize(static_cast<std::size_t>(cols) * tokens);
    fetch.residual.resize(static_cast<std::size_t>(cols) * tokens);
    fetch.ids.resize(static_cast<std::size_t>(top_k) * tokens);
    fetch.coefficients.resize(static_cast<std::size_t>(top_k) * tokens);
    CUDA_CHECK(cudaMemcpyAsync(fetch.x.data(), hidden.data, fetch.x.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(fetch.residual.data(), residual.data,
                               fetch.residual.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(fetch.ids.data(), routes.ids.data,
                               fetch.ids.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(fetch.coefficients.data(), routes.coefficients.data,
                               fetch.coefficients.size() * sizeof(float), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return fetch;
}

void host_routed_ffn(const HostRoutedFetch& fetch, const RoutedMoeParameters& moe,
                     const ops::ExpertSubset& subset, const Tensor& gpu_term, Tensor& residual,
                     DeviceExecutionView execution) {
    auto* host = moe.stream->host();
    if (host == nullptr) { throw std::runtime_error("host routed ffn: no host tier"); }
    const auto cols   = residual.ne[0];
    const auto tokens = residual.ne[1];
    const auto top_k  = moe.profile.routed.top_k;
    const auto stream = execution.stream;

    static CpuTopology topology;
    // port (sm_89): the CPU half is the whole cost of a split layer, so give it every physical
    // core the machine has instead of a fixed team of eight. A 24-core part was leaving two
    // thirds of its throughput unused.
    static const auto cpu_team = std::max<std::size_t>(
        1, cpu_worker_order(topology.info(), false).size());
    static CpuExecutor executor(topology,
                                plan_cpu_team(topology.info(), static_cast<std::uint32_t>(cpu_team)));
    const auto cpu = executor.execution_view();
    // port (sm_89): the split path calls this collective once per layer, and a parked team costs a
    // wake plus two barrier turns each time. The bench that runs the same collective back to back
    // measures 0.05 ms per expert while the engine pays 0.15-0.3, and the difference is the wake,
    // not the arithmetic. NINFER_CPU_TEAM_ACTIVE keeps the workers polling across layers.
    static std::optional<CpuExecutionView::ActiveScope> sticky_team;
    if (std::getenv("NINFER_CPU_TEAM_ACTIVE") != nullptr && !sticky_team) {
        sticky_team.emplace(cpu.keep_active());
    }

    std::vector<float> host_out(static_cast<std::size_t>(cols) * tokens);
    std::vector<float> host_gpu(static_cast<std::size_t>(cols) * tokens);

    Tensor x_cpu(const_cast<std::uint16_t*>(fetch.x.data()), Device::cpu(), DType::BF16,
                 {cols, tokens});
    Tensor out_cpu(host_out.data(), Device::cpu(), DType::FP32, {cols, tokens});
    ops::ExpertRoutes routes_cpu{
        Tensor(const_cast<std::int32_t*>(fetch.ids.data()), Device::cpu(), DType::I32, {top_k, tokens}),
        Tensor(const_cast<float*>(fetch.coefficients.data()), Device::cpu(), DType::FP32,
               {top_k, tokens})};

    // Publish only the experts this half actually reads. Publishing the whole bank made a split
    // layer stream all 256 images from the artifact even though the CPU half routes to a subset.
    std::vector<std::int32_t> wanted;
    {
        const auto* mask = subset.kind == ops::ExpertSubsetKind::MaskZeros && subset.mask.data
                               ? static_cast<const std::int32_t*>(subset.mask.data)
                               : nullptr;
        std::vector<bool> seen(static_cast<std::size_t>(moe.stream->count()), false);
        for (std::size_t i = 0; i < fetch.ids.size(); ++i) {
            const auto expert = fetch.ids[i];
            if (expert < 0 || static_cast<std::size_t>(expert) >= seen.size()) { continue; }
            if (mask != nullptr && mask[expert] != 0) { continue; }
            if (seen[static_cast<std::size_t>(expert)]) { continue; }
            seen[static_cast<std::size_t>(expert)] = true;
            wanted.push_back(expert);
        }
    }
    const auto ensure_begin = std::chrono::steady_clock::now();
    host->ensure(ExpertBankId{moe.stream->bank_index()}, wanted, nullptr);
    const auto ensure_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - ensure_begin).count();
    const auto bytes = ops::routed_swiglu_workspace_capacity_bytes(
        moe.profile.routed, ops::ExpertSubsetKind::All, tokens, tokens);
    // The CPU kernels want a 256-byte aligned cursor; a plain vector only guarantees 16.
    std::vector<std::byte> scratch(bytes + 256);
    const auto base    = reinterpret_cast<std::uintptr_t>(scratch.data());
    const auto aligned = (256 - (base & 255)) & 255;
    MemoryArena arena(
        MemorySpan{scratch.data() + aligned, bytes, Device::cpu()});
    const bool timing = std::getenv("NINFER_SPLIT_TIMING") != nullptr;
    // port (sm_89): NINFER_SPLIT_TIMING splits the host half into what it actually spends: the
    // CPU expert kernels, the wait for the Device term, the elementwise merge and the write-back.
    // The split path's cost is what decides whether holding experts at all beats host-only.
    const auto cpu_begin = std::chrono::steady_clock::now();
    ops::routed_swiglu(x_cpu, moe.profile.routed, host->experts(ExpertBankId{moe.stream->bank_index()}), routes_cpu,
                       subset, out_cpu, arena, cpu);
    const auto cpu_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - cpu_begin).count();
    if (std::getenv("NINFER_HOST_DEBUG")) {
        std::fprintf(stderr, "[host-x] bank=%u x0..3=%.6f %.6f %.6f %.6f\n",
                     moe.stream->bank_index(),
                     __bfloat162float(__ushort_as_bfloat16(fetch.x[0])),
                     __bfloat162float(__ushort_as_bfloat16(fetch.x[1])),
                     __bfloat162float(__ushort_as_bfloat16(fetch.x[2])),
                     __bfloat162float(__ushort_as_bfloat16(fetch.x[3])));
    }
    // The Device half was launched before this call and ran while the CPU kernels above worked.
    // The copy below is queued on the same stream, so it cannot start before that kernel finishes:
    // waiting for the copy alone is enough, and a separate stream-wide synchronise here would only
    // pay the WDDM round trip twice per layer.
    const auto join_begin = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(host_gpu.data(), gpu_term.data, host_gpu.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto join_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - join_begin).count();
    const auto merge_begin = std::chrono::steady_clock::now();
    std::vector<std::uint16_t> packed(host_out.size());
    for (std::size_t i = 0; i < host_out.size(); ++i) {
        // The device epilogue starts from the BF16 residual and folds the routed and shared
        // contributions into that FP32 accumulator, so keep the same order here.
        const float sum = (__bfloat162float(__ushort_as_bfloat16(fetch.residual[i])) + host_out[i]) +
                          host_gpu[i];
        std::uint32_t bits = 0;
        std::memcpy(&bits, &sum, sizeof(bits));
        // Round to nearest even, the same rule __float2bfloat16_rn applies; truncating biases
        // every layer downward and compounds over the stack.
        const std::uint32_t lsb = (bits >> 16) & 1u;
        bits += 0x7fffu + lsb;
        packed[i] = static_cast<std::uint16_t>(bits >> 16);
    }
    const auto merge_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - merge_begin).count();
    const auto write_begin = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(residual.data, packed.data(), packed.size() * sizeof(std::uint16_t),
                               cudaMemcpyHostToDevice, stream));
    if (timing) {
        static std::uint64_t calls = 0;
        static double acc[5]       = {0.0, 0.0, 0.0, 0.0, 0.0};
        acc[0] += cpu_ms;
        acc[1] += join_ms;
        acc[2] += merge_ms;
        acc[3] += std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - write_begin).count();
        acc[4] += ensure_ms;
        if (++calls % 40 == 0) {
            std::fprintf(stderr,
                         "[host] cpu=%.2fms join=%.2fms merge=%.2fms write=%.2fms ensure=%.2fms "
                         "per token\n",
                         acc[0], acc[1], acc[2], acc[3], acc[4]);
            for (double& value : acc) { value = 0.0; }
        }
    }
    if (std::getenv("NINFER_HOST_DEBUG")) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // residual is BF16, so print the packed words rather than reinterpreting them as FP32.
        double sum = 0.0, square = 0.0;
        for (const auto word : packed) {
            const double value = __bfloat162float(__ushort_as_bfloat16(word));
            sum += value;
            square += value * value;
        }
        std::fprintf(stderr, "[host-residual] bank=%u r0..3=%.6f %.6f %.6f %.6f sum=%.9f sq=%.9f\n",
                     moe.stream->bank_index(),
                     __bfloat162float(__ushort_as_bfloat16(packed[0])),
                     __bfloat162float(__ushort_as_bfloat16(packed[1])),
                     __bfloat162float(__ushort_as_bfloat16(packed[2])),
                     __bfloat162float(__ushort_as_bfloat16(packed[3])), sum, square);
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
            // Keep the shared term in FP32 so that the host fallback can fold it into the same
            // residual rounding the device epilogue performs once per layer.
            // Take the inputs the CPU half needs before the Device half is launched, so the two
            // can then run concurrently.
            auto fetch       = fetch_host_routed(hidden, routes, residual, execution.stream);
            auto shared_f32 = workspace.alloc(DType::FP32, {hidden.ne[0], columns});
            MemoryArena shared_arena(MemorySpan{scratch.data, scratch.bytes, execution.device()});
            ops::routed_swiglu(hidden, routed, moe.stream->experts(), routes,
                               ops::ExpertSubset{ops::ExpertSubsetKind::None, {}},
                               ops::ScaledSwiGLUTerm{moe.shared, shared_coefficients},
                               ops::MoeEpilogue::Write, shared_f32, shared_arena, execution);
            host_routed_ffn(fetch, moe, ops::ExpertSubset{ops::ExpertSubsetKind::All, {}},
                            shared_f32, residual, execution);
            return;
        }
    } catch (const std::exception& error) {
        throw std::invalid_argument(std::string("expert stream upload: ") + error.what());
    }
    if (!moe.stream->fully_resident()) {
        // Split layer: the pool holds the experts this layer routes to most, the host tier covers
        // the rest. Both halves land in FP32 and are folded into the residual once, so a split
        // layer rounds exactly like a resident or a fully hosted one.
        auto fetch      = fetch_host_routed(hidden, routes, residual, execution.stream);
        auto mixed_f32 = workspace.alloc(DType::FP32, {hidden.ne[0], columns});
        MemoryArena mixed_arena(MemorySpan{scratch.data, scratch.bytes, execution.device()});
        const auto mask_host = moe.stream->resident_mask_host();
        const auto mask_size = static_cast<std::int32_t>(mask_host.size());
        Tensor device_mask(const_cast<void*>(moe.stream->resident_mask()), execution.device(),
                           DType::I32, {mask_size});
        ops::routed_swiglu(hidden, routed, moe.stream->experts(), routes,
                           ops::ExpertSubset{ops::ExpertSubsetKind::MaskOnes, device_mask},
                           ops::ScaledSwiGLUTerm{moe.shared, shared_coefficients},
                           ops::MoeEpilogue::Write, mixed_f32, mixed_arena, execution);
        Tensor host_mask(const_cast<std::int32_t*>(mask_host.data()), Device::cpu(), DType::I32,
                         {mask_size});
        host_routed_ffn(fetch, moe,
                        ops::ExpertSubset{ops::ExpertSubsetKind::MaskZeros, host_mask}, mixed_f32,
                        residual, execution);
        return;
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
        const auto count = static_cast<std::size_t>(hidden.ne[0]) * hidden.ne[1];
        std::vector<std::uint16_t> all(count);
        CUDA_CHECK(cudaMemcpy(all.data(), residual.data, count * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost));
        double sum = 0.0, square = 0.0;
        for (const auto word : all) {
            const double value = __bfloat162float(__ushort_as_bfloat16(word));
            sum += value;
            square += value * value;
        }
        std::fprintf(stderr, "[residual] bank=%u r0..3=%.6f %.6f %.6f %.6f sum=%.9f sq=%.9f\n",
                     moe.stream->bank_index(),
                     __bfloat162float(__ushort_as_bfloat16(all[0])),
                     __bfloat162float(__ushort_as_bfloat16(all[1])),
                     __bfloat162float(__ushort_as_bfloat16(all[2])),
                     __bfloat162float(__ushort_as_bfloat16(all[3])), sum, square);
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
            // port (sm_89): the host fallback keeps the shared term in FP32 so that a layer
            // rounds the residual once, exactly like the device epilogue.
            const auto shared_f32 = static_cast<std::size_t>(moe->profile.routed.ffn.hidden) *
                                    static_cast<std::size_t>(last) * sizeof(float);
            return stream_route_tensor_bytes(*moe, last) + shared_f32 +
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

