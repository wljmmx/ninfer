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

// port (sm_89): `packed_device`, when the caller provides it, is a device block big enough for
// the four buffers below. The pack kernel copies them into it on the stream and one transfer brings
// the whole block down, which is what turns four submissions into one.
HostRoutedFetch fetch_host_routed(const Tensor& hidden, const ops::ExpertRoutes& routes,
                                  const Tensor& residual, cudaStream_t stream,
                                  void* packed_device) {
    const auto cols   = hidden.ne[0];
    const auto tokens = hidden.ne[1];
    const auto top_k  = routes.ids.ne[0];
    HostRoutedFetch fetch;
    fetch.x.resize(static_cast<std::size_t>(cols) * tokens);
    fetch.residual.resize(static_cast<std::size_t>(cols) * tokens);
    fetch.ids.resize(static_cast<std::size_t>(top_k) * tokens);
    fetch.coefficients.resize(static_cast<std::size_t>(top_k) * tokens);
    // The transfer itself, per token: decode speed is too noisy on this box to resolve a few
    // tenths of a millisecond, and this line is what a change here has to move.
    const auto fetch_begin = std::chrono::steady_clock::now();
    if (packed_device != nullptr) {
        const auto x_bytes    = fetch.x.size() * sizeof(std::uint16_t);
        const auto res_bytes  = fetch.residual.size() * sizeof(std::uint16_t);
        const auto ids_bytes  = fetch.ids.size() * sizeof(std::int32_t);
        const auto coef_bytes = fetch.coefficients.size() * sizeof(float);
        ops::moe_pack_host_fetch(hidden.data, residual.data, routes.ids.data,
                                 routes.coefficients.data, static_cast<int>(fetch.x.size()),
                                 static_cast<int>(fetch.ids.size()), packed_device, stream);
        const auto total = ops::moe_pack_host_fetch_bytes(x_bytes, res_bytes, ids_bytes, coef_bytes);
        // One host block, sliced back into the fields the rest of the path already uses. Same
        // alignment the pack kernel uses (16 bytes), so the slices land where it wrote them.
        static thread_local std::vector<std::byte> staging;
        staging.resize(total);
        CUDA_CHECK(cudaMemcpyAsync(staging.data(), packed_device, total, cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto align16 = [](std::size_t value) { return (value + 15U) & ~std::size_t{15}; };
        const auto off_res  = align16(x_bytes);
        const auto off_ids  = align16(off_res + res_bytes);
        const auto off_coef = align16(off_ids + ids_bytes);
        std::memcpy(fetch.x.data(), staging.data(), x_bytes);
        std::memcpy(fetch.residual.data(), staging.data() + off_res, res_bytes);
        std::memcpy(fetch.ids.data(), staging.data() + off_ids, ids_bytes);
        std::memcpy(fetch.coefficients.data(), staging.data() + off_coef, coef_bytes);
        if (std::getenv("NINFER_SPLIT_TIMING") != nullptr) {
            static double accumulated_ms = 0.0;
            static std::uint64_t calls   = 0;
            accumulated_ms += std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - fetch_begin).count();
            if (++calls % 40 == 0) {
                std::fprintf(stderr, "[fetch] %.2fms per token\n", accumulated_ms);
                accumulated_ms = 0.0;
            }
        }
        return fetch;
    }
    // port (sm_89): four independent transfers. cudaMemcpyBatchAsync (CUDA 13, with the 13.4
    // cudaMemcpyFlagPreferOverlapWithCompute hint) merges them into one submission and measured
    // 5.77/5.83 ms per token against 5.66/5.73 for the four calls: the cost is the synchronise at
    // the end, not the submission count, so the batch form was reverted rather than kept.
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
    if (std::getenv("NINFER_SPLIT_TIMING") != nullptr) {
        static double accumulated = 0.0;
        static std::uint64_t calls = 0;
        accumulated += std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - fetch_begin).count();
        if (++calls % 40 == 0) {
            std::fprintf(stderr, "[fetch] %.2fms per token\n", accumulated);
            accumulated = 0.0;
        }
    }
    return fetch;
}

void host_routed_ffn(const HostRoutedFetch& fetch, const RoutedMoeParameters& moe,
                     const ops::ExpertSubset& subset, const Tensor& gpu_term, Tensor& residual,
                     DeviceExecutionView execution, void* host_term_device) {
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
    // port (sm_89): Strata sizes its expert pool as `p_cores - 1` and lets the host thread keep the
    // last core, because the thread that drives the GPU has to run somewhere. We hand the team every
    // physical core and leave the driver to fight the spinners for the last one, which is a
    // plausible reason the per-layer cost has a 5 ms tail on a 23 ms median. NINFER_CPU_TEAM_SIZE
    // makes the trade measurable instead of a guess.
    static const auto cpu_team = [] {
        const auto selectable = cpu_worker_order(topology.info(), false).size();
        const char* value = std::getenv("NINFER_CPU_TEAM_SIZE");
        if (value != nullptr && *value != '\0') {
            const auto wanted = static_cast<std::size_t>(std::strtoul(value, nullptr, 10));
            return std::max<std::size_t>(1, std::min(wanted, selectable));
        }
        return std::max<std::size_t>(1, selectable);
    }();
    static CpuExecutor executor(topology,
                                plan_cpu_team(topology.info(), static_cast<std::uint32_t>(cpu_team)));
    // port (sm_89): the team is the cost of a split layer, not the kernel. With one or two experts
    // a layer, 24 workers spend more time waking each other than computing: measured per token at a
    // 480 MiB pool, 24 workers 38.9 ms of `cpu` against 13.0 ms for a team of four, and decode 9.7
    // against 17.6 tok/s. So small work runs on a small team, wide work keeps the full one.
    static constexpr std::size_t kSmallTeam    = 4;
    static constexpr std::size_t kSmallExperts = 4;
    static CpuExecutor small_executor(
        topology, plan_cpu_team(topology.info(),
                                static_cast<std::uint32_t>(std::min(kSmallTeam, cpu_team))));
    const auto cpu       = executor.execution_view();
    const auto cpu_small = small_executor.execution_view();
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
    // A layer that hands the CPU a handful of experts runs on the small team; anything wider keeps
    // every core, which is what prefill and the host-only path need.
    const auto& cpu_view = wanted.size() <= kSmallExperts ? cpu_small : cpu;
    ops::routed_swiglu(x_cpu, moe.profile.routed, host->experts(ExpertBankId{moe.stream->bank_index()}),
                       routes_cpu, subset, out_cpu, arena, cpu_view);
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
    if (host_term_device != nullptr) {
        // port (sm_89): the device merges. Upload the CPU term and let one kernel write
        // `RNE_bf16(residual + host + gpu)` in place - the same accumulation order and rounding the
        // loop below uses, so the result is the same value without a read-back of the device term
        // and without a second synchronise: the next layer's read-back waits on this stream anyway.
        const auto write_begin = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaMemcpyAsync(host_term_device, host_out.data(),
                                   host_out.size() * sizeof(float), cudaMemcpyHostToDevice,
                                   stream));
        ops::moe_merge_host_term(host_term_device, gpu_term.data, residual.data,
                                 static_cast<int>(host_out.size()), stream);
        if (timing) {
            static std::uint64_t calls = 0;
            static double acc[2]       = {0.0, 0.0};
            acc[0] += cpu_ms;
            acc[1] += std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - write_begin).count();
            if (++calls % 40 == 0) {
                const auto& stats = host->stats();
                std::fprintf(stderr,
                             "[host] cpu=%.2fms write=%.2fms (device merge) per token; arena=%u "
                             "slots %lluKiB hit=%llu miss=%llu evict=%llu\n",
                             acc[0], acc[1], host->slot_capacity(),
                             static_cast<unsigned long long>(host->slot_bytes() >> 10),
                             static_cast<unsigned long long>(stats.hits),
                             static_cast<unsigned long long>(stats.misses),
                             static_cast<unsigned long long>(stats.evictions));
                acc[0] = acc[1] = 0.0;
            }
        }
        return;
    }
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
            const auto& stats = host->stats();
            std::fprintf(stderr,
                         "[host] cpu=%.2fms join=%.2fms merge=%.2fms write=%.2fms ensure=%.2fms "
                         "per token; arena=%u slots %lluKiB hit=%llu miss=%llu evict=%llu\n",
                         acc[0], acc[1], acc[2], acc[3], acc[4], host->slot_capacity(),
                         static_cast<unsigned long long>(host->slot_bytes() >> 10),
                         static_cast<unsigned long long>(stats.hits),
                         static_cast<unsigned long long>(stats.misses),
                         static_cast<unsigned long long>(stats.evictions));
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

namespace {

// port (sm_89): one layer in forty carries a pair of events, so the layer budget can be split into
// "the expert tier" and "everything else" (attention, norms, head) without paying a synchronise per
// layer. NINFER_STAGE_TIMING prints the sampled layer's device time once per token.
class StageSample {
public:
    explicit StageSample(cudaStream_t stream) : stream_(stream) {
        auto& s = state();
        if (!s.enabled) { return; }
        if (!s.created) {
            (void)cudaEventCreate(&s.begin);
            (void)cudaEventCreate(&s.end);
            s.created = true;
        }
        if (s.calls % 40 != 20) { return; }
        // Read the previous sample here: it completed long ago, so this costs no synchronise. The
        // alternative - waiting inside the destructor - blocks the host for a layer and inflated the
        // sampled layer by ~40% when it was tried.
        if (s.pending) {
            float ms = 0.0F;
            (void)cudaEventElapsedTime(&ms, s.begin, s.end);
            s.sum += ms;
            s.pending = false;
            if (++s.samples % 16 == 0) {
                const double per_layer = s.sum / static_cast<double>(s.samples);
                std::fprintf(stderr,
                             "[stage] expert tier %.3f ms/layer  (x40 = %.2f ms/token)\n", per_layer,
                             per_layer * 40.0);
                s.sum     = 0.0;
                s.samples = 0;
            }
        }
        (void)cudaEventRecord(s.begin, stream_);
        armed_ = true;
    }
    ~StageSample() {
        if (!armed_) { return; }
        (void)cudaEventRecord(state().end, stream_);
        state().pending = true;
    }

    // `state()` and the name counter are used by the caller, so they stay public.
    struct State {
        bool enabled = std::getenv("NINFER_STAGE_TIMING") != nullptr;
        bool created = false;
        bool pending = false;
        cudaEvent_t begin{}, end{};
        double sum          = 0.0;
        std::uint64_t calls = 0, samples = 0;
    };
    static State& state() {
        static State s;
        return s;
    }

private:
    cudaStream_t stream_;
    bool armed_ = false;
};

} // namespace

void streamed_ffn(const Tensor& hidden, const RoutedMoeParameters& moe, Tensor& residual,
                  WorkspaceArena& workspace, DeviceExecutionView execution) {
    StageSample stage_sample(execution.stream);
    StageSample::state().calls++;
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
    // port (sm_89): one read-back per layer. The residency decision needs the 32 route ids and the
    // CPU half needs the layer inputs; they come down in the same batch with one stream synchronise.
    // The split path used to pay two - a 32-byte copy inside ensure and the inputs here - and that
    // was 40 x ~0.3 ms of the token on this driver.
    // One device block for the read-back and one for the CPU term, so the split path pays one
    // transfer per layer and the merge needs no read-back at all.
    const std::size_t handoff_x    = static_cast<std::size_t>(hidden.ne[0]) * columns;
    const std::size_t handoff_ids  = static_cast<std::size_t>(routed.top_k) * columns;
    const std::size_t handoff_term = handoff_x;
    auto pack_span      = workspace.alloc_bytes(ops::moe_pack_host_fetch_bytes(
        handoff_x * 2, handoff_x * 2, handoff_ids * 4, handoff_ids * 4));
    auto host_term_span = workspace.alloc_bytes(handoff_term * sizeof(float));
    // port (sm_89): the merged read-back measured 4.66/4.88 ms per token against 4.05 for the four
    // copies, so it stays behind a switch; the device merge replaces join+merge+write and is on.
    void* pack_device = std::getenv("NINFER_DMA_PACK") != nullptr ? pack_span.data : nullptr;
    void* host_term_device = std::getenv("NINFER_HOST_MERGE_CPU") != nullptr ? nullptr
                                                                           : host_term_span.data;
    std::optional<HostRoutedFetch> early;
    if (!moe.stream->fully_resident()) {
        early.emplace(fetch_host_routed(hidden, routes, residual, execution.stream, pack_device));
    }
    try {
        if (!moe.stream->ensure(ids.data, routed.top_k, columns, execution.stream,
                                early ? early->ids.data() : nullptr)) {
            // The pool cannot hold this layer: the shared expert still runs on the device (subset
            // None touches only the shared term), and the routed term moves to the host tier.
            // Keep the shared term in FP32 so that the host fallback can fold it into the same
            // residual rounding the device epilogue performs once per layer.
            // Take the inputs the CPU half needs before the Device half is launched, so the two
            // can then run concurrently.
            auto fetch = early ? std::move(*early)
                               : fetch_host_routed(hidden, routes, residual, execution.stream,
                                                    pack_device);
            auto shared_f32 = workspace.alloc(DType::FP32, {hidden.ne[0], columns});
            MemoryArena shared_arena(MemorySpan{scratch.data, scratch.bytes, execution.device()});
            ops::routed_swiglu(hidden, routed, moe.stream->experts(), routes,
                               ops::ExpertSubset{ops::ExpertSubsetKind::None, {}},
                               ops::ScaledSwiGLUTerm{moe.shared, shared_coefficients},
                               ops::MoeEpilogue::Write, shared_f32, shared_arena, execution);
            host_routed_ffn(fetch, moe, ops::ExpertSubset{ops::ExpertSubsetKind::All, {}},
                            shared_f32, residual, execution, host_term_device);
            return;
        }
    } catch (const std::exception& error) {
        throw std::invalid_argument(std::string("expert stream upload: ") + error.what());
    }
    if (!moe.stream->fully_resident()) {
        // Split layer: the pool holds the experts this layer routes to most, the host tier covers
        // the rest. Both halves land in FP32 and are folded into the residual once, so a split
        // layer rounds exactly like a resident or a fully hosted one.
        auto fetch      = std::move(*early);
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
                        residual, execution, host_term_device);
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

