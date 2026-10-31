#include "models/qwen3_5/execution/ffn.h"

#include "models/qwen3_5/execution/expert_stream.h"
#include "models/qwen3_5/execution/expert_path_control.h"
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
#include <functional>
#include <optional>

namespace ninfer::models::qwen3_5::execution {
namespace {

// port (sm_89): where a split token goes, node by node, on both sides of the PCIe lane.
//
// The host nodes are wall clock. The Device half cannot be: its kernel is enqueued and returns at
// once, so a host clock around the launch measures the launch and nothing else. It is timed with a
// pair of events instead, and the pair is read at the START OF THE NEXT LAYER - by then that work is
// certainly finished, so cudaEventQuery answers without a synchronise and the measurement itself
// costs no stall. That distinction matters here: this file's whole problem is per-layer stalls, and
// an instrument that adds one would measure its own cost.
//
// The numbers are summed over the forty layers of a decode token and printed once per token, because
// a decode rate is quoted per token and a per-layer number hides how the forty add up.
struct SplitTimeline {
    double wall        = 0.0; // the whole layer's FFN, host clock, entry to exit
    double fetch       = 0.0; // host: pack launch + one D2H of x/residual/ids/coef + the wait
    double ens_route   = 0.0; // host: the route ids, only when the caller has none to hand over
    double ens_layout  = 0.0; // host: hot set, slot publish, mask and offset tables
    double ens_upload  = 0.0; // host: prepare + wait_for_progress
    double gpu         = 0.0; // device: the routed expert kernel (subset or resident)
    double host_cpu    = 0.0; // host: the CPU expert GEMM for the experts the pool lacks
    double host_write  = 0.0; // host: merge submission (the device folds it in)
    std::uint64_t missing = 0; // routed entries this layer the pool did not hold
    std::uint64_t layers  = 0;
    // Last single-layer readings: the controller judges a layer, not a token average.
    double last_gpu  = 0.0;
    // Layer split for attribution: pre = entry to the read-back, ensure_win = read-back to done.
    double seg_pre = 0.0;
    double seg_ensure = 0.0;
    double last_wall = 0.0;

    void report();
};

SplitTimeline& timeline() {
    static thread_local SplitTimeline state;
    return state;
}

// One pair of events, reused. Two events per layer would be an allocation per layer; a pair reused
// in order is enough because only one layer's kernel can be in flight at a time on this stream.
struct GpuClock {
    cudaEvent_t begin = nullptr;
    cudaEvent_t end   = nullptr;
    bool pending      = false;
};

GpuClock& gpu_clock() {
    static thread_local GpuClock state;
    return state;
}

void timeline_gpu_begin(cudaStream_t stream) {
    if (std::getenv("NINFER_SPLIT_TIMING") == nullptr) { return; }
    auto& clock = gpu_clock();
    if (clock.begin == nullptr) {
        (void)cudaEventCreateWithFlags(&clock.begin, cudaEventDefault);
        (void)cudaEventCreateWithFlags(&clock.end, cudaEventDefault);
    }
    clock.pending = true;
    (void)cudaEventRecord(clock.begin, stream);
}

void timeline_gpu_end(cudaStream_t stream) {
    if (std::getenv("NINFER_SPLIT_TIMING") == nullptr) { return; }
    auto& clock = gpu_clock();
    if (clock.pending) { (void)cudaEventRecord(clock.end, stream); }
}

// Reads last layer's pair. Called at the start of a layer; never waits.
void timeline_gpu_collect() {
    auto& clock = gpu_clock();
    if (!clock.pending) { return; }
    if (cudaEventQuery(clock.end) != cudaSuccess) { return; }
    float elapsed = 0.0F;
    if (cudaEventElapsedTime(&elapsed, clock.begin, clock.end) == cudaSuccess) {
        timeline().gpu += static_cast<double>(elapsed);
        timeline().last_gpu = static_cast<double>(elapsed);
        timeline().last_gpu = static_cast<double>(elapsed);
    }
    clock.pending = false;
}

void SplitTimeline::report() {
    if (layers == 0) { return; }
    const auto per = 1.0 / static_cast<double>(layers);
    std::fprintf(stderr,
                 "[tl] token: wall=%.1fms fetch=%.1f ens(route=%.2f layout=%.2f upload=%.2f) "
                 "gpu=%.1f host(cpu=%.1f write=%.1f) miss=%.2f/layer\n",
                 wall, fetch, ens_route, ens_layout, ens_upload, gpu, host_cpu, host_write,
                 static_cast<double>(missing) * per);
    std::fprintf(stderr,
                 "[tl] per layer: wall=%.3f fetch=%.3f ens=%.3f gpu=%.3f host=%.3f "
                 "unaccounted=%.3f pre=%.3f ensure_win=%.3f\n",
                 wall * per, fetch * per, (ens_route + ens_layout + ens_upload) * per, gpu * per,
                 (host_cpu + host_write) * per,
                 wall * per - (fetch + ens_route + ens_layout + ens_upload + gpu + host_cpu +
                               host_write) * per,
                 seg_pre * per, seg_ensure * per);
    std::fflush(stderr);
    *this = SplitTimeline{};
}

// Adds one layer to the timeline on the way out, whichever of the three exits it took.
struct LayerScope {
    bool on;
    std::chrono::steady_clock::time_point begin;
    explicit LayerScope(bool enabled)
        : on(enabled), begin(std::chrono::steady_clock::now()) {}
    ~LayerScope() {
        if (!on) { return; }
        auto& state = timeline();
        const auto layer_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - begin)
                                  .count();
        state.wall += layer_ms;
        state.last_wall = layer_ms;
        if (++state.layers % 40 == 0) { state.report(); }
    }
};


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
    // True when only the route ids came down: one copy, no pack kernel. The rest of the block is
    // fetched later, and only by a layer that actually has experts to compute on the host.
    bool ids_only = false;
    std::vector<std::uint16_t> x;
    std::vector<std::uint16_t> residual;
    std::vector<std::int32_t> ids;
    std::vector<float> coefficients;
};

// port (sm_89): `packed_device`, when the caller provides it, is a device block big enough for
// the four buffers below. The pack kernel copies them into it on the stream and one transfer brings
// the whole block down, which is what turns four submissions into one.
// port (sm_89): the ids alone, in one submission. The full read-back packages x, residual, the ids
// and the coefficients into one block and brings them down in one transfer - but that is a kernel
// launch plus a copy, and on this driver a submission costs tens of microseconds. A layer whose
// experts are all resident needs none of x, residual or coefficients: it needs the 32 ids and
// nothing else. So the cheap shape is one cudaMemcpyAsync straight out of the route tensor, and the
// expensive shape is only paid by a layer that turns out to have work for the host half.
HostRoutedFetch fetch_ids_only(const ops::ExpertRoutes& routes, cudaStream_t stream) {
    const auto top_k    = routes.ids.ne[0];
    const auto tokens   = routes.ids.ne[1];
    HostRoutedFetch fetch;
    fetch.ids_only = true;
    fetch.ids.resize(static_cast<std::size_t>(top_k) * tokens);
    CUDA_CHECK(cudaMemcpyAsync(fetch.ids.data(), routes.ids.data,
                               fetch.ids.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return fetch;
}

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
        // port (sm_89): polling cudaStreamQuery instead of blocking on the stream was measured and
        // changed nothing (fetch 0.072-0.075 ms/layer either way, decode 152.9 vs 152.9 at 16 GiB).
        // The 10 KiB copy itself is ~3 us at 3 GB/s, so the cost is not the bytes and not the wait
        // primitive: it is the round trip to observe device progress at all. Removing it needs the
        // host to stop observing - a device-side decision, or a speculative run replayed at the
        // token boundary - not a cheaper way to wait.
        // Split the wait in two: what the host spent *issuing* the read-back, and what it spent
        // waiting for the Device to finish everything queued ahead of it. Only the first part is
        // overhead that removing the round trip could save; the second is work the layer needs
        // anyway (1x.4s).
        const double issue_us = std::chrono::duration<double, std::micro>(
                                    std::chrono::steady_clock::now() - fetch_begin)
                                    .count();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const double wait_us = std::chrono::duration<double, std::micro>(
                                   std::chrono::steady_clock::now() - fetch_begin)
                                   .count() - issue_us;
        if (std::getenv("NINFER_SPLIT_TIMING") != nullptr) {
            static double acc_issue = 0.0, acc_wait = 0.0;
            static std::uint64_t calls = 0;
            acc_issue += issue_us;
            acc_wait += wait_us;
            if (++calls % 40 == 0) {
                std::fprintf(stderr, "[fetch-split] issue=%.1fus wait=%.1fus\n", acc_issue / 40.0,
                             acc_wait / 40.0);
                std::fflush(stderr);
                acc_issue = 0.0;
                acc_wait = 0.0;
            }
        }
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
            const auto fetch_ms          = std::chrono::duration<double, std::milli>(
                                               std::chrono::steady_clock::now() - fetch_begin)
                                               .count();
            accumulated_ms += fetch_ms;
            timeline().fetch += fetch_ms;
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
    // port (sm_89): four independent transfers, one submission. cudaMemcpyBatchAsync (CUDA 13)
    // takes the whole list at once, which removes three WDDM submissions per layer - and a split
    // token makes forty of them. The copies are independent, so the batch's unspecified internal
    // order does not matter. NINFER_DMA_BATCH=0 keeps the per-copy form for A/B.
    static const bool batched = std::getenv("NINFER_DMA_BATCH") == nullptr;
    if (batched) {
        void* destinations[4] = {fetch.x.data(), fetch.residual.data(), fetch.ids.data(),
                                 fetch.coefficients.data()};
        const void* sources[4] = {hidden.data, residual.data, routes.ids.data,
                                  routes.coefficients.data};
        const std::size_t sizes[4] = {fetch.x.size() * sizeof(std::uint16_t),
                                      fetch.residual.size() * sizeof(std::uint16_t),
                                      fetch.ids.size() * sizeof(std::int32_t),
                                      fetch.coefficients.size() * sizeof(float)};
        // Every copy must name a source access order - the sources are device buffers the kernels
        // above wrote on this stream, so they are read in stream order - and the 13.4 flag asks the
        // driver to overlap the transfers with compute.
        cudaMemcpyAttributes attributes{};
        attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
        attributes.flags          = cudaMemcpyFlagPreferOverlapWithCompute;
        std::size_t attribute_index = 0;
        CUDA_CHECK(cudaMemcpyBatchAsync(destinations, sources, sizes, 4, &attributes,
                                        &attribute_index, 1, stream));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(fetch.x.data(), hidden.data,
                                   fetch.x.size() * sizeof(std::uint16_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(fetch.residual.data(), residual.data,
                                   fetch.residual.size() * sizeof(std::uint16_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(fetch.ids.data(), routes.ids.data,
                                   fetch.ids.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaMemcpyAsync(fetch.coefficients.data(), routes.coefficients.data,
                                   fetch.coefficients.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (std::getenv("NINFER_SPLIT_TIMING") != nullptr) {
        static double accumulated = 0.0;
        static std::uint64_t calls = 0;
        const auto fetch_ms        = std::chrono::duration<double, std::milli>(
                                         std::chrono::steady_clock::now() - fetch_begin)
                                         .count();
        accumulated += fetch_ms;
        timeline().fetch += fetch_ms;
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
    const bool timing = std::getenv("NINFER_SPLIT_TIMING") != nullptr;
    // port (sm_89): NINFER_SPLIT_TIMING splits the host half into what it actually spends: the
    // CPU expert kernels, the wait for the Device term, the elementwise merge and the write-back.
    // The split path's cost is what decides whether holding experts at all beats host-only.
    //
    // port (sm_89): a split layer whose Device half covers every routed expert has nothing for this
    // half to do - `wanted` is empty and the CPU term is zero by construction. The collective still
    // used to run, and walking the mask over 256 experts per bank cost 5.7-6.5 ms a token with
    // hit=0 miss=0 in the arena statistics to show for it. Skip the workspace and the kernels when
    // there is no expert to compute; the merge below adds a zero term and rounds exactly as before.
    double cpu_ms = 0.0;
    if (!wanted.empty()) {
        const auto bytes = ops::routed_swiglu_workspace_capacity_bytes(
            moe.profile.routed, ops::ExpertSubsetKind::All, tokens, tokens);
        // The CPU kernels want a 256-byte aligned cursor; a plain vector only guarantees 16.
        std::vector<std::byte> scratch(bytes + 256);
        const auto base    = reinterpret_cast<std::uintptr_t>(scratch.data());
        const auto aligned = (256 - (base & 255)) & 255;
        MemoryArena arena(
            MemorySpan{scratch.data() + aligned, bytes, Device::cpu()});
        const auto cpu_begin = std::chrono::steady_clock::now();
        // A layer that hands the CPU a handful of experts runs on the small team; anything wider
        // keeps every core, which is what prefill and the host-only path need.
        const auto& cpu_view = wanted.size() <= kSmallExperts ? cpu_small : cpu;
        ops::routed_swiglu(x_cpu, moe.profile.routed,
                           host->experts(ExpertBankId{moe.stream->bank_index()}), routes_cpu, subset,
                           out_cpu, arena, cpu_view);
        cpu_ms = std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - cpu_begin)
                     .count();
    }
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
            const auto write_ms        = std::chrono::duration<double, std::milli>(
                                             std::chrono::steady_clock::now() - write_begin)
                                             .count();
            acc[0] += cpu_ms;
            acc[1] += write_ms;
            timeline().host_cpu += cpu_ms;
            timeline().host_write += write_ms;
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

// port (sm_89): one CUDA graph per expert bank, covering the device half of a split layer.
//
// A layer's device work is a fixed sequence of kernel launches whose only varying inputs are device
// buffers: the routes the router just wrote, the offset/mask table, the hidden rows and the output
// row. The offsets and the mask are what a capture bakes in, so the key is their hash plus the
// addresses of every buffer the kernels touch - anything else changing would silently replay against
// stale memory. When the set changes (spill taking a different set of experts this step, a new
// generation, a different workspace slot) the key changes and the graph is rebuilt once.
//
// NINFER_LAYER_GRAPH=1 enables it; the whole-token graph is a different mechanism and stays off here.
class LayerGraphs {
public:
    static LayerGraphs& instance() {
        static LayerGraphs graphs;
        return graphs;
    }

    [[nodiscard]] bool enabled() const noexcept { return enabled_; }

    // Replays the recorded work, or records it on the first call for this key.
    void run(std::uint32_t bank, std::uint64_t key, cudaStream_t stream,
             const std::function<void()>& work) {
        Entry& entry = entries_[bank % entries_.size()];
        if (entry.exec != nullptr && entry.key == key) {
            (void)cudaGraphLaunch(entry.exec, stream);
            ++replays_;
            return;
        }
        if (entry.exec != nullptr) {
            (void)cudaGraphExecDestroy(entry.exec);
            entry.exec = nullptr;
        }
        (void)cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
        work();
        cudaGraph_t graph = nullptr;
        const auto captured = cudaStreamEndCapture(stream, &graph);
        if (captured != cudaSuccess || graph == nullptr) {
            // Nothing to fall back to here: the work was recorded, not executed.
            work();
            return;
        }
        if (cudaGraphInstantiate(&entry.exec, graph, nullptr, nullptr, 0) != cudaSuccess) {
            entry.exec = nullptr;
            (void)cudaGraphDestroy(graph);
            work();
            return;
        }
        (void)cudaGraphDestroy(graph);
        entry.key = key;
        ++captures_;
        (void)cudaGraphLaunch(entry.exec, stream);
    }

    void report() {
        if (!enabled_) { return; }
        std::fprintf(stderr, "[layer-graph] captures=%llu replays=%llu\n",
                     static_cast<unsigned long long>(captures_),
                     static_cast<unsigned long long>(replays_));
        captures_ = replays_ = 0;
    }

    ~LayerGraphs() {
        for (auto& entry : entries_) {
            if (entry.exec != nullptr) { (void)cudaGraphExecDestroy(entry.exec); }
        }
    }

private:
    struct Entry {
        cudaGraphExec_t exec = nullptr;
        std::uint64_t key    = 0;
    };
    LayerGraphs() : enabled_(std::getenv("NINFER_LAYER_GRAPH") != nullptr), entries_(40) {}
    std::vector<Entry> entries_;
    bool enabled_          = false;
    std::uint64_t captures_ = 0;
    std::uint64_t replays_  = 0;
};

} // namespace

void streamed_ffn(const Tensor& hidden, const RoutedMoeParameters& moe, Tensor& residual,
                  WorkspaceArena& workspace, DeviceExecutionView execution) {
    const bool layer_timing = std::getenv("NINFER_SPLIT_TIMING") != nullptr;
    // Last layer's Device pair is certainly finished by now, so this read is free. Doing it here,
    // before any work is queued, is what keeps the measurement out of the thing being measured.
    timeline_gpu_collect();
    LayerScope layer_scope(layer_timing);
    // Four-way split of a layer, so "unaccounted" can be attributed instead of shrugged at: what the
    // layer spent before the read-back, the read-back itself, ensure, and everything after the kernel
    // launch. The ids-only experiment moved 85 us into "unaccounted" and this is what says where.
    const auto t_layer_begin = std::chrono::steady_clock::now();
    double seg_pre_ms = 0.0;
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
    // port (sm_89): on by default. It trades three submissions for one pack kernel plus one
    // transfer, which is less work even where the wall clock is inside this box's noise.
    // NINFER_DMA_PACK=0 keeps the four copies for A/B.
    static const bool pack_read_back = std::getenv("NINFER_DMA_PACK") == nullptr ||
                                       std::getenv("NINFER_DMA_PACK")[0] != '0';
    void* pack_device = pack_read_back ? pack_span.data : nullptr;
    void* host_term_device = std::getenv("NINFER_HOST_MERGE_CPU") != nullptr ? nullptr
                                                                           : host_term_span.data;
    std::optional<HostRoutedFetch> early;
    const auto fetch_begin = std::chrono::steady_clock::now();
    seg_pre_ms = std::chrono::duration<double, std::milli>(fetch_begin - t_layer_begin).count();
    if (!moe.stream->fully_resident()) {
        // Which shape to ask for is a measurement, not a policy: the controller already tracks how
        // many experts a layer has to move, and a pool that misses almost nothing only needs the ids.
        // A pool that misses often would pay the second transfer on top of the first, so it gets the
        // full read-back in one go, exactly as before.
        const double misses = policy_controller().signals.miss_per_layer;
        // Measured at 16 GiB, 512 then 1024 tokens in the same session: the ids-only shape does cut
        // the read-back from 0.070 to 0.022-0.031 ms a layer, and yet decode came out 136.8 against
        // 148.1 - the saved time reappears as ~85 us of unaccounted time per layer, and wall does
        // not move. Whatever the driver does with the small transfer costs at least what the pack
        // kernel and the bigger copy did. So it stays off by default; NINFER_FETCH_IDS turns it on
        // for whoever wants to chase that difference down.
        static const bool ids_requested = std::getenv("NINFER_FETCH_IDS") != nullptr;
        const bool ids_only = ids_requested && misses >= 0.0 && misses < 0.20;
        early.emplace(ids_only
                          ? fetch_ids_only(routes, execution.stream)
                          : fetch_host_routed(hidden, routes, residual, execution.stream,
                                              pack_device));
    }
    const double fetch_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - fetch_begin)
                                .count();
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
    // port (sm_89): O5, first half - keep a RAM copy of every expert this step routes to.
    //
    // A split layer's upload reads the image out of the artifact and copies it into the pool: 211 us
    // per expert at 8 GiB, of which roughly two thirds is that read. The host tier already exists and
    // the cache already prefers it as a source when the expert is there; what was missing is anyone
    // filling it on the split path (the tier is only touched by the whole-layer CPU fallback). With a
    // copy in RAM, a repeat miss becomes one DMA instead of a read plus a DMA.
    if (std::getenv("NINFER_HOST_MIRROR") != nullptr && early && !moe.stream->fully_resident()) {
        if (auto* host_tier = moe.stream->host()) {
            static thread_local std::vector<std::int32_t> routes;
            static thread_local std::vector<std::int32_t> warm_list;
            routes.clear();
            routes.reserve(early->ids.size());
            for (const auto id : early->ids) {
                if (id >= 0) { routes.push_back(static_cast<TokenId>(id)); }
            }
            // Admission by probability, not by contact: only an expert that has already been routed
            // a few times is worth a RAM slot. Everything else stays on disk, which is what keeps the
            // warm tier a partition instead of a second copy of the model.
            // Admission on eviction (1y.7 step 3b): what the Device pool just dropped is exactly the
            // set with a demonstrated short reuse distance. Admission on routes flooded the tier -
            // 7.3 of 8 per layer across 40 banks rolled a 3200-slot arena over in ~10 steps, and the
            // hit rate stayed at zero.
            const auto demoted = moe.stream->demoted_this_step();
            warm_list.assign(demoted.begin(), demoted.end());
            if (!warm_list.empty()) {
                const auto fill_begin = std::chrono::steady_clock::now();
                host_tier->ensure(ExpertBankId{moe.stream->bank_index()}, warm_list, nullptr);
                const double fill_us = std::chrono::duration<double, std::micro>(
                                           std::chrono::steady_clock::now() - fill_begin)
                                           .count();
                // What one admitted expert costs to keep, so the controller can weigh it against
                // what one hit saves (1y.7 step 3f).
                auto& policy = policy_controller();
                const double per_expert = fill_us / static_cast<double>(warm_list.size());
                policy.signals.mirror_fill_us_per_expert +=
                    (per_expert - policy.signals.mirror_fill_us_per_expert) / 16.0;
                // Does the tier actually answer for what it was just given? A zero hit rate in the
                // cache is ambiguous between "the policy admits the wrong experts" and "the tier
                // cannot hand them back"; this one probe separates them.
                if (moe.stream->bank_index() == 0) {
                    std::uint64_t probe_bytes = 0;
                    const auto* probe_image = host_tier->arena_image(
                        moe.stream->bank_index(), static_cast<std::uint32_t>(warm_list.front()),
                        probe_bytes);
                    static std::uint64_t probes = 0, answered = 0;
                    ++probes;
                    if (probe_image != nullptr && probe_bytes != 0) { ++answered; }
                    if (probes % 40 == 0) {
                        std::fprintf(stderr, "[mirror] tier answers %.0f%% (bytes=%llu)\n",
                                     100.0 * static_cast<double>(answered) / 40.0,
                                     static_cast<unsigned long long>(probe_bytes));
                        std::fflush(stderr);
                        answered = 0;
                    }
                }
            }
            // One line per token: how many experts the admission rule let into the warm tier. A zero
            // here means the policy filtered everything (the tier is empty for a reason we can name);
            // a healthy number with an empty arena means the tier is not answering instead.
            static std::uint64_t mirror_layers = 0;
            static std::uint64_t mirror_admitted = 0;
            static std::uint64_t mirror_routes = 0;
            mirror_admitted += warm_list.size();
            mirror_routes += routes.size();
            if (++mirror_layers % 40 == 0) {
                std::fprintf(stderr, "[mirror] admitted=%.1f/layer of %.1f routes\n",
                             static_cast<double>(mirror_admitted) / 40.0,
                             static_cast<double>(mirror_routes) / 40.0);
                std::fflush(stderr);
                mirror_admitted = 0;
                mirror_routes = 0;
            }
        }
    }
    if (layer_timing) {
        const auto cost = moe.stream->last_cost();
        auto& state     = timeline();
        state.seg_pre += seg_pre_ms;
        state.seg_ensure += std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - fetch_begin)
                                .count() - fetch_ms;
        state.ens_route += cost.route_ms;
        state.ens_layout += cost.layout_ms;
        state.ens_upload += cost.upload_ms;
    }
    if (!moe.stream->fully_resident()) {
        // Split layer: the pool holds the experts this layer routes to most, the host tier covers
        // the rest. Both halves land in FP32 and are folded into the residual once, so a split
        // layer rounds exactly like a resident or a fully hosted one.
        auto fetch           = std::move(*early);
        const auto mask_host = moe.stream->resident_mask_host();
        // A layer that only brought the ids down has to fetch the rest now: the host half needs x,
        // the residual before the layer, and the coefficients. This is the case the cheap shape was
        // betting against, so it is the one place that pays twice.
        if (fetch.ids_only) {
            auto full = fetch_host_routed(hidden, routes, residual, execution.stream, pack_device);
            if (full.ids != fetch.ids) {
                std::fprintf(stderr,
                             "[fetch] ids changed between the two read-backs (bank=%u) - using the "
                             "later one\n",
                             moe.stream->bank_index());
                std::fflush(stderr);
            }
            fetch = std::move(full);
        }
        // The pool decides how a layer MAY run; the routing decides how it HAS to run. When every
        // route this step publishes is already resident, the host half is empty and the whole split
        // apparatus - the masked subset kernel, the FP32 intermediate, the merge - computes what the
        // resident path computes directly. Measured at 12 GiB: coverage is already 100% there and
        // decode was still 45.9 tok/s against 214-235 when the layer is fully resident, because the
        // split path was taken on the pool's shape alone.
        bool all_resident = true;
        std::uint64_t missing = 0;
        for (std::size_t i = 0; i < fetch.ids.size(); ++i) {
            const auto expert = fetch.ids[i];
            if (expert < 0 ||
                static_cast<std::size_t>(expert) >= mask_host.size() || !mask_host[expert]) {
                ++missing;
                all_resident = false;
            }
        }
        if (layer_timing) { timeline().missing += missing; }
        // port (sm_89): O1's counter. The Device half counted the routes its mask dropped; the host
        // just counted the same routes from the ids it fetched, and that host count is the only one
        // that can be checked - it comes from data, not from the mask the placement itself wrote.
        // The reading is one layer behind (the fetch that synchronised only covered the layers before
        // this one), which is what observe_missed compares against. Off unless NINFER_O1_MISS_COUNT.
        if (moe.stream->miss_count_enabled()) {
            // Diagnostic: on a prefill layer the Device count and the host count are the same
            // quantity only if both looked at the same mask. Compare the mask the kernel will read
            // with the host's copy, once.
            static bool mask_compared = false;
            if (!mask_compared && hidden.ne[1] > 1 && std::getenv("NINFER_O1_DEBUG") != nullptr) {
                mask_compared = true;
                std::vector<std::int32_t> device_mask(mask_host.size());
                CUDA_CHECK(cudaMemcpy(device_mask.data(), moe.stream->resident_mask(),
                                      device_mask.size() * sizeof(std::int32_t),
                                      cudaMemcpyDeviceToHost));
                std::size_t host_ones = 0, dev_ones = 0, differ = 0;
                for (std::size_t i = 0; i < device_mask.size(); ++i) {
                    const bool h = mask_host[i] != 0;
                    const bool d = device_mask[i] != 0;
                    host_ones += h;
                    dev_ones += d;
                    differ += h != d;
                }
                std::fprintf(stderr, "[o1m] bank=%u host_ones=%llu dev_ones=%llu differ=%llu\n",
                             moe.stream->bank_index(),
                             static_cast<unsigned long long>(host_ones),
                             static_cast<unsigned long long>(dev_ones),
                             static_cast<unsigned long long>(differ));
                std::fflush(stderr);
            }
            moe.stream->observe_missed(moe.stream->missed_routes_total(), missing,
                                       hidden.ne[1] == 1);
            const auto& tally = moe.stream->miss_tally();
            if ((tally.compared + tally.prefill_compared) % 40 == 0) {
                std::fprintf(stderr,
                             "[o1] decode=%llu mismatch=%llu | prefill=%llu mismatch=%llu | "
                             "routes device=%llu host=%llu\n",
                             static_cast<unsigned long long>(tally.compared),
                             static_cast<unsigned long long>(tally.mismatch),
                             static_cast<unsigned long long>(tally.prefill_compared),
                             static_cast<unsigned long long>(tally.prefill_mismatch),
                             static_cast<unsigned long long>(tally.device),
                             static_cast<unsigned long long>(tally.host));
                std::fflush(stderr);
            }
        }
        // The controller reads the same nodes the panel shows, one layer at a time. It is in shadow
        // mode (no Device branch in the kernels yet): it prints what it would take and the numbers
        // behind it, and changes nothing.
        if (layer_timing) {
            auto& control = policy_controller();
            if (control.layers == 0) {
                control.probe_link(static_cast<std::uint64_t>(2U << 20));
            }
            control.observe_fetch(fetch_ms * 1000.0);
            // `missing` here is the empty half of the mask, which the placement itself decided, so it
            // is always zero on a covered step. The controller needs what actually moved.
            const auto moved = moe.stream->last_cost().uploaded;
            control.observe_upload(moe.stream->last_cost().upload_ms * 1000.0, moved);
            control.observe_misses(moved);
            // Layer cost, for the actuator guardrail: a change that costs >3% gets reverted.
            control.observe_layer(moe.stream->last_cost().route_ms +
                                      moe.stream->last_cost().layout_ms +
                                      moe.stream->last_cost().upload_ms,
                                  timeline().last_gpu, timeline().last_wall);
            // GH1 has a source now: the same counters the [gh1] panel prints. Prefetch usefulness
            // still has none (the queue has no consumer), so it stays "not measured".
            const auto rates = moe.stream->gh1_rates();
            control.observe_coverage(rates.static_pct, rates.no_miss_pct, rates.capturable_pct);
            // O1's expected hit rate: how often a covered layer follows a covered one. Printed with
            // the rest so the speculative design is judged by a number before it is built.
            static std::uint64_t pred_reports = 0;
            if (++pred_reports % 40 == 0) {
                std::fprintf(stderr, "[pred] covered-after-covered=%.1f%% trials=%llu\n",
                             100.0 * moe.stream->covered_predictor_rate(),
                             static_cast<unsigned long long>(pred_reports));
                std::fflush(stderr);
            }
            // Warm-tier sides: what a cold read costs (measured in upload_one) and how often the
            // pinned copy answers instead. The mirror decision is the ratio between them.
            {
                const auto sources = moe.stream->upload_source_stats();
                if (sources.file_experts > 0) {
                    const double cold = sources.read_us / static_cast<double>(sources.file_experts);
                    control.signals.cold_read_us_per_expert +=
                        (cold - control.signals.cold_read_us_per_expert) / 16.0;
                }
                const auto total = sources.arena_experts + sources.file_experts;
                static std::uint64_t last_probes = 0;
                if (sources.arena_probes != last_probes) {
                    std::fprintf(stderr, "[warm] probes=%llu file_experts=%llu read_us=%.0f\n",
                                 static_cast<unsigned long long>(sources.arena_probes),
                                 static_cast<unsigned long long>(sources.file_experts),
                                 sources.read_us);
                    std::fflush(stderr);
                    last_probes = sources.arena_probes;
                }
                if (total > 0) {
                    const double ratio =
                        static_cast<double>(sources.arena_experts) / static_cast<double>(total);
                    control.signals.mirror_hit_ratio += (ratio - control.signals.mirror_hit_ratio) / 16.0;
                }
            }
            control.step(false, false, false, false);
            control.report();
        }
        if (all_resident) {
            // Same call as the fully resident tail: empty subset, BF16 accumulation into the
            // residual. Bit-identical by construction, which is what the PPL gate checks.
            try {
                MemoryArena arena(MemorySpan{scratch.data, scratch.bytes, execution.device()});
                timeline_gpu_begin(execution.stream);
                ops::routed_swiglu(hidden, routed, moe.stream->experts(), routes, {},
                                   ops::ScaledSwiGLUTerm{moe.shared, shared_coefficients},
                                   ops::MoeEpilogue::AddResidual, residual, arena, execution);
                timeline_gpu_end(execution.stream);
            } catch (const std::exception& error) {
                throw std::invalid_argument(std::string("expert stream run: ") + error.what());
            }
            if (std::getenv("NINFER_SPLIT_TIMING") != nullptr) {
                static std::uint64_t fast = 0;
                if (++fast % 40 == 0) {
                    std::fprintf(stderr, "[split] all-resident fast path x%llu\n",
                                 static_cast<unsigned long long>(fast));
                    std::fflush(stderr);
                }
            }
            return;
        }
        auto mixed_f32 = workspace.alloc(DType::FP32, {hidden.ne[0], columns});
        MemoryArena mixed_arena(MemorySpan{scratch.data, scratch.bytes, execution.device()});
        const auto mask_size = static_cast<std::int32_t>(mask_host.size());
        Tensor device_mask(const_cast<void*>(moe.stream->resident_mask()), execution.device(),
                           DType::I32, {mask_size});
        const auto moe_run = [&] {
            // A capture cannot record an event, so the Device clock is only taken when the graph is
            // off. With it on the panel would be wrong rather than missing, which is worse.
            const bool timed = layer_timing && !LayerGraphs::instance().enabled();
            if (timed) { timeline_gpu_begin(execution.stream); }
            ops::routed_swiglu(hidden, routed, moe.stream->experts(), routes,
                               ops::ExpertSubset{ops::ExpertSubsetKind::MaskOnes, device_mask},
                               ops::ScaledSwiGLUTerm{moe.shared, shared_coefficients},
                               ops::MoeEpilogue::Write, mixed_f32, mixed_arena, execution,
                               moe.stream->missed_routes_device());
            if (timed) { timeline_gpu_end(execution.stream); }
        };
        const auto dev_begin = std::chrono::steady_clock::now();
        auto& graphs = LayerGraphs::instance();
        if (graphs.enabled()) {
            // What the capture baked in is the argument list, so the key is the set of buffer
            // addresses the kernels read and write. A moved buffer changes an argument the capture
            // recorded and the graph has to be rebuilt.
            //
            // port (sm_89): the mask bytes are deliberately NOT part of the key. The kernels read
            // the resident mask through one fixed device pointer that `ensure` rewrites every
            // layer, so a capture stays valid whatever the mask holds; the contents are not an
            // argument, only the pointer is. Mixing in all 256 bits is what made the key change on
            // every layer and sent the graph down its rebuild branch every time - measured
            // captures=40 replays=0, i.e. the graph never ran from a cached instantiation at all,
            // which is why it was slower than plain launches instead of faster.
            std::uint64_t key = 1469598103934665603ULL;
            const auto mix    = [&key](std::uint64_t value) {
                key ^= value;
                key *= 1099511628211ULL;
            };
            mix(reinterpret_cast<std::uintptr_t>(hidden.data));
            mix(reinterpret_cast<std::uintptr_t>(residual.data));
            mix(reinterpret_cast<std::uintptr_t>(mixed_f32.data));
            mix(reinterpret_cast<std::uintptr_t>(routes.ids.data));
            mix(reinterpret_cast<std::uintptr_t>(routes.coefficients.data));
            mix(reinterpret_cast<std::uintptr_t>(moe.stream->resident_mask()));
            mix(reinterpret_cast<std::uintptr_t>(shared_coefficients.data));
            if (std::getenv("NINFER_LAYER_GRAPH_DEBUG") != nullptr) {
                static std::size_t shown = 0;
                if (shown++ < 160) {
                    std::fprintf(stderr,
                                 "[graph-key] bank=%u cols=%d key=%016llx h=%p res=%p mix=%p ids=%p "
                                 "coef=%p sh=%p\n",
                                 moe.stream->bank_index(), columns, (unsigned long long)key,
                                 static_cast<const void*>(hidden.data),
                                 static_cast<const void*>(residual.data),
                                 static_cast<const void*>(mixed_f32.data),
                                 static_cast<const void*>(routes.ids.data),
                                 static_cast<const void*>(routes.coefficients.data),
                                 static_cast<const void*>(shared_coefficients.data));
                }
            }
            graphs.run(moe.stream->bank_index(), key, execution.stream, moe_run);
            if (std::getenv("NINFER_LAYER_GRAPH_REPORT") != nullptr) {
                static std::uint64_t calls = 0;
                if (++calls % 40 == 0) { graphs.report(); }
            }
        } else {
            moe_run();
        }
        const auto dev_end = std::chrono::steady_clock::now();
        Tensor host_mask(const_cast<std::int32_t*>(mask_host.data()), Device::cpu(), DType::I32,
                         {mask_size});
        host_routed_ffn(fetch, moe,
                        ops::ExpertSubset{ops::ExpertSubsetKind::MaskZeros, host_mask}, mixed_f32,
                        residual, execution, host_term_device);
        if (std::getenv("NINFER_SPLIT_TIMING") != nullptr) {
            // Where the split path's per-layer time actually goes. Without this the 4x gap against
            // full residency could only be guessed at: the expert bookkeeping measures ~0.01 ms, so
            // the cost has to be in the device subset run or the host merge - and which one decides
            // what to remove.
            static double dev_ms = 0.0;
            static double host_ms = 0.0;
            static std::uint64_t layers = 0;
            const auto host_end = std::chrono::steady_clock::now();
            dev_ms += std::chrono::duration<double, std::milli>(dev_end - dev_begin).count();
            host_ms += std::chrono::duration<double, std::milli>(host_end - dev_end).count();
            if (++layers % 40 == 0) {
                std::fprintf(stderr, "[split] layer dev=%.3fms host=%.3fms\n", dev_ms / 40.0,
                             host_ms / 40.0);
                std::fflush(stderr);
                dev_ms = 0.0;
                host_ms = 0.0;
            }
        }
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
            // streamed_ffn() also allocates the read-back pack and the CPU merge term straight from
            // the arena, and this formula used to leave both out. Measured consequence: the plan
            // reported 72,097,792 for a 1024-column chunk while a 921-column prefill already needed
            // 72,195,840, so anything above ~900 columns died in layer 0 with std::bad_alloc. The
            // merge term alone is hidden x columns x 4 -- 8.4 MB at 1024 -- which is the size of the
            // gap that was measured.
            const std::size_t handoff_x =
                static_cast<std::size_t>(moe->profile.routed.ffn.hidden) *
                static_cast<std::size_t>(last);
            const std::size_t handoff_ids =
                static_cast<std::size_t>(moe->profile.routed.top_k) *
                static_cast<std::size_t>(last);
            const std::size_t pack = ops::moe_pack_host_fetch_bytes(
                handoff_x * 2, handoff_x * 2, handoff_ids * 4, handoff_ids * 4);
            const std::size_t host_term = handoff_x * sizeof(float);
            return stream_route_tensor_bytes(*moe, last) + shared_f32 +
                   std::max(stream_route_bytes(*moe, first, last),
                            stream_expert_bytes(*moe, first, last)) +
                   pack + host_term;
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




