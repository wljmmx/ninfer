// Times the CPU routed-SwiGLU expert kernel on the 35B-A3B shape (hidden 2048, intermediate 512,
// 256 experts, top_k 8). A split layer pays this per expert that misses the Device pool, so the
// number to drive down is milliseconds per expert at one token, not tokens per second of a whole
// decode - the decode also pays uploads, arena loads and the Device half.
#include "core/cpu/executor.h"
#include "core/cpu/system.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight_bank.h"
#include "ninfer/ops/moe.h"

#include <algorithm>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <vector>

using namespace ninfer;

namespace {

constexpr int kHidden       = 2048;
constexpr int kIntermediate = 512;
constexpr int kExperts      = 256;
constexpr int kTopK         = 8;
constexpr int kGateUpRows   = 2 * kIntermediate;

struct Bank {
    std::vector<std::byte> storage;
    std::vector<std::uint64_t> offsets;
    WeightParent parent;
    WeightBankView view;
};

void fill(std::span<std::byte> slot, const WeightBankLayout& layout, const WeightParent& parent,
          std::uint32_t seed) {
    // Codes are what the kernel actually decodes; scales stay at one so the arithmetic cannot
    // overflow into a slower path, and the high planes stay zero for the same reason.
    auto* codes = slot.data() + layout.codes.offset;
    std::uint32_t state = seed * 2654435761u + 1u;
    for (std::uint64_t i = 0; i < layout.codes.expert_bytes; ++i) {
        state      = state * 1664525u + 1013904223u;
        codes[i]   = static_cast<std::byte>(state >> 24);
    }
    if (layout.high.expert_bytes) {
        std::memset(slot.data() + layout.high.offset, 0,
                    static_cast<std::size_t>(layout.high.expert_bytes));
    }
    auto* scales = reinterpret_cast<std::uint16_t*>(slot.data() + layout.scales.offset);
    for (std::uint64_t i = 0; i < layout.scales.expert_bytes / 2; ++i) {
        scales[i] = 0x3c00; // fp16 1.0
    }
    (void)parent;
}

Bank make_bank(QType type, QuantLayout quant, std::uint64_t rows, std::uint64_t columns) {
    Bank bank;
    const std::uint64_t shape[] = {std::uint64_t{kExperts} * rows, columns};
    bank.parent.geometry        = weight_geometry(type, quant, shape);
    bank.parent.data            = nullptr;
    bank.parent.device          = Device::cpu();
    bank.parent.weight_divisors = {};

    bank.view.parent = &bank.parent;
    bank.view.layout = describe_weight_bank(bank.parent.geometry,
                                            WeightBankRegion{0, kExperts, rows, rows});
    // Mirrors the host tier: every expert is one packed segment, so the plane offsets are the
    // per-expert ones and the table carries the segment origin.
    const auto& layout = bank.view.layout;
    bank.view.layout.codes.offset  = 0;
    bank.view.layout.high.offset   = layout.codes.expert_bytes;
    bank.view.layout.scales.offset = layout.codes.expert_bytes + layout.high.expert_bytes;

    std::uint64_t slot = bank.view.layout.codes.expert_bytes +
                         bank.view.layout.high.expert_bytes +
                         bank.view.layout.scales.expert_bytes;
    slot = (slot + 131071) / 131072 * 131072;
    bank.storage.assign(static_cast<std::size_t>(slot) * kExperts, std::byte{0});
    bank.offsets.resize(kExperts);
    for (int expert = 0; expert < kExperts; ++expert) {
        bank.offsets[expert] = static_cast<std::uint64_t>(expert) * slot;
    }
    bank.parent.data           = bank.storage.data();
    bank.parent.geometry.bytes = bank.storage.size();
    bank.view.expert_offsets   = bank.offsets.data();
    for (int expert = 0; expert < kExperts; ++expert) {
        fill(std::span<std::byte>(bank.storage.data() + bank.offsets[expert], slot),
             bank.view.layout, bank.parent, static_cast<std::uint32_t>(expert));
    }
    return bank;
}

} // namespace

int main(int argc, char** argv) {
    const int iterations = argc > 1 ? std::atoi(argv[1]) : 200;
    const int workers    = argc > 2 ? std::atoi(argv[2]) : 0;

    static CpuTopology topology;
    // plan_cpu_team() rejects a team wider than the processors it selected, and an exception
    // escaping main() aborts with 0xC0000409 - which reads as a memory bug. Report the ceiling
    // instead: the bench is the tool that tells people how wide the team may be.
    const auto selectable = cpu_worker_order(topology.info(), false).size();
    const auto team       = workers > 0 ? static_cast<std::uint32_t>(workers)
                                        : static_cast<std::uint32_t>(std::max<std::size_t>(1, selectable));
    if (team > selectable) {
        std::fprintf(stderr,
                     "moe cpu bench: %u workers requested but only %zu cores are selectable "
                     "without SMT\n",
                     team, selectable);
        return 2;
    }
    static CpuExecutor executor(topology, plan_cpu_team(topology.info(), team));
    const auto execution = executor.execution_view();

    // port (sm_89): the shipped artifact stores the down projection as Q5 (one high bit per
    // weight), not Q6, so the bench has to be able to time the shape it will actually meet.
    // NINFER_BENCH_DOWN_Q5=1 switches the down bank; Q6 stays the default for the older numbers.
    static const bool down_q5 = std::getenv("NINFER_BENCH_DOWN_Q5") != nullptr;
    const auto down_type      = down_q5 ? QType::Q5_G64_FP16 : QType::Q6_G64_FP16;

    auto gate_up = make_bank(QType::Q4_G64_FP16, QuantLayout::RowSplit, kGateUpRows, kHidden);
    auto down    = make_bank(down_type, QuantLayout::RowSplit, kHidden, kIntermediate);

    ops::RoutedSwiGLUProfile profile{};
    profile.ffn.hidden         = kHidden;
    profile.ffn.intermediate   = kIntermediate;
    profile.ffn.gate_up_type   = QType::Q4_G64_FP16;
    profile.ffn.gate_up_layout = QuantLayout::RowSplit;
    profile.ffn.down_type      = down_type;
    profile.ffn.down_layout    = QuantLayout::RowSplit;
    profile.ffn.gate_up_policy = ops::LinearPolicy::A16Only;
    profile.ffn.down_policy    = ops::LinearPolicy::A16Only;
    profile.experts            = kExperts;
    profile.top_k              = kTopK;

    ops::CpuSwiGLUExperts experts{gate_up.view, down.view};

    // 2^-10 with a small spread: with unit-scale weights and a 2048-wide dot product the gate
    // pre-activation would reach a few thousand at full scale, and silu saturates there - the
    // activation plane comes out exactly zero and the check below cannot tell a correct kernel
    // from a broken one. The spread matters too: a constant row quantises losslessly, which would
    // hide the int8 path's rounding instead of measuring it.
    std::vector<std::uint16_t> x(static_cast<std::size_t>(kHidden));
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<std::uint16_t>(0x3a80 + (i % 4) * 0x08);
    }
    std::vector<float> out(static_cast<std::size_t>(kHidden), 0.0F);
    std::vector<std::int32_t> ids(kTopK);
    std::vector<float> coefficients(kTopK, 0.125F);
    std::vector<std::int32_t> mask(kExperts, 1);
    for (int j = 0; j < kTopK; ++j) { ids[j] = 40 + j * 7; }

    Tensor x_tensor(x.data(), Device::cpu(), DType::BF16, {kHidden, 1});
    Tensor out_tensor(out.data(), Device::cpu(), DType::FP32, {kHidden, 1});
    ops::ExpertRoutes routes{Tensor(ids.data(), Device::cpu(), DType::I32, {kTopK, 1}),
                             Tensor(coefficients.data(), Device::cpu(), DType::FP32, {kTopK, 1})};
    Tensor mask_tensor(mask.data(), Device::cpu(), DType::I32, {kExperts});

    const auto capacity =
        ops::routed_swiglu_workspace_capacity_bytes(profile, ops::ExpertSubsetKind::All, 1, 1);
    std::vector<std::byte> scratch(capacity + 256);
    const auto base    = reinterpret_cast<std::uintptr_t>(scratch.data());
    const auto aligned = (256 - (base & 255)) & 255;

    // The engine allocates this per layer, so its size is the cost of the per-layer scratch.
    std::printf("moe cpu bench: workspace=%.1f KiB per call (engine allocates it per layer)\n",
                capacity / 1024.0);
    std::printf("moe cpu bench: hidden=%d intermediate=%d experts=%d top_k=%d workers=%u\n", kHidden,
                kIntermediate, kExperts, kTopK, execution.worker_count());

    // port (sm_89): NINFER_BENCH_ACTIVE keeps the team spinning between calls, the same thing
    // NINFER_CPU_TEAM_ACTIVE does in the engine. Combined with NINFER_BENCH_GAP_US it separates
    // "the team is parked" from "the call is expensive".
    static std::optional<CpuExecutionView::ActiveScope> bench_active;
    if (std::getenv("NINFER_BENCH_ACTIVE") != nullptr) { bench_active.emplace(execution.keep_active()); }

    for (const int misses : {1, 2, 4, 8}) {
        const auto subset = ops::ExpertSubset{ops::ExpertSubsetKind::MaskOnes, mask_tensor};
        MemoryArena arena(MemorySpan{scratch.data() + aligned, capacity, Device::cpu()});

        // port (sm_89): rotate the routed experts every call. Repeating one set kept its 15 MB of
        // weights in L2/L3, so the bench measured instruction throughput while the real layer reads
        // a different expert every step out of a 4 GiB tier - 0.046 ms per expert against 0.077 to
        // 0.16 in the engine. Stepping through the bank with a stride coprime to the count makes
        // every call cold, which is what the engine actually pays.
        //
        // NINFER_BENCH_STATIC pins the set instead, which is the other half of that comparison:
        // the same weights stay resident, so the difference against the rotating run is what the
        // memory path costs per call - address translation included.
        static const bool fixed_set = std::getenv("NINFER_BENCH_STATIC") != nullptr;
        const auto route = [&](int iteration) {
            const auto base = fixed_set ? 0U
                                        : static_cast<std::uint32_t>(iteration * kTopK) % kExperts;
            std::fill(mask.begin(), mask.end(), 0);
            // The op rejects duplicate route IDs, so every one of the top_k entries is filled with
            // a distinct expert; the mask alone decides how many of them are computed.
            for (int j = 0; j < kTopK; ++j) {
                const auto expert = static_cast<std::int32_t>((base + j * 37U) % kExperts);
                ids[j]            = expert;
                if (j < misses) { mask[expert] = 1; }
            }
        };

        // Warm the caches and the team before timing.
        for (int i = 0; i < 16; ++i) {
            route(i);
            arena.reset();
            ops::routed_swiglu(x_tensor, profile, experts, routes, subset, out_tensor, arena,
                               execution);
        }
        // Time each call on its own and report the minimum. The mean on this box moves by 3x
        // between runs because a hybrid P/E machine schedules the team differently, and a change
        // measured against it is indistinguishable from noise; the fastest call is the one where
        // the team got out of its own way, which is the number a kernel change should move.
        // port (sm_89): the engine calls this once per layer, with a GPU half in between, so the
        // team is idle every time it is asked to run. A tight loop never pays that. This knob puts
        // a gap in front of each timed call so the bench can measure the wake, which is the差
        // between 0.08 ms here and 0.5-0.9 ms in the engine.
        static const auto gap_us = [] {
            const char* value = std::getenv("NINFER_BENCH_GAP_US");
            return value != nullptr ? std::atoi(value) : 0;
        }();
        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(iterations));
        for (int i = 0; i < iterations; ++i) {
            route(i + 16);
            arena.reset();
            if (gap_us > 0) { std::this_thread::sleep_for(std::chrono::microseconds(gap_us)); }
            const auto call_begin = std::chrono::steady_clock::now();
            ops::routed_swiglu(x_tensor, profile, experts, routes, subset, out_tensor, arena,
                               execution);
            samples.push_back(std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - call_begin)
                                  .count());
        }
        std::sort(samples.begin(), samples.end());
        const auto best   = samples.front();
        const auto median = samples[samples.size() / 2];
        std::printf("  misses=%d  min %.3f ms  median %.3f ms  %.3f ms/expert  %.1f GMAC/s\n",
                    misses, best, median, best / misses, 3.146 * misses / best);
    }

    // Numerical check: one fixed call over all eight experts, printed so the int8 path can be
    // compared against the float one. The split gate allows 1%, and the int8 activation rounding
    // is meant to stay well inside that.
    {
        std::fill(mask.begin(), mask.end(), 1);
        for (int j = 0; j < kTopK; ++j) { ids[j] = 40 + j * 7; }
        const auto subset = ops::ExpertSubset{ops::ExpertSubsetKind::MaskOnes, mask_tensor};
        MemoryArena arena(MemorySpan{scratch.data() + aligned, capacity, Device::cpu()});
        std::fill(out.begin(), out.end(), 0.0F);
        ops::routed_swiglu(x_tensor, profile, experts, routes, subset, out_tensor, arena, execution);
        double sum = 0.0, norm = 0.0;
        for (const float value : out) {
            sum += value;
            norm += static_cast<double>(value) * value;
        }
        std::printf("  check  sum %.9g  l2 %.9g  first %.9g %.9g %.9g\n", sum, std::sqrt(norm),
                    out[0], out[1], out[2]);
    }
    return 0;
}

