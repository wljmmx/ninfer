// Times the CPU half of a Qwen3.8-Flash-Next (Qwen4Exp) split MoE layer: three separate banks
// (gate Q4, up Q4, down Q6, all RowSplit) at hidden 2560 / intermediate 640, the geometry the
// shipped v5_e2 artifact carries.
//
// The number this exists to produce is the one a cost decision needs: how many seconds one expert
// costs the CPU at a given token count. The controller compares that against the expert's upload
// (bytes / probed link) per expert, so the units here are per expert, not per layer and not tokens
// per second. NINFER_MOE_HOST_ONLY cannot stand in for this: it only reaches the *fused* ensure()
// path, so on this artifact it reports cpu=0.00ms while the split path simply is not asked.
#include "core/cpu/executor.h"
#include "core/cpu/system.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight_bank.h"
#include "ninfer/ops/moe.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

using namespace ninfer;

namespace {

constexpr int kHidden       = 2560;
constexpr int kIntermediate = 640;
constexpr int kExperts      = 64;
constexpr int kTopK         = 10;
// One expert, one token: gate (I x H) + up (I x H) + down (H x I), two flops per MAC.
constexpr double kFlopsPerExpertToken =
    2.0 * 2.0 * double(kIntermediate) * double(kHidden) + 2.0 * double(kHidden) * double(kIntermediate);
// The routing the benchmark describes: every expert active, every token routed to top_k of them.

struct Bank {
    std::vector<std::byte> storage;
    std::vector<std::uint64_t> offsets;
    WeightParent parent;
    WeightBankView view;
};

void fill(std::span<std::byte> slot, const WeightBankLayout& layout, std::uint32_t seed) {
    auto* codes         = slot.data() + layout.codes.offset;
    std::uint32_t state = seed * 2654435761u + 1u;
    for (std::uint64_t i = 0; i < layout.codes.expert_bytes; ++i) {
        state    = state * 1664525u + 1013904223u;
        codes[i] = static_cast<std::byte>(state >> 24);
    }
    if (layout.high.expert_bytes) {
        std::memset(slot.data() + layout.high.offset, 0,
                    static_cast<std::size_t>(layout.high.expert_bytes));
    }
    auto* scales = reinterpret_cast<std::uint16_t*>(slot.data() + layout.scales.offset);
    for (std::uint64_t i = 0; i < layout.scales.expert_bytes / 2; ++i) {
        scales[i] = 0x3c00; // fp16 1.0
    }
}

Bank make_bank(QType type, std::uint64_t rows, std::uint64_t columns) {
    Bank bank;
    const std::uint64_t shape[] = {std::uint64_t{kExperts} * rows, columns};
    bank.parent.geometry        = weight_geometry(type, QuantLayout::RowSplit, shape);
    bank.parent.data            = nullptr;
    bank.parent.device          = Device::cpu();
    bank.parent.weight_divisors = {};

    bank.view.parent = &bank.parent;
    bank.view.layout = describe_weight_bank(bank.parent.geometry,
                                            WeightBankRegion{0, kExperts, rows, rows});
    // Mirrors the host tier: every expert is one packed segment, so the planes sit back to back
    // inside a slot and the table carries the segment origin.
    const auto& layout             = bank.view.layout;
    bank.view.layout.codes.offset  = 0;
    bank.view.layout.high.offset   = layout.codes.expert_bytes;
    bank.view.layout.scales.offset = layout.codes.expert_bytes + layout.high.expert_bytes;

    std::uint64_t slot = bank.view.layout.codes.expert_bytes + bank.view.layout.high.expert_bytes +
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
             bank.view.layout, static_cast<std::uint32_t>(expert));
    }
    return bank;
}

} // namespace

int run(int iterations, int workers) {
    static CpuTopology topology;
    const auto selectable = cpu_worker_order(topology.info(), false).size();
    const auto team       = workers > 0 ? static_cast<std::uint32_t>(workers)
                                        : static_cast<std::uint32_t>(std::max<std::size_t>(1, selectable));
    if (team > selectable) {
        std::fprintf(stderr, "moe cpu split bench: %u workers requested, %zu selectable\n", team,
                     selectable);
        return 2;
    }
    static CpuExecutor executor(topology, plan_cpu_team(topology.info(), team));
    const auto execution = executor.execution_view();

    auto gate = make_bank(QType::Q4_G64_FP16, kIntermediate, kHidden);
    auto up   = make_bank(QType::Q4_G64_FP16, kIntermediate, kHidden);
    auto down = make_bank(QType::Q6_G64_FP16, kHidden, kIntermediate);

    ops::RoutedSwiGLUProfile profile{};
    profile.ffn.hidden         = kHidden;
    profile.ffn.intermediate   = kIntermediate;
    profile.ffn.gate_up_type   = QType::Q4_G64_FP16;
    profile.ffn.gate_up_layout = QuantLayout::RowSplit;
    profile.ffn.down_type      = QType::Q6_G64_FP16;
    profile.ffn.down_layout    = QuantLayout::RowSplit;
    profile.ffn.gate_up_policy = ops::LinearPolicy::A16Only;
    profile.ffn.down_policy    = ops::LinearPolicy::A16Only;
    profile.experts            = kExperts;
    profile.top_k              = kTopK;

    ops::CpuSwiGLUExperts experts;
    // Named, not positional: the struct's first member is the fused `gate_up` view, which this form
    // must leave empty - filling it by position is exactly what the op refuses by name.
    experts.gate = gate.view;
    experts.up   = up.view;
    experts.down = down.view;

    std::printf("moe cpu split bench: hidden=%d intermediate=%d experts=%d top_k=%d workers=%u "
                "%.2f MFLOP/expert/token\n",
                kHidden, kIntermediate, kExperts, kTopK, execution.worker_count(),
                kFlopsPerExpertToken / 1e6);

    for (const int tokens : {1, 512}) {
        std::vector<std::uint16_t> x(static_cast<std::size_t>(kHidden) * tokens);
        for (std::size_t i = 0; i < x.size(); ++i) {
            x[i] = static_cast<std::uint16_t>(0x3a80 + (i % 4) * 0x08);
        }
        std::vector<float> out(static_cast<std::size_t>(kHidden) * tokens, 0.0F);
        std::vector<std::int32_t> ids(static_cast<std::size_t>(kTopK) * tokens);
        std::vector<float> coefficients(static_cast<std::size_t>(kTopK) * tokens, 0.125F);
        std::vector<std::int32_t> mask(kExperts, 1);
        // Every token routes to top_k distinct experts, walking the bank so each expert carries
        // about the same number of members at 512 tokens.
        for (int t = 0; t < tokens; ++t) {
            for (int j = 0; j < kTopK; ++j) {
                ids[static_cast<std::size_t>(t) * kTopK + j] =
                    static_cast<std::int32_t>((std::size_t(t) * kTopK + j) % kExperts);
            }
        }

        Tensor x_tensor(x.data(), Device::cpu(), DType::BF16, {kHidden, tokens});
        Tensor out_tensor(out.data(), Device::cpu(), DType::FP32, {kHidden, tokens});
        ops::ExpertRoutes routes{Tensor(ids.data(), Device::cpu(), DType::I32, {kTopK, tokens}),
                                 Tensor(coefficients.data(), Device::cpu(), DType::FP32,
                                        {kTopK, tokens})};
        Tensor mask_tensor(mask.data(), Device::cpu(), DType::I32, {kExperts});
        const auto subset = ops::ExpertSubset{ops::ExpertSubsetKind::MaskOnes, mask_tensor};

        const auto capacity = ops::split_routed_swiglu_workspace_capacity_bytes(
            profile, ops::ExpertSubsetKind::MaskOnes, tokens, tokens);
        std::vector<std::byte> scratch(capacity + 256);
        const auto base    = reinterpret_cast<std::uintptr_t>(scratch.data());
        const auto aligned = (256 - (base & 255)) & 255;

        const auto call = [&] {
            MemoryArena arena(MemorySpan{scratch.data() + aligned, capacity, Device::cpu()});
            ops::routed_swiglu(x_tensor, profile, experts, routes, subset, out_tensor, arena,
                               execution);
        };
        for (int i = 0; i < 8; ++i) { call(); } // warm the team and the pages
        std::vector<double> samples;
        for (int i = 0; i < iterations; ++i) {
            const auto begin = std::chrono::steady_clock::now();
            call();
            samples.push_back(
                std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
        }
        std::sort(samples.begin(), samples.end());
        const double best = samples.front();
        const double flops = kFlopsPerExpertToken * kExperts * tokens;
        std::printf("  tokens=%-4d min %.3f ms  %.1f us/expert  %.1f us/expert-token  %.2f GFLOPS\n",
                    tokens, best * 1e3, best / kExperts * 1e6, best / (double(kExperts) * tokens) * 1e6,
                    flops / best / 1e9);
    }
    return 0;
}

int main(int argc, char** argv) {
    const int iterations = argc > 1 ? std::atoi(argv[1]) : 3;
    const int workers    = argc > 2 ? std::atoi(argv[2]) : 0;
    // An exception escaping main() aborts with 0xC0000409, which reads as a memory bug rather than
    // "the op refused this geometry".
    try {
        return run(iterations, workers);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "moe cpu split bench: %s\n", error.what());
        return 1;
    }
}
