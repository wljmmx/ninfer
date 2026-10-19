// port (sm_89): offline diagnostic for the routed MoE kernels.
//
// The model layer is not involved beyond loading: this takes one layer's expert bank straight
// from the materialized parents and runs the shipped sparse op and the imported routed op on
// the same device weights with the same input. It reports the per-length maximum absolute
// difference, which the routing and expert kernels must keep at zero.
//
// usage: diag-moe <model.ninfer> [layer]
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight_bank.h"
#include "core/weight_view.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/model.h"
#include "models/qwen3_5/weights.h"
#include "ninfer/ops/moe.h"
#include "ninfer/ops/sparse_moe.h"
#include "ninfer/ops/weight_input.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;

void check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
    }
}

// One merged [rows,columns] matrix over a single artifact parent.
Weight merged(const WeightView& view, std::uint64_t rows, std::uint64_t columns) {
    if (view.parts.size() != 1 || !view.parts.front().parent) {
        throw std::runtime_error("matrix does not have a single parent");
    }
    const auto& part = view.parts.front();
    return native_weight({{rows, columns}, {{part.parent, part.begin, part.end}}});
}

// Two adjacent matrices of one parent, fused row-major: gate rows first, then up rows.
Weight merged_pair(const WeightView& first, const WeightView& second, std::uint64_t rows,
                   std::uint64_t columns) {
    if (first.parts.size() != 1 || second.parts.size() != 1) {
        throw std::runtime_error("paired matrix is not a single region");
    }
    const auto& a = first.parts.front();
    const auto& b = second.parts.front();
    if (a.parent != b.parent || a.end != b.begin) {
        throw std::runtime_error("paired matrices are not adjacent in one parent");
    }
    return native_weight({{rows, columns}, {{a.parent, a.begin, b.end}}});
}

} // namespace

static int run(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: diag-moe <model.ninfer> [layer]\n");
        return 2;
    }
    const std::size_t layer = argc > 2 ? static_cast<std::size_t>(std::atoi(argv[2])) : 0;

    DeviceContext device(0);
    device.bind_to_current_thread();

    std::printf("loading %s\n", argv[1]);
    auto model = models::qwen3_5::load_model(argv[1], {}, device);
    const auto& blocks = model->weights().text.layers;
    if (layer >= blocks.size()) { throw std::runtime_error("layer out of range"); }
    const auto* moe = std::get_if<models::qwen3_5::MoeWeights>(&blocks[layer].ffn);
    if (!moe) { throw std::runtime_error("selected layer is dense"); }

    const auto count        = static_cast<std::uint32_t>(moe->experts.size());
    const auto& gate        = model->weight(moe->experts.front().gate).view;
    const auto& down        = model->weight(moe->experts.front().down).view;
    const auto intermediate = gate.shape.at(0);
    const auto hidden       = gate.shape.at(1);
    std::printf("layer %zu: experts %u, hidden %llu, intermediate %llu\n", layer, count,
                static_cast<unsigned long long>(hidden),
                static_cast<unsigned long long>(intermediate));

    const auto& gu_parent = *gate.parts.at(0).parent;
    const auto& dn_parent = *down.parts.at(0).parent;
    const auto gu = bind_weight_bank(
        gu_parent,
        describe_weight_bank(gu_parent.geometry,
                             {gate.parts[0].begin / hidden, count, 2 * intermediate,
                              2 * intermediate}));
    const auto dn = bind_weight_bank(
        dn_parent,
        describe_weight_bank(dn_parent.geometry,
                             {down.parts[0].begin / down.shape.at(1), count, down.shape.at(0),
                              down.shape.at(0)}));

    // Routed operands: fused gate_up, down, and the [E+1,H] control rows.
    const auto& router = model->weight(moe->router).view;
    const auto& score  = model->weight(moe->shared_score).view;
    if (router.parts.front().parent != score.parts.front().parent ||
        router.parts.front().end != score.parts.front().begin) {
        throw std::runtime_error("control rows are not contiguous");
    }
    Weight control = native_weight({{router.shape.at(0) + 1, hidden},
                                    {{router.parts[0].parent, router.parts[0].begin,
                                      score.parts[0].end}}});

    const auto& shared_gate_v = model->weight(moe->shared.gate).view;
    const auto& shared_up_v   = model->weight(moe->shared.up).view;
    const auto& shared_down_v = model->weight(moe->shared.down).view;
    ops::SwiGLUWeights shared{
        ops::SingleProjectionWeight{
            merged_pair(shared_gate_v, shared_up_v, shared_gate_v.shape.at(0) * 2, hidden),
            ops::LinearPolicy::A16Only},
        ops::SingleProjectionWeight{merged(shared_down_v, shared_down_v.shape.at(0),
                                           shared_down_v.shape.at(1)),
                                    ops::LinearPolicy::A16Only}};

    constexpr std::int32_t top_k = 8;
    ops::CudaSwiGLUExperts routed{device_expert_bank(gu, dn), {}};
    const auto profile = ops::MoeProfile{
        ops::routed_swiglu_profile(gu, dn, top_k, ops::LinearPolicy::A16Only,
                                   ops::LinearPolicy::A16Only),
        ops::swiglu_profile(shared)};

    // Shipped sparse operands over the same artifact tensors.
    std::vector<ops::WeightInput> gate_up_in, down_in;
    gate_up_in.reserve(2 * count);
    down_in.reserve(count);
    for (const auto& expert : moe->experts) {
        gate_up_in.push_back(model->input(expert.gate));
        gate_up_in.push_back(model->input(expert.up));
        down_in.push_back(model->input(expert.down));
    }
    const auto sparse = ops::prepare_sparse_moe_weights(
        model->input(moe->router), model->input(moe->shared_score), gate_up_in, down_in,
        model->input(moe->shared.gate), model->input(moe->shared.up),
        model->input(moe->shared.down));

    const auto execution     = device.execution_view();
    const auto stream        = execution.stream;
    constexpr int max_tokens = 2048;

    __nv_bfloat16* x_device = nullptr;
    check(cudaMalloc(&x_device, hidden * max_tokens * sizeof(__nv_bfloat16)), "alloc input");
    {
        std::vector<__nv_bfloat16> host(hidden * max_tokens);
        std::mt19937 rng(12345);
        std::normal_distribution<float> normal(0.0F, 1.0F);
        for (auto& value : host) { value = __float2bfloat16_rn(normal(rng) * 1.0F); }
        check(cudaMemcpy(x_device, host.data(), host.size() * sizeof(__nv_bfloat16),
                         cudaMemcpyHostToDevice),
              "upload input");
    }

    __nv_bfloat16* out_sparse = nullptr;
    __nv_bfloat16* out_routed = nullptr;
    check(cudaMalloc(&out_sparse, hidden * max_tokens * sizeof(__nv_bfloat16)), "sparse output");
    check(cudaMalloc(&out_routed, hidden * max_tokens * sizeof(__nv_bfloat16)), "routed output");

    // Isolation: point both sides' shared experts at a zeroed buffer, which removes the shared
    // contribution from both ops without changing any geometry. If the two sides then agree, the
    // divergence lives in the shared path rather than in the routed experts.
    const auto zero_gu_bytes = sparse.shared_gate_up.payload_bytes;
    const auto zero_dn_bytes = sparse.shared_down.payload_bytes;
    void* zero_gu            = nullptr;
    void* zero_dn            = nullptr;
    check(cudaMalloc(&zero_gu, zero_gu_bytes ? zero_gu_bytes : 1), "alloc zero gate_up");
    check(cudaMalloc(&zero_dn, zero_dn_bytes ? zero_dn_bytes : 1), "alloc zero down");
    check(cudaMemset(zero_gu, 0, zero_gu_bytes ? zero_gu_bytes : 1), "clear zero gate_up");
    check(cudaMemset(zero_dn, 0, zero_dn_bytes ? zero_dn_bytes : 1), "clear zero down");
    const auto blank = [](const Weight& source, void* buffer) {
        // Zeroing the codes alone is enough: every dot product then reads zero, while the
        // geometry and the high/scale planes stay exactly as the ops expect them.
        Weight out  = source;
        out.payload = buffer;
        out.qdata   = buffer;
        return out;
    };
    auto sparse_blank           = sparse;
    sparse_blank.shared_gate_up = blank(sparse.shared_gate_up, zero_gu);
    sparse_blank.shared_down    = blank(sparse.shared_down, zero_dn);
    auto shared_blank           = shared;
    shared_blank.gate_up.weight = blank(shared.gate_up.weight, zero_gu);
    shared_blank.down.weight    = blank(shared.down.weight, zero_dn);

    struct Case {
        const char* label;
        ops::SparseMoeWeights sparse;
        ops::MoeWeights routed;
    };
    const std::vector<Case> cases{
        {"full", sparse, ops::MoeWeights{control, routed, shared}},
        {"no-shared", sparse_blank, ops::MoeWeights{control, routed, shared_blank}}};

    int failures = 0;
    for (const int tokens : {1, 2, 3, 16, 17, 32, 47, 64, 128, 768, 1024, 1560}) {
        const auto x = Tensor(x_device, execution.device(), DType::BF16,
                              {static_cast<std::int32_t>(hidden), tokens});
        auto sparse_out = Tensor(out_sparse, execution.device(), DType::BF16,
                                 {static_cast<std::int32_t>(hidden), tokens});
        auto routed_out = Tensor(out_routed, execution.device(), DType::BF16,
                                 {static_cast<std::int32_t>(hidden), tokens});
        const auto elements = static_cast<std::size_t>(hidden) * tokens;
        check(cudaMemsetAsync(out_sparse, 0, elements * sizeof(__nv_bfloat16), stream),
              "clear sparse");
        check(cudaMemsetAsync(out_routed, 0, elements * sizeof(__nv_bfloat16), stream),
              "clear routed");

        const auto routed_bytes = ops::topk_softmax_swiglu_moe_workspace_capacity_bytes(
            profile, ops::MoeEpilogue::AddResidual, tokens, tokens);
        for (const auto& item : cases) {
            check(cudaMemsetAsync(out_sparse, 0, elements * sizeof(__nv_bfloat16), stream),
                  "clear sparse");
            check(cudaMemsetAsync(out_routed, 0, elements * sizeof(__nv_bfloat16), stream),
                  "clear routed");
            const auto sparse_bytes = ops::sparse_moe_workspace_capacity_bytes(
                item.sparse.routed_gate_up.qtype, item.sparse.routed_down.qtype, tokens, tokens);
            {
                WorkspaceArena arena(sparse_bytes);
                ops::sparse_moe(x, item.sparse, ops::SparseMoeEpilogue::AddResidual, sparse_out,
                                ops::SparseMoeHints{}, arena, execution);
            }
            {
                void* buffer = nullptr;
                check(cudaMalloc(&buffer, routed_bytes), "alloc routed workspace");
                MemoryArena arena(MemorySpan{buffer, routed_bytes, execution.device()});
                ops::topk_softmax_swiglu_moe(x, profile, item.routed,
                                             ops::MoeEpilogue::AddResidual, routed_out, arena,
                                             execution);
                check(cudaStreamSynchronize(stream), "routed sync");
                check(cudaFree(buffer), "free routed workspace");
            }
            check(cudaStreamSynchronize(stream), "sparse sync");

            std::vector<__nv_bfloat16> host_sparse(elements), host_routed(elements);
            check(cudaMemcpy(host_sparse.data(), out_sparse, elements * sizeof(__nv_bfloat16),
                             cudaMemcpyDeviceToHost),
                  "read sparse");
            check(cudaMemcpy(host_routed.data(), out_routed, elements * sizeof(__nv_bfloat16),
                             cudaMemcpyDeviceToHost),
                  "read routed");

            float max_difference  = 0.0F;
            double sum_difference = 0.0;
            std::size_t worst     = 0;
            for (std::size_t i = 0; i < elements; ++i) {
                const auto a = __bfloat162float(host_sparse[i]);
                const auto b = __bfloat162float(host_routed[i]);
                const auto d = std::fabs(a - b);
                if (d > max_difference) {
                    max_difference = d;
                    worst          = i;
                }
                sum_difference += d;
            }
            std::printf("T=%-4d [%-9s] max|d|=%.6f mean|d|=%.8f (%.6f vs %.6f)\n", tokens,
                        item.label, max_difference, sum_difference / elements,
                        __bfloat162float(host_sparse[worst]),
                        __bfloat162float(host_routed[worst]));
            if (max_difference > 1e-2F) { ++failures; }
        }
    }

    check(cudaFree(out_sparse), "free sparse output");
    check(cudaFree(out_routed), "free routed output");
    check(cudaFree(x_device), "free input");
    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "diag-moe failed: %s\n", error.what());
        return 1;
    }
}
