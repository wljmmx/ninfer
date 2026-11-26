// The GGML expert banks on the CPU half of a split layer.
//
// The split path gives one share of a layer's routes to the Device and the rest to the Host. The
// Device half now reads the stored blocks directly (ops/moe/ggml/ggml_experts.cu); this is the same
// arithmetic on the Host, written against core/ggml_dequant.h so both halves decode one specification.
//
// It is deliberately plain C++: the share is a fraction of one layer's routes, and the point here is
// that the layer runs at all and agrees with the Device half. A vectorised version belongs beside the
// existing AVX kernels, not in place of correctness.

#include "ops/moe/kernels/cpu/invocation.h"
#include "ops/moe/routed_swiglu/cpu/dispatch.h"

#include "core/ggml_dequant.h"
#include "core/ggml_types.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ninfer::ops::detail::moe_routed_cpu {
namespace {

using core::decode_ggml;

// The profile is fixed by the policy, so its numbers are template parameters and the ShapePolicy
// stays the two free functions the dispatch table expects.
template <int TopK, int Intermediate, int Hidden>
struct GgmlLayer {
    static std::size_t capacity(int /*tokens*/) {
        // Two decoded rows of the reduction width plus one row of products. Allocated once per call
        // and reused across tokens and routes.
        return (std::size_t(2) * std::size_t(Hidden) + std::size_t(Intermediate)) * sizeof(float) +
               256;
    }

    static float dot(const float* a, const std::uint16_t* b, int n) {
        float acc = 0.0f;
        for (int i = 0; i < n; ++i) {
            const auto bits = b[i];
            const float value =
                __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&bits));
            acc += a[i] * value;
        }
        return acc;
    }

    // Decode `count` values of one encoded row into `out`.
    static void decode_row(const WeightBankView& bank, std::uint32_t expert, std::uint64_t row_bytes,
                           std::size_t row, int width, float* out) {
        const auto plane = bank.plane(ExpertPlaneKind::Codes, expert);
        const auto* base = static_cast<const std::uint8_t*>(plane.data) + row * row_bytes;
        const auto geometry = core::ggml_block_geometry(bank.layout.format);
        using Decode = void (*)(const std::uint8_t*, std::span<float>);
        Decode decode = nullptr;
        switch (bank.layout.format) {
        case QType::GGML_Q4_K: decode = core::decode_ggml_q4_k; break;
        case QType::GGML_Q5_K: decode = core::decode_ggml_q5_k; break;
        case QType::GGML_Q6_K: decode = core::decode_ggml_q6_k; break;
        case QType::GGML_Q2_0: decode = core::decode_ggml_q2_0; break;
        case QType::GGML_IQ4_NL: decode = core::decode_ggml_iq4_nl; break;
        case QType::GGML_IQ4_XS: decode = core::decode_ggml_iq4_xs; break;
        case QType::GGML_IQ2_XXS: decode = core::decode_ggml_iq2_xxs; break;
        case QType::GGML_IQ2_XS: decode = core::decode_ggml_iq2_xs; break;
        case QType::GGML_IQ2_S: decode = core::decode_ggml_iq2_s; break;
        case QType::GGML_IQ3_XXS: decode = core::decode_ggml_iq3_xxs; break;
        case QType::GGML_IQ3_S: decode = core::decode_ggml_iq3_s; break;
        default:
            throw std::invalid_argument("routed_swiglu CPU: not a GGML block bank");
        }
        decode_ggml(base, std::uint64_t(width), geometry.block, geometry.type_bytes, decode,
                    std::span<float>(out, std::size_t(width)));
    }

    static void launch(const moe_cpu::Invocation& invocation, MemoryArena& workspace,
                       CpuExecutionView /*execution*/) {
        const auto& gate = invocation.experts.gate;
        const auto& up   = invocation.experts.up;
        const auto& down = invocation.experts.down;
        if (gate.parent == nullptr || up.parent == nullptr || down.parent == nullptr) {
            throw std::invalid_argument(
                "routed_swiglu CPU: the GGML path needs the split gate/up/down banks");
        }
        const std::int32_t experts =
            static_cast<std::int32_t>(gate.layout.region.count);
        const auto geometry_gate = core::ggml_block_geometry(gate.layout.format);
        const auto geometry_down = core::ggml_block_geometry(down.layout.format);
        if (std::int64_t(Hidden) % std::int64_t(geometry_gate.block) != 0 ||
            std::int64_t(Intermediate) % std::int64_t(geometry_down.block) != 0) {
            throw std::invalid_argument("routed_swiglu CPU: a GGML bank width is not whole blocks");
        }
        const std::uint64_t gate_row_bytes =
            std::uint64_t(Hidden) / geometry_gate.block * geometry_gate.type_bytes;
        const std::uint64_t down_row_bytes =
            std::uint64_t(Intermediate) / geometry_down.block * geometry_down.type_bytes;

        const auto storage = workspace.alloc_bytes(capacity(invocation.tokens), 4);
        auto* row_gate     = static_cast<float*>(storage.data);
        auto* row_up       = row_gate + Hidden;
        auto* product      = row_up + Hidden;

        const int tokens = invocation.tokens;
        for (int t = 0; t < tokens; ++t) {
            const std::uint16_t* x = invocation.input + std::size_t(t) * std::size_t(Hidden);
            float* out_row         = invocation.output + std::size_t(t) * std::size_t(Hidden);
            for (int route = 0; route < TopK; ++route) {
                const std::int32_t expert = invocation.ids[std::size_t(t) * TopK + route];
                const float coefficient   = invocation.coefficients[std::size_t(t) * TopK + route];
                if (expert < 0 || expert >= experts) { continue; }
                if (invocation.subset != ExpertSubsetKind::All &&
                    !moe_cpu::participates(invocation.subset, invocation.mask, expert)) {
                    continue;
                }
                for (int i = 0; i < Intermediate; ++i) {
                    decode_row(gate, std::uint32_t(expert), gate_row_bytes, std::size_t(i), Hidden,
                               row_gate);
                    decode_row(up, std::uint32_t(expert), gate_row_bytes, std::size_t(i), Hidden,
                               row_up);
                    const float g = dot(row_gate, x, Hidden);
                    const float u = dot(row_up, x, Hidden);
                    // The same BF16 rounding the Device half applies to the down matrix's operand.
                    const float value = (g / (1.0f + std::exp(-g))) * u;
                    const auto bits   = core::ggml_float_to_bf16(value);
                    product[i] = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&bits));
                }
                for (int d = 0; d < Hidden; ++d) {
                    decode_row(down, std::uint32_t(expert), down_row_bytes, std::size_t(d),
                               Intermediate, row_gate);
                    float acc = 0.0f;
                    for (int i = 0; i < Intermediate; ++i) acc += row_gate[i] * product[i];
                    out_row[d] += coefficient * acc;
                }
            }
        }
    }
};

} // namespace

const ShapePolicy& ggml_h2560_i640_e512_k10() {
    static const ShapePolicy policy{
        .capacity = &GgmlLayer<10, 640, 2560>::capacity,
        .launch   = &GgmlLayer<10, 640, 2560>::launch};
    return policy;
}

} // namespace ninfer::ops::detail::moe_routed_cpu
