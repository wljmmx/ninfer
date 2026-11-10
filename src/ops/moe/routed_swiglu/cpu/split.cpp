// port (sm_89): the CPU half of the gate/up/down (Qwen4Exp) expert form.
//
// One expert at a time, three matrices addressed per expert:
//
//   either the host tier's packed slot form (planes at 0 / codes_bytes / scales offset inside a
//   slot, one expert offset table), or a bank addressed in place in the artifact mapping (one
//   origin per plane). Both name the same three planes per expert.
//
// The row map is the artifact's, not the fused one: the gate object and the up object hold I rows
// each, so gate row i and up row i are two images of the same output channel - never rows i and I+i
// of one parent (on this artifact row I of gate is the next expert's first row). Arithmetic runs in
// FP32 with the device kernels' group order: a 64-value partial scaled once per group.
#include "ops/moe/routed_swiglu/cpu/split.h"

#include "ops/common/cpu/avx2.h"
#include "ops/common/numeric/math.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ninfer::ops::detail::moe_routed_cpu {
namespace {

namespace ax = common::cpu::avx2;

constexpr std::int32_t kAlignment = 256;

// One row of one matrix, already advanced to its first code / high bits / scale of that row.
struct Row {
    const std::uint8_t* codes   = nullptr;
    const std::uint8_t* high    = nullptr;
    const std::uint16_t* scales = nullptr;
};

// The plane origins of one expert, before a row is selected.
struct ExpertBase {
    const std::uint8_t* codes   = nullptr;
    const std::uint8_t* high    = nullptr;
    const std::uint16_t* scales = nullptr;
};

// Row r of an expert image. The artifact lays a matrix out row-major per plane: codes hold K/2 bytes
// per row, the high bits (Bits-4 per weight) (K/8)*(Bits-4) bytes, and the group scales K/64 FP16
// values. Row K's stride is the row's, not the expert's - the expert stride only separates images.
inline Row row_at(const ExpertBase& base, std::int32_t row, std::int32_t k, int bits) noexcept {
    Row out;
    out.codes  = base.codes + static_cast<std::size_t>(row) * static_cast<std::size_t>(k / 2);
    out.high   = base.high != nullptr
                     ? base.high + static_cast<std::size_t>(row) *
                                       (static_cast<std::size_t>(k / 8) *
                                        static_cast<std::size_t>(bits - 4))
                     : nullptr;
    out.scales = base.scales != nullptr
                     ? base.scales + static_cast<std::size_t>(row) *
                                         static_cast<std::size_t>(k / 64)
                     : nullptr;
    return out;
}

// One encoded matrix of the routed bank, with the per-expert addressing the bank was built with.
struct Matrix {
    const std::uint8_t* codes   = nullptr;
    const std::uint8_t* high    = nullptr;
    const std::uint16_t* scales = nullptr;
    std::size_t code_stride     = 0;
    std::size_t high_stride     = 0;
    std::size_t scale_stride    = 0;
    const std::uint64_t* codes_offsets  = nullptr;
    const std::uint64_t* high_offsets   = nullptr;
    const std::uint64_t* scales_offsets = nullptr;
    const std::uint64_t* expert_offsets = nullptr;

    [[nodiscard]] ExpertBase expert(std::uint32_t index) const noexcept {
        const auto offset = [&](const std::uint64_t* table, std::size_t stride) -> std::size_t {
            if (table != nullptr) { return static_cast<std::size_t>(table[index]); }
            return expert_offsets != nullptr ? static_cast<std::size_t>(expert_offsets[index])
                                             : static_cast<std::size_t>(index) * stride;
        };
        ExpertBase out;
        out.codes  = codes != nullptr ? codes + offset(codes_offsets, code_stride) : nullptr;
        out.high   = high != nullptr ? high + offset(high_offsets, high_stride) : nullptr;
        out.scales = scales != nullptr
                         ? reinterpret_cast<const std::uint16_t*>(
                               reinterpret_cast<const std::byte*>(scales) +
                               offset(scales_offsets, scale_stride))
                         : nullptr;
        return out;
    }
};

int code_bits(QType format) {
    switch (format) {
    case QType::Q4_G64_FP16: return 4;
    case QType::Q6_G64_FP16: return 6;
    default:
        throw std::invalid_argument(
            "routed_swiglu CPU: the split expert form has no kernel for this code width");
    }
}

Matrix matrix_of(const WeightBankView& bank) {
    const auto& layout = bank.layout;
    if (bank.parent == nullptr || bank.parent->data == nullptr) {
        throw std::invalid_argument("routed_swiglu CPU: missing split weight bank");
    }
    const auto* data = bank.parent->data;
    Matrix out;
    out.codes          = layout.codes.expert_bytes
                             ? reinterpret_cast<const std::uint8_t*>(data + layout.codes.offset)
                             : nullptr;
    out.high           = layout.high.expert_bytes
                             ? reinterpret_cast<const std::uint8_t*>(data + layout.high.offset)
                             : nullptr;
    out.scales         = layout.scales.expert_bytes
                             ? reinterpret_cast<const std::uint16_t*>(data + layout.scales.offset)
                             : nullptr;
    out.code_stride    = static_cast<std::size_t>(layout.codes.expert_stride);
    out.high_stride    = static_cast<std::size_t>(layout.high.expert_stride);
    out.scale_stride   = static_cast<std::size_t>(layout.scales.expert_stride);
    out.codes_offsets  = bank.codes_offsets;
    out.high_offsets   = bank.high_offsets;
    out.scales_offsets = bank.scales_offsets;
    out.expert_offsets = bank.expert_offsets;
    return out;
}

// out[c] = row . x[c][0..k). Weights are decoded once per four columns and the group scale is
// applied to the group partial, so the sum has the shape every other routed kernel uses.
template <int Bits, int Columns, class Input>
inline void row_dot(const Row& row, std::int32_t k, const Input* const (&x)[Columns],
                    float (&out)[Columns]) noexcept {
    __m256 total[Columns];
    for (int c = 0; c < Columns; ++c) { total[c] = _mm256_setzero_ps(); }
    const auto groups = k / 64;
    for (std::int32_t group = 0; group < groups; ++group) {
        __m256 partial[Columns];
        for (int c = 0; c < Columns; ++c) { partial[c] = _mm256_setzero_ps(); }
        for (std::int32_t q = 0; q < 64; q += 8) {
            const auto* high = row.high != nullptr
                                   ? row.high + (group * 8 + q / 8) * (Bits - 4)
                                   : nullptr;
            const auto weights = ax::decode_signed_q<Bits>(row.codes + group * 32 + q / 2, high);
            for (int c = 0; c < Columns; ++c) {
                const auto* source = x[c] + group * 64 + q;
                __m256 values;
                if constexpr (sizeof(Input) == 2) {
                    values = ax::load_bf16(reinterpret_cast<const std::uint16_t*>(source));
                } else {
                    values = _mm256_loadu_ps(source);
                }
                partial[c] = _mm256_fmadd_ps(weights, values, partial[c]);
            }
        }
        const auto scale = _mm256_set1_ps(ax::fp16_to_f32(row.scales[group]));
        for (int c = 0; c < Columns; ++c) {
            total[c] = _mm256_fmadd_ps(partial[c], scale, total[c]);
        }
    }
    for (int c = 0; c < Columns; ++c) { out[c] = ax::reduce_sum(total[c]); }
}

struct Member {
    std::int32_t token = 0;
    float coefficient  = 0.0F;
};

struct Work {
    const std::uint16_t* x = nullptr;
    float* output          = nullptr;
    float* activation      = nullptr;
    std::int32_t tokens = 0, hidden = 0, intermediate = 0;
    int gate_bits = 4, down_bits = 6;
    Matrix gate, up, down;
    std::vector<std::int32_t> experts;
    std::vector<std::vector<Member>> members;
};

// Per-worker buffers, allocated once per collective: the rows of one expert are split across the
// team, so two workers must never share a value buffer.
struct Scratch {
    std::vector<const std::uint16_t*> inputs;
    std::vector<const float*> activations;
    std::vector<float> gate, up, down;
};

template <int Bits, class Input>
inline void row_all_tokens(const Row& row, std::int32_t k, const Input* const* inputs,
                           std::size_t count, float* out) noexcept {
    std::size_t at = 0;
    while (at < count) {
        const auto left = count - at;
        if (left >= 4) {
            const Input* xs[4] = {inputs[at], inputs[at + 1], inputs[at + 2], inputs[at + 3]};
            float values[4];
            row_dot<Bits, 4>(row, k, xs, values);
            for (int c = 0; c < 4; ++c) { out[at + c] = values[c]; }
            at += 4;
        } else if (left >= 2) {
            const Input* xs[2] = {inputs[at], inputs[at + 1]};
            float values[2];
            row_dot<Bits, 2>(row, k, xs, values);
            for (int c = 0; c < 2; ++c) { out[at + c] = values[c]; }
            at += 2;
        } else {
            const Input* xs[1] = {inputs[at]};
            float values[1];
            row_dot<Bits, 1>(row, k, xs, values);
            out[at] = values[0];
            at += 1;
        }
    }
}

void run_expert(Work& w, Scratch& s, const CpuWorkerContext& worker, std::size_t index) noexcept {
    const auto expert = static_cast<std::uint32_t>(w.experts[index]);
    // The member table is indexed by expert id, not by the position in `experts`: the two part
    // company whenever the subset skips an expert, and reading the position applies the previous
    // expert's routes (and their coefficients) to this one's weights.
    const auto& members = w.members[expert];
    if (members.empty()) { return; }
    const auto count = members.size();
    for (std::size_t m = 0; m < count; ++m) {
        s.inputs[m] = w.x + static_cast<std::size_t>(members[m].token) * w.hidden;
        s.activations[m] =
            w.activation + static_cast<std::size_t>(members[m].token) * w.intermediate;
    }
    const auto gate_base = w.gate.expert(expert);
    const auto up_base   = w.up.expert(expert);
    const auto gate_rows = worker.range(static_cast<std::size_t>(w.intermediate));
    for (auto i = gate_rows.begin; i < gate_rows.end; ++i) {
        const auto row = static_cast<std::int32_t>(i);
        const auto gate_row = row_at(gate_base, row, w.hidden, w.gate_bits);
        const auto up_row   = row_at(up_base, row, w.hidden, w.gate_bits);
        if (w.gate_bits == 4) {
            row_all_tokens<4>(gate_row, w.hidden, s.inputs.data(), count, s.gate.data());
            row_all_tokens<4>(up_row, w.hidden, s.inputs.data(), count, s.up.data());
        } else {
            row_all_tokens<6>(gate_row, w.hidden, s.inputs.data(), count, s.gate.data());
            row_all_tokens<6>(up_row, w.hidden, s.inputs.data(), count, s.up.data());
        }
        for (std::size_t m = 0; m < count; ++m) {
            w.activation[static_cast<std::size_t>(members[m].token) * w.intermediate + row] =
                common::silu_stable(s.gate[m]) * s.up[m];
        }
    }
    worker.barrier();

    // down_h . a_t folded in with the member's coefficient; the output is [hidden, tokens].
    const auto down_base = w.down.expert(expert);
    const auto out_rows  = worker.range(static_cast<std::size_t>(w.hidden));
    for (auto h = out_rows.begin; h < out_rows.end; ++h) {
        const auto row      = static_cast<std::int32_t>(h);
        const auto down_row = row_at(down_base, row, w.intermediate, w.down_bits);
        if (w.down_bits == 6) {
            row_all_tokens<6>(down_row, w.intermediate, s.activations.data(), count, s.down.data());
        } else {
            row_all_tokens<4>(down_row, w.intermediate, s.activations.data(), count, s.down.data());
        }
        for (std::size_t m = 0; m < count; ++m) {
            w.output[static_cast<std::size_t>(members[m].token) * w.hidden + row] +=
                members[m].coefficient * s.down[m];
        }
    }
    worker.barrier();
}

void run(void* arguments, const CpuWorkerContext& worker) noexcept {
    auto& work = *static_cast<Work*>(arguments);
    Scratch scratch;
    const auto room = static_cast<std::size_t>(work.tokens);
    scratch.inputs.resize(room);
    scratch.activations.resize(room);
    scratch.gate.resize(room);
    scratch.up.resize(room);
    scratch.down.resize(room);
    for (std::size_t index = 0; index < work.experts.size(); ++index) {
        run_expert(work, scratch, worker, index);
    }
}

} // namespace

std::size_t split_capacity(std::int32_t intermediate, std::int32_t tokens) {
    if (intermediate <= 0 || tokens <= 0) {
        throw std::invalid_argument("routed_swiglu CPU: invalid split shape");
    }
    const auto bytes = static_cast<std::size_t>(intermediate) * static_cast<std::size_t>(tokens) *
                       sizeof(float);
    return (bytes + kAlignment - 1) / kAlignment * kAlignment;
}

void split_launch(const Tensor& x, const CpuSwiGLUExperts& experts, const ExpertRoutes& routes,
                  ExpertSubset subset, Tensor& destination, MemoryArena& scratch,
                  CpuExecutionView execution, std::int32_t hidden, std::int32_t intermediate,
                  std::int32_t experts_count, std::int32_t top_k) {
    const auto tokens = x.ne[1];
    if (tokens <= 0 || hidden <= 0 || intermediate <= 0 || experts_count <= 0 || top_k <= 0) {
        throw std::invalid_argument("routed_swiglu CPU: invalid split invocation");
    }
    const auto* ids          = static_cast<const std::int32_t*>(routes.ids.data);
    const auto* coefficients = static_cast<const float*>(routes.coefficients.data);
    const auto* mask         = static_cast<const std::int32_t*>(subset.mask.data);
    if (subset.kind == ExpertSubsetKind::None) {
        std::memset(destination.data, 0, destination.bytes());
        return;
    }

    std::vector<std::uint8_t> selected(static_cast<std::size_t>(experts_count), 0);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t j = 0; j < top_k; ++j) {
            const auto expert = ids[static_cast<std::size_t>(t) * top_k + j];
            if (expert < 0 || expert >= experts_count) {
                throw std::invalid_argument("routed_swiglu CPU: invalid route ID on the split path");
            }
            bool member = false;
            switch (subset.kind) {
            case ExpertSubsetKind::All: member = true; break;
            case ExpertSubsetKind::MaskOnes: member = mask != nullptr && mask[expert] == 1; break;
            case ExpertSubsetKind::MaskZeros: member = mask != nullptr && mask[expert] == 0; break;
            default: break;
            }
            if (member) { selected[static_cast<std::size_t>(expert)] = 1; }
        }
    }

    Work work;
    work.x            = static_cast<const std::uint16_t*>(x.data);
    work.output       = static_cast<float*>(destination.data);
    work.activation   = static_cast<float*>(scratch.storage().data);
    work.tokens       = tokens;
    work.hidden       = hidden;
    work.intermediate = intermediate;
    work.gate         = matrix_of(experts.gate);
    work.up           = matrix_of(experts.up);
    work.down         = matrix_of(experts.down);
    work.gate_bits    = code_bits(experts.gate.layout.format);
    work.down_bits    = code_bits(experts.down.layout.format);
    if (code_bits(experts.up.layout.format) != work.gate_bits) {
        throw std::invalid_argument("routed_swiglu CPU: split gate and up differ in code width");
    }
    work.members.resize(static_cast<std::size_t>(experts_count));
    for (std::int32_t expert = 0; expert < experts_count; ++expert) {
        if (selected[static_cast<std::size_t>(expert)] != 0) { work.experts.push_back(expert); }
    }
    if (work.experts.empty()) {
        std::memset(destination.data, 0, destination.bytes());
        return;
    }
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t j = 0; j < top_k; ++j) {
            const auto expert = ids[static_cast<std::size_t>(t) * top_k + j];
            if (selected[static_cast<std::size_t>(expert)] == 0) { continue; }
            work.members[static_cast<std::size_t>(expert)].push_back(
                Member{t, coefficients[static_cast<std::size_t>(t) * top_k + j]});
        }
    }
    execution.run(run, &work);
}

} // namespace ninfer::ops::detail::moe_routed_cpu
