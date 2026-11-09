// port (sm_89): one Qwen4Exp MoE layer, end to end through ffn().
//
// A1 taught ExpertStream the gate/up/down form; A2 wires the layer. This test grades the wiring,
// not the expert kernels (tests/ops/test_moe_qwen4exp_split.cpp owns those):
//
//   * the routed term is one routed_swiglu() over THREE banks (gate and up as separate I-row
//     matrices, down as [H,I]) with the additive epilogue;
//   * the shared expert runs as its own Q5 projection pair - gate, up, SiLU*up, down - and enters
//     the residual through scaled_add() with the routing op's sigmoid coefficient;
//   * the residual is therefore old + routed + coefficient * shared, which is a different seam
//     from the fused form's single FP32 fold: this form rounds the routed term and the shared term
//     separately, and the reference below rounds the same two seams.
//
// It also grades the capacity recipe: the arena is sized by ffn_workspace_bytes() and nothing else,
// so an under-sized recipe fails here instead of in a real run.
//
// The artifact's geometry is used verbatim (H=2560, I=640, E=512, K=10, q4/q4/q6 experts, q5
// shared - the shipped artifact stores the shared expert as q5_g64_fp16/row_split_k128_v1, three
// 1,075,200-byte planes, so Q5 is what the layer has to run), and every weight is packed from a
// logical float matrix, so the reference reads the same stored codes the kernels read.
//
// The source is a double-precision evaluation of
//
//   routed[d] = sum_j c_j * down_ej(silu(gate_ej x) * up_ej x)
//   step1[d]  = bf16(residual[d] + routed[d])                  // the AddResidual epilogue
//   shared[d] = down_s(silu(gate_s x) * up_s x)                 // its own Q5 pair, BF16 stages
//   out[d]    = bf16(step1[d] + sigmoid(shared_logit[t]) * shared[d])
//
// over the same stored codes and the same per-expert scale factors, so a wrong bank, a wrong
// expert or a missing shared term shows up as a numeric difference rather than as a silent zero.
#include "models/qwen3_5/execution/ffn.h"
#include "models/qwen3_5/execution/parameters.h"

#include "core/arena.h"
#include "core/weight.h"
#include "core/weight_bank.h"
#include "ninfer/ops/moe.h"

#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
namespace qwen = ninfer::models::qwen3_5::execution;
namespace packed_weight = ninfer::test::quantized_weight;

namespace {

constexpr std::int32_t kHidden       = 2560;
constexpr std::int32_t kIntermediate = 640;
constexpr std::int32_t kExperts      = 512;
constexpr std::int32_t kTopK         = 10;

// BF16 destination plus two rounding seams (routed, then the shared term). The bound is the
// existing A16 sparse-MoE criterion; it is reported, not assumed, so a regression shows its ratio.
constexpr ReductionCriterion kLayerTolerance{
    /*relative_l2*/ 1.5e-2,
    /*gross_absolute*/ 6.0e-3,
    /*gross_relative_to_max_reference*/ 1.0e-2,
};

// One expert's three planes in device memory, addressed expert by expert through a stride - the
// layout a resident bank has when every expert image is contiguous.
struct MatrixBank {
    MatrixBank(std::size_t code, std::size_t high, std::size_t scale)
        : codes(code * kExperts), scales(scale * kExperts),
          high(high == 0 ? 0 : high * kExperts), plane_code(code), plane_high(high),
          plane_scale(scale) {}

    void upload(const packed_weight::PackedWeight& packed, std::uint32_t expert) {
        const auto* base = packed.payload.data();
        codes.copy_from_host(base, plane_code, static_cast<std::size_t>(expert) * plane_code);
        if (plane_high != 0) {
            high.copy_from_host(base + plane_code, plane_high,
                                static_cast<std::size_t>(expert) * plane_high);
        }
        scales.copy_from_host(base + plane_code + plane_high, plane_scale,
                              static_cast<std::size_t>(expert) * plane_scale);
    }

    ExpertMatrixDeviceView view() const {
        ExpertMatrixDeviceView out;
        out.codes        = static_cast<const std::byte*>(codes.p);
        out.high         = plane_high == 0 ? nullptr : static_cast<const std::byte*>(high.p);
        out.scales       = static_cast<const std::byte*>(scales.p);
        out.code_stride  = plane_code;
        out.high_stride  = plane_high;
        out.scale_stride = plane_scale;
        return out;
    }

    DeviceBuffer codes, scales, high;
    std::size_t plane_code = 0, plane_high = 0, plane_scale = 0;
};

struct PlaneBytes {
    std::size_t code = 0, high = 0, scale = 0;
};

PlaneBytes plane_bytes(const packed_weight::PackedWeight& packed) {
    const auto* base            = packed.payload.data();
    const auto* scales =
        static_cast<const std::uint8_t*>(static_cast<const void*>(packed.weight.scales));
    const std::size_t scale_off = static_cast<std::size_t>(scales - base);
    const auto* high =
        packed.weight.qhigh ? static_cast<const std::uint8_t*>(packed.weight.qhigh) : nullptr;
    const std::size_t high_off = high ? static_cast<std::size_t>(high - base) : scale_off;
    return {high_off, scale_off - high_off, packed.payload.size() - scale_off};
}

// One expert image: the base payload with its scale plane multiplied by `factor`. Scaling the scale
// plane (rather than the codes) keeps every expert a valid decode of its own bytes, which is what
// lets the reference read the same payload the kernel reads.
packed_weight::PackedWeight scaled_image(const packed_weight::PackedWeight& base,
                                         const PlaneBytes& planes, float factor) {
    packed_weight::PackedWeight out = base;
    if (factor != 1.0F) {
        auto* bytes = out.payload.data() + planes.code + planes.high;
        for (std::size_t index = 0; index + 1 < planes.scale; index += 2) {
            std::uint16_t bits = 0;
            std::memcpy(&bits, bytes + index, sizeof(bits));
            __half value;
            std::memcpy(&value, &bits, sizeof(value));
            const __half rounded = __float2half_rn(__half2float(value) * factor);
            std::uint16_t out_bits = 0;
            std::memcpy(&out_bits, &rounded, sizeof(out_bits));
            std::memcpy(bytes + index, &out_bits, sizeof(out_bits));
        }
    }
    out.weight.qdata  = out.payload.data();
    out.weight.qhigh  = planes.high == 0 ? nullptr : out.payload.data() + planes.code;
    out.weight.scales = out.payload.data() + planes.code + planes.high;
    out.weight.payload      = out.payload.data();
    out.weight.payload_bytes = out.payload.size();
    return out;
}

// A whole matrix (not a bank) resident on the device, as a value the ops read.
struct ResidentMatrix {
    explicit ResidentMatrix(const packed_weight::PackedWeight& packed, const Device& device)
        : storage(packed.payload.size()) {
        storage.copy_from_host(packed.payload.data(), packed.payload.size());
        weight        = packed.device_weight(storage.p);
        weight.device = device;
    }

    DeviceBuffer storage;
    Weight weight;
};

std::vector<float> patterned(std::int32_t rows, std::int32_t columns, std::uint32_t seed,
                             float amplitude) {
    std::vector<float> values(static_cast<std::size_t>(rows) * columns);
    fill_uniform(values, seed, -amplitude, amplitude);
    round_to_bf16(values);
    return values;
}

double silu(double x) { return x / (1.0 + std::exp(-x)); }

std::uint16_t to_bf16(double value) {
    std::uint32_t bits = 0;
    const float narrowed = static_cast<float>(value);
    std::memcpy(&bits, &narrowed, sizeof(bits));
    const std::uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return static_cast<std::uint16_t>(bits >> 16);
}

double from_bf16(std::uint16_t word) {
    const std::uint32_t bits = std::uint32_t{word} << 16;
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return static_cast<double>(value);
}

double bf16_round(double value) { return from_bf16(to_bf16(value)); }

// What expert e's matrix holds, as the double oracle sees it: the base payload's codes with its
// per-group fp16 scale multiplied by the same factor the upload applied. Nothing is copied - the
// oracle walks the base payload and applies the factor where the stored value lives.
struct ScaledMatrix {
    ScaledMatrix(const packed_weight::PackedWeight& packed, float factor)
        : packed_(&packed) {
        spec_           = packed_weight::detail::quant_spec(packed.weight.qtype);
        nibble_bytes_   = packed_weight::detail::nibble_bytes_per_group(spec_);
        high_bytes_     = packed_weight::detail::high_bytes_per_group(spec_);
        groups_per_row_ = static_cast<std::int32_t>(packed.weight.padded_shape[1]) / spec_.group_size;
        const std::size_t groups =
            static_cast<std::size_t>(packed.weight.shape[0]) * static_cast<std::size_t>(groups_per_row_);
        scale_.resize(groups);
        for (std::size_t group = 0; group < groups; ++group) {
            const auto stored = packed_weight::detail::load_u16_le(
                packed.payload, packed.scale_plane_offset + group * 2);
            const float value = packed_weight::detail::f16_to_f32(stored) * factor;
            scale_[group] =
                packed_weight::detail::f16_to_f32(packed_weight::detail::f32_to_f16(value));
        }
    }

    [[nodiscard]] double at(std::int32_t row, std::int32_t column) const {
        const std::int32_t group = column / spec_.group_size;
        const std::int32_t lane  = column % spec_.group_size;
        const std::size_t index  = static_cast<std::size_t>(row) *
                                       static_cast<std::size_t>(groups_per_row_) +
                                   static_cast<std::size_t>(group);
        const auto* nibble = packed_->payload.data() + index * static_cast<std::size_t>(nibble_bytes_);
        const auto* high   = high_bytes_ == 0
                                 ? nullptr
                                 : packed_->payload.data() + packed_->high_plane_offset +
                                       index * static_cast<std::size_t>(high_bytes_);
        const int code = packed_weight::detail::unpack_lowbit_code(nibble, high, spec_, lane);
        return static_cast<double>(code) * scale_[index];
    }

    [[nodiscard]] double dot_row(std::int32_t row, const float* x, std::int32_t k) const {
        double acc = 0.0;
        for (std::int32_t column = 0; column < k; ++column) {
            acc += at(row, column) * static_cast<double>(x[column]);
        }
        return acc;
    }

    const packed_weight::PackedWeight* packed_ = nullptr;
    packed_weight::detail::QuantSpec spec_{};
    std::vector<double> scale_;
    std::int32_t groups_per_row_ = 0;
    int nibble_bytes_            = 0;
    int high_bytes_              = 0;
};

float expert_factor(std::int32_t expert) {
    return 1.0F + 0.01F * static_cast<float>((expert * 7) % 13);
}

// A per-expert image is built on demand and kept, because a token's routes repeat across tokens.
struct ScaledBank {
    ScaledBank(const packed_weight::PackedWeight& base, float (*factor_of)(std::int32_t))
        : base_(&base), factor_of_(factor_of) {}

    const ScaledMatrix& of(std::int32_t expert) {
        auto& slot = images_[static_cast<std::size_t>(expert)];
        if (!slot) { slot.emplace(*base_, factor_of_(expert)); }
        return *slot;
    }

    const packed_weight::PackedWeight* base_ = nullptr;
    float (*factor_of_)(std::int32_t)         = nullptr;
    std::vector<std::optional<ScaledMatrix>> images_ =
        std::vector<std::optional<ScaledMatrix>>(kExperts);
};

int run_case(DeviceExecutionView execution, const Device& device, std::int32_t tokens) {
    // --- routed banks: q4 gate, q4 up, q6 down, one image per expert -----------------------------
    const auto gate_base = packed_weight::pack_q4_row_split(
        patterned(kIntermediate, kHidden, 0x51a7U, 0.09F), kIntermediate, kHidden);
    const auto up_base = packed_weight::pack_q4_row_split(
        patterned(kIntermediate, kHidden, 0x77b1U, 0.07F), kIntermediate, kHidden);
    const auto down_base = packed_weight::pack_q6_row_split(
        patterned(kHidden, kIntermediate, 0x2c93U, 0.11F), kHidden, kIntermediate);
    const auto gate_planes = plane_bytes(gate_base);
    const auto up_planes   = plane_bytes(up_base);
    const auto down_planes = plane_bytes(down_base);

    MatrixBank gate(gate_planes.code, gate_planes.high, gate_planes.scale);
    MatrixBank up(up_planes.code, up_planes.high, up_planes.scale);
    MatrixBank down(down_planes.code, down_planes.high, down_planes.scale);

    // Every expert scales its own weight by a distinct factor (through the scale plane), so a
    // wrong expert address changes the result instead of hiding behind identical images.
    for (std::int32_t expert = 0; expert < kExperts; ++expert) {
        const float factor = expert_factor(expert);
        const auto gate_image = scaled_image(gate_base, gate_planes, factor);
        const auto up_image   = scaled_image(up_base, up_planes, factor);
        const auto down_image = scaled_image(down_base, down_planes, factor);
        gate.upload(gate_image, static_cast<std::uint32_t>(expert));
        up.upload(up_image, static_cast<std::uint32_t>(expert));
        down.upload(down_image, static_cast<std::uint32_t>(expert));
    }

    // --- shared expert: q5 gate/up/down, resident whole (the artifact's own format) ----------------
    const auto shared_gate_packed = packed_weight::pack_q5_row_split(
        patterned(kIntermediate, kHidden, 0x0b3dU, 0.05F), kIntermediate, kHidden);
    const auto shared_up_packed = packed_weight::pack_q5_row_split(
        patterned(kIntermediate, kHidden, 0x1f27U, 0.05F), kIntermediate, kHidden);
    const auto shared_down_packed = packed_weight::pack_q5_row_split(
        patterned(kHidden, kIntermediate, 0x2a55U, 0.05F), kHidden, kIntermediate);
    ResidentMatrix shared_gate(shared_gate_packed, device);
    ResidentMatrix shared_up(shared_up_packed, device);
    ResidentMatrix shared_down(shared_down_packed, device);

    // --- router rows + shared logit, the [E+1,H] control matrix the routing op reads --------------
    WeightParent control_parent;
    control_parent.geometry = weight_geometry(QType::BF16, QuantLayout::Contiguous,
                                              std::vector<std::uint64_t>{kExperts + 1, kHidden});
    auto control_values = patterned(kExperts + 1, kHidden, 0x4411U, 0.05F);
    DeviceBuffer control_buffer(control_values.size() * sizeof(std::uint16_t));
    {
        std::vector<std::uint16_t> words(control_values.size());
        for (std::size_t i = 0; i < control_values.size(); ++i) {
            words[i] = to_bf16(control_values[i]);
        }
        control_buffer.copy_from_host(words.data(), words.size() * sizeof(std::uint16_t));
    }
    control_parent.data   = static_cast<const std::byte*>(control_buffer.p);
    control_parent.device = device;
    const auto control =
        native_weight(WeightView{{kExperts + 1, kHidden},
                                 {{&control_parent, 0, std::uint64_t(kExperts + 1) * kHidden}}});

    // --- the bank layouts and the routed profile ------------------------------------------------
    const std::uint64_t gate_stack_shape[2] = {std::uint64_t(kExperts) * kIntermediate, kHidden};
    const std::uint64_t down_stack_shape[2] = {std::uint64_t(kExperts) * kHidden, kIntermediate};
    const auto gate_layout = describe_weight_bank(
        weight_geometry(QType::Q4_G64_FP16, QuantLayout::RowSplit, gate_stack_shape),
        {0, kExperts, kIntermediate, kIntermediate});
    const auto down_layout = describe_weight_bank(
        weight_geometry(QType::Q6_G64_FP16, QuantLayout::RowSplit, down_stack_shape),
        {0, kExperts, kHidden, kHidden});

    qwen::RoutedMoeParameters moe;
    moe.control      = control;
    moe.profile.routed =
        ops::routed_swiglu_profile(gate_layout, gate_layout, down_layout, kTopK,
                                   ops::LinearPolicy::A16Only, ops::LinearPolicy::A16Only);
    moe.profile.shared = ops::SwiGLUProfile{kHidden, kIntermediate, QType::Q5_G64_FP16,
                                            QType::Q5_G64_FP16, QuantLayout::RowSplit,
                                            QuantLayout::RowSplit, ops::LinearPolicy::A16Only,
                                            ops::LinearPolicy::A16Only};
    moe.routed.bank.count  = kExperts;
    moe.routed.bank.device = device;
    moe.routed.gate        = gate.view();
    moe.routed.up          = up.view();
    moe.routed.bank.down   = down.view();

    qwen::MoeSplitParameters split;
    split.parameter   = "text/layers/0/moe/experts/gate";
    split.shared_gate = ops::SingleProjectionWeight{shared_gate.weight, ops::LinearPolicy::A16Only};
    split.shared_up   = ops::SingleProjectionWeight{shared_up.weight, ops::LinearPolicy::A16Only};
    split.shared_down =
        ops::SingleProjectionWeight{shared_down.weight, ops::LinearPolicy::A16Only};
    moe.split = std::move(split);

    // --- inputs ----------------------------------------------------------------------------------
    std::vector<float> hidden_values(static_cast<std::size_t>(kHidden) * tokens);
    std::vector<float> residual_values(static_cast<std::size_t>(kHidden) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        const auto x = [&] {
            std::vector<float> values(kHidden);
            fill_uniform(values, 0x1000U + static_cast<std::uint32_t>(token), -1.0F, 1.0F);
            round_to_bf16(values);
            return values;
        }();
        std::copy(x.begin(), x.end(),
                  hidden_values.begin() + static_cast<std::size_t>(token) * kHidden);
        std::vector<float> residual(kHidden);
        fill_uniform(residual, 0x9000U + static_cast<std::uint32_t>(token), -0.5F, 0.5F);
        round_to_bf16(residual);
        std::copy(residual.begin(), residual.end(),
                  residual_values.begin() + static_cast<std::size_t>(token) * kHidden);
    }
    auto hidden_device   = to_device_bf16(hidden_values);
    auto residual_device = to_device_bf16(residual_values);
    Tensor hidden(MemorySpan{hidden_device.p, hidden_device.bytes, device}, DType::BF16,
                  {kHidden, tokens});
    Tensor residual(MemorySpan{residual_device.p, residual_device.bytes, device}, DType::BF16,
                    {kHidden, tokens});

    // --- the layer's own routes, computed once through the routing op -----------------------------
    // The layer computes them again from the same input and the same control matrix; reading them
    // back here is how the reference knows which experts and coefficients the layer actually used,
    // without depending on the layer's internal arena layout. Routing itself is graded by its own
    // op tests, the expert and shared arithmetic behind it is what this one grades.
    GuardedDeviceBuffer ids_guard(std::size_t(kTopK) * tokens * sizeof(std::int32_t));
    GuardedDeviceBuffer coefficient_guard(std::size_t(kTopK) * tokens * sizeof(float));
    GuardedDeviceBuffer shared_guard(std::size_t(tokens) * sizeof(float));
    ids_guard.fill(0);
    coefficient_guard.fill(0);
    shared_guard.fill(0);
    Tensor route_ids(MemorySpan{ids_guard.data(), ids_guard.bytes(), device}, DType::I32,
                     {kTopK, tokens});
    Tensor route_coefficients(MemorySpan{coefficient_guard.data(), coefficient_guard.bytes(), device},
                              DType::FP32, {kTopK, tokens});
    Tensor shared_coefficients(MemorySpan{shared_guard.data(), shared_guard.bytes(), device},
                               DType::FP32, {tokens});
    const ops::RouteProfile route_profile{kHidden, kExperts, kTopK};
    {
        const auto bytes = ops::topk_softmax_route_workspace_capacity_bytes(route_profile, tokens,
                                                                            tokens);
        GuardedDeviceBuffer route_scratch(bytes);
        route_scratch.fill(0);
        MemoryArena arena(MemorySpan{route_scratch.data(), bytes, device});
        ops::ExpertRoutes routes{route_ids, route_coefficients};
        ops::topk_softmax_route(hidden, route_profile, moe.control, routes, shared_coefficients,
                                arena, execution);
        cuda_synchronize();
        if (route_scratch.verify_guards("qwen4exp split layer routing scratch") != 0) { return 1; }
    }

    // --- the layer, with the arena ffn_workspace_bytes() sized --------------------------------
    qwen::FfnParameters parameters{std::move(moe)};
    const auto workspace_bytes = qwen::ffn_workspace_bytes(parameters, tokens, tokens);
    GuardedDeviceBuffer scratch(workspace_bytes);
    scratch.fill(0);
    WorkspaceArena workspace(DeviceSpan{scratch.data(), workspace_bytes});
    qwen::ffn(hidden, parameters, residual, {}, workspace, execution);
    cuda_synchronize();

    const auto ids          = from_device<int>(ids_guard.data(), std::size_t(kTopK) * tokens);
    const auto coeffs_float = from_device<float>(coefficient_guard.data(),
                                                std::size_t(kTopK) * tokens);
    const auto shared_float = from_device<float>(shared_guard.data(), std::size_t(tokens));
    const std::vector<double> coeffs(coeffs_float.begin(), coeffs_float.end());
    const auto actual   = from_device_bf16(residual_device.p,
                                           static_cast<std::size_t>(kHidden) * tokens);

    // --- the reference, in double from the same stored codes ------------------------------------
    ScaledBank gate_bank(gate_base, expert_factor);
    ScaledBank up_bank(up_base, expert_factor);
    ScaledBank down_bank(down_base, expert_factor);
    ScaledMatrix shared_gate_w(shared_gate_packed, 1.0F);
    ScaledMatrix shared_up_w(shared_up_packed, 1.0F);
    ScaledMatrix shared_down_w(shared_down_packed, 1.0F);

    std::vector<double> reference(actual.size());
    std::vector<float> activation(kIntermediate);
    std::vector<float> shared_activation(kIntermediate);
    for (std::int32_t token = 0; token < tokens; ++token) {
        const float* x = hidden_values.data() + static_cast<std::size_t>(token) * kHidden;
        std::vector<double> routed(kHidden, 0.0);
        for (std::int32_t slot = 0; slot < kTopK; ++slot) {
            const auto expert = ids[static_cast<std::size_t>(token) * kTopK + slot];
            const auto weight = coeffs[static_cast<std::size_t>(token) * kTopK + slot];
            const auto& g = gate_bank.of(expert);
            const auto& u = up_bank.of(expert);
            const auto& d = down_bank.of(expert);
            for (std::int32_t row = 0; row < kIntermediate; ++row) {
                activation[static_cast<std::size_t>(row)] = static_cast<float>(
                    silu(g.dot_row(row, x, kHidden)) * u.dot_row(row, x, kHidden));
            }
            for (std::int32_t row = 0; row < kHidden; ++row) {
                routed[static_cast<std::size_t>(row)] +=
                    static_cast<double>(weight) * d.dot_row(row, activation.data(), kIntermediate);
            }
        }
        // The shared expert's own Q5 pair. Its two I-row outputs and the SiLU*up product are BF16
        // tensors in the layer, so the reference rounds them at the same seams before the H-row
        // projection reads them.
        for (std::int32_t row = 0; row < kIntermediate; ++row) {
            shared_activation[static_cast<std::size_t>(row)] = static_cast<float>(bf16_round(
                silu(shared_gate_w.dot_row(row, x, kHidden)) *
                shared_up_w.dot_row(row, x, kHidden)));
        }
        const double coefficient = static_cast<double>(shared_float[static_cast<std::size_t>(token)]);
        for (std::int32_t row = 0; row < kHidden; ++row) {
            // AddResidual: the epilogue adds the BF16 residual to its FP32 accumulation and rounds
            // once, into the BF16 destination.
            const double step1 = bf16_round(
                static_cast<double>(residual_values[static_cast<std::size_t>(token) * kHidden + row]) +
                routed[static_cast<std::size_t>(row)]);
            const double shared_term = bf16_round(
                shared_down_w.dot_row(row, shared_activation.data(), kIntermediate));
            reference[static_cast<std::size_t>(token) * kHidden + row] =
                bf16_round(step1 + coefficient * shared_term);
        }
    }
    return verify_reduction("qwen4exp split moe layer T=" + std::to_string(tokens), actual, reference,
                            kLayerTolerance);
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    DeviceContext context;
    const auto execution = context.execution_view().on_stream(nullptr);
    int failures         = 0;
    for (const std::int32_t tokens : {1, 8}) {
        failures += run_case(execution, execution.device(), tokens);
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " qwen4exp split moe layer\n";
    return failures == 0 ? 0 : 1;
}
