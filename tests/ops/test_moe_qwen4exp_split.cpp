// port (sm_89): the Qwen4Exp expert shape (H=2560, I=640, E=512, K=10) on the groupwise-int path,
// with the gate and the up matrix in two separate objects instead of one fused [2I,H] parent.
//
// The artifact keeps q4 gate/up and a q6 down, one object per expert, so this grades three things
// the 2048 row never exercised:
//
//   1. the kernels take their shape from the profile (SwiGLUH2560I640E512K10) instead of the
//      compile-time SwiGLUH2048I512E256K8 that used to be baked into the workspace header, the
//      grouping launches and the swiglu kernels;
//   2. a split bank resolves gate row r to the gate object and up row r to the up object, never to
//      the next expert's first gate row (which is what a fused row map would read);
//   3. the down projection still addresses [H,I] per expert with the same stride arithmetic.
//
// The reference is a double-precision evaluation of sum_j c_j * down_e(silu(gate_e x) * up_e x)
// over the *same stored codes*, read through the test packer's logical-weight decoder, so a wrong
// row, expert or scale shows up as a numeric difference rather than as a silent zero.
#include "core/weight.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/moe.h"

#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHidden       = 2560;
constexpr std::int32_t kIntermediate = 640;
constexpr std::int32_t kExperts      = 512;
constexpr std::int32_t kTopK         = 10;

// BF16 destination: the rounding alone is ~4e-3 relative. The bound matches the existing
// sparse-MoE A16 criterion, whose measured maximum on the same codecs was 1.117e-2.
constexpr ReductionCriterion kA16Tolerance{
    /*relative_l2*/ 1.2e-2,
    /*gross_absolute*/ 4.0e-3,
    /*gross_relative_to_max_reference*/ 8.0e-3,
};

struct PlaneBytes {
    std::size_t code, high, scale;
};

PlaneBytes plane_bytes(const quantized_weight::PackedWeight& packed) {
    const auto* base            = packed.payload.data();
    const auto* scales =
        static_cast<const std::uint8_t*>(static_cast<const void*>(packed.weight.scales));
    const std::size_t scale_off = static_cast<std::size_t>(scales - base);
    const auto* high =
        packed.weight.qhigh ? static_cast<const std::uint8_t*>(packed.weight.qhigh) : nullptr;
    const std::size_t high_off = high ? static_cast<std::size_t>(high - base) : scale_off;
    if (high_off > scale_off || scale_off > packed.payload.size()) {
        throw std::logic_error("qwen4exp moe test: packed plane offsets are not ascending");
    }
    return {high_off, scale_off - high_off, packed.payload.size() - scale_off};
}

// One expert image: a payload whose scale plane has been multiplied by `factor`. Scaling the scale
// plane (rather than the codes) keeps every expert's weights a valid decode of its own bytes, which
// is what lets the reference read the same payload the kernel reads.
quantized_weight::PackedWeight scaled_image(const quantized_weight::PackedWeight& base,
                                            const PlaneBytes& planes, float factor) {
    quantized_weight::PackedWeight out = base;
    if (factor != 1.0F) {
        auto* bytes = out.payload.data() + planes.code + planes.high;
        for (std::size_t index = 0; index + 1 < planes.scale; index += 2) {
            std::uint16_t bits = 0;
            std::memcpy(&bits, bytes + index, sizeof(bits));
            __half value;
            std::memcpy(&value, &bits, sizeof(value));
            const float scaled = __half2float(value) * factor;
            const __half rounded = __float2half_rn(scaled);
            std::memcpy(&bits, &rounded, sizeof(bits));
            std::memcpy(bytes + index, &bits, sizeof(bits));
        }
    }
    out.weight.qdata = out.payload.data();
    out.weight.qhigh = planes.high == 0 ? nullptr : out.payload.data() + planes.code;
    out.weight.scales = out.payload.data() + planes.code + planes.high;
    out.weight.payload = out.payload.data();
    out.weight.payload_bytes = out.payload.size();
    return out;
}

// A resident split bank: three plane allocations, one expert image per slot.
struct SplitBank {
    SplitBank(QType type, std::int32_t rows, std::int32_t columns, const PlaneBytes& planes)
        : geometry(planes), codes(planes.code * kExperts), scales(planes.scale * kExperts),
          high(planes.high == 0 ? 0 : planes.high * kExperts) {}

    void upload(const quantized_weight::PackedWeight& packed, std::uint32_t expert) {
        const auto* base = packed.payload.data();
        codes.copy_from_host(base, geometry.code, static_cast<std::size_t>(expert) * geometry.code);
        if (geometry.high != 0) {
            high.copy_from_host(base + geometry.code, geometry.high,
                                static_cast<std::size_t>(expert) * geometry.high);
        }
        scales.copy_from_host(base + geometry.code + geometry.high, geometry.scale,
                              static_cast<std::size_t>(expert) * geometry.scale);
    }

    ExpertMatrixDeviceView view() const {
        ExpertMatrixDeviceView out;
        out.codes        = reinterpret_cast<const std::byte*>(codes.p);
        out.high         = geometry.high == 0 ? nullptr : reinterpret_cast<const std::byte*>(high.p);
        out.scales       = reinterpret_cast<const std::byte*>(scales.p);
        out.code_stride  = geometry.code;
        out.high_stride  = geometry.high;
        out.scale_stride = geometry.scale;
        return out;
    }

    PlaneBytes geometry;
    DeviceBuffer codes, scales, high;
};

std::vector<float> patterned(std::int32_t rows, std::int32_t columns, std::uint32_t seed,
                             float amplitude) {
    std::vector<float> values(static_cast<std::size_t>(rows) * columns);
    fill_uniform(values, seed, -amplitude, amplitude);
    round_to_bf16(values);
    return values;
}

double silu(double x) { return x / (1.0 + std::exp(-x)); }

// The reference for one token: sum over routes of c_j * down_e(silu(gate_e x) * up_e x).
struct Oracle {
    std::vector<quantized_weight::PackedWeight> gate, up, down;
    std::array<int, kTopK> experts{};

    std::vector<double> run(const std::vector<float>& x, const std::array<float, kTopK>& weights) const {
        std::vector<double> result(kHidden, 0.0);
        std::vector<float> activation(kIntermediate);
        for (std::size_t route = 0; route < weights.size(); ++route) {
            const auto& g = gate[route];
            const auto& u = up[route];
            const auto& d = down[route];
            for (std::int32_t row = 0; row < kIntermediate; ++row) {
                const double gate_value = quantized_weight::dot_fp64(g, row, x.data(), kHidden);
                const double up_value   = quantized_weight::dot_fp64(u, row, x.data(), kHidden);
                activation[static_cast<std::size_t>(row)] =
                    static_cast<float>(silu(gate_value) * up_value);
            }
            for (std::int32_t row = 0; row < kHidden; ++row) {
                const double value =
                    quantized_weight::dot_fp64(d, row, activation.data(), kIntermediate);
                result[static_cast<std::size_t>(row)] +=
                    static_cast<double>(weights[route]) * value;
            }
        }
        return result;
    }
};

int run_case(DeviceExecutionView execution, const Device& device, std::int32_t tokens) {
    // One logical source per role, packed once; every expert image is the same payload with its
    // scale plane multiplied by a per-expert factor, so an addressing mistake changes the result.
    const auto gate_base = quantized_weight::pack_q4_row_split(
        patterned(kIntermediate, kHidden, 0x51a7U, 0.09F), kIntermediate, kHidden);
    const auto up_base = quantized_weight::pack_q4_row_split(
        patterned(kIntermediate, kHidden, 0x77b1U, 0.07F), kIntermediate, kHidden);
    const auto down_base = quantized_weight::pack_q6_row_split(
        patterned(kHidden, kIntermediate, 0x2c93U, 0.11F), kHidden, kIntermediate);
    const PlaneBytes gate_planes = plane_bytes(gate_base);
    const PlaneBytes up_planes   = plane_bytes(up_base);
    const PlaneBytes down_planes = plane_bytes(down_base);

    SplitBank gate(QType::Q4_G64_FP16, kIntermediate, kHidden, gate_planes);
    SplitBank up(QType::Q4_G64_FP16, kIntermediate, kHidden, up_planes);
    SplitBank down(QType::Q6_G64_FP16, kHidden, kIntermediate, down_planes);

    const std::array<int, kTopK> routed{{7, 33, 65, 128, 200, 256, 300, 400, 480, 511}};
    const std::array<float, kTopK> coefficients{{0.19F, 0.17F, 0.13F, 0.11F, 0.09F,
                                                 0.08F, 0.07F, 0.06F, 0.05F, 0.05F}};

    // Every expert gets a distinct factor; the routed set is remembered so the reference reads the
    // exact payload that was uploaded.
    std::vector<quantized_weight::PackedWeight> gate_images, up_images, down_images;
    gate_images.reserve(kExperts);
    up_images.reserve(kExperts);
    down_images.reserve(kExperts);
    const auto factor_of = [](int expert) {
        return 1.0F + 0.01F * static_cast<float>((expert * 7) % 13);
    };
    for (int expert = 0; expert < kExperts; ++expert) {
        const float factor = factor_of(expert);
        gate_images.push_back(scaled_image(gate_base, gate_planes, factor));
        up_images.push_back(scaled_image(up_base, up_planes, factor));
        down_images.push_back(scaled_image(down_base, down_planes, factor));
    }
    for (int expert = 0; expert < kExperts; ++expert) {
        gate.upload(gate_images[static_cast<std::size_t>(expert)], static_cast<std::uint32_t>(expert));
        up.upload(up_images[static_cast<std::size_t>(expert)], static_cast<std::uint32_t>(expert));
        down.upload(down_images[static_cast<std::size_t>(expert)], static_cast<std::uint32_t>(expert));
    }

    Oracle oracle;
    for (std::size_t route = 0; route < routed.size(); ++route) {
        oracle.experts[route] = routed[route];
        oracle.gate.push_back(gate_images[static_cast<std::size_t>(routed[route])]);
        oracle.up.push_back(up_images[static_cast<std::size_t>(routed[route])]);
        oracle.down.push_back(down_images[static_cast<std::size_t>(routed[route])]);
    }

    // Inputs and routes.
    std::vector<float> input(static_cast<std::size_t>(kHidden) * tokens);
    std::vector<double> reference(static_cast<std::size_t>(kHidden) * tokens);
    std::vector<int> ids(static_cast<std::size_t>(kTopK) * tokens);
    std::vector<float> coefficient(static_cast<std::size_t>(kTopK) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        std::vector<float> x(kHidden);
        fill_uniform(x, 0x1000U + static_cast<std::uint32_t>(token), -1.0F, 1.0F);
        round_to_bf16(x);
        std::copy(x.begin(), x.end(), input.begin() + static_cast<std::size_t>(token) * kHidden);
        const auto value = oracle.run(x, coefficients);
        std::copy(value.begin(), value.end(),
                  reference.begin() + static_cast<std::size_t>(token) * kHidden);
        for (std::size_t route = 0; route < routed.size(); ++route) {
            ids[static_cast<std::size_t>(token) * kTopK + route] = routed[route];
            coefficient[static_cast<std::size_t>(token) * kTopK + route] = coefficients[route];
        }
    }

    DeviceBuffer device_input = to_device_bf16(input);
    DeviceBuffer device_ids   = to_device_i32(ids);
    DeviceBuffer device_coefficient = to_device_f32(coefficient);
    GuardedDeviceBuffer device_output(
        static_cast<std::size_t>(kHidden) * static_cast<std::size_t>(tokens) * sizeof(std::uint16_t));
    device_output.fill(0);

    ops::CudaSwiGLUExperts experts{};
    experts.bank.count  = kExperts;
    experts.bank.device = device;
    experts.gate        = gate.view();
    experts.up          = up.view();
    experts.bank.down   = down.view();

    ops::RoutedSwiGLUProfile profile{{kHidden, kIntermediate, QType::Q4_G64_FP16,
                                      QType::Q6_G64_FP16, QuantLayout::RowSplit,
                                      QuantLayout::RowSplit, ops::LinearPolicy::A16Only,
                                      ops::LinearPolicy::A16Only},
                                     kExperts, kTopK};

    Tensor input_tensor(MemorySpan{device_input.p, device_input.bytes, device}, DType::BF16,
                        {kHidden, tokens});
    Tensor route_tensor(MemorySpan{device_ids.p, device_ids.bytes, device}, DType::I32,
                        {kTopK, tokens});
    Tensor coefficient_tensor(MemorySpan{device_coefficient.p, device_coefficient.bytes, device},
                              DType::FP32, {kTopK, tokens});
    Tensor destination(MemorySpan{device_output.data(), device_output.bytes(), device}, DType::BF16,
                       {kHidden, tokens});
    const ops::ExpertSubset subset{ops::ExpertSubsetKind::All, Tensor{}};

    const std::size_t workspace_bytes = ops::routed_swiglu_workspace_capacity_bytes(
        profile, subset.kind, DType::BF16, ops::MoeEpilogue::Write, tokens, tokens);
    GuardedDeviceBuffer scratch(workspace_bytes);
    MemoryArena arena(MemorySpan{scratch.data(), scratch.bytes(), device});
    ops::routed_swiglu(input_tensor, profile, experts,
                       ops::ExpertRoutes{route_tensor, coefficient_tensor}, subset,
                       ops::MoeEpilogue::Write, destination, arena, execution);
    cuda_synchronize();

    const auto actual = from_device_bf16(device_output.data(), static_cast<std::size_t>(kHidden) * tokens);
    return verify_reduction("qwen4exp split experts T=" + std::to_string(tokens), actual, reference,
                            kA16Tolerance);
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    DeviceContext context;
    const DeviceExecutionView execution = context.execution_view().on_stream(nullptr);
    int failures                        = 0;
    // T=1 is the decode path (one block per token), T=8 keeps the token traversal with several
    // tokens per block, and T=64 crosses into the grouped traversal.
    for (const std::int32_t tokens : {1, 8, 64}) {
        failures += run_case(execution, execution.device(), tokens);
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " qwen4exp split expert banks\n";
    return failures == 0 ? 0 : 1;
}
