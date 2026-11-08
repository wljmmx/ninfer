#include "core/arena.h"
#include "core/device.h"
#include "core/memory.h"
#include "ninfer/ops/moe.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
namespace qw = ninfer::test::quantized_weight;

namespace {

// Qwen3.5-35B-A3B expert geometry: the only shape the block-scaled routed kernels are compiled
// for. Two weight slots are shared by all 512 experts, so the fixture stays small while the
// offsets table still exercises per-expert addressing.
constexpr int kHidden       = 2560;
constexpr int kIntermediate = 640;
constexpr int kExperts      = 512;
constexpr int kTopK         = 10;

// One budget for the whole routed formula at A8: the FP64 oracle reads the *original* BF16 input
// and the decoded weights, so the activation quantisation inside the kernel must fit inside it.
// The absolute RMS floor (1e-7) is the near-zero tail budget the fixture deliberately plants: the
// first gate/up row tile is all zeros, so SiLU(0)=0 and those rows carry no signal for a relative
// norm to divide by (upstream 757194af).
constexpr ReductionCriterion kA8{0.08, 1e-7, 0.12, 1e-7};

int cuda_ordinal() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

CudaExecutionView execution_view() {
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, cuda_ordinal()));
    return {nullptr, props.multiProcessorCount, Device::cuda(cuda_ordinal())};
}

struct Matrix {
    int n = 0, k = 0;
    std::size_t stride = 0, scale_offset = 0;
    std::vector<std::uint8_t> payload[2];
    std::vector<float> decoded[2];
    DeviceBuffer data, offsets;

    Matrix(int rows, int columns, unsigned seed)
        : n(rows), k(columns), offsets(std::size_t(kExperts) * 8) {
        const auto code_bytes  = std::size_t(n) * std::size_t(k);
        const auto scale_bytes = std::size_t(n / 128) * std::size_t(k / 128) * 2;
        scale_offset           = code_bytes;
        for (int slot = 0; slot < 2; ++slot) {
            std::vector<std::uint8_t> bytes(code_bytes + scale_bytes);
            for (std::size_t b = 0; b < code_bytes; ++b) {
                unsigned code = qw::detail::mix64(b + seed + slot * 998123ULL) & 255;
                // 0x7f/0xff are NaN in E4M3FN; the kernel never sees them on this fixture.
                if ((code & 127) == 127) { --code; }
                bytes[b] = std::uint8_t(code);
            }
            const bool gate_up = n == 2 * kIntermediate;
            for (int nr = 0; nr < n / 128; ++nr) {
                for (int kc = 0; kc < k / 128; ++kc) {
                    // The first gate/up row tile is all zeros: an exact-zero tile must produce an
                    // exact-zero product rather than a denormal, and it also exercises SiLU(0).
                    const float scale = (gate_up && nr == 0)
                                            ? 0.0F
                                            : (1 + (nr * 3 + kc * 7 + seed + slot) % 13) * 0.0000083F;
                    qw::detail::store_u16_le(bytes, scale_offset + (nr * (k / 128) + kc) * 2,
                                             f32_to_bf16(scale));
                }
            }
            std::vector<float>& values = decoded[slot];
            values.resize(std::size_t(n) * std::size_t(k));
            for (int r = 0; r < n; ++r) {
                for (int c = 0; c < k; ++c) {
                    const auto scale = qw::detail::load_u16_le(
                        bytes, scale_offset + ((r / 128) * (k / 128) + c / 128) * 2);
                    values[std::size_t(r) * k + c] =
                        static_cast<float>(qw::detail::decode_e4m3fn(bytes[std::size_t(r) * k + c]) *
                                           qw::detail::bf16_to_f32(scale));
                }
            }
            payload[slot] = std::move(bytes);
        }
        stride = (payload[0].size() + 255) / 256 * 256;
        data   = DeviceBuffer(2 * stride);
        for (int slot = 0; slot < 2; ++slot) {
            data.copy_from_host(payload[slot].data(), payload[slot].size(), slot * stride);
        }
        publish(false);
    }

    void publish(bool flip) {
        std::vector<std::uint64_t> indices(kExperts, 0);
        for (int e = 0; e < kExperts; ++e) {
            indices[e] = std::uint64_t((e & 1) ^ int(flip)) * stride;
        }
        offsets.copy_from_host(indices.data(), std::size_t(kExperts) * 8);
    }

    ExpertMatrixDeviceView view() const {
        ExpertMatrixDeviceView result;
        result.codes           = static_cast<const std::byte*>(data.p);
        result.scales          = result.codes + scale_offset;
        result.offsets         = static_cast<const std::uint64_t*>(offsets.p);
        result.indexed_storage = MemorySpan{data.p, data.bytes, Device::cuda(cuda_ordinal())};
        return result;
    }
};

std::vector<float> random_values(std::size_t count, unsigned seed, float limit) {
    std::vector<float> result(count);
    fill_uniform(result, seed, -limit, limit);
    round_to_bf16(result);
    return result;
}

// F_e(x) = down * (SiLU(gate * x) * (up * x)) in double, from the decoded weights. Block FP8 has
// no matrix divisor, so none appears here.
std::vector<double> ffn(const std::vector<float>& x, const std::vector<float>& gate_up,
                        const std::vector<float>& down) {
    std::vector<double> h(kIntermediate), out(kHidden);
    for (int row = 0; row < kIntermediate; ++row) {
        double gate = 0, up = 0;
        for (int c = 0; c < kHidden; ++c) {
            gate += double(x[std::size_t(c)]) * gate_up[std::size_t(row) * kHidden + c];
            up += double(x[std::size_t(c)]) * gate_up[std::size_t(row + kIntermediate) * kHidden + c];
        }
        h[std::size_t(row)] = gate / (1 + std::exp(-gate)) * up;
    }
    for (int row = 0; row < kHidden; ++row) {
        for (int c = 0; c < kIntermediate; ++c) {
            out[std::size_t(row)] += h[std::size_t(c)] * down[std::size_t(row) * kIntermediate + c];
        }
    }
    return out;
}

struct Fixture {
    ops::RoutedSwiGLUProfile profile;
    Matrix gate_up, down;
    ops::CudaSwiGLUExperts experts;

    Fixture()
        : profile{{kHidden, kIntermediate, QType::FP8_E4M3FN_BLOCK128X128_BF16,
                   QType::FP8_E4M3FN_BLOCK128X128_BF16, QuantLayout::BlockScaleM128K128,
                   QuantLayout::BlockScaleM128K128, ops::LinearPolicy::AllowA8,
                   ops::LinearPolicy::AllowA8},
                  kExperts, kTopK},
          gate_up(2 * kIntermediate, kHidden, 101), down(kHidden, kIntermediate, 211) {
        experts.bank = {gate_up.view(), down.view(), kExperts, Device::cuda(cuda_ordinal())};
        experts.activation_divisors = {};
    }
};

int run(Fixture& fixture, int tokens) {
    const auto execution = execution_view();
    std::vector<float> x(std::size_t(kHidden) * std::size_t(tokens));
    for (int t = 0; t < tokens; ++t) {
        const auto values = random_values(kHidden, 1701 + unsigned(t), 0.7F);
        std::copy(values.begin(), values.end(), x.begin() + std::ptrdiff_t(t) * kHidden);
        // Deliberately zero K groups and a large last coordinate expose scale-tail handling.
        for (int c = 0; c < 128; ++c) { x[std::size_t(t) * kHidden + c] = 0; }
        x[std::size_t(t) * kHidden + kHidden - 1] = 0.875F;
    }

    std::vector<int> ids(std::size_t(kTopK) * std::size_t(tokens));
    std::vector<float> alpha(ids.size());
    for (int t = 0; t < tokens; ++t) {
        float total = 0;
        for (int j = 0; j < kTopK; ++j) {
            const int route = j * (kExperts / kTopK) + 3 * t;
            ids[std::size_t(t) * kTopK + std::size_t(j)] = route % kExperts;
            alpha[std::size_t(t) * kTopK + std::size_t(j)] = 1.0F / float(j + 1);
            total += alpha[std::size_t(t) * kTopK + std::size_t(j)];
        }
        for (int j = 0; j < kTopK; ++j) { alpha[std::size_t(t) * kTopK + std::size_t(j)] /= total; }
    }

    auto dx = to_device_bf16(x), di = to_device(ids), da = to_device(alpha);
    const auto device = Device::cuda(cuda_ordinal());
    Tensor input(MemorySpan{dx.p, dx.bytes, device}, DType::BF16, {kHidden, tokens});
    ops::ExpertRoutes routes{Tensor(MemorySpan{di.p, di.bytes, device}, DType::I32, {kTopK, tokens}),
                             Tensor(MemorySpan{da.p, da.bytes, device}, DType::FP32,
                                    {kTopK, tokens})};
    const ops::ExpertSubset subset{ops::ExpertSubsetKind::All, Tensor{}};

    const std::size_t bytes = ops::routed_swiglu_workspace_capacity_bytes(
        fixture.profile, subset.kind, DType::FP32, ops::MoeEpilogue::Write, tokens, tokens);
    GuardedDeviceBuffer scratch(bytes), result(std::size_t(kHidden) * std::size_t(tokens) * 4);
    MemoryArena arena(MemorySpan{scratch.data(), scratch.bytes(), device});
    Tensor output(MemorySpan{result.data(), result.bytes(), device}, DType::FP32,
                  {kHidden, tokens});

    ops::routed_swiglu(input, fixture.profile, fixture.experts, routes, subset,
                       ops::MoeEpilogue::Write, output, arena, execution);
    cuda_synchronize();

    const std::vector<float> got_float =
        from_device<float>(result.data(), std::size_t(kHidden) * std::size_t(tokens));
    const std::vector<double> got(got_float.begin(), got_float.end());
    std::vector<double> ref(got.size(), 0.0);
    for (int t = 0; t < tokens; ++t) {
        std::vector<double> term[2];
        bool computed[2] = {false, false};
        const std::vector<float> xt(x.begin() + std::ptrdiff_t(t) * kHidden,
                                    x.begin() + std::ptrdiff_t(t + 1) * kHidden);
        for (int j = 0; j < kTopK; ++j) {
            const int expert = ids[std::size_t(t) * kTopK + std::size_t(j)];
            const int slot   = expert & 1;
            if (!computed[slot]) {
                term[slot]     = ffn(xt, fixture.gate_up.decoded[slot], fixture.down.decoded[slot]);
                computed[slot] = true;
            }
            const double coefficient = alpha[std::size_t(t) * kTopK + std::size_t(j)];
            for (int row = 0; row < kHidden; ++row) {
                ref[std::size_t(t) * kHidden + row] += coefficient * term[slot][std::size_t(row)];
            }
        }
    }

    int failures = 0;
    failures += result.verify_guards("block-scaled fp8 output");
    failures += verify_reduction("block-scaled fp8 routed SwiGLU T=" + std::to_string(tokens), got,
                                 ref, kA8);
    return failures;
}

// T114 decision, executable: the NVFP4 expert form must refuse to run on this build instead of
// computing something else. The message names the hardware and the opt-in that unlocks it.
int nvfp4_is_dead() {
    const auto execution = execution_view();
    Fixture fixture; // FP8 payloads are irrelevant: the call must fail before touching them.
    const auto profile = ops::RoutedSwiGLUProfile{
        {kHidden, kIntermediate, QType::NVFP4, QType::NVFP4, QuantLayout::BlockScaleK16M128x4,
         QuantLayout::BlockScaleK16M128x4, ops::LinearPolicy::AllowA4, ops::LinearPolicy::AllowA4},
        kExperts, kTopK};
    ops::CudaSwiGLUExperts nvfp4_experts = fixture.experts;
    nvfp4_experts.activation_divisors    = {1.0F, 1.0F};
    // Operand validation runs before the launch: give the A4 form the per-expert divisors it
    // declares so the call reaches the kernel branch instead of stopping at the operand check.
    std::vector<float> ones(kExperts, 1.0F);
    DeviceBuffer divisors(std::size_t(kExperts) * sizeof(float));
    divisors.copy_from_host(ones.data(), std::size_t(kExperts) * sizeof(float));
    for (auto* view : {&nvfp4_experts.bank.gate_up, &nvfp4_experts.bank.down}) {
        view->divisors       = static_cast<const float*>(divisors.p);
        view->divisor_stride = 1;
    }
    auto dx = to_device_bf16(std::vector<float>(std::size_t(kHidden) * 2, 0.5F));
    std::vector<int> ids(std::size_t(kTopK) * 2, 0);
    std::vector<float> alpha(std::size_t(kTopK) * 2, 0.1F);
    auto di = to_device(ids), da = to_device(alpha);
    const auto device = Device::cuda(cuda_ordinal());
    Tensor input(MemorySpan{dx.p, dx.bytes, device}, DType::BF16, {kHidden, 2});
    ops::ExpertRoutes routes{Tensor(MemorySpan{di.p, di.bytes, device}, DType::I32, {kTopK, 2}),
                             Tensor(MemorySpan{da.p, da.bytes, device}, DType::FP32, {kTopK, 2})};
    GuardedDeviceBuffer result(std::size_t(kHidden) * 2 * 4);
    Tensor output(MemorySpan{result.data(), result.bytes(), device}, DType::FP32, {kHidden, 2});
    const std::size_t bytes =
        ops::routed_swiglu_workspace_capacity_bytes(profile, ops::ExpertSubsetKind::All, DType::FP32,
                                                    ops::MoeEpilogue::Write, 2, 2);
    GuardedDeviceBuffer scratch(bytes);
    MemoryArena arena(MemorySpan{scratch.data(), scratch.bytes(), device});
    try {
        ops::routed_swiglu(input, profile, nvfp4_experts, routes,
                           {ops::ExpertSubsetKind::All, Tensor{}}, ops::MoeEpilogue::Write, output,
                           arena, execution);
    } catch (const std::logic_error& error) {
        const std::string message = error.what();
        if (message.find("sm_100+") == std::string::npos ||
            message.find("NINFER_ENABLE_BLOCK_SCALED_MOE_MMA") == std::string::npos) {
            std::cerr << "FAIL: NVFP4 MoE refusal does not name its hardware and opt-in: " << message
                      << '\n';
            return 1;
        }
        return 0;
    }
    std::cerr << "FAIL: NVFP4 MoE experts ran on an sm_89 build\n";
    return 1;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    try {
        Fixture fixture;
        for (const int tokens : {1, 2, 3, 5, 9}) { failures += run(fixture, tokens); }
        failures += nvfp4_is_dead();
    } catch (const std::exception& error) {
        std::cerr << "block-scaled MoE test exception: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " block-scaled FP8 MoE + NVFP4 refusal\n";
    return failures == 0 ? 0 : 1;
}


