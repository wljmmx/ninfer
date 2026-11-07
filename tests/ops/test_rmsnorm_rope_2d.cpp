// Fused D64 axial RMSNorm + split-half 2-D RoPE: independent FP64 oracle,
// natural-alignment, padded gains and graph replay.
#include "ninfer/ops/rmsnorm_rope.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/norm_test_common.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::norm;

namespace {

constexpr ReductionCriterion kAxialNormBf16{2.e-3, 2.e-5, 6.9e-3};

std::vector<double> reference(const std::vector<float>& x, const std::vector<float>& gain,
                              const std::vector<int>& positions, float epsilon, float theta) {
    std::vector<double> result(x.size());
    const int tokens = static_cast<int>(positions.size() / 2);
    for (int t = 0; t < tokens; ++t)
        for (int h = 0; h < 16; ++h) {
            const auto base = (std::size_t(t) * 16 + h) * 64;
            double square   = 0;
            for (int d = 0; d < 64; ++d) square += double(x[base + d]) * x[base + d];
            const double inverse = 1 / std::sqrt(square / 64 + epsilon);
            for (int i = 0; i < 32; ++i) {
                const double angle = positions[t + (i / 16) * tokens] *
                                     std::pow(double(theta), -double(i % 16) / 16);
                const double first = x[base + i] * inverse * gain[i],
                             second = x[base + i + 32] * inverse * gain[i + 32];
                result[base + i]      = first * std::cos(angle) - second * std::sin(angle);
                result[base + i + 32] = second * std::cos(angle) + first * std::sin(angle);
            }
        }
    return result;
}

int run(DeviceContext& device, int tokens, bool graph, bool natural_gain, float theta) {
    constexpr float epsilon = 1.e-5F;
    std::vector<float> q(std::size_t(tokens) * 1024), k(q.size()), qw(64), kw(64);
    fill_uniform(q, 6301, -3.F, 3.F);
    fill_uniform(k, 6307, -2.F, 2.F);
    fill_uniform(qw, 6311, -1.F, 1.F);
    fill_uniform(kw, 6317, -1.F, 1.F);
    for (auto* x : {&q, &k, &qw, &kw}) round_to_bf16(*x);
    std::vector<int> positions(std::size_t(tokens) * 2);
    for (int t = 0; t < tokens; ++t) {
        positions[t]          = t % 61 - 2;
        positions[t + tokens] = t / 61 + 3;
    }
    const bool unaligned = natural_gain;
    DeviceInput gain_q = make_input(qw, unaligned), gain_k = make_input(kw, unaligned);
    DeviceBuffer pos   = to_device(positions);
    GuardedDeviceBuffer qdata(std::size_t(tokens) * 1024 * 2),
        kdata(std::size_t(tokens) * 1024 * 2);
    Tensor tq(qdata.data(), DType::BF16, {64, 16, tokens}),
        tk(kdata.data(), DType::BF16, {64, 16, tokens}), wq(gain_q.data, DType::BF16, {64}),
        wk(gain_k.data, DType::BF16, {64}), p(pos.p, DType::I32, {tokens, 2});
    const auto e = device.execution_view();
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    const auto launch = [&] { ops::rmsnorm_rope_2d(p, wq, wk, epsilon, theta, tq, tk, e); };
    int failures = 0;
    for (int phase = 0; phase < (graph ? 2 : 1); ++phase) {
        if (phase) {
            for (auto& x : q) x = -x;
            for (auto& x : qw) x = -x;
            for (auto& x : positions) x += 5;
        }
        const auto upload = [](void* destination, const std::vector<float>& values) {
            std::vector<std::uint16_t> bits(values.size());
            for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
            CUDA_CHECK(cudaMemcpy(destination, bits.data(), bits.size() * 2, cudaMemcpyHostToDevice));
        };
        upload(tq.data, q);
        upload(tk.data, k);
        const auto upload_gain = [&](DeviceInput& input, const std::vector<float>& values) {
            const std::size_t leading = unaligned ? 1 : 0;
            for (std::size_t i = 0; i < values.size(); ++i)
                input.expected[i + leading] = f32_to_bf16(values[i]);
            CUDA_CHECK(cudaMemcpy(input.storage.p, input.expected.data(),
                                  input.expected.size() * 2, cudaMemcpyHostToDevice));
        };
        upload_gain(gain_q, qw);
        upload_gain(gain_k, kw);
        CUDA_CHECK(cudaMemcpy(pos.p, positions.data(), positions.size() * 4,
                              cudaMemcpyHostToDevice));
        if (graph && phase == 0) {
            definition.capture(device.stream, launch);
            executable.instantiate(definition);
        }
        if (graph)
            executable.launch(device.stream);
        else
            launch();
        cuda_synchronize(device.stream);
        failures += verify_reduction("axial q norm", from_device_bf16(tq.data, q.size()),
                                     reference(q, qw, positions, epsilon, theta), kAxialNormBf16);
        failures += verify_reduction("axial k norm", from_device_bf16(tk.data, k.size()),
                                     reference(k, kw, positions, epsilon, theta), kAxialNormBf16);
        failures += verify_preserved("q gain", gain_q);
        failures += verify_preserved("k gain", gain_k);
        failures += verify_exact("positions", from_device<int>(pos, positions.size()), positions);
        failures += qdata.verify_guards("q guards") + kdata.verify_guards("k guards");
    }
    return failures;
}

int contract(DeviceContext& device) {
    const auto e = device.execution_view();
    DeviceBuffer backing(16384);
    auto at      = [&](std::size_t offset) {
        return static_cast<void*>(static_cast<std::uint8_t*>(backing.p) + offset);
    };
    Tensor q(at(0), DType::BF16, {64, 16, 2}), k(at(4096), DType::BF16, {64, 16, 2}),
        w(at(8192), DType::BF16, {64}), p(at(8320), DType::I32, {2, 2});
    int failures = 0;
    auto launch  = [&] { ops::rmsnorm_rope_2d(p, w, w, 1.e-5F, 10000.F, q, k, e); };
    auto reject  = [&] {
        try {
            launch();
            ++failures;
        } catch (const std::invalid_argument&) {}
    };
    auto saved = k;
    k.data     = q.data;
    reject();
    k       = saved;
    q.ne[0] = 72;
    reject();
    q.ne[0] = 64;
    p.ne[1] = 1;
    reject();
    p.ne[1]           = 2;
    auto saved_weight = w;
    w.data            = q.data;
    reject();
    w       = saved_weight;
    q.ne[2] = k.ne[2] = p.ne[0] = 0;
    launch();
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP rmsnorm_rope_2d (no CUDA device)\n";
        return 0;
    }
    DeviceContext device;
    int failures = 0;
    for (int tokens : {1, 2, 7, 32, 127, 1024, 4097})
        failures += run(device, tokens, tokens == 7, false, 10000.F);
    failures += run(device, 3, true, true, 5700.F);
    failures += contract(device);
    std::cout << (failures ? "FAIL" : "OK") << " rmsnorm_rope_2d\n";
    return failures ? 1 : 0;
}
