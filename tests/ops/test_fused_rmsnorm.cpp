// Paired RMSNorm+concat and residual-add RMSNorm: independent FP64 oracle,
// natural-alignment, in-place residual and graph-replay coverage.
#include "ninfer/ops/paired_rmsnorm_concat.h"
#include "ninfer/ops/residual_add_rmsnorm.h"
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

constexpr ReductionCriterion kFusedNormBf16{2.e-3, 2.e-6, 4.1e-3};
constexpr float kEpsilon = 1.e-5F;

int run(DeviceContext& device, bool paired, int d, int m, bool alias, bool unaligned,
        bool replay) {
    const std::size_t size = std::size_t(d) * std::size_t(m);
    std::vector<float> x(size), y(size), wx(d), wy(d);
    fill_uniform(x, 3801, -3.F, 3.F);
    fill_uniform(y, 3803, -1.F, 1.F);
    fill_uniform(wx, 3807, -1.5F, 1.5F);
    fill_uniform(wy, 3809, -1.5F, 1.5F);
    for (auto* v : {&x, &y, &wx, &wy}) round_to_bf16(*v);

    DeviceInput dx = make_input(x, unaligned), dy = make_input(y, unaligned),
                dwx = make_input(wx, unaligned), dwy = make_input(wy, unaligned);
    std::vector<int> enabled(m);
    for (int i = 0; i < m; ++i) enabled[i] = i % 3 != 0;
    DeviceBuffer mask = to_device_i32(enabled);

    const std::size_t leading = unaligned ? sizeof(std::uint16_t) : 0;
    GuardedDeviceBuffer out(leading + size * (paired ? 4 : 2) * sizeof(std::uint16_t));
    GuardedDeviceBuffer residual(alias ? 0 : leading + size * sizeof(std::uint16_t));
    void* out_data = static_cast<std::uint8_t*>(out.data()) + leading;
    void* residual_data =
        alias ? static_cast<std::uint8_t*>(dx.data)
              : static_cast<std::uint8_t*>(residual.data()) + leading;

    Tensor tx(dx.data, DType::BF16, {d, m}), ty(dy.data, DType::BF16, {d, m}),
        gx(dwx.data, DType::BF16, {d}), gy(dwy.data, DType::BF16, {d}),
        enable(mask.p, DType::I32, {m}),
        output(out_data, DType::BF16, {paired ? 2 * d : d, m}),
        r(residual_data, DType::BF16, {d, m});

    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    const auto launch = [&] {
        if (paired)
            ops::paired_rmsnorm_concat(tx, ty, gx, gy, enable, kEpsilon, output,
                                       device.execution_view());
        else
            ops::residual_add_rmsnorm(tx, ty, gx, kEpsilon, r, output, device.execution_view());
    };

    int failures = 0;
    for (int phase = 0; phase < (replay ? 2 : 1); ++phase) {
        if (phase) {
            for (auto& value : x) value = -value;
            for (auto& value : wx) value = -value;
            for (auto& value : enabled) value = 1 - value;
        }
        const auto upload = [&](DeviceInput& input, const std::vector<float>& values) {
            for (std::size_t i = 0; i < values.size(); ++i)
                input.expected[i + leading / 2] = f32_to_bf16(values[i]);
            CUDA_CHECK(cudaMemcpy(input.storage.p, input.expected.data(),
                                  input.expected.size() * 2, cudaMemcpyHostToDevice));
        };
        upload(dx, x);
        upload(dwx, wx);
        CUDA_CHECK(cudaMemcpy(mask.p, enabled.data(), m * 4, cudaMemcpyHostToDevice));
        out.fill(0xff);
        if (!alias) residual.fill(0xff);
        if (replay && phase == 0) {
            definition.capture(device.stream, launch);
            graph.instantiate(definition);
        }
        if (replay)
            graph.launch(device.stream);
        else
            launch();
        cuda_synchronize(device.stream);

        std::vector<double> reference(size * (paired ? 2 : 1));
        std::vector<std::uint16_t> represented(size);
        for (int column = 0; column < m; ++column) {
            for (int half = 0; half < (paired ? 2 : 1); ++half) {
                const auto base = std::size_t(column) * d;
                double sum      = 0;
                for (int i = 0; i < d; ++i) {
                    double value;
                    if (paired)
                        value = (half ? y : x)[base + i];
                    else {
                        represented[base + i] = f32_to_bf16(
                            static_cast<float>(double(x[base + i]) + double(y[base + i])));
                        value = bf16_to_f32(represented[base + i]);
                    }
                    sum += value * value;
                }
                const double inv = 1 / std::sqrt(sum / d + kEpsilon);
                for (int i = 0; i < d; ++i) {
                    const double value =
                        paired ? (half ? y : x)[base + i] : bf16_to_f32(represented[base + i]);
                    reference[std::size_t(column) * (paired ? 2 * d : d) + half * d + i] =
                        paired && half == 0 && !enabled[column]
                            ? 0
                            : value * (half ? wy : wx)[i] * inv;
                }
            }
        }
        failures += verify_reduction("fused norm", from_device_bf16(out_data, reference.size()),
                                     reference, kFusedNormBf16);
        if (!paired)
            failures += verify_exact("published residual",
                                     from_device<std::uint16_t>(residual_data, size), represented);
        else {
            const auto actual = from_device<std::uint16_t>(out_data, size * 2);
            for (int column = 0; column < m; ++column)
                if (!enabled[column])
                    for (int i = 0; i < d; ++i)
                        if (actual[std::size_t(column) * 2 * d + i] != 0) {
                            ++failures;
                            break;
                        }
        }
        if (!alias) failures += verify_preserved("first input", dx);
        failures += verify_preserved("second input", dy);
        failures += verify_preserved("first gain", dwx);
        failures += verify_preserved("second gain", dwy);
        failures += verify_exact("mask", from_device<int>(mask, m), enabled);
        failures += verify_output_storage("norm output", out, unaligned);
        if (!paired && !alias) failures += verify_output_storage("residual", residual, unaligned);
    }
    return failures;
}

int contract(DeviceContext& device, bool paired) {
    const auto e = device.execution_view();
    DeviceBuffer backing(4096);
    auto at = [&](std::size_t byte_offset) {
        return static_cast<void*>(static_cast<std::uint8_t*>(backing.p) + byte_offset);
    };
    Tensor x(at(0), DType::BF16, {64, 2}), y(at(256), DType::BF16, {64, 2}),
        g(at(512), DType::BF16, {64}), mask(at(640), DType::I32, {2}),
        out(at(768), DType::BF16, {paired ? 128 : 64, 2}), r(at(1280), DType::BF16, {64, 2});
    auto call = [&] {
        if (paired)
            ops::paired_rmsnorm_concat(x, y, g, g, mask, kEpsilon, out, e);
        else
            ops::residual_add_rmsnorm(x, y, g, kEpsilon, r, out, e);
    };
    int failures = 0;
    auto reject  = [&] {
        try {
            call();
            ++failures;
        } catch (const std::invalid_argument&) {}
    };
    auto saved = out;
    out.data   = x.data;
    reject();
    out      = saved;
    out.data = g.data;
    reject();
    out = saved;
    if (!paired) {
        auto saved_r = r;
        r.data       = static_cast<std::uint8_t*>(x.data) + 2;
        reject();
        r = saved_r;
    }
    y.ne[0] = 63;
    reject();
    y.ne[0] = 64;
    x.ne[1] = y.ne[1] = out.ne[1] = r.ne[1] = 0;
    mask.ne[0]                              = 0;
    call();
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP fused_rmsnorm (no CUDA device)\n";
        return 0;
    }
    DeviceContext device;
    int failures = 0;
    for (bool paired : {true, false}) {
        const std::string name = paired ? "paired-rmsnorm-concat" : "residual-add-rmsnorm";
        const auto add         = [&](int d, int m, bool alias, bool unaligned, bool graph) {
            const auto label = name + " d=" + std::to_string(d) + " m=" + std::to_string(m) +
                               (alias ? " inplace" : "") + (unaligned ? " unaligned" : "") +
                               (graph ? " graph" : "");
            failures += run(device, paired, d, m, alias, unaligned, graph);
        };
        for (int m : {1, 2, 8, 15, 16, 17, 64, 65, 128, 257, 1023, 1024})
            add(4096, m, false, false, m == 2);
        for (int d : {1, 33, 1024, 4151}) add(d, 3, false, false, false);
        add(4096, 5, false, true, false);
        add(33, 7, false, true, true);
        if (!paired) {
            add(4096, 4, true, false, true);
            add(1024, 129, true, false, false);
            add(1024, 17, false, false, false);
            add(1024, 257, false, false, false);
            add(33, 5, true, true, false);
        }
        failures += contract(device, paired);
    }
    std::cout << (failures ? "FAIL" : "OK") << " fused_rmsnorm\n";
    return failures ? 1 : 0;
}
