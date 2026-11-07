#include "prepare_fixture.h"
#include "core/decode_graph.h"

#include <array>
#include <bit>
#include <cmath>
#include <cuda_fp16.h>
#include <stdexcept>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::qsa::prepare;

namespace {

// A final BF16 rounding has at most 1/256 relative error. The small additional budget covers
// FP32 normalization/rotation; the pair-scaled cap remains meaningful at cancellation points.
constexpr double kRelativeL2 = 0.0041, kPairRelative = 0.0069;

struct Ideal {
    std::vector<double> values, scales;
};

Ideal oracle(const std::vector<double>& raw, const Buffer<std::uint16_t>& weight,
             std::array<int, 3> rho, const ops::QsaPrepareParameters& params) {
    const int d = raw.size();
    double sum  = 0;
    for (double x : raw) sum += x * x;
    const double inverse = 1 / std::sqrt(sum / d + double(params.rms_epsilon));
    Ideal out{std::vector<double>(d), std::vector<double>(d)};
    for (int c = 0; c < d; ++c) {
        out.values[c] = raw[c] * inverse * (1 + double(bf16_to_f32(weight.at(c))));
        out.scales[c] = std::abs(out.values[c]);
    }
    for (int i = 0; i < 32; ++i) {
        const double phase =
            double(rho[i % 3]) * std::pow(double(params.rope_theta), -double(i) / 32);
        const double cosine = std::cos(phase), sine = std::sin(phase);
        const double x = out.values[i], y = out.values[i + 32];
        out.values[i]      = x * cosine - y * sine;
        out.values[i + 32] = y * cosine + x * sine;
        out.scales[i] = out.scales[i + 32] = std::hypot(x, y);
    }
    return out;
}

struct Error {
    double sum = 0, reference = 0, max_pair = 0;
    bool finite = true;

    void add(std::uint16_t word, double expected, double scale) {
        const double actual = bf16_to_f32(word), error = actual - expected;
        finite &= std::isfinite(actual) && std::isfinite(expected);
        sum += error * error;
        reference += expected * expected;
        max_pair = std::max(max_pair, error_limit_ratio(std::abs(error), scale));
    }

    int check(const std::string& label) const {
        const double l2 = reference == 0 ? (sum == 0 ? 0 : INFINITY) : std::sqrt(sum / reference);
        const bool pass = finite && l2 <= kRelativeL2 && max_pair <= kPairRelative;
        if (!pass || error_stats_enabled())
            std::cout << label << " relative_l2=" << l2 << " max_pair=" << max_pair << '\n';
        return !pass;
    }
};

int verify(Fixture& f, const std::string& label) {
    const auto actual_q  = from_device<std::uint16_t>(f.query_output.data, f.q.host.size());
    const auto actual_iq = from_device<std::uint16_t>(f.iq_output.data, f.iq.host.size());
    const auto actual_k = f.keys.download(), actual_ik = f.compressed.download();
    auto expected_k = f.keys.host, expected_ik = f.compressed.host, expected_v = f.values.host;
    auto expected_raw  = f.records.host;
    auto expected_rope = f.record_rope.host;
    Error q_error, k_error, iq_error, ik_error;
    int failures           = 0;
    const auto coordinates = [&](int t, int b) {
        return std::array{f.rope.at(t, 0, b), f.rope.at(t, f.A == 1 ? 0 : 1, b),
                          f.rope.at(t, f.A == 1 ? 0 : 2, b)};
    };
    const auto head = [&](const auto& input, int h, int t, int b, const auto& weight,
                          const auto& actual, std::size_t offset, Error& error) {
        std::vector<double> raw(input.view.ne[0]);
        for (int c = 0; c < input.view.ne[0]; ++c) raw[c] = bf16_to_f32(input.at(c, h, t, b));
        const auto ideal = oracle(raw, weight, coordinates(t, b), f.parameters);
        for (int c = 0; c < input.view.ne[0]; ++c)
            error.add(actual[offset + c], ideal.values[c], ideal.scales[c]);
    };
    for (int b = 0; b < f.B; ++b) {
        const int n = f.valid.at(b), base = f.bases[b];
        for (int t = 0; t < f.T; ++t) {
            if (t >= n) {
                for (int h = 0; h < 24; ++h)
                    for (int c = 0; c < 256; ++c) failures += actual_q[f.q.index(c, h, t, b)] != 0;
                for (int h = 0; h < 4; ++h)
                    for (int c = 0; c < 128; ++c)
                        failures += actual_iq[f.iq.index(c, h, t, b)] != 0;
                continue;
            }
            for (int h = 0; h < 24; ++h)
                head(f.q, h, t, b, f.wq, actual_q, f.q.index(0, h, t, b), q_error);
            for (int h = 0; h < 4; ++h)
                head(f.iq, h, t, b, f.wiq, actual_iq, f.iq.index(0, h, t, b), iq_error);
            const int p = f.positions.at(t, b), page = f.tables.at(p / 64, f.table_rows.at(b));
            for (int h = 0; h < 2; ++h) {
                head(f.k, h, t, b, f.wk, actual_k, f.keys.index(0, p % 64, h, page), k_error);
                for (int c = 0; c < 256; ++c) {
                    const auto index = f.keys.index(c, p % 64, h, page);
                    // Approximate values have been checked above; retain actual words here
                    // solely to check that every other cache byte stayed untouched.
                    expected_k[index]                              = actual_k[index];
                    expected_v[f.values.index(c, p % 64, h, page)] =
                        __half_as_ushort(__float2half_rn(bf16_to_f32(f.v.at(c, h, t, b))));
                }
            }
            const int record = f.offsets.at(b) + t;
            for (int c = 0; c < 128; ++c)
                expected_raw[f.records.index(c, record, b)] = f.ik.at(c, t, b);
            const auto rho = coordinates(t, b);
            for (int a = 0; a < 3; ++a) expected_rope[f.record_rope.index(a, record, b)] = rho[a];
        }
        if (n == 0) continue;
        // Independent chronological grouping, starting with the canonical pending prefix.
        std::vector<std::vector<double>> pending;
        std::array<int, 3> first{};
        const int slot = f.slots.at(b);
        for (int j = 0; j < base % 4; ++j) {
            std::vector<double> raw(128);
            for (int c = 0; c < 128; ++c) raw[c] = bf16_to_f32(f.tail.at(c, j, slot));
            pending.push_back(raw);
        }
        if (!pending.empty())
            for (int a = 0; a < 3; ++a) first[a] = f.first.at(a, slot);
        for (int t = 0; t < n; ++t) {
            if (pending.empty()) first = coordinates(t, b);
            std::vector<double> raw(128);
            for (int c = 0; c < 128; ++c) raw[c] = bf16_to_f32(f.ik.at(c, t, b));
            pending.push_back(raw);
            if (pending.size() != 4) continue;
            std::vector<double> pooled(128);
            for (int c = 0; c < 128; ++c) {
                double sum = 0;
                for (const auto& token : pending) sum += token[c];
                pooled[c] = bf16_to_f32(f32_to_bf16(float(sum / 4)));
            }
            const auto ideal = oracle(pooled, f.wik, first, f.parameters);
            const int g = (base + t) / 4, page = f.tables.at(g / 16, f.table_rows.at(b));
            for (int c = 0; c < 128; ++c) {
                const auto idx = f.compressed.index(c, g % 16, 0, page);
                ik_error.add(actual_ik[idx], ideal.values[c], ideal.scales[c]);
                expected_ik[idx] = actual_ik[idx];
            }
            pending.clear();
        }
    }
    failures += q_error.check(label + " Q") + k_error.check(label + " K") +
                iq_error.check(label + " IQ") + ik_error.check(label + " IK");
    failures += verify_exact("K untouched", actual_k, expected_k);
    failures += verify_exact("IK untouched", actual_ik, expected_ik);
    failures += verify_exact("V words", f.values.download(), expected_v);
    failures += verify_exact("raw records", f.records.download(), expected_raw);
    failures += verify_exact("coordinate records", f.record_rope.download(), expected_rope);
    if (!(f.alias & 1)) failures += f.q.preserved("raw Q");
    if (!(f.alias & 2)) failures += f.iq.preserved("raw IQ");
    for (auto* x : {&f.k, &f.v, &f.ik, &f.wq, &f.wk, &f.wiq, &f.wik, &f.tail})
        failures += x->preserved("input");
    for (auto* x : {&f.first, &f.positions, &f.rope, &f.valid, &f.table_rows, &f.slots, &f.offsets,
                    &f.tables})
        failures += x->preserved("metadata/state");
    for (auto* x :
         {&f.q, &f.iq, &f.out_q, &f.out_iq, &f.keys, &f.values, &f.compressed, &f.records})
        failures += x->device->verify_guards(label);
    failures += f.record_rope.device->verify_guards(label);
    if (failures) std::cerr << label << ": " << failures << " failures\n";
    return failures;
}

int run(DeviceContext& device, int t, int b, int axes, int alias, int layout, int capacity = 0) {
    Fixture f(t, b, axes, alias, layout, capacity);
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    int failures = 0;
    for (int phase = 0; phase < 3; ++phase) {
        f.reset(phase, phase == 1);
        if (phase == 1) {
            f.parameters.rms_epsilon = 2e-5F;
            f.parameters.rope_theta  = 500000;
        } else {
            f.parameters.rms_epsilon = 1e-6F;
            f.parameters.rope_theta  = 1e7F;
        }
        if (phase == 0) {
            definition.capture(device.stream, [&] { f.launch(device.execution_view()); });
            graph.instantiate(definition);
        }
        if (phase == 1)
            f.launch(device.execution_view());
        else
            graph.launch(device.stream);
        device.synchronize();
        failures +=
            verify(f, "prepare T=" + std::to_string(t) + " B=" + std::to_string(b) +
                          " A=" + std::to_string(axes) + " alias=" + std::to_string(alias) +
                          " layout=" + std::to_string(layout) + " phase=" + std::to_string(phase));
    }
    return failures;
}

int continuation(DeviceContext& device) {
    Fixture f(4, 1, 3, 0, 2);
    f.base_origin = 6;
    f.reset(0, true);
    f.launch(device.execution_view());
    device.synchronize();
    int failures = verify(f, "continuation initial");
    auto base = to_device_i32({6}), count = to_device_i32({1});
    ops::qsa_tail_extract(
        f.before,
        {f.records.view, f.record_rope.view, Tensor(base.p, DType::I32, {1}), f.valid.view},
        Tensor(count.p, DType::I32, {1}), f.before, device.execution_view());
    device.synchronize();
    for (int c = 0; c < 128; ++c) f.tail.at(c, 2, 1) = f.ik.at(c, 0, 0);
    failures += f.tail.preserved("accepted-one canonical tail") +
                f.first.preserved("tail first coordinate");
    f.keys.host          = f.keys.download();
    f.values.host        = f.values.download();
    f.compressed.host    = f.compressed.download();
    f.records.host       = f.records.download();
    f.record_rope.host   = f.record_rope.download();
    f.bases[0]           = 7;
    f.valid.at(0)        = 1;
    f.offsets.at(0)      = 1;
    f.positions.at(0, 0) = 7;
    // Replace rejected token 7 while retaining actual old raw words 4/5/6 from TailExtract.
    for (int c = 0; c < 128; ++c) f.ik.at(c, 0, 0) = sample(7, 0, c, 313);
    for (int a = 0; a < 3; ++a) f.rope.at(0, a, 0) = 7000 + a * 37;
    f.ik.upload();
    for (auto* x : {&f.valid, &f.offsets, &f.positions, &f.rope}) x->upload();
    f.launch(device.execution_view());
    device.synchronize();
    failures += verify(f, "continuation replacement");
    return failures;
}

int chunking(DeviceContext& device) {
    Fixture f(9, 1, 3, 3, 2);
    f.base_origin = 63;
    f.reset(0, true);
    f.launch(device.execution_view());
    device.synchronize();
    int failures = verify(f, "whole chunk");
    const auto q = f.q.download(), iq = f.iq.download(), k = f.keys.download(),
               v = f.values.download(), ik = f.compressed.download(),
               records     = f.records.download();
    const auto coordinates = f.record_rope.download();
    f.reset(0, true);
    int offset = 0;
    auto base  = to_device_i32({63});
    for (int n : {1, 3, 2, 3}) {
        const auto slice = [&](const Tensor& input, int d, int h) {
            return Tensor(static_cast<std::uint16_t*>(input.data) + std::int64_t(offset) * d * h,
                          DType::BF16, {d, h, n, 1});
        };
        auto p          = f.projected;
        p.query         = slice(p.query, 256, 24);
        p.key           = slice(p.key, 256, 2);
        p.value         = slice(p.value, 256, 2);
        p.index_query   = slice(p.index_query, 128, 4);
        p.index_key     = Tensor(static_cast<std::uint16_t*>(p.index_key.data) + offset * 128,
                                 DType::BF16, {128, n, 1});
        f.valid.at(0)   = n;
        f.offsets.at(0) = offset;
        f.valid.upload();
        f.offsets.upload();
        auto batch = f.batch;
        batch.positions =
            Tensor(static_cast<int*>(f.positions.view.data) + offset, DType::I32, {n, 1});
        std::vector<int> coords(n * 3);
        for (int a = 0; a < 3; ++a)
            for (int t = 0; t < n; ++t) coords[t + n * a] = f.rope.at(offset + t, a, 0);
        auto rope    = to_device_i32(coords);
        Tensor q_out = p.query, iq_out = p.index_query;
        ops::qsa_prepare(p, f.parameters, batch, Tensor(rope.p, DType::I32, {n, 3, 1}), f.before,
                         f.cache, f.write, q_out, iq_out, device.execution_view());
        const int frontier = 63 + offset;
        base.copy_from_host(&frontier, 4);
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
        ops::qsa_tail_extract(f.before,
                              {f.records.view.slice(1, offset, n),
                               f.record_rope.view.slice(1, offset, n),
                               Tensor(base.p, DType::I32, {1}), f.valid.view},
                              f.valid.view, f.before, device.execution_view());
        device.synchronize();
        offset += n;
    }
    failures +=
        verify_exact("chunk Q", f.q.download(), q) + verify_exact("chunk IQ", f.iq.download(), iq);
    failures += verify_exact("chunk K", f.keys.download(), k) +
                verify_exact("chunk V", f.values.download(), v);
    failures += verify_exact("chunk IK", f.compressed.download(), ik);
    failures += verify_exact("chunk records", f.records.download(), records);
    failures += verify_exact("chunk coordinates", f.record_rope.download(), coordinates);
    for (int c = 0; c < 128; ++c)
        for (int j = 0; j < 3; ++j) f.tail.at(c, j, 1) = 0;
    for (int a = 0; a < 3; ++a) f.first.at(a, 1) = 0;
    failures += f.tail.preserved("chunk final tail") + f.first.preserved("chunk final coordinates");
    return failures;
}

int numerical_edges(DeviceContext& device, bool wide_phase) {
    Fixture f(wide_phase ? 5 : 2048, 1, 3, 3, 0);
    f.reset(2, true);
    f.parameters.rope_theta = wide_phase ? 0.5F : 1e7F;
    for (int c = 0; c < 256; ++c) {
        f.q.at(c, 0, 0, 0) = 0;
        f.q.at(c, 1, 0, 0) = f32_to_bf16((c % 3 - 1) * 1e20F);
        f.q.at(c, 2, 0, 0) = f32_to_bf16((c % 3 - 1) * 1e-30F);
    }
    f.q.upload();
    f.launch(device.execution_view());
    device.synchronize();
    return verify(f, "normalization/phase edges");
}

int epsilon_range(DeviceContext& device) {
    Fixture f(1, 1, 1);
    f.parameters.rms_epsilon = std::numeric_limits<float>::max();
    for (int c = 0; c < 256; ++c) f.q.at(c, 0, 0, 0) = f32_to_bf16(1e18F);
    f.q.upload();
    f.launch(device.execution_view());
    device.synchronize();
    return verify(f, "finite norm denominator range");
}

int coordinate_handoff(DeviceContext& device) {
    Fixture f(1, 1, 1);
    f.base_origin = 63;
    f.reset(0, true);
    f.first.at(1, 1) = 1048576;
    f.first.at(2, 1) = 2000000000;
    f.first.upload();
    f.launch(device.execution_view());
    device.synchronize();
    return verify(f, "three-axis tail to one-axis current input");
}

int unaligned_norms(DeviceContext& device) {
    Fixture f(129, 1, 1, 3, 1);
    Buffer<std::uint16_t> wq(DType::BF16, {257}), wiq(DType::BF16, {129});
    std::copy(f.wq.host.begin(), f.wq.host.end(), wq.host.begin() + 1);
    std::copy(f.wiq.host.begin(), f.wiq.host.end(), wiq.host.begin() + 1);
    wq.upload();
    wiq.upload();
    f.parameters.query_norm =
        Tensor(static_cast<std::uint16_t*>(wq.view.data) + 1, DType::BF16, {256});
    f.parameters.index_query_norm =
        Tensor(static_cast<std::uint16_t*>(wiq.view.data) + 1, DType::BF16, {128});
    int failures = 0;
    for (float theta : {1e7F, 0.5F}) {
        f.reset(2, true);
        f.parameters.rope_theta = theta;
        f.launch(device.execution_view());
        device.synchronize();
        failures += verify(f, "naturally aligned norms");
    }
    return failures + wq.preserved("shifted Q norm") + wiq.preserved("shifted IQ norm");
}

// Inert projection columns/coordinates are deliberately uninitialized. Initcheck can detect
// a masked read even if subsequent arithmetic discards it and the final output is zero.
int read_footprint(DeviceContext& device) {
    Fixture f(17, 2, 3, 0, 0);
    f.base_origin = 63;
    f.reset(0, true);
    f.valid.at(0)      = 1;
    f.valid.at(1)      = 0;
    f.slots.at(1)      = -1;
    f.table_rows.at(1) = -1;
    f.valid.upload();
    f.slots.upload();
    f.table_rows.upload();
    std::vector<DeviceBuffer> inputs;
    for (auto* view : {&f.projected.query, &f.projected.key, &f.projected.value,
                       &f.projected.index_query, &f.projected.index_key}) {
        const int width = view->ne[0] * (view == &f.projected.index_key ? 1 : view->ne[1]);
        inputs.emplace_back(view->bytes());
        CUDA_CHECK(cudaMemcpyAsync(inputs.back().p, view->data, width * 2, cudaMemcpyDeviceToDevice,
                                   device.stream));
        view->data = inputs.back().p;
    }
    DeviceBuffer pos(f.positions.view.bytes()), rope(f.rope.view.bytes());
    CUDA_CHECK(
        cudaMemcpyAsync(pos.p, f.positions.view.data, 4, cudaMemcpyDeviceToDevice, device.stream));
    for (int a = 0; a < 3; ++a)
        CUDA_CHECK(cudaMemcpyAsync(static_cast<int*>(rope.p) + a * f.T,
                                   static_cast<int*>(f.rope.view.data) + a * f.T, 4,
                                   cudaMemcpyDeviceToDevice, device.stream));
    f.batch.positions.data = pos.p;
    auto saved             = f.rope.view;
    f.rope.view.data       = rope.p;
    f.launch(device.execution_view());
    device.synchronize();
    f.rope.view = saved;
    return verify(f, "masked read footprint");
}

template <class F>
int rejects(const char* label, F&& call) {
    try {
        call();
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << label << ": invalid view accepted\n";
    return 1;
}

int validation(DeviceContext& device) {
    Fixture f(1, 1, 1);
    int failures         = 0;
    auto saved           = f.cache;
    f.cache.main.storage = KvCacheStorage::Nvfp4Group16;
    failures += rejects("unimplemented codec", [&] { f.launch(device.execution_view()); });
    f.cache = saved;
    f.cache.index.block_tables.nb[1] += 4;
    failures += rejects("different table view", [&] { f.launch(device.execution_view()); });
    f.cache                    = saved;
    f.cache.main.k_pages.nb[2] = 256;
    failures += rejects("overlapping cache heads", [&] { f.launch(device.execution_view()); });
    f.cache             = saved;
    auto q              = f.query_output;
    f.query_output.data = static_cast<std::byte*>(f.q.view.data) + 16;
    failures += rejects("partial Q alias", [&] { f.launch(device.execution_view()); });
    f.query_output           = q;
    f.parameters.rms_epsilon = 0;
    failures += rejects("zero epsilon", [&] { f.launch(device.execution_view()); });
    return failures;
}

} // namespace

int main() {
    try {
        if (cuda_unavailable()) return 77;
        DeviceContext device;
        int failures = 0;
        for (int t : {1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 33, 63, 64, 65, 128})
            failures += run(device, t, t % 2 ? 8 : 1, t % 3 ? 3 : 1, t % 4, t % 3);
        for (int t : {319, 320, 321}) failures += run(device, t, 1, 3, 3, 0);
        for (int t : {39, 40, 41}) failures += run(device, t, 8, 1, 0, 2);
        for (int alias = 0; alias < 4; ++alias)
            failures += run(device, 9, 8, 3, alias, alias % 3, 5);
        for (int t : {1024, 2048, 4096}) failures += run(device, t, 1, 3, 3, 0);
        failures += epsilon_range(device) + coordinate_handoff(device) + unaligned_norms(device) +
                    continuation(device) + chunking(device) + numerical_edges(device, true) +
                    numerical_edges(device, false) + read_footprint(device) + validation(device);
        std::cout << "QSA Prepare: " << (failures ? "FAIL" : "PASS") << '\n';
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
