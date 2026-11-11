// The Qwen4Exp PLE layer at the real geometry: a 2560-wide n-gram embedding goes through the
// artifact's single [12800,2560] projection, the per-stream gate and the four-tap convolution that
// folds into the hyper-connection residual, and then the nine-column history advances.
//
// Every comparison consumes the words the Op contract publishes as its BF16 seams (projection,
// gated U, recorded N, updated residual), and the tolerances are the Ops' own suite criteria. A
// composition is never held to a bound tighter than the seams it is built from, and none of these
// numbers was adjusted after a run.

#include "models/qwen4_exp/execution/ple_layer.h"

#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"
#include "ops/ple/conv_fixture.h"
#include "ops/ple/gate_fixture.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::models::qwen4_exp;

using ninfer::test::bf16_to_f32;
using ninfer::test::f32_to_bf16;
using ninfer::test::verify_pointwise;
using ninfer::test::verify_reduction;

// The Ops' own criteria (tests/ops/ple): the gate's normwise bound on U and norm(U), the
// convolution's pointwise bound on Y, and the norm's bound on the following block.
constexpr ninfer::test::ReductionCriterion kGate     = ninfer::test::ple::gate_criterion;
constexpr ninfer::test::PointwiseCriterion kConv     = ninfer::test::ple::conv_criterion;
constexpr ninfer::test::ReductionCriterion kConvNorm = ninfer::test::ple::norm_criterion;
// BF16 Linear's A16 criterion, from tests/ops/linear/test_bf16_a16.cpp.
constexpr ninfer::test::ReductionCriterion kLinearA16{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};

constexpr int kH = int(kPleWidth);
constexpr int kC = int(kPleStreams);
constexpr int kD = int(kPleChannels);
constexpr int kN = int(kPleProjectionRows);
constexpr int kTaps = int(kPleKernelTaps);
// The epsilon the step passes for the block that consumes `normalized`.
constexpr float kNextEpsilon = 1e-6F;

int failures = 0;

void check(bool condition, const std::string& label) {
    if (condition) return;
    std::cerr << label << ": failed\n";
    ++failures;
}

// Flat offsets. The layer's own tensors are contiguous [H,C,T,B]; the projection parent is the
// [12800, T*B] matrix the Linear writes, so a column of it holds all four key blocks then value.
std::size_t own4(int d, int s, int t, int b, int tokens) {
    return std::size_t(d) + std::size_t(kH) * (std::size_t(s) + std::size_t(kC) *
                                               (std::size_t(t) + std::size_t(tokens) * b));
}

std::size_t records3(int c, int t, int b, int tokens) {
    return std::size_t(c) + std::size_t(kD) * (std::size_t(t) + std::size_t(tokens) * b);
}

std::size_t key_flat(int d, int s, int t, int b, int tokens) {
    return std::size_t(d + kH * s) + std::size_t(kN) * (std::size_t(t) + std::size_t(tokens) * b);
}

std::size_t value_flat(int d, int t, int b, int tokens) {
    return std::size_t(kD + d) + std::size_t(kN) * (std::size_t(t) + std::size_t(tokens) * b);
}

// Deterministic BF16 fill (no dependence on the standard RNG's distribution across platforms).
void fill(std::vector<std::uint16_t>& words, std::uint32_t seed, float scale) {
    std::uint32_t state = seed * 2654435761U + 1013904223U;
    for (auto& word : words) {
        state = state * 1664525U + 1013904223U;
        const int centered = int((state >> 9) & 0xffffU) - 32768;
        word = f32_to_bf16(scale * float(centered) / 32768.0F);
    }
}

struct Fixture {
    int tokens = 0, batch = 0, slots = 1;
    std::vector<std::uint16_t> weight, embedding, residual, history, kernel;
    std::vector<std::uint16_t> norm_key, norm_query, norm_conv, next_norm;
    std::vector<int> valid, slot_of_row, accepted;
};

double conv_tap(const Fixture& f, const std::vector<std::uint16_t>& records,
                const std::vector<std::uint16_t>& history_before, int c, int t, int b, int slot) {
    double sum = 0;
    for (int j = 0; j < kTaps; ++j) {
        const int k = t - kPleHistory + 3 * j;
        const std::size_t position =
            k < 0 ? std::size_t(c) + std::size_t(kD) * (std::size_t(kPleHistory + k) +
                                                        std::size_t(kPleHistory) * slot)
                  : records3(c, k, b, f.tokens);
        sum += bf16_to_f32(f.kernel[std::size_t(c) + std::size_t(kD) * j]) *
               bf16_to_f32(k < 0 ? history_before[position] : records[position]);
    }
    return sum;
}

void run_case(const Fixture& f, const char* label) {
    const int tokens = f.tokens, batch = f.batch;
    const int columns = tokens * batch;

    test::direct_bf16_weight::HostWeight host_weight;
    host_weight.n    = kN;
    host_weight.k    = kH;
    host_weight.bits = f.weight;

    DeviceBuffer d_weight(f.weight.size() * 2), d_embedding(f.embedding.size() * 2),
        d_residual(f.residual.size() * 2), d_normalized(f.residual.size() * 2),
        d_history(f.history.size() * 2), d_kernel(f.kernel.size() * 2),
        d_nk(f.norm_key.size() * 2), d_nq(f.norm_query.size() * 2),
        d_nc(f.norm_conv.size() * 2), d_nn(f.next_norm.size() * 2),
        d_valid(f.valid.size() * 4), d_slots(f.slot_of_row.size() * 4),
        d_accepted(f.accepted.size() * 4);

    d_weight.copy_from_host(f.weight.data(), f.weight.size() * 2);
    d_embedding.copy_from_host(f.embedding.data(), f.embedding.size() * 2);
    d_residual.copy_from_host(f.residual.data(), f.residual.size() * 2);
    d_history.copy_from_host(f.history.data(), f.history.size() * 2);
    d_kernel.copy_from_host(f.kernel.data(), f.kernel.size() * 2);
    d_nk.copy_from_host(f.norm_key.data(), f.norm_key.size() * 2);
    d_nq.copy_from_host(f.norm_query.data(), f.norm_query.size() * 2);
    d_nc.copy_from_host(f.norm_conv.data(), f.norm_conv.size() * 2);
    d_nn.copy_from_host(f.next_norm.data(), f.next_norm.size() * 2);
    d_valid.copy_from_host(f.valid.data(), f.valid.size() * 4);
    d_slots.copy_from_host(f.slot_of_row.data(), f.slot_of_row.size() * 4);
    d_accepted.copy_from_host(f.accepted.data(), f.accepted.size() * 4);

    DeviceContext context;
    const auto execution = context.execution_view();
    const auto device    = Device::cuda(0);
    auto bf16 = [&](void* data, std::initializer_list<std::int32_t> shape) {
        return Tensor(data, device, DType::BF16, shape);
    };

    PleLayerWeights weights;
    weights.projection       = host_weight.device_weight(d_weight.p);
    weights.projection.device = device;
    weights.norm_key         = bf16(d_nk.p, {kH, kC});
    weights.norm_query       = bf16(d_nq.p, {kH, kC});
    weights.norm_conv        = bf16(d_nc.p, {kH, kC});
    weights.convolution      = bf16(d_kernel.p, {kD, kTaps});
    weights.rms_epsilon      = 1e-6F;

    PleLayerRuntime runtime(weights, std::uint32_t(tokens), std::uint32_t(batch),
                            std::uint32_t(f.slots));
    // The pool is seeded explicitly here; the model zeroes it at sequence start.
    runtime.history().data = d_history.p;

    Tensor embedding  = bf16(d_embedding.p, {kH, tokens, batch});
    Tensor residual   = bf16(d_residual.p, {kH, kC, tokens, batch});
    Tensor normalized = bf16(d_normalized.p, {kH, kC, tokens, batch});
    Tensor valid      = Tensor(d_valid.p, DType::I32, {batch});
    Tensor slots      = Tensor(d_slots.p, DType::I32, {batch});
    Tensor accepted   = Tensor(d_accepted.p, DType::I32, {batch});
    Tensor next_norm  = bf16(d_nn.p, {kH, kC});

    runtime.apply(embedding, residual, normalized, next_norm, kNextEpsilon, valid, slots, execution);
    context.synchronize();
    const auto projected = test::from_device<std::uint16_t>(runtime.projected().data,
                                                            std::size_t(kN) * columns);
    const auto gated   = test::from_device<std::uint16_t>(runtime.gated().data,
                                                          std::size_t(kD) * columns);
    const auto records = test::from_device<std::uint16_t>(runtime.records().data,
                                                          std::size_t(kD) * columns);
    const auto y       = test::from_device<std::uint16_t>(d_residual.p, f.residual.size());
    const auto z       = test::from_device<std::uint16_t>(d_normalized.p, f.residual.size());
    const auto before  = test::from_device<std::uint16_t>(d_history.p, f.history.size());

    runtime.commit(accepted, valid, slots, execution);
    context.synchronize();
    const auto after = test::from_device<std::uint16_t>(d_history.p, f.history.size());

    // 1. Linear: the projected words against FP64 accumulation over the represented BF16 inputs.
    {
        std::vector<double> got, expected;
        got.reserve(std::size_t(kN) * columns);
        expected.reserve(got.capacity());
        for (int m = 0; m < columns; ++m)
            for (int n = 0; n < kN; ++n) {
                const auto* row = f.weight.data() + std::size_t(n) * kH;
                double sum      = 0;
                for (int k = 0; k < kH; ++k)
                    sum += double(bf16_to_f32(row[k])) *
                           double(bf16_to_f32(f.embedding[std::size_t(k) + std::size_t(kH) * m]));
                got.push_back(bf16_to_f32(projected[std::size_t(n) + std::size_t(kN) * m]));
                expected.push_back(sum);
            }
        failures += verify_reduction(std::string(label) + " projection", got, expected, kLinearA16);
    }

    // 2. Gate: U and norm(U) from the actual projected words (the seam the contract declares).
    {
        std::vector<double> got_u, ref_u, got_n, ref_n;
        std::size_t live = 0;
        for (int b = 0; b < batch; ++b)
            for (int t = 0; t < f.valid[b]; ++t)
                for (int s = 0; s < kC; ++s) {
                    double kk = 0, qq = 0;
                    for (int d = 0; d < kH; ++d) {
                        const double k = bf16_to_f32(projected[key_flat(d, s, t, b, tokens)]);
                        const double q = bf16_to_f32(f.residual[own4(d, s, t, b, tokens)]);
                        kk += k * k;
                        qq += q * q;
                    }
                    const double ik = 1.0 / std::sqrt(kk / kH + weights.rms_epsilon);
                    const double iq = 1.0 / std::sqrt(qq / kH + weights.rms_epsilon);
                    double score    = 0;
                    for (int d = 0; d < kH; ++d) {
                        const double k =
                            bf16_to_f32(projected[key_flat(d, s, t, b, tokens)]) * ik *
                            (1.0 + bf16_to_f32(f.norm_key[std::size_t(d) + std::size_t(kH) * s]));
                        const double q =
                            bf16_to_f32(f.residual[own4(d, s, t, b, tokens)]) * iq *
                            (1.0 + bf16_to_f32(f.norm_query[std::size_t(d) + std::size_t(kH) * s]));
                        score += k * q;
                    }
                    score /= std::sqrt(double(kH));
                    const double sign = double(score > 0) - double(score < 0);
                    const double gate =
                        1.0 / (1.0 + std::exp(-sign * std::sqrt(std::max(std::abs(score), 1e-6))));
                    std::vector<double> u(kH);
                    double uu = 0;
                    for (int d = 0; d < kH; ++d) {
                        const auto position = own4(d, s, t, b, tokens);
                        const double value =
                            bf16_to_f32(projected[value_flat(d, t, b, tokens)]);
                        u[d] = bf16_to_f32(f32_to_bf16(float(gate * value)));
                        uu += u[d] * u[d];
                        got_u.push_back(bf16_to_f32(gated[position]));
                        ref_u.push_back(u[d]);
                    }
                    const double inv = 1.0 / std::sqrt(uu / kH + weights.rms_epsilon);
                    for (int d = 0; d < kH; ++d) {
                        const auto position = own4(d, s, t, b, tokens);
                        const std::uint16_t word = f32_to_bf16(float(
                            u[d] * inv *
                            (1.0 + bf16_to_f32(
                                       f.norm_conv[std::size_t(d) + std::size_t(kH) * s]) )));
                        got_n.push_back(bf16_to_f32(records[records3(d + kH * s, t, b, tokens)]));
                        ref_n.push_back(bf16_to_f32(word));
                    }
                    ++live;
                }
        check(live > 0, std::string(label) + ": the window has live columns");
        failures += verify_reduction(std::string(label) + " gate U", got_u, ref_u, kGate);
        failures += verify_reduction(std::string(label) + " gate norm(actual U)", got_n, ref_n,
                                     kGate);
    }

    // 3. Convolution + residual: Y against the double reference, consuming the actual gated and
    //    recorded words and the pre-commit history.
    {
        std::vector<double> got, expected;
        got.reserve(std::size_t(kD) * columns);
        expected.reserve(got.capacity());
        for (int b = 0; b < batch; ++b)
            for (int t = 0; t < tokens; ++t)
                for (int s = 0; s < kC; ++s)
                    for (int d = 0; d < kH; ++d) {
                        const int c         = d + kH * s;
                        const auto position = own4(d, s, t, b, tokens);
                        const bool active    = t < f.valid[b];
                        double ideal = 0;
                        if (active) {
                            const double tap =
                                conv_tap(f, records, before, c, t, b, f.slot_of_row[b]);
                            ideal = bf16_to_f32(f.residual[position]) +
                                    bf16_to_f32(gated[position]) + tap / (1.0 + std::exp(-tap));
                        }
                        got.push_back(bf16_to_f32(y[position]));
                        expected.push_back(active ? bf16_to_f32(f32_to_bf16(float(ideal))) : 0.0);
                    }
        failures += verify_pointwise(std::string(label) + " residual after PLE", got, expected, kConv);
    }

    // 4. The following block's norm, from the actual updated residual words.
    {
        std::vector<double> got, expected;
        got.reserve(std::size_t(kD) * columns);
        for (int b = 0; b < batch; ++b)
            for (int t = 0; t < tokens; ++t)
                for (int s = 0; s < kC; ++s) {
                    double yy = 0;
                    for (int d = 0; d < kH; ++d) {
                        const double value = bf16_to_f32(y[own4(d, s, t, b, tokens)]);
                        yy += value * value;
                    }
                    const double inv = 1.0 / std::sqrt(yy / kH + kNextEpsilon);
                    for (int d = 0; d < kH; ++d) {
                        const auto position = own4(d, s, t, b, tokens);
                        const std::uint16_t word =
                            t >= f.valid[b]
                                ? 0
                                : f32_to_bf16(float(
                                      bf16_to_f32(y[position]) * inv *
                                      (1.0 + bf16_to_f32(
                                                 f.next_norm[std::size_t(d) + std::size_t(kH) * s]))));
                        got.push_back(bf16_to_f32(z[position]));
                        expected.push_back(bf16_to_f32(word));
                    }
                }
        failures += verify_reduction(std::string(label) + " normalized for the next block", got,
                                     expected, kConvNorm);
    }

    // 5. Columns past each row's live prefix are exact +0 in both outputs.
    {
        std::size_t nonzero = 0;
        for (int b = 0; b < batch; ++b)
            for (int t = f.valid[b]; t < tokens; ++t)
                for (int s = 0; s < kC; ++s)
                    for (int d = 0; d < kH; ++d) {
                        const auto position = own4(d, s, t, b, tokens);
                        nonzero += (y[position] != 0) + (z[position] != 0);
                    }
        check(nonzero == 0, std::string(label) + ": masked columns are exact +0");
    }

    // 6. The history pool advanced by each row's accepted prefix, word for word.
    {
        std::size_t mismatch = 0;
        for (int b = 0; b < batch; ++b) {
            const int slot = f.slot_of_row[b];
            for (int j = 0; j < kPleHistory; ++j) {
                const int k = f.accepted[b] - kPleHistory + j;
                for (int c = 0; c < kD; ++c) {
                    const auto position = std::size_t(c) + std::size_t(kD) *
                                                              (std::size_t(j) +
                                                               std::size_t(kPleHistory) * slot);
                    const std::uint16_t source =
                        k < 0 ? before[std::size_t(c) +
                                       std::size_t(kD) * (std::size_t(kPleHistory + k) +
                                                          std::size_t(kPleHistory) * slot)]
                              : records[records3(c, k, b, tokens)];
                    mismatch += after[position] != source;
                }
            }
        }
        check(mismatch == 0, std::string(label) + ": commit advances the history pool exactly");
    }

    check(runtime.device_bytes() > 0, std::string(label) + ": device footprint is reported");
}

Fixture make_fixture(int tokens, int batch, int slots, const std::vector<int>& slot_of_row,
                     const std::vector<int>& valid, const std::vector<int>& accepted,
                     std::uint32_t seed) {
    Fixture f;
    f.tokens      = tokens;
    f.batch       = batch;
    f.slots       = slots;
    f.valid       = valid;
    f.slot_of_row = slot_of_row;
    f.accepted    = accepted;
    f.weight.resize(std::size_t(kN) * kH);
    f.embedding.resize(std::size_t(kH) * tokens * batch);
    f.residual.resize(std::size_t(kD) * tokens * batch);
    f.history.resize(std::size_t(kD) * kPleHistory * slots);
    f.kernel.resize(std::size_t(kD) * kTaps);
    f.norm_key.resize(std::size_t(kH) * kC);
    f.norm_query.resize(std::size_t(kH) * kC);
    f.norm_conv.resize(std::size_t(kH) * kC);
    f.next_norm.resize(std::size_t(kH) * kC);
    fill(f.weight, seed, 0.05F);
    fill(f.embedding, seed + 1, 1.0F);
    fill(f.residual, seed + 2, 1.0F);
    fill(f.history, seed + 3, 1.0F);
    fill(f.kernel, seed + 4, 0.5F);
    fill(f.norm_key, seed + 5, 0.05F);
    fill(f.norm_query, seed + 6, 0.05F);
    fill(f.norm_conv, seed + 7, 0.05F);
    fill(f.next_norm, seed + 8, 0.05F);
    return f;
}

// A layer whose operands do not match the Op contract is refused by name, not run.
void check_geometry_guard() {
    Fixture f = make_fixture(1, 1, 1, {0}, {1}, {1}, 11);
    test::direct_bf16_weight::HostWeight host_weight;
    host_weight.n    = kN;
    host_weight.k    = kH;
    host_weight.bits = f.weight;
    DeviceBuffer d_weight(f.weight.size() * 2), d_kernel(f.kernel.size() * 2),
        d_norm(f.norm_key.size() * 2);
    d_weight.copy_from_host(f.weight.data(), f.weight.size() * 2);
    d_kernel.copy_from_host(f.kernel.data(), f.kernel.size() * 2);
    d_norm.copy_from_host(f.norm_key.data(), f.norm_key.size() * 2);
    const auto device = Device::cuda(0);
    PleLayerWeights weights;
    weights.projection       = host_weight.device_weight(d_weight.p);
    weights.projection.device = device;
    weights.norm_key         = Tensor(d_norm.p, device, DType::BF16, {kH, kC});
    weights.norm_query       = weights.norm_key;
    weights.norm_conv        = weights.norm_key;
    weights.convolution      = Tensor(d_kernel.p, device, DType::BF16, {kD, kTaps});

    {
        PleLayerWeights short_kernel = weights;
        short_kernel.convolution = Tensor(d_kernel.p, device, DType::BF16, {kD, kTaps - 1});
        bool refused = false;
        try {
            PleLayerRuntime runtime(short_kernel, 1, 1, 1);
        } catch (const std::invalid_argument&) { refused = true; }
        check(refused, "a three-tap convolution is refused");
    }
    {
        PleLayerWeights wrong_projection = weights;
        wrong_projection.projection.n    = kN - kH;
        bool refused = false;
        try {
            PleLayerRuntime runtime(wrong_projection, 1, 1, 1);
        } catch (const std::invalid_argument&) { refused = true; }
        check(refused, "a projection that is not [H*C+H,H] is refused");
    }
    {
        bool admitted = true;
        try {
            PleLayerRuntime runtime(weights, 1, 1, 1);
        } catch (const std::exception& error) {
            admitted = false;
            std::cerr << "the contract geometry was rejected: " << error.what() << '\n';
        }
        check(admitted, "the contract geometry is admitted");
    }
}

} // namespace

int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::cout << "SKIP PLE layer (no CUDA device)\n";
            return 77;
        }
        check_geometry_guard();
        // Prefill-shaped: two columns, one row, both live.
        run_case(make_fixture(2, 1, 1, {0}, {2}, {1}, 101), "T2B1");
        // Decode-shaped, two sequences with their own slots and live prefixes: row 1 rejects its
        // only column (a = 0), so its pooled history must come back byte for byte.
        run_case(make_fixture(1, 2, 3, {2, 0}, {1, 1}, {1, 0}, 211), "T1B2");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    if (failures) return 1;
    std::cout << "OK PLE layer: projection, gate, convolution, residual and history at 2560x4\n";
    return 0;
}
