// The artifact's own form: the low-rank matrices and the four injection rows are stored as Q5
// RowSplit planes, and the fold reads them without a dequantized copy. The oracle evaluates the
// complete formula over the values those codes and FP16 scales represent, in FP64.

#include "core/device.h"
#include "ninfer/ops/gated_residual.h"
#include "ops/op_check.h"
#include "ops/op_tester.h"
#include "criteria.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_fp16.h>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::hc;

constexpr int kWidth = 2560, kStreams = 4, kRank = 320, kTotal = kWidth * kStreams;

// The geometry the artifact states. A drift here means every shape below is the wrong one.
static_assert(kStreams == 4 && kRank == 320 && kTotal == 10240);

std::uint16_t to_half_bits(float value) {
    __half h = __float2half_rn(value);
    std::uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

float from_half_bits(std::uint16_t bits) {
    __half h;
    std::memcpy(&h, &bits, sizeof(h));
    return __half2float(h);
}

// One Q5 RowSplit matrix: the three planes exactly as weight_view.cpp lays them out, plus the
// values those codes represent (what the oracle must use).
struct Q5Matrix {
    int rows = 0, columns = 0;
    std::vector<std::uint8_t> codes, high, scales;
    std::vector<float> values; // rows x columns, decoded.
    int groups = 0, code_row = 0, high_row = 0, scale_row = 0, padded_k = 0;
};

// grouped_absmax at 5 bits over 64-wide groups: scale = fp16(absmax / 15), codes round to nearest
// and saturate at [-16, 15]. The test owns the quantizer, so any self-consistent rounding is valid;
// what the kernel must reproduce is the value each stored code and scale represents.
Q5Matrix quantize(int rows, int columns, std::uint32_t seed, float gain) {
    Q5Matrix m;
    m.rows      = rows;
    m.columns   = columns;
    m.padded_k  = (columns + 127) / 128 * 128;
    m.groups    = m.padded_k / 64;
    m.code_row  = m.groups * 32;
    m.high_row  = m.groups * 8;
    m.scale_row = m.groups * 2;
    m.codes.assign(std::size_t(rows) * m.code_row, 0);
    m.high.assign(std::size_t(rows) * m.high_row, 0);
    m.scales.assign(std::size_t(rows) * m.scale_row, 0);
    m.values.assign(std::size_t(rows) * columns, 0.F);
    std::vector<float> dense(std::size_t(rows) * columns);
    fill_uniform(dense, seed, -gain, gain);
    for (int r = 0; r < rows; ++r) {
        for (int g = 0; g < m.groups; ++g) {
            const int begin = g * 64, end = std::min(columns, begin + 64);
            float peak = 0.F;
            for (int k = begin; k < end; ++k)
                peak = std::max(peak, std::abs(dense[std::size_t(r) * columns + k]));
            const std::uint16_t scale_bits = to_half_bits(peak == 0.F ? 0.F : peak / 15.F);
            const float scale              = from_half_bits(scale_bits);
            std::memcpy(&m.scales[std::size_t(r) * m.scale_row + g * 2], &scale_bits, 2);
            int codes[64] = {};
            for (int i = 0; i < 64; ++i) {
                const int k = begin + i;
                if (k >= end || scale == 0.F) continue;
                const int q = int(std::lrint(dense[std::size_t(r) * columns + k] / scale));
                codes[i]    = std::max(-16, std::min(15, q));
            }
            for (int byte = 0; byte < 32; ++byte)
                m.codes[std::size_t(r) * m.code_row + g * 32 + byte] = std::uint8_t(
                    (codes[2 * byte] & 0x0f) | ((codes[2 * byte + 1] & 0x0f) << 4));
            for (int i = 0; i < 64; ++i)
                if ((codes[i] >> 4) & 1)
                    m.high[std::size_t(r) * m.high_row + g * 8 + i / 8] |=
                        std::uint8_t(1u << (i % 8));
            for (int i = 0; i < 64; ++i) {
                const int k = begin + i;
                if (k < end) m.values[std::size_t(r) * columns + k] = float(codes[i]) * scale;
            }
        }
    }
    return m;
}

// One device allocation per matrix, planes at 256-byte offsets like the artifact's layout.
class DeviceMatrix {
    static std::size_t high_at(const Q5Matrix& s) { return (s.codes.size() + 255) / 256 * 256; }
    static std::size_t scale_at(const Q5Matrix& s) { return high_at(s) + (s.high.size() + 255) / 256 * 256; }
    static std::size_t total(const Q5Matrix& s) { return scale_at(s) + s.scales.size(); }

public:
    explicit DeviceMatrix(const Q5Matrix& source) : host(source), buffer_(total(source)) {
        auto* base                 = static_cast<std::uint8_t*>(buffer_.data());
        const std::size_t high_io  = high_at(host);
        const std::size_t scale_io = scale_at(host);
        buffer_.copy_from_host(host.codes.data(), host.codes.size(), 0);
        buffer_.copy_from_host(host.high.data(), host.high.size(), high_io);
        buffer_.copy_from_host(host.scales.data(), host.scales.size(), scale_io);
        weight_.payload          = base;
        weight_.payload_bytes    = host.codes.size();
        weight_.high_plane_bytes = host.high.size();
        weight_.qtype            = QType::Q5_G64_FP16;
        weight_.layout           = QuantLayout::RowSplit;
        weight_.group_size       = 64;
        weight_.group            = 64;
        weight_.ndim             = 2;
        weight_.scale_dtype      = DType::FP16;
        weight_.n = weight_.shape[0] = weight_.padded_shape[0] = host.rows;
        weight_.k = weight_.shape[1] = host.columns;
        weight_.padded_shape[1]      = host.padded_k;
        weight_.qdata                = base;
        weight_.qhigh                = base + high_io;
        weight_.scales               = base + scale_io;
        weight_.scale_ne[0]          = host.groups;
        weight_.scale_ne[1]          = host.rows;
        weight_.scale_nb[0]          = 2;
        weight_.scale_nb[1]          = host.scale_row;
        weight_.scale_nb[2] = weight_.scale_nb[3] = weight_.scale_nb[1] * host.rows;
    }

    const Weight& weight() const { return weight_; }
    int check(const std::string& label) const { return buffer_.verify_guards(label); }

    Q5Matrix host;

private:
    GuardedDeviceBuffer buffer_;
    Weight weight_{};
};

double sigmoid(double x) {
    if (x >= 0) return 1.0 / (1.0 + std::exp(-x));
    const double e = std::exp(x);
    return e / (1.0 + e);
}

double dot(std::span<const float> row, std::span<const float> z) {
    double sum = 0;
    for (std::size_t k = 0; k < z.size(); ++k) sum += double(row[k]) * double(z[k]);
    return sum;
}

struct Column {
    std::vector<double> output;
    std::vector<double> injection;
};

Column reference(const Q5Matrix& down, const Q5Matrix* inject, const Q5Matrix& up,
                std::span<const float> z) {
    std::vector<double> u(kRank);
    for (int r = 0; r < kRank; ++r) {
        const double a = dot(std::span{down.values.data() + std::size_t(r) * down.columns,
                                       std::size_t(down.columns)},
                             z) /
                         4.0;
        u[r] = a * sigmoid(a);
    }
    Column result{std::vector<double>(kWidth), {}};
    for (int s = 0; s < kStreams; ++s)
        for (int d = 0; d < kWidth; ++d) {
            const auto row = s * kWidth + d;
            double g       = 0;
            for (int r = 0; r < kRank; ++r)
                g += double(up.values[std::size_t(row) * kRank + r]) * u[r];
            result.output[d] += sigmoid(g) * double(z[row]) / 4.0;
        }
    if (inject != nullptr) {
        result.injection.resize(kStreams);
        for (int s = 0; s < kStreams; ++s)
            result.injection[s] =
                2.0 * sigmoid(dot(std::span{inject->values.data() + std::size_t(s) * inject->columns,
                                            std::size_t(inject->columns)},
                                  z) /
                              4.0);
    }
    return result;
}

template <class Call>
int rejects(const char* label, Call&& call) {
    try {
        call();
    } catch (const std::invalid_argument&) { return 0; } catch (const std::overflow_error&) {
        return 0;
    }
    std::cerr << "hc q5 mix " << label << ": invalid arguments were accepted\n";
    return 1;
}

// One column of operands, reused by the negative checks.
struct Rig {
    Rig(DeviceExecutionView e, const std::vector<float>& catalog)
        : execution(e), input(std::size_t(kTotal) * 2), output(std::size_t(kWidth) * 2),
          injection(std::size_t(kStreams) * 4), scratch(std::size_t(1) << 20),
          workspace(DeviceSpan{scratch.data(), scratch.bytes()}),
          z(input.data(), DType::BF16, {kWidth, kStreams, 1}),
          x(output.data(), DType::BF16, {kWidth, 1}),
          j(injection.data(), DType::FP32, {kStreams, 1}) {
        std::vector<std::uint16_t> words;
        words.resize(static_cast<std::size_t>(kTotal));
        for (int d = 0; d < kTotal; ++d)
            words[static_cast<std::size_t>(d)] = f32_to_bf16(catalog[static_cast<std::size_t>(d)]);
        input.copy_from_host(words.data(), input.bytes());
        cuda_check(cudaDeviceSynchronize(), "upload completion");
        cuda_check(cudaMemsetAsync(scratch.data(), 0xa5, scratch.bytes(), execution.stream),
                   "poison workspace");
    }

    void run(const Weight& down, const Weight& inject, const Weight& up) {
        ops::hc_mix(z, ops::HcSeparateMixWeights{down, inject, up}, workspace, x, j, execution);
    }

    DeviceExecutionView execution;
    GuardedDeviceBuffer input, output, injection, scratch;
    WorkspaceArena workspace;
    Tensor z, x, j;
};

int forge_checks(const DeviceMatrix& down, const DeviceMatrix& inject, const DeviceMatrix& up,
                 const std::vector<float>& catalog, DeviceExecutionView execution) {
    const auto d = down.weight(), i = inject.weight(), u = up.weight();
    Rig rig(execution, catalog);
    int failed = 0;
    {
        auto bad     = d;
        bad.qtype    = QType::BF16;
        failed += rejects("down format", [&] { rig.run(bad, i, u); });
    }
    {
        auto bad      = u;
        bad.layout    = QuantLayout::Contiguous;
        failed += rejects("up layout", [&] { rig.run(d, i, bad); });
    }
    {
        auto bad = i;
        bad.n    = kStreams + 1;
        failed += rejects("inject rows", [&] { rig.run(d, bad, u); });
    }
    {
        auto bad = d;
        bad.k    = kTotal + 64;
        failed += rejects("down k", [&] { rig.run(bad, i, u); });
    }
    {
        // Padded K must be a whole number of 128-wide super-blocks.
        auto bad               = u;
        bad.padded_shape[1]    = 320;
        failed += rejects("up padded k", [&] { rig.run(d, i, bad); });
    }
    {
        auto bad  = d;
        bad.qhigh = nullptr;
        failed += rejects("down high plane", [&] { rig.run(bad, i, u); });
    }
    {
        auto bad   = d;
        bad.qdata  = static_cast<const std::uint8_t*>(bad.qdata) + 2;
        failed += rejects("down alignment", [&] { rig.run(bad, i, u); });
    }
    {
        auto bad          = d;
        bad.group_size    = 32;
        failed += rejects("down group", [&] { rig.run(bad, i, u); });
    }
    {
        auto bad        = u;
        bad.payload_bytes = 0;
        failed += rejects("up payload", [&] { rig.run(d, i, bad); });
    }
    // The capacity query answers the same refusals, which is what keeps planning honest.
    {
        auto bad = d;
        bad.qtype = QType::Q8_G32_FP16;
        failed += rejects("down query format", [&] {
            (void)ops::hc_mix_workspace_capacity_bytes(ops::HcSeparateMixWeights{bad, i, u}, 1, 4,
                                                      execution);
        });
    }
    failed += rejects("query interval", [&] {
        (void)ops::hc_mix_workspace_capacity_bytes(ops::HcSeparateMixWeights{d, i, u}, 4, 1,
                                                  execution);
    });
    return failed;
}

int run_case(const DeviceMatrix& down, const DeviceMatrix* inject, const DeviceMatrix& up,
             const std::vector<float>& catalog, const std::vector<Column>& expected, int columns,
             DeviceExecutionView execution, bool block) {
    const std::size_t capacity =
        block ? ops::hc_mix_workspace_capacity_bytes(
                    ops::HcSeparateMixWeights{down.weight(), inject->weight(), up.weight()}, columns,
                    columns, execution)
              : ops::hc_mix_workspace_capacity_bytes(
                    ops::HcFinalSeparateMixWeights{down.weight(), up.weight()}, columns, columns,
                    execution);
    GuardedDeviceBuffer input(std::size_t(kTotal) * columns * 2);
    GuardedDeviceBuffer output(std::size_t(kWidth) * columns * 2);
    GuardedDeviceBuffer injection(block ? std::size_t(kStreams) * columns * 4 : 0);
    GuardedDeviceBuffer scratch(capacity);
    WorkspaceArena workspace(DeviceSpan{scratch.data(), scratch.bytes()});

    std::vector<std::uint16_t> words(std::size_t(kTotal) * columns);
    for (int m = 0; m < columns; ++m)
        for (int d = 0; d < kTotal; ++d)
            words[std::size_t(m) * kTotal + d] = f32_to_bf16(catalog[std::size_t(m) * kTotal + d]);
    input.copy_from_host(words.data(), input.bytes());
    // The upload lands on the legacy default stream, which a non-blocking execution stream is not
    // ordered against: complete it before any kernel of the call can read the operands.
    cuda_check(cudaDeviceSynchronize(), "upload completion");
    cuda_check(cudaMemsetAsync(output.data(), 0xff, output.bytes(), execution.stream), "poison");
    if (injection.bytes())
        cuda_check(cudaMemsetAsync(injection.data(), 0xff, injection.bytes(), execution.stream),
                   "poison");
    // The guard fill in the buffer constructor lands on the legacy default stream, which a
    // non-blocking execution stream does not order against; poison the workspace on the stream the
    // Op runs on so the operand state is defined before the first kernel reads it.
    cuda_check(cudaMemsetAsync(scratch.data(), 0xa5, scratch.bytes(), execution.stream),
               "poison workspace");

    Tensor z(input.data(), DType::BF16, {kWidth, kStreams, columns});
    Tensor x(output.data(), DType::BF16, {kWidth, columns});
    Tensor j(injection.data(), DType::FP32, {kStreams, columns});
    if (block)
        ops::hc_mix(z, ops::HcSeparateMixWeights{down.weight(), inject->weight(), up.weight()},
                    workspace, x, j, execution);
    else
        ops::hc_mix(z, ops::HcFinalSeparateMixWeights{down.weight(), up.weight()}, workspace, x,
                    execution);
    cuda_synchronize(execution.stream);

    int failed = input.verify_guards("input") + output.verify_guards("output") +
                 injection.verify_guards("injection") + scratch.verify_guards("scratch");
    if (workspace.used() != 0 || workspace.peak_used() != capacity) {
        std::cerr << "q5 mix workspace scope/capacity mismatch: peak=" << workspace.peak_used()
                  << " capacity=" << capacity << '\n';
        ++failed;
    }
    const auto got = from_device_bf16(x.data, std::size_t(kWidth) * columns);
    const auto got_injection =
        from_device<float>(j.data, block ? std::size_t(kStreams) * columns : 0);
    {
        // The identical call repeated must reproduce every output word, and the workspace's
        // rank-wide SiLU activation with it. A disagreement is a synchronization defect rather
        // than a numeric one, and it is never in tolerance.
        const auto activation1 = from_device_bf16(scratch.data(), std::size_t(kRank) * columns);
        cuda_check(cudaMemsetAsync(output.data(), 0xff, output.bytes(), execution.stream), "poison");
        cuda_check(cudaMemsetAsync(scratch.data(), 0xa5, scratch.bytes(), execution.stream),
                   "poison workspace");
        if (block)
            ops::hc_mix(z, ops::HcSeparateMixWeights{down.weight(), inject->weight(), up.weight()},
                        workspace, x, j, execution);
        else
            ops::hc_mix(z, ops::HcFinalSeparateMixWeights{down.weight(), up.weight()}, workspace, x,
                        execution);
        cuda_synchronize(execution.stream);
        const auto again         = from_device_bf16(x.data, std::size_t(kWidth) * columns);
        const auto activation2   = from_device_bf16(scratch.data(), std::size_t(kRank) * columns);
        std::size_t differ       = 0;
        for (std::size_t i = 0; i < got.size(); ++i)
            if (got[i] != again[i]) ++differ;
        for (std::size_t i = 0; i < activation1.size(); ++i)
            if (activation1[i] != activation2[i]) ++differ;
        if (differ != 0) {
            std::cerr << "q5 mix M=" << columns << (block ? " block" : " final") << " repeat: "
                      << differ << " words differ between two identical calls\n";
            ++failed;
        }
    }
    for (int m = 0; m < columns; ++m) {
        const auto base = std::size_t(m) * kWidth;
        const auto tag  = std::string(block ? "block" : "final") + " M=" + std::to_string(columns) +
                         " column=" + std::to_string(m);
        failed += verify_reduction(tag, std::span{got}.subspan(base, kWidth),
                                   expected[std::size_t(m)].output, kMixOutput);
        if (!block) continue;
        std::vector<double> actual(kStreams);
        for (int s = 0; s < kStreams; ++s)
            actual[std::size_t(s)] = double(got_injection[std::size_t(m) * kStreams + s]);
        failed += verify_pointwise(tag + " injection", actual,
                                   expected[std::size_t(m)].injection, kMixInjection);
        for (int s = 0; s < kStreams; ++s)
            if (actual[std::size_t(s)] < 0 || actual[std::size_t(s)] > 2) {
                std::cerr << tag << ": injection outside [0,2]\n";
                ++failed;
            }
    }
    return failed;
}
} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no CUDA device\n";
        return 77;
    }
    DeviceContext device;
    const auto execution = device.execution_view();

    const Q5Matrix down_host   = quantize(kRank, kTotal, 9001U, 0.031F);
    const Q5Matrix inject_host = quantize(kStreams, kTotal, 9002U, 0.062F);
    const Q5Matrix up_host     = quantize(kTotal, kRank, 9003U, 0.125F);
    DeviceMatrix down(down_host), inject(inject_host), up(up_host);
    int failed = down.check("q5 down") + inject.check("q5 inject") + up.check("q5 up");

    // The catalog is BF16 before it reaches the Op, so the oracle sees the same values.
    constexpr int kMaxColumns = 64;
    std::vector<float> catalog(std::size_t(kTotal) * kMaxColumns);
    fill_uniform(catalog, 4242U, -1.F, 1.F);
    round_to_bf16(catalog);

    std::vector<Column> block_ref(kMaxColumns), final_ref(kMaxColumns);
    for (int m = 0; m < kMaxColumns; ++m) {
        const auto z              = std::span{catalog}.subspan(std::size_t(m) * kTotal, kTotal);
        block_ref[std::size_t(m)] = reference(down_host, &inject_host, up_host, z);
        final_ref[std::size_t(m)] = reference(down_host, nullptr, up_host, z);
    }

    failed += forge_checks(down, inject, up, catalog, execution);

    // 51 is the shortest width whose last tile is partial with more than one token: 33 leaves a
    // one-token tail, which the guarded stores already get right.
    for (int columns : {1, 8, 17, 32, 33, 40, 51, 64}) {
        failed += run_case(down, &inject, up, catalog, block_ref, columns, execution, true);
        failed += run_case(down, &inject, up, catalog, final_ref, columns, execution, false);
    }

    std::cout << (failed ? "FAIL " : "OK ") << "hc q5 mix\n";
    return failed ? 1 : 0;
}
