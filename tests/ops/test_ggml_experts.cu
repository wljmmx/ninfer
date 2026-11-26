// The GGML routed-SwiGLU core against a host reference.
//
// The kernel (ops/moe/ggml/ggml_experts.cu) walks a token's routes serially: gate and up per row, the
// SiLU product rounded to BF16, then the down matvec accumulated with the route coefficient. The
// reference here does exactly that in double precision via core/ggml_dequant.h, so the check covers
// the three-bank addressing, the intermediate rounding and the coefficient epilogue - the three places
// a routed-expert path goes wrong.
//
// Two encoding combinations are covered, both taken from the packed artifact's own layers:
//   IQ3_S gate/up with IQ4_NL down, and IQ2_XXS gate/up with Q2_0 down.

#include "core/arena.h"
#include "core/device.h"
#include "core/ggml_dequant.h"
#include "core/ggml_dequant_fixture.h"
#include "ops/moe/ggml/ggml_experts.h"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string_view>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* what, const char* detail) {
    if (condition) { return; }
    std::fprintf(stderr, "FAIL [%s] %s\n", what, detail);
    ++failures;
}

using HostDecode = void (*)(const std::uint8_t*, std::span<float>);

struct Encoding {
    const char* name;
    ninfer::QType format;
    std::uint32_t block;
    std::uint32_t type_bytes;
    HostDecode decode;
};

constexpr Encoding kIq3s{"IQ3_S", ninfer::QType::GGML_IQ3_S, 256, 110, ninfer::core::decode_ggml_iq3_s};
constexpr Encoding kIq4nl{"IQ4_NL", ninfer::QType::GGML_IQ4_NL, 32, 18, ninfer::core::decode_ggml_iq4_nl};
constexpr Encoding kIq2xxs{"IQ2_XXS", ninfer::QType::GGML_IQ2_XXS, 256, 66,
                           ninfer::core::decode_ggml_iq2_xxs};
constexpr Encoding kQ20{"Q2_0", ninfer::QType::GGML_Q2_0, 64, 18, ninfer::core::decode_ggml_q2_0};

const ninfer::test::GgmlFixture* fixture_for(const char* name) {
    for (const auto& fixture : ninfer::test::kGgmlFixtures) {
        if (std::string_view(fixture.encoding) == name) { return &fixture; }
    }
    return nullptr;
}

// A row of `width` values for one encoding, built by tiling whole blocks from the fixture with a
// per-row starting block. Rotating *bytes* would also vary the rows, but it rewrites each block's
// scale field into an arbitrary exponent, so the row would no longer be a block stream any quantiser
// could produce - and then a large disagreement says nothing about the kernel.
void fill_row(std::vector<std::uint8_t>& out, const Encoding& encoding,
              const ninfer::test::GgmlFixture& fixture, int row, int width) {
    const int type_bytes  = int(encoding.type_bytes);
    const int blocks      = width / int(encoding.block);
    const int fixture_raw = fixture.bytes / type_bytes;
    const int row_bytes   = blocks * type_bytes;
    out.resize(std::size_t(row_bytes));
    for (int b = 0; b < blocks; ++b) {
        const int source = ((b + row) % fixture_raw) * type_bytes;
        for (int i = 0; i < type_bytes; ++i) {
            out[std::size_t(b) * type_bytes + i] = fixture.data[source + i];
        }
    }
}

std::vector<float> decode_row(const std::vector<std::uint8_t>& payload, const Encoding& encoding,
                              int width) {
    std::vector<float> values(std::size_t(width), 0.0f);
    ninfer::core::decode_ggml(payload.data(), std::size_t(width), encoding.block,
                              encoding.type_bytes, encoding.decode, values);
    return values;
}

float bf16_to_float(std::uint16_t bits) {
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&bits));
}

struct Case {
    const char* name;
    Encoding gate;
    Encoding up;
    Encoding down;
};

constexpr int kExperts = 4;
constexpr int kTopK    = 2;
constexpr int kTokens  = 2;
constexpr int kHidden  = 512;
constexpr int kInter   = 256;

} // namespace

int main() {
    ninfer::DeviceContext context(0);
    context.bind_to_current_thread();
    cudaStream_t stream = context.stream;

    const Case cases[] = {
        {"IQ3_S/IQ3_S/IQ4_NL", kIq3s, kIq3s, kIq4nl},
        {"IQ2_XXS/IQ2_XXS/Q2_0", kIq2xxs, kIq2xxs, kQ20},
    };

    for (const auto& test : cases) {
        const auto* gate_fixture = fixture_for(test.gate.name);
        const auto* up_fixture   = fixture_for(test.up.name);
        const auto* down_fixture = fixture_for(test.down.name);
        check(gate_fixture != nullptr && up_fixture != nullptr && down_fixture != nullptr, test.name,
              "a fixture is missing");
        if (gate_fixture == nullptr || up_fixture == nullptr || down_fixture == nullptr) { continue; }

        // Banks: gate/up hold experts * I rows of H; down holds experts * H rows of I.
        const int gate_rows = kExperts * kInter;
        const int down_rows = kExperts * kHidden;
        std::vector<std::uint8_t> gate_payload, up_payload, down_payload, row;
        for (int r = 0; r < gate_rows; ++r) {
            fill_row(row, test.gate, *gate_fixture, r, kHidden);
            gate_payload.insert(gate_payload.end(), row.begin(), row.end());
        }
        for (int r = 0; r < gate_rows; ++r) {
            fill_row(row, test.up, *up_fixture, r + 1, kHidden);
            up_payload.insert(up_payload.end(), row.begin(), row.end());
        }
        for (int r = 0; r < down_rows; ++r) {
            fill_row(row, test.down, *down_fixture, r + 2, kInter);
            down_payload.insert(down_payload.end(), row.begin(), row.end());
        }

        // Deterministic activation and routes.
        std::vector<float> activation(std::size_t(kHidden) * kTokens, 0.0f);
        std::vector<std::uint16_t> activation_bf16(activation.size());
        for (int t = 0; t < kTokens; ++t) {
            for (int i = 0; i < kHidden; ++i) {
                const float v = 0.35f * std::sin(0.021f * float(i + 1) + 1.3f * float(t));
                const std::uint16_t bits = ninfer::core::ggml_float_to_bf16(v);
                activation_bf16[std::size_t(t) * kHidden + i] = bits;
                activation[std::size_t(t) * kHidden + i]     = bf16_to_float(bits);
            }
        }
        std::vector<std::int32_t> ids(std::size_t(kTopK) * kTokens);
        std::vector<float> coefficients(std::size_t(kTopK) * kTokens);
        for (int t = 0; t < kTokens; ++t) {
            for (int r = 0; r < kTopK; ++r) {
                ids[std::size_t(t) * kTopK + r] = (t * 2 + r * 3) % kExperts;
                coefficients[std::size_t(t) * kTopK + r] = 0.75f / float(r + 1);
            }
        }

        // Host reference, in double, with the same BF16 rounding of the SiLU product.
        std::vector<double> expected(std::size_t(kHidden) * kTokens, 0.0);
        std::vector<std::uint8_t> raw;
        for (int t = 0; t < kTokens; ++t) {
            std::vector<float> products(std::size_t(kTopK) * kInter, 0.0f);
            for (int r = 0; r < kTopK; ++r) {
                const int expert = ids[std::size_t(t) * kTopK + r];
                for (int i = 0; i < kInter; ++i) {
                    const int row_index = expert * kInter + i;
                    const int gate_offset =
                        row_index * (kHidden / int(test.gate.block) * int(test.gate.type_bytes));
                    const int up_offset =
                        row_index * (kHidden / int(test.up.block) * int(test.up.type_bytes));
                    const int gate_bytes =
                        kHidden / int(test.gate.block) * int(test.gate.type_bytes);
                    const int up_bytes = kHidden / int(test.up.block) * int(test.up.type_bytes);
                    raw.assign(gate_payload.begin() + gate_offset,
                               gate_payload.begin() + gate_offset + gate_bytes);
                    const auto g = decode_row(raw, test.gate, kHidden);
                    raw.assign(up_payload.begin() + up_offset,
                               up_payload.begin() + up_offset + up_bytes);
                    const auto u = decode_row(raw, test.up, kHidden);
                    double acc_g = 0.0, acc_u = 0.0;
                    for (int k = 0; k < kHidden; ++k) {
                        const double xv = activation[std::size_t(t) * kHidden + k];
                        acc_g += double(g[std::size_t(k)]) * xv;
                        acc_u += double(u[std::size_t(k)]) * xv;
                    }
                    const double silu = acc_g / (1.0 + std::exp(-acc_g));
                    products[std::size_t(r) * kInter + i] =
                        bf16_to_float(ninfer::core::ggml_float_to_bf16(float(silu * acc_u)));
                }
            }
            for (int drow = 0; drow < kHidden; ++drow) {
                double acc = 0.0;
                for (int r = 0; r < kTopK; ++r) {
                    const int expert    = ids[std::size_t(t) * kTopK + r];
                    const int row_index = expert * kHidden + drow;
                    const int offset =
                        row_index * (kInter / int(test.down.block) * int(test.down.type_bytes));
                    const int down_bytes =
                        kInter / int(test.down.block) * int(test.down.type_bytes);
                    raw.assign(down_payload.begin() + offset,
                               down_payload.begin() + offset + down_bytes);
                    const auto w = decode_row(raw, test.down, kInter);
                    double dot   = 0.0;
                    for (int i = 0; i < kInter; ++i) {
                        dot += double(w[std::size_t(i)]) *
                               double(products[std::size_t(r) * kInter + i]);
                    }
                    acc += double(coefficients[std::size_t(t) * kTopK + r]) * dot;
                }
                expected[std::size_t(t) * kHidden + drow] = acc;
            }
        }

        ninfer::DeviceBuffer gate_on_device(gate_payload.size());
        ninfer::DeviceBuffer up_on_device(up_payload.size());
        ninfer::DeviceBuffer down_on_device(down_payload.size());
        ninfer::DeviceBuffer x_on_device(activation_bf16.size() * 2);
        ninfer::DeviceBuffer ids_on_device(ids.size() * 4);
        ninfer::DeviceBuffer coef_on_device(coefficients.size() * 4);
        ninfer::DeviceBuffer out_on_device(std::size_t(kHidden) * kTokens * 2);
        gate_on_device.copy_from_host(gate_payload.data(), gate_payload.size());
        up_on_device.copy_from_host(up_payload.data(), up_payload.size());
        down_on_device.copy_from_host(down_payload.data(), down_payload.size());
        x_on_device.copy_from_host(activation_bf16.data(), activation_bf16.size() * 2);
        ids_on_device.copy_from_host(ids.data(), ids.size() * 4);
        coef_on_device.copy_from_host(coefficients.data(), coefficients.size() * 4);
        out_on_device.fill(0);

        const auto make_weight = [](ninfer::DeviceBuffer& buffer, ninfer::QType qtype, int rows,
                                    int k) {
            ninfer::Weight weight{};
            weight.qdata           = buffer.p;
            weight.payload_bytes   = buffer.bytes;
            weight.qtype           = qtype;
            weight.layout          = ninfer::QuantLayout::GgmlBlock;
            weight.n               = rows;
            weight.k               = k;
            weight.shape[0]        = rows;
            weight.shape[1]        = k;
            weight.padded_shape[0] = rows;
            weight.padded_shape[1] = k;
            weight.ndim            = 2;
            return weight;
        };
        const auto gate_weight = make_weight(gate_on_device, test.gate.format, gate_rows, kHidden);
        const auto up_weight   = make_weight(up_on_device, test.up.format, gate_rows, kHidden);
        const auto down_weight = make_weight(down_on_device, test.down.format, down_rows, kInter);
        // The test's banks are one dense run of experts, so the view's per-expert stride is that
        // run's encoded width; `bank_from` derives the row width from K and the encoding.
        const auto bank_view = [](const ninfer::Weight& weight, int per_expert_rows, int row_bytes) {
            ninfer::ExpertMatrixDeviceView view{};
            view.codes       = static_cast<const std::byte*>(weight.qdata);
            view.code_stride = std::uint64_t(per_expert_rows) * std::uint64_t(row_bytes);
            return view;
        };
        const int gate_row_bytes = kHidden / int(test.gate.block) * int(test.gate.type_bytes);
        const int inter_row_bytes = kInter / int(test.down.block) * int(test.down.type_bytes);
        const ninfer::ops::detail::ggml::ExpertBanks banks{
            ninfer::ops::detail::ggml::bank_from(
                bank_view(gate_weight, kInter, gate_row_bytes), kHidden, test.gate.format),
            ninfer::ops::detail::ggml::bank_from(
                bank_view(up_weight, kInter, gate_row_bytes), kHidden, test.up.format),
            ninfer::ops::detail::ggml::bank_from(
                bank_view(down_weight, kHidden, inter_row_bytes), kInter, test.down.format)};

        ninfer::Tensor x{};
        x.data  = x_on_device.p;
        x.dtype = ninfer::DType::BF16;
        x.ne[0] = kHidden;
        x.ne[1] = kTokens;
        x.nb[0] = 2;
        x.nb[1] = std::int64_t(kHidden) * 2;

        ninfer::Tensor id_tensor{};
        id_tensor.data  = ids_on_device.p;
        id_tensor.dtype = ninfer::DType::I32;
        id_tensor.ne[0] = kTopK;
        id_tensor.ne[1] = kTokens;
        id_tensor.nb[0] = 4;
        id_tensor.nb[1] = std::int64_t(kTopK) * 4;

        ninfer::Tensor coef_tensor{};
        coef_tensor.data  = coef_on_device.p;
        coef_tensor.dtype = ninfer::DType::FP32;
        coef_tensor.ne[0] = kTopK;
        coef_tensor.ne[1] = kTokens;
        coef_tensor.nb[0] = 4;
        coef_tensor.nb[1] = std::int64_t(kTopK) * 4;

        ninfer::Tensor out{};
        out.data  = out_on_device.p;
        out.dtype = ninfer::DType::BF16;
        out.ne[0] = kHidden;
        out.ne[1] = kTokens;
        out.nb[0] = 2;
        out.nb[1] = std::int64_t(kHidden) * 2;

        ninfer::DeviceBuffer scratch_buffer(
            ninfer::ops::detail::ggml::routed_swiglu_workspace_bytes(kTopK, kInter) + 512);
        ninfer::MemoryArena workspace(
            ninfer::MemorySpan{scratch_buffer.p, scratch_buffer.bytes, ninfer::Device::cuda(0)});
        const ninfer::ops::ExpertSubset subset{};  // All: every routed expert participates
        ninfer::ops::detail::ggml::launch_routed_swiglu(
            x, banks, kHidden, kInter, kExperts, kTopK, subset, ninfer::ops::MoeEpilogue::Write,
            id_tensor, coef_tensor, nullptr, out, workspace, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<std::uint16_t> got(std::size_t(kHidden) * kTokens, 0);
        out_on_device.copy_to_host(got.data(), got.size() * 2);

        double worst = 0;
        int worst_at = -1;
        for (std::size_t i = 0; i < got.size(); ++i) {
            const double value = bf16_to_float(got[i]);
            const double want  = expected[i];
            const double bound = std::max(2e-3, (2.0 / 256.0) * std::fabs(want));
            if (std::fabs(value - want) > bound) {
                const double rel = std::fabs(want) > 0 ? std::fabs(value - want) / std::fabs(want) : 0;
                if (rel > worst) {
                    worst    = rel;
                    worst_at = int(i);
                }
            }
        }
        check(worst == 0, test.name, "routed swiglu disagrees with the host reference");
        if (worst != 0 && worst_at >= 0) {
            std::fprintf(stderr, "      y[%d]: kernel %.9g, host %.9g (rel %.3g)\n", worst_at,
                         bf16_to_float(got[std::size_t(worst_at)]), expected[std::size_t(worst_at)],
                         worst);
        }
        std::printf("%-20s experts=%d top_k=%d tokens=%d H=%d I=%d\n", test.name, kExperts, kTopK,
                    kTokens, kHidden, kInter);
    }

    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("ggml routed swiglu: both encoding combinations match the host reference\n");
    return 0;
}
