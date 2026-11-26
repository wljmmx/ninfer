// The GGML Linear kernel against the Host reference it is supposed to implement.
//
// The kernel reads a row's stored blocks and accumulates in registers (ops/linear/ggml/ggml_gemv.cuh);
// the reference decodes the same bytes with src/core/ggml_dequant.h (itself pinned to gguf-py) and
// dots in binary32. Agreement is therefore evidence about the kernel, not about one code path
// agreeing with itself.
//
// Geometry: K is two blocks wide so every decoder sees a run of blocks rather than one; two activation
// columns so the per-column walk is exercised; three rows so the warp-per-row launch has to stop at
// the row count. Rows share the same tiled payload - the arithmetic is what is under test here, and
// the fixture's per-block values are distinctive enough that a wrong block offset shows up at once.

#include "core/arena.h"
#include "core/device.h"
#include "core/ggml_dequant.h"
#include "core/ggml_dequant_fixture.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/ggml/ggml_linear.h"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdio>
#include <string_view>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* encoding, const char* what) {
    if (condition) { return; }
    std::fprintf(stderr, "FAIL [%s] %s\n", encoding, what);
    ++failures;
}

using HostDecode = void (*)(const std::uint8_t*, std::span<float>);

struct Decoder {
    const char* encoding;
    ninfer::QType format;
    std::uint32_t block;
    std::uint32_t type_bytes;
    HostDecode host;
};

// Kept beside the fixture order so a name that means a different encoding is a visible edit; the
// block geometry is re-checked against the fixture below.
constexpr Decoder kDecoders[] = {
    {"Q4_K", ninfer::QType::GGML_Q4_K, 256, 144, ninfer::core::decode_ggml_q4_k},
    {"Q5_K", ninfer::QType::GGML_Q5_K, 256, 176, ninfer::core::decode_ggml_q5_k},
    {"Q6_K", ninfer::QType::GGML_Q6_K, 256, 210, ninfer::core::decode_ggml_q6_k},
    {"Q2_0", ninfer::QType::GGML_Q2_0, 64, 18, ninfer::core::decode_ggml_q2_0},
    {"IQ4_NL", ninfer::QType::GGML_IQ4_NL, 32, 18, ninfer::core::decode_ggml_iq4_nl},
    {"IQ4_XS", ninfer::QType::GGML_IQ4_XS, 256, 136, ninfer::core::decode_ggml_iq4_xs},
    {"IQ2_XXS", ninfer::QType::GGML_IQ2_XXS, 256, 66, ninfer::core::decode_ggml_iq2_xxs},
    {"IQ2_XS", ninfer::QType::GGML_IQ2_XS, 256, 74, ninfer::core::decode_ggml_iq2_xs},
    {"IQ2_S", ninfer::QType::GGML_IQ2_S, 256, 82, ninfer::core::decode_ggml_iq2_s},
    {"IQ3_XXS", ninfer::QType::GGML_IQ3_XXS, 256, 98, ninfer::core::decode_ggml_iq3_xxs},
    {"IQ3_S", ninfer::QType::GGML_IQ3_S, 256, 110, ninfer::core::decode_ggml_iq3_s},
};

const ninfer::test::GgmlFixture* fixture_for(const char* encoding) {
    for (const auto& fixture : ninfer::test::kGgmlFixtures) {
        if (std::string_view(fixture.encoding) == encoding) { return &fixture; }
    }
    return nullptr;
}

float bf16_to_float(std::uint16_t bits) {
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&bits));
}

std::uint16_t float_to_bf16(float value) { return ninfer::core::ggml_float_to_bf16(value); }

// Deterministic activation: every value distinct, small enough that the float sum stays far inside
// the range where a relative bound means something.
std::vector<float> make_activation(int k, int columns) {
    std::vector<float> x(std::size_t(k) * columns, 0.0f);
    for (int t = 0; t < columns; ++t) {
        for (int i = 0; i < k; ++i) {
            x[std::size_t(t) * k + i] = 0.5f * std::sin(0.037f * float(i + 1) + 0.71f * float(t));
        }
    }
    return x;
}

constexpr int kColumns = 2;
constexpr int kRows    = 3;

constexpr double kRelativeTolerance = 2e-3;
constexpr double kAbsoluteFloor     = 1e-6;

} // namespace

int main() {
    ninfer::DeviceContext context(0);
    context.bind_to_current_thread();
    cudaStream_t stream = context.stream;

    for (const auto& decoder : kDecoders) {
        const auto* fixture = fixture_for(decoder.encoding);
        check(fixture != nullptr, decoder.encoding, "no fixture for this encoding");
        if (fixture == nullptr) { continue; }
        check(fixture->block == decoder.block && fixture->type_bytes == decoder.type_bytes,
              decoder.encoding, "fixture block geometry disagrees with the decoder entry");

        const int k         = int(decoder.block) * 2;
        const int row_bytes = 2 * int(decoder.type_bytes);
        std::vector<std::uint8_t> payload(std::size_t(row_bytes) * kRows);
        for (std::size_t i = 0; i < payload.size(); ++i) {
            payload[i] = fixture->data[i % fixture->bytes];
        }

        // Activation as the kernel receives it: BF16, and the reference sees the same rounded values.
        auto activation = make_activation(k, kColumns);
        std::vector<std::uint16_t> activation_bf16(activation.size());
        for (std::size_t i = 0; i < activation.size(); ++i) {
            activation_bf16[i] = float_to_bf16(activation[i]);
            activation[i]      = bf16_to_float(activation_bf16[i]);
        }

        std::vector<float> reference(std::size_t(kRows) * kColumns, 0.0f);
        std::vector<float> weights(std::size_t(k), 0.0f);
        for (int row = 0; row < kRows; ++row) {
            ninfer::core::decode_ggml(payload.data() + std::size_t(row) * row_bytes,
                                      std::size_t(k), decoder.block, decoder.type_bytes,
                                      decoder.host, weights);
            for (int t = 0; t < kColumns; ++t) {
                float acc = 0.0f;
                for (int i = 0; i < k; ++i) {
                    acc += weights[std::size_t(i)] * activation[std::size_t(t) * k + i];
                }
                reference[std::size_t(t) * kRows + row] = acc;
            }
        }

        ninfer::DeviceBuffer weights_on_device(payload.size());
        ninfer::DeviceBuffer x_on_device(activation_bf16.size() * 2);
        ninfer::DeviceBuffer y_on_device(std::size_t(kRows) * kColumns * 2);
        weights_on_device.copy_from_host(payload.data(), payload.size());
        x_on_device.copy_from_host(activation_bf16.data(), activation_bf16.size() * 2);
        y_on_device.fill(0);

        ninfer::Weight weight{};
        weight.qdata           = weights_on_device.p;
        weight.payload_bytes   = payload.size();
        weight.qtype           = decoder.format;
        weight.layout          = ninfer::QuantLayout::GgmlBlock;
        weight.n               = kRows;
        weight.k               = k;
        weight.shape[0]        = kRows;
        weight.shape[1]        = k;
        weight.padded_shape[0] = kRows;
        weight.padded_shape[1] = k;
        weight.ndim            = 2;

        ninfer::Tensor x{};
        x.data  = x_on_device.p;
        x.dtype = ninfer::DType::BF16;
        x.ne[0] = k;
        x.ne[1] = kColumns;
        x.nb[0] = 2;
        x.nb[1] = std::int64_t(k) * 2;
        x.nb[2] = x.nb[1] * kColumns;
        x.nb[3] = x.nb[2];

        ninfer::Tensor y{};
        y.data  = y_on_device.p;
        y.dtype = ninfer::DType::BF16;
        y.ne[0] = kRows;
        y.ne[1] = kColumns;
        y.nb[0] = 2;
        y.nb[1] = std::int64_t(kRows) * 2;
        y.nb[2] = y.nb[1] * kColumns;
        y.nb[3] = y.nb[2];

        ninfer::ops::detail::ggml::launch_linear(x, weight, y, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<std::uint16_t> got(std::size_t(kRows) * kColumns, 0);
        y_on_device.copy_to_host(got.data(), got.size() * 2);

        // The kernel writes BF16, so the reference is rounded the same way before comparing: a
        // difference has to be a real one, not the output format's own resolution.
        std::vector<std::uint16_t> reference_bf16(reference.size());
        for (std::size_t i = 0; i < reference.size(); ++i) {
            reference_bf16[i] = float_to_bf16(reference[i]);
        }

        double worst_relative = 0;
        int worst_at          = -1;
        int worst_ulp         = 0;
        for (std::size_t i = 0; i < got.size(); ++i) {
            const double value = bf16_to_float(got[i]);
            const double want  = bf16_to_float(reference_bf16[i]);
            const double diff  = std::fabs(value - want);
            // One BF16 ulp is 1/256 of the magnitude; allow exactly that, so a wrong grid, sign or
            // scale (which moves the result by far more) still fails.
            const double bound = std::max(kAbsoluteFloor, (1.0 / 256.0) * std::fabs(want));
            if (diff > bound) {
                const double rel = std::fabs(want) > 0 ? diff / std::fabs(want) : diff;
                if (rel > worst_relative) {
                    worst_relative = rel;
                    worst_at       = int(i);
                    worst_ulp = int(std::lround(diff / std::max(std::fabs(want), 1e-30) * 256.0));
                }
            }
        }
        check(worst_relative == 0, decoder.encoding, "kernel disagrees with the Host reference");
        if (worst_relative != 0 && worst_at >= 0) {
            std::fprintf(stderr, "      y[%d]: kernel %.9g, host %.9g (rel %.3g, %d ulp)\n",
                         worst_at, bf16_to_float(got[std::size_t(worst_at)]),
                         bf16_to_float(reference_bf16[std::size_t(worst_at)]), worst_relative,
                         worst_ulp);
        }
        std::printf("%-8s K=%d N=%d cols=%d\n", decoder.encoding, k, kRows, kColumns);
    }

    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("ggml linear: %zu encodings match the Host reference\n",
                sizeof(kDecoders) / sizeof(kDecoders[0]));
    return 0;
}
