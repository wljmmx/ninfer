// The GGML embedding gather against the Host reference.
//
// A gathered row is a decode of that row's stored blocks, so the check is: the BF16 row the kernel
// writes equals the Host decode (core/ggml_dequant.h, pinned to gguf-py) of the same bytes. Rows hold
// different tilings of the fixture on purpose - an off-by-one row stride would otherwise pass.

#include "core/arena.h"
#include "core/device.h"
#include "core/ggml_dequant.h"
#include "core/ggml_dequant_fixture.h"
#include "ninfer/ops/embedding.h"
#include "ops/linear/ggml/ggml_embedding.h"

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

constexpr int kRows     = 4;
constexpr int kTokens   = 3;
constexpr int kTableIds = 4; // rows in the table; ids below pick three of them

} // namespace

int main() {
    ninfer::DeviceContext context(0);
    context.bind_to_current_thread();
    cudaStream_t stream = context.stream;

    for (const auto& decoder : kDecoders) {
        const auto* fixture = fixture_for(decoder.encoding);
        check(fixture != nullptr, decoder.encoding, "no fixture for this encoding");
        if (fixture == nullptr) { continue; }

        const int k         = int(decoder.block) * 2;
        const int row_bytes = 2 * int(decoder.type_bytes);
        std::vector<std::uint8_t> table(std::size_t(row_bytes) * kTableIds);
        // Each table row is the fixture rotated, so a wrong row offset cannot look right.
        for (int row = 0; row < kTableIds; ++row) {
            for (int i = 0; i < row_bytes; ++i) {
                table[std::size_t(row) * row_bytes + i] = fixture->data[(i + row * 7) % fixture->bytes];
            }
        }

        const std::int32_t ids[kTokens] = {2, 0, 3};

        std::vector<float> expected(std::size_t(kTokens) * k, 0.0f);
        for (int t = 0; t < kTokens; ++t) {
            std::vector<float> row(std::size_t(k), 0.0f);
            ninfer::core::decode_ggml(table.data() + std::size_t(ids[t]) * row_bytes, std::size_t(k),
                                      decoder.block, decoder.type_bytes, decoder.host, row);
            for (int i = 0; i < k; ++i) expected[std::size_t(t) * k + i] = row[std::size_t(i)];
        }

        ninfer::DeviceBuffer table_on_device(table.size());
        ninfer::DeviceBuffer ids_on_device(sizeof(ids));
        ninfer::DeviceBuffer out_on_device(std::size_t(kTokens) * k * 2);
        table_on_device.copy_from_host(table.data(), table.size());
        ids_on_device.copy_from_host(ids, sizeof(ids));
        out_on_device.fill(0);

        ninfer::Weight weight{};
        weight.qdata           = table_on_device.p;
        weight.payload_bytes   = table.size();
        weight.qtype           = decoder.format;
        weight.layout          = ninfer::QuantLayout::GgmlBlock;
        weight.n               = kTableIds;
        weight.k               = k;
        weight.shape[0]        = kTableIds;
        weight.shape[1]        = k;
        weight.padded_shape[0] = kTableIds;
        weight.padded_shape[1] = k;
        weight.ndim            = 2;

        ninfer::Tensor id_tensor{};
        id_tensor.data  = ids_on_device.p;
        id_tensor.dtype = ninfer::DType::I32;
        id_tensor.ne[0] = kTokens;
        id_tensor.nb[0] = 4;
        id_tensor.nb[1] = 4;

        ninfer::Tensor out{};
        out.data  = out_on_device.p;
        out.dtype = ninfer::DType::BF16;
        out.ne[0] = k;
        out.ne[1] = kTokens;
        out.nb[0] = 2;
        out.nb[1] = std::int64_t(k) * 2;

        ninfer::ops::detail::ggml::launch_embedding_gather(id_tensor, weight, out, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<std::uint16_t> got(std::size_t(kTokens) * k, 0);
        out_on_device.copy_to_host(got.data(), got.size() * 2);

        double worst_relative = 0;
        int worst_at          = -1;
        for (std::size_t i = 0; i < got.size(); ++i) {
            const double value = bf16_to_float(got[i]);
            const double want  = expected[i];
            // One BF16 ulp, as in the Linear test: the output format's own resolution.
            const double bound = std::max(1e-6, (1.0 / 256.0) * std::fabs(want));
            if (std::fabs(value - want) > bound) {
                const double rel = std::fabs(want) > 0 ? std::fabs(value - want) / std::fabs(want) : 0;
                if (rel > worst_relative) {
                    worst_relative = rel;
                    worst_at       = int(i);
                }
            }
        }
        check(worst_relative == 0, decoder.encoding, "gathered row disagrees with the Host decode");
        if (worst_relative != 0 && worst_at >= 0) {
            std::fprintf(stderr, "      y[%d]: kernel %.9g, host %.9g (rel %.3g)\n", worst_at,
                         bf16_to_float(got[std::size_t(worst_at)]), expected[std::size_t(worst_at)],
                         worst_relative);
        }
        std::printf("%-8s K=%d rows=%d tokens=%d\n", decoder.encoding, k, kTableIds, kTokens);
    }

    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("ggml embedding: %zu encodings match the Host decode\n",
                sizeof(kDecoders) / sizeof(kDecoders[0]));
    return 0;
}
