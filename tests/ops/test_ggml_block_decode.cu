// The Device GGML block decoders, checked against the Host reference on the same real blocks.
//
// Two properties matter here and neither is about speed. First, the Device decode must reproduce the
// Host decode of the very same bytes: the Host side is pinned to gguf-py (tests/core/
// test_ggml_dequant.cpp), so this test extends that chain into the kernels. Second, it must decode
// whole blocks from an arbitrary offset - a weight row is a run of blocks, not one aligned block.
//
// The tolerance is the same relative bound as the host test: both sides compute one expression in
// binary32, and the Device may contract `d * q - m` into an FMA while the Host rounds the product
// first. A wrong grid, sign or scale is orders of magnitude outside that bound.

#include "core/arena.h"
#include "core/device.h"
#include "core/ggml_dequant.h"
#include "core/ggml_dequant_fixture.h"
#include "ops/linear/ggml/ggml_block_decode.cuh"

#include <cmath>
#include <cstdio>
#include <span>
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

// The Device entry per encoding: this table is also the statement of which GGML type each name is.
// The Device side is dispatched by an integer tag inside the kernel - a `__device__` function's
// address is not a host pointer and cannot be passed as a kernel argument.
struct Decoder {
    const char* encoding;
    HostDecode host;
    int type;
};

constexpr Decoder kDecoders[] = {
    {"Q4_K", ninfer::core::decode_ggml_q4_k, 0},
    {"Q5_K", ninfer::core::decode_ggml_q5_k, 1},
    {"Q6_K", ninfer::core::decode_ggml_q6_k, 2},
    {"Q2_0", ninfer::core::decode_ggml_q2_0, 3},
    {"IQ4_NL", ninfer::core::decode_ggml_iq4_nl, 4},
    {"IQ4_XS", ninfer::core::decode_ggml_iq4_xs, 5},
    {"IQ2_XXS", ninfer::core::decode_ggml_iq2_xxs, 6},
    {"IQ2_XS", ninfer::core::decode_ggml_iq2_xs, 7},
    {"IQ2_S", ninfer::core::decode_ggml_iq2_s, 8},
    {"IQ3_XXS", ninfer::core::decode_ggml_iq3_xxs, 9},
    {"IQ3_S", ninfer::core::decode_ggml_iq3_s, 10},
};

const Decoder* find(const char* encoding) {
    for (const auto& entry : kDecoders) {
        if (std::string_view(entry.encoding) == encoding) { return &entry; }
    }
    return nullptr;
}

// One block per thread: the launch this test needs, not a production shape.
__global__ void decode_blocks(const std::uint8_t* blocks, int block_bytes, int block_elements,
                              int count, int type, float* out) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) { return; }
    const std::uint8_t* src = blocks + std::size_t(index) * block_bytes;
    float* dst              = out + std::size_t(index) * block_elements;
    using namespace ninfer::ops::detail::ggml;
    switch (type) {
    case 0: decode_q4_k(src, dst); break;
    case 1: decode_q5_k(src, dst); break;
    case 2: decode_q6_k(src, dst); break;
    case 3: decode_q2_0(src, dst); break;
    case 4: decode_iq4_nl(src, dst); break;
    case 5: decode_iq4_xs(src, dst); break;
    case 6: decode_iq2_xxs(src, dst); break;
    case 7: decode_iq2_xs(src, dst); break;
    case 8: decode_iq2_s(src, dst); break;
    case 9: decode_iq3_xxs(src, dst); break;
    case 10: decode_iq3_s(src, dst); break;
    }
}

constexpr double kRelativeTolerance = 2e-6;
constexpr double kAbsoluteFloor     = 1e-7;

} // namespace

int main() {
    ninfer::DeviceContext context(0);
    context.bind_to_current_thread();
    cudaStream_t stream = context.stream;

    for (const auto& fixture : ninfer::test::kGgmlFixtures) {
        const Decoder* decoder = find(fixture.encoding);
        check(decoder != nullptr, fixture.encoding, "no decoder pair for this encoding");
        if (decoder == nullptr) { continue; }

        const std::size_t blocks = fixture.bytes / fixture.type_bytes;
        check(fixture.elements == blocks * fixture.block, fixture.encoding,
              "fixture element count disagrees with its block count");

        std::vector<float> expected(fixture.elements, 0.0f);
        ninfer::core::decode_ggml(fixture.data, fixture.elements, fixture.block, fixture.type_bytes,
                                  decoder->host, expected);

        ninfer::DeviceBuffer blocks_on_device(fixture.bytes);
        ninfer::DeviceBuffer values_on_device(fixture.elements * sizeof(float));
        blocks_on_device.copy_from_host(fixture.data, fixture.bytes);
        values_on_device.fill(0);

        const int threads = 64;
        const int grid    = int((blocks + threads - 1) / threads);
        decode_blocks<<<grid, threads, 0, stream>>>(
            static_cast<const std::uint8_t*>(blocks_on_device.p), int(fixture.type_bytes),
            int(fixture.block), int(blocks), decoder->type,
            static_cast<float*>(values_on_device.p));
        CUDA_CHECK(cudaGetLastError());
        // The download runs on the legacy default stream, which does not order against a
        // non-blocking stream, so the kernel's completion has to be awaited explicitly first.
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<float> decoded(fixture.elements, 0.0f);
        values_on_device.copy_to_host(decoded.data(), decoded.size() * sizeof(float));
        context.synchronize();

        double worst_relative   = 0;
        std::size_t worst_index = 0;
        for (std::size_t i = 0; i < decoded.size(); ++i) {
            const double got  = decoded[i];
            const double want = expected[i];
            const double diff  = std::fabs(got - want);
            const double bound = std::max(kAbsoluteFloor, kRelativeTolerance * std::fabs(want));
            if (diff > bound) {
                const double rel = std::fabs(want) > 0 ? diff / std::fabs(want) : diff;
                if (rel > worst_relative) {
                    worst_relative = rel;
                    worst_index    = i;
                }
            }
        }
        check(worst_relative == 0, fixture.encoding, "Device decode disagrees with the Host");
        if (worst_relative != 0) {
            std::fprintf(stderr, "      element %zu: device %.9g, host %.9g (rel %.3g)\n",
                         worst_index, decoded[worst_index], expected[worst_index],
                         worst_relative);
        }
        std::printf("%-8s blocks=%zu elements=%zu\n", fixture.encoding, blocks, fixture.elements);
    }

    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("device ggml decode: %zu encodings match the Host reference\n",
                sizeof(ninfer::test::kGgmlFixtures) / sizeof(ninfer::test::kGgmlFixtures[0]));
    return 0;
}
