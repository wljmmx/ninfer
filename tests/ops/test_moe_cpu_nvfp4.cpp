// T115: the CPU NVFP4 expert kernel on the Windows/AVX2 half of this port.
//
// This grades `dot` directly rather than through routed_swiglu: the op-level entry validates an
// expert bank whose per-plane strides come from a geometry that today cannot describe the
// shared-slot layout the GPU side uses (a 512-expert bank would need 512 real segments), so an
// op-level NVFP4 CPU fixture would test the metadata plumbing, not the kernel. What the port needs
// proven here is the arithmetic: E2M1 nibble decode, native 128x4 swizzled scale addressing across
// row and K tiles, per-expert divisor application, and the AVX2 code path this machine selects.

#include "core/cpu/system.h"
#include "ops/common/cpu/avx2.h"
#include "ops/moe/kernels/cpu/nvfp4/compute.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
namespace qw     = ninfer::test::quantized_weight;
namespace kernel = ninfer::ops::detail::moe_cpu::nvfp4;

namespace {

// The AVX2 instantiation of the kernel's SIMD contract, spelled exactly as
// kernels/cpu/nvfp4/avx2.cpp spells it (that one lives in an anonymous namespace).
struct Avx2 : ops::common::cpu::avx2::F32 {
    static constexpr int rows        = 2;
    static constexpr int max_columns = 2;
    static constexpr bool bf16_gate  = false;

    static Float decode(const std::uint8_t* codes, std::uint8_t scale) noexcept {
        return ops::common::cpu::avx2::decode_nvfp4(codes, scale);
    }
};

constexpr int kK        = 256;  // two K tiles of 128 and four 64-wide scale groups
constexpr int kRows     = 130;  // crosses one 128-row scale tile
constexpr float kDivisor = 2.5F;

// E2M1 magnitudes; the sign is the code's top bit.
constexpr float kMagnitudes[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};

unsigned char nibble(std::size_t index, unsigned seed) {
    return static_cast<unsigned char>(qw::detail::mix64(index + seed * 7919ULL) & 15U);
}

void write_scale(std::vector<std::uint8_t>& scales, int row, int group, std::uint8_t value) {
    // The canonical address of native NVFP4 scale words; the kernel derives the same address from
    // (row, k) with its own arithmetic, which is the part under test.
    const auto offset = ops::common::nvfp4_scale_byte_offset(row, group, kK);
    scales[static_cast<std::size_t>(offset)] = value;
}

std::uint8_t read_scale(const std::vector<std::uint8_t>& scales, int row, int group) {
    return scales[static_cast<std::size_t>(ops::common::nvfp4_scale_byte_offset(row, group, kK))];
}

int run_case(const int (&rows)[2], unsigned seed) {
    const std::size_t code_bytes = std::size_t(kRows) * (kK / 2);
    std::vector<std::uint8_t> codes(code_bytes);
    for (std::size_t i = 0; i < codes.size(); ++i) {
        codes[i] = static_cast<std::uint8_t>(nibble(2 * i, seed) | (nibble(2 * i + 1, seed) << 4));
    }
    // Two row tiles, (K/64) 512-byte blocks each.
    // plus one spare block so a generator mistake shows up as a wrong answer, not a heap overwrite.
    std::vector<std::uint8_t> scales(std::size_t(kRows / 128 + 1) * (kK / 64) * 512 + 512);
    for (int row = 0; row < kRows; ++row) {
        for (int group = 0; group < kK / 16; ++group) {
            // Finite positive E4M3FN words 0x01..0x7e: small scales, no NaN.
            const auto word = static_cast<std::uint8_t>(1 + (row * 3 + group * 5 + seed) % 126);
            write_scale(scales, row, group, word);
        }
    }

    kernel::Matrix matrix{codes.data(), scales.data(), kDivisor};

    std::vector<float> x[2];
    for (int c = 0; c < 2; ++c) {
        x[c].resize(kK);
        fill_uniform(x[c], 900 + seed + unsigned(c), -0.7F, 0.7F);
    }
    const float* inputs[2] = {x[0].data(), x[1].data()};
    float got[2][2] = {};
    kernel::dot<Avx2, kK, 2, 2>(matrix, rows, inputs, got);

    int failures = 0;
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            double reference = 0.0;
            for (int k = 0; k < kK; ++k) {
                const auto byte = codes[std::size_t(rows[r]) * (kK / 2) + std::size_t(k / 2)];
                const unsigned code = (k & 1) ? (byte >> 4) : (byte & 15U);
                const float magnitude = kMagnitudes[code & 7U];
                const float signed_value = (code & 8U) ? -magnitude : magnitude;
                const auto scale_word = read_scale(scales, rows[r], k / 16);
                reference += double(signed_value) *
                             qw::detail::decode_e4m3fn(scale_word) * double(x[c][k]);
            }
            reference /= double(kDivisor);

            const double actual = got[r][c];
            const double error  = std::abs(actual - reference);
            const double limit  = 4e-6 * std::max(1.0, std::abs(reference));
            if (!(error <= limit)) {
                std::cerr << "FAIL: row " << rows[r] << " column " << c << " actual=" << actual
                          << " reference=" << reference << " error=" << error << '\n';
                ++failures;
            }
        }
    }
    return failures;
}

} // namespace

int main() {
    if (!current_cpu_features().has(CpuFeature::Avx2)) {
        std::cout << "SKIP: no AVX2 on this CPU\n";
        return 77;
    }
    int failures = 0;
    try {
        const int first[2]  = {0, 1};
        const int second[2] = {128, 129};
        failures += run_case(first, 11);
        failures += run_case(second, 29);
    } catch (const std::exception& error) {
        std::cerr << "CPU NVFP4 test exception: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " CPU NVFP4 AVX2 dot\n";
    return failures == 0 ? 0 : 1;
}

