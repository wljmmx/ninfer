#pragma once

// Host-side scalar decode of one `q4_g64_fp16` row stored in `row_split_k128_v1` planes.
//
// The nibble rule and the arithmetic are the q4 kernels' own: a stored nibble u is the four-bit
// two's-complement word of the weight, so the value is `(u ^ 8) - 8` in [-8,7], multiplied by the
// group's binary16 multiplier and rounded to BF16 - the pair the Device side produces with
// `ops::common::Q4MmaDecodeAtom::decode_pair_with_scale` (src/ops/common/quant/integer.cuh, the
// `bfe.s32` sign extension plus `__floats2bfloat162_rn`). This header exists because that atom is
// __device__ only and the PLE table is decoded on the Host, where the n-gram embedding is gathered
// from disk; `tests/models/qwen4_exp/test_ple_q4_row.cpp` pins the equivalence against both the
// GPU kernel (a one-hot linear) and the artifact codecs' independent logical decoder.

#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace ninfer::ops::q4 {

// row_split_k128_v1 geometry for q4_g64_fp16 (docs/maintainer/storage-layouts.md §3.1, §3.3, §3.5).
inline constexpr std::size_t kGroupLanes        = 64;
inline constexpr std::size_t kCodeBytesPerGroup = 32;
inline constexpr std::size_t kScaleBytesPerGroup = 2;

// binary16 -> binary32, exact for every input including subnormals and infinities.
[[nodiscard]] inline float fp16_to_f32(std::uint16_t bits) noexcept {
    const std::uint32_t sign = std::uint32_t(bits & 0x8000u) << 16;
    const std::uint32_t exponent = (bits >> 10) & 0x1fu;
    const std::uint32_t mantissa = bits & 0x3ffu;
    if (exponent == 0) {
        if (mantissa == 0) return std::bit_cast<float>(sign);
        // Subnormal: normalise by shifting the mantissa until its implicit bit is set.
        int shift      = -1;
        std::uint32_t m = mantissa;
        do {
            ++shift;
            m <<= 1;
        } while ((m & 0x400u) == 0);
        return std::bit_cast<float>(sign | (std::uint32_t(127 - 15 - shift) << 23) |
                                    ((m & 0x3ffu) << 13));
    }
    if (exponent == 31)
        return std::bit_cast<float>(sign | 0x7f800000u | (mantissa << 13));
    return std::bit_cast<float>(sign | ((exponent + 127 - 15) << 23) | (mantissa << 13));
}

// binary32 -> binary16(bfloat16) with round-to-nearest-even, the rounding __floats2bfloat162_rn
// applies. NaN stays a quiet NaN; every product decoded here is finite.
[[nodiscard]] inline std::uint16_t f32_to_bf16(float value) noexcept {
    std::uint32_t bits                     = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return std::uint16_t((bits >> 16) | 0x0040u);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return std::uint16_t(bits >> 16);
}

// One packed byte holds two lanes: the even lane in the low nibble, the odd lane in the high one.
[[nodiscard]] inline int signed_code(std::uint8_t packed, int lane) noexcept {
    const int nibble = lane & 1 ? (packed >> 4) & 0x0f : packed & 0x0f;
    return (nibble ^ 0x08) - 0x08;
}

// One row: `codes` is the row's slice of the base-code plane (lanes/2 bytes, even lane first) and
// `scales` its slice of the binary16 scale plane (one word per 64 lanes). `output` receives the
// logical lanes in order; lanes past `output.size()` (the 128-lane padding) are not produced.
inline void decode_q4_g64_row(std::span<const std::byte> codes, std::span<const std::byte> scales,
                              std::span<std::uint16_t> output) {
    if (output.empty() || codes.size() < (output.size() + 1) / 2) {
        throw std::invalid_argument("q4 row decode: the code plane is shorter than the row");
    }
    const auto groups = (output.size() + kGroupLanes - 1) / kGroupLanes;
    if (scales.size() < groups * kScaleBytesPerGroup) {
        throw std::invalid_argument("q4 row decode: the scale plane is shorter than the row");
    }
    for (std::size_t lane = 0; lane < output.size(); ++lane) {
        const auto group   = lane / kGroupLanes;
        const auto byte    = lane / 2;
        const auto* scale  = scales.data() + group * kScaleBytesPerGroup;
        const auto bits    = std::uint16_t(std::to_integer<unsigned>(scale[0])) |
                          std::uint16_t(std::to_integer<unsigned>(scale[1]) << 8);
        const auto value   = float(signed_code(std::to_integer<std::uint8_t>(codes[byte]),
                                              int(lane & 1))) *
                           fp16_to_f32(bits);
        output[lane] = f32_to_bf16(value);
    }
}

} // namespace ninfer::ops::q4
