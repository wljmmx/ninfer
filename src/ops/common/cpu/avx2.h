#pragma once

#include "ops/common/quant/nvfp4.h"

#include <immintrin.h>
#include <cstdint>
#include <cstring>

namespace ninfer::ops::common::cpu::avx2 {

// port (sm_89): MSVC only declares _cvtsh_ss under zmmintrin.h (AVX-512), so express the same
// F16C conversion through _mm_cvtph_ps, which immintrin.h always provides.
inline float fp16_to_f32(std::uint16_t bits) noexcept {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(bits))));
}

// Private primitives for an isolated avx2/fma/f16c translation unit. Load/store
// pointers need only element alignment and must cover the complete vector extent.
inline __m256 load_bf16(const std::uint16_t* source) noexcept {
    const auto words = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source));
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(words), 16));
}

// 16 packed bytes -> 32 unsigned bytes, low nibble then high nibble for each byte.
inline __m256i unpack_u4(const std::uint8_t* source) noexcept {
    const auto words =
        _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(source)));
    return _mm256_or_si256(
        _mm256_and_si256(words, _mm256_set1_epi16(15)),
        _mm256_and_si256(_mm256_slli_epi16(words, 4), _mm256_set1_epi16(0x0f00)));
}

inline float reduce_sum(__m256 values) noexcept {
    auto sum = _mm_add_ps(_mm256_castps256_ps128(values), _mm256_extractf128_ps(values, 1));
    sum      = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(_mm_hadd_ps(sum, sum));
}

inline __m256 decode_nvfp4(const std::uint8_t* codes, std::uint8_t scale) noexcept {
    std::uint32_t packed;
    std::memcpy(&packed, codes, sizeof(packed));
    const auto words = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(packed));
    const auto bytes =
        _mm_or_si128(_mm_and_si128(words, _mm_set1_epi16(15)),
                     _mm_and_si128(_mm_slli_epi16(words, 4), _mm_set1_epi16(0x0f00)));
    const auto indices = _mm256_cvtepu8_epi32(bytes);
    const auto values =
        _mm256_permutevar8x32_ps(_mm256_load_ps(nvfp4_tables.fp32[scale].data()), indices);
    return _mm256_xor_ps(values, _mm256_castsi256_ps(_mm256_slli_epi32(
                                     _mm256_and_si256(indices, _mm256_set1_epi32(8)), 28)));
}

struct F32 {
    using Float                = __m256;
    static constexpr int lanes = 8;

    static Float zero() noexcept { return _mm256_setzero_ps(); }

    static Float load(const float* p) noexcept { return _mm256_loadu_ps(p); }

    static void store(float* p, Float v) noexcept { _mm256_storeu_ps(p, v); }

    static Float bf16(const std::uint16_t* p) noexcept { return load_bf16(p); }

    static Float fma(Float a, Float b, Float c) noexcept { return _mm256_fmadd_ps(a, b, c); }

    static float sum(Float v) noexcept { return reduce_sum(v); }
};

template <int Bits>
inline __m256 decode_signed_q(const std::uint8_t* code, const std::uint8_t* high) noexcept {
    static_assert(Bits >= 4 && Bits <= 6);
    // port (sm_89): measurement switch. NINFER_MOE_CPU_NODECODE replaces the decode with a
    // constant so the bench reports what the decode itself costs. The values are wrong on purpose;
    // only the timing is read, and it decides whether an int8 dot product is worth building.
    static const bool skip_decode = std::getenv("NINFER_MOE_CPU_NODECODE") != nullptr;
    if (skip_decode) { return _mm256_set1_ps(1.0F); }
    std::uint32_t packed;
    std::memcpy(&packed, code, sizeof(packed));
    const auto words = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(packed));
    const auto bytes =
        _mm_or_si128(_mm_and_si128(words, _mm_set1_epi16(15)),
                     _mm_and_si128(_mm_slli_epi16(words, 4), _mm_set1_epi16(0x0f00)));
    auto values = _mm256_cvtepu8_epi32(bytes);
    if constexpr (Bits == 6) {
        std::uint16_t packed_high;
        std::memcpy(&packed_high, high, sizeof(packed_high));
        auto upper = _mm256_srlv_epi32(_mm256_set1_epi32(packed_high),
                                       _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14));
        upper      = _mm256_srai_epi32(_mm256_slli_epi32(upper, 30), 26);
        values     = _mm256_add_epi32(values, upper);
    } else if constexpr (Bits == 5) {
        auto bits =
            _mm256_srlv_epi32(_mm256_set1_epi32(*high), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
        bits   = _mm256_slli_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(1)), 4);
        values = _mm256_sub_epi32(values, bits);
    } else {
        values =
            _mm256_sub_epi32(_mm256_xor_si256(values, _mm256_set1_epi32(8)), _mm256_set1_epi32(8));
    }
    return _mm256_cvtepi32_ps(values);
}

// port (sm_89): expand 32 five-bit codes in one pass. The float path decodes 8 weights per
// iteration - four 32-bit lanes for the nibble, four for the fifth bit - and repeats that eight
// times per 64-value group. The same 16 bytes of codes hold 32 weights, so one pass can build all of
// them as int8 and widen once. A 5-bit code is `nibble - 16 * bit`, so the bit plane is subtracted
// rather than sign-extended; the spread/compare trick that turns 32 bits into 32 bytes comes from
// ggml's AVX2 q5_1 path and yields one byte per weight in natural order.
inline void decode_signed_q32_q5(const std::uint8_t* code, const std::uint8_t* high,
                                 __m256 out[4]) noexcept {
    static const bool skip_decode = std::getenv("NINFER_MOE_CPU_NODECODE") != nullptr;
    if (skip_decode) {
        for (int i = 0; i < 4; ++i) { out[i] = _mm256_set1_ps(1.0F); }
        return;
    }
    // weight 2b is the low nibble of byte b and weight 2b+1 the high one, so interleaving the two
    // nibble planes rebuilds weight order.
    const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(code));
    const __m128i nibble = _mm_set1_epi8(0x0F);
    const __m128i low    = _mm_and_si128(packed, nibble);
    const __m128i high_n = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble);
    __m256i values =
        _mm256_set_m128i(_mm_unpackhi_epi8(low, high_n), _mm_unpacklo_epi8(low, high_n));
    std::uint32_t bits;
    std::memcpy(&bits, high, sizeof(bits));
    const __m256i spread = _mm256_set_epi64x(0x0303030303030303LL, 0x0202020202020202LL,
                                             0x0101010101010101LL, 0x0000000000000000LL);
    __m256i plane = _mm256_shuffle_epi8(_mm256_set1_epi32(static_cast<int>(bits)), spread);
    plane = _mm256_or_si256(plane, _mm256_set1_epi64x(0x7fbfdfeff7fbfdfeLL));
    const __m256i set = _mm256_cmpeq_epi8(plane, _mm256_set1_epi8(-1));
    values = _mm256_sub_epi8(values, _mm256_and_si256(set, _mm256_set1_epi8(0x10)));
    const __m128i halves[2] = {_mm256_castsi256_si128(values), _mm256_extracti128_si256(values, 1)};
    for (int h = 0; h < 2; ++h) {
        out[2 * h]     = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(halves[h]));
        out[2 * h + 1] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(halves[h], 8)));
    }
}

} // namespace ninfer::ops::common::cpu::avx2
