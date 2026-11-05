#pragma once

// port (sm_89): CPU-side decode substrate for the rotated-K (rk) paged KV layouts.
// Feasibility batch 1: byte-exact fp32 read-back of rk4v4 / rk8v4 pages, the groundwork
// for computing host-side attention partials over host-resident KV pages.
//
// Deliberately standalone: no CUDA header, no repo include, header-only, host-only, so
// the substrate compiles into any host translation unit without touching the GPU build.
// Every constant and addressing rule mirrors one production source; the mirror is
// documented at each definition so a GPU-side change has an exact place to land.
//
// Bit-exactness contract. The GPU kernels dequantize a stored code as
//     value = float(code) * fp32(stored FP16 scale)
// — the V dequant of grouped_mma.cuh / batch_wide_mma.cuh multiplies
// `static_cast<float>(code) * vsf` in fp32 before its fp16 rounding, the K path applies
// the same product through the int8-MMA scale factors (`q_scale * k_scale * float(c)`),
// and the shared helpers rk4_dequant_bf16x8_from / rk4_dequant_f16x8_from /
// kv_cache_int8_dequant_i8x8_from take the same fp32 product before rounding. A stored
// code is at most 8 bits and an FP16 mantissa at most 11 bits, so the product always
// fits an FP32 mantissa: the fp32 multiply is EXACT. That makes the scalar and AVX2
// paths below bit-identical by construction, and makes the substrate's output equal
// the GPU value on every kernel path, pre-rounding. Kernel-specific fp16/bf16 rounding
// happens at consumption and is out of the decode's scope.

#include <cstdint>
#include <cstring>

#if defined(__AVX2__) && !defined(NINFER_SOFTMAX_CPU_FORCE_SCALAR)
#include <immintrin.h>
#define NINFER_SOFTMAX_CPU_VECTOR 1
#else
// Scalar fallback: bit-identical to the AVX2 path (the decode arithmetic is exact in
// fp32), for translation units built without /arch:AVX2 or forced scalar via
// NINFER_SOFTMAX_CPU_FORCE_SCALAR (A/B verification switch, not a tuning knob).
#define NINFER_SOFTMAX_CPU_VECTOR 0
#endif

namespace ninfer::ops::detail::softmax_cpu {

// --- mirrored constants ------------------------------------------------------------
// All four match the GPU layout the rk kernels address (D256 head, G64 scale groups,
// 64-token pages). Sources: core/paged_kv_storage.h (kD256KVCacheHeadDim, per-plane
// extents), core/paged_kv_cache.h (kPagedKVPageSize), ops/kv_cache/int8_g64_codec.cuh
// and ops/kv_cache/rk4_codec.cuh (kKVCacheInt8Group / kKVCacheRkGroup families).

inline constexpr int kHeadDim    = 256;                // D256 KV head dimension
inline constexpr int kGroup      = 64;                 // G64 quantization group
inline constexpr int kGroups     = kHeadDim / kGroup;  // 4 groups per K/V row
inline constexpr int kPageSize   = 64;                 // tokens per physical page
inline constexpr int kInt4Plane  = kHeadDim / 2;       // 128 B/token-head, packed int4
inline constexpr int kInt8Plane  = kHeadDim;           // 256 B/token-head, plain int8
inline constexpr int kScalePlane = kGroups;            // 4 FP16 scales/token-head

static_assert(kHeadDim == 256 && kGroup == 64 && kGroups == 4);
static_assert(kPageSize == 64);
static_assert(kInt4Plane * kPageSize > 0 && kInt8Plane * kPageSize > 0);

// --- scale conversion ---------------------------------------------------------------

// FP16 bits -> fp32, exact for every bit pattern. Normals, subnormals, +-0 and +-Inf
// follow the IEEE expansion the hardware cvt performs. NaN inputs map to the canonical
// positive NaN 0x7fffffff: that is what the production read-back's __half2float
// (cvt.f32.f16) produces on sm_89 for EVERY fp16 NaN regardless of sign and payload —
// measured on RTX 4090 / CUDA 13.4 by the isolated GPU probe (half2float(0x7c01) =
// half2float(0xfc01) = half2float(0x7e00) = half2float(0xfe00) = 0x7fffffff, while
// 0x7c00 -> 0x7f800000 and 0xfc00 -> 0xff800000). Note this deliberately differs from
// Intel F16C, which preserves the payload and only quiets sNaNs. The encoder can never
// store a NaN scale (FP16-RNE of a finite absmax is finite or +-Inf), so this branch is
// robustness parity with the GPU for arbitrary on-disk bits, not a reachable case.
// Portable on purpose: the scalar fallback must not depend on F16C. The subnormal case
// is computed as mant * 2^-24 — both factors are exact in fp32 and the product lands in
// the normal fp32 range, so the multiply is exact.
inline float fp16_bits_to_f32(std::uint16_t bits) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16;
    const std::uint32_t rest = bits & 0x7fffu;
    const std::uint32_t exp5 = rest >> 10;
    const std::uint32_t mant = rest & 0x3ffu;
    if (rest == 0u) {  // +-0
        float out;
        std::memcpy(&out, &sign, sizeof out);
        return out;
    }
    if (exp5 == 0u) {  // subnormal: value = mant * 2^-24, exact as a normal fp32
        const float magnitude = static_cast<float>(mant) * 0x1p-24f;
        std::uint32_t wide;
        std::memcpy(&wide, &magnitude, sizeof wide);
        wide |= sign;  // magnitude > 0, so its sign bit is clear
        float out;
        std::memcpy(&out, &wide, sizeof out);
        return out;
    }
    std::uint32_t wide;
    if (exp5 == 31u) {
        // GPU cvt.f32.f16 canonicalizes NaN and preserves signed Inf.
        wide = mant == 0u ? (sign | 0x7f800000u) : 0x7fffffffu;
    } else {  // normal: fp32 exponent = fp16 exponent - 15 + 127
        wide = sign | ((exp5 + 112u) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &wide, sizeof out);
    return out;
}

// --- plane addressing ----------------------------------------------------------------

// Offset of `leading` inside a plane whose per-token-head extent is `leading_extent`
// units (bytes for code planes, FP16 elements for scale planes). Mirrors
// paged_kv_element_offset<LeadingExtent, HeadExtent> (ops/kernel/paged_kv_address.cuh):
//     LeadingExtent * kPagedKVPageSize * (head + HeadExtent * physical_page)
//   + LeadingExtent * page_offset + leading
// with HeadExtent = the layer's KV head count (2 or 4: ops/kv_cache/append/geometry.cuh,
// ops/softmax_attention/common/causal_geometry.h) and kPagedKVPageSize = 64. The GPU
// kernels address pages with physical ids; the caller resolves logical -> physical via
// the block table exactly like they do (paged_kv_physical_page, position >> 6).
inline std::int64_t plane_offset(int leading_extent, int kv_heads, std::int64_t physical_page,
                                 int kv_head, int token_in_page, int leading) noexcept {
    return static_cast<std::int64_t>(leading_extent) * static_cast<std::int64_t>(kPageSize) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(kv_heads) * physical_page) +
           static_cast<std::int64_t>(leading_extent) * token_in_page + leading;
}

// --- group decode primitives ----------------------------------------------------------

// Unpack one signed int4 code. Mirrors rk4_unpack (ops/kv_cache/rk4_codec.cuh):
// high=0 -> low nibble, high=1 -> high nibble; values 8..15 sign-extend to -8..-1.
// Stored codes come from an encoder that clamps to [-7, 7], but the decode accepts the
// full 4-bit range so any on-disk byte reads back the way the GPU read-back would.
inline int unpack_int4(std::uint8_t packed, int high) noexcept {
    int code = high ? static_cast<int>(packed >> 4) : static_cast<int>(packed);
    code &= 0xf;
    return code >= 8 ? code - 16 : code;
}

// One G64 group of packed int4 codes -> 64 fp32 values.
// `packed` points at the group's 32 bytes: dim d lives in byte d >> 1, low nibble for
// even d and high nibble for odd d (rk4_codec.cuh header rule). `scale_bits` is the
// group's FP16 scale. `out` receives 64 floats in dim order; no alignment required.
inline void decode_int4_group(const std::uint8_t* packed, std::uint16_t scale_bits,
                              float* out) noexcept {
    const float scale = fp16_bits_to_f32(scale_bits);
#if NINFER_SOFTMAX_CPU_VECTOR
    const __m256 scale_v = _mm256_set1_ps(scale);
    for (int chunk = 0; chunk < 2; ++chunk) {
        // 16 packed bytes = 32 codes covering dims [32*chunk, 32*chunk + 32).
        const __m128i bytes =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed + 16 * chunk));
        const __m128i low  = _mm_and_si128(bytes, _mm_set1_epi8(0x0f));
        const __m128i high = _mm_and_si128(_mm_srli_epi16(bytes, 4), _mm_set1_epi8(0x0f));
        // merged byte i = code of dim i: low nibble of byte i/2 for even i, high for odd.
        const __m256i merged = _mm256_set_m128i(_mm_unpackhi_epi8(low, high),
                                                _mm_unpacklo_epi8(low, high));
        const __m128i h0 = _mm256_castsi256_si128(merged);
        const __m128i h1 = _mm256_extracti128_si256(merged, 1);
        const __m128i eights[4] = {h0, _mm_srli_si128(h0, 8), h1, _mm_srli_si128(h1, 8)};
        for (int i = 0; i < 4; ++i) {
            // cvtepi8 sign-extends the BYTE, and the nibble values 0..15 are all
            // positive bytes, so the nibble->int8 sign extension is its own step:
            // (v ^ 8) - 8 maps 8..15 to -8..-1, mirroring rk4_sign_extend_nibbles and
            // unpack_int4. cvtepi32_ps is exact for |code| <= 8, mul_ps is the same
            // IEEE product the GPU takes — all steps are bit-exact.
            const __m256i nibbles = _mm256_cvtepi8_epi32(eights[i]);
            const __m256i codes   = _mm256_sub_epi32(_mm256_xor_si256(nibbles, _mm256_set1_epi32(8)),
                                                   _mm256_set1_epi32(8));
            _mm256_storeu_ps(out + 32 * chunk + 8 * i,
                             _mm256_mul_ps(_mm256_cvtepi32_ps(codes), scale_v));
        }
    }
#else
    for (int d = 0; d < kGroup; ++d) {
        out[d] = static_cast<float>(unpack_int4(packed[d >> 1], d & 1)) * scale;
    }
#endif
}

// One G64 group of plain int8 codes -> 64 fp32 values (rk8v4's K plane).
// Same exact-product contract as decode_int4_group; `codes` points at the group's 64
// signed bytes, dim d at codes[d] (kv_cache_int8_quant_code_index layout).
inline void decode_int8_group(const std::int8_t* codes, std::uint16_t scale_bits,
                              float* out) noexcept {
    const float scale = fp16_bits_to_f32(scale_bits);
#if NINFER_SOFTMAX_CPU_VECTOR
    const __m256 scale_v = _mm256_set1_ps(scale);
    for (int chunk = 0; chunk < 4; ++chunk) {
        const __m128i bytes =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes + 16 * chunk));
        const __m128i eights[2] = {bytes, _mm_srli_si128(bytes, 8)};
        for (int i = 0; i < 2; ++i) {
            const __m256i wide = _mm256_cvtepi8_epi32(eights[i]);
            _mm256_storeu_ps(out + 16 * chunk + 8 * i,
                             _mm256_mul_ps(_mm256_cvtepi32_ps(wide), scale_v));
        }
    }
#else
    for (int d = 0; d < kGroup; ++d) {
        out[d] = static_cast<float>(codes[d]) * scale;
    }
#endif
}

// --- row decode helpers ---------------------------------------------------------------

// Full 256-dim row over a packed-int4 plane with FP16 G64 scales: the codec shared by
// rk4v4's K and V planes and by the whole rk family's V plane. Group g owns dims
// [64g, 64g+64) = packed bytes [32g, 32g+32) of the 128-byte row and scale element g.
inline void decode_int4_row(const std::uint8_t* plane, const std::uint16_t* scale_plane,
                            int kv_heads, std::int64_t physical_page, int kv_head,
                            int token_in_page, float* out) noexcept {
    for (int g = 0; g < kGroups; ++g) {
        const std::int64_t codes = plane_offset(kInt4Plane, kv_heads, physical_page, kv_head,
                                                token_in_page, g * (kGroup / 2));
        const std::int64_t scale = plane_offset(kScalePlane, kv_heads, physical_page, kv_head,
                                                token_in_page, g);
        decode_int4_group(plane + codes, scale_plane[scale], out + g * kGroup);
    }
}

// Full 256-dim row over a plain-int8 plane with FP16 G64 scales (rk8v4's K plane).
// Group g owns signed bytes [64g, 64g+64) and scale element g.
inline void decode_int8_row(const std::uint8_t* plane, const std::uint16_t* scale_plane,
                            int kv_heads, std::int64_t physical_page, int kv_head,
                            int token_in_page, float* out) noexcept {
    for (int g = 0; g < kGroups; ++g) {
        const std::int64_t codes = plane_offset(kInt8Plane, kv_heads, physical_page, kv_head,
                                                token_in_page, g * kGroup);
        const std::int64_t scale = plane_offset(kScalePlane, kv_heads, physical_page, kv_head,
                                                token_in_page, g);
        decode_int8_group(reinterpret_cast<const std::int8_t*>(plane + codes),
                           scale_plane[scale], out + g * kGroup);
    }
}

} // namespace ninfer::ops::detail::softmax_cpu
