#pragma once

// Host reference decode of the GGML block encodings this tree carries verbatim from GGUF sources.
//
// The container stores each tensor's payload exactly as GGML wrote it (`layout = ggml_block_v1`), so
// "decoding a weight" means walking that block stream. This header is the single written-out
// statement of the format: the Device kernels mirror it, and the numbers are pinned against
// llama.cpp's own dequantisers (tools/qwen4exp/emit_ggml_codebooks.py supplies the codebooks, and
// tests/core/test_ggml_dequant.cpp compares against gguf-py's independent implementation).
//
// Every function here is scalar and allocation-free: it decodes `blocks` into `out` and writes
// nothing else. Callers that need BF16 round the float result once, on the way out.

#include <bit>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <span>

#include "core/ggml_types.h"

namespace ninfer::core {

// The codebooks are included *inside* the namespace so they stay out of the global one; the file is
// generated (tools/qwen4exp/emit_ggml_codebooks.py) and carries no namespace of its own.
#define NINFER_GGML_TABLE(type, name, count) inline constexpr type name[count] = {
#define NINFER_GGML_TABLE_END() };
#include "core/ggml_codebook.inc"
#undef NINFER_GGML_TABLE
#undef NINFER_GGML_TABLE_END

// GGML's super-block size for the K-quants and the i-quants.
inline constexpr std::size_t kGgmlQkK   = 256;
inline constexpr std::size_t kGgmlQk4Nl = 32;
inline constexpr std::size_t kGgmlQk2_0 = 64;

// binary16 -> binary32, exact for every input including subnormals and infinities.
[[nodiscard]] inline float ggml_half_to_float(std::uint16_t bits) noexcept {
    const std::uint32_t sign = std::uint32_t(bits & 0x8000u) << 16;
    const std::uint32_t exp  = (bits >> 10) & 0x1fu;
    const std::uint32_t mant = bits & 0x3ffu;
    if (exp == 0) {
        if (mant == 0) return std::bit_cast<float>(sign);
        int shift       = -1;
        std::uint32_t m = mant;
        do {
            ++shift;
            m <<= 1;
        } while ((m & 0x400u) == 0);
        return std::bit_cast<float>(sign | (std::uint32_t(127 - 15 - shift) << 23) |
                                    ((m & 0x3ffu) << 13));
    }
    if (exp == 31) return std::bit_cast<float>(sign | 0x7f800000u | (mant << 13));
    return std::bit_cast<float>(sign | ((exp + 127 - 15) << 23) | (mant << 13));
}

[[nodiscard]] inline std::uint16_t ggml_float_to_bf16(float value) noexcept {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return std::uint16_t((bits >> 16) | 0x0040u);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return std::uint16_t(bits >> 16);
}

namespace detail {

inline std::uint16_t load_u16(const std::uint8_t* p) noexcept {
    return std::uint16_t(p[0]) | std::uint16_t(p[1]) << 8;
}

inline std::uint32_t load_u32(const std::uint8_t* p) noexcept {
    return std::uint32_t(p[0]) | std::uint32_t(p[1]) << 8 | std::uint32_t(p[2]) << 16 |
           std::uint32_t(p[3]) << 24;
}

inline float half_at(const std::uint8_t* p) noexcept { return ggml_half_to_float(load_u16(p)); }

// Q4_K/Q5_K's six-bit scale/minimum pair, upstream's get_scale_min_k4.
inline void scale_min_k4(int j, const std::uint8_t* q, std::uint8_t& d, std::uint8_t& m) noexcept {
    if (j < 4) {
        d = q[j] & 63u;
        m = q[j + 4] & 63u;
    } else {
        d = std::uint8_t((q[j + 4] & 0x0fu) | ((q[j - 4] >> 6) << 4));
        m = std::uint8_t((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

inline float sign_of(std::uint8_t signs, int j) noexcept {
    return (signs & kIq2xsMask[j]) != 0 ? -1.0f : 1.0f;
}

} // namespace detail

// Q4_K: 144 B per 256 values - d, dmin (binary16), 12 packed scale/min bytes, 128 nibble bytes.
inline void decode_ggml_q4_k(const std::uint8_t* block, std::span<float> out) {
    const float d   = detail::half_at(block);
    const float min = detail::half_at(block + 2);
    const std::uint8_t* scales = block + 4;
    const std::uint8_t* q      = block + 16;
    int is = 0;
    std::size_t o = 0;
    for (int j = 0; j < int(kGgmlQkK); j += 64) {
        std::uint8_t sc = 0, m = 0;
        detail::scale_min_k4(is + 0, scales, sc, m);
        const float d1 = d * float(sc), m1 = min * float(m);
        detail::scale_min_k4(is + 1, scales, sc, m);
        const float d2 = d * float(sc), m2 = min * float(m);
        for (int l = 0; l < 32; ++l) out[o++] = d1 * float(q[l] & 0x0fu) - m1;
        for (int l = 0; l < 32; ++l) out[o++] = d2 * float(q[l] >> 4) - m2;
        q += 32;
        is += 2;
    }
}

// Q5_K: 176 B per 256 values - as Q4_K but with a 32-byte high-bit plane after the scales.
inline void decode_ggml_q5_k(const std::uint8_t* block, std::span<float> out) {
    const float d   = detail::half_at(block);
    const float min = detail::half_at(block + 2);
    const std::uint8_t* scales = block + 4;
    const std::uint8_t* qh     = block + 16;
    const std::uint8_t* ql     = block + 48;
    int is = 0;
    std::uint8_t u1 = 1, u2 = 2;
    std::size_t o = 0;
    for (int j = 0; j < int(kGgmlQkK); j += 64) {
        std::uint8_t sc = 0, m = 0;
        detail::scale_min_k4(is + 0, scales, sc, m);
        const float d1 = d * float(sc), m1 = min * float(m);
        detail::scale_min_k4(is + 1, scales, sc, m);
        const float d2 = d * float(sc), m2 = min * float(m);
        for (int l = 0; l < 32; ++l)
            out[o++] = d1 * float((ql[l] & 0x0fu) + ((qh[l] & u1) ? 16 : 0)) - m1;
        for (int l = 0; l < 32; ++l)
            out[o++] = d2 * float((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
        ql += 32;
        is += 2;
        u1 = std::uint8_t(u1 << 2);
        u2 = std::uint8_t(u2 << 2);
    }
}

// Q6_K: 210 B per 256 values - 128 low-nibble bytes, 64 two-bit high bytes, 16 int8 scales, d.
inline void decode_ggml_q6_k(const std::uint8_t* block, std::span<float> out) {
    const float d = detail::half_at(block + 208);
    const std::uint8_t* ql = block;
    const std::uint8_t* qh = block + 128;
    const std::uint8_t* sc = block + 192;
    std::size_t o = 0;
    for (int n = 0; n < int(kGgmlQkK); n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = int((ql[l + 0] & 0x0fu) | (((qh[l] >> 0) & 3u) << 4)) - 32;
            const int q2 = int((ql[l + 32] & 0x0fu) | (((qh[l] >> 2) & 3u) << 4)) - 32;
            const int q3 = int((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3u) << 4)) - 32;
            const int q4 = int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3u) << 4)) - 32;
            out[o + l + 0]  = d * float(std::int8_t(sc[is + 0])) * float(q1);
            out[o + l + 32] = d * float(std::int8_t(sc[is + 2])) * float(q2);
            out[o + l + 64] = d * float(std::int8_t(sc[is + 4])) * float(q3);
            out[o + l + 96] = d * float(std::int8_t(sc[is + 6])) * float(q4);
        }
        o += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

// Q2_0: 18 B per 64 values - d (binary16) then sixteen two-bit codes, four per byte.
// The code is (q - 1): 00 = -1, 01 = 0, 10 = +1, 11 = +2.
inline void decode_ggml_q2_0(const std::uint8_t* block, std::span<float> out) {
    const float d = detail::half_at(block);
    for (std::size_t j = 0; j < kGgmlQk2_0; ++j) {
        const auto q = std::uint8_t((block[2 + j / 4] >> ((j % 4) * 2)) & 0x03u);
        out[j] = (float(q) - 1.0f) * d;
    }
}

// IQ4_NL: 18 B per 32 values - d then sixteen bytes holding two codebook indices each.
inline void decode_ggml_iq4_nl(const std::uint8_t* block, std::span<float> out) {
    const float d = detail::half_at(block);
    for (std::size_t j = 0; j < kGgmlQk4Nl / 2; ++j) {
        out[j]                = d * float(kIq4nlValues[block[2 + j] & 0x0fu]);
        out[j + kGgmlQk4Nl / 2] = d * float(kIq4nlValues[block[2 + j] >> 4]);
    }
}

// IQ4_XS: 136 B per 256 values - d, scales_h (binary16), 4 low-scale bytes, 128 code bytes.
// Eight 32-wide groups share d; each group's scale is a six-bit field biased by 32.
inline void decode_ggml_iq4_xs(const std::uint8_t* block, std::span<float> out) {
    const float d              = detail::half_at(block);
    const std::uint16_t high   = detail::load_u16(block + 2);
    const std::uint8_t* low    = block + 4;
    const std::uint8_t* qs     = block + 8;
    std::size_t o = 0;
    for (int ib = 0; ib < int(kGgmlQkK) / 32; ++ib) {
        const int ls = int((low[ib / 2] >> (4 * (ib % 2))) & 0x0fu) |
                       int(((high >> (2 * ib)) & 3u) << 4);
        const float dl = d * float(ls - 32);
        for (int j = 0; j < 16; ++j) {
            out[o + j]      = dl * float(kIq4nlValues[qs[j] & 0x0fu]);
            out[o + j + 16] = dl * float(kIq4nlValues[qs[j] >> 4]);
        }
        o += 32;
        qs += 16;
    }
}

// IQ2_XXS: 66 B per 256 values - d, then eight 8-byte chunks; each chunk holds four grid indices
// (one byte each) and a shared sign/scale word whose top nibble picks the group scale.
inline void decode_ggml_iq2_xxs(const std::uint8_t* block, std::span<float> out) {
    const float d = detail::half_at(block);
    const std::uint8_t* qs = block + 2;
    std::size_t o = 0;
    for (int ib32 = 0; ib32 < int(kGgmlQkK) / 32; ++ib32) {
        const std::uint8_t* aux8 = qs + 8 * ib32;
        const std::uint32_t aux1 = detail::load_u32(aux8 + 4);
        const float db = d * (0.5f + float(aux1 >> 28)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
            const auto* grid = reinterpret_cast<const std::uint8_t*>(&kIq2xxsGrid[aux8[l]]);
            const std::uint8_t signs = kIq2xsSigns[(aux1 >> (7 * l)) & 127u];
            for (int j = 0; j < 8; ++j) out[o + j] = db * float(grid[j]) * detail::sign_of(signs, j);
            o += 8;
        }
    }
}

// IQ2_XS: 74 B per 256 values - d, eight uint16 (9-bit grid index + 7-bit sign index), 8 scale
// bytes. A group of two 8-value grids shares one four-bit scale field.
inline void decode_ggml_iq2_xs(const std::uint8_t* block, std::span<float> out) {
    const float d = detail::half_at(block);
    const std::uint8_t* qs = block + 2;
    const std::uint8_t* scales = block + 66;
    std::size_t o = 0;
    for (int ib32 = 0; ib32 < int(kGgmlQkK) / 32; ++ib32) {
        const float db[2] = {d * (0.5f + float(scales[ib32] & 0x0fu)) * 0.25f,
                             d * (0.5f + float(scales[ib32] >> 4)) * 0.25f};
        for (int l = 0; l < 4; ++l) {
            const std::uint16_t w = detail::load_u16(qs + 2 * (4 * ib32 + l));
            const auto* grid = reinterpret_cast<const std::uint8_t*>(&kIq2xsGrid[w & 511u]);
            const std::uint8_t signs = kIq2xsSigns[w >> 9];
            for (int j = 0; j < 8; ++j)
                out[o + j] = db[l / 2] * float(grid[j]) * detail::sign_of(signs, j);
            o += 8;
        }
    }
}

// IQ2_S: 82 B per 256 values - d, 64 code bytes, 8 high-index bytes, 8 scale bytes, then the sign
// plane as the second half of the code area (qs + 32).
inline void decode_ggml_iq2_s(const std::uint8_t* block, std::span<float> out) {
    const float d = detail::half_at(block);
    const std::uint8_t* qs = block + 2;
    const std::uint8_t* qh = block + 66;
    const std::uint8_t* scales = block + 74;
    const std::uint8_t* signs0 = qs + 32;
    std::size_t o = 0;
    for (int ib32 = 0; ib32 < int(kGgmlQkK) / 32; ++ib32) {
        const float db[2] = {d * (0.5f + float(scales[ib32] & 0x0fu)) * 0.25f,
                             d * (0.5f + float(scales[ib32] >> 4)) * 0.25f};
        const std::uint8_t* q = qs + 4 * ib32;
        const std::uint8_t* s = signs0 + 4 * ib32;
        for (int l = 0; l < 4; ++l) {
            const std::uint32_t index =
                std::uint32_t(q[l]) | (std::uint32_t(qh[ib32] << (8 - 2 * l)) & 0x300u);
            const auto* grid = reinterpret_cast<const std::uint8_t*>(&kIq2sGrid[index]);
            for (int j = 0; j < 8; ++j)
                out[o + j] = db[l / 2] * float(grid[j]) * detail::sign_of(s[l], j);
            o += 8;
        }
    }
}

// IQ3_XXS: 98 B per 256 values - d, 64 bytes of grid indices (two per 8-value group), then eight
// 4-byte scale/sign words. One scale per 32 values; four sign indices each cover eight lanes.
inline void decode_ggml_iq3_xxs(const std::uint8_t* block, std::span<float> out) {
    const float d = detail::half_at(block);
    const std::uint8_t* qs = block + 2;
    const std::uint8_t* scales_and_signs = qs + 64;
    std::size_t o = 0;
    for (int ib32 = 0; ib32 < int(kGgmlQkK) / 32; ++ib32) {
        const std::uint32_t aux =
            detail::load_u32(scales_and_signs + 4 * ib32);
        const float db = d * (0.5f + float(aux >> 28)) * 0.5f;
        const std::uint8_t* q = qs + 8 * ib32;
        for (int l = 0; l < 4; ++l) {
            const std::uint8_t signs = kIq2xsSigns[(aux >> (7 * l)) & 127u];
            const auto* g1 = reinterpret_cast<const std::uint8_t*>(&kIq3xxsGrid[q[2 * l + 0]]);
            const auto* g2 = reinterpret_cast<const std::uint8_t*>(&kIq3xxsGrid[q[2 * l + 1]]);
            for (int j = 0; j < 4; ++j) {
                out[o + j]     = db * float(g1[j]) * detail::sign_of(signs, j);
                out[o + j + 4] = db * float(g2[j]) * detail::sign_of(signs, j + 4);
            }
            o += 8;
        }
    }
}

// IQ3_S: 110 B per 256 values - d, 64 code bytes, 8 high-index bytes, 32 sign bytes, 4 scale bytes.
// Each scale byte covers 64 values: low nibble for the first 32, high nibble for the second.
inline void decode_ggml_iq3_s(const std::uint8_t* block, std::span<float> out) {
    const float d = detail::half_at(block);
    const std::uint8_t* qs = block + 2;
    const std::uint8_t* qh = block + 66;
    const std::uint8_t* signs = block + 74;
    const std::uint8_t* scales = block + 106;
    std::size_t o = 0;
    for (int ib32 = 0; ib32 < int(kGgmlQkK) / 32; ib32 += 2) {
        const float db1 = d * float(1 + 2 * int(scales[ib32 / 2] & 0x0fu));
        const float db2 = d * float(1 + 2 * int(scales[ib32 / 2] >> 4));
        const std::uint8_t* q = qs + 8 * ib32;
        const std::uint8_t* s = signs + 4 * ib32;
        // The high-index plane advances one byte per 32-value group, so a pair consumes two:
        // `half` selects within the pair, `ib32` locates the pair.
        const std::uint8_t* h = qh + ib32;
        for (int half = 0; half < 2; ++half) {
            const float db = half == 0 ? db1 : db2;
            for (int l = 0; l < 4; ++l) {
                const std::uint32_t i1 =
                    std::uint32_t(q[2 * l + 0]) | (std::uint32_t(h[half] << (8 - 2 * l)) & 256u);
                const std::uint32_t i2 =
                    std::uint32_t(q[2 * l + 1]) | (std::uint32_t(h[half] << (7 - 2 * l)) & 256u);
                const auto* g1 = reinterpret_cast<const std::uint8_t*>(&kIq3sGrid[i1]);
                const auto* g2 = reinterpret_cast<const std::uint8_t*>(&kIq3sGrid[i2]);
                for (int j = 0; j < 4; ++j) {
                    out[o + j]     = db * float(g1[j]) * detail::sign_of(s[l], j);
                    out[o + j + 4] = db * float(g2[j]) * detail::sign_of(s[l], j + 4);
                }
                o += 8;
            }
            q += 8;
            s += 4;
        }
    }
}

// Decoded values for `elements` weights starting at `data`, for an encoding whose block size is
// `block` and whose encoded block is `type_bytes` wide. Refuses a mismatched pairing instead of
// decoding past the payload, because the caller's geometry and this table must agree.
inline void decode_ggml(const std::uint8_t* data, std::uint64_t elements, std::uint64_t block,
                        std::uint64_t type_bytes, void (*decode)(const std::uint8_t*,
                                                                 std::span<float>),
                        std::span<float> out) {
    if (block == 0 || elements % block != 0 || out.size() < elements) {
        throw std::invalid_argument("ggml decode: element count does not cover whole blocks");
    }
    const std::size_t count = std::size_t(elements / block);
    for (std::size_t b = 0; b < count; ++b) {
        decode(data + b * type_bytes, out.subspan(b * block, block));
    }
}

// Elements per block and encoded bytes per block live in core/ggml_types.h; this header owns the
// decoding algorithms only.

} // namespace ninfer::core
