#pragma once

// Device decode of the GGML block encodings, one function per type, in the block order the container
// stores (layout = ggml_block_v1).
//
// These are the kernels' own copies of what src/core/ggml_dequant.h states for the Host. The host
// reference is the arbiter: tests/ops/test_ggml_block_decode.cu decodes the same real blocks on the
// Device and compares, and the host side is itself pinned against gguf-py. Keeping the two written
// separately is deliberate - a single shared implementation would only prove it agrees with itself.
//
// The codebooks are the same generated file, instantiated here as Device constant data inside this
// namespace so a translation unit can hold both the host and the device tables.

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail::ggml {

#define NINFER_GGML_TABLE(type, name, count) static __device__ const type name[count] = {
#define NINFER_GGML_TABLE_END() };
#include "core/ggml_codebook.inc"
#undef NINFER_GGML_TABLE
#undef NINFER_GGML_TABLE_END

inline constexpr int kQkK   = 256;
inline constexpr int kQk4Nl = 32;
inline constexpr int kQk2_0 = 64;

__device__ __forceinline__ float half_to_float(std::uint16_t bits) noexcept {
    const std::uint32_t sign = std::uint32_t(bits & 0x8000u) << 16;
    const std::uint32_t exp  = (bits >> 10) & 0x1fu;
    const std::uint32_t mant = bits & 0x3ffu;
    if (exp == 0) {
        if (mant == 0) { return __int_as_float(int(sign)); }
        int shift       = -1;
        std::uint32_t m = mant;
        do {
            ++shift;
            m <<= 1;
        } while ((m & 0x400u) == 0);
        return __int_as_float(
            int(sign | (std::uint32_t(127 - 15 - shift) << 23) | ((m & 0x3ffu) << 13)));
    }
    if (exp == 31) { return __int_as_float(int(sign | 0x7f800000u | (mant << 13))); }
    return __int_as_float(int(sign | ((exp + 127 - 15) << 23) | (mant << 13)));
}

__device__ __forceinline__ std::uint16_t load_u16(const std::uint8_t* p) noexcept {
    return std::uint16_t(p[0]) | std::uint16_t(p[1]) << 8;
}

__device__ __forceinline__ std::uint32_t load_u32(const std::uint8_t* p) noexcept {
    return std::uint32_t(p[0]) | std::uint32_t(p[1]) << 8 | std::uint32_t(p[2]) << 16 |
           std::uint32_t(p[3]) << 24;
}

__device__ __forceinline__ float half_at(const std::uint8_t* p) noexcept {
    return half_to_float(load_u16(p));
}

__device__ __forceinline__ void scale_min_k4(int j, const std::uint8_t* q, int& d,
                                             int& m) noexcept {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

__device__ __forceinline__ float sign_of(std::uint8_t signs, int j) noexcept {
    return (signs & kIq2xsMask[j]) != 0 ? -1.0f : 1.0f;
}

// One grid entry packs eight lanes, low lane first.
__device__ __forceinline__ float grid8(std::uint64_t entry, int lane) noexcept {
    return float(std::uint8_t(entry >> (8 * lane)));
}

__device__ __forceinline__ float grid4(std::uint32_t entry, int lane) noexcept {
    return float(std::uint8_t(entry >> (8 * lane)));
}

__device__ inline void decode_q4_k(const std::uint8_t* block, float* out) {
    const float d              = half_at(block);
    const float min            = half_at(block + 2);
    const std::uint8_t* scales = block + 4;
    const std::uint8_t* q      = block + 16;
    int is = 0;
    int o  = 0;
    for (int j = 0; j < kQkK; j += 64) {
        int sc = 0, m = 0;
        scale_min_k4(is + 0, scales, sc, m);
        const float d1 = d * float(sc), m1 = min * float(m);
        scale_min_k4(is + 1, scales, sc, m);
        const float d2 = d * float(sc), m2 = min * float(m);
#pragma unroll
        for (int l = 0; l < 32; ++l) out[o + l] = d1 * float(q[l] & 0x0f) - m1;
#pragma unroll
        for (int l = 0; l < 32; ++l) out[o + 32 + l] = d2 * float(q[l] >> 4) - m2;
        o += 64;
        q += 32;
        is += 2;
    }
}

__device__ inline void decode_q5_k(const std::uint8_t* block, float* out) {
    const float d              = half_at(block);
    const float min            = half_at(block + 2);
    const std::uint8_t* scales = block + 4;
    const std::uint8_t* qh     = block + 16;
    const std::uint8_t* ql     = block + 48;
    int is = 0;
    int u1 = 1, u2 = 2;
    int o = 0;
    for (int j = 0; j < kQkK; j += 64) {
        int sc = 0, m = 0;
        scale_min_k4(is + 0, scales, sc, m);
        const float d1 = d * float(sc), m1 = min * float(m);
        scale_min_k4(is + 1, scales, sc, m);
        const float d2 = d * float(sc), m2 = min * float(m);
#pragma unroll
        for (int l = 0; l < 32; ++l)
            out[o + l] = d1 * float((ql[l] & 0x0f) + ((qh[l] & u1) ? 16 : 0)) - m1;
#pragma unroll
        for (int l = 0; l < 32; ++l)
            out[o + 32 + l] = d2 * float((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
        o += 64;
    }
}

__device__ inline void decode_q6_k(const std::uint8_t* block, float* out) {
    const float d          = half_at(block + 208);
    const std::uint8_t* ql = block;
    const std::uint8_t* qh = block + 128;
    const std::uint8_t* sc = block + 192;
    int o = 0;
    for (int n = 0; n < kQkK; n += 128) {
#pragma unroll
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = int((ql[l + 0] & 0x0f) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = int((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = int((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
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

__device__ inline void decode_q2_0(const std::uint8_t* block, float* out) {
    const float d = half_at(block);
#pragma unroll
    for (int j = 0; j < kQk2_0; ++j) {
        const auto q = std::uint8_t((block[2 + j / 4] >> ((j % 4) * 2)) & 0x03u);
        out[j]       = (float(q) - 1.0f) * d;
    }
}

__device__ inline void decode_iq4_nl(const std::uint8_t* block, float* out) {
    const float d = half_at(block);
#pragma unroll
    for (int j = 0; j < kQk4Nl / 2; ++j) {
        out[j]              = d * float(kIq4nlValues[block[2 + j] & 0x0fu]);
        out[j + kQk4Nl / 2] = d * float(kIq4nlValues[block[2 + j] >> 4]);
    }
}

__device__ inline void decode_iq4_xs(const std::uint8_t* block, float* out) {
    const float d            = half_at(block);
    const std::uint32_t high = load_u16(block + 2);
    const std::uint8_t* low  = block + 4;
    const std::uint8_t* qs   = block + 8;
    int o = 0;
    for (int ib = 0; ib < kQkK / 32; ++ib) {
        const int ls = int((low[ib / 2] >> (4 * (ib % 2))) & 0x0f) |
                       int(((high >> (2 * ib)) & 3u) << 4);
        const float dl = d * float(ls - 32);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            out[o + j]      = dl * float(kIq4nlValues[qs[j] & 0x0f]);
            out[o + j + 16] = dl * float(kIq4nlValues[qs[j] >> 4]);
        }
        o += 32;
        qs += 16;
    }
}

__device__ inline void decode_iq2_xxs(const std::uint8_t* block, float* out) {
    const float d          = half_at(block);
    const std::uint8_t* qs = block + 2;
    int o = 0;
    for (int ib32 = 0; ib32 < kQkK / 32; ++ib32) {
        const std::uint8_t* aux8 = qs + 8 * ib32;
        const std::uint32_t aux1 = load_u32(aux8 + 4);
        const float db           = d * (0.5f + float(aux1 >> 28)) * 0.25f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const std::uint64_t entry = kIq2xxsGrid[aux8[l]];
            const std::uint8_t signs  = kIq2xsSigns[(aux1 >> (7 * l)) & 127u];
#pragma unroll
            for (int j = 0; j < 8; ++j) out[o + j] = db * grid8(entry, j) * sign_of(signs, j);
            o += 8;
        }
    }
}

__device__ inline void decode_iq2_xs(const std::uint8_t* block, float* out) {
    const float d              = half_at(block);
    const std::uint8_t* qs     = block + 2;
    const std::uint8_t* scales = block + 66;
    int o = 0;
    for (int ib32 = 0; ib32 < kQkK / 32; ++ib32) {
        const float db[2] = {d * (0.5f + float(scales[ib32] & 0x0f)) * 0.25f,
                             d * (0.5f + float(scales[ib32] >> 4)) * 0.25f};
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const std::uint32_t w    = load_u16(qs + 2 * (4 * ib32 + l));
            const std::uint64_t e    = kIq2xsGrid[w & 511u];
            const std::uint8_t signs = kIq2xsSigns[w >> 9];
#pragma unroll
            for (int j = 0; j < 8; ++j) out[o + j] = db[l / 2] * grid8(e, j) * sign_of(signs, j);
            o += 8;
        }
    }
}

__device__ inline void decode_iq2_s(const std::uint8_t* block, float* out) {
    const float d              = half_at(block);
    const std::uint8_t* qs     = block + 2;
    const std::uint8_t* qh     = block + 66;
    const std::uint8_t* scales = block + 74;
    const std::uint8_t* signs0 = qs + 32;
    int o = 0;
    for (int ib32 = 0; ib32 < kQkK / 32; ++ib32) {
        const float db[2] = {d * (0.5f + float(scales[ib32] & 0x0f)) * 0.25f,
                             d * (0.5f + float(scales[ib32] >> 4)) * 0.25f};
        const std::uint8_t* q = qs + 4 * ib32;
        const std::uint8_t* s = signs0 + 4 * ib32;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const std::uint32_t index =
                std::uint32_t(q[l]) | (std::uint32_t(qh[ib32] << (8 - 2 * l)) & 0x300u);
            const std::uint64_t e = kIq2sGrid[index];
#pragma unroll
            for (int j = 0; j < 8; ++j) out[o + j] = db[l / 2] * grid8(e, j) * sign_of(s[l], j);
            o += 8;
        }
    }
}

__device__ inline void decode_iq3_xxs(const std::uint8_t* block, float* out) {
    const float d                        = half_at(block);
    const std::uint8_t* qs               = block + 2;
    const std::uint8_t* scales_and_signs = qs + 64;
    int o = 0;
    for (int ib32 = 0; ib32 < kQkK / 32; ++ib32) {
        const std::uint32_t aux = load_u32(scales_and_signs + 4 * ib32);
        const float db          = d * (0.5f + float(aux >> 28)) * 0.5f;
        const std::uint8_t* q   = qs + 8 * ib32;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const std::uint8_t signs = kIq2xsSigns[(aux >> (7 * l)) & 127u];
            const std::uint32_t g1   = kIq3xxsGrid[q[2 * l + 0]];
            const std::uint32_t g2   = kIq3xxsGrid[q[2 * l + 1]];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                out[o + j]     = db * grid4(g1, j) * sign_of(signs, j);
                out[o + j + 4] = db * grid4(g2, j) * sign_of(signs, j + 4);
            }
            o += 8;
        }
    }
}

__device__ inline void decode_iq3_s(const std::uint8_t* block, float* out) {
    const float d               = half_at(block);
    const std::uint8_t* qs      = block + 2;
    const std::uint8_t* qh      = block + 66;
    const std::uint8_t* signs   = block + 74;
    const std::uint8_t* scales  = block + 106;
    int o = 0;
    for (int ib32 = 0; ib32 < kQkK / 32; ib32 += 2) {
        const float db1 = d * float(1 + 2 * int(scales[ib32 / 2] & 0x0f));
        const float db2 = d * float(1 + 2 * int(scales[ib32 / 2] >> 4));
        const std::uint8_t* q = qs + 8 * ib32;
        const std::uint8_t* s = signs + 4 * ib32;
        const std::uint8_t* h = qh + ib32;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const float db = half == 0 ? db1 : db2;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const std::uint32_t i1 =
                    std::uint32_t(q[2 * l + 0]) | (std::uint32_t(h[half] << (8 - 2 * l)) & 256u);
                const std::uint32_t i2 =
                    std::uint32_t(q[2 * l + 1]) | (std::uint32_t(h[half] << (7 - 2 * l)) & 256u);
                const std::uint32_t g1 = kIq3sGrid[i1];
                const std::uint32_t g2 = kIq3sGrid[i2];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    out[o + j]     = db * grid4(g1, j) * sign_of(s[l], j);
                    out[o + j + 4] = db * grid4(g2, j) * sign_of(s[l], j + 4);
                }
                o += 8;
            }
            q += 8;
            s += 4;
        }
    }
}

} // namespace ninfer::ops::detail::ggml
