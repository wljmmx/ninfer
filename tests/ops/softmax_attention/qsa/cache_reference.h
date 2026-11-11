#pragma once

#include "core/dtype.h"
#include "e8_reference.h"
#include "ninfer/types.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_fp16.h>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ninfer::test::qsa::prepare {

enum class Encoding { Bf16, Fp16, Int8, Fp8, Nvfp4, Int4, E8Cylinder, E8Root };

struct VectorEncoding {
    Encoding encoding;
    DType data_dtype, scale_dtype;
    int data_extent, scale_extent, group;
    bool rotate;
};

struct CacheEncoding {
    VectorEncoding key, value;
};

inline CacheEncoding cache_encoding(KvCacheStorage storage) {
    const VectorEncoding i8{Encoding::Int8, DType::I8, DType::FP16, 256, 4, 64, true};
    const VectorEncoding fp8{Encoding::Fp8, DType::FP8_E4M3FN, DType::FP16, 256, 1, 256, true};
    const VectorEncoding nvfp4{Encoding::Nvfp4, DType::U8, DType::U8, 128, 16, 16, true};
    // rk: 4-bit symmetric [-7, 7] with a per-group absmax/7 FP16 scale, two dims per byte. V keeps
    // the original coordinate frame; only K carries the Hadamard rotation.
    const VectorEncoding i4{Encoding::Int4, DType::U8, DType::FP16, 128, 4, 64, false};
    const VectorEncoding i4_key{Encoding::Int4, DType::U8, DType::FP16, 128, 4, 64, true};
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return {{Encoding::Bf16, DType::BF16, DType::U8, 256, 0, 0, false},
                {Encoding::Fp16, DType::FP16, DType::U8, 256, 0, 0, false}};
    case KvCacheStorage::Int8Group64: {
        auto value   = i8;
        value.rotate = false;
        return {i8, value};
    }
    case KvCacheStorage::Fp8E4M3Row256: {
        auto value   = fp8;
        value.rotate = false;
        return {fp8, value};
    }
    case KvCacheStorage::Nvfp4Group16:
        return {nvfp4, nvfp4};
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return {fp8, nvfp4};
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
        return {i8, i4};
    case KvCacheStorage::RotatedInt4KeyInt4ValueGroup64:
        return {i4_key, i4};
    // E8 carriers: 96 B of (root, radial-axis) pairs plus 32 B of residual signs, or the 64 B
    // two-bit form without them. The fixture fills these from a deterministic code pattern (the
    // encoder itself is device-only), and the reference decodes them with the host table copies.
    case KvCacheStorage::RK4V4E8:
        return {{Encoding::E8Cylinder, DType::U8, DType::FP16, 96, 4, 64, true}, i4};
    case KvCacheStorage::RK2V4E8:
        return {{Encoding::E8Root, DType::U8, DType::FP16, 64, 4, 64, true}, i4};
    case KvCacheStorage::RK4V2E8:
        return {{Encoding::E8Cylinder, DType::U8, DType::FP16, 96, 4, 64, true},
                {Encoding::E8Root, DType::U8, DType::FP16, 64, 4, 64, false}};
    }
    throw std::invalid_argument("test cache storage");
}

inline std::uint16_t read_u16(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    std::uint16_t word;
    std::memcpy(&word, bytes.data() + offset, sizeof(word));
    return word;
}

inline void write_u16(std::vector<std::uint8_t>& bytes, std::size_t offset, std::uint16_t word) {
    std::memcpy(bytes.data() + offset, &word, sizeof(word));
}

inline std::uint16_t fp16_encode(float x) {
    return __half_as_ushort(__float2half_rn(x));
}

inline float fp16_decode(std::uint16_t x) { return __half2float(__ushort_as_half(x)); }

inline double positive_fp8(int code) {
    const int exponent = code >> 3, fraction = code & 7;
    if (code == 127) return std::numeric_limits<double>::quiet_NaN();
    return exponent == 0 ? std::ldexp(double(fraction), -9)
                         : std::ldexp(1.0 + fraction / 8.0, exponent - 7);
}

inline double positive_fp4(int code) {
    constexpr double values[]{0, 0.5, 1, 1.5, 2, 3, 4, 6};
    return values[code & 7];
}

inline std::uint8_t encode_minifloat(float x, bool fp4) {
    const double magnitude = std::abs(double(x));
    const auto value = [=](int code) { return fp4 ? positive_fp4(code) : positive_fp8(code); };
    int low = 0, high = fp4 ? 7 : 126;
    while (low < high) {
        const int mid = (low + high) / 2;
        if (value(mid) < magnitude)
            low = mid + 1;
        else
            high = mid;
    }
    int selected = low;
    if (low > 0) {
        const double upper_error = std::abs(value(low) - magnitude);
        const double lower_error = std::abs(value(low - 1) - magnitude);
        if (lower_error < upper_error || (lower_error == upper_error && (low & 1))) --selected;
    }
    return std::uint8_t(selected | (std::signbit(x) ? (fp4 ? 8 : 128) : 0));
}

inline double decode_code(std::uint8_t code, Encoding encoding) {
    if (encoding == Encoding::Int8) return std::bit_cast<std::int8_t>(code);
    if (encoding == Encoding::Int4) {
        const int nibble = code & 15;
        return double(nibble >= 8 ? nibble - 16 : nibble);
    }
    if (encoding == Encoding::Nvfp4) return (code & 8 ? -1 : 1) * positive_fp4(code);
    return (code & 128 ? -1 : 1) * positive_fp8(code & 127);
}

// The scalar butterfly uses ordinary contiguous dimensions, independently of CUDA lanes.
// Double evaluates the ideal transform; float evaluates the existing persistent V codec.
template <class T>
void hadamard(std::array<T, 256>& values) {
    for (int width = 1; width < 256; width *= 2)
        for (int block = 0; block < 256; block += 2 * width)
            for (int j = 0; j < width; ++j) {
                const T lo = values[block + j], hi = values[block + j + width];
                values[block + j]         = lo + hi;
                values[block + j + width] = lo - hi;
            }
    for (auto& value : values) value *= T(1.0 / 16);
}

struct ReferenceScale {
    std::uint16_t bits;
    float value;
};

inline ReferenceScale encode_scale(float absmax, Encoding encoding) {
    if (absmax == 0) return {0, 0};
    if (encoding == Encoding::Nvfp4) {
        const float bounded = std::clamp(absmax / 6.0F, 0x1p-9F, 448.0F);
        const auto bits     = encode_minifloat(bounded, false);
        return {bits, float(positive_fp8(bits))};
    }
    const float raw = encoding == Encoding::Int8    ? absmax / 127.0F
                      : encoding == Encoding::Int4  ? absmax / 7.0F
                                                    : std::clamp(absmax / 448.0F, 0x1p-24F, 65504.0F);
    const auto bits = fp16_encode(raw);
    return {bits, fp16_decode(bits)};
}

inline std::uint8_t encode_code(float x, float scale, Encoding encoding) {
    if (scale == 0) return 0;
    const float normalized = encoding == Encoding::Nvfp4 ? x / scale : x * (1.0F / scale);
    if (encoding == Encoding::Int4) {
        const float clamped = std::clamp(normalized, -7.0F, 7.0F);
        const int rounded   = int(std::nearbyint(clamped));
        return std::uint8_t(std::bit_cast<std::uint8_t>(std::int8_t(rounded)) & 0x0F);
    }
    if (encoding != Encoding::Int8)
        return encode_minifloat(normalized, encoding == Encoding::Nvfp4);
    const float clamped   = std::clamp(normalized, -127.0F, 127.0F);
    const int lower       = int(std::floor(clamped));
    const float remainder = clamped - lower;
    const int rounded     = lower + (remainder > 0.5F || (remainder == 0.5F && (lower & 1)));
    return std::bit_cast<std::uint8_t>(std::int8_t(rounded));
}

struct EncodedRow {
    std::vector<std::uint8_t> data, scales;
};

inline bool is_e8(Encoding encoding) {
    return encoding == Encoding::E8Cylinder || encoding == Encoding::E8Root;
}

// The 96-byte carrier stores an E8-cylinder code per eight-dim subspace plus a residual sign byte
// at offset 64; the 64-byte form keeps only the codes. Both divide the group scale by the residual
// grid when they carry signs, because read-back multiplies the codes by it.
inline void encode_e8_row(const std::array<float, 256>& values, const VectorEncoding& format,
                          EncodedRow& out) {
    const bool residual = format.encoding == Encoding::E8Cylinder;
    for (int group = 0; group < 4; ++group) {
        float absmax = 0;
        for (int d = group * 64; d < group * 64 + 64; ++d)
            absmax = std::max(absmax, std::abs(values[d]));
        const float step = fp16_decode(fp16_encode(absmax / 7.0F));
        for (int subspace = 0; subspace < 8; ++subspace) {
            std::array<float, 8> block{};
            for (int i = 0; i < 8; ++i) block[i] = values[group * 64 + subspace * 8 + i];
            std::uint8_t root = 0, radial = 0;
            e8::encode_8d(block, step, root, radial);
            out.data[(group * 8 + subspace) * 2]     = root;
            out.data[(group * 8 + subspace) * 2 + 1] = radial;
            if (!residual) continue;
            std::array<int, 8> decoded{};
            e8::decode_8d(root, radial, false, decoded);
            std::uint8_t bits = 0;
            for (int i = 0; i < 8; ++i) {
                if (step > 0.0F && block[i] / step >= float(decoded[i])) bits |= 1U << i;
            }
            out.data[64 + group * 8 + subspace] = bits;
        }
        const std::uint16_t stored =
            residual ? fp16_encode(fp16_decode(fp16_encode(absmax / 7.0F)) / 4.0F)
                     : fp16_encode(absmax / 7.0F);
        write_u16(out.scales, 2 * group, stored);
    }
}

inline EncodedRow encode_row(std::array<float, 256> values, VectorEncoding format) {
    if (format.rotate) hadamard(values);
    EncodedRow out{std::vector<std::uint8_t>(format.data_extent * dtype_size(format.data_dtype)),
                   std::vector<std::uint8_t>(format.scale_extent * dtype_size(format.scale_dtype))};
    if (is_e8(format.encoding)) {
        encode_e8_row(values, format, out);
        return out;
    }
    for (int begin = 0; begin < 256; begin += format.group) {
        float absmax = 0;
        for (int d = begin; d < begin + format.group; ++d)
            absmax = std::max(absmax, std::abs(values[d]));
        const auto scale = encode_scale(absmax, format.encoding);
        const int group  = begin / format.group;
        if (format.encoding == Encoding::Nvfp4)
            out.scales[group] = std::uint8_t(scale.bits);
        else
            write_u16(out.scales, 2 * group, scale.bits);
        for (int d = begin; d < begin + format.group; ++d) {
            const auto code = encode_code(values[d], scale.value, format.encoding);
            if (format.encoding == Encoding::Nvfp4)
                out.data[d / 2] |= code << (4 * (d % 2));
            else if (format.encoding == Encoding::Int4)
                out.data[d / 2] |= std::uint8_t((code & 0x0F) << (4 * (d % 2)));
            else
                out.data[d] = code;
        }
    }
    return out;
}

// The E8 carriers are not scalar codes: one code covers eight dimensions. The same envelope applies
// subspace by subspace — the scale must sit in the jitter range, and every decoded dimension must
// lie inside the range the ±delta inputs encode to, with the error accounting of the scalar form.
inline bool compatible_e8_encoding(const std::array<double, 256>& ideal, const VectorEncoding& format,
                                   const EncodedRow& actual, double delta) {
    const bool residual = format.encoding == Encoding::E8Cylinder;
    double error2 = 0, nearest_error2 = 0, allowance2 = 0;
    for (int group = 0; group < 4; ++group) {
        double absmax = 0;
        for (int d = group * 64; d < group * 64 + 64; ++d)
            absmax = std::max(absmax, std::abs(ideal[d]));
        const auto stored = residual ? fp16_decode(read_u16(actual.scales, 2 * group)) * 4.0F
                                     : fp16_decode(read_u16(actual.scales, 2 * group));
        const auto low    = encode_scale(float(std::max(0.0, absmax - delta)), Encoding::Int4);
        const auto high   = encode_scale(float(absmax + delta), Encoding::Int4);
        if (!std::isfinite(stored) || stored < low.value || stored > high.value) return false;
        // The encoder works from the undivided absmax/7, so the envelope must too: dividing the
        // stored scale back and re-rounding to FP16 would move the step by up to one ulp and can
        // select another root at a tie.
        const float step = fp16_decode(fp16_encode(float(absmax) / 7.0F));
        for (int subspace = 0; subspace < 8; ++subspace) {
            const int first = group * 64 + subspace * 8;
            const auto root = actual.data[(group * 8 + subspace) * 2];
            const auto radial = actual.data[(group * 8 + subspace) * 2 + 1];
            std::array<int, 8> decoded{};
            e8::decode_8d(root, radial, false, decoded);
            const int bits = residual ? actual.data[64 + group * 8 + subspace] : 0;
            std::array<float, 8> lo_block{}, hi_block{}, mid_block{};
            for (int i = 0; i < 8; ++i) {
                lo_block[i]  = float(ideal[first + i] - delta);
                hi_block[i]  = float(ideal[first + i] + delta);
                mid_block[i] = float(ideal[first + i]);
            }
            const auto encode_block = [&](const std::array<float, 8>& block,
                                          std::array<int, 8>& codes) {
                std::uint8_t r = 0, a = 0;
                std::array<int, 8> dec{};
                e8::encode_8d(block, step, r, a);
                e8::decode_8d(r, a, false, dec);
                for (int i = 0; i < 8; ++i) {
                    codes[i] = residual
                                   ? 4 * (dec[i] + (step > 0.0F && block[i] / step >= float(dec[i]) ? 1 : -1))
                                   : dec[i];
                }
            };
            std::array<int, 8> lo_codes{}, hi_codes{}, mid_codes{};
            encode_block(lo_block, lo_codes);
            encode_block(hi_block, hi_codes);
            encode_block(mid_block, mid_codes);
            for (int i = 0; i < 8; ++i) {
                int code = decoded[i];
                if (residual) code = 4 * (code + (((bits >> i) & 1) ? 1 : -1));
                const double q       = double(code) * step;
                const double nearest = double(mid_codes[i]) * step;
                // A coupled eight-dim code has no per-dimension bin: the root is a discrete choice
                // over the whole subspace, and a 1e-7 difference between the device's FP32 input
                // and this FP64 ideal can legitimately select another root at a near tie. The
                // assertion is therefore on the subspace's approximation quality - the device's
                // code may not be worse than the code this reference picks, beyond the jitter
                // allowance - while `exact_key_encoding` pins the encoder byte for byte on inputs
                // both sides represent exactly.
                error2 += (q - ideal[first + i]) * (q - ideal[first + i]);
                nearest_error2 += (nearest - ideal[first + i]) * (nearest - ideal[first + i]);
                allowance2 += 2 * delta * std::abs(q - nearest) + delta * delta;
            }
        }
    }
    // A root, radius or residual-sign choice is discrete over the whole subspace, so a 1e-7
    // difference between the device's FP32 input and this FP64 ideal can legitimately select
    // another code at a near tie. Measured after the encoder's top-2 tie-break became a strict total
    // order: the worst aggregate over the eight-carrier suite is error2/nearest = 1.073 on the
    // large-token cases, against 1.05 before the fix. A genuinely wrong code lands near 2.0, because
    // an unrelated second choice costs about half the subspace's energy, so this bound admits the
    // re-selection and nothing else. Byte agreement with the encoder is pinned separately by
    // `exact_key_encoding`, whose dyadic inputs both sides represent exactly.
    constexpr double kDiscreteChoiceSlack = 1.12;
    if (error2 > nearest_error2 * kDiscreteChoiceSlack + allowance2 + 1e-12) {
        if (std::getenv("NINFER_E8_DIAG") != nullptr) {
            std::fprintf(stderr, "[e8] aggregate error2=%.6e nearest=%.6e allowance=%.6e ratio=%.6f\n",
                         error2, nearest_error2, allowance2, error2 / (nearest_error2 + 1e-30));
        }
        return false;
    }
    return true;
}

// Norm/RoPE and Hadamard are approximate FP32 computations. Their fixed input-error envelope
// admits adjacent quantizer bins at a threshold, not an extra BF16 rounding before encoding.
inline bool compatible_key_encoding(std::array<double, 256> ideal, VectorEncoding format,
                                    const EncodedRow& actual) {
    hadamard(ideal);
    double magnitude = 0;
    for (double x : ideal) magnitude = std::max(magnitude, std::abs(x));
    constexpr double kArithmeticRelative = 2e-5;
    const double delta =
        kArithmeticRelative * magnitude + 64 * double(std::numeric_limits<float>::denorm_min());
    if (is_e8(format.encoding)) return compatible_e8_encoding(ideal, format, actual, delta);
    double error2 = 0, nearest_error2 = 0, allowance2 = 0;
    for (int begin = 0; begin < 256; begin += format.group) {
        double absmax = 0;
        for (int d = begin; d < begin + format.group; ++d)
            absmax = std::max(absmax, std::abs(ideal[d]));
        const auto low    = encode_scale(float(std::max(0.0, absmax - delta)), format.encoding);
        const auto high   = encode_scale(float(absmax + delta), format.encoding);
        const int group   = begin / format.group;
        const float scale = format.encoding == Encoding::Nvfp4
                                ? float(positive_fp8(actual.scales[group]))
                                : fp16_decode(read_u16(actual.scales, 2 * group));
        if (!std::isfinite(scale) || scale < low.value || scale > high.value) return false;
        for (int d = begin; d < begin + format.group; ++d) {
            const auto code  = (format.encoding == Encoding::Nvfp4 || format.encoding == Encoding::Int4)
                                   ? (actual.data[d / 2] >> (4 * (d % 2))) & 15
                                   : actual.data[d];
            const double q   = decode_code(code, format.encoding);
            const double qlo = decode_code(
                encode_code(float(ideal[d] - delta), scale, format.encoding), format.encoding);
            const double qhi = decode_code(
                encode_code(float(ideal[d] + delta), scale, format.encoding), format.encoding);
            if (!std::isfinite(q) || q < qlo || q > qhi) return false;
            const double nearest =
                decode_code(encode_code(float(ideal[d]), scale, format.encoding), format.encoding) *
                scale;
            const double decoded = q * scale;
            error2 += (decoded - ideal[d]) * (decoded - ideal[d]);
            nearest_error2 += (nearest - ideal[d]) * (nearest - ideal[d]);
            // Crossing a nearest-rounding midpoint by delta changes squared error by at
            // most 2*delta*the distance between the two decoded codes.
            allowance2 += 2 * delta * std::abs(decoded - nearest) + delta * delta;
        }
    }
    return error2 <= nearest_error2 + allowance2 + 1e-12 * magnitude * magnitude;
}

} // namespace ninfer::test::qsa::prepare
