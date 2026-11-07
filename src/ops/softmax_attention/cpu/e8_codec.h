#pragma once

// port (sm_89): CPU-side decode substrate for the E8 lattice/root/cylinder K-V codecs of the
// rk -e8 KV-cache family (rk4v4-e8 / rk2v4-e8 / rk4v2-e8), the P4 extension of decode.h's
// int4/int8 substrate. Header-only, host-only, standalone — mirrors the E8 device codec in
// ops/kernel/e8_root_codec.cuh (tables + e8_root_decode_8d_fast/_v + rk4_apply_e8_residual)
// and the plane geometry in ops/kv_cache/rk4_codec.cuh (kRkE8KLatticeExtent, FracGrid/Theta)
// and core/paged_kv_storage.h (RK4V4E8 / RK2V4E8 / RK4V2E8 extents).
//
// Codec semantics (all mirrored bit-exactly; see the audit comment at each definition):
//   * Every 8D subspace of a row stores one (root_code, rad_axis_code) byte pair: root at
//     plane byte 2*s, rad_axis at 2*s+1 (append writer rk_kernel.cuh:74-93/143-156, read-back
//     grouped_mma.cuh:549-577/628-638).
//   * Decode (e8_root_codec.cuh:712-767): rad_idx = rad_axis>>4; 0 -> eight zero codes;
//     otherwise int8 root vector (stage-1 table) + int8 unit axis (axis table), saturating add
//     (vadd4.sat on sm_89), times scale = c_radius_scale[rad_idx] * radius_mult, RNE-rounded
//     to int8. The K plane and every stock consumer use radius_mult = 1.0; the rk4v2-e8 V
//     plane is calibrated per-root: (root < 112) ? 1.0 : kE8VRadiusCentreB.
//   * The rk4v4-e8 / rk4v2-e8 K plane additionally carries one residual sign byte per
//     subspace at plane byte 64+s (32 B for 32 subspaces; writer rk_kernel.cuh:168-185,
//     reader grouped_mma.cuh:726-733), applied per element as bit i -> dim i of the decoded
//     subspace: with the shipped kRkE8KFracTheta == kRkE8KFracGrid the reconstruction is
//     code' = Grid * (code +- 1) with the group scale stored as absmax/7/Grid (writer
//     rk_kernel.cuh:215-218), so the decoded value stays code' * stored_scale — the decode
//     contract does not care that the scale was pre-divided.
//
// Numeric exactness contract (same class as decode.h): every arithmetic step here is the
// integer/fp32 operation the device takes — table lookup (integer), saturating int8 add
// (integer), one fp32 multiply, one RNE float->int32 round, and the row-level code*scale
// product that decode.h already proves exact. No fp16/bf16 rounding happens here.

#include "ops/softmax_attention/cpu/decode.h"

#include <array>
#include <cstdint>

#if defined(__AVX2__) && !defined(NINFER_SOFTMAX_CPU_FORCE_SCALAR)
#define NINFER_SOFTMAX_CPU_E8_VECTOR 1
#else
// Scalar fallback is bit-identical (see above); forced off via the same A/B knob as decode.h.
#define NINFER_SOFTMAX_CPU_E8_VECTOR 0
#endif

namespace ninfer::ops::detail::softmax_cpu {

// --- mirrored plane extents ------------------------------------------------------------
// 64 B: one cylinder plane (rk2v4-e8 K, rk4v2-e8 V) — 32 subspaces x 2 B, 2 bits/dim.
// 96 B: the E8Lattice K plane — 64 B cylinder codes [0,64) + 32 B residual signs [64,96)
// (rk4_codec.cuh::kRkE8KLatticeExtent; core/paged_kv_storage.h RK4V4E8 / RK4V2E8 extents).
inline constexpr int kE8CylinderPlane  = 64;
inline constexpr int kE8LatticePlane    = 96;
inline constexpr int kE8Subspaces       = kHeadDim / 8;  // 32 subspaces per 256-dim row
inline constexpr int kE8RadiusBits      = 4;              // rad_axis high nibble: radius idx
inline constexpr int kE8AxisBits        = 4;              // rad_axis low nibble: axis idx

static_assert(kE8CylinderPlane == kE8Subspaces * 2);
static_assert(kE8LatticePlane == kE8Subspaces * 2 + kE8Subspaces);

// --- mirrored tables (ops/kernel/e8_root_codec.cuh) ------------------------------------
// c_e8_stage1_i8x8 as int8[8] per code and GENERATED from the code layout the
// device table encodes (verified entry-by-entry against the device uint64 words — the
// semantic map below reproduced all 256 entries with zero mismatches):
//   codes 0..111 (Type A):  code = pair_idx<<2 | s_i<<1 | s_j over the 28 lexicographic
//     {i,j} pairs; the root is +-4 at dims (i,j) (unit-amp roots scaled x4 by the table).
//   codes 112..239 (Type B): code = 112 | b where bit i (i<7) marks dim i positive;
//     dim 7 keeps the even-minus-parity rule -> +-2 at every dim (half-roots scaled x4).
//   codes 240..255: unreachable (240 roots); the device table reads them back as zero.
// A generator instead of a transcription removes the row-swap class of hand-copy errors;
// the GPU dump parity check re-verifies every entry against hardware on every audit run.
inline constexpr auto kE8Stage1 = [] {
    std::array<std::array<std::int8_t, 8>, 256> table{};
    for (int code = 0; code < 240; ++code) {
        for (int d = 0; d < 8; ++d) { table[code][d] = 0; }
        if (code < 112) {
            int pair = 0;
            for (int i = 0; i < 7; ++i) {
                for (int j = i + 1; j < 8; ++j) {
                    if (pair == (code >> 2)) {
                        table[code][i] = (code >> 1) & 1 ? 4 : -4;  // s_i bit
                        table[code][j] = code & 1 ? 4 : -4;         // s_j bit
                    }
                    ++pair;
                }
            }
        } else {
            const int b = code - 112;
            int minus = 0;
            for (int i = 0; i < 7; ++i) {
                table[code][i] = (b >> i) & 1 ? 2 : -2;
                if (((b >> i) & 1) == 0) { ++minus; }
            }
            table[code][7] = (minus % 2 == 0) ? 2 : -2;  // even minus parity
        }
    }
    return table;
}();

// c_axis_i8x8 as int8[8]: one signed unit axis per (dim, sign) nibble index.
inline constexpr std::int8_t kE8Axis[16][8] = {
    { 1, 0, 0, 0, 0, 0, 0, 0}, {-1, 0, 0, 0, 0, 0, 0, 0},
    { 0, 1, 0, 0, 0, 0, 0, 0}, { 0,-1, 0, 0, 0, 0, 0, 0},
    { 0, 0, 1, 0, 0, 0, 0, 0}, { 0, 0,-1, 0, 0, 0, 0, 0},
    { 0, 0, 0, 1, 0, 0, 0, 0}, { 0, 0, 0,-1, 0, 0, 0, 0},
    { 0, 0, 0, 0, 1, 0, 0, 0}, { 0, 0, 0, 0,-1, 0, 0, 0},
    { 0, 0, 0, 0, 0, 1, 0, 0}, { 0, 0, 0, 0, 0,-1, 0, 0},
    { 0, 0, 0, 0, 0, 0, 1, 0}, { 0, 0, 0, 0, 0, 0,-1, 0},
    { 0, 0, 0, 0, 0, 0, 0, 1}, { 0, 0, 0, 0, 0, 0, 0,-1},
};

// c_radius_scale[16]: the 4-bit log-radius multipliers (e8_root_codec.cuh:418-435).
inline constexpr float kE8RadiusScale[16] = {
    0.0000f, 0.0992f, 0.1250f, 0.1575f, 0.1984f, 0.2500f, 0.3150f, 0.3969f,
    0.5000f, 0.6300f, 0.7937f, 1.0000f, 1.2599f, 1.5874f, 2.0000f, 2.5198f,
};

// kE8VRadiusCentreB (e8_root_codec.cuh:709): the calibrated Type-B radius centre the rk4v2-e8
// V-plane read-back multiplies in ((root < 112) ? 1.0 : this).
inline constexpr float kE8VRadiusCentreB = 1.115f;

// kRkE8KFracGrid / kRkE8KFracTheta (rk4_codec.cuh:85-86). The shipped theta == grid makes the
// residual reconstruction code' = Grid * (code +- 1), an exact integer shift-scale; the
// general clamped form is NOT mirrored because the constants live in the repo and a theta
// change must move this assert with it (the device mirrors the clamp branch in
// rk4_apply_e8_residual, so an asymmetric edit shows up as a decode mismatch immediately).
inline constexpr int kE8LatticeFracGrid  = 4;
inline constexpr int kE8LatticeFracTheta = 4;
static_assert(kE8LatticeFracTheta == kE8LatticeFracGrid);

// --- decode primitives -------------------------------------------------------------------

// RNE float -> int32 for the decoder's bounded range. The device uses __float2int_rn
// (cvt.rni.s32.f32); the magic-number add/sub is the same round-half-to-even for every
// |x| < 2^22 and is exponent-safe here (|products| <= 5 * 2.5198 * 1.115 < 15).
inline int e8_round_rn(float value) noexcept {
    const float magic = 12582912.0f;  // 1.5 * 2^23: forces RNE of the low mantissa bits
    return static_cast<int>((value + magic) - magic);
}

// Saturating int8 add, the CPU mirror of vadd4.s32.s32.s32.sat (e8_root_codec.cuh:729-734).
// The reachable table values (|root| <= 4, |axis| <= 1) never saturate; the guard exists so
// arbitrary on-disk bytes read back the way the PTX does.
inline int e8_sat_add_i8(int a, int b) noexcept {
    const int value = a + b;
    return value < -128 ? -128 : (value > 127 ? 127 : value);
}

// One (root_code, rad_axis_code) pair -> eight int8 codes; the CPU mirror of
// e8_root_decode_8d_fast_scaled (e8_root_codec.cuh:712-754). `radius_mult` is 1.0f for the K
// plane / stock consumers and the V-plane calibration for the rk4v2-e8 value read-back.
inline void e8_root_decode_8d(std::uint8_t root_code, std::uint8_t rad_axis_code,
                              float radius_mult, std::int8_t (&out)[8]) noexcept {
    const unsigned rad_idx  = rad_axis_code >> kE8RadiusBits;
    const unsigned axis_idx = rad_axis_code & ((1u << kE8AxisBits) - 1u);
    if (rad_idx == 0u) {
        for (int i = 0; i < 8; ++i) { out[i] = 0; }
        return;
    }
    const auto& root = kE8Stage1[root_code];  // std::array<int8_t, 8>
    const std::int8_t* axis = kE8Axis[axis_idx];
    const float scale = kE8RadiusScale[rad_idx] * radius_mult;  // the device's single fp32 mul
    for (int i = 0; i < 8; ++i) {
        out[i] = static_cast<std::int8_t>(
            e8_round_rn(static_cast<float>(e8_sat_add_i8(root[i], axis[i])) * scale));
    }
}

// The V-plane calibrated variant, mirroring e8_root_decode_8d_fast_v (e8_root_codec.cuh:763-767).
inline void e8_root_decode_8d_v(std::uint8_t root_code, std::uint8_t rad_axis_code,
                                 std::int8_t (&out)[8]) noexcept {
    e8_root_decode_8d(root_code, rad_axis_code,
                      root_code < 112 ? 1.0f : kE8VRadiusCentreB, out);
}

// The per-subspace residual of the E8Lattice K plane, mirroring rk4_apply_e8_residual with
// the shipped theta == grid constants (rk4_codec.cuh:93-106): code' = Grid * (code +- 1),
// sign bit i of `sign_bits` steering dim i (1 -> +1, 0 -> -1). |code| <= 2 keeps
// Grid * (code +- 1) inside int8, exactly the device argument for the clamp-free branch.
inline void e8_lattice_apply_residual(std::int8_t (&dec)[8],
                                      std::uint32_t sign_bits) noexcept {
    for (int i = 0; i < 8; ++i) {
        const int sign = static_cast<int>((sign_bits >> i) & 1u);
        const int value = kE8LatticeFracGrid * (static_cast<int>(dec[i]) + (sign ? 1 : -1));
        dec[i] = static_cast<std::int8_t>(value);
    }
}

// --- row decode helpers -----------------------------------------------------------------

// One 256-dim row over a cylinder plane (64 B/token-head: 2 B per 8D subspace, root first)
// with FP16 G64 scales — the K plane of rk2v4-e8 and the V plane of rk4v2-e8. Group g owns
// subspaces [8g, 8g+8) and scale element g, exactly the G64 partition of the int4/int8 rows.
// `v_calibrated` selects the rk4v2-e8 V-plane per-root radius centre.
inline void decode_e8_cylinder_row(const std::uint8_t* plane,
                                   const std::uint16_t* scale_plane, int kv_heads,
                                   std::int64_t physical_page, int kv_head, int token_in_page,
                                   float* out, bool v_calibrated) noexcept {
    const std::int64_t row =
        plane_offset(kE8CylinderPlane, kv_heads, physical_page, kv_head, token_in_page, 0);
    for (int g = 0; g < kGroups; ++g) {
        const std::int64_t scale =
            plane_offset(kScalePlane, kv_heads, physical_page, kv_head, token_in_page, g);
        const float scale_value = fp16_bits_to_f32(scale_plane[scale]);
        for (int s = 0; s < kGroup / 8; ++s) {  // 8 subspaces per G64 group
            const int subspace = g * (kGroup / 8) + s;
            const std::uint8_t root = plane[row + 2 * subspace];
            const std::uint8_t rad  = plane[row + 2 * subspace + 1];
            std::int8_t codes[8];
            if (v_calibrated) {
                e8_root_decode_8d_v(root, rad, codes);
            } else {
                e8_root_decode_8d(root, rad, 1.0f, codes);
            }
            float* dst = out + g * kGroup + s * 8;
            for (int i = 0; i < 8; ++i) { dst[i] = static_cast<float>(codes[i]) * scale_value; }
        }
    }
}

// One 256-dim row over the E8Lattice K plane (96 B/token-head: cylinder codes at [0,64),
// one residual sign byte per subspace at [64,96)) with FP16 G64 scales — the K plane of
// rk4v4-e8 and rk4v2-e8. The scale plane stores absmax/7/Grid (writer rk_kernel.cuh:215-218);
// the decode multiplies the stored bits exactly like every other row, which reconstructs the
// Grid-scaled code semantics because the residual already folded Grid into the codes.
inline void decode_e8_lattice_row(const std::uint8_t* plane,
                                  const std::uint16_t* scale_plane, int kv_heads,
                                  std::int64_t physical_page, int kv_head, int token_in_page,
                                  float* out) noexcept {
    const std::int64_t row =
        plane_offset(kE8LatticePlane, kv_heads, physical_page, kv_head, token_in_page, 0);
    for (int g = 0; g < kGroups; ++g) {
        const std::int64_t scale =
            plane_offset(kScalePlane, kv_heads, physical_page, kv_head, token_in_page, g);
        const float scale_value = fp16_bits_to_f32(scale_plane[scale]);
        for (int s = 0; s < kGroup / 8; ++s) {
            const int subspace = g * (kGroup / 8) + s;
            const std::uint8_t root = plane[row + 2 * subspace];
            const std::uint8_t rad  = plane[row + 2 * subspace + 1];
            const std::uint32_t signs = plane[row + kE8CylinderPlane + subspace];
            std::int8_t codes[8];
            e8_root_decode_8d(root, rad, 1.0f, codes);
            e8_lattice_apply_residual(codes, signs);
            float* dst = out + g * kGroup + s * 8;
            for (int i = 0; i < 8; ++i) { dst[i] = static_cast<float>(codes[i]) * scale_value; }
        }
    }
}

// Code-only row decode over a cylinder plane: the int8 codes of one 256-dim row WITHOUT the
// group scales — the split the int8-MMA partial wants (integer dot per group, FP32 group
// scales applied after). Same addressing and calibration as decode_e8_cylinder_row.
inline void e8_cylinder_codes_row(const std::uint8_t* plane, int kv_heads,
                                  std::int64_t physical_page, int kv_head, int token_in_page,
                                  int* codes, bool v_calibrated) noexcept {
    const std::int64_t row =
        plane_offset(kE8CylinderPlane, kv_heads, physical_page, kv_head, token_in_page, 0);
    for (int g = 0; g < kGroups; ++g) {
        for (int s = 0; s < kGroup / 8; ++s) {
            const int subspace = g * (kGroup / 8) + s;
            const std::uint8_t root = plane[row + 2 * subspace];
            const std::uint8_t rad  = plane[row + 2 * subspace + 1];
            std::int8_t decoded[8];
            if (v_calibrated) {
                e8_root_decode_8d_v(root, rad, decoded);
            } else {
                e8_root_decode_8d(root, rad, 1.0f, decoded);
            }
            int* dst = codes + g * kGroup + s * 8;
            for (int i = 0; i < 8; ++i) { dst[i] = decoded[i]; }
        }
    }
}

// Code-only row decode over the E8Lattice K plane (cylinder codes + residual signs).
inline void e8_lattice_codes_row(const std::uint8_t* plane, int kv_heads,
                                 std::int64_t physical_page, int kv_head, int token_in_page,
                                 int* codes) noexcept {
    const std::int64_t row =
        plane_offset(kE8LatticePlane, kv_heads, physical_page, kv_head, token_in_page, 0);
    for (int g = 0; g < kGroups; ++g) {
        for (int s = 0; s < kGroup / 8; ++s) {
            const int subspace = g * (kGroup / 8) + s;
            const std::uint8_t root = plane[row + 2 * subspace];
            const std::uint8_t rad  = plane[row + 2 * subspace + 1];
            const std::uint32_t signs = plane[row + kE8CylinderPlane + subspace];
            std::int8_t decoded[8];
            e8_root_decode_8d(root, rad, 1.0f, decoded);
            e8_lattice_apply_residual(decoded, signs);
            int* dst = codes + g * kGroup + s * 8;
            for (int i = 0; i < 8; ++i) { dst[i] = decoded[i]; }
        }
    }
}

} // namespace ninfer::ops::detail::softmax_cpu
