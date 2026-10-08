#pragma once

// 4-bit value-plane codec for the rank-compressed KV-cache layouts (rk4v4, rk4v4-e8,
// rk2v4-e8, rk8v4). Keys and values are Hadamard-rotated in the existing int8 path
// (hadamard_d256.cuh); this header only adds the int4 pack/unpack and the /7 scale
// convention that the 4-bit symmetric [-7, 7] range requires.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kKVCacheRkHeadDim  = 256;
inline constexpr int kKVCacheRkGroup    = 64;
inline constexpr int kKVCacheRkGroups   = kKVCacheRkHeadDim / kKVCacheRkGroup;

// 4-bit packed code index: two dimensions per byte, so the byte stride per head is
// head_dim / 2 = 128. Dimension d lives in byte d / 2 (low nibble = even d, high = odd).
template <typename Geometry>
__device__ __forceinline__ std::int64_t
rk4_v_code_index(int physical_page, int kv_head, int packed_d, int page_offset) {
    return paged_kv_element_offset<128, Geometry::KVHeads>(physical_page, kv_head, page_offset,
                                                           packed_d);
}

// G64 scale index (same as int8: 4 groups per 256-dim head).
template <typename Geometry>
__device__ __forceinline__ std::int64_t
rk4_v_scale_index(int physical_page, int kv_head, int group, int page_offset) {
    return paged_kv_element_offset<4, Geometry::KVHeads>(physical_page, kv_head, page_offset,
                                                          group);
}

// 4-bit symmetric quantization: range [-7, 7], scale = absmax / 7.
__device__ __forceinline__ float rk4_absmax_to_scale(float absmax) {
    return absmax > 0.0f ? absmax / 7.0f : 0.0f;
}

__device__ __forceinline__ __half rk4_absmax_to_scale_h(float absmax) {
    return __float2half_rn(absmax > 0.0f ? absmax / 7.0f : 0.0f);
}

__device__ __forceinline__ std::int8_t rk4_quant_code(float x, float inv_scale) {
    if (inv_scale == 0.0f) { return 0; }
    int q = __float2int_rn(x * inv_scale);
    q     = max(-7, min(7, q));
    return static_cast<std::int8_t>(q);
}

// Pack two int4 codes into one byte (low nibble = first, high nibble = second).
__device__ __forceinline__ std::uint8_t rk4_pack(std::int8_t lo, std::int8_t hi) {
    return static_cast<std::uint8_t>(
        (static_cast<std::uint8_t>(lo) & 0xF) |
        ((static_cast<std::uint8_t>(hi) & 0xF) << 4));
}

// Unpack one int4 from a byte. high=0 → low nibble, high=1 → high nibble.
// Sign-extends from 4-bit to 8-bit: values 8..15 become -8..-1.
__device__ __forceinline__ std::int8_t rk4_unpack(std::uint8_t packed, int high) {
    std::int8_t code = high ? static_cast<std::int8_t>(packed >> 4)
                            : static_cast<std::int8_t>(packed);
    code &= 0xF;
    if (code >= 8) { code -= 16; }
    return code;
}

// Sign-extend four 4-bit nibbles (one per byte) into four 8-bit signed codes without
// cross-byte carries. The previous (nibble ^ 0x08) + 0xF8 form used a 32-bit add whose
// carries crossed byte boundaries whenever a nibble was below 8, corrupting the adjacent
// code; replicating bit 3 (the sign bit) into bits 4..7 cannot carry.
__device__ __forceinline__ std::uint32_t rk4_sign_extend_nibbles(std::uint32_t masked) {
    const std::uint32_t sign = masked & 0x08080808u;
    return masked | ((sign << 1) | (sign << 2) | (sign << 3) | (sign << 4));
}

// Fast 8-byte (16 int4) unpack using PTX prmt for byte permutation.
// Given 8 packed bytes, extracts all 16 int4 values into 16 int8 values in one
// vectorized operation, replacing 16 separate rk4_unpack calls with 2 prmt +
// 2 arithmetic ops. The sign extension uses the fact that nibble values 8-15
// (unsigned) map to -8..-1 (signed) via `val - 16` when val >= 8.
__device__ __forceinline__ void rk4_unpack_x8_fast(std::uint32_t packed_lo, std::uint32_t packed_hi,
                                                    std::int8_t out[8]) {
    // Use prmt to extract low nibbles: each byte → low nibble in its position.
    // prmt.b32 with selector 0x76543210 extracts bytes as-is; we need to isolate
    // nibbles. Instead, use a mask + conditional subtract approach.
    // Extract low nibbles: mask each byte with 0x0F
    const std::uint32_t lo_masked = packed_lo & 0x0F0F0F0Fu;
    const std::uint32_t hi_masked = (packed_lo >> 4) & 0x0F0F0F0Fu;
    const std::uint32_t lo_signed = rk4_sign_extend_nibbles(lo_masked);
    const std::uint32_t hi_signed = rk4_sign_extend_nibbles(hi_masked);
    // Reinterpret as 4 int8 per uint32
    const auto* lo_p = reinterpret_cast<const std::int8_t*>(&lo_signed);
    const auto* hi_p = reinterpret_cast<const std::int8_t*>(&hi_signed);
    // Low nibbles (even positions)
    out[0] = lo_p[0]; out[2] = lo_p[1]; out[4] = lo_p[2]; out[6] = lo_p[3];
    // High nibbles (odd positions)
    out[1] = hi_p[0]; out[3] = hi_p[1]; out[5] = hi_p[2]; out[7] = hi_p[3];
}

// Fast batch unpack: 4 packed bytes → 8 int4 values as a 32-bit packed int8.
// Returns a uint32_t holding 4 int8 low-nibble values in the low half and
// 4 int8 high-nibble values in the high half — suitable for direct
// float conversion + pack_f16x2.
__device__ __forceinline__ std::uint32_t rk4_unpack_lo_fast(std::uint32_t packed) {
    // Extract low nibbles and sign-extend in one vectorized operation.
    const std::uint32_t masked = packed & 0x0F0F0F0Fu;
    return rk4_sign_extend_nibbles(masked);
}

__device__ __forceinline__ std::uint32_t rk4_unpack_hi_fast(std::uint32_t packed) {
    // Extract high nibbles and sign-extend.
    const std::uint32_t masked = (packed >> 4) & 0x0F0F0F0Fu;
    return rk4_sign_extend_nibbles(masked);
}

// Unpack 16 int4 values (8 bytes) into 16 int8 values.
__device__ __forceinline__ void rk4_unpack_x16(const std::uint8_t* src8,
                                                std::int8_t* dst16) {
    const int2 raw       = load_vec<int2>(src8);
    const std::uint8_t* b = reinterpret_cast<const std::uint8_t*>(&raw);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        dst16[2 * i]     = rk4_unpack(b[i], 0);
        dst16[2 * i + 1] = rk4_unpack(b[i], 1);
    }
}

// Dequantize 8 packed int4 values (4 bytes) to 8 bf16 values, given an FP16 scale.
__device__ __forceinline__ int4 rk4_dequant_bf16x8_from(const std::uint8_t* codes4,
                                                         __half scale) {
    const int raw = *reinterpret_cast<const int*>(codes4);
    const std::uint8_t* b = reinterpret_cast<const std::uint8_t*>(&raw);
    const float s = __half2float(scale);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float x0 = static_cast<float>(rk4_unpack(b[i], 0)) * s;
        const float x1 = static_cast<float>(rk4_unpack(b[i], 1)) * s;
        packed[i] = pack_bf16x2(x0, x1);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

// Dequantize 8 packed int4 values to 8 fp16 values, given an FP16 scale.
__device__ __forceinline__ int4 rk4_dequant_f16x8_from(const std::uint8_t* codes4,
                                                        __half scale) {
    const int raw = *reinterpret_cast<const int*>(codes4);
    const std::uint8_t* b = reinterpret_cast<const std::uint8_t*>(&raw);
    const __half2 s2 = __halves2half2(scale, scale);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __half2 code2 = __floats2half2_rn(
            static_cast<float>(rk4_unpack(b[i], 0)),
            static_cast<float>(rk4_unpack(b[i], 1)));
        const __half2 value2 = __hmul2(code2, s2);
        packed[i] = *reinterpret_cast<const unsigned*>(&value2);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

// Packed int4 V occupies half the smem extent: the 16 dims of one chunk live at the
// (d >> 1) half-offset and cover exactly 8 bytes. The masked-key zero fill must use
// that offset AND that width. A 16-byte store at the half-offset faults with
// cudaErrorMisalignedAddress (the tiled kernel did this for width > 256); a store at
// the full offset leaves the packed region uninitialised (the grouped kernel did this).
// One definition, shared by both causal kernels so they cannot diverge again.
template <bool PackedV>
__device__ __forceinline__ void rk4_zero_packed_v(std::int8_t* v_row, std::int32_t d) {
    if constexpr (PackedV) {
        store_vec(&v_row[d >> 1], make_int2(0, 0));
    } else {
        store_vec(&v_row[d], make_int4(0, 0, 0, 0));
    }
}
} // namespace ninfer::ops
