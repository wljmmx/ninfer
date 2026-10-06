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

} // namespace ninfer::ops
