#pragma once

#include "ninfer/types.h"
#include "ops/kernel/e8_root_codec.cuh"
#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/kv_cache/nvfp4_group16_codec.cuh"
#include "ops/kv_cache/rk4_codec.cuh"
#include "ops/softmax_attention/qsa/norm_rope.cuh"
#include "ops/softmax_attention/qsa/prepare_arguments.cuh"

namespace ninfer::ops::detail::qsa {

// The rank-compressed family: the same Hadamard-rotated K as the int8 path, then an int8/int4 or
// E8-cylinder carrier, with an int4 or 2-bit E8-cylinder V. V is written in the *original*
// coordinate frame (the rk append path documents why: rotating it corrupted every appended token).
template <KvCacheStorage Storage>
inline constexpr bool kPrepareRk =
    Storage == KvCacheStorage::RotatedInt8KeyInt4ValueGroup64 ||
    Storage == KvCacheStorage::RotatedInt4KeyInt4ValueGroup64 || Storage == KvCacheStorage::RK4V4E8 ||
    Storage == KvCacheStorage::RK2V4E8 || Storage == KvCacheStorage::RK4V2E8;

// 96-byte E8-cylinder carrier with a per-element 1-bit residual (rk4v4-e8, rk4v2-e8 key plane).
template <KvCacheStorage Storage>
inline constexpr bool kPrepareRkE8Key =
    Storage == KvCacheStorage::RK4V4E8 || Storage == KvCacheStorage::RK4V2E8;

// 64-byte 2-bit E8-cylinder carrier without a residual (rk2v4-e8 key, rk4v2-e8 value).
template <KvCacheStorage Storage>
inline constexpr bool kPrepareRkRootValue = Storage == KvCacheStorage::RK4V2E8;

__device__ __forceinline__ void store_int8(const float (&values)[8], std::uint8_t* data,
                                           std::uint8_t* scales, int lane) {
    auto* codes      = reinterpret_cast<std::int8_t*>(data);
    auto* out_scales = reinterpret_cast<__half*>(scales);
#pragma unroll
    for (int group = 0; group < 4; ++group) {
        const float x0 = values[2 * group], x1 = values[2 * group + 1];
        const auto quant = kv_cache_int8_quant_params(warp_max(fmaxf(fabsf(x0), fabsf(x1))));
        codes[group * 64 + lane]      = kv_cache_int8_quant_code(x0, quant.inverse_scale);
        codes[group * 64 + lane + 32] = kv_cache_int8_quant_code(x1, quant.inverse_scale);
        if (lane == 0) out_scales[group] = quant.scale;
    }
}

__device__ __forceinline__ void store_fp8(const float (&values)[8], std::uint8_t* codes,
                                          std::uint8_t* scales, int lane) {
    float absmax = 0;
#pragma unroll
    for (float value : values) absmax = fmaxf(absmax, fabsf(value));
    const auto quant = kv_cache_fp8_quant_params(warp_max(absmax));
#pragma unroll
    for (int j = 0; j < 8; ++j)
        codes[lane + 32 * j] = kv_cache_fp8_quant_code(values[j], quant.inverse_scale);
    if (lane == 0) *reinterpret_cast<__half*>(scales) = quant.scale;
}

__device__ __forceinline__ void store_nvfp4(const float (&values)[8], std::uint8_t* codes,
                                            std::uint8_t* scales, int lane, float* scratch) {
#pragma unroll
    for (int j = 0; j < 8; ++j) scratch[lane + 32 * j] = values[j];
    __syncwarp();
    if (lane < 16) {
        const auto encoded = kv_cache_nvfp4_quantize_group16(scratch + 16 * lane);
        auto* out          = codes + 8 * lane;
        if (reinterpret_cast<std::uintptr_t>(out) % 8 == 0)
            *reinterpret_cast<uint2*>(out) = make_uint2(encoded.codes_lo, encoded.codes_hi);
        else {
            // Quantized planes promise natural byte alignment, including padded layer views.
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                out[j]     = std::uint8_t(encoded.codes_lo >> (8 * j));
                out[j + 4] = std::uint8_t(encoded.codes_hi >> (8 * j));
            }
        }
        scales[lane] = encoded.scale;
    }
    // Every reader must finish before the warp reuses this row for V.
    __syncwarp();
}

template <KvCacheStorage Storage>
__device__ __forceinline__ void store_quantized(const float (&values)[8], std::uint8_t* codes,
                                                std::uint8_t* scales, int lane, float* scratch) {
    if constexpr (Storage == KvCacheStorage::Int8Group64)
        store_int8(values, codes, scales, lane);
    else if constexpr (Storage == KvCacheStorage::Fp8E4M3Row256)
        store_fp8(values, codes, scales, lane);
    else {
        static_assert(Storage == KvCacheStorage::Nvfp4Group16);
        store_nvfp4(values, codes, scales, lane, scratch);
    }
}

// ---------------------------------------------------------------------------
// rk family. The arithmetic is the rk append kernel's, mirrored one-to-one so a prepare-written
// token and an append-written token are the same bytes: per-group absmax scales, the int4 packing
// pattern, and the E8-cylinder carrier with its residual sign bytes.
constexpr unsigned kRkFullMask = 0xffffffffU;

// Quantized planes promise natural byte alignment, including padded layer views, and the packed
// writers assemble four bytes at a time: fall back to byte stores when the row is not four-aligned.
__device__ __forceinline__ void store_rk_u32(std::uint8_t* address, std::uint32_t value) {
    if ((reinterpret_cast<std::uintptr_t>(address) & 3U) == 0) {
        *reinterpret_cast<std::uint32_t*>(address) = value;
    } else {
        address[0] = static_cast<std::uint8_t>(value);
        address[1] = static_cast<std::uint8_t>(value >> 8);
        address[2] = static_cast<std::uint8_t>(value >> 16);
        address[3] = static_cast<std::uint8_t>(value >> 24);
    }
}

// 4-bit symmetric [-7, 7] plane, two dims per byte. Every lane holds dim lane + 32*r, so dims 2k and
// 2k+1 sit in adjacent lanes: an even lane builds the byte and a leader lane gathers four of them.
__device__ __forceinline__ void store_rk_int4(const float (&values)[8], std::uint8_t* data,
                                              std::uint8_t* scales, int lane) {
    for (int group = 0; group < 4; ++group) {
        const float absmax =
            warp_max(fmaxf(fabsf(values[2 * group]), fabsf(values[2 * group + 1])), kRkFullMask);
        const __half scale = rk4_absmax_to_scale_h(absmax);
        const float step   = __half2float(scale);
        const float inverse = step > 0.0f ? 1.0f / step : 0.0f;
        for (int half = 0; half < 2; ++half) {
            const int code      = rk4_quant_code(values[2 * group + half], inverse);
            const int neighbour = __shfl_xor_sync(kRkFullMask, code, 1);
            std::uint32_t packed = 0;
            if ((lane & 1) == 0) {
                packed = rk4_pack(static_cast<std::int8_t>(code),
                                  static_cast<std::int8_t>(neighbour));
            }
            const std::uint32_t b1 = __shfl_xor_sync(kRkFullMask, packed, 2);
            const std::uint32_t b2 = __shfl_xor_sync(kRkFullMask, packed, 4);
            const std::uint32_t b3 = __shfl_xor_sync(kRkFullMask, packed, 6);
            if ((lane & 7) == 0) {
                const int index = (group * 64 + half * 32 + lane) >> 1;
                store_rk_u32(data + index, packed | (b1 << 8) | (b2 << 16) | (b3 << 24));
            }
        }
        if (lane == 0) { reinterpret_cast<__half*>(scales)[group] = scale; }
    }
}

// E8-cylinder carrier: one (root, radial-axis) byte pair per eight-dim subspace, and — with a
// residual — one sign byte per subspace at offset 64. The stored group scale is divided by the
// residual grid because read-back multiplies the codes by it.
template <bool Residual>
__device__ __forceinline__ void store_rk_e8(const float (&values)[8], std::uint8_t* data,
                                           std::uint8_t* scales, int lane,
                                           const E8Trace* trace_base = nullptr) {
    for (int group = 0; group < 4; ++group) {
        const float absmax =
            warp_max(fmaxf(fabsf(values[2 * group]), fabsf(values[2 * group + 1])), kRkFullMask);
        const __half scale = rk4_absmax_to_scale_h(absmax);
        const float step   = __half2float(scale);
        std::uint8_t root0 = 0, rad0 = 0, root1 = 0, rad1 = 0;
        E8Trace trace0{}, trace1{};
        if (trace_base != nullptr) {
            trace0           = *trace_base;
            trace0.group     = group;
            trace0.subspace  = group * 8 + int((lane >> 3) & 3);
            trace1           = trace0;
            trace1.subspace  = trace0.subspace + 4;
        }
        e8_encode_cylinder_8d_warp(values[2 * group], step, root0, rad0, lane,
                                   trace_base != nullptr ? &trace0 : nullptr);
        e8_encode_cylinder_8d_warp(values[2 * group + 1], step, root1, rad1, lane,
                                   trace_base != nullptr ? &trace1 : nullptr);
        const std::uint32_t pair0      = root0 | (rad0 << 8);
        const std::uint32_t pair1      = root1 | (rad1 << 8);
        const std::uint32_t pair0_next = __shfl_xor_sync(kRkFullMask, pair0, 8);
        const std::uint32_t pair1_next = __shfl_xor_sync(kRkFullMask, pair1, 8);
        std::uint32_t up0 = 0, up1 = 0;
        if constexpr (Residual) {
            __align__(8) std::int8_t dec0[8];
            __align__(8) std::int8_t dec1[8];
            e8_root_decode_8d_fast(root0, rad0, dec0);
            e8_root_decode_8d_fast(root1, rad1, dec1);
            const bool over0 = values[2 * group] / step >= static_cast<float>(dec0[lane & 7]);
            const bool over1 = values[2 * group + 1] / step >= static_cast<float>(dec1[lane & 7]);
            up0 = __ballot_sync(kRkFullMask, over0);
            up1 = __ballot_sync(kRkFullMask, over1);
        }
        // Lanes 0 and 16 own a four-byte window: the XOR-8 gather pairs each writer with the next
        // subspace, so one writer covers subspaces {s, s+1} and {s+4, s+5}.
        if ((lane & 15) == 0) {
            const int first = lane >> 3;
            store_rk_u32(data + (group * 8 + first) * 2, pair0 | (pair0_next << 16));
            store_rk_u32(data + (group * 8 + first + 4) * 2, pair1 | (pair1_next << 16));
        }
        if constexpr (Residual) {
            if (lane == 0) {
                // One sign byte per subspace: ballot bits 8s..8s+7 are subspace s's eight dims.
                const auto pack = [](std::uint32_t bits, int first) {
                    return ((bits >> (8 * first)) & 0xffu) |
                           (((bits >> (8 * (first + 1))) & 0xffu) << 8) |
                           (((bits >> (8 * (first + 2))) & 0xffu) << 16) |
                           (((bits >> (8 * (first + 3))) & 0xffu) << 24);
                };
                store_rk_u32(data + 64 + group * 8, pack(up0, 0));
                store_rk_u32(data + 64 + group * 8 + 4, pack(up1, 0));
            }
        }
        if (lane == 0) {
            reinterpret_cast<__half*>(scales)[group] =
                Residual ? __float2half_rn(step / static_cast<float>(kRkE8KFracGrid)) : scale;
        }
    }
}

template <KvCacheStorage Storage>
__device__ __forceinline__ void store_rk_key(const float (&values)[8], std::uint8_t* data,
                                             std::uint8_t* scales, int lane,
                                             const E8Trace* trace = nullptr) {
    if constexpr (Storage == KvCacheStorage::RotatedInt8KeyInt4ValueGroup64) {
        // 8-bit Hadamard key: the int8-group64 codec verbatim, 256 B of codes and four scales.
        store_int8(values, data, scales, lane);
    } else if constexpr (Storage == KvCacheStorage::RotatedInt4KeyInt4ValueGroup64) {
        store_rk_int4(values, data, scales, lane);
    } else if constexpr (kPrepareRkE8Key<Storage>) {
        store_rk_e8<true>(values, data, scales, lane, trace);
    } else {
        static_assert(Storage == KvCacheStorage::RK2V4E8);
        store_rk_e8<false>(values, data, scales, lane, trace);
    }
}

template <KvCacheStorage Storage>
__device__ __forceinline__ void store_rk_value(const float (&values)[8], std::uint8_t* data,
                                               std::uint8_t* scales, int lane,
                                               const E8Trace* trace = nullptr) {
    if constexpr (kPrepareRkRootValue<Storage>) {
        store_rk_e8<false>(values, data, scales, lane, trace);
    } else {
        store_rk_int4(values, data, scales, lane);
    }
}

template <KvCacheStorage Storage, int KvHeads>
__device__ __forceinline__ void prepare_kv(const PrepareArguments& a, int head, int t, int b,
                                           int lane, float2 sc, float* scratch = nullptr) {
    const auto m          = t + std::int64_t(b) * a.tokens;
    const int p           = a.positions[m];
    const int page        = a.tables[p / 64 + std::int64_t(a.table_rows[b]) * a.table_stride];
    auto* key             = a.keys.at(p % 64, head, page);
    auto* value           = a.values.at(p % 64, head, page);
    const bool trace_row  = a.e8_dump != 0 && (a.e8_dump_row < 0 || a.e8_dump_row == int(m));
    const bool trace_key  = trace_row && a.e8_dump_plane <= 0;
    const bool trace_value = trace_row && (a.e8_dump_plane < 0 || a.e8_dump_plane == 1);
    const auto* raw_key   = a.k + (m * KvHeads + head) * 256;
    const auto* raw_value = a.v + (m * KvHeads + head) * 256;
    if constexpr (Storage == KvCacheStorage::BFloat16) {
        prepare_head<256>(raw_key, a.wk, reinterpret_cast<__nv_bfloat16*>(key), a.epsilon, sc,
                          lane);
#pragma unroll
        for (int j = 0; j < 8; ++j)
            reinterpret_cast<__half*>(value)[lane + 32 * j] =
                __float2half_rn(__bfloat162float(raw_value[lane + 32 * j]));
    } else {
        float values[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) values[j] = __bfloat162float(raw_key[lane + 32 * j]);
        norm_rotate<256>(values, a.wk, a.epsilon, sc, lane);
        // The transformed FP32 key enters the codec directly, without a BF16 intermediate.
        normalized_hadamard_d256_inplace(values, lane);
        constexpr auto key_storage =
            Storage == KvCacheStorage::Fp8KeyNvfp4Value ? KvCacheStorage::Fp8E4M3Row256 : Storage;
        constexpr auto value_storage =
            Storage == KvCacheStorage::Fp8KeyNvfp4Value ? KvCacheStorage::Nvfp4Group16 : Storage;
        if constexpr (kPrepareRk<Storage>) {
            E8Trace trace{};
            trace.row   = int(m);
            trace.head  = head;
            trace.plane = 0;
            store_rk_key<Storage>(values, key, a.key_scales.at(p % 64, head, page), lane,
                                  trace_key ? &trace : nullptr);
        } else {
            store_quantized<key_storage>(values, key, a.key_scales.at(p % 64, head, page), lane,
                                         scratch);
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) values[j] = __bfloat162float(raw_value[lane + 32 * j]);
        if constexpr (value_storage == KvCacheStorage::Nvfp4Group16)
            normalized_hadamard_d256_inplace(values, lane);
        if constexpr (kPrepareRk<Storage>) {
            // rk keeps V in the original coordinate frame, exactly as the append path does.
            E8Trace trace{};
            trace.row   = int(m);
            trace.head  = head;
            trace.plane = 1;
            store_rk_value<Storage>(values, value, a.value_scales.at(p % 64, head, page), lane,
                                    trace_value ? &trace : nullptr);
        } else {
            store_quantized<value_storage>(values, value, a.value_scales.at(p % 64, head, page),
                                           lane, scratch);
        }
    }
}

} // namespace ninfer::ops::detail::qsa
