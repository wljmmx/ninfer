#pragma once

// port (sm_89): rk8v4 (RotatedInt8KeyInt4ValueGroup64) CPU page decode — bit-exact fp32
// read-back of one (physical_page, kv_head, token) K/V row from the raw page planes.
//
// Physical schema (core/paged_kv_storage.h, RotatedInt8KeyInt4ValueGroup64): K is a
// DType::I8 plane of 256 signed bytes per token-head — one int8 code per dim, at the
// kv_cache_int8_quant_code_index address (ops/kv_cache/int8_g64_codec.cuh) — with an
// FP16 G64 scale plane; V matches rk4v4's packed-int4 plane (128 B/token-head, low
// nibble = even dim, ops/kv_cache/rk4_codec.cuh) with its own FP16 G64 scale plane. The
// serving name maps at src/product/config/options.cpp kv_storage(); the attention dispatch
// that consumes this layout is RkVariant::Plain (int8 K, PackedV only) in
// src/ops/softmax_attention/dense/causal_cache/int8/launch.cu.
//
// Stored frame. Identical story to rk4v4: K is Hadamard-rotated before quantization
// (ops/kv_cache/append/rk_kernel.cuh, int8 branch), so the decoded K row is in the
// ROTATED frame the GPU scores in after rotating Q; V is stored unrotated and the merges
// run with InverseRotation=false, so no inverse rotation belongs to this decode.
//
// Quantization conventions written by the append: K codes are RTN int8 clamped to
// [-127, 127] with scale = FP16-RNE(group_absmax / 127) (kv_cache_int8_quant_params /
// kv_cache_int8_quant_code, int8_g64_codec.cuh); V codes are RTN int4 clamped to
// [-7, 7] with scale = FP16-RNE(group_absmax / 7) (rk4_codec.cuh) — the same value
// codec as rk4v4's V. The decode contract itself only relies on the stored bytes.

#include "ops/softmax_attention/cpu/decode.h"

namespace ninfer::ops::detail::softmax_cpu {

// Decode the 256-dim K row of (physical_page, kv_head, token_in_page) into `out`
// (rotated frame). `k_plane` / `k_scale_plane` are the bases of the layer's K code
// plane (int8 bytes) and K scale plane (FP16 bits) — PagedKVLayerView's k_pages /
// k_scale_pages.
inline void rk8v4_decode_k_row(const std::uint8_t* k_plane,
                               const std::uint16_t* k_scale_plane, int kv_heads,
                               std::int64_t physical_page, int kv_head, int token_in_page,
                               float* out) noexcept {
    decode_int8_row(k_plane, k_scale_plane, kv_heads, physical_page, kv_head, token_in_page,
                    out);
}

// Decode the 256-dim V row (original frame). Packed-int4 codec, same as rk4v4's V.
inline void rk8v4_decode_v_row(const std::uint8_t* v_plane,
                               const std::uint16_t* v_scale_plane, int kv_heads,
                               std::int64_t physical_page, int kv_head, int token_in_page,
                               float* out) noexcept {
    decode_int4_row(v_plane, v_scale_plane, kv_heads, physical_page, kv_head, token_in_page,
                    out);
}

// Decode both rows of one (page, head, token) triple in one pass.
inline void rk8v4_decode_kv_row(const std::uint8_t* k_plane, const std::uint8_t* v_plane,
                                const std::uint16_t* k_scale_plane,
                                const std::uint16_t* v_scale_plane, int kv_heads,
                                std::int64_t physical_page, int kv_head, int token_in_page,
                                float* k_out, float* v_out) noexcept {
    rk8v4_decode_k_row(k_plane, k_scale_plane, kv_heads, physical_page, kv_head,
                       token_in_page, k_out);
    rk8v4_decode_v_row(v_plane, v_scale_plane, kv_heads, physical_page, kv_head,
                       token_in_page, v_out);
}

} // namespace ninfer::ops::detail::softmax_cpu
