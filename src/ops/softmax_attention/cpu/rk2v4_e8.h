#pragma once

// port (sm_89): rk2v4-e8 (KvCacheStorage::RK2V4E8) CPU page decode — bit-exact fp32 read-back
// of one (physical_page, kv_head, token) K/V row from the raw page planes.
//
// Physical schema (core/paged_kv_storage.h, RK2V4E8): the K plane is a DType::U8 plane of 64
// bytes per token-head — one (root_code, rad_axis_code) byte pair per 8D subspace, root at
// plane byte 2s and rad_axis at 2s+1 (the 2-bit E8-cylinder codec, ops/kernel/e8_root_codec.cuh
// e8_encode_cylinder_8d / e8_root_decode_8d_fast; append writer ops/kv_cache/append/rk_kernel.cuh
// :47-99, attention read-back grouped_mma.cuh:549-577) — with an FP16 G64 scale plane. The V
// plane matches rk4v4's packed-int4 plane (128 B/token-head, low nibble = even dim,
// ops/kv_cache/rk4_codec.cuh) with its own FP16 G64 scale plane. ~13 KiB/token
// (docs/zh-CN/kv-cache-and-eviction.md). The attention dispatch that consumes this layout is
// RkVariant::E8Root (PackedV=true, E8Root=true) in
// src/ops/softmax_attention/dense/causal_cache/int8/launch.cu.
//
// Stored frame. Identical story to rk4v4.h: K is quantized AFTER the normalized D256 Hadamard
// rotation (rk_kernel.cuh rotates, then encodes), so a decoded K row is in the ROTATED frame
// the GPU scores in after rotating Q at kernel entry; V is stored unrotated, so no inverse
// rotation belongs to this decode. The E8 root codes describe the normalized direction of each
// rotated 8D subspace and the 4-bit log-radius its length relative to the group scale — the
// decode below rebuilds the int8 codes the GPU read-back reconstructs, and the row value is
// the same exact code*scale product as every rk layout. The K plane uses the radius table
// unchanged (radius_mult = 1.0; only the rk4v2-e8 V plane carries a calibration).

#include "ops/softmax_attention/cpu/decode.h"
#include "ops/softmax_attention/cpu/e8_codec.h"

namespace ninfer::ops::detail::softmax_cpu {

// Decode the 256-dim K row of (physical_page, kv_head, token_in_page) into `out` (rotated
// frame). `k_plane` / `k_scale_plane` are the bases of the layer's 64-byte cylinder K code
// plane (uint8) and K scale plane (FP16 bits) — the tensors PagedKVLayerView names
// k_pages / k_scale_pages.
inline void rk2v4_e8_decode_k_row(const std::uint8_t* k_plane,
                                  const std::uint16_t* k_scale_plane, int kv_heads,
                                  std::int64_t physical_page, int kv_head, int token_in_page,
                                  float* out) noexcept {
    decode_e8_cylinder_row(k_plane, k_scale_plane, kv_heads, physical_page, kv_head,
                           token_in_page, out, /*v_calibrated=*/false);
}

// Decode the 256-dim V row (original frame). Packed-int4 codec, same as rk4v4's V.
inline void rk2v4_e8_decode_v_row(const std::uint8_t* v_plane,
                                  const std::uint16_t* v_scale_plane, int kv_heads,
                                  std::int64_t physical_page, int kv_head, int token_in_page,
                                  float* out) noexcept {
    decode_int4_row(v_plane, v_scale_plane, kv_heads, physical_page, kv_head, token_in_page,
                    out);
}

// Decode both rows of one (page, head, token) triple in one pass: the entry point the
// host-partial consumer should prefer over the split calls.
inline void rk2v4_e8_decode_kv_row(const std::uint8_t* k_plane, const std::uint8_t* v_plane,
                                   const std::uint16_t* k_scale_plane,
                                   const std::uint16_t* v_scale_plane, int kv_heads,
                                   std::int64_t physical_page, int kv_head, int token_in_page,
                                   float* k_out, float* v_out) noexcept {
    rk2v4_e8_decode_k_row(k_plane, k_scale_plane, kv_heads, physical_page, kv_head,
                          token_in_page, k_out);
    rk2v4_e8_decode_v_row(v_plane, v_scale_plane, kv_heads, physical_page, kv_head,
                          token_in_page, v_out);
}

} // namespace ninfer::ops::detail::softmax_cpu
