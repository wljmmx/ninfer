#pragma once

// port (sm_89): rk4v2-e8 (KvCacheStorage::RK4V2E8) CPU page decode — bit-exact fp32 read-back
// of one (physical_page, kv_head, token) K/V row from the raw page planes.
//
// Physical schema (core/paged_kv_storage.h, RK4V2E8): the K plane is the E8Lattice carrier —
// 96 B/token-head, 64 B of E8-cylinder codes plus 32 B of per-element 1-bit residual signs,
// identical to rk4v4-e8's K plane (rk4_codec.cuh::kRkE8KLatticeExtent) — with an FP16 G64
// scale plane stored as absmax/7/Grid. The V plane is a 2-bit E8-cylinder plane (64
// B/token-head, one (root, rad_axis) byte pair per 8D subspace; rk_kernel.cuh:263-296 writer,
// grouped_mma.cuh:664-688 read-back) with its own FP16 G64 scale plane. ~13 KiB/token
// (docs/zh-CN/kv-cache-and-eviction.md). The attention dispatch that consumes this layout is
// RkVariant::E8LatticeV2 (PackedK=true, E8Lattice=true, V8Root=true; PackedV=false — the V
// plane is NOT int4-packed) in src/ops/softmax_attention/dense/causal_cache/int8/launch.cu.
//
// Stored frame. Same rotation story as rk4v4.h: K is encoded after the normalized D256
// Hadamard rotation (rotated frame out), V is stored unrotated (original frame out, the
// rk_kernel.cuh V8Root branch never rotates).
//
// V-plane radius calibration (e8_root_codec.cuh:697-709): the raw V plane reconstructs
// systematically deflated (100% Type-B root selection offline); folding the factor 1.115 into
// the radius INSIDE the round (a table re-centring) was the only calibration that improved
// end-to-end PPL (rk4v2-e8 8.327146 -> 8.300312). The factor is Type-B-only (root >= 112):
// the raw V data never selects a Type-A root, and the calibrated decode keeps 1.0 for A.
// Mirrored as e8_root_decode_8d_v in e8_codec.h.

#include "ops/softmax_attention/cpu/decode.h"
#include "ops/softmax_attention/cpu/e8_codec.h"

namespace ninfer::ops::detail::softmax_cpu {

// Decode the 256-dim K row of (physical_page, kv_head, token_in_page) into `out` (rotated
// frame). Identical codec to rk4v4-e8's K row: E8Lattice carrier (96 B) + FP16 G64 scales.
inline void rk4v2_e8_decode_k_row(const std::uint8_t* k_plane,
                                  const std::uint16_t* k_scale_plane, int kv_heads,
                                  std::int64_t physical_page, int kv_head, int token_in_page,
                                  float* out) noexcept {
    decode_e8_lattice_row(k_plane, k_scale_plane, kv_heads, physical_page, kv_head,
                          token_in_page, out);
}

// Decode the 256-dim V row (original frame): the 2-bit E8-cylinder plane with the Type-B
// radius-centre calibration (e8_root_decode_8d_v).
inline void rk4v2_e8_decode_v_row(const std::uint8_t* v_plane,
                                  const std::uint16_t* v_scale_plane, int kv_heads,
                                  std::int64_t physical_page, int kv_head, int token_in_page,
                                  float* out) noexcept {
    decode_e8_cylinder_row(v_plane, v_scale_plane, kv_heads, physical_page, kv_head,
                           token_in_page, out, /*v_calibrated=*/true);
}

// Decode both rows of one (page, head, token) triple in one pass.
inline void rk4v2_e8_decode_kv_row(const std::uint8_t* k_plane, const std::uint8_t* v_plane,
                                   const std::uint16_t* k_scale_plane,
                                   const std::uint16_t* v_scale_plane, int kv_heads,
                                   std::int64_t physical_page, int kv_head, int token_in_page,
                                   float* k_out, float* v_out) noexcept {
    rk4v2_e8_decode_k_row(k_plane, k_scale_plane, kv_heads, physical_page, kv_head,
                          token_in_page, k_out);
    rk4v2_e8_decode_v_row(v_plane, v_scale_plane, kv_heads, physical_page, kv_head,
                          token_in_page, v_out);
}

} // namespace ninfer::ops::detail::softmax_cpu
