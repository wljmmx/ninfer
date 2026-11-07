#pragma once

// port (sm_89): rk4v4-e8 (KvCacheStorage::RK4V4E8) CPU page decode — bit-exact fp32 read-back
// of one (physical_page, kv_head, token) K/V row from the raw page planes.
//
// Physical schema (core/paged_kv_storage.h, RK4V4E8): the K plane is a DType::U8 plane of 96
// bytes per token-head — 64 B of E8-cylinder codes ([0,64), one (root, rad_axis) byte pair
// per 8D subspace, root at 2s) plus 32 B of per-element 1-bit residual signs ([64,96), one
// sign byte per subspace at 64+s, bit i steering dim i; rk4_codec.cuh::kRkE8KLatticeExtent,
// [D3] converged extent) — with an FP16 G64 scale plane that stores absmax/7/kRkE8KFracGrid.
// The V plane matches rk4v4's packed-int4 plane (128 B/token-head) with its own FP16 G64
// scale plane. ~15-17 KiB/token (docs/zh-CN/kv-cache-and-eviction.md lists 17 KiB). The
// attention dispatch that consumes this layout is RkVariant::E8Lattice (PackedV=true,
// PackedK=true, E8Lattice=true) in
// src/ops/softmax_attention/dense/causal_cache/int8/launch.cu.
//
// Stored frame. Same rotation story as rk4v4.h: K is encoded after the normalized D256
// Hadamard rotation (rotated frame out), V is stored unrotated (original frame out). The
// K quantization is the [EXPERIMENT E8+1bit residual] carrier: each rotated 8D subspace is
// projected to its nearest E8 cylinder root with a log-radius index, and the per-element
// sign of the code-unit residual is carried alongside (rk_kernel.cuh:110-185). Read-back
// (grouped_mma.cuh:628-638 + decode_e8_tile :715-743): e8_root_decode_8d_fast, then
// rk4_apply_e8_residual folds code' = Grid * (code +- 1) with the shipped theta == grid —
// mirrored in e8_codec.h. The decode below reproduces that value times the stored scale,
// which is Grid-predivided so the product is the original-unit reconstruction.

#include "ops/softmax_attention/cpu/decode.h"
#include "ops/softmax_attention/cpu/e8_codec.h"

namespace ninfer::ops::detail::softmax_cpu {

// Decode the 256-dim K row of (physical_page, kv_head, token_in_page) into `out` (rotated
// frame). `k_plane` / `k_scale_plane` are the bases of the layer's 96-byte lattice K code
// plane (uint8) and K scale plane (FP16 bits) — the tensors PagedKVLayerView names
// k_pages / k_scale_pages.
inline void rk4v4_e8_decode_k_row(const std::uint8_t* k_plane,
                                  const std::uint16_t* k_scale_plane, int kv_heads,
                                  std::int64_t physical_page, int kv_head, int token_in_page,
                                  float* out) noexcept {
    decode_e8_lattice_row(k_plane, k_scale_plane, kv_heads, physical_page, kv_head,
                          token_in_page, out);
}

// Decode the 256-dim V row (original frame). Packed-int4 codec, same as rk4v4's V.
inline void rk4v4_e8_decode_v_row(const std::uint8_t* v_plane,
                                  const std::uint16_t* v_scale_plane, int kv_heads,
                                  std::int64_t physical_page, int kv_head, int token_in_page,
                                  float* out) noexcept {
    decode_int4_row(v_plane, v_scale_plane, kv_heads, physical_page, kv_head, token_in_page,
                    out);
}

// Decode both rows of one (page, head, token) triple in one pass.
inline void rk4v4_e8_decode_kv_row(const std::uint8_t* k_plane, const std::uint8_t* v_plane,
                                   const std::uint16_t* k_scale_plane,
                                   const std::uint16_t* v_scale_plane, int kv_heads,
                                   std::int64_t physical_page, int kv_head, int token_in_page,
                                   float* k_out, float* v_out) noexcept {
    rk4v4_e8_decode_k_row(k_plane, k_scale_plane, kv_heads, physical_page, kv_head,
                          token_in_page, k_out);
    rk4v4_e8_decode_v_row(v_plane, v_scale_plane, kv_heads, physical_page, kv_head,
                          token_in_page, v_out);
}

} // namespace ninfer::ops::detail::softmax_cpu
