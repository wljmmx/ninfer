#pragma once

// port (sm_89): rk4v4 (RotatedInt4KeyInt4ValueGroup64) CPU page decode — bit-exact fp32
// read-back of one (physical_page, kv_head, token) K/V row from the raw page planes.
//
// Physical schema (core/paged_kv_storage.h, RotatedInt4KeyInt4ValueGroup64): K and V are
// both DType::U8 planes of 128 packed bytes per token-head — two int4 codes per byte, low
// nibble = even dim (ops/kv_cache/rk4_codec.cuh) — and each carries its own FP16 G64
// scale plane of 4 scales per token-head. The serving name maps at
// src/product/config/options.cpp kv_storage(); the attention dispatch that consumes this
// layout is RkVariant::PackedKOnly (src/ops/softmax_attention/dense/causal_cache/int8/launch.cu).
//
// Stored frame. K is quantized AFTER the normalized D256 Hadamard rotation
// (ops/kv_cache/append/rk_kernel.cuh rotates, then quantizes), so a decoded K row is in
// the ROTATED frame — the same frame the GPU scores in after rotating Q at kernel entry
// (grouped_mma.cuh / batch_wide_mma.cuh / tiled_mma.cuh all call
// normalized_hadamard_d256_inplace on Q). V is stored unrotated (rk_kernel.cuh: the V
// rotation was measured as a net loss and reverted), so a decoded V row is in the
// original coordinate frame and the PV accumulation needs no inverse rotation. The rk
// merges run with InverseRotation=false (launch.cu passes false as the merge's fifth
// template argument); causal_merge.cuh's InverseRotation=true path exists for layouts
// with a rotated V plane and is never selected for rk4v4. Consequence: NO inverse
// rotation is part of this decode; a consumer that wants original-frame K applies the
// inverse transform itself, exactly like the GPU never does during scoring.
//
// Quantization conventions written by the append (verified in the same headers): K and V
// codes are plain RTN int4 clamped to [-7, 7], scale = FP16-RNE(group_absmax / 7). The
// decode contract does not depend on those conventions — it maps any stored byte and any
// FP16 scale to the same product the GPU read-back computes.

#include "ops/softmax_attention/cpu/decode.h"

namespace ninfer::ops::detail::softmax_cpu {

// Decode the 256-dim K row of (physical_page, kv_head, token_in_page) into `out`
// (rotated frame, see the header comment). `k_plane` / `k_scale_plane` are the bases of
// the layer's K code plane (uint8) and K scale plane (FP16 bits), exactly the tensors
// PagedKVLayerView names k_pages / k_scale_pages.
inline void rk4v4_decode_k_row(const std::uint8_t* k_plane,
                               const std::uint16_t* k_scale_plane, int kv_heads,
                               std::int64_t physical_page, int kv_head, int token_in_page,
                               float* out) noexcept {
    decode_int4_row(k_plane, k_scale_plane, kv_heads, physical_page, kv_head, token_in_page,
                    out);
}

// Decode the 256-dim V row (original frame). Same packed-int4 codec on the V planes.
inline void rk4v4_decode_v_row(const std::uint8_t* v_plane,
                               const std::uint16_t* v_scale_plane, int kv_heads,
                               std::int64_t physical_page, int kv_head, int token_in_page,
                               float* out) noexcept {
    decode_int4_row(v_plane, v_scale_plane, kv_heads, physical_page, kv_head, token_in_page,
                    out);
}

// Decode both rows of one (page, head, token) triple in one pass: `k_out` and `v_out`
// each receive kHeadDim floats. The shared-plane geometry (kv_heads, page, head, token)
// is resolved once per plane, so this is the entry point the host-partial consumer
// should prefer over the split calls.
inline void rk4v4_decode_kv_row(const std::uint8_t* k_plane, const std::uint8_t* v_plane,
                                const std::uint16_t* k_scale_plane,
                                const std::uint16_t* v_scale_plane, int kv_heads,
                                std::int64_t physical_page, int kv_head, int token_in_page,
                                float* k_out, float* v_out) noexcept {
    rk4v4_decode_k_row(k_plane, k_scale_plane, kv_heads, physical_page, kv_head,
                       token_in_page, k_out);
    rk4v4_decode_v_row(v_plane, v_scale_plane, kv_heads, physical_page, kv_head,
                       token_in_page, v_out);
}

} // namespace ninfer::ops::detail::softmax_cpu
