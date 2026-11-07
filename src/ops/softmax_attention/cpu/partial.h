#pragma once

// port (sm_89): CPU partition partial over host-resident rk-family KV pages.
//
// Feasibility batch 2, built on the Phase A decode substrate (decode.h / rk4v4.h / rk8v4.h):
// compute one KV split's attention partial for a decode-shaped query tile, in the exact
// numeric form the production GPU grouped instance uses, and write it into the partial
// buffers the GPU merge kernel consumes. Header-only, host-only, no CUDA include, no runtime
// wiring — the controller that decides which splits run on the host is a later batch (cost.h
// is its decision input).
//
// P4 batch: the codec axis (RkKCodec / RkVCodec below) extends the partial over the rk -e8
// layouts — rk4v4-e8 (E8Lattice K), rk2v4-e8 (E8Root K), rk4v2-e8 (E8Lattice K + E8Root V) —
// on the e8_codec.h decode substrate. The e8 rows feed the SAME int8-dot / FP16-group-scale
// numeric form (every e8 code decodes to int8 and every scale plane stays FP16 G64), so the
// contract below is unchanged; only the code-fetch in Stage 1 varies by codec.
//
// --- partial contract (extracted, read-only) ------------------------------------------
// CausalPartialView { float* acc; float* maximum; float* sum; } — unnormalized FP32
// numerator / running max / probability sum (causal_operands.h:33-39). Buffers are allocated
// FP32 with layouts (allocate_causal_partials, causal_operands.h:63-69):
//     acc:     {kCausalHeadDim=256, heads, width, splits*batch}
//     maximum: {heads, width, splits*batch}
//     sum:     {heads, width, splits*batch}
// with element addressing causal_partial_index / causal_stat_index (causal_tile_io.cuh:22-34):
//     acc:     d + 256 * (head + QHeads * (token + width * split))
//     maximum: head + QHeads * (token + width * split)
// The batch slot is added to the base pointer by the caller (grouped_mma.cuh:148-155 adds
// batch * 256*QHeads*width*split_count to acc and the stat analogues), so this kernel writes
// batch slot 0 and the caller offsets. `maximum` is in natural scaled-score units: the score
// here already includes attention scale (causal_operands.h:33-34).
//
// --- split geometry (grouped_mma.cuh:159-168) ------------------------------------------
// window = positions[width-1] + 1; active = partition.active(window) (causal_partition.h:17-20,
// count = ceil(window / 2^key_shift), clamped to target). Splits own key ranges:
//     logical_tiles = ceil(window / KeyTile)      (KeyTile = Schedule::kKeyRows: 32 for every
//     first_owned   = split * logical_tiles / active      decode grouped instance of both
//     end_owned     = (split+1) * logical_tiles / active  production geometries; schedule.cuh,
//     begin         = first_owned * KeyTile               instances.h:11-16)
//     end           = min(end_owned * KeyTile, window)
// Splits >= active write nothing and are never read by the merge.
//
// --- numeric form (int8 grouped instance, grouped_mma.cuh) -----------------------------
// Q: bf16 -> fp32 (exact), normalized Hadamard at entry (:461-469), then per-G64-group int8
//    quantization with an FP32 scale qs = absmax/127 (:472-484, kv_cache_int8_quant_code =
//    RNE-round, clamp [-127,127], int8_g64_codec.cuh:62-67). K scales are the stored FP16
//    scales widened exactly. QK^T is an int8xint8->int32 dot per group — exact on both sides —
//    and each group contributes (qs * ks) * float(dot) accumulated in FP32 in group order
//    (:786-813). Attention scale multiplies the unmasked score (:830-845); masked keys
//    (absent page / outside the split / causally future) score -inf.
// Online softmax per KeyTile-key tile (:823-956): tile max, nm = max(m, tile_max),
//    alpha = (m == -inf) ? 0 : exp(m - nm), p = (nm > -inf && s > -inf) ? exp(s - nm) : 0,
//    bl = tile sum of p (pairs p[key]+p[key+1] accumulated in key order, :921-931),
//    l = l*alpha + bl, m = nm.
// PV (:1033-1058): acc *= alpha, then acc += fp16(p) * fp16(code * vs) accumulated in FP32.
//    The fp32 product float(code)*vs is exact (decode.h contract), so both sides round the
//    same value to FP16; p is likewise FP16-RNE rounded (:923-928).
// exp: the GPU uses ex2.approx.f32 (causal_softmax.cuh:11-13 via exp2_approx, math.cuh:26-30);
//    this kernel uses host expf. Residual analysis: ex2.approx.f32 relative error is ~2^-22,
//    host expf ~1 ulp, so p/l deviate by ~3e-7 relative — an order below the 1e-5 bar, and
//    acc is dominated anyway by the shared FP16 rounding of p (2^-11). The merge weights use
//    device expf on the GPU (causal_merge.cuh:30) and host expf here (same 1-2 ulp class).
//
// --- merge contract (causal_merge.cuh:11-38, 92-109) -----------------------------------
// maximum = max over active splits; weight[s] = (l[s] > 0 && maximum > -inf) ? expf(m[s] -
// maximum) : 0; denominator = sum(l * weight); if denominator <= 0 every weight is zeroed;
// value[d] = (sum over splits of acc[d][s] * weight[s]) / denominator, or 0; stored bf16-RNE.
// rk_merge_partials below mirrors this for host-side end-to-end use; the InverseRotation
// branch never applies to rk layouts (launch.cu passes false).

#include "ops/softmax_attention/cpu/decode.h"
#include "ops/softmax_attention/cpu/e8_codec.h"
#include "ops/softmax_attention/cpu/hadamard_d256.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ninfer::ops::detail::softmax_cpu {

// --- geometry mirror -------------------------------------------------------------------
// Compile-time head mapping of CausalGeometry (causal_geometry.h:7-14,
// AttentionHeadMapping, head_mapping.cuh:7-15). Production decode shapes:
// rk_partial<24, 4, ...> (CausalD256H24Kv4) and rk_partial<16, 2, ...> (CausalD256H16Kv2).
template <int QueryHeads, int KVHeads>
struct HostCausalGeometry {
    static_assert(QueryHeads > 0 && KVHeads > 0 && QueryHeads % KVHeads == 0);
    static_assert(kHeadDim == 256);  // decode.h mirror of kCausalHeadDim
    static constexpr int q_heads    = QueryHeads;
    static constexpr int kv_heads   = KVHeads;
    static constexpr int group_size = QueryHeads / KVHeads;
};

// --- conversions ------------------------------------------------------------------------

// BF16 bits -> fp32: exact widening (the mirror of __bfloat162float on finite values).
inline float bf16_bits_to_f32(std::uint16_t bits) noexcept {
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
    float out;
    std::memcpy(&out, &wide, sizeof out);
    return out;
}

// fp32 -> FP16 bits, round-to-nearest-even: the mirror of __float2half_rn / __floats2half2_rn
// (pack_f16x2) for finite inputs, which is the only reachable domain here — p in [0,1] and the
// exact products float(code)*vs are finite. Subnormals round like the hardware cvt.rn.f16.f32;
// |x| >= 65520 rounds to Inf (the RNE tie at 65520 goes to the even neighbor, Inf); NaN
// canonicalizes to 0x7e00 with the sign cleared, mirroring the decode-side rule in decode.h.
inline std::uint16_t f32_to_f16_bits_rn(float value) noexcept {
    std::uint32_t x;
    std::memcpy(&x, &value, sizeof x);
    const std::uint16_t sign = static_cast<std::uint16_t>((x >> 16) & 0x8000u);
    x &= 0x7fffffffu;
    if (x >= 0x7f800000u) {  // Inf / NaN
        return x == 0x7f800000u ? static_cast<std::uint16_t>(sign | 0x7c00u) : 0x7e00u;
    }
    if (x >= 0x477ff000u) {  // |x| >= 65520 -> Inf (ties at 65520 round to even = Inf)
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    if (x < 0x33000000u) {  // |x| < 2^-25, and the tie 2^-25 -> even -> zero
        return sign;
    }
    if (x < 0x38800000u) {  // FP16 subnormal: |x| in [2^-25, 2^-14)
        const std::uint32_t exponent = x >> 23;                       // 103..112
        const std::uint32_t mantissa = (x & 0x007fffffu) | 0x00800000u;
        const std::uint32_t shift    = 126 - exponent;               // 14..23
        const std::uint32_t half     = 1u << (shift - 1);
        const std::uint32_t rest     = mantissa & ((1u << shift) - 1u);
        std::uint32_t quantum        = mantissa >> shift;            // value = quantum * 2^-24
        if (rest > half || (rest == half && (quantum & 1u) != 0u)) { ++quantum; }
        return static_cast<std::uint16_t>(sign | quantum);  // quantum == 1024 encodes 2^-14
    }
    const int exponent = static_cast<int>(x >> 23) - 127;  // -14..15
    std::uint32_t mantissa = (x & 0x007fffffu) >> 13;  // top 10 of the 23-bit mantissa
    const std::uint32_t rest = x & 0x1fffu;
    if (rest > 0x1000u || (rest == 0x1000u && (mantissa & 1u) != 0u)) { ++mantissa; }
    if (mantissa == 0x400u) {  // carry into the exponent (unreachable past the Inf guard)
        mantissa = 0;
        return static_cast<std::uint16_t>(sign | ((static_cast<std::uint32_t>(exponent + 16)) << 10));
    }
    return static_cast<std::uint16_t>(sign |
                                      ((static_cast<std::uint32_t>(exponent + 15)) << 10) |
                                      mantissa);
}

// fp32 -> BF16 bits, round-to-nearest-even: the mirror of __float2bfloat16 (merge epilogue)
// for finite values — the merge's numerator/denominator are finite, so Inf/NaN are robustness
// parity only: Inf keeps its sign, NaN canonicalizes to 0x7fc0 with the sign cleared (the same
// canonicalization rule decode.h applies to widened scales).
inline std::uint16_t f32_to_bf16_bits_rn(float value) noexcept {
    std::uint32_t x;
    std::memcpy(&x, &value, sizeof x);
    const std::uint16_t sign = static_cast<std::uint16_t>((x >> 16) & 0x8000u);
    if ((x & 0x7fffffffu) > 0x7f800000u) { return 0x7fc0u; }  // NaN -> canonical
    if ((x & 0x7fffffffu) == 0x7f800000u) {  // +-Inf
        return static_cast<std::uint16_t>(sign | 0x7f80u);
    }
    const std::uint32_t lsb = (x >> 16) & 1u;
    x += 0x7fffu + lsb;
    return static_cast<std::uint16_t>(x >> 16);
}

// --- K/V codec selection -----------------------------------------------------------------
// The rk family's per-layout plane codecs, as a template axis of the partial. Int4/Int8 are
// the launch.cu PackedK pair (rk4v4 / rk8v4); E8Root is rk2v4-e8's 64 B cylinder K plane;
// E8Lattice is the 96 B cylinder+residual K plane of rk4v4-e8 / rk4v2-e8; E8Root on the V side
// is rk4v2-e8's calibrated 64 B cylinder V plane (VCodec::Int4 covers every other variant).
// The K plane extents and decode semantics mirror e8_codec.h; the numeric form below never
// changes — every codec still feeds the instance's exact int8 dot with FP16 G64 scales.
enum class RkKCodec { Int4, Int8, E8Root, E8Lattice };
enum class RkVCodec { Int4, E8Root };

// --- partition mirror --------------------------------------------------------------------

// Host mirror of CausalKvPartition (causal_partition.h:11-21).
struct RkPartition {
    int capacity  = 1;
    int target    = 1;
    int key_shift = 6;  // log2 of the minimum KV keys per split

    int active(int visible) const noexcept {
        const int count = (visible + (1 << key_shift) - 1) >> key_shift;
        return count < target ? count : target;
    }
};

// Host mirror of the split key-range assignment (grouped_mma.cuh:159-168).
struct RkPartialTokenRange {
    int begin = 0;
    int end   = 0;
};

inline RkPartialTokenRange rk_partial_split_range(int window, int split, int active_splits,
                                                  int key_tile) noexcept {
    const int logical_tiles = (window + key_tile - 1) / key_tile;
    const int first_owned   = split * logical_tiles / active_splits;
    const int end_owned     = (split + 1) * logical_tiles / active_splits;
    const int owned_end     = end_owned * key_tile;
    return {first_owned * key_tile, owned_end < window ? owned_end : window};
}

// --- kernel inputs / outputs --------------------------------------------------------------

// Host-resident page planes of one layer: the same bases PagedKVLayerView names k_pages /
// k_scale_pages / v_pages / v_scale_pages. `block_table` is the row's table: logical page =
// position >> 6 -> physical page, a negative entry marks the page absent
// (paged_kv_address.cuh:115-142); absent keys score -inf exactly like causally masked keys.
struct RkPartialKvPages {
    const std::uint8_t* k_codes    = nullptr;  // K code plane (int4-packed or int8)
    const std::uint8_t* v_codes    = nullptr;  // V code plane (int4-packed, low nibble = even dim)
    const std::uint16_t* k_scales  = nullptr;  // FP16 G64 scale plane bits
    const std::uint16_t* v_scales  = nullptr;  // FP16 G64 scale plane bits
    const std::int32_t* block_table = nullptr;
    int kv_heads                    = 0;       // == HostCausalGeometry::kv_heads of the call
};

// One split's work item. `q_bf16` holds BF16 bits in causal_q_index order
// (causal_tile_io.cuh:10-13): element (token, head, d) at d + 256*(head + QHeads*token).
// `positions[t]` is the absolute KV position of query token t and must be non-negative and
// increasing; the caller guarantees pos[width-1] < visible capacity (the kernel-side guard is
// grouped_mma.cuh:158). `split` selects the output slot; `range` comes from
// rk_partial_split_range; `key_tile` is the online-softmax granularity (the grouped instance's
// kKeyRows — 32 for every production decode schedule; the numerics only depend on it through
// the per-tile rescale points, so a caller may mirror a different schedule faithfully).
struct RkPartialRequest {
    const std::uint16_t* q_bf16 = nullptr;
    const std::int32_t* positions = nullptr;
    float scale   = 0.0f;  // attention scale (1/sqrt(256) for the production shapes)
    int width     = 0;     // query-tile tokens (== the grouped instance's TokenTile)
    int tokens    = 0;     // active query tokens (<= width; the unmasked decode path passes width)
    int split     = 0;     // output slot, < partition capacity
    RkPartialTokenRange range{};
    int key_tile  = 32;    // Schedule::kKeyRows of the mirrored instance
};

// The partial buffers, GPU-contract layout (see the header comment). Base pointers of the
// caller's batch slot; the kernel indexes by (token, head, split) only.
struct RkPartialOutput {
    float* acc     = nullptr;  // [256][QHeads][width][splits]
    float* maximum = nullptr;  // [QHeads][width][splits]
    float* sum     = nullptr;  // [QHeads][width][splits]
};

// --- Q preparation -------------------------------------------------------------------------

// The grouped instance's Q entry path (grouped_mma.cuh:457-485): bf16 -> fp32, normalized
// Hadamard, per-G64-group absmax, FP32 scale qs = absmax/127 (NOTE: unlike the K/V cache
// scales the Q scale stays FP32), RNE int8 codes clamped to [-127,127]
// (kv_cache_int8_quant_code, int8_g64_codec.cuh:62-67).
//
// Outputs (caller-allocated, shared with the double oracle so both consume identical codes):
//   q_codes[(token * QueryHeads + head) * 256 + d]   int8 code in [-127,127]
//   q_scales[(token * QueryHeads + head) * 4 + g]   FP32 group scale
template <int QueryHeads, int KVHeads>
void rk_prepare_query(const std::uint16_t* q_bf16, int width, int tokens, int* q_codes,
                      float* q_scales) {
    static_assert(HostCausalGeometry<QueryHeads, KVHeads>::group_size > 0);
    (void)width;  // Q rows are laid out per (token, head); only `tokens` rows are computed
    for (int token = 0; token < tokens; ++token) {
        for (int head = 0; head < QueryHeads; ++head) {
            const std::uint16_t* src = q_bf16 + static_cast<std::int64_t>(token) * QueryHeads * 256;
            float row[256];
            for (int d = 0; d < 256; ++d) { row[d] = bf16_bits_to_f32(src[head * 256 + d]); }
            hadamard_d256_inplace(row);
            int* codes  = q_codes + (static_cast<std::int64_t>(token) * QueryHeads + head) * 256;
            float* scale = q_scales + (static_cast<std::int64_t>(token) * QueryHeads + head) * 4;
            for (int g = 0; g < kGroups; ++g) {
                float absmax = 0.0f;
                for (int j = 0; j < kGroup; ++j) {
                    const float magnitude = std::fabs(row[g * kGroup + j]);
                    absmax                = absmax > magnitude ? absmax : magnitude;
                }
                const float qs = absmax > 0.0f ? absmax / 127.0f : 0.0f;
                const float inv = qs > 0.0f ? 1.0f / qs : 0.0f;
                scale[g]        = qs;
                for (int j = 0; j < kGroup; ++j) {
                    int code = 0;
                    if (inv != 0.0f) {
                        code = std::lrintf(row[g * kGroup + j] * inv);
                        if (code > 127) code = 127;
                        if (code < -127) code = -127;
                    }
                    codes[g * kGroup + j] = code;
                }
            }
        }
    }
}

// --- the partial kernel ---------------------------------------------------------------------

namespace rk_partial_detail {

inline constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// K code of dimension d (grouped_mma.cuh:645-660 PackedK read-back / :662 int8 cp_async path).
template <bool PackedK>
inline int k_code_at(const std::uint8_t* k_plane, int kv_heads, std::int64_t page, int kv_head,
                     int token_in_page, int d) noexcept {
    if constexpr (PackedK) {
        return unpack_int4(
            k_plane[plane_offset(kInt4Plane, kv_heads, page, kv_head, token_in_page, d >> 1)],
            d & 1);
    } else {
        return static_cast<int>(static_cast<std::int8_t>(
            k_plane[plane_offset(kInt8Plane, kv_heads, page, kv_head, token_in_page, d)]));
    }
}

inline int v_code_at(const std::uint8_t* v_plane, int kv_heads, std::int64_t page, int kv_head,
                      int token_in_page, int d) noexcept {
    return unpack_int4(
        v_plane[plane_offset(kInt4Plane, kv_heads, page, kv_head, token_in_page, d >> 1)], d & 1);
}

// One key's 256 int8 K codes, codec-selected (the read-back paths the grouped instance
// stages: rk4_unpack for PackedK (:645-660), the int8 cp_async plane (:662), the E8Root
// cylinder (:549-577) and the E8Lattice carrier + residual (:628-638 + decode_e8_tile)).
// The E8 codecs are the code-only rows of e8_codec.h; the sign window the GPU stages 4-byte
// aligned collapses to the same per-subspace byte 64+s both sides read.
template <RkKCodec KCodec>
inline void k_codes_row(const std::uint8_t* k_plane, int kv_heads, std::int64_t page, int kv_head,
                        int token_in_page, int* codes) noexcept {
    if constexpr (KCodec == RkKCodec::Int4) {
        for (int d = 0; d < kHeadDim; ++d) {
            codes[d] = k_code_at<true>(k_plane, kv_heads, page, kv_head, token_in_page, d);
        }
    } else if constexpr (KCodec == RkKCodec::Int8) {
        for (int d = 0; d < kHeadDim; ++d) {
            codes[d] = k_code_at<false>(k_plane, kv_heads, page, kv_head, token_in_page, d);
        }
    } else if constexpr (KCodec == RkKCodec::E8Root) {
        e8_cylinder_codes_row(k_plane, kv_heads, page, kv_head, token_in_page, codes, false);
    } else {
        e8_lattice_codes_row(k_plane, kv_heads, page, kv_head, token_in_page, codes);
    }
}

// One key's 256 int8 V codes: int4-packed (grouped_mma.cuh:689-697) or the calibrated
// rk4v2-e8 cylinder plane (grouped_mma.cuh:664-688, e8_root_decode_8d_fast_v).
template <RkVCodec VCodec>
inline void v_codes_row(const std::uint8_t* v_plane, int kv_heads, std::int64_t page, int kv_head,
                        int token_in_page, int* codes) noexcept {
    if constexpr (VCodec == RkVCodec::Int4) {
        for (int d = 0; d < kHeadDim; ++d) {
            codes[d] = v_code_at(v_plane, kv_heads, page, kv_head, token_in_page, d);
        }
    } else {
        e8_cylinder_codes_row(v_plane, kv_heads, page, kv_head, token_in_page, codes, true);
    }
}

// One key's unscaled score, the instance's exact arithmetic (grouped_mma.cuh:786-813): an
// exact int32 dot per G64 group (the int8 MMA is integer-exact), then the FP32 group
// contribution (qs * ks) * float(dot) accumulated in group order. The dot is order-free
// (integer); the FP32 sequence is fixed by this shared body so the kernel and the double
// oracle compile the identical operations.
inline float rk_score_key(const int* q_codes_row, const float* q_scales_row,
                          const int* k_codes_row, const float* k_scales_row) noexcept {
    float score = 0.0f;
    for (int g = 0; g < kGroups; ++g) {  // grouped_mma.cuh:809-812
        std::int32_t dot = 0;
        for (int j = 0; j < kGroup; ++j) {
            dot += q_codes_row[g * kGroup + j] * k_codes_row[g * kGroup + j];
        }
        score += q_scales_row[g] * k_scales_row[g] * static_cast<float>(dot);
    }
    return score;
}

} // namespace rk_partial_detail

// Compute one split's partial for every query head of the tile. KCodec/VCodec select the
// plane codecs: (Int4, Int4) = rk4v4, (Int8, Int4) = rk8v4, (E8Root, Int4) = rk2v4-e8,
// (E8Lattice, Int4) = rk4v4-e8, (E8Lattice, E8Root) = rk4v2-e8 — every combination the
// launch.cu rk variants dispatch. Both codecs must be spelled as the enum explicitly:
// an earlier bool-PackedK overload of this name silently captured enum spellings (an enum
// value is a valid converted constant expression for a bool template parameter, and MSVC
// resolved the ambiguity to the bool overload, running the int4 codec for every E8 call),
// so the legacy bool form was REMOVED and its two callers spell the enum directly
// (host_partial.h and apps/diag-cpu-partial).
// All arithmetic mirrors the int8 grouped MMA instance (see the header comment). Two
// deviation classes vs the GPU instance, both verified bounded on hardware: (1) accumulation
// order inside a tile differs from the warp schedule — ulp-level; (2) the expf-vs-ex2.approx
// residual can flip which side of a FP16 boundary p rounds to — a single FP16 quantum of one
// key's contribution on rare elements. Measured: every element of every probe case either
// meets 1e-5 relative or stays inside one 2^-11 quantum (see the batch-2 report).
template <int QueryHeads, int KVHeads, RkKCodec KCodec = RkKCodec::Int4,
          RkVCodec VCodec = RkVCodec::Int4>
void rk_attention_partial(const RkPartialKvPages& pages, const RkPartialRequest& request,
                          const RkPartialOutput& partial) {
    using G = HostCausalGeometry<QueryHeads, KVHeads>;
    using rk_partial_detail::kNegInf;
    constexpr int D = kHeadDim;  // 256

    const int width  = request.width;
    const int tokens = request.tokens > 0 ? request.tokens : width;
    if (width < 1 || tokens > width || request.key_tile < 1 || !pages.k_codes || !pages.v_codes ||
        !pages.k_scales || !pages.v_scales || !pages.block_table || !request.q_bf16 ||
        !request.positions || !partial.acc || !partial.maximum || !partial.sum ||
        pages.kv_heads != KVHeads)
        throw std::invalid_argument("rk attention partial: invalid operands");
    const int rows = tokens * G::group_size;
    if (rows > 64)  // the GPU schedule bound (grouped_mma.cuh:71)
        throw std::invalid_argument("rk attention partial: query tile exceeds 64 rows");

    // Q preparation, shared with the double oracle (identical codes on both sides).
    std::vector<int> q_codes(static_cast<std::size_t>(tokens) * QueryHeads * D);
    std::vector<float> q_scales(static_cast<std::size_t>(tokens) * QueryHeads * kGroups);
    rk_prepare_query<QueryHeads, KVHeads>(request.q_bf16, width, tokens, q_codes.data(),
                                          q_scales.data());

    const int key_tile = request.key_tile;
    const int begin     = request.range.begin;
    const int end       = request.range.end;

    std::vector<float> acc(static_cast<std::size_t>(rows) * D);
    std::vector<float> running_max(static_cast<std::size_t>(rows), kNegInf);
    std::vector<float> running_sum(static_cast<std::size_t>(rows), 0.0f);
    std::vector<float> tile_scores(static_cast<std::size_t>(key_tile) * rows);
    std::vector<std::uint16_t> tile_p16(static_cast<std::size_t>(key_tile) * rows);
    std::vector<float> tile_alpha(static_cast<std::size_t>(rows));
    std::vector<std::uint16_t> tile_v16(static_cast<std::size_t>(key_tile) * D);
    std::vector<int> tile_k_codes(static_cast<std::size_t>(key_tile) * D);
    std::vector<int> tile_v_codes(static_cast<std::size_t>(key_tile) * D);
    std::vector<float> tile_k_scales(static_cast<std::size_t>(key_tile) * kGroups);
    std::vector<unsigned char> tile_present(static_cast<std::size_t>(key_tile));

    for (int kv_head = 0; kv_head < KVHeads; ++kv_head) {
        std::fill(acc.begin(), acc.end(), 0.0f);
        std::fill(running_max.begin(), running_max.end(), kNegInf);
        std::fill(running_sum.begin(), running_sum.end(), 0.0f);

        for (int tile_base = begin; tile_base < end; tile_base += key_tile) {
            const int tile_keys =
                (end - tile_base) < key_tile ? (end - tile_base) : key_tile;
            // Stage 1: decode the tile's K codes/scales and FP16 V products, score every row.
            // Out-of-range / absent / causally masked keys keep score -inf and p 0, exactly
            // like the instance's masked lanes (grouped_mma.cuh:830-845, 955-956, 1009-1011).
            for (int k = 0; k < tile_keys; ++k) {
                const int key          = tile_base + k;
                const std::int32_t entry = pages.block_table[key >> 6];
                const bool present    = entry >= 0;
                tile_present[k]       = present ? 1u : 0u;
                if (!present) {
                    for (int row = 0; row < rows; ++row) {
                        tile_scores[static_cast<std::size_t>(k) * rows + row] = kNegInf;
                    }
                    continue;
                }
                const auto page    = static_cast<std::int64_t>(entry);
                const int tip      = key & (kPageSize - 1);
                int* k_codes        = &tile_k_codes[static_cast<std::size_t>(k) * D];
                float* k_scales    = &tile_k_scales[static_cast<std::size_t>(k) * kGroups];
                std::uint16_t* v16 = &tile_v16[static_cast<std::size_t>(k) * D];
                for (int g = 0; g < kGroups; ++g) {
                    const std::int64_t scale_offset =
                        plane_offset(kScalePlane, KVHeads, page, kv_head, tip, g);
                    k_scales[g] = fp16_bits_to_f32(pages.k_scales[scale_offset]);
                }
                rk_partial_detail::k_codes_row<KCodec>(pages.k_codes, KVHeads, page, kv_head,
                                                       tip, k_codes);
                rk_partial_detail::v_codes_row<VCodec>(pages.v_codes, KVHeads, page, kv_head,
                                                       tip, &tile_v_codes[static_cast<std::size_t>(k) * D]);
                for (int g = 0; g < kGroups; ++g) {
                    const float vs = fp16_bits_to_f32(
                        pages.v_scales[plane_offset(kScalePlane, KVHeads, page, kv_head, tip, g)]);
                    const int* v_row = &tile_v_codes[static_cast<std::size_t>(k) * D];
                    for (int j = 0; j < kGroup; ++j) {
                        const int d = g * kGroup + j;
                        const float product = static_cast<float>(v_row[d]) * vs;
                        v16[d] = f32_to_f16_bits_rn(product);  // exact fp32 product, one RNE
                    }
                }
                for (int row = 0; row < rows; ++row) {
                    const int token = row / G::group_size;
                    const int local  = row - token * G::group_size;
                    const int head   = kv_head * G::group_size + local;
                    const int* q_row = &q_codes[(static_cast<std::size_t>(token) * QueryHeads +
                                                head) * D];
                    const float* q_scale = &q_scales[(static_cast<std::size_t>(token) * QueryHeads +
                                                     head) * kGroups];
                    const float score =
                        rk_partial_detail::rk_score_key(q_row, q_scale, k_codes, k_scales);
                    tile_scores[static_cast<std::size_t>(k) * rows + row] =
                        key <= request.positions[token] ? score * request.scale : kNegInf;
                }
            }
            for (int k = tile_keys; k < key_tile; ++k) {  // masked tail of the last tile
                for (int row = 0; row < rows; ++row) {
                    tile_scores[static_cast<std::size_t>(k) * rows + row] = kNegInf;
                }
                tile_present[k] = 0u;
            }

            // Stage 2: per-row online update (grouped_mma.cuh:846-956).
            for (int row = 0; row < rows; ++row) {
                float tile_max = kNegInf;
                for (int k = 0; k < key_tile; ++k) {
                    const float s = tile_scores[static_cast<std::size_t>(k) * rows + row];
                    tile_max     = tile_max > s ? tile_max : s;
                }
                const float m_old = running_max[row];
                const float m_new = m_old > tile_max ? m_old : tile_max;
                const float alpha = m_old == kNegInf ? 0.0f : std::exp(m_old - m_new);
                float bl = 0.0f;  // pair-wise accumulation in key order, like the producer lanes
                for (int k = 0; k + 1 < key_tile; k += 2) {
                    const float p0 = m_new > kNegInf &&
                                    tile_scores[static_cast<std::size_t>(k) * rows + row] > kNegInf
                                ? std::exp(tile_scores[static_cast<std::size_t>(k) * rows + row] -
                                           m_new)
                                : 0.0f;
                    const float p1 = m_new > kNegInf &&
                                    tile_scores[static_cast<std::size_t>(k + 1) * rows + row] > kNegInf
                                ? std::exp(tile_scores[static_cast<std::size_t>(k + 1) * rows + row] -
                                           m_new)
                                : 0.0f;
                    bl += p0 + p1;
                    tile_p16[static_cast<std::size_t>(k) * rows + row] = f32_to_f16_bits_rn(p0);
                    tile_p16[static_cast<std::size_t>(k + 1) * rows + row] =
                        f32_to_f16_bits_rn(p1);
                }
                if ((key_tile & 1) != 0) {  // odd tiles never occur in the production schedules
                    const auto k = static_cast<std::size_t>(key_tile - 1);
                    const float p = m_new > kNegInf && tile_scores[k * rows + row] > kNegInf
                                        ? std::exp(tile_scores[k * rows + row] - m_new)
                                        : 0.0f;
                    bl += p;
                    tile_p16[k * rows + row] = f32_to_f16_bits_rn(p);
                }
                running_sum[row] = running_sum[row] * alpha + bl;
                running_max[row] = m_new;
                tile_alpha[row]  = alpha;
            }

            // Stage 3: rescale, then PV with the FP16-rounded p and V (grouped_mma.cuh:1033-1058).
            for (int row = 0; row < rows; ++row) {
                const float alpha = tile_alpha[row];
                float* row_acc    = &acc[static_cast<std::size_t>(row) * D];
                for (int d = 0; d < D; ++d) { row_acc[d] *= alpha; }
            }
            for (int k = 0; k < tile_keys; ++k) {
                if (!tile_present[k]) { continue; }
                const std::uint16_t* v16 = &tile_v16[static_cast<std::size_t>(k) * D];
                for (int row = 0; row < rows; ++row) {
                    const std::uint16_t bits = tile_p16[static_cast<std::size_t>(k) * rows + row];
                    if (bits == 0u) { continue; }  // p16 == +0 contributes nothing (sign-exact)
                    const float p     = fp16_bits_to_f32(bits);
                    float* row_acc    = &acc[static_cast<std::size_t>(row) * D];
                    for (int d = 0; d < D; ++d) {
                        row_acc[d] += p * fp16_bits_to_f32(v16[d]);  // exact product, FP32 add
                    }
                }
            }
        }

        // Stage 4: write the GPU-contract partial (grouped_mma.cuh:1069-1116).
        for (int row = 0; row < rows; ++row) {
            const int token = row / G::group_size;
            const int local = row - token * G::group_size;
            const int head  = kv_head * G::group_size + local;
            const std::int64_t stat = head +
                static_cast<std::int64_t>(QueryHeads) * (token + static_cast<std::int64_t>(width) * request.split);
            partial.maximum[stat] = running_max[row];
            partial.sum[stat]     = running_sum[row];
            float* dst = partial.acc +
                static_cast<std::int64_t>(D) * stat;
            const float* row_acc = &acc[static_cast<std::size_t>(row) * D];
            for (int d = 0; d < D; ++d) { dst[d] = row_acc[d]; }
        }
    }
}

// --- host merge mirror ------------------------------------------------------------------------

// Merge the active splits exactly like causal_natural_merge_kernel (causal_merge.cuh:11-38,
// 92-109) with InverseRotation=false: max over splits, weight = (l > 0 && max > -inf) ?
// expf(m - max) : 0, denominator = sum(l * weight) — zeroed together when it is <= 0 — and
// value = (sum of acc * weight) / denominator, stored BF16-RNE. `active_splits` is
// partition.active(positions[width-1] + 1); slots >= active_splits are never read, exactly
// like the GPU. `out_bf16` follows causal_q_index order: (token, head, d) at
// d + 256*(head + QueryHeads*token). Multi-batch callers loop with per-batch output bases.
template <int QueryHeads>
void rk_merge_partials(const float* acc, const float* maximum, const float* sum, int width,
                       int active_splits, std::uint16_t* out_bf16) {
    constexpr int D = 256;
    if (active_splits < 1 || active_splits > 256) {
        throw std::invalid_argument("rk merge partials: invalid split count");
    }
    std::vector<float> weights(static_cast<std::size_t>(active_splits));
    for (int token = 0; token < width; ++token) {
        for (int head = 0; head < QueryHeads; ++head) {
            const std::int64_t stat_base = head +
                static_cast<std::int64_t>(QueryHeads) * token;
            float maximum_all = rk_partial_detail::kNegInf;
            for (int split = 0; split < active_splits; ++split) {
                const float m = maximum[stat_base +
                                       static_cast<std::int64_t>(QueryHeads) * width * split];
                maximum_all   = maximum_all > m ? maximum_all : m;
            }
            float denominator = 0.0f;
            for (int split = 0; split < active_splits; ++split) {
                const std::int64_t stat = stat_base +
                                          static_cast<std::int64_t>(QueryHeads) * width * split;
                const float l = sum[stat];
                const float w = l > 0.0f && maximum_all > rk_partial_detail::kNegInf
                                    ? std::exp(maximum[stat] - maximum_all)
                                    : 0.0f;
                weights[static_cast<std::size_t>(split)] = w;
                denominator += l * w;
            }
            if (denominator <= 0.0f) {
                for (auto& w : weights) { w = 0.0f; }
            }
            std::uint16_t* out_row =
                out_bf16 + static_cast<std::int64_t>(token) * QueryHeads * D + head * D;
            for (int d = 0; d < D; ++d) {
                float numerator = 0.0f;
                for (int split = 0; split < active_splits; ++split) {
                    if (weights[static_cast<std::size_t>(split)] == 0.0f) { continue; }
                    const std::int64_t element =
                        d + static_cast<std::int64_t>(D) *
                                (head + static_cast<std::int64_t>(QueryHeads) *
                                            (token + static_cast<std::int64_t>(width) * split));
                    numerator += acc[element] * weights[static_cast<std::size_t>(split)];
                }
                const float value = denominator > 0.0f ? numerator / denominator : 0.0f;
                out_row[d]       = f32_to_bf16_bits_rn(value);
            }
        }
    }
}

} // namespace ninfer::ops::detail::softmax_cpu
