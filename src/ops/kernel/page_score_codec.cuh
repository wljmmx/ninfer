#pragma once

// P6 2c: order-preserving, lossless float <-> unsigned codec for the per-page relevance score.
//
// CUDA has no native float atomicMax (appendix BL.1(1)). The textbook shortcut - reinterpret a
// non-negative float's IEEE-754 bits as unsigned and compare - is monotone only for non-negative
// values, and attention logits can be negative, which is why the first cut used an integer hit
// count instead. Folding the sign bit repairs the shortcut: for non-negative values set the sign
// bit, for negative values flip every bit. The resulting unsigned order matches the float order
// across the whole range (-inf .. +inf) and the map is a bijection, so the counter stays exact -
// no fixed-point scaling, no quantisation, no overflow - and a reader can recover the original
// score or rank directly on the raw code (the order is preserved).
//
//   -inf -> 0x007FFFFF   +inf -> 0xFF800000   -0.0 -> 0x7FFFFFFF   +0.0 -> 0x80000000
//
// 0 sits below every encodable value and is reserved for "no score recorded", so callers must skip
// -inf before encoding. See appendices CL.4/CM for why the statistic became a maximising store.

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define NINFER_PAGE_SCORE_HD __host__ __device__ __forceinline__
#else
#define NINFER_PAGE_SCORE_HD inline
#endif

namespace ninfer::ops {

/// Reserved counter value: no score was recorded for this page and row.
inline constexpr std::uint32_t kPagedKVPageScoreUnset = 0u;

/// Encode a finite attention logit into an order-preserving unsigned counter value.
NINFER_PAGE_SCORE_HD std::uint32_t paged_kv_page_score_encode(float value) {
#if defined(__CUDA_ARCH__)
    const std::uint32_t bits = __float_as_uint(value);
#else
    std::uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
#endif
    return (bits & 0x80000000u) != 0u ? ~bits : (bits | 0x80000000u);
}

/// Inverse of paged_kv_page_score_encode(); maps the reserved 0 back to -inf.
NINFER_PAGE_SCORE_HD float paged_kv_page_score_decode(std::uint32_t code) {
    if (code == kPagedKVPageScoreUnset) { return -INFINITY; }
    const std::uint32_t bits = (code & 0x80000000u) != 0u ? (code & 0x7FFFFFFFu) : ~code;
#if defined(__CUDA_ARCH__)
    return __uint_as_float(bits);
#else
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
#endif
}

/// P6 2c (appendix EH): fixed-point scale for the attention-mass probe. A page's mass is the sum of
/// softmax probabilities over the span's rows and the page's keys, i.e. at most one per row, so a span of
/// a few thousand rows stays far inside a u32 at 2^20 while keeping a single tiny contribution visible.
inline constexpr float kPagedKVPageScoreMassScale = 1048576.0f;

/// P6 2c B4 (appendix DH): the absolute position of the oldest query row allowed to vote on a page's
/// score, given the position of the call's last token and the requested span of newest rows.
///
/// The count statistic adds one record per (row, page) where that page is the row's new best, so it is
/// dominated by whichever rows are numerous: in a 1024-token prefill chunk the body rows outnumber the
/// question's by two orders of magnitude and a handful of shared pages win for most of them, while the
/// query signal lives in the newest rows only (measured in DG.2/DG.4). A span <= 0 disables the filter
/// and admits every row, which is the behaviour the archives were written with.
NINFER_PAGE_SCORE_HD std::int32_t paged_kv_page_score_row_begin(std::int32_t last_pos,
                                                               std::int32_t span) {
    return span > 0 ? last_pos - span + 1 : std::int32_t{0};
}

}  // namespace ninfer::ops
