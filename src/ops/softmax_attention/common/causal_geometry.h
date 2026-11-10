#pragma once
#include "ops/softmax_attention/common/head_mapping.cuh"

namespace ninfer::ops::detail {
inline constexpr int kCausalHeadDim = 256;

template <int HeadDim, int QueryHeads, int KVHeads>
struct CausalGeometry : AttentionHeadMapping<QueryHeads, KVHeads> {
    static_assert(HeadDim > 0 && HeadDim % 64 == 0);
    static constexpr int kHeadDim = HeadDim;
};

using CausalD256H24Kv4 = CausalGeometry<256, 24, 4>;
using CausalD256H16Kv2 = CausalGeometry<256, 16, 2>;
// Qwen4Exp indexed (QSA) layers: 24 query heads over 2 KV heads at D256, a GQA group of 12. It runs
// the same D256 MMA path as the other two; only the query/KV head ratio differs.
using CausalD256H24Kv2 = CausalGeometry<256, 24, 2>;
} // namespace ninfer::ops::detail
