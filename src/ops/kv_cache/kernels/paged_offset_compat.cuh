#pragma once

// port (sm_89): the KV codecs and the rk append kernel carried over from the earlier implementation
// were written in ninfer::ops against headers that have since moved and been re-namespaced. Their
// bodies are worth keeping verbatim, so the translation lives here once:
//
//     load_vec / store_vec / pack_bf16x2    -> ninfer::ops::common   (common/cuda/memory.cuh,
//                                                                     common/numeric/convert.cuh)
//     warp_max / warp_sum / kFullWarpMask   -> ninfer::ops::common   (common/cuda/reduce.cuh)
//     hadamard_d32_columns_inplace,
//     hadamard_d64_fragment_inplace,
//     normalized_hadamard_d256_inplace      -> ninfer::ops::detail::kv
//                                                                     (kv_cache/kernels/hadamard.cuh;
//                                                                      identical signatures)
//     paged_kv_element_offset               -> kv::page_major_offset (same formula, term for term)
//
// The offset formula is identical:
//
//     earlier:   Leading*64*(head + Heads*page) + Leading*page_offset + component
//     this tree: page_major_head_offset(page, head) + Leading*offset + component
//              = Leading*64*(head + Heads*page) + Leading*offset + component
#include "ops/common/cuda/memory.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/kv_cache/kernels/hadamard.cuh"
#include "ops/kv_cache/kernels/layout.cuh"

#include <cstdint>

namespace ninfer::ops {

using common::kFullWarpMask;
using common::load_vec;
using common::pack_bf16x2;
using common::store_vec;
using common::warp_max;
using common::warp_sum;

using detail::kv::hadamard_d32_columns_inplace;
using detail::kv::hadamard_d64_fragment_inplace;
using detail::kv::normalized_hadamard_d256_inplace;

template <int LeadingExtent, int HeadExtent>
__device__ __forceinline__ std::int64_t paged_kv_element_offset(std::int32_t physical_page,
                                                               std::int32_t head,
                                                               std::int32_t page_offset,
                                                               std::int32_t leading) {
    return detail::kv::page_major_offset<LeadingExtent, HeadExtent>(physical_page, head, page_offset,
                                                                   leading);
}

} // namespace ninfer::ops
