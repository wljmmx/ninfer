#pragma once

#include "core/paged_kv_cache.h"

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kPagedKVPageShift = 6;
inline constexpr int kPagedKVPageMask  = kPagedKVPageSize - 1;

static_assert(kPagedKVPageSize == (1 << kPagedKVPageShift));

struct PagedKVDirectMetadata {
    const std::int32_t* table;

    __device__ __forceinline__ std::int32_t valid_tokens(std::int32_t width) const { return width; }

    __device__ __forceinline__ const std::int32_t* block_table() const { return table; }
};

template <bool Masked>
struct PagedKVBatchMetadata {
    const std::int32_t* tables;
    const std::int32_t* valid_columns;
    const std::int32_t* table_rows;
    std::int32_t table_stride;

    __device__ __forceinline__ std::int32_t valid_tokens(std::int32_t width) const {
        if constexpr (Masked) {
            const std::int32_t valid = valid_columns[0];
            return valid <= 0 ? 0 : (valid < width ? valid : width);
        }
        return width;
    }

    __device__ __forceinline__ const std::int32_t* block_table() const {
        return tables + static_cast<std::int64_t>(table_rows[0]) * table_stride;
    }
};

__device__ __forceinline__ std::int32_t paged_kv_physical_page(const std::int32_t* block_table,
                                                               std::int32_t position) {
    return block_table[position >> kPagedKVPageShift];
}

// A negative block-table entry marks a logical page as *absent*: the page holds no live
// key/value data, so every score that would read it must be forced to -inf exactly like a key
// that is causally out of range. Producing an absent entry is a policy decision (KV eviction,
// host tiering); the value must be negative and must not collide with a physical page id.
//
// Addressing stays in bounds: consumers clamp an absent entry to page 0 before computing
// offsets (see paged_kv_page_for_address), so a kernel never dereferences the sentinel. With no
// absent entry the predicate is constant true and results are unchanged.
inline constexpr std::int32_t kPagedKVPageAbsent = -1;

__device__ __forceinline__ bool paged_kv_page_present(const std::int32_t* block_table,
                                                      std::int32_t position) {
    return paged_kv_physical_page(block_table, position) >= 0;
}

// Physical page to address for a logical position: absent pages are redirected to page 0 so the
// load stays in bounds. The value read there is masked out of the attention by
// paged_kv_page_present().
__device__ __forceinline__ std::int32_t paged_kv_page_for_address(const std::int32_t* block_table,
                                                                 std::int32_t position) {
    const std::int32_t page = paged_kv_physical_page(block_table, position);
    return page < 0 ? 0 : page;
}

template <int LeadingExtent, int HeadExtent>
__device__ __forceinline__ std::int64_t paged_kv_page_head_offset(std::int32_t physical_page,
                                                                  std::int32_t head) {
    return static_cast<std::int64_t>(LeadingExtent) * kPagedKVPageSize *
           (static_cast<std::int64_t>(head) +
            static_cast<std::int64_t>(HeadExtent) * physical_page);
}

template <int LeadingExtent, int HeadExtent>
__device__ __forceinline__ std::int64_t
paged_kv_element_offset(std::int32_t physical_page, std::int32_t head, std::int32_t page_offset,
                        std::int32_t leading) {
    return paged_kv_page_head_offset<LeadingExtent, HeadExtent>(physical_page, head) +
           static_cast<std::int64_t>(LeadingExtent) * page_offset + leading;
}

template <int LeadingExtent, int HeadExtent>
__device__ __forceinline__ std::int64_t
paged_kv_element_offset(const std::int32_t* block_table, std::int32_t head, std::int32_t position,
                        std::int32_t leading) {
    return paged_kv_element_offset<LeadingExtent, HeadExtent>(
        paged_kv_physical_page(block_table, position), head, position & kPagedKVPageMask, leading);
}

} // namespace ninfer::ops
