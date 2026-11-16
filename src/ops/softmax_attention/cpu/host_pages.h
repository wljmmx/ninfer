#pragma once

// Split ownership, mirror knife 1: the host page mirror.
//
// WHAT THIS IS. `softmax_cpu::rk_attention_partial` (cpu/partial.h) reads a split's KV pages
// through four plane bases and `plane_offset()`, whose page stride is *inside one plane*:
//
//     offset = leading * page_tokens * (kv_head + kv_heads * page) + leading * token + d
//
// i.e. a plane is a contiguous [page][head][token][leading] array. `core/host_kv_arena.h`
// stores the same page content in a RECORD per page: [page][plane][head][token][leading],
// with `page_stride` spanning all of a layer's planes and their alignment padding. The two
// addresses agree inside one page and differ across pages, so a page set cannot be handed to
// the operator as a set of offsets - it has to be repacked into the operator's plane blocks.
// That repack is this header, and it is the whole reason a "mirror" exists.
//
// The mirror is therefore explicitly a copy, and its price is stated rather than hidden:
// every mirrored page costs `sum(plane payload bytes) = layout.page_stride` rounded down to
// the payload (typically 68 KiB for rk4v4 D256 x 4 KV heads, 132 KiB for rk8v4), once per
// mirror call. Callers that re-mirror per step pay that per step; see rk_host_mirror_layout's
// `total_bytes` for the exact number of the page set they asked for.
//
// WHICH PLANE IS WHICH. The planner writes one layer's planes in a fixed inventory
// (`models/qwen3_5/state/decoder_state.cpp::plan_cache`): key data, value data, key scale,
// value scale. This header never assumes that order; it identifies the layout family by
// matching each candidate `paged_kv_storage_layout()` against the layout's per-layer plane
// bytes and dtypes, and only then names the four planes. A layout that matches no family the
// CPU partial covers is refused (return false) instead of being mis-indexed.
//
// The block table handed to the operator is the row's logical-page table with the mirrored
// pages renumbered to 0..page_count-1 and every other entry negative (absent), which is the
// same convention paged_kv_address.cuh gives the GPU instance.

#include "core/host_kv_arena.h"
#include "core/paged_kv_storage.h"
#include "ops/softmax_attention/cpu/partial.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::softmax_cpu {

// The four planes of one layer, named, plus the codec identity they carry.
//
// The identity is the (K, V) codec PAIR and nothing else. A `packed_k` boolean used to sit beside
// it as a coarse "int4 vs int8 K" flag; it had no production reader (the seam goes
// `rk_host_kv_dtype_of(k_codec, v_codec)` -> RkHostKvDType -> `rk_host_kv_codecs`), and it was not
// even faithful for the E8 carriers, which are neither 128 B nor 256 B. The pair is the axis: it
// names exactly one family per row and one carrier size per side.
struct RkHostKvPlanes {
    int k_codes    = -1;
    int v_codes    = -1;
    int k_scales   = -1;
    int v_scales   = -1;
    int per_layer  = 0;
    int layers     = 0;
    RkKCodec k_codec = RkKCodec::Int4;
    RkVCodec v_codec = RkVCodec::Int4;
};

namespace rk_host_pages_detail {

struct Candidate {
    KvCacheStorage storage;
    RkKCodec k_codec;
    RkVCodec v_codec;
};

// The five families partial.h decodes; the list, not the order, is what makes the probe a
// derivation rather than an index guess.
inline constexpr Candidate kCandidates[] = {
    {KvCacheStorage::RotatedInt4KeyInt4ValueGroup64, RkKCodec::Int4, RkVCodec::Int4},
    {KvCacheStorage::RotatedInt8KeyInt4ValueGroup64, RkKCodec::Int8, RkVCodec::Int4},
    {KvCacheStorage::RK4V4E8, RkKCodec::E8Lattice, RkVCodec::Int4},
    {KvCacheStorage::RK2V4E8, RkKCodec::E8Root, RkVCodec::Int4},
    {KvCacheStorage::RK4V2E8, RkKCodec::E8Lattice, RkVCodec::E8Root},
};

// One plane of one page in the Host records: payload bytes and dtype must both match, so a
// U8 plane of one extent cannot be mistaken for an FP16 plane of another.
inline bool plane_matches(const HostKVPageLayout& layout, int index, const KVPlaneGeometry& want) {
    const KVPlaneGeometry& have = layout.geometry.planes[static_cast<std::size_t>(index)];
    if (have.dtype != want.dtype || have.leading_extent != want.leading_extent) { return false; }
    const std::size_t head_bytes =
        static_cast<std::size_t>(want.leading_extent) * layout.geometry.page_tokens *
        dtype_size(want.dtype);
    const std::size_t page_bytes = head_bytes * static_cast<std::size_t>(want.head_extent);
    const HostKVPlaneLayout& packed = layout.planes[static_cast<std::size_t>(index)];
    return packed.head_payload_bytes == head_bytes && packed.page_payload_bytes == page_bytes;
}

} // namespace rk_host_pages_detail

// Identify the layout family and the layer's four plane indices. `layer` selects which
// layer's block to name; the family is identified from layer 0 so every layer agrees.
[[nodiscard]] inline bool plan_rk_host_kv_planes(const HostKVPageLayout& layout, int layer,
                                                 RkHostKvPlanes& out) noexcept {
    using namespace rk_host_pages_detail;
    if (layer < 0) { return false; }
    for (const Candidate& candidate : kCandidates) {
        PagedKVStorageLayout storage{};
        try {
            storage = paged_kv_storage_layout(candidate.storage, kHeadDim);
        } catch (...) {
            continue;
        }
        const int per_layer = static_cast<int>(storage.planes_per_layer());
        if (per_layer <= 0 || layout.planes.size() < static_cast<std::size_t>(per_layer) ||
            layout.planes.size() % static_cast<std::size_t>(per_layer) != 0) {
            continue;
        }
        const int layers = static_cast<int>(layout.planes.size() / static_cast<std::size_t>(per_layer));
        if (layer >= layers) { continue; }
        const int base = layer * per_layer;
        const int heads = layout.geometry.planes[0].head_extent;
        // Every layer must carry the same inventory, or a block index means different planes
        // on different layers and the mirror would read another layer's bytes.
        for (int other = 0; other < layers; ++other) {
            const KVPlaneGeometry* plane = layout.geometry.planes.data() + other * per_layer;
            const KVPlaneGeometry key{.dtype = storage.key.data_dtype,
                                      .leading_extent = storage.key.data_leading_extent,
                                      .head_extent = heads};
            const KVPlaneGeometry value{.dtype = storage.value.data_dtype,
                                        .leading_extent = storage.value.data_leading_extent,
                                        .head_extent = heads};
            if (!plane_matches(layout, other * per_layer + 0, key) ||
                !plane_matches(layout, other * per_layer + 1, value)) {
                break;
            }
            if (storage.key.has_scale()) {
                const KVPlaneGeometry scale{.dtype = storage.key.scale_dtype,
                                            .leading_extent = storage.key.scale_leading_extent,
                                            .head_extent = heads};
                if (!plane_matches(layout, other * per_layer + 2, scale)) { break; }
            }
            if (storage.value.has_scale()) {
                const KVPlaneGeometry scale{.dtype = storage.value.scale_dtype,
                                            .leading_extent = storage.value.scale_leading_extent,
                                            .head_extent = heads};
                if (!plane_matches(layout, other * per_layer + 3, scale)) { break; }
            }
            if (other + 1 == layers) {
                int cursor    = base;
                out.k_codes   = cursor++;
                out.v_codes   = cursor++;
                out.k_scales  = storage.key.has_scale() ? cursor++ : -1;
                out.v_scales  = storage.value.has_scale() ? cursor++ : -1;
                out.per_layer = per_layer;
                out.layers    = layers;
                out.k_codec   = candidate.k_codec;
                out.v_codec   = candidate.v_codec;
                return out.k_scales >= 0 && out.v_scales >= 0;  // the partial needs both
            }
        }
    }
    return false;
}

// Where each of the four channels lives in the compact mirror, and what one page costs.
struct RkHostMirrorLayout {
    std::size_t plane_offset[4] = {0, 0, 0, 0};  // byte offset of each plane's block
    std::size_t plane_page[4]   = {0, 0, 0, 0};  // bytes one mirrored page takes in its block
    std::size_t page_bytes      = 0;             // sum of the four page terms
    std::size_t total_bytes     = 0;             // page_bytes * page_count
    std::uint32_t page_count    = 0;
    std::int64_t page_tokens    = 0;
};

// The operator's addressing fixes the page granularity (decode.h kPageSize and the
// block_table[key >> 6] index), so a layout with a different page size cannot be mirrored
// into it - refused here instead of silently mis-indexed later.
[[nodiscard]] inline RkHostMirrorLayout rk_host_mirror_layout(const HostKVPageLayout& layout,
                                                              const RkHostKvPlanes& planes,
                                                              std::uint32_t page_count) {
    if (static_cast<std::int64_t>(layout.geometry.page_tokens) != kPageSize) {
        throw std::invalid_argument("host page mirror: the layout's page size is not the operator's");
    }
    if (planes.k_codes < 0 || planes.v_codes < 0 || planes.k_scales < 0 || planes.v_scales < 0) {
        throw std::invalid_argument("host page mirror: layout has no rk K/V/scale planes");
    }
    RkHostMirrorLayout out;
    out.page_count  = page_count;
    out.page_tokens = static_cast<std::int64_t>(layout.geometry.page_tokens);
    const int indices[4] = {planes.k_codes, planes.v_codes, planes.k_scales, planes.v_scales};
    std::size_t cursor   = 0;
    for (int channel = 0; channel < 4; ++channel) {
        const HostKVPlaneLayout& plane = layout.planes[static_cast<std::size_t>(indices[channel])];
        if (plane.page_payload_bytes == 0) {
            throw std::invalid_argument("host page mirror: empty plane payload");
        }
        out.plane_page[channel]   = plane.page_payload_bytes;
        out.plane_offset[channel] = cursor;
        cursor += plane.page_payload_bytes * static_cast<std::size_t>(page_count);
        out.page_bytes += plane.page_payload_bytes;
    }
    out.total_bytes = cursor;
    return out;
}

// Repack `page_count` Host page records starting at record `first_page` into the compact
// mirror. Every page's four plane payloads are copied verbatim: the bytes are the same bytes
// the device page holds (the Host record is written by a plain page copy), and the only
// change is where they sit, so no encoding is transcoded here.
inline void rk_host_mirror_pages(std::byte* mirror, const RkHostMirrorLayout& target,
                                 const std::byte* records, const HostKVPageLayout& layout,
                                 const RkHostKvPlanes& planes, std::uint32_t first_page,
                                 std::uint32_t page_count) {
    if (mirror == nullptr || records == nullptr) {
        throw std::invalid_argument("host page mirror: null buffer");
    }
    if (page_count == 0 || page_count != target.page_count) {
        throw std::invalid_argument("host page mirror: page count does not match the sizing");
    }
    const int indices[4] = {planes.k_codes, planes.v_codes, planes.k_scales, planes.v_scales};
    for (int channel = 0; channel < 4; ++channel) {
        const HostKVPlaneLayout& plane = layout.planes[static_cast<std::size_t>(indices[channel])];
        for (std::uint32_t index = 0; index < page_count; ++index) {
            const std::byte* source =
                records + static_cast<std::size_t>(first_page + index) * layout.page_stride +
                plane.offset;
            std::byte* destination = mirror + target.plane_offset[channel] +
                                     static_cast<std::size_t>(index) * plane.page_payload_bytes;
            std::memcpy(destination, source, plane.page_payload_bytes);
        }
    }
}

// The operator's block table: the row's logical-page table with mirrored pages renumbered to
// [0, page_count) and everything else absent (-1). A mirrored entry that falls outside the
// window would address a page the mirror does not hold, so it is refused rather than clamped.
inline void rk_host_mirror_block_table(std::int32_t* destination, int destination_pages,
                                       const std::int32_t* source, int source_pages,
                                       std::uint32_t first_page, std::uint32_t page_count) {
    if (destination == nullptr || source == nullptr) {
        throw std::invalid_argument("host page mirror: null block table");
    }
    if (destination_pages < source_pages || source_pages <= 0) {
        throw std::invalid_argument("host page mirror: block table is too small");
    }
    for (int logical = 0; logical < source_pages; ++logical) {
        const std::int32_t physical = source[logical];
        std::int32_t mirrored       = -1;
        if (physical >= 0) {
            const auto page = static_cast<std::uint32_t>(physical);
            if (page >= first_page && page - first_page < page_count) {
                mirrored = static_cast<std::int32_t>(page - first_page);
            } else {
                throw std::invalid_argument(
                    "host page mirror: the row maps a page outside the mirrored window");
            }
        }
        destination[logical] = mirrored;
    }
}

// The operator's view over a filled mirror. `block_table` must already be the mirrored table
// (see rk_host_mirror_block_table); `kv_heads` is the call's KV head count, which must be the
// layout's head extent - the address arithmetic is shared between them.
[[nodiscard]] inline RkPartialKvPages rk_host_partial_pages(const std::byte* mirror,
                                                           const RkHostMirrorLayout& target,
                                                           const std::int32_t* block_table,
                                                           int kv_heads) {
    if (mirror == nullptr || block_table == nullptr || kv_heads <= 0) {
        throw std::invalid_argument("host page mirror: invalid operator view");
    }
    RkPartialKvPages pages;
    pages.k_codes = reinterpret_cast<const std::uint8_t*>(mirror + target.plane_offset[0]);
    pages.v_codes = reinterpret_cast<const std::uint8_t*>(mirror + target.plane_offset[1]);
    pages.k_scales = reinterpret_cast<const std::uint16_t*>(mirror + target.plane_offset[2]);
    pages.v_scales = reinterpret_cast<const std::uint16_t*>(mirror + target.plane_offset[3]);
    pages.block_table = block_table;
    pages.kv_heads    = kv_heads;
    return pages;
}

} // namespace ninfer::ops::detail::softmax_cpu
