// Split ownership, mirror knife 1: the Host page record -> operator plane mirror.
//
// The operator (ops/softmax_attention/cpu/partial.h) addresses KV pages with
// `plane_offset()`: one contiguous [page][head][token][leading] array per channel. The Host
// tier stores a page RECORD per page: [page][plane][head][token][leading] plus alignment.
// The two agree inside a page and disagree across pages, so the mirror is a repack and the
// only honest way to accept it is to decode with the operator's own read-back paths.
//
// This test therefore:
//   * builds a Host page layout through plan_host_kv_page_layout (the same planner the engine
//     uses) for two layers of rk4v4 D256 x 4 KV heads;
//   * fills pages with distinct, per-(layer, page, head, token, dim) code nibbles and G64 scale
//     bits, written through the RECORD addressing;
//   * mirrors a window of those pages and reads them back with rk_partial_detail::k_codes_row /
//     v_codes_row (the int4 decode the partial kernel runs) and the FP16 scale plane;
//   * proves the logical -> mirrored block table remap, including the out-of-window refusal.
//
// A stride mistake cannot pass: each page carries its own nibble pattern, so reading page 1
// through page 0's stride yields the wrong codes.
#include "ops/softmax_attention/cpu/host_pages.h"

#include "ops/op_tester.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::ops::detail::softmax_cpu;

namespace {

constexpr int kKvHeads = 4;
constexpr int kLayers  = 2;
constexpr int kHeadDim = 256;
constexpr int kGroup   = 64;
constexpr int kGroups  = kHeadDim / kGroup;

int expect(const std::string& label, const std::vector<std::int64_t>& got,
           const std::vector<std::int64_t>& expected) {
    return verify_exact(label.c_str(), got, expected);
}

int expect(const std::string& label, std::int64_t got, std::int64_t expected) {
    return expect(label, std::vector<std::int64_t>{got}, std::vector<std::int64_t>{expected});
}

// One page's content, addressed exactly like the Host record the arena writes:
// plane_base + head * (leading * page_tokens * element_bytes) + token * leading * element_bytes
// + element * element_bytes. Codes are 1 byte per element, FP16 scales 2.
std::uint8_t* record_plane_element(std::byte* records, const HostKVPageLayout& layout, int layer,
                                   int channel, std::uint32_t page, int head, int token,
                                   int element_bytes, int element) {
    const int index                   = layer * 4 + channel;
    const HostKVPlaneLayout& plane    = layout.planes[static_cast<std::size_t>(index)];
    const int leading = layout.geometry.planes[static_cast<std::size_t>(index)].leading_extent;
    std::byte* base = records + static_cast<std::size_t>(page) * layout.page_stride + plane.offset;
    const std::size_t head_bytes = static_cast<std::size_t>(leading) *
                                   layout.geometry.page_tokens *
                                   static_cast<std::size_t>(element_bytes);
    return reinterpret_cast<std::uint8_t*>(base) + static_cast<std::size_t>(head) * head_bytes +
           static_cast<std::size_t>(token) * static_cast<std::size_t>(leading) *
               static_cast<std::size_t>(element_bytes) +
           static_cast<std::size_t>(element) * static_cast<std::size_t>(element_bytes);
}

std::uint8_t* record_plane_byte(std::byte* records, const HostKVPageLayout& layout, int layer,
                                int channel, std::uint32_t page, int head, int token,
                                int element) {
    return record_plane_element(records, layout, layer, channel, page, head, token, 1, element);
}

std::uint16_t* record_plane_scale(std::byte* records, const HostKVPageLayout& layout, int layer,
                                  int channel, std::uint32_t page, int head, int token,
                                  int group) {
    return reinterpret_cast<std::uint16_t*>(record_plane_element(
        records, layout, layer, channel, page, head, token, 2, group));
}

// Packed int4: byte d/2 holds element d in its low nibble and d+1 in its high nibble
// (decode.h unpack_int4 / rk4_codec.cuh). Codes stay in [-8, 7] so the 4-bit field is exactly
// the value the kernel's sign-extending rk4 decode returns.
int fixture_code(int layer, int channel, std::uint32_t page, int head, int token, int d) {
    return ((static_cast<int>(page) * 7 + layer * 5 + channel * 3 + head * 11 + token + d) & 0xf) -
           8;
}

KVPageGeometry rk4v4_geometry() {
    KVPageGeometry geometry;
    for (int layer = 0; layer < kLayers; ++layer) {
        geometry.planes.push_back({DType::U8, 128, kKvHeads, 256});   // K codes, int4-packed
        geometry.planes.push_back({DType::U8, 128, kKvHeads, 256});   // V codes, int4-packed
        geometry.planes.push_back({DType::FP16, 4, kKvHeads, 256});   // K G64 scales
        geometry.planes.push_back({DType::FP16, 4, kKvHeads, 256});   // V G64 scales
    }
    return geometry;
}

std::vector<std::int32_t> decoded_codes(const RkPartialKvPages& pages, int channel,
                                        std::int64_t mirrored_page, int head, int token) {
    std::vector<std::int32_t> codes(kHeadDim, 0);
    if (channel == 0) {
        rk_partial_detail::k_codes_row<RkKCodec::Int4>(pages.k_codes, pages.kv_heads,
                                                       mirrored_page, head, token, codes.data());
    } else {
        rk_partial_detail::v_codes_row<RkVCodec::Int4>(pages.v_codes, pages.kv_heads,
                                                       mirrored_page, head, token, codes.data());
    }
    return codes;
}

int mirror_round_trip() {
    const HostKVPageLayout layout = plan_host_kv_page_layout(rk4v4_geometry());
    RkHostKvPlanes planes;
    if (!plan_rk_host_kv_planes(layout, 0, planes)) {
        std::cout << "FAIL: rk4v4 host layout was not identified\n";
        return 1;
    }
    int failures = 0;
    failures += expect("plane inventory (layer 0)",
                       {planes.k_codes, planes.v_codes, planes.k_scales, planes.v_scales,
                        planes.per_layer, planes.layers},
                       {0, 1, 2, 3, 4, kLayers});
    RkHostKvPlanes second;
    const bool second_ok = plan_rk_host_kv_planes(layout, 1, second);
    failures += expect("plane inventory (layer 1, same family)",
                       {second_ok ? 1 : 0, second.k_codes, second.k_scales}, {1, 4, 6});

    // Eight source pages, mirroring the window [5, 8): an offset source window is what makes a
    // stride/renumber mistake visible.
    constexpr std::uint32_t kPages    = 8;
    constexpr std::uint32_t kFirst    = 5;
    constexpr std::uint32_t kMirrored = 3;
    std::vector<std::byte> records(layout.page_stride * kPages);
    for (std::uint32_t page = 0; page < kPages; ++page) {
        for (int layer = 0; layer < kLayers; ++layer) {
            for (int channel = 0; channel < 4; ++channel) {
                for (int head = 0; head < kKvHeads; ++head) {
                    for (int token = 0; token < 64; ++token) {
                        if (channel < 2) {
                            for (int d = 0; d < kHeadDim; d += 2) {
                                const int low  = fixture_code(layer, channel, page, head, token, d);
                                const int high =
                                    fixture_code(layer, channel, page, head, token, d + 1);
                                *record_plane_byte(records.data(), layout, layer, channel, page,
                                                   head, token, d / 2) =
                                    static_cast<std::uint8_t>(((high & 0xf) << 4) | (low & 0xf));
                            }
                        } else {
                            for (int g = 0; g < kGroups; ++g) {
                                // Distinct finite FP16 bit patterns per (layer, page, group).
                                *record_plane_scale(records.data(), layout, layer, channel, page,
                                                    head, token, g) = static_cast<std::uint16_t>(
                                    0x3c00u + 0x10u * g + 0x100u * page + 0x40u * layer);
                            }
                        }
                    }
                }
            }
        }
    }

    const RkHostMirrorLayout target = rk_host_mirror_layout(layout, planes, kMirrored);
    failures += expect("mirror page cost is the record's payload", target.page_bytes,
                       std::int64_t{128} * 64 * kKvHeads * 2 + std::int64_t{4} * 2 * 64 * kKvHeads * 2);
    std::vector<std::byte> mirror(target.total_bytes);
    rk_host_mirror_pages(mirror.data(), target, records.data(), layout, planes, kFirst, kMirrored);

    const std::vector<std::int32_t> source_table{5, 6, 7, -1, -1};
    std::vector<std::int32_t> table(source_table.size(), 0);
    rk_host_mirror_block_table(table.data(), static_cast<int>(table.size()), source_table.data(),
                               static_cast<int>(source_table.size()), kFirst, kMirrored);
    failures += expect("block table renumbers the window",
                       {table[0], table[1], table[2], table[3], table[4]}, {0, 1, 2, -1, -1});

    const RkPartialKvPages pages =
        rk_host_partial_pages(mirror.data(), target, table.data(), kKvHeads);

    // Read back every mirrored page through the operator's own decode, at several heads/tokens.
    for (std::uint32_t index = 0; index < kMirrored; ++index) {
        for (int channel = 0; channel < 2; ++channel) {
            for (int head = 0; head < kKvHeads; head += 3) {
                for (int token = 0; token < 64; token += 21) {
                    const auto codes = decoded_codes(pages, channel, index, head, token);
                    std::int64_t mismatches = 0;
                    for (int d = 0; d < kHeadDim; ++d) {
                        const int want = fixture_code(0, channel, index + kFirst, head, token, d);
                        mismatches += codes[static_cast<std::size_t>(d)] == want ? 0 : 1;
                    }
                    failures += expect("decoded " + std::string(channel == 0 ? "K" : "V") +
                                           " codes page " + std::to_string(index) + " head " +
                                           std::to_string(head) + " token " +
                                           std::to_string(token),
                                       mismatches, 0);
                }
            }
        }
    }

    // The scales are read by the partial through the same plane_offset the codes use.
    std::int64_t scale_mismatches = 0;
    for (std::uint32_t index = 0; index < kMirrored; ++index) {
        for (int g = 0; g < kGroups; ++g) {
            const std::int64_t offset =
                plane_offset(kScalePlane, kKvHeads, static_cast<std::int64_t>(index), 2, 17, g);
            const std::uint16_t want =
                static_cast<std::uint16_t>(0x3c00u + 0x10u * g + 0x100u * (index + kFirst));
            scale_mismatches += pages.k_scales[offset] == want ? 0 : 1;
            scale_mismatches += pages.v_scales[offset] == want ? 0 : 1;
        }
    }
    failures += expect("scale plane read-back", scale_mismatches, 0);

    // Out-of-window sources are refused, not clamped into another page's bytes.
    bool refused = false;
    try {
        std::vector<std::int32_t> stray{6, 2};  // logical 1 -> physical 2, outside [5, 8)
        rk_host_mirror_block_table(stray.data(), 2, stray.data(), 2, kFirst, kMirrored);
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    failures += expect("out-of-window page refused", refused ? 1 : 0, 1);
    return failures;
}

// A layout whose page granularity is not the operator's must be refused: the operator indexes
// block_table[key >> 6] and strides a page by kPageSize.
int wrong_page_size_refused() {
    KVPageGeometry geometry = rk4v4_geometry();
    geometry.page_tokens    = 32;
    const HostKVPageLayout layout = plan_host_kv_page_layout(geometry);
    RkHostKvPlanes planes;
    bool refused = false;
    try {
        (void)rk_host_mirror_layout(layout, planes, 1);
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    return expect("a 32-token page is refused by the operator's 64-token addressing",
                  refused ? 1 : 0, 1);
}

// The five rk families are told apart by their plane inventory, not by an index.
int family_identification() {
    int failures = 0;
    struct Case {
        KvCacheStorage storage;
        RkKCodec k;
        RkVCodec v;
        bool packed;
    };
    const Case cases[] = {
        {KvCacheStorage::RotatedInt4KeyInt4ValueGroup64, RkKCodec::Int4, RkVCodec::Int4, true},
        {KvCacheStorage::RotatedInt8KeyInt4ValueGroup64, RkKCodec::Int8, RkVCodec::Int4, false},
        {KvCacheStorage::RK2V4E8, RkKCodec::E8Root, RkVCodec::Int4, false},
        {KvCacheStorage::RK4V4E8, RkKCodec::E8Lattice, RkVCodec::Int4, true},
        {KvCacheStorage::RK4V2E8, RkKCodec::E8Lattice, RkVCodec::E8Root, true},
    };
    for (const Case& one : cases) {
        const PagedKVStorageLayout storage = paged_kv_storage_layout(one.storage, kHeadDim);
        KVPageGeometry geometry;
        geometry.planes = {
            {storage.key.data_dtype, storage.key.data_leading_extent, kKvHeads, 256},
            {storage.value.data_dtype, storage.value.data_leading_extent, kKvHeads, 256},
            {storage.key.scale_dtype, storage.key.scale_leading_extent, kKvHeads, 256},
            {storage.value.scale_dtype, storage.value.scale_leading_extent, kKvHeads, 256},
        };
        const HostKVPageLayout layout = plan_host_kv_page_layout(geometry);
        RkHostKvPlanes planes;
        const bool identified = plan_rk_host_kv_planes(layout, 0, planes);
        failures += expect("family " + std::to_string(static_cast<int>(one.storage)), 
                           {identified ? static_cast<std::int64_t>(planes.k_codec) : -1,
                            static_cast<std::int64_t>(planes.v_codec), planes.packed_k ? 1 : 0},
                           {static_cast<std::int64_t>(one.k), static_cast<std::int64_t>(one.v),
                            one.packed ? 1 : 0});
    }
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += mirror_round_trip();
    failures += wrong_page_size_refused();
    failures += family_identification();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " host page mirror\n";
    return failures == 0 ? 0 : 1;
}
