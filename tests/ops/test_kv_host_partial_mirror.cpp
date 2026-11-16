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
//
// The second half of the file repeats the round trip for ALL FIVE rk families the CPU partial
// decodes (rk4v4, rk8v4, rk4v4-e8, rk2v4-e8, rk4v2-e8). The mirror derives a family from the
// layer's plane inventory, the family names exactly one (K, V) codec pair, and the family's own
// carrier bytes must reach the operator unchanged - so each family is read back with ITS codecs,
// not with whichever codec the int4 path happens to use.
#include "ops/softmax_attention/cpu/host_pages.h"

#include "ops/softmax_attention/dense/causal_cache/int8/host_partial.h"  // RkHostKvDType axis

#include "ops/op_tester.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
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

// The five rk families are told apart by their plane inventory, not by an index. What is graded is
// the axis production consumes: the (K, V) codec pair the inventory yields, and the RkHostKvDType
// that pair names (host_partial.h's single mapping). A former `packed_k` boolean is gone - it was a
// coarse int4-vs-int8 flag with no production reader, and it could not describe the E8 carriers.
int family_identification() {
    int failures = 0;
    struct Case {
        KvCacheStorage storage;
        RkKCodec k;
        RkVCodec v;
        ops::detail::RkHostKvDType dtype;
    };
    const Case cases[] = {
        {KvCacheStorage::RotatedInt4KeyInt4ValueGroup64, RkKCodec::Int4, RkVCodec::Int4,
         ops::detail::RkHostKvDType::Rk4v4},
        {KvCacheStorage::RotatedInt8KeyInt4ValueGroup64, RkKCodec::Int8, RkVCodec::Int4,
         ops::detail::RkHostKvDType::Rk8v4},
        {KvCacheStorage::RK2V4E8, RkKCodec::E8Root, RkVCodec::Int4,
         ops::detail::RkHostKvDType::Rk2v4E8},
        {KvCacheStorage::RK4V4E8, RkKCodec::E8Lattice, RkVCodec::Int4,
         ops::detail::RkHostKvDType::Rk4v4E8},
        {KvCacheStorage::RK4V2E8, RkKCodec::E8Lattice, RkVCodec::E8Root,
         ops::detail::RkHostKvDType::Rk4v2E8},
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
        ops::detail::RkHostKvDType derived{};
        const bool named =
            identified && ops::detail::rk_host_kv_dtype_of(planes.k_codec, planes.v_codec, derived);
        failures += expect("family " + std::to_string(static_cast<int>(one.storage)),
                           {identified ? static_cast<std::int64_t>(planes.k_codec) : -1,
                            static_cast<std::int64_t>(planes.v_codec),
                            named ? static_cast<std::int64_t>(derived) : -1},
                           {static_cast<std::int64_t>(one.k), static_cast<std::int64_t>(one.v),
                            static_cast<std::int64_t>(one.dtype)});
    }
    return failures;
}

// --- every rk family through the same mirror ------------------------------------------------
//
// The mirror's job is the REPACK, and the seam's job is to pick the family's own codec. So each
// family is graded on three things:
//   * the (K, V) codec pair host_pages.h identifies names exactly one RkHostKvDType, and that
//     dtype maps back to the same pair (the axis is one mapping, not two);
//   * the layer's carriers are the byte sizes that codec reads (a 96 B lattice plane decoded as
//     a 64 B cylinder would read the next channel's bytes);
//   * the mirrored page decodes to the same 256 codes as the same payload written in the
//     operator's own [head][token][leading] layout, and - for the int4/int8 carriers, whose
//     fixture code IS the decoded code - to the fixture value itself.
// The E8 carriers have no closed-form expectation here (their decode is a table plus a radius
// lookup), so their equality oracle is the reference plane, plus two properties only an E8
// reader can have: codes outside the int4 range, and (rk4v2-e8 V) a calibrated radius centre.

enum class PlaneKind { Int4, Int8, E8Cylinder, E8Lattice };

PlaneKind k_kind(RkKCodec codec) {
    switch (codec) {
    case RkKCodec::Int4: return PlaneKind::Int4;
    case RkKCodec::Int8: return PlaneKind::Int8;
    case RkKCodec::E8Root: return PlaneKind::E8Cylinder;
    case RkKCodec::E8Lattice: return PlaneKind::E8Lattice;
    }
    return PlaneKind::Int4;
}

PlaneKind v_kind(RkVCodec codec) {
    return codec == RkVCodec::Int4 ? PlaneKind::Int4 : PlaneKind::E8Cylinder;
}

// The carrier size paged_kv_storage_layout() must give each kind: two dims per int4 byte, one
// byte per int8 dim, 2 B per 8D subspace on an E8 cylinder, plus one residual byte per subspace
// on the lattice carrier.
std::int64_t expected_leading(PlaneKind kind) {
    switch (kind) {
    case PlaneKind::Int4: return kHeadDim / 2;
    case PlaneKind::Int8: return kHeadDim;
    case PlaneKind::E8Cylinder: return kE8CylinderPlane;
    case PlaneKind::E8Lattice: return kE8LatticePlane;
    }
    return -1;
}

// int8 K: the carrier holds one signed byte per dim, so the fixture byte IS the decoded code.
int int8_fixture(int layer, int channel, std::uint32_t page, int head, int token, int d) {
    const int raw = (static_cast<int>(page) * 131 + head * 29 + token * 7 + d * 3 + channel * 11 +
                     layer * 5) & 0xff;
    return static_cast<std::int8_t>(static_cast<std::uint8_t>(raw));
}

// E8 fixture. Three root codes whose stage-1 decode is stated here rather than computed: root 0
// is (-4,-4,0,...), root 112 is (-2 x 8) and root 239 is (+2 x 8). rad_axis keeps the axis nibble
// at 0 (+1 on dim 0) and walks the radius nibble through 0 (an all-zero subspace), 11 (radius
// 1.0) and 15 (radius 2.5198), so the decoded magnitudes reach 8. Subspace 0 of every row is
// pinned to root 239 / rad 0xF0: 3 * 2.5198 * 1.115 = 8.4 vs 3 * 2.5198 = 7.6 is what makes
// rk4v2-e8's calibrated V read-back differ from the stock cylinder read-back.
constexpr std::uint8_t kE8RootFixture[3] = {0, 112, 239};
constexpr std::uint8_t kE8RadFixture[3]  = {0x00, 0xB0, 0xF0};
constexpr std::uint8_t kE8PinnedRoot     = 239;
constexpr std::uint8_t kE8PinnedRad      = 0xF0;

// One (layer, channel, page, head, token) row of a carrier, in the carrier's own byte order.
std::vector<std::uint8_t> plane_row(PlaneKind kind, int layer, int channel, std::uint32_t page,
                                    int head, int token) {
    if (kind == PlaneKind::Int4) {
        std::vector<std::uint8_t> row(kHeadDim / 2, 0);
        for (int d = 0; d < kHeadDim; d += 2) {
            const int low  = fixture_code(layer, channel, page, head, token, d);
            const int high = fixture_code(layer, channel, page, head, token, d + 1);
            row[static_cast<std::size_t>(d / 2)] =
                static_cast<std::uint8_t>(((high & 0xf) << 4) | (low & 0xf));
        }
        return row;
    }
    if (kind == PlaneKind::Int8) {
        std::vector<std::uint8_t> row(kHeadDim, 0);
        for (int d = 0; d < kHeadDim; ++d) {
            row[static_cast<std::size_t>(d)] = static_cast<std::uint8_t>(
                int8_fixture(layer, channel, page, head, token, d));
        }
        return row;
    }
    const bool lattice = kind == PlaneKind::E8Lattice;
    std::vector<std::uint8_t> row(static_cast<std::size_t>(lattice ? kE8LatticePlane
                                                                  : kE8CylinderPlane),
                                  0);
    for (int s = 0; s < kE8Subspaces; ++s) {
        const std::size_t subspace = static_cast<std::size_t>(s);
        const int pick = (static_cast<int>(page) * 3 + head + token * 5 + s * 7 + channel * 2) % 3;
        row[2 * subspace]     = kE8RootFixture[pick];
        row[2 * subspace + 1] = kE8RadFixture[(static_cast<int>(page) + head * 2 + token + s +
                                               channel) % 3];
        if (lattice) {
            row[static_cast<std::size_t>(kE8CylinderPlane) + subspace] = static_cast<std::uint8_t>(
                static_cast<int>(page) * 37 + head * 13 + token * 7 + s * 5 + channel * 3);
        }
    }
    row[0] = kE8PinnedRoot;
    row[1] = kE8PinnedRad;
    return row;
}

std::uint16_t scale_fixture(std::uint32_t page, int channel, int group) {
    return static_cast<std::uint16_t>(0x3c00u + 0x10u * static_cast<unsigned>(group) +
                                      0x100u * (page & 0x7u) + 0x40u * static_cast<unsigned>(channel));
}

std::vector<std::int32_t> read_k_codes(RkKCodec codec, const RkPartialKvPages& pages,
                                       std::int64_t page, int head, int token) {
    std::vector<std::int32_t> codes(kHeadDim, 0);
    switch (codec) {
    case RkKCodec::Int4:
        rk_partial_detail::k_codes_row<RkKCodec::Int4>(pages.k_codes, pages.kv_heads, page, head,
                                                       token, codes.data());
        break;
    case RkKCodec::Int8:
        rk_partial_detail::k_codes_row<RkKCodec::Int8>(pages.k_codes, pages.kv_heads, page, head,
                                                       token, codes.data());
        break;
    case RkKCodec::E8Root:
        rk_partial_detail::k_codes_row<RkKCodec::E8Root>(pages.k_codes, pages.kv_heads, page,
                                                         head, token, codes.data());
        break;
    case RkKCodec::E8Lattice:
        rk_partial_detail::k_codes_row<RkKCodec::E8Lattice>(pages.k_codes, pages.kv_heads, page,
                                                            head, token, codes.data());
        break;
    }
    return codes;
}

std::vector<std::int32_t> read_v_codes(RkVCodec codec, const RkPartialKvPages& pages,
                                       std::int64_t page, int head, int token) {
    std::vector<std::int32_t> codes(kHeadDim, 0);
    if (codec == RkVCodec::Int4) {
        rk_partial_detail::v_codes_row<RkVCodec::Int4>(pages.v_codes, pages.kv_heads, page, head,
                                                       token, codes.data());
    } else {
        rk_partial_detail::v_codes_row<RkVCodec::E8Root>(pages.v_codes, pages.kv_heads, page, head,
                                                         token, codes.data());
    }
    return codes;
}

int one_family(KvCacheStorage storage, ops::detail::RkHostKvDType want_dtype) {
    int failures = 0;
    const std::string tag = ops::detail::rk_host_kv_codecs(want_dtype).name;

    // --- the layer's plane inventory, as the engine plans it --------------------------------
    const PagedKVStorageLayout stored = paged_kv_storage_layout(storage, kHeadDim);
    KVPageGeometry geometry;
    geometry.planes = {
        {stored.key.data_dtype, stored.key.data_leading_extent, kKvHeads, 256},
        {stored.value.data_dtype, stored.value.data_leading_extent, kKvHeads, 256},
        {stored.key.scale_dtype, stored.key.scale_leading_extent, kKvHeads, 256},
        {stored.value.scale_dtype, stored.value.scale_leading_extent, kKvHeads, 256},
    };
    const HostKVPageLayout layout = plan_host_kv_page_layout(geometry);
    RkHostKvPlanes planes;
    if (!plan_rk_host_kv_planes(layout, 0, planes)) {
        std::cout << "FAIL: " << tag << " host layout was not identified\n";
        return 1;
    }

    // --- the axis: one dtype per family, one codec pair per dtype ---------------------------
    ops::detail::RkHostKvDType derived{};
    const bool named = ops::detail::rk_host_kv_dtype_of(planes.k_codec, planes.v_codec, derived);
    failures += expect(tag + ": identified codec pair names the family",
                       {named ? 1 : 0, static_cast<std::int64_t>(derived)},
                       {1, static_cast<std::int64_t>(want_dtype)});
    const ops::detail::RkHostKvCodecs codecs = ops::detail::rk_host_kv_codecs(want_dtype);
    failures += expect(tag + ": family -> (K, V) codec pair",
                       {static_cast<std::int64_t>(codecs.k), static_cast<std::int64_t>(codecs.v)},
                       {static_cast<std::int64_t>(planes.k_codec),
                        static_cast<std::int64_t>(planes.v_codec)});

    const PlaneKind kk = k_kind(planes.k_codec);
    const PlaneKind vk = v_kind(planes.v_codec);
    const int k_leading = layout.geometry.planes[0].leading_extent;
    const int v_leading = layout.geometry.planes[1].leading_extent;
    failures += expect(tag + ": K carrier bytes per token-head", k_leading, expected_leading(kk));
    failures += expect(tag + ": V carrier bytes per token-head", v_leading, expected_leading(vk));

    // --- the fixture: 8 source pages, mirror the window [5, 8) ------------------------------
    constexpr std::uint32_t kPages    = 8;
    constexpr std::uint32_t kFirst    = 5;
    constexpr std::uint32_t kMirrored = 3;
    std::vector<std::byte> records(layout.page_stride * kPages);
    for (std::uint32_t page = 0; page < kPages; ++page) {
        for (int channel = 0; channel < 4; ++channel) {
            for (int head = 0; head < kKvHeads; ++head) {
                for (int token = 0; token < kPageSize; ++token) {
                    if (channel >= 2) {
                        for (int group = 0; group < kGroups; ++group) {
                            *record_plane_scale(records.data(), layout, 0, channel, page, head,
                                                token, group) = scale_fixture(page, channel, group);
                        }
                        continue;
                    }
                    const auto row = plane_row(channel == 0 ? kk : vk, 0, channel, page, head,
                                               token);
                    std::memcpy(record_plane_byte(records.data(), layout, 0, channel, page, head,
                                                  token, 0),
                                row.data(), row.size());
                }
            }
        }
    }

    const RkHostMirrorLayout target = rk_host_mirror_layout(layout, planes, kMirrored);
    std::vector<std::byte> mirror(target.total_bytes);
    rk_host_mirror_pages(mirror.data(), target, records.data(), layout, planes, kFirst, kMirrored);

    const std::vector<std::int32_t> source_table{5, 6, 7, -1, -1};
    std::vector<std::int32_t> table(source_table.size(), -1);
    rk_host_mirror_block_table(table.data(), static_cast<int>(table.size()), source_table.data(),
                               static_cast<int>(source_table.size()), kFirst, kMirrored);
    const RkPartialKvPages view_pages =
        rk_host_partial_pages(mirror.data(), target, table.data(), kKvHeads);

    // --- the same payload at the operator's own addresses, for one page ---------------------
    const std::size_t k_block = static_cast<std::size_t>(k_leading) * kPageSize * kKvHeads;
    const std::size_t v_block = static_cast<std::size_t>(v_leading) * kPageSize * kKvHeads;
    std::vector<std::vector<std::uint8_t>> reference_k(kMirrored);
    std::vector<std::vector<std::uint8_t>> reference_v(kMirrored);
    for (std::uint32_t index = 0; index < kMirrored; ++index) {
        reference_k[index].assign(k_block, 0);
        reference_v[index].assign(v_block, 0);
        for (int head = 0; head < kKvHeads; ++head) {
            for (int token = 0; token < kPageSize; ++token) {
                const std::size_t at =
                    static_cast<std::size_t>(token) + static_cast<std::size_t>(kPageSize) * head;
                const auto k_row = plane_row(kk, 0, 0, index + kFirst, head, token);
                const auto v_row = plane_row(vk, 0, 1, index + kFirst, head, token);
                std::memcpy(reference_k[index].data() + static_cast<std::size_t>(k_leading) * at,
                            k_row.data(), k_row.size());
                std::memcpy(reference_v[index].data() + static_cast<std::size_t>(v_leading) * at,
                            v_row.data(), v_row.size());
            }
        }
    }

    // --- read every sampled row back through the operator's decode --------------------------
    int rows_checked = 0;
    int rows_equal   = 0;
    int k_max_abs    = 0;
    int v_max_abs    = 0;
    for (std::uint32_t index = 0; index < kMirrored; ++index) {
        RkPartialKvPages reference;
        reference.k_codes  = reference_k[index].data();
        reference.v_codes  = reference_v[index].data();
        reference.kv_heads = kKvHeads;
        for (const int head : {0, 2, 3}) {
            for (const int token : {0, 17, 63}) {
                const auto codec_page = static_cast<std::int64_t>(index);
                const auto got_k = read_k_codes(codecs.k, view_pages, codec_page, head, token);
                const auto ref_k = read_k_codes(codecs.k, reference, 0, head, token);
                const auto got_v = read_v_codes(codecs.v, view_pages, codec_page, head, token);
                const auto ref_v = read_v_codes(codecs.v, reference, 0, head, token);
                std::int64_t mirror_k = 0;
                std::int64_t mirror_v = 0;
                for (int d = 0; d < kHeadDim; ++d) {
                    const auto e = static_cast<std::size_t>(d);
                    mirror_k += got_k[e] == ref_k[e] ? 0 : 1;
                    mirror_v += got_v[e] == ref_v[e] ? 0 : 1;
                    k_max_abs = std::max(k_max_abs, std::abs(got_k[e]));
                    v_max_abs = std::max(v_max_abs, std::abs(got_v[e]));
                }
                const std::string where = " " + std::to_string(index) + "/" +
                                          std::to_string(head) + "/" + std::to_string(token);
                failures += expect(tag + ": mirror == operator layout" + where, {mirror_k, mirror_v},
                                   {0, 0});

                // Carriers whose fixture code IS the decoded code: check the value itself.
                if (kk == PlaneKind::Int4 || kk == PlaneKind::Int8) {
                    std::int64_t analytic = 0;
                    for (int d = 0; d < kHeadDim; ++d) {
                        const auto e = static_cast<std::size_t>(d);
                        const int want_k =
                            kk == PlaneKind::Int4
                                ? fixture_code(0, 0, index + kFirst, head, token, d)
                                : int8_fixture(0, 0, index + kFirst, head, token, d);
                        analytic += got_k[e] == want_k ? 0 : 1;
                        // Both int4-K families carry a packed int4 V, so this covers the V side too.
                        if (vk != PlaneKind::Int4) { continue; }
                        analytic += got_v[e] == fixture_code(0, 1, index + kFirst, head, token, d)
                                        ? 0
                                        : 1;
                    }
                    failures += expect(tag + ": decoded == fixture code" + where, analytic, 0);
                }
                ++rows_checked;
                rows_equal += (mirror_k == 0 && mirror_v == 0) ? 1 : 0;
            }
        }
    }
    failures += expect(tag + ": every sampled row matched", {rows_equal, rows_checked},
                       {rows_checked, rows_checked});

    // --- E8 readers: magnitudes int4 cannot hold, and the calibrated V branch ---------------
    if (kk != PlaneKind::Int4 && kk != PlaneKind::Int8) {
        failures += expect(tag + ": K decode leaves the int4 range", k_max_abs >= 8 ? 1 : 0, 1);
    }
    if (vk == PlaneKind::E8Cylinder) {
        failures += expect(tag + ": V decode leaves the int4 range", v_max_abs >= 8 ? 1 : 0, 1);
    }
    if (codecs.v == RkVCodec::E8Root) {
        std::vector<int> calibrated(kHeadDim, 0);
        std::vector<int> stock(kHeadDim, 0);
        e8_cylinder_codes_row(view_pages.v_codes, kKvHeads, 0, 0, 0, calibrated.data(), true);
        e8_cylinder_codes_row(view_pages.v_codes, kKvHeads, 0, 0, 0, stock.data(), false);
        std::int64_t moved = 0;
        for (int d = 0; d < kHeadDim; ++d) {
            moved += calibrated[static_cast<std::size_t>(d)] == stock[static_cast<std::size_t>(d)]
                         ? 0
                         : 1;
        }
        failures += expect(tag + ": the V radius centre moves the codes", moved > 0 ? 1 : 0, 1);
        const auto operator_v = read_v_codes(codecs.v, view_pages, 0, 0, 0);
        std::int64_t differs = 0;
        for (int d = 0; d < kHeadDim; ++d) {
            differs += operator_v[static_cast<std::size_t>(d)] ==
                               calibrated[static_cast<std::size_t>(d)]
                           ? 0
                           : 1;
        }
        failures += expect(tag + ": the operator's V read is the calibrated one", differs, 0);
    }

    // --- both scale planes, read through the operator's own plane_offset --------------------
    std::int64_t scale_mismatches = 0;
    for (std::uint32_t index = 0; index < kMirrored; ++index) {
        for (int group = 0; group < kGroups; ++group) {
            const std::int64_t offset =
                plane_offset(kScalePlane, kKvHeads, static_cast<std::int64_t>(index), 2, 17, group);
            scale_mismatches += view_pages.k_scales[offset] == scale_fixture(index + kFirst, 2, group)
                                    ? 0
                                    : 1;
            scale_mismatches += view_pages.v_scales[offset] == scale_fixture(index + kFirst, 3, group)
                                    ? 0
                                    : 1;
        }
    }
    failures += expect(tag + ": scale planes read back", scale_mismatches, 0);

    std::printf("family %-9s dtype=%d codec=(K %d, V %d) carriers=(K %d B, V %d B) "
                "rows %d/%d equal max|K code|=%d max|V code|=%d -> %s\n",
                tag.c_str(), static_cast<int>(want_dtype), static_cast<int>(codecs.k),
                static_cast<int>(codecs.v), k_leading, v_leading, rows_equal, rows_checked,
                k_max_abs, v_max_abs, failures == 0 ? "PASS" : "FAIL");
    return failures;
}

int families_round_trip() {
    struct Case {
        KvCacheStorage storage;
        ops::detail::RkHostKvDType dtype;
    };
    // The family list is the axis itself (host_partial.h), so a family added there cannot be
    // skipped here without changing this loop's input.
    const Case cases[] = {
        {KvCacheStorage::RotatedInt4KeyInt4ValueGroup64, ops::detail::RkHostKvDType::Rk4v4},
        {KvCacheStorage::RotatedInt8KeyInt4ValueGroup64, ops::detail::RkHostKvDType::Rk8v4},
        {KvCacheStorage::RK4V4E8, ops::detail::RkHostKvDType::Rk4v4E8},
        {KvCacheStorage::RK2V4E8, ops::detail::RkHostKvDType::Rk2v4E8},
        {KvCacheStorage::RK4V2E8, ops::detail::RkHostKvDType::Rk4v2E8},
    };
    static_assert(std::size(cases) == std::size(ops::detail::kRkHostKvDTypes),
                  "the family list must cover the dtype axis");
    int failures = 0;
    for (const Case& one : cases) { failures += one_family(one.storage, one.dtype); }
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += mirror_round_trip();
    failures += wrong_page_size_refused();
    failures += family_identification();
    failures += families_round_trip();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " host page mirror\n";
    return failures == 0 ? 0 : 1;
}
