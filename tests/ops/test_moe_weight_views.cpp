#include "core/weight_view.h"

#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;

namespace {

int failures = 0;

void expect(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

template <class Function>
void expect_throws(Function&& function, const std::string& what) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    std::cerr << "FAIL: " << what << " did not reject its input\n";
    ++failures;
}

// One 1280x2560 matrix: the 35B-A3B gate/up geometry, which is what the block-scaled expert bank
// carries. Every number below is derived by hand from the layout, not from the implementation.
void test_geometry(QType format, QuantLayout layout, std::uint64_t block) {
    constexpr std::uint64_t rows = 1280, columns = 2560;
    const auto g = weight_geometry(format, layout, std::vector<std::uint64_t>{rows, columns});

    const auto elements         = rows * columns;
    const auto scale_row_bytes  = columns / block * 2;
    const auto scale_bytes      = rows / block * scale_row_bytes;
    expect(g.format == format && g.layout == layout, "geometry keeps its format and layout");
    expect(g.group_size == block && g.scale_block_rows == block, "geometry keeps the tile size");
    expect(g.code_bytes_per_row == columns, "one FP8 code per element");
    expect(g.code_bytes == elements, "code plane covers every element");
    expect(g.scale_bytes_per_row == scale_row_bytes, "one BF16 scale per K tile");
    expect(g.scale_bytes == scale_bytes, "scale plane covers every row tile");
    expect(g.scale_offset == (elements + 255) / 256 * 256, "the scale plane is 256-byte aligned");
    expect(g.bytes == g.scale_offset + scale_bytes, "payload covers codes and scales");
    expect(g.divisor_count == 0, "block FP8 has no matrix divisor");

    // The tile is the only difference between the two forms: a coarser tile needs proportionally
    // fewer scale words, so the payload must shrink by the tile ratio squared.
    const auto other = weight_geometry(
        block == 64 ? QType::FP8_E4M3FN_BLOCK128X128_BF16 : QType::FP8_E4M3FN_BLOCK64X64_BF16,
        block == 64 ? QuantLayout::BlockScaleM128K128 : QuantLayout::BlockScaleM64K64,
        std::vector<std::uint64_t>{rows, columns});
    const auto& coarse = block == 64 ? other : g;
    const auto& fine   = block == 64 ? g : other;
    expect(other.code_bytes == g.code_bytes, "the tile size does not change the code plane");
    expect(fine.scale_bytes == coarse.scale_bytes * 4, "64x64 needs four times the scale bytes");
}

void test_rejections() {
    expect_throws(
        [] {
            (void)weight_geometry(QType::FP8_E4M3FN_BLOCK64X64_BF16,
                                  QuantLayout::BlockScaleM128K128,
                                  std::vector<std::uint64_t>{128, 128});
        },
        "a 64x64 format under the 128 tile");
    expect_throws(
        [] {
            (void)weight_geometry(QType::FP8_E4M3FN_BLOCK128X128_BF16,
                                  QuantLayout::BlockScaleM64K64,
                                  std::vector<std::uint64_t>{128, 128});
        },
        "a 128x128 format under the 64 tile");
    expect_throws(
        [] {
            (void)weight_geometry(QType::FP8_E4M3FN_BLOCK64X64_BF16,
                                  QuantLayout::BlockScaleM64K64,
                                  std::vector<std::uint64_t>{96, 128});
        },
        "N that is not a multiple of the tile");
    expect_throws(
        [] {
            (void)weight_geometry(QType::FP8_E4M3FN_BLOCK64X64_BF16,
                                  QuantLayout::BlockScaleM64K64,
                                  std::vector<std::uint64_t>{128, 200});
        },
        "K that is not a multiple of the tile");
}

// native_weight is the only place the scale tensor metadata is written for a quantized parent;
// a wrong stride there is a silent wrong answer, so pin every field.
void test_native_weight(QType format, QuantLayout layout, std::uint64_t block) {
    constexpr std::uint64_t rows = 256, columns = 256;
    const auto geometry = weight_geometry(format, layout,
                                          std::vector<std::uint64_t>{rows, columns});
    std::vector<std::byte> storage(geometry.bytes);
    std::memset(storage.data(), 0, storage.size());
    const WeightParent parent{geometry, storage.data()};

    WeightView view;
    view.shape = {rows, columns};
    view.parts.push_back({&parent, 0, rows * columns});
    const auto weight = native_weight(view);

    expect(weight.qtype == format && weight.layout == layout, "native weight keeps its encoding");
    expect(weight.payload == storage.data() && weight.payload_bytes == geometry.bytes,
           "native weight keeps the whole payload");
    expect(weight.qdata == storage.data(), "codes start at the parent origin");
    expect(weight.scales == storage.data() + geometry.scale_offset, "scales start at the plane");
    expect(weight.qhigh == nullptr && weight.high_plane_bytes == 0, "block FP8 has no high plane");
    expect(weight.scale_dtype == DType::BF16, "block scales are BF16");
    expect(weight.scale_ne[0] == static_cast<std::int32_t>(columns / block) &&
               weight.scale_ne[1] == static_cast<std::int32_t>(rows / block),
           "scale tensor is [K tiles, N tiles]");
    expect(weight.scale_nb[0] == 2 &&
               weight.scale_nb[1] == static_cast<std::int64_t>(geometry.scale_bytes_per_row) &&
               weight.scale_nb[2] == static_cast<std::int64_t>(geometry.scale_bytes),
           "scale byte strides follow the plane");
    expect(weight.group_size == block && weight.group == static_cast<std::int32_t>(block),
           "native weight reports its tile as the group");

    // A view that starts on a tile boundary keeps its codes and advances the scale plane by whole
    // tiles: this is the property a bank split relies on.
    WeightView second;
    second.shape = {block, columns};
    second.parts.push_back({&parent, block * columns, 2 * block * columns});
    const auto partial = native_weight(second);
    expect(partial.scales == storage.data() + geometry.scale_offset +
                                 geometry.scale_bytes_per_row,
           "a tile-aligned row view advances the scale plane");

    // ...and one that starts inside a tile must be refused rather than read the neighbour.
    WeightView split;
    split.shape = {block, columns};
    split.parts.push_back({&parent, block * columns / 2,
                           block * columns / 2 + block * columns});
    expect_throws([&] { (void)native_weight(split); }, "a row view that splits a scale tile");
}

void test_activation_scale_round_trip() {
    // The 64 tile must resolve scale words where weight_scale_offset says they are.
    const auto g = weight_geometry(QType::FP8_E4M3FN_BLOCK64X64_BF16, QuantLayout::BlockScaleM64K64,
                                   std::vector<std::uint64_t>{128, 128});
    expect(weight_scale_offset(g, 0, 0) == g.scale_offset, "first tile starts the plane");
    expect(weight_scale_offset(g, 64, 0) == g.scale_offset + g.scale_bytes_per_row,
           "second row tile advances by one scale row");
    expect(weight_scale_offset(g, 0, 1) == g.scale_offset + 2, "one BF16 word per K tile");
}

} // namespace

int main() {
    try {
        test_geometry(QType::FP8_E4M3FN_BLOCK64X64_BF16, QuantLayout::BlockScaleM64K64, 64);
        test_geometry(QType::FP8_E4M3FN_BLOCK128X128_BF16, QuantLayout::BlockScaleM128K128, 128);
        test_rejections();
        test_native_weight(QType::FP8_E4M3FN_BLOCK64X64_BF16, QuantLayout::BlockScaleM64K64, 64);
        test_native_weight(QType::FP8_E4M3FN_BLOCK128X128_BF16, QuantLayout::BlockScaleM128K128, 128);
        test_activation_scale_round_trip();
    } catch (const std::exception& error) {
        std::cerr << "weight view test exception: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " block-scaled FP8 weight views\n";
    return failures == 0 ? 0 : 1;
}
