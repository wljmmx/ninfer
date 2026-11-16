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

// port (sm_89) V10-13: a batched NVFP4 parent keeps one matrix per expert behind a leading axis.
// The scale plane repeats once per matrix and a declared divisor array adds one word per expert.
// Every number below is derived by hand from the layout, not from the implementation.
void test_batched_geometry() {
    constexpr std::uint64_t experts = 3, rows = 256, columns = 128;
    const auto g = weight_geometry(QType::NVFP4, QuantLayout::BatchedBlockScaleK16M128x4,
                                   std::vector<std::uint64_t>{experts, rows, columns},
                                   WeightDivisors{experts});
    const auto elements         = experts * rows * columns;
    const auto scale_row_bytes  = columns / 16;
    const auto scale_per_matrix = rows * scale_row_bytes;
    expect(g.shape.size() == 3 && g.shape[0] == experts && g.shape[1] == rows &&
               g.shape[2] == columns,
           "batched geometry keeps its [E, N, K] shape");
    expect(g.elements == elements, "batched geometry covers every expert element");
    expect(g.code_bytes_per_row == columns / 2 && g.code_bytes == elements / 2,
           "the code plane packs every expert");
    expect(g.scale_bytes_per_row == scale_row_bytes && g.scale_block_rows == 1,
           "one E4M3 word per K group, one scale row per matrix row");
    expect(g.matrix_rows == rows && g.scale_bytes_per_matrix == scale_per_matrix,
           "one expert matrix owns its own scale block");
    expect(g.scale_bytes == experts * scale_per_matrix, "the scale plane repeats per expert");
    expect(g.scale_offset == (elements / 2 + 255) / 256 * 256,
           "codes then 256-byte aligned scales");
    expect(g.divisor_offset == g.scale_offset + g.scale_bytes, "divisors follow the scale plane");
    expect(g.divisor_count == experts, "a declared array counts one divisor per expert");
    expect(g.divisor_rows == rows, "one divisor word covers one expert matrix");
    expect(g.bytes == g.divisor_offset + experts * 4, "the payload ends after the divisor array");
    // Each expert's scale block starts where the swizzle helper resolves its first row; that is the
    // property an expert bank split relies on.
    for (std::uint64_t expert = 1; expert < experts; ++expert) {
        expect(weight_scale_offset(g, expert * rows, 0) ==
                   g.scale_offset + expert * g.scale_bytes_per_matrix,
               "expert scale blocks are independently addressable");
    }
    // ...and inside one expert the swizzle is the rank-two one, unchanged.
    expect(weight_scale_offset(g, rows + 1, 1) ==
               g.scale_offset + g.scale_bytes_per_matrix + 16 + 1,
           "the swizzle inside an expert repeats the rank-two plane");
}

// A block-scale layout may state its leading matrix axis by shape too: the tile plane then repeats
// once per matrix, exactly as the NVFP4 batched bank repeats its scale plane.
void test_tiled_block_rank_three() {
    constexpr std::uint64_t matrices = 4, rows = 128, columns = 128, tile = 64;
    const auto g = weight_geometry(QType::FP8_E4M3FN_BLOCK64X64_BF16, QuantLayout::BlockScaleM64K64,
                                   std::vector<std::uint64_t>{matrices, rows, columns});
    const auto scale_per_matrix = rows / tile * (columns / tile * 2);
    expect(g.matrix_rows == rows && g.scale_bytes_per_matrix == scale_per_matrix,
           "one matrix owns one tile plane");
    expect(g.code_bytes == matrices * rows * columns, "the code plane covers every matrix");
    expect(g.scale_bytes == matrices * scale_per_matrix, "the tile plane repeats per matrix");
    expect(g.divisor_count == 0 && g.divisor_rows == 0, "block FP8 still has no divisor plane");
    expect(weight_scale_offset(g, rows, 0) == g.scale_offset + g.scale_bytes_per_matrix,
           "the second matrix's tile plane starts after the first");
}

// The expert-file form packs the same experts as consecutive row blocks of one rank-two matrix.
// Both carriers must describe the same bytes, or an artifact and a batched bank would disagree
// about where an expert's scales and divisor begin.
void test_stacked_per_expert_divisors() {
    constexpr std::uint64_t experts = 3, rows = 256, columns = 128;
    const auto batched = weight_geometry(QType::NVFP4, QuantLayout::BatchedBlockScaleK16M128x4,
                                         std::vector<std::uint64_t>{experts, rows, columns},
                                         WeightDivisors{experts});
    const auto stacked = weight_geometry(QType::NVFP4, QuantLayout::BlockScaleK16M128x4,
                                         std::vector<std::uint64_t>{experts * rows, columns},
                                         WeightDivisors{experts});
    expect(stacked.divisor_count == experts && stacked.divisor_rows == rows,
           "a rank-two stack declares the same per-expert domains");
    expect(stacked.matrix_rows == experts * rows,
           "the rank-two shape names one matrix over every expert");
    expect(stacked.scale_bytes / stacked.divisor_count == batched.scale_bytes_per_matrix,
           "one expert's slice of the stacked plane is the batched scale block");
    expect(stacked.scale_bytes == batched.scale_bytes &&
               stacked.divisor_offset == batched.divisor_offset && stacked.bytes == batched.bytes,
           "the two expert carriers describe the same bytes");
}

// The undeclared default is still one matrix divisor, so a rank-two NVFP4 parent keeps exactly the
// payload it had before a divisor array could be declared.
void test_single_divisor_unchanged() {
    constexpr std::uint64_t rows = 256, columns = 128;
    const auto g = weight_geometry(QType::NVFP4, QuantLayout::BlockScaleK16M128x4,
                                   std::vector<std::uint64_t>{rows, columns});
    const auto codes = rows * columns / 2;
    expect(g.divisor_count == 1 && g.divisor_rows == rows, "the default is one matrix divisor");
    expect(g.divisor_offset == (codes + 255) / 256 * 256 + rows * (columns / 16),
           "the divisor follows the scale plane");
    expect(g.bytes == g.divisor_offset + 4, "the default payload ends after one word");
    expect(g.matrix_rows == rows && g.scale_bytes_per_matrix == g.scale_bytes,
           "one matrix owns the whole scale plane");
}

void test_format_rejections() {
    expect_throws(
        [] {
            (void)weight_geometry(QType::NVFP4, QuantLayout::BatchedBlockScaleK16M128x4,
                                  std::vector<std::uint64_t>{256, 128});
        },
        "a batched layout with rank two");
    expect_throws(
        [] {
            (void)weight_geometry(QType::NVFP4, QuantLayout::BatchedBlockScaleK16M128x4,
                                  std::vector<std::uint64_t>{2, 2, 256, 128});
        },
        "a batched layout with rank four");
    expect_throws(
        [] {
            (void)weight_geometry(QType::NVFP4, QuantLayout::BlockScaleK16M128x4,
                                  std::vector<std::uint64_t>{3, 256, 128});
        },
        "a rank-three tensor under the rank-two NVFP4 layout");
    expect_throws(
        [] {
            (void)weight_geometry(QType::NVFP4, QuantLayout::BatchedBlockScaleK16M128x4,
                                  std::vector<std::uint64_t>{2, 96, 128});
        },
        "a batched N that is not a whole 128-row tile");
    expect_throws(
        [] {
            (void)weight_geometry(QType::NVFP4, QuantLayout::BatchedBlockScaleK16M128x4,
                                  std::vector<std::uint64_t>{2, 256, 96});
        },
        "a batched K that is not a multiple of the 16-group times four");
    expect_throws(
        [] {
            (void)weight_geometry(QType::FP8_E4M3FN_BLOCK128X128_BF16,
                                  QuantLayout::BatchedBlockScaleK16M128x4,
                                  std::vector<std::uint64_t>{2, 256, 128});
        },
        "a batched layout carrying a non-NVFP4 format");
    expect_throws(
        [] {
            (void)weight_geometry(QType::NVFP4, QuantLayout::BatchedBlockScaleK16M128x4,
                                  std::vector<std::uint64_t>{3, 256, 128}, WeightDivisors{2});
        },
        "a per-expert array whose count differs from the expert axis");
    expect_throws(
        [] {
            (void)weight_geometry(QType::NVFP4, QuantLayout::BlockScaleK16M128x4,
                                  std::vector<std::uint64_t>{256, 128}, WeightDivisors{3});
        },
        "a rank-two stack whose rows do not divide by the declared count");
    expect_throws(
        [] {
            (void)weight_geometry(QType::NVFP4, QuantLayout::BlockScaleK16M128x4,
                                  std::vector<std::uint64_t>{384, 128}, WeightDivisors{2});
        },
        "a declared divisor domain that is not a whole 128-row tile");
    expect_throws(
        [] {
            (void)weight_geometry(QType::Q4_G64_FP16, QuantLayout::RowSplit,
                                  std::vector<std::uint64_t>{2, 256, 128});
        },
        "a grouped integer layout with a leading matrix axis");
    expect_throws(
        [] {
            (void)weight_geometry(QType::FP8_E4M3FN_BLOCK64X64_BF16, QuantLayout::BlockScaleM64K64,
                                  std::vector<std::uint64_t>{2, 2, 128, 128});
        },
        "a block-scaled FP8 layout with rank four");
    expect_throws(
        [] {
            (void)weight_geometry(QType::Q4_G64_FP16, QuantLayout::RowSplit,
                                  std::vector<std::uint64_t>{256, 128}, WeightDivisors{2});
        },
        "a per-expert array on a layout that has no divisor plane");
    expect_throws(
        [] {
            (void)weight_geometry(QType::NVFP4, QuantLayout::BlockScaleK16M128x4,
                                  std::vector<std::uint64_t>{256, 128}, WeightDivisors{0});
        },
        "a declared empty divisor array");
}

void test_batched_native_weight() {
    constexpr std::uint64_t experts = 2, rows = 256, columns = 128;
    const auto geometry = weight_geometry(QType::NVFP4, QuantLayout::BatchedBlockScaleK16M128x4,
                                          std::vector<std::uint64_t>{experts, rows, columns},
                                          WeightDivisors{experts});
    std::vector<std::byte> storage(geometry.bytes);
    std::memset(storage.data(), 0, storage.size());
    const std::vector<float> divisors{2.0F, 5.0F};
    WeightParent parent{geometry, storage.data()};
    parent.weight_divisors = divisors;

    WeightView second;
    second.shape = {rows, columns};
    second.parts.push_back({&parent, rows * columns, 2 * rows * columns});
    const auto weight = native_weight(second);
    expect(weight.layout == QuantLayout::BlockScaleK16M128x4,
           "a matrix view reports the rank-two layout");
    expect(weight.scales ==
               storage.data() + geometry.scale_offset + geometry.scale_bytes_per_matrix,
           "the scale plane rebases to the expert's own block");
    expect(weight.weight_scale_divisor == 5.0F, "the expert's own divisor is reported");

    WeightView first;
    first.shape = {rows, columns};
    first.parts.push_back({&parent, 0, rows * columns});
    expect(native_weight(first).weight_scale_divisor == 2.0F, "the first expert keeps its word");

    // A view that runs from one expert domain into the next has no single divisor to report.
    WeightView crossed;
    crossed.shape = {rows, columns};
    crossed.parts.push_back(
        {&parent, rows / 2 * columns, rows / 2 * columns + rows * columns});
    expect_throws([&] { (void)native_weight(crossed); }, "a view across two divisor domains");

    // Without the declared array there is nothing to read, so the view must not inherit a zero.
    WeightParent bare{geometry, storage.data()};
    WeightView third;
    third.shape = {rows, columns};
    third.parts.push_back({&bare, rows * columns, 2 * rows * columns});
    expect_throws([&] { (void)native_weight(third); }, "a missing per-expert divisor array");
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
        test_batched_geometry();
        test_tiled_block_rank_three();
        test_stacked_per_expert_divisors();
        test_single_divisor_unchanged();
        test_format_rejections();
        test_batched_native_weight();
    } catch (const std::exception& error) {
        std::cerr << "weight view test exception: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " block-scaled FP8 weight views\n";
    return failures == 0 ? 0 : 1;
}
