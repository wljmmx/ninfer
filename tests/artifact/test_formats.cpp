// T117: the artifact registry must be able to name every encoding the expert path can execute.
//
// Before this, a container could not describe block-scaled FP8 experts at all: the format and the
// layouts were missing from the closed registry, so an artifact carrying them failed to parse with
// "unknown tensor format" - a name problem, not a data problem. This test pins the names, the
// round trip, and the fact that each registered name leads to a geometry that accepts it.

#include "artifact/formats.h"
#include "artifact/schema.h"
#include "core/weight_view.h"

#include <iostream>
#include <string>
#include <string_view>
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

void expect_round_trip(std::string_view name, QType format) {
    expect(artifact::parse_format(name) == format, "parse_format(" + std::string(name) + ")");
    expect(artifact::format_name(format) == name, "format_name round trip for " + std::string(name));
}

void expect_layout_round_trip(std::string_view name, QuantLayout layout) {
    expect(artifact::parse_layout(name) == layout, "parse_layout(" + std::string(name) + ")");
    expect(artifact::layout_name(layout) == name,
           "layout_name round trip for " + std::string(name));
}

void expect_rejected(std::string_view name) {
    try {
        (void)artifact::parse_format(name);
    } catch (const artifact::ArtifactError&) {
        return;
    }
    std::cerr << "FAIL: unknown format " << name << " was accepted\n";
    ++failures;
}

} // namespace

int main() {
    // The block-scaled pair the routed expert path executes on Ada, in both tile sizes.
    expect_round_trip("fp8_e4m3fn_block128x128_bf16", QType::FP8_E4M3FN_BLOCK128X128_BF16);
    expect_round_trip("fp8_e4m3fn_block64x64_bf16", QType::FP8_E4M3FN_BLOCK64X64_BF16);
    expect_round_trip("nvfp4", QType::NVFP4);
    expect_round_trip("nf4_g16_e4m3fn", QType::NF4_G16_E4M3FN);
    expect_layout_round_trip("block_scale_m128_k128_v1", QuantLayout::BlockScaleM128K128);
    expect_layout_round_trip("block_scale_m64_k64_v1", QuantLayout::BlockScaleM64K64);
    expect_layout_round_trip("block_scale_k16_m128x4_v1", QuantLayout::BlockScaleK16M128x4);
    expect_layout_round_trip("batched_block_scale_k16_m128x4_v1",
                             QuantLayout::BatchedBlockScaleK16M128x4);

    // A name is only useful if the geometry behind it accepts the encoding: cross-check the chain
    // name -> layout -> weight_geometry for both block sizes.
    const std::vector<std::uint64_t> shape{1280, 2560};
    const auto block128 = weight_geometry(artifact::parse_format("fp8_e4m3fn_block128x128_bf16"),
                                          artifact::parse_layout("block_scale_m128_k128_v1"),
                                          shape);
    expect(block128.group_size == 128 && block128.scale_block_rows == 128,
           "128x128 name resolves to a 128 tile");
    const auto block64 = weight_geometry(artifact::parse_format("fp8_e4m3fn_block64x64_bf16"),
                                         artifact::parse_layout("block_scale_m64_k64_v1"), shape);
    expect(block64.group_size == 64 && block64.scale_block_rows == 64,
           "64x64 name resolves to a 64 tile");

    expect_rejected("fp8_e4m3fn_block32x32_bf16");
    expect_rejected("");
    expect(artifact::format_name(static_cast<QType>(0x5a5a)).empty(),
           "an unregistered format has no name");
    expect(artifact::layout_name(static_cast<QuantLayout>(0x5a5a)).empty(),
           "an unregistered layout has no name");

    std::cout << (failures == 0 ? "OK" : "FAIL") << " artifact format registry\n";
    return failures == 0 ? 0 : 1;
}
