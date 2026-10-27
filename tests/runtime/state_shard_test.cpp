// Does the shard plan belong to one model, or to the format?
//
// The first version of the planner refused outright when a single conv block or a single layer
// exceeded the frame budget, so it worked for the 35B model it was written against and would have
// failed on a larger one. These scenarios pin that down: the real 64 MB image, a bigger model whose
// layers still fit, a model whose single layer does not fit, and one whose conv block does not.
//
// The real image anchors the first case rather than a synthetic buffer, because a synthetic one
// would not show the awkward remainder (30 layers do not divide evenly into 6 MiB frames).

#include "core/state_image_shards.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace sis = ninfer::core::state_shards;

int failures = 0;

void expect(bool condition, const char* what) {
    std::printf("%s %s\n", condition ? "ok  " : "FAIL", what);
    if (!condition) {
        ++failures;
    }
}

bool read_file(const char* path, std::string& out) {
    std::FILE* file = std::fopen(path, "rb");
    if (file == nullptr) {
        return false;
    }
    std::fseek(file, 0, SEEK_END);
    const long size = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    if (size <= 0) {
        std::fclose(file);
        return false;
    }
    out.resize(static_cast<std::size_t>(size));
    const std::size_t got = std::fread(out.data(), 1, out.size(), file);
    std::fclose(file);
    return got == out.size();
}

// Contents do not matter, only that they survive the round trip.
std::string synthetic(std::uint64_t bytes, unsigned char seed) {
    std::string out(static_cast<std::size_t>(bytes), '\0');
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<char>((i * 31u + seed) & 0xFFu);
    }
    return out;
}

// One full scenario: plan, split, reassemble, and check the invariants that hold for any model.
void run_case(const char* name, const std::string& image, const sis::RegionSizes& sizes,
              std::uint64_t max_payload, bool expect_split_of_layer, bool expect_split_of_conv) {
    std::printf("\n--- %s ---\n", name);
    std::printf("image %zu B; conv %llu, layer %llu x %u, tail %llu; budget %llu\n", image.size(),
                static_cast<unsigned long long>(sizes.conv_bytes),
                static_cast<unsigned long long>(sizes.layer_bytes), sizes.layers,
                static_cast<unsigned long long>(sizes.tail_bytes),
                static_cast<unsigned long long>(max_payload));

    const auto plan = sis::plan_shards(sizes, max_payload);
    expect(!plan.empty(), "a plan exists");
    if (plan.empty()) {
        return;
    }

    bool all_fit = true;
    std::uint32_t summed_layers = 0;
    std::uint32_t conv_shards = 0;
    std::uint32_t layer_pieces = 0;
    std::uint64_t covered = 0;
    for (const auto& piece : plan) {
        if (piece.length > max_payload) {
            all_fit = false;
        }
        summed_layers += piece.layer_count;
        covered += piece.length;
        if (piece.region == sis::Region::Conv) {
            ++conv_shards;
        }
        if (piece.region == sis::Region::Recurrent) {
            ++layer_pieces;
        }
    }
    std::printf("   %zu shards (%u of them recurrent pieces)\n", plan.size(), layer_pieces);
    expect(all_fit, "every shard fits the payload limit");
    expect(covered == sis::image_bytes(sizes), "the plan covers the whole image");
    expect(summed_layers == sizes.layers, "layers are counted exactly once");
    if (expect_split_of_conv) {
        expect(conv_shards > 1, "an oversized conv block is split rather than refused");
    }
    if (expect_split_of_layer) {
        // One layer needs more than one shard, and the continuation pieces must not claim layers.
        expect(layer_pieces > sizes.layers,
               "an oversized layer is split rather than refused");
    }

    const auto split = sis::split(image, sizes, max_payload);
    expect(split.ok, "split succeeds");
    if (!split.ok) {
        std::printf("   %s\n", split.error.c_str());
        return;
    }
    std::size_t largest = 0;
    for (const auto& shard : split.shards) {
        largest = shard.size() > largest ? shard.size() : largest;
    }
    expect(largest - sis::kHeaderBytes <= max_payload,
           "no shard exceeds the limit once its header is included");

    std::string rebuilt;
    std::string error;
    const bool joined = sis::join(split.shards, rebuilt, error);
    expect(joined, "shards reassemble");
    if (!joined) {
        std::printf("   %s\n", error.c_str());
        return;
    }
    expect(rebuilt == image, "reassembled bytes are identical");
}

// The refusal cases only need one well-formed shard set, so they run once on the real image.
void run_refusals(const std::string& image, const sis::RegionSizes& sizes, std::uint64_t budget) {
    std::printf("\n--- refusals ---\n");
    const auto split = sis::split(image, sizes, budget);
    if (!split.ok) {
        std::printf("FAIL split failed, cannot test refusals\n");
        ++failures;
        return;
    }
    {
        auto missing = split.shards;
        missing.erase(missing.begin() + 3);
        std::string out;
        std::string why;
        expect(!sis::join(missing, out, why), "a missing shard is refused");
        std::printf("   %s\n", why.c_str());
    }
    {
        auto corrupt = split.shards;
        // Corrupt the magic. The first version of this test flipped byte 7, which is reserved and
        // unvalidated, so it asserted a refusal that never had a reason to happen.
        corrupt[1][0] = static_cast<char>(corrupt[1][0] ^ 0xFF);
        std::string out;
        std::string why;
        expect(!sis::join(corrupt, out, why), "a corrupted header is refused");
        std::printf("   %s\n", why.c_str());
    }
    {
        auto swapped = split.shards;
        std::swap(swapped[3], swapped[5]);
        std::string out;
        std::string why;
        expect(!sis::join(swapped, out, why),
               "out-of-order shards are refused rather than written to the wrong offset");
        std::printf("   %s\n", why.c_str());
    }
    {
        auto truncated = split.shards;
        truncated[2].resize(truncated[2].size() - 1);
        std::string out;
        std::string why;
        expect(!sis::join(truncated, out, why), "a truncated payload is refused");
        std::printf("   %s\n", why.c_str());
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) {
        std::printf("usage: state_shard_test <ckpt-*.state>\n");
        return 2;
    }
    const std::uint64_t budget = 6ull * 1024 * 1024;

    std::string file;
    if (!read_file(argv[1], file)) {
        std::printf("FAIL cannot read %s\n", argv[1]);
        return 1;
    }
    const std::size_t dump_header = 24;  // six uint32 of layout numbers
    if (file.size() <= dump_header) {
        std::printf("FAIL file is too short to hold a dump header\n");
        return 1;
    }
    const std::string real_image = file.substr(dump_header);

    // --- the model this was written against, real bytes -------------------
    sis::RegionSizes ours;
    ours.conv_bytes  = 49152ull * 30;          // 8192 channels x 3 x bf16
    ours.layer_bytes = 32ull * 128 * 128 * 4;  // 32 heads x 128 x 128, fp32
    ours.layers      = 30;
    ours.tail_bytes  = 4096;
    run_case("35B (real image)", real_image, ours, budget, false, false);
    run_refusals(real_image, ours, budget);

    // --- a bigger model: more layers, bigger recurrent matrix --------------
    sis::RegionSizes big;
    big.conv_bytes  = 3ull * 1024 * 1024;
    big.layer_bytes = 4ull * 1024 * 1024;
    big.layers      = 64;
    big.tail_bytes  = 8192;
    run_case("bigger model (64 layers x 4 MiB)", synthetic(sis::image_bytes(big), 0x21), big,
             budget, false, false);

    // --- one layer larger than a frame ------------------------------------
    sis::RegionSizes huge_layer;
    huge_layer.conv_bytes  = 1ull * 1024 * 1024;
    huge_layer.layer_bytes = 8ull * 1024 * 1024;
    huge_layer.layers      = 8;
    huge_layer.tail_bytes  = 4096;
    run_case("layer larger than a frame", synthetic(sis::image_bytes(huge_layer), 0x37),
             huge_layer, budget, true, false);

    // --- conv block larger than a frame -----------------------------------
    sis::RegionSizes huge_conv;
    huge_conv.conv_bytes  = 9ull * 1024 * 1024;
    huge_conv.layer_bytes = 2ull * 1024 * 1024;
    huge_conv.layers      = 4;
    huge_conv.tail_bytes  = 0;
    run_case("conv larger than a frame", synthetic(sis::image_bytes(huge_conv), 0x4D), huge_conv,
             budget, false, true);

    // --- a small model: one shard would suffice ---------------------------
    sis::RegionSizes small;
    small.conv_bytes  = 64ull * 1024;
    small.layer_bytes = 256ull * 1024;
    small.layers      = 8;
    small.tail_bytes  = 2048;
    run_case("small model (8 x 256 KiB)", synthetic(sis::image_bytes(small), 0x5F), small, budget,
             false, false);

    std::printf("\n%d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
