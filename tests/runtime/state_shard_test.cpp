// Does the shard plan actually tile a real state image, and does it refuse a bad one?
//
// Runs against the 64 MB image dumped from a live run rather than a synthetic buffer: the whole
// point of the split is that the real sizes have awkward remainders (30 layers do not divide by 3
// evenly into an 8 MB frame), and a synthetic case would not show it.
//
// The negative cases matter as much as the positive one -- a shard set arriving over a network is
// untrusted, and the failure mode to avoid is a silent reassembly with a hole in it.

#include "core/state_image_shards.h"

#include <cstdio>
#include <cstring>
#include <utility>
#include <string>
#include <vector>

namespace {

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

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) {
        std::printf("usage: state_shard_test <ckpt-*.state> [max_payload_bytes]\n");
        return 2;
    }
    const std::uint64_t max_payload =
        argc > 2 ? std::strtoull(argv[2], nullptr, 10) : (6ull * 1024 * 1024);

    std::string file;
    if (!read_file(argv[1], file)) {
        std::printf("FAIL cannot read %s\n", argv[1]);
        return 1;
    }
    // The dump writes six uint32 of layout numbers ahead of the image.
    const std::size_t dump_header = 24;
    if (file.size() <= dump_header) {
        std::printf("FAIL file is too short to hold a dump header\n");
        return 1;
    }
    const std::string image = file.substr(dump_header);

    // Sizes as measured from the live run: 30 linear-attention layers.
    ninfer::core::state_shards::RegionSizes sizes;
    sizes.conv_bytes   = 49152ull * 30;   // 8192 channels x 3 x bf16, all layers contiguous
    sizes.layer_bytes  = 32ull * 128 * 128 * 4;  // 32 heads x 128 x 128, fp32
    sizes.layers       = 30;
    sizes.tail_bytes   = 4096;            // continuation_hidden, bf16 x 2048

    const std::uint64_t described = ninfer::core::state_shards::image_bytes(sizes);
    std::printf("image      : %zu B\n", image.size());
    std::printf("described  : %llu B\n", static_cast<unsigned long long>(described));
    std::printf("max payload: %llu B (%.2f MiB)\n\n",
                static_cast<unsigned long long>(max_payload),
                static_cast<double>(max_payload) / 1048576.0);
    expect(described == image.size(), "the region sizes describe exactly this image");

    const auto plan = ninfer::core::state_shards::plan_shards(sizes, max_payload);
    expect(!plan.empty(), "a plan exists for these sizes");
    std::printf("   %zu shards\n", plan.size());
    bool all_fit = true;
    bool layers_only = true;
    for (const auto& piece : plan) {
        if (piece.length > max_payload) {
            all_fit = false;
        }
        std::printf("   [%s] offset %11llu  length %9u  layers %u+%u\n",
                    ninfer::core::state_shards::region_name(piece.region),
                    static_cast<unsigned long long>(piece.offset), piece.length,
                    piece.first_layer, piece.layer_count);
        // Recurrent shards must carry whole layers or none: a partial layer is unusable on its own.
        if (piece.region == ninfer::core::state_shards::Region::Recurrent &&
            piece.length % sizes.layer_bytes != 0) {
            layers_only = false;
        }
    }
    expect(all_fit, "every shard fits the payload limit");
    expect(layers_only, "every recurrent shard ends on a layer boundary");

    std::uint32_t summed_layers = 0;
    for (const auto& piece : plan) {
        summed_layers += piece.layer_count;
    }
    expect(summed_layers == sizes.layers, "the shards cover all 30 layers exactly once");

    // --- split and reassemble ---------------------------------------------
    const auto split = ninfer::core::state_shards::split(image, sizes, max_payload);
    expect(split.ok, "split succeeds");
    if (!split.ok) {
        std::printf("   %s\n", split.error.c_str());
        std::printf("\n%d failure(s)\n", failures);
        return 1;
    }
    std::size_t largest = 0;
    for (const auto& shard : split.shards) {
        largest = shard.size() > largest ? shard.size() : largest;
    }
    std::printf("   largest shard on the wire: %zu B (%.2f MiB)\n", largest,
                static_cast<double>(largest) / 1048576.0);
    expect(largest - ninfer::core::state_shards::kHeaderBytes <= max_payload,
           "no shard exceeds the limit once its header is included");

    std::string rebuilt;
    std::string error;
    const bool joined = ninfer::core::state_shards::join(split.shards, rebuilt, error);
    expect(joined, "shards reassemble");
    if (!joined) {
        std::printf("   %s\n", error.c_str());
    }
    expect(rebuilt == image, "reassembled bytes are identical to the source image");

    // --- the negative cases ------------------------------------------------
    {
        auto missing = split.shards;
        missing.erase(missing.begin() + 3);
        std::string out;
        std::string why;
        expect(!ninfer::core::state_shards::join(missing, out, why),
               "a missing shard is refused, not silently joined");
        std::printf("   refused for: %s\n", why.c_str());
    }
    {
        auto corrupt = split.shards;
        // Corrupt the magic, not a reserved byte: the first version of this test flipped byte 7,
        // which nothing validates, so it "passed" a header the reader had no reason to reject.
        corrupt[1][0] = static_cast<char>(corrupt[1][0] ^ 0xFF);
        std::string out;
        std::string why;
        expect(!ninfer::core::state_shards::join(corrupt, out, why),
               "a corrupted header is refused");
        std::printf("   refused for: %s\n", why.c_str());
    }
    {
        auto swapped = split.shards;
        std::swap(swapped[3], swapped[5]);
        std::string out;
        std::string why;
        expect(!ninfer::core::state_shards::join(swapped, out, why),
               "out-of-order shards are refused rather than written to the wrong offset");
        std::printf("   refused for: %s\n", why.c_str());
    }
    {
        auto short_payload = split.shards;
        short_payload[2].resize(short_payload[2].size() - 1);
        std::string out;
        std::string why;
        expect(!ninfer::core::state_shards::join(short_payload, out, why),
               "a truncated payload is refused");
        std::printf("   refused for: %s\n", why.c_str());
    }

    std::printf("\n%d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
