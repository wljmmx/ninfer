#include "core/decode_graph.h"
#include "core/nf4.h"
#include "ninfer/ops/ple.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <bit>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kHeads = 16, kWidth = 160, kRowsPerPage = 45, kRowBytes = 90;
constexpr int kHe = kHeads * kWidth;

struct Shard {
    std::uint32_t first, rows;
    std::vector<std::uint8_t> bytes;
    std::vector<std::uint16_t> lut;
    GuardedDeviceBuffer pages, decode;

    Shard(std::uint32_t begin, std::uint32_t count, int number)
        : first(begin), rows(count),
          bytes(std::size_t((count + kRowsPerPage - 1) / kRowsPerPage) * 4096, 0xa5),
          lut(128 * 16, 0), pages(bytes.size()), decode(lut.size() * 2) {
        const auto table = make_nf4_bf16_lut(0.75F + number * 0.03125F);
        std::copy(table.words.begin(), table.words.end(), lut.begin());
        for (std::uint32_t r = 0; r < count; ++r) {
            auto* p = bytes.data() + (r / kRowsPerPage) * 4096 + (r % kRowsPerPage) * kRowBytes;
            for (int c = 0; c < 80; ++c) p[c] = (r * 71 + c * 29 + number * 13) & 255;
            for (int g = 0; g < 10; ++g) p[80 + g] = g ? (r * 17 + g * 13 + number) % 127 : 0;
            p[0] = 0x80; // signed-zero pair
        }
        pages.copy_from_host(bytes.data(), bytes.size());
        decode.copy_from_host(lut.data(), lut.size() * 2);
    }

    Tensor decode_view() {
        return Tensor(decode.data(), Device::cuda(0), DType::BF16, {16, 128});
    }

    ops::Nf4EmbeddingShard view() {
        return {{pages.data(), bytes.size(), Device::cuda(0), false}, decode_view(), first, rows};
    }

    std::uint16_t at(std::uint32_t row, int feature) const {
        const auto r      = row - first;
        const auto base   = (r / kRowsPerPage) * 4096 + (r % kRowsPerPage) * kRowBytes;
        const auto packed = bytes[base + feature / 2];
        const auto code   = (packed >> (4 * (feature % 2))) & 15;
        return lut[bytes[base + 80 + feature / 16] * 16 + code];
    }

    int preserved(const char* label) {
        const auto lut_label = std::string(label) + " LUT";
        return verify_exact(label, from_device<std::uint8_t>(pages.data(), bytes.size()), bytes) +
               verify_exact(lut_label.c_str(),
                            from_device<std::uint16_t>(decode.data(), lut.size()), lut) +
               pages.verify_guards(label) + decode.verify_guards(label);
    }
};

struct Fixture {
    int tokens, batch, stride, tail_stride;
    std::vector<int> ids, tail, valid, slots;
    GuardedDeviceBuffer d_ids, d_tail, d_valid, d_slots, output;
    Tensor input, counts, chosen, result;
    ops::NgramHashParameters hash;
    std::vector<std::unique_ptr<Shard>> table;

    Fixture(int t, int b, bool padded)
        : tokens(t), batch(b), stride(t + (padded ? 3 : 0)), tail_stride(padded ? 5 : 2),
          ids(stride * b), tail(tail_stride * (b + 2)), valid(b), slots(b),
          d_ids(ids.size() * 4), d_tail(tail.size() * 4), d_valid(b * 4), d_slots(b * 4),
          output(std::size_t(kHe) * t * b * 2),
          input(d_ids.data(), Device::cuda(0), DType::I32, {t, b}),
          counts(d_valid.data(), Device::cuda(0), DType::I32, {b}),
          chosen(d_slots.data(), Device::cuda(0), DType::I32, {b}),
          result(output.data(), Device::cuda(0), DType::BF16, {kHe, t, b}) {
        input.nb[1] = stride * 4;
        hash.multipliers = {0xd6e8feb86659fd93ULL, 0xa5a3564e27f8862bULL, 0x9e3779b97f4a7c15ULL};
        for (int h = 0; h < kHeads; ++h) {
            hash.vocab_sizes[h] = 120;
            hash.offsets[h]     = 0;
        }
        table.push_back(std::make_unique<Shard>(0, 60, 0));
        table.push_back(std::make_unique<Shard>(60, 60, 1));
        prepare();
    }

    std::vector<ops::Nf4EmbeddingShard> views() {
        std::vector<ops::Nf4EmbeddingShard> result;
        for (auto& shard : table) result.push_back(shard->view());
        return result;
    }

    void prepare() {
        for (int b = 0; b < batch; ++b)
            for (int t = 0; t < tokens; ++t) ids[b * stride + t] = (b * 37 + t * 101) % 97 + 1;
        for (int s = 0; s < batch + 2; ++s)
            for (int i = 0; i < tail_stride; ++i) tail[s * tail_stride + i] = (s * 13 + i * 7) % 89;
        for (int b = 0; b < batch; ++b) {
            valid[b] = b ? std::max(1, tokens - 2) : tokens;
            slots[b] = b % (batch + 2);
        }
        // eos is 7: force one predecessor to it, which masks only the older ID.
        if (ids.size() > 2) ids[2] = 7;
        d_ids.copy_from_host(ids.data(), ids.size() * 4);
        d_tail.copy_from_host(tail.data(), tail.size() * 4);
        d_valid.copy_from_host(valid.data(), valid.size() * 4);
        d_slots.copy_from_host(slots.data(), slots.size() * 4);
        cuda_synchronize();
    }

    void launch(DeviceContext& device) {
        auto shards = views();
        ops::ngram_embedding_lookup(input, counts, tail_view(), hash, 7, shards, result,
                                    device.execution_view());
    }

    Tensor tail_tensor() {
        auto tensor = Tensor(d_tail.data(), Device::cuda(0), DType::I32, {2, batch + 2});
        tensor.nb[1] = tail_stride * 4;
        return tensor;
    }

    ops::NgramTokenTailView tail_view() { return {tail_tensor(), chosen}; }

    std::uint32_t row_of(int h, int t, int b, int& owner) const {
        const auto previous = t ? ids[b * stride + t - 1]
                                : tail[slots[b] * tail_stride + 1];
        const auto older    = t >= 2 ? ids[b * stride + t - 2]
                             : t == 1 ? tail[slots[b] * tail_stride + 1]
                                      : tail[slots[b] * tail_stride];
        auto word = std::uint64_t(ids[b * stride + t]) * hash.multipliers[0] ^
                    std::uint64_t(previous) * hash.multipliers[1];
        const auto effective = previous == 7 ? 7 : older;
        if (h >= 8) word ^= std::uint64_t(effective) * hash.multipliers[2];
        auto row = std::bit_cast<std::int64_t>(word) % std::int64_t(hash.vocab_sizes[h]);
        if (row < 0) row += hash.vocab_sizes[h];
        const auto global = std::uint32_t(row + hash.offsets[h]);
        owner             = global < table[1]->first ? 0 : 1;
        return global;
    }

    int verify(const char* label) {
        const auto got = from_device<std::uint16_t>(output.data(), std::size_t(kHe) * tokens * batch);
        int failures   = 0;
        for (int b = 0; b < batch; ++b)
            for (int t = 0; t < tokens; ++t)
                for (int h = 0; h < kHeads; ++h) {
                    const auto base = std::size_t(b * tokens + t) * kHe;
                    if (t >= valid[b]) {
                        for (int d = 0; d < kWidth; ++d) failures += got[base + h * kWidth + d] != 0;
                        continue;
                    }
                    int owner = 0;
                    const auto row = row_of(h, t, b, owner);
                    for (int d = 0; d < kWidth; ++d)
                        failures += got[base + h * kWidth + d] != table[owner]->at(row, d);
                }
        if (failures) std::cerr << label << ": " << failures << " mismatched words\n";
        for (auto& shard : table) failures += shard->preserved(label);
        return failures;
    }
};

int rejects(const char* label, auto&& body) {
    try {
        body();
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << label << ": did not reject\n";
    return 1;
}

} // namespace

int main() {
    if (cuda_unavailable()) return 77;
    try {
        DeviceContext device;
        int failures = 0;
        for (const auto& [t, b, padded] : {std::tuple{1, 1, false}, std::tuple{5, 3, false},
                                           std::tuple{9, 2, true}, std::tuple{33, 8, true},
                                           std::tuple{257, 4, true}}) {
            Fixture f(t, b, padded);
            f.launch(device);
            device.synchronize();
            failures += f.verify("ngram lookup");

            auto shards = f.views();
            failures += rejects("lookup output alias", [&] {
                ops::ngram_embedding_lookup(f.input, f.counts, f.tail_view(), f.hash, 7, shards,
                                            f.input, device.execution_view());
            });
            auto bad = f.hash;
            bad.vocab_sizes[0] = 0;
            failures += rejects("lookup empty vocabulary", [&] {
                ops::ngram_embedding_lookup(f.input, f.counts, f.tail_view(), bad, 7, shards,
                                            f.result, device.execution_view());
            });
        }

        {
            Fixture f(17, 4, true);
            auto shards = f.views();
            DecodeGraphDefinition definition;
            DecodeGraphExecutable executable;
            definition.capture(device.stream, [&] {
                ops::ngram_embedding_lookup(f.input, f.counts, f.tail_view(), f.hash, 7, shards,
                                            f.result, device.execution_view());
            });
            executable.instantiate(definition);
            for (int phase = 0; phase < 2; ++phase) {
                f.prepare();
                f.ids[0] = 3 + phase;
                f.ids[f.stride] = 11 + phase;
                f.d_ids.copy_from_host(f.ids.data(), f.ids.size() * 4);
                executable.launch(device.stream);
                device.synchronize();
                failures += f.verify("ngram lookup graph");
            }
        }
        if (failures) return 1;
        std::cout << "OK sharded ngram embedding lookup, masked rows, Graph replay\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
