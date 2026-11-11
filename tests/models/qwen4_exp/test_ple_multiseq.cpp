// PLE for multi-sequence passes: the verify rectangle and the merged prefill batch.
//
// Both passes put several sequences in one window, and they differ in exactly one thing: a prefill
// accepts every live column, while a verify rectangle's acceptance is decided after the verifier
// runs and may be shorter. This test drives the stage the way those two entries do and compares
// each sequence against the same sequence run alone:
//
//   1. Verify: one window holds two sequences over four candidate columns; the layer is applied to
//      all four (the verifier needs every column's logits) and then committed by an accepted prefix
//      of two and three columns. Each row's nine-column pool must equal the pool of a run that only
//      ever saw that prefix, so the unaccepted columns never reached the history.
//   2. Prefill batch: one window holds two sequences over three columns, each row naming its own
//      slot, and both pools must equal two single-sequence runs of three columns.
//   3. The two slots' pools must differ, which is what "the convolution never reads one sequence's
//      columns as another's past" means in numbers.

#include "models/qwen4_exp/execution/ple_layer.h"

#include "artifact/reader.h"
#include "core/weight_view.h"
#include "ninfer/ops/gated_residual.h"
#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::json;
using namespace ninfer;
using namespace ninfer::models::qwen4_exp;

using ninfer::test::f32_to_bf16;

constexpr int kH              = int(kPleWidth);
constexpr int kC              = int(kPleStreams);
constexpr int kD              = int(kPleChannels);
constexpr int kN              = int(kPleProjectionRows);
constexpr int kTaps           = int(kPleKernelTaps);
constexpr std::int32_t kEos   = 2;
constexpr float kBlockEpsilon = 1e-6F;
constexpr std::uint32_t kSlots = 2;

int failures = 0;

void check(bool condition, const std::string& label) {
    if (condition) return;
    std::cerr << label << ": failed\n";
    ++failures;
}

// ---------------------------------------------------------------- the NF4 fixture artifact

constexpr std::uint32_t kRows      = 5;
constexpr std::uint32_t kPageBytes = 4096;

struct ArtifactDir {
    std::filesystem::path directory;
    std::filesystem::path entry;

    ArtifactDir() {
        const auto base = std::filesystem::temp_directory_path();
        for (int attempt = 0; attempt < 64; ++attempt) {
            const auto candidate =
                base / ("ninfer-ple-multiseq-" +
                        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                        "-" + std::to_string(attempt));
            std::error_code error;
            if (std::filesystem::create_directory(candidate, error)) {
                directory = candidate;
                break;
            }
        }
        if (directory.empty()) throw std::runtime_error("cannot create the PLE fixture directory");
        entry = directory / "model.ninfer";
    }

    ~ArtifactDir() {
        if (directory.empty()) return;
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    }

    ArtifactDir(const ArtifactDir&)            = delete;
    ArtifactDir& operator=(const ArtifactDir&) = delete;
    ArtifactDir(ArtifactDir&& other) noexcept
        : directory(std::move(other.directory)), entry(std::move(other.entry)) {
        other.directory.clear();
    }
};

void put_word(std::span<std::byte> bytes, std::size_t offset, std::uint64_t value, unsigned size) {
    for (unsigned i = 0; i < size; ++i) bytes[offset + i] = std::byte((value >> (8 * i)) & 255);
}

std::vector<std::byte> encode_row(int row) {
    std::vector<std::byte> bytes(kPleRowBytes);
    for (int i = 0; i < 80; ++i)
        bytes[std::size_t(i)] =
            std::byte(((row * 37 + i * 29) & 0x0f) | (((row * 11 + i * 7) & 0x0f) << 4));
    for (int i = 0; i < 10; ++i)
        bytes[std::size_t(80 + i)] = std::byte(8 + (row * 5 + i * 3) % 100);
    return bytes;
}

ArtifactDir write_fixture() {
    const std::size_t payload_bytes = kPageBytes + 4 + 280;
    std::vector<std::byte> payload(payload_bytes, std::byte{0});
    for (std::uint32_t row = 0; row < kRows; ++row) {
        const auto encoded = encode_row(int(row));
        std::copy(encoded.begin(), encoded.end(), payload.begin() + row * kPleRowBytes);
    }
    put_word(payload, kPageBytes, std::bit_cast<std::uint32_t>(1.0f), 4);
    std::array<std::int64_t, 35> controls{};
    controls[0] = 3;
    controls[1] = 5;
    controls[2] = 7;
    for (std::size_t i = 0; i < kPleHeads; ++i) {
        controls[3 + i]             = kRows;
        controls[3 + kPleHeads + i] = 0;
    }
    for (std::size_t i = 0; i < controls.size(); ++i)
        put_word(payload, kPageBytes + 4 + 8 * i, std::bit_cast<std::uint64_t>(controls[i]), 8);

    Json root = {
        {"components", {{"text", {{"config", Json::object()}, {"resources", Json::object()}}}}},
        {"objects",
         Json::array({{{"id", "ngram"},
                       {"kind", "tensor"},
                       {"shape", {kRows, kPleRowElements}},
                       {"format", "nf4_g16_e4m3fn"},
                       {"layout", "row_paged_g16_4k_v1"},
                       {"offset", 0},
                       {"bytes", kPageBytes + 4}},
                      {{"id", "controls"},
                       {"kind", "resource"},
                       {"encoding", "raw_bytes_v1"},
                       {"offset", kPageBytes + 4},
                       {"bytes", 280}}})},
        {"bindings", Json::object()},
        {"uses", Json::array()},
        {"files", Json::array({{{"path", nullptr}, {"payload_bytes", payload.size()}}})},
        {"metadata", {{"name", "ple-multiseq-fixture"}}}};

    ArtifactDir result;
    const auto text = root.dump();
    std::array<std::byte, 32> header{};
    const std::array<unsigned char, 8> magic{'N', 'I', 'N', 'F', 'E', 'R', 0, 3};
    for (std::size_t i = 0; i < magic.size(); ++i) header[i] = std::byte(magic[i]);
    put_word(header, 8, text.size(), 8);
    header[16] = std::byte{0x71};
    std::ofstream file(result.entry, std::ios::binary | std::ios::trunc);
    file.exceptions(std::ios::badbit | std::ios::failbit);
    file.write(reinterpret_cast<const char*>(header.data()), header.size());
    file.write(text.data(), static_cast<std::streamsize>(text.size()));
    const auto position = 32 + text.size();
    const auto aligned  = (position + 4095) / 4096 * 4096;
    const std::string padding(aligned - position, '\0');
    file.write(padding.data(), static_cast<std::streamsize>(padding.size()));
    file.write(reinterpret_cast<const char*>(payload.data()),
               static_cast<std::streamsize>(payload.size()));
    return result;
}

// ---------------------------------------------------------------- one stage and its step

void fill(std::vector<std::uint16_t>& words, std::uint32_t seed, float scale) {
    std::uint32_t state = seed * 2654435761U + 1013904223U;
    for (auto& word : words) {
        state              = state * 1664525U + 1013904223U;
        const int centered = int((state >> 9) & 0xffffU) - 32768;
        word               = f32_to_bf16(scale * float(centered) / 32768.0F);
    }
}

class Stage {
public:
    // `max_tokens`/`max_batch` are the widest step this stage must accept; `slots` is the history
    // pool depth. A stage owns its own pool, so two stages compared below start from the same zero
    // history.
    Stage(const PleTableSource& table, std::uint32_t max_tokens, std::uint32_t max_batch,
          std::uint32_t slots)
        : context_(), device_(Device::cuda(context_.device)) {
        std::vector<std::uint16_t> projection(static_cast<std::size_t>(kN) * kH);
        std::vector<std::uint16_t> gain_key(std::size_t(kH) * kC), gain_query(std::size_t(kH) * kC),
            gain_conv(std::size_t(kH) * kC), next_norm(std::size_t(kH) * kC),
            kernel(std::size_t(kD) * kTaps);
        fill(projection, 11, 0.02F);
        fill(gain_key, 12, 0.05F);
        fill(gain_query, 13, 0.05F);
        fill(gain_conv, 14, 0.05F);
        fill(next_norm, 17, 0.05F);
        fill(kernel, 15, 0.5F);
        projection_ = DeviceBuffer(projection.size() * 2);
        nk_         = DeviceBuffer(gain_key.size() * 2);
        nq_         = DeviceBuffer(gain_query.size() * 2);
        nc_         = DeviceBuffer(gain_conv.size() * 2);
        next_       = DeviceBuffer(next_norm.size() * 2);
        kernel_     = DeviceBuffer(kernel.size() * 2);
        projection_.copy_from_host(projection.data(), projection_.bytes);
        nk_.copy_from_host(gain_key.data(), nk_.bytes);
        nq_.copy_from_host(gain_query.data(), nq_.bytes);
        nc_.copy_from_host(gain_conv.data(), nc_.bytes);
        next_.copy_from_host(next_norm.data(), next_.bytes);
        kernel_.copy_from_host(kernel.data(), kernel_.bytes);

        test::direct_bf16_weight::HostWeight host_projection;
        host_projection.n    = kN;
        host_projection.k    = kH;
        host_projection.bits = projection;

        PleLayerWeights ple;
        ple.projection        = host_projection.device_weight(projection_.p);
        ple.projection.device = device_;
        ple.norm_key          = bf16(nk_.p, {kH, kC});
        ple.norm_query        = bf16(nq_.p, {kH, kC});
        ple.norm_conv         = bf16(nc_.p, {kH, kC});
        ple.convolution       = bf16(kernel_.p, {kD, kTaps});
        ple.rms_epsilon       = kBlockEpsilon;
        block_norm_           = DeviceBuffer(std::size_t(kH) * kC * 2);
        block_norm_.copy_from_host(next_norm.data(), block_norm_.bytes);

        std::vector<std::optional<PleLayerWeights>> layers;
        layers.push_back(std::nullopt);
        layers.push_back(ple);
        PlePrepareOptions options;
        options.max_positions = max_tokens * max_batch;
        options.windows       = 2;
        options.io_depth      = 4;
        options.cache_bytes   = 0;
        stage_ = std::make_unique<PleTextStage>(table, kEos, context_, std::move(layers),
                                                max_tokens, max_batch, slots, options);
        tokens_    = max_tokens;
        max_batch_ = max_batch;
    }

    // One pass: a window of `tokens` columns over `ids.size()` sequences. `ids` is row-major
    // (row b owns ids[b*tokens, (b+1)*tokens)), `valid` bounds each row's live columns, and
    // `accepted` - when given - is the prefix the pass may commit (a verify rectangle); otherwise
    // every live column is accepted (a prefill).
    void run(const std::vector<std::int32_t>& ids, std::span<const std::int32_t> slots,
             std::span<const std::uint32_t> valid, std::int32_t tokens,
             const std::vector<std::int32_t>* accepted) {
        const auto batch   = std::int32_t(slots.size());
        const auto columns = std::size_t(tokens) * std::size_t(batch);
        DeviceBuffer residual(std::size_t(kD) * columns * 2);
        DeviceBuffer normalized(std::size_t(kD) * columns * 2);
        std::vector<std::uint16_t> seed(static_cast<std::size_t>(kD) * columns);
        fill(seed, 21, 1.0F);
        residual.copy_from_host(seed.data(), residual.bytes);
        normalized.fill(0);

        Tensor residual4 = bf16(residual.p, {kH, kC, tokens, batch});
        Tensor normalized4 = bf16(normalized.p, {kH, kC, tokens, batch});
        stage_->begin(PleStepWindow{.ids = ids, .slots = slots, .valid_columns = valid,
                                    .tokens = tokens},
                      context_.execution_view());
        stage_->apply(1, residual4, normalized4, bf16(block_norm_.p, {kH, kC}), kBlockEpsilon,
                      context_.execution_view());
        if (accepted == nullptr) {
            stage_->commit(1, context_.execution_view());
        } else {
            DeviceBuffer d_accepted(accepted->size() * 4);
            d_accepted.copy_from_host(accepted->data(), d_accepted.bytes);
            stage_->commit(1, Tensor(d_accepted.p, Device::cuda(context_.device), DType::I32,
                                     {std::int32_t(accepted->size())}),
                           context_.execution_view());
        }
        context_.synchronize();
        stage_->end(context_.execution_view());
        context_.synchronize();
    }

    // The nine-column pool of one layer, as BF16 words [D,9,slots].
    [[nodiscard]] std::vector<std::uint16_t> history(std::uint32_t layer) const {
        return test::from_device<std::uint16_t>(stage_->history(layer).data,
                                                std::size_t(kD) * kPleHistory * kSlots);
    }

    [[nodiscard]] const Tensor& records(std::uint32_t layer) const { return stage_->records(layer); }

    [[nodiscard]] DeviceContext& context() noexcept { return context_; }

private:
    Tensor bf16(void* data, std::initializer_list<std::int32_t> shape) const {
        return Tensor(data, device_, DType::BF16, shape);
    }

    DeviceContext context_;
    Device device_;
    DeviceBuffer projection_, nk_, nq_, nc_, next_, kernel_, block_norm_;
    std::unique_ptr<PleTextStage> stage_;
    std::uint32_t tokens_    = 0;
    std::uint32_t max_batch_ = 0;
};

std::vector<std::uint16_t> column_of(const std::vector<std::uint16_t>& pool, std::size_t slot,
                                     std::size_t column) {
    std::vector<std::uint16_t> out(static_cast<std::size_t>(kD));
    for (std::size_t c = 0; c < std::size_t(kD); ++c)
        out[c] = pool[c + std::size_t(kD) * (column + kPleHistory * slot)];
    return out;
}

// One slot's [D,9] block of a pool laid out [D,9,slots].
std::vector<std::uint16_t> slot_block(const std::vector<std::uint16_t>& pool, std::size_t slot) {
    std::vector<std::uint16_t> out(static_cast<std::size_t>(kD) * kPleHistory);
    for (std::size_t j = 0; j < kPleHistory; ++j)
        for (std::size_t c = 0; c < std::size_t(kD); ++c)
            out[c + std::size_t(kD) * j] = pool[c + std::size_t(kD) * (j + kPleHistory * slot)];
    return out;
}

// The pool the stage would hold for a slot after being run alone for `tokens` columns and
// committing all of them. `ids` is the row's full candidate list; only its prefix is run.
std::vector<std::uint16_t> run_alone(const PleTableSource& table, std::int32_t slot,
                                     const std::vector<std::int32_t>& ids, std::int32_t tokens) {
    Stage stage(table, std::uint32_t(tokens), 1, kSlots);
    const std::array<std::int32_t, 1> slots{slot};
    const std::array<std::uint32_t, 1> valid{std::uint32_t(tokens)};
    const std::vector<std::int32_t> prefix(ids.begin(), ids.begin() + tokens);
    stage.run(prefix, slots, valid, tokens, nullptr);
    return slot_block(stage.history(1), std::size_t(slot));
}

// One row's record column `t` of the batch window, [D].
std::vector<std::uint16_t> record_of(const Tensor& records, std::int32_t tokens, std::size_t row,
                                     std::size_t column) {
    const auto words = test::from_device<std::uint16_t>(
        records.data, std::size_t(kD) * std::size_t(tokens) * 2);
    std::vector<std::uint16_t> out(static_cast<std::size_t>(kD));
    for (std::size_t c = 0; c < std::size_t(kD); ++c)
        out[c] = words[c + std::size_t(kD) * (column + std::size_t(tokens) * row)];
    return out;
}

double relative_l2(const std::vector<std::uint16_t>& got, const std::vector<std::uint16_t>& want) {
    double difference = 0, magnitude = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double a = double(ninfer::test::bf16_to_f32(got[i]));
        const double b = double(ninfer::test::bf16_to_f32(want[i]));
        difference += (a - b) * (a - b);
        magnitude += b * b;
    }
    return magnitude > 0 ? std::sqrt(difference / magnitude) : std::sqrt(difference);
}

// The linear Op's own A16 criterion: its gate allows one BF16 unit roundoff on a column's relative
// L2 and twice that as the gross bound, because the storage rounding and the accumulation order
// are both implementation choices. Two runs of the same sequence at different step shapes pick
// different kernels (and therefore different reduction orders), so their words are compared the way
// the Op's gate compares them, not bit for bit. What must be exact is the *structure* - which
// columns entered the pool - and that is checked separately, word for word, inside one window.
void check_within_criterion(const std::vector<std::uint16_t>& got,
                            const std::vector<std::uint16_t>& want, const std::string& label) {
    constexpr double kGrossBound = 2.0 / 256.0;
    for (std::size_t column = 0; column < kPleHistory; ++column) {
        const auto begin = got.begin() + std::ptrdiff_t(column) * kD;
        const std::vector<std::uint16_t> a(begin, begin + kD);
        const std::vector<std::uint16_t> b(want.begin() + std::ptrdiff_t(column) * kD,
                                           want.begin() + std::ptrdiff_t(column + 1) * kD);
        const auto error = relative_l2(a, b);
        if (error <= kGrossBound) continue;
        std::cerr << label << ": pool column " << column << " relative L2 " << error << '\n';
        ++failures;
        return;
    }
}

void check_verify_prefix() {
    const auto fixture = write_fixture();
    artifact::Reader reader(fixture.entry);
    const auto shard    = reader.find("ngram");
    const auto controls = reader.find("controls");
    PleTableSource table(reader, std::span(&shard, 1), controls);

    // Two sequences in one verify rectangle of four candidate columns.
    constexpr std::int32_t kTokens = 4;
    const std::array<std::int32_t, 8> ids{1, 2, 3, 4, 5, 6, 7, 8};
    const std::array<std::int32_t, 2> slots{0, 1};
    const std::array<std::uint32_t, 2> valid{4, 4};
    const std::vector<std::int32_t> accepted{2, 3};

    Stage verify(table, kTokens, 2, kSlots);
    verify.run(std::vector<std::int32_t>(ids.begin(), ids.end()), slots, valid, kTokens,
               &accepted);
    const auto pool = verify.history(1);
    // The pool itself: a fresh pool starts at zero, so only the accepted columns' records are in it,
    // in the newest positions. This is the structural proof that an unaccepted column never enters
    // the history, and it is bit-exact because both sides are the same window's words.
    for (std::size_t slot = 0; slot < kSlots; ++slot) {
        const auto count  = std::size_t(accepted[slot]);
        std::size_t wrong = 0, nonzero_old = 0;
        for (std::size_t j = 0; j < kPleHistory; ++j) {
            const auto column = column_of(pool, slot, j);
            if (j + count < kPleHistory) {
                nonzero_old += std::count_if(column.begin(), column.end(),
                                             [](std::uint16_t w) { return w != 0; });
                continue;
            }
            const auto expected =
                record_of(verify.records(1), kTokens, slot, j + count - kPleHistory);
            wrong += column != expected;
        }
        check(nonzero_old == 0,
              "a fresh pool's columns before the accepted prefix are exactly zero");
        check(wrong == 0, "every accepted column's record is the pool column the extract wrote");
    }

    // Each row against the same sequence run alone for its accepted prefix. The two runs are
    // different step shapes (2 or 3 columns against this window's 8), and a projection's
    // accumulation order follows the shape, so the comparison uses the linear Op's own A16
    // criterion - one BF16 unit roundoff on the relative L2 of a column - instead of bit equality.
    {
        const std::vector<std::int32_t> first{1, 2, 3, 4};
        check_within_criterion(slot_block(pool, 0), run_alone(table, 0, first, accepted[0]),
                               "slot 0's pool agrees with its accepted prefix run alone");
    }
    {
        const std::vector<std::int32_t> second{5, 6, 7, 8};
        check_within_criterion(slot_block(pool, 1), run_alone(table, 1, second, accepted[1]),
                               "slot 1's pool agrees with its accepted prefix run alone");
    }

    // The two slots really hold their own histories.
    check(column_of(pool, 0, kPleHistory - 1) != column_of(pool, 1, kPleHistory - 1),
          "the two sequences' pools are different, so no row read the other's past");
    std::cout << "  verify rectangle: 2 sequences x 4 candidate columns, accepted prefix 2/3\n";
}

void check_prefill_batch() {
    const auto fixture = write_fixture();
    artifact::Reader reader(fixture.entry);
    const auto shard    = reader.find("ngram");
    const auto controls = reader.find("controls");
    PleTableSource table(reader, std::span(&shard, 1), controls);

    constexpr std::int32_t kTokens = 3;
    const std::array<std::int32_t, 6> ids{1, 2, 3, 5, 6, 7};
    const std::array<std::int32_t, 2> slots{0, 1};
    const std::array<std::uint32_t, 2> valid{3, 3};

    Stage batch(table, kTokens, 2, kSlots);
    batch.run(std::vector<std::int32_t>(ids.begin(), ids.end()), slots, valid, kTokens, nullptr);
    const auto pool = batch.history(1);

    check_within_criterion(slot_block(pool, 0),
                           run_alone(table, 0, std::vector<std::int32_t>{1, 2, 3}, kTokens),
                           "the batch's first row agrees with that sequence run alone");
    check_within_criterion(slot_block(pool, 1),
                           run_alone(table, 1, std::vector<std::int32_t>{5, 6, 7}, kTokens),
                           "the batch's second row agrees with that sequence run alone");
    std::cout << "  prefill batch: 2 sequences x 3 columns, committed per slot\n";
}

} // namespace
int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::cout << "SKIP PLE multi-sequence (no CUDA device)\n";
            return 77;
        }
        check_verify_prefix();
        check_prefill_batch();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    if (failures) return 1;
    std::cout << "OK PLE multi-sequence: a verify rectangle commits its accepted prefix and a "
                 "batch prefill advances each sequence's own history\n";
    return 0;
}

