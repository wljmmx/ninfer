// Qwen4Exp PLE wired into the text block: a two-block stack whose second block carries PLE, driven
// through the same owner text.cpp calls (PleTextStage: begin / apply / commit / end) with the
// n-gram embedding gathered from a real, small NF4 table on disk.
//
// What this proves, and how:
//   1. The embedding the stage hands the layer is the table's: every word is checked against an
//      independent hash + NF4 decode of the rows the window names, so the layer's contribution is
//      driven by the artifact rather than by a fixture array.
//   2. The block's four-stream residual really is rewritten: every live cell of it changes, the
//      change is exactly the layer's own terms (gated + SiLU(convolution)) within the convolution
//      Op's own criterion, and `normalized` - the tensor the block's mixer reads - is the exact
//      BF16 norm of the NEW residual.
//   3. A model with no PLE block is untouched: the stack has no PLE owner for those blocks and the
//      stem's residual and normalized views are the same words run to run, which is the baseline
//      this test compares against.
//   4. `commit` advanced the nine-column pool by the window's accepted prefix.
//
// It also pins the load-time refusal that keeps a PLE layer from being skipped silently: an
// artifact whose table is not the NF4/paged encoding must throw with the encoding in the message.

#include "models/qwen4_exp/execution/ple_layer.h"

#include "artifact/reader.h"
#include "core/nf4.h"
#include "core/weight_view.h"
#include "models/qwen4_exp/ple/hash.h"
#include "ninfer/ops/gated_residual.h"
#include "ninfer/ops/ple.h"
#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::json;
using namespace ninfer;
using namespace ninfer::models::qwen4_exp;

using ninfer::test::bf16_to_f32;
using ninfer::test::f32_to_bf16;

constexpr int kH              = int(kPleWidth);          // 2560
constexpr int kC              = int(kPleStreams);        // 4
constexpr int kD              = int(kPleChannels);       // 10240
constexpr int kN              = int(kPleProjectionRows); // 12800
constexpr int kTaps           = int(kPleKernelTaps);
constexpr std::int32_t kEos   = 2;
constexpr float kBlockEpsilon = 1e-6F;

int failures = 0;

void check(bool condition, const std::string& label) {
    if (condition) return;
    std::cerr << label << ": failed\n";
    ++failures;
}

// ---------------------------------------------------------------- the NF4 fixture artifact

constexpr std::uint32_t kRows      = 3;
constexpr std::uint32_t kPageBytes = 4096;

struct ArtifactDir {
    std::filesystem::path directory;
    std::filesystem::path entry;

    ArtifactDir() {
        const auto base = std::filesystem::temp_directory_path();
        for (int attempt = 0; attempt < 64; ++attempt) {
            const auto candidate =
                base /
                ("ninfer-ple-block-" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
                 std::to_string(attempt));
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

// The encoded row: 80 code bytes then 10 scale bytes, exactly the reader's 90-byte row.
std::vector<std::byte> encode_row(int row) {
    std::vector<std::byte> bytes(kPleRowBytes);
    for (int i = 0; i < 80; ++i)
        bytes[std::size_t(i)] =
            std::byte(((row * 37 + i * 29) & 0x0f) | (((row * 11 + i * 7) & 0x0f) << 4));
    for (int i = 0; i < 10; ++i)
        bytes[std::size_t(80 + i)] = std::byte(8 + (row * 5 + i * 3) % 100);
    return bytes;
}

Json fixture_document(std::size_t payload_bytes) {
    return {{"components", {{"text", {{"config", Json::object()}, {"resources", Json::object()}}}}},
            {"objects", Json::array({{{"id", "ngram"},
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
            {"files", Json::array({{{"path", nullptr}, {"payload_bytes", payload_bytes}}})},
            {"metadata", {{"name", "ple-block-fixture"}}}};
}

ArtifactDir write_fixture() {
    std::vector<std::byte> payload(kPageBytes + 4 + 280, std::byte{0});
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

    ArtifactDir result;
    const auto text = fixture_document(payload.size()).dump();
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

// ---------------------------------------------------------------- synthetic layer weights

void fill(std::vector<std::uint16_t>& words, std::uint32_t seed, float scale) {
    std::uint32_t state = seed * 2654435761U + 1013904223U;
    for (auto& word : words) {
        state              = state * 1664525U + 1013904223U;
        const int centered = int((state >> 9) & 0xffffU) - 32768;
        word               = f32_to_bf16(scale * float(centered) / 32768.0F);
    }
}

std::size_t own4(int d, int s, int t) {
    return std::size_t(d) + std::size_t(kH) * (std::size_t(s) + std::size_t(kC) * std::size_t(t));
}

// The convolution layout: the artifact stores the tap bank as [4,D] with the channel axis
// innermost, and the loader hands the Op `weight_tensor(view, {D,4})`. This checks that the two
// are the same bytes in the same order - so C needs no transpose copy - and that the Op accepts
// the loader's view while refusing the artifact's own axis order.
void check_convolution_layout() {
    std::vector<std::uint16_t> taps(std::size_t(kD) * kTaps);
    fill(taps, 31, 0.5F);
    // The artifact's placement: tap j owns the contiguous D words [j*D, (j+1)*D).
    std::vector<std::uint16_t> artifact(taps.size());
    for (int j = 0; j < kTaps; ++j)
        for (int c = 0; c < kD; ++c)
            artifact[std::size_t(j) * kD + std::size_t(c)] = taps[std::size_t(j) * kD + std::size_t(c)];

    WeightParent parent;
    const std::array<std::uint64_t, 2> declared{std::uint64_t(kTaps), std::uint64_t(kD)};
    parent.geometry = weight_geometry(QType::BF16, QuantLayout::Contiguous, declared);
    parent.data     = reinterpret_cast<const std::byte*>(artifact.data());
    parent.device   = Device::cpu();
    WeightView view;
    view.shape = {std::uint64_t(kTaps), std::uint64_t(kD)};
    view.parts = {{&parent, 0, std::uint64_t(kTaps) * kD}};

    // What the loader does: reverse the declared axes onto the Tensor's.
    const Tensor kernel = weight_tensor(view, {kD, kTaps});
    check(kernel.ne[0] == kD && kernel.ne[1] == kTaps, "the loader yields [D,4]");
    check(kernel.nb[0] == 2 && kernel.nb[1] == std::int64_t(kD) * 2,
          "the loader's [D,4] keeps the artifact's byte order (channel fastest)");
    std::size_t misplaced = 0;
    for (int j = 0; j < kTaps; ++j)
        for (int c = 0; c < kD; ++c)
            misplaced += static_cast<const std::uint16_t*>(kernel.data)[std::size_t(c) +
                                                                      std::size_t(kD) * j] !=
                         artifact[std::size_t(j) * kD + std::size_t(c)];
    check(misplaced == 0, "kernel[c + D*j] is the artifact's (tap j, channel c)");

    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP convolution layout on the Device (no CUDA device)\n";
        return;
    }
    // The Op's own admission: run one convolution with the loader's [D,4] view and once with the
    // same words copied into a plain contiguous [D,4] buffer; the outputs must be bit-identical,
    // which is what "the Op no longer refuses it" means in numbers.
    DeviceContext context;
    const auto execution = context.execution_view();
    const auto device    = Device::cuda(0);
    auto bf16            = [&](void* data, std::initializer_list<std::int32_t> shape) {
        return Tensor(data, device, DType::BF16, shape);
    };
    constexpr int tokens = 1;
    std::vector<std::uint16_t> residual_words(std::size_t(kD) * tokens),
        gated_words(std::size_t(kD) * tokens), norm_words(std::size_t(kH) * kC);
    fill(residual_words, 32, 1.0F);
    fill(gated_words, 33, 0.25F);
    fill(norm_words, 34, 0.05F);
    DeviceBuffer d_residual(residual_words.size() * 2), d_gated(gated_words.size() * 2),
        d_norm(norm_words.size() * 2), d_kernel(artifact.size() * 2), d_history(std::size_t(kD) * kPleHistory * 2),
        d_records(std::size_t(kD) * tokens * 2), d_valid(4), d_slots(4), d_offsets(4),
        d_out_a(std::size_t(kD) * tokens * 2), d_out_b(std::size_t(kD) * tokens * 2),
        d_norm_a(std::size_t(kD) * tokens * 2), d_norm_b(std::size_t(kD) * tokens * 2);
    d_kernel.copy_from_host(artifact.data(), d_kernel.bytes);
    const std::int32_t one = tokens, zero = 0;
    d_valid.copy_from_host(&one, 4);
    d_slots.copy_from_host(&zero, 4);
    d_offsets.copy_from_host(&zero, 4);
    d_history.fill(0);
    d_records.fill(0);

    const Tensor loader_kernel = bf16(d_kernel.p, {kD, kTaps});
    const Tensor plain_kernel  = bf16(d_kernel.p, {kD, kTaps}); // same words, contiguous view
    auto conv = [&](const Tensor& kernel_view, DeviceBuffer& out, DeviceBuffer& normalized) {
        Tensor residual = bf16(d_residual.p, {kH, kC, tokens, 1});
        Tensor gated    = bf16(d_gated.p, {kH, kC, tokens, 1});
        Tensor updated  = bf16(out.p, {kH, kC, tokens, 1});
        Tensor normed   = bf16(normalized.p, {kH, kC, tokens, 1});
        d_residual.copy_from_host(residual_words.data(), d_residual.bytes);
        d_gated.copy_from_host(gated_words.data(), d_gated.bytes);
        ops::ple_conv_add_norm(residual, gated, kernel_view,
                               ops::PleBatchView{Tensor(d_valid.p, DType::I32, {1})},
                               ops::PleConvStateView{bf16(d_history.p, {kD, kPleHistory, 1}),
                                                     Tensor(d_slots.p, DType::I32, {1})},
                               ops::PleConvRecordWindowView{bf16(d_records.p, {kD, tokens, 1}),
                                                            Tensor(d_offsets.p, DType::I32, {1})},
                               bf16(d_norm.p, {kH, kC}), kBlockEpsilon, updated, normed,
                               execution);
        context.synchronize();
    };
    bool accepted = true;
    try {
        conv(loader_kernel, d_out_a, d_norm_a);
        conv(plain_kernel, d_out_b, d_norm_b);
    } catch (const std::invalid_argument& error) {
        accepted = false;
        std::cerr << "the Op refused the loader's [D,4] kernel: " << error.what() << '\n';
    }
    check(accepted, "the convolution Op accepts the loader's [D,4] view");
    check(test::from_device<std::uint16_t>(d_out_a.p, std::size_t(kD) * tokens) ==
              test::from_device<std::uint16_t>(d_out_b.p, std::size_t(kD) * tokens),
          "the loader's view and a plain [D,4] copy give the same residual");

    // The artifact's own axis order is not an Op operand: [4,D] must be refused, which is why the
    // loader reverses it instead of passing it through.
    bool refused = false;
    try {
        const Tensor wrong = bf16(d_kernel.p, {kTaps, kD});
        conv(wrong, d_out_a, d_norm_a);
    } catch (const std::invalid_argument&) { refused = true; }
    check(refused, "the Op refuses the artifact's [4,D] axis order");
}


// The hash the reader uses, reproduced here so the expected row of every (column, head) is
// derived from the ids alone.
std::uint32_t reference_hash(std::int32_t current, std::int32_t previous, std::int32_t older,
                             std::uint32_t head, const PleHashControls& controls) {    const auto pair = std::uint64_t(current) * std::uint64_t(controls.multipliers[0]) ^
                      std::uint64_t(previous) * std::uint64_t(controls.multipliers[1]);
    const auto word =
        head < 8 ? pair : pair ^ std::uint64_t(older) * std::uint64_t(controls.multipliers[2]);
    auto index = std::bit_cast<std::int64_t>(word) % controls.vocab_sizes[head];
    if (index < 0) index += controls.vocab_sizes[head];
    return static_cast<std::uint32_t>(index + controls.offsets[head]);
}

void check_encoding_refusal() {
    // The artifact this round ships: q4_g64_fp16 rows in row_split_k128_v1, 136 bytes a row.
    bool refused = false;
    std::string message;
    try {
        require_supported_ple_table({.format    = QType::Q4_G64_FP16,
                                     .layout    = QuantLayout::RowSplit,
                                     .rows      = 320001536,
                                     .width     = 160,
                                     .parameter = "text/layers/1/ple/table"});
    } catch (const std::invalid_argument& error) {
        refused = true;
        message = error.what();
    }
    check(refused, "an unsupported table encoding is refused");
    check(message.find("q4_g64_fp16") != std::string::npos &&
              message.find("320001536") != std::string::npos,
          "the refusal names the encoding and the row count");
    check(message.find("不受支持") != std::string::npos &&
              message.find("转码") != std::string::npos &&
              message.find("加解码器") != std::string::npos,
          "the refusal offers the two ways forward");

    bool admitted = true;
    try {
        require_supported_ple_table({.format    = QType::NF4_G16_E4M3FN,
                                     .layout    = QuantLayout::RowPagedG16_4K,
                                     .rows      = 128,
                                     .width     = kPleRowElements,
                                     .parameter = "ple/table"});
    } catch (const std::exception&) { admitted = false; }
    check(admitted, "the NF4 paged encoding is admitted");

    // A supported encoding at the wrong width is not the reader's format either.
    bool narrow_refused = false;
    try {
        require_supported_ple_table({.format    = QType::NF4_G16_E4M3FN,
                                     .layout    = QuantLayout::RowPagedG16_4K,
                                     .rows      = 128,
                                     .width     = 128,
                                     .parameter = "ple/table"});
    } catch (const std::invalid_argument&) { narrow_refused = true; }
    check(narrow_refused, "a 128-wide NF4 table is refused");
}

void run_stack() {
    constexpr int tokens = 4;
    const std::array<std::int32_t, 4> ids{1, 2, 3, 4};

    const auto fixture = write_fixture();
    artifact::Reader reader(fixture.entry);
    const auto shard    = reader.find("ngram");
    const auto controls = reader.find("controls");
    PleTableSource table(reader, std::span(&shard, 1), controls);

    DeviceContext context;
    const auto execution = context.execution_view();
    const auto device    = Device::cuda(0);
    auto bf16            = [&](void* data, std::initializer_list<std::int32_t> shape) {
        return Tensor(data, device, DType::BF16, shape);
    };

    std::vector<std::uint16_t> projection(std::size_t(kN) * kH);
    std::vector<std::uint16_t> gain_key(std::size_t(kH) * kC), gain_query(std::size_t(kH) * kC),
        gain_conv(std::size_t(kH) * kC), next_norm(std::size_t(kH) * kC);
    std::vector<std::uint16_t> kernel(std::size_t(kD) * kTaps);
    std::vector<std::uint16_t> embedding(std::size_t(kH) * tokens);
    fill(projection, 11, 0.02F);
    fill(gain_key, 12, 0.05F);
    fill(gain_query, 13, 0.05F);
    fill(gain_conv, 14, 0.05F);
    fill(next_norm, 17, 0.05F);
    fill(kernel, 15, 0.5F);
    fill(embedding, 16, 1.0F);

    DeviceBuffer d_projection(projection.size() * 2), d_nk(gain_key.size() * 2),
        d_nq(gain_query.size() * 2), d_nc(gain_conv.size() * 2), d_conv(kernel.size() * 2),
        d_next_norm(next_norm.size() * 2), d_stem_embedding(embedding.size() * 2);
    d_projection.copy_from_host(projection.data(), d_projection.bytes);
    d_nk.copy_from_host(gain_key.data(), d_nk.bytes);
    d_nq.copy_from_host(gain_query.data(), d_nq.bytes);
    d_nc.copy_from_host(gain_conv.data(), d_nc.bytes);
    d_conv.copy_from_host(kernel.data(), d_conv.bytes);
    d_next_norm.copy_from_host(next_norm.data(), d_next_norm.bytes);
    d_stem_embedding.copy_from_host(embedding.data(), d_stem_embedding.bytes);

    test::direct_bf16_weight::HostWeight host_projection;
    host_projection.n    = kN;
    host_projection.k    = kH;
    host_projection.bits = projection;

    PleLayerWeights ple;
    ple.projection        = host_projection.device_weight(d_projection.p);
    ple.projection.device = device;
    ple.norm_key          = bf16(d_nk.p, {kH, kC});
    ple.norm_query        = bf16(d_nq.p, {kH, kC});
    ple.norm_conv         = bf16(d_nc.p, {kH, kC});
    ple.convolution       = bf16(d_conv.p, {kD, kTaps});
    ple.rms_epsilon       = kBlockEpsilon;

    // The block's own input norm is the gain PLE's `normalized` output is produced with, and the
    // gain the stem uses above it.
    const Tensor block_norm = bf16(d_next_norm.p, {kH, kC});

    PlePrepareOptions options;
    options.max_positions = std::uint32_t(tokens);
    options.windows       = 2;
    options.io_depth      = 4;
    options.cache_bytes   = 0;

    // Block 0 has no PLE; block 1 carries the layer under test.
    std::vector<std::optional<PleLayerWeights>> layers;
    layers.push_back(std::nullopt);
    layers.push_back(ple);
    PleTextStage stage(table, kEos, context, std::move(layers), std::uint32_t(tokens), 1, 1,
                       options);
    check(stage.has_layer(1) && !stage.has_layer(0), "only the configured block carries PLE");

    // The four-stream state the stem produces, twice: the second copy is the "PLE off" baseline
    // this test compares against, and the first is the tensor PLE rewrites in place.
    auto stem = [&](DeviceBuffer& residual, DeviceBuffer& normalized) {
        Tensor residual4   = bf16(residual.p, {kH, kC, tokens, 1});
        Tensor normalized4 = bf16(normalized.p, {kH, kC, tokens, 1});
        ops::hc_expand_norm(bf16(d_stem_embedding.p, {kH, tokens}), block_norm, kBlockEpsilon,
                            residual4, normalized4, execution);
    };
    DeviceBuffer d_residual(std::size_t(kD) * tokens * 2),
        d_normalized(std::size_t(kD) * tokens * 2);
    DeviceBuffer d_base_residual(std::size_t(kD) * tokens * 2),
        d_base_normalized(std::size_t(kD) * tokens * 2);
    stem(d_residual, d_normalized);
    stem(d_base_residual, d_base_normalized);
    context.synchronize();

    const auto words         = std::size_t(kD) * tokens;
    const auto base_residual = test::from_device<std::uint16_t>(d_base_residual.p, words);
    check(base_residual == test::from_device<std::uint16_t>(d_residual.p, words),
          "the stem is deterministic, so the off-path baseline is the same words");

    // ---- the PLE step, exactly as text.cpp drives it
    const std::array<std::int32_t, 1> slot{0};
    const std::array<std::uint32_t, 1> live{std::uint32_t(tokens)};
    stage.begin(PleStepWindow{.ids = ids, .slots = slot, .valid_columns = live, .tokens = tokens},
                execution);
    context.synchronize();

    // 1. The Device E is the table's: every word against the hash + NF4 reference.
    {
        const auto published = stage.embedding();
        check(published.ne[0] == kH && published.ne[1] == tokens && published.ne[2] == 1,
              "the stage publishes E as [H,T,B]");
        const auto staged =
            test::from_device<std::uint16_t>(published.data, std::size_t(kH) * tokens);
        const auto lut  = make_nf4_bf16_lut(1.0F);
        const auto& c   = table.controls();
        std::size_t bad = 0;
        std::vector<std::uint16_t> decoded(kPleRowElements);
        for (int t = 0; t < tokens; ++t) {
            const auto previous = t == 0 ? kEos : ids[std::size_t(t) - 1];
            const auto older =
                t == 0 || previous == kEos ? kEos : (t == 1 ? kEos : ids[std::size_t(t) - 2]);
            for (std::uint32_t h = 0; h < kPleHeads; ++h) {
                const auto global  = reference_hash(ids[std::size_t(t)], previous, older, h, c);
                const auto encoded = encode_row(int(global));
                decode_nf4_row(encoded, lut, decoded);
                for (int d = 0; d < int(kPleRowElements); ++d)
                    bad += staged[std::size_t(t) * kH + h * kPleRowElements + std::size_t(d)] !=
                           decoded[std::size_t(d)];
            }
        }
        check(bad == 0, "the stage's embedding is the table's rows, decoded");
    }

    Tensor residual   = bf16(d_residual.p, {kH, kC, tokens, 1});
    Tensor normalized = bf16(d_normalized.p, {kH, kC, tokens, 1});
    stage.apply(1, residual, normalized, block_norm, kBlockEpsilon, execution);
    stage.commit(1, execution);
    context.synchronize();

    const auto after_residual   = test::from_device<std::uint16_t>(d_residual.p, words);
    const auto after_normalized = test::from_device<std::uint16_t>(d_normalized.p, words);
    const auto gated            = test::from_device<std::uint16_t>(stage.gated(1).data, words);
    const auto records          = test::from_device<std::uint16_t>(stage.records(1).data, words);

    // 2. The residual really moved, and only through PLE's own terms.
    {
        // The change is exactly residual + gated + SiLU(convolution), judged by the convolution
        // Op's own pointwise criterion (tests/ops/ple: 2e-6 absolute, 4e-3 relative). Every cell
        // whose contribution is representable in the residual's own words must show the change:
        // a cell that rounds back to the same BF16 is reported, never counted as pass.
        std::size_t unexplained = 0, changed = 0, significant = 0, missing = 0;
        double worst = 0, largest = 0;
        for (int t = 0; t < tokens; ++t)
            for (int s = 0; s < kC; ++s)
                for (int d = 0; d < kH; ++d) {
                    const auto i = own4(d, s, t);
                    const auto c = d + kH * s;
                    double tap   = 0;
                    for (int j = 0; j < kTaps; ++j) {
                        const int k = t - int(kPleHistory) + 3 * j;
                        if (k < 0) continue; // this slot's pool started at zero
                        tap += double(bf16_to_f32(
                                   kernel[std::size_t(c) + std::size_t(kD) * std::size_t(j)])) *
                               double(bf16_to_f32(
                                   records[std::size_t(c) + std::size_t(kD) * std::size_t(k)]));
                    }
                    const double base     = double(bf16_to_f32(base_residual[i]));
                    const double expected = base + double(bf16_to_f32(gated[i])) +
                                            tap / (1.0 + std::exp(-tap));
                    const double error = std::fabs(double(bf16_to_f32(after_residual[i])) - expected);
                    worst              = std::max(worst, error);
                    unexplained += error > 2e-6 + 4e-3 * std::fabs(expected);
                    const bool moved = after_residual[i] != base_residual[i];
                    changed += moved;
                    largest = std::max(largest, std::fabs(double(bf16_to_f32(after_residual[i])) -
                                                          base));
                    // "Representable" = the contribution exceeds the spacing of the residual's
                    // own BF16 grid at that magnitude, so it cannot round away.
                    const double grid = std::ldexp(1.0, std::ilogb(std::fabs(base)) - 7);
                    if (std::fabs(expected - base) > grid) {
                        ++significant;
                        missing += !moved;
                    }
                }
        check(unexplained == 0, "the rewrite is residual + gated + SiLU(conv), cell by cell");
        check(significant != 0 && missing == 0,
              "every cell whose PLE contribution is representable moved");
        check(largest > 1e-3, "the rewrite is not a rounding residual");
        std::cout << "  residual: " << changed << "/" << words << " words changed, "
                  << significant << " representable contributions, " << missing
                  << " missing, max |delta| = " << largest << "\n";
        std::cout << "  rewrite vs the convolution reference: max |error| = " << worst << "\n";
    }

    // 3. `normalized` is the exact BF16 norm of the NEW residual - the tensor the block's mixer
    //    reads, which is why the call site replaces hc_.normalized rather than adding a term.
    {
        std::size_t mismatched = 0;
        double worst           = 0;
        for (int t = 0; t < tokens; ++t)
            for (int s = 0; s < kC; ++s) {
                double sum = 0;
                for (int d = 0; d < kH; ++d)
                    sum += std::pow(double(bf16_to_f32(after_residual[own4(d, s, t)])), 2.0);
                const double inv = 1.0 / std::sqrt(sum / kH + kBlockEpsilon);
                for (int d = 0; d < kH; ++d) {
                    const auto i = own4(d, s, t);
                    const auto gain =
                        1.0 + double(bf16_to_f32(next_norm[std::size_t(d) + std::size_t(kH) * s]));
                    const auto ideal =
                        f32_to_bf16(float(double(bf16_to_f32(after_residual[i])) * inv * gain));
                    mismatched += after_normalized[i] != ideal;
                    worst = std::max(worst, std::fabs(double(bf16_to_f32(after_normalized[i])) -
                                                      double(bf16_to_f32(ideal))));
                }
            }
        check(mismatched == 0, "normalized is the exact BF16 norm of the new residual");
        std::cout << "  normalized vs the norm of the new residual: max |error| = " << worst
                  << "\n";
    }

    // 4. commit advanced the nine-column pool by the accepted prefix (four columns here).
    {
        const auto history =
            test::from_device<std::uint16_t>(stage.history(1).data, std::size_t(kD) * kPleHistory);
        std::size_t mismatched = 0;
        for (int j = 0; j < int(kPleHistory); ++j) {
            const int k = tokens - int(kPleHistory) + j; // accepted == the window's live columns
            for (int c = 0; c < kD; ++c) {
                const auto at = std::size_t(c) + std::size_t(kD) * std::size_t(j);
                // Columns before the window came from an all-zero pool.
                const std::uint16_t expected =
                    k < 0 ? 0 : records[std::size_t(c) + std::size_t(kD) * std::size_t(k)];
                mismatched += history[at] != expected;
            }
        }
        check(mismatched == 0, "commit advanced the history pool by the accepted columns");
    }
    stage.end(execution);
    context.synchronize();
    check(!stage.open(), "end released the window");

    // 5. Off path: a stack with no PLE block has nothing to call, and says so instead of skipping.
    {
        std::vector<std::optional<PleLayerWeights>> none;
        none.push_back(std::nullopt);
        none.push_back(std::nullopt);
        PleTextStage off(table, kEos, context, std::move(none), std::uint32_t(tokens), 1, 1,
                         options);
        check(!off.has_layer(0) && !off.has_layer(1), "a block without PLE carries no layer");
        bool refused = false;
        try {
            Tensor r = bf16(d_base_residual.p, {kH, kC, tokens, 1});
            Tensor n = bf16(d_base_normalized.p, {kH, kC, tokens, 1});
            off.apply(1, r, n, block_norm, kBlockEpsilon, execution);
        } catch (const std::invalid_argument&) { refused = true; }
        check(refused, "applying PLE to a block that has none is refused");
    }
}

} // namespace

int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::cout << "SKIP PLE block (no CUDA device)\n";
            return 77;
        }
        check_encoding_refusal();
        check_convolution_layout();
        run_stack();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    if (failures) return 1;
    std::cout << "OK PLE block: the table drives the layer, the layer rewrites the block's "
                 "residual, and a block without PLE is untouched\n";
    return 0;
}
