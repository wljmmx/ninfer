// The q4_g64 PLE table: the row decoder the disk reader now serves, checked three ways.
//
//   1. Against the q4 kernel itself. A real `q4_g64_fp16` / `row_split_k128_v1` weight is built by
//      the quantized-weight fixture, its payload is uploaded, and `ops::linear` is run with a
//      one-hot activation: out[r, j] is then exactly the decoder's word for lane j of row r, as
//      produced by the shipped CUDA kernels (SIMT/MMA/GEMV, selected by T). The host decoder must
//      agree word for word - it is the same arithmetic, so any drift is a defect.
//   2. Against the artifact codecs' independent logical decoder (`logical_weight_fp64`), rounded to
//      the BF16 seam the Ops store. That path never shares a line of code with the row decoder.
//   3. End to end through the disk table: a synthetic artifact whose rows are q4_g64 rows in the
//      row-split planes is hashed and prepared by PlePrepareService, and every published word must
//      be the decode of the row the hash named - which is what proves the reader maps the code and
//      scale planes of a row at the right offsets.
//
// It also pins the load-time refusal: q4_g64 is now an admitted encoding, and an encoding the
// reader still cannot serve is refused with its name and row count in the message.

#include "artifact/reader.h"
#include "core/device.h"
#include "models/qwen4_exp/execution/ple_layer.h"
#include "models/qwen4_exp/ple/hash.h"
#include "models/qwen4_exp/ple/prepare.h"
#include "models/qwen4_exp/ple/table.h"
#include "ninfer/ops/linear.h"
#include "ops/direct_bf16_weight.h"
#include "ops/linear/linear_test_common.h"
#include "ops/linear/q4/q4_row_decode.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

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
#include <thread>
#include <vector>

namespace {

using Json = nlohmann::json;
using namespace ninfer;
using namespace ninfer::models::qwen4_exp;

using ninfer::test::bf16_to_f32;
using ninfer::test::f32_to_bf16;

int failures = 0;

void check(bool condition, const std::string& label) {
    if (condition) return;
    std::cerr << label << ": failed\n";
    ++failures;
}

void put_word(std::span<std::byte> bytes, std::size_t offset, std::uint64_t value, unsigned size) {
    for (unsigned i = 0; i < size; ++i) bytes[offset + i] = std::byte((value >> (8 * i)) & 255);
}

// ---------------------------------------------------------------- the q4 row's two planes

constexpr std::uint32_t kRows     = 3;
constexpr std::size_t kRowLanes   = kPleRowElements;               // 160
constexpr std::size_t kGroups     = 4;                             // 256 padded lanes / 64
constexpr std::size_t kCodeBytes  = kGroups * 32;                  // 128
constexpr std::size_t kScaleBytes = kGroups * 2;                   // 8
constexpr std::size_t kScaleOffset = 512;                          // align_up(3*128, 256)

// A row's codes and scales, deterministic and covering every code value.
std::vector<std::uint8_t> row_codes(int row) {
    std::vector<std::uint8_t> bytes(kCodeBytes);
    for (std::size_t i = 0; i < bytes.size(); ++i)
        bytes[i] = std::uint8_t(((row * 37 + int(i) * 29) & 0x0f) |
                                (((row * 11 + int(i) * 7) & 0x0f) << 4));
    return bytes;
}

std::vector<std::uint16_t> row_scales(int row) {
    // Exact binary16 words: 0.5, 0.25, 0.125, 0.0625 scaled per row so no value is trivial.
    static constexpr std::array<std::uint16_t, 4> kBase{0x3800, 0x3400, 0x3000, 0x2c00};
    std::vector<std::uint16_t> words(kGroups);
    for (std::size_t g = 0; g < words.size(); ++g) words[g] = kBase[(g + std::size_t(row)) % 4];
    return words;
}

std::vector<std::uint16_t> expected_row(int row) {
    std::vector<std::uint16_t> out(kRowLanes);
    const auto codes  = row_codes(row);
    const auto scales = row_scales(row);
    for (std::size_t lane = 0; lane < kRowLanes; ++lane) {
        const auto group  = lane / 64;
        const auto packed = codes[lane / 2]; // a byte holds lanes 2b and 2b+1 of its group
        const auto nibble = (lane & 1) ? (packed >> 4) : (packed & 0x0f);
        const auto code   = (nibble ^ 0x08) - 0x08;
        out[lane] = f32_to_bf16(float(code) * ninfer::ops::q4::fp16_to_f32(scales[group]));
    }
    return out;
}

// ---------------------------------------------------------------- the synthetic q4 artifact

struct ArtifactDir {
    std::filesystem::path directory;
    std::filesystem::path entry;

    ArtifactDir() {
        const auto base = std::filesystem::temp_directory_path();
        for (int attempt = 0; attempt < 64; ++attempt) {
            const auto candidate =
                base / ("ninfer-ple-q4-" +
                        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                        "-" + std::to_string(attempt));
            std::error_code error;
            if (std::filesystem::create_directory(candidate, error)) {
                directory = candidate;
                break;
            }
        }
        if (directory.empty()) throw std::runtime_error("cannot create the PLE q4 fixture");
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

ArtifactDir write_fixture() {
    const std::size_t payload_bytes = kScaleOffset + kRows * kScaleBytes;
    std::vector<std::byte> payload(payload_bytes, std::byte{0});
    for (std::uint32_t row = 0; row < kRows; ++row) {
        const auto codes = row_codes(int(row));
        for (std::size_t i = 0; i < kCodeBytes; ++i)
            payload[row * kCodeBytes + i] = std::byte(codes[i]);
        const auto scales = row_scales(int(row));
        for (std::size_t g = 0; g < kGroups; ++g)
            put_word(payload, kScaleOffset + row * kScaleBytes + g * 2, scales[g], 2);
    }
    // The 280-byte hash controls: one head per row, triple-hash multipliers.
    const std::size_t controls_offset = payload_bytes;
    payload.resize(payload_bytes + 280, std::byte{0});
    std::array<std::int64_t, 35> controls{};
    controls[0] = 3;
    controls[1] = 5;
    controls[2] = 7;
    for (std::size_t i = 0; i < kPleHeads; ++i) {
        controls[3 + i]             = kRows;
        controls[3 + kPleHeads + i] = 0;
    }
    for (std::size_t i = 0; i < controls.size(); ++i)
        put_word(payload, controls_offset + 8 * i, std::bit_cast<std::uint64_t>(controls[i]), 8);

    Json root = {
        {"components", {{"text", {{"config", Json::object()}, {"resources", Json::object()}}}}},
        {"objects",
         Json::array({{{"id", "ngram"},
                       {"kind", "tensor"},
                       {"shape", {kRows, kPleRowElements}},
                       {"format", "q4_g64_fp16"},
                       {"layout", "row_split_k128_v1"},
                       {"offset", 0},
                       {"bytes", kScaleOffset + kRows * kScaleBytes}},
                      {{"id", "controls"},
                       {"kind", "resource"},
                       {"encoding", "raw_bytes_v1"},
                       {"offset", controls_offset},
                       {"bytes", 280}}})},
        {"bindings", Json::object()},
        {"uses", Json::array()},
        {"files", Json::array({{{"path", nullptr}, {"payload_bytes", payload.size()}}})},
        {"metadata", {{"name", "ple-q4-fixture"}}}};

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

// ---------------------------------------------------------------- 1. the shipped q4 kernel

void check_against_the_kernel() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP q4 kernel cross-check (no CUDA device)\n";
        return;
    }
    // A shape the q4 dispatch registers, so the linear below runs the shipped kernels.
    constexpr std::int32_t kRowsOfWeight = 2048;
    constexpr std::int32_t kWidth        = 2560; // 40 whole 64-lane groups
    constexpr std::int32_t kWeightGroups = kWidth / 64;
    const auto packed = ninfer::test::linear::make_q4_g64_fp16_weight(kRowsOfWeight, kWidth, 941U);

    DeviceContext context;
    const auto execution = context.execution_view();
    DeviceBuffer d_weight(packed.payload.size());
    d_weight.copy_from_host(packed.payload.data(), d_weight.bytes);
    Weight weight    = packed.device_weight(d_weight.p);
    weight.device    = Device::cuda(context.device);

    // x is the identity over the whole width, so out[r, t] is the weight's own word at (r, t).
    constexpr std::int32_t kColumns = 2560;
    std::vector<std::uint16_t> one_hot(std::size_t(kWidth) * kColumns, 0);
    for (std::int32_t j = 0; j < kColumns; ++j)
        one_hot[std::size_t(j) + std::size_t(kWidth) * std::size_t(j)] = 0x3f80;
    DeviceBuffer d_x(one_hot.size() * 2), d_out(std::size_t(kRowsOfWeight) * kColumns * 2);
    d_x.copy_from_host(one_hot.data(), d_x.bytes);
    WorkspaceArena work(512ULL * 1024 * 1024);
    Tensor x(d_x.p, Device::cuda(context.device), DType::BF16, {kWidth, kColumns});
    Tensor out(d_out.p, Device::cuda(context.device), DType::BF16, {kRowsOfWeight, kColumns});
    ops::linear(x, weight, out, ops::LinearPolicy::A16Only, work, execution.stream);
    context.synchronize();

    const auto produced = ninfer::test::from_device<std::uint16_t>(d_out.p,
                                                                  std::size_t(kRowsOfWeight) * kColumns);
    const auto produced_unused = produced;
    (void)produced_unused;
    std::size_t lane_mismatch = 0, lanes = 0;
    std::size_t reference_mismatch = 0;
    for (std::int32_t row = 0; row < 8; ++row) {
        // A row of the parent is 40 groups wide; the decoder reads only the lanes it is asked for,
        // so the row's whole code and scale slices are the honest operand - and lanes 0..159 of it
        // are the first two and a half groups, which is a PLE row's span.
        const auto* codes  = packed.payload.data() + std::size_t(row) * kWeightGroups * 32;
        const auto* scales = packed.payload.data() + packed.scale_plane_offset +
                             std::size_t(row) * kWeightGroups * 2;
        std::vector<std::uint16_t> decoded(kWidth, 0xdead);
        ops::q4::decode_q4_g64_row(
            std::span<const std::byte>(reinterpret_cast<const std::byte*>(codes),
                                       std::size_t(kWeightGroups) * 32),
            std::span<const std::byte>(reinterpret_cast<const std::byte*>(scales),
                                       std::size_t(kWeightGroups) * 2),
            decoded);
        for (std::size_t lane = 0; lane < std::size_t(kColumns); ++lane) {
            // Dimension zero is fastest, so out[r, t] lives at r + N*t.
            const auto at = std::size_t(row) + std::size_t(kRowsOfWeight) * lane;
            if (produced[at] != decoded[lane] && lane_mismatch < 6)
                std::cerr << "    kernel row=" << row << " lane=" << lane << " decoded=" << std::hex
                          << decoded[lane] << " kernel=" << produced[at] << std::dec << '\n';
            lane_mismatch += produced[at] != decoded[lane];
            // The same word through the artifact codecs' independent logical decoder.
            const auto logical = ninfer::test::quantized_weight::logical_weight_fp64(
                packed, row, std::int32_t(lane));
            reference_mismatch += f32_to_bf16(static_cast<float>(logical)) != decoded[lane];
            ++lanes;
        }
    }
    check(lanes == 8 * std::size_t(kColumns), "the cross-check covered eight rows of the whole width");
    check(lane_mismatch == 0,
          "the host row decoder is word-identical to the shipped q4 kernel (one-hot linear)");
    check(reference_mismatch == 0,
          "the host row decoder is word-identical to the artifact codecs' logical decoder");
    std::cout << "  q4 kernel cross-check: " << lanes << " lanes, " << lane_mismatch
              << " kernel mismatches, " << reference_mismatch << " reference mismatches\n";
}

// ---------------------------------------------------------------- 3. the disk table

void check_disk_table() {
    const auto fixture  = write_fixture();
    artifact::Reader reader(fixture.entry);
    const auto shard    = reader.find("ngram");
    const auto controls = reader.find("controls");
    PleTableSource table(reader, std::span(&shard, 1), controls);

    check(table.encoding() == PleRowEncoding::Q4G64RowSplit, "the table declares the q4 encoding");
    check(table.rows() == kRows, "q4 table row count");
    check(table.row_parts() == 2 && table.row_encoded_bytes() == 136,
          "a q4 row is a 128-byte code plane and an 8-byte scale plane");
    for (std::uint32_t row = 0; row < kRows; ++row) {
        const auto source = table.row(row);
        check(source.parts == 2 && source.part_offsets[0] == row * kCodeBytes &&
                  source.part_bytes[0] == kCodeBytes &&
                  source.part_offsets[1] == kScaleOffset + row * kScaleBytes &&
                  source.part_bytes[1] == kScaleBytes,
              "the row's two spans are the code plane slice and the scale plane slice");
    }

    constexpr std::uint32_t max_positions = 8;
    std::vector<std::uint16_t> first(max_positions * kPleEmbeddingElements, 0x7fa1);
    std::vector<std::uint16_t> second(max_positions * kPleEmbeddingElements, 0x7fa1);
    std::vector<MemorySpan> outputs{
        {first.data(), first.size() * 2, Device::cpu(), false},
        {second.data(), second.size() * 2, Device::cpu(), false}};
    PlePrepareOptions options;
    options.max_positions = max_positions;
    options.windows       = 2;
    options.io_depth      = 4;
    options.cache_bytes   = 0;

    PlePrepareService service(table, 2, options, outputs);
    const std::array<std::int32_t, 4> ids{1, 2, 3, 4};
    const PleSequenceIdentity identity{7, 1, 4, {2, 0}};
    std::array<PleInputRow, 1> rows{{{identity, ids}}};
    const auto ticket = service.prepare(4, rows);
    check(ticket.has_value(), "the q4 table admits a window");
    const std::array<std::uint32_t, 1> valid{4};
    service.seal(*ticket, valid);
    PleWindowSnapshot state{};
    for (int attempt = 0; attempt < 500; ++attempt) {
        state = service.poll(*ticket);
        if (state.status != PlePrepareStatus::Preparing) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    check(state.status == PlePrepareStatus::HostReady, "the q4 window reaches the Host");
    check(state.drained, "the q4 window drained");

    const auto embedding = service.embedding(*ticket);
    const auto words     = static_cast<const std::uint16_t*>(embedding.values.data);
    auto hash_of = [&](std::uint32_t t, std::uint32_t head) {
        const auto previous = t ? ids[t - 1] : identity.raw_tail[1];
        const auto older    = previous == 2 ? 2
                                            : (t >= 2 ? ids[t - 2]
                                                      : (t == 1 ? identity.raw_tail[1]
                                                                : identity.raw_tail[0]));
        const auto pair = std::uint64_t(ids[t]) * std::uint64_t(table.controls().multipliers[0]) ^
                          std::uint64_t(previous) * std::uint64_t(table.controls().multipliers[1]);
        const auto word = head < 8
                              ? pair
                              : pair ^ std::uint64_t(older) * std::uint64_t(table.controls().multipliers[2]);
        auto index = std::bit_cast<std::int64_t>(word) % table.controls().vocab_sizes[head];
        if (index < 0) index += table.controls().vocab_sizes[head];
        return static_cast<std::uint32_t>(index + table.controls().offsets[head]);
    };
    std::size_t bad = 0;
    for (std::uint32_t t = 0; t < valid[0]; ++t) {
        for (std::uint32_t head = 0; head < kPleHeads; ++head) {
            const auto row      = hash_of(t, head);
            const auto expected = expected_row(int(row));
            for (std::size_t d = 0; d < kRowLanes; ++d) {
                const auto got = words[std::size_t(t) * kPleEmbeddingElements + head * kRowLanes + d];
                if (got != expected[d] && bad < 6)
                    std::cerr << "    mismatch t=" << t << " head=" << head << " row=" << row
                              << " lane=" << d << " expected=" << std::hex << expected[d]
                              << " got=" << got << std::dec << '\n';
                bad += got != expected[d];
            }
        }
    }
    check(bad == 0, "every prepared word is the decoded row the hash named");
    std::cout << "  q4 disk table: " << valid[0] * kPleHeads << " rows gathered, " << bad
              << " wrong words\n";
    service.release(*ticket);
}

// ---------------------------------------------------------------- 4. the refusal

void check_refusal() {
    bool admitted = false;
    try {
        require_supported_ple_table({.format    = QType::Q4_G64_FP16,
                                     .layout    = QuantLayout::RowSplit,
                                     .rows      = 320001536,
                                     .width     = kPleRowElements,
                                     .parameter = "text/layers/1/ple/table"});
        admitted = true;
    } catch (const std::exception&) {}
    check(admitted, "q4_g64/row_split_k128_v1 is an admitted table encoding");

    bool refused = false;
    std::string message;
    try {
        require_supported_ple_table({.format    = QType::Q5_G64_FP16,
                                     .layout    = QuantLayout::RowSplit,
                                     .rows      = 320001536,
                                     .width     = kPleRowElements,
                                     .parameter = "text/layers/1/ple/table"});
    } catch (const std::invalid_argument& error) {
        refused = true;
        message = error.what();
    }
    check(refused, "an encoding the reader cannot serve is refused");
    check(message.find("q5_g64_fp16") != std::string::npos &&
              message.find("320001536") != std::string::npos,
          "the refusal names the encoding and the row count");
    check(message.find("q4_g64_fp16") != std::string::npos,
          "the refusal states the encodings the reader does serve");
}

} // namespace

int main() {
    try {
        check_refusal();
        check_against_the_kernel();
        check_disk_table();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    if (failures) return 1;
    std::cout << "OK PLE q4_g64 rows: the host decoder is the kernel's arithmetic, and the disk "
                 "table serves the planes at the right offsets\n";
    return 0;
}
