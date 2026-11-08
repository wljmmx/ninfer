#include "artifact/reader.h"
#include "core/device.h"
#include "core/nf4.h"
#include "models/qwen4_exp/ple/gpu.h"
#include "models/qwen4_exp/ple/hash.h"
#include "models/qwen4_exp/ple/prepare.h"

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

int failures = 0;

void check(bool condition, const std::string& label) {
    if (condition) return;
    std::cerr << label << ": failed\n";
    ++failures;
}

std::uint16_t rounded_bf16(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, 4);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return std::uint16_t((bits >> 16) | 0x0040u);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return std::uint16_t(bits >> 16);
}

// E4M3FN byte to exact value; bytes above 126 are excluded by the format.
double e4m3_value(unsigned byte) {
    const unsigned exponent = byte >> 3, mantissa = byte & 7;
    if (exponent == 0) return std::ldexp(double(mantissa) / 8.0, -6);
    return (1.0 + double(mantissa) / 8.0) * std::ldexp(1.0, int(exponent) - 7);
}

std::vector<std::uint16_t> reference_table() {
    std::vector<std::uint16_t> table(128 * 16, 0);
    for (unsigned scale = 0; scale < 127; ++scale)
        for (unsigned code = 0; code < 16; ++code) {
            const double value = double(nf4_code_value(code)) * e4m3_value(scale);
            table[scale * 16 + code] = rounded_bf16(static_cast<float>(value));
        }
    return table;
}

// The encoded row: 80 code bytes then 10 scale bytes.
std::vector<std::byte> encode_row(int row) {
    std::vector<std::byte> bytes(kPleRowBytes);
    for (int i = 0; i < 80; ++i)
        bytes[i] = std::byte(((row * 37 + i * 29) & 0x0f) | (((row * 11 + i * 7) & 0x0f) << 4));
    for (int i = 0; i < 10; ++i) bytes[80 + i] = std::byte(8 + (row * 5 + i * 3) % 100);
    return bytes;
}

struct ArtifactDir {
    std::filesystem::path directory;
    std::filesystem::path entry;

    explicit ArtifactDir() {
        const auto base = std::filesystem::temp_directory_path();
        for (int attempt = 0; attempt < 64; ++attempt) {
            const auto candidate =
                base / ("ninfer-ple-" + std::to_string(std::chrono::steady_clock::now()
                                                          .time_since_epoch()
                                                          .count()) +
                        "-" + std::to_string(attempt));
            std::error_code error;
            if (std::filesystem::create_directory(candidate, error)) {
                directory = candidate;
                break;
            }
        }
        if (directory.empty()) throw std::runtime_error("cannot create PLE fixture directory");
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

constexpr std::uint32_t kRows     = 3;
constexpr std::uint32_t kPageBytes = 4096;

void put_word(std::span<std::byte> bytes, std::size_t offset, std::uint64_t value, unsigned size) {
    for (unsigned i = 0; i < size; ++i) bytes[offset + i] = std::byte((value >> (8 * i)) & 255);
}

// One container file: 32-byte header, manifest, 4096-aligned payload.
ArtifactDir write_fixture(std::vector<std::byte>& payload) {
    payload.assign(kPageBytes + 4 + 280, std::byte{0});
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
        controls[3 + i]           = kRows;
        controls[3 + kPleHeads + i] = 0;
    }
    for (std::size_t i = 0; i < controls.size(); ++i)
        put_word(payload, kPageBytes + 4 + 8 * i, std::bit_cast<std::uint64_t>(controls[i]), 8);

    Json root = {
        {"components",
         {{"text", {{"config", Json::object()}, {"resources", Json::object()}}}}},
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
        {"metadata", {{"name", "ple-disk-fixture"}}}};

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

std::uint32_t reference_hash(std::int32_t current, std::int32_t previous, std::int32_t older,
                             std::uint32_t head, const PleHashControls& controls) {
    const auto pair = std::uint64_t(current) * std::uint64_t(controls.multipliers[0]) ^
                      std::uint64_t(previous) * std::uint64_t(controls.multipliers[1]);
    const auto word = head < 8 ? pair
                               : pair ^ std::uint64_t(older) * std::uint64_t(controls.multipliers[2]);
    auto index      = std::bit_cast<std::int64_t>(word) % controls.vocab_sizes[head];
    if (index < 0) index += controls.vocab_sizes[head];
    return static_cast<std::uint32_t>(index + controls.offsets[head]);
}

void check_nf4() {
    const auto table = reference_table();
    const auto lut   = make_nf4_bf16_lut(1.0f);
    int mismatch     = 0;
    for (std::size_t i = 0; i < table.size(); ++i) mismatch += table[i] != lut.words[i];
    check(mismatch == 0, "NF4 LUT matches the exact codebook * E4M3 reference");
    check(lut.words[127 * 16] == 0 && lut.words[127 * 16 + 15] == 0, "reserved scale row is zero");

    bool rejected = false;
    try {
        (void)make_nf4_bf16_lut(0.0f);
    } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "zero NF4 divisor rejected");

    std::vector<std::uint16_t> out(160);
    const auto encoded  = encode_row(2);
    decode_nf4_row(encoded, lut, out);
    int words  = 0;
    const auto* row = reinterpret_cast<const std::uint8_t*>(encoded.data());
    for (int group = 0; group < 10; ++group)
        for (int pair = 0; pair < 8; ++pair) {
            words += out[group * 16 + 2 * pair] == table[row[80 + group] * 16 + (row[group * 8 + pair] & 15)];
            words += out[group * 16 + 2 * pair + 1] ==
                     table[row[80 + group] * 16 + (row[group * 8 + pair] >> 4)];
        }
    check(words == 160, "NF4 row decode matches the reference table");

    auto bad_scale = encoded;
    bad_scale[80]  = std::byte{127};
    bool scale_rejected = false;
    try {
        decode_nf4_row(bad_scale, lut, out);
    } catch (const std::invalid_argument&) { scale_rejected = true; }
    check(scale_rejected, "reserved NF4 scale byte rejected");
}

void check_table_and_prepare() {    std::vector<std::byte> payload;
    const auto fixture = write_fixture(payload);
    const auto table_ref = reference_table();

    artifact::Reader reader(fixture.entry);
    const auto shard   = reader.find("ngram");
    const auto controls = reader.find("controls");
    PleTableSource table(reader, std::span(&shard, 1), controls);
    check(table.rows() == kRows, "table row count");
    check(table.file_count() == 1, "table file count");

    auto& c = table.controls();
    check(c.multipliers[0] == 3 && c.multipliers[1] == 5 && c.multipliers[2] == 7,
          "hash multipliers");
    check(c.vocab_sizes[0] == std::int64_t(kRows) && c.offsets[15] == 0, "hash head domain");

    for (std::uint32_t row = 0; row < kRows; ++row)
        check(table.row(row).offset == row * kPleRowBytes, "paged row offset");

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
    check(ticket.has_value(), "prepare admits the window");

    const std::array<std::uint32_t, 1> valid{3};
    service.seal(*ticket, valid);
    PleWindowSnapshot state{};
    for (int attempt = 0; attempt < 200; ++attempt) {
        state = service.poll(*ticket);
        if (state.status != PlePrepareStatus::Preparing) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    check(state.status == PlePrepareStatus::HostReady, "window reaches HostReady");
    check(state.drained, "window is drained");
    check(state.tokens == 4 && state.batch == 1, "window shape");
    const auto stats = service.stats();
    check(stats.rows == valid[0] * kPleHeads && stats.reads >= 1, "read stats");

    const auto embedding = service.embedding(*ticket);
    check(embedding.values.bytes == 4 * kPleEmbeddingBytes, "embedding extent");

    auto expected_row = [&](std::uint32_t t, std::uint32_t h) {
        const auto previous = t ? ids[t - 1] : identity.raw_tail[1];
        const auto older    = previous == 2
                                  ? 2
                                  : (t >= 2 ? ids[t - 2]
                                            : (t == 1 ? identity.raw_tail[1]
                                                      : identity.raw_tail[0]));
        return reference_hash(ids[t], previous, older, h, c);
    };
    const auto words = static_cast<const std::uint16_t*>(embedding.values.data);
    int bad = 0;
    for (std::uint32_t t = 0; t < valid[0]; ++t)
        for (std::uint32_t h = 0; h < kPleHeads; ++h) {
            const auto global  = expected_row(t, h);
            const auto encoded = encode_row(int(global));
            const auto* bytes  = reinterpret_cast<const std::uint8_t*>(encoded.data());
            for (std::uint32_t d = 0; d < kPleRowElements; ++d) {
                const auto group = d / 16, pair = (d % 16) / 2;
                const auto code = (d % 2) ? (bytes[group * 8 + pair] >> 4)
                                          : (bytes[group * 8 + pair] & 15);
                const auto reference = table_ref[bytes[80 + group] * 16 + code];
                bad += words[t * kPleEmbeddingElements + h * kPleRowElements + d] != reference;
            }
        }
    check(bad == 0, "prepared embedding words match hash + NF4 reference");

    check(std::all_of(first.begin() + valid[0] * kPleEmbeddingElements,
                      first.begin() + 4 * kPleEmbeddingElements,
                      [](auto word) { return word == 0; }),
          "columns past the seal are exact +0");

    const auto snapshot = service.poll(*ticket);
    check(snapshot.identities[0].sequence == 7, "identity is preserved");
    service.release(*ticket);
}

void check_gpu_staging() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP PLE GPU staging (no CUDA device)\n";
        return;
    }
    std::vector<std::byte> payload;
    const auto fixture = write_fixture(payload);
    artifact::Reader reader(fixture.entry);
    const auto shard    = reader.find("ngram");
    const auto controls = reader.find("controls");
    PleTableSource table(reader, std::span(&shard, 1), controls);

    DeviceContext device;
    PlePrepareOptions options;
    options.max_positions = 8;
    options.windows       = 1;
    options.io_depth      = 4;
    options.cache_bytes   = 0;
    PleGpuPreparation gpu(table, 2, device, options);

    const std::array<std::int32_t, 4> ids{5, 6, 7, 8};
    const PleSequenceIdentity identity{9, 2, 4, {1, 3}};
    std::array<PleInputRow, 1> rows{{{identity, ids}}};
    const auto ticket = gpu.host().prepare(4, rows);
    check(ticket.has_value(), "GPU prepare admits the window");
    const std::array<std::uint32_t, 1> valid{4};
    gpu.host().seal(*ticket, valid);
    for (int attempt = 0; attempt < 200; ++attempt) {
        if (gpu.host().poll(*ticket).status != PlePrepareStatus::Preparing) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    check(gpu.host().poll(*ticket).status == PlePrepareStatus::HostReady, "GPU window HostReady");

    const auto expected = gpu.host().embedding(*ticket);
    gpu.upload(*ticket, device.execution_view());
    bool uploaded = false;
    for (int attempt = 0; attempt < 200 && !uploaded; ++attempt) {
        uploaded = gpu.upload_complete(*ticket);
        if (!uploaded) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    check(uploaded, "GPU upload completes");
    const auto view = gpu.embedding(*ticket, device.execution_view());
    check(view.ne[0] == std::int32_t(kPleEmbeddingElements) && view.ne[1] == 4 && view.ne[2] == 1,
          "device slot shape");
    std::vector<std::uint16_t> staged(view.ne[0] * view.ne[1] * view.ne[2]);
    check(cudaMemcpy(staged.data(), view.data, staged.size() * 2, cudaMemcpyDeviceToHost) ==
              cudaSuccess,
          "device slot copy back");
    gpu.record_consumed(*ticket, device.execution_view());
    device.synchronize();
    check(staged == std::vector<std::uint16_t>(
                        static_cast<const std::uint16_t*>(expected.values.data),
                        static_cast<const std::uint16_t*>(expected.values.data) + staged.size()),
          "device slot matches the prepared Host words");
    bool released = false;
    for (int attempt = 0; attempt < 200 && !released; ++attempt) {
        released = gpu.try_release(*ticket);
        if (!released) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    check(released, "GPU slot releases after consumption");
    const auto memory = gpu.memory();
    check(memory.pinned_host_bytes >= 8 * kPleEmbeddingBytes &&
              memory.device_bytes == memory.pinned_host_bytes,
          "GPU staging memory accounting");
}

} // namespace

int main() {
    try {
        check_nf4();
        check_table_and_prepare();
        check_gpu_staging();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    if (failures) return 1;
    std::cout << "OK PLE NF4 LUT/row decode, paged table, hash, asynchronous preparation and GPU "
                 "staging\n";
    return 0;
}
