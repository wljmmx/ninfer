#include "artifact/binder.h"
#include "artifact/fixture.h"
#include "artifact/reader.h"
#include "artifact/views.h"
#include "core/device.h"
#include "core/weight_bank.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::artifact;
using namespace ninfer::test::artifact_fixture;

// port (sm_89) V10-13 acceptance shape: one 128x128 NVFP4 matrix per expert behind a leading axis,
// each with its own FP32 divisor. Every constant is derived by hand from the stored planes: codes
// are four-bit words in K order, the scale plane repeats once per expert matrix, and the payload
// ends with one divisor word per expert.
constexpr std::uint64_t kExperts        = 2;
constexpr std::uint64_t kRows           = 128;
constexpr std::uint64_t kColumns        = 128;
constexpr std::uint64_t kCodeBytes      = kExperts * kRows * kColumns / 2; // 16384
constexpr std::uint64_t kScaleOffset    = kCodeBytes;                      // already 256-aligned
constexpr std::uint64_t kScalePerMatrix = kRows * (kColumns / 16);         // 1024
constexpr std::uint64_t kScaleBytes     = kExperts * kScalePerMatrix;      // 2048
constexpr std::uint64_t kDivisorOffset  = kScaleOffset + kScaleBytes;      // 18432
constexpr std::uint64_t kPayloadBytes   = kDivisorOffset + 4 * kExperts;   // 18440
constexpr std::uint32_t kDivisorTwo     = 0x40000000U;                     // 2.0F
constexpr std::uint32_t kDivisorThree   = 0x40400000U;                     // 3.0F

template <class Error = ArtifactError, class Fn>
void rejects_with(Fn&& fn, std::string_view needle, const char* message) {
    try {
        fn();
    } catch (const Error& error) {
        if (std::string_view(error.what()).find(needle) != std::string_view::npos) { return; }
        throw std::runtime_error(std::string(message) + ": unexpected message " + error.what());
    }
    throw std::runtime_error(message);
}

// The two decoders below are the documented NVFP4 meaning of a stored byte pair, written out here
// so the test compares against a hand-computed value rather than against the loader's own numbers.
float e2m1_nibble(unsigned code) {
    constexpr float kMagnitudes[] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
    const auto magnitude          = kMagnitudes[code & 0x7U];
    return (code & 0x8U) ? -magnitude : magnitude;
}

float e4m3fn_word(unsigned word) {
    const float sign    = (word & 0x80U) ? -1.0F : 1.0F;
    const auto exponent = (word >> 3) & 0xFU;
    const auto fraction = word & 0x7U;
    if (exponent == 0) { return sign * static_cast<float>(fraction) * std::ldexp(1.0F, -9); }
    return sign * (1.0F + static_cast<float>(fraction) / 8.0F) *
           std::ldexp(1.0F, static_cast<int>(exponent) - 7);
}

// One expert matrix: 128 rows of 128 four-bit columns, expert 0 taking 2.0 and expert 1 taking 3.0.
struct BankFixture {
    Fixture fixture;

    BankFixture() {
        fixture.payload.assign(kPayloadBytes, std::byte{0});
        fixture.root = {
            {"components", {{"text", {{"config", Json::object()}}}}},
            {"objects", Json::array({{{"id", "bank"},
                                      {"kind", "tensor"},
                                      {"shape", {kExperts, kRows, kColumns}},
                                      {"format", "nvfp4"},
                                      {"layout", "batched_block_scale_k16_m128x4_v1"},
                                      {"offset", 0},
                                      {"bytes", kPayloadBytes}}})},
            {"bindings", {{"bank", {{"object", "bank"}}}}},
            {"uses", Json::array()},
            {"files", Json::array({{{"path", nullptr}, {"payload_bytes", kPayloadBytes}}})}};
        // Codes: the first two columns of every expert's first row are the E2M1 word 2 (1.0).
        for (std::uint64_t expert = 0; expert < kExperts; ++expert) {
            fixture.payload[expert * kRows * kColumns / 2] = std::byte{0x22};
        }
        // Scales: the first K group of every expert's first row stores E4M3FN 0x40 (2.0).
        for (std::uint64_t expert = 0; expert < kExperts; ++expert) {
            fixture.payload[kScaleOffset + expert * kScalePerMatrix] = std::byte{0x40};
        }
        put_word(fixture.payload, kDivisorOffset, kDivisorTwo, 4);
        put_word(fixture.payload, kDivisorOffset + 4, kDivisorThree, 4);
        fixture.write();
    }

    void rewrite() const { fixture.write(); }
};

std::byte read_device_byte(const void* address) {
    std::byte value{};
    CUDA_CHECK(cudaMemcpy(&value, address, 1, cudaMemcpyDeviceToHost));
    return value;
}

// The artifact must describe the bank, refuse a logical reshape of it, and hand both planes and
// both divisors to the view that addresses one expert.
void describe_bind_and_decode(DeviceContext& device) {
    BankFixture fixture;
    ObjectHandle bank;
    ParameterReference reference;
    std::optional<MaterializedArtifact> backing;
    {
        Reader reader(fixture.fixture.entry);
        bank                 = reader.find("bank");
        const auto& geometry = reader.geometry(bank);
        require(geometry.layout == QuantLayout::BatchedBlockScaleK16M128x4,
                "the batched layout name did not resolve");
        require(geometry.shape == Shape{kExperts, kRows, kColumns},
                "the described shape was reshaped");
        require(geometry.matrix_rows == kRows && geometry.scale_bytes_per_matrix == kScalePerMatrix,
                "one expert matrix did not own one scale block");
        require(geometry.scale_offset == kScaleOffset && geometry.scale_bytes == kScaleBytes &&
                    geometry.divisor_offset == kDivisorOffset && geometry.bytes == kPayloadBytes,
                "the described planes differ from the hand-computed payload");
        require(geometry.divisor_count == kExperts && geometry.divisor_rows == kRows,
                "a batched bank declares one divisor per expert matrix");
        Binder binder(reader);
        reference = binder.parameter("bank", {kExperts, kRows, kColumns});
        require(reference.binding.whole_object &&
                    reference.binding.elements == kExperts * kRows * kColumns,
                "the whole bank binding lost its element coverage");
        rejects([&] { (void)binder.parameter("bank", {kRows, kColumns}); },
                "the whole bank was accepted as one matrix");
        backing.emplace(materialize(reader, std::move(binder).finish(), device));
    }
    const auto& parent = backing->device_parent(bank);
    require(parent.weight_divisors.size() == kExperts,
            "the materialized bank lost its per-expert divisors");
    require(parent.weight_divisors[0] == 2.0F && parent.weight_divisors[1] == 3.0F,
            "the materialized divisor words were reordered or changed");
    require(parent.weight_scale_divisor == 0.0F,
            "a mixture of divisor domains was reported as one scalar");

    const auto view = bind_view(reference, *backing);
    require(view.shape.size() == 3 && view.parts.size() == 1 &&
                view.parts[0].end - view.parts[0].begin == kExperts * kRows * kColumns,
            "the bound bank view changed shape or coverage");
    for (std::uint64_t expert = 0; expert < kExperts; ++expert) {
        WeightView matrix;
        matrix.shape = {kRows, kColumns};
        matrix.parts.push_back(
            {&parent, expert * kRows * kColumns, (expert + 1) * kRows * kColumns});
        const auto weight = native_weight(matrix);
        require(weight.layout == QuantLayout::BlockScaleK16M128x4,
                "an expert matrix view did not report its rank-two layout");
        require(weight.qdata == parent.data + expert * kRows * kColumns / 2,
                "the code plane did not follow the expert axis");
        require(weight.scales == parent.data + kScaleOffset + expert * kScalePerMatrix,
                "the scale plane did not follow the expert axis");
        const auto divisor = expert ? 3.0F : 2.0F;
        require(weight.weight_scale_divisor == divisor,
                "the view reported another expert's divisor");
        const auto code  = read_device_byte(weight.qdata);
        const auto scale = read_device_byte(weight.scales);
        // 1.0 (E2M1 word 2) * 2.0 (E4M3FN 0x40) * the expert's own divisor: 4.0 and 6.0.
        const auto value = e2m1_nibble(std::to_integer<unsigned>(code) & 0xFU) *
                           e4m3fn_word(std::to_integer<unsigned>(scale)) * divisor;
        require(value == (expert ? 6.0F : 4.0F),
                "the decoded expert value differs from the hand-computed one");
    }

    // The bank ABI reads the same array: one matrix region that covers expert 1 only.
    const auto expert_one =
        bind_weight_bank(parent, describe_weight_bank(parent.geometry, {kRows, 1, kRows, kRows}));
    require(expert_one.divisor(0) == 3.0F, "the bank view reported another expert's divisor");
    const auto matrix = matrix_weight(expert_one, 0);
    require(matrix.scales == parent.data + kScaleOffset + kScalePerMatrix &&
                matrix.weight_scale_divisor == 3.0F,
            "the bank matrix lost its scale block or divisor");
}

// A rank-two stack of the same experts has no leading axis to count, so only a stated count can
// name its divisor array; describing it must produce the same bytes as the batched form.
void stacked_declared_divisors() {
    BankFixture fixture;
    Json root                      = fixture.fixture.root;
    root["objects"][0]["shape"]    = {kExperts * kRows, kColumns};
    root["objects"][0]["layout"]   = "block_scale_k16_m128x4_v1";
    root["objects"][0]["divisors"] = kExperts;
    fixture.fixture.write(false, root.dump());
    {
        Reader reader(fixture.fixture.entry);
        const auto& geometry = reader.geometry(reader.find("bank"));
        require(geometry.divisor_count == kExperts && geometry.divisor_rows == kRows,
                "a stated count did not name one divisor per expert");
        require(geometry.divisor_offset == kDivisorOffset && geometry.bytes == kPayloadBytes,
                "the stacked form described different bytes than the batched form");
    }
    // Without the count the same payload is one matrix with one divisor, which these bytes cannot
    // hold: the descriptor is refused rather than read as if it did.
    root["objects"][0].erase("divisors");
    fixture.fixture.write(false, root.dump());
    Reader plain(fixture.fixture.entry);
    rejects_with<ArtifactError>([&] { (void)plain.geometry(plain.find("bank")); }, "encoded size",
                                "a bank payload was read as one matrix divisor");
}

void describe_refusals() {
    BankFixture fixture;
    const auto refuse = [&](const std::function<void(Json&)>& mutate, std::string_view needle,
                            const char* message) {
        Json root = fixture.fixture.root;
        mutate(root);
        fixture.fixture.write(false, root.dump());
        // A stated count is a descriptor member, so a bad one is refused while the directory is
        // parsed; a count that disagrees with the payload is refused when the tensor is described.
        rejects_with<ArtifactError>(
            [&] {
                Reader reader(fixture.fixture.entry);
                (void)reader.geometry(reader.find("bank"));
            },
            needle, message);
    };
    refuse([](Json& root) { root["objects"][0]["shape"] = {kRows, kColumns}; },
           "quantized weight rank differs from its layout", "a rank-two shape was accepted");
    refuse([](Json& root) { root["objects"][0]["shape"] = {1, kExperts, kRows, kColumns}; },
           "quantized weight rank differs from its layout", "a rank-four shape was accepted");
    refuse([](Json& root) { root["objects"][0]["divisors"] = 3; },
           "per-expert divisor count differs from the expert count",
           "a divisor count that differs from the expert axis was accepted");
    refuse([](Json& root) { root["objects"][0]["divisors"] = 1; }, "encoded size",
           "a single divisor word was accepted for a bank payload");
    refuse([](Json& root) { root["objects"][0]["divisors"] = 0; }, "divisors must be positive",
           "an empty divisor array was accepted");
    refuse(
        [](Json& root) {
            root["objects"][0]["shape"]    = {kRows, kColumns};
            root["objects"][0]["format"]   = "q4_g64_fp16";
            root["objects"][0]["layout"]   = "row_split_k128_v1";
            root["objects"][0]["bytes"]    = 64;
            root["objects"][0]["divisors"] = kExperts;
        },
        "per-expert divisors require a block-scaled NVFP4 layout",
        "a divisor array was accepted on a layout without a divisor plane");
    refuse(
        [](Json& root) {
            root["objects"][0]["shape"] = {kExperts, 96, kColumns};
            root["objects"][0]["bytes"] = kExperts * 96 * kColumns / 2;
        },
        "NVFP4 BlockScale requires N%128=0 and K%64=0",
        "a bank whose N is not a whole 128-row tile was accepted");
}

// A mixture is only usable if every word is a real multiplier: one bad expert must not load with a
// silent 1.0.
void divisor_refusals(DeviceContext& device) {
    for (const auto word : {0x7FC00000U /* NaN */, 0x7F800000U /* +inf */, 0x00000000U /* 0 */,
                            0xBF800000U /* -1 */}) {
        BankFixture fixture;
        put_word(fixture.fixture.payload, kDivisorOffset + 4, word, 4);
        fixture.rewrite();
        Reader reader(fixture.fixture.entry);
        Binder binder(reader);
        (void)binder.parameter("bank", {kExperts, kRows, kColumns});
        rejects_with<ArtifactError>(
            [&] { (void)materialize(reader, std::move(binder).finish(), device); },
            "invalid NVFP4 weight divisor 1 of 2",
            "a bank with a non-finite or non-positive divisor was materialized");
    }
}

// Optional real artifact: describe every object and report whether any batched bank is inside.
void inspect(const std::filesystem::path& path) {
    Reader reader(path);
    std::uint64_t objects = 0;
    std::uint64_t banks   = 0;
    for (std::size_t i = 0; i < reader.directory().objects.size(); ++i) {
        reader.validate_object({i});
        ++objects;
        const auto& object = reader.directory().objects[i];
        if (!std::holds_alternative<TensorObject>(object)) { continue; }
        const auto& geometry = reader.geometry({i});
        if (geometry.layout != QuantLayout::BatchedBlockScaleK16M128x4) { continue; }
        ++banks;
        std::cout << "bank " << object_id(object) << " experts=" << geometry.shape.front()
                  << " divisors=" << geometry.divisor_count << " bytes=" << geometry.bytes << '\n';
    }
    std::cout << path.filename().string() << ": objects=" << objects
              << " batched_block_scale_objects=" << banks << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2) {
            inspect(argv[1]);
            return 0;
        }
        if (argc != 1) { throw std::invalid_argument("expected an optional artifact path"); }
        int count         = 0;
        const auto result = cudaGetDeviceCount(&count);
        if (result == cudaErrorNoDevice || result == cudaErrorInsufficientDriver ||
            (result == cudaSuccess && count == 0)) {
            return 77;
        }
        CUDA_CHECK(result);
        DeviceContext device;
        describe_refusals();
        stacked_declared_divisors();
        describe_bind_and_decode(device);
        divisor_refusals(device);
        std::cout << "artifact batched bank checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
