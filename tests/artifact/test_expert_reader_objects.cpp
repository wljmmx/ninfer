// port (sm_89): the expert reader resolves an object index through the artifact's file table.
//
// An `ExpertFileChunk` names an object and an offset inside it, the same convention
// host_expert_reader.h reads through model_.host_bytes(ObjectHandle). The reader used to hand that
// index to Reader::open_file(), which returns the ENTRY file whatever index it is given, so every
// chunk that lived in a continuation part was served with the entry's bytes at the same offset - a
// silent read of another object's payload. This fixture puts objects in the second and third parts
// and asserts:
//
//   1. the bytes read now equal the object's own bytes, and differ from what the entry file holds
//      at the same offset (the defect's bytes, reproduced here so the difference is on record);
//   2. a chunk whose object crosses a part boundary is read from both parts;
//   3. read_range() / object_base() answer for the same object the chunks name;
//   4. one entry may name different objects per chunk, each read from its own object.

#include "artifact/expert_reader.h"
#include "artifact/reader.h"
#include "core/expert_file.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::artifact;
using Json = nlohmann::json;

// Entry payload [0,3072) / part 1 [3072,11264) / part 2 [11264,15360).
constexpr std::size_t kPayloadBytes      = 15360;
constexpr std::uint64_t kFirstFileBytes  = 3072;
constexpr std::uint64_t kSecondFileBytes = 8192;

namespace Object {
// Ported positions: the entry's own object, one that crosses the entry/part-1 boundary, one that
// lies entirely in part 1, and one that lies entirely in part 2.
constexpr std::size_t head      = 0;
constexpr std::size_t spanning  = 1;
constexpr std::size_t in_first  = 2;
constexpr std::size_t in_second = 3;
constexpr std::array<std::uint64_t, 4> offsets{0, 2800, 3600, 11264};
constexpr std::array<std::uint64_t, 4> bytes{64, 800, 4096, 736};
constexpr std::array<const char*, 4> ids{"head", "spanning", "in_first_part", "in_second_part"};
} // namespace Object

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

// Every payload byte carries its logical position, so a read from the wrong file (or the wrong
// object) cannot compare equal by accident. The mix is not periodic in 256, because the file
// boundaries this fixture uses are whole multiples of 256.
std::vector<std::byte> payload_bytes() {
    std::vector<std::byte> bytes(kPayloadBytes);
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        std::uint32_t mixed = static_cast<std::uint32_t>(i) * 0x9E3779B1u;
        mixed ^= mixed >> 15;
        mixed *= 0x85EBCA77u;
        bytes[i] = std::byte(static_cast<unsigned char>(mixed & 0xff));
    }
    return bytes;
}

std::vector<std::byte> object_bytes(const std::vector<std::byte>& payload, std::size_t object) {
    return {payload.begin() + static_cast<std::ptrdiff_t>(Object::offsets[object]),
            payload.begin() +
                static_cast<std::ptrdiff_t>(Object::offsets[object] + Object::bytes[object])};
}

std::filesystem::path make_directory() {
    const auto base  = std::filesystem::temp_directory_path();
    const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
    for (int attempt = 0; attempt < 64; ++attempt) {
        const auto candidate = base / ("ninfer-expert-objects-" + std::to_string(ticks) + "-" +
                                       std::to_string(attempt));
        std::error_code error;
        if (std::filesystem::create_directory(candidate, error)) { return candidate; }
    }
    throw std::runtime_error("cannot create fixture directory");
}

void put_u64(std::span<std::byte> bytes, std::uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) { bytes[i] = std::byte((value >> (8 * i)) & 0xff); }
}

void write_file(const std::filesystem::path& path, bool entry, std::uint64_t part_index,
                std::span<const std::byte> json, std::span<const std::byte> payload) {
    std::array<std::byte, 32> header{};
    const std::array<unsigned char, 8> magic =
        entry ? std::array<unsigned char, 8>{'N', 'I', 'N', 'F', 'E', 'R', 0, 3}
              : std::array<unsigned char, 8>{'N', 'I', 'N', 'P', 'R', 'T', 0, 3};
    for (std::size_t i = 0; i < magic.size(); ++i) { header[i] = std::byte(magic[i]); }
    put_u64(std::span<std::byte>(header).subspan(8, 8), entry ? json.size() : part_index);
    std::fill(header.begin() + 16, header.end(), std::byte{0x5a});
    const std::uint64_t written = 32 + (entry ? json.size() : 0);
    const std::uint64_t aligned = (written + 4095) / 4096 * 4096;
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.exceptions(std::ios::badbit | std::ios::failbit);
    file.write(reinterpret_cast<const char*>(header.data()), header.size());
    if (entry) { file.write(reinterpret_cast<const char*>(json.data()), json.size()); }
    const std::string padding(static_cast<std::size_t>(aligned - written), '\0');
    file.write(padding.data(), static_cast<std::streamsize>(padding.size()));
    file.write(reinterpret_cast<const char*>(payload.data()),
               static_cast<std::streamsize>(payload.size()));
}

struct Fixture {
    std::filesystem::path directory;
    std::filesystem::path entry;
    std::filesystem::path second   = directory / "model.ninfer.second";
    std::filesystem::path third    = directory / "model.ninfer.third";
    std::vector<std::byte> payload = payload_bytes();

    Fixture() {
        directory    = make_directory();
        entry        = directory / "model.ninfer";
        second       = directory / "model.ninfer.second";
        third        = directory / "model.ninfer.third";
        Json objects = Json::array();
        for (std::size_t i = 0; i < Object::ids.size(); ++i) {
            objects.push_back({{"id", Object::ids[i]},
                               {"kind", "resource"},
                               {"encoding", "raw_bytes_v1"},
                               {"offset", Object::offsets[i]},
                               {"bytes", Object::bytes[i]}});
        }
        const Json root = {
            {"components", {{"text", {{"config", Json::object()}}}}},
            {"objects", objects},
            {"bindings", Json::object()},
            {"uses", Json::array()},
            {"files",
             Json::array(
                 {{{"path", nullptr}, {"payload_bytes", kFirstFileBytes}},
                  {{"path", "model.ninfer.second"}, {"payload_bytes", kSecondFileBytes}},
                  {{"path", "model.ninfer.third"},
                   {"payload_bytes", kPayloadBytes - kFirstFileBytes - kSecondFileBytes}}})}};
        const auto text = root.dump();
        write_file(entry, true, 0, std::as_bytes(std::span(text)),
                   std::span(payload).first(kFirstFileBytes));
        write_file(second, false, 1, {},
                   std::span(payload).subspan(kFirstFileBytes, kSecondFileBytes));
        write_file(third, false, 2, {},
                   std::span(payload).subspan(kFirstFileBytes + kSecondFileBytes));
    }

    ~Fixture() {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    }

    Fixture(const Fixture&)            = delete;
    Fixture& operator=(const Fixture&) = delete;
};

std::string hex8(std::span<const std::byte> bytes) {
    std::string out;
    for (std::size_t i = 0; i < bytes.size() && i < 8; ++i) {
        char text[3] = {};
        std::snprintf(text, sizeof(text), "%02x", std::to_integer<int>(bytes[i]));
        out += text;
    }
    return out;
}

void one_object_per_chunk() {
    Fixture fixture;
    ArtifactExpertReader reader(fixture.entry);
    const auto& directory = reader.reader().directory();

    // The chunk names the object that lives in the last part, the way describe_matrix_expert_files
    // and the host forms build one.
    ExpertFileEntry entry;
    entry.image_bytes = Object::bytes[Object::in_second];
    entry.chunks.push_back(
        {static_cast<std::uint32_t>(Object::in_second), 0, entry.image_bytes, 0});
    std::vector<std::byte> image(entry.image_bytes, std::byte{0xcc});
    reader.read_entry(entry, image);
    const auto expected = object_bytes(fixture.payload, Object::in_second);
    require(image == expected, "a chunk in the last part did not read its own object");

    // The defect's bytes: open_file(any index) answers the entry file, read at the chunk's
    // object-relative offset, which lands on the first object's payload.
    std::ifstream whole(fixture.entry, std::ios::binary);
    whole.exceptions(std::ios::badbit);
    whole.seekg(4096);
    std::vector<std::byte> wrong(image.size());
    whole.read(reinterpret_cast<char*>(wrong.data()), static_cast<std::streamsize>(wrong.size()));
    require(wrong != expected, "the fixture cannot tell the entry's bytes from the object's");
    require(std::equal(wrong.begin(), wrong.begin() + 64, fixture.payload.begin()),
            "the defect's bytes are not the first object's payload");
    std::fprintf(stderr,
                 "[expert-objects] object %zu (%s) in model.ninfer.third: read=%s entry-file=%s "
                 "(the entry file's bytes are object %s at that offset)\n",
                 Object::in_second, object_id(directory.objects[Object::in_second]).c_str(),
                 hex8(expected).c_str(), hex8(wrong).c_str(), Object::ids[Object::head]);

    // The cached path answers the same bytes.
    std::vector<std::byte> cached(entry.image_bytes, std::byte{0xcc});
    reader.read_entry_cached(entry, cached);
    require(cached == expected, "the cached path did not read its own object");

    // object_base names the object itself, not the file it happens to share with other objects.
    const auto* base = reader.object_base(static_cast<std::uint32_t>(Object::in_second));
    require(base != nullptr, "object_base lost an object that lives in one part");
    require(std::equal(expected.begin(), expected.end(), base),
            "object_base did not point at the object's first byte");

    // read_range answers the same object at the same offsets.
    std::vector<std::byte> range(entry.image_bytes, std::byte{0xcc});
    require(reader.read_range(static_cast<std::uint32_t>(Object::in_second), 0, range),
            "read_range refused a range inside the object");
    require(range == expected, "read_range read another object's bytes");
}

void chunk_across_a_part_boundary() {
    Fixture fixture;
    ArtifactExpertReader reader(fixture.entry);
    ExpertFileEntry entry;
    entry.image_bytes = Object::bytes[Object::spanning];
    entry.chunks.push_back({static_cast<std::uint32_t>(Object::spanning), 0, entry.image_bytes, 0});
    std::vector<std::byte> image(entry.image_bytes, std::byte{0xcc});
    reader.read_entry(entry, image);
    require(image == object_bytes(fixture.payload, Object::spanning),
            "an object that crosses a part boundary was not read from both parts");
    require(reader.object_base(static_cast<std::uint32_t>(Object::spanning)) == nullptr,
            "object_base named an object that has no single mapping");

    // The same range through read_range, split at the boundary by the reader.
    std::vector<std::byte> range(entry.image_bytes, std::byte{0xcc});
    require(reader.read_range(static_cast<std::uint32_t>(Object::spanning), 0, range),
            "read_range refused a range that merely crosses a file boundary");
    require(range == image, "read_range and read_entry disagree across a file boundary");
    // Out of the object, not merely out of the file: refused rather than clamped.
    std::vector<std::byte> beyond(64, std::byte{0xcc});
    require(!reader.read_range(static_cast<std::uint32_t>(Object::spanning),
                               Object::bytes[Object::spanning] - 8, beyond),
            "read_range accepted a range that runs past the object");
}

void chunks_name_their_own_objects() {
    Fixture fixture;
    ArtifactExpertReader reader(fixture.entry);
    // One entry image assembled from three different objects, the shape a bank has when its planes
    // are separate objects.
    ExpertFileEntry entry;
    entry.chunks.push_back(
        {static_cast<std::uint32_t>(Object::in_first), 0, Object::bytes[Object::in_first], 0});
    entry.chunks.push_back(
        {static_cast<std::uint32_t>(Object::in_second), 128, 256, Object::bytes[Object::in_first]});
    entry.chunks.push_back({static_cast<std::uint32_t>(Object::spanning), 776, 24,
                            Object::bytes[Object::in_first] + 256});
    entry.image_bytes = Object::bytes[Object::in_first] + 256 + 24;
    std::vector<std::byte> image(entry.image_bytes, std::byte{0xcc});
    reader.read_entry(entry, image);

    const auto first    = object_bytes(fixture.payload, Object::in_first);
    const auto second   = object_bytes(fixture.payload, Object::in_second);
    const auto spanning = object_bytes(fixture.payload, Object::spanning);
    require(std::equal(first.begin(), first.end(), image.begin()),
            "the first chunk did not read the object it names");
    require(
        std::equal(second.begin() + 128, second.begin() + 128 + 256,
                   image.begin() + static_cast<std::ptrdiff_t>(Object::bytes[Object::in_first])),
        "the second chunk did not read the object it names");
    require(std::equal(spanning.begin() + 776, spanning.begin() + 776 + 24,
                       image.begin() +
                           static_cast<std::ptrdiff_t>(Object::bytes[Object::in_first] + 256)),
            "the third chunk did not read the object it names");

    // A chunk that names a missing object is an error, not a fallback to whatever the index means
    // as a file number.
    ExpertFileEntry absent;
    absent.image_bytes = 16;
    absent.chunks.push_back({900, 0, 16, 0});
    std::vector<std::byte> sink(16);
    bool threw = false;
    try {
        reader.read_entry(absent, sink);
    } catch (const std::exception&) { threw = true; }
    require(threw, "a chunk naming an absent object was accepted");
}

} // namespace

int main() {
    try {
        one_object_per_chunk();
        chunk_across_a_part_boundary();
        chunks_name_their_own_objects();
        std::cout << "expert reader object resolution checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
