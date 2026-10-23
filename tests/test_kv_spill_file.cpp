#include "core/kv_spill_file.h"

#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

std::vector<std::byte> image(std::size_t bytes, std::uint8_t seed) {
    std::vector<std::byte> out(bytes);
    for (std::size_t i = 0; i < bytes; ++i) {
        out[i] = static_cast<std::byte>((seed + i) & 0xFFU);
    }
    return out;
}

std::filesystem::path scratch_path(const char* name) {
    auto path = std::filesystem::temp_directory_path() / (std::string("ninfer-spill-") + name);
    std::filesystem::remove(path);
    return path;
}

void test_round_trip_and_cleanup() {
    constexpr std::size_t kExtent = 4096;
    const auto path               = scratch_path("round-trip.bin");
    {
        ninfer::KVSpillFile spill(path, kExtent * 4, kExtent);
        expect(spill.capacity_extents() == 4, "capacity rounds to whole extents");
        expect(std::filesystem::exists(path), "the file is preallocated at construction");

        const auto first  = image(kExtent, 7);
        const auto second = image(kExtent / 2, 200);
        const auto slot_a = spill.write(first, 1);
        const auto slot_b = spill.write(second, 2);
        expect(slot_a != slot_b, "two images take two slots");
        expect(spill.resident_extents() == 2, "resident count follows the writes");

        std::vector<std::byte> back(kExtent);
        spill.read(slot_a, back);
        expect(std::memcmp(back.data(), first.data(), first.size()) == 0, "slot A round-trips");
        spill.read(slot_b, std::span<std::byte>(back.data(), second.size()));
        expect(std::memcmp(back.data(), second.data(), second.size()) == 0, "slot B round-trips");
        expect(spill.spilled_bytes() == first.size() + second.size(), "spill accounting");
        expect(spill.restored_bytes() == first.size() + second.size(), "restore accounting");
    }
    expect(!std::filesystem::exists(path), "destruction removes the scratch file");
}

void test_lru_eviction() {
    constexpr std::size_t kExtent = 1024;
    const auto path               = scratch_path("lru.bin");
    ninfer::KVSpillFile spill(path, kExtent * 2, kExtent);
    const auto a = image(kExtent, 1);
    const auto b = image(kExtent, 2);
    const auto c = image(kExtent, 3);

    const auto slot_a = spill.write(a, 10);
    const auto slot_b = spill.write(b, 20);
    expect(slot_a != slot_b && spill.resident_extents() == 2, "two images fill the file");

    const auto slot_c = spill.write(c, 30);
    expect(slot_c == slot_a, "the oldest epoch is the victim");
    expect(spill.resident_extents() == 2, "eviction keeps the file full, not overfull");

    std::vector<std::byte> back(kExtent);
    spill.read(slot_c, back);
    expect(std::memcmp(back.data(), c.data(), c.size()) == 0, "the new image is intact");
    spill.read(slot_b, back);
    expect(std::memcmp(back.data(), b.data(), b.size()) == 0, "the survivor is intact");
}

void test_release_reuse_and_bounds() {
    constexpr std::size_t kExtent = 512;
    const auto path               = scratch_path("reuse.bin");
    ninfer::KVSpillFile spill(path, kExtent * 2, kExtent);
    const auto image_a = image(kExtent, 5);
    const auto image_b = image(kExtent, 9);

    const auto slot_a = spill.write(image_a, 1);
    const auto slot_b = spill.write(image_b, 2);
    spill.release(slot_a);
    expect(spill.resident_extents() == 1, "release frees the slot");

    const auto slot_c = spill.write(image_a, 3);
    expect(slot_c == slot_a, "a released slot is reused before eviction");
    expect(spill.resident_extents() == 2, "reuse keeps the count honest");

    spill.release(slot_b);
    spill.release(slot_b);
    expect(spill.resident_extents() == 1, "releasing twice is a no-op");

    const auto too_big = image(kExtent * 2, 0);
    bool rejected      = false;
    try {
        spill.write(too_big, 4);
    } catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "an oversized image is refused");

    std::vector<std::byte> back(kExtent);
    bool empty_rejected = false;
    try {
        spill.read(1, back);
    } catch (const std::invalid_argument&) { empty_rejected = true; }
    expect(empty_rejected, "reading a released slot is refused");
}

} // namespace

int main() {
    try {
        test_round_trip_and_cleanup();
        test_lru_eviction();
        test_release_reuse_and_bounds();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return failures == 0 ? 0 : 1;
}
