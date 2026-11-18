// V10-83: the artifact read leg's probe, pinned to its failure contract.
//
// The probe runs inside a model constructor, on the path that prices a bet. It is allowed to find
// nothing - a missing file, a truncated object, a source that cannot name its reader - and the one
// thing it must never do then is report a number: a zero rate is the bet's documented fallback to
// the priority rule, while a small number would be a price for traffic that was never measured.
// That is what this test asserts, on a source it owns, with no artifact and no Device.
//
// The positive case uses an in-memory source so the test does not depend on a model being present:
// it pins that a source that answers every read produces a rate, the payload shape that produced
// it, and the number of reads, i.e. that the probe reports its own shape and not just a speed.

#include "core/transfer_rate.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

constexpr std::uint64_t kMib = 1024ULL * 1024ULL;

// A source whose reads always fail: the "file opened, read failed" shape.
std::size_t read_nothing(void*, std::uint64_t, std::span<std::byte>) {
    return 0;
}

// A source that answers a short read: a truncated object, which must not be priced as if it moved
// the whole payload.
std::size_t read_short(void*, std::uint64_t, std::span<std::byte> destination) {
    return destination.size() / 2;
}

struct Memory {
    std::vector<std::byte> bytes;
};

// The positive control: every read is served from a resident buffer, at the offset asked for.
std::size_t read_memory(void* context, std::uint64_t offset, std::span<std::byte> destination) {
    const auto& source = *static_cast<const Memory*>(context);
    if (offset + destination.size() > source.bytes.size()) { return 0; }
    std::memcpy(destination.data(), source.bytes.data() + offset, destination.size());
    return destination.size();
}

void test_a_probe_that_cannot_run_measures_nothing() {
    std::uint32_t copies = 1;
    std::uint64_t span   = 1;
    // No read at all: the caller's reader is missing (a retained Host replica, a failed open).
    check(ninfer::core::measure_disk_read_rate({}, kMib, copies, span) == 0.0,
          "a source with no reader measures nothing");
    check(copies == 0 && span == 0, "and reports no shape");
    // A zero payload cannot be multiplied by a rate.
    const ninfer::core::DiskReadSource zero{
        .context = nullptr, .read = read_memory, .bytes = 8 * kMib, .mode = "buffered"};
    check(ninfer::core::measure_disk_read_rate(zero, 0, copies, span) == 0.0,
          "a zero payload measures nothing");
}

void test_a_failed_read_leaves_the_rate_at_zero() {
    std::uint32_t copies = 1;
    std::uint64_t span   = 1;
    const ninfer::core::DiskReadSource failing{
        .context = nullptr, .read = read_nothing, .bytes = 64 * kMib, .mode = "buffered"};
    check(ninfer::core::measure_disk_read_rate(failing, kMib, copies, span) == 0.0,
          "a read that returns nothing leaves the rate at zero");
    check(copies == 0 && span == 0, "and the probe claims no shape");

    const ninfer::core::DiskReadSource short_read{
        .context = nullptr, .read = read_short, .bytes = 64 * kMib, .mode = "buffered"};
    check(ninfer::core::measure_disk_read_rate(short_read, kMib, copies, span) == 0.0,
          "a short read leaves the rate at zero");

    // An object smaller than one payload cannot supply the shape being measured.
    const ninfer::core::DiskReadSource tiny{
        .context = nullptr, .read = read_memory, .bytes = kMib / 2, .mode = "buffered"};
    check(ninfer::core::measure_disk_read_rate(tiny, kMib, copies, span) == 0.0,
          "an object smaller than the payload measures nothing");
}

void test_a_working_source_reports_its_shape() {
    Memory memory;
    memory.bytes.assign(static_cast<std::size_t>(96 * kMib), std::byte{7});
    const ninfer::core::DiskReadSource source{
        .context = &memory, .read = read_memory, .bytes = 96 * kMib, .mode = "buffered"};
    std::uint32_t copies = 0;
    std::uint64_t span   = 0;
    const double rate =
        ninfer::core::measure_disk_read_rate(source, 2 * kMib, copies, span);
    check(rate > 0.0, "a source that answers every read produces a rate");
    check(copies >= 2, "the probe says how many reads formed the number");
    check(span == copies * 2 * kMib, "and how many bytes it moved");
    // The rate must be the one its own shape describes: bytes over the time it took.
    check(span > 0 && rate > 1e8, "the rate is bytes per second, not MiB per second");
}

} // namespace

int main() {
    test_a_probe_that_cannot_run_measures_nothing();
    test_a_failed_read_leaves_the_rate_at_zero();
    test_a_working_source_reports_its_shape();
    if (failures == 0) { std::printf("ok\n"); }
    return failures == 0 ? 0 : 1;
}
