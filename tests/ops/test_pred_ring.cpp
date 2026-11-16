// S3/P1: the prediction ring's contract, against a host model of it.
//
// The ring exists to answer two questions a single buffer cannot: which entry belongs to which layer,
// and what was lost. Both are checkable without a GPU consumer, so this test drives the kernel exactly
// as the stream does - write an entry, bump the head - and then checks:
//
//   * the head is monotonic and one per write, so a consumer's "tail -> head" walk enumerates every
//     entry once, in order, with the sequence number the header carries (no loss, no duplicate);
//   * an entry wider than the ring's id capacity is refused as a whole (`truncated`), because half a
//     prediction placed as if it were whole is a wrong placement, not a missing one;
//   * a producer that outruns the consumer by more than `depth` leaves the newest `depth` entries
//     readable and the older slots overwritten - which is what makes "dropped" a count rather than a
//     guess: the consumer sees a sequence mismatch instead of stale bytes.
//
// Test-local numbers only: the ring's depth and id capacity are the model's in production (see
// ExpertStream::ensure_pred_ring), so nothing here may be read as an engine constant.
#include "ninfer/ops/moe.h"

#include "ops/op_tester.h"

#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::uint32_t kDepth    = 3;
constexpr std::uint32_t kCapacity = 8;

// Straight from the kernel's contract: slot = sequence % depth, and the header names the ticket the
// entry was written under.
struct Entry {
    ops::PredRingHeader header;
    std::vector<std::int32_t> ids;
};

std::vector<Entry> drain(const std::vector<ops::PredRingHeader>& headers,
                         const std::vector<std::int32_t>& ids, std::uint64_t head) {
    std::vector<Entry> entries;
    const auto readable = head < kDepth ? head : static_cast<std::uint64_t>(kDepth);
    for (auto sequence = head - readable; sequence < head; ++sequence) {
        const auto slot = static_cast<std::size_t>(sequence % kDepth);
        entries.push_back(Entry{headers[slot], {}});
        if (headers[slot].truncated != 0) { continue; }
        for (std::uint32_t i = 0; i < headers[slot].count; ++i) {
            entries.back().ids.push_back(ids[slot * kCapacity + i]);
        }
    }
    return entries;
}

int sequence_and_payload(cudaStream_t stream) {
    DeviceBuffer headers(kDepth * sizeof(ops::PredRingHeader));
    DeviceBuffer ids(kDepth * kCapacity * sizeof(std::int32_t));
    DeviceBuffer head(sizeof(unsigned long long));
    const unsigned long long zero = 0;
    head.copy_from_host(&zero, sizeof(zero));

    int failures = 0;
    // Six writes over a depth of three: the first three fill the ring, the last three overwrite it.
    for (std::uint32_t round = 0; round < 6; ++round) {
        std::vector<std::int32_t> host_ids(kCapacity);
        for (std::uint32_t i = 0; i < kCapacity; ++i) {
            host_ids[i] = static_cast<std::int32_t>(1000 * round + i);
        }
        DeviceBuffer device_ids = to_device(host_ids);
        ops::moe_pred_ring_write(static_cast<const std::int32_t*>(device_ids.p),
                                 static_cast<int>(kCapacity), /*source_bank=*/round,
                                 /*target_bank=*/round + 1, /*layer=*/round,
                                 static_cast<ops::PredRingHeader*>(headers.p),
                                 static_cast<std::int32_t*>(ids.p), kCapacity, kDepth,
                                 static_cast<unsigned long long*>(head.p), stream);
        cuda_check_last_launch("moe_pred_ring_write");
        cuda_synchronize(stream);
        const auto entries = drain(from_device<ops::PredRingHeader>(headers, kDepth),
                                   from_device<std::int32_t>(ids, kDepth * kCapacity), round + 1);
        // The newest entry is the one just written, with the round's own ids.
        const auto& newest = entries.back();
        failures += verify_exact("ring sequence",
                                 std::vector<std::uint64_t>{newest.header.sequence},
                                 std::vector<std::uint64_t>{static_cast<std::uint64_t>(round)});
        failures += verify_exact("ring ids in order", newest.ids, host_ids);
        failures += verify_exact("ring target bank",
                                 std::vector<std::uint64_t>{static_cast<std::uint64_t>(newest.header.target_bank)},
                                 std::vector<std::uint64_t>{static_cast<std::uint64_t>(round + 1)});
    }
    // After six writes the head must be six and only the newest `depth` entries readable, each with
    // the sequence its slot names - the older three left no sequence behind that a consumer could
    // mistake for its own.
    unsigned long long final_head = 0;
    head.copy_to_host(&final_head, sizeof(final_head));
    failures += verify_exact("ring head monotonic", std::vector<std::uint64_t>{final_head},
                             std::vector<std::uint64_t>{6});
    const auto entries = drain(from_device<ops::PredRingHeader>(headers, kDepth),
                               from_device<std::int32_t>(ids, kDepth * kCapacity), final_head);
    failures += verify_exact("ring readable entries",
                             std::vector<std::uint64_t>{static_cast<std::uint64_t>(entries.size())},
                             std::vector<std::uint64_t>{kDepth});
    const auto expected_sequences = std::vector<std::uint64_t>{3, 4, 5};
    failures += verify_exact("ring surviving sequences",
                             std::vector<std::uint64_t>{entries[0].header.sequence,
                                                        entries[1].header.sequence,
                                                        entries[2].header.sequence},
                             expected_sequences);
    return failures;
}

// A guess wider than an entry is refused as a whole: no ids are written, and the header says so. A
// consumer that read the count without the flag would place a half-prediction.
int truncation_is_refused(cudaStream_t stream) {
    DeviceBuffer headers(kDepth * sizeof(ops::PredRingHeader));
    DeviceBuffer ids(kDepth * kCapacity * sizeof(std::int32_t));
    DeviceBuffer head(sizeof(unsigned long long));
    const unsigned long long zero = 0;
    head.copy_from_host(&zero, sizeof(zero));

    std::vector<std::int32_t> host_ids(kCapacity + 4);
    for (std::size_t i = 0; i < host_ids.size(); ++i) { host_ids[i] = static_cast<std::int32_t>(i); }
    DeviceBuffer device_ids = to_device(host_ids);
    ops::moe_pred_ring_write(static_cast<const std::int32_t*>(device_ids.p),
                             static_cast<int>(host_ids.size()), 0, 1, 0,
                             static_cast<ops::PredRingHeader*>(headers.p),
                             static_cast<std::int32_t*>(ids.p), kCapacity, kDepth,
                             static_cast<unsigned long long*>(head.p), stream);
    cuda_check_last_launch("moe_pred_ring_write oversized");
    cuda_synchronize(stream);

    int failures = 0;
    const auto stored = from_device<ops::PredRingHeader>(headers, kDepth);
    failures += verify_exact("ring truncated flag",
                             std::vector<std::uint64_t>{static_cast<std::uint64_t>(stored[0].truncated)},
                             std::vector<std::uint64_t>{1});
    failures += verify_exact("ring truncated count",
                             std::vector<std::uint64_t>{stored[0].count},
                             std::vector<std::uint64_t>{static_cast<std::uint64_t>(host_ids.size())});
    // The ids are left untouched rather than partially written.
    const auto untouched = from_device<std::int32_t>(ids, kCapacity);
    failures += verify_exact("ring truncation writes no ids",
                             std::vector<std::uint64_t>{static_cast<std::uint64_t>(untouched[0])},
                             std::vector<std::uint64_t>{0});
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");

    int failures = 0;
    failures += sequence_and_payload(stream);
    failures += truncation_is_refused(stream);

    cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    std::cout << (failures == 0 ? "OK" : "FAIL") << " pred_ring contract\n";
    return failures == 0 ? 0 : 1;
}
