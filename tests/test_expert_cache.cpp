#include "core/expert_cache.h"

#include <cstdio>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;

int fail(const char* message) {
    std::fprintf(stderr, "expert cache test: %s\n", message);
    return 1;
}

} // namespace

int main() {
    DeviceContext device(0);
    const std::uint64_t shape[] = {8, 8};
    const auto geometry = weight_geometry(QType::BF16, QuantLayout::Contiguous, shape);

    std::vector<std::vector<std::byte>> storage(2, std::vector<std::byte>(geometry.bytes));
    std::vector<WeightParent> parents(2);
    std::vector<const WeightParent*> pointers(2);
    for (std::size_t i = 0; i < parents.size(); ++i) {
        storage[i].assign(geometry.bytes, static_cast<std::byte>(i + 1));
        parents[i].geometry = geometry;
        parents[i].data     = storage[i].data();
        parents[i].device   = Device::cpu();
        pointers[i]         = &parents[i];
    }

    ExpertBankView view{pointers.data(), 2};
    ExpertBankPair pair{view, view};
    ExpertBankSpec spec{{geometry}, {geometry}, 2, false};

    ExpertCacheConfig config;
    config.capacity_bytes = 0;

    // Size the pool from the planned entry image: both experts fit, so nothing is evicted.
    config.capacity_bytes = 4096;
    const auto requirements = describe_expert_cache(std::span<const ExpertBankSpec>(&spec, 1), config);
    if (requirements.banks.size() != 1 || requirements.banks[0].count != 2) {
        return fail("bank plan is wrong");
    }
    if (requirements.banks[0].entry_bytes > config.capacity_bytes) {
        return fail("entry does not fit the pool");
    }

    ExpertCache cache(std::span<const ExpertBankSpec>(&spec, 1), requirements, device,
                      std::span<const ExpertBankPair>(&pair, 1));
    if (cache.capacity_bytes() != config.capacity_bytes) { return fail("capacity mismatch"); }

    const ExpertKey keys[] = {{0, 0}, {0, 1}};
    cache.prepare(keys);
    cache.wait_for_progress();

    if (cache.uploads() != 2) { return fail("expected two uploads"); }
    if (!cache.offset({0, 0}) || !cache.offset({0, 1})) {
        return fail("both experts should be resident");
    }
    if (cache.evictions() != 0) { return fail("one entry each fits without eviction"); }

    // port (sm_89): the file source must be usable without any Host replica. Passing null
    // parent data proves the artifact path is the one that runs.
    const auto image_bytes = requirements.banks[0].entry_bytes;
    std::vector<std::byte> image(static_cast<std::size_t>(image_bytes), std::byte{0x5a});
    struct MemoryReader final : ExpertBlobReader {
        const std::vector<std::byte>* data = nullptr;
        void read_entry(const ExpertFileEntry& entry,
                        std::span<std::byte> destination) const override {
            std::memcpy(destination.data(), data->data() + entry.chunks.front().file_offset,
                        static_cast<std::size_t>(entry.image_bytes));
        }
    } reader;
    reader.data = &image;

    ExpertFileBank file_bank;
    for (std::uint32_t e = 0; e < 2; ++e) {
        ExpertFileEntry entry;
        entry.image_bytes = image_bytes;
        entry.chunks.push_back({0, 0, image_bytes, 0});
        file_bank.experts.push_back(std::move(entry));
    }

    std::vector<WeightParent> detached(2);
    std::vector<const WeightParent*> detached_pointers(2);
    for (std::size_t i = 0; i < detached.size(); ++i) {
        detached[i].geometry = geometry;
        detached[i].data     = nullptr; // any Host read would fail
        detached[i].device   = Device::cpu();
        detached_pointers[i] = &detached[i];
    }
    ExpertBankView detached_view{detached_pointers.data(), 2};
    ExpertBankPair detached_pair{detached_view, detached_view};

    ExpertCache file_cache(std::span<const ExpertBankSpec>(&spec, 1), requirements, device,
                           std::span<const ExpertBankPair>(&detached_pair, 1),
                           std::span<const ExpertFileBank>(&file_bank, 1), &reader);
    file_cache.prepare(keys);
    file_cache.wait_for_progress();
    if (file_cache.uploads() != 2) { return fail("file tier should upload both experts"); }
    if (!file_cache.offset({0, 1})) { return fail("file tier entry should be resident"); }

    std::printf("expert cache test: uploads=%llu evictions=%llu used=%llu file_uploads=%llu\n",
                static_cast<unsigned long long>(cache.uploads()),
                static_cast<unsigned long long>(cache.evictions()),
                static_cast<unsigned long long>(cache.used_bytes()),
                static_cast<unsigned long long>(file_cache.uploads()));
    return 0;
}



