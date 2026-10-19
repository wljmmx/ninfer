#include "core/expert_cache.h"

#include <cstdio>
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
    const auto probe      = [&] {
        auto sizing         = config;
        sizing.capacity_bytes = 1;
        return sizing;
    };
    (void)probe;

    // Size the pool from the planned entry image: one expert fits, so the second upload
    // must evict the first.
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

    std::printf("expert cache test: uploads=%llu evictions=%llu used=%llu\n",
                static_cast<unsigned long long>(cache.uploads()),
                static_cast<unsigned long long>(cache.evictions()),
                static_cast<unsigned long long>(cache.used_bytes()));
    return 0;
}


