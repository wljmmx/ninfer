// Guards the runtime-elastic expert pool: the reservation must be able to give device memory back
// and take it again while every published offset stays meaningful.
#include "core/expert_cache.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <span>
#include <vector>

namespace {

using namespace ninfer;

int failures = 0;

void expect(const char* what, bool ok) {
    std::printf("%s %s\n", ok ? "[ok]" : "[FAIL]", what);
    if (!ok) { ++failures; }
}

} // namespace

int main() {
    DeviceContext device(0);
    constexpr std::uint32_t kBankCount  = 4;
    constexpr std::uint32_t kExpertCount = 2;

    // 1024x1024 bf16 is 2 MiB per matrix, so one expert entry is 4 MiB and two slots of one bank
    // are exactly one 8 MiB VMM segment.
    const std::uint64_t shape[] = {1024, 1024};
    const auto geometry = weight_geometry(QType::BF16, QuantLayout::Contiguous, shape);

    std::vector<std::vector<std::byte>> storage(kBankCount * kExpertCount,
                                                std::vector<std::byte>(geometry.bytes));
    std::vector<WeightParent> parents(kBankCount * kExpertCount);
    std::vector<std::vector<const WeightParent*>> pointers(kBankCount,
                                                           std::vector<const WeightParent*>(2));
    std::vector<ExpertBankView> views(kBankCount);
    std::vector<ExpertBankPair> pairs(kBankCount);
    std::vector<ExpertBankSpec> specs;
    for (std::uint32_t b = 0; b < kBankCount; ++b) {
        for (std::uint32_t e = 0; e < kExpertCount; ++e) {
            const auto flat = b * kExpertCount + e;
            std::memset(storage[flat].data(), static_cast<int>(0x10 + flat),
                        static_cast<std::size_t>(geometry.bytes));
            parents[flat].geometry = geometry;
            parents[flat].data     = storage[flat].data();
            parents[flat].device   = Device::cpu();
            pointers[b][e]         = &parents[flat];
        }
        views[b] = ExpertBankView{pointers[b].data(), kExpertCount};
        pairs[b] = ExpertBankPair{views[b], views[b]};
        specs.push_back(ExpertBankSpec{{geometry}, {geometry}, kExpertCount, false});
    }

    constexpr std::uint64_t kCapacity = 64ull << 20;
    const auto requirements = describe_expert_cache(specs, {.capacity_bytes = kCapacity});
    expect("one entry fits the pool", requirements.max_entry_bytes <= kCapacity / 8);

    ExpertCache cache(specs, requirements, device, pairs);
    const auto full = cache.capacity_bytes();
    expect("the whole reservation is usable", full == kCapacity);
    expect("the reservation holds eight bank runs", cache.windows() == 8);

    const auto free_bytes = [] {
        std::size_t free_now = 0, total = 0;
        cudaMemGetInfo(&free_now, &total);
        return free_now;
    };

    std::vector<ExpertCacheKey> keys;
    for (std::uint32_t b = 0; b < kBankCount; ++b) { keys.push_back({b, 0}); }
    cache.prepare(keys);
    cache.wait_for_progress();
    for (std::uint32_t b = 0; b < kBankCount; ++b) {
        expect("expert is resident after upload", cache.offset({b, 0}).has_value());
    }

    // 13.5 (step 5): a resize moves the layout - experts above the new edge lose their slots -
    // and the stream republishes its offset tables only when the generations move. A shrink or
    // a grow that left them still would serve a table naming a released segment (the illegal
    // address the resize-under-load acceptance caught on the e2 artifact), so every generation a
    // publisher compares against must advance; a same-size call is the documented no-op.
    const auto layout_before = cache.layout_generation();
    const auto bank_before   = cache.bank_generation(0);
    const auto hot_before    = cache.hot_generation(0);
    const auto step_before   = cache.step_generation(0);
    const auto before = free_bytes();
    const auto shrunk_now = cache.resize(16ull << 20);
    const auto after = free_bytes();
    expect("a shrink advances the published-table generations",
           cache.layout_generation() > layout_before && cache.bank_generation(0) > bank_before &&
               cache.hot_generation(0) > hot_before && cache.step_generation(0) > step_before);
    expect("shrink keeps two bank runs", shrunk_now == (16ull << 20) && cache.windows() == 2);
    expect("shrunk bytes reach the driver", after >= before + (40ull << 20));
    expect("a dropped run is gone", !cache.window_resident(3) && !cache.offset({3, 0}).has_value());
    expect("a kept run still answers", cache.window_resident(1) && cache.offset({1, 0}).has_value());

    const auto layout_mid = cache.layout_generation();
    const auto bank_mid   = cache.bank_generation(0);
    expect("a same-size resize is the documented no-op",
           cache.resize(16ull << 20) == shrunk_now && cache.layout_generation() == layout_mid &&
               cache.bank_generation(0) == bank_mid);

    // Growing back must make the reservation usable again, and the data that was dropped has to
    // come back through the normal upload path with its original bytes. Slots are shared, so the
    // bank asks for one instead of assuming its old address is still there.
    const auto layout_grown = cache.layout_generation();
    expect("grow restores the reservation", cache.resize(kCapacity) == kCapacity);
    expect("a grow advances the published-table generations",
           cache.layout_generation() > layout_grown);
    const std::uint32_t expert_zero = 0;
    cache.set_resident_experts(3, std::span<const std::uint32_t>(&expert_zero, 1));
    cache.prepare(std::span<const ExpertCacheKey>(&keys[3], 1));
    cache.wait_for_progress();
    const auto restored = cache.offset({3, 0});
    expect("dropped expert streams back in", restored.has_value());
    if (restored) {
        std::vector<std::byte> readback(static_cast<std::size_t>(geometry.bytes));
        const auto* base = static_cast<const std::byte*>(cache.pool().data) + *restored;
        cudaMemcpy(readback.data(), base, readback.size(), cudaMemcpyDeviceToHost);
        expect("restored bytes match the source",
               std::memcmp(readback.data(), storage[3 * kExpertCount].data(), readback.size()) == 0);
    }

    std::printf("expert pool resize test: full=%llu shrunk=%llu windows=%llu uploads=%llu\n",
                static_cast<unsigned long long>(full),
                static_cast<unsigned long long>(shrunk_now),
                static_cast<unsigned long long>(cache.windows()),
                static_cast<unsigned long long>(cache.uploads()));
    std::printf("%s expert pool resize\n", failures == 0 ? "OK" : "FAIL");
    return failures == 0 ? 0 : 1;
}
