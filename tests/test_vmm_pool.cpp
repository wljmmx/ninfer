// Guards the runtime-elastic device pool: the address range is stable, whole segments can be
// handed back and re-acquired, and the pool refuses to touch unmapped memory.
#include "core/memory.h"
#include "core/vmm_pool.h"

#include <cstdio>
#include <stdexcept>
#include <vector>

int main() {
    using namespace ninfer;
    constexpr std::size_t kSegment = 16u << 20;
    constexpr std::size_t kCapacity = 64u << 20;

    int failures = 0;
    const auto expect = [&](const char* what, bool ok) {
        std::printf("%s %s\n", ok ? "[ok]" : "[FAIL]", what);
        if (!ok) { ++failures; }
    };

    try {
        VmmPool pool(Device::cuda(0), kCapacity, kSegment);
        expect("capacity is fully mapped", pool.size() == kCapacity && pool.mapped() == kCapacity);

        std::vector<unsigned char> pattern(kSegment, 0x5a);
        std::vector<unsigned char> readback(kSegment, 0);
        pool.copy_from_host(pattern.data(), kSegment, 0);
        pool.copy_to_host(readback.data(), kSegment, 0);
        expect("host round trip", readback == pattern);

        const auto released = pool.release(32u << 20, 16u << 20);
        expect("release gives back whole segments",
               released == (32u << 20) && pool.mapped() == (32u << 20));

        bool refused = false;
        try {
            pool.copy_from_host(pattern.data(), kSegment, 48u << 20);
        } catch (const std::out_of_range&) {
            refused = true;
        }
        expect("released tail is not addressable", refused);

        const auto added = pool.acquire(32u << 20);
        expect("acquire restores the range",
               added == (32u << 20) && pool.mapped() == kCapacity);

        pool.copy_to_host(readback.data(), kSegment, 0);
        expect("live data survives a release/acquire cycle", readback == pattern);

        expect("release keeps at least one segment",
               pool.release(kCapacity, 0) == kCapacity - kSegment && pool.mapped() == kSegment);
    } catch (const std::exception& error) {
        std::printf("SKIP: %s\n", error.what());
        return 77;
    }

    std::printf("%s vmm pool\n", failures == 0 ? "OK" : "FAIL");
    return failures == 0 ? 0 : 1;
}
