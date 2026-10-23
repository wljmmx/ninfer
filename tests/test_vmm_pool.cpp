// Guards the runtime-elastic device pool: the address range is stable, whole segments can be handed
// back to the driver and re-acquired, and the pool refuses to touch unmapped memory.
#include "core/memory.h"
#include "core/vmm_pool.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <exception>
#include <stdexcept>
#include <vector>

int main() {
    using namespace ninfer;
    constexpr std::size_t kSegment  = 32u << 20;
    constexpr std::size_t kCapacity = 128u << 20;

    int failures = 0;
    const auto expect = [&](const char* what, bool ok) {
        std::printf("%s %s\n", ok ? "[ok]" : "[FAIL]", what);
        if (!ok) { ++failures; }
    };
    const auto free_bytes = [] {
        std::size_t free_now = 0, total = 0;
        cudaMemGetInfo(&free_now, &total);
        return free_now;
    };

    try {
        VmmPool pool(Device::cuda(0), kCapacity, kSegment);
        expect("capacity is fully mapped", pool.size() == kCapacity && pool.mapped() == kCapacity);

        std::vector<unsigned char> pattern(kSegment, 0x5a);
        std::vector<unsigned char> readback(kSegment, 0);
        pool.copy_from_host(pattern.data(), kSegment, 0);
        pool.copy_to_host(readback.data(), kSegment, 0);
        expect("host round trip", readback == pattern);

        // The property the expert cache relies on: what was released is really free again.
        const auto before = free_bytes();
        const auto released = pool.release(64u << 20, 32u << 20);
        const auto after = free_bytes();
        expect("release gives back whole segments",
               released == (64u << 20) && pool.mapped() == (64u << 20));
        expect("released bytes reach the driver", after >= before + (64u << 20) * 9 / 10);

        bool refused = false;
        try {
            pool.copy_from_host(pattern.data(), kSegment, 96u << 20);
        } catch (const std::out_of_range&) {
            refused = true;
        }
        expect("released tail is not addressable", refused);

        std::vector<unsigned char> scratch(kSegment, 0);
        pool.copy_to_host(scratch.data(), kSegment, 0);
        expect("data below the boundary survives the release", scratch == pattern);

        const auto added = pool.acquire(64u << 20);
        expect("acquire restores the range",
               added == (64u << 20) && pool.mapped() == kCapacity);

        pool.copy_to_host(readback.data(), kSegment, 0);
        expect("live data survives a release/acquire cycle", readback == pattern);

        expect("release keeps at least one segment",
               pool.release(kCapacity, 0) == kCapacity - kSegment && pool.mapped() == kSegment);

        // A pool that was shrunk and grown must still release everything it owns on destruction;
        // the run is repeated here so a leaked segment shows up as a lower free count.
        const auto at_exit = free_bytes();
        {
            VmmPool cycle(Device::cuda(0), 64u << 20, kSegment);
            cycle.release(32u << 20, 0);
            cycle.acquire(32u << 20);
            cycle.release(32u << 20, 0);
        }
        expect("a shrunk pool releases every segment on destruction", free_bytes() >= at_exit);
    } catch (const std::exception& error) {
        std::printf("SKIP: %s\n", error.what());
        return 77;
    }

    std::printf("%s vmm pool\n", failures == 0 ? "OK" : "FAIL");
    return failures == 0 ? 0 : 1;
}
