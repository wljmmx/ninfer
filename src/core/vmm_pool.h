#pragma once

#include "core/device.h"

#include <cstddef>
#include <vector>

namespace ninfer {

// port (sm_89): device memory whose physical backing can be handed back to the driver at runtime
// and re-acquired later. The virtual address range is reserved once and stays stable, so offsets
// already handed out by the expert cache remain valid across a release/acquire cycle; only the
// physical segments behind them come and go.
//
// Segments are fixed size and mapped independently, so `release` can only give back whole segments
// from the top of the range. Callers must keep their live data below `mapped()`.
class VmmPool {
public:
    VmmPool(Device device, std::size_t capacity_bytes, std::size_t segment_bytes);
    ~VmmPool();

    VmmPool(const VmmPool&)            = delete;
    VmmPool& operator=(const VmmPool&) = delete;
    VmmPool(VmmPool&& other) noexcept;
    VmmPool& operator=(VmmPool&& other) noexcept;

    [[nodiscard]] void* data() const noexcept { return base_; }
    // Address-space capacity: stable for the lifetime of the pool.
    [[nodiscard]] std::size_t size() const noexcept { return capacity_; }
    // Physically backed bytes right now (always a multiple of segment_bytes()).
    [[nodiscard]] std::size_t mapped() const noexcept { return mapped_; }
    [[nodiscard]] std::size_t segment_bytes() const noexcept { return segment_; }
    [[nodiscard]] MemorySpan span() const noexcept { return {data(), mapped_, device_, false}; }

    // Unmaps whole segments from the top so that up to `wanted_bytes` are returned, never dropping
    // below `keep_mapped_bytes` and never below one segment. Returns the bytes actually released.
    std::size_t release(std::size_t wanted_bytes, std::size_t keep_mapped_bytes);

    // Maps segments back, up to `wanted_bytes` or the reserved capacity. Returns the bytes added.
    std::size_t acquire(std::size_t wanted_bytes);

    // Both require the whole range to be mapped.
    void fill(int byte_value = 0);
    void copy_from_host(const void* source, std::size_t count, std::size_t offset = 0);
    void copy_to_host(void* destination, std::size_t count, std::size_t offset = 0) const;

private:
    void map_range(std::size_t offset, std::size_t bytes);
    void unmap_range(std::size_t offset, std::size_t bytes);
    void destroy() noexcept;

    Device device_{};
    void* base_               = nullptr;
    // One physical allocation per segment; nullptr means that segment is not mapped.
    std::vector<void*> handles_;
    std::size_t capacity_     = 0;
    std::size_t segment_      = 0;
    std::size_t mapped_       = 0;
    int device_ordinal_       = 0;
};

} // namespace ninfer
