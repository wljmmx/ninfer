#pragma once

#include "core/device.h"

namespace ninfer::runtime {

// Startup has one Device owner. Snapshots are taken with its transfers and work settled.
// Re-read after destroying a candidate: retained driver storage is already charged.
class StartupMemoryBudget {
public:
    StartupMemoryBudget(DeviceMemorySnapshot initial, std::size_t limit, std::size_t headroom,
                        std::size_t reserve = 0);
    [[nodiscard]] std::size_t available(DeviceMemorySnapshot current) const;
    [[nodiscard]] bool fits(DeviceMemorySnapshot current) const noexcept;
    [[nodiscard]] std::int64_t retained(DeviceMemorySnapshot current,
                                        std::size_t owned_bytes) const;

    [[nodiscard]] std::size_t limit() const noexcept { return limit_; }

    [[nodiscard]] std::size_t headroom() const noexcept { return headroom_; }

private:
    DeviceMemorySnapshot initial_;
    std::size_t limit_, headroom_;
    // port (sm_89): excluded from available() only; limit_ still measures real consumption.
    std::size_t reserve_ = 0;
};

} // namespace ninfer::runtime
