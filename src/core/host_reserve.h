#pragma once

#include "core/memory.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer {

// port (sm_89): a fixed virtual range whose physical pages are committed on demand.
//
// The address never moves, so weight views that point into it stay valid while experts are
// released and reloaded. Released bytes are either returned to the OS (Windows MEM_DECOMMIT,
// POSIX MADV_DONTNEED) or simply unmapped; a later access to a released page is a fault, which
// is why callers must commit what they are about to touch.
class HostReserve {
public:
    HostReserve() noexcept = default;
    explicit HostReserve(std::size_t bytes);
    ~HostReserve();
    HostReserve(const HostReserve&)            = delete;
    HostReserve& operator=(const HostReserve&) = delete;
    HostReserve(HostReserve&& other) noexcept;
    HostReserve& operator=(HostReserve&& other) noexcept;

    [[nodiscard]] std::byte* data() noexcept { return static_cast<std::byte*>(base_); }

    [[nodiscard]] const std::byte* data() const noexcept {
        return static_cast<const std::byte*>(base_);
    }

    [[nodiscard]] std::size_t size() const noexcept { return bytes_; }

    // Allocation granularity of the platform; every committed range is rounded to it.
    [[nodiscard]] static std::size_t granularity() noexcept;

    void commit(std::uint64_t offset, std::uint64_t bytes);
    void decommit(std::uint64_t offset, std::uint64_t bytes);
    [[nodiscard]] bool resident(std::uint64_t offset, std::uint64_t bytes) const noexcept;

    [[nodiscard]] std::size_t committed_bytes() const noexcept { return committed_; }

    [[nodiscard]] std::size_t peak_committed_bytes() const noexcept { return peak_committed_; }

    // Gives back everything. The reservation itself stays until destruction.
    void release_all();

private:
    void* base_                 = nullptr;
    std::size_t bytes_          = 0;
    std::size_t committed_      = 0;
    std::size_t peak_committed_ = 0;
    std::vector<std::uint8_t> pages_; // One flag per granularity unit.
};

} // namespace ninfer
