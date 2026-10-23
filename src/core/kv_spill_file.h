#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <vector>

namespace ninfer {

// The third KV tier: a fixed-size file holding whole page-group images.
//
// The Host arena is pinned RAM, so it cannot grow past the machine. When a live sequence needs
// more room than the arena has, the coldest page group is written here and its arena extent is
// released; reading it back is an ordinary buffered file read. The file is preallocated so a
// write never fails halfway, and slots are addressed by index so nothing depends on the OS
// allocation order.
//
// This type owns storage only. Which page group may leave the arena is a policy decision that
// belongs to the caller (see the live/checkpoint split in HostContextArena).
class KVSpillFile {
public:
    // `capacity_bytes` is rounded down to whole extents. The file is created (or truncated) at
    // construction and its directory is created when missing.
    KVSpillFile(std::filesystem::path path, std::size_t capacity_bytes, std::size_t extent_bytes);
    ~KVSpillFile();

    KVSpillFile(const KVSpillFile&)            = delete;
    KVSpillFile& operator=(const KVSpillFile&) = delete;

    [[nodiscard]] std::size_t extent_bytes() const noexcept { return extent_bytes_; }

    [[nodiscard]] std::uint32_t capacity_extents() const noexcept {
        return static_cast<std::uint32_t>(slots_.size());
    }

    [[nodiscard]] std::size_t resident_extents() const noexcept { return resident_; }

    [[nodiscard]] std::size_t spilled_bytes() const noexcept { return spilled_bytes_; }

    [[nodiscard]] std::size_t restored_bytes() const noexcept { return restored_bytes_; }

    // Copies `bytes` into a free slot, or into the least recently used one when the file is full.
    // `use_epoch` is the caller's monotonic access counter; it is what LRU compares.
    // The returned slot stays owned by the caller until release(). Throws on I/O failure.
    [[nodiscard]] std::uint32_t write(std::span<const std::byte> bytes, std::uint64_t use_epoch);

    // Reads a slot back. `out` must be exactly extent_bytes() unless `bytes` was smaller at write
    // time, in which case the trailing bytes are the ones the file already held.
    void read(std::uint32_t slot, std::span<std::byte> out);

    void release(std::uint32_t slot) noexcept;

private:
    struct Slot {
        std::uint64_t use_epoch = 0;
        bool occupied           = false;
    };

    void seek(std::uint32_t slot, std::streamoff offset);
    [[nodiscard]] std::uint32_t least_recently_used() const noexcept;

    std::filesystem::path path_;
    std::fstream file_;
    std::size_t extent_bytes_   = 0;
    std::size_t resident_       = 0;
    std::size_t spilled_bytes_  = 0;
    std::size_t restored_bytes_ = 0;
    std::vector<Slot> slots_;
};

} // namespace ninfer
