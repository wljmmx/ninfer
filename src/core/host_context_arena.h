#pragma once

#include "core/arena.h"

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace ninfer {

class HostContextArena;

// A unique, immovable physical extent. Keeping this owner alive pins its bytes; views never
// allocate or charge a second copy. Transfer destinations remain reserved until publication.
// appendix FF (M3 3.2a): which consumer a Host allocation belongs to. Live = a live sequence's
// KV; Checkpoint = a reusable prefix snapshot. They differ in replaceability, so they get separate
// accounting and the live side gets a floor.
enum class HostConsumer { Live, Checkpoint };

class HostContextAllocation {
public:
    HostContextAllocation() noexcept = default;
    ~HostContextAllocation();
    HostContextAllocation(const HostContextAllocation&)            = delete;
    HostContextAllocation& operator=(const HostContextAllocation&) = delete;
    HostContextAllocation(HostContextAllocation&& other) noexcept;
    HostContextAllocation& operator=(HostContextAllocation&& other) noexcept;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

    [[nodiscard]] bool reserved() const noexcept { return valid() && reserved_; }

    [[nodiscard]] std::byte* data() const noexcept;
    void publish() noexcept;
    bool release() noexcept;

private:
    friend class HostContextArena;

    HostContextAllocation(HostContextArena& owner, std::size_t offset, std::size_t bytes,
                          bool reserved = true,
                          HostConsumer consumer = HostConsumer::Live) noexcept
        : owner_(&owner), offset_(offset), bytes_(bytes), reserved_(reserved),
          consumer_(consumer) {}

    void disarm() noexcept;

    HostContextArena* owner_ = nullptr;
    std::size_t offset_      = 0;
    std::size_t bytes_       = 0;
    bool reserved_           = false;
    HostConsumer consumer_   = HostConsumer::Live;
};

// One startup-fixed pinned backing shared by every optional Host context representation.
// minimum_allocation_bytes is the smallest supported complete geometry, not a quota split.
// It bounds metadata at startup; all allocations and split pieces must meet that minimum.
// The arena must outlive its allocations and the typed pools that own them.
class HostContextArena {
public:
    static constexpr std::size_t alignment = 256;

    HostContextArena(std::size_t capacity_bytes, std::size_t minimum_allocation_bytes);
    HostContextArena(const HostContextArena&)            = delete;
    HostContextArena& operator=(const HostContextArena&) = delete;
    HostContextArena(HostContextArena&&)                 = delete;
    HostContextArena& operator=(HostContextArena&&)      = delete;

    [[nodiscard]] std::size_t capacity_bytes() const noexcept { return capacity_bytes_; }

    [[nodiscard]] std::size_t minimum_allocation_bytes() const noexcept { return minimum_bytes_; }

    [[nodiscard]] std::size_t occupied_bytes() const noexcept { return occupied_bytes_; }

    [[nodiscard]] std::size_t reserved_bytes() const noexcept { return reserved_bytes_; }

    [[nodiscard]] std::size_t live_bytes() const noexcept {
        return occupied_bytes_ - reserved_bytes_;
    }

    [[nodiscard]] std::size_t free_bytes() const noexcept {
        return capacity_bytes_ - occupied_bytes_;
    }

    [[nodiscard]] std::size_t peak_occupied_bytes() const noexcept { return peak_occupied_bytes_; }

    [[nodiscard]] std::size_t allocation_count() const noexcept { return allocation_count_; }

    // appendix FF (M3 3.2a): tagged forms. The untagged ones delegate with Live and a zero floor,
    // which is provably identical to the previous behaviour (see the file header comment).
    [[nodiscard]] bool can_allocate(std::size_t bytes) const noexcept {
        return can_allocate(bytes, HostConsumer::Live);
    }
    [[nodiscard]] bool can_allocate(std::size_t bytes, HostConsumer consumer) const noexcept;
    [[nodiscard]] std::optional<HostContextAllocation> allocate(std::size_t bytes) noexcept {
        return allocate(bytes, HostConsumer::Live);
    }
    [[nodiscard]] std::optional<HostContextAllocation> allocate(std::size_t bytes,
                                                               HostConsumer consumer) noexcept;

    // Bytes the Checkpoint consumer must never take, so a live sequence can always demote a page.
    void set_live_floor_bytes(std::size_t bytes) noexcept { live_floor_bytes_ = bytes; }
    [[nodiscard]] std::size_t live_floor_bytes() const noexcept { return live_floor_bytes_; }
    [[nodiscard]] std::size_t live_kv_bytes_held() const noexcept { return live_kv_bytes_; }
    [[nodiscard]] std::size_t checkpoint_bytes_held() const noexcept { return checkpoint_bytes_; }
    [[nodiscard]] std::pair<HostContextAllocation, HostContextAllocation>
    split(HostContextAllocation&& allocation, std::size_t byte_offset);

private:
    friend class HostContextAllocation;

    struct FreeExtent {
        std::size_t offset = 0;
        std::size_t bytes  = 0;
    };

    [[nodiscard]] std::size_t allocation_bytes(std::size_t bytes) const noexcept;
    [[nodiscard]] std::optional<std::size_t> find_free_extent(std::size_t bytes) const noexcept;
    void insert_free_extent(FreeExtent extent) noexcept;
    void release(std::size_t offset, std::size_t bytes, bool reserved,
                 HostConsumer consumer) noexcept;

    std::optional<PinnedHostBuffer> backing_;
    std::size_t capacity_bytes_      = 0;
    std::size_t minimum_bytes_       = 0;
    std::size_t occupied_bytes_      = 0;
    std::size_t reserved_bytes_      = 0;
    std::size_t peak_occupied_bytes_ = 0;
    std::size_t live_floor_bytes_    = 0;
    std::size_t live_kv_bytes_       = 0;
    std::size_t checkpoint_bytes_    = 0;
    std::size_t allocation_count_    = 0;
    std::vector<FreeExtent> free_extents_;
};

} // namespace ninfer
