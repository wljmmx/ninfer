#pragma once

#include "core/arena.h"

#include <cstddef>
#include <cstdint>
#include <memory>
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

    HostContextAllocation(HostContextArena& owner, std::uint32_t extent, std::size_t offset,
                          std::size_t bytes, bool reserved = true,
                          HostConsumer consumer = HostConsumer::Live) noexcept
        : owner_(&owner), extent_(extent), offset_(offset), bytes_(bytes), reserved_(reserved),
          consumer_(consumer) {}

    void disarm() noexcept;

    HostContextArena* owner_ = nullptr;
    std::uint32_t extent_    = 0;
    std::size_t offset_      = 0;
    std::size_t bytes_       = 0;
    bool reserved_           = false;
    HostConsumer consumer_   = HostConsumer::Live;
};

// One startup-fixed pinned budget shared by every optional Host context representation.
// minimum_allocation_bytes is the smallest supported complete geometry, not a quota split.
// It bounds metadata at startup; all allocations and split pieces must meet that minimum.
// The budget is a CAP, not a pre-pin: pinned extents are created on demand and an allocation
// never crosses an extent boundary, so every allocation stays one contiguous pinned span for
// the consumers that copy into it. The arena must outlive its allocations and the typed pools
// that own them.
class HostContextArena {
public:
    static constexpr std::size_t alignment = 256;

    HostContextArena(std::size_t capacity_bytes, std::size_t minimum_allocation_bytes);
    ~HostContextArena();
    HostContextArena(const HostContextArena&)            = delete;
    HostContextArena& operator=(const HostContextArena&) = delete;
    HostContextArena(HostContextArena&&)                 = delete;
    HostContextArena& operator=(HostContextArena&&)      = delete;

    // False once any extent had to fall back to pageable memory; true while every extent pinned.
    [[nodiscard]] bool pinned() const noexcept { return pageable_extent_bytes_ == 0; }

    [[nodiscard]] std::size_t capacity_bytes() const noexcept { return capacity_bytes_; }

    // What is actually pinned right now, and how many extents that took. Startup pins nothing;
    // both grow with the allocations and never exceed capacity_bytes().
    [[nodiscard]] std::size_t pinned_bytes() const noexcept { return pinned_extent_bytes_; }

    [[nodiscard]] std::size_t mapped_bytes() const noexcept { return extent_bytes_; }

    [[nodiscard]] std::size_t extent_count() const noexcept { return extents_.size(); }

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

    // One on-demand backing. `backing` owns the pin when it holds a value; `pageable` is the
    // private fallback of a build that has one. Allocations are carved out of a single extent's
    // free runs, so no allocation is ever split across two backing blocks.
    struct Extent {
        Extent()                              = default;
        ~Extent();
        Extent(const Extent&)                 = delete;
        Extent& operator=(const Extent&)      = delete;

        std::optional<PinnedHostBuffer> backing;
        std::byte* pageable = nullptr;
        std::size_t bytes   = 0;
        std::vector<FreeExtent> free_runs;

        [[nodiscard]] std::byte* data() const noexcept {
            return backing.has_value() ? static_cast<std::byte*>(backing->data()) : pageable;
        }
    };

    [[nodiscard]] std::size_t allocation_bytes(std::size_t bytes) const noexcept;
    [[nodiscard]] std::optional<std::pair<std::size_t, std::size_t>>
    find_free_extent(std::size_t bytes) const noexcept;
    [[nodiscard]] std::unique_ptr<Extent> make_extent(std::size_t bytes) noexcept;
    [[nodiscard]] HostContextAllocation charge(std::uint32_t extent, std::size_t offset,
                                               std::size_t bytes, HostConsumer consumer) noexcept;
    void insert_free_extent(std::vector<FreeExtent>& runs, FreeExtent extent) const;
    void release(std::uint32_t extent, std::size_t offset, std::size_t bytes, bool reserved,
                 HostConsumer consumer) noexcept;

    std::vector<std::unique_ptr<Extent>> extents_;
    std::size_t capacity_bytes_        = 0;
    std::size_t extent_bytes_          = 0;
    std::size_t pinned_extent_bytes_   = 0;
    std::size_t pageable_extent_bytes_ = 0;
    std::size_t minimum_bytes_         = 0;
    std::size_t occupied_bytes_        = 0;
    std::size_t reserved_bytes_        = 0;
    std::size_t peak_occupied_bytes_   = 0;
    std::size_t live_floor_bytes_      = 0;
    std::size_t live_kv_bytes_         = 0;
    std::size_t checkpoint_bytes_      = 0;
    std::size_t allocation_count_      = 0;
};

} // namespace ninfer
