#include "core/host_context_arena.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace ninfer {
namespace {

// Granularity a lazy extent grows by. It is a floor, not a ceiling: a single allocation larger
// than this gets an extent of its own size, because an allocation must stay one contiguous span.
constexpr std::size_t kExtentBytes = 64ULL << 20;

std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + (alignment - 1)) & ~(alignment - 1);
}

} // namespace

HostContextAllocation::~HostContextAllocation() { (void)release(); }

HostContextAllocation::HostContextAllocation(HostContextAllocation&& other) noexcept
    : owner_(other.owner_), extent_(other.extent_), offset_(other.offset_), bytes_(other.bytes_),
      reserved_(other.reserved_), consumer_(other.consumer_) {
    other.disarm();
}

HostContextAllocation& HostContextAllocation::operator=(HostContextAllocation&& other) noexcept {
    if (this == &other) { return *this; }
    (void)release();
    owner_    = other.owner_;
    extent_   = other.extent_;
    offset_   = other.offset_;
    bytes_    = other.bytes_;
    reserved_ = other.reserved_;
    consumer_ = other.consumer_;
    other.disarm();
    return *this;
}

std::byte* HostContextAllocation::data() const noexcept {
    if (!valid() || extent_ >= owner_->extents_.size()) { return nullptr; }
    return owner_->extents_[extent_]->data() + offset_;
}

void HostContextAllocation::publish() noexcept {
    if (!reserved()) { return; }
    owner_->reserved_bytes_ -= bytes_;
    reserved_ = false;
}

bool HostContextAllocation::release() noexcept {
    if (!valid()) { return false; }
    owner_->release(extent_, offset_, bytes_, reserved_, consumer_);
    disarm();
    return true;
}

void HostContextAllocation::disarm() noexcept {
    owner_    = nullptr;
    extent_   = 0;
    offset_   = 0;
    bytes_    = 0;
    reserved_ = false;
    consumer_ = HostConsumer::Live;
}

HostContextArena::HostContextArena(std::size_t capacity_bytes, std::size_t minimum_allocation_bytes)
    : capacity_bytes_(capacity_bytes) {
    if (minimum_allocation_bytes == 0 ||
        minimum_allocation_bytes > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
        throw std::invalid_argument("Host context minimum allocation is invalid");
    }
    minimum_bytes_ = align_up(minimum_allocation_bytes, alignment);
    if (capacity_bytes_ == 0) { return; }
    // The budget is charged, not pinned: nothing is locked here. This is what lets a 24 GiB Host
    // tier start on a box that cannot take a single 24 GiB cudaMallocHost - measured as
    // cudaErrorMemoryAllocation at 19.9 GB free and cudaErrorAlreadyMapped at 26.7 GB free - while
    // the logical page count the operator asked for stays exactly the same.
    constexpr double kMib = 1024.0 * 1024.0;
    std::fprintf(stderr,
                 "[host-context] budget: %.0f MiB charged, lazily pinned in %.0f MiB extents "
                 "(nothing is locked before the first page arrives)\n",
                 static_cast<double>(capacity_bytes_) / kMib,
                 static_cast<double>(kExtentBytes) / kMib);
    std::fflush(stderr);
}

HostContextArena::~HostContextArena() = default;

HostContextArena::Extent::~Extent() {
#if defined(_WIN32)
    if (pageable != nullptr) { _aligned_free(pageable); }
#endif
}

std::size_t HostContextArena::allocation_bytes(std::size_t bytes) const noexcept {
    if (bytes == 0 || bytes > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
        return 0;
    }
    const std::size_t rounded = align_up(bytes, alignment);
    return rounded < minimum_bytes_ ? 0 : rounded;
}

std::optional<std::pair<std::size_t, std::size_t>>
HostContextArena::find_free_extent(std::size_t bytes) const noexcept {
    for (std::size_t extent = 0; extent < extents_.size(); ++extent) {
        const std::vector<FreeExtent>& runs = extents_[extent]->free_runs;
        for (std::size_t run = 0; run < runs.size(); ++run) {
            if (runs[run].bytes >= bytes) { return std::make_pair(extent, run); }
        }
    }
    return std::nullopt;
}

std::unique_ptr<HostContextArena::Extent>
HostContextArena::make_extent(std::size_t bytes) noexcept {
    auto extent = std::unique_ptr<Extent>(new (std::nothrow) Extent());
    if (!extent) { return nullptr; }
    try {
        extent->backing.emplace(bytes);
        pinned_extent_bytes_ += bytes;
    } catch (const std::exception& error) {
        // A pinned request this size is not guaranteed: it competes with every other lock the
        // process holds (the expert arena, the KV staging ring) and with the driver's own budget.
        // The tier only needs addressable host memory - the copies become driver staged instead of
        // DMA - so a refused pin degrades one extent instead of aborting a 256K plan.
        constexpr double kMib = 1024.0 * 1024.0;
        std::fprintf(stderr,
                     "[host-context] pinned %.0f MiB extent refused (%s); using pageable backing "
                     "(host demotion copies will stage through the driver)\n",
                     static_cast<double>(bytes) / kMib, error.what());
        std::fflush(stderr);
#if defined(_WIN32)
        extent->pageable = static_cast<std::byte*>(_aligned_malloc(bytes, alignment));
        if (extent->pageable != nullptr) {
            std::memset(extent->pageable, 0, bytes);
            pageable_extent_bytes_ += bytes;
        }
#endif
        if (extent->pageable == nullptr) { return nullptr; }
    }
    extent->bytes = bytes;
    extent->free_runs.push_back(FreeExtent{0, bytes});
    return extent;
}

HostContextAllocation HostContextArena::charge(std::uint32_t extent, std::size_t offset,
                                               std::size_t bytes, HostConsumer consumer) noexcept {
    occupied_bytes_ += bytes;
    reserved_bytes_ += bytes;
    if (consumer == HostConsumer::Checkpoint) {
        checkpoint_bytes_ += bytes;
    } else {
        live_kv_bytes_ += bytes;
    }
    peak_occupied_bytes_ = std::max(peak_occupied_bytes_, occupied_bytes_);
    ++allocation_count_;
    return HostContextAllocation(*this, extent, offset, bytes, true, consumer);
}

bool HostContextArena::can_allocate(std::size_t bytes, HostConsumer consumer) const noexcept {
    const std::size_t rounded = allocation_bytes(bytes);
    if (rounded == 0) { return false; }
    // A request fits either in an extent already mapped, or in the budget that is still unpinned.
    // Both are bounded by capacity_bytes_, so the answer never promises bytes the budget lacks.
    if (!find_free_extent(rounded).has_value() && capacity_bytes_ - extent_bytes_ < rounded) {
        return false;
    }
    // appendix FF: the checkpoint consumer must leave the live floor free. With a zero floor this is
    // already implied by the free extent, so the untagged form keeps its exact behaviour.
    return consumer != HostConsumer::Checkpoint ||
           occupied_bytes_ + rounded + live_floor_bytes_ <= capacity_bytes_;
}

std::optional<HostContextAllocation> HostContextArena::allocate(std::size_t bytes,
                                                                HostConsumer consumer) noexcept {
    const std::size_t rounded = allocation_bytes(bytes);
    if (rounded == 0) { return std::nullopt; }
    // appendix FF: same floor rule as can_allocate, so the two can never disagree.
    if (consumer == HostConsumer::Checkpoint &&
        occupied_bytes_ + rounded + live_floor_bytes_ > capacity_bytes_) {
        return std::nullopt;
    }
    if (const auto found = find_free_extent(rounded)) {
        FreeExtent& free         = extents_[found->first]->free_runs[found->second];
        const std::size_t offset = free.offset;
        free.offset += rounded;
        free.bytes -= rounded;
        if (free.bytes == 0) {
            auto& runs = extents_[found->first]->free_runs;
            runs.erase(runs.begin() + static_cast<std::ptrdiff_t>(found->second));
        }
        return charge(static_cast<std::uint32_t>(found->first), offset, rounded, consumer);
    }
    // Nothing mapped can take it: grow one extent, capped by the configured budget. At least the
    // request itself, at most one granularity more, and never more than what is still unpinned.
    const std::size_t spare = capacity_bytes_ - extent_bytes_;
    if (spare < rounded) { return std::nullopt; }
    const std::size_t extent_bytes =
        std::min(align_up(std::max(rounded, std::min(kExtentBytes, spare)), alignment), spare);
    std::unique_ptr<Extent> extent = make_extent(extent_bytes);
    if (!extent) { return std::nullopt; }
    const std::uint32_t index = static_cast<std::uint32_t>(extents_.size());
    extent_bytes_ += extent_bytes;
    // The whole extent starts free; the request consumes its head, the rest stays available.
    extent->free_runs.clear();
    extent->free_runs.push_back(FreeExtent{rounded, extent_bytes - rounded});
    if (extent->free_runs.back().bytes == 0) { extent->free_runs.clear(); }
    extents_.push_back(std::move(extent));
    return charge(index, 0, rounded, consumer);
}

std::pair<HostContextAllocation, HostContextAllocation>
HostContextArena::split(HostContextAllocation&& allocation, std::size_t byte_offset) {
    if (allocation.owner_ != this || allocation.extent_ >= extents_.size()) {
        throw std::invalid_argument("Cannot split a foreign or released Host context allocation");
    }
    if (byte_offset % alignment != 0 || byte_offset < minimum_bytes_ ||
        byte_offset > allocation.bytes_ || allocation.bytes_ - byte_offset < minimum_bytes_) {
        throw std::out_of_range("Host context split must retain two aligned complete geometries");
    }
    // appendix FF (M3 3.2d): both halves keep the original's consumer tag. Without it they defaulted to
    // Live, so releasing a split allocation decremented live_kv_bytes_ and left checkpoint_bytes_
    // inflated - the mis-attribution FF.2 recorded. The arena total is untouched either way.
    // Both halves stay inside the same extent, so each one remains a contiguous span.
    HostContextAllocation left(*this, allocation.extent_, allocation.offset_, byte_offset,
                               allocation.reserved_, allocation.consumer_);
    HostContextAllocation right(*this, allocation.extent_, allocation.offset_ + byte_offset,
                                allocation.bytes_ - byte_offset, allocation.reserved_,
                                allocation.consumer_);
    allocation.disarm();
    ++allocation_count_;
    return {std::move(left), std::move(right)};
}

void HostContextArena::release(std::uint32_t extent, std::size_t offset, std::size_t bytes,
                               bool reserved, HostConsumer consumer) noexcept {
    occupied_bytes_ -= bytes;
    if (reserved) { reserved_bytes_ -= bytes; }
    // appendix FF: the per-consumer counters must come back down here, or they would only ever grow.
    if (consumer == HostConsumer::Checkpoint) {
        checkpoint_bytes_ -= bytes;
    } else {
        live_kv_bytes_ -= bytes;
    }
    --allocation_count_;
    if (extent < extents_.size()) { insert_free_extent(extents_[extent]->free_runs, {offset, bytes}); }
}

void HostContextArena::insert_free_extent(std::vector<FreeExtent>& runs, FreeExtent extent) const {
    const auto position = std::lower_bound(
        runs.begin(), runs.end(), extent.offset,
        [](const FreeExtent& candidate, std::size_t offset) { return candidate.offset < offset; });
    auto inserted = runs.insert(position, extent);
    if (inserted != runs.begin()) {
        auto previous = inserted - 1;
        if (previous->offset + previous->bytes == inserted->offset) {
            previous->bytes += inserted->bytes;
            runs.erase(inserted);
            inserted = previous;
        }
    }
    const auto next = inserted + 1;
    if (next != runs.end() && inserted->offset + inserted->bytes == next->offset) {
        inserted->bytes += next->bytes;
        runs.erase(next);
    }
}

} // namespace ninfer
