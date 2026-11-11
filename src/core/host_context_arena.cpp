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

HostContextAllocation::~HostContextAllocation() { (void)release(); }

HostContextAllocation::HostContextAllocation(HostContextAllocation&& other) noexcept
    : owner_(other.owner_), offset_(other.offset_), bytes_(other.bytes_),
      reserved_(other.reserved_) {
    other.disarm();
}

HostContextAllocation& HostContextAllocation::operator=(HostContextAllocation&& other) noexcept {
    if (this == &other) { return *this; }
    (void)release();
    owner_    = other.owner_;
    offset_   = other.offset_;
    bytes_    = other.bytes_;
    reserved_ = other.reserved_;
    other.disarm();
    return *this;
}

std::byte* HostContextAllocation::data() const noexcept {
    return valid() ? owner_->base() + offset_ : nullptr;
}

void HostContextAllocation::publish() noexcept {
    if (!reserved()) { return; }
    owner_->reserved_bytes_ -= bytes_;
    reserved_ = false;
}

bool HostContextAllocation::release() noexcept {
    if (!valid()) { return false; }
    owner_->release(offset_, bytes_, reserved_, consumer_);
    disarm();
    return true;
}

void HostContextAllocation::disarm() noexcept {
    owner_    = nullptr;
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
    minimum_bytes_ = (minimum_allocation_bytes + alignment - 1) & ~(alignment - 1);
    if (capacity_bytes_ == 0) { return; }
    // At most N allocated extents separate N+1 free runs; releases and splits never allocate
    // metadata. The unused trailing bytes remain part of the charged physical backing.
    free_extents_.reserve(capacity_bytes_ / minimum_bytes_ + 1);
    try {
        backing_.emplace(capacity_bytes_);
    } catch (const std::exception& error) {
        // A pinned request this size is not guaranteed: it competes with every other lock the
        // process holds (the expert arena, the KV staging ring) and with the driver's own budget.
        // The tier only needs addressable host memory - the copies become driver staged instead of
        // DMA - so a refused pin degrades the tier instead of aborting a 256K plan.
        std::fprintf(stderr,
                     "[host-context] pinned %.0f MiB refused (%s); using pageable backing "
                     "(host demotion copies will stage through the driver)\n",
                     static_cast<double>(capacity_bytes_) / (1024.0 * 1024.0), error.what());
        std::fflush(stderr);
#if defined(_WIN32)
        pageable_ = static_cast<std::byte*>(_aligned_malloc(capacity_bytes_, alignment));
        if (pageable_ != nullptr) {
            std::memset(pageable_, 0, capacity_bytes_);
        }
#endif
        if (pageable_ == nullptr) { throw; }
    }
    free_extents_.push_back({0, capacity_bytes_});
}

HostContextArena::~HostContextArena() {
#if defined(_WIN32)
    if (pageable_ != nullptr) { _aligned_free(pageable_); }
#endif
}

std::size_t HostContextArena::allocation_bytes(std::size_t bytes) const noexcept {
    if (bytes == 0 || bytes > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
        return 0;
    }
    const std::size_t rounded = (bytes + alignment - 1) & ~(alignment - 1);
    return rounded < minimum_bytes_ ? 0 : rounded;
}

std::optional<std::size_t> HostContextArena::find_free_extent(std::size_t bytes) const noexcept {
    for (std::size_t index = 0; index < free_extents_.size(); ++index) {
        if (free_extents_[index].bytes >= bytes) { return index; }
    }
    return std::nullopt;
}

bool HostContextArena::can_allocate(std::size_t bytes, HostConsumer consumer) const noexcept {
    const std::size_t rounded = allocation_bytes(bytes);
    if (rounded == 0 || !find_free_extent(rounded).has_value()) { return false; }
    // appendix FF: the checkpoint consumer must leave the live floor free. With a zero floor this is
    // already implied by find_free_extent, so the untagged form keeps its exact behaviour.
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
    const auto free_index = find_free_extent(rounded);
    if (!free_index) { return std::nullopt; }
    FreeExtent& free         = free_extents_[*free_index];
    const std::size_t offset = free.offset;
    free.offset += rounded;
    free.bytes -= rounded;
    if (free.bytes == 0) {
        free_extents_.erase(free_extents_.begin() + static_cast<std::ptrdiff_t>(*free_index));
    }
    occupied_bytes_ += rounded;
    reserved_bytes_ += rounded;
    if (consumer == HostConsumer::Checkpoint) {
        checkpoint_bytes_ += rounded;
    } else {
        live_kv_bytes_ += rounded;
    }
    peak_occupied_bytes_ = std::max(peak_occupied_bytes_, occupied_bytes_);
    ++allocation_count_;
    return HostContextAllocation(*this, offset, rounded, true, consumer);
}

std::pair<HostContextAllocation, HostContextAllocation>
HostContextArena::split(HostContextAllocation&& allocation, std::size_t byte_offset) {
    if (allocation.owner_ != this) {
        throw std::invalid_argument("Cannot split a foreign or released Host context allocation");
    }
    if (byte_offset % alignment != 0 || byte_offset < minimum_bytes_ ||
        byte_offset > allocation.bytes_ || allocation.bytes_ - byte_offset < minimum_bytes_) {
        throw std::out_of_range("Host context split must retain two aligned complete geometries");
    }
    // appendix FF (M3 3.2d): both halves keep the original's consumer tag. Without it they defaulted to
    // Live, so releasing a split allocation decremented live_kv_bytes_ and left checkpoint_bytes_
    // inflated - the mis-attribution FF.2 recorded. The arena total is untouched either way.
    HostContextAllocation left(*this, allocation.offset_, byte_offset, allocation.reserved_,
                               allocation.consumer_);
    HostContextAllocation right(*this, allocation.offset_ + byte_offset,
                                allocation.bytes_ - byte_offset, allocation.reserved_,
                                allocation.consumer_);
    allocation.disarm();
    ++allocation_count_;
    return {std::move(left), std::move(right)};
}

void HostContextArena::release(std::size_t offset, std::size_t bytes, bool reserved,
                               HostConsumer consumer) noexcept {
    occupied_bytes_ -= bytes;
    if (reserved) { reserved_bytes_ -= bytes; }
    // appendix FF: the per-consumer counters must come back down here, or they would only ever grow.
    if (consumer == HostConsumer::Checkpoint) {
        checkpoint_bytes_ -= bytes;
    } else {
        live_kv_bytes_ -= bytes;
    }
    --allocation_count_;
    insert_free_extent({offset, bytes});
}

void HostContextArena::insert_free_extent(FreeExtent extent) noexcept {
    const auto position = std::lower_bound(
        free_extents_.begin(), free_extents_.end(), extent.offset,
        [](const FreeExtent& candidate, std::size_t offset) { return candidate.offset < offset; });
    auto inserted = free_extents_.insert(position, extent);
    if (inserted != free_extents_.begin()) {
        auto previous = inserted - 1;
        if (previous->offset + previous->bytes == inserted->offset) {
            previous->bytes += inserted->bytes;
            free_extents_.erase(inserted);
            inserted = previous;
        }
    }
    const auto next = inserted + 1;
    if (next != free_extents_.end() && inserted->offset + inserted->bytes == next->offset) {
        inserted->bytes += next->bytes;
        free_extents_.erase(next);
    }
}

} // namespace ninfer
