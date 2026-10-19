#include "core/host_reserve.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <sysinfoapi.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace ninfer {
namespace {

std::size_t platform_granularity() {
#ifdef _WIN32
    SYSTEM_INFO info{};
    GetSystemInfo(&info);
    return static_cast<std::size_t>(info.dwAllocationGranularity);
#else
    const long page = sysconf(_SC_PAGESIZE);
    const auto base = page > 0 ? static_cast<std::size_t>(page) : std::size_t{4096};
    return std::max<std::size_t>(base, 64 * 1024);
#endif
}

void* reserve(std::size_t bytes) {
#ifdef _WIN32
    void* base = VirtualAlloc(nullptr, bytes, MEM_RESERVE, PAGE_NOACCESS);
    if (!base) { throw std::bad_alloc(); }
    return base;
#else
    void* base = mmap(nullptr, bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (base == MAP_FAILED) { throw std::bad_alloc(); }
    return base;
#endif
}

void unreserve(void* base, std::size_t bytes) {
#ifdef _WIN32
    (void)VirtualFree(base, 0, MEM_RELEASE);
#else
    (void)munmap(base, bytes);
#endif
}

void commit_pages(void* address, std::size_t bytes) {
#ifdef _WIN32
    if (!VirtualAlloc(address, bytes, MEM_COMMIT, PAGE_READWRITE)) {
        throw std::runtime_error("host reserve commit failed");
    }
#else
    if (mprotect(address, bytes, PROT_READ | PROT_WRITE) != 0) {
        throw std::runtime_error("host reserve commit failed");
    }
#endif
}

void decommit_pages(void* address, std::size_t bytes) {
#ifdef _WIN32
    if (!VirtualFree(address, bytes, MEM_DECOMMIT)) {
        throw std::runtime_error("host reserve decommit failed");
    }
#else
    if (madvise(address, bytes, MADV_DONTNEED) != 0) {
        throw std::runtime_error("host reserve decommit failed");
    }
    if (mprotect(address, bytes, PROT_NONE) != 0) {
        throw std::runtime_error("host reserve decommit failed");
    }
#endif
}

std::size_t round_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

} // namespace

std::size_t HostReserve::granularity() noexcept {
    static const std::size_t value = platform_granularity();
    return value;
}

HostReserve::HostReserve(std::size_t bytes) {
    if (!bytes) { return; }
    const auto unit = granularity();
    bytes_          = round_up(bytes, unit);
    base_           = reserve(bytes_);
    pages_.assign(bytes_ / unit, 0);
}

HostReserve::~HostReserve() {
    if (base_) { unreserve(base_, bytes_); }
}

HostReserve::HostReserve(HostReserve&& other) noexcept
    : base_(other.base_), bytes_(other.bytes_), committed_(other.committed_),
      peak_committed_(other.peak_committed_), pages_(std::move(other.pages_)) {
    other.base_ = nullptr;
    other.bytes_ = other.committed_ = other.peak_committed_ = 0;
}

HostReserve& HostReserve::operator=(HostReserve&& other) noexcept {
    if (this != &other) {
        if (base_) { unreserve(base_, bytes_); }
        base_           = other.base_;
        bytes_          = other.bytes_;
        committed_      = other.committed_;
        peak_committed_ = other.peak_committed_;
        pages_          = std::move(other.pages_);
        other.base_ = nullptr;
        other.bytes_ = other.committed_ = other.peak_committed_ = 0;
    }
    return *this;
}

void HostReserve::commit(std::uint64_t offset, std::uint64_t bytes) {
    if (!bytes) { return; }
    if (offset > bytes_ || bytes > bytes_ - offset) {
        throw std::out_of_range("host reserve commit exceeds its range");
    }
    const auto unit  = granularity();
    const auto first = static_cast<std::size_t>(offset / unit);
    const auto last  = static_cast<std::size_t>((offset + bytes + unit - 1) / unit);
    for (std::size_t index = first; index < last; ++index) {
        if (pages_[index]) { continue; }
        commit_pages(static_cast<std::byte*>(base_) + index * unit, unit);
        pages_[index] = 1;
    }
    committed_ = static_cast<std::size_t>(std::count(pages_.begin(), pages_.end(), std::uint8_t{1})) *
                 unit;
    peak_committed_ = std::max(peak_committed_, committed_);
}

void HostReserve::decommit(std::uint64_t offset, std::uint64_t bytes) {
    if (!bytes) { return; }
    if (offset > bytes_ || bytes > bytes_ - offset) {
        throw std::out_of_range("host reserve decommit exceeds its range");
    }
    const auto unit  = granularity();
    const auto first = static_cast<std::size_t>(offset / unit);
    const auto last  = static_cast<std::size_t>((offset + bytes + unit - 1) / unit);
    for (std::size_t index = first; index < last; ++index) {
        if (!pages_[index]) { continue; }
        decommit_pages(static_cast<std::byte*>(base_) + index * unit, unit);
        pages_[index] = 0;
    }
    committed_ = static_cast<std::size_t>(std::count(pages_.begin(), pages_.end(), std::uint8_t{1})) *
                 unit;
}

bool HostReserve::resident(std::uint64_t offset, std::uint64_t bytes) const noexcept {
    if (!bytes) { return true; }
    if (offset > bytes_ || bytes > bytes_ - offset) { return false; }
    const auto unit  = granularity();
    const auto first = static_cast<std::size_t>(offset / unit);
    const auto last  = static_cast<std::size_t>((offset + bytes + unit - 1) / unit);
    for (std::size_t index = first; index < last; ++index) {
        if (!pages_[index]) { return false; }
    }
    return true;
}

void HostReserve::release_all() {
    if (!base_) { return; }
    const auto unit = granularity();
    for (std::size_t index = 0; index < pages_.size(); ++index) {
        if (!pages_[index]) { continue; }
        decommit_pages(static_cast<std::byte*>(base_) + index * unit, unit);
        pages_[index] = 0;
    }
    committed_ = 0;
}

} // namespace ninfer
