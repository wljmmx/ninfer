#include "core/vmm_pool.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer {
namespace {

[[noreturn]] void fail(CUresult result, const char* operation) {
    const char* name = nullptr;
    cuGetErrorName(result, &name);
    throw std::runtime_error(std::string("vmm_pool: ") + operation + " failed: " +
                             (name != nullptr ? name : "unknown"));
}

void check(CUresult result, const char* operation) {
    if (result != CUDA_SUCCESS) { fail(result, operation); }
}

CUmemGenericAllocationHandle as_handle(void* handle) {
    return static_cast<CUmemGenericAllocationHandle>(
        reinterpret_cast<std::uintptr_t>(handle));
}

void* as_opaque(CUmemGenericAllocationHandle handle) {
    return reinterpret_cast<void*>(static_cast<std::uintptr_t>(handle));
}

} // namespace

VmmPool::VmmPool(Device device, std::size_t capacity_bytes, std::size_t segment_bytes)
    : device_(device), capacity_(capacity_bytes), segment_(segment_bytes) {
    if (capacity_bytes == 0 || segment_bytes == 0) {
        throw std::invalid_argument("vmm_pool requires a nonzero capacity and segment");
    }
    // Callers size their pool from free memory, which is never segment aligned; round up so every
    // segment can be mapped and returned on its own.
    capacity_bytes = (capacity_bytes + segment_bytes - 1) / segment_bytes * segment_bytes;
    capacity_      = capacity_bytes;
    if (capacity_bytes % 256 != 0) {
        throw std::invalid_argument("vmm_pool capacity must be 256-byte aligned");
    }
    CudaDeviceScope scope(device_);
    int index = 0;
    check(cuInit(0), "cuInit");
    if (cudaGetDevice(&index) != cudaSuccess) { throw std::runtime_error("vmm_pool: no device"); }

    check(cuMemAddressReserve(reinterpret_cast<CUdeviceptr*>(&base_), capacity_, segment_, 0, 0),
          "cuMemAddressReserve");

    device_ordinal_ = index;
    handles_.assign(capacity_ / segment_, nullptr);
    map_range(0, capacity_);
    mapped_ = capacity_;
}

VmmPool::~VmmPool() { destroy(); }

VmmPool::VmmPool(VmmPool&& other) noexcept
    : device_(other.device_), base_(std::exchange(other.base_, nullptr)),
      handles_(std::move(other.handles_)), capacity_(other.capacity_),
      segment_(other.segment_), mapped_(other.mapped_) {
    other.capacity_ = 0;
    other.segment_  = 0;
    other.mapped_   = 0;
}

VmmPool& VmmPool::operator=(VmmPool&& other) noexcept {
    if (this != &other) {
        destroy();
        device_   = other.device_;
        base_     = std::exchange(other.base_, nullptr);
        handles_  = std::move(other.handles_);
        capacity_ = other.capacity_;
        segment_  = other.segment_;
        mapped_   = other.mapped_;
        other.capacity_ = 0;
        other.segment_  = 0;
        other.mapped_   = 0;
    }
    return *this;
}

void VmmPool::map_range(std::size_t offset, std::size_t bytes) {
    if (bytes == 0) { return; }
    if (bytes % segment_ != 0) { throw std::invalid_argument("vmm_pool: partial segment map"); }
    CudaDeviceScope scope(device_);
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id   = device_ordinal_;
    access.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    // Each segment owns its own physical allocation: mapping one handle at several addresses is
    // rejected, and an independent handle is what makes a segment returnable on its own.
    for (std::size_t done = 0; done < bytes; done += segment_) {
        const auto index = (offset + done) / segment_;
        if (handles_[index] != nullptr) { continue; }
        CUmemAllocationProp property{};
        property.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
        property.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        property.location.id   = device_ordinal_;
        CUmemGenericAllocationHandle handle{};
        check(cuMemCreate(&handle, segment_, &property, 0), "cuMemCreate");
        const auto address = reinterpret_cast<CUdeviceptr>(base_) + offset + done;
        check(cuMemMap(address, segment_, 0, handle, 0), "cuMemMap");
        check(cuMemSetAccess(address, segment_, &access, 1), "cuMemSetAccess");
        handles_[index] = as_opaque(handle);
    }
}

void VmmPool::unmap_range(std::size_t offset, std::size_t bytes) {
    if (bytes == 0) { return; }
    if (bytes % segment_ != 0) { throw std::invalid_argument("vmm_pool: partial segment unmap"); }
    CudaDeviceScope scope(device_);
    for (std::size_t done = 0; done < bytes; done += segment_) {
        const auto index = (offset + done) / segment_;
        if (handles_[index] == nullptr) { continue; }
        check(cuMemUnmap(reinterpret_cast<CUdeviceptr>(base_) + offset + done, segment_),
              "cuMemUnmap");
        check(cuMemRelease(as_handle(handles_[index])), "cuMemRelease");
        handles_[index] = nullptr;
    }
}

std::size_t VmmPool::release(std::size_t wanted_bytes, std::size_t keep_mapped_bytes) {
    if (base_ == nullptr || mapped_ == 0) { return 0; }
    const auto floor = std::max<std::size_t>(keep_mapped_bytes, segment_);
    if (mapped_ <= floor) { return 0; }
    auto target = wanted_bytes >= mapped_ ? 0 : mapped_ - wanted_bytes;
    target      = std::max(target, floor);
    target      = target / segment_ * segment_;
    if (target >= mapped_) { return 0; }
    const auto released = mapped_ - target;
    unmap_range(target, released);
    mapped_ = target;
    return released;
}

std::size_t VmmPool::acquire(std::size_t wanted_bytes) {
    if (base_ == nullptr || mapped_ >= capacity_ || wanted_bytes == 0) { return 0; }
    auto target = std::min(capacity_, mapped_ + wanted_bytes);
    target      = std::min(capacity_, (target + segment_ - 1) / segment_ * segment_);
    if (target <= mapped_) { return 0; }
    const auto added = target - mapped_;
    map_range(mapped_, added);
    mapped_ = target;
    return added;
}

bool VmmPool::is_mapped_range(std::size_t offset, std::size_t bytes) const noexcept {
    if (base_ == nullptr || bytes == 0 || offset + bytes > capacity_) { return false; }
    for (std::size_t done = 0; done < bytes; done += std::min(segment_, bytes - done)) {
        const auto index = (offset + done) / segment_;
        if (index >= handles_.size() || handles_[index] == nullptr) { return false; }
    }
    return true;
}

std::size_t VmmPool::unmap_at(std::size_t offset, std::size_t bytes) {
    if (base_ == nullptr || bytes == 0) { return 0; }
    if (offset % segment_ != 0 || bytes % segment_ != 0 || offset + bytes > mapped_) {
        throw std::out_of_range("vmm_pool: unmap_at needs an aligned mapped range");
    }
    unmap_range(offset, bytes);
    mapped_ -= bytes;
    return bytes;
}

std::size_t VmmPool::map_at(std::size_t offset, std::size_t bytes) {
    if (base_ == nullptr || bytes == 0) { return 0; }
    if (offset % segment_ != 0 || bytes % segment_ != 0 || offset + bytes > capacity_) {
        throw std::out_of_range("vmm_pool: map_at needs an aligned reserved range");
    }
    map_range(offset, bytes);
    mapped_ += bytes;
    return bytes;
}

void VmmPool::fill(int byte_value) {
    CudaDeviceScope scope(device_);
    if (cudaMemset(base_, byte_value, mapped_) != cudaSuccess) {
        throw std::runtime_error("vmm_pool: cudaMemset failed");
    }
}

void VmmPool::copy_from_host(const void* source, std::size_t count, std::size_t offset) {
    if (!is_mapped_range(offset, count)) {
        throw std::out_of_range("vmm_pool: copy touches an unmapped segment");
    }
    CudaDeviceScope scope(device_);
    if (cudaMemcpy(static_cast<std::byte*>(base_) + offset, source, count,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error("vmm_pool: cudaMemcpy failed");
    }
}

void VmmPool::copy_to_host(void* destination, std::size_t count, std::size_t offset) const {
    if (!is_mapped_range(offset, count)) {
        throw std::out_of_range("vmm_pool: copy touches an unmapped segment");
    }
    CudaDeviceScope scope(device_);
    if (cudaMemcpy(destination, static_cast<const std::byte*>(base_) + offset, count,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("vmm_pool: cudaMemcpy failed");
    }
}

void VmmPool::destroy() noexcept {
    if (base_ == nullptr) { return; }
    try {
        CudaDeviceScope scope(device_);
        for (std::size_t done = 0; done < mapped_; done += segment_) {
            const auto index = done / segment_;
            if (index >= handles_.size() || handles_[index] == nullptr) { continue; }
            cuMemUnmap(reinterpret_cast<CUdeviceptr>(base_) + done, segment_);
            cuMemRelease(as_handle(handles_[index]));
            handles_[index] = nullptr;
        }
        cuMemAddressFree(reinterpret_cast<CUdeviceptr>(base_), capacity_);
    } catch (...) {
    }
    base_     = nullptr;
    capacity_ = 0;
    mapped_   = 0;
}

} // namespace ninfer
