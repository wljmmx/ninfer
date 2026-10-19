#pragma once

#include "core/device.h"

#include <cstddef>

namespace ninfer {

// port (sm_89): a pinned Host range and its address in one CUDA device's mapping.
//
// The expert cache uploads from pinned Host memory; this is the minimal view it needs.
// The backing and its registration must outlive every transfer that uses the view.
struct CudaHostView {
    void* host              = nullptr;
    std::size_t bytes       = 0;
    void* device_address    = nullptr; // cudaHostGetDevicePointer result
    int cuda_ordinal = -1;

    [[nodiscard]] bool valid() const noexcept {
        return host != nullptr && device_address != nullptr &&
               cuda_ordinal >= 0;
    }
};

// Registers an existing Host range as pinned for one device. Not copyable; movable.
class CudaHostRegistration {
public:
    CudaHostRegistration() noexcept = default;
    CudaHostRegistration(void* host, std::size_t bytes, int cuda_ordinal);
    ~CudaHostRegistration();

    CudaHostRegistration(const CudaHostRegistration&)            = delete;
    CudaHostRegistration& operator=(const CudaHostRegistration&) = delete;
    CudaHostRegistration(CudaHostRegistration&& other) noexcept;
    CudaHostRegistration& operator=(CudaHostRegistration&& other) noexcept;

    [[nodiscard]] const CudaHostView& view() const noexcept { return view_; }

private:
    void release() noexcept;

    CudaHostView view_;
};

} // namespace ninfer

