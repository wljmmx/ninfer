#pragma once

#include "core/memory.h"

#include <cstddef>

namespace ninfer {

// port (sm_89): registers an existing Host range as pinned for one device and exposes the
// bench-compatible CudaHostView (Host span + device mapping address).
class CudaHostRegistration {
public:
    CudaHostRegistration() noexcept = default;
    CudaHostRegistration(void* host, std::size_t bytes, int cuda_ordinal);
    ~CudaHostRegistration();

    CudaHostRegistration(const CudaHostRegistration&) = delete;
    CudaHostRegistration& operator=(const CudaHostRegistration&) = delete;
    CudaHostRegistration(CudaHostRegistration&& other) noexcept;
    CudaHostRegistration& operator=(CudaHostRegistration&& other) noexcept;

    [[nodiscard]] const CudaHostView& view() const noexcept { return view_; }

private:
    void release() noexcept;

    CudaHostView view_;
};

} // namespace ninfer
