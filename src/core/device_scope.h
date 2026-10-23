#pragma once

#include "core/device.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace ninfer {

// port (sm_89): binds one device for the lifetime of a scope and restores whatever was current
// before. The expert pool and the VMM reservation call the driver API, which does not take a device
// from the caller, so they have to make the device current themselves.
class DeviceScope {
public:
    explicit DeviceScope(Device device) : device_(device) {
        if (cudaGetDevice(&previous_) != cudaSuccess) { previous_ = -1; }
        const cudaError_t err = cudaSetDevice(device_.index());
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("device scope: cudaSetDevice failed: ") +
                                     cudaGetErrorName(err));
        }
    }
    ~DeviceScope() {
        if (previous_ >= 0) { (void)cudaSetDevice(previous_); }
    }

    DeviceScope(const DeviceScope&)            = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;
    DeviceScope(DeviceScope&&)                 = delete;
    DeviceScope& operator=(DeviceScope&&)      = delete;

private:
    Device device_{};
    int previous_ = -1;
};

} // namespace ninfer
