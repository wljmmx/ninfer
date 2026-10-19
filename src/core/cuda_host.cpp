#include "core/cuda_host.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer {
namespace {

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

} // namespace

CudaHostRegistration::CudaHostRegistration(void* host, std::size_t bytes, int cuda_ordinal) {
    if (host == nullptr || bytes == 0 || cuda_ordinal < 0) {
        throw std::invalid_argument("pinned registration requires a Host range and a CUDA device");
    }
    check(cudaSetDevice(cuda_ordinal), "cudaSetDevice");
    check(cudaHostRegister(host, bytes, cudaHostRegisterPortable), "cudaHostRegister");
    void* device_address = nullptr;
    const auto status    = cudaHostGetDevicePointer(&device_address, host, 0);
    if (status != cudaSuccess) {
        (void)cudaHostUnregister(host);
        check(status, "cudaHostGetDevicePointer");
    }
    view_ = {host, bytes, device_address, cuda_ordinal};
}

CudaHostRegistration::~CudaHostRegistration() { release(); }

CudaHostRegistration::CudaHostRegistration(CudaHostRegistration&& other) noexcept
    : view_(std::exchange(other.view_, CudaHostView{})) {}

CudaHostRegistration& CudaHostRegistration::operator=(CudaHostRegistration&& other) noexcept {
    if (this != &other) {
        release();
        view_ = std::exchange(other.view_, CudaHostView{});
    }
    return *this;
}

void CudaHostRegistration::release() noexcept {
    if (view_.host != nullptr) { (void)cudaHostUnregister(view_.host); }
    view_ = CudaHostView{};
}

} // namespace ninfer

