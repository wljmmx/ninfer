#pragma once

// One name for a Device transfer, so a reader can tell direction from the call site and so the
// error cannot be dropped by forgetting a macro. The body is the same cudaMemcpyAsync the tree
// already used; this header exists to make the DIRECTION explicit and checked, not to hide CUDA.
//
// Direction is an argument rather than a derived property on purpose: deriving it from pointer
// attributes would need a query per copy and would silently pick a default when the query fails -
// exactly the class of mistake the explicit kind prevents.

#include "core/device.h"

#include <cstddef>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace ninfer::core {

enum class TransferDirection : std::uint8_t {
    DeviceToDevice,
    HostToDevice,
    DeviceToHost,
};

[[nodiscard]] inline constexpr const char* transfer_direction_name(
    TransferDirection direction) noexcept {
    switch (direction) {
    case TransferDirection::DeviceToDevice: return "d2d";
    case TransferDirection::HostToDevice: return "h2d";
    case TransferDirection::DeviceToHost: return "d2h";
    }
    return "unknown";
}

[[nodiscard]] inline constexpr cudaMemcpyKind transfer_kind(TransferDirection direction) noexcept {
    switch (direction) {
    case TransferDirection::DeviceToDevice: return cudaMemcpyDeviceToDevice;
    case TransferDirection::HostToDevice: return cudaMemcpyHostToDevice;
    case TransferDirection::DeviceToHost: return cudaMemcpyDeviceToHost;
    }
    return cudaMemcpyDefault;
}

// A zero-byte copy is legal and cheap, but a null side is not: CUDA would report an invalid value
// far from the caller. Both are checked here, once.
inline void copy_bytes(void* destination, const void* source, std::size_t bytes,
                       TransferDirection direction, cudaStream_t stream) {
    if (bytes == 0) { return; }
    if (destination == nullptr || source == nullptr) {
        throw std::invalid_argument(std::string("transfer ") + transfer_direction_name(direction) +
                                    ": null operand for " + std::to_string(bytes) + " bytes");
    }
    CUDA_CHECK(cudaMemcpyAsync(destination, source, bytes, transfer_kind(direction), stream));
}

// The same transfer, with the outcome REPORTED instead of thrown. The diagnostic read-backs in the
// execution layer deliberately branch on failure - a probe that cannot run must not abort a
// generation - and spelled that by hand, as `if (cudaMemcpyAsync(...) == cudaSuccess)` or a cast to
// void. Same call and same direction argument; only the error shape differs. A null operand reports
// failure rather than throwing, and a zero-byte copy is a success that touches nothing, so both
// guards behave exactly like their thrown counterparts before CUDA is asked anything.
[[nodiscard]] inline bool try_copy_bytes(void* destination, const void* source, std::size_t bytes,
                                         TransferDirection direction,
                                         cudaStream_t stream) noexcept {
    if (bytes == 0) { return true; }
    if (destination == nullptr || source == nullptr) { return false; }
    return cudaMemcpyAsync(destination, source, bytes, transfer_kind(direction), stream) ==
           cudaSuccess;
}

} // namespace ninfer::core
