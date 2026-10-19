#pragma once

#include "core/io/file.h"

#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::io {

// port (sm_89): reads an arbitrary range from a Direct-mode file.
//
// Direct I/O requires the destination, the file offset and the length to be aligned to the
// volume sector. Expert chunks sit at 1 MiB strides, so most reads hit the fast path; the rest
// go through one aligned window plus a copy, which still avoids the system cache.
inline void read_direct_into(const ReadOnlyFile& file, std::uint64_t offset,
                             std::span<std::byte> destination) {
    if (destination.empty()) { return; }
    const auto view      = file.view();
    const auto alignment = view.memory_alignment ? view.memory_alignment : std::size_t{1};
    const auto window    = direct_read_window(view, offset, destination.size());
    const auto address   = reinterpret_cast<std::uintptr_t>(destination.data());
    const bool direct_ok =
        window.prefix == 0 && window.bytes == destination.size() && address % alignment == 0;
    if (direct_ok) {
        const auto received = file.read_direct(window.offset, destination);
        if (received < destination.size()) { throw std::runtime_error("short direct read"); }
        return;
    }
    thread_local std::vector<std::byte> scratch;
    if (scratch.size() < window.bytes + alignment) { scratch.resize(window.bytes + alignment); }
    const auto base_address =
        (reinterpret_cast<std::uintptr_t>(scratch.data()) + alignment - 1) / alignment * alignment;
    auto* base          = reinterpret_cast<std::byte*>(base_address);
    const auto received = file.read_direct(window.offset, {base, window.bytes});
    if (received < window.required) { throw std::runtime_error("short direct read"); }
    std::memcpy(destination.data(), base + window.prefix, destination.size());
}

// Reads bytes from either kind of file.
inline void read_range(const ReadOnlyFile& file, std::uint64_t offset,
                       std::span<std::byte> destination) {
    if (file.view().mode == ReadMode::Direct) {
        read_direct_into(file, offset, destination);
    } else {
        file.read_exact(offset, destination);
    }
}

} // namespace ninfer::io
