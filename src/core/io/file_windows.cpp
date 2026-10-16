// port (route B / P1): Win32 implementation of the direct/buffered read-only file.
// Same header and semantics as upstream's Linux file.cpp; that file is left untouched so upstream
// merges stay mechanical. Differences that matter:
//   * O_DIRECT                -> FILE_FLAG_NO_BUFFERING (offset/size/address alignment required)
//   * statx(STATX_DIOALIGN)   -> the volume's bytes-per-sector from GetDiskFreeSpaceW
//   * pread/pread64           -> ReadFile with an OVERLAPPED offset (the handle is always
//                                FILE_FLAG_OVERLAPPED so both paths use the same call)
#include "core/io/file.h"

#ifndef NOMINMAX
#define NOMINMAX // std::numeric_limits<...>::max() must survive <windows.h>
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <algorithm>
#include <bit>
#include <limits>
#include <memory>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace ninfer::io {
namespace {

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation,
                       DWORD error = GetLastError()) {
    throw std::system_error(static_cast<int>(error), std::system_category(),
                            path.string() + ": " + operation);
}

HANDLE handle_of(std::intptr_t descriptor) { return reinterpret_cast<HANDLE>(descriptor); }

// One synchronous read at an absolute offset through the same overlapped machinery the async path
// uses, because the handle carries FILE_FLAG_OVERLAPPED.
std::size_t read_at(std::intptr_t descriptor, std::uint64_t offset, std::byte* data,
                    std::size_t bytes) {
    if (!bytes) return 0;
    OVERLAPPED overlapped{};
    overlapped.Offset     = static_cast<DWORD>(offset & 0xFFFFFFFFULL);
    overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32);
    DWORD transferred = 0;
    const BOOL ok = ReadFile(handle_of(descriptor), data, static_cast<DWORD>(bytes), nullptr,
                             &overlapped);
    if (!ok) {
        const auto error = GetLastError();
        if (error != ERROR_IO_PENDING && !GetOverlappedResult(handle_of(descriptor), &overlapped,
                                                              &transferred, TRUE))
            throw std::system_error(static_cast<int>(error), std::system_category(),
                                    "overlapped read");
    } else if (!GetOverlappedResult(handle_of(descriptor), &overlapped, &transferred, TRUE)) {
        throw std::system_error(static_cast<int>(GetLastError()), std::system_category(),
                                "overlapped read result");
    }
    return static_cast<std::size_t>(transferred);
}

std::uint32_t volume_sector_bytes(const std::filesystem::path& path) {
    std::error_code code;
    auto root = path.parent_path();
    while (!root.empty() && !std::filesystem::exists(root, code)) root = root.parent_path();
    if (root.empty()) root = std::filesystem::temp_directory_path(code);
    std::wstring volume = root.wstring();
    if (volume.size() < 2 || volume[1] != L':') return 4096;
    volume = volume.substr(0, 2) + L"\\";
    DWORD sectors_per_cluster = 0, bytes_per_sector = 0, free_clusters = 0, total_clusters = 0;
    if (!GetDiskFreeSpaceW(volume.c_str(), &sectors_per_cluster, &bytes_per_sector, &free_clusters,
                           &total_clusters))
        return 4096;
    const auto sector = static_cast<std::uint32_t>(bytes_per_sector);
    // Unbuffered I/O on this platform requires the sector size, and the engine needs a power of two.
    return std::has_single_bit(sector) ? sector : 4096;
}

} // namespace

ReadOnlyFile::ReadOnlyFile(const std::filesystem::path& path, ReadMode mode) : path_(path) {
    if (mode != ReadMode::Buffered && mode != ReadMode::Direct)
        throw std::invalid_argument("invalid file read mode");
    const DWORD flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS |
                        (mode == ReadMode::Direct ? FILE_FLAG_NO_BUFFERING : 0);
    const HANDLE handle = CreateFileW(path.wstring().c_str(), GENERIC_READ,
                                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                      nullptr, OPEN_EXISTING, flags, nullptr);
    if (handle == INVALID_HANDLE_VALUE) fail(path, "open");
    view_.descriptor = reinterpret_cast<std::intptr_t>(handle);
    view_.mode       = mode;
    try {
        if (GetFileType(handle) != FILE_TYPE_DISK) fail(path, "expected regular file", ERROR_INVALID_PARAMETER);
        LARGE_INTEGER size{};
        if (!GetFileSizeEx(handle, &size) || size.QuadPart < 0)
            fail(path, "GetFileSizeEx");
        view_.bytes = static_cast<std::uint64_t>(size.QuadPart);
        if (mode == ReadMode::Direct) {
            const auto sector      = volume_sector_bytes(path);
            view_.memory_alignment = sector;
            view_.offset_alignment = sector;
            if (!std::has_single_bit(view_.memory_alignment) ||
                !std::has_single_bit(view_.offset_alignment))
                fail(path, "unsupported direct I/O alignment", ERROR_NOT_SUPPORTED);
        }
    } catch (...) {
        CloseHandle(handle);
        view_.descriptor = -1;
        throw;
    }
}

ReadOnlyFile::~ReadOnlyFile() {
    if (view_.descriptor >= 0) CloseHandle(handle_of(view_.descriptor));
}

ReadOnlyFile::ReadOnlyFile(ReadOnlyFile&& other) noexcept
    : path_(std::move(other.path_)), view_(std::exchange(other.view_, {})) {}

ReadOnlyFile& ReadOnlyFile::operator=(ReadOnlyFile&& other) noexcept {
    if (this != &other) {
        if (view_.descriptor >= 0) CloseHandle(handle_of(view_.descriptor));
        path_ = std::move(other.path_);
        view_ = std::exchange(other.view_, {});
    }
    return *this;
}

void ReadOnlyFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (view_.mode != ReadMode::Buffered || view_.descriptor < 0)
        throw std::invalid_argument("read_exact requires a buffered file");
    if (offset > view_.bytes || destination.size() > view_.bytes - offset)
        fail(path_, "read exceeds file length", ERROR_HANDLE_EOF);
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024);
        std::size_t n;
        try {
            n = read_at(view_.descriptor, offset, destination.data(), count);
        } catch (const std::system_error& error) {
            throw std::system_error(error.code().value(), std::system_category(),
                                    path_.string() + ": read_exact");
        }
        if (!n) fail(path_, "unexpected EOF", ERROR_HANDLE_EOF);
        offset += static_cast<std::uint64_t>(n);
        destination = destination.subspan(n);
    }
}

void validate_direct_read(FileView file, std::uint64_t offset, std::span<std::byte> destination) {
    if (file.descriptor < 0 || file.mode != ReadMode::Direct ||
        !std::has_single_bit(file.memory_alignment) ||
        !std::has_single_bit(file.offset_alignment) || offset % file.offset_alignment ||
        destination.size() % file.offset_alignment ||
        (!destination.empty() &&
         (!destination.data() ||
          reinterpret_cast<std::uintptr_t>(destination.data()) % file.memory_alignment)) ||
        destination.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
        throw std::invalid_argument("invalid direct read range or alignment");
}

std::size_t ReadOnlyFile::read_direct(std::uint64_t offset,
                                      std::span<std::byte> destination) const {
    validate_direct_read(view_, offset, destination);
    if (destination.empty()) return 0;
    try {
        return read_at(view_.descriptor, offset, destination.data(), destination.size());
    } catch (const std::system_error& error) {
        throw std::system_error(error.code().value(), std::system_category(),
                                path_.string() + ": direct read");
    }
}

DirectReadWindow direct_read_window(FileView file, std::uint64_t offset, std::size_t bytes) {
    if (file.descriptor < 0 || file.mode != ReadMode::Direct || !bytes ||
        !std::has_single_bit(file.offset_alignment) || offset > file.bytes ||
        bytes > file.bytes - offset)
        throw std::invalid_argument("invalid direct read window");
    const auto alignment = file.offset_alignment;
    const auto prefix    = offset % alignment;
    if (bytes > std::numeric_limits<std::size_t>::max() - prefix - (alignment - 1))
        throw std::overflow_error("direct read window size overflow");
    const auto required = bytes + prefix;
    const auto count    = (required + alignment - 1) / alignment * alignment;
    return {offset - prefix, count, prefix, required};
}

} // namespace ninfer::io
