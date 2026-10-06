#include "artifact/file_io.h"

#include "artifact/framing.h"
#include "artifact/schema.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ninfer::artifact {
namespace {

#ifdef _WIN32

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
    throw ArtifactError(path.string() + ": " + operation + ": error " +
                        std::to_string(static_cast<int>(::GetLastError())));
}

void overlapped_offset(OVERLAPPED& operation, std::uint64_t offset) {
    operation.Offset     = static_cast<DWORD>(offset & 0xffffffffULL);
    operation.OffsetHigh = static_cast<DWORD>(offset >> 32U);
}

#else

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
    throw ArtifactError(path.string() + ": " + operation + ": " + std::strerror(errno));
}

off_t file_offset(std::uint64_t offset) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        throw ArtifactError("file offset exceeds positional I/O range");
    }
    return static_cast<off_t>(offset);
}

#endif

} // namespace

InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
#ifdef _WIN32
    HANDLE file = ::CreateFileW(path_.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) { fail(path_, "open"); }
    file_ = file;

    LARGE_INTEGER size{};
    if (!::GetFileSizeEx(file, &size) || size.QuadPart < 0) {
        const auto error = static_cast<int>(::GetLastError());
        ::CloseHandle(file);
        file_ = nullptr;
        ::SetLastError(static_cast<DWORD>(error));
        fail(path_, "size");
    }
    bytes_ = static_cast<std::uint64_t>(size.QuadPart);
#else
    fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) { fail(path_, "open"); }

    struct stat status {};

    if (::fstat(fd_, &status) != 0) {
        const auto error = errno;
        ::close(fd_);
        fd_   = -1;
        errno = error;
        fail(path_, "fstat");
    }
    if (status.st_size < 0 || !S_ISREG(status.st_mode)) {
        ::close(fd_);
        fd_ = -1;
        throw ArtifactError(path_.string() + ": expected a regular file");
    }
    bytes_ = static_cast<std::uint64_t>(status.st_size);
#endif
}

InputFile::~InputFile() {
#ifdef _WIN32
    if (direct_file_ != nullptr) { ::CloseHandle(static_cast<HANDLE>(direct_file_)); }
    if (file_ != nullptr) { ::CloseHandle(static_cast<HANDLE>(file_)); }
#else
    if (direct_fd_ >= 0) { ::close(direct_fd_); }
    if (fd_ >= 0) { ::close(fd_); }
#endif
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
#ifdef _WIN32
    std::size_t total = 0;
    while (total < destination.size()) {
        const auto amount = static_cast<DWORD>(
            std::min<std::size_t>(destination.size() - total, 64ULL * 1024 * 1024));
        OVERLAPPED operation{};
        overlapped_offset(operation, offset + total);
        DWORD read = 0;
        if (!::ReadFile(static_cast<HANDLE>(file_), destination.data() + total, amount, &read,
                        &operation)) {
            const auto error = static_cast<int>(::GetLastError());
            ::SetLastError(static_cast<DWORD>(error));
            fail(path_, "read");
        }
        if (read == 0) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        total += read;
    }
#else
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024);
        const auto read  = ::pread(fd_, destination.data(), count, file_offset(offset));
        if (read < 0) {
            if (errno == EINTR) { continue; }
            fail(path_, "pread");
        }
        if (!read) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        offset += static_cast<std::uint64_t>(read);
        destination = destination.subspan(static_cast<std::size_t>(read));
    }
#endif
}

std::size_t InputFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset % kPayloadAlignment || destination.size() % kPayloadAlignment ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment ||
        destination.size() > static_cast<std::size_t>(std::numeric_limits<std::size_t>::max())) {
        throw ArtifactError(path_.string() + ": unaligned or oversized direct read");
    }
    if (destination.empty()) { return 0; }
#ifdef _WIN32
    if (direct_file_ == nullptr) {
        HANDLE direct = ::CreateFileW(path_.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                      OPEN_EXISTING,
                                      FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING |
                                          FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN,
                                      nullptr);
        if (direct == INVALID_HANDLE_VALUE) { fail(path_, "open direct"); }
        direct_file_ = direct;
    }
    std::size_t total = 0;
    while (total < destination.size()) {
        constexpr std::size_t max_read = 1ULL << 30;
        const auto amount = static_cast<DWORD>(std::min(max_read, destination.size() - total));
        OVERLAPPED operation{};
        overlapped_offset(operation, offset + total);
        DWORD read = 0;
        const BOOL started = ::ReadFile(static_cast<HANDLE>(direct_file_), destination.data() + total,
                                        amount, &read, &operation);
        if (!started) {
            const auto error = ::GetLastError();
            if (error == ERROR_HANDLE_EOF) { break; }
            if (error != ERROR_IO_PENDING ||
                !::GetOverlappedResult(static_cast<HANDLE>(direct_file_), &operation, &read,
                                       TRUE)) {
                const auto final_error = error == ERROR_IO_PENDING ? ::GetLastError() : error;
                ::SetLastError(static_cast<DWORD>(final_error));
                fail(path_, "direct read");
            }
        }
        total += read;
        if (read != amount) { break; }
    }
    return total;
#else
    if (direct_fd_ < 0) {
        direct_fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (direct_fd_ < 0) { fail(path_, "open direct"); }
    }
    ssize_t read;
    do {
        read = ::pread(direct_fd_, destination.data(), destination.size(), file_offset(offset));
    } while (read < 0 && errno == EINTR);
    if (read < 0) { fail(path_, "direct pread"); }
    return static_cast<std::size_t>(read);
#endif
}

} // namespace ninfer::artifact
