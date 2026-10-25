#include "artifact/file_io.h"

#include "artifact/framing.h"
#include "artifact/schema.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#include <winioctl.h>
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

bool volume_incurs_seek_penalty(const std::filesystem::path& path) {
#ifdef _WIN32
    const auto root = path.root_name();
    if (root.empty()) { return true; }
    const std::wstring volume = L"\\\\.\\" + root.wstring();
    HANDLE handle = ::CreateFileW(volume.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                                  OPEN_EXISTING, 0, nullptr);
    if (handle == INVALID_HANDLE_VALUE) { return true; }
    STORAGE_PROPERTY_QUERY query{};
    query.PropertyId = StorageDeviceSeekPenaltyProperty;
    query.QueryType  = PropertyStandardQuery;
    DEVICE_SEEK_PENALTY_DESCRIPTOR descriptor{};
    DWORD returned = 0;
    const BOOL ok  = ::DeviceIoControl(handle, IOCTL_STORAGE_QUERY_PROPERTY, &query, sizeof(query),
                                       &descriptor, sizeof(descriptor), &returned, nullptr);
    ::CloseHandle(handle);
    if (!ok) { return true; }
    return descriptor.IncursSeekPenalty != FALSE;
#else
    (void)path;
    return false;
#endif
}

bool artifact_fits_in_page_cache(std::uint64_t bytes) {
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (bytes == 0 || ::GlobalMemoryStatusEx(&status) == 0) { return false; }
    // Leave half the machine to the process itself: weights, KV, the pinned host arena and whatever
    // else is running. 22.8 GB of 64 GB qualifies; a 118 GB model on the same box does not.
    return bytes <= status.ullTotalPhys / 2;
#else
    (void)bytes;
    return false;
#endif
}

ReadMode expert_read_mode(const std::filesystem::path& path, std::uint64_t artifact_bytes) {
    const bool seeks = volume_incurs_seek_penalty(path);
    if (const char* value = std::getenv("NINFER_EXPERT_IO")) {
        if (std::strcmp(value, "buffered") == 0) { return ReadMode::Buffered; }
        if (std::strcmp(value, "direct") == 0) {
            return seeks ? ReadMode::Buffered : ReadMode::Direct;
        }
    }
    if (seeks) { return ReadMode::Buffered; }
    // port (sm_89): a model that fits in RAM should be read through the page cache. An expert the
    // split path reads again on the next step then costs a memcpy (0.2 ms per 1.8 MiB) instead of a
    // physical read (8.6 ms) - measured 14 ms against 560-830 ms of ensure per token on the 35B
    // artifact. Reading around the cache only pays when the working set cannot be held, which is the
    // 118 GB model on a 64 GB box; that case keeps the uncached path.
    if (artifact_fits_in_page_cache(artifact_bytes)) { return ReadMode::Buffered; }
    return ReadMode::Direct;
}

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
    if (view_ != nullptr) { ::UnmapViewOfFile(view_); }
    if (mapping_ != nullptr) { ::CloseHandle(static_cast<HANDLE>(mapping_)); }
    if (direct_file_ != nullptr) { ::CloseHandle(static_cast<HANDLE>(direct_file_)); }
    if (file_ != nullptr) { ::CloseHandle(static_cast<HANDLE>(file_)); }
#else
    if (direct_fd_ >= 0) { ::close(direct_fd_); }
    if (fd_ >= 0) { ::close(fd_); }
#endif
}

const std::byte* InputFile::map_view() const {
#ifdef _WIN32
    if (view_ == nullptr && mapping_ == nullptr && file_ != nullptr && bytes_ > 0) {
        // Read-only, whole file, demand-paged: nothing is read until a slot is actually touched,
        // which is what makes this cheap next to copying every expert into the arena.
        mapping_ = ::CreateFileMappingW(static_cast<HANDLE>(file_), nullptr, PAGE_READONLY, 0, 0,
                                        nullptr);
        if (mapping_ != nullptr) {
            view_ = static_cast<const std::byte*>(
                ::MapViewOfFile(static_cast<HANDLE>(mapping_), FILE_MAP_READ, 0, 0, 0));
        }
    }
    return view_;
#else
    return nullptr;
#endif
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
#ifdef _WIN32
    // The buffered handle is shared by every caller, and SetFilePointerEx followed by ReadFile
    // is not atomic: without this guard two threads interleave and read each other's offsets,
    // which shows up as randomly corrupted payloads (a tokenizer that fails to parse at a
    // different line every run).
    const std::lock_guard<std::mutex> guard(read_mutex_);
    std::size_t total = 0;
    while (total < destination.size()) {
        const auto amount = static_cast<DWORD>(
            std::min<std::size_t>(destination.size() - total, 64ULL * 1024 * 1024));
        // Synchronous read: this handle is not opened overlapped (ReadFile rejects a non-null
        // OVERLAPPED on such a handle), and the device path never reaches this code because it
        // uploads through direct reads.
        LARGE_INTEGER position{};
        position.QuadPart = static_cast<LONGLONG>(offset + total);
        if (!::SetFilePointerEx(static_cast<HANDLE>(file_), position, nullptr, FILE_BEGIN)) {
            fail(path_, "seek");
        }
        DWORD read = 0;
        if (!::ReadFile(static_cast<HANDLE>(file_), destination.data() + total, amount, &read,
                        nullptr)) {
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

#ifdef _WIN32
DirectReadState::DirectReadState() {
    op_event_ = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
    op_       = ::HeapAlloc(::GetProcessHeap(), HEAP_ZERO_MEMORY, sizeof(OVERLAPPED));
    if (op_event_ == nullptr || op_ == nullptr) { fail(std::filesystem::path(), "alloc read state"); }
}

DirectReadState::~DirectReadState() {
    if (issued_ && file_ != nullptr) {
        // Never abandon an in-flight read: join it (the destination buffer is
        // guaranteed to outlive the state) so the kernel stops touching it.
        DWORD received = 0;
        (void)::GetOverlappedResult(static_cast<HANDLE>(file_), static_cast<OVERLAPPED*>(op_),
                                   &received, TRUE);
        issued_ = false;
    }
    if (op_ != nullptr) { ::HeapFree(::GetProcessHeap(), 0, op_); }
    if (op_event_ != nullptr) { ::CloseHandle(static_cast<HANDLE>(op_event_)); }
}
#else
DirectReadState::DirectReadState()  = default;
DirectReadState::~DirectReadState() = default;
#endif

void InputFile::begin_direct_read(std::uint64_t offset, std::span<std::byte> destination,
                                  DirectReadState& state) const {
#ifdef _WIN32
    if (offset % kPayloadAlignment || destination.size() % kPayloadAlignment ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment) {
        throw ArtifactError(path_.string() + ": unaligned async direct read");
    }
    if (destination.empty() || destination.size() > (1ULL << 30)) {
        throw ArtifactError(path_.string() + ": invalid async direct read size");
    }
    if (state.issued_) { throw ArtifactError(path_.string() + ": read state already in flight"); }
    if (direct_file_ == nullptr) {
        HANDLE direct = ::CreateFileW(path_.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                       OPEN_EXISTING,
                                       FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING |
                                           FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN,
                                       nullptr);
        if (direct == INVALID_HANDLE_VALUE) { fail(path_, "open direct"); }
        direct_file_ = direct;
    }
    OVERLAPPED* op   = static_cast<OVERLAPPED*>(state.op_);
    HANDLE event     = static_cast<HANDLE>(state.op_event_);
    *op              = OVERLAPPED{};
    overlapped_offset(*op, offset);
    op->hEvent       = event;
    state.file_      = direct_file_;
    if (!::ResetEvent(event)) { fail(path_, "reset read event"); }
    DWORD read = 0;
    state.inline_completed_ = false;
    state.inline_bytes_     = 0;
    const BOOL started      = ::ReadFile(static_cast<HANDLE>(direct_file_), destination.data(),
                                         static_cast<DWORD>(destination.size()), &read, op);
    if (started) {
        // Completed synchronously (rare for NO_BUFFERING).
        state.inline_completed_ = true;
        state.inline_bytes_     = read;
        return;
    }
    const auto error = ::GetLastError();
    if (error == ERROR_IO_PENDING) {
        state.issued_ = true;
        return;
    }
    if (error == ERROR_HANDLE_EOF) {
        state.inline_completed_ = true;
        state.inline_bytes_     = 0;
        return;
    }
    ::SetLastError(static_cast<DWORD>(error));
    fail(path_, "begin direct read");
#else
    state.offset_     = offset;
    state.destination_ = destination.data();
    state.requested_  = destination.size();
#endif
}

std::size_t InputFile::complete_direct_read(DirectReadState& state) const {
#ifdef _WIN32
    if (state.inline_completed_) {
        state.inline_completed_ = false;
        return state.inline_bytes_;
    }
    if (!state.issued_) { throw ArtifactError(path_.string() + ": no direct read in flight"); }
    OVERLAPPED* op      = static_cast<OVERLAPPED*>(state.op_);
    DWORD received      = 0;
    const BOOL joined   = ::GetOverlappedResult(static_cast<HANDLE>(direct_file_), op, &received, TRUE);
    state.issued_       = false;
    if (!joined) {
        const auto error = ::GetLastError();
        if (error == ERROR_HANDLE_EOF) { return 0; }
        ::SetLastError(static_cast<DWORD>(error));
        fail(path_, "complete direct read");
    }
    return received;
#else
    return read_direct(state.offset_, {state.destination_, state.requested_});
#endif
}

} // namespace ninfer::artifact
