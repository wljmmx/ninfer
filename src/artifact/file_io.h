#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <span>

namespace ninfer::artifact {

// port (sm_89): the artifact is plane-major, so one expert's codes and scales are hundreds of MiB
// apart and reading a single expert costs several seeks. Buffered I/O lets the OS read-ahead hide
// that on a disk; unbuffered I/O only wins where seeks are cheap. Measured on this host: HDD 2.3
// tok/s unbuffered against 61.4 buffered, NVMe 55.8 against 39.3.
enum class ReadMode : std::uint8_t { Buffered, Direct };

// True when the volume behind `path` reports a seek penalty (rotating media). A failed query is
// treated as "no penalty" only for a path without a local volume name; otherwise the conservative
// buffered default wins.
[[nodiscard]] bool volume_incurs_seek_penalty(const std::filesystem::path& path);

// NINFER_EXPERT_IO=buffered|direct overrides; "direct" is still demoted to buffered on a volume
// with a seek penalty, because unbuffered I/O cannot benefit from read-ahead there.
// `artifact_bytes` is the whole model: a model that fits in RAM is worth caching, one that does not
// gains nothing from the page cache and would evict everything else.
[[nodiscard]] ReadMode expert_read_mode(const std::filesystem::path& path,
                                        std::uint64_t artifact_bytes = 0);

// State for one in-flight aligned direct read. Windows keeps a OVERLAPPED op
// with its own manual-reset event so several reads can be issued concurrently
// on the same FILE_FLAG_OVERLAPPED handle (NVMe queue depth); POSIX has no
// equivalent here and completes synchronously inside complete_direct_read.
class DirectReadState {
public:
    DirectReadState();
    ~DirectReadState();
    DirectReadState(const DirectReadState&)            = delete;
    DirectReadState& operator=(const DirectReadState&) = delete;
    DirectReadState(DirectReadState&&)                 = delete;
    DirectReadState& operator=(DirectReadState&&)     = delete;

private:
    friend class InputFile;
#ifdef _WIN32
    void* op_event_ = nullptr;  // HANDLE, manual reset
    void* op_       = nullptr;  // OVERLAPPED, heap-stable across the call
    void* file_     = nullptr;  // HANDLE captured at begin for abandoned joins
    bool issued_    = false;
    bool inline_completed_ = false;
    std::uint32_t inline_bytes_ = 0;
#else
    std::uint64_t offset_    = 0;
    std::byte* destination_ = nullptr;
    std::size_t requested_  = 0;
#endif
};

// Direct reads require aligned offsets and buffers. A short final direct block is allowed;
// read_exact always requires the complete requested byte range.
class InputFile {
public:
    explicit InputFile(std::filesystem::path path);
    ~InputFile();
    InputFile(const InputFile&)            = delete;
    InputFile& operator=(const InputFile&) = delete;

    [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }

    void read_exact(std::uint64_t offset, std::span<std::byte> destination) const;
    [[nodiscard]] std::size_t read_direct(std::uint64_t offset,
                                          std::span<std::byte> destination) const;

    // Async pair: begin_direct_read issues one aligned NO_BUFFERING read without
    // blocking (Windows) and returns immediately; complete_direct_read joins it.
    // A single state object may host one read at a time. begin_direct_read may
    // complete inline (short EOF); complete_direct_read then returns that count.
    void begin_direct_read(std::uint64_t offset, std::span<std::byte> destination,
                           DirectReadState& state) const;
    [[nodiscard]] std::size_t complete_direct_read(DirectReadState& state) const;

    // port (sm_89): read-only view of the whole object, created on first use. A bank that lives in
    // one object can then be addressed where it already is instead of being copied into an arena
    // slot - the copy is what made the host tier pay 20 ms per layer. Null when the mapping fails,
    // which keeps the copying path as the fallback.
    [[nodiscard]] const std::byte* map_view() const;

private:
    std::filesystem::path path_;
    // Buffered positional reads move the kernel file pointer, so concurrent read_exact callers
    // must not interleave their seek/read pairs.
    mutable std::mutex read_mutex_;
#ifdef _WIN32
    void* file_               = nullptr;  // HANDLE, buffered positional reads
    mutable void* direct_file_ = nullptr; // HANDLE, FILE_FLAG_NO_BUFFERING positional reads
#else
    int fd_                = -1;
    mutable int direct_fd_ = -1;
#endif
    std::uint64_t bytes_   = 0;
    mutable void* mapping_      = nullptr; // HANDLE, read-only file mapping
    mutable const std::byte* view_ = nullptr; // MapViewOfFile base
};

} // namespace ninfer::artifact
