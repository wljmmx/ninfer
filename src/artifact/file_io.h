#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>

namespace ninfer::artifact {

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

private:
    std::filesystem::path path_;
#ifdef _WIN32
    void* file_               = nullptr;  // HANDLE, buffered positional reads
    mutable void* direct_file_ = nullptr; // HANDLE, FILE_FLAG_NO_BUFFERING positional reads
#else
    int fd_                = -1;
    mutable int direct_fd_ = -1;
#endif
    std::uint64_t bytes_   = 0;
};

} // namespace ninfer::artifact
