#pragma once

#include "artifact/backing.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace ninfer::models::qwen4_exp::ple_detail {

enum class ReadAdmission { Accepted, Full };

// One aligned page read: `buffer` indexes the staging slab supplied at construction.
struct ReadRequest {
    std::size_t file    = 0;
    std::size_t buffer  = 0;
    std::uint64_t offset = 0;
    std::uint64_t tag    = 0;
};

struct ReadCompletion {
    std::size_t buffer = 0;
    std::size_t bytes  = 0;
    std::uint64_t tag  = 0;
    std::exception_ptr error;
};

// port (sm_89): upstream uses io_uring through core/io/read_queue. Windows has no io_uring, so
// this queue keeps the same contract (io_depth aligned NO_BUFFERING reads in flight, retire by
// completion tag) on artifact::InputFile's overlapped + event pairs. Every request is a
// kPayloadAlignment-sized read from a kPayloadAlignment-aligned offset into an aligned buffer.
class ReadQueue {
public:
    ReadQueue(const artifact::FileBacking&, std::span<std::byte> slab, std::uint32_t depth,
              std::size_t read_bytes);
    ~ReadQueue();
    ReadQueue(const ReadQueue&)            = delete;
    ReadQueue& operator=(const ReadQueue&) = delete;

    [[nodiscard]] ReadAdmission try_enqueue(const ReadRequest&);
    void submit();
    [[nodiscard]] std::size_t poll(std::span<ReadCompletion>);
    // Blocks until at least one in-flight read retires, or until wake() with no outstanding read.
    void wait();
    void wake();
    [[nodiscard]] std::size_t outstanding() const noexcept;

private:
    struct Slot {
        std::unique_ptr<artifact::DirectReadState> state;
        std::size_t file  = 0;
        std::uint64_t tag = 0;
        bool in_flight    = false;
    };

    std::size_t free_buffer() const;
    void submit_locked();

    const artifact::FileBacking& backing_;
    std::span<std::byte> slab_;
    std::vector<Slot> slots_;
    std::size_t read_bytes_ = 0;
    std::deque<ReadRequest> pending_;
    mutable std::mutex mutex_;
    std::condition_variable wake_;
    bool woken_ = false;
};

// Page-aligned staging storage for direct reads. The slab outlives every queued read.
class AlignedSlab {
public:
    AlignedSlab(std::size_t bytes, std::size_t alignment);
    ~AlignedSlab();
    AlignedSlab(const AlignedSlab&)            = delete;
    AlignedSlab& operator=(const AlignedSlab&) = delete;

    [[nodiscard]] std::byte* data() noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return bytes_; }

private:
    std::byte* data_  = nullptr;
    std::size_t bytes_ = 0;
};

} // namespace ninfer::models::qwen4_exp::ple_detail
