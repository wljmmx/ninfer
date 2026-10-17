// port (route B / P1): Win32 implementation of the direct read queue.
// Upstream maps the queue onto io_uring (registered files + fixed buffers + eventfd wake).
// The Win32 mapping is an I/O completion port:
//   * registered files/buffers  -> handles associated with one completion port; the buffer pointer
//                                 is carried by a per-buffer OVERLAPPED, so no registration is
//                                 needed and buffers stay stable forever
//   * io_uring_submit           -> the ReadFile calls in submit()
//   * io_uring_peek_cqe          -> a non-blocking GetQueuedCompletionStatus + the harvested queue
//   * poll(ring_fd, eventfd)     -> one blocking GetQueuedCompletionStatus; wake() is a posted
//                                 completion packet with a reserved key
// Errors are reported as positive Win32 error codes (the Linux build reports positive errno).
#include "core/io/read_queue.h"

#ifndef NOMINMAX
#define NOMINMAX // std::numeric_limits<...>::max() must survive <windows.h>
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace ninfer::io {
namespace {

constexpr ULONG_PTR kWakeCompletionKey =
    static_cast<ULONG_PTR>(~static_cast<ULONG_PTR>(0)); // never a buffer index

struct OverlappedSlot {
    OVERLAPPED overlapped{}; // must stay first: the completion handler casts back from it
    std::uint32_t buffer_index = 0;
};

[[noreturn]] void fail(const char* operation) {
    throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), operation);
}

} // namespace

struct DirectReadQueue::Impl {
    struct Slot {
        DirectRead request{};
        bool busy = false;
    };

    HANDLE port            = nullptr;
    std::uint32_t capacity = 0;
    std::size_t pending    = 0;
    bool closed            = false;
    std::vector<FileView> files;
    std::vector<MemorySpan> buffers;
    std::vector<Slot> slots;
    std::vector<OverlappedSlot> overlapped; // one stable address per buffer
    std::vector<std::uint32_t> queued;      // admitted, not yet submitted
    std::vector<ReadCompletion> ready;      // completions harvested by wait()

    ~Impl() {
        if (port) CloseHandle(port);
    }

    void complete(std::uint32_t index, std::uint32_t bytes, int error) {
        auto& slot = slots[index];
        ready.push_back({slot.request.tag, index, bytes, error});
        slot.busy = false;
        if (pending) --pending;
    }
};

DirectReadQueue::DirectReadQueue(std::span<const FileView> files,
                                 std::span<const MemorySpan> buffers, std::uint32_t capacity)
    : impl_(std::make_unique<Impl>()) {
    if (!capacity || capacity > buffers.size() || files.empty() || buffers.size() > 65535)
        throw std::invalid_argument("invalid direct read queue capacity");
    auto& p    = *impl_;
    p.capacity = capacity;
    p.files.assign(files.begin(), files.end());
    p.buffers.assign(buffers.begin(), buffers.end());
    p.slots.resize(buffers.size());
    p.overlapped.resize(buffers.size());
    for (std::size_t i = 0; i < files.size(); ++i) {
        if (files[i].mode != ReadMode::Direct || files[i].descriptor < 0)
            throw std::invalid_argument("read queue requires direct files");
    }
    for (std::size_t i = 0; i < buffers.size(); ++i) {
        const auto buffer = buffers[i];
        validate_memory(buffer);
        if (buffer.device != Device::cpu() || !buffer.bytes ||
            buffer.bytes > std::numeric_limits<unsigned>::max())
            throw std::invalid_argument("read queue requires nonempty Host buffers");
        for (const auto file : files)
            validate_direct_read(file, 0, {static_cast<std::byte*>(buffer.data), buffer.bytes});
        const auto begin = reinterpret_cast<std::uintptr_t>(buffer.data);
        for (std::size_t j = 0; j < i; ++j) {
            const auto other = reinterpret_cast<std::uintptr_t>(buffers[j].data);
            if (begin < other + buffers[j].bytes && other < begin + buffer.bytes)
                throw std::invalid_argument("read queue buffers overlap");
        }
        p.overlapped[i].buffer_index = static_cast<std::uint32_t>(i);
    }
    p.port = CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, 0);
    if (!p.port) fail("create direct read completion port");
    for (const auto file : files) {
        const auto handle = reinterpret_cast<HANDLE>(file.descriptor);
        // port (sm_89): Win32 associates a *file object* with one completion port, so a second queue
        // over the same artifact cannot associate and this throws. That case needs a per-queue
        // handle opened from the path (the queue only sees a borrowed HANDLE), so it is a known
        // limitation rather than a silent fallback -- a fallback that polled its own OVERLAPPEDs
        // deadlocked the concurrent PLE row reader.
        if (!CreateIoCompletionPort(handle, p.port, 0, 0))
            fail("associate direct read file with the completion port");
    }
}

DirectReadQueue::~DirectReadQueue() {
    try {
        close();
        drain();
    } catch (...) {
        // Releasing borrowed buffers with unknown kernel writers would violate ownership.
        std::terminate();
    }
}

ReadAdmission DirectReadQueue::try_enqueue(DirectRead request) {
    auto& p = *impl_;
    if (p.closed) return ReadAdmission::Closed;
    if (p.pending == p.capacity) return ReadAdmission::Full;
    if (request.file_index >= p.files.size() || request.buffer_index >= p.buffers.size() ||
        !request.bytes)
        throw std::invalid_argument("invalid direct read request");
    auto& slot        = p.slots[request.buffer_index];
    const auto buffer = p.buffers[request.buffer_index];
    if (slot.busy || request.bytes > buffer.bytes)
        throw std::invalid_argument("direct read buffer is busy or too small");
    validate_direct_read(p.files[request.file_index], request.file_offset,
                         {static_cast<std::byte*>(buffer.data), request.bytes});
    slot.request = request;
    slot.busy    = true;
    ++p.pending;
    p.queued.push_back(request.buffer_index);
    return ReadAdmission::Accepted;
}

void DirectReadQueue::submit() {
    auto& p = *impl_;
    for (const auto index : p.queued) {
        auto& slot        = p.slots[index];
        const auto buffer = p.buffers[index];
        auto& overlapped  = p.overlapped[index];
        overlapped.overlapped.Offset =
            static_cast<DWORD>(slot.request.file_offset & 0xFFFFFFFFULL);
        overlapped.overlapped.OffsetHigh = static_cast<DWORD>(slot.request.file_offset >> 32);
        overlapped.overlapped.hEvent     = nullptr;
        const auto handle = reinterpret_cast<HANDLE>(p.files[slot.request.file_index].descriptor);
        const BOOL issued =
            ReadFile(handle, buffer.data, slot.request.bytes, nullptr, &overlapped.overlapped);
        if (!issued) {
            const auto error = GetLastError();
            // ERROR_IO_PENDING is the normal asynchronous path; anything else never starts, so it
            // is reported as that request's completion instead of throwing.
            if (error != ERROR_IO_PENDING) {
                p.ready.push_back({slot.request.tag, index, 0, static_cast<int>(error)});
                continue;
            }
        }

    }
    p.queued.clear();
}

std::size_t DirectReadQueue::poll(std::span<ReadCompletion> output) {
    auto& p           = *impl_;
    std::size_t count = 0;

    while (count < output.size() && !p.ready.empty()) {
        const auto staged = p.ready.front();
        p.ready.erase(p.ready.begin());
        output[count++] = staged;
        // port (sm_89) fix: a completion is only released here, so this is where the slot becomes
        // reusable and the in-flight count drops. wait() stages completions without releasing them,
        // otherwise a caller loop of `while (outstanding()) poll/wait` can exit before the copy.
        p.slots[staged.buffer_index].busy = false;
        --p.pending;
    }
    while (count < output.size()) {
        DWORD bytes       = 0;
        ULONG_PTR key     = 0;
        LPOVERLAPPED done = nullptr;
        const BOOL ok     = GetQueuedCompletionStatus(p.port, &bytes, &key, &done, 0);
        if (!done) break; // nothing queued (or a bare wake packet)
        if (!ok && GetLastError() == WAIT_TIMEOUT) break;
        auto* slot_overlapped = reinterpret_cast<OverlappedSlot*>(done);
        const auto index      = slot_overlapped->buffer_index;
        if (index >= p.slots.size() || !p.slots[index].busy)
            throw std::logic_error("unmatched direct read completion");
        output[count++] = {p.slots[index].request.tag, index,
                           ok ? static_cast<std::uint32_t>(bytes) : 0u,
                           ok ? 0 : static_cast<int>(GetLastError())};
        p.slots[index].busy = false;
        --p.pending;
    }
    return count;
}

ReadWake DirectReadQueue::wait() {
    auto& p = *impl_;
    for (;;) {
        DWORD bytes       = 0;
        ULONG_PTR key     = 0;
        LPOVERLAPPED done = nullptr;
        const BOOL ok     = GetQueuedCompletionStatus(p.port, &bytes, &key, &done, INFINITE);
        if (!done) {
            if (!ok) continue; // a spurious/terminated packet; keep waiting like EINTR does
            throw std::runtime_error("unexpected direct read wake");
        }
        if (key == kWakeCompletionKey) return ReadWake::Signalled;
        auto* slot_overlapped = reinterpret_cast<OverlappedSlot*>(done);
        const auto index      = slot_overlapped->buffer_index;
        if (index >= p.slots.size() || !p.slots[index].busy)
            throw std::logic_error("unmatched direct read completion");
        p.ready.push_back({p.slots[index].request.tag, index,
                           ok ? static_cast<std::uint32_t>(bytes) : 0u,
                           ok ? 0 : static_cast<int>(GetLastError())});
        // The slot stays busy and `pending` unchanged: the completion is released by the next
        // poll(), which is the call the consumer's loop is waiting to make.
        return ReadWake::Completion;
    }
}

void DirectReadQueue::wake() {
    auto& p = *impl_;
    if (!PostQueuedCompletionStatus(p.port, 0, kWakeCompletionKey, nullptr))
        fail("wake direct read queue");
}

void DirectReadQueue::close() {
    impl_->closed = true;
    submit();
}

void DirectReadQueue::drain() {
    submit();
    std::array<ReadCompletion, 128> completions;
    while (impl_->pending)
        if (!poll(completions)) (void)wait();
}

std::size_t DirectReadQueue::outstanding() const noexcept { return impl_->pending; }

} // namespace ninfer::io
