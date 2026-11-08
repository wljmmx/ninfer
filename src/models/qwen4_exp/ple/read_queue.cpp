#include "models/qwen4_exp/ple/read_queue.h"

#include "artifact/file_io.h"

#include <algorithm>
#include <cstdlib>
#include <new>
#include <stdexcept>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace ninfer::models::qwen4_exp::ple_detail {

ReadQueue::ReadQueue(const artifact::FileBacking& backing, std::span<std::byte> slab,
                     std::uint32_t depth, std::size_t read_bytes)
    : backing_(backing), slab_(slab), slots_(depth), read_bytes_(read_bytes) {
    if (!depth || !read_bytes || slab.size() < std::size_t(depth) * read_bytes)
        throw std::invalid_argument("invalid PLE read queue resources");
    for (auto& slot : slots_) slot.state = std::make_unique<artifact::DirectReadState>();
}

ReadQueue::~ReadQueue() = default;

ReadAdmission ReadQueue::try_enqueue(const ReadRequest& request) {
    std::lock_guard lock(mutex_);
    if (request.buffer >= slots_.size() || slots_[request.buffer].in_flight) return ReadAdmission::Full;
    pending_.push_back(request);
    return ReadAdmission::Accepted;
}

std::size_t ReadQueue::free_buffer() const {
    for (std::size_t i = 0; i < slots_.size(); ++i)
        if (!slots_[i].in_flight) return i;
    return slots_.size();
}

void ReadQueue::submit() {
    std::lock_guard lock(mutex_);
    submit_locked();
}

void ReadQueue::submit_locked() {
    while (!pending_.empty()) {
        const auto request = pending_.front();
        if (request.buffer >= slots_.size()) throw std::logic_error("PLE read buffer index");
        auto& slot = slots_[request.buffer];
        if (slot.in_flight) throw std::logic_error("PLE read buffer admission mismatch");
        pending_.pop_front();
        auto bytes = slab_.subspan(request.buffer * read_bytes_, read_bytes_);
        backing_.file(request.file).begin_direct_read(request.offset, bytes, *slot.state);
        slot.file      = request.file;
        slot.tag       = request.tag;
        slot.in_flight = true;
    }
}

std::size_t ReadQueue::poll(std::span<ReadCompletion> output) {
    std::size_t count = 0;
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        auto& slot = slots_[i];
        if (!slot.in_flight) continue;
        const auto& file = backing_.file(slot.file);
        if (!file.direct_read_ready(*slot.state)) continue;
        try {
            const auto bytes = file.complete_direct_read(*slot.state);
            output[count++]  = {i, bytes, slot.tag, {}};
        } catch (...) {
            output[count++] = {i, 0, slot.tag, std::current_exception()};
        }
        slot.in_flight = false;
        slot.tag       = 0;
    }
    return count;
}

void ReadQueue::wait() {
    std::size_t file_index = 0;
    artifact::DirectReadState* state = nullptr;
    {
        std::unique_lock lock(mutex_);
        std::size_t in_flight = 0;
        for (const auto& slot : slots_)
            if (slot.in_flight) ++in_flight;
        if (!in_flight) {
            if (!pending_.empty()) return;
            wake_.wait(lock, [&] { return woken_ || !pending_.empty(); });
            woken_ = false;
            return;
        }
        for (const auto& slot : slots_) {
            if (!slot.in_flight) continue;
            file_index = slot.file;
            state      = slot.state.get();
            break;
        }
    }
    // Joining one event is enough to make progress; poll() then retires every ready read.
    (void)backing_.file(file_index).wait_direct_read(*state, 0xffffffffU);
}

void ReadQueue::wake() {
    std::lock_guard lock(mutex_);
    woken_ = true;
    wake_.notify_all();
}

std::size_t ReadQueue::outstanding() const noexcept {
    std::lock_guard lock(mutex_);
    std::size_t count = 0;
    for (const auto& slot : slots_)
        if (slot.in_flight) ++count;
    return count;
}

AlignedSlab::AlignedSlab(std::size_t bytes, std::size_t alignment) : bytes_(bytes) {
    if (!bytes || alignment < alignof(std::max_align_t) || alignment & (alignment - 1))
        throw std::invalid_argument("invalid PLE staging slab geometry");
    const auto rounded = (bytes + alignment - 1) / alignment * alignment;
#if defined(_WIN32)
    data_ = static_cast<std::byte*>(_aligned_malloc(rounded, alignment));
#else
    data_ = static_cast<std::byte*>(std::aligned_alloc(alignment, rounded));
#endif
    if (data_ == nullptr) throw std::bad_alloc();
    bytes_ = rounded;
}

AlignedSlab::~AlignedSlab() {
#if defined(_WIN32)
    _aligned_free(data_);
#else
    std::free(data_);
#endif
}

} // namespace ninfer::models::qwen4_exp::ple_detail
