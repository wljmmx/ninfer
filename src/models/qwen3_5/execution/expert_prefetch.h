#pragma once

// port (sm_89): the process-scoped read-ahead queue (plan D1, infrastructure items 1 and 2).
//
// Why one queue for the whole process: the fill is per bank, and there are forty banks, so a queue
// owned by ExpertStream would be forty queues and forty worker threads all reading the same disk.
// The queue therefore takes the bank as part of the job, and the residency as a weak reference.
//
// Why the reference is weak: the worker outlives any single request but must never outlive the
// model. A job holds the residency only for the length of one fill, and a fill issued just before
// the model is destroyed finds an expired reference instead of a dangling pointer. That is the
// correct form of the fix - a "still alive" flag would be a race with no lock able to close it.
//
// The worker itself only exists while a residency is attached: retain() creates it, release() stops
// and joins it. That ordering is what guarantees the pinned arena is never freed under a running
// fill, and it is why the release happens in ModelInstance's destructor body rather than in a
// static destructor at exit.

#include "core/prefetch.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>

namespace ninfer::models::qwen3_5::execution {

class HostExpertResidency;

class ExpertPrefetchQueue {
public:
    struct Stats {
        core::PrefetchStats queue;
        std::uint64_t ran     = 0; // jobs the worker executed
        std::uint64_t expired = 0; // jobs whose residency was already gone
        std::size_t pending   = 0; // running + waiting, against NINFER_PREFETCH_PENDING
        std::size_t max_pending = 0;
    };

    static ExpertPrefetchQueue& instance();

    // Attach/detach a residency. Counted, because the engine may build more than one model in a
    // process; the worker runs while at least one is attached.
    void retain();
    void release();

    // Offers one read-ahead. Returns false when the worker is gone or the budget refused it; the
    // caller is never expected to fall back to a synchronous fill - a read-ahead that did not fit
    // is simply not done.
    bool enqueue(std::weak_ptr<HostExpertResidency> host, std::uint32_t bank,
                 std::span<const std::int32_t> experts, std::size_t bytes, std::uint64_t due,
                 std::uint64_t now);

    // D4: the consumer reports the bytes a prefetched expert actually saved.
    void note_used(std::size_t bytes);
    // D3: the step boundary the deadlines are measured against.
    void advance(std::uint64_t now);
    // The controller's decision, pushed down: the queue measures, it does not judge.
    void set_enabled(bool value);
    [[nodiscard]] bool enabled() const;
    [[nodiscard]] double usefulness() const;

    [[nodiscard]] Stats stats() const;
    [[nodiscard]] bool gate_open() const;
    [[nodiscard]] int users() const;
private:
    ExpertPrefetchQueue()  = default;
    ~ExpertPrefetchQueue() = default;
    ExpertPrefetchQueue(const ExpertPrefetchQueue&)            = delete;
    ExpertPrefetchQueue& operator=(const ExpertPrefetchQueue&) = delete;

    mutable std::mutex mutex_;
    std::shared_ptr<core::Prefetcher> worker_;
    std::uint64_t ran_     = 0;
    std::uint64_t expired_ = 0;
    int users_             = 0;
};

} // namespace ninfer::models::qwen3_5::execution
