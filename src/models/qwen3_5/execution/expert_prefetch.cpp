#include "models/qwen3_5/execution/expert_prefetch.h"

#include "models/qwen3_5/execution/host_expert_residency.h"

#include <cstdlib>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::execution {
namespace {

// One read-ahead, whole: which bank it belongs to, which experts it asks for, and a weak reference
// to the residency that would serve them. The bank and the list travel with the job so one worker
// can serve forty banks in the order they asked, instead of each bank owning a thread.
struct Job {
    std::weak_ptr<HostExpertResidency> host;
    std::uint32_t bank = 0;
    std::vector<std::int32_t> experts;
};

// D2's budget. One window is a step: at the plan's default shape that is 8 experts a layer over 40
// layers at the 1.8 MiB slot image this artifact produces, about 576 MiB. The default below is a
// ceiling for that and not a tuned number - what turns the budget into a decision is item 3.
std::size_t budget_bytes() {
    if (const char* value = std::getenv("NINFER_PREFETCH_BUDGET_MB");
        value != nullptr && *value != '\0') {
        const auto mb = std::strtoull(value, nullptr, 10);
        if (mb != 0) { return static_cast<std::size_t>(mb) << 20; }
    }
    return std::size_t{1024} << 20;
}

// The backpressure valve: how far the producer may run ahead of the worker. Eight is a step's worth
// of layers divided by five, i.e. enough that a burst of a few layers never blocks, and small enough
// that a disk which has stopped answering holds a bounded amount of queued work instead of the whole
// conversation. 0 is a legal value and turns every read-ahead into a refusal, which is what the
// positive control in tests/runtime/kv_seam_test.cpp uses to prove the gate is wired.
std::size_t max_pending() {
    if (const char* value = std::getenv("NINFER_PREFETCH_PENDING");
        value != nullptr && *value != '\0') {
        return static_cast<std::size_t>(std::strtoull(value, nullptr, 10));
    }
    return 8;
}

} // namespace

ExpertPrefetchQueue& ExpertPrefetchQueue::instance() {
    static ExpertPrefetchQueue queue;
    return queue;
}

void ExpertPrefetchQueue::retain() {
    std::lock_guard lock(mutex_);
    if (users_++ != 0) { return; }
    // The worker starts with the first residency, not with the process: a run that leaves the
    // read-ahead off pays neither the thread nor its shutdown.
    worker_ = std::make_shared<core::Prefetcher>(budget_bytes(), 0.10, max_pending());
}

void ExpertPrefetchQueue::release() {
    std::shared_ptr<core::Prefetcher> doomed;
    {
        std::lock_guard lock(mutex_);
        if (users_ == 0) { return; }
        if (--users_ != 0) { return; }
        doomed = std::move(worker_);
    }
    // Stopped outside the lock. A job takes the lock only to count itself, so this cannot deadlock,
    // but the join must not happen while another thread is waiting to enqueue.
    if (doomed) { doomed->stop(); }
}

bool ExpertPrefetchQueue::enqueue(std::weak_ptr<HostExpertResidency> host, std::uint32_t bank,
                                  std::span<const std::int32_t> experts, std::size_t bytes,
                                  std::uint64_t due, std::uint64_t now) {
    std::shared_ptr<core::Prefetcher> worker;
    {
        std::lock_guard lock(mutex_);
        worker = worker_;
    }
    if (!worker) { return false; }
    auto job = std::make_shared<Job>(
        Job{std::move(host), bank, std::vector<std::int32_t>(experts.begin(), experts.end())});
    return worker->enqueue(
        [this, job]() {
            // port (sm_89): D1's body, still a placeholder - it stages nothing. What it does do is
            // the ownership handshake the rest of D1 rests on: the residency is taken by strong
            // reference for the length of the job, so a model destroyed under a running fill is
            // seen here as an expired reference instead of a dangling pointer.
            auto residency = job->host.lock();
            std::lock_guard lock(mutex_);
            if (!residency) {
                ++expired_;
                return false;
            }
            ++ran_;
            return false; // the fill itself is item 4, deliberately not here yet
        },
        bytes, due, now);
}

void ExpertPrefetchQueue::note_used(std::size_t bytes) {
    std::shared_ptr<core::Prefetcher> worker;
    {
        std::lock_guard lock(mutex_);
        worker = worker_;
    }
    if (worker) { worker->note_used(bytes); }
}

void ExpertPrefetchQueue::advance(std::uint64_t now) {
    std::shared_ptr<core::Prefetcher> worker;
    {
        std::lock_guard lock(mutex_);
        worker = worker_;
    }
    if (worker) { worker->advance(now); }
}

ExpertPrefetchQueue::Stats ExpertPrefetchQueue::stats() const {
    std::shared_ptr<core::Prefetcher> worker;
    Stats out;
    {
        std::lock_guard lock(mutex_);
        worker          = worker_;
        out.ran         = ran_;
        out.expired     = expired_;
        out.staged      = staged_;
    }
    if (worker) {
        out.queue       = worker->stats();
        out.pending     = worker->pending();
        out.max_pending = worker->max_pending();
    }
    return out;
}

bool ExpertPrefetchQueue::gate_open() const {
    std::shared_ptr<core::Prefetcher> worker;
    {
        std::lock_guard lock(mutex_);
        worker = worker_;
    }
    return worker ? worker->gate_open() : false;
}

int ExpertPrefetchQueue::users() const {
    std::lock_guard lock(mutex_);
    return users_;
}

} // namespace ninfer::models::qwen3_5::execution
