#include "models/qwen3_5/execution/expert_prefetch.h"

#include "models/qwen3_5/execution/host_expert_residency.h"

#include <cstdlib>
#include <cstring>
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

// D2's budget. Measured, not invented: with the gate held open over the eight-slice long dialogue
// (1024 tokens, 12 GiB), 1 GiB spent on read-ahead was 0.16 useful and 256 MiB was 0.18 - the
// budget prunes exactly the attempts that would not have paid, so the smaller window is the better
// one. NINFER_PREFETCH_BUDGET_MB overrides it.
std::size_t budget_bytes() {
    if (const char* value = std::getenv("NINFER_PREFETCH_BUDGET_MB");
        value != nullptr && *value != '\0') {
        const auto mb = std::strtoull(value, nullptr, 10);
        if (mb != 0) { return static_cast<std::size_t>(mb) << 20; }
    }
    return std::size_t{256} << 20;
}

// The gate's threshold (D4). 0.10 is the plan's number and stays the default; a sweep has to be
// able to hold it open, because a gate that closes after sixteen samples makes every later
// configuration look identical - the pool size cannot matter to a mechanism that has stopped.
double min_usefulness() {
    if (const char* value = std::getenv("NINFER_PREFETCH_MIN_USEFUL");
        value != nullptr && *value != '\0') {
        return std::strtod(value, nullptr);
    }
    return 0.10;
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

// Q69: which target a read-ahead should take.
//
//   tier (default) - fill an arena slot. That is what D1 was built for and what every recorded
//                    `useful=` number was measured with; it also competes with the routes of the
//                    layer that follows, which is why four fifths of it was never routed to.
//   warm           - pull the bytes through the buffered reader and let the OS hold them. No slot is
//                    claimed and nothing can be evicted; the fill that follows a real miss becomes a
//                    memcpy out of the page cache (the same 8.6 ms -> 0.2 ms this project measured
//                    on the fill path).
//
// The two are alternatives, not layers: doing both would read every expert twice.
bool warm_target() {
    const char* value = std::getenv("NINFER_PREFETCH_MODE");
    return value != nullptr && std::strcmp(value, "warm") == 0;
}

ExpertPrefetchQueue& ExpertPrefetchQueue::instance() {
    static ExpertPrefetchQueue queue;
    return queue;
}

void ExpertPrefetchQueue::retain() {
    std::lock_guard lock(mutex_);
    if (users_++ != 0) { return; }
    // The workers start with the first residency, not with the process: a run that leaves the
    // read-ahead off pays neither the threads nor their shutdown.
    //
    // NINFER_PREFETCH_WORKERS (default 2). One was the original shape and it serialised forty banks'
    // offers behind one disk read; the budget is split so N workers cost the same bytes in total.
    static const std::size_t count = [] {
        const char* value = std::getenv("NINFER_PREFETCH_WORKERS");
        if (value != nullptr && *value != '\0') {
            return static_cast<std::size_t>(std::strtoul(value, nullptr, 10));
        }
        return std::size_t{2};
    }();
    const std::size_t n = std::max<std::size_t>(1, count);
    const std::size_t per = std::max<std::size_t>(1, budget_bytes() / n);
    workers_.clear();
    workers_.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        workers_.push_back(std::make_shared<core::Prefetcher>(per, min_usefulness(), max_pending()));
    }
    workers_enabled_ = true;
}

void ExpertPrefetchQueue::release() {
    std::vector<std::shared_ptr<core::Prefetcher>> doomed;
    {
        std::lock_guard lock(mutex_);
        if (users_ == 0) { return; }
        if (--users_ != 0) { return; }
        doomed = std::move(workers_);
        workers_.clear();
    }
    // Stopped outside the lock. A job takes the lock only to count itself, so this cannot deadlock,
    // but the join must not happen while another thread is waiting to enqueue.
    for (auto& w : doomed) { if (w) { w->stop(); } }
}

bool ExpertPrefetchQueue::enqueue(std::weak_ptr<HostExpertResidency> host, std::uint32_t bank,
                                  std::span<const std::int32_t> experts, std::size_t bytes,
                                  std::uint64_t due, std::uint64_t now) {
    std::shared_ptr<core::Prefetcher> worker;
    {
        std::lock_guard lock(mutex_);
        if (workers_.empty()) { return false; }
        worker = workers_[worker_cursor_++ % workers_.size()];
    }
    if (!worker) { return false; }
    auto job = std::make_shared<Job>(
        Job{std::move(host), bank, std::vector<std::int32_t>(experts.begin(), experts.end())});
    return worker->enqueue(
        [this, job]() {
            // D1's body. The residency is taken by strong reference for the length of the fill, so
            // a model destroyed under a running job is seen as an expired reference instead of a
            // dangling pointer; `->fill_prefetch` is the ordinary arena fill with the slots marked
            // so a later hit on one of them can be counted as usefulness (D4).
            auto residency = job->host.lock();
            if (!residency) {
                std::lock_guard lock(mutex_);
                ++expired_;
                return false;
            }
            if (warm_target()) {
                residency->warm(ExpertBankId{job->bank},
                                std::span<const std::int32_t>(job->experts));
            } else {
                residency->fill_prefetch(ExpertBankId{job->bank},
                                         std::span<const std::int32_t>(job->experts));
            }
            std::lock_guard lock(mutex_);
            ++ran_;
            return true;
        },
        bytes, due, now);
}

void ExpertPrefetchQueue::note_used(std::size_t bytes) {
    // The meter lives on the first worker: it is the one whose usefulness decides the gate, and a
    // broadcast would count the same bytes against every worker's window.
    std::shared_ptr<core::Prefetcher> meter;
    {
        std::lock_guard lock(mutex_);
        if (!workers_.empty()) { meter = workers_.front(); }
    }
    if (meter) { meter->note_used(bytes); }
}

void ExpertPrefetchQueue::advance(std::uint64_t now) {
    std::vector<std::shared_ptr<core::Prefetcher>> workers;
    {
        std::lock_guard lock(mutex_);
        workers = workers_;
    }
    // Every worker has its own deadline window, so every worker has to move its clock.
    for (auto& w : workers) { if (w) { w->advance(now); } }
}

ExpertPrefetchQueue::Stats ExpertPrefetchQueue::stats() const {
    std::vector<std::shared_ptr<core::Prefetcher>> workers;
    Stats out;
    {
        std::lock_guard lock(mutex_);
        workers     = workers_;
        out.ran     = ran_;
        out.expired = expired_;
    }
    // Counters add up; usefulness is a ratio and cannot, so the meter's own value is reported (it is
    // the one the gate is driven by) while bytes are summed so the panel still shows total traffic.
    for (auto& w : workers) {
        if (!w) { continue; }
        const auto s = w->stats();
        out.queue.enqueued         += s.enqueued;
        out.queue.dropped_budget   += s.dropped_budget;
        out.queue.dropped_gate     += s.dropped_gate;
        out.queue.dropped_deadline += s.dropped_deadline;
        out.queue.dropped_backlog  += s.dropped_backlog;
        out.queue.dropped_stopped  += s.dropped_stopped;
        out.queue.bytes_requested  += s.bytes_requested;
        out.queue.bytes_used       += s.bytes_used;
        out.pending                += w->pending();
        out.max_pending             = std::max(out.max_pending, w->max_pending());
    }
    if (!workers.empty() && workers.front()) { out.queue.bytes_used = workers.front()->stats().bytes_used; }
    return out;
}

bool ExpertPrefetchQueue::gate_open() const {
    std::shared_ptr<core::Prefetcher> meter;
    {
        std::lock_guard lock(mutex_);
        if (!workers_.empty()) { meter = workers_.front(); }
    }
    return meter ? meter->enabled() : false;
}

void ExpertPrefetchQueue::set_enabled(bool value) {
    std::vector<std::shared_ptr<core::Prefetcher>> workers;
    {
        std::lock_guard lock(mutex_);
        workers          = workers_;
        workers_enabled_ = value;
    }
    // One decision, every worker: a gate that only closed on the meter would let the other workers
    // keep reading after the controller said stop.
    for (auto& w : workers) { if (w) { w->set_enabled(value); } }
}

bool ExpertPrefetchQueue::enabled() const { return gate_open(); }

double ExpertPrefetchQueue::usefulness() const { return stats().queue.usefulness(); }

int ExpertPrefetchQueue::users() const {
    std::lock_guard lock(mutex_);
    return users_;
}

} // namespace ninfer::models::qwen3_5::execution
