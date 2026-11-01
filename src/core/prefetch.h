#pragma once

// The prefetch seam: a budgeted, deadline-aware queue with a usefulness gate, shaped after what
// Strata measured (src/core/expert_source.cpp:1064-1111, :1976-2004): a byte budget per window, a
// deadline per job so a late fetch is dropped instead of competing with the uploads on the critical
// path, and a gate that closes the whole mechanism when the bytes it fetched were not used.
//
// Strata's own numbers are the reason the gate is not optional: with an 18 GiB RAM tier that already
// held the predictions, read-ahead cost 9-15% of decode at 0% usefulness. The same trap exists here
// at 12-14 GiB, where an expert upload already costs 0.02 ms/token.
//
// Header-only and unwired: the expert prefetch (plan D1-D4) and the KV prefix prefetch (F2) both
// enqueue here, and nothing else in the engine has to know how the budget is spent.

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>

namespace ninfer::core {

struct PrefetchStats {
    std::uint64_t enqueued         = 0;
    std::uint64_t dropped_budget   = 0;
    // The gate closed (the bytes fetched so far were not used), so the mechanism is off until the
    // usefulness it did measure recovers. Counted apart from dropped_budget: one is "this window is
    // spent", the other is "this mechanism is not paying for itself", and folding them together
    // made a closed gate look like a budget problem.
    std::uint64_t dropped_gate     = 0;
    std::uint64_t dropped_deadline = 0;
    // Refused because the worker already has max_pending jobs (the one it is running plus the ones
    // waiting). This is the backpressure valve: a byte budget bounds what a window may spend, not
    // how far the producer may run ahead of a slow disk.
    std::uint64_t dropped_backlog  = 0;
    // Offered after stop(). A process-scoped owner is stopped before it is destroyed, so this is
    // the count of callers that raced the shutdown rather than a scheduling decision.
    std::uint64_t dropped_stopped  = 0;
    std::uint64_t bytes_requested  = 0;
    std::uint64_t bytes_used       = 0;

    [[nodiscard]] double usefulness() const noexcept {
        return bytes_requested == 0
                   ? 0.0
                   : static_cast<double>(bytes_used) / static_cast<double>(bytes_requested);
    }
};

class Prefetcher {
public:
    struct Job {
        std::function<bool()> run;  // returns true when the bytes were actually staged
        std::size_t bytes   = 0;
        std::uint64_t due   = 0;    // logical step by which this must have started
    };

    Prefetcher(std::size_t budget_bytes, double min_usefulness = 0.10,
               std::size_t max_pending = 8)
        : budget_(budget_bytes), min_usefulness_(min_usefulness), max_pending_(max_pending) {
        worker_ = std::thread([this] { run(); });
    }

    // Stops and joins the worker. A process-scoped owner has to be able to say "the fill is over"
    // before the objects a job points into are destroyed, and a destructor is too late to say it:
    // by then the last strong reference may already be the one a job is holding.
    //
    // Idempotent, and safe to call from any thread but the worker itself.
    void stop() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        wake_.notify_all();
        std::lock_guard join(join_mutex_);
        if (!worker_.joinable()) { return; }
        if (worker_.get_id() == std::this_thread::get_id()) { return; }
        worker_.join();
    }

    ~Prefetcher() { stop(); }

    Prefetcher(const Prefetcher&)            = delete;
    Prefetcher& operator=(const Prefetcher&) = delete;

    // Enqueues a job if it fits the window budget and is not already past its deadline.
    bool enqueue(std::function<bool()> run, std::size_t bytes, std::uint64_t due, std::uint64_t now) {
        std::lock_guard lock(mutex_);
        if (stopping_) {
            ++stats_.dropped_stopped;
            return false;
        }
        if (bytes == 0 || due <= now) {
            ++stats_.dropped_deadline;
            return false;
        }
        if (!enabled_) {
            ++stats_.dropped_gate;
            return false;
        }
        if (window_bytes_ + bytes > budget_) {
            ++stats_.dropped_budget;
            return false;
        }
        // Backpressure (D2's second gate). The byte budget says how much a window may ask for; this
        // says how far the caller may run ahead of the worker. Without it a slow disk turns the
        // read-ahead into an unbounded queue of jobs whose deadline has long passed by the time they
        // run, and the drop counter would blame the budget for a backlog.
        if (running_ + jobs_.size() >= max_pending_) {
            ++stats_.dropped_backlog;
            return false;
        }
        window_bytes_ += bytes;
        stats_.bytes_requested += bytes;
        ++stats_.enqueued;
        jobs_.push_back(Job{std::move(run), bytes, due});
        wake_.notify_one();
        return true;
    }

    // Called by the consumer when a prefetched payload was actually used (the "useful" side of the
    // gate). The caller reports bytes, not jobs, because a job may stage more than is used.
    void note_used(std::size_t bytes) {
        std::lock_guard lock(mutex_);
        stats_.bytes_used += bytes;
    }

    // Advances the window: drops jobs whose deadline has passed and resets the byte budget. `now` is
    // the logical step the engine is about to run. It deliberately does NOT decide whether the queue
    // should be on - that is the controller's call, and a second decision point here is how a gate
    // ends up fighting the policy that is supposed to own it.
    void advance(std::uint64_t now) {
        std::lock_guard lock(mutex_);
        jobs_.erase(std::remove_if(jobs_.begin(), jobs_.end(),
                                   [&](const Job& job) {
                                       if (job.due <= now) {
                                           ++stats_.dropped_deadline;
                                           return true;
                                       }
                                       return false;
                                   }),
                    jobs_.end());
        window_bytes_ = 0;
    }

    [[nodiscard]] PrefetchStats stats() const {
        std::lock_guard lock(mutex_);
        return stats_;
    }

    // Jobs the worker has taken but not finished, plus the ones still waiting: the number the
    // backpressure gate compares against max_pending.
    [[nodiscard]] std::size_t pending() const {
        std::lock_guard lock(mutex_);
        return running_ + jobs_.size();
    }

    [[nodiscard]] std::size_t max_pending() const {
        std::lock_guard lock(mutex_);
        return max_pending_;
    }

    // The gate is no longer decided here (2026-11-01). A threshold on usefulness is a number that
    // only holds for the machine it was measured on, and the runtime already has one place that
    // decides what to enable - PolicyController. This queue therefore measures and obeys: it counts
    // what it was asked for and what was used, and takes its on/off from the controller.
    void set_enabled(bool value) {
        std::lock_guard lock(mutex_);
        enabled_ = value;
    }
    [[nodiscard]] bool enabled() const {
        std::lock_guard lock(mutex_);
        return enabled_;
    }

    void set_budget(std::size_t bytes) {
        std::lock_guard lock(mutex_);
        budget_ = bytes;
    }

private:
    void run() {
        for (;;) {
            Job job;
            {
                std::unique_lock lock(mutex_);
                wake_.wait(lock, [this] { return stopping_ || !jobs_.empty(); });
                if (stopping_) { return; }
                job = std::move(jobs_.front());
                jobs_.pop_front();
                running_ = 1;
            }
            if (job.run) { (void)job.run(); }
            // Cleared after the job returns, so "pending" cannot dip below one while the worker is
            // inside a fill and let the producer overshoot the cap by the width of that window.
            std::lock_guard lock(mutex_);
            running_ = 0;
        }
    }

    mutable std::mutex mutex_;
    // Held only across the join, so two stop() calls cannot both wait on the same thread object.
    // Deliberately not `mutex_`: the worker needs that one to observe `stopping_` and exit.
    std::mutex join_mutex_;
    std::condition_variable wake_;
    std::deque<Job> jobs_;
    std::thread worker_;
    std::size_t budget_              = 0;
    std::size_t window_bytes_        = 0;
    std::size_t max_pending_         = 8;
    std::size_t running_             = 0; // 0 or 1: the worker's job, if it is carrying one
    double min_usefulness_           = 0.10; // kept for reporting only; nothing acts on it
    bool enabled_                    = true; // the controller's decision, not this queue's
    bool stopping_                   = false;
    PrefetchStats stats_{};
};

} // namespace ninfer::core
