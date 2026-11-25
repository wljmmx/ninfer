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
#include <cstdio>
#include <deque>
#include <exception>
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
    // Refused by the queue's hard ceiling on resident job bytes (enqueue_item). Separate from
    // dropped_budget, which belongs to the read-ahead window the queue has to allocate for: a
    // publish arrives with its payload already materialised, so the only memory bound it needs is
    // the ceiling on how many such payloads may be resident at once.
    std::uint64_t dropped_ceiling  = 0;
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
    // What counts as a large payload for the one-at-a-time rule in enqueue_item(): anything that
    // copies a whole prefix (one 163k checkpoint is 2545 MiB, one 256k is about 4 GiB).
    static constexpr std::size_t kLargePayloadBytes = 1ULL << 30;

    struct Job {
        std::function<bool()> run;  // returns true when the bytes were actually staged
        std::size_t bytes   = 0;
        std::uint64_t due   = 0;    // logical step by which this must have started
    };

    Prefetcher(std::size_t budget_bytes, double min_usefulness = 0.10,
               std::size_t max_pending = 8, std::size_t hard_bytes = 0)
        : budget_(budget_bytes), hard_bytes_(hard_bytes), max_pending_(max_pending),
          min_usefulness_(min_usefulness) {
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
        pending_bytes_ += bytes;
        stats_.bytes_requested += bytes;
        ++stats_.enqueued;
        jobs_.push_back(Job{std::move(run), bytes, due});
        wake_.notify_one();
        return true;
    }

    // Item-bounded enqueue for a payload the caller has ALREADY materialised: a KV checkpoint that
    // was exported into memory before it was handed over. The byte window above exists to bound
    // read-ahead the queue itself would have to allocate, so charging it again for bytes that
    // already exist refuses the job for a memory cost the caller has already paid. That is not
    // hypothetical: an exported 2.5 GiB checkpoint (one 163k prefix) measured
    // `[publish] timing: enqueue REFUSED 0.000 s (2545 MiB)` against the 1024 MiB window, which
    // either stored it synchronously on the serving lane (84 s of boundary, every queued request
    // starved) or, after 67d4acc2, dropped it entirely.
    //
    // What still bounds memory is the count (running + waiting, the same backpressure valve) and
    // the hard ceiling on resident job bytes. Both are needed: the count alone lets three 163k
    // checkpoints queue (7.5 GiB) on a box that cannot hold them, and the ceiling alone lets a
    // stream of tiny jobs grow without limit. Bytes are returned when the job completes or is
    // dropped by advance(), so the ceiling tracks what is actually resident, not what was asked for.
    bool enqueue_item(std::function<bool()> run, std::size_t bytes, std::uint64_t due,
                      std::uint64_t now) {
        std::lock_guard lock(mutex_);
        if (stopping_) {
            ++stats_.dropped_stopped;
            return false;
        }
        if (due <= now) {
            ++stats_.dropped_deadline;
            return false;
        }
        if (!enabled_) {
            ++stats_.dropped_gate;
            return false;
        }
        if (running_ + jobs_.size() >= max_pending_) {
            ++stats_.dropped_backlog;
            return false;
        }
        // Two large payloads must not be resident at once. Each one is a whole prefix copied into
        // host memory (2545 MiB at 163k), and the box that produced it is still holding the KV it is
        // about to restore. Measured on this one: the second 163k export, made while the first
        // payload was still in the writer, took 3.648 s instead of 1.355 s and then failed the next
        // request with `bad allocation`. The rule is on the payloads, not on a byte window: one large
        // one plus any number of small ones is fine, two large ones is the shape that does not fit.
        if (bytes >= kLargePayloadBytes && pending_bytes_ >= kLargePayloadBytes) {
            ++stats_.dropped_ceiling;
            return false;
        }
        if (hard_bytes_ != 0 && pending_bytes_ + bytes > hard_bytes_) {
            ++stats_.dropped_ceiling;
            return false;
        }
        pending_bytes_ += bytes;
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
        for (auto it = jobs_.begin(); it != jobs_.end();) {
            if (it->due <= now) {
                ++stats_.dropped_deadline;
                release_bytes_locked(it->bytes);
                it = jobs_.erase(it);
            } else {
                ++it;
            }
        }
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

    // Bytes of the jobs this queue is holding right now (the one being run plus the ones waiting).
    // This is the number the hard ceiling compares against, so it has to be observable: a ceiling
    // nobody can read is indistinguishable from a drop caused by the count.
    [[nodiscard]] std::size_t pending_bytes() const {
        std::lock_guard lock(mutex_);
        return pending_bytes_;
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
            // A job that throws must not take the process with it: this is a std::thread function,
            // and an escaping exception calls terminate(). Measured 2026-11-25 (0xC0000409,
            // subcode 7 FAST_FAIL_FATAL_APP_EXIT): a cache publish copied a 1.35 GB
            // CachedCheckpoint under memory pressure, bad_alloc escaped here, and the engine died
            // mid-load with every in-flight request reset. The job's payload is a cache entry, not
            // a client result, so losing it is the correct failure.
            if (job.run) {
                try {
                    (void)job.run();
                } catch (const std::exception& error) {
                    std::fprintf(stderr, "[prefetch] job failed: %s\n", error.what());
                    std::fflush(stderr);
                } catch (...) {
                    std::fprintf(stderr, "[prefetch] job failed: non-standard exception\n");
                    std::fflush(stderr);
                }
            }
            // Cleared after the job returns, so "pending" cannot dip below one while the worker is
            // inside a fill and let the producer overshoot the cap by the width of that window.
            std::lock_guard lock(mutex_);
            running_ = 0;
            release_bytes_locked(job.bytes);
        }
    }

    // Bytes leave the ceiling exactly when the payload leaves the queue: on completion, or when
    // advance() drops a job that outlived its deadline. The min() guards the accounting against a
    // double release instead of wrapping a std::size_t into a ceiling that refuses everything.
    void release_bytes_locked(std::size_t bytes) noexcept {
        pending_bytes_ -= std::min(pending_bytes_, bytes);
    }

    mutable std::mutex mutex_;
    // Held only across the join, so two stop() calls cannot both wait on the same thread object.
    // Deliberately not `mutex_`: the worker needs that one to observe `stopping_` and exit.
    std::mutex join_mutex_;
    std::condition_variable wake_;
    std::deque<Job> jobs_;
    std::thread worker_;
    std::size_t budget_              = 0;
    std::size_t hard_bytes_          = 0; // 0 = no ceiling (the window-budgeted read-ahead case)
    std::size_t window_bytes_        = 0;
    std::size_t pending_bytes_       = 0;
    std::size_t max_pending_         = 8;
    std::size_t running_             = 0; // 0 or 1: the worker's job, if it is carrying one
    double min_usefulness_           = 0.10; // kept for reporting only; nothing acts on it
    bool enabled_                    = true; // the controller's decision, not this queue's
    bool stopping_                   = false;
    PrefetchStats stats_{};
};

} // namespace ninfer::core
