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
    std::uint64_t dropped_deadline = 0;
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

    Prefetcher(std::size_t budget_bytes, double min_usefulness = 0.10)
        : budget_(budget_bytes), min_usefulness_(min_usefulness) {
        worker_ = std::thread([this] { run(); });
    }

    ~Prefetcher() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        wake_.notify_all();
        if (worker_.joinable()) { worker_.join(); }
    }

    Prefetcher(const Prefetcher&)            = delete;
    Prefetcher& operator=(const Prefetcher&) = delete;

    // Enqueues a job if it fits the window budget and is not already past its deadline.
    bool enqueue(std::function<bool()> run, std::size_t bytes, std::uint64_t due, std::uint64_t now) {
        if (bytes == 0 || due <= now) {
            std::lock_guard lock(mutex_);
            ++stats_.dropped_deadline;
            return false;
        }
        {
            std::lock_guard lock(mutex_);
            if (!gate_open_ || window_bytes_ + bytes > budget_) {
                ++stats_.dropped_budget;
                return false;
            }
            window_bytes_ += bytes;
            stats_.bytes_requested += bytes;
            ++stats_.enqueued;
            jobs_.push_back(Job{std::move(run), bytes, due});
        }
        wake_.notify_one();
        return true;
    }

    // Called by the consumer when a prefetched payload was actually used (the "useful" side of the
    // gate). The caller reports bytes, not jobs, because a job may stage more than is used.
    void note_used(std::size_t bytes) {
        std::lock_guard lock(mutex_);
        stats_.bytes_used += bytes;
    }

    // Advances the window: drops jobs whose deadline has passed, recomputes the gate, resets the
    // byte budget. `now` is the logical step the engine is about to run.
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
        const auto used = stats_.usefulness();
        if (stats_.enqueued >= 16) {  // too few samples to judge
            if (used < min_usefulness_) {
                gate_open_ = false;
            } else if (used >= min_usefulness_ * 2.0) {
                gate_open_ = true;
            }
        }
    }

    [[nodiscard]] PrefetchStats stats() const {
        std::lock_guard lock(mutex_);
        return stats_;
    }

    [[nodiscard]] bool gate_open() const {
        std::lock_guard lock(mutex_);
        return gate_open_;
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
            }
            if (job.run) { (void)job.run(); }
        }
    }

    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<Job> jobs_;
    std::thread worker_;
    std::size_t budget_              = 0;
    std::size_t window_bytes_        = 0;
    double min_usefulness_           = 0.10;
    bool gate_open_                  = true;
    bool stopping_                   = false;
    PrefetchStats stats_{};
};

} // namespace ninfer::core
