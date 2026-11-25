#pragma once

// Small fixed-capacity request execution for every backend.

#include "core/device.h"
#include "core/nvtx.h"
#include "ninfer/types.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/resources.h"
#include "runtime/engine/request_record.h"
#include "runtime/engine/context_cache/resource_manager.h"
#include "runtime/engine/scheduler.h"
#include "runtime/engine/generation_budget.h"
#include "runtime/engine/kv_capacity.h"
#include "models/qwen3_5/execution/shape_class.h"
#include "models/qwen3_5/program/program.h"
#include "text/grammar.h"
#include "models/qwen3_5/execution/expert_path_control.h"
#include "models/qwen3_5/execution/ffn.h"
#include "models/qwen3_5/execution/param_bus.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace ninfer::runtime {

template <class Instance>
class EngineCore {

public:
    using ModelContract      = typename Instance::ModelContract;
    using ExecutionUnit      = typename ModelContract::ExecutionUnit;
    using UnitKind           = typename ModelContract::ExecutionUnitKind;
    using Checkpoint         = typename ModelContract::CheckpointHandle;
    using SequenceHandle     = typename ModelContract::SequenceHandle;
    using PendingBatch       = typename ModelContract::PendingBatch;
    using PreparedPrompt     = typename ModelContract::PreparedPrompt;
    using PublishedOutput    = typename ModelContract::PublishedOutput;
    using Request            = RequestRecord<ModelContract>;
    using Scheduling         = Scheduler<Request>;
    using RoundMembership    = typename Scheduling::RoundMembership;
    using ControlMembership  = typename Scheduling::ControlMembership;
    using ResourceManagement = ResourceManager<ModelContract>;
    using Clock              = std::chrono::steady_clock;

    // What an external cache must be registered with, from the live layout rather than a constant:
    // layer count, head width, plane count and state dtype all differ per model.
    [[nodiscard]] std::size_t cache_page_bytes() const {
        return instance_.program == nullptr
                   ? 0U
                   : instance_.program->checkpoint_geometry().page_bytes;
    }
    [[nodiscard]] std::int64_t cache_page_tokens() const {
        return instance_.program == nullptr
                   ? 0
                   : instance_.program->checkpoint_geometry().page_tokens;
    }
    [[nodiscard]] std::size_t cache_state_image_bytes() const {
        return instance_.program == nullptr
                   ? 0U
                   : instance_.program->checkpoint_geometry().state_image_bytes;
    }
    [[nodiscard]] std::uint32_t cache_kv_dtype_tag() const {
        return instance_.program == nullptr
                   ? 0U
                   : instance_.program->checkpoint_geometry().kv_dtype_tag;
    }
    [[nodiscard]] std::uint32_t cache_kv_planes() const {
        return instance_.program == nullptr
                   ? 0U
                   : instance_.program->checkpoint_geometry().kv_planes;
    }

    class RoundMasks final : public TokenMaskProvider {
    public:
        std::array<decltype(&std::declval<Request&>().output), kMaximumConcurrency> outputs{};

        bool constrained(std::size_t row) const noexcept override {
            return outputs[row] && outputs[row]->constrained();
        }

        std::uint32_t fill(std::size_t row, std::span<const TokenId> drafts,
                           std::span<std::uint32_t> words) override {
            return outputs[row]->grammar_masks(drafts, words);
        }

        void uploaded(std::size_t row, std::size_t bytes) noexcept override {
            outputs[row]->constraint_uploaded(bytes);
        }
    };

    // Read-only view of the loaded model: how many bytes one step's routed activation needs is a
    // property of the model, not of the core, so both cores hand the instance back here.
    [[nodiscard]] const Instance& instance() const noexcept { return instance_; }

    EngineCore(Instance& instance, DeviceContext& device, const EngineOptions& options,
               ContextMachineCostModel context_cost)        : instance_(instance), device_(device), max_context_(options.max_context),
          max_concurrency_(options.max_concurrency),
          max_outstanding_(static_cast<std::size_t>(options.max_concurrency) +
                           options.max_pending_requests),
          active_concurrency_(options.max_concurrency),
          pending_timeout_(std::chrono::milliseconds(options.pending_timeout_ms)),
          resources_(options.context_cache.enabled, std::move(context_cost),
                     options.checkpoint_cache) {
        if (max_concurrency_ == 0 || max_concurrency_ > kMaximumConcurrency ||
            options.max_pending_requests == 0 || pending_timeout_.count() <= 0) {
            throw std::invalid_argument("Engine core bounds are invalid");
        }
        // 13.5: the merged prefill starts from the environment's answer when one was given, and off
        // otherwise - the controller turns it on from the measured fit, never from a default. "Off
        // unless asked" is what keeps the single-request paths (CLI, perplexity, the tests' per-lane
        // arm) exactly as they were.
        prefill_merge_enabled_.store(prefill_merge_from_env_, std::memory_order_relaxed);
        prefill_admit_first_.store(prefill_admit_from_env_, std::memory_order_relaxed);
        // One line, once: the queue budget the engine will judge waiting requests by. It is derived
        // from the context when the caller left it unset, and that has to be visible at startup - the
        // only other way to see it was to wait for a 503.
        std::fprintf(stderr, "[queue] pending timeout %lld ms (max_context %u, max_concurrency %u)\n",
                     static_cast<long long>(pending_timeout_.count()), max_context_,
                     max_concurrency_);
        std::fflush(stderr);
        paused_.reserve(max_outstanding_);
        std::promise<void> startup;
        std::future<void> started = startup.get_future();
        worker_                   = std::thread([this, startup = std::move(startup)]() mutable {
            try {
                device_.bind_to_current_thread();
                startup.set_value();
            } catch (...) {
                startup.set_exception(std::current_exception());
                return;
            }
            worker_loop();
        });
        try {
            started.get();
        } catch (...) {
            if (worker_.joinable()) { worker_.join(); }
            throw;
        }
        // A separate thread, because a hang is exactly the case where the worker cannot report on
        // itself: on 2026-11-25 the engine stopped logging for 85 minutes in the middle of a
        // prefill (no throughput lines at all, then "host 4798.7% (4m 0.2s)" when the blocked call
        // returned) and 38 client requests were lost with nothing in the log to say where it was.
        watchdog_ = std::thread([this] { hang_watchdog_loop(); });
    }

    ~EngineCore() noexcept {
        request_stop();
        if (watchdog_.joinable()) { watchdog_.join(); }
        if (worker_.joinable()) { worker_.join(); }
    }

    // May be called while consumers are waiting; only the worker releases model resources. Split out
    // of the destructor so the serve layer can stop the engine BEFORE joining its HTTP workers: a
    // generation handler blocked in a wait has to see "stopping" and return, and it cannot do that
    // while the join in the destructor is the only place the flag is raised.
    void request_stop() noexcept {
        watchdog_stop_.store(true, std::memory_order_release);
        {
            std::lock_guard lock(queue_mutex_);
            stopping_ = true;
        }
        queue_cv_.notify_all();
    }

    EngineCore(const EngineCore&)            = delete;
    EngineCore& operator=(const EngineCore&) = delete;

    class Submission {
    public:
        Submission() noexcept = default;

        ~Submission() { reset(); }

        Submission(Submission&& other) noexcept
            : owner_(std::exchange(other.owner_, nullptr)), request_(std::move(other.request_)) {}

        Submission& operator=(Submission&& other) noexcept {
            if (this != &other) {
                reset();
                owner_   = std::exchange(other.owner_, nullptr);
                request_ = std::move(other.request_);
            }
            return *this;
        }

        Submission(const Submission&)            = delete;
        Submission& operator=(const Submission&) = delete;

        void wait_for_admission(const CancellationView& cancellation) {
            if (owner_ == nullptr || request_ == nullptr) {
                throw std::logic_error("concurrent submission is empty");
            }
            owner_->wait_for_admission(request_, cancellation);
        }

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
            if (owner_ == nullptr || request_ == nullptr) {
                throw std::logic_error("concurrent submission is empty");
            }
            const bool streaming = request_->consumer_mode == OutputConsumerMode::Streaming;
            if (streaming != (sink != nullptr)) {
                throw std::invalid_argument(
                    "GenerationHandle wait sink does not match its submitted consumer mode");
            }
            EngineCore* owner = std::exchange(owner_, nullptr);
            return owner->wait_for_request(std::exchange(request_, nullptr), sink, cancellation);
        }

    private:
        Submission(EngineCore& owner, std::shared_ptr<Request> request) noexcept
            : owner_(&owner), request_(std::move(request)) {}

        void reset() noexcept {
            if (owner_ != nullptr && request_ != nullptr) {
                owner_->abandon_request(std::move(request_));
            }
            owner_ = nullptr;
        }

        EngineCore* owner_ = nullptr;
        std::shared_ptr<Request> request_;

        friend class EngineCore;
    };

    Submission submit(PreparedPrompt prompt, PromptSummary prompt_summary, double prepare_seconds,
                      ResolvedRequestOptions options, OutputConsumerMode consumer_mode,
                      GenerationObservationOptions observation,
                      Clock::time_point pending_deadline = {}) {
        const Clock::time_point submitted = Clock::now();
        if (pending_deadline == Clock::time_point{}) {
            pending_deadline = submitted + pending_timeout_;
        }
        if (submitted >= pending_deadline) {
            throw RequestError(RequestErrorKind::QueueTimeout,
                               "inference request expired before submission");
        }

        std::uint64_t request_id        = 0;
        std::uint64_t publication_order = 0;
        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_ || failed_) {
                throw RequestError(RequestErrorKind::Unavailable,
                                   "inference engine is unavailable");
            }
            if (outstanding_ >= max_outstanding_) {
                throw RequestError(RequestErrorKind::Overloaded, "inference request queue is full");
            }
            // Capacity planning, not a stall: the queue is served in submission order, so this
            // request's wait is every queued request plus its own prefill. When that work cannot
            // drain inside the 15-minute ceiling the deadline logic enforces, admitting it only
            // becomes a 503 after minutes of waiting (measured 2026-11-25: 16x88k tokens, 340 s
            // of waiting, then 503 while the engine was healthy). Refuse it now as Overloaded.
            {
                double queued_seconds =
                    static_cast<double>(prompt_summary.prompt_tokens) / kPrefillTokensPerSecond;
                for (const auto& queued : pending_) {
                    queued_seconds += static_cast<double>(queued->prompt_summary.prompt_tokens) /
                                      kPrefillTokensPerSecond;
                }
                if (queued_seconds > kQueueDrainCeilingSeconds) {
                    throw RequestError(RequestErrorKind::Overloaded,
                                       "inference request queue is full");
                }
            }
            if (next_request_id_ == 0 || next_publication_order_ == 0) {
                throw std::overflow_error("request identity space exhausted");
            }
            ++outstanding_;
            request_id        = next_request_id_++;
            publication_order = next_publication_order_++;
        }

        std::shared_ptr<Request> request;
        try {
            const auto constraint_started = observation.phase_timings ? Clock::now() : submitted;
            auto output                   = instance_.frontend.make_output_session(
                prompt, options.stop, options.output, options.execution.thinking,
                options.constraint, options.tool_choice);
            if (Clock::now() >= pending_deadline) {
                throw RequestError(RequestErrorKind::QueueTimeout,
                                   "inference request expired during grammar preparation");
            }
            const std::uint32_t capacity_output =
                max_context_ - prompt_summary.prompt_tokens + static_cast<std::uint32_t>(1);
            try {
                output.validate_generation_capacity(
                    std::min(options.execution.requested_output_tokens, capacity_output));
            } catch (const std::invalid_argument& error) {
                throw RequestError(RequestErrorKind::ThinkingBudgetCapacityInsufficient,
                                   error.what());
            }
            const auto ready = Clock::now();
            output.observe_constraint(
                observation.phase_timings,
                std::chrono::duration<double>(ready - constraint_started).count());
            prepare_seconds += std::chrono::duration<double>(ready - submitted).count();
            request = std::make_shared<Request>(request_id, publication_order, std::move(prompt),
                                                std::move(output), prompt_summary, prepare_seconds,
                                                std::move(options), consumer_mode,
                                                std::move(observation), pending_deadline, ready);
        } catch (...) {
            release_reserved_capacity();
            throw;
        }

        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_ || failed_) {
                --outstanding_;
                throw RequestError(RequestErrorKind::Unavailable,
                                   "inference engine is unavailable");
            }
            pending_.push_back(request);
        }
        request_admission_check();
        queue_cv_.notify_one();
        return Submission(*this, std::move(request));
    }

    // The gauges a monitor reads, from the copy the worker published with the stats.
    //
    // This used to take execution_mutex_, which the worker holds for a whole execution round. A
    // long prefill runs many chunks inside ONE round, so /monitor/snapshot answered only after it
    // finished: measured 4.4 s in a 6.0 s prefill round, and a 163k-token prompt (~108 s) made the
    // console's 1 Hz poll time out, which it drew as a gap in the curve. The numbers were already
    // stale-by-design (a 1 Hz poll of a 5 s stats interval), so reading the published copy costs
    // nothing a reader had before and never waits for the GPU work.
    [[nodiscard]] MemorySummary memory_summary() const {
        MemorySummary out;
        bool published = false;
        {
            std::lock_guard lock(stats_mutex_);
            published = memory_published_;
            if (published) { out = published_memory_; }
        }
        // Before the first publication (a monitor poll between construction and the first round)
        // there is no copy yet; the engine cannot be busy then, so the live read cannot block.
        if (!published) { out = instance_.program->memory_summary(); }
        const KvCapacityResolution& resolution = instance_.kv_capacity_resolution;
        out.kv_capacity_mode                   = resolution.mode;
        // 13.4: the program reports the pool's LIVE edge (a runtime shrink/grow moves it); the
        // resolution is only the fallback for a model without a paged text cache.
        out.kv_capacity_page_groups            = out.kv_capacity_page_groups != 0
                                                     ? out.kv_capacity_page_groups
                                                     : resolution.main_page_groups;
        out.kv_capacity_max_page_groups        = resolution.maximum_main_page_groups;
        // port (sm_89): the residency split. The Program reports what it was built with; the
        // resolution is the fallback for a model without a paged text cache.
        out.kv_logical_page_groups =
            out.kv_logical_page_groups != 0 ? out.kv_logical_page_groups
                                            : resolution.logical_page_groups;
        out.kv_resident_page_groups =
            out.kv_resident_page_groups != 0 ? out.kv_resident_page_groups
                                             : resolution.resident_page_groups;
        out.kv_host_payload_bytes = out.kv_host_payload_bytes != 0
                                        ? out.kv_host_payload_bytes
                                        : resolution.host_kv_payload_bytes;
        out.evict_contract_effective =
            resolution.evict_contract_effective != 0 ? EvictionContract::On : EvictionContract::Off;
        out.evict_contract_reasons = resolution.evict_contract_reasons;
        out.minimum_runtime_reservation_bytes  = resolution.minimum_runtime_reservation_bytes;
        out.kv_capacity_increment_bytes        = resolution.bytes_per_additional_main_page_group;
        out.runtime_reservation_bytes          = resolution.runtime_reservation_bytes;
        out.available_after_weights_bytes      = resolution.available_after_weights_bytes;
        out.available_after_startup_bytes      = resolution.available_after_startup_bytes;
        out.free_after_startup_bytes           = resolution.free_after_startup_bytes;
        out.kv_capacity_headroom_bytes         = resolution.automatic_headroom_bytes;
        out.planned_slack_bytes                = resolution.planned_slack_bytes;
        out.startup_retained_bytes             = instance_.startup_retained_bytes;
        return out;
    }

    [[nodiscard]] RuntimeStats runtime_stats() const {
        std::lock_guard lock(stats_mutex_);
        return published_stats_;
    }

    // The per-request projection published with the last stats snapshot. Copied under the same lock
    // as the stats: a reader gets a consistent row set, never a half-built one.
    [[nodiscard]] std::vector<RequestSnapshot> request_snapshots() const {
        std::lock_guard lock(stats_mutex_);
        return published_requests_;
    }

    [[nodiscard]] bool is_available() const {
        std::lock_guard lock(queue_mutex_);
        return !stopping_ && !failed_;
    }

    // appendix HX (plan item 9b): the engine's monotonic clock in milliseconds, for the policy
    // layer, which has no clock of its own (appendix HN.1).
    [[nodiscard]] static std::uint64_t monotonic_ms() {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch())
                .count());
    }

    // appendix HX (9b): 0 or absent means the sweep is off, and then nothing about the engine's
    // behaviour changes at all (criterion 4).
    [[nodiscard]] static std::uint64_t read_idle_reclaim_ms() {
        const char* text = std::getenv("NINFER_KV_IDLE_RECLAIM_MS");
        if (text == nullptr || *text == '\0') { return 0; }
        const long value = std::strtol(text, nullptr, 10);
        return value > 0 ? static_cast<std::uint64_t>(value) : 0;
    }

    void reset_memory_peaks() noexcept {
        try {
            std::scoped_lock lock(execution_mutex_);
            instance_.program->reset_memory_peaks();
        } catch (...) {}
    }

private:
    enum class HostWorkClass : std::uint8_t {
        Decode,
        Prefill,
        Control,
    };

    using EngineHostPhase = RequestEngineHostPhase;

    [[nodiscard]] static nvtx::Name phase_range_name(EngineHostPhase phase) noexcept;

    struct ActiveExposure {
        std::shared_ptr<Request> request;
        std::uint32_t lane = 0;
    };

    struct ActiveExposureSet {
        std::array<ActiveExposure, kMaximumConcurrency> entries{};
        std::size_t size = 0;
    };

    struct HostPhaseMeasurement {
        Clock::time_point started;
        std::uint64_t accounted_before = 0;
        ActiveExposureSet exposed;
    };

    [[nodiscard]] static std::uint64_t elapsed_ns(Clock::time_point started,
                                                  Clock::time_point finished) noexcept;

    [[nodiscard]] ActiveExposureSet active_exposure_set() const;

    [[nodiscard]] HostPhaseMeasurement begin_host_phase() const;

    void set_host_work_class(HostWorkClass work_class,
                             std::span<const std::uint32_t> decode_lanes = {}) noexcept;

    [[nodiscard]] bool current_decode_contains(std::uint32_t lane) const noexcept;

    void add_class_host_time(std::uint64_t host_ns, std::uint64_t device_wait_ns) noexcept;

    void expose_engine_phase(const ActiveExposureSet& exposed, EngineHostPhase phase,
                             std::uint64_t elapsed) noexcept;

    void finish_engine_phase(const HostPhaseMeasurement& measurement,
                             EngineHostPhase phase) noexcept;

    void record_program_timing(runtime::ExecutionTiming timing,
                               const ActiveExposureSet& exposed) noexcept;

    void finish_program_call(const HostPhaseMeasurement& measurement,
                             runtime::ExecutionTiming timing) noexcept;

    void record_detail(std::uint64_t RuntimeHostWorkStats::*elapsed_member,
                       std::uint64_t RuntimeHostWorkStats::*invocation_member,
                       Clock::time_point started) noexcept;

    class EnginePhaseScope {
    public:
        EnginePhaseScope(EngineCore& owner, EngineHostPhase phase)
            : owner_(owner), phase_(phase), measurement_(owner.begin_host_phase()) {
            range_.emplace(phase_range_name(phase), nvtx::Category::Runtime);
        }

        ~EnginePhaseScope() { finish(); }

        EnginePhaseScope(const EnginePhaseScope&)            = delete;
        EnginePhaseScope& operator=(const EnginePhaseScope&) = delete;

        void pause_range() noexcept { range_.reset(); }

        void resume_range() noexcept {
            if (active_ && !range_) {
                range_.emplace(phase_range_name(phase_), nvtx::Category::Runtime);
            }
        }

        void checkpoint() noexcept {
            if (!active_) { return; }
            owner_.finish_engine_phase(measurement_, phase_);
            measurement_ = owner_.begin_host_phase();
        }

        void finish() noexcept {
            if (!active_) { return; }
            range_.reset();
            owner_.finish_engine_phase(measurement_, phase_);
            active_ = false;
        }

    private:
        EngineCore& owner_;
        EngineHostPhase phase_;
        HostPhaseMeasurement measurement_;
        std::optional<nvtx::ScopedRange> range_;
        bool active_ = true;
    };

    class ProgramCallScope {
    public:
        explicit ProgramCallScope(EngineCore& owner)
            : owner_(owner), measurement_(owner.begin_host_phase()) {}

        ~ProgramCallScope() noexcept { finish(failed_timing_); }

        ProgramCallScope(const ProgramCallScope&)            = delete;
        ProgramCallScope& operator=(const ProgramCallScope&) = delete;

        [[nodiscard]] runtime::ExecutionTiming& failed_timing() noexcept { return failed_timing_; }

        void finish(runtime::ExecutionTiming timing) noexcept {
            if (!active_) { return; }
            owner_.finish_program_call(measurement_, timing);
            active_ = false;
        }

    private:
        EngineCore& owner_;
        HostPhaseMeasurement measurement_;
        runtime::ExecutionTiming failed_timing_;
        bool active_ = true;
    };

    void publish_runtime_stats();

    void wait_for_admission(const std::shared_ptr<Request>& request,
                            const CancellationView& cancellation) {
        try {
            for (;;) {
                const bool cancelled = cancellation.requested();
                std::unique_lock lock(request->mutex);
                if (request->error) { std::rethrow_exception(request->error); }
                if (cancelled || request->cancelled.load(std::memory_order_acquire) ||
                    (request->response_done &&
                     request->result.finish_reason == FinishReason::Cancelled)) {
                    throw RequestError(RequestErrorKind::Cancelled,
                                       "inference request cancelled while waiting for admission");
                }
                if (request->generation_start || request->response_done) { return; }
                request->cv.wait_for(lock, std::chrono::milliseconds(10), [&] {
                    return request->generation_start.has_value() || request->response_done;
                });
            }
        } catch (...) {
            request->cancelled.store(true, std::memory_order_release);
            request_admission_check();
            queue_cv_.notify_one();
            throw;
        }
    }

    GenerationResult wait_for_request(std::shared_ptr<Request> request, OutputSink* sink,
                                      const CancellationView& cancellation) {
        struct ConsumerGuard {
            EngineCore* owner;
            std::shared_ptr<Request> request;

            ~ConsumerGuard() { owner->release_consumer(request); }
        } guard{this, request};

        std::exception_ptr caller_error;
        bool scheduling_failed   = false;
        const auto fail_consumer = [&] {
            if (caller_error == nullptr) { caller_error = std::current_exception(); }
            request->cancelled.store(true, std::memory_order_release);
            request_admission_check();
            queue_cv_.notify_one();
        };
        std::optional<GenerationStart> start;
        bool start_delivered = false;
        std::optional<PromptProgress> progress;
        std::vector<typename Request::StreamEvent> events;
        for (;;) {
            start.reset();
            progress.reset();
            events.clear();
            bool done = false;
            {
                std::unique_lock lock(request->mutex);
                request->cv.wait_for(lock, std::chrono::milliseconds(10), [&] {
                    return request->response_done || (!start_delivered && request->generation_start) ||
                           request->stream_progress.has_value() || !request->events.empty();
                });
                if (!start_delivered && request->generation_start) {
                    start           = request->generation_start;
                    start_delivered = true;
                }
                progress = std::move(request->stream_progress);
                request->stream_progress.reset();
                events.swap(request->events);
                done = request->response_done;
            }

            if (caller_error == nullptr && sink != nullptr) {
                try {
                    if (start) { sink->start(std::move(*start)); }
                    if (progress) { sink->progress(std::move(*progress)); }
                } catch (...) { fail_consumer(); }
            }
            for (auto& event : events) {
                if (const auto* scheduling =
                        std::get_if<std::unique_ptr<GenerationSchedulingObservation>>(&event)) {
                    if (scheduling_failed) { continue; }
                    try {
                        request->observation.scheduling(**scheduling);
                    } catch (...) {
                        scheduling_failed = true;
                        fail_consumer();
                    }
                } else if (const auto* first_token =
                               std::get_if<GenerationFirstTokenObservation>(&event)) {
                    if (caller_error == nullptr) {
                        try {
                            request->observation.first_token(*first_token);
                        } catch (...) { fail_consumer(); }
                    }
                } else if (caller_error == nullptr && sink != nullptr) {
                    try {
                        if (auto* timing = std::get_if<GenerationTimingObservation>(&event)) {
                            sink->timing(std::move(*timing));
                        } else {
                            sink->publish(std::move(std::get<OutputEvent>(event)));
                        }
                    } catch (...) { fail_consumer(); }
                }
            }

            if (caller_error == nullptr) {
                try {
                    if (cancellation.requested()) {
                        request->cancelled.store(true, std::memory_order_release);
                        request_admission_check();
                        queue_cv_.notify_one();
                    }
                } catch (...) { fail_consumer(); }
            }
            if (!done) { continue; }

            if (caller_error != nullptr) { std::rethrow_exception(caller_error); }
            std::lock_guard lock(request->mutex);
            if (request->error != nullptr) { std::rethrow_exception(request->error); }
            return std::move(request->result);
        }
    }

    // Coalesces admission-visible queue/resource changes. Ordinary prefill/decode progress does not
    // re-arm a temporarily blocked inspection.
    void request_admission_check() noexcept {
        admission_check_pending_.store(true, std::memory_order_release);
    }

    [[nodiscard]] bool consume_admission_check() noexcept {
        return admission_check_pending_.exchange(false, std::memory_order_acq_rel);
    }

    // v8 13.4 item 4 (joiner pacing): does the queued head still belong to the current merge
    // window? The active prefill lanes' remaining tokens plus the head's prompt form the
    // candidate rectangle; when prefill_batch_fit prices the whole set into one pass, or when
    // nothing merges anyway (each row runs alone), binding the head now is free. When the
    // column budget cuts the window short of the head, the head waits for the next window: it
    // would bind mid-flight into a pass it cannot join and push the batch into narrow rows.
    // The hold is bounded: as the current window's lanes finish, their remaining counts drop
    // out and the head becomes the only row, which always binds.
    [[nodiscard]] bool queued_head_joins_prefill_window() {
        std::array<std::uint32_t, kMaximumConcurrency> remaining{};
        std::size_t queued = 0;
        {
            std::lock_guard lock(queue_mutex_);
            if (pending_.empty()) { return true; }
            for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                if (!slots_[lane]) { continue; }
                const std::uint32_t left = instance_.program->prefill_remaining(lane);
                if (left != 0) { remaining[queued++] = left; }
            }
            remaining[queued++] = pending_.front()->prompt_summary.prompt_tokens;
        }
        if (queued <= 1) { return true; }
        const auto fit = models::qwen3_5::execution::prefill_batch_fit(
            std::span<const std::uint32_t>(remaining.data(), queued));
        return fit.width == 0 || fit.lanes == queued;
    }

    struct MaterializingRequest {
        std::shared_ptr<Request> request;
        LaneId destination;
        BeginSummary summary;
        bool resumed = false;
        typename ResourceManagement::SourceChoice source;
    };

    struct AdmissionDecision {
        std::shared_ptr<Request> request;
        bool restoring = false;
        std::vector<typename ResourceManagement::SourceChoice> sources;
        std::size_t current = 0;
        typename ResourceManagement::ReclaimCursor reclaim;
    };

    struct CaptureDecision {
        SequenceHandle sequence;
        std::uint32_t frontier = 0;
        typename ResourceManagement::ReclaimCursor reclaim;
    };

    [[nodiscard]] std::optional<GenerationTimingObservation>
    record_committed_output(const std::shared_ptr<Request>& request,
                            std::uint32_t accepted_tokens) {
        if (accepted_tokens == 0) { return std::nullopt; }
        cumulative_stats_.generated_tokens += accepted_tokens;
        const bool observe_wall =
            request->observation.phase_timings || request->observation.live_timings;
        const bool need_now         = !request->first_token || observe_wall;
        const Clock::time_point now = need_now ? Clock::now() : Clock::time_point{};
        if (!request->first_token) {
            request->first_token = now;
            if (request->observation.first_token) {
                GenerationFirstTokenObservation observation{
                    .prepare_seconds = request->prepare_seconds,
                    .elapsed_since_submit_seconds =
                        std::chrono::duration<double>(now - request->submitted).count(),
                    .queue_wait_seconds = request->admitted_at
                                              ? std::chrono::duration<double>(
                                                    *request->admitted_at - request->submitted)
                                                    .count()
                                              : 0.0,
                };
                {
                    std::lock_guard lock(request->mutex);
                    request->events.emplace_back(observation);
                }
                request->cv.notify_one();
            }
        }
        if (!observe_wall) { return std::nullopt; }
        if (!request->admitted_at || !request->first_token) {
            throw std::logic_error("committed output has no observed admission boundary");
        }
        request->last_token = now;
        if (!request->observation.live_timings) { return std::nullopt; }
        if (request->generated.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("generated token count exceeds observation domain");
        }
        return GenerationTimingObservation{
            .generated_tokens      = static_cast<std::uint32_t>(request->generated.size()),
            .prompt_elapsed_ns     = elapsed_ns(*request->admitted_at, *request->first_token),
            .generation_elapsed_ns = elapsed_ns(*request->first_token, *request->last_token),
        };
    }

    void record_first_output(const std::shared_ptr<Request>& request);
    static void record_execution_work(GenerationWorkTiming& work, ExecutionTiming timing) noexcept;

    void append_output(const std::shared_ptr<Request>& request, PublishedOutput output,
                       std::optional<GenerationTimingObservation> timing = std::nullopt,
                       EnginePhaseScope* phase                           = nullptr) {
        if (output.empty() && !timing) { return; }
        if (!request->first_output_timing &&
            std::any_of(output.begin(), output.end(), [](const OutputEvent& event) {
                if (const auto* text = std::get_if<OutputDelta>(&event))
                    return !text->text.empty();
                if (const auto* arguments = std::get_if<ToolArgumentsDelta>(&event))
                    return !arguments->text.empty();
                return false;
            })) {
            if (phase) { phase->checkpoint(); }
            record_first_output(request);
        }
        const bool streaming = request->consumer_mode == OutputConsumerMode::Streaming;
        {
            std::lock_guard lock(request->mutex);
            if (streaming && timing) { request->events.emplace_back(std::move(*timing)); }
            for (OutputEvent& event : output) {
                if (const auto* delta = std::get_if<OutputDelta>(&event)) {
                    auto& full = delta->channel == OutputChannel::Reasoning ? request->reasoning
                                                                            : request->content;
                    full += delta->text;
                }
                if (streaming) { request->events.emplace_back(std::move(event)); }
            }
        }
        if (streaming) { request->cv.notify_one(); }
    }

    // Model-format failures settle only this already-committed row. Native abort still owns the
    // sequence cleanup; decoder errors must not escape to the Engine-wide invariant handler.
    bool fail_output(const std::shared_ptr<Request>& request) {
        const auto error = request->output.output_error();
        if (!error) { return false; }
        auto aborted = instance_.program->abort(*request->sequence);
        request->generation_timings = aborted.timings;
        request->speculative_stats  = std::move(aborted.speculative);
        complete_error(request, error->exception());
        return true;
    }

    void publish_prompt_progress(const std::shared_ptr<Request>& request) {
        if (!request->observation.prompt_progress) { return; }
        if (!request->admitted_begin || !request->admitted_at) {
            throw std::logic_error("prompt progress has no admitted request boundary");
        }
        const BeginSummary& begin = *request->admitted_begin;
        if (begin.reused_prompt_tokens > begin.prompt_tokens ||
            request->computed_prompt_tokens > begin.prompt_tokens - begin.reused_prompt_tokens) {
            throw std::logic_error("prompt progress exceeds the admitted prompt frontier");
        }
        const PromptProgress progress{
            .total_prompt_tokens     = begin.prompt_tokens,
            .reused_prompt_tokens    = begin.reused_prompt_tokens,
            .processed_prompt_tokens = begin.reused_prompt_tokens + request->computed_prompt_tokens,
            .elapsed_ns              = elapsed_ns(*request->admitted_at, Clock::now()),
        };
        {
            std::lock_guard lock(request->mutex);
            if (request->response_done) { return; }
            request->stream_progress = progress;
        }
        request->cv.notify_one();
    }

    void publish_generation_start(const std::shared_ptr<Request>& request, BeginSummary begin) {
        if (request->admitted_begin) {
            throw std::logic_error("request admission published generation start twice");
        }
        request->admitted_begin = begin;
        if (request->observation.phase_timings || request->observation.live_timings ||
            request->observation.prompt_progress) {
            if (!request->admitted_at) {
                throw std::logic_error("generation start has no binding timestamp");
            }
        }
        {
            std::lock_guard lock(request->mutex);
            if (request->generation_start || request->response_done) {
                throw std::logic_error("request has an invalid generation-start state");
            }
            request->generation_start = GenerationStart{
                .prompt               = request->prompt_summary,
                .reused_prompt_tokens = begin.reused_prompt_tokens,
            };
        }
        request->cv.notify_one();
    }

    void release_reserved_capacity() noexcept {
        std::lock_guard lock(queue_mutex_);
        if (outstanding_ != 0) { --outstanding_; }
    }

    void release_consumer(const std::shared_ptr<Request>& request) noexcept {
        bool release = false;
        {
            std::lock_guard lock(request->mutex);
            request->consumer_released = true;
            if (request->response_done && !request->capacity_released) {
                request->capacity_released = true;
                release                    = true;
            }
        }
        if (release) { release_reserved_capacity(); }
    }

    void abandon_request(std::shared_ptr<Request> request) noexcept {
        request->cancelled.store(true, std::memory_order_release);
        request_admission_check();
        queue_cv_.notify_one();
        release_consumer(request);
    }

    bool prepare_completion(const std::shared_ptr<Request>& request) {
        {
            std::lock_guard lock(request->mutex);
            if (request->response_done) { return false; }
        }
        // 13.5: the queue is alive. A waiting request's deadline is extended while this stamp keeps
        // moving (see extend_pending_deadlines_locked) - a queue timeout means the engine is stuck,
        // not that it was busy. Measured shape of the wrong reading: eight 800-token requests need
        // about 32 s of service, the third batch would have been admitted two seconds later, and the
        // 30 s pending timeout killed it at 30.48 s with a 503.
        last_completion_ms_.store(monotonic_ms(), std::memory_order_relaxed);
        if (request->preemption_count) {
            observe_scheduling(request, GenerationSchedulingTransition::Terminal);
        }
        if (admission_decision_ && admission_decision_->request == request) {
            admission_decision_.reset();
        }
        if (!request->admitted_at) {
            request->admission.revoked_checkpoints = resources_.source_revocations(request->id);
            finish_source_wait(request, Clock::now());
        }
        resources_.release_source(*instance_.program, request->id);
        if (request->continuation_owner) {
            resources_.abandon(*instance_.program, request->continuation_owner);
            request->continuation_owner = 0;
        }
        request->suspended.reset();
        request->base_plan.reset();
        request->prompt      = {};
        request->model_state = EngineRequestState::ModelFinished;
        return true;
    }

    void release_request(const std::shared_ptr<Request>& request) noexcept {
        scheduler_.finished(request->id);
        request_admission_check();
        if (request->lane) {
            const auto lane = request->lane->value;
            capture_decisions_[lane].reset();
            runnable_units_[lane] = false;
            slots_[lane].reset();
        }
        request->sequence.reset();
        request->lane.reset();
        request->budget.reset();
        request->terminal_reason.reset();
    }

    // upstream a754fcf3: local execution is synchronous, so a unit's own measured latency is the
    // service slice it consumed. Waiting for a grant, reclamation and replay never consume it.
    static std::uint64_t execution_elapsed_ns(const runtime::ExecutionTiming& timing) noexcept {
        return timing.submit_host_ns + timing.device_wait_ns + timing.post_host_ns +
               timing.gpu_elapsed_ns;
    }

    void record_service(std::uint64_t request, bool progressed,
                        std::uint64_t elapsed_ns) noexcept {
        if (progressed) { scheduler_.served(request, elapsed_ns); }
    }

    void record_recovery(std::uint64_t request, std::uint64_t elapsed_ns) noexcept {
        scheduler_.recovery_work(request, elapsed_ns);
    }

    void publish_completion(const std::shared_ptr<Request>& request, GenerationResult result,
                            std::exception_ptr error = {}) {
        bool release = false;
        {
            std::lock_guard lock(request->mutex);
            if (request->response_done) { return; }
            request->result        = std::move(result);
            request->error         = std::move(error);
            request->response_done = true;
            if (request->consumer_released && !request->capacity_released) {
                request->capacity_released = true;
                release                    = true;
            }
        }
        if (release) { release_reserved_capacity(); }
        request->cv.notify_one();
    }

    void complete_error(const std::shared_ptr<Request>& request_ref, std::exception_ptr error) {
        // Keep the request alive across release_request: that call clears its lane slot, which
        // may be the very shared_ptr this reference aliases.
        const auto request = request_ref;
        if (!prepare_completion(request)) { return; }
        release_request(request);
        publish_runtime_stats();
        publish_completion(request, {}, std::move(error));
    }

    void complete_success(const std::shared_ptr<Request>& request_ref, FinishReason reason) {
        const auto request = request_ref;
        HostPhaseMeasurement completion = begin_host_phase();
        if (!prepare_completion(request)) { return; }
        double prompt_wall_seconds     = 0.0;
        double generation_wall_seconds = 0.0;
        if (request->observation.phase_timings && request->first_token) {
            if (!request->admitted_at || !request->last_token) {
                throw std::logic_error("observed request completed without stable timing bounds");
            }
            prompt_wall_seconds =
                std::chrono::duration<double>(*request->first_token - *request->admitted_at)
                    .count();
            generation_wall_seconds =
                std::chrono::duration<double>(*request->last_token - *request->first_token).count();
        }
        if (!request->queue_wait_recorded) {
            request->host_timing.queue_wait_ns = elapsed_ns(request->submitted, Clock::now());
            request->queue_wait_recorded       = true;
        }
        GenerationResult result;
        result.engine_request_id            = request->id;
        result.admission                    = request->admission;
        result.computed_prefill_tokens      = request->computed_prompt_tokens;
        result.scheduling.preemptions       = request->preemption_count;
        result.scheduling.replay_restores   = request->replay_restores;
        result.scheduling.snapshot_restores = request->snapshot_restores;
        result.scheduling.replayed_tokens   = request->replayed_tokens;
        if (request->paused_at) {
            request->paused_ns += elapsed_ns(*request->paused_at, Clock::now());
            request->paused_at.reset();
        }
        result.scheduling.paused_ns            = request->paused_ns;
        result.scheduling.device_to_host_bytes = request->device_to_host_bytes;
        result.scheduling.host_to_device_bytes = request->host_to_device_bytes;
        result.prompt                          = request->prompt_summary;
        result.generated_token_ids             = std::move(request->generated);
        result.content                         = std::move(request->content);
        result.reasoning                       = std::move(request->reasoning);
        result.constraint                      = request->output.constraint_observation();
        result.tool_calls                      = request->output.take_tool_calls();
        result.tool_call_parse                 = request->output.tool_call_parse_diagnostics();
        result.reasoning_tokens                = request->output.reasoning_tokens();
        result.finish_reason                   = reason;
        result.matched_stop_string             = request->output.matched_stop_string();
        if (request->begin) {
            result.reused_prompt_tokens = request->begin->reused_prompt_tokens;
            result.prefix_reuse_path    = request->begin->prefix_reuse_path;
        }
        result.timings                 = request->generation_timings;
        result.timings.prepare_seconds = request->prepare_seconds;
        result.speculative             = std::move(request->speculative_stats);
        result.thinking                = request->output.thinking_stats();
        if (request->first_token) {
            result.timings.first_token_seconds =
                request->prepare_seconds +
                std::chrono::duration<double>(*request->first_token - request->submitted).count();
        }
        result.timings.prompt_wall_seconds     = prompt_wall_seconds;
        result.timings.generation_wall_seconds = generation_wall_seconds;
        result.timings.total_seconds =
            request->prepare_seconds +
            std::chrono::duration<double>(Clock::now() - request->submitted).count();
        release_request(request);
        finish_engine_phase(completion, EngineHostPhase::CommitOutput);
        result.engine_timing       = request->host_timing.public_snapshot();
        result.first_output_timing = request->first_output_timing;
        publish_runtime_stats();
        publish_completion(request, std::move(result));
    }

    void complete_cancelled(const std::shared_ptr<Request>& request_ref) {
        const auto request = request_ref;
        try {
            (void)request->output.preview_terminal(FinishReason::Cancelled);
            append_output(request, request->output.commit_preview());
            complete_success(request, FinishReason::Cancelled);
        } catch (...) {
            const std::exception_ptr error = std::current_exception();
            complete_error(request, error);
            throw;
        }
    }

    // port (sm_89): refuse ONE request that the Device KV residency cannot hold.
    //
    // The pool is deliberately smaller than max_context - the Host tier prices the difference - and a
    // full-attention layer reads its whole history every step, so "longer than the resident set" is a
    // property of the request, not a transient shortage. Before this existed the capacity path treated
    // it as a shortage: reserve_resident_units() paused the very lane that could never fit, and the
    // request then stayed in the paused set with nothing driving it back (measured with the pre-flight
    // disabled: `paused=1 resident=0 pending=0 tx=0` for as long as the run was observed, no
    // completion, and the Program's own refusal at storage/sequence.cpp's ensure_sequence_kv_mapped()
    // was never reached because the unit reservation refuses first). The numbers are part of the
    // message because the caller's only remedy is to shorten the prompt or to plan more residency.
    void refuse_kv_residency(const std::shared_ptr<Request>& request, std::uint32_t live_tokens,
                             std::uint32_t resident_tokens, std::uint32_t priced_tokens = 0) {
        const std::uint32_t prompt_tokens =
            request->base_plan ? request->base_plan->summary().prompt_tokens : live_tokens;
        // port (sm_89, V10-84): under read-set pricing the number that was compared is the read
        // set, so it is named separately instead of being confused with the history.
        const std::string priced_note =
            priced_tokens == 0 ? std::string()
                               : " read_set_tokens=" + std::to_string(priced_tokens);
        complete_error(request, std::make_exception_ptr(RequestError(
                                    RequestErrorKind::ContextLengthExceeded,
                                    "KV residency cannot serve this request: prompt_tokens=" +
                                        std::to_string(prompt_tokens) +
                                        " live_tokens=" + std::to_string(live_tokens) +
                                        " resident_tokens=" + std::to_string(resident_tokens) +
                                        priced_note +
                                        kv_residency_limit_note(priced_tokens != 0))));
    }

    // The sentence that names the REMAINING limit, which differs between the two arms: with the
    // page-in leg off a page on the Host tier has no reader at all, with it on the pool itself is
    // the limit - a page group holds every layer's planes and the full-attention layers address the
    // whole history on every step, so a step cannot address fewer pages than it does (V10-79).
    // P8: the leg is the default, so the leg-off arm is the run that named NINFER_KV_NO_PAGE_IN (or
    // NINFER_KV_PAGE_IN=0); the leg-on sentence now names the one thing that lifts the whole-history
    // bound - a bounded read set (NINFER_KV_EVICT_WINDOW).
    // V10-84 adds a third state: with the leg on AND a window installed, a step addresses only sink
    // + newest window + the selector's keep set, so that set (plus the unit's growth) is what the
    // pool is priced by, and the pool edge is the limit on the SUM of those sets over live lanes.
    [[nodiscard]] static std::string kv_residency_limit_note(bool read_priced = false) {
        if (read_priced) {
            return std::string(
                " (the Device KV pool is tiered and the page-in leg is on: a step addresses only "
                "the sink pages, the newest window and the selector's keep set - the set the "
                "eviction window masks - so each lane is priced by that set plus the pages its own "
                "unit writes, and the pool edge has to cover their sum. Shorten the window "
                "(NINFER_KV_EVICT_WINDOW / NINFER_KV_EVICT_KEEP_FROM_SCORES), lower "
                "max_concurrency, or give the KV leg more pages (--kv-residency-slack-mib).");
        }
        const bool leg = [] {
            // P8: same tri-state the Program uses - the leg is on by default, and either variable
            // names the opt-out.
            if (const char* opt_out = std::getenv("NINFER_KV_NO_PAGE_IN");
                opt_out != nullptr && *opt_out != '\0' && std::strtol(opt_out, nullptr, 10) != 0) {
                return false;
            }
            const char* value = std::getenv("NINFER_KV_PAGE_IN");
            if (value == nullptr || *value == '\0') { return true; }
            return std::strtol(value, nullptr, 10) != 0;
        }();
        return leg ? std::string(
                         " (the Device KV pool is tiered and the page-in leg is on: it copies a "
                         "page "
                         "back from the Host tier when a step addresses one the Device no longer "
                         "holds, but a full-attention step addresses its whole history, so the pool "
                         "still has to cover the live history itself. Bound the read set with "
                         "NINFER_KV_EVICT_WINDOW - sink + newest window + the selector's keep set "
                         "is what a step addresses then - or give the KV leg more pages "
                         "(--kv-residency-slack-mib, or a smaller max_context).")
                   : std::string(
                         " (the Device KV pool is tiered: it holds the resident part of "
                         "max_context, and the full-attention layers read the whole history every "
                         "step, so the history cannot be streamed from the Host tier). Shorten the "
                         "prompt, or leave more Device memory to the pool "
                         "(--kv-residency-slack-mib, or a smaller max_context).");
    }

    // port (sm_89): the aggregate half of the residency contract, now an admission answer rather than
    // a refusal. The Device pool is one physical page pool shared by every lane (LogicalKVPageStore's
    // physical pool, sized `kv_resident_page_groups`), and under residency a lane's whole history has
    // to stay in it for every step. The per-request bound alone therefore lets two legal requests
    // over-subscribe the pool: each then fails to reserve a page, each pauses itself
    // (reserve_resident_units), and neither is ever driven back - measured non-terminating. That is
    // why the aggregate has to be asked; what changed is the answer when it does not fit.
    // port (sm_89): an aggregate shortfall is a WAIT, not a length property.
    //
    // `kv_residency_share_fits()` answers an ADMISSION question: the Device pool is one physical page
    // pool shared by every lane, and the lanes already resident give their pages back when they
    // finish. So when the request fits the pool on its own (`request_tokens <= capacity_tokens`) the
    // only thing in its way is who else is resident right now, and the answer is to stay in pending_
    // and be offered again at the next admission generation - release_request() arms one - rather
    // than fail a request that would be legal a minute later. Measured on this box (max_context
    // 262144, max_concurrency 4, 2769 resident pages = 177,216 tokens): a 153,600-token request was
    // refused in this shape 20 times in one run (prompts 27.6k-31.8k against a live 155,264-token
    // lane, and 153,600 against a live 33,856-token lane), and every client retry met the same
    // refusal because nothing about the pool had changed.
    //
    // The refusal that remains - `request_tokens > capacity_tokens` - is the one no lane in front of
    // it can fix, and refuse_kv_residency() is the message for it.
    // Physical-memory headroom for admission. Everything else in the admission path prices
    // TOKENS, but a request's KV lands in host memory first, and spending the last of it makes a
    // CUDA call inside the prefill fail - which cuda_check turns into a process-wide fast-fail
    // (0xC0000409 measured 2026-11-25: twelve 88k-token requests, host-kv at 13.9 GiB beside a
    // 16 GiB host context, no free physical memory). A resident lane gives its bytes back when it
    // finishes, so this waits rather than refuses; the queue deadline, which now includes the
    // queued work, bounds the wait.
    static constexpr std::uint64_t kHostAdmissionReserveBytes = 4ULL * 1024 * 1024 * 1024;

    [[nodiscard]] static std::uint64_t available_physical_bytes() noexcept {
#ifdef _WIN32
        MEMORYSTATUSEX status{};
        status.dwLength = sizeof(status);
        if (GlobalMemoryStatusEx(&status) != 0) { return status.ullAvailPhys; }
#endif
        // Unknown must not refuse: a platform without the query keeps its previous behaviour.
        return UINT64_MAX;
    }

    void note_kv_residency_wait(const std::shared_ptr<Request>& request,
                                std::uint32_t prompt_tokens, std::uint32_t request_tokens,
                                const models::qwen3_5::Program::KvResidencyBudget& budget) {
        ++request->residency_waits;
        // One line per request, not one per pass: a wait that lasts is reported by the queue timeout,
        // which already names the request, its prompt and the work still ahead of it.
        if (request->residency_waits != 1) { return; }
        const auto remaining =
            std::chrono::duration<double>(request->deadline - Clock::now()).count();
        // The per-lane composition, because which lane holds what is what the operator's remedy
        // depends on (the same numbers the refusal used to carry).
        std::string lanes;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const std::uint32_t held = budget.lane_tokens[lane];
            if (held == 0) { continue; }
            if (!lanes.empty()) { lanes += '+'; }
            lanes += std::to_string(held);
        }
        std::fprintf(stderr,
                     "[admit] req#%llu waits for KV residency: prompt_tokens=%u request_tokens=%u "
                     "live_lanes=%u live_tokens=%u live_lane_tokens=%s aggregate_tokens=%u "
                     "resident_tokens=%u (queue deadline in %.0f s)\n",
                     static_cast<unsigned long long>(request->log_id()), prompt_tokens,
                                 request_tokens, budget.lane_count, budget.live_tokens,
                     lanes.empty() ? "-" : lanes.c_str(), budget.live_tokens + request_tokens,
                     budget.capacity_tokens, remaining);
        std::fflush(stderr);
    }

    // port (sm_89): refuse ONE lane whose GROWTH has nothing left to preempt.
    //
    // The aggregate admission rule prices `live + one step`, which is what makes a pair of lanes
    // admissible. A later step of the SAME round can still cross the edge (two lanes at 49.8% of the
    // pool have ~190 tokens each before their sum reaches it), and then reserve_units() fails for a
    // lane that the pool cannot grow: the only resident candidates are the requester itself and its
    // own round batch, so the preemption that would follow pauses someone who is executing this very
    // pass. The paused lane is never driven back (the other one keeps decode_ready), and the load
    // never terminates - measured as `paused=1 running=1 decode_ready=1 preemptions=48` for 120 s.
    //
    // Pausing is therefore the wrong answer exactly when no OTHER lane can give its pages back. The
    // request is refused as the length property it is (its history no longer fits beside its peers),
    // the other lanes keep running, and the engine keeps serving new requests.
    void refuse_kv_residency_growth(std::uint32_t lane, const std::shared_ptr<Request>& request) {
        // The numbers describe the state the step failed in, so they are read BEFORE the abort below
        // gives this lane's pages back.
        const auto budget = instance_.program->kv_residency_budget();
        const std::uint32_t needed_pages =
            lane < budget.lane_pages.size() ? budget.lane_pages[lane] : 0U;
        std::string lanes;
        for (std::uint32_t other = 0; other < max_concurrency_; ++other) {
            const std::uint32_t held = budget.lane_pages[other];
            if (held == 0) { continue; }
            if (!lanes.empty()) { lanes += '+'; }
            lanes += std::to_string(held);
        }
        // The native sequence owns the lane's Device KV, and completing the request does not release
        // it by itself: without the abort the pages stay mapped, the next pass still finds the pool
        // over its edge, and every other lane of the batch is refused in turn.
        if (request->sequence) {
            auto aborted                = instance_.program->abort(*request->sequence);
            request->generation_timings = aborted.timings;
            request->speculative_stats  = std::move(aborted.speculative);
        }
        // port (sm_89, V10-84): under read-set pricing growth_pages_needed is the set the step
        // addresses (plus its own growth), not the whole history, so the two arms describe
        // different quantities and say so.
        const char* const demand_note =
            instance_.program->kv_read_priced()
                ? " (growth_pages_needed is this lane's read set - the sink, the newest window, "
                  "the "
                  "selector's keep set and its own growth - in the pool's page groups; the Device "
                  "KV "
                  "pool is tiered and shared, so the sum over the live lanes has to fit it. "
                  "Shorten "
                  "the window, lower max_concurrency, or leave more Device memory to the pool "
                  "(--kv-residency-slack-mib).)"
                : " (growth_pages_needed is this lane's whole demand after the step, in the pool's "
                  "page groups; the Device KV pool is tiered and shared, and every live lane's "
                  "history has to stay in it at every step, so the growth cannot be served by "
                  "pausing a lane of the same batch. Shorten the prompt, lower max_concurrency, or "
                  "leave more Device memory to the pool (--kv-residency-slack-mib, or a "
                  "smaller "
                  "max_context).)";
        complete_error(
            request,
            std::make_exception_ptr(RequestError(
                RequestErrorKind::ContextLengthExceeded,
                "KV residency cannot grow this lane beside the lanes already resident: "
                "growth_pages_needed=" +
                    std::to_string(needed_pages) + " capacity_pages=" +
                    std::to_string(budget.capacity_pages) + " live_lanes=" +
                    std::to_string(budget.lane_count) + " live_pages=" +
                    std::to_string(budget.live_pages) + " live_lane_pages=" +
                    (lanes.empty() ? std::string("-") : lanes) + " live_tokens=" +
                    std::to_string(budget.live_tokens) + " resident_tokens=" +
                    std::to_string(budget.capacity_tokens) + demand_note)));
    }

    [[nodiscard]] std::array<bool, kMaximumConcurrency> snapshot_cancellations() const noexcept {
        std::array<bool, kMaximumConcurrency> cancelled{};
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) {
                cancelled[lane] = slots_[lane]->cancelled.load(std::memory_order_acquire);
            }
        }
        return cancelled;
    }

    bool settle_terminal_requests(HostPhaseMeasurement& boundary) {
        if (instance_.program->has_context_transaction()) { return false; }

        bool changed = false;
        for (;;) {
            std::optional<std::uint32_t> selected;
            for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                const auto& request = slots_[lane];
                if (request == nullptr || !request->terminal_reason) { continue; }
                if (!selected ||
                    request->publication_order < slots_[*selected]->publication_order) {
                    selected = lane;
                }
            }
            if (!selected) { break; }

            const std::uint32_t lane = *selected;
            const auto request       = slots_[lane];
            if (!request->is_model_finished() || request->capture_pending || !request->sequence ||
                !request->lane || request->lane->value != lane) {
                throw std::logic_error("terminal-pending request has invalid ownership");
            }
            const FinishReason reason = *request->terminal_reason;
            const auto finish_started = Clock::now();
            worker_phase_            = "finish";
            auto finished             = instance_.program->finish(*request->sequence);
            if (finished.status != ConsumeStatus::Consumed) {
                throw std::logic_error("terminal native sequence could not finish");
            }
                report_boundary_work(
                    "program finish", request->log_id(),
                std::chrono::duration<double>(Clock::now() - finish_started).count(),
                request->base_plan ? request->base_plan->summary().prompt_tokens : 0U);
            if (finished.checkpoint) {
                const auto publish_started = Clock::now();
                worker_phase_             = "publish";
                // A cache export that cannot be held is a LOST CACHE ENTRY, not a lost request.
                // Letting it escape here failed the worker and turned every later request into a
                // 500 (measured 2026-11-25: a 1.35 GB record under memory pressure reported
                // "worker failed: bad allocation | phase=publish to cache", and the service
                // answered 500 from then on).
                try {
                    resources_.publish(*instance_.program, request->continuation_owner,
                                       *finished.checkpoint, monotonic_ms());
                    // The finish-time checkpoint is the one a later request can actually match: it
                    // covers the whole prompt plus everything generated. Without this hand-off the
                    // record is retained for pause/replay only and the external cache never sees a
                    // byte, because publish_to_cache() used to be reachable from the active-capture
                    // path alone.
                    if (request->base_plan) {
                        worker_phase_ = "publish to cache";
                        resources_.publish_to_cache(*instance_.program, *request->base_plan,
                                                    *finished.checkpoint,
                                                    request->continuation_owner);
                    }
                } catch (const std::bad_alloc&) {
                    std::fprintf(stderr,
                                 "[publish] cache export skipped: host allocation failed "
                                 "(the request keeps its result)\n");
                    std::fflush(stderr);
                } catch (const std::exception& error) {
                    std::fprintf(stderr, "[publish] cache export skipped: %s\n", error.what());
                    std::fflush(stderr);
                }
                report_boundary_work(
                    "publish", request->log_id(),
                    std::chrono::duration<double>(Clock::now() - publish_started).count(),
                    request->base_plan ? request->base_plan->summary().prompt_tokens : 0U);
            } else if (std::getenv("NINFER_PUBLISH_PROBE") != nullptr) {
                // A finish without a checkpoint explains a cache that never receives the prefix: the
                // whole publish path below is skipped, and the only symptom is a miss on the next
                // request. Print it rather than leaving it silent.
                std::fprintf(stderr,
                             "[publish] finish produced NO checkpoint | req %llu | prompt %u\n",
                             static_cast<unsigned long long>(request->log_id()),
                             request->base_plan ? request->base_plan->summary().prompt_tokens : 0U);
                std::fflush(stderr);
            }
            const auto release_started = Clock::now();
            worker_phase_             = "resources finish";
            resources_.finish(*instance_.program, request->continuation_owner, *request->base_plan,
                              request->publication_order);
            request->continuation_owner = 0;
            request->generation_timings = finished.timings;
            request->speculative_stats  = std::move(finished.speculative);
            request->terminal_reason.reset();
            report_boundary_work(
                "resources finish", request->log_id(),
                std::chrono::duration<double>(Clock::now() - release_started).count(),
                request->base_plan ? request->base_plan->summary().prompt_tokens : 0U);

            finish_engine_phase(boundary, EngineHostPhase::Boundary);
            const auto complete_started = Clock::now();
            complete_success(request, reason);
            report_boundary_work(
                "complete", request->log_id(),
                std::chrono::duration<double>(Clock::now() - complete_started).count(),
                request->base_plan ? request->base_plan->summary().prompt_tokens : 0U);
            boundary = begin_host_phase();
            changed  = true;
        }
        return changed;
    }

    void cancel_active_requests(const std::array<bool, kMaximumConcurrency>& cancelled_at_boundary,
                                HostPhaseMeasurement& boundary) {
        if (instance_.program->has_context_transaction()) { return; }
        bool changed = false;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto request = slots_[lane];
            if (request == nullptr || !cancelled_at_boundary[lane]) { continue; }
            if (request->capture_pending) { continue; }
            if (!request->sequence || !request->lane || request->lane->value != lane) {
                throw std::logic_error("active cancellation has no sequence binding");
            }
            (void)request->output.preview_terminal(FinishReason::Cancelled);
            auto aborted                = instance_.program->abort(*request->sequence);
            request->generation_timings = aborted.timings;
            request->speculative_stats  = std::move(aborted.speculative);
            finish_engine_phase(boundary, EngineHostPhase::Boundary);
            append_output(request, request->output.commit_preview());
            complete_success(request, FinishReason::Cancelled);
            boundary = begin_host_phase();
            changed  = true;
        }
    }

    // A queued request's deadline comes from its own length, which cannot know what stands in front
    // of it: at submit the engine already stretches the pending timeout by the prompt (measured
    // `[queue] deadline extended 29975 ms -> 163846 ms (prompt 163687, budget 8)` for the long
    // request itself), but a 67-token probe has no length to stretch, so it keeps the launch's
    // pending timeout - 30 s here - and expires while the lane ahead still has a minute and a half
    // of work. What it cannot see, the engine can: the Program already reports how many prompt tokens
    // a lane still has to prefill (Q67/S1), which is a measurement rather than an estimate of queue
    // depth. The extension is bounded twice: at most kMaxQueueExtensions per request, and never past
    // the 15-minute ceiling the service itself uses.
    static constexpr std::uint32_t kMaxQueueExtensions = 4;
    // Queued work is priced at the measured long-prefill rate (1.4-1.6k tok/s on this box; 1200
    // is the conservative side, so a deadline never expires a request the engine is still working
    // towards). The ceiling matches the 15-minute bound the deadline logic already enforces.
    static constexpr double kPrefillTokensPerSecond   = 1200.0;
    static constexpr double kQueueDrainCeilingSeconds = 900.0;

    // How recently a request must have finished for the queue to count as "moving". Five seconds is
    // several service windows on this box (a 400-token request at 4 lanes finishes in ~2 s), so a
    // stalled engine ages past it quickly and the ordinary expiry fires again.
    static constexpr std::uint64_t kProgressStallMs = 5000;
    // monotonic_ms() of the last completed request; 0 means none yet (a fresh engine is not
    // "moving" until something finishes, so the first requests are judged by the lane estimate).
    std::atomic<std::uint64_t> last_completion_ms_{0};

    void extend_pending_deadlines_locked(Clock::time_point now) {
        if (pending_.empty()) { return; }
        // Off switch for the A/B: the comparison is "the same probe behind the same prefill, one
        // expires and one is admitted", and it has to be the same binary for that to mean anything.
        if (const char* flag = std::getenv("NINFER_QUEUE_EXTEND"); flag != nullptr &&
            (std::strcmp(flag, "0") == 0 || std::strcmp(flag, "off") == 0)) {
            return;
        }
        // A lane that is decoding instead of prefilling still owns its lane for a whole response,
        // so it is charged a flat few seconds rather than a fraction of a budget this function
        // cannot price.
        constexpr double kDecodingLaneSeconds = 5.0;
        double lane_ahead                     = 0.0;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& other = slots_[lane];
            if (!other || !other->sequence) { continue; }
            const std::uint32_t left = instance_.program->prefill_remaining(lane);
            lane_ahead += left != 0 ? static_cast<double>(left) / kPrefillTokensPerSecond
                                    : kDecodingLaneSeconds;
        }
        if (lane_ahead < 1.0 && pending_.empty()) { return; }
        const bool probe = std::getenv("NINFER_QUEUE_PROBE") != nullptr;
        // A queue timeout has to mean "the engine is stuck", not "it was busy while you waited".
        // Measured on this box: eight 800-token requests need about 32 s of service, so the third
        // batch was two seconds from admission when the 30 s pending timeout answered 503 at 30.48 s
        // - the queue was healthy the whole time (waiting rose to 66 and drained to zero). While the
        // engine keeps completing requests a deadline is therefore pushed out; once it stops, the
        // stamp ages and the ordinary expiry fires again.
        const auto now_ms       = std::chrono::duration_cast<std::chrono::milliseconds>(
                                      now.time_since_epoch())
                                      .count();
        const auto last_done_ms = static_cast<long long>(
            last_completion_ms_.load(std::memory_order_relaxed));
        const bool progressing =
            last_done_ms != 0 && now_ms >= last_done_ms && now_ms - last_done_ms < kProgressStallMs;
        if (lane_ahead < 1.0 && !progressing) { return; }
        // A request's wait is the lanes' remaining work PLUS every request ahead of it in the
        // queue. Charging only the lanes made the 10th of sixteen 88k-token requests believe it
        // had 240 s ahead when it had nine whole prefills (~9 min), so it expired at 340 s with a
        // 503 while the engine was making steady progress (2026-11-25).
        double queued_ahead = 0.0;
        for (auto& request : pending_) {
            const double ahead = lane_ahead + queued_ahead;
            queued_ahead += static_cast<double>(request->prompt_summary.prompt_tokens) /
                            kPrefillTokensPerSecond;
            if (request->queue_extensions >= kMaxQueueExtensions) { continue; }
            if (ahead < 1.0) { continue; }
            // 1.5x the remaining work, with a 30 s floor: a request that is about to be admitted
            // must not be expired in the step before its lane frees up.
            const auto extension = std::chrono::milliseconds(
                std::max<long long>(30000, static_cast<long long>(ahead * 1500.0)));
            auto next = std::min(now + extension, request->submitted + std::chrono::minutes(15));
            if (progressing) {
                // The queue is moving: never expire a request the engine is still working towards.
                // One service window per extension, so a trickle of progress cannot keep a request
                // alive forever - kMaxQueueExtensions bounds it (4 x 30 s past the base timeout).
                next = std::min(std::max(next, now + std::chrono::seconds(30)),
                                request->submitted + std::chrono::minutes(15));
            }
            if (next <= request->deadline) { continue; }
            ++request->queue_extensions;
            request->deadline = next;
            if (probe) {
                std::fprintf(stderr,
                             "[queue] req#%llu deadline extended: %.0f s of work ahead, "
                             "progressing=%d, +%lld ms (extension %u)\n",
                             static_cast<unsigned long long>(request->log_id()), ahead,
                             progressing ? 1 : 0, static_cast<long long>(extension.count()),
                             static_cast<unsigned>(request->queue_extensions));
                std::fflush(stderr);
            }
        }
    }

    [[nodiscard]] bool expire_pending_requests() {
        std::vector<std::shared_ptr<Request>> cancelled;
        std::vector<std::shared_ptr<Request>> expired;
        bool have_pending = false;
        {
            std::lock_guard lock(queue_mutex_);
            const auto now = Clock::now();
            extend_pending_deadlines_locked(now);
            for (auto it = pending_.begin(); it != pending_.end();) {
                if ((*it)->cancelled.load(std::memory_order_acquire)) {
                    cancelled.push_back(*it);
                    it = pending_.erase(it);
                } else if (now >= (*it)->deadline) {
                    expired.push_back(*it);
                    it = pending_.erase(it);
                } else {
                    ++it;
                }
            }
            have_pending = !pending_.empty();
        }
        try {
            for (const auto& request : cancelled) { complete_cancelled(request); }
            for (const auto& request : expired) {
                complete_error(request,
                               std::make_exception_ptr(RequestError(
                                   RequestErrorKind::QueueTimeout,
                                   "inference request expired while waiting for admission")));
            }
        } catch (...) {
            const std::exception_ptr error = std::current_exception();
            for (const auto& request : cancelled) { complete_error(request, error); }
            for (const auto& request : expired) { complete_error(request, error); }
            throw;
        }
        if (!cancelled.empty() || !expired.empty()) { request_admission_check(); }
        return have_pending;
    }

    void commit_pending(PendingBatch&& pending, std::span<const std::uint32_t> lane_indices,
                        bool decode_round,
                        const std::array<bool, kMaximumConcurrency>& cancelled_at_unit_start) {
        EnginePhaseScope phase(*this, EngineHostPhase::CommitOutput);
        const std::size_t row_count = lane_indices.size();
        if (row_count == 0 || row_count != pending.row_count() || pending.row_stride() == 0 ||
            (!pending.row_counts().empty() && pending.row_counts().size() != row_count) ||
            pending.tokens().size() < static_cast<std::size_t>(pending.row_stride()) * row_count) {
            (void)instance_.program->abort_pending(std::move(pending));
            throw std::logic_error("pending batch returned an invalid ragged layout");
        }
        const auto service_ns = execution_elapsed_ns(pending.execution_timing());

        std::array<CommitDecision, kMaximumConcurrency> decisions{};
        std::array<FinishReason, kMaximumConcurrency> finish_reasons{};
        std::array<ContinuationAction, kMaximumConcurrency> continuations{};
        std::array<std::size_t, kMaximumConcurrency> generated_sizes{};
        std::array<bool, kMaximumConcurrency> cancelled{};
        std::array<bool, kMaximumConcurrency> failed{};
        bool generated_staged = false;
        std::array<std::shared_ptr<Request>, kMaximumConcurrency> terminal_requests{};
        std::array<FinishReason, kMaximumConcurrency> terminal_reasons{};
        std::size_t terminal_count    = 0;
        const auto rollback_generated = [&]() {
            for (std::size_t row = 0; row < row_count; ++row) {
                const auto& request = slots_[lane_indices[row]];
                if (request != nullptr) { request->output.discard_preview(); }
                if (generated_staged && request != nullptr &&
                    request->generated.size() >= generated_sizes[row]) {
                    request->generated.resize(generated_sizes[row]);
                }
            }
            generated_staged = false;
        };
        try {
            for (std::size_t row = 0; row < row_count; ++row) {
                const std::uint32_t lane = lane_indices[row];
                const auto& request      = slots_[lane];
                if (request == nullptr || !request->sequence || !request->lane ||
                    request->lane->value != lane || !request->budget) {
                    throw std::logic_error("pending row has no active Engine request");
                }
                if (decode_round) { ++request->host_timing.decode_rounds; }
                cancelled[row] = cancelled_at_unit_start[lane];
                failed[row]    = !cancelled[row] && pending.constraint_failed(row);
                const std::int32_t raw_count =
                    pending.row_counts().empty() ? 1 : pending.row_counts()[row];
                if (raw_count <= 0 || raw_count > static_cast<std::int32_t>(pending.row_stride())) {
                    throw std::logic_error("pending row has an invalid licensed extent");
                }
                const std::uint32_t count = static_cast<std::uint32_t>(raw_count);
                const auto row_tokens     = pending.tokens().subspan(row * pending.row_stride(),
                                                                     static_cast<std::size_t>(count));
                generated_sizes[row]      = request->generated.size();
                if (failed[row]) {
                    decisions[row] = CommitDecision{.terminal = true, .failed = true};
                    continue;
                }
                if (cancelled[row]) {
                    (void)request->output.preview_terminal(FinishReason::Cancelled);
                    decisions[row] = CommitDecision{
                        .accepted_tokens = 0,
                        .terminal        = true,
                        .cancelled       = true,
                    };
                    finish_reasons[row] = FinishReason::Cancelled;
                    continue;
                }
                OutputDecision decision;
                if (std::getenv("NINFER_GRAMMAR_DEBUG") != nullptr) {
                    std::fprintf(stderr, "[grammar] matcher got | row %zu | %u token(s):", row, count);
                    for (std::size_t index = 0; index < row_tokens.size() && index < 8; ++index) {
                        std::fprintf(stderr, " %lld", static_cast<long long>(row_tokens[index]));
                    }
                    std::fprintf(stderr, "\n");
                    std::fflush(stderr);
                }
                try {
                    decision = request->output.preview_model(
                        row_tokens, request->budget->remaining(), request->budget->limit_reason());
                } catch (const text::GrammarViolation& violation) {
                    // This request's token set and the matcher disagree - a speculative draft the
                    // mask never saw, measured 2026-11-21 as 7 of 8 drafts. It is the request's
                    // outcome, not the engine's: fail this row and keep every other lane serving.
                    // Escaping to the worker's catch used to set failed_ and answer 503 to every
                    // later request, which is what made a single constrained turn look like an
                    // engine outage.
                    //
                    // The row is marked on the batch first: commit() only accepts a failed decision
                    // for a row the program already considers constraint-dead, and the program's own
                    // mark comes from the mask, which is exactly the side that accepted these tokens.
                    std::fprintf(stderr, "[grammar] request %llu failed: %s\n",
                                 static_cast<unsigned long long>(request->log_id()), violation.what());
                    std::fflush(stderr);
                    pending.mark_constraint_failed(row, true);
                    failed[row]    = true;   // expected disposition: FailedReleased (see below)
                    decisions[row] = CommitDecision{.terminal = true, .failed = true};
                    continue;
                }
                if (decision.accepted_tokens == 0 || decision.accepted_tokens > count ||
                    (!decision.finished() && decision.accepted_tokens != count) ||
                    (decision.finished() && decision.continuation != ContinuationAction::Decode) ||
                    (decision.prefix_execution_split_after &&
                     (*decision.prefix_execution_split_after == 0 ||
                      *decision.prefix_execution_split_after > decision.accepted_tokens))) {
                    throw std::logic_error("output policy returned an invalid licensed prefix");
                }
                decisions[row] = CommitDecision{
                    .accepted_tokens              = decision.accepted_tokens,
                    .terminal                     = decision.finished(),
                    .cancelled                    = false,
                    .prefix_execution_split_after = decision.prefix_execution_split_after,
                };
                finish_reasons[row] = decision.finish_reason;
                continuations[row]  = decision.continuation;
            }
            generated_staged = true;
            for (std::size_t row = 0; row < row_count; ++row) {
                const std::uint32_t accepted = decisions[row].accepted_tokens;
                if (accepted == 0) { continue; }
                const auto& request = slots_[lane_indices[row]];
                if (request->generated.size() > request->generated.capacity() ||
                    accepted > request->generated.capacity() - request->generated.size()) {
                    throw std::logic_error("admission did not reserve generated-token capacity");
                }
                const auto first = pending.tokens().begin() +
                                   static_cast<std::ptrdiff_t>(row * pending.row_stride());
                request->generated.insert(request->generated.end(), first,
                                          first + static_cast<std::ptrdiff_t>(accepted));
                // Token-level trace for the replay question (E9). Text cannot carry it: the chat
                // template re-renders and detokenisation both move the bytes, which is why an
                // HTTP-level comparison contradicted a bit-identical PPL run. The sampled ids do
                // not have that problem - two runs agree or they do not, token by token.
                if (const char* dump_path = std::getenv("NINFER_TOKEN_DUMP")) {
                    static std::FILE* dump = std::fopen(dump_path, "ab");
                    if (dump != nullptr) {
                        for (std::size_t k = 0; k < accepted; ++k) {
                            std::fprintf(dump, "%llu %d\n",
                                         static_cast<unsigned long long>(request->id),
                                         static_cast<int>(*(first + static_cast<std::ptrdiff_t>(k))));
                        }
                        std::fflush(dump);
                    }
                }
            }
        } catch (...) {
            const std::exception_ptr error = std::current_exception();
            rollback_generated();
            const auto discarded = instance_.program->abort_pending(std::move(pending));
            if (discarded.status != ConsumeStatus::Consumed) {
                throw std::logic_error("failed pending batch could not be discarded");
            }
            std::rethrow_exception(error);
        }

        std::optional<typename ModelContract::CommitResult> committed_storage;
        try {
            phase.pause_range();
            ProgramCallScope program_call(*this);
            auto committed = instance_.program->commit(
                std::move(pending), std::span<const CommitDecision>(decisions.data(), row_count),
                CommitObservation::ReleasedRowsOnly, &program_call.failed_timing());
            program_call.finish(committed.timing);
            committed_storage.emplace(std::move(committed));
            phase.resume_range();
        } catch (...) {
            rollback_generated();
            throw;
        }
        generated_staged = false;
        auto& committed  = *committed_storage;
        if (committed.row_count != row_count) {
            throw std::logic_error("Runtime commit result is not row aligned");
        }
        for (std::size_t row = 0; row < row_count; ++row) {
            const CommitDisposition expected =
                failed[row]               ? CommitDisposition::FailedReleased
                : cancelled[row]          ? CommitDisposition::CancelledReleased
                : decisions[row].terminal ? CommitDisposition::Finishable
                                          : CommitDisposition::Active;
            if (committed.rows[row].disposition != expected) {
                throw std::logic_error("Runtime commit row disposition is invalid");
            }
            if (committed.capture_ready[row] &&
                (decode_round || expected != CommitDisposition::Active)) {
                throw std::logic_error("Runtime exposed a capture outside a committed Begin row");
            }
        }
        const bool terminal_in_batch = std::any_of(
            decisions.begin(), decisions.begin() + static_cast<std::ptrdiff_t>(row_count),
            [](const CommitDecision& decision) { return decision.terminal; });

        for (std::size_t row = 0; row < row_count; ++row) {
            const auto& request  = slots_[lane_indices[row]];
            auto& observed       = request->speculative_stats;
            const auto& counters = committed.rows[row].speculative_counters;
            cumulative_stats_.speculative_rounds += counters.rounds - observed.rounds;
            cumulative_stats_.speculative_draft_tokens +=
                counters.drafted_tokens - observed.drafted_tokens;
            cumulative_stats_.speculative_accepted_tokens +=
                counters.accepted_tokens - observed.accepted_tokens;
            cumulative_stats_.speculative_fallback_steps +=
                counters.fallback_steps - observed.fallback_steps;
            observed.rounds          = counters.rounds;
            observed.drafted_tokens  = counters.drafted_tokens;
            observed.accepted_tokens = counters.accepted_tokens;
            observed.fallback_steps  = counters.fallback_steps;
            if (cancelled[row] || failed[row]) {
                request->generation_timings = committed.rows[row].timings;
                request->speculative_stats  = std::move(committed.rows[row].speculative);
            }
        }

        if (decode_round) {
            ++cumulative_stats_.decode_rounds;
            cumulative_stats_.decode_row_rounds += row_count;
            for (std::size_t row = 0; row < row_count; ++row) {
                if (!cancelled[row]) {
                    cumulative_stats_.committed_decode_tokens += decisions[row].accepted_tokens;
                    slots_[lane_indices[row]]->committed_decode_tokens +=
                        decisions[row].accepted_tokens;
                }
            }
        }

        try {
            for (std::size_t row = 0; row < row_count; ++row) {
                const std::uint32_t lane     = lane_indices[row];
                const auto& request          = slots_[lane];
                const std::uint32_t accepted = decisions[row].accepted_tokens;
                if (failed[row]) {
                    complete_error(request, std::make_exception_ptr(
                                                RequestError(RequestErrorKind::ConstraintDeadEnd,
                                                             "grammar has no valid next token")));
                    continue;
                }
                if (!cancelled[row]) { request->budget->commit(accepted); }
                record_service(request->id, accepted != 0, service_ns);
                update_recovery(request, cancelled[row]);
                auto published = request->output.commit_preview();
                auto timing    = record_committed_output(request, accepted);
                append_output(request, std::move(published), std::move(timing), &phase);
                if (request->output.output_error()) {
                    if (committed.capture_ready[row]) {
                        instance_.program->skip_capture(*request->sequence);
                        committed.capture_ready[row] = false;
                    }
                    if (fail_output(request)) { continue; }
                }
                if (decisions[row].terminal) {
                    if (cancelled[row]) {
                        terminal_requests[terminal_count] = request;
                        terminal_reasons[terminal_count]  = finish_reasons[row];
                        ++terminal_count;
                    } else {
                        request->model_state     = EngineRequestState::ModelFinished;
                        request->terminal_reason = finish_reasons[row];
                    }
                } else if (committed.capture_ready[row]) {
                    if (!request->is_prefilling()) {
                        throw std::logic_error("prompt-frontier capture lost its prefill owner");
                    }
                    const EngineRequestState post_capture_state =
                        continuations[row] == ContinuationAction::ApplyTargetControl
                            ? EngineRequestState::ControlReady
                            : EngineRequestState::DecodeReady;
                    // A prompt-frontier capture exports the WHOLE prefix to the Host tier before the
                    // first token can be produced: 163k tokens = 2,558 pages = 2.4 GiB, measured as
                    // TTFT 1.9 s (one run 3.4 s) against 0.57-0.68 s with the checkpoint cache off,
                    // reported by the operator as "TTFT is always ~5 s".
                    //
                    // The same prefix is published again when the request FINISHES (the checkpoint
                    // handed back by finish()), and that is the record a later turn or a restarted
                    // engine actually reuses - so for a turn-by-turn conversation the prompt
                    // frontier capture buys nothing on the critical path and only delays the first
                    // token. NINFER_CAPTURE_PROMPT_FRONTIER=0 skips it; the finish-time publish is
                    // untouched, so cross-process reuse still gets its record.
                    if (terminal_in_batch || !capture_prompt_frontier_) {
                        instance_.program->skip_capture(*request->sequence);
                        request->model_state = post_capture_state;
                        if (!terminal_in_batch) {
                            std::fprintf(stderr,
                                         "[ckpt] prompt-frontier capture skipped | frontier %u | "
                                         "finish-time publish keeps the record\n",
                                         request->base_plan->summary().prompt_tokens);
                            std::fflush(stderr);
                        }
                    } else {
                        reserve_active_capture(request, post_capture_state);
                    }
                    committed.capture_ready[row] = false;
                    if (!request->capture_pending) {
                        request->model_state =
                            continuations[row] == ContinuationAction::ApplyTargetControl
                                ? EngineRequestState::ControlReady
                                : EngineRequestState::DecodeReady;
                    }
                } else {
                    request->model_state =
                        continuations[row] == ContinuationAction::ApplyTargetControl
                            ? EngineRequestState::ControlReady
                            : EngineRequestState::DecodeReady;
                }
            }
        } catch (...) {
            phase.finish();
            for (std::size_t index = 0; index < terminal_count; ++index) {
                complete_success(terminal_requests[index], terminal_reasons[index]);
            }
            throw;
        }
        phase.finish();
        for (std::size_t index = 0; index < terminal_count; ++index) {
            complete_success(terminal_requests[index], terminal_reasons[index]);
        }
    }

    // Boundary attribution. The done line shows a "boundary" phase of 1.5 s on an agent-shaped turn
    // (163k prompt, +30 tokens) against 189 ms when the same prompt is asked twice, and the phase
    // total alone cannot say which of the three boundary jobs it is. These lines answer that -
    // capture staging, finish-time publish, fork - and stay silent unless they are worth printing.
    static void report_boundary_work(const char* what, std::uint64_t id, double seconds,
                                     std::uint32_t pages) {
        if (seconds < 0.02) { return; }
        std::fprintf(stderr, "[ckpt] %s | req %llu | %u page(s) | %.0f ms\n", what,
                     static_cast<unsigned long long>(id), pages, seconds * 1000.0);
        std::fflush(stderr);
    }

    void reserve_active_capture(const std::shared_ptr<Request>& request, EngineRequestState next) {
        request->model_state        = next;
        request->capture_pending    = true;
        request->post_capture_state = next;
        if (instance_.program->has_context_transaction()) { return; }
        if (scheduler_.progress_owner()) {
            instance_.program->skip_capture(*request->sequence);
            request->capture_pending = false;
            return;
        }
        bool started = instance_.program->start_capture(*request->sequence);
        if (!started) {
            instance_.program->skip_capture(*request->sequence);
            request->capture_pending = false;
            return;
        }
        context_owner_ = request;
    }

    bool prepare_semantic_capture(const std::shared_ptr<Request>& request) {
        auto& decision = capture_decisions_[request->lane->value];
        if (decision && decision->sequence != *request->sequence) { decision.reset(); }
        if (instance_.program->capture_is_input(*request->sequence) && !decision) {
            (void)resources_.recycle_input(*instance_.program, request->continuation_owner);
        }
        while (const auto capture = instance_.program->prepare_capture(*request->sequence)) {
            if (scheduler_.progress_owner()) {
                instance_.program->skip_capture(*request->sequence);
                decision.reset();
                continue;
            }
            if (capture->reserved) {
                decision.reset();
                return true;
            }
            if (instance_.program->has_context_transaction()) { return false; }
            const auto admission = resources_.capture_admission(
                *instance_.program, request->continuation_owner, *request->base_plan,
                capture->frontier,
                instance_.program->capture_is_input(*request->sequence) &&
                    !capture->shortage.main_kv_pages && !capture->shortage.backend_kv_pages);
            if (!decision || decision->frontier != capture->frontier) {
                decision.emplace(
                    CaptureDecision{*request->sequence, capture->frontier,
                                    resources_.begin_reclaim(*instance_.program, admission)});
            }
            const auto usage = instance_.program->physical_usage();
            if (capture->host_bytes && capture->host_bytes <= usage.capacity.host_bytes) {
                if (capture->host_bytes > usage.capacity.host_bytes - usage.occupied.host_bytes) {
                    (void)instance_.program->release_redundant_host({}, request->sequence);
                }
                if (const auto victims = resources_.host_victims(
                        *instance_.program, capture->host_bytes, std::nullopt, {}, {}, admission,
                        &decision->reclaim)) {
                    resources_.commit_host_victims(*instance_.program, *victims, decision->reclaim);
                    if (!victims->empty()) { request_admission_check(); }
                    if (const auto retry = instance_.program->prepare_capture(*request->sequence);
                        retry && retry->reserved) {
                        decision.reset();
                        return true;
                    }
                }
            }
            const auto progress =
                resources_.reclaim(*instance_.program, capture->shortage, {}, decision->reclaim);
            if (progress == ReclaimProgress::Transferring) { return false; }
            if (progress == ReclaimProgress::Changed) { continue; }
            instance_.program->skip_capture(*request->sequence);
            decision.reset();
        }
        decision.reset();
        return true;
    }

    void
    resolve_prefill_progress(const std::shared_ptr<Request>& request,
                             typename ModelContract::PrefillProgress&& progress,
                             const std::array<bool, kMaximumConcurrency>& cancelled_at_unit_start) {
        EnginePhaseScope phase(*this, EngineHostPhase::CommitOutput);
        ++cumulative_stats_.host_work.prefill_units;
        const auto service_ns = execution_elapsed_ns(progress.timing);
        ++request->host_timing.prefill_units;
        cumulative_stats_.computed_prefill_tokens += progress.processed_prompt_tokens;
        if (!request->admitted_begin) {
            throw std::logic_error("prefill progress has no committed admission summary");
        }
        const BeginSummary& begin = *request->admitted_begin;
        if (begin.reused_prompt_tokens > begin.prompt_tokens) {
            throw std::logic_error("admitted prefix exceeds its prompt");
        }
        const std::uint32_t suffix_tokens = begin.prompt_tokens - begin.reused_prompt_tokens;
        if (request->computed_prompt_tokens > suffix_tokens ||
            progress.processed_prompt_tokens > suffix_tokens - request->computed_prompt_tokens) {
            throw std::logic_error("prefill unit exceeded the admitted prompt suffix");
        }
        const auto computed_begin = begin.reused_prompt_tokens + request->computed_prompt_tokens;
        request->computed_prompt_tokens += progress.processed_prompt_tokens;
        record_service(request->id, progress.processed_prompt_tokens != 0, service_ns);
        if (!request->cancelled.load(std::memory_order_acquire)) {
            resources_.observe_committed(*request->base_plan, request->id, computed_begin,
                                         computed_begin + progress.processed_prompt_tokens);
        }
        if (progress.complete && request->computed_prompt_tokens != suffix_tokens) {
            throw std::logic_error("completed prefill did not reach the admitted prompt frontier");
        }
        if (progress.processed_prompt_tokens != 0) { publish_prompt_progress(request); }
        if (progress.capture_ready) {
            if (progress.complete || progress.pending) {
                throw std::logic_error("prefill capture offer overlaps prompt completion");
            }
            reserve_active_capture(request, EngineRequestState::Prefill);
            return;
        }
        if (!progress.complete) { return; }
        if (!request->lane || !progress.pending) {
            throw std::logic_error("completed prefill has no lane or pending token");
        }
        if (!request->admitted_begin || progress.summary != *request->admitted_begin) {
            throw std::logic_error("runtime Begin summary differs from committed admission");
        }
        const std::uint32_t lane = request->lane->value;
        request->begin           = progress.summary;
        const std::array<std::uint32_t, 1> lanes{lane};
        phase.finish();
        commit_pending(std::move(*progress.pending), lanes, false, cancelled_at_unit_start);
        progress.pending.reset();
    }

    void run_prefill_step(std::uint32_t lane,
                          const std::array<bool, kMaximumConcurrency>& cancelled_at_unit_start) {
        nvtx::ScopedRange prefill_range(nvtx::Name::Prefill, nvtx::Category::Prefill);
        EnginePhaseScope setup(*this, EngineHostPhase::CommitOutput);
        const auto request = slots_[lane];
        if (request == nullptr || !request->is_prefilling() || request->capture_pending) {
            throw std::logic_error("staged prefill lane has invalid request state");
        }
        if (!request->sequence) {
            throw std::logic_error("prefill request has no sequence handle");
        }
        setup.finish();
        RoundMasks masks;
        masks.outputs[0] = &request->output;
        auto* provider   = request->output.constrained() ? &masks : nullptr;
        ProgramCallScope program_call(*this);
        auto progress = instance_.program->advance_prefill(*request->sequence,
                                                           &program_call.failed_timing(), provider);
        program_call.finish(progress.timing);
        if (!request->first_output_timing) {
            record_execution_work(request->prefill_work, progress.timing);
        }
        resolve_prefill_progress(request, std::move(progress), cancelled_at_unit_start);
        publish_runtime_stats();
    }

    void ensure_base_plan(const std::shared_ptr<Request>& request) {
        if (!request->base_plan) {
            request->base_plan.emplace(instance_.program->plan_request(std::move(request->prompt),
                                                                       request->options.execution));
            const auto& summary = request->base_plan->summary();
            request->budget.emplace(summary.effective_output_tokens,
                                    summary.effective_limit_reason);
            request->generated.reserve(summary.effective_output_tokens);
        }
    }

    // 13.5: the state the dispatch reads. Merging and "hold admission first" are one decision now -
    // a merged pass needs the lanes bound before it can carry them - with the env switches kept as
    // explicit overrides for an A/B arm.
    [[nodiscard]] bool prefill_merge_active() const noexcept {
        return prefill_merge_enabled_.load(std::memory_order_acquire);
    }
    [[nodiscard]] bool prefill_hold_active() const noexcept {
        return prefill_admit_first_.load(std::memory_order_acquire) || prefill_merge_active();
    }

    // The bus's accepted executor for "prefill_merge". Refused when the state is pinned by the
    // environment: a run that says "merge off" is an A/B arm, and the controller must not merge into
    // it halfway through.
    bool apply_prefill_merge(double value) {
        if (prefill_merge_pinned_) { return false; }
        const bool enabled = value > 0.5;
        const bool previous = prefill_merge_enabled_.exchange(enabled, std::memory_order_acq_rel);
        if (previous != enabled) {
            std::fprintf(stderr, "[ctl] engine prefill merge: %s (controller)\n",
                         enabled ? "on" : "off");
            std::fflush(stderr);
        }
        return true;
    }

    // v8 9.3: the lane limit in force - the controller may hold it below the configured maximum.
    [[nodiscard]] std::uint32_t effective_concurrency() const noexcept {
        const std::uint32_t active = active_concurrency_.load(std::memory_order_acquire);
        return active < max_concurrency_ ? active : max_concurrency_;
    }

    // The bus's accepted executor for "concurrency" (v8 12.2). The bounds are the configured
    // maximum: the engine never admits beyond what its buffers were sized for, and never below
    // one lane. The step from one value to the next costs nothing - it takes effect at the next
    // admission decision, so cost_ms=0 in the spec.
    bool apply_active_concurrency(double value) {
        const auto lanes = static_cast<std::uint32_t>(value);
        if (lanes < 1U || lanes > max_concurrency_) {
            std::fprintf(stderr, "[ctl] engine concurrency refused: %u outside [1,%u]\n", lanes,
                         max_concurrency_);
            std::fflush(stderr);
            return false;
        }
        const std::uint32_t previous =
            active_concurrency_.exchange(lanes, std::memory_order_acq_rel);
        if (previous != lanes) {
            std::fprintf(stderr, "[ctl] engine concurrency: %u -> %u (controller)\n", previous,
                         lanes);
            std::fflush(stderr);
        }
        return true;
    }

    [[nodiscard]] std::optional<std::uint32_t> free_lane() const noexcept {
        // v8 9.3: admission is bounded by the ACTIVE limit - the controller may have serialized
        // serving. Lanes above the limit stay allocated (in-flight requests there keep draining
        // to completion; the round/prefill membership still scans every configured lane), but no
        // new request is admitted into one.
        const std::uint32_t limit = effective_concurrency();
        for (std::uint32_t lane = 0; lane < limit; ++lane) {
            if (!slots_[lane] && (!materializing_ || materializing_->destination.value != lane)) {
                return lane;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] bool resident_empty() const noexcept {
        return !materializing_ &&
               std::all_of(slots_.begin(), slots_.end(), [](const auto& p) { return !p; });
    }

    // 13.5 (step 5): the live counterpart of the idle the worker itself waits for. The
    // published RuntimeStats gauges are a boundary-consistent snapshot, and a long merged
    // prefill does not publish while it runs, so a caller can read all nine gauges as zero
    // with a four-way batch in flight - the double read in Engine::resize_* then sailed
    // straight through and the KV shrink ran concurrently with the decode it should have
    // refused (measured on the e2 artifact: the in-flight batch diverged after the resize).
    // The worker holds execution_mutex_ for the whole of an execution round and drops it to
    // idle on the condition variable, so the lock itself answers the question: when the
    // try-lock fails the worker is inside a round and the caller is refused NOW instead of
    // parking behind a multi-second prefill. When it succeeds, the queue state is read under
    // both locks in the worker's own order.
public:
    // v8 12.5 step 4 + 13.5-5: THE idle contract for a reallocation, in ONE place, for both the
    // expert pool and the KV pages. Engine::resize_expert_pool / resize_kv_pages forward external
    // callers here, and the worker's own scheduling point (apply_pending_pool_resize) walks the
    // SAME code, so there is no second, weaker copy to drift. resize_mutex_ serializes the entry
    // points.
    //
    // The window itself (open #1, measured on the e2 artifact 2026-11-05 as one refusal and one
    // penetration in the same run): the two-read form only *observed* idleness, it never held it.
    // Executing between the check and the mutation needs execution_mutex_ - the worker's round
    // takes it - so a submission arriving in that gap could be admitted and start binding while
    // the pool was being rebuilt. The contract is now: try the round lock (refuse immediately if a
    // round is running, never park behind a multi-second prefill), and on success HOLD it across
    // the verdict, the device synchronize and the mutation. With it held the worker cannot start a
    // round, so nothing can be admitted or bound; the only thing that can still change is the
    // request-lifecycle counter (a submission), which the second read refuses. Idle observation and
    // idle preservation are the same lock.
    struct IdleVerdict {
        bool idle = false;
        std::string why;
    };

    // Requires BOTH execution_mutex_ and queue_mutex_ to be held by the caller (see above); reads
    // the queue side and the request-lifecycle counter, which is what makes "a request that exists
    // at all" answer busy. The resize entries hold the queue lock from the verdict through the
    // mutation, so no submission can increment outstanding_ inside the window the verdict clears.
    [[nodiscard]] IdleVerdict idle_state_held() const {
        IdleVerdict verdict;
        const bool holding_decision =
            admission_decision_.has_value() || context_owner_ != nullptr ||
            std::any_of(capture_decisions_.begin(), capture_decisions_.end(),
                        [](const auto& decision) { return decision.has_value(); });
        verdict.idle = outstanding_ == 0 && pending_.empty() && resident_empty() &&
                       paused_.empty() && !holding_decision;
        if (!verdict.idle) {
            verdict.why = std::string("request lifecycle: outstanding=") +
                          std::to_string(outstanding_) + " queued=" +
                          std::to_string(pending_.size()) +
                          " resident=" + (resident_empty() ? "0" : "1") +
                          " paused=" + std::to_string(paused_.size()) +
                          " decision=" + (holding_decision ? "1" : "0");
        }
        return verdict;
    }

    [[nodiscard]] std::uint64_t resize_expert_pool_checked(std::uint64_t target_bytes) {        std::lock_guard serialize(resize_mutex_);
        // A bounded grace, not an admit-anywhere window: a request that just finished leaves the
        // worker in its settle round for a few milliseconds, and refusing that tail would make an
        // "idle" engine look busy to the caller that waited for it. A multi-second prefill still
        // refuses, at the same 80 ms bound.
        std::unique_lock execution(execution_mutex_, std::defer_lock);
        if (!execution.try_lock_for(std::chrono::milliseconds(80))) {
            log_resize_verdict("expert pool", "refused", "an execution round holds the engine");
            throw std::logic_error("expert pool resize requires an idle engine");
        }
        IdleVerdict verdict;
        {
            // Held from the verdict through the mutation: a submission blocks on this lock and
            // therefore cannot increment outstanding_ between the cleared verdict and the
            // reallocation. Before, the second read only *observed* idleness; the mutation ran
            // after it was released (the 2026-11-05 penetration).
            std::unique_lock queue(queue_mutex_);
            verdict = idle_state_held();
            if (!verdict.idle) {
                log_resize_verdict("expert pool", "refused", verdict.why);
                throw std::logic_error("expert pool resize requires an idle engine");
            }
            // The synchronize runs with both locks held: no round can start and no request can be
            // submitted meanwhile, so this is the whole in-flight set, not a snapshot racing one.
            device_.synchronize();
            verdict = idle_state_held();
            if (!verdict.idle) {
                log_resize_verdict("expert pool", "refused",
                                   "lost its idle window to a submission: " + verdict.why);
                throw std::logic_error("expert pool resize lost its idle window");
            }
            log_resize_verdict("expert pool", "accepted", verdict.why);
            // The verdict above required outstanding_ == 0 under this lock; if that ever stops
            // being true the counter makes it visible to the acceptance test instead of hiding in
            // a log line.
            if (outstanding_.load(std::memory_order_acquire) != 0) {
                resize_accepted_inflight_.fetch_add(1, std::memory_order_relaxed);
            }
            return instance_.resize_expert_pool(target_bytes);
        }
    }

    // 13.5-5 / open #1: the KV half, same acceptance, same lock discipline, same verdict log.
    [[nodiscard]] std::uint32_t resize_kv_pages_checked(std::uint32_t target_pages) {
        std::lock_guard serialize(resize_mutex_);
        if (target_pages == 0) { throw std::invalid_argument("KV page target must be positive"); }
        std::unique_lock execution(execution_mutex_, std::defer_lock);
        if (!execution.try_lock_for(std::chrono::milliseconds(80))) {
            log_resize_verdict("kv", "refused", "an execution round holds the engine");
            throw std::logic_error("KV pool resize requires an idle engine");
        }
        IdleVerdict verdict;
        {
            // Same lock discipline as the expert half: verdict and mutation under one queue lock.
            std::unique_lock queue(queue_mutex_);
            verdict = idle_state_held();
            if (!verdict.idle) {
                log_resize_verdict("kv", "refused", verdict.why);
                throw std::logic_error("KV pool resize requires an idle engine");
            }
            device_.synchronize();
            verdict = idle_state_held();
            if (!verdict.idle) {
                log_resize_verdict("kv", "refused",
                                   "lost its idle window to a submission: " + verdict.why);
                throw std::logic_error("KV pool resize lost its idle window");
            }
            const std::uint32_t before = instance_.program->physical_usage().capacity.main_kv_pages;
            const std::uint32_t after  = instance_.resize_kv_capacity(target_pages);
            log_resize_verdict("kv", "accepted", std::string("pages ") + std::to_string(before) +
                                                     " -> " + std::to_string(after) + " (target " +
                                                     std::to_string(target_pages) + ")");
            if (outstanding_.load(std::memory_order_acquire) != 0) {
                resize_accepted_inflight_.fetch_add(1, std::memory_order_relaxed);
            }
            return after;
        }
    }


    // The bus's accepted executor for "expert_pool_mb" (v8 12.5 step 4). It runs on the worker
    // thread itself - the model code calling bus.set IS this thread, mid-round - so an actual
    // resize needs an idle engine that this thread, right now, can never offer. The executor
    // therefore only parks the target and wakes the worker's own scheduling point, which
    // applies it through the same idle contract an external caller walks. The dial reads the
    // capacity back from the bus, so whatever happens - applied, clamped, refused - is what
    // the next verdict sees.
    bool park_expert_pool_resize(double value) {
        pending_pool_resize_mb_.store(value, std::memory_order_release);
        {
            // Take the lock so a worker sitting in the wait predicate cannot miss the store
            // between its check and its sleep (the notify alone would be a race).
            std::lock_guard lock(queue_mutex_);
        }
        queue_cv_.notify_all();
        return true;
    }

    // The KV half of the slow dial's executor (13.4 + v8 12.3's L1 rows). Same handshake, same
    // parking discipline: the worker applies it at its own idle scheduling point through
    // resize_kv_pages_checked, i.e. the very contract an external caller walks. A value of 0 means
    // "nothing parked".
    bool park_kv_resize_pages(double value) {
        pending_kv_pages_.store(value > 0.0 ? static_cast<double>(static_cast<std::uint32_t>(value))
                                           : 0.0,
                                std::memory_order_release);
        {
            std::lock_guard lock(queue_mutex_);
        }
        queue_cv_.notify_all();
        return true;
    }

    // ---- T131: the control plane's parked write -------------------------------------------
    //
    // The console's three deployment modes and the expert freeze policy reach the engine from an
    // HTTP worker thread, but the things they change - PolicyController::mode, the thread-local
    // ParamBus rows, every ExpertStream's FreezePolicy - live on the worker thread (the
    // controller and the bus are thread_local BY DESIGN: v8 12.1, "one thread is the shared
    // state"). This mirrors the slow dial's handshake exactly: an external caller parks a
    // request, wakes the worker's own scheduling point, and the worker applies it between
    // rounds. The queue mutex is taken on the way out so a worker sitting in its wait predicate
    // cannot miss the store between its check and its sleep.
    //
    // `freeze_keep_initial` means "the policy this process started with" (NINFER_EXPERT_FREEZE
    // and friends, read once by the bus); the mode bundles use it for auto/stable, where the
    // configured warm-up is the policy, and pass an explicit 0 for performance (no freeze).
    struct ControlRequest {
        int mode                 = -1; // -1: leave the mode alone
        bool freeze_valid        = false;
        bool freeze_keep_initial = false;
        std::uint32_t freeze_after    = 0;
        std::uint32_t freeze_reselect = 0;
        bool freeze_thaw_on_request   = false;
        std::uint64_t sequence        = 0;
    };
    struct ControlState {
        int mode = 0;
        std::uint32_t freeze_after = 0;
        std::uint32_t freeze_reselect = 0;
        bool freeze_thaw_on_request = false;
        bool freeze_armed = false;
        bool frozen       = false;
        std::uint32_t freeze_banks = 0; // routed banks the policy reaches (0 = dense model)
        bool live         = false; // a control-plane write has landed
        bool mode_live    = false; // ... and it included the mode
        std::uint64_t sequence = 0;
        std::string detail;
    };

    [[nodiscard]] std::uint64_t park_control(ControlRequest request) {
        std::uint64_t sequence = 0;
        {
            std::lock_guard lock(control_mutex_);
            sequence = ++control_sequence_;
            request.sequence = sequence;
            pending_control_ = request;
        }
        control_pending_.store(true, std::memory_order_release);
        {
            std::lock_guard lock(queue_mutex_);
        }
        queue_cv_.notify_all();
        return sequence;
    }

    [[nodiscard]] ControlState control_state() const {
        std::lock_guard lock(control_mutex_);
        return control_state_;
    }

    // Returns true when the worker applied (or observed an equal) sequence within the timeout;
    // false means the engine is mid-round and the write is still parked - the caller reports
    // "queued", never a value it cannot see.
    [[nodiscard]] bool wait_control(std::uint64_t sequence, std::chrono::milliseconds timeout) {
        std::unique_lock lock(control_mutex_);
        return control_cv_.wait_for(lock, timeout, [&] {
            return control_state_.sequence >= sequence;
        });
    }

    // The worker's own scheduling point for the control plane. Runs OUTSIDE execution_mutex_ and
    // touches only worker-thread state, so it is safe between rounds and while idle.
    void apply_pending_control() {
        if (!control_pending_.load(std::memory_order_acquire)) { return; }
        ControlRequest request;
        {
            std::lock_guard lock(control_mutex_);
            if (!control_pending_.load(std::memory_order_relaxed)) { return; }
            request = pending_control_;
            control_pending_.store(false, std::memory_order_release);
        }
        using models::qwen3_5::execution::param_bus;
        std::string detail = "applied";
        if (request.mode >= 0) {
            const auto mode = static_cast<ninfer::ControllerMode>(request.mode);
            models::qwen3_5::execution::policy_controller().mode = mode;
            // The same value in the process-wide cell: the [bus] panel and every other reader
            // print the mode in force, so a control-plane change must move the cell too or the
            // two panels would name different deployments in the same process.
            ninfer::set_controller_mode_runtime(mode);
            mode_from_control_ = true;
            std::fprintf(stderr, "[ctl] mode: %s (control plane)\n",
                         ninfer::controller_mode_name(mode));
            std::fflush(stderr);
        }
        if (request.freeze_valid) {
            freeze_after_    = request.freeze_keep_initial ? initial_freeze_after_
                                                           : request.freeze_after;
            freeze_reselect_ = request.freeze_keep_initial ? initial_freeze_reselect_
                                                           : request.freeze_reselect;
            freeze_thaw_     = request.freeze_keep_initial ? initial_freeze_thaw_
                                                           : request.freeze_thaw_on_request;
            instance_.set_expert_freeze_policy(freeze_after_, freeze_reselect_, freeze_thaw_);
            // The bus rows are the panel's copy of the same policy; note_value is a mirror of a
            // value the mechanism already holds, not a second write path.
            auto& bus = param_bus();
            bus.note_value("freeze_after", static_cast<double>(freeze_after_));
            bus.note_value("freeze_reselect", static_cast<double>(freeze_reselect_));
            bus.note_value("freeze_thaw_on_request", freeze_thaw_ ? 1.0 : 0.0);
            if (freeze_after_ == 0) {
                detail = "applied (freeze disabled: placement follows the router)";
            } else if (instance_.expert_bank_count() == 0) {
                detail = "applied (dense model: no routed expert banks to freeze)";
            } else if (!instance_.expert_freeze_armed()) {
                detail = "applied (no freeze mechanism armed: policy stored for the next freeze)";
            }
        }
        publish_control_state(detail.c_str(), request.sequence);
    }

    // Publishes the worker's current view. `sequence == 0` refreshes the live-state half
    // (armed/frozen) without claiming a new write; the last write's sequence and detail stay.
    // `detail == nullptr` likewise keeps the last detail line.
    void publish_control_state(const char* detail, std::uint64_t sequence) {
        ControlState next;
        next.mode                   = static_cast<int>(models::qwen3_5::execution::policy_controller().mode);
        next.freeze_after           = freeze_after_;
        next.freeze_reselect        = freeze_reselect_;
        next.freeze_thaw_on_request = freeze_thaw_;
        next.freeze_armed           = instance_.expert_freeze_armed();
        next.frozen                 = instance_.expert_frozen();
        next.freeze_banks           = static_cast<std::uint32_t>(instance_.expert_bank_count());
        {
            std::lock_guard lock(control_mutex_);
            next.detail    = detail != nullptr ? detail : control_state_.detail;
            next.sequence  = sequence != 0 ? sequence : control_state_.sequence;
            next.live      = sequence != 0 || control_state_.live;
            next.mode_live = mode_from_control_ || control_state_.mode_live;
            control_state_ = std::move(next);
        }
        control_cv_.notify_all();
    }

    // The worker's own scheduling point: apply a parked pool target while idle. Runs OUTSIDE
    // execution_mutex_ on purpose - any_inflight()'s try-lock must be able to see the queue, and
    // the worker must not hold the very lock the contract probes. A refusal is logged and the
    // bus is noted back to the pre-step capacity; it is never bypassed.
    void apply_pending_pool_resize() {
        apply_pending_expert_pool_resize();
        apply_pending_kv_resize();
    }

    void apply_pending_expert_pool_resize() {
        const double mb = pending_pool_resize_mb_.exchange(0.0, std::memory_order_acq_rel);
        if (mb <= 0.0) { return; }
        const auto target    = static_cast<std::uint64_t>(mb) << 20;
        const double from_mb = static_cast<double>(instance_.expert_pool_capacity() >> 20);
        try {
            const auto after      = resize_expert_pool_checked(target);
            const double after_mb = static_cast<double>(after >> 20);
            // A moved layout invalidates every captured layer graph - the baked offset-table
            // addresses are the exact illegal-address trap expert_cache.cpp's resize note
            // describes - so they are dropped here and the next layer captures fresh.
            models::qwen3_5::execution::discard_layer_graph_captures();
            models::qwen3_5::execution::param_bus().note_value("expert_pool_mb", after_mb);
            std::fprintf(stderr,
                         "[ctl] expert pool %.0f -> %.0f MiB (controller slow dial, idle window)\n",
                         from_mb, after_mb);
            std::fflush(stderr);
        } catch (const std::exception& error) {
            models::qwen3_5::execution::param_bus().note_value("expert_pool_mb", from_mb);
            std::fprintf(stderr,
                         "[ctl] expert pool resize refused: %s (capacity stays %.0f MiB)\n",
                         error.what(), from_mb);
            std::fflush(stderr);
        }
    }

    // 13.4: the KV half. The bus row is the single read-back channel, exactly like the pool: the
    // worker notes the capacity the resize actually produced (a shrink that a retained checkpoint
    // refused leaves the old edge in place), so the dial's next verdict judges what is in force.
    void apply_pending_kv_resize() {
        const double parked = pending_kv_pages_.exchange(0.0, std::memory_order_acq_rel);
        if (parked <= 0.0) { return; }
        const auto target    = static_cast<std::uint32_t>(parked);
        const std::uint32_t before = instance_.program->physical_usage().capacity.main_kv_pages;
        try {
            const std::uint32_t after = resize_kv_pages_checked(target);
            models::qwen3_5::execution::param_bus().note_value("kv_pages", after);
            std::fprintf(stderr,
                         "[ctl] kv pages %u -> %u (controller slow dial, idle window)\n", before,
                         after);
            std::fflush(stderr);
        } catch (const std::exception& error) {
            models::qwen3_5::execution::param_bus().note_value("kv_pages", before);
            std::fprintf(stderr,
                         "[ctl] kv resize refused: %s (capacity stays %u pages)\n", error.what(),
                         before);
            std::fflush(stderr);
        }
    }

    // THE idle question, in one place: is any request anywhere in its lifecycle? Used by the
    // offline scoring core's resize entry (which has no worker scheduling point of its own) and
    // by tests; the generation core's own contract goes through idle_state_held() while HOLDING
    // both the round and the queue lock, which is the stronger statement. `outstanding_` is the
    // request-lifecycle counter (bumped under queue_mutex_ by submit() *before* the request object
    // exists, released when its consumer lets go), so a request that is pre-queue, queued, binding,
    // resident or terminal-pending all answer "busy" - the queue -> lane hand-off has no gap left
    // to shrink.
    [[nodiscard]] bool any_inflight(std::string* why = nullptr) const {
        std::unique_lock execution_lock(execution_mutex_, std::try_to_lock);
        if (!execution_lock.owns_lock()) {
            if (why != nullptr) { *why = "an execution round holds the engine"; }
            return true;
        }
        const IdleVerdict verdict = [&] {
            std::lock_guard queue(queue_mutex_);
            return idle_state_held();
        }();
        if (why != nullptr) {
            *why = verdict.idle
                       ? std::string("idle under the queue lock (outstanding=0, queued=0, "
                                     "resident=0, paused=0, no decision)")
                       : verdict.why;
        }
        return !verdict.idle;
    }

    // One line per verdict, so a refusal recorded in a log names the state that refused it and
    // an acceptance names the edge it moved. Cheap: only the resize entry points call it.
    void log_resize_verdict(const char* which, const char* verdict, const std::string& detail) const {
        const RuntimeStats stats = runtime_stats();
        std::fprintf(stderr,
                     "[resize] %s %s: %s | gauges run=%u pref=%u ready=%u wait=%u pause=%u "
                     "replay=%u mat=%u capt=%u term=%u outstanding=%zu\n",
                     which, verdict, detail.c_str(), stats.running_requests,
                     stats.prefilling_requests, stats.decode_ready_requests,
                     stats.waiting_requests, stats.paused_requests, stats.replaying_requests,
                     stats.materializing_requests, stats.capture_pending_requests,
                     stats.terminal_pending_requests, outstanding_snapshot());
        std::fflush(stderr);
    }

    [[nodiscard]] std::size_t outstanding_snapshot() const {
        // Lock-free on purpose: the resize verdict logging calls this while it already holds
        // queue_mutex_ (the verdict and the mutation share that lock), so taking it again would
        // self-deadlock. The counter itself is still only mutated under queue_mutex_.
        return outstanding_.load(std::memory_order_acquire);
    }

    // Every accepted reallocation must have cleared a verdict that saw outstanding_ == 0. This
    // counter exists so the acceptance test can assert that invariant engine-side instead of
    // sampling a counter from outside (where a request finishing during the 80 ms settle grace,
    // or the next one being submitted right after the return, is indistinguishable from a
    // penetration). It must stay zero.
    [[nodiscard]] std::uint64_t resize_accepted_inflight_snapshot() const {
        return resize_accepted_inflight_.load(std::memory_order_relaxed);
    }

private:

    void record_context_work(const typename ModelContract::ContextProgress& progress,
                             const std::shared_ptr<Request>& owner);

    bool progress_context_transaction(HostPhaseMeasurement& boundary) {
        if (!instance_.program->has_context_transaction()) { return false; }
        const auto owner = materializing_ ? materializing_->request : context_owner_;
        const auto cancellation =
            owner ? CancellationFlagView{&owner->cancelled} : CancellationFlagView{};
        auto progress = instance_.program->poll_context(cancellation);
        if (!progress.complete) {
            // Watchdog: nothing else will ever time a stuck materialization out, and the lane stays
            // occupied meanwhile (`running 0 | materializing 1` is what it looks like from outside).
            constexpr std::uint64_t kMaterializeTimeoutMs = 120000;
            if (materializing_ && materializing_since_ms_ != 0 &&
                monotonic_ms() - materializing_since_ms_ > kMaterializeTimeoutMs) {
                const auto request = materializing_->request;
                materializing_.reset();
                materializing_since_ms_ = 0;
                complete_error(
                    request,
                    std::make_exception_ptr(RequestError(
                        RequestErrorKind::Unavailable,
                        "context materialization stalled for " +
                            std::to_string(kMaterializeTimeoutMs / 1000) +
                            " s (no progress from the binding path)")));
                request_admission_check();
                return true;
            }
            return progress.advanced;
        }
        record_context_work(progress, owner);
        if (owner && progress.request_timings) {
            owner->generation_timings = *progress.request_timings;
            owner->speculative_stats  = std::move(progress.request_speculative);
        }
        if (materializing_) {
            const auto control  = std::move(*materializing_);
            const auto& request = control.request;
            if (!progress.published) {
                materializing_.reset();
                complete_cancelled(request);
                request_admission_check();
                return true;
            }
            auto carried = progress.private_points;
            if (control.resumed) {
                for (const auto point : resources_.points(request->continuation_owner)) {
                    if (instance_.program->valid_checkpoint(point) &&
                        std::find(carried.begin(), carried.end(), point) == carried.end()) {
                        carried.push_back(point);
                    }
                }
            }
            request->continuation_owner =
                resources_.adopt(*instance_.program, control.source, *request->base_plan,
                                 request->publication_order, monotonic_ms(), carried,
                                 progress.retired_checkpoints);
            resources_.release_source(*instance_.program, request->id);
            request->lane     = control.destination;
            request->sequence = progress.sequence;
            if (!request->sequence) {
                throw std::logic_error("binding completed without a native sequence");
            }
            request->model_state = progress.recovery.route == GenerationRecoveryRoute::Replay
                                   ? EngineRequestState::Replay
                               : control.resumed ? request->resume_phase
                                                 : EngineRequestState::Prefill;
            if (control.resumed) {
                // Report the route the session actually took. Direct (no committed history) and
                // Snapshot are both non-replay restorations, and they used to be counted as one.
                request->recovery = progress.recovery;
                if (progress.recovery.route == GenerationRecoveryRoute::Snapshot) {
                    ++cumulative_stats_.snapshot_restores;
                    ++request->snapshot_restores;
                } else if (progress.recovery.route == GenerationRecoveryRoute::Direct) {
                    ++cumulative_stats_.direct_restores;
                    ++request->direct_restores;
                }
                request->suspended.reset();
                request->recovery_pending = instance_.program->recovery_pending(*request->sequence);
                if (request->paused_at) {
                    request->paused_ns += elapsed_ns(*request->paused_at, Clock::now());
                }
                request->paused_at.reset();
                observe_scheduling(request, GenerationSchedulingTransition::Restored);
            } else {
                request->initial_binding_ns = elapsed_ns(*request->admitted_at, Clock::now());
                if (control.summary.reused_prompt_tokens) {
                    ++cumulative_stats_.checkpoint_selections;
                } else {
                    ++cumulative_stats_.root_selections;
                }
                cumulative_stats_.reused_prompt_tokens += control.summary.reused_prompt_tokens;
                cumulative_stats_.prompt_tokens += control.summary.prompt_tokens;
                cumulative_stats_.last_selected_frontier_tokens =
                    control.summary.reused_prompt_tokens;
                // Generation start was published when the first binding obtained its resources.
            }
            finish_engine_phase(boundary, EngineHostPhase::Boundary);
            slots_[control.destination.value] = request;
            materializing_.reset();
            boundary = begin_host_phase();
        } else if (context_owner_) {
            const auto request = context_owner_;
            if (request->model_state == EngineRequestState::Pausing) {
                const auto lane = request->lane->value;
                if (!progress.published) {
                    finish_engine_phase(boundary, EngineHostPhase::Boundary);
                    context_owner_.reset();
                    complete_cancelled(request);
                    request_admission_check();
                    boundary = begin_host_phase();
                    return true;
                }
                request->suspended      = std::move(progress.paused);
                request->model_state    = EngineRequestState::Paused;
                request->recovery_route = request->suspended->has_snapshot()
                                              ? GenerationRecoveryRoute::Snapshot
                                              : GenerationRecoveryRoute::Replay;
                observe_scheduling(request, GenerationSchedulingTransition::Paused);
                paused_.push_back(request);
                std::sort(paused_.begin(), paused_.end(),
                          [](const auto& a, const auto& b) { return a->id < b->id; });
                finish_engine_phase(boundary, EngineHostPhase::Boundary);
                slots_[lane].reset();
                request->lane.reset();
                request->sequence.reset();
                request_admission_check();
                boundary = begin_host_phase();
            } else {
                if (progress.published) {
                    ++cumulative_stats_.active_captures_completed;
                } else {
                    ++cumulative_stats_.active_captures_aborted;
                }
                request->capture_pending = false;
                request->model_state     = request->post_capture_state;
                for (const auto point : progress.captured_checkpoints) {
                    resources_.publish(*instance_.program, request->continuation_owner, point,
                                       monotonic_ms());
                    // Only after the record exists and is owned: publishing an unowned checkpoint
                    // would hand the cache bytes that a later prune() is about to release.
                    if (request->base_plan) {
                        resources_.publish_to_cache(*instance_.program, *request->base_plan, point,
                                                   request->continuation_owner);
                    }
                }
            }
            context_owner_.reset();
        }
        request_admission_check();
        return true;
    }

    bool reclaim(ContextResourceUsage shortage, bool allow_pause,
                 std::span<const Checkpoint> excluded               = {},
                 typename ResourceManagement::ReclaimCursor* cursor = nullptr) {
        if (!shortage.state_slots && !shortage.main_kv_pages && !shortage.backend_kv_pages &&
            !shortage.host_bytes) {
            return false;
        }
        if (instance_.program->reclaim_capture_reservation(shortage)) {
            request_admission_check();
            return true;
        }
        const auto progress =
            cursor ? resources_.reclaim(*instance_.program, shortage, excluded, *cursor)
                   : resources_.reclaim(*instance_.program, shortage, excluded);
        if (progress != ReclaimProgress::Blocked) {
            if (progress == ReclaimProgress::Changed) {
                request_admission_check();
            }
            return true;
        }
        if (cursor && cursor->rights.purpose != ReclaimPurpose::Execution) { return false; }
        for (auto it = paused_.rbegin(); it != paused_.rend(); ++it) {
            auto& request = **it;
            if (!request.suspended || !request.suspended->has_snapshot()) { continue; }
            const auto handle = *request.suspended->snapshot_handle();
            if (std::find(excluded.begin(), excluded.end(), handle) != excluded.end()) { continue; }
            const auto resources = instance_.program->snapshot_resources(*request.suspended);
            if (((shortage.state_slots && resources.state_slots) ||
                 (shortage.main_kv_pages && resources.main_kv_pages) ||
                 (shortage.backend_kv_pages && resources.backend_kv_pages) ||
                 (shortage.host_bytes && resources.host_bytes)) &&
                instance_.program->revoke_snapshot(*request.suspended)) {
                request.recovery_route = GenerationRecoveryRoute::Replay;
                observe_scheduling(*it, GenerationSchedulingTransition::SnapshotRevoked);
                if (admission_decision_ && admission_decision_->request.get() == &request) {
                    admission_decision_.reset();
                }
                request_admission_check();
                return true;
            }
        }
        return allow_pause && pause_reclaim_victim();
    }

    bool pause_reclaim_victim() {
        const auto victim = scheduler_.preemption_victim(slots_, max_concurrency_);
        if (!victim) { return false; }
        return pause_resident(*victim);
    }

    // port (sm_89): is there a lane whose pages could be given back to grow `lane`? A candidate has
    // to sit OUTSIDE this round's batch - a batch member is executing the same pass, so pausing it
    // only moves the shortage to the next row of the same pass - and it has to hold resident KV at
    // all. When nothing answers, the shortage is not transient and pausing is the wrong answer (see
    // refuse_kv_residency_growth()).
    [[nodiscard]] bool resident_growth_can_preempt(
        std::uint32_t lane, std::span<const std::uint32_t> batch) const {
        for (std::uint32_t other = 0; other < max_concurrency_; ++other) {
            if (other == lane) { continue; }
            if (std::find(batch.begin(), batch.end(), other) != batch.end()) { continue; }
            const auto& candidate = slots_[other];
            if (!candidate || !candidate->sequence || candidate->is_model_finished() ||
                candidate->model_state == EngineRequestState::Pausing) {
                continue;
            }
            if (instance_.program->sequence_execution_frontier(*candidate->sequence) == 0) {
                continue;
            }
            return true;
        }
        return false;
    }

    // Could pausing somebody hand `lane` the pages it is short of? Another resident lane can be
    // paused, and a paused request can have its snapshot revoked. When neither exists, pausing this
    // lane is a self-inflicted deadlock: the only victim is the request itself, and nothing is left
    // running to drive it back.
    [[nodiscard]] bool other_pause_candidate(std::uint32_t lane) const {
        for (std::uint32_t other = 0; other < max_concurrency_; ++other) {
            if (other != lane && slots_[other]) { return true; }
        }
        for (const auto& request : paused_) {
            if (request && request->suspended && request->suspended->has_snapshot()) { return true; }
        }
        return false;
    }

    // The pool could not produce the page this step needs and there is no other holder to pause for
    // it. That is the arena being full, not a transient shortage, so the request is refused as the
    // length property it is. The sequence is aborted first so its pages go back: without that the
    // pool stays over its edge and the next request is refused in turn.
    void refuse_kv_pool_exhausted(const std::shared_ptr<Request>& request,
                                  ContextResourceUsage shortage) {
        const auto usage = instance_.program->physical_usage();
        if (request->sequence) {
            auto aborted                = instance_.program->abort(*request->sequence);
            request->generation_timings = aborted.timings;
            request->speculative_stats  = std::move(aborted.speculative);
        }
        complete_error(
            request,
            std::make_exception_ptr(RequestError(
                RequestErrorKind::ContextLengthExceeded,
                "the KV pool is full and nothing can be reclaimed for this step: needed_main_pages=" +
                    std::to_string(shortage.main_kv_pages) + " needed_backend_pages=" +
                    std::to_string(shortage.backend_kv_pages) + " needed_state_slots=" +
                    std::to_string(shortage.state_slots) + " needed_host_bytes=" +
                    std::to_string(shortage.host_bytes) + " resident=" +
                    std::to_string(usage.occupied.main_kv_pages) + " of " +
                    std::to_string(usage.capacity.main_kv_pages) +
                    " page(s) (a retained prefix and a request this size do not both fit: lower "
                    "max_context, or serve one long prefix at a time)")));
    }

    bool pause_resident(std::uint32_t lane) {
        if (instance_.program->has_context_transaction()) { return false; }
        auto request       = slots_[lane];
        bool save_snapshot = false;
        if (const auto bytes = instance_.program->pause_host_bytes(*request->sequence)) {
            const auto usage = instance_.program->physical_usage();
            if (*bytes <= usage.capacity.host_bytes &&
                *bytes > usage.capacity.host_bytes - usage.occupied.host_bytes) {
                (void)instance_.program->release_redundant_host({}, request->sequence);
            }
            // appendix HB (plan item 5): order the sacrificed paused snapshots by RECOMPUTE COST rather
            // than by arrival. Revoking a snapshot forces its context to be replayed, whose cost is
            // linear in the tokens involved (appendix GQ measured 181 us/token), so picking the wrong
            // victim is ~300x worse than picking the right one. `later`'s index IS the snapshot_order
            // tie-break that resource_manager.h ranks by, so ordering here is the whole change.
            // ResumeState::frontier() (program.h:244) is the monotone cost proxy: a higher committed
            // frontier means more to replay. stable_sort keeps the previous youngest-first order among
            // equal costs, which also preserves its fairness property.
            std::vector<std::pair<std::uint32_t, Checkpoint>> later_by_cost;
            for (auto it = paused_.rbegin(); it != paused_.rend(); ++it) {
                if ((*it)->id > request->id && (*it)->suspended &&
                    (*it)->suspended->has_snapshot()) {
                    later_by_cost.emplace_back((*it)->suspended->frontier(),
                                               *(*it)->suspended->snapshot_handle());
                }
            }
            std::stable_sort(later_by_cost.begin(), later_by_cost.end(),
                             [](const auto& left, const auto& right) {
                                 return left.first < right.first;
                             });
            std::vector<Checkpoint> later;
            later.reserve(later_by_cost.size());
            for (const auto& entry : later_by_cost) { later.push_back(entry.second); }
            if (const auto victims =
                    resources_.host_victims(*instance_.program, *bytes, std::nullopt, later)) {
                for (const auto handle : *victims) {
                    const auto usage = instance_.program->physical_usage();
                    if (usage.capacity.host_bytes - usage.occupied.host_bytes >= *bytes) { break; }
                    if (resources_.erase(*instance_.program, handle)) { continue; }
                    const auto owner =
                        std::find_if(paused_.begin(), paused_.end(), [&](const auto& other) {
                            return other->suspended &&
                                   other->suspended->snapshot_handle() == handle;
                        });
                    if (owner == paused_.end() ||
                        !instance_.program->revoke_snapshot(*(*owner)->suspended)) {
                        throw std::logic_error("pause Host victim changed after preflight");
                    }
                    (*owner)->recovery_route = GenerationRecoveryRoute::Replay;
                    observe_scheduling(*owner, GenerationSchedulingTransition::SnapshotRevoked);
                }
                save_snapshot = true;
            }
        }
        ProgramCallScope call(*this);
        const auto pause_started = Clock::now();
        if (!instance_.program->start_pause(*request->sequence, save_snapshot,
                                            &call.failed_timing())) {
            return false;
        }
        call.finish(call.failed_timing());
        if (!request->is_replaying()) { request->resume_phase = request->model_state; }
        request->model_state = EngineRequestState::Pausing;
        request->paused_at   = pause_started;
        ++request->preemption_count;
        ++cumulative_stats_.preemptions;
        request->recovery_route = GenerationRecoveryRoute::None;
        observe_scheduling(request, GenerationSchedulingTransition::PauseStarted, pause_started);
        context_owner_ = request;
        capture_decisions_[lane].reset();
        return true;
    }

    void finish_source_wait(const std::shared_ptr<Request>& request,
                            Clock::time_point now) noexcept {
        if (!request->source_wait_started) { return; }
        request->admission.source_wait_seconds +=
            static_cast<double>(elapsed_ns(*request->source_wait_started, now)) * 1.0e-9;
        request->source_wait_started.reset();
    }

    void retain_admission_source(const std::shared_ptr<Request>& request,
                                 const typename ResourceManagement::SourceChoice& choice,
                                 bool restoring) {
        resources_.retain_source(*instance_.program, request->id, choice);
        if (!restoring && choice.source.reused_tokens && !request->source_wait_started) {
            request->source_wait_started = Clock::now();
        } else if (!restoring && !choice.source.reused_tokens) {
            finish_source_wait(request, Clock::now());
        }
    }

    bool try_admit_one(bool restoring) {
        // Q67/S1: the batch container's step 0 is to overlap the next lane's bind/capture with the
        // current lane's chunk, so this probe counts what blocks admission today. Printed with the
        // prefill batch report; the counters only move under that switch.
        const bool admit_probe = prefill_batch_report_ && !restoring;
        if (admit_probe) { ++admit_probe_calls_; }
        admission_deferred_ = false;
        if (instance_.program->has_context_transaction()) {
            if (admit_probe) { ++admit_probe_tx_busy_; }
            return false;
        }
        const auto lane = free_lane();
        if (admit_probe && !lane) { ++admit_probe_no_lane_; }
        // upstream a754fcf3: only the pressure service owner restores; a suspended request
        // that merely happens to be oldest cannot take the turn's capacity.
        const auto owner = scheduler_.progress_owner();
        const auto suspended =
            restoring ? std::find_if(paused_.begin(), paused_.end(),
                                     [&](const auto& request) { return request->id == owner; })
                      : paused_.end();
        if (restoring && suspended == paused_.end()) { return false; }
        if (!lane && restoring) {
            if (const auto victim = scheduler_.preemption_victim(slots_, max_concurrency_)) {
                return pause_resident(*victim);
            }
            return false;
        }
        if (admission_decision_ && (admission_decision_->restoring != restoring ||
                                    (restoring && *suspended != admission_decision_->request))) {
            admission_decision_.reset();
        }
        std::vector<std::shared_ptr<Request>> candidates;
        if (admission_decision_) {
            candidates.push_back(admission_decision_->request);
        } else if (restoring) {
            candidates.push_back(*suspended);
        } else {
            std::lock_guard lock(queue_mutex_);
            candidates = scheduler_.fresh_candidates(pending_, max_concurrency_);
        }
        for (const auto& request : candidates) {
            const ReclaimRights rights{restoring ? ReclaimPurpose::Execution
                                                 : ReclaimPurpose::FreshAdmission,
                                       request->id};
            if (!admission_decision_) {
                if (!restoring) {
                    scheduler_.admission_candidate_checked(request->id);
                    if (request->admission_generation == admission_generation_) { continue; }
                    request->admission_generation = admission_generation_;
                }
                ensure_base_plan(request);
                // port (sm_89): a prompt the residency contract can never hold is refused HERE, before
                // any unit, snapshot or pause is booked for it. This is the only place with the whole
                // answer: the prompt length, the live pool edge, and no execution state to unwind.
                // The alternative was the pause-then-strand path described at refuse_kv_residency().
                if (!restoring) {
                    const std::uint32_t resident_limit =
                        instance_.program->kv_residency_token_limit();
                    const std::uint32_t prompt_tokens = request->base_plan->summary().prompt_tokens;
                    // port (sm_89, V10-84): with the page-in leg on, a prompt LONGER than the arena
                    // is not the impossibility it used to be - the step addresses sink + newest
                    // window + the selector's keep set, and a page the Device does not hold is
                    // copied back. So this whole-history pre-flight does not apply; the aggregate
                    // question below (which prices the read set) is the one that decides.
                    const bool read_priced = instance_.program->kv_read_priced();
                    if (!read_priced && resident_limit != 0 && prompt_tokens > resident_limit) {
                        {
                            std::lock_guard lock(queue_mutex_);
                            std::erase(pending_, request);
                        }
                        refuse_kv_residency(request, prompt_tokens, resident_limit);
                        request_admission_check();
                        continue;
                    }
                    // port (sm_89): the pool is shared, so "this request fits" is not enough. Ask
                    // the Program what the lanes already resident hold (page-exact per lane, one
                    // Decode step of growth included - the part every resident lane is guaranteed),
                    // add this request's own page-rounded prompt + step, and refuse here when the
                    // sum does not fit the live pool edge. Without this, two legal requests could
                    // each reserve against the whole pool and then pause each other forever.
                    if (resident_limit != 0) {
                        const auto budget = instance_.program->kv_residency_budget();
                        const std::uint32_t request_tokens =
                            instance_.program->kv_residency_request_tokens(prompt_tokens);
                        const KvResidencyShare share{budget.capacity_tokens, budget.live_tokens,
                                                     request_tokens};
                        if (!kv_residency_share_fits(share)) {
                            // terminal: a prompt the pool could not hold even with every lane empty.
                            // Nothing resident can make room for it, so it is the length property
                            // refuse_kv_residency() names, not a scheduling wait.
                            if (request_tokens > budget.capacity_tokens) {
                                {
                                    std::lock_guard lock(queue_mutex_);
                                    std::erase(pending_, request);
                                }
                                refuse_kv_residency(request, prompt_tokens,
                                                    budget.capacity_tokens);
                                request_admission_check();
                                continue;
                            }
                            // transient: what is missing is the resident lanes' share and they hand
                            // it back when they finish. Stay queued (the queue deadline, which
                            // already stretches with the work ahead, bounds the wait) and be
                            // offered again at the next admission generation, which the completing
                            // lane arms. Marking the scan deferred is what keeps this pass from
                            // re-arming itself: the retry must ride on a real capacity event.
                            note_kv_residency_wait(request, prompt_tokens, request_tokens, budget);
                            admission_deferred_ = true;
                            continue;
                        }
                        // The token arithmetic above is the contract with the KV pool, not with the
                        // box. When physical memory is nearly gone the next prefill's CUDA call
                        // fails and cuda_check aborts the process, so the newcomer waits for a
                        // resident lane to release its bytes.
                        if (available_physical_bytes() < kHostAdmissionReserveBytes) {
                            note_kv_residency_wait(request, prompt_tokens, request_tokens, budget);
                            admission_deferred_ = true;
                            continue;
                        }
                    }
                }
                const auto retained = resources_.retained_source(request->id);
                const auto revoked  = resources_.source_revocations(request->id);
                if (!restoring && revoked > request->admission.revoked_checkpoints) {
                    request->admission.revoked_checkpoints = revoked;
                    if (!retained) {
                        request->admission.fallback_reason = AdmissionFallbackReason::SourceRevoked;
                    }
                }
                std::vector<typename ResourceManagement::SourceChoice> sources;
                if (restoring && request->suspended->has_snapshot()) {
                    auto& choice                 = sources.emplace_back();
                    choice.resume_owner          = request->continuation_owner
                                                       ? std::optional(request->continuation_owner)
                                                       : std::nullopt;
                    choice.source.private_points = resources_.points(request->continuation_owner);
                    choice.source.take_private =
                        resources_.can_transfer_private(request->continuation_owner, request->id);
                } else {
                    sources = resources_.candidates(*instance_.program, *request->base_plan,
                                                    restoring ? request->suspended->frontier()
                                                              : UINT32_MAX,
                                                    restoring && request->continuation_owner
                                                        ? std::optional(request->continuation_owner)
                                                        : std::nullopt,
                                                    request->id, request->publication_order);
                }
                if (!restoring && !sources.empty()) {
                    const auto preferred = sources.front().source.reused_tokens;
                    if (!request->admission_observed) {
                        request->admission.preferred_reused_tokens = preferred;
                        request->admission_observed                = true;
                    } else if (retained && preferred < retained->source.reused_tokens) {
                        request->admission.fallback_reason = AdmissionFallbackReason::CostChanged;
                    }
                }
                admission_decision_.emplace(AdmissionDecision{
                    request, restoring, std::move(sources), 0,
                    resources_.begin_reclaim(*instance_.program, std::nullopt, rights)});
            }
            auto& decision = *admission_decision_;
            ContextResourceUsage shortage;
            bool deferred = false;
            while (decision.current < decision.sources.size()) {
                auto& choice            = decision.sources[decision.current];
                const bool own_snapshot = restoring && request->suspended->has_snapshot();
                if (!own_snapshot &&
                    !resources_.prepare_source(*instance_.program, *request->base_plan, choice,
                                               request->id)) {
                    if (!restoring) {
                        request->admission.fallback_reason =
                            resources_.source_revocations(request->id) &&
                                    !resources_.retained_source(request->id)
                                ? AdmissionFallbackReason::SourceRevoked
                                : AdmissionFallbackReason::SourceInvalid;
                    }
                    ++decision.current;
                    continue;
                }
                if (!lane) {
                    retain_admission_source(request, choice, restoring);
                    deferred = true;
                    break;
                }
                // A new selection replaces its old waiting reference before capacity is
                // checked. The replacement is retained before any old point is released.
                if (resources_.has_source_record(request->id)) {
                    retain_admission_source(request, choice, restoring);
                }
                auto binding_started = Clock::now();
                worker_phase_        = "binding";
                const auto bind      = [&](const auto& source) {
                    return instance_.program->start_binding(
                        *request->base_plan, LaneId{*lane}, source,
                        restoring ? &*request->suspended : nullptr,
                        request->resume_phase == EngineRequestState::Prefill ? UnitKind::Prefill
                             : request->resume_phase == EngineRequestState::ControlReady
                                 ? UnitKind::Control
                                 : UnitKind::Decode,
                        request->resume_phase == EngineRequestState::Prefill ? 0U
                             : request->resume_phase == EngineRequestState::ControlReady
                                 ? static_cast<std::uint32_t>(
                                  request->output.pending_control_tokens().size())
                                 : 1U);
                };
                auto reservation = bind(choice.source);
                if (!reservation && !reservation.source_valid) {
                    if (!restoring) {
                        request->admission.fallback_reason =
                            resources_.source_revocations(request->id) &&
                                    !resources_.retained_source(request->id)
                                ? AdmissionFallbackReason::SourceRevoked
                                : AdmissionFallbackReason::SourceInvalid;
                    }
                    ++decision.current;
                    continue;
                }
                if (!reservation) {
                    shortage = reservation.shortage;
                    if (!own_snapshot) { retain_admission_source(request, choice, restoring); }
                    const auto protected_source = own_snapshot
                                                      ? request->suspended->snapshot_handle()
                                                      : choice.source.checkpoint;
                    const auto excluded         = protected_source
                                                      ? std::span<const Checkpoint>(&*protected_source, 1)
                                                      : std::span<const Checkpoint>{};
                    if (reclaim(shortage, restoring, excluded, &decision.reclaim)) {
                        if (instance_.program->has_context_transaction()) { return true; }
                        continue;
                    }
                    // Another waiting reference may be the only obstacle to a Move.
                    // Grant its revocation only when the resulting complete bind succeeds.
                    if (!own_snapshot) {
                        auto transferable = choice;
                        if (resources_.prepare_source(*instance_.program, *request->base_plan,
                                                      transferable, request->id, rights) &&
                            (transferable.source.consume_source != choice.source.consume_source ||
                             transferable.source.retired_points != choice.source.retired_points)) {
                            const auto transfer_started = Clock::now();
                            auto attempt                = bind(transferable.source);
                            if (attempt.source_valid) {
                                reservation.capacity_possible |= attempt.capacity_possible;
                            }
                            if (attempt) {
                                binding_started = transfer_started;
                                reservation     = std::move(attempt);
                                choice          = std::move(transferable);
                            }
                        }
                    }
                    if (!reservation) {
                        // Resident progress can release/unlock capacity. A root chunk's
                        // smaller initial allocation is not a reason to discard this source.
                        if (reservation.capacity_possible &&
                            (!resident_empty() || (!restoring && !paused_.empty()))) {
                            deferred = true;
                            break;
                        }
                        if (!restoring) {
                            request->admission.fallback_reason =
                                reservation.capacity_possible
                                    ? AdmissionFallbackReason::IsolatedCapacity
                                    : AdmissionFallbackReason::CapacityLimit;
                        }
                        ++decision.current;
                        continue;
                    }
                }
                resources_.binding_started(request->id, reservation.retired_points,
                                           reservation.consumed_source);
                const auto& source = choice.source;
                // A third-tier import is found inside binding, so there is no candidate here to read
                // a frontier from: the reservation reports what it actually reused. Reporting zero
                // would label a 296-token reuse as a root prefill in every observation.
                const std::uint32_t reused = reservation.reused_tokens != 0
                                                 ? reservation.reused_tokens
                                                 : source.reused_tokens;
                const BeginSummary begin{
                    .prompt_tokens        = request->base_plan->summary().prompt_tokens,
                    .reused_prompt_tokens = reused,
                    .prefix_reuse_path =
                        reused ? PrefixReusePath::Checkpoint : PrefixReusePath::Root};
                if (restoring) {
                    std::erase(paused_, request);
                    observe_scheduling(request, GenerationSchedulingTransition::RestoreStarted,
                                       binding_started);
                } else {
                    std::lock_guard lock(queue_mutex_);
                    scheduler_.admitted(pending_, *request, max_concurrency_);
                    std::erase(pending_, request);
                }
                request->model_state = EngineRequestState::Materializing;
                if (admit_probe) { ++admit_probe_ok_; }
                // The batching window's clock: a lane admitted now means the rest of the burst may
                // still be arriving, and the merged pass needs them to start together.
                prefill_admission_pass_ = scheduling_pass_;
                materializing_.emplace(
                    MaterializingRequest{request, LaneId{*lane}, begin, restoring, choice});
                materializing_since_ms_ = monotonic_ms();
                admission_decision_.reset();
                if (!restoring) {
                    finish_source_wait(request, binding_started);
                    request->admission.revoked_checkpoints =
                        resources_.source_revocations(request->id);
                    if (reservation.reused_tokens != 0) {
                        // The candidate said "root"; the binding turned out to reuse a third-tier
                        // run. Record what happened, not what was predicted.
                        request->admission.preferred_reused_tokens = reservation.reused_tokens;
                    }
                    request->admitted_at = binding_started;
                    request->host_timing.queue_wait_ns =
                        elapsed_ns(request->submitted, *request->admitted_at);
                    request->queue_wait_recorded = true;
                    publish_generation_start(request, begin);
                }
                return true;
            }
                admission_decision_.reset();
                if (deferred) {
                    // Capacity, not ordering, is what refused this candidate. Remember it so the loop
                    // stops re-arming the scan until a real capacity event arrives.
                    admission_deferred_ = true;
                if (restoring) {
                    return false;
                }
                continue;
            }
            if (admission_check_pending_.load(std::memory_order_acquire)) { return true; }
            if (resident_empty()) {
                throw std::logic_error("isolated request cannot reserve its first legal unit after "
                                       "context reclamation: state=" +
                                       std::to_string(shortage.state_slots) +
                                       ", main_kv=" + std::to_string(shortage.main_kv_pages) +
                                       ", backend_kv=" + std::to_string(shortage.backend_kv_pages) +
                                       ", host_bytes=" + std::to_string(shortage.host_bytes));
            }
            if (restoring) {
                return false;
            }
        }
        return false;
    }

    void observe_scheduling(const std::shared_ptr<Request>& request,
                            GenerationSchedulingTransition transition, Clock::time_point now = {}) {
        if (!request->observation.scheduling) { return; }
        if (now == Clock::time_point{}) { now = Clock::now(); }
        const GenerationSchedulingObservation observation{
            .transition              = transition,
            .route                   = request->recovery_route,
            .engine_request_id       = request->id,
            .steady_ns               = elapsed_ns(Clock::time_point{}, now),
            .elapsed_ns              = elapsed_ns(request->submitted, now),
            .preemption_index        = request->preemption_count,
            .global_prefill_tokens   = cumulative_stats_.computed_prefill_tokens,
            .global_decode_tokens    = cumulative_stats_.committed_decode_tokens,
            .global_replayed_tokens  = cumulative_stats_.replayed_tokens,
            .request_prefill_tokens  = request->computed_prompt_tokens,
            .request_decode_tokens   = request->committed_decode_tokens,
            .request_replayed_tokens = request->replayed_tokens,
        };
        {
            std::lock_guard lock(request->mutex);
            request->events.emplace_back(
                std::make_unique<GenerationSchedulingObservation>(observation));
        }
        request->cv.notify_one();
    }

    void update_recovery(const std::shared_ptr<Request>& request, bool released = false) {
        if (!request->recovery_pending) { return; }
        request->recovery_pending = !released && request->sequence &&
                                    instance_.program->recovery_pending(*request->sequence);
        if (!request->recovery_pending) {
            if (!released) {
                observe_scheduling(request, GenerationSchedulingTransition::RecoveryComplete);
            }
            request_admission_check();
        }
    }

    enum class ReservationScope { Round, Prefill };

    void reserve_resident_units(ReservationScope scope = ReservationScope::Round) {
        runnable_units_.fill(false);
        // upstream a754fcf3: the service turn owns the next capacity handoff. Capacity released
        // for a suspended owner cannot be borrowed by resident growth, and non-owners may not
        // change capacity ownership while the turn is active.
        const auto owner = scheduler_.progress_owner();
        std::optional<std::uint32_t> priority;
        std::optional<std::uint32_t> recovering;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (!request || !request->sequence || request->is_model_finished()) { continue; }
            if (request->id == owner) { priority = lane; }
            if (request->recovery_pending) {
                if (recovering) { throw std::logic_error("multiple protected recoveries"); }
                recovering = lane;
            }
        }
        if (owner && !priority) { return; }
        // The owner's prefill/replay turn is the one this pass staffs first. Local prefill has
        // no continuation handle to yield (the merged-prefill container owns its rows), so the
        // owner simply takes the prefill slot instead of someone else's.
        const auto priority_prefill = priority && (slots_[*priority]->is_prefilling() ||
                                                   slots_[*priority]->is_replaying())
                                          ? priority
                                          : std::nullopt;
        const auto prefill = priority_prefill ? priority_prefill
                                              : scheduler_.next_prefill(slots_, max_concurrency_);
        std::vector<std::uint32_t> candidates;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (!request || request->capture_pending || !request->sequence ||
                request->cancelled.load(std::memory_order_acquire) ||
                instance_.program->context_blocks(*request->sequence)) {
                continue;
            }
            if (prefill == lane || (scope == ReservationScope::Round &&
                                    (request->is_control_ready() || request->is_decode_ready()))) {
                candidates.push_back(lane);
            }
        }
        std::sort(candidates.begin(), candidates.end(), [&](auto a, auto b) {
            if (a == b) { return false; }
            if (priority == a || priority == b) { return priority == a; }
            if (recovering == a || recovering == b) { return recovering == a; }
            return slots_[a]->id < slots_[b]->id;
        });
        // port (sm_89): at most ONE growth refusal per pass. A refusal frees that lane's pages, so the
        // lanes still waiting in this list have room again on the next pass; refusing them all here
        // would answer a shared edge by failing every participant.
        bool refused_lane = false;
        for (const auto lane : candidates) {
            if (refused_lane) { break; }
            const auto request = slots_[lane];
            // Reclaiming for an earlier row may already have paused this candidate.
            if (!request || !request->sequence || request->capture_pending ||
                request->cancelled.load(std::memory_order_acquire) ||
                instance_.program->context_blocks(*request->sequence) ||
                !(request->is_control_ready() || request->is_decode_ready() ||
                  request->is_prefilling() || request->is_replaying())) {
                if (priority == lane) { return; }
                continue;
            }
            const ExecutionUnit unit{
                *request->sequence,
                request->is_control_ready()  ? UnitKind::Control
                : request->is_decode_ready() ? UnitKind::Decode
                : request->is_replaying()    ? UnitKind::Replay
                                             : UnitKind::Prefill,
                request->is_control_ready()
                    ? static_cast<std::uint32_t>(request->output.pending_control_tokens().size())
                : request->is_decode_ready()
                    ? (request->recovery_pending ? 1U
                                                 : request->output.model_token_budget_remaining(
                                                       request->budget->remaining()))
                    : 0U};
            // port (sm_89): the same bound the Program checks at the step, asked BEFORE the unit is
            // reserved - so an overrun is refused as one request instead of entering the
            // reclaim/pause path, where a request that can never obtain a legal unit ends up paused
            // with nobody driving it back. Prefill/Replay grow at most to the prompt, which admission
            // already bounded; a Control step grows by its own token count.
            if (const std::uint32_t resident_limit = instance_.program->kv_residency_token_limit();
                resident_limit != 0) {
                const std::uint32_t live =
                    instance_.program->sequence_execution_frontier(*request->sequence);
                const std::uint32_t growth =
                    request->is_control_ready()  ? unit.tokens
                    : request->is_decode_ready() ? instance_.program->kv_residency_step_growth()
                                                 : 0U;
                // port (sm_89, V10-84): under read-set pricing the quantity that has to fit is the
                // set this step addresses (plus its own growth), not the history the frontier
                // measures - a lane may legitimately grow past the arena, and then the step pages
                // its non-addressed pages back in. 0 means the lane is not priced that way (or is
                // empty), and the whole-history bound below is the right one for it. A Control step
                // keeps the whole-history bound on purpose: its write set is the control block's
                // own token count, which the Program never sees, so its price would be short.
                const bool read_priced_unit = instance_.program->kv_read_priced() &&
                                              !request->is_control_ready();
                const std::uint32_t priced = read_priced_unit
                                                 ? instance_.program->kv_read_set_lane_tokens(lane)
                                                 : 0U;
                if ((priced != 0 ? priced : live + growth) > resident_limit) {
                    refuse_kv_residency(request, live, resident_limit, priced);
                    continue;
                }
            }
            std::optional<typename ResourceManagement::ReclaimCursor> cursor;
            for (;;) {
                const auto result = instance_.program->reserve_units({&unit, 1});
                if (result) {
                    runnable_units_[lane] = true;
                    break;
                }
                if (instance_.program->has_context_transaction()) {
                    if (priority == lane) { return; }
                    break;
                }
                // Only the turn owner may change capacity ownership under pressure; a new
                // contender claims the next turn instead of borrowing the current one's.
                if (scheduler_.progress_owner() && scheduler_.progress_owner() != request->id) {
                    scheduler_.require_progress(request->id);
                    break;
                }
                if (!cursor) { cursor.emplace(resources_.begin_reclaim(*instance_.program)); }
                if (reclaim(result.shortage, false, {}, &*cursor)) {
                    if (!instance_.program->has_context_transaction()) { continue; }
                    break;
                }
                if (request->recovery_pending) {
                    throw std::logic_error("protected recovery exceeded its Native reservation");
                }
                // port (sm_89): under a residency contract a growth with no candidate to preempt is
                // not a shortage - it is this request's history no longer fitting beside its peers.
                // Only the Device KV pool turns a shortage into that property; a state-slot or Host
                // shortage is transient, and preemption stays the answer there.
                if (result.shortage.main_kv_pages != 0 &&
                    instance_.program->kv_residency_token_limit() != 0 &&
                    !resident_growth_can_preempt(lane, candidates)) {
                    refuse_kv_residency_growth(lane, request);
                    if (priority == lane) { return; }
                    // Leave the retry loop for this candidate (not the pass): the next lane gets its
                    // own turn, and the refused one is already finished.
                    refused_lane = true;
                    break;
                }
                // A pause is a promise that some other holder can give the pages back. When this is
                // the only resident request and nothing is paused with a snapshot, nobody can: the
                // victim would be this request, and no timer drives a paused lane back. Measured at
                // 256K (2026-11-22, reproduced with --spec none, so it is not the draft backend): a
                // 163k prompt cold-prefilling beside a 2558-page restored prefix filled the
                // 4,682-page pool, the next page would not come, and the only request paused itself
                // -- paused=1 running=0, 2% host, ten minutes, until the client gave up. Refusing it
                // is the length property it is: the caller gets an answer, and the engine keeps
                // serving.
                if (!other_pause_candidate(lane)) {
                    refuse_kv_pool_exhausted(request, result.shortage);
                    if (priority == lane) { return; }
                    refused_lane = true;
                    break;
                }
                scheduler_.require_progress(request->id);
                if (pause_resident(lane) || pause_reclaim_victim()) { return; }
                throw std::logic_error("progress owner cannot obtain its legal unit");
            }
        }
        if (prefill && runnable_units_[*prefill]) {
            const auto capture_started = Clock::now();
            const bool capture_ready   = prepare_semantic_capture(slots_[*prefill]);
            report_boundary_work(
                "capture stage", slots_[*prefill]->id,
                std::chrono::duration<double>(Clock::now() - capture_started).count(),
                slots_[*prefill]->base_plan ? slots_[*prefill]->base_plan->summary().prompt_tokens
                                            : 0U);
            if (!capture_ready) { runnable_units_[*prefill] = false; }
        }
        // A pause may have selected an already licensed younger row. Its permit belongs
        // to Native cleanup, and it must not enter this cycle's compact execution batch.
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (!slots_[lane] || slots_[lane]->model_state == EngineRequestState::Pausing ||
                (slots_[lane]->sequence &&
                 instance_.program->context_blocks(*slots_[lane]->sequence))) {
                runnable_units_[lane] = false;
            }
        }
    }

    // port (sm_89): NINFER_EXPERT_POOL_CYCLE="<round>,<percent>" hands part of the expert pool
    // back to the driver at a round boundary and restores it on the next round. This is a
    // measurement knob: dropping a window only costs that layer a re-upload, because every bank
    // keeps `window_slots` slots whatever the window count is.
    void maybe_cycle_expert_pool() {
        if (!expert_pool_cycle_parsed_) {
            expert_pool_cycle_parsed_ = true;
            if (const char* value = std::getenv("NINFER_EXPERT_POOL_CYCLE")) {
                unsigned long long step = 0;
                int percent             = 0;
                if (std::sscanf(value, "%llu,%d", &step, &percent) == 2 && percent > 0 &&
                    percent < 100) {
                    expert_pool_step_    = step;
                    expert_pool_percent_ = percent;
                    expert_pool_full_    = instance_.expert_pool_capacity();
                }
            }
        }
        const auto round = ++expert_pool_rounds_;
        if (expert_pool_percent_ == 0 || expert_pool_full_ == 0) { return; }
        if (round == expert_pool_step_) {
            (void)instance_.resize_expert_pool(expert_pool_full_ / 100 * expert_pool_percent_);
        } else if (round == expert_pool_step_ + 1) {
            (void)instance_.resize_expert_pool(expert_pool_full_);
        }
    }

    void run_decode_round(const RoundMembership& membership,
                          const std::array<bool, kMaximumConcurrency>& cancelled_at_unit_start) {
        nvtx::ScopedRange decode_range(nvtx::Name::Decode, nvtx::Category::Decode,
                                       static_cast<std::uint64_t>(membership.size));
        ProgramCallScope program_call(*this);
        RoundMasks masks;
        bool constrained = false;
        for (std::size_t row = 0; row < membership.size; ++row) {
            masks.outputs[row] = &slots_[membership.lane_span()[row]]->output;
            constrained |= masks.outputs[row]->constrained();
        }
        auto pending = instance_.program->decode(
            membership.sequence_span(), membership.budget_span(), &program_call.failed_timing(),
            constrained ? &masks : nullptr);
        program_call.finish(pending.execution_timing());
        commit_pending(std::move(pending), membership.lane_span(), true, cancelled_at_unit_start);
        publish_runtime_stats();
    }

    void run_control_batch(const ControlMembership& membership) {
        nvtx::ScopedRange control_range(nvtx::Name::ControlBatch, nvtx::Category::Control,
                                        static_cast<std::uint64_t>(membership.size));
        EnginePhaseScope phase(*this, EngineHostPhase::CommitOutput);
        if (membership.empty() || membership.row_stride == 0 ||
            membership.tokens.size() !=
                static_cast<std::size_t>(membership.row_stride) * membership.size) {
            throw std::logic_error("thinking control membership is invalid");
        }

        std::array<std::size_t, kMaximumConcurrency> generated_sizes{};
        std::array<std::optional<std::uint32_t>, kMaximumConcurrency> prefix_execution_splits{};
        bool generated_staged         = false;
        const auto rollback_generated = [&]() {
            if (!generated_staged) { return; }
            for (std::size_t row = 0; row < membership.size; ++row) {
                const auto& request = slots_[membership.lanes[row]];
                if (request != nullptr) { request->output.discard_preview(); }
                if (request != nullptr && request->generated.size() >= generated_sizes[row]) {
                    request->generated.resize(generated_sizes[row]);
                }
            }
            generated_staged = false;
        };

        for (std::size_t row = 0; row < membership.size; ++row) {
            const auto& request = slots_[membership.lanes[row]];
            if (request == nullptr) {
                throw std::logic_error("thinking control membership lost its request");
            }
            generated_sizes[row] = request->generated.size();
        }
        generated_staged = true;
        runtime::ExecutionTiming control_timing{};
        try {
            for (std::size_t row = 0; row < membership.size; ++row) {
                const std::uint32_t lane = membership.lanes[row];
                const auto& request      = slots_[lane];
                if (request == nullptr || !request->is_control_ready() ||
                    request->capture_pending || !request->budget || !request->sequence ||
                    !request->lane || request->lane->value != lane) {
                    throw std::logic_error("thinking control row lost its active request");
                }
                const std::span<const TokenId> tokens =
                    std::span<const TokenId>(membership.tokens)
                        .subspan(row * membership.row_stride, membership.row_stride);
                const OutputDecision decision =
                    request->output.preview_control(tokens, request->budget->remaining());
                if (decision.accepted_tokens != membership.row_stride || decision.finished() ||
                    decision.continuation != ContinuationAction::Decode ||
                    (decision.prefix_execution_split_after &&
                     (*decision.prefix_execution_split_after == 0 ||
                      *decision.prefix_execution_split_after > decision.accepted_tokens))) {
                    throw std::logic_error("target control preview returned an invalid decision");
                }
                prefix_execution_splits[row] = decision.prefix_execution_split_after;
                if (request->generated.size() > request->generated.capacity() ||
                    tokens.size() > request->generated.capacity() - request->generated.size()) {
                    throw std::logic_error(
                        "admission did not reserve thinking-control token capacity");
                }
                request->generated.insert(request->generated.end(), tokens.begin(), tokens.end());
            }
            phase.pause_range();
            ProgramCallScope program_call(*this);
            const runtime::ExecutionTiming timing = instance_.program->append_forced_tokens(
                membership.sequence_span(), membership.tokens, membership.row_stride,
                std::span<const std::optional<std::uint32_t>>(prefix_execution_splits.data(),
                                                              membership.size),
                &program_call.failed_timing());
            control_timing = timing;
            program_call.finish(timing);
            phase.resume_range();
        } catch (...) {
            rollback_generated();
            throw;
        }
        generated_staged = false;

        ++cumulative_stats_.host_work.control_units;
        for (const std::uint32_t lane : membership.lane_span()) {
            ++slots_[lane]->host_timing.control_units;
            slots_[lane]->committed_decode_tokens += membership.row_stride;
        }
        cumulative_stats_.committed_decode_tokens += membership.size * membership.row_stride;

        for (std::size_t row = 0; row < membership.size; ++row) {
            const std::uint32_t lane = membership.lanes[row];
            const auto& request      = slots_[lane];
            request->budget->commit(membership.row_stride);
            record_service(request->id, membership.row_stride != 0,
                           execution_elapsed_ns(control_timing));
            auto timing = record_committed_output(request, membership.row_stride);
            append_output(request, request->output.commit_preview(), std::move(timing), &phase);
            if (fail_output(request)) { continue; }
            update_recovery(request);
            request->model_state = EngineRequestState::DecodeReady;
        }
        publish_runtime_stats();
    }

    // The worker holds execution_mutex_ across the failing operation and this cleanup, so no
    // Program introspection can observe a partially cleared physical state.
    //
    // worker_phase_ is the last thing the pass said it was doing: a plain pointer store on the hot
    // path, read only when the worker dies. A failure that names only "KV address-space handle is
    // stale" does not say WHICH call held the stale handle, and the addresses involved are gone by
    // the time anyone reads the log - measured 2026-11-24, a checkpoint released at a publish left a
    // handle stale and the message alone left four candidate call sites.
    // Read by the worker on its failure path and by the hang watchdog from another thread, so it is
    // atomic: a plain pointer written by one thread and read by two is exactly the kind of race a
    // diagnostic must not introduce.
    std::atomic<const char*> worker_phase_{"idle"};

    // A hang is the one failure the worker cannot report: it is inside a turn that never returns.
    // Measured 2026-11-25: the engine logged nothing for 85 minutes mid-prefill (no throughput lines
    // at all, then "host 4798.7% (4m 0.2s)" when the blocked call finally returned) and 38 client
    // requests were lost with no indication of where the time went. This thread watches the heartbeat
    // and says which phase went quiet, once per stall and again every five minutes.
    static constexpr std::uint64_t kHangReportMs   = 20000;
    static constexpr std::uint64_t kHangReminderMs = 300000;

    void report_hang(std::uint64_t stalled_ms) const noexcept {
        RuntimeStats stats;
        {
            std::lock_guard lock(stats_mutex_);
            stats = published_stats_;
        }
        std::size_t waiting = 0;
        {
            std::lock_guard lock(queue_mutex_);
            waiting = pending_.size();
        }
        // Only a stall with work outstanding is a hang. An idle engine (nothing queued, no lane busy)
        // legitimately takes no turns - warmup and the gap between requests both look like that - and
        // reporting those would bury the real one.
        if (waiting == 0 && stats.prefilling_requests == 0 && stats.decode_ready_requests == 0) {
            return;
        }
        std::fprintf(stderr,
                     "[hang] no scheduling pass for %.1f s | phase=%s | waiting=%zu | "
                     "prefilling=%u decode_ready=%u\n",
                     static_cast<double>(stalled_ms) / 1000.0,
                     worker_phase_.load(std::memory_order_relaxed), waiting,
                     stats.prefilling_requests, stats.decode_ready_requests);
        std::fflush(stderr);
    }

    void hang_watchdog_loop() noexcept {
        bool reported              = false;
        std::uint64_t reported_at  = 0;
        while (!watchdog_stop_.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            const auto last = last_pass_ms_.load(std::memory_order_relaxed);
            if (last == 0) { continue; }
            const auto now = monotonic_ms();
            if (now <= last) { continue; }
            const auto stalled = now - last;
            if (stalled < kHangReportMs) {
                reported = false;
                continue;
            }
            if (reported && now - reported_at < kHangReminderMs) { continue; }
            reported    = true;
            reported_at = now;
            report_hang(stalled);
        }
    }

    void log_worker_failure(std::exception_ptr error) const noexcept {
        const auto report = [&](const char* reason) {
            std::fprintf(stderr,
                         "NInfer worker failed: %s | phase=%s progress_owner=%llu paused=%zu "
                         "context=%d materializing=%d\n",
                         reason, worker_phase_.load(std::memory_order_relaxed),
                         static_cast<unsigned long long>(scheduler_.progress_owner().value_or(0)),
                         paused_.size(), instance_.program->has_context_transaction() ? 1 : 0,
                         materializing_.has_value() ? 1 : 0);
            for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                const auto& request = slots_[lane];
                if (request) {
                    std::fprintf(stderr, "  request=%llu lane=%u state=%u recovery=%d\n",
                                 static_cast<unsigned long long>(request->id), lane,
                                 static_cast<unsigned>(request->model_state),
                                 request->recovery_pending ? 1 : 0);
                }
            }
            std::fflush(stderr);
        };
        try {
            std::rethrow_exception(error);
        } catch (const std::exception& cause) { report(cause.what()); } catch (...) {
            report("non-standard exception");
        }
    }

    void fail_all_locked(std::exception_ptr error) noexcept {
        std::deque<std::shared_ptr<Request>> pending;
        {
            std::lock_guard lock(queue_mutex_);
            failed_ = true;
            pending.swap(pending_);
        }
        scheduler_ = Scheduling{};
        const std::shared_ptr<Request> materializing_request =
            materializing_ ? materializing_->request : nullptr;
        // Native retires all transfer readers before any external checkpoint owner is destroyed.
        instance_.program->fail_all_cleanup();
        materializing_.reset();
        admission_decision_.reset();
        for (auto& decision : capture_decisions_) { decision.reset(); }
        context_owner_.reset();
        std::vector<std::shared_ptr<Request>> paused;
        paused.swap(paused_);
        const auto residents = slots_;
        const auto release   = [&](const std::shared_ptr<Request>& request) {
            if (request && prepare_completion(request)) { release_request(request); }
        };
        for (const auto& request : paused) { release(request); }
        for (const auto& request : residents) { release(request); }
        release(materializing_request);
        for (const auto& request : pending) { release(request); }
        resources_.release_all(*instance_.program);
        publish_runtime_stats();
        const auto publish = [&](const std::shared_ptr<Request>& request) {
            if (request) { publish_completion(request, {}, error); }
        };
        for (const auto& request : paused) { publish(request); }
        for (const auto& request : residents) { publish(request); }
        publish(materializing_request);
        for (const auto& request : pending) { publish(request); }
    }

    void worker_loop() noexcept {
        // v8 9.3 + 12.2: register this engine's concurrency executor on the parameter bus. The bus
        // is thread-local, and this worker thread is the thread that both runs the model (whose
        // PolicyController decides) and schedules lanes (whose count is the parameter), so the
        // registration and every later write are race-free by construction. Until this runs, the
        // concurrency actuator stays shadow - which is exactly right for every caller that is
        // not this engine (the single-threaded CLI, perplexity, tests).
        models::qwen3_5::execution::param_bus().attach_applier(
            "concurrency", max_concurrency_, 1.0, static_cast<double>(max_concurrency_),
            [this](double value) { return apply_active_concurrency(value); });
        models::qwen3_5::execution::param_bus().note_value("lanes", max_concurrency_);
        // 13.5: the merged prefill's executor, on the same thread and for the same reason. It is
        // attached only when the environment has not pinned the state: a pinned arm must not be
        // merged into by the controller, and without an applier the actuator stays a shadow that
        // reports what it would do.
        if (!prefill_merge_pinned_) {
            models::qwen3_5::execution::param_bus().attach_applier(
                "prefill_merge", prefill_merge_active() ? 1.0 : 0.0, 0.0, 1.0,
                [this](double value) { return apply_prefill_merge(value); });
        }
        // v8 12.5 step 4: the L1 slow dial's executor. Registered on the same thread-local bus,
        // pinned to the pool's real capacity (a model with no expert pool - dense - attaches
        // nothing, and the dial stays honestly no-executor there). The range is v8 12.3's
        // 1024..19456 MiB; the executor only parks a target, the worker applies it when idle.
        const double pool_mb = static_cast<double>(instance_.expert_pool_capacity() >> 20);
        if (pool_mb > 0.0) {
            models::qwen3_5::execution::param_bus().attach_applier(
                "expert_pool_mb", pool_mb, 1024.0, 19456.0,
                [this](double value) { return park_expert_pool_resize(value); });
            std::fprintf(stderr, "[ctl] expert pool dial attached at %.0f MiB\n", pool_mb);
            std::fflush(stderr);
        }
        // 13.4 + v8 12.3: the KV half of the L1 slow dial. Same shape as the pool row - pinned to
        // the pool's real edge, executor parks a target the worker applies through the idle
        // contract. The range's floor is the smallest edge the dial may ask for (16 page groups =
        // 1024 tokens at this page size); the ceiling is the construction capacity, because the
        // physical backing above it was never laid out.
        const std::uint32_t kv_edge = instance_.program->physical_usage().capacity.main_kv_pages;
        if (kv_edge > 0) {
            const double floor_pages =
                std::max(16.0, static_cast<double>(kv_edge) / 4.0);
            models::qwen3_5::execution::param_bus().attach_applier(
                "kv_pages", static_cast<double>(kv_edge), floor_pages,
                static_cast<double>(kv_edge),
                [this](double value) { return park_kv_resize_pages(value); });
            std::fprintf(stderr, "[ctl] kv page dial attached at %u page(s) (floor %.0f)\n", kv_edge,
                         floor_pages);
            std::fflush(stderr);
        }
        // The construction-time whole-step graph decision, mirrored for print (the [bus] row is        // no-executor by the honest register): expert_pool_pinned is set from graph_ok in
        // model_instance.cpp, and a pinned pool is exactly the state a captured decode graph
        // froze its offsets against.
        models::qwen3_5::execution::param_bus().note_value(
            "graph_on", instance_.expert_pool_pinned ? 1.0 : 0.0);
        // T131: the control plane's starting view. initial_freeze_* is the policy this process
        // configured (the bus read the environment once, at construction); auto/stable's bundle
        // restores exactly this value. Publishing here also means a GET before any write answers
        // the worker's real mode instead of a guess.
        {
            auto& bus = models::qwen3_5::execution::param_bus();
            const auto clamp_u32 = [](double value) {
                return value <= 0.0 ? 0U : static_cast<std::uint32_t>(value);
            };
            initial_freeze_after_    = clamp_u32(bus.get("freeze_after"));
            initial_freeze_reselect_ = clamp_u32(bus.get("freeze_reselect"));
            initial_freeze_thaw_     = bus.get("freeze_thaw_on_request") > 0.0;
            freeze_after_            = initial_freeze_after_;
            freeze_reselect_         = initial_freeze_reselect_;
            freeze_thaw_             = initial_freeze_thaw_;
            publish_control_state("startup", 0);
        }
        for (;;) {
            // Heartbeat for the hang watchdog: stamped once per loop turn, so a turn that never
            // returns shows up as "no pass for N s" instead of an 85-minute silence.
            last_pass_ms_.store(monotonic_ms(), std::memory_order_relaxed);
            bool idle_for_pool_dial = false;
            {
                std::unique_lock lock(queue_mutex_);
                const bool pool_dial_pending =
                    pending_pool_resize_mb_.load(std::memory_order_acquire) > 0.0;
                const bool kv_dial_pending =
                    pending_kv_pages_.load(std::memory_order_acquire) > 0.0;
                if (!stopping_ && pending_.empty() && resident_empty() && paused_.empty() &&
                    !instance_.program->has_context_transaction() && !pool_dial_pending &&
                    !kv_dial_pending && !control_pending_.load(std::memory_order_acquire)) {
                    queue_cv_.wait(lock, [&] {
                        return stopping_ || !pending_.empty() ||
                               pending_pool_resize_mb_.load(std::memory_order_acquire) > 0.0 ||
                               pending_kv_pages_.load(std::memory_order_acquire) > 0.0 ||
                               control_pending_.load(std::memory_order_acquire);
                    });
                }
                if (stopping_) {
                    lock.unlock();
                    std::scoped_lock execution_lock(execution_mutex_);
                    fail_all_locked(std::make_exception_ptr(RequestError(
                        RequestErrorKind::Unavailable, "inference engine is shutting down")));
                    return;
                }
                // The idle window the slow dial needs, observed under the queue lock: nothing
                // queued, nothing resident, nothing paused, no transaction, and no request
                // anywhere in its lifecycle (outstanding_ - the same counter any_inflight()
                // answers from, so the worker's own scheduling point and the external entry
                // cannot disagree about what idle means). When this is false the parked target
                // STAYS parked - a busy engine is not a refusal, it is a "not yet", and the
                // dial's next window re-reads the bus either way.
                idle_for_pool_dial =
                    outstanding_ == 0 && pending_.empty() && resident_empty() && paused_.empty() &&
                    !instance_.program->has_context_transaction() &&
                    (pending_pool_resize_mb_.load(std::memory_order_acquire) > 0.0 ||
                     pending_kv_pages_.load(std::memory_order_acquire) > 0.0);
            }
            if (idle_for_pool_dial) {
                apply_pending_pool_resize();
                continue; // re-evaluate the queue before the next round starts
            }
            // T131: the control plane's own scheduling point. A parked mode/freeze write is
            // applied here, on the thread that owns the controller and the streams, before the
            // next round reads them.
            apply_pending_control();
            // The frozen flag moves on its own clock (the freeze rule installs/thaws after 32
            // layers), so the published state is refreshed on the same window beat instead of
            // only when someone writes. Cheap: one modulo and one small copy per round.
            if (models::qwen3_5::execution::param_bus().layers() % 32 == 0) {
                publish_control_state(nullptr, 0);
                // 13.4: hand the KV dial this window's occupancy peak (see publish_runtime_stats)
                // and start a fresh window. One observation per window, on the thread that owns
                // both the controller and the pool counters.
                models::qwen3_5::execution::policy_controller().observe_kv_pool(kv_window_peak_);
                kv_window_peak_ = 0.0;
            }
            std::unique_lock execution_lock(execution_mutex_);
            bool executed = false;
            try {
                // v8 9.3: the serial rule is judged on the lanes actually in service, not the
                // configured ceiling - serializing one busy lane is a no-op. One observation per
                // scheduling pass; the model code running later in this pass reads the previous
                // pass's count, which is the freshest input available at decision time.
                {
                    std::uint32_t busy_lanes = materializing_ ? 1U : 0U;
                    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                        if (slots_[lane]) { ++busy_lanes; }
                    }
                    models::qwen3_5::execution::policy_controller().observe_lanes(busy_lanes);
                }
                set_host_work_class(HostWorkClass::Control);
                auto boundary = begin_host_phase();
                (void)expire_pending_requests();
                executed = progress_context_transaction(boundary);
                if (!instance_.program->has_context_transaction()) {
                    for (const auto& request : slots_) {
                        if (request && request->capture_pending) {
                            reserve_active_capture(request, request->post_capture_state);
                            break;
                        }
                    }
                }
                (void)settle_terminal_requests(boundary);
                cancel_active_requests(snapshot_cancellations(), boundary);
                for (auto it = paused_.begin();
                     !instance_.program->has_context_transaction() && it != paused_.end();) {
                    const auto request = *it;
                    if (!request->cancelled.load(std::memory_order_acquire)) {
                        ++it;
                        continue;
                    }
                    request->suspended.reset();
                    it = paused_.erase(it);
                    complete_cancelled(request);
                }
                const bool recovering =
                    std::any_of(slots_.begin(), slots_.end(),
                                [](const auto& r) { return r && r->recovery_pending; }) ||
                    (materializing_ && materializing_->resumed);
                if (!instance_.program->has_context_transaction() &&
                    scheduler_.advance_service(!paused_.empty(), recovering)) {
                    // An unused permit belongs to the previous turn. Retained recovery capacity
                    // is released only by Native after actual progress, never by this handoff.
                    for (const auto& request : slots_) {
                        if (request && request->sequence) {
                            instance_.program->release_units({&*request->sequence, 1});
                        }
                    }
                    admission_decision_.reset();
                    request_admission_check();
                }
                if (!recovering && scheduler_.progress_owner()) {
                    (void)try_admit_one(true);
                    executed |= progress_context_transaction(boundary);
                }
                reserve_resident_units();
                // appendix HJ (plan item 4): the idle trigger - a SUPPLEMENT to the capacity-driven path
                // above, for the workload where sessions sit idle between turns while holding device
                // pages (chat / meeting style). Guards, all three required (appendix HC.1.3):
                //   * reachability: this is inert unless residency is actually kept high - the item-1
                //     demote default collapses a lane to its window, so the arena never fills;
                //   * REFUSE rather than degrade: if the Host arena cannot hold the snapshot, skip the
                //     pause entirely. Unlike the capacity path there is no emergency to pay seconds of
                //     recompute for (appendix GQ: ~8.4 s to replay against ~28 ms to fetch);
                //   * demote semantics: pause_resident keeps the bytes, so a resumed session fetches.
                // Off by default: no switch means no behavioural change whatsoever.
                // appendix HX (plan item 9b): the idle trigger, aimed at the object that actually
                // holds residency. Item 4 fired at a live LANE and was measured unreachable in every
                // workload (appendix HO/HP: a decode-ready lane always has work, a finished request
                // releases its lane, and the output path has no back-pressure, so nothing could
                // idle) - but a COMPLETED session does keep its KV: 9a made it a reusable
                // SharedPrefix checkpoint (appendix HU.1) whose pages stay on the device until
                // capacity pressure takes them. This gives those pages up when nobody has asked for
                // them for NINFER_KV_IDLE_RECLAIM_MS, KEEPING the bytes in the Host arena, so a later
                // request still reuses the prefix by fetching (appendix HW: h2d 49 pages,
                // `cache 46,283 (100.0%)`, TTFT 119 ms against 19.9 s to re-prefill).
                // Three guards, all deliberate: the sweep is skipped while the capacity path is
                // already active (a non-empty `paused_` means it had to pause someone) AND while the
                // device arena is under capacity pressure (appendix IE: the pool cannot already
                // spare the footprint the demotion would free - appendix ID measured the pressure
                // state this missed), so it never competes with it (criterion 5); and it runs at
                // most once per threshold window, so a refusal is counted per sweep instead of once
                // per round - the mistake appendix HM had to fix for the lane trigger. Off by
                // default: no switch, no change at all.
                if (idle_reclaim_ms_ != 0 && paused_.empty() && !scheduler_.progress_owner() &&
                    !instance_.program->has_context_transaction()) {
                    const std::uint64_t now_ms = monotonic_ms();
                    if (now_ms >= last_idle_sweep_ms_ &&
                        now_ms - last_idle_sweep_ms_ >= idle_reclaim_ms_) {
                        last_idle_sweep_ms_ = now_ms;
                        const auto idle = resources_.idle_shared(now_ms, idle_reclaim_ms_);
                        if (!idle.empty()) {
                            switch (resources_.demote_idle(*instance_.program, idle)) {
                            case ResourceManagement::IdleDemotion::Started:
                                ++cumulative_stats_.idle_reclaims;
                                break;
                            case ResourceManagement::IdleDemotion::Blocked:
                                // No capacity emergency here, so REFUSE rather than degrade: if the
                                // Host arena cannot take the bytes the demotion is simply not made
                                // (appendix HC.1.3 guard 1), which is why this counts refusals rather
                                // than failing anything.
                                ++cumulative_stats_.idle_reclaims_refused;
                                break;
                            case ResourceManagement::IdleDemotion::Deferred:
                                // appendix IE (criterion 5): the arena is under capacity pressure,
                                // so the capacity path keeps the decision. Counted separately so
                                // that "stood down" is distinguishable from "had no candidate".
                                ++cumulative_stats_.idle_reclaims_deferred;
                                break;
                            case ResourceManagement::IdleDemotion::Nothing:
                                break;
                            }
                        }
                    }
                }
                // A failed restore leaves spare resources available to fresh requests.
                // An in-flight restoration/reclaim keeps its own candidate until settled.
                if (!scheduler_.progress_owner() &&
                    (!admission_decision_ || !admission_decision_->restoring)) {
                    if (!admission_decision_ && consume_admission_check()) {
                        ++admission_generation_;
                        scheduler_.begin_admission_scan();
                    }
                    if (admission_decision_ || scheduler_.admission_scan_pending()) {
                        (void)try_admit_one(false);
                    } else if (!admission_deferred_ && free_lane() &&
                               !instance_.program->has_context_transaction() &&
                               (!prefill_pacing_ || queued_head_joins_prefill_window())) {
                        // Q67/S1 step 0: a queued request used to wait for the previous request's
                        // whole prefill, because the admission scan was only armed by a submission or
                        // a capacity event. While a lane is free and something is queued, arm it here
                        // instead - that is what lets the next lane's bind materialize while the
                        // current lane's chunk runs, which is the precondition a merged prefill has.
                        // With pacing on, the queued head only arms when it belongs to the current
                        // merge window (see queued_head_joins_prefill_window).
                        std::lock_guard lock(queue_mutex_);
                        if (!pending_.empty()) { scheduler_.begin_admission_scan(); }
                    }
                }
                if (instance_.program->has_context_transaction()) {
                    executed |= progress_context_transaction(boundary);
                    if (!instance_.program->has_context_transaction()) { reserve_resident_units(); }
                }
                const auto cancelled = snapshot_cancellations();
                auto controls = scheduler_.build_control_membership(slots_, max_concurrency_);
                ControlMembership selected_control;
                selected_control.row_stride = controls.row_stride;
                for (std::size_t row = 0; row < controls.size; ++row) {
                    if (!runnable_units_[controls.lanes[row]] ||
                        slots_[controls.lanes[row]]->cancelled.load(std::memory_order_acquire)) {
                        continue;
                    }
                    selected_control.lanes[selected_control.size]       = controls.lanes[row];
                    selected_control.sequences[selected_control.size++] = controls.sequences[row];
                    const auto begin = controls.tokens.begin() + row * controls.row_stride;
                    selected_control.tokens.insert(selected_control.tokens.end(), begin,
                                                   begin + controls.row_stride);
                }
                controls = std::move(selected_control);
                finish_engine_phase(boundary, EngineHostPhase::Boundary);
                if (!controls.empty()) {
                    set_host_work_class(HostWorkClass::Control);
                    // v8 9.3: control rounds are decode-family work for the serial rule's
                    // purposes (spec-decode tokens, one layer pass = one token).
                    models::qwen3_5::execution::policy_controller().observe_prefill(false);
                    run_control_batch(controls);
                    executed = true;
                }
                // Control rows become decode-ready next cycle, after reserving their new unit.
                auto decode = scheduler_.build_round_membership(slots_, max_concurrency_);
                {
                    RoundMembership selected;
                    for (std::size_t row = 0; row < decode.size; ++row) {
                        if (!runnable_units_[decode.lanes[row]] ||
                            slots_[decode.lanes[row]]->cancelled.load(std::memory_order_acquire)) {
                            continue;
                        }
                        if (std::find(controls.lane_span().begin(), controls.lane_span().end(),
                                      decode.lanes[row]) != controls.lane_span().end()) {
                            continue;
                        }
                        selected.lanes[selected.size]     = decode.lanes[row];
                        selected.sequences[selected.size] = decode.sequences[row];
                        selected.budgets[selected.size++] = decode.budgets[row];
                    }
                    decode = selected;
                }
                if (!decode.empty()) {
                    maybe_cycle_expert_pool();
                    set_host_work_class(HostWorkClass::Decode, decode.lane_span());
                    // v8 9.3: the serial rule is decided and verified on decode measurements
                    // only; a prefill layer pass carries a whole chunk and its interval reads as
                    // a fraction-of-a-token tps. This is the decode dispatch's own marker.
                    models::qwen3_5::execution::policy_controller().observe_prefill(false);
                    worker_phase_ = "decode round";
                    run_decode_round(decode, cancelled);
                    executed = true;
                }
                if (instance_.program->has_context_transaction()) {
                    set_host_work_class(HostWorkClass::Control);
                    auto progress_boundary = begin_host_phase();
                    worker_phase_         = "context transaction";
                    executed |= progress_context_transaction(progress_boundary);
                    if (!instance_.program->has_context_transaction()) {
                        // Control/Decode permits were consumed. Only the still-unexecuted
                        // Prefill/Replay turn can receive a new permit in this round.
                        reserve_resident_units(ReservationScope::Prefill);
                    }
                    finish_engine_phase(progress_boundary, EngineHostPhase::Boundary);
                }
                auto prefill_slots = slots_;
                for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                    if (!runnable_units_[lane] ||
                        (prefill_slots[lane] &&
                         prefill_slots[lane]->cancelled.load(std::memory_order_acquire))) {
                        prefill_slots[lane].reset();
                    }
                }
                // Q67/S1: what a merged prefill would be, before one exists. The batch container's
                // decision is pure arithmetic on the queued lanes' remaining prompt tokens, so the
                // prediction is printable while prefill is still one lane at a time: if the fit is
                // wrong here it is wrong in the batch too, and this is the cheap place to see it.
                // NINFER_PREFILL_BATCH turns the report on; the default keeps this branch compiled
                // out of the hot path entirely.
                // 13.5: the fit is arithmetic on the queued lanes' remaining prompt tokens, so the
                // controller can price a merge on every pass without a measurement window of its
                // own. Computed whenever more than one lane could contribute - the only case where
                // there is anything to decide - and the report below reuses the same numbers.
                std::size_t queued       = 0;
                std::size_t engine_state = 0;
                std::size_t capture_held = 0;
                std::size_t bound_ready  = 0;
                models::qwen3_5::execution::PrefillBatchFit fit{};
                ++scheduling_pass_;
                if (max_concurrency_ > 1) {
                    std::array<std::uint32_t, kMaximumConcurrency> remaining{};
                    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                        if (!prefill_slots[lane]) { continue; }
                        if (prefill_slots[lane]->is_prefilling()) { ++engine_state; }
                        if (prefill_slots[lane]->capture_pending) { ++capture_held; }
                        const std::uint32_t left = instance_.program->prefill_remaining(lane);
                        if (left == 0) { continue; }
                        ++bound_ready;
                        remaining[queued++] = left;
                    }
                    fit = models::qwen3_5::execution::prefill_batch_fit(
                        std::span<const std::uint32_t>(remaining.data(), queued));
                    {
                        auto& control = models::qwen3_5::execution::policy_controller();
                        // How many requests have prefill work but no row yet: the merge's bootstrap.
                        // Counted even when no row exists yet - that is exactly the state the
                        // bootstrap exists for, and gating the observation on `queued != 0` is what
                        // kept the switch off: a burst that had not started a row never armed it.
                        std::size_t waiting = 0;
                        {
                            std::lock_guard lock(queue_mutex_);
                            waiting = pending_.size() + (materializing_.has_value() ? 1U : 0U);
                        }
                        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                            const auto& resident = slots_[lane];
                            if (resident == nullptr || resident->is_prefilling()) { continue; }
                            if (resident->model_state == EngineRequestState::DecodeReady ||
                                resident->model_state == EngineRequestState::ModelFinished) {
                                continue;
                            }
                            ++waiting;
                        }
                        control.observe_prefill_waiting(static_cast<std::uint32_t>(waiting));
                        prefill_merge_waiting_ = waiting;
                        if (queued != 0) {
                            control.observe_prefill_fit(static_cast<std::uint32_t>(queued),
                                                        fit.merged_ms, fit.separate_ms);
                        }
                        // The flip is applied here: this thread owns the criterion, the executor and
                        // the dispatch, so it does not need the parameter bus (whose tick only the
                        // expert-stream path advances). The actuator records for the panel.
                        const bool want = control.want_prefill_merge();
                        if (want != prefill_merge_active()) {
                            if (++prefill_merge_agree_ >= 2U) {
                                prefill_merge_agree_ = 0;
                                if (!prefill_merge_pinned_) {
                                    (void)apply_prefill_merge(want ? 1.0 : 0.0);
                                    control.note_prefill_merge(
                                        want, want ? "fit: one pass cheaper" : "fit: rows apart cheaper");
                                }
                            }
                        } else {
                            prefill_merge_agree_ = 0;
                        }
                    }
                }
                if (prefill_batch_report_) {
                    ++prefill_batch_reports_;
                    // NINFER_PREFILL_BATCH=all prints every dispatch, not just the first few: a
                    // sampled report hid whether the later dispatches already see two ready lanes,
                    // and that is the whole question this counter exists to answer.
                    if (prefill_batch_verbose_ || prefill_batch_reports_ <= 8 ||
                        prefill_batch_reports_ % 32 == 0) {
                        std::fprintf(stderr,
                                     "[prefill-batch] queued=%zu pick=%d lanes=%zu "
                                     "merged=%.0fms separate=%.0fms | merge=%d waiting=%u |",
                                     queued, fit.width, fit.lanes, fit.merged_ms,
                                     fit.separate_ms, prefill_merge_active() ? 1 : 0,
                                     static_cast<unsigned>(prefill_merge_waiting_));
                        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                            const auto& resident = slots_[lane];
                            std::fprintf(stderr, " l%u[%d/%d/rem%u/%d]", lane,
                                         resident != nullptr ? 1 : 0,
                                         resident != nullptr && resident->is_prefilling() ? 1 : 0,
                                         instance_.program->prefill_remaining(lane),
                                         resident != nullptr
                                             ? static_cast<int>(resident->model_state)
                                             : -1);
                        }
                        std::fprintf(stderr, "\n");
                        std::fflush(stderr);
                    }
                    if (prefill_batch_verbose_) {
                        // Every lane the Program still owes prefill work, not just the ones this pass
                        // would dispatch: a lane held by a capture is exactly a lane a merged pass
                        // would want to include, and filtering on runnable_units_ hid that.
                        std::array<std::uint32_t, kMaximumConcurrency> ids{};
                        std::size_t id_count = 0;
                        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                            if (instance_.program->prefill_remaining(lane) == 0) { continue; }
                            ids[id_count++] = lane;
                        }
                        instance_.program->report_prefill_rows(
                            std::span<const std::uint32_t>(ids.data(), id_count));
                    }
                }
                // Q67/S1 step 0, second half: admission runs behind prefill. A queued lane is bound
                // only after the lane before it has finished its head chunk, so two lanes are never
                // both at "a full chunk of work" at the same dispatch (the per-call trace shows
                // 1570 -> 546 -> 7 for one lane while the next lane's 1570 starts only after that
                // lane is down to its 7-token tail). Holding the prefill dispatch for a pass while
                // a queued lane can still be bound is what puts two heads in the same pass; it is
                // opt-in until it has been measured.
                bool hold_prefill = false;
                if (prefill_hold_active()) {
                    // Hold while anything is still queued - not only while a bind can start right
                    // now. Binding is a context transaction that opens and closes over several loop
                    // passes; holding only when it is free let a head slip in between two binds, and
                    // then no two lanes ever had a chunk at the same time.
                    std::lock_guard lock(queue_mutex_);
                    hold_prefill = (!pending_.empty() || materializing_.has_value()) &&
                                   !admission_deferred_;
                }
                // 13.5: a batching window after the most recent admission. A merged pass can only
                // form if the lanes start their chunk together, and in the serve path each request is
                // admitted into a lane and dispatched the very next pass, so a burst never overlaps.
                // Holding for a few passes after an admission lets the rest of the burst arrive;
                // the window expires on its own, so a lone request pays a few passes of deferral and
                // nothing else - and at expiry the merged attempt runs BEFORE the single-lane turn,
                // which is what gives it the first shot at the lanes it just waited for.
                if (!hold_prefill && prefill_merge_active() &&
                    scheduling_pass_ - prefill_admission_pass_ < kMergeWindowPasses) {
                    hold_prefill = true;
                }
                // The merged attempt is NOT gated by the hold: the hold exists to keep the lanes
                // staged long enough for a batch to form, and the pass is what consumes it. Gating
                // the attempt on "not holding" meant the merge was only ever tried in the passes
                // where the hold had already released the lanes to the single-lane turn - measured:
                // zero [prefill-reject] lines in a whole burst, i.e. the pass was never attempted.
                auto merged = prefill_merge_active() && !scheduler_.progress_owner()
                                  ? instance_.program->advance_prefill_batch()
                                  : std::nullopt;
                if (merged) {
                    // Q67/S1: one merged pass carried several lanes' chunks. Every row that finished
                    // owns a pending token, and committing a multi-row pending is the same call the
                    // decode batch uses.
                    for (std::size_t row = 0; row < merged->count; ++row) {
                        if (auto& request = slots_[merged->lanes[row]]; request != nullptr) {
                            request->begin = merged->summaries[row];
                        }
                    }
                    if (merged->progress.pending) {
                        commit_pending(
                            std::move(*merged->progress.pending),
                            std::span<const std::uint32_t>(merged->lanes.data(), merged->count), false,
                            snapshot_cancellations());
                    }
                    executed = true;
                } else if (hold_prefill) {
                    // A deliberate pass, not an idle one: the loop must not sleep waiting for work
                    // it just chose not to do.
                    executed = true;
                } else if (const auto lane = scheduler_.next_prefill(prefill_slots, max_concurrency_)) {
                    // v8 9.3: prefill/replay dispatch - the layers this pass runs are prefill
                    // layers, whatever the serial rule thinks of the traffic.
                    models::qwen3_5::execution::policy_controller().observe_prefill(true);
                    const auto request = slots_[*lane];
                    set_host_work_class(HostWorkClass::Prefill);
                    if (request->is_replaying()) {
                        ProgramCallScope call(*this);
                        auto progress = instance_.program->advance_replay(*request->sequence,
                                                                          &call.failed_timing());
                        call.finish(progress.timing);
                        record_recovery(request->id, execution_elapsed_ns(progress.timing));
                        if (!request->first_output_timing) {
                            record_execution_work(request->replay_work, progress.timing);
                        }
                        request->replayed_tokens += progress.processed_tokens;
                        cumulative_stats_.replayed_tokens += progress.processed_tokens;
                        // The frontier advances only with completed work, and it is what a monitor
                        // reports as "how far the rebuild has got".
                        request->recovery.frontier = progress.frontier;
                        if (progress.capture_ready) {
                            reserve_active_capture(request, EngineRequestState::Replay);
                        }
                        if (progress.complete) {
                            ++cumulative_stats_.replay_restores;
                            ++request->replay_restores;
                            request->model_state = request->resume_phase;
                            observe_scheduling(request,
                                               GenerationSchedulingTransition::ReplayComplete);
                        }
                    } else {
                        worker_phase_ = "prefill step";
                        run_prefill_step(*lane, snapshot_cancellations());
                    }
                    scheduler_.prefill_executed(*lane, max_concurrency_);
                    update_recovery(request);
                    executed = true;
                }
                // docs/design-kv-address-lifecycle.md step 3: one reclaim per scheduling pass, on
                // the thread that owns the addresses. It runs before the gauges are published so
                // the reported occupancy already includes what this pass gave back. An address only
                // becomes Retired when its owner ends, so with nothing retired this is one counter
                // compare -- and a Retired address whose pages are still pinned stays Retired for
                // the next pass instead of aborting the engine.
                if (instance_.program != nullptr) {
                    worker_phase_ = "reclaim";
                    instance_.program->reclaim_retired();
                }
                publish_runtime_stats();
            } catch (...) {
                auto error = std::current_exception();
                log_worker_failure(error);
                fail_all_locked(error);
                return;
            }
            execution_lock.unlock();
            if (!executed) {
                std::unique_lock lock(queue_mutex_);
                queue_cv_.wait_for(lock, std::chrono::milliseconds(1));
            }
        }
    }

    Instance& instance_;
    DeviceContext& device_;
    const std::uint32_t max_context_;    const std::uint32_t max_concurrency_;
    const std::size_t max_outstanding_;
    // v8 9.3/12.2: the controller-owned lane limit. The buffers above stay sized for
    // max_concurrency_ forever - shrinking is an admission decision, never a reallocation - so
    // the only writers are the worker thread itself (bus applier) and the only reader is the
    // scheduling code on the same thread; the atomic is for the public setter's contract, not
    // for contention.
    std::atomic<std::uint32_t> active_concurrency_;
    const std::chrono::milliseconds pending_timeout_;
    ResourceManagement resources_;

    // v8 12.5 step 4: the parked expert-pool target, in MiB (0 = none). Written by the bus
    // executor from the worker thread itself (park_expert_pool_resize), read by the worker's
    // own scheduling point between rounds; the atomic is for the notify/wait handshake, not
    // for contention. resize_mutex_ serializes the two entry points of the idle contract
    // (external Engine::resize_expert_pool and the worker's apply_pending_pool_resize).
    std::atomic<double> pending_pool_resize_mb_{0.0};
    // 13.4: the parked KV page target (0 = none). Same handshake as the pool one; kept in page
    // groups because that is the unit the pool's edge is measured in.
    std::atomic<double> pending_kv_pages_{0.0};
    std::mutex resize_mutex_;

    // T131: the control plane's parked write (see park_control). control_pending_ is the wake
    // predicate the worker's wait consults; the payload and the published state live behind
    // control_mutex_. The freeze values below are the worker's own copy: initial_* is what the
    // process configured (the "keep initial" half of the mode bundles), the bare names are what
    // is in force after the last control write.
    std::atomic<bool> control_pending_{false};
    mutable std::mutex control_mutex_;
    std::condition_variable control_cv_;
    ControlRequest pending_control_{};
    ControlState control_state_{};
    std::uint64_t control_sequence_      = 0;
    std::uint32_t initial_freeze_after_    = 4;
    std::uint32_t initial_freeze_reselect_ = 0;
    bool          initial_freeze_thaw_     = false;
    std::uint32_t freeze_after_    = 4;
    std::uint32_t freeze_reselect_ = 0;
    bool          freeze_thaw_     = false;
    bool          mode_from_control_ = false;

    // Timed so a caller that arrives during the settle round of a just-finished request can wait
    // a bounded 80 ms for it instead of being told the engine is busy (see the resize entries).
    mutable std::timed_mutex execution_mutex_;
    mutable std::mutex queue_mutex_;
    mutable std::mutex stats_mutex_;
    std::condition_variable queue_cv_;
    std::deque<std::shared_ptr<Request>> pending_;
    // Mutated only under queue_mutex_; atomic so the resize verdict log can read it while it
    // already holds that lock (see outstanding_snapshot).
    std::atomic<std::size_t> outstanding_{0};
    std::atomic<std::uint64_t> resize_accepted_inflight_{0};
    std::uint64_t next_request_id_        = 1;
    std::uint64_t next_publication_order_ = 1;
    std::array<std::shared_ptr<Request>, kMaximumConcurrency> slots_{};
    std::optional<MaterializingRequest> materializing_;
    // When the current materialization started. A materialization that neither completes nor
    // advances is a hang: measured 2026-11-23, `materializing 1` with 0% host for minutes after a
    // cold 163k request. The lane is held for the whole time and nothing times it out, so a
    // watchdog here is the only exit.
    std::uint64_t materializing_since_ms_ = 0;
    std::optional<AdmissionDecision> admission_decision_;
    std::uint64_t admission_generation_ = 1;
    std::array<std::optional<CaptureDecision>, kMaximumConcurrency> capture_decisions_;
    std::shared_ptr<Request> context_owner_;
    // appendix HX (plan item 9b): the idle trigger's state. The threshold is read ONCE (not per
    // round, as item 4 did) and the sweep runs at most once per threshold window, so a refusal
    // cannot be counted once per round - the mistake appendix HM had to fix for the lane trigger.
    // 13.4: the KV slow dial's window input (peak edge occupancy since the last window).
    double kv_window_peak_ = 0.0;
    std::uint64_t idle_reclaim_ms_ = read_idle_reclaim_ms();
    std::uint64_t last_idle_sweep_ms_ = 0;
    // port (sm_89): one-shot validation of the elastic expert pool. NINFER_EXPERT_POOL_CYCLE
    // carries "step,percent": at that decode round the pool is handed back down to `percent` of
    // its capacity and restored on the next round. The knob exists to measure what the elastic
    // reservation costs and saves; nothing chooses it automatically yet.
    bool expert_pool_cycle_parsed_  = false;
    std::uint64_t expert_pool_step_ = 0;
    std::uint64_t expert_pool_full_ = 0;
    int expert_pool_percent_        = 0;
    std::uint64_t expert_pool_rounds_ = 0;
    std::vector<std::shared_ptr<Request>> paused_;
    Scheduling scheduler_;
    // Q67/S1: read once, so the report costs a pointer comparison per prefill pass and nothing on
    // the decode path. NINFER_PREFILL_BATCH prints what a merged prefill would be.
    bool prefill_batch_report_ = std::getenv("NINFER_PREFILL_BATCH") != nullptr;
    // 13.5: the merged prefill's state, promoted out of the environment. The controller owns it
    // (PolicyController::prefill_merge, driven by the measured fit of the queued lanes' token
    // counts); NINFER_PREFILL_BATCH_EXEC / NINFER_PREFILL_ADMIT_FIRST survive as explicit overrides
    // so an A/B arm can pin the state, and pinning turns the actuator into a shadow - a run that
    // says "merge off" must not be merged into by a controller halfway through.
    bool prefill_merge_from_env_      = std::getenv("NINFER_PREFILL_BATCH_EXEC") != nullptr;
    bool prefill_admit_from_env_      = std::getenv("NINFER_PREFILL_ADMIT_FIRST") != nullptr;
    bool prefill_merge_pinned_        = prefill_merge_from_env_ || prefill_admit_from_env_;
    std::atomic<bool> prefill_merge_enabled_{false};
    std::atomic<bool> prefill_admit_first_{false};
    // Last count of requests with prefill work that no row of a pass has picked up yet; printed by
    // the prefill report, which is how the merge bootstrap is observed.
    std::size_t prefill_merge_waiting_ = 0;
    // Consecutive passes that asked for the same flip; two is enough because the state it reacts to
    // lasts only a few passes per request burst.
    std::uint32_t prefill_merge_agree_ = 0;
    // Scheduling-pass clock and the last admission's stamp: the batching window that lets a burst
    // reach a dispatch together. Iterations of the scheduling loop, not milliseconds - the loop's
    // period varies with the work it does.
    static constexpr std::uint64_t kMergeWindowPasses = 8;
    std::uint64_t scheduling_pass_         = 0;
    std::uint64_t prefill_admission_pass_  = 0;
    // v8 13.4 item 4, second half: opt-in joiner pacing. NINFER_PREFILL_ADMIT_FIRST holds the
    // prefill until every queued joiner is bound; this knob holds a *new* joiner when the
    // merge-window column budget has no room for it, so it does not bind into a pass it cannot
    // join and waits for the next window instead.
    bool prefill_pacing_ = std::getenv("NINFER_PREFILL_PACING") != nullptr;
    // "all" (or any value starting with 'a') prints every dispatch instead of the first few.
    bool prefill_batch_verbose_ = [] {
        const char* value = std::getenv("NINFER_PREFILL_BATCH");
        return value != nullptr && (value[0] == 'a' || value[0] == 'A');
    }();
    // See the capture_ready branch: 0 keeps the prompt-frontier capture off the first-token path
    // and relies on the finish-time publish for the reusable record.
    bool capture_prompt_frontier_ = [] {
        const char* value = std::getenv("NINFER_CAPTURE_PROMPT_FRONTIER");
        return value == nullptr || value[0] != '0';
    }();
    std::uint64_t prefill_batch_reports_ = 0;
    // Q67/S1: admission probe counters, see try_admit_one.
    std::uint64_t admit_probe_calls_   = 0;
    std::uint64_t admit_probe_tx_busy_  = 0;
    std::uint64_t admit_probe_no_lane_  = 0;
    std::uint64_t admit_probe_ok_       = 0;
    // Q67/S1: set by try_admit_one when a candidate was refused for capacity rather than ordering.
    bool admission_deferred_ = false;
    std::array<bool, kMaximumConcurrency> runnable_units_{};
    std::atomic<bool> admission_check_pending_{false};
    std::uint64_t worker_accounted_elapsed_ns_ = 0;
    HostWorkClass current_host_work_class_     = HostWorkClass::Control;
    std::array<std::uint32_t, kMaximumConcurrency> current_decode_lanes_{};
    std::size_t current_decode_lane_count_ = 0;
    RuntimeStats cumulative_stats_;
    RuntimeStats published_stats_;
    std::vector<RequestSnapshot> published_requests_;
    // The memory gauges published with published_stats_ (see memory_summary). Written by the
    // worker at publication, read by the monitor threads under stats_mutex_.
    MemorySummary published_memory_;
    bool memory_published_ = false;
    bool stopping_ = false;
    bool failed_   = false;
    std::thread worker_;
    // Hang watchdog (see hang_watchdog_loop): the worker stamps last_pass_ms_ every loop turn, and a
    // second thread reports when the stamp stops moving. The worker cannot do this itself - the whole
    // failure mode is a turn that never returns.
    std::atomic<std::uint64_t> last_pass_ms_{0};
    std::atomic<bool> watchdog_stop_{false};
    std::thread watchdog_;
};

} // namespace ninfer::runtime

#include "runtime/engine/engine_metrics.inl"
