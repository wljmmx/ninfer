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
#include "models/qwen3_5/execution/shape_class.h"
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

    EngineCore(Instance& instance, DeviceContext& device, const EngineOptions& options,
               ContextMachineCostModel context_cost)
        : instance_(instance), device_(device), max_context_(options.max_context),
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
    }

    ~EngineCore() noexcept {
        {
            std::lock_guard lock(queue_mutex_);
            stopping_ = true;
        }
        queue_cv_.notify_all();
        if (worker_.joinable()) { worker_.join(); }
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

    [[nodiscard]] MemorySummary memory_summary() const {
        std::scoped_lock lock(execution_mutex_);
        MemorySummary out                      = instance_.program->memory_summary();
        const KvCapacityResolution& resolution = instance_.kv_capacity_resolution;
        out.kv_capacity_mode                   = resolution.mode;
        out.kv_capacity_page_groups            = resolution.main_page_groups;
        out.kv_capacity_max_page_groups        = resolution.maximum_main_page_groups;
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
        return out;
    }

    [[nodiscard]] RuntimeStats runtime_stats() const {
        std::lock_guard lock(stats_mutex_);
        return published_stats_;
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
                    return request->response_done || request->stream_start.has_value() ||
                           request->stream_progress.has_value() || !request->events.empty();
                });
                start = std::move(request->stream_start);
                request->stream_start.reset();
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
                            sink->publish(std::move(std::get<OutputDelta>(event)));
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
            std::any_of(output.begin(), output.end(),
                        [](const auto& delta) { return !delta.text.empty(); })) {
            if (phase) { phase->checkpoint(); }
            record_first_output(request);
        }
        const bool streaming = request->consumer_mode == OutputConsumerMode::Streaming;
        {
            std::lock_guard lock(request->mutex);
            if (streaming && timing) { request->events.emplace_back(std::move(*timing)); }
            for (OutputDelta& delta : output) {
                std::string& full = delta.channel == OutputChannel::Reasoning ? request->reasoning
                                                                              : request->content;
                full += delta.text;
                if (streaming) { request->events.emplace_back(std::move(delta)); }
            }
        }
        if (streaming) { request->cv.notify_one(); }
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
        if (request->consumer_mode != OutputConsumerMode::Streaming) { return; }
        {
            std::lock_guard lock(request->mutex);
            if (request->stream_start || request->response_done) {
                throw std::logic_error("streaming request has an invalid generation-start state");
            }
            request->stream_start = GenerationStart{
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

    bool mark_completed(const std::shared_ptr<Request>& request) noexcept {
        bool release = false;
        {
            std::lock_guard lock(request->mutex);
            if (request->consumer_released && !request->capacity_released) {
                request->capacity_released = true;
                release                    = true;
            }
        }
        return release;
    }

    void release_planning_state(const std::shared_ptr<Request>& request) noexcept {
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
    }

    void complete_error(const std::shared_ptr<Request>& request, std::exception_ptr error) {
        if (request->preemption_count) {
            observe_scheduling(request, GenerationSchedulingTransition::Terminal);
        }
        release_planning_state(request);
        request->prompt      = {};
        request->model_state = EngineRequestState::ModelFinished;
        request->sequence.reset();
        request->lane.reset();
        request->budget.reset();
        request->terminal_reason.reset();
        {
            std::lock_guard lock(request->mutex);
            if (request->response_done) { return; }
            request->error         = std::move(error);
            request->response_done = true;
        }
        if (mark_completed(request)) { release_reserved_capacity(); }
        request->cv.notify_one();
    }

    void complete_success(const std::shared_ptr<Request>& request, FinishReason reason) {
        HostPhaseMeasurement completion = begin_host_phase();
        if (request->preemption_count) {
            observe_scheduling(request, GenerationSchedulingTransition::Terminal);
        }
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
        release_planning_state(request);
        request->prompt      = {};
        request->model_state = EngineRequestState::ModelFinished;
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
        request->sequence.reset();
        request->lane.reset();
        request->budget.reset();
        request->terminal_reason.reset();
        finish_engine_phase(completion, EngineHostPhase::CommitOutput);
        result.engine_timing       = request->host_timing.public_snapshot();
        result.first_output_timing = request->first_output_timing;
        {
            std::lock_guard lock(request->mutex);
            if (request->response_done) { return; }
            request->result        = std::move(result);
            request->response_done = true;
        }
        if (mark_completed(request)) { release_reserved_capacity(); }
        request->cv.notify_one();
    }

    void complete_cancelled(const std::shared_ptr<Request>& request) {
        (void)request->output.preview_terminal(FinishReason::Cancelled);
        append_output(request, request->output.commit_preview());
        complete_success(request, FinishReason::Cancelled);
    }

    void complete_detached_cancelled(const std::shared_ptr<Request>& request) {
        try {
            complete_cancelled(request);
        } catch (...) {
            const std::exception_ptr error = std::current_exception();
            complete_error(request, error);
            throw;
        }
    }

    void remove_completed_slot(std::uint32_t lane) {
        capture_decisions_[lane].reset();
        slots_[lane].reset();
        request_admission_check();
        scheduler_.capacity_released();
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
            auto finished             = instance_.program->finish(*request->sequence);
            if (finished.status != ConsumeStatus::Consumed) {
                throw std::logic_error("terminal native sequence could not finish");
            }
            if (finished.checkpoint) {
                resources_.publish(*instance_.program, request->continuation_owner,
                                   *finished.checkpoint, monotonic_ms());
            }
            resources_.finish(*instance_.program, request->continuation_owner, *request->base_plan,
                              request->publication_order);
            request->continuation_owner = 0;
            request->generation_timings = finished.timings;
            request->speculative_stats  = std::move(finished.speculative);
            request->terminal_reason.reset();

            finish_engine_phase(boundary, EngineHostPhase::Boundary);
            complete_success(request, reason);
            remove_completed_slot(lane);
            boundary = begin_host_phase();
            changed  = true;
        }
        if (changed) { publish_runtime_stats(); }
        return changed;
    }

    void cancel_active_requests(const std::array<bool, kMaximumConcurrency>& cancelled_at_boundary,
                                HostPhaseMeasurement& boundary) {
        if (instance_.program->has_context_transaction()) { return; }
        bool changed = false;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
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
            remove_completed_slot(lane);
            boundary = begin_host_phase();
            changed  = true;
        }
        if (changed) { publish_runtime_stats(); }
    }

    [[nodiscard]] bool expire_pending_requests() {
        std::vector<std::shared_ptr<Request>> cancelled;
        std::vector<std::shared_ptr<Request>> expired;
        bool have_pending = false;
        {
            std::lock_guard lock(queue_mutex_);
            const auto now = Clock::now();
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
            for (const auto& request : cancelled) { complete_detached_cancelled(request); }
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
        if (!cancelled.empty() || !expired.empty()) {
            request_admission_check();
            publish_runtime_stats();
        }
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

        std::array<CommitDecision, kMaximumConcurrency> decisions{};
        std::array<FinishReason, kMaximumConcurrency> finish_reasons{};
        std::array<ContinuationAction, kMaximumConcurrency> continuations{};
        std::array<std::size_t, kMaximumConcurrency> generated_sizes{};
        std::array<bool, kMaximumConcurrency> cancelled{};
        std::array<bool, kMaximumConcurrency> failed{};
        bool generated_staged = false;
        std::array<std::shared_ptr<Request>, kMaximumConcurrency> terminal_requests{};
        std::array<std::uint32_t, kMaximumConcurrency> terminal_lanes{};
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
                const OutputDecision decision = request->output.preview_model(
                    row_tokens, request->budget->remaining(), request->budget->limit_reason());
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
                    remove_completed_slot(lane);
                    continue;
                }
                if (!cancelled[row]) { request->budget->commit(accepted); }
                update_recovery(request, cancelled[row]);
                auto published = request->output.commit_preview();
                auto timing    = record_committed_output(request, accepted);
                append_output(request, std::move(published), std::move(timing), &phase);
                if (decisions[row].terminal) {
                    if (cancelled[row]) {
                        terminal_requests[terminal_count] = request;
                        terminal_lanes[terminal_count]    = lane;
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
                    if (terminal_in_batch) {
                        instance_.program->skip_capture(*request->sequence);
                        request->model_state = post_capture_state;
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
                remove_completed_slot(terminal_lanes[index]);
            }
            throw;
        }
        phase.finish();
        for (std::size_t index = 0; index < terminal_count; ++index) {
            complete_success(terminal_requests[index], terminal_reasons[index]);
            remove_completed_slot(terminal_lanes[index]);
        }
    }

    void reserve_active_capture(const std::shared_ptr<Request>& request, EngineRequestState next) {
        request->model_state        = next;
        request->capture_pending    = true;
        request->post_capture_state = next;
        if (instance_.program->has_context_transaction()) { return; }
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
                    if (!victims->empty()) { scheduler_.capacity_released(); }
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
    // v8 12.5 step 4 + 13.5-5: THE idle contract for an expert-pool resize, in ONE place. The
    // nine gauge counters plus the live try-lock check first, then a device synchronize and a
    // second read - a request that started during the synchronize is the one the second check
    // refuses (found by the resize-under-load acceptance on the e2 artifact, 2026-11-05).
    // Engine::resize_expert_pool forwards external callers here, and the worker's own
    // scheduling point (apply_pending_pool_resize) walks the SAME code, so there is no second,
    // weaker copy to drift. resize_mutex_ serializes the two entry points; the queue-side race
    // is still covered by the double read, which is the part that makes a wrong-but-atomic
    // answer impossible.
    [[nodiscard]] std::uint64_t resize_expert_pool_checked(std::uint64_t target_bytes) {
        std::lock_guard serialize(resize_mutex_);
        const RuntimeStats stats = runtime_stats();
        const std::uint64_t busy = static_cast<std::uint64_t>(stats.running_requests) +
                                   stats.prefilling_requests + stats.decode_ready_requests +
                                   stats.waiting_requests + stats.paused_requests +
                                   stats.replaying_requests + stats.materializing_requests +
                                   stats.capture_pending_requests + stats.terminal_pending_requests;
        if (busy != 0 || any_inflight()) {
            throw std::logic_error("expert pool resize requires an idle engine");
        }
        device_.synchronize();
        const RuntimeStats again = runtime_stats();
        if (any_inflight() || again.running_requests != 0 || again.waiting_requests != 0 ||
            again.paused_requests != 0 || again.prefilling_requests != 0 ||
            again.decode_ready_requests != 0) {
            throw std::logic_error("expert pool resize lost its idle window");
        }
        return instance_.resize_expert_pool(target_bytes);
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

    // The worker's own scheduling point: apply a parked pool target while idle. Runs OUTSIDE
    // execution_mutex_ on purpose - any_inflight()'s try-lock must be able to see the queue, and
    // the worker must not hold the very lock the contract probes. A refusal is logged and the
    // bus is noted back to the pre-step capacity; it is never bypassed.
    void apply_pending_pool_resize() {
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

    [[nodiscard]] bool any_inflight() const {
        std::unique_lock execution_lock(execution_mutex_, std::try_to_lock);
        if (!execution_lock.owns_lock()) { return true; }
        std::lock_guard lock(queue_mutex_);
        // The decision containers are the boundary states a request can sit in between the
        // queue and a lane: a mid-admission scan, an armed capture, or a transaction owner.
        // The first draft of this check read only the settled states and a resize slipped
        // through exactly one of these - the controller had just flipped the lane count and
        // the batch was neither queued nor resident for the length of the flip.
        const bool holding_decision =
            admission_decision_.has_value() || context_owner_ != nullptr ||
            std::any_of(capture_decisions_.begin(), capture_decisions_.end(),
                        [](const auto& decision) { return decision.has_value(); });
        return !pending_.empty() || !resident_empty() || !paused_.empty() || holding_decision;
    }

private:

    [[nodiscard]] bool has_older_resident(std::uint64_t ticket) const noexcept {
        return std::any_of(slots_.begin(), slots_.end(), [ticket](const auto& request) {
            return request && request->id < ticket;
        });
    }

    void record_context_work(const typename ModelContract::ContextProgress& progress,
                             const std::shared_ptr<Request>& owner);

    bool progress_context_transaction(HostPhaseMeasurement& boundary) {
        if (!instance_.program->has_context_transaction()) { return false; }
        const auto owner = materializing_ ? materializing_->request : context_owner_;
        const auto cancellation =
            owner ? CancellationFlagView{&owner->cancelled} : CancellationFlagView{};
        auto progress = instance_.program->poll_context(cancellation);
        if (!progress.complete) { return progress.advanced; }
        record_context_work(progress, owner);
        if (owner && progress.request_timings) {
            owner->generation_timings = *progress.request_timings;
            owner->speculative_stats  = std::move(progress.request_speculative);
        }
        if (materializing_) {
            const auto control  = std::move(*materializing_);
            const auto& request = control.request;
            if (!progress.published) {
                complete_detached_cancelled(request);
                materializing_.reset();
                scheduler_.capacity_released();
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
            request->model_state = progress.replaying ? EngineRequestState::Replay
                                   : control.resumed  ? request->resume_phase
                                                      : EngineRequestState::Prefill;
            if (control.resumed) {
                if (!progress.replaying) {
                    ++cumulative_stats_.snapshot_restores;
                    ++request->snapshot_restores;
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
                    slots_[lane].reset();
                    request->lane.reset();
                    request->sequence.reset();
                    context_owner_.reset();
                    complete_detached_cancelled(request);
                    scheduler_.capacity_released();
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
                // Yielding a borrower may unblock an older suspended request. The newly
                // paused request itself waits for a later event, avoiding immediate churn.
                if (paused_.front() != request) { scheduler_.capacity_released(); }
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
                        resources_.publish_to_cache(*instance_.program, *request->base_plan, point);
                    }
                }
            }
            context_owner_.reset();
        }
        if (progress.kind == decltype(progress.kind)::Demote) { scheduler_.capacity_released(); }
        request_admission_check();
        return true;
    }

    bool reclaim(ContextResourceUsage shortage, bool allow_pause,
                 std::span<const Checkpoint> excluded               = {},
                 typename ResourceManagement::ReclaimCursor* cursor = nullptr,
                 std::optional<std::uint64_t> priority_ticket       = std::nullopt) {
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
                scheduler_.capacity_released();
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
        return allow_pause && pause_reclaim_victim(priority_ticket);
    }

    bool pause_reclaim_victim(std::optional<std::uint64_t> priority_ticket) {
        const auto victim = scheduler_.youngest_victim(slots_, max_concurrency_, priority_ticket);
        if (!victim) { return false; }
        return pause_resident(*victim);
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
        scheduler_.preempted();
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
        if (!lane && restoring) {
            if (!paused_.empty() && !has_older_resident(paused_.front()->id)) {
                if (const auto victim =
                        scheduler_.youngest_victim(slots_, max_concurrency_, paused_.front()->id)) {
                    return pause_resident(*victim);
                }
            }
            return false;
        }
        if (admission_decision_ &&
            (admission_decision_->restoring != restoring ||
             (restoring && (paused_.empty() || paused_.front() != admission_decision_->request)))) {
            admission_decision_.reset();
        }
        std::vector<std::shared_ptr<Request>> candidates;
        if (admission_decision_) {
            candidates.push_back(admission_decision_->request);
        } else if (restoring) {
            if (!paused_.empty()) { candidates.push_back(paused_.front()); }
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
                    if (reclaim(shortage, restoring && !has_older_resident(request->id), excluded,
                                &decision.reclaim,
                                restoring ? std::optional(request->id) : std::nullopt)) {
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
                materializing_.emplace(
                    MaterializingRequest{request, LaneId{*lane}, begin, restoring, choice});
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
                    scheduler_.restoration_blocked();
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
                scheduler_.restoration_blocked();
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
            scheduler_.capacity_released();
            request_admission_check();
        }
    }

    enum class ReservationScope { Round, Prefill };

    void reserve_resident_units(ReservationScope scope = ReservationScope::Round) {
        runnable_units_.fill(false);
        std::optional<std::uint32_t> recovering;
        std::optional<std::uint32_t> oldest;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (!request || !request->sequence || request->is_model_finished()) { continue; }
            if (!oldest || request->id < slots_[*oldest]->id) { oldest = lane; }
            if (request->recovery_pending) {
                if (recovering) { throw std::logic_error("multiple protected recoveries"); }
                recovering = lane;
            }
        }
        const auto prefill = recovering && (slots_[*recovering]->is_prefilling() ||
                                            slots_[*recovering]->is_replaying())
                                 ? recovering
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
            if (recovering == a || recovering == b) { return recovering == a; }
            return slots_[a]->id < slots_[b]->id;
        });
        std::optional<std::uint32_t> blocked;
        for (const auto lane : candidates) {
            const auto request = slots_[lane];
            // Reclaiming for an earlier row may already have paused this candidate.
            if (!request || !request->sequence || request->capture_pending ||
                request->cancelled.load(std::memory_order_acquire) ||
                instance_.program->context_blocks(*request->sequence) ||
                !(request->is_control_ready() || request->is_decode_ready() ||
                  request->is_prefilling() || request->is_replaying())) {
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
            std::optional<typename ResourceManagement::ReclaimCursor> cursor;
            for (;;) {
                const auto result = instance_.program->reserve_units({&unit, 1});
                if (result) {
                    runnable_units_[lane] = true;
                    break;
                }
                if (instance_.program->has_context_transaction()) { break; }
                if (!cursor) { cursor.emplace(resources_.begin_reclaim(*instance_.program)); }
                if (reclaim(result.shortage, false, {}, &*cursor)) {
                    if (!instance_.program->has_context_transaction()) { continue; }
                    break;
                }
                if (recovering == lane) {
                    throw std::logic_error("protected recovery exceeded its Native reservation");
                }
                if (!recovering && oldest == lane) {
                    // Only the oldest required unit can displace younger residents. Other
                    // failed rows wait or yield their own lane; they cannot block ready rows.
                    // The same shortage has already exhausted cache and paused snapshots.
                    if (pause_reclaim_victim(request->id)) { break; }
                    throw std::logic_error("oldest resident cannot obtain its legal unit");
                }
                if (oldest != lane && (!blocked || request->id > slots_[*blocked]->id)) {
                    blocked = lane;
                }
                break;
            }
        }
        if (blocked && !instance_.program->has_context_transaction()) {
            (void)pause_resident(*blocked);
        }
        if (prefill && runnable_units_[*prefill] && !prepare_semantic_capture(slots_[*prefill])) {
            runnable_units_[*prefill] = false;
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
            auto timing = record_committed_output(request, membership.row_stride);
            append_output(request, request->output.commit_preview(), std::move(timing), &phase);
            update_recovery(request);
            request->model_state = EngineRequestState::DecodeReady;
        }
        publish_runtime_stats();
    }

    // The worker holds execution_mutex_ across the failing operation and this cleanup, so no
    // Program introspection can observe a partially cleared physical state.
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
        for (auto& request : paused_) {
            request->suspended.reset();
            complete_error(request, error);
        }
        paused_.clear();
        resources_.release_all(*instance_.program);
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) {
                complete_error(slots_[lane], error);
                slots_[lane].reset();
            }
        }
        if (materializing_request != nullptr) { complete_error(materializing_request, error); }
        for (const auto& request : pending) { complete_error(request, error); }
        publish_runtime_stats();
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
        // The construction-time whole-step graph decision, mirrored for print (the [bus] row is
        // no-executor by the honest register): expert_pool_pinned is set from graph_ok in
        // model_instance.cpp, and a pinned pool is exactly the state a captured decode graph
        // froze its offsets against.
        models::qwen3_5::execution::param_bus().note_value(
            "graph_on", instance_.expert_pool_pinned ? 1.0 : 0.0);
        for (;;) {
            bool idle_for_pool_dial = false;
            {
                std::unique_lock lock(queue_mutex_);
                const bool pool_dial_pending =
                    pending_pool_resize_mb_.load(std::memory_order_acquire) > 0.0;
                if (!stopping_ && pending_.empty() && resident_empty() && paused_.empty() &&
                    !instance_.program->has_context_transaction() && !pool_dial_pending) {
                    queue_cv_.wait(lock, [&] {
                        return stopping_ || !pending_.empty() ||
                               pending_pool_resize_mb_.load(std::memory_order_acquire) > 0.0;
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
                // queued, nothing resident, nothing paused, no transaction. When this is false
                // the parked target STAYS parked - a busy engine is not a refusal, it is a
                // "not yet", and the dial's next window re-reads the bus either way.
                idle_for_pool_dial =
                    pending_.empty() && resident_empty() && paused_.empty() &&
                    !instance_.program->has_context_transaction() &&
                    pending_pool_resize_mb_.load(std::memory_order_acquire) > 0.0;
            }
            if (idle_for_pool_dial) {
                apply_pending_pool_resize();
                continue; // re-evaluate the queue before the next round starts
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
                    complete_detached_cancelled(request);
                    scheduler_.capacity_released();
                }
                const bool recovering =
                    std::any_of(slots_.begin(), slots_.end(),
                                [](const auto& r) { return r && r->recovery_pending; }) ||
                    (materializing_ && materializing_->resumed);
                if (paused_.empty() && !context_owner_ && !recovering) {
                    scheduler_.paused_queue_empty();
                }
                if (!recovering &&
                    ((admission_decision_ && admission_decision_->restoring) ||
                     scheduler_.should_restore(!paused_.empty(), resident_empty()))) {
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
                if (idle_reclaim_ms_ != 0 && paused_.empty() &&
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
                if (!admission_decision_ || !admission_decision_->restoring) {
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
                    run_decode_round(decode, cancelled);
                    executed = true;
                }
                if (instance_.program->has_context_transaction()) {
                    set_host_work_class(HostWorkClass::Control);
                    auto progress_boundary = begin_host_phase();
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
                if (prefill_batch_report_) {
                    std::array<std::uint32_t, kMaximumConcurrency> remaining{};
                    std::size_t queued      = 0;
                    std::size_t engine_state = 0;  // engine says prefilling
                    std::size_t capture_held = 0;  // ...but a capture has not landed yet
                    std::size_t bound_ready   = 0;  // Program agrees the lane still has a chunk
                    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                        if (!prefill_slots[lane]) { continue; }
                        if (prefill_slots[lane]->is_prefilling()) { ++engine_state; }
                        if (prefill_slots[lane]->capture_pending) { ++capture_held; }
                        const std::uint32_t left = instance_.program->prefill_remaining(lane);
                        if (left == 0) { continue; }
                        ++bound_ready;
                        remaining[queued++] = left;
                    }
                    const auto fit = models::qwen3_5::execution::prefill_batch_fit(
                        std::span<const std::uint32_t>(remaining.data(), queued));
                    ++prefill_batch_reports_;
                    // NINFER_PREFILL_BATCH=all prints every dispatch, not just the first few: a
                    // sampled report hid whether the later dispatches already see two ready lanes,
                    // and that is the whole question this counter exists to answer.
                    if (prefill_batch_verbose_ || prefill_batch_reports_ <= 8 ||
                        prefill_batch_reports_ % 32 == 0) {
                        std::fprintf(stderr,
                                     "[prefill-batch] queued=%zu pick=%d lanes=%zu "
                                     "merged=%.0fms separate=%.0fms | engine_prefill=%zu "
                                     "capture_held=%zu bound_ready=%zu | admit calls=%llu "
                                     "tx_busy=%llu no_lane=%llu ok=%llu\n",
                                     queued, fit.width, fit.lanes, fit.merged_ms,
                                     fit.separate_ms, engine_state, capture_held, bound_ready,
                                     static_cast<unsigned long long>(admit_probe_calls_),
                                     static_cast<unsigned long long>(admit_probe_tx_busy_),
                                     static_cast<unsigned long long>(admit_probe_no_lane_),
                                     static_cast<unsigned long long>(admit_probe_ok_));
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
                if (prefill_admit_first_) {
                    // Hold while anything is still queued - not only while a bind can start right
                    // now. Binding is a context transaction that opens and closes over several loop
                    // passes; holding only when it is free let a head slip in between two binds, and
                    // then no two lanes ever had a chunk at the same time.
                    std::lock_guard lock(queue_mutex_);
                    hold_prefill = (!pending_.empty() || materializing_.has_value()) &&
                                   !admission_deferred_;
                }
                // The merged attempt obeys the same hold as the single-lane turn: admit-first
                // exists precisely so a queued batch binds before anything prefills, and a merge
                // that fires with half the lanes bound consumes the first two rows in one pass
                // and the rest in another - two narrow passes instead of the one the cost model
                // priced.
                auto merged = prefill_batch_exec_ && !hold_prefill
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
                        if (!request->first_output_timing) {
                            record_execution_work(request->replay_work, progress.timing);
                        }
                        request->replayed_tokens += progress.processed_tokens;
                        cumulative_stats_.replayed_tokens += progress.processed_tokens;
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
                        run_prefill_step(*lane, snapshot_cancellations());
                    }
                    scheduler_.prefill_executed(*lane, max_concurrency_);
                    update_recovery(request);
                    executed = true;
                }
                publish_runtime_stats();
            } catch (...) {
                auto error = std::current_exception();
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
    const std::uint32_t max_context_;
    const std::uint32_t max_concurrency_;
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
    std::mutex resize_mutex_;

    mutable std::mutex execution_mutex_;
    mutable std::mutex queue_mutex_;
    mutable std::mutex stats_mutex_;
    std::condition_variable queue_cv_;
    std::deque<std::shared_ptr<Request>> pending_;
    std::size_t outstanding_              = 0;
    std::uint64_t next_request_id_        = 1;
    std::uint64_t next_publication_order_ = 1;
    std::array<std::shared_ptr<Request>, kMaximumConcurrency> slots_{};
    std::optional<MaterializingRequest> materializing_;
    std::optional<AdmissionDecision> admission_decision_;
    std::uint64_t admission_generation_ = 1;
    std::array<std::optional<CaptureDecision>, kMaximumConcurrency> capture_decisions_;
    std::shared_ptr<Request> context_owner_;
    // appendix HX (plan item 9b): the idle trigger's state. The threshold is read ONCE (not per
    // round, as item 4 did) and the sweep runs at most once per threshold window, so a refusal
    // cannot be counted once per round - the mistake appendix HM had to fix for the lane trigger.
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
    // Q67/S1: opt-in "admit before prefilling" hold; see the prefill dispatch.
    bool prefill_admit_first_ = std::getenv("NINFER_PREFILL_ADMIT_FIRST") != nullptr;
    // Q67/S1: opt-in merged prefill dispatch (interior chunks only).
    bool prefill_batch_exec_ = std::getenv("NINFER_PREFILL_BATCH_EXEC") != nullptr;
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
    bool stopping_ = false;
    bool failed_   = false;
    std::thread worker_;
};

} // namespace ninfer::runtime

#include "runtime/engine/engine_metrics.inl"
