#include "serve/operational_log.h"

#include "product/logging/pretty_format.h"
#include "product/speculative_options.h"

#include <spdlog/logger.h>

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

namespace ninfer::serve {
namespace {

const char* phase_name(RequestFailurePhase phase) noexcept {
    switch (phase) {
    case RequestFailurePhase::Prepare:
        return "prepare";
    case RequestFailurePhase::Generation:
        return "generation";
    case RequestFailurePhase::ResponseRender:
        return "response rendering";
    case RequestFailurePhase::ResponseStore:
        return "response storage";
    case RequestFailurePhase::Transport:
        return "transport";
    case RequestFailurePhase::Http:
        return "http";
    }
    return "unknown";
}

const char* classification_name(RequestFailureClass classification) noexcept {
    switch (classification) {
    case RequestFailureClass::ClientInput:
        return "client input";
    case RequestFailureClass::ClientDisconnected:
        return "client disconnected";
    case RequestFailureClass::Overload:
        return "overload";
    case RequestFailureClass::Timeout:
        return "timeout";
    case RequestFailureClass::Unavailable:
        return "unavailable";
    case RequestFailureClass::Upstream:
        return "upstream error";
    case RequestFailureClass::Internal:
        return "internal error";
    }
    return "unknown";
}

OperationalSeverity failure_severity(RequestFailureClass classification) noexcept {
    switch (classification) {
    case RequestFailureClass::ClientInput:
    case RequestFailureClass::ClientDisconnected:
        return OperationalSeverity::Info;
    case RequestFailureClass::Overload:
    case RequestFailureClass::Timeout:
    case RequestFailureClass::Unavailable:
    case RequestFailureClass::Upstream:
        return OperationalSeverity::Warning;
    case RequestFailureClass::Internal:
        return OperationalSeverity::Error;
    }
    return OperationalSeverity::Error;
}

const char* finish_reason_name(ninfer::FinishReason reason) noexcept {
    switch (reason) {
    case ninfer::FinishReason::None:
        return "none";
    case ninfer::FinishReason::OutputLimit:
        return "output limit";
    case ninfer::FinishReason::ContextCapacity:
        return "context capacity";
    case ninfer::FinishReason::StopToken:
        return "stop token";
    case ninfer::FinishReason::StopString:
        return "stop string";
    case ninfer::FinishReason::Cancelled:
        return "cancelled";
    }
    return "unknown";
}

const char* prefix_reuse_path_name(ninfer::PrefixReusePath path) noexcept {
    switch (path) {
    case ninfer::PrefixReusePath::Root:
        return "root";
    case ninfer::PrefixReusePath::Checkpoint:
        return "checkpoint";
    }
    return "unknown";
}

std::string_view requested_effort_name(const RequestLogContext& context) noexcept {
    if (context.requested_reasoning_effort)
        return requested_reasoning_effort_name(*context.requested_reasoning_effort);
    return "template default";
}

const char* protocol_name(std::string_view protocol) noexcept {
    if (protocol == "openai_chat_completions") { return "openai-chat"; }
    if (protocol == "openai_responses") { return "openai-responses"; }
    if (protocol == "anthropic_messages") { return "anthropic"; }
    if (protocol == "openai_responses_input_tokens") { return "openai-input-tokens"; }
    if (protocol == "anthropic_count_tokens") { return "anthropic-count-tokens"; }
    return "http";
}

const char* kv_cache_name(ninfer::KvCacheStorage storage) noexcept {
    switch (storage) {
    case ninfer::KvCacheStorage::BFloat16:
        return "bf16";
    case ninfer::KvCacheStorage::Int8Group64:
        return "int8";
    case ninfer::KvCacheStorage::Fp8E4M3Row256:
        return "fp8";
    case ninfer::KvCacheStorage::Nvfp4Group16:
        return "nvfp4";
    case ninfer::KvCacheStorage::Fp8KeyNvfp4Value:
        return "k8v4";
    }
    return "unknown";
}

const char* kv_capacity_mode_name(ninfer::KvCapacityMode mode) noexcept {
    return mode == ninfer::KvCapacityMode::Automatic ? "auto" : "explicit";
}

void append_clause(std::ostringstream& out, std::string_view clause) { out << " | " << clause; }

void append_counted_clause(std::ostringstream& out, std::string_view label, std::uint64_t count) {
    out << " | " << label << ' ' << product::format_pretty_count(count);
}

std::string pretty_code(std::string_view code) {
    std::string result;
    result.reserve(code.size());
    for (const char ch : code) { result.push_back(ch == '_' ? ' ' : ch); }
    return result;
}

template <class T>
T monotonic_delta(T previous, T current) noexcept {
    return current >= previous ? current - previous : T{};
}

std::uint64_t host_active_ns(const ThroughputReport& report) noexcept {
    const ninfer::RuntimeHostWorkStats& previous = report.previous.host_work;
    const ninfer::RuntimeHostWorkStats& current  = report.current.host_work;
    return monotonic_delta(previous.engine_boundary_ns, current.engine_boundary_ns) +
           monotonic_delta(previous.program_submit_ns, current.program_submit_ns) +
           monotonic_delta(previous.program_post_ns, current.program_post_ns) +
           monotonic_delta(previous.engine_commit_output_ns, current.engine_commit_output_ns) +
           monotonic_delta(previous.engine_maintenance_ns, current.engine_maintenance_ns);
}

void append_failure_fields(std::ostringstream& out, const RequestFailure& failure) {
    if (failure.http_status != 0) { out << " | HTTP " << failure.http_status; }
    if (!failure.error_code.empty()) {
        append_clause(out, pretty_code(failure.error_code));
    } else {
        append_clause(out, classification_name(failure.classification));
    }
    // appendix GH: the Internal class is the one whose rendering said nothing about the CAUSE, and
    // the record has carried it all along (machine_message, filled from the exception's what()).
    // Render a bounded single-line form of it, so an unexplained 500 is diagnosable from the log
    // instead of only from a re-run under a debugger. Every other class keeps its exact old line.
    if (failure.classification == RequestFailureClass::Internal &&
        !failure.machine_message.empty()) {
        std::string detail = failure.machine_message;
        for (char& character : detail) {
            if (character == '\n' || character == '\r' || character == '\t') { character = ' '; }
        }
        constexpr std::size_t kMaximumDetail = 240;
        if (detail.size() > kMaximumDetail) {
            detail.resize(kMaximumDetail);
            detail += "...";
        }
        append_clause(out, "detail " + detail);
    }
}

} // namespace

OperationalRecord render_request_start(const RequestLogContext& context) {
    std::ostringstream out;
    out << "req#" << context.id << " started | " << protocol_name(context.protocol) << ' '
        << (context.stream ? "stream" : "non-stream") << " | "
        << product::format_pretty_count(context.message_count)
        << (context.message_count == 1 ? " message" : " messages") << " | max output "
        << product::format_pretty_count(
               static_cast<std::uint64_t>(std::max(context.requested_output_tokens, 0)));
    out << " | thinking ";
    if (context.enable_thinking) {
        out << requested_effort_name(context);
        if (context.thinking_budget) {
            out << ", budget " << product::format_pretty_count(*context.thinking_budget);
        }
    } else {
        out << "off";
    }
    if (context.media_item_count != 0) {
        append_counted_clause(out, "media", context.media_item_count);
        if (context.preparation.seconds > 0.0) {
            out << ", prepared " << product::format_pretty_duration(context.preparation.seconds);
        }
    }
    if (context.tool_count != 0) { append_counted_clause(out, "tools", context.tool_count); }
    if (context.preserve_thinking == true) { append_clause(out, "preserve thinking"); }
    return {.severity = OperationalSeverity::Info, .message = out.str()};
}

OperationalRecord render_request_rejected(const RequestRejectionLogContext& context) {
    const RequestFailure failure =
        make_request_failure(RequestFailurePhase::Prepare, context.error);
    std::ostringstream out;
    const char* status = "failed";
    if (failure.classification == RequestFailureClass::ClientDisconnected) {
        status = "cancelled";
    } else if (failure.classification == RequestFailureClass::ClientInput ||
               failure.classification == RequestFailureClass::Overload) {
        status = "rejected";
    }
    out << "req#" << context.id << ' ' << status << " during prepare | "
        << protocol_name(context.protocol) << ' ' << (context.stream ? "stream" : "non-stream");
    append_failure_fields(out, failure);
    append_counted_clause(out, "messages", context.message_count);
    if (context.media_item_count != 0) {
        append_counted_clause(out, "media", context.media_item_count);
    }
    if (context.tool_count != 0) { append_counted_clause(out, "tools", context.tool_count); }
    return {.severity = failure_severity(failure.classification), .message = out.str()};
}

OperationalRecord render_request_done(const RequestLogContext& context,
                                      const GenerationOutcome& outcome) {
    const GenerationMetrics& metrics     = outcome.metrics;
    const double computed_prefill_tokens = static_cast<double>(metrics.computed_prefill_tokens);
    const double decode_tokens =
        outcome.completion_tokens > 0 ? static_cast<double>(outcome.completion_tokens - 1) : 0.0;
    std::ostringstream out;
    out << "req#" << context.id << " done | " << protocol_name(context.protocol) << " | ";
    if (outcome.tool_calls.empty()) {
        out << finish_reason_name(outcome.finish_reason);
    } else {
        out << "tool calls " << product::format_pretty_count(outcome.tool_calls.size());
    }
    out << " | prompt "
        << product::format_pretty_count(
               static_cast<std::uint64_t>(std::max(outcome.prompt_tokens, 0)))
        << " | output "
        << product::format_pretty_count(
               static_cast<std::uint64_t>(std::max(outcome.completion_tokens, 0)));
    const double cache_ratio = outcome.prompt_tokens > 0
                                   ? static_cast<double>(metrics.prefix_cache_hit_tokens) /
                                         static_cast<double>(outcome.prompt_tokens)
                                   : 0.0;
    out << " | cache " << product::format_pretty_count(metrics.prefix_cache_hit_tokens) << " ("
        << product::format_pretty_percent(cache_ratio);
    if (metrics.prefix_reuse_path != ninfer::PrefixReusePath::Root) {
        out << ", " << prefix_reuse_path_name(metrics.prefix_reuse_path);
    }
    out << ") | TTFT " << product::format_pretty_duration(metrics.ttft_seconds) << " | total "
        << product::format_pretty_duration(metrics.total_seconds);
    if (metrics.engine_timing.queue_wait_seconds >= 0.01) {
        out << " | queue "
            << product::format_pretty_duration(metrics.engine_timing.queue_wait_seconds);
    }
    if (metrics.prefill_seconds > 0.0) {
        out << " | prefill "
            << product::format_pretty_rate(computed_prefill_tokens / metrics.prefill_seconds,
                                           "tok");
    }
    if (metrics.decode_seconds > 0.0) {
        out << " | decode "
            << product::format_pretty_rate(decode_tokens / metrics.decode_seconds, "tok");
    }
    if (metrics.speculative_draft_tokens != 0) {
        const double acceptance = static_cast<double>(metrics.speculative_accepted_tokens) /
                                  static_cast<double>(metrics.speculative_draft_tokens);
        out << " | " << product::speculative_backend_name(metrics.speculative_backend)
            << " accepted " << product::format_pretty_count(metrics.speculative_accepted_tokens)
            << '/' << product::format_pretty_count(metrics.speculative_draft_tokens) << " ("
            << product::format_pretty_percent(acceptance) << ')';
    }
    if (outcome.thinking.configured_budget) {
        out << " | thinking "
            << product::format_pretty_count(outcome.thinking.model_thinking_tokens) << '/'
            << product::format_pretty_count(*outcome.thinking.configured_budget);
        if (outcome.thinking.injected_tokens != 0) {
            out << ", control " << product::format_pretty_count(outcome.thinking.injected_tokens);
        }
    }
    return {.severity = OperationalSeverity::Info, .message = out.str()};
}

std::optional<OperationalRecord> render_tool_call_fallback(const RequestLogContext& context,
                                                           const GenerationOutcome& outcome) {
    const ninfer::ToolCallParseFallbackReason reason = outcome.tool_call_parse.fallback_reason;
    if (!outcome.tool_call_parse.marker_seen ||
        reason == ninfer::ToolCallParseFallbackReason::None) {
        return std::nullopt;
    }
    return OperationalRecord{
        .severity = OperationalSeverity::Warning,
        .message  = "req#" + std::to_string(context.id) + " tool markup returned as text | " +
                   pretty_code(ninfer::tool_call_parse_fallback_reason_name(reason)),
    };
}

OperationalRecord render_request_failure(const RequestLogContext& context,
                                         const RequestFailure& failure) {
    std::ostringstream out;
    out << "req#" << context.id
        << (failure.classification == RequestFailureClass::ClientDisconnected ? " cancelled during "
                                                                              : " failed during ")
        << phase_name(failure.phase) << " | " << protocol_name(context.protocol);
    append_failure_fields(out, failure);
    return {.severity = failure_severity(failure.classification), .message = out.str()};
}

OperationalRecord render_response_failure(std::uint64_t request_id, const RequestFailure& failure) {
    std::ostringstream out;
    out << "req#" << request_id << " response failed during " << phase_name(failure.phase);
    append_failure_fields(out, failure);
    return {.severity = failure_severity(failure.classification), .message = out.str()};
}

OperationalRecord render_throughput(const ThroughputReport& report) {
    const double prefill_rate =
        report.interval_seconds > 0.0
            ? static_cast<double>(report.computed_prefill_tokens) / report.interval_seconds
            : 0.0;
    const double decode_rate =
        report.interval_seconds > 0.0
            ? static_cast<double>(report.committed_decode_tokens) / report.interval_seconds
            : 0.0;
    std::ostringstream out;
    out << "throughput | " << product::format_pretty_duration(report.interval_seconds);
    if (report.computed_prefill_tokens != 0) {
        out << " | prefill " << product::format_pretty_rate(prefill_rate, "tok") << " ("
            << product::format_pretty_count(report.computed_prefill_tokens) << " tok)";
    }
    if (report.committed_decode_tokens != 0) {
        out << " | decode " << product::format_pretty_rate(decode_rate, "tok") << " ("
            << product::format_pretty_count(report.committed_decode_tokens) << " tok)";
    }
    out << " | running " << report.current.running_requests;
    if (report.current.prefilling_requests != 0 || report.current.decode_ready_requests != 0) {
        out << " (";
        bool separator = false;
        if (report.current.prefilling_requests != 0) {
            out << "prefill " << report.current.prefilling_requests;
            separator = true;
        }
        if (report.current.decode_ready_requests != 0) {
            if (separator) { out << ", "; }
            out << "decode-ready " << report.current.decode_ready_requests;
        }
        out << ')';
    }
    if (report.current.waiting_requests != 0) {
        out << " | waiting " << report.current.waiting_requests;
    }
    if (report.current.paused_requests != 0) {
        out << " | paused " << report.current.paused_requests;
    }
    if (report.current.replaying_requests != 0) {
        out << " | replaying " << report.current.replaying_requests;
    }
    // appendix GT: recomputed tokens are the price of degrading a recovery to Replay, and this line
    // used to show only the LIVE gauge above - the cumulative cost was reachable only through
    // /metrics or the JSON request log. appendix GQ priced one 46,290-token recompute at ~8.4 s of
    // wall, so the number belongs where the other interval statistics are.
    if (report.current.replayed_tokens > report.previous.replayed_tokens) {
        out << " | recomputed "
            << product::format_pretty_count(report.current.replayed_tokens -
                                            report.previous.replayed_tokens)
            << " tok";
    }
    // appendix HM (plan item 4): the idle trigger is an independent reason a lane leaves the device,
    // so it gets its own readout. It is invisible in `paused` (a live gauge) and in the `kv tiers`
    // `spilled_pages` (appendix HH.2: that counts the demote path and stays 0 while residency is
    // kept), which is why criteria #2/#4/#5 had no number to be judged by. Cumulative and printed
    // only when it moves, so a default run (counter 0) logs exactly as before. `idle-refused` counts
    // LANES refused, not rounds - a refusal concludes the idle period just as a reclaim does.
    if (report.current.idle_reclaims > report.previous.idle_reclaims) {
        out << " | idle-reclaims " << report.current.idle_reclaims;
    }
    if (report.current.idle_reclaims_refused > report.previous.idle_reclaims_refused) {
        out << " | idle-refused " << report.current.idle_reclaims_refused;
    }
    if (report.current.materializing_requests != 0) {
        out << " | materializing " << report.current.materializing_requests;
    }
    if (report.current.capture_pending_requests != 0) {
        out << " | capture-pending " << report.current.capture_pending_requests;
    }
    if (report.current.terminal_pending_requests != 0) {
        out << " | terminal-pending " << report.current.terminal_pending_requests;
    }
    if (report.decode_rounds != 0) {
        out << " | batch " << std::fixed << std::setprecision(2)
            << static_cast<double>(report.decode_row_rounds) /
                   static_cast<double>(report.decode_rounds);
    }
    const double host_seconds = static_cast<double>(host_active_ns(report)) * 1.0e-9;
    const double host_ratio =
        report.interval_seconds > 0.0 ? host_seconds / report.interval_seconds : 0.0;
    out << " | host " << product::format_pretty_percent(host_ratio) << " ("
        << product::format_pretty_duration(host_seconds) << ')';
    return {.severity = OperationalSeverity::Info, .message = out.str()};
}

OperationalLog::OperationalLog(std::shared_ptr<spdlog::logger> logger)
    : logger_(std::move(logger)) {}

void OperationalLog::write(OperationalRecord record) const {
    switch (record.severity) {
    case OperationalSeverity::Info:
        logger_->info("{}", record.message);
        return;
    case OperationalSeverity::Warning:
        logger_->warn("{}", record.message);
        return;
    case OperationalSeverity::Error:
        logger_->error("{}", record.message);
        return;
    }
}

void OperationalLog::request_start(const RequestLogContext& context) const {
    write(render_request_start(context));
}

void OperationalLog::request_rejected(const RequestRejectionLogContext& context) const {
    write(render_request_rejected(context));
}

void OperationalLog::request_done(const RequestLogContext& context,
                                  const GenerationOutcome& outcome) const {
    write(render_request_done(context, outcome));
    if (std::optional<OperationalRecord> fallback = render_tool_call_fallback(context, outcome)) {
        write(std::move(*fallback));
    }
}

void OperationalLog::request_failure(const RequestLogContext& context,
                                     const RequestFailure& failure) const {
    write(render_request_failure(context, failure));
}

void OperationalLog::response_failure(std::uint64_t request_id,
                                      const RequestFailure& failure) const {
    write(render_response_failure(request_id, failure));
}

void OperationalLog::throughput(const ThroughputReport& report) const {
    write(render_throughput(report));
}

void OperationalLog::http_failure(std::string_view endpoint, const RequestFailure& failure,
                                  std::string_view request_id) const {
    std::ostringstream out;
    out << "HTTP request failed during " << phase_name(failure.phase) << " | "
        << protocol_name(endpoint);
    if (!request_id.empty()) { out << " | request " << product::format_pretty_text(request_id); }
    append_failure_fields(out, failure);
    write({.severity = failure_severity(failure.classification), .message = out.str()});
}

void OperationalLog::engine_capacity(const GenerationService& service) const {
    const ninfer::MemorySummary memory            = service.memory_summary();
    const ninfer::EngineOptions& engine           = service.engine_options();
    const ninfer::ContextCacheOptions& cache      = engine.context_cache;
    const ninfer::ContextCostSummary context_cost = service.load_summary().context_cost;

    logger_->info("capacity | KV {} tokens, {}, {} | pages {}/{} | runtime {} | free {}",
                  product::format_pretty_count(memory.kv_capacity), kv_cache_name(memory.kv_cache),
                  kv_capacity_mode_name(memory.kv_capacity_mode),
                  product::format_pretty_count(memory.kv_capacity_page_groups),
                  product::format_pretty_count(memory.kv_capacity_max_page_groups),
                  product::format_pretty_bytes(memory.runtime_reservation_bytes),
                  product::format_pretty_bytes(memory.available_after_startup_bytes));

    logger_->info("context | history {} | {} active + {} extra device states | host {}",
                  cache.enabled ? "on" : "off", engine.max_concurrency, *cache.device_state_slots,
                  product::format_pretty_bytes(*cache.host_capacity_bytes));
    // appendix GO: say what the arena can actually hold at once. Silence here is what let an
    // "8-way" run measure one lane at a time for two days (appendices GM and GL.6).
    const std::uint32_t page_tokens =
        memory.kv_capacity_page_groups != 0
            ? memory.kv_capacity / memory.kv_capacity_page_groups
            : 0;
    const std::uint32_t per_lane_pages =
        (page_tokens != 0 && memory.max_context != 0)
            ? (memory.max_context + page_tokens - 1U) / page_tokens
            : 0;
    if (per_lane_pages != 0) {
        const std::uint32_t lanes = memory.kv_capacity_page_groups / per_lane_pages;
        if (lanes < engine.max_concurrency) {
            // The advice has to fit the mode: with automatic sizing the arena is already as large
            // as the free memory allows, so the lever is a SMALLER headroom, not `auto`.
            const bool automatic = memory.kv_capacity_mode == KvCapacityMode::Automatic;
            const char* const mode_advice =
                automatic ? "The arena is already memory-sized: free device memory with a smaller "
                            "--kv-capacity-headroom-mib, or enable the eviction contract so "
                            "residency can be smaller than the context"
                          : "Raise it with --kv-capacity auto (tune with "
                            "--kv-capacity-headroom-mib), or enable the eviction contract so "
                            "residency can be smaller than the context";
            // appendix GY: spell the trade-off out instead of assuming the memory can be found. The
            // degradation bands are MEASURED on the gate corpus (appendices GV/GW): reaching 77%
            // retention already reproduces the untouched perplexity bit for bit, 70% costs 0.15%,
            // 64% 0.92%, 58% 1.78%, 52% 2.02%, 27% 8.6%. The user's requirement is <2%, i.e. >=58%.
            const std::uint32_t retained_pages =
                memory.kv_capacity_page_groups / engine.max_concurrency;
            const std::uint32_t retained_percent = 100U * retained_pages / per_lane_pages;
            const char* const measured =
                retained_percent >= 77   ? "no measurable loss (bit-identical perplexity)"
                : retained_percent >= 70 ? "+0.15%"
                : retained_percent >= 64 ? "+0.92%"
                : retained_percent >= 58 ? "+1.78%"
                : retained_percent >= 52 ? "+2.02%"
                : retained_percent >= 27 ? "+8.6%"
                                         : "more than +15%";
            std::string retention_note =
                " Running all " + std::to_string(engine.max_concurrency) +
                " together would need each lane to retain " +
                std::to_string(retained_percent) + "% of its context, measured at " + measured;
            if (retained_percent >= 58) {
                retention_note += " - inside the <2% band.";
            } else {
                const std::uint64_t need_pages =
                    static_cast<std::uint64_t>(58U * per_lane_pages + 99U) / 100U *
                    engine.max_concurrency;
                const std::uint64_t deficit = need_pages > memory.kv_capacity_page_groups
                                                  ? need_pages - memory.kv_capacity_page_groups
                                                  : 0;
                retention_note += " - outside the <2% band; reaching 58% needs " +
                                  product::format_pretty_bytes(
                                      deficit * memory.kv_capacity_increment_bytes) +
                                  " more arena, i.e. that much more free device memory " +
                                  "(other GPU applications) or a lower --max-concurrency.";
            }
            logger_->warn(
                "concurrency | the arena holds {} full-length {}-token lane(s) but "
                "--max-concurrency is {}: further requests are queued, not run together.{} {}",
                lanes, memory.max_context, engine.max_concurrency, retention_note, mode_advice);
        }
    }
    // appendix GR: a Host arena below one full preemption's spill turns the MANDATORY recovery
    // into a recompute. Measured (appendix GQ): 1 GiB of Host against 8 x 0.66 GiB of spill cost
    // +22% wall time with 202,752 tokens recomputed, 512 MiB cost +33%, while fetching the same
    // bytes back costs ~28 ms - a factor of ~300. The default arena (8 GiB plus eight
    // StateImages, startup.cpp:907-914) is above the line, so this only fires when one was set
    // explicitly, which is the mistake it exists to catch.
    if (per_lane_pages != 0 && memory.kv_capacity_increment_bytes != 0 &&
        cache.host_capacity_bytes) {
        const std::size_t per_lane_bytes =
            static_cast<std::size_t>(per_lane_pages) * memory.kv_capacity_increment_bytes;
        const std::size_t needed = per_lane_bytes * engine.max_concurrency;
        if (*cache.host_capacity_bytes < needed) {
            logger_->warn(
                "preemption | the Host context arena holds {} but one full preemption of every "
                "lane needs {} ({} lanes x {}): a pause that cannot keep its snapshot is "
                "recovered by RECOMPUTING the context, measured at +22..33% wall time. Raise "
                "--host-context-mib",
                product::format_pretty_bytes(*cache.host_capacity_bytes),
                product::format_pretty_bytes(needed), engine.max_concurrency,
                product::format_pretty_bytes(per_lane_bytes));
        }
    }
    // appendix GO: the arena was sized from the free memory measured at planning time; compare
    // that plan with what is left now. A shortfall means the device had less free memory a few
    // seconds later - on this machine ~2.2 GiB is held by graphics applications that WDDM does
    // not attribute per process, and their footprint moves - so the arena may have eaten the
    // slack the caller asked for (appendix GO.3).
    const std::size_t slack_shortfall =
        memory.planned_slack_bytes > memory.available_after_startup_bytes
            ? memory.planned_slack_bytes - memory.available_after_startup_bytes
            : 0;
    if (slack_shortfall > (256U * 1024U * 1024U)) {
        logger_->warn(
            "memory | {} free after startup against {} planned: the device lost {} of free "
            "memory between planning and startup. Use a larger --kv-capacity-headroom-mib",
            product::format_pretty_bytes(memory.available_after_startup_bytes),
            product::format_pretty_bytes(memory.planned_slack_bytes),
            product::format_pretty_bytes(slack_shortfall));
    }

    if (service.options().enable_vision) {
        const ninfer::MediaCacheSummary media = service.media_cache_summary();
        logger_->info("media | {} preprocess workers | cache {} | live {}",
                      media.preprocess_threads, product::format_pretty_bytes(media.capacity_bytes),
                      product::format_pretty_bytes(media.live_capacity_bytes));
    }

    logger_->debug("memory ledger | after weights {} | after startup {} | headroom {} | slack {} | "
                   "CUDA graphs {}",
                   product::format_pretty_bytes(memory.available_after_weights_bytes),
                   product::format_pretty_bytes(memory.available_after_startup_bytes),
                   product::format_pretty_bytes(memory.kv_capacity_headroom_bytes),
                   product::format_pretty_bytes(memory.planned_slack_bytes),
                   product::format_pretty_bytes(memory.cuda_graph_allowance_bytes));
    logger_->debug("context cost | transfer {} | prefill {} | hardware {} | prefill signature {}",
                   ninfer::context_cost_preset_source_name(context_cost.transfer_source),
                   ninfer::context_cost_preset_source_name(context_cost.prefill_source),
                   product::format_pretty_text(context_cost.hardware_class),
                   product::format_pretty_text(context_cost.prefill_signature));
}

void OperationalLog::warmup_started() const { logger_->debug("warming up"); }

void OperationalLog::warmup_complete(double seconds) const {
    if (seconds >= 0.25) {
        logger_->info("warmup complete | {}", product::format_pretty_duration(seconds));
    } else {
        logger_->debug("warmup complete | {}", product::format_pretty_duration(seconds));
    }
}

void OperationalLog::warmup_failure(double seconds, std::string_view detail) const {
    logger_->critical("warmup failed | {} | {}", product::format_pretty_duration(seconds),
                      product::format_pretty_text(detail));
}

void OperationalLog::bind_failure(std::string_view host, int port) const {
    logger_->error("cannot bind {}:{}", product::format_pretty_text(host), port);
}

void OperationalLog::listen_failure(std::string_view host, int port) const {
    logger_->error("server listen failed | {}:{}", product::format_pretty_text(host), port);
}

void OperationalLog::server_ready(std::string_view host, int port, std::string_view model_id,
                                  bool auth_enabled) const {
    logger_->info("listening on http://{}:{} | model {} | auth {}",
                  product::format_pretty_text(host), port, product::format_pretty_text(model_id),
                  auth_enabled ? "bearer" : "disabled");
}

void OperationalLog::server_stopped() const { logger_->info("server stopped"); }

void OperationalLog::server_failure(bool serving, std::string_view detail) const {
    logger_->critical("server failed during {} | {}", serving ? "serving" : "startup",
                      product::format_pretty_text(detail));
}

} // namespace ninfer::serve
