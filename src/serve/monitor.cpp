#include "serve/http_server.h"

#include "product/config/options.h"
#include "serve/control.h"
#include "serve/http_transport.h"
#include "serve/request_events.h"

#include <chrono>
#include <string_view>

namespace ninfer::serve {
namespace {
using Json = nlohmann::json;

std::uint64_t steady_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

Json identity(const ServiceObservations& observations) {
    return {{"server_instance_id", observations.instance_id()},
            {"start_time_seconds", observations.start_seconds()},
            {"timestamp_unix_ms", std::chrono::duration_cast<std::chrono::milliseconds>(
                                      std::chrono::system_clock::now().time_since_epoch())
                                      .count()}};
}

const char* kv_cache_name(KvCacheStorage storage) noexcept {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return "bfloat16";
    case KvCacheStorage::Int8Group64:
        return "int8-group64";
    case KvCacheStorage::Fp8E4M3Row256:
        return "fp8-e4m3-row256";
    case KvCacheStorage::Nvfp4Group16:
        return "nvfp4-group16";
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return "fp8-key-nvfp4-value";
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
        return "rotated-int8-key-int4-value-group64";
    }
    return "unknown";
}

Json host_work_json(const RuntimeHostWorkStats& work) {
    return Json{{"engine_boundary_ns", work.engine_boundary_ns},
                {"program_submit_ns", work.program_submit_ns},
                {"program_post_ns", work.program_post_ns},
                {"engine_commit_output_ns", work.engine_commit_output_ns},
                {"engine_maintenance_ns", work.engine_maintenance_ns},
                {"device_wait_ns", work.device_wait_ns},
                {"constraint_draft_wait_ns", work.constraint_draft_wait_ns},
                {"decode_host_ns", work.decode_host_ns},
                {"decode_device_wait_ns", work.decode_device_wait_ns},
                {"prefill_host_ns", work.prefill_host_ns},
                {"prefill_device_wait_ns", work.prefill_device_wait_ns},
                {"control_host_ns", work.control_host_ns},
                {"control_device_wait_ns", work.control_device_wait_ns},
                {"prefill_units", work.prefill_units},
                {"control_units", work.control_units},
                {"stats_publication_ns", work.stats_publication_ns},
                {"stats_publication_invocations", work.stats_publication_invocations}};
}

Json runtime_json(const RuntimeStats& stats, const ServiceSnapshot& service) {
    // Every number below is a value this Engine actually publishes. Host phases stay separate from
    // device waits and from Context transfer time per docs/performance/methodology.md; nothing is
    // summed for the consumer. A quantity this port does not measure is absent, not zero.
    return Json{{"requests", Json{{"submitted", service.counters.submitted},
                                  {"completed", service.counters.completed},
                                  {"cancelled", service.counters.cancelled},
                                  {"failed", service.counters.failed},
                                  {"rejected", service.counters.rejected}}},
                {"host_work", host_work_json(stats.host_work)},
                {"prompt_tokens", stats.prompt_tokens},
                {"generated_tokens", stats.generated_tokens},
                {"speculative_rounds", stats.speculative_rounds},
                {"speculative_draft_tokens", stats.speculative_draft_tokens},
                {"speculative_accepted_tokens", stats.speculative_accepted_tokens},
                {"speculative_fallback_steps", stats.speculative_fallback_steps},
                {"speculative_attempted_per_position", Json::array()},
                {"speculative_accepted_per_position", Json::array()},
                {"computed_prefill_tokens", stats.computed_prefill_tokens},
                {"committed_decode_tokens", stats.committed_decode_tokens},
                {"decode_rounds", stats.decode_rounds},
                {"decode_row_rounds", stats.decode_row_rounds},
                {"running_requests", stats.running_requests},
                {"prefilling_requests", stats.prefilling_requests},
                {"decode_ready_requests", stats.decode_ready_requests},
                {"waiting_requests", stats.waiting_requests},
                {"paused_requests", stats.paused_requests},
                {"replaying_requests", stats.replaying_requests},
                {"materializing_requests", stats.materializing_requests},
                {"capture_pending_requests", stats.capture_pending_requests},
                {"terminal_pending_requests", stats.terminal_pending_requests},
                {"active_captures_completed", stats.active_captures_completed},
                {"active_captures_aborted", stats.active_captures_aborted},
                {"preemptions", stats.preemptions},
                {"snapshot_restores", stats.snapshot_restores},
                {"replay_restores", stats.replay_restores},
                {"replayed_tokens", stats.replayed_tokens},
                {"idle_reclaims", stats.idle_reclaims},
                {"idle_reclaims_refused", stats.idle_reclaims_refused},
                {"idle_reclaims_deferred", stats.idle_reclaims_deferred},
                {"root_selections", stats.root_selections},
                {"checkpoint_selections", stats.checkpoint_selections},
                {"reused_prompt_tokens", stats.reused_prompt_tokens},
                {"last_selected_frontier_tokens", stats.last_selected_frontier_tokens},
                {"state_moves", stats.state_moves},
                {"state_forks", stats.state_forks},
                {"materialization_state_forks", stats.materialization_state_forks},
                {"state_restores", stats.state_restores},
                {"state_d2h_bytes", stats.state_d2h_bytes},
                {"state_h2d_bytes", stats.state_h2d_bytes},
                {"state_d2d_bytes", stats.state_d2d_bytes},
                {"state_d2h_seconds", stats.state_d2h_seconds},
                {"state_h2d_seconds", stats.state_h2d_seconds},
                {"state_d2d_seconds", stats.state_d2d_seconds},
                {"main_kv_d2h_pages", stats.main_kv_d2h_pages},
                {"main_kv_h2d_pages", stats.main_kv_h2d_pages},
                {"main_kv_d2d_pages", stats.main_kv_d2d_pages},
                {"main_kv_d2h_bytes", stats.main_kv_d2h_bytes},
                {"main_kv_h2d_bytes", stats.main_kv_h2d_bytes},
                {"main_kv_d2d_bytes", stats.main_kv_d2d_bytes},
                {"backend_kv_d2h_bytes", stats.backend_kv_d2h_bytes},
                {"backend_kv_h2d_bytes", stats.backend_kv_h2d_bytes},
                {"backend_kv_d2d_bytes", stats.backend_kv_d2d_bytes},
                {"pressure_spill_pages", stats.pressure_spill_pages},
                {"partial_tail_cow_pages", stats.partial_tail_cow_pages},
                {"device_state_occupied_slots", stats.device_state_occupied_slots},
                {"host_state_occupied_slots", stats.host_state_occupied_slots},
                {"device_main_kv_occupied_pages", stats.device_main_kv_occupied_pages},
                {"device_backend_kv_occupied_pages", stats.device_backend_kv_occupied_pages},
                {"host_kv_occupied_bytes", stats.host_kv_occupied_bytes},
                {"host_context_occupied_bytes", stats.host_context_occupied_bytes},
                {"host_context_reserved_bytes", stats.host_context_reserved_bytes},
                {"host_context_peak_occupied_bytes", stats.host_context_peak_occupied_bytes},
                {"actual_context_transfer_seconds", stats.actual_context_transfer_seconds},
                // Prefix-cache and per-position speculative arrays are not published by this
                // port's Engine statistics; they stay empty objects/arrays instead of fakes.
                {"prefix_cache", Json::object()}};
}

Json arena_json(const ArenaMemorySummary& arena) {
    return Json{{"capacity_bytes", arena.capacity_bytes},
                {"used_bytes", arena.used_bytes},
                {"peak_used_bytes", arena.peak_used_bytes}};
}

Json memory_json(const MemorySummary& memory) {
    return Json{{"device", memory.device},
                {"max_context", memory.max_context},
                {"kv_capacity", memory.kv_capacity},
                {"kv_capacity_page_groups", memory.kv_capacity_page_groups},
                {"kv_capacity_max_page_groups", memory.kv_capacity_max_page_groups},
                {"kv_payload_bytes", memory.kv_payload_bytes},
                {"available_after_weights_bytes", memory.available_after_weights_bytes},
                {"available_after_startup_bytes", memory.available_after_startup_bytes},
                {"free_after_startup_bytes", memory.free_after_startup_bytes},
                {"runtime_reservation_bytes", memory.runtime_reservation_bytes},
                {"kv_capacity_headroom_bytes", memory.kv_capacity_headroom_bytes},
                {"planned_slack_bytes", memory.planned_slack_bytes},
                {"workspace_logical_peak_bytes", memory.workspace_logical_peak_bytes},
                {"cuda_graph_allowance_bytes", memory.cuda_graph_allowance_bytes},
                {"startup_retained_bytes", memory.startup_retained_bytes},
                {"host_context_capacity_bytes", memory.host_context_capacity_bytes},
                {"host_context_occupied_bytes", memory.host_context_occupied_bytes},
                {"host_context_reserved_bytes", memory.host_context_reserved_bytes},
                {"host_state_occupied_slots", memory.host_state_occupied_slots},
                {"host_kv_occupied_bytes", memory.host_kv_occupied_bytes}};
}

Json media_json(const MediaCacheSummary& media) {
    return Json{{"capacity_bytes", media.capacity_bytes},
                {"live_capacity_bytes", media.live_capacity_bytes},
                {"retained_bytes", media.retained_bytes},
                {"live_bytes", media.live_bytes},
                {"entries", media.entries},
                {"inflight", media.inflight},
                {"queued_tasks", media.queued_tasks},
                {"active_tasks", media.active_tasks},
                {"preprocess_threads", media.preprocess_threads},
                {"hits", media.hits},
                {"misses", media.misses},
                {"singleflight_waits", media.singleflight_waits},
                {"evictions", media.evictions},
                {"oversize_bypasses", media.oversize_bypasses}};
}

Json rank_json(const MemorySummary& memory, const RuntimeStats& stats) {
    Json allocations = arena_json(memory.weights);
    allocations["sequence_capacity_bytes"]  = memory.sequence.capacity_bytes;
    allocations["sequence_used_bytes"]      = memory.sequence.used_bytes;
    allocations["sequence_peak_used_bytes"] = memory.sequence.peak_used_bytes;
    Json vision                             = Json(nullptr);
    if (memory.vision_workspace) {
        vision = Json{{"general_capacity_bytes", memory.vision_workspace->general_capacity_bytes},
                      {"encode_peak_bytes", memory.vision_workspace->encode_peak_bytes},
                      {"handoff_capacity_bytes", memory.vision_workspace->handoff_capacity_bytes},
                      {"handoff_active_bytes", memory.vision_workspace->handoff_active_bytes},
                      {"handoff_peak_bytes", memory.vision_workspace->handoff_peak_bytes}};
    }
    return Json{{"rank", 0},
                {"node", 0},
                {"sample_age_ns", 0},
                // Physical device memory and GPU interval counters are not sampled by this port.
                {"physical_memory_age_ns", nullptr},
                {"allocations", allocations},
                {"workspace", arena_json(memory.workspace)},
                {"physical_memory", nullptr},
                {"gpu_intervals", Json::object()},
                {"host", host_work_json(stats.host_work)},
                {"attention_work", Json::object()},
                {"communication_eager", Json::object()},
                {"communication_graph", Json::object()},
                {"vision", vision},
                {"experts", nullptr},
                {"ple", nullptr}};
}

Json resources_json(const MemorySummary& memory) {
    Json budget = memory_json(memory);
    return Json{{"ranks", Json::array({{{"rank", 0}, {"budget", budget}}})},
                {"total_allocations",
                 Json{{"weights_capacity_bytes", memory.weights.capacity_bytes},
                      {"weights_used_bytes", memory.weights.used_bytes},
                      {"sequence_capacity_bytes", memory.sequence.capacity_bytes},
                      {"workspace_capacity_bytes", memory.workspace.capacity_bytes},
                      {"workspace_peak_used_bytes", memory.workspace.peak_used_bytes}}}};
}

Json load_json(const LoadSummary& load) {
    return Json{{"architecture", load.architecture},
                {"model_name", load.model_name},
                {"cuda_sync_mode", load.cuda_sync_mode},
                {"weight_formats", load.weight_formats},
                {"prefill_signature", load.prefill_signature},
                {"load_seconds", load.load_seconds},
                {"upload_seconds", load.upload_seconds},
                {"artifact_bytes_read", load.artifact_bytes_read},
                {"host_to_device_bytes", load.host_to_device_bytes},
                {"peak_staging_bytes", load.peak_staging_bytes},
                {"device_object_count", load.device_object_count},
                {"host_object_count", load.host_object_count},
                {"kv_resolved_tokens", load.kv_resolved_tokens},
                {"kv_main_page_groups", load.kv_main_page_groups},
                {"kv_max_main_page_groups", load.kv_max_main_page_groups},
                {"kv_runtime_reservation_bytes", load.kv_runtime_reservation_bytes},
                {"kv_planned_slack_bytes", load.kv_planned_slack_bytes},
                {"kv_available_after_weights_bytes", load.kv_available_after_weights_bytes},
                {"kv_available_after_startup_bytes", load.kv_available_after_startup_bytes},
                {"kv_bytes_per_additional_page_group",
                 load.kv_bytes_per_additional_page_group}};
}
} // namespace

void HttpServer::register_monitor_routes() {
    server_.Get("/monitor/info", [this](const httplib::Request&, httplib::Response& res) {
        auto out              = identity(observations_);
        auto options          = options_;
        options.launch.engine = service_->engine_options();
        // configuration_view owns normalized presentation and credential redaction.
        for (const auto& item : product::configuration_view(options).items) {
            out["configuration"][item.key] =
                std::visit([](const auto& value) { return Json(value); }, item.value);
        }
        const auto& engine = service_->engine_options();
        const auto load     = service_->load_summary();
        out["model"]        = {{"id", public_model_id_},
                               {"architecture", load.architecture},
                               {"name", load.model_name},
                               {"max_context", engine.max_context},
                               {"max_concurrency", engine.max_concurrency},
                               {"prefill_chunk", engine.prefill_chunk},
                               {"kv_cache", kv_cache_name(engine.kv_cache)},
                               {"context_token_capacity_estimate", load.kv_resolved_tokens}};
        out["ranks"] = Json::array({{{"rank", 0},
                                     {"node", 0},
                                     {"device", engine.device},
                                     {"hardware", "cuda:" + std::to_string(engine.device)},
                                     {"shared_physical_memory", false},
                                     {"tp", {{"rank", 0}, {"size", 1}}},
                                     {"ep", {{"rank", 0}, {"size", 1}}},
                                     {"dp", {{"rank", 0}, {"size", 1}}},
                                     {"pp", {{"rank", 0}, {"size", 1}}},
                                     {"adp", {{"rank", 0}, {"size", 1}}}}});
        out["placements"] = Json::array();
        out["resources"]  = resources_json(service_->memory_summary());
        out["load"]       = load_json(load);
        // T131: the control plane's in-force state, read-only here. The writes live on
        // /control/mode and /control/freeze; this panel shows the worker's own answer, so the
        // console can label a deployment with the mode the engine actually runs (and whether the
        // freeze policy it holds is armed at all), instead of a UI-only selection.
        out["controller"] = controller_control_json(service_->controller_control_state());
        res.set_header("Cache-Control", "no-store");
        res.set_content(out.dump(), "application/json");
    });

    server_.Get("/monitor/snapshot", [this](const httplib::Request&, httplib::Response& res) {
        const auto stats    = service_->runtime_stats();
        const auto memory   = service_->memory_summary();
        const auto media    = service_->media_cache_summary();
        const auto service  = observations_.snapshot(true);
        auto out            = identity(observations_);
        out["engine"]       = Json{{"sequence", observations_.next_sample_sequence()},
                                   {"sampled_ns", steady_ns()},
                                   {"sample_age_ns", 0},
                                   {"status", service_->is_available() ? "ready" : "unavailable"},
                                   {"runtime", runtime_json(stats, service)},
                                   // Execution-kind latency groups are owned by the model programs in
                                   // the upstream tree; this port publishes request-level timings
                                   // instead, so the groups stay empty rather than duplicated.
                                   {"execution_latency_bounds_ns", Json::array()},
                                   {"executions", Json::object()},
                                   {"decode_by_batch", Json::array()},
                                   {"context",
                                    Json{{"pools", Json::array()},
                                         {"resources", Json::array()},
                                         {"metadata", Json::object()},
                                         // T166 (upstream e5e18b97): the resolved Main KV
                                         // capacity is the context token capacity estimate.
                                         {"context_token_capacity_estimate",
                                          memory.kv_capacity}}},
                                   {"placements", Json::array()},
                                   {"ranks", Json::array({rank_json(memory, stats)})},
                                   {"media", media_json(media)},
                                   {"requests", Json::array()}};
        out["service"] = service_observation_json(service);
        res.set_header("Cache-Control", "no-store");
        res.set_content(out.dump(), "application/json");
    });

    server_.Get("/monitor/events", [this](const httplib::Request&, httplib::Response& res) {
        auto subscription = observations_.events.subscribe();
        if (!subscription) {
            res.status = 503;
            res.set_content(Json{{"error", "monitor subscriber capacity unavailable"}}.dump(),
                            "application/json");
            return;
        }
        const auto hello = "event: connected\ndata: " + identity(observations_).dump() + "\n\n";
        prepare_sse_response(res);
        res.set_chunked_content_provider(
            "text/event-stream",
            [subscription, hello](std::size_t offset, httplib::DataSink& sink) {
                std::atomic<bool> cancelled{false};
                SseTransport transport(sink, cancelled);
                try {
                    if (!offset) { transport.write(hello); }
                    while (!subscription->closed()) {
                        if (auto event = subscription->next(std::chrono::milliseconds(500))) {
                            transport.write(*event);
                        }
                        if (transport.poll()) { break; }
                    }
                } catch (const ClientDisconnected&) {}
                subscription->close();
                return false;
            },
            [subscription](bool) { subscription->close(); });
    });
}
} // namespace ninfer::serve
