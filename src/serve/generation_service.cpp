#include <charconv>
#include "serve/generation_service.h"

#include "product/media_acquire/acquire.h"
#include "serve/translate.h"

#if NINFER_HAVE_LMCACHE
#include "core/lmcache_bridge.h"
#endif

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::serve {

#if NINFER_HAVE_LMCACHE
namespace {

// The engine's whole view of the cache: one lookup, no transport types. Kept separate from Bridge so
// the engine side never sees a key, a connection or a chunk.
class LmcacheBackend final : public ninfer::CheckpointCacheBackend {
public:
    // Second step of a two-step handoff. The backend has to exist before the engine because the
    // engine takes it in its options; it cannot be configured until the pools exist, because it
    // needs page_bytes and the image size. fetch() stays silent until this succeeds, so nothing is
    // ever registered against a zero-size layout.
    bool configure(const ninfer::CacheGeometry& geometry, const std::string& model_name,
                   int connections) {
        if (geometry.page_bytes == 0 || geometry.page_tokens <= 0 ||
            geometry.state_image_bytes == 0) {
            error_ = "geometry is not ready";
            return false;
        }
        ninfer::core::lmcache::BridgeConfig config;
        config.model_name      = model_name;
        config.page_tokens     = geometry.page_tokens;
        config.pages_per_chunk = 2;
        config.page_bytes      = geometry.page_bytes;
        // Opaque bytes, one plane (v6 1p): a quantized format has four physical planes per layer and
        // the server can express two, so the shape states the byte count and nothing else.
        config.kv =
            ninfer::core::lmcache::GroupSpec::exact("uint8", 2 * geometry.page_bytes);
        config.state =
            ninfer::core::lmcache::GroupSpec::exact("uint8", geometry.state_image_bytes);
        config.state.window_chunks = 1;
        config.state.recurrent     = true;
        // The key is the model and the tokens; the sizes are what change when the format or the page
        // geometry does, so they are what isolates one from the other.
        // ':' and not '@': LMCache forbids '@' in a salt because it separates the fields of its own
        // cache key, and it answers a violating request by raising -- no reply frame at all, so the
        // client's only symptom is a receive timeout. Measured: a two-minute stall.
        config.cache_salt = model_name + ":" + std::to_string(geometry.page_bytes) + ":" +
                            std::to_string(geometry.page_tokens) + ":" +
                            std::to_string(geometry.state_image_bytes);
        page_bytes_ = geometry.page_bytes;
        // A lookup resolves complete chunks only. Below one chunk there is nothing to ask about --
        // and the server answers such a request with no frames at all, so the client sits on its
        // receive timeout. Measured: a 22-token prompt held a request for two minutes.
        chunk_tokens_ =
            static_cast<std::size_t>(geometry.page_tokens) * static_cast<std::size_t>(2);

        ninfer::core::lmcache::ClientOptions client;
        if (const char* endpoint = std::getenv("NINFER_LMCACHE_ENDPOINT");
            endpoint != nullptr && *endpoint != '\0') {
            client.endpoint = endpoint;
        }
        // Restore is best-effort: a healthy round trip is about 1.3 s, so a much longer wait only
        // means the server is not answering -- and there it answers with silence, not an error. The
        // default 120 s would turn one bad payload into a two-minute request.
        client.receive_timeout_ms = 5000;
        // A fresh id per process start: the server does not replace the layout of an id it already
        // knows.
        const std::int64_t instance =
            static_cast<std::int64_t>(std::chrono::system_clock::now().time_since_epoch().count());
        bridge_ = std::make_unique<ninfer::core::lmcache::Bridge>();
        if (!bridge_->open(client, connections, config, instance)) {
            error_ = bridge_->error();
            bridge_.reset();
            return false;
        }
        ready_.store(true);
        return true;
    }

    [[nodiscard]] const std::string& error() const { return error_; }

    [[nodiscard]] std::optional<ninfer::CachedCheckpoint>
    fetch(std::span<const TokenId> prompt_tokens) override {
        if (!ready_.load() || bridge_ == nullptr || prompt_tokens.empty()) { return std::nullopt; }
        // Fewer tokens than one chunk means no complete chunk to look up. The server replies to such
        // a lookup with nothing at all, so asking costs a full receive timeout.
        if (chunk_tokens_ != 0 && prompt_tokens.size() < chunk_tokens_) { return std::nullopt; }
        ninfer::core::lmcache::BridgeKey key;
        key.tokens.assign(prompt_tokens.begin(), prompt_tokens.end());
        // The server keys its token session by request_id and does not reset the rolling hash when
        // the token buffer is replaced, so this has to be unique per call, not per process.
        key.request_id = "svc-" + std::to_string(instance_tag_) + "-" + std::to_string(++sequence_);

        std::vector<std::byte> kv;
        std::vector<std::byte> state;
        if (!bridge_->restore(key, &kv, &state)) { return std::nullopt; }
        if (page_bytes_ == 0 || kv.size() % page_bytes_ != 0) { return std::nullopt; }

        ninfer::CachedCheckpoint out;
        out.page_bytes = page_bytes_;
        out.kv_bytes   = std::move(kv);
        // One element: the image travels whole (v6 1n). The sharded layout exists for transport
        // sizing, not for import.
        out.state_shards.emplace_back(reinterpret_cast<const char*>(state.data()), state.size());
        return out;
    }

private:
    std::unique_ptr<ninfer::core::lmcache::Bridge> bridge_;
    std::atomic<bool> ready_{false};
    std::atomic<std::uint64_t> sequence_{0};
    std::size_t page_bytes_ = 0;
    std::size_t chunk_tokens_ = 0;
    std::int64_t instance_tag_ =
        static_cast<std::int64_t>(std::chrono::system_clock::now().time_since_epoch().count());
    std::string error_;
};

}  // namespace

struct GenerationService::CheckpointCache {
    std::shared_ptr<LmcacheBackend> backend;
};
#else
struct GenerationService::CheckpointCache {};
#endif

GenerationService::~GenerationService() = default;

bool GenerationService::attach_checkpoint_cache() {
#if !NINFER_HAVE_LMCACHE
    return false;
#else
    // The switch was read at construction, where the backend had to exist so the engine could take
    // it in its options. No holder means it was off then.
    if (!checkpoint_cache_ || !checkpoint_cache_->backend) { return false; }

    const ninfer::CacheGeometry geometry = cache_geometry();
    int connections = 4;
    if (const char* value = std::getenv("NINFER_LMCACHE_CONNECTIONS")) {
        const int parsed = std::atoi(value);
        if (parsed > 0) { connections = parsed; }
    }
    if (!checkpoint_cache_->backend->configure(geometry, load_summary().model_name, connections)) {
        std::fprintf(stderr, "[lmcache] could not attach (%s); the cache stays off\n",
                     checkpoint_cache_->backend->error().c_str());
        checkpoint_cache_.reset();
        return false;
    }
    std::fprintf(stderr,
                 "[lmcache] attached | page %zu B, %lld token(s), state image %zu B, %d connection(s)"
                 "\n",
                 geometry.page_bytes, static_cast<long long>(geometry.page_tokens),
                 geometry.state_image_bytes, connections);
    return true;
#endif
}

struct RequestCapacity {
    explicit RequestCapacity(std::size_t limit) : maximum(limit) {}

    std::mutex mutex;
    std::size_t active = 0;
    const std::size_t maximum;
};

struct RequestLifetime {
    RequestLifetime(std::shared_ptr<RequestCapacity> owner,
                    std::chrono::steady_clock::time_point begin,
                    std::chrono::steady_clock::time_point limit)
        : capacity(std::move(owner)), started(begin), deadline(limit) {}

    ~RequestLifetime() {
        std::lock_guard lock(capacity->mutex);
        --capacity->active;
    }

    std::shared_ptr<RequestCapacity> capacity;
    std::chrono::steady_clock::time_point started;
    std::chrono::steady_clock::time_point deadline;
};

ApiError request_error_to_api_error(const ninfer::RequestError& exception,
                                    std::string_view constraint_param,
                                    std::span<const std::string> tool_schema_params) {
    ApiError error;
    error.param   = "messages";
    error.message = exception.what();
    const auto tool_param = [&] {
        const std::string_view pointer = exception.pointer();
        const auto split               = pointer.find('/', 1);
        std::size_t index              = 0;
        if (pointer.starts_with('/') && split != std::string_view::npos) {
            const auto number = pointer.substr(1, split - 1);
            const auto parsed =
                std::from_chars(number.data(), number.data() + number.size(), index);
            if (parsed.ec == std::errc{} && parsed.ptr == number.data() + number.size() &&
                index < tool_schema_params.size() && !tool_schema_params[index].empty() &&
                (pointer.substr(split) == "/parameters" ||
                 pointer.substr(split).starts_with("/parameters/")))
                return tool_schema_params[index] + std::string(pointer.substr(split + 11));
        }
        return "tools" + exception.pointer();
    };
    switch (exception.kind()) {
    case ninfer::RequestErrorKind::InvalidToolConstraint:
        error.status = 400;
        error.code   = "invalid_tool_constraint";
        error.param  = tool_param();
        break;
    case ninfer::RequestErrorKind::InvalidGrammar:
    case ninfer::RequestErrorKind::InvalidChoice:
    case ninfer::RequestErrorKind::InvalidRegex:
    case ninfer::RequestErrorKind::ConstraintDeadEnd:
        error.status = 400;
        error.param  = constraint_param;
        error.code =
            exception.kind() == ninfer::RequestErrorKind::InvalidGrammar  ? "invalid_grammar"
            : exception.kind() == ninfer::RequestErrorKind::InvalidChoice ? "invalid_choice"
            : exception.kind() == ninfer::RequestErrorKind::InvalidRegex  ? "invalid_regex"
                                                                          : "constraint_dead_end";
        break;
    case ninfer::RequestErrorKind::InvalidJsonSchema:
    case ninfer::RequestErrorKind::UnsupportedJsonSchema:
    case ninfer::RequestErrorKind::UnsatisfiableJsonSchema:
        error.status = 400;
        error.param  = exception.source() == RequestErrorSource::Tools
                           ? tool_param()
                           : std::string(constraint_param) + exception.pointer();
        error.code = exception.kind() == ninfer::RequestErrorKind::InvalidJsonSchema
                         ? "invalid_json_schema"
                     : exception.kind() == ninfer::RequestErrorKind::UnsupportedJsonSchema
                         ? "unsupported_json_schema"
                         : "unsatisfiable_json_schema";
        break;
    case ninfer::RequestErrorKind::ContextLengthExceeded:
        error.status = 400;
        error.code   = "context_length_exceeded";
        break;
    case ninfer::RequestErrorKind::ThinkingBudgetCapacityInsufficient:
        error.param.clear();
        error.status = 400;
        error.code   = "thinking_budget_capacity_insufficient";
        break;
    case ninfer::RequestErrorKind::MediaBudgetExceeded:
        error.status = 400;
        error.code   = "media_budget_exceeded";
        break;
    case ninfer::RequestErrorKind::InvalidMedia:
        error.status = 400;
        error.code   = "invalid_media";
        break;
    case ninfer::RequestErrorKind::Overloaded:
        error.param.clear();
        error.status = 429;
        error.type   = "rate_limit_error";
        error.code   = "server_overloaded";
        break;
    case ninfer::RequestErrorKind::QueueTimeout:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "request_queue_timeout";
        break;
    case ninfer::RequestErrorKind::Cancelled:
        error.param.clear();
        error.status = 499;
        error.type   = "request_cancelled";
        error.code   = "client_disconnected";
        break;
    case ninfer::RequestErrorKind::Unavailable:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "service_unavailable";
        break;
    }
    return error;
}

namespace {

using Clock = std::chrono::steady_clock;

[[noreturn]] void throw_preparation_cancelled();

[[noreturn]] void throw_media_error(const ninfer::product::media_acquire::Error& exception) {
    ApiError error;
    error.param   = "messages";
    error.message = exception.what();
    switch (exception.kind()) {
    case ninfer::product::media_acquire::ErrorKind::BudgetExceeded:
        error.status = 400;
        error.code   = "media_budget_exceeded";
        break;
    case ninfer::product::media_acquire::ErrorKind::RemoteUnavailable:
        error.status = 502;
        error.type   = "server_error";
        error.code   = "media_fetch_failed";
        break;
    case ninfer::product::media_acquire::ErrorKind::RemoteTimeout:
        error.status = 504;
        error.type   = "server_error";
        error.code   = "media_fetch_timeout";
        break;
    case ninfer::product::media_acquire::ErrorKind::DeadlineExceeded:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "request_queue_timeout";
        break;
    case ninfer::product::media_acquire::ErrorKind::Cancelled:
        throw_preparation_cancelled();
    }
    throw ApiException(std::move(error));
}

[[noreturn]] void throw_invalid_input(const std::exception& exception, const char* code) {
    ApiError error;
    error.status  = 400;
    error.param   = "messages";
    error.code    = code;
    error.message = exception.what();
    throw ApiException(std::move(error));
}

[[noreturn]] void throw_preparation_cancelled() {
    ApiError error;
    error.status  = 499;
    error.type    = "request_cancelled";
    error.code    = "client_disconnected";
    error.message = "client disconnected during media preparation";
    throw ApiException(std::move(error));
}

ninfer::OwnedMedia acquire_media(const ContentPart& part, Clock::time_point deadline,
                                 const std::function<bool()>& is_cancelled,
                                 std::size_t& remaining_bytes) {
    if (remaining_bytes == 0) {
        throw_media_error(ninfer::product::media_acquire::Error(
            ninfer::product::media_acquire::ErrorKind::BudgetExceeded,
            "request media exceeds aggregate byte limit"));
    }
    ninfer::product::media_acquire::Policy policy;
    policy.max_bytes    = std::min(policy.max_bytes, remaining_bytes);
    policy.deadline     = deadline;
    policy.is_cancelled = is_cancelled;
    std::vector<std::uint8_t> source_bytes;
    try {
        source_bytes = ninfer::product::media_acquire::acquire_bytes(part.source, policy);
    } catch (const ninfer::product::media_acquire::Error& exception) {
        throw_media_error(exception);
    } catch (const std::invalid_argument& exception) {
        throw_invalid_input(exception, "invalid_media");
    }

    remaining_bytes -= source_bytes.size();
    ninfer::OwnedMedia media;
    media.kind =
        part.kind == ContentKind::Image ? ninfer::MediaKind::Image : ninfer::MediaKind::Video;
    media.media_type = part.source.media_type;
    switch (part.source.kind) {
    case ninfer::product::media_acquire::SourceKind::Path:
    case ninfer::product::media_acquire::SourceKind::Url:
        media.source_name = part.source.value;
        break;
    case ninfer::product::media_acquire::SourceKind::Data:
        media.source_name = "inline-data";
        break;
    case ninfer::product::media_acquire::SourceKind::Bytes:
        media.source_name = "inline-bytes";
        break;
    }
    media.bytes               = std::move(source_bytes);
    media.image_resize_policy = part.image_resize_policy;
    return media;
}

[[noreturn]] void
throw_request_error(const ninfer::RequestError& exception,
                    std::string_view constraint_param               = "structured_outputs.grammar",
                    std::span<const std::string> tool_schema_params = {}) {
    throw ApiException(request_error_to_api_error(exception, constraint_param, tool_schema_params));
}

void check_preparation_control(Clock::time_point deadline,
                               const std::function<bool()>& is_cancelled) {
    if (is_cancelled && is_cancelled()) { throw_preparation_cancelled(); }
    if (Clock::now() >= deadline) {
        throw_request_error(ninfer::RequestError(RequestErrorKind::QueueTimeout,
                                                 "inference request expired during preparation"));
    }
}

class ServiceOutputSink final : public ninfer::OutputSink {
public:
    explicit ServiceOutputSink(const StreamSink& sink) : sink_(&sink) {}

    void start(ninfer::GenerationStart start) override {
        deliver([&] {
            if (sink_->on_start) { sink_->on_start(start); }
        });
    }

    void progress(ninfer::PromptProgress progress) override {
        deliver([&] {
            if (sink_->on_progress) { sink_->on_progress(progress); }
        });
    }

    void timing(ninfer::GenerationTimingObservation timing) override {
        deliver([&] {
            if (sink_->on_timing) { sink_->on_timing(timing); }
        });
    }

    void publish(ninfer::OutputDelta delta) override {
        if (delta.text.empty()) { return; }
        deliver([&] {
            if (delta.channel == ninfer::OutputChannel::Reasoning) {
                if (sink_->on_reasoning) { sink_->on_reasoning(delta.text); }
            } else {
                if (sink_->on_content) { sink_->on_content(delta.text); }
            }
        });
    }

    [[nodiscard]] bool disconnected() const noexcept { return disconnected_; }

private:
    template <class Callback>
    void deliver(Callback&& callback) {
        if (disconnected_) { return; }
        try {
            callback();
        } catch (const ClientDisconnected&) { disconnected_ = true; }
    }

    const StreamSink* sink_ = nullptr;
    bool disconnected_      = false;
};

} // namespace

GenerationService::GenerationService(ServeOptions options, StartupObserver startup_observer)
    : options_(std::move(options)) {
    ninfer::EngineOptions engine_options;
    engine_options.artifact_path            = options_.artifact_path;
    engine_options.chat_template_path       = options_.chat_template_path;
    engine_options.device                   = options_.device;
    engine_options.max_context              = options_.max_context;
    engine_options.kv_capacity              = options_.kv_capacity;
    engine_options.max_concurrency          = options_.max_concurrency;
    engine_options.max_pending_requests     = options_.max_pending_requests;
    engine_options.pending_timeout_ms       = options_.pending_timeout_ms;
    engine_options.prefill_chunk            = options_.prefill_chunk;
    engine_options.kv_cache                 = options_.kv_cache;
    engine_options.enable_vision            = options_.enable_vision;
    engine_options.use_cuda_graph           = options_.use_cuda_graph;
    engine_options.wddm_evictable_budget     = options_.wddm_evictable_budget;
    engine_options.speculative              = options_.speculative;
    engine_options.context_cache            = options_.context_cache;
    engine_options.evict_contract           = options_.evict_contract;
    engine_options.evict_window_pages       = options_.evict_window_pages;
    engine_options.evict_sink_pages         = options_.evict_sink_pages;
    engine_options.context_cost.preset_path = options_.context_cost_presets;
    engine_options.media_cache_bytes        = options_.media_cache_bytes;
    engine_options.media_live_bytes         = options_.media_live_bytes;
    engine_options.media_preprocess_threads = options_.media_preprocess_threads;
    engine_options.startup_observer         = std::move(startup_observer);
    // appendix IL: the idle demotion (9b) is ON by default through the CLI; the environment still
    // wins when it is set, so the recorded experiments stay reproducible.
    if (options_.idle_reclaim_ms != 0 && std::getenv("NINFER_KV_IDLE_RECLAIM_MS") == nullptr) {
        const std::string text = std::to_string(options_.idle_reclaim_ms);
#ifdef _WIN32
        (void)_putenv_s("NINFER_KV_IDLE_RECLAIM_MS", text.c_str());
#else
        (void)setenv("NINFER_KV_IDLE_RECLAIM_MS", text.c_str(), 1);
#endif
    }
    // Two-step handoff. The engine takes the backend through its options, so it has to exist before
    // the engine is built; but it cannot be configured until the pools exist, because it needs the
    // page size and the image size. Hand over an empty one now, fill it in at attach time.
    if (const char* enabled = std::getenv("NINFER_LMCACHE_RESTORE");
        enabled != nullptr && *enabled != '\0' && std::strcmp(enabled, "0") != 0) {
#if NINFER_HAVE_LMCACHE
        auto backend                   = std::make_shared<LmcacheBackend>();
        checkpoint_cache_              = std::make_unique<CheckpointCache>();
        checkpoint_cache_->backend     = backend;
        engine_options.checkpoint_cache = backend;
#endif
    }
    engine_           = std::make_unique<ninfer::Engine>(std::move(engine_options));
    request_capacity_ = std::make_shared<RequestCapacity>(
        static_cast<std::size_t>(options_.max_concurrency) + options_.max_pending_requests);
}

std::shared_ptr<RequestLifetime>
GenerationService::acquire_request_lifetime(DeadlinePolicy deadline_policy) const {
    const auto started = Clock::now();
    {
        std::lock_guard lock(request_capacity_->mutex);
        if (request_capacity_->active >= request_capacity_->maximum) {
            throw_request_error(ninfer::RequestError(RequestErrorKind::Overloaded,
                                                     "inference request queue is full"));
        }
        ++request_capacity_->active;
    }
    try {
        const Clock::time_point deadline =
            deadline_policy == DeadlinePolicy::UnboundedStartup
                ? Clock::time_point::max()
                : started + std::chrono::milliseconds(options_.pending_timeout_ms);
        return std::make_shared<RequestLifetime>(request_capacity_, started, deadline);
    } catch (...) {
        std::lock_guard lock(request_capacity_->mutex);
        --request_capacity_->active;
        throw;
    }
}

PreparedRequest GenerationService::prepare(const GenerationRequest& request,
                                           GenerationConsumerMode consumer_mode,
                                           ninfer::GenerationObservationOptions observation,
                                           std::function<bool()> is_cancelled,
                                           ContextCacheHints context_cache) const {
    return prepare_impl(request, consumer_mode, std::move(observation), std::move(is_cancelled),
                        std::move(context_cache),
                        options_.allow_prefix_reuse ? CacheParticipation::ReadWrite
                                                    : CacheParticipation::Disabled,
                        DeadlinePolicy::ClientPendingTimeout);
}

PreparedRequest GenerationService::prepare_impl(const GenerationRequest& request,
                                                GenerationConsumerMode consumer_mode,
                                                ninfer::GenerationObservationOptions observation,
                                                std::function<bool()> is_cancelled,
                                                ContextCacheHints context_cache,
                                                CacheParticipation cache_participation,
                                                DeadlinePolicy deadline_policy) const {
    PreparedRequest prepared;
    prepared.constraint_param               = request.constraint_param;
    for (const auto& tool : request.tools) prepared.tool_schema_params.push_back(tool.schema_param);
    const ResolvedPromptSemantics semantics = resolve_prompt_semantics(request, options_);
    ninfer::RequestOptions request_options  = to_request_options(
        request, options_, semantics, cache_participation == CacheParticipation::ReadWrite);
    prepared.thinking_budget     = request_options.execution.thinking.budget;
    prepared.reasoning_effort    = semantics.reasoning_effort;
    prepared.preserve_thinking   = semantics.preserve_thinking;
    const bool request_has_media = request.media_item_count() != 0;
    if (request_has_media && !options_.enable_vision) {
        const std::invalid_argument error("Vision is disabled for this server");
        throw_invalid_input(error, "vision_disabled");
    }
    prepared.lifetime = acquire_request_lifetime(deadline_policy);

    try {
        const auto acquisition_started = Clock::now();
        std::size_t remaining_media_bytes =
            std::min(options_.max_request_bytes, ninfer::kMaximumPromptMediaBytes);
        ninfer::PromptInput input =
            to_prompt_input(request, semantics, [&](const ContentPart& part) {
                return acquire_media(part, prepared.lifetime->deadline, is_cancelled,
                                     remaining_media_bytes);
            });
        std::vector<PromptCacheMarker> protocol_markers = std::move(input.context_cache.markers);
        const bool protocol_allows_engine_automatic =
            input.context_cache.allow_engine_automatic_shared_prefixes;
        input.context_cache = std::move(context_cache);
        input.context_cache.markers.insert(input.context_cache.markers.end(),
                                           std::make_move_iterator(protocol_markers.begin()),
                                           std::make_move_iterator(protocol_markers.end()));
        input.context_cache.allow_engine_automatic_shared_prefixes =
            input.context_cache.allow_engine_automatic_shared_prefixes &&
            protocol_allows_engine_automatic;
        prepared.acquisition_seconds =
            std::chrono::duration<double>(Clock::now() - acquisition_started).count();
        check_preparation_control(prepared.lifetime->deadline, is_cancelled);
        const PreparationControl control{
            .deadline     = prepared.lifetime->deadline,
            .cancellation = CancellationView(is_cancelled),
        };
        ninfer::PreparedPrompt prompt = engine_->prepare(std::move(input), control);
        check_preparation_control(prepared.lifetime->deadline, is_cancelled);
        prepared.enable_thinking = prompt.summary().starts_in_reasoning;
        if (!prepared.enable_thinking) {
            request_options.execution.thinking.budget.reset();
            prepared.thinking_budget.reset();
        }
        prepared.prompt_tokens           = static_cast<int>(prompt.summary().prompt_tokens);
        prepared.preparation             = prompt.preparation_stats();
        prepared.service_prepare_seconds = std::max(
            0.0, std::chrono::duration<double>(Clock::now() - prepared.lifetime->started).count() -
                     prepared.preparation.seconds);
        if (observation.first_token) {
            observation.first_token = [callback = std::move(observation.first_token),
                                       seconds  = prepared.service_prepare_seconds](
                                          ninfer::GenerationFirstTokenObservation first) {
                first.prepare_seconds += seconds;
                callback(first);
            };
        }
        prepared.generation = engine_->submit(std::move(prompt), std::move(request_options),
                                              consumer_mode == GenerationConsumerMode::Streaming
                                                  ? ninfer::OutputConsumerMode::Streaming
                                                  : ninfer::OutputConsumerMode::Aggregate,
                                              std::move(observation), prepared.lifetime->deadline);
        prepared.sampling   = prepared.generation.resolved_sampling();
    } catch (const ApiException&) { throw; } catch (const ninfer::RequestError& exception) {
        throw_request_error(exception, request.constraint_param, prepared.tool_schema_params);
    } catch (const std::invalid_argument& exception) {
        throw_invalid_input(exception, "invalid_prompt");
    }
    return prepared;
}

int GenerationService::count_prompt_tokens(const GenerationRequest& request,
                                           std::function<bool()> is_cancelled) const {
    const bool request_has_media = request.media_item_count() != 0;
    if (request_has_media && !options_.enable_vision) {
        const std::invalid_argument error("Vision is disabled for this server");
        throw_invalid_input(error, "vision_disabled");
    }
    const Clock::time_point deadline =
        Clock::now() + std::chrono::milliseconds(options_.pending_timeout_ms);
    const ResolvedPromptSemantics semantics = resolve_prompt_semantics(request, options_);
    try {
        std::size_t remaining_media_bytes =
            std::min(options_.max_request_bytes, ninfer::kMaximumPromptMediaBytes);
        ninfer::PromptInput input =
            to_prompt_input(request, semantics, [&](const ContentPart& part) {
                return acquire_media(part, deadline, is_cancelled, remaining_media_bytes);
            });
        check_preparation_control(deadline, is_cancelled);
        const PreparationControl control{
            .deadline     = deadline,
            .cancellation = CancellationView(is_cancelled),
        };
        const int prompt_tokens =
            static_cast<int>(engine_->count_tokens(std::move(input), control));
        check_preparation_control(deadline, is_cancelled);
        return prompt_tokens;
    } catch (const ApiException&) { throw; } catch (const ninfer::RequestError& exception) {
        throw_request_error(exception);
    } catch (const std::invalid_argument& exception) {
        throw_invalid_input(exception, "invalid_prompt");
    }
}

GenerationOutcome GenerationService::run(PreparedRequest& prepared, const StreamSink* sink,
                                         std::function<bool()> is_cancelled) {
    std::unique_ptr<ServiceOutputSink> output_sink;
    if (sink != nullptr) { output_sink = std::make_unique<ServiceOutputSink>(*sink); }
    ninfer::OutputSink* public_sink = output_sink.get();
    ninfer::CancellationView cancellation;
    if (is_cancelled || sink != nullptr) {
        cancellation = ninfer::CancellationView(
            [external = std::move(is_cancelled), sink, observed = output_sink.get()]() {
                return (observed && observed->disconnected()) || (external && external()) ||
                       (sink != nullptr && sink->is_cancelled && sink->is_cancelled());
            });
    }

    ninfer::GenerationResult result;
    try {
        result = prepared.generation.wait(public_sink, cancellation);
    } catch (const ninfer::RequestError& exception) {
        throw_request_error(exception, prepared.constraint_param, prepared.tool_schema_params);
    }
    GenerationOutcome outcome;
    outcome.text                = std::move(result.content);
    outcome.reasoning           = std::move(result.reasoning);
    outcome.prompt_tokens       = static_cast<int>(result.prompt.prompt_tokens);
    outcome.completion_tokens   = static_cast<int>(result.generated_token_ids.size());
    outcome.generated_token_ids = std::move(result.generated_token_ids);
    outcome.reasoning_tokens    = static_cast<int>(result.reasoning_tokens);
    outcome.thinking            = result.thinking;
    outcome.constraint          = result.constraint;
    outcome.finish_reason       = result.finish_reason;
    outcome.matched_stop_string = std::move(result.matched_stop_string);

    outcome.metrics.prepare_seconds =
        prepared.service_prepare_seconds + result.timings.prepare_seconds;
    outcome.metrics.ttft_seconds =
        outcome.metrics.prepare_seconds +
        std::max(0.0, result.timings.first_token_seconds - result.timings.prepare_seconds);
    outcome.metrics.vision_seconds          = result.timings.vision_seconds;
    outcome.metrics.prefill_seconds         = result.timings.prefill_seconds;
    outcome.metrics.decode_seconds          = result.timings.decode_seconds;
    outcome.metrics.prompt_wall_seconds     = result.timings.prompt_wall_seconds;
    outcome.metrics.generation_wall_seconds = result.timings.generation_wall_seconds;
    outcome.metrics.total_seconds = prepared.service_prepare_seconds + result.timings.total_seconds;
    outcome.metrics.engine_timing = result.engine_timing;
    outcome.metrics.first_output_timing         = std::move(result.first_output_timing);
    outcome.metrics.scheduling                  = result.scheduling;
    outcome.metrics.admission                   = result.admission;
    outcome.metrics.engine_request_id           = result.engine_request_id;
    outcome.metrics.computed_prefill_tokens     = result.computed_prefill_tokens;
    outcome.metrics.prefix_cache_hit_tokens     = result.reused_prompt_tokens;
    outcome.metrics.prefix_reuse_path           = result.prefix_reuse_path;
    outcome.metrics.speculative_backend         = result.speculative.backend;
    outcome.metrics.speculative_draft_window    = result.speculative.draft_window;
    outcome.metrics.speculative_rounds          = result.speculative.rounds;
    outcome.metrics.speculative_draft_tokens    = result.speculative.drafted_tokens;
    outcome.metrics.speculative_accepted_tokens = result.speculative.accepted_tokens;
    outcome.metrics.speculative_fallback_steps  = result.speculative.fallback_steps;
    outcome.metrics.speculative_accepted_per_position =
        std::move(result.speculative.accepted_per_position);

    outcome.tool_calls      = std::move(result.tool_calls);
    outcome.tool_call_parse = result.tool_call_parse;
    return outcome;
}

void GenerationService::warmup() {
    GenerationRequest request;
    ChatTurn turn;
    turn.role = ChatRole::User;
    ContentPart content;
    content.kind     = ContentKind::Text;
    content.text     = "hi";
    content.type_raw = "text";
    turn.content.push_back(std::move(content));
    request.messages.push_back(std::move(turn));
    request.max_tokens = 4;
    PreparedRequest prepared =
        prepare_impl(request, GenerationConsumerMode::Aggregate, {}, {}, {},
                     CacheParticipation::Disabled, DeadlinePolicy::UnboundedStartup);
    run(prepared, nullptr);
}

} // namespace ninfer::serve

