#include "product/config/schema.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ninfer::product::config_detail {
namespace {

constexpr FieldSpec kFields[] = {
    {Field::ModelPath, "model.path", "", "", "", "", "FILE", "Model artifact (.ninfer)."},
    {Field::ChatTemplate, "model.chat_template", "--chat-template", "", "", "", "FILE",
     "Chat template override; empty uses the artifact."},
    {Field::Vision, "model.vision", "--vision", "", "", "", "", "Load Vision resources."},
    {Field::Device, "engine.device", "--device", "", "", "", "N", "Local CUDA device ordinal."},
    {Field::MaxContext, "engine.max_context", "--max-context", "", "", "", "N",
     "Per-request context limit (CLI 2048, serve 8192)."},
    {Field::KvDtype, "engine.kv_dtype", "--kv-dtype", "", "", "",
     "bf16|int8|fp8|nvfp4|k8v4|rk8v4|rk4v4|rk4v4-e8|rk2v4-e8|rk4v2-e8", "KV representation."},
    {Field::KvCapacity, "engine.kv_capacity", "--kv-capacity", "", "", "", "N|auto",
     "Shared Main KV capacity; omitted follows --max-context (serve: auto when --max-concurrency "
     "is above 1)."},
    {Field::KvCapacityHeadroom, "engine.kv_capacity_headroom_mib", "--kv-capacity-headroom-mib", "",
     "", "", "MIB",
     "Sizing headroom left free by --kv-capacity auto, in decimal MiB resolved to whole bytes. "
     "This is KV sizing, not the startup memory planning reserve.",
     Scope::Serve, ConfigurationUnit::Bytes},
    {Field::PrefillChunk, "engine.prefill_chunk", "--prefill-chunk", "", "", "", "N",
     "Prefill chunk; a positive multiple of 128."},
    {Field::CudaGraph, "engine.cuda_graph", "", "", "", "--no-cuda-graph", "",
     "Whole-step CUDA Graph execution (on by default)."},
    {Field::MaxConcurrency, "engine.max_concurrency", "--max-concurrency", "", "", "", "N",
     "Resident execution lanes (1..8).", Scope::Serve},
    {Field::MaxPendingRequests, "engine.max_pending_requests", "--max-pending-requests", "", "", "",
     "N", "Pending request capacity (16).", Scope::Serve},
    {Field::PendingTimeout, "engine.pending_timeout_ms", "--pending-timeout-ms", "", "", "", "N",
     "Queue timeout before execution in milliseconds (30000).", Scope::Serve},
    {Field::PrefixReuse, "context_cache.prefix_reuse", "", "", "", "--no-prefix-reuse", "",
     "Cross-request history reuse (serve, on by default).", Scope::Serve},
    {Field::DeviceStateSlots, "context_cache.device_state_slots", "--device-state-slots", "", "",
     "", "N", "Extra complete StateImages beyond the active lanes.", Scope::Serve},
    {Field::HostContext, "context_cache.host_context_mib", "--host-context-mib", "", "", "", "MIB",
     "Shared Host State/KV and in-flight capacity in decimal MiB resolved to whole bytes; 0 "
     "disables backing.",
     Scope::Serve},
    {Field::CostPresets, "context_cache.cost_presets", "--context-cost-presets", "", "", "", "FILE",
     "Context-cost JSON overrides; empty uses the built-in defaults.", Scope::Serve},
    {Field::EvictContract, "context_cache.evict", "--kv-evict", "", "", "", "off|on|auto",
     "KV eviction contract, resolved once at startup.", Scope::Serve},
    {Field::EvictWindow, "context_cache.evict_window_pages", "--kv-evict-window", "", "", "", "N",
     "Eviction window in pages (4).", Scope::Serve},
    {Field::EvictSink, "context_cache.evict_sink_pages", "--kv-evict-sink", "", "", "", "N",
     "Eviction sink pages (1).", Scope::Serve},
    {Field::IdleReclaim, "context_cache.idle_reclaim_ms", "--kv-idle-reclaim-ms", "", "", "", "N",
     "Idle retained-prefix demotion delay in milliseconds (60000); 0 disables it.", Scope::Serve},
    {Field::WddmBudget, "engine.wddm_evictable_budget", "--wddm-evictable-budget", "", "", "", "",
     "Windows: budget against total VRAM minus the DWM floor.", Scope::Serve},
    {Field::SpecBackend, "speculative.backend", "--spec", "", "", "", "none|mtp|dflash|dflash2",
     "Speculative backend."},
    {Field::DraftTokens, "speculative.draft_tokens", "--draft-tokens", "", "", "", "N",
     "Draft window; requires --spec."},
    {Field::ProposalHead, "speculative.proposal_head", "--lm-head-draft", "", "", "", "",
     "Optimized proposal head (requires --spec)."},
    {Field::MediaCache, "media.cache_mib", "--media-cache-mib", "", "", "", "N",
     "Retained media cache in MiB (1024); 0 disables reuse.", Scope::Serve,
     ConfigurationUnit::Bytes},
    {Field::MediaLive, "media.live_mib", "--media-live-mib", "", "", "", "N",
     "All live BF16 patch payloads in MiB (2048).", Scope::Serve, ConfigurationUnit::Bytes},
    {Field::MediaThreads, "media.preprocess_threads", "--media-preprocess-threads", "", "", "", "N",
     "Explicit workers 0..64; 0 selects auto (at most 16).", Scope::Serve},
    {Field::ControllerMode, "engine.controller_mode", "--controller-mode", "", "", "",
     "auto|stable|performance",
     "Who the runtime controller optimizes for (v8 section 12 initial value)."},
    {Field::MaxTokens, "generation.max_tokens", "--max-tokens", "--max-new",
     "--default-max-tokens", "", "N", "Output budget; CLI default 128, serve request default 8192."},
    {Field::Thinking, "generation.thinking", "", "", "", "--no-thinking", "",
     "Thinking override; omitted follows the model and prompt template."},
    {Field::PreserveThinking, "generation.preserve_thinking", "--preserve-thinking", "", "", "", "",
     "Retain closed-turn reasoning in later prompts.", Scope::Serve},
    {Field::ThinkingBudget, "generation.thinking_budget", "--thinking-budget", "",
     "--default-thinking-budget", "", "N", "Model-origin thinking token budget."},
    {Field::Temperature, "generation.sampling.temperature", "--temperature", "", "", "", "F|model",
     "Sampling temperature [0,2]; 0 selects greedy."},
    {Field::TopK, "generation.sampling.top_k", "--top-k", "", "", "", "N|model",
     "Candidate limit [0,20]."},
    {Field::TopP, "generation.sampling.top_p", "--top-p", "", "", "", "F|model",
     "Cumulative probability cutoff [0,1]."},
    {Field::MinP, "generation.sampling.min_p", "--min-p", "", "", "", "F|model",
     "Relative probability cutoff [0,1]."},
    {Field::PresencePenalty, "generation.sampling.presence_penalty", "--presence-penalty", "", "",
     "", "F|model", "Presence penalty [-2,2]."},
    {Field::FrequencyPenalty, "generation.sampling.frequency_penalty", "--frequency-penalty", "", "",
     "", "F|model", "Frequency penalty [-2,2]."},
    {Field::Seed, "generation.sampling.seed", "--seed", "", "", "", "N|model",
     "uint64 seed; CLI default 0, serve default random per request."},
    {Field::Greedy, "generation.greedy", "--greedy", "", "", "", "",
     "Exact argmax: forces temperature 0."},
    {Field::ServerHost, "server.host", "--host", "", "", "", "ADDRESS", "Listen address (127.0.0.1).",
     Scope::Serve},
    {Field::ServerPort, "server.port", "--port", "", "", "", "N", "Listen port (8080).",
     Scope::Serve},
    {Field::ApiKey, "server.api_key", "--api-key", "", "", "", "KEY",
     "Authentication key; empty disables authentication.", Scope::Serve},
    {Field::ModelId, "server.model_id", "--model-id", "", "", "", "ID",
     "Public model alias; omitted uses the artifact metadata.name.", Scope::Serve},
    {Field::Cors, "server.cors", "--cors", "", "", "", "", "Permissive CORS headers.",
     Scope::Serve},
    {Field::MaxRequestSize, "server.max_request_mib", "--max-request-mib", "", "", "", "N",
     "HTTP body limit in MiB before JSON parsing (384).", Scope::Serve,
     ConfigurationUnit::Bytes},
    {Field::ResponseRecords, "server.responses.max_records", "--response-store-max-records", "", "",
     "", "N", "Responses store record limit (1024).", Scope::Serve},
    {Field::ResponseSize, "server.responses.max_mib", "--response-store-max-mib", "", "", "", "N",
     "Responses store capacity in MiB (256).", Scope::Serve, ConfigurationUnit::Bytes},
    {Field::LogLevel, "logging.level", "--log-level", "", "", "", "LEVEL",
     "trace/debug/info/warning/error/critical/off."},
    {Field::StatsInterval, "logging.stats_interval_ms", "--log-stats-interval-ms", "", "", "", "N",
     "Throughput log interval in milliseconds (5000); 0 disables it.", Scope::Serve},
    {Field::RequestJsonl, "logging.request_jsonl", "--request-log-jsonl", "", "", "", "FILE",
     "Structured startup/request log; empty disables it.", Scope::Serve},
    {Field::Prompt, "cli.prompt", "--prompt", "", "", "", "TEXT",
     "Prompt text, mutually exclusive with --messages.", Scope::Cli},
    {Field::Messages, "cli.messages", "--messages", "", "", "", "FILE",
     "Messages/tools JSON; local media paths are relative to that file.", Scope::Cli},
    {Field::ReasoningEffort, "cli.reasoning_effort", "--reasoning-effort", "", "", "",
     "EFFORT|model", "none/minimal/low/medium/high/xhigh/max; none disables thinking.",
     Scope::Cli},
    {Field::RawOutput, "cli.raw_output", "--raw-output", "", "", "", "",
     "Emit the raw output representation.", Scope::Cli},
    {Field::PrintTokenIds, "cli.print_token_ids", "--print-token-ids", "", "", "", "",
     "Print generated token IDs to stderr.", Scope::Cli},
    {Field::StopIds, "cli.stop_token_ids", "--stop-token-id", "", "", "", "N",
     "Token stop condition; repeat to append.", Scope::Cli},
    {Field::Stop, "cli.stop", "--stop", "", "", "", "TEXT", "Content stop string; repeat to append.",
     Scope::Cli},
    {Field::ReasoningStop, "cli.reasoning_stop", "--reasoning-stop", "", "", "", "TEXT",
     "Reasoning stop string; repeat to append.", Scope::Cli},
    {Field::ConstraintKind, "cli.constraint.kind", "--constraint", "", "", "", "none",
     "Clear the inherited output constraint.", Scope::Cli},
    {Field::GrammarFile, "cli.constraint.grammar_file", "--grammar-file", "", "", "", "FILE",
     "GBNF output grammar.", Scope::Cli},
    {Field::JsonSchemaFile, "cli.constraint.json_schema_file", "--json-schema-file", "", "", "",
     "FILE", "JSON Schema output constraint.", Scope::Cli},
    {Field::Regex, "cli.constraint.regex", "--regex", "", "", "", "PATTERN",
     "Regex output constraint; empty is a valid regex.", Scope::Cli},
    {Field::Choices, "cli.constraint.choices", "--choice", "", "", "", "TEXT",
     "Literal output choice; repeat to append, empty entries allowed.", Scope::Cli},
};
static_assert(std::size(kFields) == static_cast<std::size_t>(Field::Count));

constexpr auto kU32Max = std::numeric_limits<std::uint32_t>::max();
constexpr auto kU64Max = std::numeric_limits<std::uint64_t>::max();

[[noreturn]] void invalid(std::string_view message) {
    throw std::invalid_argument(std::string(message));
}

std::uint64_t decimal(std::string_view text) {
    std::uint64_t result = 0;
    const auto parsed    = std::from_chars(text.data(), text.data() + text.size(), result);
    if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
        invalid("expected an unsigned decimal integer");
    return result;
}

std::uint64_t integer(const RawValue& value, bool allow_string, std::uint64_t min,
                      std::uint64_t max) {
    std::uint64_t result = 0;
    if (const auto* number = std::get_if<std::int64_t>(&value)) {
        if (*number < 0) invalid("expected a nonnegative integer");
        result = static_cast<std::uint64_t>(*number);
    } else if (const auto* text = std::get_if<std::string>(&value); text && allow_string) {
        result = decimal(*text);
    } else
        invalid("expected an integer");
    if (result < min || result > max)
        invalid("expected an integer in [" + std::to_string(min) + "," + std::to_string(max) + "]");
    return result;
}

double real(const RawValue& value, bool command_line, double min, double max) {
    double result = 0;
    if (const auto* number = std::get_if<double>(&value))
        result = *number;
    else if (const auto* number = std::get_if<std::int64_t>(&value))
        result = static_cast<double>(*number);
    else if (const auto* text = std::get_if<std::string>(&value); text && command_line) {
        const auto parsed = std::from_chars(text->data(), text->data() + text->size(), result);
        if (text->empty() || parsed.ec != std::errc{} || parsed.ptr != text->data() + text->size())
            invalid("expected a finite number");
    } else
        invalid("expected a number");
    if (!std::isfinite(result) || result < min || result > max)
        invalid("expected a finite number in [" + std::to_string(min) + "," + std::to_string(max) +
                "]");
    return result;
}

std::string text(const RawValue& value, bool nonempty = false) {
    const auto* result = std::get_if<std::string>(&value);
    if (!result || (nonempty && result->empty()))
        invalid(nonempty ? "expected a nonempty string" : "expected a string");
    return *result;
}

std::string enumeration(const RawValue& value, std::initializer_list<std::string_view> choices) {
    const auto result = text(value);
    if (std::find(choices.begin(), choices.end(), result) == choices.end())
        invalid("invalid enum value");
    return result;
}

// Exact decimal MiB -> bytes. Fractional MiB is accepted only when it resolves to a whole number
// of bytes, and the conversion is integer-only so a requested budget is never rounded upward.
// TOML integers are plain MiB counts.
std::uint64_t mib_bytes(const RawValue& value, bool positive) {
    constexpr std::uint64_t bytes_per_mib = 1ULL << 20;
    constexpr std::uint64_t maximum       = std::numeric_limits<std::size_t>::max();
    if (const auto* number = std::get_if<std::int64_t>(&value)) {
        if (*number < 0) invalid("expected a nonnegative MiB count");
        const auto mib = static_cast<std::uint64_t>(*number);
        if (mib > maximum / bytes_per_mib) invalid("MiB count is out of range");
        const std::uint64_t bytes = mib * bytes_per_mib;
        if (positive && bytes == 0) invalid("MiB count must be positive");
        return bytes;
    }
    const auto* text_value = std::get_if<std::string>(&value);
    if (!text_value) invalid("expected an integer MiB count");
    const std::string_view input(*text_value);
    const std::size_t point      = input.find('.');
    const std::string_view whole = input.substr(0, point);
    std::string_view fraction =
        point == std::string_view::npos ? std::string_view{} : input.substr(point + 1);
    if (whole.empty() || whole.find_first_not_of("0123456789") != std::string_view::npos ||
        (point != std::string_view::npos &&
         (fraction.empty() || fraction.find_first_not_of("0123456789") != std::string_view::npos)))
        invalid("expected a nonnegative decimal MiB value");

    std::uint64_t whole_mib = 0;
    for (const char character : whole) {
        const std::uint64_t digit = static_cast<std::uint64_t>(character - '0');
        if (whole_mib > (maximum / bytes_per_mib - digit) / 10)
            invalid("MiB value is out of range");
        whole_mib = whole_mib * 10 + digit;
    }

    // With k fraction digits, bytes = digits * 2^20 / 10^k = (digits / 5^k) * 2^(20-k); 5^20
    // still fits a 64-bit word, so no wider arithmetic is required.
    while (!fraction.empty() && fraction.back() == '0') { fraction.remove_suffix(1); }
    if (fraction.size() > 20) invalid("MiB value must resolve to a whole number of bytes");
    std::uint64_t power_of_five = 1;
    for (std::size_t index = 0; index < fraction.size(); ++index) { power_of_five *= 5U; }
    std::uint64_t remainder = 0; // digit stream modulo 5^k
    std::uint64_t quotient  = 0; // digit stream divided by 5^k
    for (const char character : fraction) {
        const std::uint64_t digit  = static_cast<std::uint64_t>(character - '0');
        const std::uint64_t merged = remainder * 10U + digit;
        quotient                   = quotient * 10U + merged / power_of_five;
        remainder                  = merged % power_of_five;
    }
    if (remainder != 0) invalid("MiB value must resolve to a whole number of bytes");
    const auto fractional_bytes =
        static_cast<std::uint64_t>(quotient << (20U - static_cast<unsigned>(fraction.size())));
    const std::uint64_t whole_bytes = whole_mib * bytes_per_mib;
    if (fractional_bytes > maximum - whole_bytes) invalid("MiB value is out of range");
    const std::uint64_t bytes = whole_bytes + fractional_bytes;
    if (positive && bytes == 0) invalid("MiB value must be positive");
    return bytes;
}

// Integer MiB -> bytes for the options that have always taken whole MiB only. --host-context-mib
// and --kv-capacity-headroom-mib stay on mib_bytes because their historical contract is exact
// bytes; the rest keep their original integer-only range.
std::uint64_t whole_mib_bytes(const RawValue& value, bool positive) {
    if (const auto* text_value = std::get_if<std::string>(&value);
        text_value && text_value->find('.') != std::string::npos)
        invalid("expected an integer MiB count");
    return mib_bytes(value, positive);
}

} // namespace

std::span<const FieldSpec> fields() { return kFields; }

const FieldSpec& field_spec(Field field) { return kFields[static_cast<std::size_t>(field)]; }

bool applies_to(const FieldSpec& spec, EntryPoint entry) {
    return spec.scope == Scope::Common || (spec.scope == Scope::Cli && entry == EntryPoint::Cli) ||
           (spec.scope == Scope::Serve && entry == EntryPoint::Serve);
}

std::string_view flag_for(const FieldSpec& spec, EntryPoint entry) {
    if (entry == EntryPoint::Cli && !spec.cli_flag.empty()) return spec.cli_flag;
    if (entry == EntryPoint::Serve && !spec.serve_flag.empty()) return spec.serve_flag;
    return spec.flag;
}

bool accepts_flag(const FieldSpec& spec, EntryPoint entry, std::string_view spelling) {
    if (!spec.cli_flag.empty() && entry == EntryPoint::Cli && spelling == spec.cli_flag) return true;
    if (!spec.serve_flag.empty() && entry == EntryPoint::Serve && spelling == spec.serve_flag)
        return true;
    return !spec.flag.empty() && spelling == spec.flag;
}

FieldValue parse_field_value(Field field, const RawValue& value, bool command_line) {
    const auto* string = std::get_if<std::string>(&value);
    const auto is      = [&](std::string_view word) { return string && *string == word; };
    switch (field) {
    case Field::KvCapacityHeadroom:
    case Field::HostContext:
        return mib_bytes(value, false);
    case Field::MediaCache:
        return whole_mib_bytes(value, false);
    case Field::MediaLive:
    case Field::MaxRequestSize:
    case Field::ResponseSize:
        return whole_mib_bytes(value, true);
    case Field::PrefillChunk: {
        const auto count = integer(value, command_line, 1, kU32Max);
        if (count % 128 != 0) invalid("expected a positive multiple of 128");
        return count;
    }
    case Field::PrefixReuse:
    case Field::DeviceStateSlots:
    case Field::MaxTokens:
    case Field::Seed:
        if (is("default")) return DefaultValue{};
        break;
    case Field::Thinking:
    case Field::PreserveThinking:
    case Field::Temperature:
    case Field::TopK:
    case Field::TopP:
    case Field::MinP:
    case Field::PresencePenalty:
    case Field::FrequencyPenalty:
    case Field::ReasoningEffort:
        if (is("model")) return DefaultValue{};
        break;
    case Field::ThinkingBudget:
        if (is("none")) return DefaultValue{};
        break;
    case Field::MediaThreads:
        if (is("auto")) return DefaultValue{};
        break;
    default:
        break;
    }
    switch (field) {
    case Field::Vision:
    case Field::CudaGraph:
    case Field::PrefixReuse:
    case Field::Thinking:
    case Field::PreserveThinking:
    case Field::Cors:
    case Field::RawOutput:
    case Field::PrintTokenIds:
    case Field::WddmBudget:
    case Field::Greedy:
        if (const auto* boolean = std::get_if<bool>(&value)) return *boolean;
        if (command_line && (is("true") || is("false"))) return is("true");
        invalid("expected a boolean");
    case Field::Device:
        return integer(value, command_line, 0, std::numeric_limits<int>::max());
    case Field::MaxContext:
    case Field::MaxPendingRequests:
    case Field::MaxTokens:
    case Field::ThinkingBudget:
    case Field::PendingTimeout:
        return integer(value, command_line, 1, kU32Max);
    case Field::EvictWindow:
        return integer(value, command_line, 1, kU32Max);
    case Field::DeviceStateSlots:
    case Field::DraftTokens:
    case Field::EvictSink:
    case Field::StatsInterval:
        return integer(value, command_line, 0, kU32Max);
    case Field::IdleReclaim:
        return integer(value, command_line, 0, kU64Max);
    case Field::MaxConcurrency:
        return integer(value, command_line, 1, kMaximumConcurrency);
    case Field::ServerPort:
        return integer(value, command_line, 1, 65535);
    case Field::ResponseRecords:
        return integer(value, command_line, 1, kU64Max);
    case Field::TopK:
        return integer(value, command_line, 0, 20);
    case Field::Seed:
        if (is("random")) return DefaultValue{};
        return integer(value, true, 0, kU64Max);
    case Field::KvCapacity:
        if (is("auto") || is("context")) return *string;
        return integer(value, command_line, 1, kU32Max);
    case Field::MediaThreads:
        return integer(value, command_line, 0, 64);
    case Field::Temperature:
        return real(value, command_line, 0, 2);
    case Field::TopP:
    case Field::MinP:
        return real(value, command_line, 0, 1);
    case Field::PresencePenalty:
    case Field::FrequencyPenalty:
        return real(value, command_line, -2, 2);
    case Field::KvDtype:
        return enumeration(value, {"bf16", "int8", "fp8", "nvfp4", "k8v4", "rk8v4", "rk4v4",
                                   "rk4v4-e8", "rk2v4-e8", "rk4v2-e8"});
    case Field::SpecBackend:
        return enumeration(value, {"none", "mtp", "dflash", "dflash2"});
    case Field::ProposalHead:
        if (command_line && (is("true") || is("false"))) return std::string(is("true") ? "optimized"
                                                                                       : "full");
        return enumeration(value, {"full", "optimized"});
    case Field::EvictContract:
        return enumeration(value, {"off", "on", "auto"});
    case Field::ControllerMode:
        return enumeration(value, {"auto", "stable", "performance"});
    case Field::ReasoningEffort:
        return enumeration(value, {"none", "minimal", "low", "medium", "high", "xhigh", "max"});
    case Field::ConstraintKind:
        if (command_line) return enumeration(value, {"none"});
        return enumeration(value,
                           {"none", "grammar", "json_object", "json_schema", "regex", "choice"});
    case Field::LogLevel:
        return enumeration(value,
                           {"trace", "debug", "info", "warning", "error", "critical", "off"});
    case Field::ModelPath:
    case Field::ServerHost:
    case Field::Prompt:
    case Field::Messages:
    case Field::GrammarFile:
    case Field::JsonSchemaFile:
        return text(value, true);
    case Field::ModelId:
        return text(value, true);
    case Field::CostPresets:
        return command_line ? text(value, true) : text(value);
    case Field::RequestJsonl:
        return command_line ? text(value, true) : text(value);
    case Field::ChatTemplate:
    case Field::ApiKey:
    case Field::Regex:
        return text(value);
    case Field::StopIds: {
        std::vector<std::uint32_t> result;
        if (command_line) {
            result.push_back(static_cast<std::uint32_t>(
                integer(value, true, 0, std::numeric_limits<TokenId>::max())));
        } else {
            const auto* values = std::get_if<std::vector<std::int64_t>>(&value);
            if (!values) invalid("expected an array of integer token IDs");
            for (auto token : *values) {
                if (token < 0 || token > std::numeric_limits<TokenId>::max())
                    invalid("token ID is out of range");
                result.push_back(static_cast<std::uint32_t>(token));
            }
        }
        return result;
    }
    case Field::Stop:
    case Field::ReasoningStop:
    case Field::Choices: {
        std::vector<std::string> result;
        if (command_line)
            result.push_back(text(value));
        else {
            const auto* values = std::get_if<std::vector<std::string>>(&value);
            if (!values) invalid("expected an array of strings");
            result = *values;
        }
        if (field == Field::Choices) {
            if (result.empty()) invalid("choices must contain at least one string");
        } else if (std::any_of(result.begin(), result.end(),
                               [](const auto& v) { return v.empty(); }))
            invalid("stop strings must be nonempty");
        return result;
    }
    case Field::KvCapacityHeadroom:
    case Field::HostContext:
    case Field::MediaCache:
    case Field::MediaLive:
    case Field::MaxRequestSize:
    case Field::ResponseSize:
        break;
    case Field::Count:
        break;
    }
    throw std::logic_error("unhandled startup field");
}

void field_error(const FieldSpec& spec, const ConfigSource& source,
                 const std::filesystem::path& config_path, std::string_view message) {
    std::string location;
    if (source.kind == ConfigSourceKind::File)
        location = config_path.string() + ":" + std::to_string(source.line) + ":" +
                   std::to_string(source.column) + ": ";
    else if (source.kind == ConfigSourceKind::CommandLine)
        location = "argument " + std::to_string(source.argument) + " (" +
                   std::string(spec.flag.empty() ? spec.key : spec.flag) + "): ";
    throw std::invalid_argument(location + std::string(spec.key) + ": " + std::string(message));
}

} // namespace ninfer::product::config_detail
