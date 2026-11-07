#pragma once

// Unified startup configuration for the two product entry points.
//
// One field catalog (product/config/schema.h) describes every startup setting: its TOML key, its
// command-line flag, its parser/range and its help text. Both `ninfer` and `ninfer-serve` parse
// through the same layered pipeline (defaults -> optional TOML file -> argv), so a setting cannot
// exist on one entry point and silently mean something else on the other, and every value keeps
// its provenance for the startup log.
//
// Scope, deliberately unchanged from v8 section 12: everything here is a STARTUP value. It seeds
// EngineOptions / request defaults once, before the engine exists. The runtime parameter bus
// (models/qwen3_5/execution/param_bus.h) is still the single registry of record afterwards - the
// NINFER_* environment variables are its initial values, not this layer's, and the controller may
// move the L2/L3 parameters at runtime. Nothing in this file re-reads the environment or writes a
// bus parameter; a config file or flag is a launch decision, not a live control.

#include "ninfer/types.h"
#include "product/logging/logging.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace ninfer::product {

enum class EntryPoint { Cli, Serve };
enum class ConfigSourceKind { Default, File, CommandLine };

// Where one startup value came from. `argument` is the argv index (command line) and `line` /
// `column` point into the TOML file; both survive into the startup log.
struct ConfigSource {
    ConfigSourceKind kind = ConfigSourceKind::Default;
    std::uint32_t line    = 0;
    std::uint32_t column  = 0;
    int argument          = 0;
};

struct StartupProvenance {
    std::filesystem::path config_path;
    // Exact argv with secret-bearing values replaced. Provenance only; never drives execution.
    std::vector<std::string> argv;
    std::map<std::string, ConfigSource, std::less<>> fields;
};

struct GenerationDefaults {
    std::uint32_t max_tokens = 128;
    std::optional<bool> enable_thinking;
    std::optional<bool> preserve_thinking;
    std::optional<std::uint32_t> thinking_budget;
    // A missing seed asks for a fresh seed when each request is formed (serve); the CLI keeps
    // the historical default of 0.
    SamplingOverrides sampling;
    // --greedy: exact argmax. Kept as a flag (not just temperature 0) because serve re-applies it
    // per request in translate.cpp, overriding a request's own temperature.
    bool greedy = false;
};

struct StartupLoggingOptions {
    LogLevel level                  = LogLevel::Info;
    std::uint32_t stats_interval_ms = 5000;
    std::filesystem::path request_jsonl;
};

// What both entry points share: the engine knobs, the request generation defaults, logging and
// the provenance of every value above.
struct LaunchOptions {
    EngineOptions engine;
    GenerationDefaults generation;
    StartupLoggingOptions logging;
    StartupProvenance provenance;
    // v8 section 12: "auto" | "stable" | "performance". Applied through
    // set_controller_mode_initial() before the engine exists; the environment variable
    // NINFER_CONTROLLER_MODE is the equivalent channel for launches without the flag.
    std::string controller_mode = "auto";
};

[[nodiscard]] LaunchOptions default_launch_options(EntryPoint entry);
[[nodiscard]] std::uint64_t random_sampling_seed();

enum class ConstraintKind { None, Grammar, JsonObject, JsonSchema, Regex, Choice };

struct ConstraintOptions {
    ConstraintKind kind = ConstraintKind::None;
    std::filesystem::path path;
    std::string regex;
    std::vector<std::string> choices;
};

struct CliOptions {
    bool help_requested  = false;
    LaunchOptions launch = default_launch_options(EntryPoint::Cli);
    std::string prompt;
    std::filesystem::path messages_path;
    std::optional<ReasoningEffort> reasoning_effort;
    bool raw_output      = false;
    bool print_token_ids = false;
    std::vector<TokenId> stop_token_ids;
    std::vector<StopString> stop_strings;
    ConstraintOptions constraint;
};

inline constexpr std::uint32_t kDefaultServeMaxTokens     = 8192;
inline constexpr std::size_t kDefaultMaxRequestBytes      = 384ULL << 20;
inline constexpr std::size_t kDefaultResponseStoreRecords = 1024;
inline constexpr std::size_t kDefaultResponseStoreBytes   = 256ULL << 20;

struct ServeOptions {
    bool help_requested  = false;
    LaunchOptions launch = default_launch_options(EntryPoint::Serve);
    std::string host     = "127.0.0.1";
    int port             = 8080;
    std::string api_key;                          // empty => no auth
    std::optional<std::string> model_id_override; // unset => artifact metadata.name
    bool enable_cors                       = false;
    std::size_t max_request_bytes          = kDefaultMaxRequestBytes;
    std::size_t response_store_max_records = kDefaultResponseStoreRecords;
    std::size_t response_store_max_bytes   = kDefaultResponseStoreBytes;
    // appendix IL: serve-only. Zero disables the idle demotion. It is exported as the
    // NINFER_KV_IDLE_RECLAIM_MS initial value before the engine starts, which is why it stays a
    // serve option rather than an EngineOptions field.
    std::uint64_t idle_reclaim_ms = 60000;
};

[[nodiscard]] CliOptions parse_cli_options(int argc, char** argv);
[[nodiscard]] ServeOptions parse_serve_options(int argc, char** argv);
[[nodiscard]] std::string startup_usage_text(std::string_view executable, EntryPoint entry);
[[nodiscard]] std::string resolve_public_model_id(const ServeOptions& options,
                                                  std::string_view artifact_model_name);

using ConfigurationValue = std::variant<bool, std::uint64_t, double, std::string,
                                        std::vector<std::uint32_t>, std::vector<std::string>>;
enum class ConfigurationUnit { None, Bytes, Milliseconds };

struct ConfigurationItem {
    std::string key;
    ConfigurationValue value;
    ConfigSource source;
    ConfigurationUnit unit = ConfigurationUnit::None;
};

struct StartupConfigurationView {
    EntryPoint entry;
    std::filesystem::path config_path;
    std::vector<std::string> argv;
    std::vector<ConfigurationItem> items;
    std::map<std::string, std::string> environment;
};

[[nodiscard]] StartupConfigurationView configuration_view(const CliOptions& options);
[[nodiscard]] StartupConfigurationView configuration_view(const ServeOptions& options);

} // namespace ninfer::product
