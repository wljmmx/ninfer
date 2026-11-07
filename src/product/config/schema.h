#pragma once

// The single catalog of startup fields shared by `ninfer` and `ninfer-serve`.
//
// A FieldSpec is the one place a startup setting is declared: TOML key, command-line flag(s),
// value syntax, range/parser and help text. The parser (options.cpp), the TOML reader
// (file.cpp), the startup display (configuration.cpp) and the help text all consume this table,
// so adding a setting cannot leave one of them behind.

#include "product/config/options.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace ninfer::product::config_detail {

enum class Field {
    ModelPath,
    ChatTemplate,
    Vision,
    Device,
    MaxContext,
    KvDtype,
    KvCapacity,
    KvCapacityHeadroom,
    PrefillChunk,
    CudaGraph,
    MaxConcurrency,
    MaxPendingRequests,
    PendingTimeout,
    PrefixReuse,
    DeviceStateSlots,
    HostContext,
    CostPresets,
    EvictContract,
    EvictWindow,
    EvictSink,
    IdleReclaim,
    WddmBudget,
    SpecBackend,
    DraftTokens,
    ProposalHead,
    MediaCache,
    MediaLive,
    MediaThreads,
    ControllerMode,
    MaxTokens,
    Thinking,
    PreserveThinking,
    ThinkingBudget,
    Temperature,
    TopK,
    TopP,
    MinP,
    PresencePenalty,
    FrequencyPenalty,
    Seed,
    Greedy,
    ServerHost,
    ServerPort,
    ApiKey,
    ModelId,
    Cors,
    MaxRequestSize,
    ResponseRecords,
    ResponseSize,
    LogLevel,
    StatsInterval,
    RequestJsonl,
    Prompt,
    Messages,
    ReasoningEffort,
    RawOutput,
    PrintTokenIds,
    StopIds,
    Stop,
    ReasoningStop,
    ConstraintKind,
    GrammarFile,
    JsonSchemaFile,
    Regex,
    Choices,
    Count,
};

enum class Scope { Common, Cli, Serve };

struct FieldSpec {
    Field field = Field::Count;
    std::string_view key;
    // `flag` is the canonical spelling and is accepted on both entry points. An entry point may
    // add its own historical spelling through cli_flag / serve_flag; the help text prints the
    // spelling that entry point honours, while the parser accepts every listed one.
    std::string_view flag;
    std::string_view cli_flag;
    std::string_view serve_flag;
    std::string_view negative_flag;
    std::string_view argument;
    std::string_view description;
    Scope scope            = Scope::Common;
    ConfigurationUnit unit = ConfigurationUnit::None;
};

struct DefaultValue {};

using FieldValue = std::variant<DefaultValue, bool, std::uint64_t, double, std::string,
                                std::vector<std::uint32_t>, std::vector<std::string>>;
// A TOML or command-line value before field-specific parsing. Command-line values arrive as
// strings; TOML keeps the native type so an out-of-type value can be refused instead of coerced.
using RawValue = std::variant<bool, std::int64_t, double, std::string, std::vector<std::int64_t>,
                              std::vector<std::string>>;

struct Assignment {
    FieldValue value;
    ConfigSource source;
};

using StartupPatch = std::array<std::optional<Assignment>, static_cast<std::size_t>(Field::Count)>;

[[nodiscard]] std::span<const FieldSpec> fields();
[[nodiscard]] const FieldSpec& field_spec(Field field);
[[nodiscard]] bool applies_to(const FieldSpec& spec, EntryPoint entry);
// The spelling this entry point documents in its help text.
[[nodiscard]] std::string_view flag_for(const FieldSpec& spec, EntryPoint entry);
// Whether `spelling` names this field on this entry point (canonical, historical or negative).
[[nodiscard]] bool accepts_flag(const FieldSpec& spec, EntryPoint entry, std::string_view spelling);
[[nodiscard]] FieldValue parse_field_value(Field field, const RawValue& value, bool command_line);
[[nodiscard]] StartupPatch read_config_file(const std::filesystem::path& path);
[[noreturn]] void field_error(const FieldSpec& spec, const ConfigSource& source,
                              const std::filesystem::path& config_path, std::string_view message);

inline auto& assignment(StartupPatch& patch, Field field) {
    return patch[static_cast<std::size_t>(field)];
}

inline const auto& assignment(const StartupPatch& patch, Field field) {
    return patch[static_cast<std::size_t>(field)];
}

} // namespace ninfer::product::config_detail
