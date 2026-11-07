#pragma once

#include "ninfer/types.h"
#include "product/config/options.h"

#include <string>

namespace spdlog {
class logger;
}

namespace ninfer::product {

// One aligned block: field key, resolved value, and where the value came from ([argv:N] /
// [file:line] / [default]). Secrets are already redacted by the parser.
[[nodiscard]] std::string format_configuration(const StartupConfigurationView& view);
void log_configuration(spdlog::logger& logger, const StartupConfigurationView& view);

// The values Engine actually resolved (capacity after page alignment, Host budget after the
// target plan). Logged once per launch so "what was asked for" and "what the engine chose" are
// both on the record.
void log_resolved_configuration(spdlog::logger& logger, const EngineOptions& options,
                                const MemorySummary& memory);

} // namespace ninfer::product
