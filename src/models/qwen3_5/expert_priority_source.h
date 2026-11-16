#pragma once

// V10-14: where a model's cold-start expert prior comes from.
//
// Three sources, in this precedence:
//
//   1. an explicit override - `LoadOptions::expert_priority_file`, a JSON file whose
//      `expert_priority` key holds the same rows an artifact would carry. Always wins (row by row),
//      and a file that is named but missing or malformed is an error, not a silent fallback;
//   2. the artifact itself, either the metadata the .ninfer already parses at load time
//      (`metadata.expert_priority`) or, when the metadata carries none, the conversion report the
//      converter writes next to it (`<artifact>.conversion.json`). The report is a build artifact,
//      not the model: a report that is missing, carries no table, or does not parse resolves to "no
//      table" instead of failing the load. In the metadata a malformed table IS an error - the
//      model itself stated it;
//   3. nothing - the seat order stays the id order, i.e. the behaviour before this existed.
//
// No table is compiled in. Upstream hard-codes its 35B calibration in `moe_priority.cpp`; this port
// reads whatever the artifact carries instead, and an artifact with no calibration (all of them
// today) resolves to "no prior" rather than a number nobody can re-derive here.
//
// `NINFER_EXPERT_PRIORITY=off|0` disables the whole feature: no source is read, no log line printed,
// and every comparator produces the id order.

#include "artifact/schema.h"
#include "models/qwen3_5/expert_priority.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace ninfer::models::qwen3_5 {

enum class ExpertPrioritySource : std::uint8_t {
    None,     // no table anywhere: the previous behaviour
    Artifact, // the artifact's metadata, or its conversion report
    Override, // LoadOptions::expert_priority_file
};

struct ResolvedExpertPriority {
    ExpertPriorityTable table;
    ExpertPrioritySource source = ExpertPrioritySource::None;
    // What was read, for the startup line ("artifact metadata", "<path>.conversion.json", the
    // override's path). Empty when `source == None`.
    std::string origin;
};

// Parses one `expert_priority` document into a table for a model of `layers` blocks and `experts`
// logical experts per block.
//
// Accepted forms: an object with an `expert_priority` key (what an artifact's metadata looks like),
// or the row array itself (what a conversion report key or an override file holds), e.g.
//
//   {"expert_priority": [[3, 1, 0, 2], [0, 2, 3, 1]]}   // one row per text block
//
// Every row must be either empty or a permutation of [0, experts); anything else throws
// std::invalid_argument with the row index, because a table that names a nonexistent or repeated
// expert is a calibration bug that would otherwise seat the wrong experts silently. A document
// without the key yields an empty table - "this artifact carries no calibration" is a legitimate
// answer, not an error.
[[nodiscard]] ExpertPriorityTable parse_expert_priority(const artifact::Json& document,
                                                        std::size_t layers, std::size_t experts,
                                                        std::string_view label);

// Reads and parses a JSON file. Throws artifact::ArtifactError when the file is missing or is not
// JSON, naming `what` (the caller's description of the file) so the failed source is identifiable.
[[nodiscard]] artifact::Json read_expert_priority_json(const std::filesystem::path& path,
                                                       std::string_view what);

// The override path NINFER_EXPERT_PRIORITY_FILE names, or an empty path when the variable is unset
// or empty. Read once per load, so a caller that builds no LoadOptions of its own (the engine) still
// has one way to name an override.
[[nodiscard]] std::filesystem::path expert_priority_override_path();

// Resolves this load's prior. `directory` is the artifact's parsed directory (its `metadata` is the
// first place a table is looked for) and `artifact` its entry path (the conversion report is sought
// beside it). `override_file` is the explicit override, already resolved to the effective path
// (`expert_priority_override_path()` when LoadOptions names none). `layers` is the text block count
// and `experts` the routed expert count - the geometry every row is validated against; nothing about
// 35B is assumed here.
[[nodiscard]] ResolvedExpertPriority resolve_expert_priority(const artifact::Directory& directory,
                                                             const std::filesystem::path& artifact,
                                                             const std::filesystem::path& override_file,
                                                             std::size_t layers,
                                                             std::size_t experts);

} // namespace ninfer::models::qwen3_5
