#include "models/qwen3_5/expert_priority_source.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace ninfer::models::qwen3_5 {
namespace {

// The row array of one `expert_priority` document, or nullptr when the document does not carry one.
// Both shapes are accepted because two different files legitimately hold the same rows: an artifact's
// metadata is an object with the key, while a conversion report or a hand-written override names the
// key itself (or holds the bare array).
const artifact::Json* priority_rows(const artifact::Json& document) {
    if (document.is_array()) { return &document; }
    if (!document.is_object()) { return nullptr; }
    const auto found = document.find("expert_priority");
    return found == document.end() ? nullptr : &*found;
}

bool feature_disabled() {
    const char* value = std::getenv("NINFER_EXPERT_PRIORITY");
    if (value == nullptr || *value == '\0') { return false; }
    return value[0] == '0' || std::strcmp(value, "off") == 0 || std::strcmp(value, "OFF") == 0;
}

// A conversion report is a per-object manifest - megabytes for a 35B artifact - and today it carries
// no priority table, so the key is looked for in the bytes before the document is parsed. A false
// positive only costs the parse; a false negative (the key spelled differently) is caught because the
// key this reads is the same one the parser accepts.
std::optional<artifact::Json> read_report_table(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) { return std::nullopt; }
    const std::string text((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
    if (text.empty() || text.find("\"expert_priority\"") == std::string::npos) {
        return std::nullopt;
    }
    return artifact::parse_json(text, "conversion report (" + path.string() + ")");
}

} // namespace

artifact::Json read_expert_priority_json(const std::filesystem::path& path, std::string_view what) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw artifact::ArtifactError(std::string(what) + ": cannot open " + path.string());
    }
    const std::string text((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
    if (text.empty()) {
        throw artifact::ArtifactError(std::string(what) + ": " + path.string() + " is empty");
    }
    return artifact::parse_json(text, std::string(what) + " (" + path.string() + ")");
}

ExpertPriorityTable parse_expert_priority(const artifact::Json& document, std::size_t layers,
                                          std::size_t experts, std::string_view label) {
    ExpertPriorityTable out;
    const auto* rows = priority_rows(document);
    if (rows == nullptr) { return out; }
    const std::string where(label);
    if (!rows->is_array()) {
        throw std::invalid_argument(where + ": expert_priority is not an array of rows");
    }
    if (rows->size() != layers) {
        throw std::invalid_argument(where + ": expert_priority has " +
                                    std::to_string(rows->size()) + " rows but the model has " +
                                    std::to_string(layers) + " text blocks");
    }
    out.rows.resize(layers);
    for (std::size_t layer = 0; layer < layers; ++layer) {
        const auto& row = (*rows)[layer];
        if (row.is_null()) { continue; }
        if (!row.is_array()) {
            throw std::invalid_argument(where + ": row " + std::to_string(layer) +
                                        " is not an array of expert ids");
        }
        if (row.empty()) { continue; } // a dense/uncalibrated block: that layer keeps the id order
        if (row.size() != experts) {
            throw std::invalid_argument(where + ": row " + std::to_string(layer) + " names " +
                                        std::to_string(row.size()) + " experts but the model has " +
                                        std::to_string(experts));
        }
        std::vector<std::uint8_t> seen(experts, 0);
        auto& order = out.rows[layer];
        order.reserve(experts);
        for (std::size_t position = 0; position < row.size(); ++position) {
            const auto& value = row[position];
            if (!value.is_number_integer()) {
                throw std::invalid_argument(where + ": row " + std::to_string(layer) + ", entry " +
                                            std::to_string(position) + " is not an integer");
            }
            const auto expert = value.get<std::int64_t>();
            if (expert < 0 || static_cast<std::uint64_t>(expert) >= experts) {
                throw std::invalid_argument(where + ": row " + std::to_string(layer) + ", entry " +
                                            std::to_string(position) + " is expert " +
                                            std::to_string(expert) + ", outside [0, " +
                                            std::to_string(experts) + ")");
            }
            const auto id = static_cast<std::uint32_t>(expert);
            if (seen[id] != 0) {
                throw std::invalid_argument(where + ": row " + std::to_string(layer) +
                                            " names expert " + std::to_string(id) + " twice");
            }
            seen[id] = 1;
            order.push_back(id);
        }
    }
    return out;
}

std::filesystem::path expert_priority_override_path() {
    const char* value = std::getenv("NINFER_EXPERT_PRIORITY_FILE");
    if (value == nullptr || *value == '\0') { return {}; }
    return std::filesystem::path(value);
}

ResolvedExpertPriority resolve_expert_priority(const artifact::Directory& directory,
                                               const std::filesystem::path& artifact,
                                               const std::filesystem::path& override_file,
                                               std::size_t layers, std::size_t experts) {
    ResolvedExpertPriority out;
    if (feature_disabled() || layers == 0 || experts == 0) { return out; }

    // 1. The artifact's own metadata: already parsed by the reader, so this source costs no I/O.
    auto table = parse_expert_priority(directory.metadata, layers, experts, "artifact metadata");
    if (!table.empty()) {
        out.table  = std::move(table);
        out.source = ExpertPrioritySource::Artifact;
        out.origin = "artifact metadata";
    }

    // 2. The conversion report the converter writes beside the entry file. Only consulted when the
    //    metadata carried no table: the artifact's own statement outranks a build-time report. A
    //    report that exists but cannot be read as JSON is not this load's problem - it is a build
    //    artifact, not the model - so it resolves to "no table" rather than failing the load.
    if (out.source == ExpertPrioritySource::None && !artifact.empty()) {
        const std::filesystem::path report(artifact.string() + ".conversion.json");
        std::error_code error;
        if (std::filesystem::exists(report, error) && !error) {
            try {
                if (const auto document = read_report_table(report)) {
                    auto reported = parse_expert_priority(*document, layers, experts,
                                                          report.string());
                    if (!reported.empty()) {
                        out.table  = std::move(reported);
                        out.source = ExpertPrioritySource::Artifact;
                        out.origin = report.string();
                    }
                }
            } catch (const artifact::ArtifactError&) {
                // Deliberately ignored: see above.
            }
        }
    }

    // 3. The explicit override, merged row by row over whatever the artifact said. A named file that
    //    is missing, malformed or empty is an error: the caller asked for this prior.
    if (!override_file.empty()) {
        const auto document = read_expert_priority_json(override_file, "expert priority override");
        auto override_table = parse_expert_priority(document, layers, experts,
                                                    override_file.string());
        if (override_table.empty()) {
            throw std::invalid_argument(override_file.string() +
                                        ": the override carries no expert_priority rows");
        }
        out.table  = merge_expert_priority(out.table, override_table);
        out.source = ExpertPrioritySource::Override;
        out.origin = override_file.string();
    }
    return out;
}

} // namespace ninfer::models::qwen3_5
