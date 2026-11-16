// V10-14 (upstream c996baa8, T161): the cold-start expert prior - its sources and its ordering.
//
// The stream has one order today: the decayed route counts it observes. A fresh process has observed
// nothing, so every seat decision falls back to the logical id order (experts 0..capacity-1 take the
// first layer). This test pins the four properties the injection point promises:
//
//   1. a table (an artifact's metadata, or a conversion report, or an override file) initialises the
//      order: with no observation yet, the seats follow the row, layer by layer;
//   2. no table anywhere changes nothing: the ranking is the id order it always was, which is what
//      makes the feature safe to ship ahead of any artifact that carries a calibration;
//   3. an explicit override wins over the table, row by row - including a partial override, which
//      leaves the rows it does not name to the artifact;
//   4. the prior is a PRIOR: an expert the router has observed outranks it, so no measured hot order
//      can be overridden by a table.
//
// Host only: no CUDA, no artifact file, no ExpertStream instance. The stream's comparator is
// `precedes()` from models/qwen3_5/expert_priority.h - the same function every seat decision in
// expert_stream.cpp now calls - so the ordering asserted here is the ordering the stream uses.
#include "models/qwen3_5/expert_priority.h"
#include "models/qwen3_5/expert_priority_source.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

using namespace ninfer;
using namespace ninfer::models::qwen3_5;

namespace {

int failures = 0;

void check(bool condition, const char* label) {
    if (!condition) {
        std::cerr << "expert priority: " << label << " failed\n";
        ++failures;
    }
}

void check_order(std::span<const std::uint32_t> got, std::span<const std::uint32_t> want,
                 const char* label) {
    if (std::equal(got.begin(), got.end(), want.begin(), want.end())) { return; }
    std::cerr << "expert priority: " << label << " expected [";
    for (std::size_t i = 0; i < want.size(); ++i) {
        std::cerr << (i == 0 ? "" : ", ") << want[i];
    }
    std::cerr << "] got [";
    for (std::size_t i = 0; i < got.size(); ++i) {
        std::cerr << (i == 0 ? "" : ", ") << got[i];
    }
    std::cerr << "]\n";
    ++failures;
}

void check_order(std::span<const std::uint32_t> got, std::initializer_list<std::uint32_t> want,
                 const char* label) {
    check_order(got, std::span<const std::uint32_t>(want.begin(), want.size()), label);
}

template <typename Call>
bool throws_invalid_argument(Call&& call) {
    try {
        call();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (const std::exception& error) {
        std::cerr << "expert priority: unexpected exception: " << error.what() << "\n";
        return false;
    }
    return false;
}

void set_switch(const char* value) {
#if defined(_WIN32)
    _putenv_s("NINFER_EXPERT_PRIORITY", value);
#else
    if (value == nullptr || *value == '\0') {
        unsetenv("NINFER_EXPERT_PRIORITY");
    } else {
        setenv("NINFER_EXPERT_PRIORITY", value, 1);
    }
#endif
}

// ---- 1/2: a table initialises the cold order; no table keeps it ------------------------------

void check_table_initialises_the_order() {
    const auto document = artifact::Json::parse(R"({"expert_priority": [[2, 0, 3, 1], [1, 3, 0, 2]]})");
    const auto table    = parse_expert_priority(document, 2, 4, "test metadata");
    check(table.layers() == 2, "both rows parsed");
    check_order(table.row(0), {2, 0, 3, 1}, "a row keeps the file's order");

    const std::vector<float> cold(4, 0.0F); // nothing observed: the cold start this exists for
    check_order(table.rank(0, 4), {1, 3, 0, 2}, "row 0 as ranks");
    check_order(seat_ranking(cold, table.rank(0, 4)), {2, 0, 3, 1},
                "a cold start follows the row");
    check_order(seat_ranking(cold, table.rank(1, 4)), {1, 3, 0, 2},
                "each block takes its own row");

    // A row that names nobody (a dense block, or one the calibration skipped) must not disturb the
    // other rows: the miss is per layer, not a reason to refuse the table.
    const auto partial = parse_expert_priority(
        artifact::Json::parse(R"({"expert_priority": [[3, 2, 1, 0], []]})"), 2, 4, "partial metadata");
    check(partial.layers() == 2, "an empty row is still a row");
    check(partial.row(1).empty(), "an empty row names no expert");
    check_order(seat_ranking(cold, partial.rank(1, 4)), {0, 1, 2, 3},
                "an empty row keeps the id order");
}

void check_no_table_keeps_the_order() {
    const auto none = parse_expert_priority(artifact::Json::object(), 2, 4, "no table");
    check(none.empty(), "a document without the key carries no table");

    const std::vector<float>      cold(4, 0.0F);
    const std::vector<std::uint32_t> no_rank;
    check_order(seat_ranking(cold, no_rank), {0, 1, 2, 3}, "no prior, nothing observed: id order");

    // Ties among observed experts are still the id order when no prior exists - the exact ordering
    // the comparators produced before this feature (ascending id over an equal score).
    std::vector<float> tied(4, 2.0F);
    check_order(seat_ranking(tied, no_rank), {0, 1, 2, 3}, "no prior, equal heat: id order");
}

// ---- 3: an explicit override wins over the table, row by row ---------------------------------

void check_override_wins_over_table() {
    const auto base = parse_expert_priority(
        artifact::Json::parse(R"({"expert_priority": [[3, 2, 1, 0], [3, 2, 1, 0]]})"), 2, 4,
        "artifact table");
    // A bare row array is the override's shape, and an empty row means "this layer keeps the table's".
    const auto override_rows =
        parse_expert_priority(artifact::Json::parse("[[0, 1, 2, 3], []]"), 2, 4, "override");
    const auto merged = merge_expert_priority(base, override_rows);

    check_order(merged.row(0), {0, 1, 2, 3}, "the override's row replaces the table's");
    check_order(merged.row(1), {3, 2, 1, 0}, "a row the override leaves empty keeps the table's");
}

// ---- 4: the prior never outranks something the router has used -------------------------------

void check_heat_outranks_the_prior() {
    const auto table = parse_expert_priority(
        artifact::Json::parse(R"({"expert_priority": [[2, 0, 3, 1]]})"), 1, 4, "test metadata");
    const auto rank = table.rank(0, 4); // 2 first, 3 third
    std::vector<float> warm(4, 0.0F);
    warm[3] = 1.0F; // the router used expert 3, which the table puts third
    check_order(seat_ranking(warm, rank), {3, 2, 0, 1},
                "an observed expert leads the prior's first choice");

    // A prior entry with no observation still beats an unobserved expert outside the table: that is
    // the only tie it is allowed to decide.
    warm.assign(4, 0.0F);
    const ExpertPriorityTable empty_row{std::vector<std::vector<std::uint32_t>>{{}, {}}};
    const auto empty_rank = empty_row.rank(0, 4);
    check(empty_rank.size() == 4 && cold_priority_rank(empty_rank, 0) == kNoColdPriority &&
              cold_priority_rank(empty_rank, 3) == kNoColdPriority,
          "an empty row ranks every expert as unnamed");
}

// ---- the parser refuses a table it cannot mean ------------------------------------------------

void check_invalid_rows_are_refused() {
    check(throws_invalid_argument([] {
              (void)parse_expert_priority(artifact::Json::parse("[[0, 1, 1, 2]]"), 1, 4, "dup");
          }),
          "a repeated expert is refused");
    check(throws_invalid_argument([] {
              (void)parse_expert_priority(artifact::Json::parse("[[0, 1, 2, 4]]"), 1, 4, "range");
          }),
          "an out-of-range expert is refused");
    check(throws_invalid_argument([] {
              (void)parse_expert_priority(artifact::Json::parse("[[0, 1, 2]]"), 1, 4, "short");
          }),
          "a row that does not name every expert is refused");
    check(throws_invalid_argument([] {
              (void)parse_expert_priority(artifact::Json::parse("[[0, 1, 2, 3]]"), 2, 4, "rows");
          }),
          "a row count that does not match the model is refused");
    check(throws_invalid_argument([] {
              (void)parse_expert_priority(artifact::Json::parse("[[0, 1, 2.5, 3]]"), 1, 4, "float");
          }),
          "a non-integer entry is refused");
}

// ---- the three sources, in precedence order ---------------------------------------------------

void check_sources() {
    // The artifact's metadata is the first source and needs no file I/O.
    artifact::Directory directory;
    directory.metadata = artifact::Json::parse(
        R"({"name": "test", "expert_priority": [[1, 0], [1, 0]]})");
    const auto from_artifact = resolve_expert_priority(directory, {}, {}, 2, 2);
    check(from_artifact.source == ExpertPrioritySource::Artifact, "metadata is read as the artifact");
    check(from_artifact.origin == "artifact metadata", "the origin names the metadata");
    check_order(from_artifact.table.row(0), {1, 0}, "the metadata's row is installed");

    // Nothing anywhere: the previous behaviour, stated rather than implied.
    artifact::Directory bare;
    bare.metadata     = artifact::Json::parse(R"({"name": "test"})");
    const auto empty  = resolve_expert_priority(bare, {}, {}, 2, 2);
    check(empty.source == ExpertPrioritySource::None, "no source resolves to none");
    check(empty.table.empty(), "no source installs no table");

    // The conversion report beside the entry file is the artifact's second place: the metadata here
    // carries nothing, so the table has to come from `<artifact>.conversion.json`. The entry file
    // itself does not exist - the resolver only stats the report.
    const auto stamp = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto fake_artifact = std::filesystem::temp_directory_path() / ("ninfer_priority_" + stamp + ".ninfer");
    const auto report_path   = std::filesystem::path(fake_artifact.string() + ".conversion.json");
    {
        std::ofstream out(report_path, std::ios::binary);
        out << R"({"name": "test", "expert_priority": [[0, 1], [1, 0]]})";
    }
    const auto from_report = resolve_expert_priority(bare, fake_artifact, {}, 2, 2);
    check(from_report.source == ExpertPrioritySource::Artifact, "the report is the artifact's second place");
    check(from_report.origin == report_path.string(), "the origin names the report");
    check_order(from_report.table.row(0), {0, 1}, "the report's row is installed");

    // A report that does not parse is a build artifact's problem, not the model's: it resolves to "no
    // table" and the load keeps going.
    {
        std::ofstream out(report_path, std::ios::binary);
        out << R"({"expert_priority": [[0, 1],)";
    }
    const auto broken_report = resolve_expert_priority(bare, fake_artifact, {}, 2, 2);
    check(broken_report.source == ExpertPrioritySource::None, "a broken report is ignored");
    {
        std::error_code ignored;
        std::filesystem::remove(report_path, ignored);
    }

    // The explicit override, over the artifact's table. A tiny file is the only way to exercise the
    // file branch; it is removed before the test returns. The name carries a clock reading so two
    // concurrent runs of this test cannot write each other's file.
    const auto override_path = std::filesystem::temp_directory_path() /
                               ("ninfer_expert_priority_override_" +
                                std::to_string(std::chrono::steady_clock::now()
                                                   .time_since_epoch()
                                                   .count()) +
                                ".json");
    {
        std::ofstream out(override_path, std::ios::binary);
        out << R"({"expert_priority": [[0, 1], []]})";
    }
    const auto overridden = resolve_expert_priority(directory, {}, override_path, 2, 2);
    check(overridden.source == ExpertPrioritySource::Override, "the override is reported as such");
    check_order(overridden.table.row(0), {0, 1}, "the override wins over the artifact's row");
    check_order(overridden.table.row(1), {1, 0}, "an empty override row keeps the artifact's row");

    // NINFER_EXPERT_PRIORITY=off turns every source off - including the override path, which must not
    // even be opened (this one does not exist).
    set_switch("off");
    const auto disabled =
        resolve_expert_priority(directory, {}, override_path.string() + ".missing", 2, 2);
    check(disabled.source == ExpertPrioritySource::None, "the switch disables the feature");
    check(disabled.table.empty(), "the switch installs no table");
    set_switch("");

    std::error_code error;
    std::filesystem::remove(override_path, error);
}

} // namespace

int main() {
    check_table_initialises_the_order();
    check_no_table_keeps_the_order();
    check_override_wins_over_table();
    check_heat_outranks_the_prior();
    check_invalid_rows_are_refused();
    check_sources();

    if (failures != 0) {
        std::cerr << "expert priority: " << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "expert priority: ok\n";
    return 0;
}
