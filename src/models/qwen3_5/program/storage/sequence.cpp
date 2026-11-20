#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/kv_tier_stats.h"
#include "core/device.h"
#include "core/transfer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <fstream>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

namespace {

// Optional KV eviction window, read once from the environment. A zero window leaves the block
// table exactly as published, so the default configuration is untouched and bit-identical.
// Sink pages are always kept; the window keeps the most recent pages.
struct KvEvictionWindow {
    std::uint32_t sink_pages   = 0;
    std::uint32_t window_pages = 0;
    // Explicit logical pages to keep as well: stands in for a query-aware keep set until a
    // relevance scorer exists, so the mechanism and the keep budget can be measured first.
    std::vector<std::uint32_t> keep_pages;

    [[nodiscard]] bool enabled() const noexcept {
        return window_pages != 0 || !keep_pages.empty();
    }
};

// P6 2c (appendix EU): a named preset for the configurations this work verified. Every knob is
// opt-in and independent, so using the feature meant knowing four variables whose values only existed in
// the appendices. The table is the measured design rule - budget-first, small guaranteed window, sink = 1
// (EO/ER/EP/ET) - and an explicit per-variable environment always overrides it, so the default (unset)
// path is byte-identical to before.
struct KvEvictionPreset {
    std::uint32_t sink_pages;
    std::uint32_t window_pages;
    std::uint32_t budget;
};

std::optional<KvEvictionPreset> kv_eviction_preset() {
    static const std::optional<KvEvictionPreset> preset = []() -> std::optional<KvEvictionPreset> {
        const char* value = std::getenv("NINFER_KV_EVICT_PROFILE");
        if (value == nullptr) { return std::nullopt; }
        const std::string name(value);
        if (name.empty() || name == "off") { return std::nullopt; }
        if (name == "sparse") { return KvEvictionPreset{1u, 4u, 2u}; }
        if (name == "bal") { return KvEvictionPreset{1u, 4u, 8u}; }
        throw std::invalid_argument(
            "NINFER_KV_EVICT_PROFILE must be sparse, bal or off (see the P6 2c appendices)");
    }();
    return preset;
}

// P6 2c: the page scores are per-call by default (appendix CO). They are a marginal over the streams
// of one call, so an accumulating buffer would conflate calls. The two statistics also live in two
// separate planes - `page_score` carries the prefill hit count, `page_score_peak` the decode peak
// logit - precisely so that a step running both shapes cannot mix units in one row (appendices
// DD/DE). NINFER_KV_EVICT_SCORES_CUMULATIVE restores the old accumulating profile for A/B against
// the archives in .temp/kv.
bool kv_page_scores_cumulative_enabled() {
    static const bool enabled = std::getenv("NINFER_KV_EVICT_SCORES_CUMULATIVE") != nullptr;
    return enabled;
}

// P6 2c B4 (appendices DN/DM): the keep set the selector derives from the page-score planes.
//
// NINFER_KV_EVICT_KEEP_FROM_SCORES=<k> asks for up to 2k pages on top of the sink/window pages the
// eviction window keeps anyway. The choice is the UNION of two top-k lists over the candidate pages the
// window does not keep:
//   * the contrast D[p] = recent[p]/K - (all[p] - recent[p])/(W' - K) - what the newest rows prefer
//     over and above the body rows, which DM validated 11/11 on the fixture set;
//   * the every-row profile itself, because a needle that sits on a structurally hot page is exactly
//     what D cancels (DM.4/DM.5, reproduced twice) while the raw profile still sees it.
// K is the same newest-rows span the scoring hooks read, and W' is nominal: D survives W' anywhere in
// [128, 1024] for the spans in use (DO.3), so the selector never needs the call's true row count.
std::uint32_t keep_from_scores_budget() {
    static const std::uint32_t budget = [] {
        const char* value = std::getenv("NINFER_KV_EVICT_KEEP_FROM_SCORES");
        if (value == nullptr) {
            const auto preset = kv_eviction_preset();
            return preset ? preset->budget : std::uint32_t{0};
        }
        const unsigned long parsed = std::strtoul(value, nullptr, 10);
        return parsed > 0 ? static_cast<std::uint32_t>(parsed) : std::uint32_t{0};
    }();
    return budget;
}

// P6 2c (appendix EX): a target RETENTION ratio, so the knob stops depending on how long the document is.
// The budget above is a page count, and its meaning collapses as documents grow - measured on a real
// 106-page document the preset `bal` (k=8) retained only 17% and answered 1/8, while restoring the
// fraction recovered 7/8 (appendix EW). With this set the budget is derived from `live` and the keep set
// is capped to the same number, so the fraction is what the caller asked for.
std::uint32_t keep_from_scores_ratio() {
    static const std::uint32_t ratio = [] {
        const char* value = std::getenv("NINFER_KV_EVICT_KEEP_RATIO");
        if (value == nullptr) { return std::uint32_t{0}; }
        const long parsed = std::strtol(value, nullptr, 10);
        if (parsed <= 0 || parsed >= 100) {
            throw std::invalid_argument(
                "NINFER_KV_EVICT_KEEP_RATIO must be a percentage in 1..99 (0 or unset disables it;"
                " 100% would mean no eviction at all)");
        }
        return static_cast<std::uint32_t>(parsed);
    }();
    return ratio;
}

// Pages the selector may add so that sink + window + selection reaches the requested fraction.
std::uint32_t keep_from_scores_target(std::uint32_t live, std::uint32_t sink_pages,
                                      std::uint32_t window_pages) {
    const std::uint32_t ratio = keep_from_scores_ratio();
    if (ratio == 0 || live == 0) { return 0; }
    const std::uint64_t wanted = (static_cast<std::uint64_t>(live) * ratio + 99u) / 100u;
    const std::uint32_t fixed  = sink_pages + window_pages;
    if (wanted <= fixed) { return 1u; }
    return static_cast<std::uint32_t>(wanted - fixed);
}

// P6 2c (appendix DW.3): delay the eviction until the prompt has been read. The scorer measures the
// model's attention under the visibility in force, so evicting during prefill sparsifies its own input
// and can push a marginal page under the threshold (measured on n4-q2: D=+16.25 fully visible, -1.66
// after eviction). With this on, a prefill-shaped call only REMEMBERS the selection; the window is
// applied from the first decision after the prompt is complete. Off by default.
bool defer_eviction_enabled() {
    static const bool enabled = std::getenv("NINFER_KV_EVICT_DEFER") != nullptr;
    return enabled;
}

// appendix GC (FZ layer 0, O1): start the per-round eviction scans at the frontier the previous
// round already retired instead of re-walking [sink, live - window) from the sink every round. The
// evicted prefix never comes back, so the walk is pure overhead - FW.5 measured 4.4 us per dropped
// page per round for exactly this. Off by default, and only used on rounds whose keep set is
// EMPTY: the selector recomputes keep every call (appendix FI), and a page it may name again has to
// stay reachable, so a selector round falls back to the original full walk and the contract's
// capacity is unchanged.
bool kv_eviction_frontier_enabled() {
    const char* value = std::getenv("NINFER_KV_EVICT_FRONTIER");
    return value != nullptr && *value != '\0' && std::strtol(value, nullptr, 10) != 0;
}

// appendix GE (O2): write only the newly evicted interval of the execution row instead of
// rebuilding the whole row from the keep rule on every round. Equivalent on a frontier round
// (see the caller for the three conditions), and a separate knob from the frontier so the two
// changes can be attributed instead of landing as one lump. Off by default.
bool kv_eviction_interval_enabled() {
    const char* value = std::getenv("NINFER_KV_EVICT_INTERVAL");
    return value != nullptr && *value != '\0' && std::strtol(value, nullptr, 10) != 0;
}

// Which list(s) the keep set is built from (appendix DS.4). `union` is the validated default; the two
// single-list modes exist to find out whether half the budget does the same job.
enum class KeepMode { Union, Duo, Contrast, Raw, Peak };

KeepMode keep_mode() {
    static const KeepMode mode = [] {
        const char* value = std::getenv("NINFER_KV_EVICT_KEEP_MODE");
        if (value == nullptr) { return KeepMode::Union; }
        const std::string_view text(value);
        if (text == "duo") { return KeepMode::Duo; }
        if (text == "contrast") { return KeepMode::Contrast; }
        if (text == "raw") { return KeepMode::Raw; }
        if (text == "peak") { return KeepMode::Peak; }
        return KeepMode::Union;
    }();
    return mode;
}

// The same knob the INT8 scoring hooks read (int8/template_launch.cuh); both read it once, and the
// scoring header is deliberately not includable from here (appendix C3 keeps the scoring path to one TU).
std::int32_t score_row_span() {
    static const std::int32_t span = [] {
        const char* value = std::getenv("NINFER_KV_PAGE_SCORE_ROW_SPAN");
        if (value == nullptr) {
            // The preset and the retention ratio both need the newest-rows plane; 8 is the value
            // every measurement used.
            if (kv_eviction_preset().has_value() ||
                std::getenv("NINFER_KV_EVICT_KEEP_RATIO") != nullptr) {
                return std::int32_t{8};
            }
            return std::int32_t{0};
        }
        const long parsed = std::strtol(value, nullptr, 10);
        return parsed > 0 ? static_cast<std::int32_t>(parsed) : std::int32_t{0};
    }();
    return span;
}

// Nominal row count for the contrast's normalisation (DO.3 measured that anything in [128, 1024]
// keeps the ranking for the spans in use).
constexpr std::int32_t kNominalScoreRows = 1024;

// P6 2c (appendix EB): let the window be a fraction of the context rather than a fixed page count. The
// window is unconditional retention, so a fixed count is a fixed fraction that hurts short corpora
// hardest - 4 pages is 8% of a 52-page context but 13% of a 31-page one, which is the whole of the
// short-corpus overhead measured in EA.4. With NINFER_KV_EVICT_WINDOW_PCT set the window becomes
// min(requested, ceil(live * pct / 100)), never below one page; it only ever shrinks the absolute knob,
// so leaving the variable unset keeps the old behaviour exactly.
// P6 2c (appendix EY, P3a): release the Device replicas of the pages the window just masked. The
// kernel never reads an absent page, so this can only return capacity, never change a result - the
// gate is a bit-identical PPL. Default off, like every other knob on this path.
bool kv_eviction_release_enabled() {
    const char* value = std::getenv("NINFER_KV_EVICT_RELEASE");
    return value != nullptr && *value != '\0' && std::strtol(value, nullptr, 10) != 0;
}

// appendix FF (M3 3.2b): demote the masked pages to Host instead of dropping them - mutually
// exclusive with the release switch, see demote_absent_pages_to_host().
//
// appendix GU (item 1b): read as a TRI-state. The enable predicate alone cannot express
// "explicitly off", because a numeric 0 is indistinguishable from unset - so once item 1 made
// demote the default when the contract names no fate, the pre-item-1 behaviour (masked pages stay
// resident) had no reachable switch combination left, which would have made its historical
// numbers unreproducible.
enum class DemoteRequest : std::uint8_t { Unset, Off, On };

DemoteRequest kv_host_demote_request() {
    const char* value = std::getenv("NINFER_KV_EVICT_HOST_DEMOTE");
    if (value == nullptr || *value == '\0') { return DemoteRequest::Unset; }
    return std::strtol(value, nullptr, 10) != 0 ? DemoteRequest::On : DemoteRequest::Off;
}

bool kv_host_demote_enabled() {
    return kv_host_demote_request() == DemoteRequest::On;
}

// port (sm_89, V10-79): the resident-KV leg's step half. With the arena smaller than the logical
// page space, a page the Device does not hold has to be copied back before the layer that reads it
// runs - and a page this step does not read is what makes room. It is a CONTRACT in the same sense
// the demotion is: it is the only knob under which a page may be absent from the Device while the
// row still addresses it.
//
// P8: the leg is ON by default - a tiered arena is exactly the state in which a page can be absent,
// and a leg that only runs when someone remembers an environment variable leaves every demoted page
// unreadable. The tri-state keeps every recorded experiment reproducible: an explicit
// NINFER_KV_PAGE_IN wins (0 restores the pre-V10-79 step path), and NINFER_KV_NO_PAGE_IN is the
// opt-out spelled for the new default, so an A/B needs one variable with the opposite polarity.
// The leg still only ACTS when the arena is tiered and a Host tier exists (page_in_active()), so a
// fully resident run - and every frozen gate - is byte-identical to before.
bool kv_page_in_requested() {
    if (const char* opt_out = std::getenv("NINFER_KV_NO_PAGE_IN");
        opt_out != nullptr && *opt_out != '\0' && std::strtol(opt_out, nullptr, 10) != 0) {
        return false;
    }
    const char* value = std::getenv("NINFER_KV_PAGE_IN");
    if (value == nullptr || *value == '\0') { return true; }
    return std::strtol(value, nullptr, 10) != 0;
}

// Verification knob (default off): how many of the addressed pages the pass just staged to give
// back once, so the same call has to read them out of the Host tier again. It exists because the
// leg's failure mode is silent (attention would read a block-table entry that names a recycled
// page), so the honest test is "evict and read back inside a real step, and the tokens do not
// move".
std::uint32_t kv_page_in_probe_pages() {
    const char* value = std::getenv("NINFER_KV_PAGE_IN_PROBE");
    if (value == nullptr || *value == '\0') { return 0; }
    const long parsed = std::strtol(value, nullptr, 10);
    return parsed > 0 ? static_cast<std::uint32_t>(parsed) : 0U;
}

std::uint32_t effective_window_pages(std::uint32_t requested, std::uint32_t live) {
    const char* value = std::getenv("NINFER_KV_EVICT_WINDOW_PCT");
    if (value == nullptr || *value == '\0' || live == 0) { return requested; }
    const long pct = std::strtol(value, nullptr, 10);
    if (pct <= 0) { return requested; }
    const std::uint64_t scaled =
        (static_cast<std::uint64_t>(live) * static_cast<std::uint64_t>(pct) + 99u) / 100u;
    const std::uint32_t pages = scaled == 0 ? 1u : static_cast<std::uint32_t>(scaled);
    return pages < requested ? pages : requested;
}

std::vector<std::uint32_t> keep_pages_from_scores(const KVExecutionTablePool& tables,
                                                 KVExecutionRowHandle row, std::uint32_t live,
                                                 std::uint32_t sink_pages, std::uint32_t window_pages,
                                                 std::uint32_t budget, std::uint32_t cap,
                                                 cudaStream_t stream) {
    const Tensor counts  = tables.page_score_row(row);
    const Tensor recents = tables.page_score_recent_row(row);
    const Tensor peaks   = tables.page_score_peak_row(row);
    const std::uint32_t pages = static_cast<std::uint32_t>(counts.ne[0]);
    const std::uint32_t rows  = live < pages ? live : pages;
    if (rows == 0) { return {}; }
    std::vector<std::uint32_t> host_counts(static_cast<std::size_t>(rows), 0u);
    std::vector<std::uint32_t> host_recents(static_cast<std::size_t>(rows), 0u);
    std::vector<std::uint32_t> host_peaks(static_cast<std::size_t>(rows), 0u);
    const std::size_t bytes = static_cast<std::size_t>(rows) * sizeof(std::uint32_t);
    // port (sm_89): three independent slices on one stream, so one batch submission replaces three.
    // The caller synchronises immediately after, which is what makes the batch the right shape: the
    // whole cost here is submission, not transfer.
    {
        void* const       destinations[3] = {host_counts.data(), host_recents.data(),
                                             host_peaks.data()};
        const void* const sources[3]      = {counts.data, recents.data, peaks.data};
        const std::size_t sizes[3]        = {bytes, bytes, bytes};
        cudaMemcpyAttributes attributes{};
        attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
        attributes.flags          = 0;
        std::size_t attribute_index = 0;
        const auto submitted        = cudaMemcpyBatchAsync(destinations, sources, sizes, 3,
                                                           &attributes, &attribute_index, 1, stream);
        if (submitted != cudaSuccess) {
            core::copy_bytes(host_counts.data(), counts.data, bytes,
                             core::TransferDirection::DeviceToHost, stream);
            core::copy_bytes(host_recents.data(), recents.data, bytes,
                             core::TransferDirection::DeviceToHost, stream);
            core::copy_bytes(host_peaks.data(), peaks.data, bytes,
                             core::TransferDirection::DeviceToHost, stream);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Candidates: pages the window does not already keep. Sink pages and the newest `window_pages`
    // are addressed regardless, so selecting them would spend the budget for nothing.
    // Only a call that actually recorded counts may move the keep set. A decode-shaped call writes the
    // PEAK plane instead, leaving both count planes zero; without this guard the contrast ties at 0
    // everywhere and the top-k degenerates to the first candidates in index order, silently evicting
    // whatever the last prefill chose (appendix DQ). The guard deliberately stays on the COUNT planes
    // even though the peak plane is now one of the ranked lists, so a decode step still cannot move the
    // keep set - and at the informative call the peak plane holds exactly that call's peaks.
    const std::uint64_t recorded = std::accumulate(host_counts.begin(), host_counts.end(),
                                                   std::uint64_t{0});
    if (recorded == 0) { return {}; }

    const std::uint32_t last_kept = rows > window_pages ? rows - window_pages : 0;
    std::vector<std::uint32_t> candidates;
    for (std::uint32_t page = sink_pages; page < last_kept; ++page) { candidates.push_back(page); }
    if (candidates.empty()) { return {}; }

    const double span = static_cast<double>(score_row_span());
    const double denom = static_cast<double>(kNominalScoreRows) > span
                             ? static_cast<double>(kNominalScoreRows) - span
                             : 1.0;
    // Three ranked lists, each of a different nature. The contrast rewards pages the query rows prefer,
    // the every-row count rewards pages the body returns to, and the peak rewards pages some row attended
    // to hardest - it is an order-preserving encoding, so its raw code already ranks correctly and needs
    // no decode. The default is the union of all three (appendix DV.4(b), DT.1).
    const auto top_k = [&](int kind) {
        std::vector<std::pair<double, std::uint32_t>> scored;
        scored.reserve(candidates.size());
        for (std::uint32_t page : candidates) {
            const double all    = static_cast<double>(host_counts[page]);
            const double recent = static_cast<double>(host_recents[page]);
            const double value  = kind == 0 ? recent / span - (all - recent) / denom
                                : kind == 1 ? all
                                            : static_cast<double>(host_peaks[page]);
            scored.emplace_back(value, page);
        }
        // Ties are broken by the evidence itself (the every-row count), not by page index: an
        // all-zero page must not outrank one that was actually recorded (appendix DQ).
        std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
            if (a.first != b.first) { return a.first > b.first; }
            return a.second > b.second;
        });
        return scored;
    };
    const KeepMode mode = keep_mode();
    std::vector<std::uint32_t> keep;
    for (const int kind : {0, 1, 2}) {
        // `duo` is the union as it stood before the peak list existed - the control that isolates it.
        if (mode == KeepMode::Duo && kind == 2) { continue; }
        if (mode == KeepMode::Contrast && kind != 0) { break; }
        if (mode == KeepMode::Raw && kind != 1) { continue; }
        if (mode == KeepMode::Peak && kind != 2) { continue; }
        const auto scored = top_k(kind);
        for (std::uint32_t i = 0; i < budget && i < scored.size(); ++i) {
            const std::uint32_t page = scored[i].second;
            if (std::find(keep.begin(), keep.end(), page) == keep.end()) { keep.push_back(page); }
        }
    }
    // P6 2c (appendix EX): a ratio caller wants an exact page count, so trim the union to it. The order
    // is the existing list priority (contrast, then every-row, then peak), so the query-aware list spends
    // the budget first. cap == 0 leaves the union exactly as the page-count path always produced it.
    if (cap != 0 && keep.size() > cap) { keep.resize(cap); }
    return keep;
}

// N3/DE: the run-level "did this row ever record anything" flag is keyed on the plane's own block -
// the slice handed out per execution row is stable and distinct - rather than on the process, so a
// batched run cannot let one row's traffic make a silent row look healthy. Each plane keeps its own
// flag: after the split a plane carries exactly one unit, so "this plane recorded something" and
// "this statistic was measured" are the same statement (appendix C4/CW).
bool record_plane_activity(const void* plane, std::uint32_t maximum) {
    static std::map<const void*, bool> recorded;
    bool& flag = recorded[plane];
    flag       = flag || maximum != 0u;
    return flag;
}

// appendix IK: an explicit install that overrides the environment, so a caller that only learns
// the geometry after the plan is resolved (the startup contract decision) can still reach the
// per-round application. Unset means "use the environment", which is what every recorded
// experiment used.
std::optional<KvEvictionWindow>& installed_eviction_window() {
    static std::optional<KvEvictionWindow> window;
    return window;
}

const KvEvictionWindow& kv_eviction_window() {
    if (const auto& installed = installed_eviction_window(); installed) { return *installed; }
    static const KvEvictionWindow window = [] {
        KvEvictionWindow result;
        // The preset only fills what the caller left unset; explicit variables win below.
        if (const auto preset = kv_eviction_preset()) {
            result.sink_pages   = preset->sink_pages;
            result.window_pages = preset->window_pages;
        }
        if (const char* sink = std::getenv("NINFER_KV_EVICT_SINK")) {
            result.sink_pages = static_cast<std::uint32_t>(std::strtoul(sink, nullptr, 10));
        }
        if (const char* span = std::getenv("NINFER_KV_EVICT_WINDOW")) {
            result.window_pages = static_cast<std::uint32_t>(std::strtoul(span, nullptr, 10));
        }
        if (const char* path = std::getenv("NINFER_KV_EVICT_SCORES")) {
            // Query-aware selection: pick the pages the scorer ranks highest. A later kernel hook
            // only has to fill this file so the decision path is already exercised.
            std::ifstream scores(path);
            std::vector<std::pair<double, std::uint32_t>> ranked;
            std::uint32_t page  = 0;
            double score        = 0.0;
            while (scores >> page >> score) { ranked.emplace_back(score, page); }
            std::sort(ranked.begin(), ranked.end(), [](const auto& lhs, const auto& rhs) {
                return lhs.first != rhs.first ? lhs.first > rhs.first : lhs.second < rhs.second;
            });
            std::uint32_t top = 4;
            if (const char* count = std::getenv("NINFER_KV_EVICT_KEEP_TOP")) {
                top = static_cast<std::uint32_t>(std::strtoul(count, nullptr, 10));
            }
            for (std::uint32_t index = 0; index < top && index < ranked.size(); ++index) {
                if (ranked[index].first > 0.0) { result.keep_pages.push_back(ranked[index].second); }
            }
        }
        if (const char* list = std::getenv("NINFER_KV_EVICT_KEEP_PAGES")) {
            const char* cursor = list;
            while (*cursor != '\0') {
                char* end = nullptr;
                const unsigned long value = std::strtoul(cursor, &end, 10);
                if (end == cursor) { break; }
                result.keep_pages.push_back(static_cast<std::uint32_t>(value));
                cursor = *end == ',' ? end + 1 : end;
            }
        }
        return result;
    }();
    return window;
}

// The causal attention kernels that honour the absent-page contract: BF16, FP8 and K8V4 plus the
// INT8-routed INT8-G64 and rank-compressed family. NVFP4 is Blackwell-only and still addresses the
// raw block-table entry, so the window is silently skipped for it.
bool kv_absent_pages_supported(KvCacheStorage storage) noexcept {
    switch (storage) {
    case KvCacheStorage::BFloat16:
    case KvCacheStorage::Int8Group64:
    case KvCacheStorage::Fp8E4M3Row256:
    case KvCacheStorage::Fp8KeyNvfp4Value:
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
    case KvCacheStorage::RotatedInt4KeyInt4ValueGroup64:
    case KvCacheStorage::RK4V4E8:
    case KvCacheStorage::RK2V4E8:
    case KvCacheStorage::RK4V2E8:
        return true;
    default:
        return false;
    }
}

std::uint32_t kv_live_pages(std::uint32_t tokens) noexcept {
    const auto page = static_cast<std::uint32_t>(kPagedKVPageSize);
    return (tokens + page - 1) / page;
}

} // namespace

SequenceHandle ProgramImpl::sequence_handle(std::uint32_t lane) const noexcept {
    return ContractAccess::make_sequence(this, runtime::LaneId{lane}, lane_epochs[lane]);
}

bool ProgramImpl::valid_sequence(SequenceHandle handle) const noexcept {
    const auto lane = ContractAccess::lane(handle).value;
    return ContractAccess::owner(handle) == this && lane < max_concurrency &&
           ContractAccess::epoch(handle) == lane_epochs[lane] &&
           requests[lane].lifecycle != Lifecycle::Empty;
}

bool ProgramImpl::recovery_pending(SequenceHandle handle) const noexcept {
    return valid_sequence(handle) &&
           requests[ContractAccess::lane(handle).value].recovery.has_value();
}

bool ProgramImpl::valid_pending(const PendingBatch& pending) const noexcept {
    if (!pending_transaction_ || ContractAccess::owner(pending) != this ||
        ContractAccess::transaction(pending) != pending_transaction_->id ||
        pending.row_count() != pending_transaction_->size) {
        return false;
    }
    const auto rows = ContractAccess::rows(pending);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (!valid_sequence(rows[i]) ||
            ContractAccess::lane(rows[i]).value != pending_transaction_->lanes[i] ||
            ContractAccess::epoch(rows[i]) != pending_transaction_->epochs[i]) {
            return false;
        }
    }
    return true;
}

void ProgramImpl::invalidate_lane(std::uint32_t lane) noexcept {
    if (++lane_epochs[lane] == 0) { ++lane_epochs[lane]; }
}

SequenceState& ProgramImpl::active_sequence(std::uint32_t lane) {
    if (lane >= max_concurrency || requests[lane].lifecycle == Lifecycle::Empty) {
        throw std::logic_error("execution lane is empty");
    }
    return sequences[lane];
}

const SequenceState& ProgramImpl::active_sequence(std::uint32_t lane) const {
    if (lane >= max_concurrency || requests[lane].lifecycle == Lifecycle::Empty) {
        throw std::logic_error("execution lane is empty");
    }
    return sequences[lane];
}

StateImageSelectors ProgramImpl::state_selectors(const SequenceState& sequence) const {
    return {state_store->physical_slot(sequence.state.read),
            state_store->physical_slot(sequence.state.write)};
}

void ProgramImpl::refresh_state_views(SequenceState& sequence) {
    sequence.tail_hidden =
        state_images->continuation_hidden_slot(state_store->physical_slot(sequence.state.read));
}

void ProgramImpl::settle_state_fork(SequenceState& sequence) {
    if (!sequence.state.fork_pending) { return; }
    state_store->commit_fork(sequence.state.read, sequence.state.write);
    sequence.state = {.read = sequence.state.write, .write = sequence.state.write};
    refresh_state_views(sequence);
}

void ProgramImpl::release_sequence_state(SequenceState& sequence) noexcept {
    try {
        if (sequence.state.fork_pending) {
            state_store->abort_fork(sequence.state.read, sequence.state.write);
        }
        if (state_store->valid(sequence.state.write) &&
            !state_store->release(sequence.state.write)) {
            std::terminate();
        }
        sequence.state       = {};
        sequence.tail_hidden = {};
    } catch (...) { std::terminate(); }
}

void ProgramImpl::release_sequence_kv(SequenceState& sequence) noexcept {
    if (!sequence.kv) { return; }
    // appendix FF (M3 3.2c): the execution rows must be read BEFORE deactivation drops them, and the Host
    // extents this sequence's demotion created must go back with it - otherwise the generation path
    // keeps those Host bytes for the life of the process (the scoring path already returns them).
    const std::int32_t text_row = text_kv_addresses->bound_row(sequence.kv->text);
    const std::int32_t backend_row = sequence.kv->backend
                                         ? backend_kv_addresses->bound_row(*sequence.kv->backend)
                                         : -1;
    if (sequence.kv->backend && backend_kv_addresses->active(*sequence.kv->backend)) {
        backend_kv_addresses->deactivate(*sequence.kv->backend);
    }
    if (text_kv_addresses->active(sequence.kv->text)) {
        text_kv_addresses->deactivate(sequence.kv->text);
    }
    release_live_host_extents(text_row);
    release_live_host_extents(backend_row);
    refresh_history_requirements(sequence.kv, true);
    sequence.kv.reset();
}

void ProgramImpl::clear_lane(SequenceState& sequence, RequestControl& request) noexcept {
    // The caller settles real GPU readers before reaching this destruction boundary.
    const auto lane = sequence.lane;
    request.prefill.reset();
    release_sequence_kv(sequence);
    release_sequence_state(sequence);
    sequence      = {};
    sequence.lane = lane;
    request       = {};
    invalidate_lane(lane);
    // The lane's n-gram history dies with the sequence that filled it: the next request to own this
    // lane starts from an all-+0 nine-column pool and an EOS hash tail, which is what a freshly
    // built pool holds. A window a failed pass left open is closed first, without committing it.
    if (ple) {
        try {
            ple->abort_window(device.execution_view());
            ple->reset_lane(static_cast<std::int32_t>(lane), device.execution_view());
        } catch (...) {}
    }
}

void ProgramImpl::clear_execution_failure_lanes(std::span<const std::uint32_t> lanes) noexcept {
    for (const auto lane : lanes) {
        if (lane < max_concurrency) { clear_lane(sequences[lane], requests[lane]); }
    }
}

PagedKVCache* ProgramImpl::backend_kv_cache() noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return decoder->mtp_cache(); }
    return dflash && dflash->full ? &*dflash->full : nullptr;
}

const PagedKVCache* ProgramImpl::backend_kv_cache() const noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return decoder->mtp_cache(); }
    return dflash && dflash->full ? &*dflash->full : nullptr;
}

std::uint32_t ProgramImpl::backend_kv_valid(const SequenceState& state) const noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return state.mtp_kv_valid; }
    return speculative_backend == SpeculativeBackend::DFlash ? state.dflash_context_frontier : 0;
}

// appendix GF.6: accumulate the host time the eviction decision costs, on destruction, so every
// early return is covered. Disabled unless the timing switch is set - read once, like every other
// switch on this path, so the default path pays neither a getenv nor a clock read.
namespace {
bool kv_host_timing_enabled() {
    static const bool enabled = std::getenv("NINFER_KV_EVICT_HOST_TIMING") != nullptr;
    return enabled;
}

struct HostEvictionTimer {
    std::chrono::steady_clock::time_point begin;
    bool enabled = false;
    explicit HostEvictionTimer(bool on) : enabled(on) {
        if (enabled) { begin = std::chrono::steady_clock::now(); }
    }
    ~HostEvictionTimer() {
        if (!enabled) { return; }
        const auto end = std::chrono::steady_clock::now();
        kv_tier_stats().host_eviction_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
        kv_tier_stats().host_eviction_calls += 1;
    }
};

// appendix GJ: the whole per-lane KV management call, so the append path is included this time.
struct HostKvMgmtTimer {
    std::chrono::steady_clock::time_point begin;
    bool enabled = false;
    explicit HostKvMgmtTimer(bool on) : enabled(on) {
        if (enabled) { begin = std::chrono::steady_clock::now(); }
    }
    ~HostKvMgmtTimer() {
        if (!enabled) { return; }
        const auto end = std::chrono::steady_clock::now();
        kv_tier_stats().host_kv_mgmt_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
        kv_tier_stats().host_kv_mgmt_calls += 1;
    }
};
} // namespace

// port (sm_89): the Device page pool's LIVE edge. The control plane's KV dial can shrink the pool at
// runtime, so both the refusal below and the Runtime's pre-flight read this and not the plan.
std::uint32_t ProgramImpl::residency_device_page_capacity() const noexcept {
    return text_kv_pages != nullptr ? text_kv_pages->physical_pool().capacity_pages()
                                    : kv_resident_page_groups;
}

// port (sm_89, V10-79): is the Device arena smaller than the logical page space? Under a residency
// plan the two plan fields say so; under a DECLARED arena (NINFER_KV_EVICT_CAPACITY) they are equal
// while the pool is not, so the live edge is the authority. Every branch that used to compare the
// plan fields asks this instead - on a fully resident run the two answers agree by construction,
// which is why no frozen gate moves.
bool ProgramImpl::kv_arena_tiered() const noexcept {
    return residency_device_page_capacity() < kv_logical_page_groups;
}

// P8: is the leg actually able to run here? The request (default on) is only half the answer: the
// leg copies a page back from a Host tier, so without one there is nothing to copy from and the
// step path has to stay the pre-V10-79 one. Every caller asks this instead of the environment, so
// the switch cannot promise a reader the pool does not have.
bool ProgramImpl::page_in_active() const noexcept {
    return kv_page_in_requested() && host_kv_extents != nullptr;
}

// port (sm_89, V10-80): which space's page-in counters a helper is reporting into. The two spaces
// are told apart exactly as the page store is chosen in the helpers, which is by the address store
// they belong to, so a helper needs no extra argument.
ProgramImpl::PageInCounters&
ProgramImpl::page_in_counters(KVAddressSpaceStore& addresses) noexcept {
    return last_page_in_[&addresses == text_kv_addresses.get() ? 0 : 1];
}

std::uint32_t ProgramImpl::kv_residency_token_limit() const noexcept {
    // No residency contract: the Device pool holds every logical page, so there is no bound to
    // report (0 is the "unbounded" answer the Runtime tests for).
    if (!kv_arena_tiered()) { return 0; }
    return residency_device_page_capacity() * static_cast<std::uint32_t>(kPagedKVPageSize);
}

std::uint32_t ProgramImpl::sequence_execution_frontier(SequenceHandle handle) const noexcept {
    if (!valid_sequence(handle)) { return 0; }
    return sequences[ContractAccess::lane(handle).value].execution_frontier;
}

std::uint32_t ProgramImpl::kv_residency_step_growth() const noexcept {
    // The most a single Decode step can add to the main frontier: its own token, plus the draft
    // window when a speculative backend appends ahead of the committed frontier. Prefill and Replay
    // never grow past the prompt, which the Runtime already bounded against the same limit when it
    // admitted the request.
    return 1U + (backend_kv_cache() != nullptr ? draft_window : 0U);
}

// port (sm_89): one lane's share of the shared Device pool, in the units the Runtime compares.
// Each lane's address owns whole pages, so the demand is rounded up HERE, per lane: two lanes at 63
// and 1 tokens own two pages, not ceil(64 / 64) = 1 - summing tokens first and dividing once
// under-counts the pool exactly at the edge.
namespace {
std::uint32_t page_rounded_tokens(std::uint32_t tokens) noexcept {
    const auto page = static_cast<std::uint32_t>(kPagedKVPageSize);
    if (tokens == 0) { return 0U; }
    const std::uint32_t pages = 1U + (tokens - 1U) / page;
    return pages * page;
}
} // namespace

namespace {
// port (sm_89, V10-84): the price of a lane's STEP, not of its history.
//
// With the page-in leg on, a step does not read its history: `stage_step_kv_pages()` asks
// "is this page addressed", and that predicate is page < sink || page >= end - eff_window ||
// page in keep - the sink pages, the newest `eff_window <= window_pages` pages, and the selector's
// stored keep set, exactly the three sources `apply_kv_eviction_window()` built the row's mask
// from. What the shared Device pool has to hold per lane is THAT set plus the pages the lane's own
// unit is about to write. Pricing the history instead is what refused a four-lane batch whose
// windows fit a small arena (measured: 4 x 2000 tokens against a 120-page arena ->
// `aggregate_tokens=8192 resident_tokens=7680`, while the windows those lanes actually read are
// 4 x (1 sink + 8 window) pages).
//
// 0 means "the window does not describe this step, price the history as before":
//   * no window installed - every mapped page of the history is addressed;
//   * NINFER_KV_EVICT_KEEP_RATIO - the keep set is a FRACTION of `live`, so it has no constant
//   bound;
//   * NINFER_KV_EVICT_DEFER with a selector - a prefill-shaped call leaves the block table whole on
//     purpose, so that row reads its whole history until the prompt is done;
//   * a KV storage whose kernels do not honour the absent-page contract (the mask is skipped).
std::uint32_t window_read_cap_pages() {
    const KvEvictionWindow& window = kv_eviction_window();
    if (!window.enabled()) { return 0; }
    // Presence, not value: `keep_from_scores_ratio()` validates and can throw, and this answer is
    // read from noexcept queries. Any configured ratio means the keep set is a fraction of `live`,
    // which is exactly the case with no constant bound (the validation reports a bad value on its
    // own path).
    if (std::getenv("NINFER_KV_EVICT_KEEP_RATIO") != nullptr) { return 0; }
    const std::uint32_t budget = keep_from_scores_budget();
    if (budget != 0 && defer_eviction_enabled()) { return 0; }
    std::uint32_t pages = window.sink_pages + window.window_pages +
                          static_cast<std::uint32_t>(window.keep_pages.size());
    if (budget != 0) {
        // `keep_pages_from_scores()` unions up to `budget` pages from each list its mode ranks (all
        // three by default), so that is the bound the selector's contribution can reach.
        const KeepMode mode       = keep_mode();
        const std::uint32_t lists = mode == KeepMode::Union ? 3U : mode == KeepMode::Duo ? 2U : 1U;
        pages += budget * lists;
    }
    return pages;
}
} // namespace

// port (sm_89, V10-84): is read-set pricing in force? It needs the leg (a page the Device does not
// hold must have a reader), a tiered arena (a fully resident pool has no edge to price against), a
// window (which is what bounds the addressed set) and a storage whose kernels honour absent pages.
std::uint32_t ProgramImpl::read_set_fixed_pages() const noexcept {
    if (!page_in_active() || !kv_arena_tiered()) { return 0; }
    if (decoder == nullptr || !kv_absent_pages_supported(decoder->text_kv.storage())) { return 0; }
    // P8: the residency plan's own contract installs a window that keeps EVERY logical page (sink 0,
    // span == the logical capacity - model_instance.cpp's kv_host_residency_active arm), because such
    // a plan addresses its whole history. Nothing is masked there, so the read set is the history and
    // pricing it as if it were a window would price the whole logical capacity against a smaller
    // pool - which refuses every request, a 53-token warmup included. Report "not in force" and let
    // the whole-history price do the admitting, exactly as the leg-off arm did.
    const KvEvictionWindow& window = kv_eviction_window();
    if (window.sink_pages == 0 && window.keep_pages.empty() &&
        window.window_pages >= kv_logical_page_groups) {
        return 0;
    }
    // A/B switch, default on: with NINFER_KV_READ_PRICING=0 the same binary prices the whole
    // history the way every build before this one did, so "the read set is what admits the batch"
    // is measured against its own control instead of inferred (NINFER_KV_BORROW's shape).
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_KV_READ_PRICING");
        return value == nullptr || *value == '\0' || std::strtol(value, nullptr, 10) != 0;
    }();
    if (!enabled) { return 0; }
    return window_read_cap_pages();
}

// The pages one lane's next unit can add on top of the read set: the unit's own frontier growth.
// `next_unit()` is the authority for what that unit is, so the same three shapes are used here -
// a prefill/replay chunk (at most `prefill_chunk` tokens), or one Decode step (its token plus the
// draft window when a speculative backend runs). The growth is measured from the prefix the row has
// already MAPPED, and during prefill/replay that is the cursor, not the committed frontier (the
// frontier is published when the prompt completes, so it trails the chunks the mapping already
// covers); measuring from the frontier would charge the whole chunk again on every step.
std::uint32_t ProgramImpl::residency_unit_growth_pages(std::uint32_t lane) const noexcept {
    const auto& request          = requests[lane];
    const std::uint32_t frontier = sequences[lane].execution_frontier;
    std::uint32_t base           = frontier;
    std::uint32_t horizon        = frontier + kv_residency_step_growth();
    if (request.prefill) {
        base    = std::max(request.prefill->cursor, frontier);
        horizon = std::min(request.prefill->prompt_tokens, base + prefill_chunk);
    } else if (request.replay) {
        base    = std::max(request.replay_cursor, frontier);
        horizon = std::min(request.replay_target, base + prefill_chunk);
    }
    const std::uint32_t pages = kv_pages_for_frontier(horizon);
    const std::uint32_t held  = kv_pages_for_frontier(base);
    return pages > held ? pages - held : 0;
}

// The pages a lane's row holds on top of its read set because the MAPPING runs ahead of the
// eviction mask: the mask is applied against the COMMITTED frontier, so every page the mapping
// already covers past that frontier is still resident when the next unit reserves. Asking the row
// is exact and costs two reads - measured, this is what the pool account showed: 4 x (1 + 8 + 2)
// through the steady prefill of a 4 x 6000-token batch at chunk 128, and 4 x (1 + 8 + 5) at its
// prefill -> decode transition (two chunks plus one page for a unit that straddles a page
// boundary). A lane that is not live yet has nothing mapped, so its admission price carries the
// growth alone.
std::uint32_t ProgramImpl::residency_pending_mapping_pages(std::uint32_t lane) const noexcept {
    if (lane >= max_concurrency || !sequences[lane].kv) { return 0; }
    const KVAddressSpaceHandle address = sequences[lane].kv->text;
    const std::uint32_t mapped         = text_kv_addresses->mapped_pages(address);
    const std::uint32_t committed = kv_live_pages(text_kv_addresses->committed_frontier(address));
    return mapped > committed ? mapped - committed : 0;
}

std::uint32_t ProgramImpl::read_set_lane_pages(std::uint32_t lane) const noexcept {
    const std::uint32_t fixed = read_set_fixed_pages();
    if (fixed == 0 || lane >= max_concurrency) { return 0; }
    if (requests[lane].lifecycle == Lifecycle::Empty || !sequences[lane].kv) { return 0; }
    return fixed + residency_pending_mapping_pages(lane) + residency_unit_growth_pages(lane);
}

std::uint32_t ProgramImpl::read_set_request_pages(std::uint32_t prompt_tokens) const noexcept {
    const std::uint32_t fixed = read_set_fixed_pages();
    if (fixed == 0) { return 0; }
    // A request that is not live yet has mapped nothing, so what it needs is the read set plus the
    // first prefill chunk it is about to write (`min(prompt, prefill_chunk)`).
    return fixed + kv_pages_for_frontier(std::min(prompt_tokens, prefill_chunk));
}

std::uint32_t ProgramImpl::read_set_lane_tokens(std::uint32_t lane) const noexcept {
    return read_set_lane_pages(lane) * static_cast<std::uint32_t>(kPagedKVPageSize);
}

std::uint32_t ProgramImpl::read_set_request_tokens(std::uint32_t prompt_tokens) const noexcept {
    return read_set_request_pages(prompt_tokens) * static_cast<std::uint32_t>(kPagedKVPageSize);
}

std::uint32_t ProgramImpl::residency_lane_tokens(std::uint32_t lane,
                                                 std::uint32_t growth) const noexcept {
    if (lane >= max_concurrency) { return 0U; }
    const auto& request = requests[lane];
    const auto& state   = sequences[lane];
    if (request.lifecycle == Lifecycle::Empty || !state.kv) { return 0U; }
    // A lane that is still prefilling (or replaying) has to reach its own prompt, and that prompt
    // was priced against this pool when it was admitted. Counting the cursor instead of the
    // destination is what made a second lane look affordable and then collide later.
    std::uint32_t demand = state.execution_frontier;
    if (request.prefill) { demand = std::max(demand, request.prefill->prompt_tokens); }
    if (request.replay) { demand = std::max(demand, request.replay->prompt_tokens); }
    if (demand > std::numeric_limits<std::uint32_t>::max() - growth) { return demand; }
    return page_rounded_tokens(demand + growth);
}

Program::KvResidencyBudget ProgramImpl::kv_residency_budget() const noexcept {
    Program::KvResidencyBudget budget{};
    // No residency contract: every logical page is resident, so there is no shared bound to report.
    if (!kv_arena_tiered()) { return budget; }
    budget.capacity_tokens =
        residency_device_page_capacity() * static_cast<std::uint32_t>(kPagedKVPageSize);
    budget.capacity_pages = residency_device_page_capacity();
    // port (sm_89, V10-84): with the page-in leg on, a lane's share is what its step ADDRESSES, so
    // the same field carries the read-set price (see read_set_lane_pages()). The shape of the
    // answer - page-exact per lane, summed, against the same live pool edge - is unchanged, which
    // is what lets the Runtime's admission rule and its refusals keep working without a new
    // contract.
    if (read_set_fixed_pages() != 0) {
        for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
            const std::uint32_t pages = read_set_lane_pages(lane);
            if (pages == 0) { continue; }
            budget.lane_pages[lane] = pages;
            budget.lane_tokens[lane] = pages * static_cast<std::uint32_t>(kPagedKVPageSize);
            budget.live_tokens += budget.lane_tokens[lane];
            budget.live_pages += pages;
            ++budget.lane_count;
        }
        return budget;
    }
    const std::uint32_t growth = kv_residency_step_growth();
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        const std::uint32_t tokens = residency_lane_tokens(lane, growth);
        if (tokens == 0) { continue; }
        budget.lane_tokens[lane] = tokens;
        // `residency_lane_tokens()` already rounded to whole pages, so this division is exact.
        budget.lane_pages[lane] = tokens / static_cast<std::uint32_t>(kPagedKVPageSize);
        budget.live_tokens += tokens;
        budget.live_pages += budget.lane_pages[lane];
        ++budget.lane_count;
    }
    return budget;
}

std::uint32_t ProgramImpl::kv_residency_request_tokens(std::uint32_t prompt_tokens) const noexcept {
    if (!kv_arena_tiered()) { return 0U; }
    if (const std::uint32_t pages = read_set_request_pages(prompt_tokens); pages != 0) {
        return pages * static_cast<std::uint32_t>(kPagedKVPageSize);
    }
    const std::uint32_t growth = kv_residency_step_growth();
    if (prompt_tokens > std::numeric_limits<std::uint32_t>::max() - growth) { return prompt_tokens; }
    return page_rounded_tokens(prompt_tokens + growth);
}

void ProgramImpl::ensure_sequence_kv_mapped(SequenceState& state, std::uint32_t main,
                                            std::uint32_t backend) {
    const HostKvMgmtTimer kv_mgmt_timer(kv_host_timing_enabled());
    // port (sm_89): a residency Program keeps the Device pool at `kv_resident_page_groups`. A step
    // whose live history is longer than that cannot be served by fetching: the full-attention layers
    // read the whole history every step, so every live page has to be resident at once. Refuse here,
    // with the numbers, rather than letting the append die inside the page pool - and far rather than
    // letting attention read a block-table entry whose Device replica was already given back.
    // This is the LAST line of defence: the Runtime refuses such a request before it grants a unit
    // (see kv_residency_token_limit()), because a refusal thrown out of a step fails every lane.
    if (kv_arena_tiered() && main != 0) {
        const std::uint32_t page_tokens = static_cast<std::uint32_t>(kPagedKVPageSize);
        // port (sm_89, V10-84): under read-set pricing the step does not address its history - it
        // addresses sink + the newest window + the selector's keep set, and the pages its own
        // unit is about to write. That priced set is the bound this step has to fit, and
        // admission has already promised the sum over live lanes fits the pool.
        const std::uint32_t read_priced = read_set_lane_pages(state.lane);
        const std::uint32_t needed =
            read_priced != 0 ? read_priced : 1U + (main - 1U) / page_tokens;
        // The live edge, not the construction number: the control plane's KV dial can shrink the
        // pool at runtime, and a guard that read the plan would then promise room that is gone.
        const std::uint32_t resident_capacity = residency_device_page_capacity();
        if (needed > resident_capacity) {
            const std::string head =
                "KV residency cannot serve this step: it needs " + std::to_string(needed) +
                " resident page group(s) for " + std::to_string(main) +
                " token(s), but the Device "
                "pool holds " +
                std::to_string(resident_capacity) + " of " +
                std::to_string(kv_logical_page_groups) + " logical page group(s) (" +
                std::to_string(host_kv_payload_bytes >> 20) +
                " MiB priced onto the Host tier). ";
            if (read_priced != 0) {
                // The windowed arm: the priced set is the read set, so the pool is short of the
                // pages one step addresses rather than of the history - the remedy is the same
                // shape (a smaller window, fewer lanes, more pages), and the page-in leg cannot
                // help because a step cannot address fewer page groups than it reads.
                throw std::runtime_error(
                    head + "The window this lane runs under addresses (sink + newest window + "
                    "the selector's keep set) and the unit's own growth; that set does not fit the "
                    "pool, so no page could be restored for it. Shorten the window "
                    "(NINFER_KV_EVICT_WINDOW / NINFER_KV_EVICT_KEEP_FROM_SCORES), lower "
                    "max_concurrency, or give the pool more pages (--kv-residency-slack-mib).");
            }
            // V10-79: the two arms say different things because they are different states. With the
            // page-in leg off the pages on the Host tier have no reader at all; with it on they do
            // - but the step still ADDRESSES all of them, because one page group holds every
            // layer's planes and every full-attention layer reads the whole history on every step.
            // So the leg does not lift this bound; it moves the refusal down to the arena's own
            // size.
            throw std::runtime_error(
                head + "The " +
                std::to_string(parameters.model.config().text.full_attention_layers) +
                " full-attention layer(s) read the whole history every step, so the step needs all "
                "of it resident at once; " +
                (page_in_active()
                     ? "the page-in leg can copy a page back from Host, but a page group spans "
                       "every layer's planes, so it cannot make the step address fewer pages. "
                       "Bound the read set with NINFER_KV_EVICT_WINDOW (sink + newest window + the "
                       "selector's keep set is what a step addresses then), or give the pool more "
                       "pages (--kv-residency-slack-mib, or a smaller max_context)."
                     : "it cannot be streamed out of the Host tier (the page-in leg would copy a "
                       "page back from Host; NINFER_KV_PAGE_IN=0 or NINFER_KV_NO_PAGE_IN=1 turned "
                       "it off for this run)."));
        }
    }
    text_kv_addresses->ensure_mapped_to_tokens(state.kv->text, main, device.stream);
    if (state.kv->backend) {
        backend_kv_addresses->ensure_mapped_to_tokens(*state.kv->backend, backend, device.stream);
    }
    // An optional eviction window (default off) marks the pages the attention may skip. It runs
    // after the mapping is extended and before the forward reads the table; it only ever drops
    // pages whose tokens are already committed, so the next append never targets an absent page.
    // Both the Main and the backend (MTP / DFlash) growing caches are windowed.
    apply_kv_eviction_window(*text_kv_addresses, decoder->text_kv, state.kv->text);
    if (state.kv->backend) {
        if (PagedKVCache* cache = backend_kv_cache(); cache != nullptr) {
            apply_kv_eviction_window(*backend_kv_addresses, *cache, *state.kv->backend);
        }
    }
    // port (sm_89, V10-79): the resident-KV leg's step half. It runs AFTER the eviction window
    // (which is what decides which pages a step addresses) and BEFORE the forward reads the row:
    // every addressed page the Device does not hold is copied back from the Host tier, and a page
    // this step does not address is given back to make room. The window is not required - a page
    // can be given back by an idle reclaim, a checkpoint demotion or a dial shrink, and this is
    // what makes it readable again. Two guards keep a default run identical: the switch, and a
    // tiered arena (a fully resident pool never enters, so no frozen gate can move).
    if (page_in_active() && kv_arena_tiered()) {
        stage_step_kv_pages(*text_kv_addresses, decoder->text_kv, state.kv->text);
        // port (sm_89, V10-80): the speculative backend's cache is a second, separately paged space
        // that the same demotion empties and the same draft attention reads (MTP: the whole
        // history; DFlash: the full-attention draft layers). Its pass follows the text one for the
        // reason above - the backend's own window has just run - and its helpers are the same
        // three.
        stage_step_backend_kv_pages(state);
    }
}

// The last informative selection per execution row, keyed on the plane row's data pointer (stable and
// distinct per row, exactly as the N3 flag is keyed). Without it a decode-step decision - which sees no
// counts at all - would replace a good prefill selection with a degenerate one (appendix DQ).
static std::map<const void*, std::vector<std::uint32_t>> g_scored_keep_by_row;

// appendix GL: WHERE the addressed KV pages physically live, read from the very block table the
// decode kernel reads. The release-cost work (GI/GJ/GK) leaves one hypothesis standing - that
// releasing the evicted prefix changes the physical placement of the pages still read every
// round - and this is its cheapest falsifiable test: no profile capture (GK.1 could not make one
// self-consistent) and no pool-policy change (which would need a header edit). The addressed set
// is sink + newest window + keep, identical in both arms by construction, so the arms compare
// exactly the thing the hypothesis is about. The ids are printed in the ORDER THE KERNEL WALKS
// them, so a burst of jumps is visible directly rather than as a summary statistic.
// Read-only, and gated on its own path: a default run computes nothing, so no frozen gate moves.
static std::uint64_t placement_every() {
    const char* text = std::getenv("NINFER_KV_EVICT_PLACEMENT_EVERY");
    if (text == nullptr || *text == '\0') { return 1; }
    const long value = std::strtol(text, nullptr, 10);
    return value > 0 ? static_cast<std::uint64_t>(value) : 1;
}

static void dump_page_placement(PagedKVCache& cache, KVAddressSpaceStore& addresses,
                                KVAddressSpaceHandle address, cudaStream_t stream,
                                std::uint32_t live, std::uint32_t sink_pages,
                                std::uint32_t window_pages,
                                const std::vector<std::uint32_t>& keep_pages) {
    const char* path = std::getenv("NINFER_KV_EVICT_PLACEMENT_DUMP");
    if (path == nullptr || *path == '\0') { return; }
    // Sampling is not cosmetic: the copy below synchronises the stream, so an every-round dump
    // would perturb the very decode rate it is meant to explain. every=1 (the default) dumps all.
    static std::uint64_t placement_calls = 0;
    const std::uint64_t every = placement_every();
    if (every > 1 && (placement_calls++ % every) != 0) { return; }
    const std::uint32_t capacity = cache.execution_tables().logical_page_capacity();
    const std::uint32_t end = std::min(live, capacity);
    if (end == 0) { return; }
    std::vector<std::int32_t> table(static_cast<std::size_t>(end), -1);
    const Tensor row = cache.execution_tables().row(addresses.execution_row(address).handle());
    core::copy_bytes(table.data(), row.data,
                     static_cast<std::size_t>(end) * sizeof(std::int32_t),
                     core::TransferDirection::DeviceToHost, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::uint32_t window_begin = end > window_pages ? end - window_pages : 0;
    std::vector<std::int32_t> ids;
    ids.reserve(static_cast<std::size_t>(sink_pages) + window_pages + keep_pages.size());
    for (std::uint32_t page = 0; page < end; ++page) {
        const bool addressed =
            page < sink_pages || page >= window_begin ||
            std::find(keep_pages.begin(), keep_pages.end(), page) != keep_pages.end();
        if (!addressed || table[page] < 0) { continue; }
        ids.push_back(table[page]);
    }
    std::vector<std::int32_t> sorted = ids;
    std::sort(sorted.begin(), sorted.end());
    std::uint32_t runs = 0;
    std::int32_t max_gap = 0;
    for (std::size_t index = 0; index < sorted.size(); ++index) {
        if (index == 0 || sorted[index] != sorted[index - 1] + 1) { ++runs; }
        if (index != 0) {
            const std::int32_t gap = sorted[index] - sorted[index - 1];
            if (gap > max_gap) { max_gap = gap; }
        }
    }
    const DeviceKVPagePool& pool = cache.page_pool();
    std::ofstream out(path, std::ios::app);
    if (!out) { return; }
    out << "# placement live=" << end << " sink=" << sink_pages << " window=" << window_pages
        << " keep=" << keep_pages.size() << " resident=" << ids.size() << " runs=" << runs
        << " min=" << (sorted.empty() ? -1 : sorted.front())
        << " max=" << (sorted.empty() ? -1 : sorted.back())
        << " span=" << (sorted.empty() ? 0 : sorted.back() - sorted.front())
        << " maxgap=" << max_gap << " pool_allocated=" << pool.allocated_pages()
        << " pool_capacity=" << pool.capacity_pages() << '\n';
    out << "# placement_ids";
    for (const std::int32_t id : ids) { out << ' ' << id; }
    out << '\n';
}

void ProgramImpl::apply_kv_eviction_window(KVAddressSpaceStore& addresses, PagedKVCache& cache,
                                           KVAddressSpaceHandle address) {
    const HostEvictionTimer host_timer(kv_host_timing_enabled());
    const KvEvictionWindow& window = kv_eviction_window();
    const std::uint32_t scored_budget = keep_from_scores_budget();
    const std::uint32_t ratio_percent = keep_from_scores_ratio();
    const bool selector_requested     = scored_budget != 0 || ratio_percent != 0;
    if (!window.enabled() && !selector_requested) { return; }
    if (!kv_absent_pages_supported(cache.storage())) { return; }
    if (selector_requested && window.window_pages == 0) {
        // Without a window there is nothing to evict against, and apply_absent_window() would leave
        // the row untouched - so the knob would silently do nothing. Say so instead (appendix DN).
        throw std::invalid_argument(
            "NINFER_KV_EVICT_KEEP_FROM_SCORES needs NINFER_KV_EVICT_WINDOW to define what is kept"
            " unconditionally; otherwise the row is left untouched and the selector has no effect");
    }
    if (selector_requested && score_row_span() <= 0) {
        // With no newest-rows span the second count plane stays zero, and the contrast degenerates to
        // the negated every-row profile - i.e. it would rank the WORST pages first. Fail loudly.
        throw std::invalid_argument(
            "NINFER_KV_EVICT_KEEP_FROM_SCORES needs NINFER_KV_PAGE_SCORE_ROW_SPAN > 0: without the"
            " newest-rows plane the contrast is not defined and would invert the ranking");
    }
    const std::uint32_t frontier = addresses.committed_frontier(address);
    if (frontier == 0) { return; }
    const KVExecutionRowHandle row = addresses.execution_row(address).handle();
    const std::uint32_t live       = kv_live_pages(frontier);
    const std::uint32_t eff_window = effective_window_pages(window.window_pages, live);
    // The ratio supersedes the page-count budget: setting it expresses an intent about the fraction,
    // and the union is capped to the same number so the fraction is actually met (appendix EX).
    const std::uint32_t ratio_cap =
        keep_from_scores_target(live, window.sink_pages, eff_window);
    const std::uint32_t effective_budget = ratio_cap != 0 ? ratio_cap : scored_budget;
    std::vector<std::uint32_t> keep       = window.keep_pages;
    bool defer_apply                      = false;
    if (effective_budget != 0) {
        const void* const key = cache.execution_tables().page_score_row(row).data;
        const auto selected = keep_pages_from_scores(cache.execution_tables(), row, live,
                                                     window.sink_pages, eff_window,
                                                     effective_budget, ratio_cap, device.stream);
        if (!selected.empty()) {
            g_scored_keep_by_row[key] = selected;
            // Counts were recorded, so the prompt is still being read: keep the cache whole and let the
            // next decision - whose statistics are therefore still un-sparsified - decide for real.
            if (defer_eviction_enabled()) { defer_apply = true; }
        }
        const std::vector<std::uint32_t>& effective = g_scored_keep_by_row[key];
        keep.insert(keep.end(), effective.begin(), effective.end());
    }
    // appendix FI (M3 3.3): is the contract's invariant ('sink + newest window only move forward,
    // so the kernel never reads a dropped page again') still true once the selector recomputes the
    // keep set every call? Walk the addressed set and count the pages whose Device replica is gone.
    // Gated on the dump: a default run computes nothing, so no frozen gate can move.
    if (const char* tier_path = std::getenv("NINFER_KV_EVICT_POOL_DUMP");
        tier_path != nullptr && *tier_path != '\0') {
        LogicalKVPageStore* const tier_pages =
            &addresses == text_kv_addresses.get() ? text_kv_pages.get() : backend_kv_pages.get();
        last_addressed_absent_      = 0;
        last_addressed_absent_host_ = 0;
        if (tier_pages != nullptr) {
            // Clamp exactly as demote_absent_pages_to_host() and release_absent_device_pages() do:
            // `live` is the COMMITTED live page count, which can exceed what this address has
            // mapped (the defer path and every non-PPL caller reach that state). Walking to `live`
            // reads logical_page() past the mapping, which corrupted the heap in the preemption
            // real test - the PPL path never sees it, which is why the byte gates stayed green.
            const std::uint32_t absent_end = std::min(live, addresses.mapped_pages(address));
            const std::uint32_t absent_begin =
                absent_end > eff_window ? absent_end - eff_window : 0;
            for (std::uint32_t page = 0; page < absent_end; ++page) {
                const bool addressed =
                    page < window.sink_pages || page >= absent_begin ||
                    std::find(keep.begin(), keep.end(), page) != keep.end();
                if (!addressed) { continue; }
                const LogicalKVPageHandle handle = addresses.logical_page(address, page);
                if (tier_pages->device_resident(handle)) { continue; }
                ++last_addressed_absent_;
                if (tier_pages->host_resident(handle)) { ++last_addressed_absent_host_; }
            }
        }
    }
    // port (sm_89, V10-81): remember whether this row's block table still addresses its whole
    // history. A deferred mask leaves the table untouched on purpose, so the window predicate stops
    // describing what this row reads, and a cross-row borrow must not treat one of its pages as spare.
    {
        const void* const key = cache.execution_tables().page_score_row(row).data;
        const auto found = std::find(deferred_mask_rows_.begin(), deferred_mask_rows_.end(), key);
        if (defer_apply && found == deferred_mask_rows_.end()) {
            deferred_mask_rows_.push_back(key);
        } else if (!defer_apply && found != deferred_mask_rows_.end()) {
            deferred_mask_rows_.erase(found);
        }
    }
    if (!defer_apply) {
        // appendix GC (FZ layer 0): the scans' own left edge, computed once so the row write
        // and all three scans clamp the frontier the same way. The frontier is advanced once
        // they have all run.
        const bool frontier_on       = kv_eviction_frontier_enabled() && keep.empty();
        std::uint32_t window_begin   = 0;
        std::uint32_t frontier_start = 0;
        if (frontier_on) {
            const std::uint32_t frontier_end = std::min(live, addresses.mapped_pages(address));
            window_begin   = frontier_end > eff_window ? frontier_end - eff_window : 0;
            frontier_start = std::min(addresses.eviction_frontier(address), window_begin);
            kv_tier_stats().frontier_rounds += 1;
        }
        // appendix GE (O2): the interval write is equivalent to the whole-row rebuild only when
        // the scans' own edge and the rebuild's edge agree, i.e. when the whole committed frontier
        // is mapped. Otherwise the rebuild masks up to live - window, which is past what is mapped,
        // and an interval that stopped at the mapped edge would leave those pages addressed. So it
        // is used only on a frontier round with live == mapped; the rebuild runs everywhere else.
        const bool interval_ok =
            kv_eviction_interval_enabled() && frontier_on && window_begin != 0 &&
            live == addresses.mapped_pages(address);
        if (interval_ok) {
            const std::uint32_t interval_begin = std::max(window.sink_pages, frontier_start);
            cache.execution_tables().apply_absent_interval(row, interval_begin, window_begin,
                                                           device.stream);
        } else {
            cache.execution_tables().apply_absent_window(row, live, window.sink_pages, eff_window,
                                                         keep, device.stream);
        }
        // appendix FF (M3 3.2b) vs appendix EY (P3a): the same masked pages, two different fates.
        // Demote keeps their bytes on Host, so a later query can still reach them; release throws them
        // away. Both return the Device page, so the capacity effect is identical - only reversibility
        // differs. Demote wins when both switches are set, because release would destroy what it saved.
        std::uint32_t demoted = 0;
        // appendix GS (item 1): prefer KEEPING the bytes. When the caller named no fate for the
        // masked pages, DEMOTE instead of leaving them resident: appendices GP/GQ measured 4-5x the
        // aggregate decode throughput for window-sized residency over one full-length lane, demotion
        // costs one spill (5,784 pages, fetched back 0 times) as cheap as discarding, and the bytes
        // survive so a later, better decision can still fetch them. Output-neutral by construction -
        // what attention reads is decided by the mask, not by the replica's fate, and the contract's
        // gate is a bit-identical PPL. Inert when the contract is off, so no frozen gate can move.
        // appendix GU: only an UNSET switch falls through to the default; an explicit 0 keeps the
        // masked pages resident, which is the pre-item-1 behaviour.
        const DemoteRequest demote_request = kv_host_demote_request();
        const bool demote_unasked = demote_request == DemoteRequest::Unset &&
                                    !kv_eviction_release_enabled() && host_context_arena != nullptr;
        if (demote_request == DemoteRequest::On || demote_unasked) {
            demoted = demote_absent_pages_to_host(addresses, cache, address, live, window.sink_pages,
                                                  eff_window, keep, frontier_start);
        }
        if (kv_eviction_release_enabled() && demoted == 0) {
            // Also the fallback when the demotion demoted nothing - most importantly when no Host
            // arena is configured. Otherwise the demote switch would win the fork, do nothing, and
            // leave every masked page materialized with nothing saying so.
            // Deferred rounds never reach here, so the fate always follows the mask it came from.
            (void)addresses.release_absent_device_pages(address, live, window.sink_pages, eff_window,
                                                        keep, frontier_start);
        }
        // appendix FL (M3 3.3, option (c)): retire the pages whose Device replica the demotion or
        // the release just gave back. The shadow is what publish() wrote and what
        // apply_absent_window() reads, so leaving it naming a returned page is exactly how a
        // released page could be re-addressed and read from a page the pool already handed on.
        {
            LogicalKVPageStore* const retired_pages =
                &addresses == text_kv_addresses.get() ? text_kv_pages.get() : backend_kv_pages.get();
            if (retired_pages != nullptr) {
                const std::uint32_t retire_end = std::min(live, addresses.mapped_pages(address));
                const std::uint32_t retire_begin =
                    retire_end > eff_window ? retire_end - eff_window : 0;
                std::vector<std::uint32_t> retired;
                for (std::uint32_t page =
                         std::max(std::min(window.sink_pages, retire_begin),
                                  std::min(frontier_start, retire_begin));
                     page < retire_begin; ++page) {
                    if (std::find(keep.begin(), keep.end(), page) != keep.end()) { continue; }
                    if (retired_pages->device_resident(addresses.logical_page(address, page))) {
                        continue;
                    }
                    retired.push_back(page);
                }
                if (!retired.empty()) {
                    cache.execution_tables().mark_absent(row, retired, device.stream);
                }
            }
        }
        // appendix GC (FZ layer 0): everything in [sink, window_begin) has now been given back by
        // exactly one of the three paths above, so the next round may start there. Only advanced on
        // a frontier round, so a selector round leaves the bound untouched and the full walk runs.
        if (frontier_on && window_begin != 0) {
            addresses.advance_eviction_frontier(address, window_begin);
            kv_tier_stats().frontier_advanced_to = window_begin;
        }
        // appendix FJ (M3 3.3): the pages the new mask drops have now given their replicas back,
        // which is exactly the room this needs - so the fetch runs after the demotion/release, and
        // a page dropped this round can never be demanded back this round. Both directions are on
        // device.stream, so the read that follows is ordered after both.
        (void)restore_addressed_host_pages(addresses, cache, address, live, window.sink_pages,
                                           eff_window, keep);
        // port (sm_89): under residency an addressed page that could not be brought back is not a
        // miss to tolerate - the step is about to read it. Fail with the count instead of running the
        // kernel against a block table entry whose replica is gone.
        if (kv_resident_page_groups < kv_logical_page_groups && last_fetch_shortfall_ != 0) {
            throw std::runtime_error(
                "KV residency could not restore every addressed page before the step: addressed=" +
                std::to_string(last_fetch_candidates_) +
                " restored=" + std::to_string(last_fetched_pages_) +
                " shortfall=" + std::to_string(last_fetch_shortfall_) + " (Device pool " +
                std::to_string(kv_resident_page_groups) + " page group(s) of " +
                std::to_string(kv_logical_page_groups) + " logical, Host payload " +
                std::to_string(host_kv_payload_bytes >> 20) + " MiB)");
        }
        dump_pool_account(cache);
        dump_page_placement(cache, addresses, address, device.stream, live, window.sink_pages,
                            eff_window, keep);
    }
    dump_page_scores(addresses, cache, address, keep);
    // P6 2c (appendix DU): the page scores are per-call whether or not a dump is being written. The
    // reset used to live inside dump_page_scores, which made the statistic - and now the keep set the
    // selector derives from it - depend on whether a diagnostic was enabled. Consuming the planes and
    // clearing them belong to the decision point; the dump only reports. Clearing after the dump keeps
    // the dump-on sequence byte-identical to before.
    if (!kv_page_scores_cumulative_enabled()) {
        cache.execution_tables().reset_page_scores(row, device.stream);
    }
}

// appendix FF (M3 3.2b): move every masked page in [sink, live - window) that the selector did not
// keep onto Host, then drop its Device lease. The Host extent store already attaches the replica and
// unpins the source inside publish(), so this is: collect the pages, prepare, copy, publish, demote.
// Returns the number of pages actually demoted; 0 means "nothing eligible, nothing changed".
// appendix FF (M3 3.2c): release the Host extents recorded for one execution row. release() detaches
// each page's replica and, through it, retires the descriptors that were kept alive for the extent -
// which is why nothing else can do this job.
// appendix FF (M3 3.2c): release the globally OLDEST demoted extent, whatever its owner, and report
// how many pages that gave up. Recency is the agreed rule ("keep the most recent N"); a score-based
// ranking would need the page-index mapping the record is only now starting to carry.
// Kept although it can only ever report 0: release() refuses a live demoted extent, so this can
// only give up extents that are already unreferenced - which release_unreferenced() covers. Left as
// the documented hook for a future contract that allows evicting live demoted pages.
std::uint32_t ProgramImpl::evict_oldest_host_extent() noexcept {
    if (host_kv_extents == nullptr) { return 0; }
    std::int32_t victim_row = -1;
    std::size_t victim_index = 0;
    std::uint64_t oldest     = std::numeric_limits<std::uint64_t>::max();
    for (const auto& [row, extents] : live_host_extents_) {
        for (std::size_t index = 0; index < extents.size(); ++index) {
            if (extents[index].age < oldest) {
                oldest       = extents[index].age;
                victim_row   = row;
                victim_index = index;
            }
        }
    }
    if (victim_row < 0) { return 0; }
    auto& extents               = live_host_extents_[victim_row];
    const LiveHostExtent victim = extents[victim_index];
    const std::uint32_t pages   = victim.page_end > victim.page_begin
                                    ? victim.page_end - victim.page_begin
                                    : 0U;
    if (!host_kv_extents->release(victim.capability)) {
        // The bytes are still occupied, so the record must survive: erasing it would lose the only
        // handle to them and leak exactly what this function exists to avoid. Report no progress so the
        // caller stops instead of pretending room was made.
        return 0;
    }
    extents.erase(extents.begin() + static_cast<std::ptrdiff_t>(victim_index));
    if (extents.empty()) { live_host_extents_.erase(victim_row); }
    return pages;
}

void ProgramImpl::release_live_host_extents(std::int32_t row) noexcept {
    if (host_kv_extents == nullptr) { return; }
    const auto found = live_host_extents_.find(row);
    if (found == live_host_extents_.end()) { return; }
    const std::size_t extent_count = found->second.size();  // before erase: the iterator dies with it
    std::uint32_t released         = 0;
    for (const LiveHostExtent& extent : found->second) {
        if (host_kv_extents->release(extent.capability)) { ++released; }
    }
    live_host_extents_.erase(found);
    if (const char* path = std::getenv("NINFER_KV_EVICT_POOL_DUMP"); path != nullptr && *path != '\0') {
        std::ofstream out(path, std::ios::app);
        if (out) {
            out << "# reclaim row=" << row << " extents=" << extent_count
                << " released=" << released << " host_occupied=" << host_kv_extents->occupied()
                << '\n';
        }
    }
}

std::uint32_t ProgramImpl::demote_absent_pages_to_host(KVAddressSpaceStore& addresses,
                                                       PagedKVCache& cache,
                                                       KVAddressSpaceHandle address,
                                                       std::uint32_t live, std::uint32_t sink_pages,
                                                       std::uint32_t window_pages,
                                                       const std::vector<std::uint32_t>& keep_pages,
                                                       std::uint32_t first_page) {
    last_demote_stage_      = 0;
    last_demote_candidates_ = 0;
    if (host_kv_extents == nullptr) { last_demote_stage_ = 1; return 0; }
    LogicalKVPageStore* pages = &addresses == text_kv_addresses.get() ? text_kv_pages.get()
                                                                     : backend_kv_pages.get();
    if (pages == nullptr) { last_demote_stage_ = 2; return 0; }
    const std::uint32_t end        = std::min(live, addresses.mapped_pages(address));
    const std::uint32_t keep_begin = end > window_pages ? end - window_pages : 0;
    std::vector<LogicalKVPageHandle> membership;
    membership.reserve(keep_begin);
    std::uint32_t candidates_begin = 0;  // logical page span, recorded with the extent (appendix FF)
    std::uint32_t candidates_end   = 0;
    // appendix GC (FZ layer 0): start where the last round stopped, clamped to keep_begin.
    const std::uint32_t demote_start =
        std::max(std::min(sink_pages, keep_begin), std::min(first_page, keep_begin));
    for (std::uint32_t page = demote_start; page < keep_begin; ++page) {
        if (std::find(keep_pages.begin(), keep_pages.end(), page) != keep_pages.end()) { continue; }
        // logical_page() already exists on the address space; the loop bound keeps it in range.
        const LogicalKVPageHandle candidate = addresses.logical_page(address, page);
        // Already on Host (moved by an earlier round, or by a checkpoint): offering it again would make
        // prepare() reject the WHOLE batch, so every later round would do nothing.
        if (pages->host_resident(candidate)) { continue; }
        if (membership.empty()) { candidates_begin = page; }
        candidates_end = page + 1U;
        membership.push_back(candidate);
    }
    last_demote_candidates_ = static_cast<std::uint32_t>(membership.size());
    if (membership.empty()) { last_demote_stage_ = 3; return 0; }
    auto reservation = host_kv_extents->prepare(*pages, membership);
    if (!reservation) {
        // appendix FF (M3 3.2c): the arena is full while the owner is still alive, so this is NOT the
        // "owner is gone" case - do not release by owner. Give back whatever is already unreferenced,
        // then make room by evicting the least valuable (oldest) demoted extents, and only then retry.
        // HostKVExtentReservation is move-only WITHOUT move-assignment, so each retry constructs the
        // optional in place rather than assigning to it.
        // Only already-unreferenced extents can be given back here. An extent whose pages are still
        // address-referenced cannot be released at all (host_kv_store.h:264-265 refuses the demoted
        // state on purpose), so "evict the least valuable live demoted extent" is not implementable
        // without first dropping those pages - which is P3a's lossy release, the fallback below. The
        // pressure valve therefore already exists; what is missing is only PRIORITISATION, so that a
        // checkpoint gives up its room before a live page gives up reversibility.
        (void)host_kv_extents->release_unreferenced();
        if (auto retry = host_kv_extents->prepare(*pages, membership)) {
            reservation.emplace(std::move(*retry));
        }
        if (!reservation) {
            // appendix FF (M3 3.2c): prioritisation. A checkpoint CAN give its Host room back (its
            // extents become unreferenced and release_unreferenced frees them); a live demoted extent
            // cannot be released at all. So give up the least recent checkpoints, keeping the configured
            // number, and try once more - this is the lever, not evicting live pages.
            const std::uint32_t stale = release_stale_checkpoints(host_keep_snapshots_);
            last_stale_checkpoints_ += stale;
            // appendix FH (M3 3.2d): count them where they happen. FE.3 requires the preemption
            // to be reported, so a silent multi-turn regression cannot hide behind a number that
            // is only ever read once.
            kv_tier_stats().preempted_checkpoints += stale;
            if (auto retry = host_kv_extents->prepare(*pages, membership)) {
                reservation.emplace(std::move(*retry));
            }
        }
        if (!reservation) { last_demote_stage_ = 4; return 0; }
    }
    const auto tier_copy =
        cache.page_pool().copy_to_host(host_kv_extents->device_sources(*reservation),
                                       host_kv_extents->writable_view(*reservation),
                                       device.stream);
    const HostKVExtentCapability capability = host_kv_extents->publish(std::move(*reservation));
    if (const std::int32_t row = addresses.bound_row(address); row >= 0) {
        live_host_extents_[row].push_back(
            LiveHostExtent{.capability = capability,
                           .age        = host_extent_age_++,
                           .page_begin = candidates_begin,
                           .page_end   = candidates_end});
    }
    std::uint32_t demoted = 0;
    std::uint32_t block_mask = 0;
    for (const LogicalKVPageHandle page : membership) {
        if (pages->demote_device_replica(page)) {
            ++demoted;
        } else {
            block_mask |= pages->demote_block_mask(page);
        }
    }
    last_demote_block_mask_ = block_mask;
    last_demoted_pages_     = demoted;
    last_demote_stage_      = 5;
    if (demoted == 0 && std::getenv("NINFER_KV_DEMOTE_DEBUG") != nullptr) {
        std::fprintf(stderr,
                     "[demote] nothing demoted: candidates=%zu block_mask=0x%x live=%u "
                     "window=%u sink=%u keep=%zu\n",
                     membership.size(), block_mask, live, window_pages, sink_pages, keep_pages.size());
        std::fflush(stderr);
    }
    // appendix FH (M3 3.2d): the `kv tiers` counters for the Device -> Host direction. Pages are
    // what actually reached Host; the bytes are the batch's payload - the whole reservation was
    // copied, so this is the transfer's real cost, not just the surviving subset's.
    kv_tier_stats().spilled_pages += demoted;
    kv_tier_stats().spilled_bytes += tier_copy.payload_bytes;
    return demoted;
}

// appendix FJ (M3 3.3): fetch back the pages the mask is about to address whose Device replica the
// eviction already gave away. Appendix FI measured the set (with the selector on, 17 of 20 decision
// points address a page with no Device replica); the address came from a block table that still names
// the old, recycled page. Restoring the replica before the read makes the row name the right page
// again, and since the bytes are the same bytes, no number may move - which is the gate.
std::uint32_t ProgramImpl::restore_addressed_host_pages(KVAddressSpaceStore& addresses,
                                                        PagedKVCache& cache,
                                                        KVAddressSpaceHandle address,
                                                        std::uint32_t live,
                                                        std::uint32_t sink_pages,
                                                        std::uint32_t window_pages,
                                                        const std::vector<std::uint32_t>& keep_pages) {
    last_fetch_candidates_ = 0;
    last_fetched_pages_    = 0;
    last_fetch_shortfall_  = 0;
    if (host_kv_extents == nullptr) { return 0; }
    LogicalKVPageStore* const pages =
        &addresses == text_kv_addresses.get() ? text_kv_pages.get() : backend_kv_pages.get();
    if (pages == nullptr) { return 0; }
    const std::uint32_t end        = std::min(live, addresses.mapped_pages(address));
    const std::uint32_t keep_begin = end > window_pages ? end - window_pages : 0;
    std::vector<LogicalKVPageHandle> wanted;
    for (std::uint32_t page = 0; page < end; ++page) {
        const bool addressed = page < sink_pages || page >= keep_begin ||
                               std::find(keep_pages.begin(), keep_pages.end(), page) !=
                                   keep_pages.end();
        if (!addressed) { continue; }
        const LogicalKVPageHandle handle = addresses.logical_page(address, page);
        if (pages->device_resident(handle)) { continue; }
        ++last_fetch_candidates_;
        // One page at a time, and only when the pool hands back a page for it: the alternative -
        // one big reservation - either fails whole or succeeds whole, and a partial fetch is what
        // this call is for. Eligibility is re-checked because reserve_device_replica() throws.
        if (!pages->can_reserve_device_replica(handle)) { continue; }
        // A page whose Host copy was spilled has to be back in the arena before the copy can read
        // it. Done before the Device reservation so a skip leaks nothing; the bytes stay in the
        // file, so a later attempt can still take them.
        if (!host_kv_extents->ensure_host_resident(*pages, handle)) { continue; }
        auto slot = pages->physical_pool().reserve(1);
        if (!slot) { continue; }
        const DeviceKVPageHandle destination = pages->reserve_device_replica(handle, *slot);
        const HostKVPageReplica replica       = pages->host_replica(handle);
        const auto host = host_kv_extents->view(replica.extent).subview(replica.page_offset, 1);
        const std::span<const DeviceKVPageHandle> one(&destination, 1);
        const TransferWork work = pages->physical_pool().copy_from_host(host, one, device.stream);
        pages->publish_device_replica(handle);
        // appendix FL (M3 3.3): and publish it back into the ADDRESS. reserve/publish only touch
        // the descriptor, while apply_absent_window() rebuilds the device row from the shadow - so
        // without this the restore was invisible and the kernel kept reading the page the demotion
        // gave back. This is the half that makes the fetch actually mean something.
        {
            const KVExecutionRowHandle fetch_row = addresses.execution_row(address).handle();
            const std::span<const DeviceKVPageHandle> restored(&destination, 1);
            cache.execution_tables().publish(fetch_row, page, restored, device.stream);
        }
        ++last_fetched_pages_;
        kv_tier_stats().fetched_pages += 1;
        kv_tier_stats().fetched_bytes += work.payload_bytes;
    }
    last_fetch_shortfall_ = last_fetch_candidates_ - last_fetched_pages_;
    return last_fetched_pages_;
}

// port (sm_89, V10-79): give ONE page back to the pool, keeping its bytes on the Host tier, so a
// page the current step does not read can be traded for one it does. This is the demotion of
// demote_absent_pages_to_host() applied to a single, explicitly named page: prepare -> copy ->
// publish
// -> demote, in that order, because the Host extent pins the Device source until publish() attaches
// the replica. Returns false when the page cannot be given back (no Host room, a protected page, no
// replica) - the caller then leaves the shortage to be reported instead of inventing room.
bool ProgramImpl::demote_page_for_room(KVAddressSpaceStore& addresses, PagedKVCache& cache,
                                       KVAddressSpaceHandle address, std::uint32_t page) {
    if (host_kv_extents == nullptr) { return false; }
    LogicalKVPageStore* const pages =
        &addresses == text_kv_addresses.get() ? text_kv_pages.get() : backend_kv_pages.get();
    if (pages == nullptr) { return false; }
    const LogicalKVPageHandle handle = addresses.logical_page(address, page);
    if (!pages->device_resident(handle) || pages->host_resident(handle)) { return false; }
    const std::span<const LogicalKVPageHandle> one(&handle, 1);
    auto reservation = host_kv_extents->prepare(*pages, one);
    if (!reservation) { return false; }
    const TransferWork tier_copy =
        cache.page_pool().copy_to_host(host_kv_extents->device_sources(*reservation),
                                       host_kv_extents->writable_view(*reservation), device.stream);
    const HostKVExtentCapability capability = host_kv_extents->publish(std::move(*reservation));
    if (const std::int32_t row = addresses.bound_row(address); row >= 0) {
        live_host_extents_[row].push_back(LiveHostExtent{.capability = capability,
                                                         .age        = host_extent_age_++,
                                                         .page_begin = page,
                                                         .page_end   = page + 1U});
    }
    if (!pages->demote_device_replica(handle)) { return false; }
    ++page_in_counters(addresses).evicted;
    kv_tier_stats().spilled_pages += 1;
    kv_tier_stats().spilled_bytes += tier_copy.payload_bytes;
    return true;
}

namespace {
// port (sm_89, V10-81): one row's read set for this step, in the vocabulary the window and
// stage_step_kv_pages() use. `keep` is owned because it is the union of the installed window's list
// and the selector's stored result for THAT row - the same two sources apply_kv_eviction_window()
// masks with, which is what makes "this page is spare" true rather than assumed.
struct StepReadSet {
    std::uint32_t sink_pages = 0;
    std::uint32_t keep_begin = 0; // first page of the newest window
    std::vector<std::uint32_t> keep;

    [[nodiscard]] bool addresses(std::uint32_t page) const noexcept {
        return page < sink_pages || page >= keep_begin ||
               std::find(keep.begin(), keep.end(), page) != keep.end();
    }
};

StepReadSet step_read_set(const KvEvictionWindow& window, const KVExecutionTablePool& tables,
                          KVExecutionRowHandle row, std::uint32_t live, std::uint32_t end) {
    StepReadSet out;
    if (!window.enabled()) {
        // No window: every mapped page of the live history is read, so nothing is spare.
        out.keep_begin = end;
        return out;
    }
    out.sink_pages          = window.sink_pages;
    const std::uint32_t eff = effective_window_pages(window.window_pages, live);
    out.keep_begin          = end > eff ? end - eff : 0U;
    out.keep                = window.keep_pages;
    if (const auto found = g_scored_keep_by_row.find(tables.page_score_row(row).data);
        found != g_scored_keep_by_row.end()) {
        out.keep.insert(out.keep.end(), found->second.begin(), found->second.end());
    }
    return out;
}
} // namespace

// port (sm_89, V10-81): cross-row borrowing. `demote_page_for_room` can only give back a page of
// the row it is serving, so a row whose own non-addressed pages are gone fails even when a sibling
// row is holding pages this step does not read: one physical page pool holds every lane's window at
// once. What may be taken is exactly what NO active row of the space addresses this step -
// correctness first, because a page a step reads must not leave the Device after that step's
// staging ran, and the victim's own window would then read a recycled page.
//
// Victim selection: only a row with at least two spare pages qualifies, and at most half of that
// row's slack is taken per call, so a victim never loses its last spare page and always keeps half
// of what it was holding - enough to serve its own next step without a refetch. The qualifying rows
// are visited most-slack-first from a rotating start, so equal-slack rows take turns instead of the
// lowest row index being drained every time. A taken page leaves the victim's row marked absent
// (the representation a demotion leaves) and its Host extent recorded against the victim's row, so
// the victim's own page-in restores it through the existing path.
bool ProgramImpl::borrow_slot_from_another_row(KVAddressSpaceStore& addresses, PagedKVCache& cache,
                                               KVAddressSpaceHandle requester, std::uint32_t live,
                                               std::uint32_t sink_pages, std::uint32_t window_pages,
                                               const std::vector<std::uint32_t>& keep_pages) {
    if (host_kv_extents == nullptr) { return false; }
    // A/B switch, default on: without it the same binary reproduces the pre-V10-81 behaviour, so
    // the "the borrow was what made room" claim is measured against its own control instead of
    // inferred.
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_KV_BORROW");
        return value == nullptr || *value == '\0' || std::strtol(value, nullptr, 10) != 0;
    }();
    if (!enabled) { return false; }
    const KvEvictionWindow& window = kv_eviction_window();
    if (!window.enabled()) { return false; }
    LogicalKVPageStore* const pages =
        &addresses == text_kv_addresses.get() ? text_kv_pages.get() : backend_kv_pages.get();
    if (pages == nullptr) { return false; }
    // Every handle ANY active row reads this step, the requester included. Compared on the handle
    // and not on the page index because a forked prefix is one logical page shared by two rows:
    // page 3 of the victim may be page 3 of a third row, and taking it would break that row
    // instead.
    std::vector<LogicalKVPageHandle> addressed;
    const auto collect = [&](KVAddressSpaceHandle other) {
        const std::uint32_t frontier = addresses.committed_frontier(other);
        if (frontier == 0) { return; }
        const std::uint32_t other_live = kv_live_pages(frontier);
        const std::uint32_t end        = std::min(other_live, addresses.mapped_pages(other));
        if (end == 0) { return; }
        const StepReadSet reads =
            other == requester
                ? StepReadSet{.sink_pages = sink_pages,
                              .keep_begin = end > window_pages ? end - window_pages : 0U,
                              .keep       = keep_pages}
                : step_read_set(window, cache.execution_tables(),
                                addresses.execution_row(other).handle(), other_live, end);
        for (std::uint32_t page = 0; page < end; ++page) {
            if (reads.addresses(page)) { addressed.push_back(addresses.logical_page(other, page)); }
        }
    };
    addresses.for_each_active_address(collect);
    const auto is_addressed = [&](LogicalKVPageHandle handle) {
        return std::find(addressed.begin(), addressed.end(), handle) != addressed.end();
    };

    struct Candidate {
        KVAddressSpaceHandle address;
        std::int32_t row = -1;
        std::vector<std::uint32_t> slack; // spare pages, oldest first
    };

    std::vector<Candidate> candidates;
    addresses.for_each_active_address([&](KVAddressSpaceHandle other) {
        if (other == requester) { return; }
        const std::uint32_t frontier = addresses.committed_frontier(other);
        if (frontier == 0) { return; }
        const std::uint32_t victim_live = kv_live_pages(frontier);
        const std::uint32_t end         = std::min(victim_live, addresses.mapped_pages(other));
        if (end == 0) { return; }
        // A row whose mask was deferred reads its whole history this step: nothing of it is spare.
        const void* const victim_key =
            cache.execution_tables().page_score_row(addresses.execution_row(other).handle()).data;
        if (std::find(deferred_mask_rows_.begin(), deferred_mask_rows_.end(), victim_key) !=
            deferred_mask_rows_.end()) {
            return;
        }
        Candidate candidate;
        candidate.address = other;
        candidate.row     = addresses.bound_row(other);
        for (std::uint32_t page = 0; page < end; ++page) {
            const LogicalKVPageHandle handle = addresses.logical_page(other, page);
            if (!pages->device_resident(handle) || pages->host_resident(handle)) { continue; }
            if (is_addressed(handle)) { continue; }
            // The same eligibility a borrow's demotion needs; a protected or pinned page would make
            // the whole attempt a no-op, so it is not counted as slack in the first place.
            if (!pages->can_demote_device_replica(handle)) { continue; }
            candidate.slack.push_back(page);
        }
        if (candidate.slack.size() >= 2) { candidates.push_back(std::move(candidate)); }
    });
    if (candidates.empty()) { return false; }
    const std::size_t space = &addresses == text_kv_addresses.get() ? 0 : 1;
    // Rotation first (so ties take turns), then most slack first (so the row that can afford it
    // gives the page). The cursor advances only on a successful take.
    std::rotate(candidates.begin(),
                candidates.begin() +
                    static_cast<std::ptrdiff_t>(borrow_cursor_[space] % candidates.size()),
                candidates.end());
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const Candidate& left, const Candidate& right) {
                         return left.slack.size() > right.slack.size();
                     });
    for (const Candidate& candidate : candidates) {
        const std::size_t budget = candidate.slack.size() / 2;
        for (std::size_t index = 0; index < budget; ++index) {
            const std::uint32_t page = candidate.slack[index];
            if (!demote_page_for_room(addresses, cache, candidate.address, page)) { continue; }
            if (candidate.row >= 0) {
                const KVExecutionRowHandle victim_row =
                    addresses.execution_row(candidate.address).handle();
                const std::array<std::uint32_t, 1> taken{page};
                cache.execution_tables().mark_absent(victim_row, taken, device.stream);
            }
            PageInCounters& counters = page_in_counters(addresses);
            ++counters.borrowed;
            ++borrow_cursor_[space];
            std::fprintf(stderr,
                         "[kv-page-in] borrow %s: requester_row=%d victim_row=%d page=%u "
                         "slack=%zu of %u mapped (pool %u/%u)\n",
                         space == 0 ? "text" : "backend", addresses.bound_row(requester),
                         candidate.row, page, candidate.slack.size(),
                         addresses.mapped_pages(candidate.address),
                         cache.page_pool().capacity_pages(), kv_logical_page_groups);
            std::fflush(stderr);
            return true;
        }
    }
    return false;
}

// port (sm_89, V10-79): the read half of the resident-KV leg. Every page this step ADDRESSES whose
// Device replica is gone is copied back into a free slot and republished into the row before the
// layer that reads it runs; when the pool has no free slot, the oldest page this step does NOT
// address is given back first (the recency rule the window's own scan uses - the scan walks [0,
// end), so the earliest page is the least recently useful one), and when THIS row has nothing left
// to give back, a sibling row that holds pages no active row reads this step lends one
// (borrow_slot_from_another_row, V10-81). A page that cannot be brought back is counted, never
// assumed away: the caller turns a non-zero answer into a refusal that names the remaining limit
// instead of letting attention read a table entry naming a recycled page. Returns the number of
// addressed pages still without a Device replica.
std::uint32_t ProgramImpl::page_in_addressed_pages(KVAddressSpaceStore& addresses,
                                                   PagedKVCache& cache,
                                                   KVAddressSpaceHandle address, std::uint32_t live,
                                                   std::uint32_t sink_pages,
                                                   std::uint32_t window_pages,
                                                   const std::vector<std::uint32_t>& keep_pages) {
    PageInCounters& counters = page_in_counters(addresses);
    counters.candidates      = 0;
    counters.fetched         = 0;
    if (host_kv_extents == nullptr) { return 0; }
    LogicalKVPageStore* const pages =
        &addresses == text_kv_addresses.get() ? text_kv_pages.get() : backend_kv_pages.get();
    if (pages == nullptr) { return 0; }
    const std::uint32_t end        = std::min(live, addresses.mapped_pages(address));
    const std::uint32_t keep_begin = end > window_pages ? end - window_pages : 0;
    const auto addressed           = [&](std::uint32_t page) {
        return page < sink_pages || page >= keep_begin ||
               std::find(keep_pages.begin(), keep_pages.end(), page) != keep_pages.end();
    };
    std::vector<std::uint32_t> wanted; // addressed, no Device replica
    std::vector<std::uint32_t> spare;  // not addressed: room may be made here, oldest first
    for (std::uint32_t page = 0; page < end; ++page) {
        const LogicalKVPageHandle handle = addresses.logical_page(address, page);
        if (addressed(page)) {
            if (!pages->device_resident(handle)) { wanted.push_back(page); }
        } else if (pages->device_resident(handle) && !pages->host_resident(handle)) {
            // A page already on Host needs no room trade: its bytes are not using the pool.
            spare.push_back(page);
        }
    }
    counters.candidates = static_cast<std::uint32_t>(wanted.size());
    if (std::getenv("NINFER_KV_PAGE_IN_SLACK_LOG") != nullptr) {
        std::fprintf(stderr,
                     "[kv-page-in] slack %s: live=%u end=%u addressed_missing=%zu spare=%zu "
                     "pool=%u/%u\n",
                     &addresses == text_kv_addresses.get() ? "text" : "backend", live, end,
                     wanted.size(), spare.size(), cache.page_pool().capacity_pages(),
                     kv_logical_page_groups);
        std::fflush(stderr);
    }
    if (wanted.empty()) { return 0; }
    std::size_t spare_cursor = 0;
    for (const std::uint32_t page : wanted) {
        const LogicalKVPageHandle handle = addresses.logical_page(address, page);
        // Same eligibility the window's fetch asks for; a page whose bytes were RELEASED (the lossy
        // switch) has none, and that is exactly the case this counter exists to make visible.
        if (!pages->can_reserve_device_replica(handle)) { continue; }
        if (!host_kv_extents->ensure_host_resident(*pages, handle)) { continue; }
        if (!pages->physical_pool().reserve(1)) {
            bool made_room = false;
            while (spare_cursor < spare.size() && !made_room) {
                made_room = demote_page_for_room(addresses, cache, address, spare[spare_cursor++]);
            }
            // port (sm_89, V10-81): this row's own non-addressed pages are gone, which under a
            // multi-lane batch is not the same as "nobody can give a page back". Take one from a
            // row that holds pages no active row reads this step.
            if (!made_room) {
                made_room = borrow_slot_from_another_row(addresses, cache, address, live,
                                                         sink_pages, window_pages, keep_pages);
            }
            if (!made_room) { continue; }
        }
        auto slot = pages->physical_pool().reserve(1);
        if (!slot) { continue; }
        const DeviceKVPageHandle destination = pages->reserve_device_replica(handle, *slot);
        const HostKVPageReplica replica      = pages->host_replica(handle);
        const auto host = host_kv_extents->view(replica.extent).subview(replica.page_offset, 1);
        const std::span<const DeviceKVPageHandle> one(&destination, 1);
        const TransferWork work = pages->physical_pool().copy_from_host(host, one, device.stream);
        pages->publish_device_replica(handle);
        {
            const KVExecutionRowHandle fetch_row = addresses.execution_row(address).handle();
            const std::span<const DeviceKVPageHandle> restored(&destination, 1);
            cache.execution_tables().publish(fetch_row, page, restored, device.stream);
        }
        ++counters.fetched;
        kv_tier_stats().fetched_pages += 1;
        kv_tier_stats().fetched_bytes += work.payload_bytes;
    }
    return counters.candidates - counters.fetched;
}

// port (sm_89, V10-79): one step's page-in, with the step's own addressed set. The window decides
// what the step reads: with one installed, sink + newest window + the selector's keep set; without
// one, every mapped page of the live history - which is why a full-attention model cannot run with
// an arena smaller than its history, and why that refusal names the arena rather than the Host
// tier.
void ProgramImpl::stage_step_kv_pages(KVAddressSpaceStore& addresses, PagedKVCache& cache,
                                      KVAddressSpaceHandle address) {
    if (host_kv_extents == nullptr) {
        // A page that leaves the Device has to land somewhere: without a Host KV tier the leg has
        // nothing to copy back from, and the honest answer is to say so rather than to run with a
        // tiered arena whose evicted pages are unreachable.
        throw std::runtime_error(
            "NINFER_KV_PAGE_IN needs a Host KV tier: configure the Host context budget, or use a "
            "residency plan that prices the Host payload the tiered arena relies on");
    }
    const std::uint32_t frontier = addresses.committed_frontier(address);
    if (frontier == 0) { return; }
    const KVExecutionRowHandle row = addresses.execution_row(address).handle();
    const KvEvictionWindow& window = kv_eviction_window();
    const std::uint32_t live       = kv_live_pages(frontier);
    // The addressed set, in the window's own vocabulary. With NO window installed every page of the
    // live history is addressed - that is what a full-attention layer does - so the window is named
    // as "the newest `live` pages", which makes the shared predicate below address [0, live)
    // exactly, and leaves nothing spare: a step that reads everything cannot trade pages with
    // itself.
    const bool windowed = window.enabled();
    const std::uint32_t eff_window =
        windowed ? effective_window_pages(window.window_pages, live) : live;
    const std::uint32_t sink_pages = windowed ? window.sink_pages : 0U;
    std::vector<std::uint32_t> keep;
    if (windowed) {
        keep = window.keep_pages;
        // The selector recomputes its keep set once per call inside apply_kv_eviction_window();
        // read the decision that call stored rather than ranking again - a second ranking would
        // read the planes after their reset and could name a different set than the mask the row
        // was built from.
        if (const auto found =
                g_scored_keep_by_row.find(cache.execution_tables().page_score_row(row).data);
            found != g_scored_keep_by_row.end()) {
            keep.insert(keep.end(), found->second.begin(), found->second.end());
        }
    }
    const std::uint32_t missing =
        page_in_addressed_pages(addresses, cache, address, live, sink_pages, eff_window, keep);
    const PageInCounters& staged = page_in_counters(addresses);
    if (missing != 0) {
        // The pool in the message is the one this pass drew from, not the text pool: with a
        // speculative backend the two differ, and naming the wrong one would send a reader looking
        // at the space that is not short.
        throw std::runtime_error(
            "KV page-in could not restore every addressed page before the step: addressed=" +
            std::to_string(staged.candidates) + " restored=" + std::to_string(staged.fetched) +
            " missing=" + std::to_string(missing) + " borrowed=" + std::to_string(staged.borrowed) +
            " (Device pool " + std::to_string(cache.page_pool().capacity_pages()) +
            " page group(s) of " + std::to_string(kv_logical_page_groups) +
            " logical, Host payload " + std::to_string(host_kv_payload_bytes >> 20) +
            " MiB). A page this step does not address has to exist to be given back; the "
            "full-attention layers address the whole history, so the pool must cover it.");
    }
    // Verification knob (default off): give back part of the set the pass just staged and stage it
    // again, so the round trip Host -> Device -> Host -> Device is executed inside a real step and
    // the tokens must not move. One shot per address space: the point is a deterministic,
    // inspectable event, not a standing cost - and with a speculative backend the backend space
    // needs its own shot, otherwise the second space would never be exercised.
    const std::uint32_t probe = kv_page_in_probe_pages();
    bool& probe_done          = page_in_probe_done_[&addresses == text_kv_addresses.get() ? 0 : 1];
    if (probe == 0 || probe_done || live == 0) { return; }
    probe_done                    = true;
    const std::uint32_t probe_end = std::min({live, addresses.mapped_pages(address), probe});
    std::uint32_t given_back      = 0;
    for (std::uint32_t page = 0; page < probe_end; ++page) {
        if (demote_page_for_room(addresses, cache, address, page)) { ++given_back; }
    }
    const std::uint32_t staged_before = staged.fetched;
    const std::uint32_t again =
        page_in_addressed_pages(addresses, cache, address, live, sink_pages, eff_window, keep);
    std::fprintf(stderr,
                 "[kv-page-in] probe %s: staged=%u demoted=%u refetched=%u missing=%u borrowed=%u "
                 "live=%u pool=%u/%u\n",
                 &addresses == text_kv_addresses.get() ? "text" : "backend", staged_before,
                 given_back, staged.fetched, again, staged.borrowed, live,
                 cache.page_pool().capacity_pages(), kv_logical_page_groups);
    std::fflush(stderr);
    if (again != 0) {
        throw std::runtime_error("KV page-in probe: the re-fetch after the induced demotion left " +
                                 std::to_string(again) +
                                 " addressed page(s) without a Device replica");
    }
}

// port (sm_89, V10-80): the speculative backend's half of the same leg. The draft cache has its own
// page store and its own execution row (io.backend_kv_table_row / active_backend_kv_table_rows_),
// but the demotion gives its pages back by exactly the same rules (apply_kv_eviction_window runs
// for it too), so the step that makes the draft address them has to restore them by the same rules.
// The helpers are already parameterised by address space and cache, so this is only the call - and
// the two guards that keep a default run identical: the switch, and a tiered arena.
void ProgramImpl::stage_step_backend_kv_pages(SequenceState& state) {
    // port (sm_89, V10-86): one line, once, naming why this space is (not) staged. The DFlash2
    // drafter keeps its draft window in the cyclic local cache; whether it also has a paged "full"
    // half decides whether the leg has an address space to work on at all, and that difference is
    // not visible from the counters (both report zero fetched pages).
    if (std::getenv("NINFER_KV_PAGE_IN_TRACE") != nullptr) {
        static bool told = false;
        if (!told) {
            told                  = true;
            const bool has_bundle = state.kv && state.kv->backend;
            const std::uint32_t frontier =
                (has_bundle && backend_kv_addresses)
                    ? backend_kv_addresses->committed_frontier(*state.kv->backend)
                    : 0U;
            std::fprintf(stderr,
                         "[kv-page-in] backend space: cache=%s bundle=%d leg=%d tiered=%d "
                         "frontier=%u backend=%u\n",
                         backend_kv_cache() != nullptr ? "paged" : "none", has_bundle ? 1 : 0,
                         page_in_active() ? 1 : 0, kv_arena_tiered() ? 1 : 0, frontier,
                         static_cast<unsigned>(speculative_backend));
            std::fflush(stderr);
        }
    }
    if (!state.kv || !state.kv->backend || !page_in_active() || !kv_arena_tiered()) { return; }
    PagedKVCache* const cache = backend_kv_cache();
    if (cache == nullptr) { return; }
    stage_step_kv_pages(*backend_kv_addresses, *cache, *state.kv->backend);
}

// appendix EY (P3a): the Device page pool's own account - allocated / reserved / available /
// capacity, one line per decision point. It lives in its own file, and not in the score dump, because
// a diagnostic must not move the bytes a frozen regression gate watches. Off unless
// NINFER_KV_EVICT_POOL_DUMP names a path; written whether or not the release is enabled, so the
// disabled run is a control rather than a separate experiment.
void ProgramImpl::dump_pool_account(PagedKVCache& cache) {
    const char* path = std::getenv("NINFER_KV_EVICT_POOL_DUMP");
    if (path == nullptr || *path == '\0') { return; }
    const DeviceKVPagePool& pool = cache.page_pool();
    std::ofstream out(path, std::ios::app);
    if (!out) { return; }
    out << pool.allocated_pages() << '\t' << pool.reserved_pages() << '\t' << pool.available_pages()
        << '\t' << pool.capacity_pages() << '\n';
    // appendix FF (M3 3.2b): Host occupancy and how many pages the last demotion moved there. A '#'
    // marker, so the numeric columns above keep their meaning for existing readers.
    out << "# demote host_occupied="
        << (host_kv_extents ? host_kv_extents->occupied() : 0U)
        << " host_capacity=" << (host_kv_extents ? host_kv_extents->capacity() : 0U)
        << " last=" << last_demoted_pages_ << " stage=" << last_demote_stage_
        << " candidates=" << last_demote_candidates_ << " block=0x" << std::hex
        << last_demote_block_mask_ << std::dec << " keep_snapshots=" << host_keep_snapshots_
        << " live_floor=" << (host_context_arena ? host_context_arena->live_floor_bytes() : 0U)
        << " evicted_pages=" << last_host_evicted_pages_
        << " addressed_absent=" << last_addressed_absent_
        << " addressed_absent_host=" << last_addressed_absent_host_
        << " stale_checkpoints=" << last_stale_checkpoints_ << '\n';
    out << "# frontier rounds=" << kv_tier_stats().frontier_rounds
        << " advanced_to=" << kv_tier_stats().frontier_advanced_to << '\n';
    // appendix GF.6: host time inside the eviction decision. us_per_call is the per-lane per-round
    // host cost; compare it against the round wall time to see whether this is the C=8 cost.
    const std::uint64_t host_calls = kv_tier_stats().host_eviction_calls;
    out << "# host_eviction calls=" << host_calls
        << " total_ms=" << (kv_tier_stats().host_eviction_ns / 1000000ULL)
        << " us_per_call="
        << (host_calls == 0 ? 0ULL : kv_tier_stats().host_eviction_ns / host_calls / 1000ULL)
        << '\n';
    const std::uint64_t mgmt_calls = kv_tier_stats().host_kv_mgmt_calls;
    out << "# host_kv_mgmt calls=" << mgmt_calls
        << " total_ms=" << (kv_tier_stats().host_kv_mgmt_ns / 1000000ULL)
        << " us_per_call="
        << (mgmt_calls == 0 ? 0ULL : kv_tier_stats().host_kv_mgmt_ns / mgmt_calls / 1000ULL)
        << '\n';
    // appendix FH (M3 3.2d): `kv tiers`, in Strata's "expert tiers" style - the two tiers as they
    // stand now, then, since the start, what has moved between them. "spilled" is the demotion's
    // Device -> Host payload (reversible: the matrix says whether `--host-kv-mib` is doing work);
    // "fetched" is the planner's Host -> Device restore payload (the pages whose Device replica the
    // demotion gave up and a later plan asked back); "preempted_checkpoints" is FE.3's rule
    // actually firing. A '#' marker, so the numeric columns above keep their meaning for existing
    // readers.
    out << "# kv tiers device=" << pool.allocated_pages() << '/' << pool.capacity_pages()
        << " host=" << (host_kv_extents ? host_kv_extents->occupied() : 0U) << '/'
        << (host_kv_extents ? host_kv_extents->capacity() : 0U)
        << " host_live=" << (host_context_arena ? host_context_arena->live_kv_bytes_held() : 0U)
        << " host_checkpoint="
        << (host_context_arena ? host_context_arena->checkpoint_bytes_held() : 0U)
        << " fetch_candidates=" << last_fetch_candidates_
        << " fetched_now=" << last_fetched_pages_
        << " fetch_shortfall=" << last_fetch_shortfall_
        << " spilled_pages=" << kv_tier_stats().spilled_pages
        << " spilled_bytes=" << kv_tier_stats().spilled_bytes
        << " fetched_pages=" << kv_tier_stats().fetched_pages
        << " fetched_bytes=" << kv_tier_stats().fetched_bytes
        << " preempted_checkpoints=" << kv_tier_stats().preempted_checkpoints << '\n';
    // V10-79: the page-in leg's own pass, so "a page left the Device and was read back" is a line a
    // reader can check instead of an inference from the cumulative counters above. V10-80 keeps the
    // text line's meaning by giving the speculative backend's pass a line of its own: both run
    // inside the same step, so one set of counters would report whichever ran last.
    out << "# page_in candidates=" << last_page_in_[0].candidates
        << " fetched=" << last_page_in_[0].fetched << " evicted=" << last_page_in_[0].evicted
        << " tiered=" << (kv_arena_tiered() ? 1 : 0) << '\n';
    if (backend_kv_cache() != nullptr) {
        out << "# page_in_backend candidates=" << last_page_in_[1].candidates
            << " fetched=" << last_page_in_[1].fetched << " evicted=" << last_page_in_[1].evicted
            << '\n';
    }
}

void ProgramImpl::dump_page_scores(KVAddressSpaceStore& addresses, PagedKVCache& cache,
                                   KVAddressSpaceHandle address,
                                   const std::vector<std::uint32_t>& keep_pages) {
    const char* path = std::getenv("NINFER_KV_EVICT_SCORES_DUMP");
    if (path == nullptr || *path == '\0') { return; }
    const std::uint32_t frontier = addresses.committed_frontier(address);
    if (frontier == 0) { return; }
    const KVExecutionRowHandle row = addresses.execution_row(address).handle();
    const Tensor counts = cache.execution_tables().page_score_row(row);
    const Tensor peaks  = cache.execution_tables().page_score_peak_row(row);
    const Tensor recents = cache.execution_tables().page_score_recent_row(row);
    const std::uint32_t pages = static_cast<std::uint32_t>(counts.ne[0]);
    const std::uint32_t live  = kv_live_pages(frontier);
    const std::uint32_t rows  = live < pages ? live : pages;
    std::vector<std::uint32_t> host_counts(static_cast<std::size_t>(rows), 0u);
    std::vector<std::uint32_t> host_peaks(static_cast<std::size_t>(rows), 0u);
    std::vector<std::uint32_t> host_recents(static_cast<std::size_t>(rows), 0u);
    if (rows != 0) {
        const std::size_t bytes = static_cast<std::size_t>(rows) * sizeof(std::uint32_t);
        core::copy_bytes(host_counts.data(), counts.data, bytes,
                         core::TransferDirection::DeviceToHost, device.stream);
        core::copy_bytes(host_peaks.data(), peaks.data, bytes,
                         core::TransferDirection::DeviceToHost, device.stream);
        core::copy_bytes(host_recents.data(), recents.data, bytes,
                         core::TransferDirection::DeviceToHost, device.stream);
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
    }
    // Append with a call marker: the number of markers says how often the decision point ran, and
    // consecutive blocks say whether the question tail ever reaches a dumped call (appendix CD.5.1).
    // Each statistic is written raw into its own table, because its unit is fixed by the plane and
    // not by the shape that happened to run (appendices CO.4/DD/DE): the count plane stores a
    // running-best hit count, the peak plane an order-preserving peak logit (recover it with
    // paged_kv_page_score_decode). 0 means "nothing recorded".
    //
    // The count table keeps the original "page<TAB>raw" lines so a dump still lines up with the
    // archives written before the split. The peak rows are prefixed with 'p' on purpose: a reader
    // that only understands the old format then fails loudly on them instead of silently merging the
    // two units back into one table - which is exactly how appendix DE.1 read counts as logits and
    // reported phantom NaNs.
    std::ofstream out(path, std::ios::app);
    if (!out) { return; }
    out << "# call\n";
    // The keep set the selector just handed to apply_absent_window(), as a marker line so the page
    // tables stay machine-readable (appendix DN: the selection has to be observable to be verifiable).
    if (!keep_pages.empty()) {
        out << "# keep";
        for (const std::uint32_t page : keep_pages) { out << ' ' << page; }
        out << '\n';
    }
    std::uint32_t count_max     = 0;
    std::uint32_t count_nonzero = 0;
    for (std::uint32_t page = 0; page < rows; ++page) {
        out << page << '\t' << host_counts[page] << '\n';
        if (host_counts[page] > count_max) { count_max = host_counts[page]; }
        count_nonzero += host_counts[page] != 0u ? 1u : 0u;
    }
    // C4: a silently all-zero dump is ambiguous - most KV storages (bf16, fp8, nvfp4, k8v4) have no
    // P6 2c scoring hook at all, so a plane stays zero no matter how important a page is - and that
    // is easy to misread as "no page is important". Each plane therefore states its own maximum, its
    // own nonzero count, and whether any call of this run ever recorded anything on it. Lines
    // beginning with '#' are markers, so this stays out of the page tables.
    out << "# stats max=" << count_max << " nonzero=" << count_nonzero
        << " ever_nonzero=" << (record_plane_activity(counts.data, count_max) ? 1 : 0) << '\n';
    std::uint32_t peak_max     = 0;
    std::uint32_t peak_nonzero = 0;
    out << "# peak\n";
    for (std::uint32_t page = 0; page < rows; ++page) {
        out << "p\t" << page << '\t' << host_peaks[page] << '\n';
        if (host_peaks[page] > peak_max) { peak_max = host_peaks[page]; }
        peak_nonzero += host_peaks[page] != 0u ? 1u : 0u;
    }
    out << "# stats_peak max=" << peak_max << " nonzero=" << peak_nonzero
        << " ever_nonzero=" << (record_plane_activity(peaks.data, peak_max) ? 1 : 0) << '\n';
    // The newest-rows count table (appendix DN). It is all zero unless a row span was requested, and
    // the selector needs it beside the every-row table above to form D = A - B for the SAME call.
    std::uint32_t recent_max     = 0;
    std::uint32_t recent_nonzero = 0;
    out << "# recent\n";
    for (std::uint32_t page = 0; page < rows; ++page) {
        out << "r\t" << page << '\t' << host_recents[page] << '\n';
        if (host_recents[page] > recent_max) { recent_max = host_recents[page]; }
        recent_nonzero += host_recents[page] != 0u ? 1u : 0u;
    }
    out << "# stats_recent max=" << recent_max << " nonzero=" << recent_nonzero
        << " ever_nonzero=" << (record_plane_activity(recents.data, recent_max) ? 1 : 0) << '\n';
    // The per-call reset lives in apply_kv_eviction_window(), not here: it belongs to the decision
    // point, which is also where the selector consumes the planes (appendix DU).
}

void ProgramImpl::commit_sequence_kv(SequenceState& state, std::uint32_t main,
                                     std::uint32_t backend) {
    text_kv_addresses->commit_frontier(state.kv->text, main);
    if (state.kv->backend) { backend_kv_addresses->commit_frontier(*state.kv->backend, backend); }
}

void ProgramImpl::trim_sequence_kv(SequenceState& state, std::uint32_t main,
                                   std::uint32_t backend) {
    text_kv_addresses->destructive_truncate(state.kv->text, main);
    if (state.kv->backend) {
        backend_kv_addresses->destructive_truncate(*state.kv->backend, backend);
    }
}

PagedKVCacheView ProgramImpl::text_kv_view(const SequenceState& state) const {
    return decoder->text_kv.execution_view(text_kv_addresses->execution_row(state.kv->text));
}

PagedKVCacheView ProgramImpl::mtp_kv_view(const SequenceState& state) const {
    return speculative_backend == SpeculativeBackend::Mtp
               ? decoder->mtp_cache()->execution_view(
                     backend_kv_addresses->execution_row(*state.kv->backend))
               : PagedKVCacheView{};
}

void ProgramImpl::set_device_i32(Tensor& tensor, std::int32_t value) {
    core::copy_bytes(tensor.data, &value, sizeof(value), core::TransferDirection::HostToDevice,
                     device.stream);
}

void ProgramImpl::ordered_reset(SequenceState& state) {
    refresh_state_views(state);
    // P6 2c: the relevance counters accumulate, so a reused execution row must start from zero.
    decoder->text_kv.execution_tables().reset_page_scores(
        text_kv_addresses->execution_row(state.kv->text).handle(), device.stream);
    if (state.kv->backend) {
        if (PagedKVCache* cache = backend_kv_cache(); cache != nullptr) {
            cache->execution_tables().reset_page_scores(
                backend_kv_addresses->execution_row(*state.kv->backend).handle(), device.stream);
        }
    }
    state.text_kv_valid = state.mtp_kv_valid = state.dflash_context_frontier = 0;
    work.reset();
}

} // namespace ninfer::models::qwen3_5::detail

namespace ninfer::models::qwen3_5 {

// appendix IK/IL: see the declarations in program.h.
void install_kv_eviction_window(std::uint32_t sink_pages, std::uint32_t window_pages) {
    detail::KvEvictionWindow window;
    window.sink_pages   = sink_pages;
    window.window_pages = window_pages;
    detail::installed_eviction_window() = std::move(window);
}

bool kv_absent_pages_supported_for(KvCacheStorage storage) noexcept {
    return detail::kv_absent_pages_supported(storage);
}

bool kv_eviction_window_from_environment_enabled() noexcept {
    return detail::kv_eviction_window().enabled();
}

} // namespace ninfer::models::qwen3_5
