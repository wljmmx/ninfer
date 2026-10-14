#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/kv_tier_stats.h"
#include "core/device.h"

#include <algorithm>
#include <array>
#include <cstdlib>
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

// appendix FF (M3 3.2b): demote the masked pages to Host instead of dropping them. Off by default,
// and mutually exclusive with the release switch - see demote_absent_pages_to_host().
bool kv_host_demote_enabled() {
    const char* value = std::getenv("NINFER_KV_EVICT_HOST_DEMOTE");
    return value != nullptr && *value != '\0' && std::strtol(value, nullptr, 10) != 0;
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
    CUDA_CHECK(cudaMemcpyAsync(host_counts.data(), counts.data, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(host_recents.data(), recents.data, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(host_peaks.data(), peaks.data, bytes, cudaMemcpyDeviceToHost, stream));
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

const KvEvictionWindow& kv_eviction_window() {
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

void ProgramImpl::ensure_sequence_kv_mapped(SequenceState& state, std::uint32_t main,
                                            std::uint32_t backend) {
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
}

// The last informative selection per execution row, keyed on the plane row's data pointer (stable and
// distinct per row, exactly as the N3 flag is keyed). Without it a decode-step decision - which sees no
// counts at all - would replace a good prefill selection with a degenerate one (appendix DQ).
static std::map<const void*, std::vector<std::uint32_t>> g_scored_keep_by_row;

void ProgramImpl::apply_kv_eviction_window(KVAddressSpaceStore& addresses, PagedKVCache& cache,
                                           KVAddressSpaceHandle address) {
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
    if (!defer_apply) {
        cache.execution_tables().apply_absent_window(row, live, window.sink_pages, eff_window,
                                                     keep, device.stream);
        // appendix FF (M3 3.2b) vs appendix EY (P3a): the same masked pages, two different fates.
        // Demote keeps their bytes on Host, so a later query can still reach them; release throws them
        // away. Both return the Device page, so the capacity effect is identical - only reversibility
        // differs. Demote wins when both switches are set, because release would destroy what it saved.
        std::uint32_t demoted = 0;
        if (kv_host_demote_enabled()) {
            demoted = demote_absent_pages_to_host(addresses, cache, address, live, window.sink_pages,
                                                  eff_window, keep);
        }
        if (kv_eviction_release_enabled() && demoted == 0) {
            // Also the fallback when the demotion demoted nothing - most importantly when no Host
            // arena is configured. Otherwise the demote switch would win the fork, do nothing, and
            // leave every masked page materialized with nothing saying so.
            // Deferred rounds never reach here, so the fate always follows the mask it came from.
            (void)addresses.release_absent_device_pages(address, live, window.sink_pages, eff_window,
                                                        keep);
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
                for (std::uint32_t page = std::min(window.sink_pages, retire_begin);
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
        // appendix FJ (M3 3.3): the pages the new mask drops have now given their replicas back,
        // which is exactly the room this needs - so the fetch runs after the demotion/release, and
        // a page dropped this round can never be demanded back this round. Both directions are on
        // device.stream, so the read that follows is ordered after both.
        (void)restore_addressed_host_pages(addresses, cache, address, live, window.sink_pages,
                                           eff_window, keep);
        dump_pool_account(cache);
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
                                                       const std::vector<std::uint32_t>& keep_pages) {
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
    for (std::uint32_t page = std::min(sink_pages, keep_begin); page < keep_begin; ++page) {
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
    last_demoted_pages_ = demoted;
    last_demote_stage_  = 5;
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
        CUDA_CHECK(cudaMemcpyAsync(host_counts.data(), counts.data, bytes,
                                   cudaMemcpyDeviceToHost, device.stream));
        CUDA_CHECK(cudaMemcpyAsync(host_peaks.data(), peaks.data, bytes,
                                   cudaMemcpyDeviceToHost, device.stream));
        CUDA_CHECK(cudaMemcpyAsync(host_recents.data(), recents.data, bytes,
                                   cudaMemcpyDeviceToHost, device.stream));
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
    CUDA_CHECK(
        cudaMemcpyAsync(tensor.data, &value, sizeof(value), cudaMemcpyHostToDevice, device.stream));
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
