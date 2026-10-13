#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "core/device.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <numeric>
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
        if (value == nullptr) { return std::uint32_t{0}; }
        const unsigned long parsed = std::strtoul(value, nullptr, 10);
        return parsed > 0 ? static_cast<std::uint32_t>(parsed) : std::uint32_t{0};
    }();
    return budget;
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
        if (value == nullptr) { return std::int32_t{0}; }
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
                                                 std::uint32_t budget, cudaStream_t stream) {
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
    if (sequence.kv->backend && backend_kv_addresses->active(*sequence.kv->backend)) {
        backend_kv_addresses->deactivate(*sequence.kv->backend);
    }
    if (text_kv_addresses->active(sequence.kv->text)) {
        text_kv_addresses->deactivate(sequence.kv->text);
    }
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
    if (!window.enabled() && scored_budget == 0) { return; }
    if (!kv_absent_pages_supported(cache.storage())) { return; }
    if (scored_budget != 0 && window.window_pages == 0) {
        // Without a window there is nothing to evict against, and apply_absent_window() would leave
        // the row untouched - so the knob would silently do nothing. Say so instead (appendix DN).
        throw std::invalid_argument(
            "NINFER_KV_EVICT_KEEP_FROM_SCORES needs NINFER_KV_EVICT_WINDOW to define what is kept"
            " unconditionally; otherwise the row is left untouched and the selector has no effect");
    }
    if (scored_budget != 0 && score_row_span() <= 0) {
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
    std::vector<std::uint32_t> keep       = window.keep_pages;
    bool defer_apply                      = false;
    if (scored_budget != 0) {
        const void* const key = cache.execution_tables().page_score_row(row).data;
        const auto selected = keep_pages_from_scores(cache.execution_tables(), row, live,
                                                     window.sink_pages, eff_window,
                                                     scored_budget, device.stream);
        if (!selected.empty()) {
            g_scored_keep_by_row[key] = selected;
            // Counts were recorded, so the prompt is still being read: keep the cache whole and let the
            // next decision - whose statistics are therefore still un-sparsified - decide for real.
            if (defer_eviction_enabled()) { defer_apply = true; }
        }
        const std::vector<std::uint32_t>& effective = g_scored_keep_by_row[key];
        keep.insert(keep.end(), effective.begin(), effective.end());
    }
    if (!defer_apply) {
        cache.execution_tables().apply_absent_window(row, live, window.sink_pages, eff_window,
                                                     keep, device.stream);
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
