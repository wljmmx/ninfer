#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "core/device.h"

#include <algorithm>
#include <array>
#include <cstdlib>
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

void ProgramImpl::apply_kv_eviction_window(KVAddressSpaceStore& addresses, PagedKVCache& cache,
                                           KVAddressSpaceHandle address) {
    const KvEvictionWindow& window = kv_eviction_window();
    if (!window.enabled()) { return; }
    if (!kv_absent_pages_supported(cache.storage())) { return; }
    const std::uint32_t frontier = addresses.committed_frontier(address);
    if (frontier == 0) { return; }
    cache.execution_tables().apply_absent_window(addresses.execution_row(address).handle(),
                                                 kv_live_pages(frontier), window.sink_pages,
                                                 window.window_pages, window.keep_pages,
                                                 device.stream);
    dump_page_scores(addresses, cache, address);
}

void ProgramImpl::dump_page_scores(KVAddressSpaceStore& addresses, PagedKVCache& cache,
                                   KVAddressSpaceHandle address) {
    const char* path = std::getenv("NINFER_KV_EVICT_SCORES_DUMP");
    if (path == nullptr || *path == '\0') { return; }
    const std::uint32_t frontier = addresses.committed_frontier(address);
    if (frontier == 0) { return; }
    const KVExecutionRowHandle row = addresses.execution_row(address).handle();
    const Tensor counts = cache.execution_tables().page_score_row(row);
    const Tensor peaks  = cache.execution_tables().page_score_peak_row(row);
    const std::uint32_t pages = static_cast<std::uint32_t>(counts.ne[0]);
    const std::uint32_t live  = kv_live_pages(frontier);
    const std::uint32_t rows  = live < pages ? live : pages;
    std::vector<std::uint32_t> host_counts(static_cast<std::size_t>(rows), 0u);
    std::vector<std::uint32_t> host_peaks(static_cast<std::size_t>(rows), 0u);
    if (rows != 0) {
        const std::size_t bytes = static_cast<std::size_t>(rows) * sizeof(std::uint32_t);
        CUDA_CHECK(cudaMemcpyAsync(host_counts.data(), counts.data, bytes,
                                   cudaMemcpyDeviceToHost, device.stream));
        CUDA_CHECK(cudaMemcpyAsync(host_peaks.data(), peaks.data, bytes,
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
    if (!kv_page_scores_cumulative_enabled()) {
        // Per call is the semantics the planes are designed around (appendix CO): clearing here -
        // the dump runs before the forward - means the next dump reads exactly one call's
        // contribution, whatever shape produced it. Both planes are cleared together.
        cache.execution_tables().reset_page_scores(row, device.stream);
    }
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
