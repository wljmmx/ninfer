#include "models/qwen3_5/execution/expert_stream.h"
#include "models/qwen3_5/execution/expert_prefetch.h"
#include "models/qwen3_5/execution/expert_path_control.h"
#include "models/qwen3_5/execution/cpu_path_policy.h"
#include "models/qwen3_5/execution/host_expert_residency.h"
#include "core/memory_budget.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <numeric>
#include <chrono>
#include <map>
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {
namespace {

// One read of the switch per process: the fill path asks this on every layer, and a getenv scan
// there is pure overhead when the trace is off (which is always, outside a diagnosis).
inline bool freeze_trace() {
    static const bool enabled = std::getenv("NINFER_FREEZE_TRACE") != nullptr;
    return enabled;
}

// port (sm_89): Q68a. Each bank remembers what it routed on its last small-batch step, so a probe
// can ask the question the read-ahead's next selection depends on: do adjacent layers route the
// same experts? Indices are compared, not bytes - different layers own different matrices; the
// hypothesis is that the router prefers similar indices from one layer to the next, and that is
// exactly what this measures. Nothing here changes behaviour; it only counts.
struct LayerTrace {
    std::array<std::vector<std::int32_t>, 64> sets;
    std::array<bool, 64> valid{};
};

LayerTrace& layer_trace() {
    static LayerTrace trace;
    return trace;
}


std::uint64_t rows_of(const WeightGeometry& parent, std::uint64_t elements) {
    if (parent.shape.empty() || parent.shape.back() == 0) {
        throw std::invalid_argument("streamed expert parent has no column axis");
    }
    return elements / parent.shape.back();
}

// port (sm_89): how the hot set of a bank follows the router. Only small batches train it, so a
// prefill chunk (which touches most experts anyway) cannot wash the ranking out; the score decays
// so the set follows the conversation instead of remembering the first prompt forever.
constexpr std::int32_t kHotSetUpdateTokens  = 4;
constexpr std::uint32_t kHotSetReorderPeriod = 32;

// port (sm_89): Q11/Q12. The freeze rule is judged on a window, not on the last layer: the controller
// uses 32 agreeing layers for its own actuators, and the same number here keeps the two from
// disagreeing about what "a while" means.
constexpr std::int32_t kFreezeWindowCalls = 32;

// port (sm_89): Q11/Q12. The old rule was a percentage copied from 1y.3. A percentage cannot answer
// the question a freeze has to answer - "does holding the set still earn more than following it" -
// and 99% was never met on the pool sizes this box runs. The rule now has two halves:
//
//   freeze  when the last 32 layers were all clean. That is a state, not a threshold: the pool
//           answered every route, so there is nothing for the router to buy.
//   keep    only if the freeze raised the static rate or did not lower throughput. If it did
//           neither, it is churn that bought nothing, and the set goes back to following.
//
// Either way the decision is made from this run's numbers. NINFER_FREEZE_RULE=1 enables it; the old
// NINFER_FREEZE_ON_NO_MISS name is accepted as the same switch, with its number deliberately
// ignored - keeping it would keep the threshold the rule exists to remove.
bool freeze_rule_enabled() {
    static const bool value = [] {
        if (const char* text = std::getenv("NINFER_FREEZE_RULE"); text != nullptr && text[0] != '0') {
            return true;
        }
        if (const char* text = std::getenv("NINFER_FREEZE_ON_NO_MISS");
            text != nullptr && text[0] != '\0') {
            std::fprintf(stderr,
                         "[freeze] NINFER_FREEZE_ON_NO_MISS is now a switch: the measured rule "
                         "replaces the percentage\n");
            std::fflush(stderr);
            return text[0] != '0';
        }
        return false;
    }();
    return value;
}

// port (sm_89): how long the ranking remembers. The 32-step half-life is ~64 tokens of memory, which
// is too short for a fixed hot set to mean anything: the plan asked for a long window or an offline
// profile (section 6, item 3). NINFER_EXPERT_DECAY_STEPS overrides it, 0 keeps the scores forever.
std::uint32_t hot_decay_period() {
    static const std::uint32_t period = [] {
        const char* value = std::getenv("NINFER_EXPERT_DECAY_STEPS");
        // 256 steps is ~8x the old 32 (~64 tokens of memory): a conversation's hot set is stable
        // over hundreds of tokens, and a short half-life chased the last few dozen instead.
        if (value == nullptr || *value == '\0') { return std::uint32_t{256}; }
        return static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10));
    }();
    return period;
}

// port (sm_89): an offline hot-set profile. One float32 row per bank, `count` experts each, in bank
// order - what NINFER_EXPERT_PROFILE_DUMP writes at exit. A later run starts with the set already
// known instead of learning it from the first 64 tokens, which is the difference between "the fixed
// set covers the routes" and "the fixed set is a guess" (section 6, item 3).
const std::vector<float>& profile_rows() {
    static const std::vector<float> rows = [] {
        std::vector<float> out;
        const char* path = std::getenv("NINFER_EXPERT_PROFILE");
        if (path == nullptr || *path == '\0') { return out; }
        std::FILE* file = std::fopen(path, "rb");
        if (file == nullptr) {
            std::fprintf(stderr, "[experts] profile %s cannot be read, learning the set instead\n",
                         path);
            return out;
        }
        std::fseek(file, 0, SEEK_END);
        const auto bytes = static_cast<std::size_t>(std::ftell(file));
        std::fseek(file, 0, SEEK_SET);
        out.resize(bytes / sizeof(float));
        if (!out.empty() && std::fread(out.data(), sizeof(float), out.size(), file) != out.size()) {
            out.clear();
        }
        std::fclose(file);
        std::fprintf(stderr, "[experts] profile %s: %zu floats\n", path, out.size());
        return out;
    }();
    return rows;
}

// The frozen set a profile asks for: its top `capacity` experts for this bank, or empty when the
// profile does not cover it.
std::vector<std::uint32_t> profile_top_experts(std::uint32_t bank, std::uint32_t count,
                                               std::uint32_t capacity) {
    const auto& rows = profile_rows();
    if (rows.empty() || count == 0) { return {}; }
    const auto stride = static_cast<std::size_t>(count);
    if ((static_cast<std::size_t>(bank) + 1) * stride > rows.size()) { return {}; }
    const auto* row = rows.data() + static_cast<std::size_t>(bank) * stride;
    std::vector<std::uint32_t> order(count);
    std::iota(order.begin(), order.end(), 0U);
    std::stable_sort(order.begin(), order.end(),
                     [row](std::uint32_t a, std::uint32_t b) { return row[a] > row[b]; });
    order.resize(std::min<std::size_t>(capacity, order.size()));
    return order;
}

// Dump support: every bank registers its scores, one file at exit.
std::mutex& profile_mutex() {
    static std::mutex mutex;
    return mutex;
}
std::map<std::uint32_t, const std::vector<float>*>& profile_registry() {
    // Intentionally never destroyed, like the request-boundary list above: entries are deregistered
    // by ~ExpertStream(), which may itself run during static teardown.
    static auto* registry = new std::map<std::uint32_t, const std::vector<float>*>();
    return *registry;
}
void dump_profiles() {
    const char* path = std::getenv("NINFER_EXPERT_PROFILE_DUMP");
    if (path == nullptr || *path == '\0') { return; }
    std::lock_guard<std::mutex> lock(profile_mutex());
    auto& registry = profile_registry();
    // Rows are borrowed from the live streams, so a dump only happens while they exist. Writing
    // only the rows that are still owned keeps an empty registry from truncating a good snapshot.
    std::size_t rows = 0, width = 0;
    for (const auto& [bank, scores] : registry) {
        (void)bank;
        if (!scores->empty()) {
            ++rows;
            width = scores->size();
        }
    }
    if (rows == 0) { return; }
    std::FILE* file = std::fopen(path, "wb");
    if (file == nullptr) { return; }
    for (const auto& [bank, scores] : registry) {
        (void)bank;
        if (scores->size() == width) {
            std::fwrite(scores->data(), sizeof(float), scores->size(), file);
        }
    }
    std::fclose(file);
}

void register_profile_dump(std::uint32_t bank, const std::vector<float>* scores) {
    const char* path = std::getenv("NINFER_EXPERT_PROFILE_DUMP");
    if (path == nullptr || *path == '\0') { return; }
    std::lock_guard<std::mutex> lock(profile_mutex());
    profile_registry()[bank] = scores;
}
// port (sm_89): the switches that used to be refused on the gate/up/down form are applied per run
// now (see ensure_split / publish_split_run / freeze_split / stage_predicted): a tier bank, a GH3
// placement word, a freeze placement and a prediction region exist once per matrix, which is exactly
// what three pool runs need. The one mechanism that is NOT per run is the device-admitted ring:
//
//   device_admit_ring() copies one whole pool slot out of one whole tier slot and refuses unless
//   `host_->slot_bytes() == cache_->slot_bytes()` (expert_stream.cpp, the "no mapped arena or slot
//   size mismatch" guard). A tier prices every bank at ONE slot size - slot_bytes_for(widest entry),
//   host_expert_residency.h - while a three-run layer has three different pool strides (0.87 MiB for
//   the q4 runs, 1.28 MiB for the q6 run). One uniform tier slot cannot be copied into all three, so
//   on this form the ring is the HOST-side one (a fixed seat ring the routes fill, NINFER_EXPERT_RING_SLOTS)
//   and NINFER_RING_DEVICE_ADMIT is reported as host-side rather than silently doing nothing.

} // namespace

// The request boundary (see the declaration in the header). attach() registers a stream here and
// ~ExpertStream() removes it again: the registry must only ever hold live streams, because a
// process that tears one Engine down and builds the next (a test, a serving reload) would
// otherwise leave freed streams in the list. The mutex exists because the boundary fires from
// the request path while a second model could be loading on another thread. The list itself is
// intentionally never destroyed: a deregistration during static teardown must not touch a
// vector that already died.
namespace {
std::mutex g_request_boundary_mutex;
std::vector<ExpertStream*>& request_boundary_streams() {
    static auto* streams = new std::vector<ExpertStream*>();
    return *streams;
}

// One freeze-window resolver per form, declared here because the stream's own defaults are read by
// init_stream_state(). The split form owns three per-run sets, so a short warm-up window is what
// stops each run from re-scanning its slots; the fused form owns one set and its measured default is
// always-rerank, because a frozen single set cannot admit a new route without a host read.
// NINFER_EXPERT_FREEZE overrides either default.
std::uint32_t freeze_after_for_form(bool split) {
    if (const char* value = std::getenv("NINFER_EXPERT_FREEZE"); value != nullptr && *value != '\0') {
        return static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10));
    }
    return split ? std::uint32_t{4} : std::uint32_t{0};
}
} // namespace

void note_expert_request_begin() {
    std::lock_guard<std::mutex> guard(g_request_boundary_mutex);
    for (auto* stream : request_boundary_streams()) { stream->note_request_begin(); }
}

ExpertStream::ExpertStream(std::uint32_t count, WeightBankLayout gate_up, WeightBankLayout down,
                           WeightGeometry gate_up_parent, WeightGeometry down_parent,
                           std::uint64_t gate_up_object, std::uint64_t down_object)
    : ExpertStream(count, gate_up, down, std::move(gate_up_parent), std::move(down_parent),
                   std::vector<std::uint64_t>(count, gate_up_object),
                   std::vector<std::uint64_t>(count, down_object)) {}

ExpertStream::ExpertStream(std::uint32_t count, WeightBankLayout gate_up, WeightBankLayout down,
                           WeightGeometry gate_up_parent, WeightGeometry down_parent,
                           std::vector<std::uint64_t> gate_up_objects,
                           std::vector<std::uint64_t> down_objects)
    : count_(count), gate_up_(gate_up), down_(down), gate_up_parent_(std::move(gate_up_parent)),
      down_parent_(std::move(down_parent)), gate_up_objects_(std::move(gate_up_objects)),
      down_objects_(std::move(down_objects)) {
    if (gate_up_objects_.size() != count_ || down_objects_.size() != count_) {
        throw std::invalid_argument("expert stream needs one object per expert");
    }
    // A bank whose experts all live in one object is the fused layout: the loader describes that
    // object once and addresses experts by row range.
    per_expert_objects_ =
        std::any_of(gate_up_objects_.begin(), gate_up_objects_.end(),
                    [&](std::uint64_t object) { return object != gate_up_objects_.front(); }) ||
        std::any_of(down_objects_.begin(), down_objects_.end(),
                    [&](std::uint64_t object) { return object != down_objects_.front(); });
    if (std::getenv("NINFER_EXPERT_TIMING") != nullptr) {
        // One line, once: which artifact layout this bank is built from. The two are described and
        // materialized differently (per-expert objects vs one fused object addressed by row range),
        // and without this line a run cannot be attributed to either.
        static bool artifact_logged = false;
        if (!artifact_logged) {
            artifact_logged = true;
            std::fprintf(stderr, "[experts] artifact layout: %s\n",
                         per_expert_objects_ ? "per-expert objects (e2)"
                                             : "fused (one object per bank)");
            std::fflush(stderr);
        }
    }
    const std::array<std::uint64_t, 2> gate_up_shape{gate_up_.region.rows, gate_up_.columns};
    const std::array<std::uint64_t, 2> down_shape{down_.region.rows, down_.columns};
    gate_up_expert_ = weight_geometry(gate_up_.format, gate_up_.layout, gate_up_shape);
    down_expert_    = weight_geometry(down_.format, down_.layout, down_shape);
    init_stream_state();
}

// port (sm_89): the gate/up/down form. Each descriptor is one artifact bank (its own layout, parent
// and one object per expert), and the three expert geometries are derived from each layout exactly
// as the fused form derives its two: `region.rows` is the rows of one expert, `columns` its width.
ExpertStream::ExpertStream(std::uint32_t count, ExpertMatrixBankDesc gate, ExpertMatrixBankDesc up,
                           ExpertMatrixBankDesc down)
    : count_(count) {
    split_banks_      = true;
    split_[0].desc    = std::move(gate);
    split_[1].desc    = std::move(up);
    split_[2].desc    = std::move(down);
    for (std::size_t run = 0; run < split_.size(); ++run) {
        auto& entry = split_[run];
        if (entry.desc.objects.size() != count_) {
            throw std::invalid_argument("expert stream needs one object per expert per bank");
        }
        const std::array<std::uint64_t, 2> shape{entry.desc.layout.region.rows,
                                                 entry.desc.layout.columns};
        entry.expert = weight_geometry(entry.desc.layout.format, entry.desc.layout.layout, shape);
        // The parent is metadata for a loader that names the objects, but it must describe the same
        // matrix the layout does: a parent of another format or width would size the operator
        // against one matrix and stream another.
        if (entry.desc.parent.format != entry.desc.layout.format ||
            entry.desc.parent.layout != entry.desc.layout.layout ||
            entry.desc.parent.shape.size() != 2 ||
            entry.desc.parent.shape.back() != entry.desc.layout.columns) {
            throw std::invalid_argument(
                "expert stream: a bank's parent does not describe its layout");
        }
        entry.offsets.assign(count_, 0);
        entry.mask.assign(count_, 0);
    }
    // Every bank stores its experts as its own objects on this artifact; the flag is derived the
    // same way as the fused form's so a bank that repeats one handle stays describable.
    per_expert_objects_ = false;
    for (const auto& run : split_) {
        for (const auto object : run.desc.objects) {
            if (object != run.desc.objects.front()) {
                per_expert_objects_ = true;
                break;
            }
        }
    }
    if (std::getenv("NINFER_EXPERT_TIMING") != nullptr) {
        static bool artifact_logged = false;
        if (!artifact_logged) {
            artifact_logged = true;
            std::fprintf(stderr, "[experts] artifact layout: %s\n",
                         "gate/up/down, one object per expert per bank");
            std::fflush(stderr);
        }
    }
    init_stream_state();
}

// The ranking buffer, the per-expert keys and the policy switches: identical for both forms, so the
// fused path and the three-run path cannot disagree about a knob.
void ExpertStream::init_stream_state() {
    keys_.reserve(count_);
    for (std::uint32_t expert = 0; expert < count_; ++expert) {
        keys_.push_back(ExpertCacheKey{0, expert});
    }
    hot_score_.assign(count_, 0.0F);
    // The policy is chosen per process: NINFER_EXPERT_POLICY=slru switches the streaming path to the
    // 50/50 segmented LRU. Everything measured so far was taken with the score fill, so the default
    // stays there until this one is measured against it.
    epoch_steps_ = [] {
        const char* value = std::getenv("NINFER_EXPERT_EPOCH");
        return value != nullptr ? static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10)) : 0U;
    }();
    auto_slru_ = [] {
        const char* value = std::getenv("NINFER_EXPERT_AUTO_SLRU");
        return value == nullptr || value[0] != 0;
    }();
    regions_ = [] {
        if (const char* value = std::getenv("NINFER_EXPERT_REGIONS");
            value != nullptr && *value != '\0') {
            return value[0] != '0';
        }
        // Q20: host-direct is the path with a device-side second leg, and the fixed-S ring is what
        // gives that path an R2 region whose shape does not move. Measured at 8 GiB: S = 8 moved
        // 8.4% of routes off the host leg (21.1% -> 12.7%) at neutral throughput (46.6 vs 46.4);
        // S = 16 and 32 cost 1-5% for little more coverage, so the default stays the same fixed few
        // slots the split path already measured. NINFER_EXPERT_RING_SLOTS overrides; 0 disables.
        return std::getenv("NINFER_HOST_DIRECT") != nullptr;
    }();
    slru_policy_ = [] {
        const char* value = std::getenv("NINFER_EXPERT_POLICY");
        return value != nullptr && std::strcmp(value, "slru") == 0;
    }();
    static_warm_ = std::getenv("NINFER_WARM_STATIC") != nullptr &&
                   std::getenv("NINFER_HOST_WARM") != nullptr &&
                   std::getenv("NINFER_HOST_DIRECT") != nullptr;
    // P1: the per-layer frozen set. A separate mode rather than a sixth environment variable to
    // remember: with NINFER_HOST_FREEZE the static set is not "a feature you can also enable", it is
    // the placement, and a run that forgot one of the three switches above would silently be the
    // streaming path wearing a freeze label. The mode implies the warm tier; the caller still owns
    // the arena and the budget.
    freeze_mode_ = [] {
        const char* value = std::getenv("NINFER_HOST_FREEZE");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    if (freeze_mode_) { static_warm_ = true; }
    freeze_after_ = [] {
        const char* value = std::getenv("NINFER_EXPERT_FREEZE");
        // The form is not known until attach_split(), so the default is the split form's window
        // here and the fused form's zero is applied at its own use site
        // (freeze_after_for_form(/*split=*/false)). An explicit value applies to both.
        return value != nullptr && *value != '\0'
                   ? static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10))
                   : freeze_after_for_form(/*split=*/true);
    }();
    freeze_refresh_ = [] {
        const char* value = std::getenv("NINFER_FREEZE_REFRESH");
        return value != nullptr && *value != '\0'
                   ? static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10))
                   : 0U;
    }();
    thaw_on_request_ = [] {
        const char* value = std::getenv("NINFER_FREEZE_THAW_REQUEST");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
}

ExpertStream::~ExpertStream() {
    // port (sm_89): both process-wide registries hold raw pointers into this object, and the old
    // "registration is process-lifetime" assumption behind them is only true for a process that
    // never tears an Engine down. ninfer_qwen3_5_dflash_real_test builds a Vision Engine, destroys
    // it, then builds the next one; the destroyed Engine's streams stayed registered, and the
    // next Engine's first admitted request walked them: note_expert_request_begin() called
    // note_request_begin() on freed memory, where the reused bytes made every thaw() guard read
    // as true, and static_member_.assign() then wrote into and freed a stale heap block - c0000374
    // at the next free (RTX 4090, cdb: note_expert_request_begin+0x116 -> vector<uint8_t>::assign
    // -> free_base). Unregister here so a registry only ever holds streams that are still alive.
    if (boundary_registered_) {
        std::lock_guard<std::mutex> guard(g_request_boundary_mutex);
        auto& streams = request_boundary_streams();
        const auto it = std::find(streams.begin(), streams.end(), this);
        if (it != streams.end()) { streams.erase(it); }
        boundary_registered_ = false;
    }
    // The profile rows are borrowed from the live streams too, and the registry is keyed by bank:
    // a second Engine reuses bank numbers, so only drop entries that point at THIS stream's row.
    std::lock_guard<std::mutex> guard(profile_mutex());
    auto& registry = profile_registry();
    for (auto it = registry.begin(); it != registry.end();) {
        if (it->second == &hot_score_) {
            it = registry.erase(it);
        } else {
            ++it;
        }
    }
    // S2: the slot-table mirror and the lookup scratch are this object's own cudaMalloc'd memory, so
    // they go with it (the registries above never reference them).
    release_slot_mirror_device();
    if (lookup_slots_dev_ != nullptr) {
        if (const auto status = cudaFree(lookup_slots_dev_); status != cudaSuccess) {
            std::fprintf(stderr, "[pf-mirror] cudaFree(id->slot scratch) failed: %s\n",
                         cudaGetErrorString(status));
            std::fflush(stderr);
        }
        lookup_slots_dev_ = nullptr;
        lookup_capacity_  = 0;
    }
    if (lookup_misses_dev_ != nullptr) {
        if (const auto status = cudaFree(lookup_misses_dev_); status != cudaSuccess) {
            std::fprintf(stderr, "[pf-mirror] cudaFree(miss counter) failed: %s\n",
                         cudaGetErrorString(status));
            std::fflush(stderr);
        }
        lookup_misses_dev_ = nullptr;
    }
    // P1: the prediction ring is this object's own Device memory too, and it is read by a kernel, so it
    // goes with the stream rather than with any host container.
    release_pred_ring();
}

void ExpertStream::attach_host(std::shared_ptr<HostExpertResidency> host) noexcept {
    host_      = host.get();
    host_weak_ = std::move(host);
    // Q53: the second-chance band's admission compares this bank's score against the occupant it would
    // displace, so the tier needs the ranking. The vector is allocated once in the constructor, so
    // the pointer stays valid for the stream's life. On the three-run form every run is its own tier
    // bank, and each one's fill is scored by the same ranking (the router names the expert, not the
    // matrix), so all three registers point at this stream's row.
    if (host_ != nullptr) {
        host_->attach_scores(bank_, hot_score_.data(), count_);
        if (split_banks_) {
            for (const auto& run : split_) {
                host_->attach_scores(run.bank, hot_score_.data(), count_);
            }
        }
    }
    // Q20/B: the static stripe must exist before the first dynamic fill - configuring it later moves
    // the dynamic bump pointer under a pointer that has already advanced, and the next eviction then
    // scans untouched slots and reports "nothing to evict" (the tier now refuses that ordering
    // outright). The quota is a process-wide decision, so the first stream installs it and the rest
    // are no-ops. The freeze route always installs it: its S stripe is the placement, not an option.
    //
    // The three-run form keeps its frozen placement in the pool (freeze_split) and lets the tier's
    // dynamic range serve the rest: the stripe's quota is ONE number per bank, while these three runs
    // have three different complement sizes (count - capacity(bank)), so a single stripe would either
    // overflow the narrowest run or waste the widest. The dynamic range has no such shape.
    if (host_ != nullptr && !split_banks_ && static_warm_ &&
        (freeze_mode_ || std::getenv("NINFER_WARM_STATIC_RESERVE") != nullptr)) {
        host_->configure_static_slots(static_quota());
    }
}

void ExpertStream::attach(ExpertCache& cache, std::uint32_t bank, const ExpertBankLayout& entry) {
    if (split_banks_) {
        throw std::logic_error("expert stream: attach() describes one fused bank; this stream was "
                               "built from gate/up/down - use attach_split()");
    }
    register_profile_dump(bank, &hot_score_);
    if (bank > 0xffffU || cache.capacity_bytes() < entry.entry_bytes) {
        throw std::invalid_argument("expert stream cannot attach to this cache");
    }
    cache_ = &cache;
    bank_  = bank;
    entry_ = entry;
    // Once attached, this bank can act on the request boundary. attach() runs once per stream at
    // model wiring, so the flag also guards a hypothetical second call.
    if (!boundary_registered_) {
        boundary_registered_ = true;
        std::lock_guard<std::mutex> guard(g_request_boundary_mutex);
        request_boundary_streams().push_back(this);
    }
    if (host_ != nullptr) { host_->attach_scores(bank_, hot_score_.data(), count_); }
    for (auto& key : keys_) { key.bank = bank; }
    

    const auto pool = cache.pool();
    const auto base = static_cast<const std::byte*>(pool.data);
    auto& gate_up_view = experts_.bank.gate_up;
    gate_up_view.codes = base + entry.codes.offset;
    gate_up_view.high  = entry.high.bytes ? base + entry.high.offset : nullptr;
    gate_up_view.scales = base + entry.scales.offset;
    gate_up_view.code_stride  = cache.slot_stride(bank);
    gate_up_view.high_stride  = entry.high.bytes ? cache.slot_stride(bank) : 0;
    gate_up_view.scale_stride = cache.slot_stride(bank);
    auto& down_view   = experts_.bank.down;
    down_view         = gate_up_view;
    down_view.codes   = base + entry.down_codes.offset;
    down_view.high    = entry.down_high.bytes ? base + entry.down_high.offset : nullptr;
    down_view.scales  = base + entry.down_scales.offset;
    down_view.code_stride  = cache.slot_stride(bank);
    down_view.high_stride  = entry.down_high.bytes ? cache.slot_stride(bank) : 0;
    down_view.scale_stride = cache.slot_stride(bank);
    experts_.bank.count  = count_;
    experts_.bank.device = pool.device;

    // Publish the per-expert offset table. port (sm_89): the pool is a flat slot space, so an entry
    // is its global slot times the slot stride; the kernel reads its plane base through this table,
    // so a partial resident set only rewrites the entries.
    offsets_.assign(count_, 0);
    std::vector<std::int32_t> mask(count_, 0);
    for (std::uint32_t expert = 0; expert < count_; ++expert) {
        const auto slot = cache.expert_slot(bank, expert);
        if (!slot) { continue; }
        offsets_[expert] = cache.slot_offset(static_cast<std::uint32_t>(*slot));
        mask[expert]     = 1;
    }
    const auto table_bytes = offsets_.size() * sizeof(std::uint64_t);
    offsets_device_        = std::make_unique<DeviceBuffer>(table_bytes);
    const auto copied      = cudaMemcpy(offsets_device_->p, offsets_.data(), table_bytes,
                                        cudaMemcpyHostToDevice);
    if (copied != cudaSuccess) {
        throw std::runtime_error(std::string("expert offset table: ") + cudaGetErrorString(copied));
    }
    const auto* table = static_cast<const std::uint64_t*>(offsets_device_->p);
    gate_up_view.offsets = table;
    down_view.offsets    = table;
    // Indexed addressing names its backing store, so the pool has to travel with the table.
    gate_up_view.indexed_storage = pool;
    down_view.indexed_storage    = pool;

    // Publish which experts the pool will hold. With room for the whole layer every entry is set;
    // otherwise only the experts that own a slot are, and the caller masks the rest out of the GPU
    // job list.
    mask_device_ = std::make_unique<DeviceBuffer>(mask.size() * sizeof(std::int32_t));
    mask_host_   = mask;
    const auto mask_bytes = mask.size() * sizeof(std::int32_t);
    const auto mask_copy  = cudaMemcpy(mask_device_->p, mask.data(), mask_bytes,
                                       cudaMemcpyHostToDevice);
    if (mask_copy != cudaSuccess) {
        throw std::runtime_error(std::string("expert residency mask: ") + cudaGetErrorString(mask_copy));
    }
    // Q20: the ring was reserved by the cache at construction (a slice of each bank's window).
    // Read its size here; configuring it after seeding would need slots the pool does not have.
    device_ring_slots_ = cache.ring_slot_count(bank);
    // port (sm_89): with a fixed slot run per bank the resident set never changes, so this stream
    // can skip the routing round trip entirely and a captured graph stays valid.
    pinned_ = cache.can_hold_every_layer();
}

void ExpertStream::rebase_window() {
    // port (sm_89): the pool is a flat slot space, so a bank has no per-bank origin to add; only
    // the pool base and the entry's plane offsets matter, and both are fixed for the pool's life.
    const auto base = static_cast<const std::byte*>(cache_->pool().data);
    experts_.bank.gate_up.codes  = base + entry_.codes.offset;
    experts_.bank.gate_up.high   = entry_.high.bytes ? base + entry_.high.offset : nullptr;
    experts_.bank.gate_up.scales = base + entry_.scales.offset;
    experts_.bank.down.codes     = base + entry_.down_codes.offset;
    experts_.bank.down.high      = entry_.down_high.bytes ? base + entry_.down_high.offset : nullptr;
    experts_.bank.down.scales    = base + entry_.down_scales.offset;
}

// port (sm_89): one run of the three-run form. The view this builds is exactly what the operator
// reads for that matrix: plane bases inside the run's entry image, the run's own slot stride, the
// per-expert offset table this stream publishes, and the pool as the indexed backing.
void ExpertStream::attach_split_one(std::size_t index, ExpertCache& cache, std::uint32_t bank,
                                    const ExpertBankLayout& entry) {
    auto& run  = split_[index];
    const auto pool = cache.pool();
    const auto base = static_cast<const std::byte*>(pool.data);
    const auto stride = cache.slot_stride(bank);
    auto& view = mutable_view(index);
    view = ExpertMatrixDeviceView{};
    view.codes  = base + entry.codes.offset;
    view.high   = entry.high.bytes ? base + entry.high.offset : nullptr;
    view.scales = entry.scales.bytes ? base + entry.scales.offset : nullptr;
    view.code_stride  = stride;
    view.high_stride  = entry.high.bytes ? stride : 0;
    view.scale_stride = entry.scales.bytes ? stride : 0;
    run.bank      = bank;
    run.entry     = entry;
    run.attached  = true;
    // Publish the table the pool already seeds: the cache gives a fresh bank the identity mapping
    // up to its seat count, so a stream attached to a pinned pool is addressable before the first
    // ensure() - the same contract the fused attach() has.
    run.offsets.assign(count_, 0);
    run.mask.assign(count_, 0);
    for (std::uint32_t expert = 0; expert < count_; ++expert) {
        const auto slot = cache.expert_slot(bank, expert);
        if (!slot) { continue; }
        run.offsets[expert] = cache.slot_offset(static_cast<std::uint32_t>(*slot));
        run.mask[expert]    = 1;
    }
    run.resident = static_cast<std::uint32_t>(
        std::count(run.mask.begin(), run.mask.end(), std::int32_t{1}));
    run.offsets_device = std::make_unique<DeviceBuffer>(run.offsets.size() * sizeof(std::uint64_t));
    run.mask_device    = std::make_unique<DeviceBuffer>(run.mask.size() * sizeof(std::int32_t));
    const auto table_bytes = run.offsets.size() * sizeof(std::uint64_t);
    const auto mask_bytes  = run.mask.size() * sizeof(std::int32_t);
    auto status = cudaMemcpy(run.offsets_device->p, run.offsets.data(), table_bytes,
                             cudaMemcpyHostToDevice);
    if (status == cudaSuccess) {
        status = cudaMemcpy(run.mask_device->p, run.mask.data(), mask_bytes,
                            cudaMemcpyHostToDevice);
    }
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("expert split tables: ") + cudaGetErrorString(status));
    }
    view.offsets         = static_cast<const std::uint64_t*>(run.offsets_device->p);
    view.indexed_storage = pool;
    // The generation the two tables now hold: the cache's own counters for this run, so the A3b
    // check in ensure_split() compares like with like (the same rule the fused form uses).
    run.pub_hot = run.table_hot = cache.hot_generation(bank);
    run.pub_step = run.table_step = cache.step_generation(bank);
    run.pub_prefetch = run.table_prefetch = cache.prefetch_generation(bank);
    run.published = true;
}

void ExpertStream::attach_split(ExpertCache& cache, std::uint32_t gate_bank,
                                const ExpertBankLayout& gate_entry, std::uint32_t up_bank,
                                const ExpertBankLayout& up_entry, std::uint32_t down_bank,
                                const ExpertBankLayout& down_entry) {
    if (!split_banks_) {
        throw std::logic_error("expert stream: attach_split() requires the gate/up/down form");
    }
    // A tier attached before the cache is kept: it is the second leg of these three runs.
    const std::array<std::uint32_t, 3> banks{gate_bank, up_bank, down_bank};
    const std::array<const ExpertBankLayout*, 3> entries{&gate_entry, &up_entry, &down_entry};
    for (std::size_t index = 0; index < banks.size(); ++index) {
        if (banks[index] > 0xffffU || cache.capacity_bytes() < entries[index]->entry_bytes) {
            throw std::invalid_argument("expert stream cannot attach to this cache");
        }
        if (banks[index] == banks[(index + 1) % 3]) {
            throw std::invalid_argument("expert stream needs three distinct cache banks");
        }
    }
    // The gate run is the layer's identity: diagnostics, graph keys and the score registry all name
    // one number per layer, and the router's own matrix is the one they mean.
    register_profile_dump(gate_bank, &hot_score_);
    cache_ = &cache;
    bank_  = gate_bank;
    for (auto& key : keys_) { key.bank = gate_bank; }
    if (!boundary_registered_) {
        boundary_registered_ = true;
        std::lock_guard<std::mutex> guard(g_request_boundary_mutex);
        request_boundary_streams().push_back(this);
    }
    for (std::size_t index = 0; index < banks.size(); ++index) {
        attach_split_one(index, cache, banks[index], *entries[index]);
    }
    // The fused view must stay empty: the operator refuses a request that names both forms at once,
    // and a leftover pointer here would be exactly that request.
    experts_.bank.gate_up = ExpertMatrixDeviceView{};
    experts_.bank.count   = count_;
    experts_.bank.device  = cache.pool().device;
    // port (sm_89): the device-admitted ring is the one mechanism that is not per run - it copies a
    // whole pool slot out of a whole tier slot and needs both to be the same image, which a tier with
    // one slot size cannot be for three different pool strides. The host-side ring (a fixed seat
    // slice the routes fill) is what this form uses; see the note at the top of ensure_split.
    device_ring_slots_ = 0;
    if (std::getenv("NINFER_RING_DEVICE_ADMIT") != nullptr) {
        static bool announced = false;
        if (!announced) {
            announced = true;
            std::fprintf(stderr,
                         "[ring] NINFER_RING_DEVICE_ADMIT: host-side admission on the gate/up/down "
                         "form - a device-admitted ring copies one whole pool slot from one whole "
                         "tier slot, and this tier prices every bank at a single slot size while "
                         "these three runs have three strides\n");
            std::fflush(stderr);
        }
    }
    pinned_ = cache.can_hold_every_layer();
}

void ExpertStream::rebase_split() {
    // The pool is one flat reservation, so no plane base moves; re-reading them keeps the views
    // honest against a re-attach or a pool that was rebased before this layer (the fused path's
    // rebase_window has the same contract).
    const auto pool = cache_->pool();
    const auto base = static_cast<const std::byte*>(pool.data);
    for (std::size_t index = 0; index < split_.size(); ++index) {
        auto& run  = split_[index];
        auto& view = mutable_view(index);
        const auto stride = cache_->slot_stride(run.bank);
        view.codes  = base + run.entry.codes.offset;
        view.high   = run.entry.high.bytes ? base + run.entry.high.offset : nullptr;
        view.scales = run.entry.scales.bytes ? base + run.entry.scales.offset : nullptr;
        view.code_stride  = stride;
        view.high_stride  = run.entry.high.bytes ? stride : 0;
        view.scale_stride = run.entry.scales.bytes ? stride : 0;
        view.offsets         = static_cast<const std::uint64_t*>(run.offsets_device->p);
        view.indexed_storage = pool;
    }
    experts_.bank.gate_up = ExpertMatrixDeviceView{};
    experts_.bank.count   = count_;
    experts_.bank.device  = pool.device;
}

const ExpertMatrixBankDesc& ExpertStream::split_desc(ExpertMatrixRole role) const {
    if (!split_banks_) {
        throw std::logic_error("expert stream: this stream is not in the gate/up/down form");
    }
    return split_[split_index(role)].desc;
}

const ExpertMatrixDeviceView& ExpertStream::view_for(std::size_t index) const noexcept {
    if (index == split_index(ExpertMatrixRole::Gate)) { return experts_.gate; }
    if (index == split_index(ExpertMatrixRole::Up)) { return experts_.up; }
    return experts_.bank.down;
}

ExpertMatrixDeviceView& ExpertStream::mutable_view(std::size_t index) noexcept {
    if (index == split_index(ExpertMatrixRole::Gate)) { return experts_.gate; }
    if (index == split_index(ExpertMatrixRole::Up)) { return experts_.up; }
    return experts_.bank.down;
}

std::uint32_t ExpertStream::split_resident_capacity(ExpertMatrixRole role) const noexcept {
    if (!split_banks_ || cache_ == nullptr) { return 0; }
    return cache_->resident_capacity(split_[split_index(role)].bank);
}

std::uint32_t ExpertStream::spill_slots(std::uint32_t capacity) const noexcept {
    // Still a fixed eight, and the measurement below says why: on the 35B at a 12 GiB pool the
    // measured p99 of a decode step's churn against the previous set is exactly eight - one token's
    // routing - so a quantile rule reproduces the constant and buys nothing, and the earlier report
    // measured a *smaller* measured size (six) at 63-68 tok/s against 82-86 for eight. The histogram
    // is therefore a diagnostic: it now holds real counts (it used to read zero by construction, see
    // the sampling site), and a future decision - for example one per generation rather than per step
    // - can read it without changing this placement.
    return std::min<std::uint32_t>(8U, std::max<std::uint32_t>(2U, capacity / 3));
}

void ExpertStream::refresh_spill_measure(std::uint32_t capacity) noexcept {
    // Sixteen decode layers per window; the histogram itself decays by halves every 256 samples, so a
    // window reads the recent tail rather than the whole run.
    constexpr std::uint32_t kWindow = 16;
    if (++spill_window_samples_ < kWindow) { return; }
    spill_window_samples_ = 0;
    ++spill_windows_;
    std::uint64_t total = 0;
    for (const auto bucket : miss_hist_) { total += bucket; }
    if (total < kWindow) { return; }
    // p99, not p95: the region has to hold the tail of a step, and the two quantiles differ by one
    // slot while getting it wrong costs a host read per miss.
    const auto threshold =
        static_cast<std::uint64_t>(static_cast<double>(total) * 0.99 + 0.5);
    std::uint64_t cumulative = 0;
    std::uint32_t p99        = 0;
    for (std::uint32_t bucket = 0; bucket < miss_hist_.size(); ++bucket) {
        cumulative += miss_hist_[bucket];
        if (cumulative >= threshold) {
            p99 = bucket;
            break;
        }
    }
    const std::uint32_t upper = std::max<std::uint32_t>(2U, capacity / 2);
    spill_measured_           = std::min(std::max<std::uint32_t>(2U, p99), upper);
    if (bank_ == 0 && std::getenv("NINFER_SPILL_TRACE") != nullptr) {
        std::fprintf(stderr, "[spill] window=%u p99=%u (ring stays %u, capacity %u)\n",
                     spill_windows_, p99, spill_slots(capacity), capacity);
        std::fflush(stderr);
    }
}

bool ExpertStream::fully_resident() const noexcept {
    // This stream's own bank, not the narrowest one: a wider bank may hold its whole layer while a
    // narrower one cannot, and the answer decides whether this layer needs the split path at all.
    if (split_banks_) {
        // Three runs, so the layer is only resident when all three can hold it: the narrowest run
        // (the q6 down matrix) is the binding one and it is read from its own seats.
        if (cache_ == nullptr) { return false; }
        for (const auto& run : split_) {
            if (cache_->resident_capacity(run.bank) < count_) { return false; }
        }
        return true;
    }
    return cache_ != nullptr && cache_->resident_capacity(bank_) >= count_;
}

// S1/C1: two mirrors, one per table, each with its own generation.
//
// The PLACEMENT mirror (expert -> pool slot, export_slots) is versioned by step_generation(bank): the
// counter the pool bumps whenever a bank's set changes. It legitimately re-exports every layer on a
// streaming pool, because the addresses really did move - that table answers "where are this expert's
// bytes", which is a question about the resident set.
//
// The REGION mirror (region slot -> expert, export_prefetch_owners) is versioned by
// prefetch_generation(bank): the counter the pool bumps only when set_prefetch_experts() places
// something. That table answers "what is in the prediction ring", which is a question about a
// placement, not about residency. Before this split both questions read the placement table with the
// step generation, so the fill path's view of the region was only as fresh as the last residency move
// and the mirror re-exported once per bank per layer (measured 9921 re-exports on a decode run).
//
// S2: each mirror is also uploaded to one contiguous Device image, in the same bank order the host
// accumulated it in, so `mirror.dev + mirror.off[bank]` is exactly the table the export returned.
// Nothing here decides placement or size: the bank count comes from the pool, each segment's length
// from the export's return value, and the offsets from those lengths.
namespace {
// `~0` is never a generation the pool reports (they start at 0 and only move up), so it means "the
// Device copy is not the staged generation" for both "never uploaded" and "the buffer was rebuilt".
constexpr std::uint64_t kMirrorStaleGeneration = std::numeric_limits<std::uint64_t>::max();
constexpr std::size_t   kMirrorProbeEntries   = 4096; // export reports the real count and we resize

// C5: the read-ahead's logical clock. One tick per offer and ONE clock for the process, because the
// forty banks are forty layers of the same step: a per-bank clock would let one bank's backlog expire
// against another's progress. Both the demand path's offer (ensure) and the prediction's fill ask for
// it here, so the two cannot drift into two windows.
std::atomic<std::uint64_t>& prefetch_layer_tick() {
    static std::atomic<std::uint64_t> tick{0};
    return tick;
}
std::uint64_t prefetch_window() {
    static const std::uint64_t window = [] {
        const char* value = std::getenv("NINFER_PREFETCH_WINDOW");
        if (value != nullptr && *value != '\0') {
            return static_cast<std::uint64_t>(std::strtoull(value, nullptr, 10));
        }
        return std::uint64_t{40};
    }();
    return window;
}
// Defined with the prediction ring below; declared here because stage_predicted() asks whether the
// `prev` signal owns this hand-off (it does, and stage_history() already offered the read).
bool history_signal_switched();
// The mode in which a prediction region is actually staged. Read here as well as in stage_predicted()
// because the offer of a read is only worth making on a path that will place the bytes: a read-ahead
// into a region nothing stages is pure I/O.
bool prefetch_vram_mode();
} // namespace

void ExpertStream::release_mirror_device(SlotMirror& mirror) noexcept {
    if (mirror.dev != nullptr) {
        const auto status = cudaFree(mirror.dev);
        if (status != cudaSuccess) {
            std::fprintf(stderr, "[pf-mirror] cudaFree(%zu bytes) failed: %s\n", mirror.bytes,
                         cudaGetErrorString(status));
            std::fflush(stderr);
            ++mirror.dev_fails;
        }
        mirror.dev = nullptr;
    }
    mirror.bytes = 0;
    std::fill(mirror.dev_gen.begin(), mirror.dev_gen.end(), kMirrorStaleGeneration);
    std::fill(mirror.dev_off.begin(), mirror.dev_off.end(), 0U);
}

void ExpertStream::release_slot_mirror_device() noexcept {
    release_mirror_device(placement_mirror_);
    release_mirror_device(prefetch_mirror_);
}

bool ExpertStream::serves_bank(std::uint32_t bank) const noexcept {
    if (split_.empty()) { return bank == bank_; }
    for (const auto& run : split_) {
        if (run.bank == bank) { return true; }
    }
    return false;
}

const std::uint32_t* ExpertStream::mirror_segment(const SlotMirror& mirror, std::uint32_t bank,
                                                  std::uint32_t* entries) const noexcept {
    if (entries != nullptr) { *entries = 0; }
    if (mirror.dev == nullptr || !serves_bank(bank) || bank >= mirror.off.size()) {
        return nullptr;
    }
    // Also require the Device copy to be this bank's current generation: a pointer to bytes that the
    // last upload left behind would answer every id with the previous placement's slot, which is worse
    // than no answer at all.
    if (bank >= mirror.dev_gen.size() || mirror.dev_gen[bank] != mirror.gen[bank]) {
        return nullptr;
    }
    if (entries != nullptr) {
        *entries = bank < mirror.host.size() ? static_cast<std::uint32_t>(mirror.host[bank].size())
                                             : 0U;
    }
    return mirror.dev + mirror.off[bank];
}

const std::uint32_t* ExpertStream::slot_mirror_segment(std::uint32_t bank, std::uint32_t* entries,
                                                       ops::RouteTableKind kind) const noexcept {
    return kind == ops::RouteTableKind::PrefetchRegion
               ? mirror_segment(prefetch_mirror_, bank, entries)
               : mirror_segment(placement_mirror_, bank, entries);
}

void ExpertStream::sync_mirror(SlotMirror& mirror, ops::RouteTableKind kind, cudaStream_t stream) {
    if (cache_ == nullptr) { return; }
    const bool region = kind == ops::RouteTableKind::PrefetchRegion;
    if (region && cache_->prefetch_slots_per_bank() == 0) {
        // No region was carved, so there is no table to mirror. Staging one empty row per bank and
        // counting that as an upload would put 144 meaningless re-exports on the panel and make the
        // region's upload count unreadable exactly when it should read zero.
        release_mirror_device(mirror);
        mirror.gen.clear();
        mirror.host.clear();
        mirror.dev_gen.clear();
        mirror.dev_off.clear();
        mirror.off.clear();
        return;
    }
    const auto banks = cache_->bank_count();
    if (mirror.gen.size() != banks) {
        // A different bank count means every offset below is invalid, so the Device image goes too.
        release_mirror_device(mirror);
        mirror.gen.assign(banks, kMirrorStaleGeneration);
        mirror.host.assign(banks, {});
        mirror.dev_gen.assign(banks, kMirrorStaleGeneration);
        mirror.dev_off.assign(banks, 0);
        mirror.entries = 0;
    }
    // 1) Re-export the banks whose own generation moved, and only those. The generation is the pool's,
    //    and it is the one that belongs to THIS table: step_generation for the placement table,
    //    prefetch_generation for the region's.
    for (std::uint32_t bank = 0; bank < banks; ++bank) {
        const auto generation = region ? cache_->prefetch_generation(bank)
                                       : cache_->step_generation(bank);
        if (mirror.gen[bank] == generation) { continue; }
        auto& staged = mirror.host[bank];
        staged.assign(kMirrorProbeEntries, 0);   // probe buffer: the real count comes back from the export
        const auto entries = region ? cache_->export_prefetch_owners(bank, staged)
                                    : cache_->export_slots(bank, staged);
        staged.resize(entries);
        mirror.gen[bank] = generation;
        ++mirror.uploads;
        mirror.entries += entries;
    }
    // 2) The Device layout: bank order, each segment as long as that bank's export. Recomputed rather
    //    than cached, because an export that returns a different length must move every later segment.
    mirror.off.assign(banks, 0);
    std::size_t total = 0;
    for (std::uint32_t bank = 0; bank < banks; ++bank) {
        mirror.off[bank] = static_cast<std::uint32_t>(total);
        total += mirror.host[bank].size();
    }
    if (total == 0) { return; }
    // 3) (Re)allocate when the image is absent or a bank outgrew it. A rebuild makes every segment's
    //    Device copy stale, so each one is uploaded again below.
    if (mirror.dev == nullptr || total > mirror.bytes / sizeof(std::uint32_t)) {
        release_mirror_device(mirror);
        const std::size_t bytes = total * sizeof(std::uint32_t);
        void*  allocated        = nullptr;
        const auto status       = cudaMalloc(&allocated, bytes);
        if (status != cudaSuccess) {
            ++mirror.dev_fails;
            std::fprintf(stderr,
                         "[pf-mirror] cudaMalloc(%zu bytes for %zu entries over %u banks, %s) "
                         "failed: %s\n",
                         bytes, total, banks, region ? "region" : "placement",
                         cudaGetErrorString(status));
            std::fflush(stderr);
            return;
        }
        mirror.dev   = static_cast<std::uint32_t*>(allocated);
        mirror.bytes = bytes;
    }
    // 4) Upload the segments whose Device copy is not the staged generation - but only the banks this
    //    stream resolves (its own runs). The image is per stream and the pool has banks for every layer
    //    in it, so uploading the whole pool from each of them would copy the same bytes once per layer
    //    per step for tables no consumer of this stream ever reads. `serves_bank()` is the same test the
    //    lookup uses, so "current" and "readable" cannot drift apart.
    for (std::uint32_t bank = 0; bank < banks; ++bank) {
        if (!serves_bank(bank)) { continue; }
        const auto& staged = mirror.host[bank];
        if (staged.empty()) { continue; }
        if (mirror.dev_gen[bank] == mirror.gen[bank] && mirror.dev_off[bank] == mirror.off[bank]) {
            continue;
        }
        const auto bytes = staged.size() * sizeof(std::uint32_t);
        if (!slot_mirror_stage_ || slot_mirror_stage_->size() < bytes) {
            try {
                slot_mirror_stage_ = std::make_unique<PinnedHostBuffer>(bytes);
            } catch (const std::exception& error) {
                ++mirror.dev_fails;
                std::fprintf(stderr, "[pf-mirror] pinned staging (%zu bytes) failed: %s\n", bytes,
                             error.what());
                std::fflush(stderr);
                return;
            }
        }
        std::memcpy(slot_mirror_stage_->data(), staged.data(), bytes);
        const auto status = cudaMemcpyAsync(mirror.dev + mirror.off[bank],
                                            slot_mirror_stage_->data(), bytes,
                                            cudaMemcpyHostToDevice, stream);
        if (status != cudaSuccess) {
            // Leave the generation stale: the next pass retries instead of trusting a short copy.
            ++mirror.dev_fails;
            std::fprintf(stderr,
                         "[pf-mirror] cudaMemcpyAsync bank=%u entries=%zu -> dev+%u (%zu bytes, %s) "
                         "failed: %s\n",
                         bank, staged.size(), mirror.off[bank], bytes,
                         region ? "region" : "placement", cudaGetErrorString(status));
            std::fflush(stderr);
            continue;
        }
        mirror.dev_gen[bank] = mirror.gen[bank];
        mirror.dev_off[bank] = mirror.off[bank];
        ++mirror.dev_uploads;
        mirror.dev_bytes += bytes;
    }
}

void ExpertStream::sync_slot_mirror(cudaStream_t stream) {
    sync_mirror(placement_mirror_, ops::RouteTableKind::Placement, stream);
    sync_mirror(prefetch_mirror_, ops::RouteTableKind::PrefetchRegion, stream);
}


// S2-5/C2: the staging path's own consumer of the mirrors. The caller names WHICH table it is
// resolving against, because the two answers are in different spaces:
//
//   Placement      - pool slots (addresses). "Where are this expert's bytes."
//   PrefetchRegion - region indices (positions in this bank's ring, 0..slots-1). "Does the fill's
//                    target already hold this expert, and in which region slot."
//
// The routes the table cannot answer (id outside the bank, or nothing held) are counted where they
// happen and shown in the [pf-mirror] line. Explicit switch only on the demand path: the shipped
// placement does not consume the pool slots yet, so NINFER_SLOT_LOOKUP is the only thing that makes
// the placement lookup run - and that is what keeps the default path bit-identical to S1.
void ExpertStream::lookup_routed_slots(const void* device_ids, std::int32_t id_values,
                                       cudaStream_t stream, ops::RouteTableKind kind, bool armed) {
    static const bool switched = std::getenv("NINFER_SLOT_LOOKUP") != nullptr;
    const bool enabled         = switched || armed;
    if (!enabled || device_ids == nullptr || id_values <= 0) { return; }
    const bool region = kind == ops::RouteTableKind::PrefetchRegion;
    SlotMirror& mirror = region ? prefetch_mirror_ : placement_mirror_;
    if (mirror.dev == nullptr) {
        // No image means there is nothing to resolve against. Said out loud once, because a silently
        // skipped lookup reads exactly like a lookup that found every id.
        static bool said[2] = {false, false};
        if (!said[region ? 1 : 0]) {
            said[region ? 1 : 0] = true;
            std::fprintf(stderr,
                         "[pf-mirror] id->%s lookup armed with no Device image (region slots=%u)\n",
                         region ? "region" : "slot",
                         cache_ != nullptr ? cache_->prefetch_slots_per_bank() : 0U);
            std::fflush(stderr);
        }
        return;
    }
    const auto values = static_cast<std::size_t>(id_values);
    // Output scratch: one word per id, per space. Grown, never shrunk, freed with the stream.
    if (lookup_capacity_ < values) {
        for (auto** pointer : {&lookup_slots_dev_, &lookup_region_dev_}) {
            if (*pointer == nullptr) { continue; }
            const auto free_status = cudaFree(*pointer);
            *pointer              = nullptr;
            if (free_status != cudaSuccess) {
                ++mirror.dev_fails;
                std::fprintf(stderr, "[pf-mirror] cudaFree(id->slot scratch) failed: %s\n",
                             cudaGetErrorString(free_status));
                std::fflush(stderr);
                return;
            }
        }
        lookup_capacity_ = 0;
        const auto bytes = values * sizeof(std::uint32_t);
        for (auto** pointer : {&lookup_slots_dev_, &lookup_region_dev_}) {
            void*      allocated = nullptr;
            const auto status    = cudaMalloc(&allocated, bytes);
            if (status != cudaSuccess) {
                ++mirror.dev_fails;
                std::fprintf(stderr,
                             "[pf-mirror] cudaMalloc(%zu bytes, id->%s scratch) failed: %s\n", bytes,
                             pointer == &lookup_region_dev_ ? "region" : "slot",
                             cudaGetErrorString(status));
                std::fflush(stderr);
                return;
            }
            *pointer = static_cast<std::uint32_t*>(allocated);
        }
        lookup_capacity_ = values;
    }
    // One cumulative miss counter per space: zeroed once so the print can report a total, and never
    // reset, which keeps the read-back off the layer path.
    auto& miss_counter = region ? lookup_region_misses_dev_ : lookup_misses_dev_;
    if (miss_counter == nullptr) {
        void*      allocated = nullptr;
        const auto status    = cudaMalloc(&allocated, sizeof(std::uint32_t));
        if (status != cudaSuccess) {
            ++mirror.dev_fails;
            std::fprintf(stderr, "[pf-mirror] cudaMalloc(%s miss counter) failed: %s\n",
                         region ? "region" : "slot", cudaGetErrorString(status));
            std::fflush(stderr);
            return;
        }
        miss_counter = static_cast<std::uint32_t*>(allocated);
        if (const auto zeroed = cudaMemset(miss_counter, 0, sizeof(std::uint32_t));
            zeroed != cudaSuccess) {
            ++mirror.dev_fails;
            std::fprintf(stderr, "[pf-mirror] cudaMemset(%s miss counter) failed: %s\n",
                         region ? "region" : "slot", cudaGetErrorString(zeroed));
            std::fflush(stderr);
            return;
        }
    }
    auto& out    = region ? lookup_region_dev_ : lookup_slots_dev_;
    auto& calls  = region ? lookup_region_calls_ : lookup_calls_;
    // C2: the two spaces are stated once, at the point where their sizes are known, so a reader of the
    // panel can see that a region index is not a pool slot on this bank rather than take it on trust.
    static bool spaces_said = false;
    if (region && !spaces_said && cache_ != nullptr) {
        spaces_said = true;
        const auto first = split_.empty() ? bank_ : split_.front().bank;
        const auto slot0 = cache_->prefetch_region_slot(first, 0);
        std::fprintf(stderr,
                     "[pf-space] bank=%u region index space 0..%u | it is NOT a pool slot space "
                     "(region slot 0 names pool slot %u; this bank has %u experts)\n",
                     first, cache_->prefetch_slots_per_bank() - 1,
                     slot0 ? *slot0 : kAbsentSlot, count_);
        std::fflush(stderr);
    }
    // One lookup per matrix run of this layer: the three runs place independently, so an id can have a
    // slot in one and none in another - resolving only the gate run would hide exactly that. The
    // segment comes from the accessor, so a bank this stream does not keep current resolves to null and
    // is reported rather than read as stale bytes.
    std::size_t resolved_runs = 0;
    const auto resolve = [&](std::uint32_t bank) {
        std::uint32_t       entries = 0;
        const std::uint32_t* table  = mirror_segment(mirror, bank, &entries);
        if (table == nullptr || entries == 0) {
            ++mirror.dev_fails;
            std::fprintf(stderr,
                         "[pf-mirror] id->%s skipped for bank %u: no current Device segment "
                         "(served=%d, entries=%u)\n",
                         region ? "region" : "slot", bank, int(serves_bank(bank)), entries);
            std::fflush(stderr);
            return;
        }
        if (region) {
            ops::moe_route_region_slots(static_cast<const std::int32_t*>(device_ids),
                                        static_cast<int>(values), table, entries, out, miss_counter,
                                        stream);
        } else {
            ops::moe_route_slots(static_cast<const std::int32_t*>(device_ids),
                                 static_cast<int>(values), table, entries, out, miss_counter, stream);
        }
        ++calls;
    };
    if (split_.empty()) {
        resolve(bank_);
        resolved_runs = 1;
    } else {
        for (const auto& run : split_) { resolve(run.bank); }
        resolved_runs = split_.size();
    }
    // S3-3: a prediction set has to reach every matrix run the layer has, so the number of runs that
    // were actually fed is checked against what the layout reports instead of being assumed. A fused
    // layer has one run, the gate/up/down form three; a disagreement means a run is cold that the
    // layer will read, and it is said out loud rather than left to a hit rate.
    if (armed) {
        ++predicted_runs_resolved_;
        const auto expected = static_cast<std::size_t>(runs());
        if (resolved_runs != expected) {
            ++predicted_run_mismatches_;
            std::fprintf(stderr,
                         "[pf-pred] run mismatch on bank %u: fed %zu run(s), layout reports %u "
                         "(split=%d, ids=%zu)\n",
                         bank_, resolved_runs, runs(), int(split_banks_), values);
            std::fflush(stderr);
        }
    }
    if (const auto status = cudaGetLastError(); status != cudaSuccess) {
        ++mirror.dev_fails;
        std::fprintf(stderr, "[pf-mirror] id->%s launch (bank=%u, %zu ids) failed: %s\n",
                     region ? "region" : "slot", bank_, values, cudaGetErrorString(status));
        std::fflush(stderr);
    }
}

// C2: one space used as the other. The check needs both numbers, so it runs where both are on the
// host: `value` is a region index and `cache_->prefetch_region_slot(bank, value)` is the pool slot it
// names. Equal is the only correct answer for a value that came from the region table; anything else
// (an index outside the region, an index whose slot is not a region slot, or a pool slot passed where
// an index was expected) is counted and printed with both values.
void ExpertStream::note_slot_space(std::uint32_t bank, ops::RouteTableKind kind, std::uint32_t value) {
    if (cache_ == nullptr || value == kAbsentSlot) { return; }
    if (kind == ops::RouteTableKind::PrefetchRegion) {
        const auto region = cache_->prefetch_slots_per_bank();
        const auto slot   = cache_->prefetch_region_slot(bank, value);
        if (value >= region || !slot) {
            ++slot_space_mixes_;
            std::fprintf(stderr,
                         "[pf-space] MIX bank=%u: %u is not a region index (region holds %u slot(s)); "
                         "as a pool slot it would be %s\n",
                         bank, value, region,
                         cache_->prefetch_slot(value) ? "a region slot" : "a non-region slot");
            std::fflush(stderr);
            return;
        }
        if (!cache_->prefetch_slot(*slot)) {
            ++slot_space_mixes_;
            std::fprintf(stderr,
                         "[pf-space] MIX bank=%u: region index %u names pool slot %u, which is not a "
                         "region slot (region=%u)\n",
                         bank, value, *slot, region);
            std::fflush(stderr);
        }
        return;
    }
    // A placement slot that turns out to be a region slot means the region's numbers were read as the
    // placement's. The region index is printed beside it, because that is the value that was meant.
    if (cache_->prefetch_slot(value)) {
        const auto index = cache_->prefetch_region_index(bank, value);
        ++slot_space_mixes_;
        std::fprintf(stderr,
                     "[pf-space] MIX bank=%u: placement slot %u is a REGION slot (region holds %u "
                     "slot(s), region index %d)\n",
                     bank, value, cache_->prefetch_slots_per_bank(), index ? int(*index) : -1);
        std::fflush(stderr);
    }
}

void ExpertStream::gh1_report(cudaStream_t stream) {
    sync_slot_mirror(stream);
    {
        static std::uint64_t reported = 0;
        if (++reported % 8 == 0) {
            // The id->slot misses are counted where they happen (an atomic add in the kernel), so the
            // only read-back is here, once per print. Nothing else touches this counter.
            std::uint32_t lookup_misses = 0;
            if (lookup_misses_dev_ != nullptr) {
                const auto status = cudaMemcpy(&lookup_misses, lookup_misses_dev_,
                                               sizeof(lookup_misses), cudaMemcpyDeviceToHost);
                if (status != cudaSuccess) {
                    std::fprintf(stderr, "[pf-mirror] read-back of the id->slot miss counter failed: %s\n",
                                 cudaGetErrorString(status));
                    std::fflush(stderr);
                }
            }
            std::uint32_t region_misses = 0;
            if (lookup_region_misses_dev_ != nullptr) {
                const auto status = cudaMemcpy(&region_misses, lookup_region_misses_dev_,
                                               sizeof(region_misses), cudaMemcpyDeviceToHost);
                if (status != cudaSuccess) {
                    std::fprintf(stderr,
                                 "[pf-mirror] read-back of the region miss counter failed: %s\n",
                                 cudaGetErrorString(status));
                    std::fflush(stderr);
                }
            }
            // C1's two rows: the placement mirror and the region mirror, side by side because their
            // upload counts are the evidence that they are driven by different events. `uploads` is a
            // host re-export, so it is the number that used to be one per bank per layer.
            std::fprintf(stderr,
                         "[pf-mirror] banks=%u | placement(entries=%llu uploads=%llu dev_upload=%llu "
                         "dev_bytes=%llu fails=%llu) | region(entries=%llu uploads=%llu "
                         "dev_upload=%llu dev_bytes=%llu fails=%llu) | lookup calls=%llu absent=%u "
                         "region_calls=%llu region_absent=%u region_stale=%llu space_mix=%llu\n",
                         cache_ != nullptr ? cache_->bank_count() : 0U,
                         static_cast<unsigned long long>(placement_mirror_.entries),
                         static_cast<unsigned long long>(placement_mirror_.uploads),
                         static_cast<unsigned long long>(placement_mirror_.dev_uploads),
                         static_cast<unsigned long long>(placement_mirror_.dev_bytes),
                         static_cast<unsigned long long>(placement_mirror_.dev_fails),
                         static_cast<unsigned long long>(prefetch_mirror_.entries),
                         static_cast<unsigned long long>(prefetch_mirror_.uploads),
                         static_cast<unsigned long long>(prefetch_mirror_.dev_uploads),
                         static_cast<unsigned long long>(prefetch_mirror_.dev_bytes),
                         static_cast<unsigned long long>(prefetch_mirror_.dev_fails),
                         static_cast<unsigned long long>(lookup_calls_), lookup_misses,
                         static_cast<unsigned long long>(lookup_region_calls_), region_misses,
                         static_cast<unsigned long long>(prefetch_region_stale_),
                         static_cast<unsigned long long>(slot_space_mixes_));
            std::fflush(stderr);
        }
    }
    // The prediction region's own hit rate. `[pf]`'s `hit=` is the WARM TIER's, a different question:
    // this one is "of the routes this layer had, how many the region already held", which is the only
    // number that says whether the slots it took are earning their keep.
    //
    // S3-4 fixes the denominator: it used to be `prefetch_calls_`, the warm tier's ask count, which is
    // zero unless NINFER_PREFETCH attaches a queue - so the percentage printed 0/0 as 0.0% and read as
    // "the region never hits" on runs where it was never even asked. It now counts the routes the hit
    // loop examined (`prefetch_route_samples_`), which is the same population as the numerator.
    if (cache_ != nullptr && cache_->prefetch_slots_per_bank() != 0) {
        std::fprintf(stderr,
                     "[pf-region] %u slot(s)/bank | hits=%llu calls=%llu (%.1f%%) | routes examined, "
                     "all runs must hold the expert in a region slot\n",
                     cache_->prefetch_slots_per_bank(),
                     static_cast<unsigned long long>(prefetch_hits_),
                     static_cast<unsigned long long>(prefetch_route_samples_),
                     prefetch_route_samples_ != 0
                         ? 100.0 * static_cast<double>(prefetch_hits_) /
                               static_cast<double>(prefetch_route_samples_)
                         : 0.0);
        std::fflush(stderr);
    }
    if (std::getenv("NINFER_EXPERT_TIMING") == nullptr || gh1_calls_ % 8 != 0) { return; }
    const auto pct = [](std::uint64_t part, std::uint64_t all) {
        return all == 0 ? 0.0 : 100.0 * static_cast<double>(part) / static_cast<double>(all);
    };
    // static/no_miss are per STEP on the split path, capturable is per CALL (layer). Saying which
    // denominator each number uses is the difference between a metric and a mystery.
    //
    // Q20: `capturable` and `capt_free` are printed side by side on purpose. The first is a capacity
    // statement ("the pool could hold the layer"), the second is the execution statement ("the layer
    // did not touch the host"). F3 was the first number standing in for the second.
    //
    // The three fail counters attribute the gap between capt_free and capt_ready. A rate names the
    // problem; the counters name whose lifetime caused it - an unaddressable route is the data path,
    // a publish is the placement, an upload is the loader - and a controller can only act on the
    // second kind. They can overlap on one layer, so they print as attribution, not as a partition.
    std::fprintf(stderr,
                 "[gh1] bank=%u calls=%llu static=%.1f%% no_miss=%.1f%% capturable=%.1f%% "
                 "capt_free=%.1f%% capt_ready=%.1f%% fail_unaddr=%llu fail_publish=%llu "
                 "fail_upload=%llu residency=%.1f%% (%llu/%llu routed experts were held by every "
                 "run) | published_slot=%.1f%% (%llu/%llu published routes had slot+offset - a publish "
                 "rate, not residency)\n",
                  bank_, static_cast<unsigned long long>(gh1_calls_),
                  pct(gh1_static_calls_, gh1_calls_), pct(gh1_no_miss_calls_, gh1_calls_),
                  pct(gh1_calls_ - gh1_split_calls_, gh1_calls_),
                  pct(gh1_capt_free_calls_, gh1_calls_),
                  pct(gh1_capt_ready_calls_, gh1_calls_),
                  static_cast<unsigned long long>(gh1_captfail_unaddressable_),
                  static_cast<unsigned long long>(gh1_captfail_publish_),
                  static_cast<unsigned long long>(gh1_captfail_upload_),
                  pct(residency_inside_, residency_routes_),
                  static_cast<unsigned long long>(residency_inside_),
                  static_cast<unsigned long long>(residency_routes_),
                  pct(coverage_.inside, coverage_.routes),
                  static_cast<unsigned long long>(coverage_.inside),
                  static_cast<unsigned long long>(coverage_.routes));
    std::fflush(stderr);

}
// port (sm_89): Q10's instrument. The plan's gate is "static ≥95%", and `static` is incremented in
// two places - the covered fast path and "the placement moved nothing" - so the number alone does not
// say what to fix. This separates the reasons a layer is not static:
//
//   absent  : a route has no slot at all      -> the placement policy is not holding it
//   unready : a route has a slot, image not in -> the upload did not land before the next layer
//   moved   : rows that changed in the table   (ins + outs; only ins cost an upload)
//
// and it is a counter, not an inference: it prints every 400 layers on one stream. Off by default -
// the probe adds two hash lookups per route, which is the thing being measured.
namespace {
void static_probe(std::uint32_t bank, std::int32_t tokens, bool covered, std::uint32_t absent,
                  std::uint32_t unready, std::uint64_t moved, std::uint64_t ins,
                  std::uint64_t staged, bool capt_free) {
    static const bool on = std::getenv("NINFER_STATIC_PROBE") != nullptr;
    if (!on) { return; }
    struct Totals {
        std::uint64_t calls = 0, cov = 0, absent = 0, absent_layers = 0;
        std::uint64_t unready = 0, unready_layers = 0, moved = 0, ins = 0, staged = 0;
        // Step accounting. A round is 40 layers and a graph is captured per round, so layer coverage
        // says nothing about it: if one layer in forty misses, the round cannot be replayed from a
        // captured graph at all. A step is closed when bank 0 comes round again.
        //
        // Q20 keeps TWO step rates, because "clean" has been asked to mean two things:
        //   clean      - no layer was absent (a capacity statement: the pool held every route)
        //   clean_capt - no layer needed the host (an execution statement: nothing was fetched,
        //                published or uploaded). This is the one a capture can be built on.
        std::uint64_t steps = 0, clean_steps = 0, clean_capt_steps = 0, step_layers = 0;
        bool step_dirty = false;
        bool step_capt  = true;
    };
    static thread_local Totals narrow_win, narrow_all;  // decode: one (or two) tokens a call
    static thread_local Totals wide_all;                // prefill: many tokens a call
    // The split is not cosmetic. A prefill call carries tokens x top_k routes at once - a 2,300-token
    // prompt is 18,000 routes in ONE call - so a counter that mixes the two widths reports the prompt
    // length, not the pool. Every number this probe is asked for is a decode property.
    Totals& t   = tokens <= 2 ? narrow_win : wide_all;
    Totals& all = tokens <= 2 ? narrow_all : wide_all;
    if (tokens <= 2) {
        if (bank == 0 && t.step_layers != 0) {
            ++t.steps;
            ++all.steps;
            if (!t.step_dirty) { ++t.clean_steps; ++all.clean_steps; }
            if (t.step_capt) { ++t.clean_capt_steps; ++all.clean_capt_steps; }
            t.step_layers = 0;
            t.step_dirty  = false;
            t.step_capt   = true;
        }
        ++t.step_layers;
        if (absent != 0) { t.step_dirty = true; }
        if (!capt_free) { t.step_capt = false; }
    }
    ++t.calls;
    ++all.calls;
    if (covered) { ++t.cov; ++all.cov; }
    if (absent != 0) { t.absent += absent; ++t.absent_layers; all.absent += absent; ++all.absent_layers; }
    if (unready != 0) { t.unready += unready; ++t.unready_layers; all.unready += unready; ++all.unready_layers; }
    t.moved += moved; all.moved += moved;
    t.ins += ins; all.ins += ins;
    t.staged += staged; all.staged += staged;
    if (tokens > 2 || narrow_win.calls % 400 != 0) { return; }
    const double n  = static_cast<double>(narrow_win.calls);
    const double na = static_cast<double>(narrow_all.calls);
    const double ns = static_cast<double>(narrow_all.steps);
    std::fprintf(stderr,
                 "[st] win=%llu decode_layers=%llu | window covered=%.1f%% absent=%llu (%llu L) "
                 "unready=%llu | cum covered=%.1f%% absent=%.3f/L | steps=%llu clean=%.1f%% "
                 "clean_capt=%.1f%% "
                 "| moved=%.2f ins=%.2f staged=%.2f | prefill calls=%llu absent=%llu\n",
                 static_cast<unsigned long long>(narrow_all.calls / 400),
                 static_cast<unsigned long long>(narrow_win.calls),
                 100.0 * static_cast<double>(narrow_win.cov) / n,
                 static_cast<unsigned long long>(narrow_win.absent),
                 static_cast<unsigned long long>(narrow_win.absent_layers),
                 static_cast<unsigned long long>(narrow_win.unready),
                 100.0 * static_cast<double>(narrow_all.cov) / na,
                 static_cast<double>(narrow_all.absent) / na,
                 static_cast<unsigned long long>(narrow_all.steps),
                 100.0 * static_cast<double>(narrow_all.clean_steps) / ns,
                 100.0 * static_cast<double>(narrow_all.clean_capt_steps) / ns,
                 static_cast<double>(narrow_win.moved) / n, static_cast<double>(narrow_win.ins) / n,
                 static_cast<double>(narrow_win.staged) / n,
                 static_cast<unsigned long long>(wide_all.calls),
                 static_cast<unsigned long long>(wide_all.absent));
    std::fflush(stderr);
    narrow_win = Totals{};
}
} // namespace

bool ExpertStream::set_predicted_device(const std::int32_t* ids, std::int32_t count,
                                        cudaStream_t stream) {
    // A prediction that cannot be stored must not linger: the previous layer's ids would be placed as
    // if they were this one's, which is a silently wrong placement rather than a missing one.
    predicted_device_active_ = false;
    predicted_device_count_  = 0;
    predicted_.clear();
    if (ids == nullptr || count <= 0) {
        ++predicted_device_failures_;
        std::fprintf(stderr, "[pf-pred] hand-off refused: %s (count=%d)\n",
                     ids == nullptr ? "no Device ids" : "empty id list", count);
        std::fflush(stderr);
        return false;
    }
    const std::size_t bytes = static_cast<std::size_t>(count) * sizeof(std::int32_t);
    if (!predicted_ids_device_ || predicted_ids_device_->bytes < bytes) {
        try {
            predicted_ids_device_ = std::make_unique<DeviceBuffer>(bytes);
        } catch (const std::exception& error) {
            ++predicted_device_failures_;
            std::fprintf(stderr, "[pf-pred] Device buffer (%zu bytes) failed: %s\n", bytes,
                         error.what());
            std::fflush(stderr);
            return false;
        }
    }
    // Device-to-Device on the caller's stream, so the ids are ready for the layer that consumes them
    // and the workspace arena the router wrote into can be recycled in the meantime. No host wait.
    const auto status = cudaMemcpyAsync(predicted_ids_device_->p, ids, bytes,
                                       cudaMemcpyDeviceToDevice, stream);
    if (status != cudaSuccess) {
        ++predicted_device_failures_;
        std::fprintf(stderr, "[pf-pred] Device hand-off copy (%zu bytes) failed: %s\n", bytes,
                     cudaGetErrorString(status));
        std::fflush(stderr);
        return false;
    }
    predicted_device_count_  = count;
    predicted_device_active_ = true;
    ++predicted_device_handoffs_;
    return true;
}

// C5: hand one fill's BYTE SOURCE to the process-wide read-ahead worker. What moves is the artifact
// read (the slow part: a pageable read is measured at 8.6 ms an expert on this box, a page-cache hit at
// 0.2 ms). The region's placement and the Device copy stay on the calling thread, because they must be
// ordered before the GEMM that consumes them - a stream is a queue, and a copy submitted from another
// thread is not ordered against this one without an event and a wait, which is exactly the per-layer
// synchronise this path does not have. So the worker fills the pinned tier under the layer that is
// computing, and `cache_->prepare()` then finds the bytes in pinned memory instead of behind a read.
//
// The offer goes through the existing queue, so the budget, the deadline and the usefulness counters
// are the ones that were already measured (`[pf] ... ran= expired= useful=`). A closed gate or a full
// window refuses the offer; a refused read-ahead is simply not done (the fill that follows reads the
// artifact, which is what it did before this existed).
std::size_t ExpertStream::queue_prefetch_fill(std::span<const std::uint32_t> experts,
                                              std::uint32_t bank, std::uint64_t tick,
                                              std::uint64_t window) {
    if (host_ == nullptr || experts.empty()) { return 0; }
    auto& queue = ExpertPrefetchQueue::instance();
    if (!queue.gate_open()) { return 0; }
    std::size_t offered = 0;
    const auto  tick_at = tick == 0 ? prefetch_layer_tick().load() : tick;
    const auto  due     = tick_at + (window == 0 ? prefetch_window() : window);
    for (const auto expert : experts) {
        if (expert > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
            continue;
        }
        const std::int32_t one[1] = {static_cast<std::int32_t>(expert)};
        if (queue.enqueue(host_weak_, bank, std::span<const std::int32_t>(one, 1),
                          static_cast<std::size_t>(host_->slot_bytes()), due, tick_at)) {
            ++offered;
        }
    }
    return offered;
}

// C3/C4: the port, assembled from what this stream measured. One function so the arming decision and
// the line printed beside it cannot disagree - the same rule the controller's other ports follow
// (nothing on the prediction side decides its own size).
PrefetchPlan ExpertStream::prefetch_plan() const {
    PrefetchSignals signals;
    signals.cold_share = residency_rate(); // 1 - residency: the share a prediction would have to add
    // The region's own rate, from the population the panel counts (routes examined), not the warm
    // tier's ask count - that one is zero on this path and would leave the rate permanently unknown.
    signals.hit_rate = prefetch_route_samples_ == 0
                           ? -1.0
                           : static_cast<double>(prefetch_hits_) /
                                 static_cast<double>(prefetch_route_samples_);
    const auto& ledger = cpu_path_ledger();
    signals.cpu_share  = ledger.routes_seen == 0
                             ? -1.0
                             : static_cast<double>(ledger.cpu_routes) /
                                   static_cast<double>(ledger.routes_seen);
    signals.staging_reachable =
        cache_ != nullptr && cache_->prefetch_slots_per_bank() != 0 && prefetch_vram_mode();
    signals.freeze_active       = frozen_;
    signals.lanes               = 1;
    // The model's routing width, not this guess's width: a prefill guess is tens of ids and a decode
    // guess is ten, and the region's demand must not change with the prompt length.
    signals.top_k               = static_cast<int>(top_k_);
    signals.runs_per_layer      = static_cast<int>(runs());
    signals.pool_slots_per_bank = static_cast<int>(pool_slots_per_bank());
    signals.region_slots = cache_ != nullptr ? static_cast<int>(cache_->prefetch_slots_per_bank()) : 0;
    const int per_layer = std::max(1, static_cast<int>(runs()));
    signals.layers      = pool_banks() != 0 ? static_cast<int>(pool_banks()) / per_layer : 0;
    signals.ring_depth  = static_cast<int>(ring_depth_);
    // 0: the controller derives the depth cap from the model's layer count and the ring, which is what
    // C3 asks for - a caller-chosen number here would be the magic constant the cutoff exists to avoid.
    signals.max_layers_ahead = 0;
    note_prefetch_signals(signals);
    return decide_prefetch_plan();
}

void ExpertStream::stage_predicted(cudaStream_t stream) {
    const bool vram = prefetch_vram_mode();
    // One line for the first few calls, whatever the reason: "no placement happened" and "the region
    // is off" look identical in the counters, and that ambiguity already cost a measurement round.
    static std::uint64_t calls = 0;
    // Entry trace, deliberately NOT gated on the bank: the per-call trace below only speaks for the
    // first four banks, so "every bank says nothing" and "stage_predicted is never called" read the
    // same. This line answers the second question first, with the bank that asked.
    static std::uint64_t entries = 0;
    if (std::getenv("NINFER_EXPERT_TIMING") != nullptr && ++entries <= 8) {
        std::fprintf(stderr,
                     "[pfv] enter #%llu bank=%u vram=%d cache=%d predicted=%zu device=%d region=%u\n",
                     (unsigned long long)entries, bank_, vram ? 1 : 0, cache_ != nullptr ? 1 : 0,
                     predicted_.size(), int(predicted_is_device()),
                     cache_ != nullptr ? cache_->prefetch_slots_per_bank() : 0U);
        std::fflush(stderr);
    }
    const bool trace =
        bank_ < 4 && std::getenv("NINFER_EXPERT_TIMING") != nullptr && ++calls <= 6;
    const auto say = [&](const char* why) {
        if (!trace) { return; }
        std::fprintf(stderr,
                     "[pfv] skip bank=%u: %s (vram=%d cache=%d predicted=%zu device=%d region=%u)\n",
                     bank_, why, vram ? 1 : 0, cache_ != nullptr ? 1 : 0, predicted_.size(),
                     int(predicted_is_device()),
                     cache_ != nullptr ? cache_->prefetch_slots_per_bank() : 0U);
        std::fflush(stderr);
    };
    if (!vram) { say("mode is not vram"); return; }
    if (cache_ == nullptr) { say("no cache"); return; }

    // S3: the two forms of the same hand-off.
    //
    // Device form (S3-1): the ids never crossed PCIe. S3-2's resolution half runs where they are -
    // moe_route_slots() against the pool's own slot mirror, once per matrix run, so an expert's slot in
    // the gate run and its absence from the down run are both visible. The placement half - the words
    // set_prefetch_experts() writes (owner, region cursor, generation, entry readiness) and the bytes
    // that then have to leave the artifact - cannot be applied from the Device, and the reason is not
    // a missing kernel: the pool's tables are host-authoritative (every publication reads
    // cache_->expert_slot()), and the only Device-addressable source of an expert's bytes is the
    // pinned tier, so a kernel-side fill would have to read PCIe at SM rates. The house precedent for
    // "the Device decides, the host applies" is device_admit_ring(), which drains its descriptors
    // with cudaStreamSynchronize and issues cudaMemcpyAsync itself. So this form resolves and counts,
    // and says that the fill did not happen instead of pretending it did.
    if (predicted_is_device()) {
        const auto count = predicted_device_count();
        ++predicted_layers_;
        if (cache_->prefetch_slots_per_bank() != 0) {
            lookup_routed_slots(predicted_device_ids(), count, stream,
                                ops::RouteTableKind::PrefetchRegion, /*armed=*/true);
        }
        static std::uint64_t reports = 0;
        if (bank_ < 4 && (++reports % 40) == 0) {
            std::fprintf(stderr,
                         "[pf-pred] Device form: layers=%llu handoffs=%llu handoff_fail=%llu | "
                         "runs_resolved=%llu run_mismatch=%llu | ids/layer=%d fill=host-only "
                         "(0 placed by the Device)\n",
                         (unsigned long long)predicted_layers_,
                         (unsigned long long)predicted_device_handoffs_,
                         (unsigned long long)predicted_device_failures_,
                         (unsigned long long)predicted_runs_resolved_,
                         (unsigned long long)predicted_run_mismatches_, int(count));
            std::fflush(stderr);
        }
        return;
    }

    if (predicted_.empty()) { say("predicted_ is empty"); return; }
    if (cache_->prefetch_slots_per_bank() == 0) { say("no region"); return; }
    // C3/C4: the arm/cutoff decision, taken once per call from what this stream measured. `armed` is
    // a RESULT (slots x cold delta x measured rate), not a switch: a region whose rate does not clear
    // the bar computes zero slots, and a region too small to hold one layer's cold delta computes a
    // zero cutoff. Both are printed with the reason, on the same line as the numbers they came from.
    const auto plan = prefetch_plan();
    {
        const auto& port = prefetch_signals();
        if (pf_ctl_reports_++ % 64 == 0) {
            std::fprintf(stderr,
                         "[pf-ctl] cutoff=%d armed=%s slots=%d reason=%s | cold=%.1f%% hit=%.1f%% "
                         "cpu=%.1f%% delta=%d layer(s)=%d ring=%d top_k=%d region=%u/bank\n",
                         plan.cutoff, plan.armed ? "Y" : "N", plan.slots_per_bank, plan.why,
                         port.cold_share >= 0.0 ? 100.0 * port.cold_share : -1.0,
                         port.hit_rate >= 0.0 ? 100.0 * port.hit_rate : -1.0,
                         port.cpu_share >= 0.0 ? 100.0 * port.cpu_share : -1.0,
                         plan.delta_per_layer, port.layers, port.ring_depth, port.top_k,
                         cache_->prefetch_slots_per_bank());
            std::fflush(stderr);
        }
        if (!plan.armed) {
            // A shut region is a decision, not an absence: the fill is refused and counted, and the
            // reason is on the line above. Nothing is staged, so the next layer stays a demand miss.
            ++predicted_shut_layers_;
            return;
        }
    }
    // port (sm_89): the prediction region exists per bank, so the three-run form stages the same
    // predicted set into each of its runs. One run's region is sized per bank by the cache, so a run
    // that cannot take the whole set takes what fits and the rest stays a demand miss next layer.
    std::array<std::uint32_t, 4> banks{};
    std::size_t bank_count = 1;
    banks[0] = bank_;
    if (split_banks_) {
        bank_count = split_.size();
        for (std::size_t index = 0; index < split_.size(); ++index) { banks[index] = split_[index].bank; }
    }
    // S3-3: every run the layer has is fed, and the count is checked against the layout instead of
    // assumed - a fused layer has one run, gate/up/down has three, and a disagreement leaves a run
    // cold that this layer will read.
    const std::size_t expected_runs = static_cast<std::size_t>(runs());
    if (bank_count != expected_runs) {
        ++predicted_run_mismatches_;
        std::fprintf(stderr,
                     "[pf-pred] run mismatch on bank %u: feeding %zu run(s), layout reports %zu "
                     "(split=%d, top_k=%d)\n",
                     bank_, bank_count, expected_runs, int(split_banks_),
                     static_cast<int>(predicted_.size()));
        std::fflush(stderr);
    }
    for (std::size_t index = 0; index < bank_count; ++index) {
        const auto bank = banks[index];
        std::vector<std::uint32_t> absent;
        absent.reserve(predicted_.size());
        for (const auto expert : predicted_) {
            if (expert < 0 || static_cast<std::uint32_t>(expert) >= count_) {
                // Out of this bank's range: a prediction that cannot be placed is counted, not dropped
                // silently - it means the router and the pool disagree about the bank's width.
                ++prefetch_out_of_range_;
                continue;
            }
            const auto id = static_cast<std::uint32_t>(expert);
            // C2: the two spaces, checked where both numbers are on the host. An expert the region
            // already holds has a region index AND a pool slot. If the index names no slot at all, or
            // names a pool slot that is not a region slot, the two spaces have been used as one -
            // counted and printed, never smoothed over. A row whose expert the placement has since
            // moved is NOT a mix: a prediction is not a member of the resident set, so the resident set
            // recomputes every layer and leaves the region row behind. That case gets its own counter
            // (`prefetch_region_stale_`) so the mix count stays meaningful.
            if (const auto region_index = cache_->prefetch_region_index(bank, id); region_index) {
                note_slot_space(bank, ops::RouteTableKind::PrefetchRegion, *region_index);
                const auto from_index = cache_->prefetch_region_slot(bank, *region_index);
                const auto direct     = cache_->expert_slot(bank, id);
                if (from_index && (!direct || *direct != *from_index)) {
                    ++prefetch_region_stale_;
                }
            }
            if (cache_->expert_slot(bank, id)) { continue; } // already resident: not a prediction
            absent.push_back(id);
        }
        prefetch_delta_ += absent.size();
        // C5: the read ahead of the copy. Offered to the process-wide worker first, so the artifact
        // read happens under this layer's compute; the placement and the Device copy stay ordered on
        // `stream` below. A closed gate refuses and nothing changes - the copy then reads the artifact
        // exactly as it did before this existed.
        //
        // Not offered here on the `prev` path: stage_history() already offered the same set one hook
        // earlier (before this layer's GEMM), and offering it twice would spend the window's budget on
        // a read that is then immediately duplicated by the inline one this call still does.
        const bool offered_early = history_signal_switched() && !history_.empty();
        if (!absent.empty() && !offered_early) {
            const auto tick = prefetch_layer_tick().fetch_add(1) + 1;
            const auto window = prefetch_window();
            if (window != 0 && (tick % window) == 0) { ExpertPrefetchQueue::instance().advance(tick); }
            const auto offered = queue_prefetch_fill(
                std::span<const std::uint32_t>(absent), bank, tick, window);
            prefetch_queued_ += offered;
        }
        std::vector<ExpertCacheKey> placed;
        const auto filled = absent.empty()
                                ? 0U
                                : cache_->set_prefetch_experts(bank, absent, placed);
        if (filled != 0) {
            cache_->prepare(std::span<const ExpertCacheKey>(placed), stream);
            (void)cache_->progress(stream);
            prefetch_placed_ += placed.size();
            ++predicted_runs_filled_;
            // C5/D4: the read-ahead's bytes that the copy actually used. The pool prefers the pinned
            // tier when it holds an expert, so a slot the worker filled before this call shows up here
            // - and on this path that is the ONLY consumer of those bytes (the experts go to the
            // Device, not to the CPU half), which is why the routed-hit counter reads zero.
            if (host_ != nullptr) {
                if (const auto marks = host_->take_prefetch_marks(bank, absent); marks != 0) {
                    ExpertPrefetchQueue::instance().note_used(
                        static_cast<std::size_t>(marks) * host_->slot_bytes());
                }
            }
        } else if (!absent.empty()) {
            // A non-empty delta that placed nothing is a refusal (no region on this bank, or the bank
            // is outside the pool), and it has to be visible rather than read as "nothing to do".
            ++predicted_fill_failures_;
            std::fprintf(stderr,
                         "[pf-pred] fill refused bank=%u: %zu absent expert(s), name no slot "
                         "(region=%u)\n",
                         bank, absent.size(), cache_->prefetch_slots_per_bank());
            std::fflush(stderr);
        }
    }
    static std::uint64_t reports = 0;
    if (bank_ < 4 && (++reports % 40) == 0) {
        // C5: the read-ahead's own counters, on the line the fill produces them from. The split path
        // never reaches the fused path's `[pf]` panel, so without this the queue could run, expire and
        // be useful without a single number saying so.
        const auto queued = ExpertPrefetchQueue::instance().stats();
        std::fprintf(stderr,
                     "[pfv] bank=%u delta=%llu queued=%llu placed=%llu hits=%llu region=%u/bank | "
                     "runs=%zu filled=%llu fill_fail=%llu out_of_range=%llu mismatch=%llu shut=%llu | "
                     "guess ids=%llu hit=%llu over=%llu layer(s) (%.1f%%) | q enq=%llu ran=%llu "
                     "expired=%llu useful=%.2f pending=%zu\n",
                     bank_, (unsigned long long)prefetch_delta_,
                     (unsigned long long)prefetch_queued_,
                     (unsigned long long)prefetch_placed_,
                     (unsigned long long)prefetch_hits_, cache_->prefetch_slots_per_bank(),
                     expected_runs, (unsigned long long)predicted_runs_filled_,
                     (unsigned long long)predicted_fill_failures_,
                     (unsigned long long)prefetch_out_of_range_,
                     (unsigned long long)predicted_run_mismatches_,
                     (unsigned long long)predicted_shut_layers_,
                     (unsigned long long)prediction_ids_seen_,
                     (unsigned long long)prediction_overlap_,
                     (unsigned long long)prediction_layers_checked_,
                     prediction_ids_seen_ != 0
                         ? 100.0 * static_cast<double>(prediction_overlap_) /
                               static_cast<double>(prediction_ids_seen_)
                         : 0.0,
                     (unsigned long long)queued.queue.enqueued, (unsigned long long)queued.ran,
                     (unsigned long long)queued.expired, queued.queue.usefulness(),
                     queued.pending);
        std::fflush(stderr);
    }
}

// port (sm_89): P1 - the prediction hand-off as a ring the Device writes and the host drains.
//
// What this is for. The prediction is made by a kernel at the end of layer L-1 and consumed by the
// host at layer L. S3's hand-off is one Device buffer, which answers "the last ids the producer
// wrote" - enough for one reader that always wants the latest, and blind to the two things a pipeline
// needs: whether the entry the host read is the one produced for the layer it is placing, and whether
// it missed one. The ring answers both with one monotonic counter (moe_pred_ring_write, ops level) and
// three consequences for the host:
//
//   * the drain issues its copies in `begin_fetch`, in front of the layer's own existing
//     cudaStreamSynchronize (ensure_split, the route read-back). Nothing here waits: the copies ride
//     the same stream, ordered behind the producer's kernel and in front of nothing but the call the
//     layer already makes. That is the constructive half of "no new per-layer synchronise" - the layer
//     had a synchronise before this change and has exactly one after it;
//   * `complete` reads the pinned bytes and says what it got: entries drained, ids drained, entries it
//     never saw (a producer that outran the consumer), entries whose slot was overwritten, entries
//     addressed to another layer, and a guess wider than an entry. Every one of those is a refusal to
//     place, counted where it is detected;
//   * nothing is written down: the id capacity is the guess's own length (a layer's top_k times the
//     tokens it ran), the depth is what the in-flight bound allows (see ensure_pred_ring).
//
// Explicit switch only. The shipped path (no NINFER_PREFETCH*/SLOT_LOOKUP) never allocates here, so
// the gate's numbers cannot move because of this file.
namespace {
bool pred_ring_switched() {
    static const bool value = [] {
        const char* v = std::getenv("NINFER_PRED_RING");
        return v != nullptr && *v != '\0' && std::strtol(v, nullptr, 10) != 0;
    }();
    return value;
}
// The mode in which the region is staged: NINFER_PREFETCH_MODE=vram. Everything else (including the
// default) leaves stage_predicted() a no-op, so no read is worth offering.
bool prefetch_vram_mode() {
    static const bool value = [] {
        const char* v = std::getenv("NINFER_PREFETCH_MODE");
        return v != nullptr && std::strcmp(v, "vram") == 0;
    }();
    return value;
}
// P2: the signal source. `residual` is the default and the only one that needs a kernel; `prev` is
// this layer's own routes from the previous step, which the host already holds. `mtp` has no source on
// this artifact (see predict_next_layer in ffn.cpp) and parses here only so the two sites agree on the
// spelling instead of one silently treating it as the default.
bool history_signal_switched() {
    static const bool value = [] {
        const char* v = std::getenv("NINFER_PREFETCH_SIGNAL");
        return v != nullptr && std::strcmp(v, "prev") == 0;
    }();
    return value;
}
} // namespace

bool ExpertStream::pred_ring_armed() const noexcept { return pred_ring_switched(); }

void ExpertStream::release_pred_ring() noexcept {
    const auto drop = [](auto*& pointer, const char* what) {
        if (pointer == nullptr) { return; }
        const auto status = cudaFree(pointer);
        if (status != cudaSuccess) {
            std::fprintf(stderr, "[pf-pred] cudaFree(%s) failed: %s\n", what,
                         cudaGetErrorString(status));
            std::fflush(stderr);
        }
        pointer = nullptr;
    };
    drop(ring_headers_dev_, "ring headers");
    drop(ring_ids_dev_, "ring ids");
    drop(ring_head_dev_, "ring head");
    ring_depth_       = 0;
    ring_id_capacity_ = 0;
    ring_tail_        = 0;
}

// The ring's two dimensions, both derived rather than chosen:
//
//   * id capacity = the guess's own length. The producer knows it (top_k times the tokens the layer
//     ran), and it is the only number that changes with the model - a top_k of 10 and an eight-token
//     verify rectangle need 80 ids, a single decode token needs 10, and neither is a constant here;
//   * depth = one entry in flight plus the one being drained. The in-flight bound is structural, not
//     a policy: the consumer drains at its own ensure(), and that call synchronises for the route
//     read-back (ensure_split), so at most one produced entry can be waiting to be drained. Two is
//     therefore the smallest depth at which head and tail never alias, and a "full" ring is
//     detectable instead of being read as an empty one. NINFER_PRED_RING_DEPTH widens it for an
//     experiment; it is clamped to the pool's own layer count, because a ring deeper than the pool
//     has banks cannot be filled by a walk over them.
bool ExpertStream::ensure_pred_ring(std::int32_t ids) noexcept {
    if (ids <= 0) { return false; }
    const auto capacity = static_cast<std::uint32_t>(ids);
    if (ring_headers_dev_ != nullptr && ring_id_capacity_ >= capacity) { return true; }
    // A wider guess rebuilds the ring: the entry stride changes, so the old image cannot be reused
    // with a new `id_capacity` (that is exactly the mixture the sequence check exists to catch).
    release_pred_ring();
    std::uint32_t depth = 2;
    if (const char* value = std::getenv("NINFER_PRED_RING_DEPTH");
        value != nullptr && *value != '\0') {
        const auto asked = std::strtoul(value, nullptr, 10);
        if (asked != 0) { depth = static_cast<std::uint32_t>(asked); }
    }
    if (cache_ != nullptr) {
        const auto layers = cache_->bank_count();
        if (layers != 0 && depth > layers) { depth = layers; }
    }
    if (depth < 2) { depth = 2; }
    void* headers = nullptr;
    void* entry   = nullptr;
    void* head    = nullptr;
    auto status   = cudaMalloc(&headers, static_cast<std::size_t>(depth) *
                                             sizeof(ops::PredRingHeader));
    if (status == cudaSuccess) {
        status = cudaMalloc(&entry, static_cast<std::size_t>(depth) * capacity * sizeof(std::int32_t));
    }
    if (status == cudaSuccess) { status = cudaMalloc(&head, sizeof(unsigned long long)); }
    if (status != cudaSuccess) {
        ++ring_stats_.copy_failures;
        std::fprintf(stderr,
                     "[pf-pred] ring cudaMalloc(depth=%u ids=%u, %zu bytes) failed: %s\n", depth,
                     capacity,
                     static_cast<std::size_t>(depth) * capacity * sizeof(std::int32_t),
                     cudaGetErrorString(status));
        std::fflush(stderr);
        if (headers != nullptr) { (void)cudaFree(headers); }
        if (entry != nullptr) { (void)cudaFree(entry); }
        return false;
    }
    // The head starts at zero, and that is the only host write into the ring's Device memory - it
    // happens once, at construction, never on a layer.
    if (const auto zeroed = cudaMemset(head, 0, sizeof(unsigned long long)); zeroed != cudaSuccess) {
        ++ring_stats_.copy_failures;
        std::fprintf(stderr, "[pf-pred] ring cudaMemset(head) failed: %s\n",
                     cudaGetErrorString(zeroed));
        std::fflush(stderr);
        (void)cudaFree(headers);
        (void)cudaFree(entry);
        (void)cudaFree(head);
        return false;
    }
    ring_headers_dev_  = static_cast<ops::PredRingHeader*>(headers);
    ring_ids_dev_      = static_cast<std::int32_t*>(entry);
    ring_head_dev_     = static_cast<unsigned long long*>(head);
    ring_depth_        = depth;
    ring_id_capacity_  = capacity;
    ring_tail_         = 0;
    ring_fetch_armed_  = false;
    return true;
}

bool ExpertStream::pred_ring_write(const std::int32_t* ids, std::int32_t count,
                                   std::uint32_t source_bank, std::uint32_t layer,
                                   cudaStream_t stream) {
    if (!pred_ring_switched()) { return false; }
    if (!ensure_pred_ring(count)) { return false; }
    if (static_cast<std::uint32_t>(count) > ring_id_capacity_) {
        ++ring_stats_.truncated;
        std::fprintf(stderr,
                     "[pf-pred] ring refuses %d id(s): the entry holds %u (bank=%u layer=%u)\n",
                     count, ring_id_capacity_, bank_, layer);
        std::fflush(stderr);
        return false;
    }
    ops::moe_pred_ring_write(ids, count, source_bank, bank_, layer, ring_headers_dev_,
                             ring_ids_dev_, ring_id_capacity_, ring_depth_, ring_head_dev_, stream);
    if (const auto status = cudaGetLastError(); status != cudaSuccess) {
        ++ring_stats_.copy_failures;
        std::fprintf(stderr, "[pf-pred] ring write launch (bank=%u, %d ids) failed: %s\n", bank_,
                     count, cudaGetErrorString(status));
        std::fflush(stderr);
        return false;
    }
    ++ring_stats_.entries_written;
    ring_stats_.ids_written += static_cast<std::uint64_t>(count);
    if (ring_stats_.entries_written == 1) {
        // Once per stream: which bank owns the ring, how deep it is and what one entry holds. The
        // whole hand-off is invisible otherwise - a consumer that never sees a write and a producer
        // that wrote into nothing look the same in the drain counts.
        std::fprintf(stderr, "[pf-pred] ring armed bank=%u depth=%u capacity=%u ids=%d from bank=%u\n",
                     bank_, ring_depth_, ring_id_capacity_, count, source_bank);
        std::fflush(stderr);
    }
    static std::uint64_t reports = 0;
    if ((++reports % 64) == 0) {
        std::fprintf(stderr,
                     "[pf-pred] written=%llu ids=%llu (bank=%u) | drained=%llu ids=%llu empty=%llu "
                     "dropped=%llu\n",
                     static_cast<unsigned long long>(ring_stats_.entries_written),
                     static_cast<unsigned long long>(ring_stats_.ids_written), bank_,
                     static_cast<unsigned long long>(ring_stats_.entries_drained),
                     static_cast<unsigned long long>(ring_stats_.ids_drained),
                     static_cast<unsigned long long>(ring_stats_.empty_fetches),
                     static_cast<unsigned long long>(ring_stats_.dropped));
        std::fflush(stderr);
    }
    return true;
}

void ExpertStream::pred_ring_begin_fetch(cudaStream_t stream) {
    ring_fetch_armed_ = false;
    if (!pred_ring_switched() || ring_head_dev_ == nullptr || ring_headers_dev_ == nullptr) {
        if (pred_ring_switched() && ring_head_dev_ == nullptr) {
            // A consumer with no ring has nothing to drain: either no layer has predicted into this
            // one yet (the first layer), or the producer wrote elsewhere. Printed once so the two are
            // distinguishable from a drain that found an empty ring.
            static bool said = false;
            if (!said) {
                said = true;
                std::fprintf(stderr,
                             "[pf-pred] drain armed with no ring on bank=%u (no entry has been "
                             "written for this stream yet)\n",
                             bank_);
                std::fflush(stderr);
            }
        }
        return;
    }
    const std::size_t header_bytes =
        static_cast<std::size_t>(ring_depth_) * sizeof(ops::PredRingHeader);
    const std::size_t id_bytes = static_cast<std::size_t>(ring_depth_) * ring_id_capacity_ *
                                 sizeof(std::int32_t);
    const std::size_t total = sizeof(unsigned long long) + header_bytes + id_bytes;
    if (!ring_stage_ || ring_stage_->size() < total) {
        try {
            ring_stage_ = std::make_unique<PinnedHostBuffer>(total);
        } catch (const std::exception& error) {
            ++ring_stats_.copy_failures;
            std::fprintf(stderr, "[pf-pred] ring pinned staging (%zu bytes) failed: %s\n", total,
                         error.what());
            std::fflush(stderr);
            return;
        }
    }
    auto* base = static_cast<std::byte*>(ring_stage_->data());
    const auto copy = [&](void* destination, const void* source, std::size_t bytes) {
        const auto status =
            cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToHost, stream);
        if (status != cudaSuccess) {
            ++ring_stats_.copy_failures;
            std::fprintf(stderr, "[pf-pred] ring drain copy (%zu bytes) failed: %s\n", bytes,
                         cudaGetErrorString(status));
            std::fflush(stderr);
            return false;
        }
        return true;
    };
    if (!copy(base, ring_head_dev_, sizeof(unsigned long long))) { return; }
    if (!copy(base + sizeof(unsigned long long), ring_headers_dev_, header_bytes)) { return; }
    if (!copy(base + sizeof(unsigned long long) + header_bytes, ring_ids_dev_, id_bytes)) { return; }
    ring_fetch_armed_ = true;
}

std::int32_t ExpertStream::pred_ring_complete(std::vector<std::int32_t>& ids,
                                              std::int32_t expected) {
    ids.clear();
    if (!ring_fetch_armed_) { return 0; }
    ring_fetch_armed_ = false;
    const auto* base = static_cast<const std::byte*>(ring_stage_->data());
    unsigned long long head = 0;
    std::memcpy(&head, base, sizeof(head));
    const auto* headers =
        reinterpret_cast<const ops::PredRingHeader*>(base + sizeof(unsigned long long));
    const auto* entry_ids = reinterpret_cast<const std::int32_t*>(
        base + sizeof(unsigned long long) + static_cast<std::size_t>(ring_depth_) *
                                                sizeof(ops::PredRingHeader));
    ring_stats_.head = static_cast<std::uint64_t>(head);
    if (head == ring_tail_) {
        ++ring_stats_.empty_fetches;
        return 0;
    }
    // A producer that outran this consumer left entries in slots that have since been reused. Those
    // entries are gone, not stale-but-readable, so they are counted and skipped rather than placed.
    if (head - ring_tail_ > ring_depth_) {
        // The entries between `tail_` and `head - depth` left no slot behind: their count is what the
        // pipeline lost, so it is added as a count, not as "this drain found one problem".
        ring_stats_.dropped += head - ring_tail_ - ring_depth_;
        ring_tail_ = head - ring_depth_;
    }
    std::int32_t delivered = 0;
    for (auto sequence = ring_tail_; sequence < head; ++sequence) {
        const auto slot = static_cast<std::uint32_t>(sequence % ring_depth_);
        const auto& header = headers[slot];
        if (header.sequence != sequence) {
            // The slot was overwritten between the producer's write and this drain: the entry the
            // ticket named is not there any more.
            ++ring_stats_.stale;
            continue;
        }
        if (header.target_bank != bank_) {
            ++ring_stats_.wrong_target;
            std::fprintf(stderr,
                         "[pf-pred] ring entry seq=%llu was addressed to bank %u, drained by bank "
                         "%u\n",
                         static_cast<unsigned long long>(sequence), header.target_bank, bank_);
            std::fflush(stderr);
            continue;
        }
        if (header.truncated != 0) {
            ++ring_stats_.truncated;
            std::fprintf(stderr,
                         "[pf-pred] ring entry seq=%llu dropped: %u id(s) did not fit %u\n",
                         static_cast<unsigned long long>(sequence), header.count, header.capacity);
            std::fflush(stderr);
            continue;
        }
        if (header.count > ring_id_capacity_) {
            ++ring_stats_.count_mismatch;
            std::fprintf(stderr,
                         "[pf-pred] ring entry seq=%llu claims %u id(s), the entry holds %u\n",
                         static_cast<unsigned long long>(sequence), header.count,
                         ring_id_capacity_);
            std::fflush(stderr);
            continue;
        }
        const auto* source = entry_ids + static_cast<std::size_t>(slot) * ring_id_capacity_;
        for (std::uint32_t i = 0; i < header.count; ++i) {
            const auto expert = source[i];
            if (expert < 0 || static_cast<std::uint32_t>(expert) >= count_) {
                // The producer and this bank disagree about the bank's width: counted, not used.
                ++ring_stats_.out_of_range;
                continue;
            }
            ids.push_back(expert);
        }
        ++ring_stats_.entries_drained;
        ring_stats_.ids_drained += header.count;
        delivered += static_cast<std::int32_t>(header.count);
    }
    ring_tail_ = static_cast<std::uint64_t>(head);
    // The acceptance question for the hand-off, asked where the answer is: did every id the producer
    // wrote arrive? A short delivery is a lost prediction, not a wrong one, so it is counted apart
    // from the guess's own precision.
    if (expected > 0 && delivered != expected) {
        ++ring_stats_.count_mismatch;
        std::fprintf(stderr,
                     "[pf-pred] drain count mismatch bank=%u: %d id(s) arrived, this layer's route "
                     "grid is %d (depth=%u capacity=%u)\n",
                     bank_, delivered, expected, ring_depth_, ring_id_capacity_);
        std::fflush(stderr);
    }
    // One line per 8 drains: entries in and out, ids in and out, and the two failure kinds that mean
    // the pipeline lost a prediction rather than that the guess was wrong.
    static std::uint64_t reports = 0;
    if ((++reports % 8) == 0) {
        std::fprintf(stderr,
                     "[pf-pred] drained=%llu ids=%llu dropped=%llu gen=%llu | written=%llu "
                     "ids=%llu empty=%llu stale=%llu truncated=%llu target=%llu range=%llu "
                     "depth=%u capacity=%u\n",
                     static_cast<unsigned long long>(ring_stats_.entries_drained),
                     static_cast<unsigned long long>(ring_stats_.ids_drained),
                     static_cast<unsigned long long>(ring_stats_.dropped),
                     static_cast<unsigned long long>(ring_stats_.head),
                     static_cast<unsigned long long>(ring_stats_.entries_written),
                     static_cast<unsigned long long>(ring_stats_.ids_written),
                     static_cast<unsigned long long>(ring_stats_.empty_fetches),
                     static_cast<unsigned long long>(ring_stats_.stale),
                     static_cast<unsigned long long>(ring_stats_.truncated),
                     static_cast<unsigned long long>(ring_stats_.wrong_target),
                     static_cast<unsigned long long>(ring_stats_.out_of_range), ring_depth_,
                     ring_id_capacity_);
        std::fflush(stderr);
    }
    return delivered;
}

// P2: `prev` - the signal that can actually be prefetched. A layer's routes on the previous step are
// on the host before the current layer runs, so the fill can be queued while the PREVIOUS layer's
// kernels execute (text.cpp, before ffn()). That is the one placement order the pool rewards: the
// bytes are already in the slot by the time the route asks for it. The residual signal cannot be
// staged that way by construction - it is produced at the end of the layer it predicts for, and the
// host cannot read it before that layer's own synchronise without adding one.
void ExpertStream::note_routes(std::span<const std::int32_t> ids) {
    if (!history_signal_switched()) { return; }
    history_.assign(ids.begin(), ids.end());
    std::sort(history_.begin(), history_.end());
    history_.erase(std::unique(history_.begin(), history_.end()), history_.end());
    ++history_notes_;
    history_ids_ += history_.size();
}

void ExpertStream::stage_history(cudaStream_t stream) {
    if (!history_signal_switched() || history_.empty()) { return; }
    predicted_               = history_;
    predicted_device_active_ = false;
    predicted_device_count_  = 0;
    // C5: the READ, and only the read, is offered here - this hook runs BEFORE the layer's routed
    // GEMM (text.cpp's first call), which is the one interval where a cold read can hide. The
    // placement and the Device copy stay at the post-GEMM stage_predicted() call, because the copy
    // must be ordered before the layer that reads it while the read must not be in front of anything.
    //
    // The old shape did the whole fill here, and that is the serialisation the worker exists to remove:
    // set_prefetch_experts() + prepare() + progress() read the artifact on the calling thread in the
    // same breath, so the copy's source could never be a read-ahead that had not happened yet
    // (measured: `ran` climbed while `useful` stayed 0.00).
    if (host_ != nullptr && cache_ != nullptr && cache_->prefetch_slots_per_bank() != 0 &&
        prefetch_vram_mode()) {
        const auto plan = prefetch_plan();
        if (plan.armed) {
            std::vector<std::uint32_t> absent;
            absent.reserve(predicted_.size());
            for (const auto expert : predicted_) {
                if (expert < 0 || static_cast<std::uint32_t>(expert) >= count_) { continue; }
                const auto id = static_cast<std::uint32_t>(expert);
                if (cache_->expert_slot(bank_, id)) { continue; }
                // NOT filtered by "the tier already holds it": a re-offer of an expert the arena
                // already holds costs no read (fill_locked's hit path) and is the same evidence -
                // those bytes were fetched ahead and are being used. Filtering here made the offer
                // disappear exactly when the read-ahead was working, which left `queued=0` and
                // `useful` undefined (a ratio with a zero denominator).
                absent.push_back(id);
            }
            if (!absent.empty()) {
                const auto tick = prefetch_layer_tick().fetch_add(1) + 1;
                const auto window = prefetch_window();
                if (window != 0 && (tick % window) == 0) {
                    ExpertPrefetchQueue::instance().advance(tick);
                }
                prefetch_queued_ += queue_prefetch_fill(std::span<const std::uint32_t>(absent),
                                                        bank_, tick, window);
            }
        }
    }
}

std::uint32_t ExpertStream::publish_static_table() {
    ring_snapshot_host_.assign(count_, ops::kRingAdmitAbsent);
    host_->snapshot_slots(bank_, ring_snapshot_host_);
    std::uint32_t present = 0;
    for (const auto slot : ring_snapshot_host_) {
        if (slot != HostExpertResidency::kAbsentSlot) { ++present; }
    }
    const std::size_t bytes = static_cast<std::size_t>(count_) * sizeof(std::uint32_t);
    if (!ring_candidates_device_ || ring_candidates_device_->bytes < bytes) {
        ring_candidates_device_ = std::make_unique<DeviceBuffer>(bytes);
    }
    ring_candidates_device_->copy_from_host(ring_snapshot_host_.data(), bytes);
    ++static_table_refreshes_;
    static_candidates_ready_ = true;
    static_candidates_count_ = present;
    static_table_generation_ = host_->bank_slot_generation(bank_);
    if (bank_ == 0 && std::getenv("NINFER_RING_PROBE") != nullptr) {
        std::fprintf(stderr, "[warm-static] table copy #%llu present=%u\n",
                     static_cast<unsigned long long>(static_table_refreshes_), present);
        std::fflush(stderr);
    }
    return present;
}

std::uint32_t ExpertStream::static_quota() const {
    if (host_ == nullptr) { return 0; }
    const auto room = host_->routed_slots_per_bank() > 1
                          ? host_->routed_slots_per_bank() - 1
                          : 0;
    if (const char* value = std::getenv("NINFER_WARM_STATIC_SLOTS");
        value != nullptr && *value != '\0') {
        const auto requested = static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10));
        const auto clamped    = std::min({requested, room, count_});
        if (clamped != requested) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                std::fprintf(stderr,
                             "[freeze] bank=%u: NINFER_WARM_STATIC_SLOTS=%u clamped to %u "
                             "(routed/bank=%u, count=%u) - a stripe with no dynamic slot would be "
                             "refused\n",
                             bank_, requested, clamped, host_->routed_slots_per_bank(), count_);
                std::fflush(stderr);
            }
        }
        return clamped;
    }
    if (freeze_mode_ && cache_ != nullptr) {
        // The host set is the complement of the Device set: the pool keeps the top `capacity`
        // experts, the tier keeps the rest of the bank. Clamped to the routed grant minus one
        // dynamic slot, because the static stripe and the dynamic stripe share it.
        const auto device = std::min(cache_->resident_capacity(bank_), count_);
        const auto keep   = count_ - device;
        return std::min({keep, room, count_});
    }
    const auto banks = std::max<std::uint32_t>(1U, host_->bank_count());
    auto quota = std::max<std::uint32_t>(8U, host_->routed_capacity() / banks);
    return std::min(quota, count_);
}

void ExpertStream::refresh_static_candidates() {
    if (!static_warm_ || host_ == nullptr) { return; }
    // P1: a frozen set is not refreshed. Re-ranking while the Device holds the set is exactly what
    // made the two halves disagree - the table moved every epoch while the pool did not, and every
    // move cost a republish.
    if (static_frozen_) { return; }
    const auto keep = static_quota();
    std::vector<std::uint32_t> ranked(count_);
    std::iota(ranked.begin(), ranked.end(), 0U);
    std::stable_sort(ranked.begin(), ranked.end(),
                     [&](std::uint32_t a, std::uint32_t b) { return hot_score_[a] > hot_score_[b]; });
    // P3: the host set is the complement of the Device set, not another copy of it. Both were taking
    // the top of the same score, so the 12 host slots named experts the pool already held and served
    // none of the routes that actually reached the host (measured: static_present=12, route_static=0).
    // Slicing at the Device's per-bank capacity makes the two sets disjoint by construction.
    std::size_t offset = 0;
    if (std::getenv("NINFER_WARM_STATIC_SLICE") != nullptr && cache_ != nullptr) {
        // Sliced at this bank's capacity: the host set must be the complement of this bank's Device
        // set, and per-bank pricing makes that a different number for a wide bank.
        offset = std::min<std::size_t>(cache_->resident_capacity(bank_), ranked.size());
        if (offset + keep > ranked.size()) {
            offset = ranked.size() - std::min<std::size_t>(keep, ranked.size());
        }
    }
    std::vector<std::int32_t> candidates(
        ranked.begin() + static_cast<std::ptrdiff_t>(offset),
        ranked.begin() + static_cast<std::ptrdiff_t>(offset + keep));
    static_member_.assign(count_, 0);
    for (const auto expert : candidates) {
        static_member_[static_cast<std::size_t>(expert)] = 1;
    }
    // Q20/B: with the split on, the set gets its own stripe and the fill may only evict inside it.
    // Without the split the old shared-range fill is kept byte-identical.
    static const bool reserve = std::getenv("NINFER_WARM_STATIC_RESERVE") != nullptr;
    if (reserve && !static_frozen_) { host_->configure_static_slots(keep); }
    host_->ensure_scored(ExpertBankId{bank_}, candidates, hot_score_.data(), reserve && !static_frozen_);
    publish_static_table();
    if (bank_ == 0 && std::getenv("NINFER_RING_PROBE") != nullptr) {
        std::fprintf(stderr, "[warm-static] bank=%u asked=%u held=%u per_bank=%u\n", bank_, keep,
                     static_candidates_count_, host_->static_slots_per_bank());
        std::fflush(stderr);
    }
}

// P1: choose the whole placement once, from one ranking, and never let the two halves drift.
//
// The Device keeps the top `capacity` of the decayed score; the host keeps the next `q`, so the
// two sets are disjoint by construction. `q` is the complement of the Device set (clamped to the
// routed grant), which is the capacity the plan measured: a host tier smaller than the complement
// churns, one equal to it never misses.
//
// The fill is synchronous on purpose. The set is 150 slots a bank on this model, and a freeze
// happens once per bank: the alternative - backfilling across layers - leaves routes outside both
// halves during the transition, which is the CPU divergence the gate exists to catch. One stall at
// the boundary is the honest trade until the backfill can be shown to be complete before a layer
// needs it.
void ExpertStream::probe_tables(const char* where) const {
    if (std::getenv("NINFER_TABLE_PROBE") == nullptr || bank_ != 0) { return; }
    std::uint64_t placement = 1469598103934665603ULL;
    std::uint64_t flags     = 1469598103934665603ULL;
    for (const auto word : placement_) {
        placement ^= word;
        placement *= 1099511628211ULL;
    }
    for (const auto flag : host_flags_) {
        flags ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(flag));
        flags *= 1099511628211ULL;
    }
    std::fprintf(stderr, "[table] bank=0 where=%s placement=%016llx flags=%016llx pool0=%d\n",
                 where, static_cast<unsigned long long>(placement),
                 static_cast<unsigned long long>(flags),
                 static_cast<int>(pool_slot_.empty() ? -2 : pool_slot_[0]));
    std::fflush(stderr);
}

void ExpertStream::probe_device_tables(const char* where, const void* device_ids,
                                       std::size_t id_values) const {
    if (std::getenv("NINFER_DEVICE_TABLE_PROBE") == nullptr || bank_ != 0) { return; }
    if (!offsets_device_ || !mask_device_ || !host_offsets_device_ || !host_flags_device_) { return; }
    static std::uint32_t shown = 0;
    if (shown >= 48) { return; }
    const auto seq = shown++;
    const auto hash_of = [](const void* device, std::size_t bytes) {
        std::vector<std::uint8_t> host(bytes);
        if (cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
            (void)cudaGetLastError();
            return std::uint64_t{0};
        }
        std::uint64_t h = 1469598103934665603ULL;
        for (const auto byte : host) {
            h ^= byte;
            h *= 1099511628211ULL;
        }
        return h;
    };
    const auto offsets = hash_of(offsets_device_->p, count_ * sizeof(std::uint64_t));
    const auto mask    = hash_of(mask_device_->p, count_ * sizeof(std::int32_t));
    const auto hoffs   = hash_of(host_offsets_device_->p, count_ * sizeof(std::uint64_t));
    const auto hflags  = hash_of(host_flags_device_->p, count_ * sizeof(std::int32_t));
    const auto ids     = device_ids == nullptr
                             ? std::uint64_t{0}
                             : hash_of(device_ids, id_values * sizeof(std::int32_t));
    std::uint64_t host_placement = 1469598103934665603ULL;
    for (const auto word : placement_) {
        host_placement ^= word;
        host_placement *= 1099511628211ULL;
    }
    // The last input the kernel has that no table hash covers: the bytes at the address the table
    // names. Same offset on a different base, or a different payload at the same offset, both read
    // as "identical addressing" from the tables alone.
    const auto base = reinterpret_cast<const std::byte*>(experts_.bank.gate_up.codes);
    const auto down_base = reinterpret_cast<const std::byte*>(experts_.bank.down.codes);
    std::int32_t first_id = -1;
    std::uint64_t slot_off = ~std::uint64_t{0};
    if (device_ids != nullptr) {
        if (cudaMemcpy(&first_id, device_ids, sizeof(first_id), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
            (void)cudaGetLastError();
            first_id = -1;
        }
    }
    if (first_id >= 0 && static_cast<std::size_t>(first_id) < count_) {
        std::vector<std::uint64_t> offs(count_);
        if (cudaMemcpy(offs.data(), offsets_device_->p, count_ * sizeof(std::uint64_t),
                       cudaMemcpyDeviceToHost) == cudaSuccess) {
            slot_off = offs[static_cast<std::size_t>(first_id)];
        } else {
            (void)cudaGetLastError();
        }
    }
    // NOTE: `codes + offsets[e]` is NOT the expert's payload - measuring it reads zero padding and
    // reports the same hash for different experts. Sampling the weights needs the plane addressing
    // (stride/layout), so this probe deliberately stops at the pointers and the offset value.
    std::fprintf(stderr,
                 "[devtab] seq=%u where=%s offsets=%016llx mask=%016llx host_offsets=%016llx "
                 "host_flags=%016llx ids=%016llx placement=%016llx gu=%p dn=%p hgu=%p id=%d "
                 "off=%llu\n",
                 seq, where, static_cast<unsigned long long>(offsets),
                 static_cast<unsigned long long>(mask), static_cast<unsigned long long>(hoffs),
                 static_cast<unsigned long long>(hflags), static_cast<unsigned long long>(ids),
                 static_cast<unsigned long long>(host_placement), static_cast<const void*>(base),
                 static_cast<const void*>(down_base),
                 static_cast<const void*>(experts_.bank.gate_up.host_codes), first_id,
                 static_cast<unsigned long long>(slot_off));
    std::fflush(stderr);
}

void ExpertStream::freeze_now(std::uint32_t /*tokens*/, cudaStream_t stream) {
    if (frozen_ || host_ == nullptr || cache_ == nullptr) { return; }
    if (freeze_trace()) {
        std::fprintf(stderr, "[freeze] bank=%u enter clock=%llu\n", bank_,
                     static_cast<unsigned long long>(freeze_clock_));
        std::fflush(stderr);
    }
    std::vector<std::uint32_t> ranked(count_);
    std::iota(ranked.begin(), ranked.end(), 0U);
    std::stable_sort(ranked.begin(), ranked.end(), [&](std::uint32_t a, std::uint32_t b) {
        if (hot_score_[a] != hot_score_[b]) { return hot_score_[a] > hot_score_[b]; }
        return a < b; // a deterministic tie-break: the same run must freeze the same set
    });
    // The frozen Device set is this bank's seats, so it is this bank's capacity that sizes it.
    const auto capacity = std::min<std::size_t>(cache_->resident_capacity(bank_), ranked.size());
    frozen_wanted_.assign(ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(capacity));
    const auto q = static_quota();
    std::vector<std::int32_t> candidates;
    const auto offset = std::min<std::size_t>(capacity, ranked.size());
    const auto take   = std::min<std::size_t>(q, ranked.size() - offset);
    candidates.reserve(take);
    for (std::size_t i = 0; i < take; ++i) {
        candidates.push_back(static_cast<std::int32_t>(ranked[offset + i]));
    }
    static_member_.assign(count_, 0);
    for (const auto expert : candidates) {
        static_member_[static_cast<std::size_t>(expert)] = 1;
    }
    // The stripe was installed at attach_host; if it was refused there is no static range to fill
    // into, and a static fill would silently fall back to the dynamic one. Say so once.
    if (host_->static_slots_per_bank() == 0 && !candidates.empty()) {
        std::fprintf(stderr,
                     "[freeze] bank=%u: static stripe is not configured (routed/bank=%u) - the "
                     "freeze keeps the Device set only\n",
                     bank_, host_->routed_slots_per_bank());
        std::fflush(stderr);
    }
    // Replace the warm-up contents wholesale. The reset refuses while a slot is still being read;
    // the holds rotate every layer, so a retry on the next ensure always succeeds. Freezing before
    // the rewrite lands would publish a table that names a half-rewritten stripe.
    if (host_->static_slots_per_bank() != 0 && !candidates.empty() &&
        !host_->reset_static_stripe(bank_)) {
        return;
    }
    if (freeze_trace()) {
        std::fprintf(stderr, "[freeze] bank=%u reset ok candidates=%zu\n", bank_,
                     candidates.size());
        std::fflush(stderr);
    }
    frozen_         = true;
    static_frozen_  = true;
    freeze_table_published_ = false;
    freeze_members_ = static_cast<std::uint32_t>(candidates.size());
    if (host_->static_slots_per_bank() != 0) { host_->set_static_frozen(bank_, false); }
    if (!candidates.empty()) {
        host_->ensure_scored(ExpertBankId{bank_}, candidates, hot_score_.data(),
                             /*static_fill=*/true);
    }
    if (host_->static_slots_per_bank() != 0) { host_->set_static_frozen(bank_, true); }
    freeze_present_ = host_->static_present(bank_, candidates);
    // Install the frozen device set NOW. The layer's normal `set_resident_experts` is gated on
    // `refresh`, which is false during prefill - so the freeze boundary could otherwise publish a
    // placement for a pool that still held the streaming set.
    if (cache_ != nullptr && !frozen_wanted_.empty()) {
        (void)cache_->set_resident_experts(bank_, frozen_wanted_,
                                           static_cast<std::uint32_t>(frozen_wanted_.size()));
        // Materialise the whole set before publishing anything. set_resident_experts only assigns
        // slots: every expert it places comes back `ready == false`, and a mask built from the
        // mapping alone calls 133 of 163 experts resident while the pool still holds the bytes of
        // whoever used those slots before. The layer then runs the right arithmetic on the wrong
        // weights, which is silent until the tokens drift. `covers_all` is a promise that a route
        // needs no read-back to be served, so it has to be a promise about the bytes.
        std::vector<ExpertCacheKey> keys;
        keys.reserve(frozen_wanted_.size());
        for (const auto expert : frozen_wanted_) { keys.push_back(ExpertCacheKey{bank_, expert}); }
        for (std::uint32_t turn = 0; turn < 256; ++turn) {
            if (cache_->unready_count(bank_, frozen_wanted_) == 0) { break; }
            cache_->prepare(keys, stream);
            cache_->wait_for_progress(stream);
        }
        freeze_pool_unready_ = cache_->unready_count(bank_, frozen_wanted_);
    }
    // Can every route be addressed without reading the ids back? This is the property a fully
    // static layer would need; kept as a counter, not as a branch (see the note at the covered
    // check: the 0.5 ms fetch is in ffn, not in ensure).
    const auto device = std::min<std::uint32_t>(cache_->resident_capacity(bank_), count_);
    std::uint32_t pool_missing = 0;
    for (const auto expert : frozen_wanted_) {
        if (!cache_->expert_slot(bank_, expert)) { ++pool_missing; }
    }
    freeze_pool_missing_ = pool_missing;
    freeze_covers_all_ = (freeze_members_ == count_ - device) &&
                         (freeze_present_ == freeze_members_) && (pool_missing == 0) &&
                         (freeze_pool_unready_ == 0);
    publish_static_table();
    std::fprintf(stderr,
                 "[freeze] bank=%u state=frozen device=%zu host_static=%u present=%u d=%u c=%u "
                 "p=%u covers_all=%d pool_missing=%u pool_unready=%u clock=%llu\n",
                 bank_, frozen_wanted_.size(), freeze_members_, freeze_present_,
                 host_->dynamic_slots_per_bank(), host_->second_chance_per_bank(),
                 host_->prefetch_slots_per_bank(), int(freeze_covers_all_),
                 freeze_pool_missing_, freeze_pool_unready_,
                 static_cast<unsigned long long>(freeze_clock_));
    std::fflush(stderr);
    ++freeze_membership_moves_; // the one move a set is allowed: the move into the freeze
}

void ExpertStream::set_freeze_policy(const FreezePolicy& policy) {
    freeze_after_    = policy.freeze_after;
    freeze_refresh_  = policy.reselect_after;
    thaw_on_request_ = policy.thaw_on_request;
    if (bank_ == 0) {
        std::fprintf(stderr,
                     "[freeze] policy: freeze_after=%u reselect_after=%u thaw_on_request=%d\n",
                     freeze_after_, freeze_refresh_, int(thaw_on_request_));
        std::fflush(stderr);
    }
}

ExpertStream::FreezePolicy ExpertStream::freeze_policy() const noexcept {
    return FreezePolicy{freeze_after_, freeze_refresh_, thaw_on_request_};
}

bool ExpertStream::freeze_armed() const noexcept {
    // freeze_mode_ is NINFER_HOST_FREEZE (the mode installs its own set on a clock); the rule is
    // the Q11/Q12 measured freeze (NINFER_FREEZE_RULE, or the old switch name). Either way the
    // freeze decision is live in this stream, and the controller should say so.
    return freeze_mode_ || freeze_rule_enabled();
}

void ExpertStream::note_request_begin() {
    if (thaw_on_request_) { thaw(); }
}

void ExpertStream::thaw() {
    if (!freeze_mode_ || (!frozen_ && !static_frozen_)) { return; }
    frozen_         = false;
    static_frozen_  = false;
    freeze_clock_   = 0;
    freeze_present_ = 0;
    freeze_covers_all_ = false;
    freeze_table_published_ = false;
    frozen_wanted_.clear();
    for (auto& set : split_frozen_) { set.clear(); }
    static_member_.assign(count_, 0);
    // The stripe returns to warm-up: its fill may churn again until the next freeze rewrites it.
    if (host_ != nullptr) { host_->set_static_frozen(bank_, false); }
    if (bank_ == 0) {
        std::fprintf(stderr, "[freeze] bank=0 thawed (placement follows the router again)\n");
        std::fflush(stderr);
    }
}

std::size_t ExpertStream::device_admit_ring(const void* device_ids, std::int32_t values,
                                            std::uint32_t ring_base, std::uint32_t ring_slots,
                                            cudaStream_t stream, std::vector<std::uint32_t>& wanted,
                                            std::vector<bool>& chosen) {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_RING_DEVICE_ADMIT");
        return value != nullptr && *value != '\0' && std::strtol(value, nullptr, 10) != 0;
    }();
    if (!enabled || host_ == nullptr || device_ids == nullptr || values <= 0 || ring_slots == 0 ||
        count_ == 0) {
        return 0;
    }
    const std::byte* arena = host_->arena_device_base();
    if (arena == nullptr || host_->slot_bytes() != cache_->slot_bytes()) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::fprintf(stderr,
                         "[ring] device admission disabled: no mapped arena or slot size mismatch "
                         "(host %llu, pool %llu)\n",
                         static_cast<unsigned long long>(host_->slot_bytes()),
                         static_cast<unsigned long long>(cache_->slot_bytes()));
            std::fflush(stderr);
        }
        return 0;
    }
    const std::size_t candidate_bytes = static_cast<std::size_t>(count_) * sizeof(std::uint32_t);
    if (!ring_candidates_device_ || ring_candidates_device_->bytes < candidate_bytes) {
        ring_candidates_device_ = std::make_unique<DeviceBuffer>(candidate_bytes);
    }
    if (!ring_resident_device_ || ring_resident_device_->bytes < candidate_bytes) {
        ring_resident_device_ = std::make_unique<DeviceBuffer>(candidate_bytes);
    }
    if (!ring_descriptors_device_) {
        ring_descriptors_device_ =
            std::make_unique<DeviceBuffer>(
                static_cast<std::size_t>(cache_->resident_capacity(bank_)) *
                sizeof(ops::RingAdmitDescriptor));
    }
    if (!ring_count_device_) { ring_count_device_ = std::make_unique<DeviceBuffer>(sizeof(std::uint32_t)); }
    // Membership at this instant, not the previous layer's published mask: "already resident" is
    // what makes an expert not a miss, and a hot expert admitted into the ring would lose its slot.
    ring_resident_host_.assign(count_, 0);
    for (std::uint32_t expert = 0; expert < count_; ++expert) {
        if (cache_->expert_slot(bank_, expert)) { ring_resident_host_[expert] = 1; }
    }
    ring_resident_device_->copy_from_host(ring_resident_host_.data(), candidate_bytes);
    std::size_t candidates_seen = static_candidates_count_;
    // The Device-side table is only valid for the warm tier generation it was copied at. Any fill
    // (this bank's uncovered fallback, or another bank's fill) bumps that generation; a mismatch
    // means a slot this table names may already hold another expert, so refresh before admitting.
    if (!(static_warm_ && static_candidates_ready_ &&
          host_->bank_slot_generation(bank_) == static_table_generation_)) {
        candidates_seen = publish_static_table();
    }
    if (bank_ == 0 && std::getenv("NINFER_RING_PROBE") != nullptr) {
        static int shown = 0;
        if (shown++ < 4) {
            std::fprintf(stderr, "[ring] admit call values=%d ring=%u base=%u warm_candidates=%zu\n",
                         values, ring_slots, ring_base, candidates_seen);
            std::fflush(stderr);
        }
    }
    ring_count_device_->fill(0);
    const auto capacity = std::min(cache_->resident_capacity(bank_), ring_slots);
    ops::ring_admit_launch(device_ids, values,
                           static_cast<const std::uint32_t*>(ring_candidates_device_->p),
                           static_cast<const std::int32_t*>(ring_resident_device_->p), count_,
                           ring_base, ring_slots,
                           static_cast<ops::RingAdmitDescriptor*>(ring_descriptors_device_->p),
                           static_cast<std::uint32_t*>(ring_count_device_->p), capacity, stream);
    // The queue head is the only thing the host must see before it can act; the copies themselves
    // stay on the stream and are ordered before the layer that reads them.
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        throw std::runtime_error(std::string("ring admission: ") +
                                 cudaGetErrorString(cudaGetLastError()));
    }
    std::uint32_t requested = 0;
    ring_count_device_->copy_to_host(&requested, sizeof(requested));
    requested = std::min(requested, capacity);
    if (bank_ == 0 && std::getenv("NINFER_RING_PROBE") != nullptr) {
        static int shown = 0;
        if (shown++ < 4) {
            std::fprintf(stderr, "[ring] descriptors=%u capacity=%u\n", requested, capacity);
            std::fflush(stderr);
        }
    }
    if (requested == 0) { return 0; }
    ring_descriptors_host_.resize(requested);
    ring_descriptors_device_->copy_to_host(
        ring_descriptors_host_.data(),
        static_cast<std::size_t>(requested) * sizeof(ops::RingAdmitDescriptor));
    const std::size_t slot_bytes = cache_->slot_bytes();
    std::size_t admitted = 0;
    for (const auto& descriptor : ring_descriptors_host_) {
        if (descriptor.expert >= count_ || descriptor.ring_slot >= ring_slots) { continue; }
        const auto mapped = cache_->ring_slot(bank_, descriptor.ring_slot);
        if (!mapped) { continue; }
        const std::uint32_t ring_slot = *mapped;
        const std::byte* source = arena + static_cast<std::size_t>(descriptor.host_slot) *
                                              host_->slot_bytes();
        std::byte* destination =
            const_cast<std::byte*>(static_cast<const std::byte*>(cache_->pool().data)) +
            cache_->slot_offset(ring_slot);
        // The copy reads the pinned slot asynchronously; keep the tier from handing it out until a
        // later layer has waited on the stream.
        host_->hold_for_device(bank_, descriptor.expert);
        if (cudaMemcpyAsync(destination, source, slot_bytes, cudaMemcpyDeviceToDevice, stream) !=
            cudaSuccess) {
            throw std::runtime_error(std::string("ring admission copy: ") +
                                     cudaGetErrorString(cudaGetLastError()));
        }
        if (!cache_->adopt_resident_slot(bank_, descriptor.expert, ring_slot)) { continue; }
        if (bank_ == 0 && std::getenv("NINFER_RING_PROBE") != nullptr) {
            static bool checked = false;
            if (!checked) {
                checked = true;
                std::vector<std::byte> from_pool(slot_bytes), from_host(slot_bytes);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaMemcpy(from_pool.data(), destination, slot_bytes,
                                      cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(from_host.data(), source, slot_bytes,
                                      cudaMemcpyDeviceToHost));
                std::fprintf(stderr, "[ring] copy check expert=%u slot=%u bytes=%zu %s\n",
                             descriptor.expert, ring_slot, slot_bytes,
                             std::memcmp(from_pool.data(), from_host.data(), slot_bytes) == 0
                                 ? "identical"
                                 : "MISMATCH");
                std::fflush(stderr);
            }
        }
        if (!chosen[descriptor.expert]) {
            chosen[descriptor.expert] = true;
            wanted.push_back(descriptor.expert);
        }
        ++admitted;
    }
    return admitted;
}

// port (sm_89): the layer as three runs.
//
// The router names an expert, not a matrix, so all three runs hold the same expert *identity* - but
// sized per run, because the cache prices each run at its own stride and the q6 down matrix is wider
// than the q4 gate/up pair (on Qwen4Exp: 1.28 MiB against 0.87 MiB, so the down run keeps fewer
// experts for the same bytes). Everything below is per-run; nothing is derived from a sibling run's
// slot numbering.
//
// Order inside the layer, and why it is this order:
//
//   1. read the routes back (or take the caller's copy) - the placement needs them;
//   2. per run: the set is this step's routes first (or the held/frozen set), then the decayed
//      score, capped by that run's own seat count; set_resident_experts() installs it;
//   3. the pinned tier is asked to hold, per run, the routes that run's pool will NOT hold. The fill
//      happens before the tables are built because the tables name slots: a fill that ran afterwards
//      would name the previous layer's expert (the fused path measured exactly that);
//   4. per run: one placement word per expert (pool slot / pinned slot / absent) is expanded on the
//      Device into that run's offsets, mask and host flags - with that run's pool stride and bias;
//   5. one prepare()/wait_for_progress() for all three runs' routed keys - the uploader is per key,
//      so the three runs share one turn instead of three;
//   6. rebase the views and compare each run's published generation against the cache's.
//
// The mask that reaches the kernels means "this route resolves to an address" (pool or pinned tier),
// which is the same rule GH3 writes and the same rule ffn's CPU complement reads: MaskZeros over it
// is "neither leg has this expert", not "the pool lacks it".
namespace {
// The layer policy constants, shared with the fused path (see the notes at their use sites there).
constexpr std::int32_t kSplitSetUpdateTokens = 8;
// One criterion instead of two constants: a pool that seats fewer than this many experts of a layer
// cannot hold a layer's working set, so a set built from a prefill is a set decode did not ask for
// and neither form protects it. Both forms used to spell this number out separately.
constexpr std::uint32_t kPoolCoversLayerSlots = 96;
constexpr bool pool_covers_layer(std::uint32_t slots) { return slots >= kPoolCoversLayerSlots; }
} // namespace

// port (sm_89): one run's tables, from one placement word.
//
// `placement_mode` says the pinned tier can be addressed by the Device. It is re-derived on EVERY
// layer even when the pool's set did not move: the tier keeps churning while the pool sits still, so
// "the pool did not move" says nothing about whether a host slot still holds the expert the table
// names. The held set (`hold_slots`) is exactly the slots this layer's table names.
void ExpertStream::publish_split_run(std::size_t index, cudaStream_t stream, bool placement_mode) {
    auto& run = split_[index];
    // Diagnostic: one marker per call, so a trace can be read per token instead of as one stream.
    if (const char* wanted = std::getenv("NINFER_SLOT_TRACE")) {
        if (static_cast<std::uint32_t>(std::atoi(wanted)) == run.bank) {
            static std::FILE* calls = std::fopen((std::string(wanted) + ".calls").c_str(), "wb");
            if (calls != nullptr) {
                static std::uint64_t seq = 0;
                std::fprintf(calls, "%llu %u %u\n", static_cast<unsigned long long>(++seq),
                             run.bank, count_);
                std::fflush(calls);
            }
        }
    }
    if (run.pool_slot.size() != count_) { run.pool_slot.assign(count_, -1); }
    for (std::uint32_t expert = 0; expert < count_; ++expert) {
        const auto slot       = cache_->expert_slot(run.bank, expert);
        const auto previous   = run.pool_slot[expert];
        run.pool_slot[expert] = slot ? static_cast<std::int32_t>(*slot) : -1;
        // Leaving the pool is the one signal that says "this expert was hot and is not hot enough
        // any more": it is the warm tier's admission criterion, and the gate run's journal is the
        // one the model reads.
        if (previous >= 0 && run.pool_slot[expert] < 0 && index == 0) {
            demoted_.push_back(static_cast<std::int32_t>(expert));
        }
    }
    if (run.placement.size() != count_) {
        run.placement.assign(count_, ops::kExpertPlacementAbsent);
        run.placement_published.assign(count_, ops::kExpertPlacementAbsent);
        run.host_slots.assign(count_, HostExpertResidency::kAbsentSlot);
        run.host_flags.assign(count_, 0);
    }
    const auto* host_base = placement_mode ? host_->arena_device_base() : nullptr;
    bool tier_usable      = host_base != nullptr;
    if (tier_usable) {
        std::uint64_t gu_plane[3] = {}, dn_plane[3] = {}, gu_high = 0, dn_high = 0;
        if (host_->plane_offsets(run.bank, gu_plane, dn_plane, gu_high, dn_high)) {
            for (std::size_t k = 0; k < 3; ++k) { run.host_plane[k] = gu_plane[k]; }
            run.host_high_bytes = gu_high;
            host_->snapshot_slots(run.bank, run.host_slots);
        } else {
            // The tier cannot describe this bank: nothing it holds is offerable, so every route the
            // pool does not hold goes to the CPU half rather than to a slot table we cannot trust.
            tier_usable = false;
        }
    }

    bool changed = !run.placement_published_once;
    run.placed_slots.clear();
    run.placed_slots.reserve(count_);
    // Hoisted out of the 512-row loop: std::getenv() walks the environment block on this platform,
    // and running it once per expert (x3 banks a layer) measured 3.2 ms of pure CPU per layer -
    // more than the upload it was sitting next to. The trace switch is read once per publish.
    static const char* const slot_trace_env = std::getenv("NINFER_SLOT_TRACE");
    const bool trace_bank =
        slot_trace_env != nullptr &&
        static_cast<std::uint32_t>(std::atoi(slot_trace_env)) == run.bank;
    static std::FILE* slot_trace_file = nullptr;
    static bool slot_trace_opened     = false;
    if (trace_bank && !slot_trace_opened) {
        slot_trace_file =
            std::fopen((std::string(slot_trace_env) + ".slots").c_str(), "wb");
        slot_trace_opened = true;
    }
    for (std::uint32_t expert = 0; expert < count_; ++expert) {
        // The pool wins a tie: an expert resident in both is addressed in video memory.
        //
        // P3: an expert the layer's CPU half owns is absent as far as this publication is
        // concerned - even if a pool seat still happens to hold it, and even if the pinned tier
        // does. The mask is what the GEMM's MaskOnes subset reads, so marking it absent here is
        // what makes the CPU half's half of the routes the CPU's: one place decides, and the
        // Device never sees an address for a route it is not supposed to compute.
        const bool cpu_half =
            !split_cpu_only_[index].empty() && split_cpu_only_[index][expert] != 0;
        std::uint32_t placement = ops::kExpertPlacementAbsent;
        if (!cpu_half) {
            if (run.pool_slot[expert] >= 0) {
                placement = static_cast<std::uint32_t>(run.pool_slot[expert]);
            } else if (tier_usable && run.host_slots[expert] != HostExpertResidency::kAbsentSlot) {
                placement = ops::kExpertPlacementHost | run.host_slots[expert];
            }
        }
        if (placement != run.placement_published[expert]) { changed = true; }
        run.placement[expert] = placement;
        // Diagnostic: NINFER_SLOT_TRACE=<bank> records every placement change of one bank, so two
        // artifacts that upload identical images can be compared on what the kernel is told.
        if (trace_bank && placement != run.placement_published[expert] &&
            slot_trace_file != nullptr) {
            std::fprintf(slot_trace_file, "%u %u %d %d\n", run.bank, expert,
                         static_cast<int>(run.placement_published[expert]),
                         static_cast<int>(placement));
            std::fflush(slot_trace_file);
        }
        const bool absent     = placement == ops::kExpertPlacementAbsent;
        const bool on_host    = !absent && (placement & ops::kExpertPlacementHost) != 0;
        run.host_flags[expert] = on_host ? 1 : 0;
        if (on_host) { run.placed_slots.push_back(placement & ~ops::kExpertPlacementHost); }
    }
    if (tier_usable) { host_->hold_slots(run.placed_slots); }

    // The host mirror of the tables the Device derives. Readers that are not kernels ask for these
    // (the tests, the coverage attribution, the demotion journal); they are the same four rules
    // `moe_expand_expert_placements` applies, so the two cannot disagree about a resident expert.
    if (run.offsets.size() != count_) { run.offsets.assign(count_, 0); }
    if (run.mask.size() != count_) { run.mask.assign(count_, 0); }
    std::uint32_t pool_resident = 0, host_resident = 0;
    for (std::uint32_t expert = 0; expert < count_; ++expert) {
        const auto placement = run.placement[expert];
        const bool absent    = placement == ops::kExpertPlacementAbsent;
        const bool on_host   = !absent && (placement & ops::kExpertPlacementHost) != 0;
        run.mask[expert]    = absent ? 0 : 1;
        run.offsets[expert] = absent || on_host
                                  ? 0
                                  : cache_->slot_offset(static_cast<std::uint32_t>(placement));
        pool_resident += on_host || absent ? 0U : 1U;
        host_resident += on_host ? 1U : 0U;
    }
    run.resident        = pool_resident;
    run.host_routes     = 0; // filled by ensure_split from the routes it just placed
    run.host_leg        = tier_usable && host_resident != 0;

    if (run.placement_device == nullptr) {
        run.placement_device   = std::make_unique<DeviceBuffer>(count_ * sizeof(std::uint32_t));
        run.placement_stage[0] = std::make_unique<PinnedHostBuffer>(count_ * sizeof(std::uint32_t));
        run.placement_stage[1] = std::make_unique<PinnedHostBuffer>(count_ * sizeof(std::uint32_t));
    }
    if (run.host_leg) {
        if (run.host_offsets_device == nullptr) {
            run.host_offsets_device = std::make_unique<DeviceBuffer>(count_ * sizeof(std::uint64_t));
            run.host_flags_device   = std::make_unique<DeviceBuffer>(count_ * sizeof(std::int32_t));
        }
    }
    auto& view = mutable_view(index);
    if (run.host_leg) {
        view.host_codes   = host_base + run.host_plane[0];
        view.host_high    = run.host_high_bytes != 0 ? host_base + run.host_plane[1] : nullptr;
        view.host_scales  = host_base + run.host_plane[2];
        view.host_offsets = static_cast<const std::uint64_t*>(run.host_offsets_device->p);
        view.host_flags   = static_cast<const std::int32_t*>(run.host_flags_device->p);
    } else {
        // Leaving last layer's addresses behind would be worse than having none: the kernel would
        // keep reading the tier for a route the pool now expects to own.
        view.host_codes   = nullptr;
        view.host_high    = nullptr;
        view.host_scales  = nullptr;
        view.host_offsets = nullptr;
        view.host_flags   = nullptr;
    }

    if (changed) {
        // The publication is one word per expert: 4 bytes x 512 = 2 KiB, which the Device expands
        // into the four tables the GEMM reads. A whole-table copy is cheaper than the bookkeeping a
        // per-row batch would need at this size - and the driver refused the batch form on this
        // platform (cudaMemcpyBatchAsync -> invalid argument, measured here), so the table goes over
        // in one submission per run.
        auto& stage = run.placement_stage[run.placement_stage_next];
        run.placement_stage_next ^= 1;
        std::memcpy(stage->data(), run.placement.data(), count_ * sizeof(std::uint32_t));
        if (cudaMemcpyAsync(run.placement_device->p, stage->data(),
                            count_ * sizeof(std::uint32_t), cudaMemcpyHostToDevice, stream) !=
            cudaSuccess) {
            throw std::runtime_error(std::string("expert split placement copy: ") +
                                     cudaGetErrorString(cudaGetLastError()));
        }
        ops::moe_expand_expert_placements(
            static_cast<const std::uint32_t*>(run.placement_device->p), static_cast<int>(count_),
            cache_->slot_stride(run.bank), cache_->slot_bias(run.bank),
            run.host_leg ? host_->slot_bytes() : 0,
            static_cast<std::uint64_t*>(run.offsets_device->p),
            static_cast<std::int32_t*>(run.mask_device->p),
            run.host_leg ? static_cast<std::uint64_t*>(run.host_offsets_device->p) : nullptr,
            run.host_leg ? static_cast<std::int32_t*>(run.host_flags_device->p) : nullptr, stream);
        run.placement_published = run.placement;
        run.placement_published_once = true;
        publish_calls_ += 1;
        publish_bytes_ += static_cast<std::uint64_t>(count_) * sizeof(std::uint32_t);
        if (const auto latched = cudaGetLastError(); latched != cudaSuccess) {
            // A placement that did not reach the Device is not a slowdown: the kernels would read the
            // previous layer's table. Refuse it here rather than compute with it.
            throw std::runtime_error(std::string("expert split placement publish: ") +
                                     cudaGetErrorString(latched));
        }
    }

    run.published    = true;
    run.pub_hot      = cache_->hot_generation(run.bank);
    run.pub_step     = cache_->step_generation(run.bank);
    run.pub_prefetch = cache_->prefetch_generation(run.bank);
    run.table_hot      = run.pub_hot;
    run.table_step     = run.pub_step;
    run.table_prefetch = run.pub_prefetch;
}

// The frozen pool set: this run's seats, taken from the decayed ranking once and held. The pinned
// tier keeps filling whatever the pool does not hold, layer by layer; what the freeze removes is the
// pool's churn, not the second leg.
void ExpertStream::freeze_split(cudaStream_t stream) {
    if (frozen_ || cache_ == nullptr) { return; }
    std::vector<std::uint32_t> ranked(count_);
    std::iota(ranked.begin(), ranked.end(), 0U);
    std::stable_sort(ranked.begin(), ranked.end(), [&](std::uint32_t a, std::uint32_t b) {
        if (hot_score_[a] != hot_score_[b]) { return hot_score_[a] > hot_score_[b]; }
        return a < b; // deterministic: the same run must freeze the same set
    });
    std::vector<ExpertCacheKey> preload;
    for (std::size_t index = 0; index < split_.size(); ++index) {
        const auto bank_seats = std::min<std::size_t>(
            cache_->resident_capacity(split_[index].bank), ranked.size());
        auto& set = split_frozen_[index];
        set.assign(ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(bank_seats));
        (void)cache_->set_resident_experts(split_[index].bank, set,
                                           static_cast<std::uint32_t>(set.size()));
        for (const auto expert : set) {
            preload.push_back(ExpertCacheKey{split_[index].bank, expert});
        }
    }
    // Materialise the whole set before anything publishes it: a mask built from the mapping alone
    // calls an expert resident while the pool still holds whoever used that slot before.
    for (std::uint32_t turn = 0; turn < 256 && !preload.empty(); ++turn) {
        bool unready = false;
        for (std::size_t index = 0; index < split_.size() && !unready; ++index) {
            if (cache_->unready_count(split_[index].bank, split_frozen_[index]) != 0) {
                unready = true;
            }
        }
        if (!unready) { break; }
        cache_->prepare(std::span<const ExpertCacheKey>(preload), stream);
        cache_->wait_for_progress(stream);
    }
    frozen_ = true;
    std::fprintf(stderr,
                 "[freeze] bank=%u state=frozen gate/up/down sets=%zu/%zu/%zu (the pool placement "
                 "is held; the pinned tier keeps serving the rest every layer, and the static host "
                 "stripe stays on the fused form because its quota is one number per bank while "
                 "these three runs have three complement sizes)\n",
                 bank_, split_frozen_[0].size(), split_frozen_[1].size(), split_frozen_[2].size());
    std::fflush(stderr);
}

bool ExpertStream::ensure_split(const void* device_ids, std::int32_t top_k, std::int32_t tokens,
                                cudaStream_t stream, const std::int32_t* host_ids) {
    if (top_k <= 0 || tokens <= 0) {
        throw std::invalid_argument("expert stream needs a positive route grid");
    }
    const auto values = static_cast<std::size_t>(top_k) * static_cast<std::size_t>(tokens);
    const bool timing = std::getenv("NINFER_SPLIT_TIMING") != nullptr;
    // The model's routing width, for the prefetch port (see prefetch_plan): recorded here because this
    // is the one place that is told it, and the port runs outside the layer.
    top_k_ = top_k;

    std::uint32_t capacity[3] = {};
    bool fully                = true;
    std::uint32_t narrowest   = 0xffffffffu;
    for (std::size_t index = 0; index < split_.size(); ++index) {
        const auto& run = split_[index];
        if (cache_ == nullptr || !run.attached) {
            throw std::logic_error("expert stream is not attached");
        }
        capacity[index] = cache_->resident_capacity(run.bank);
        // A run with no seat cannot hold a single route, so this layer cannot run on the pool at
        // all. False hands the whole routed term to the caller's fallback, which is the honest
        // answer: two of three runs resident is not three.
        if (capacity[index] == 0) { return false; }
        if (capacity[index] < count_) { fully = false; }
        narrowest = std::min(narrowest, capacity[index]);
    }
    ++gh1_calls_;
    if (!fully) {
        ++gh1_split_calls_;
    } else {
        ++gh1_no_miss_calls_;
        ++gh1_static_calls_;
    }

    // port (sm_89): a pinned pool holds every expert of every run from construction, so the window
    // cannot move and the router does not have to be read back.
    if (pinned_) {
        bool resident = true;
        for (const auto& run : split_) {
            if (!cache_->window_resident(run.bank)) { resident = false; break; }
        }
        if (resident) {
            rebase_split();
            last_cost_ = StepCost{};
            ++gh1_no_miss_calls_;
            ++gh1_static_calls_;
            ++gh1_capt_free_calls_;
            ++gh1_capt_ready_calls_;
            return true;
        }
        // A resize dropped one of the runs; fall through to the normal path.
    }

    // P1: the freeze clock counts ensure() calls, not branch hits - a covered layer must still
    // reach the boundary that freezes the set.
    if (freeze_mode_) {
        ++freeze_clock_;
        if (frozen_ && freeze_refresh_ != 0 && freeze_clock_ >= freeze_refresh_) {
            frozen_        = false;
            freeze_clock_  = 0;
            for (auto& set : split_frozen_) { set.clear(); }
            if (bank_ == 0) {
                std::fprintf(stderr, "[freeze] bank=0 re-select after %u calls\n", freeze_refresh_);
                std::fflush(stderr);
            }
        } else if (!frozen_ && freeze_after_ != 0 && freeze_clock_ >= freeze_after_) {
            freeze_split(stream);
        }
    }

    double route_ms               = 0.0;
    const std::int32_t* route_ids = host_ids;
    if (route_ids == nullptr) {
        const auto bytes = values * sizeof(std::int32_t);
        if (!route_staging_ || route_staging_->size() < bytes) {
            route_staging_ = std::make_unique<PinnedHostBuffer>(bytes);
        }
        const auto begin  = std::chrono::steady_clock::now();
        const auto status = cudaMemcpyAsync(route_staging_->data(), device_ids, bytes,
                                            cudaMemcpyDeviceToHost, stream);
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("route copy: ") + cudaGetErrorString(status));
        }
        // P1: the prediction ring's copies are queued in the same breath, so the one synchronise below
        // covers both reads. This is the whole "no new per-layer synchronise" claim: the layer had one
        // route read-back and still has exactly one, and the ring rides it.
        pred_ring_begin_fetch(stream);
        const auto sync = cudaStreamSynchronize(stream);
        if (sync != cudaSuccess) {
            throw std::runtime_error(std::string("route sync: ") + cudaGetErrorString(sync));
        }
        route_ms  = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                             begin).count();
        route_ids = static_cast<const std::int32_t*>(route_staging_->data());
    } else {
        // The caller handed the routes over already read, and this path does not synchronise at all
        // (it is the prefill/static one). Draining the ring here would need a wait this path does not
        // have, so it is left alone: the entry stays in the ring and is counted as dropped when the
        // consumer next runs. Adding a synchronise to keep a prediction would be a worse trade than
        // losing the prediction, and the count says which happened.
    }
    for (std::size_t i = 0; i < values; ++i) {
        const auto expert = route_ids[i];
        if (expert < 0 || static_cast<std::uint32_t>(expert) >= count_) {
            throw std::out_of_range("expert stream received a route outside its bank");
        }
    }
    // P2/`prev`: record this layer's set for the NEXT step's prediction. Recorded here, where the ids
    // are already on the host, and read by the previous layer's stage_history() - which is the entry
    // that puts the fill one layer EARLY instead of one layer late. A no-op unless the signal says
    // prev, so the residual path and the default path are untouched.
    note_routes(std::span<const std::int32_t>(route_ids, values));
    // S3-4: the prediction region's numerator, counted where the routes are - and only when EVERY run
    // holds the expert in its region, because a route one run cannot address is the CPU half's
    // whatever the other two hold. Until S3 this loop existed only on the fused path, so
    // `[pf-region] hits=0` on the gate/up/down form was a property of the counter, not of the region.
    if (cache_ != nullptr && cache_->prefetch_slots_per_bank() != 0) {
        for (std::size_t i = 0; i < values; ++i) {
            ++prefetch_route_samples_;
            const auto expert = static_cast<std::uint32_t>(route_ids[i]);
            bool all_runs    = true;
            for (const auto& run : split_) {
                const auto slot = cache_->expert_slot(run.bank, expert);
                if (!slot || !cache_->prefetch_slot(*slot)) { all_runs = false; break; }
            }
            if (all_runs) { ++prefetch_hits_; }
        }
    }
    // P1: drain the prediction ring here. The order is the measurement's correctness:
    //
    //   * after the region's hit count above, so that rate measures what the region held when the layer
    //     ARRIVED - filling first would make every route a hit and the number meaningless (the mistake
    //     GH1's `no_miss` made once already);
    //   * before the precision count below, so a drained guess is attributed to the layer that used it.
    //
    // The drain itself waits for nothing: its copies were queued in front of the layer's own route
    // synchronise, and this reads the pinned landing bytes that call already ordered.
    if (pred_ring_switched()) {
        std::vector<std::int32_t> drained;
        const auto delivered =
            pred_ring_complete(drained, static_cast<std::int32_t>(values));
        if (delivered > 0) {
            set_predicted(std::move(drained));
            // The fill, through the same host-form path a residual guess uses: predicted-and-not-
            // resident goes into the region and the copies are queued on this stream, in front of the
            // layer's own GEMMs. It is queued here, at the ensure of the layer that was predicted -
            // which is the honest place for a signal produced at the end of the layer before it.
            if (cache_ != nullptr && cache_->prefetch_slots_per_bank() != 0) {
                stage_predicted(stream);
            }
        }
    }
    // S3-4: how good the guess was, independent of whether it was placed. The region's hit rate
    // confounds two questions - "was the router predicted" and "did the fill survive to the layer that
    // reads it" - and a hit rate near zero cannot say which failed. This counts the first directly:
    // predicted id set against the routes this layer actually produced. Host form only (the Device
    // form's ids never land here by construction).
    if (!predicted_.empty()) {
        ++prediction_layers_checked_;
        prediction_ids_seen_ += predicted_.size();
        for (const auto expert : predicted_) {
            for (std::size_t i = 0; i < values; ++i) {
                if (route_ids[i] == expert) {
                    ++prediction_overlap_;
                    break;
                }
            }
        }
    }
    // port (sm_89): the three-run form has no route trace of its own, so a divergence between two
    // artifacts that carry identical weights cannot be attributed to routing at all. Same shape as
    // the fused trace: one line per (bank, token) with the experts that token routed to.
    if (const char* dump = std::getenv("NINFER_EXPERT_ROUTE_DUMP")) {
        static std::FILE* trace = std::fopen(dump, "wb");
        if (trace != nullptr) {
            for (std::size_t token = 0; token < static_cast<std::size_t>(tokens); ++token) {
                for (std::size_t index = 0; index < split_.size(); ++index) {
                    std::fprintf(trace, "%u %zu", split_[index].bank, token);
                    for (std::int32_t k = 0; k < top_k; ++k) {
                        std::fprintf(
                            trace, " %d",
                            route_ids[static_cast<std::size_t>(k) * static_cast<std::size_t>(tokens) +
                                      token]);
                    }
                    std::fputc('\n', trace);
                }
            }
            std::fflush(trace);
        }
    }
    const auto layout_begin = std::chrono::steady_clock::now();

    // port (sm_89): whether the pinned tier can be addressed by the Device decides whether this
    // layer must re-derive its placement every time (it must: the tier churns under a still pool).
    const bool placement_mode = host_ != nullptr && host_->arena_device_base() != nullptr &&
                                std::getenv("NINFER_SKIP_PLACEMENT") == nullptr;

    // A layer needs nothing when every route already has a ready image in every run - and, with a
    // live second leg, only when the placement it published is still true, which is a per-layer
    // property of the tier and not of the pool. So the covered shortcut is taken on the pool-only
    // path and nowhere else.
    bool covered = !placement_mode;
    // Count residency over EVERY route against EVERY run before the shortcut above can return: the
    // loop below breaks at the first route that is not held, which is what the shortcut needs but not
    // what a rate needs.
    if (!split_.empty()) {
        std::uint32_t held_routes = 0;
        for (std::size_t i = 0; i < values; ++i) {
            const auto expert = static_cast<std::uint32_t>(route_ids[i]);
            bool held = true;
            for (const auto& run : split_) {
                if (!cache_->expert_slot(run.bank, expert) ||
                    !cache_->offset(ExpertCacheKey{run.bank, expert})) {
                    held = false;
                    break;
                }
            }
            if (held) { ++held_routes; }
        }
        residency_routes_ += values;
        residency_inside_ += held_routes;
    }
    for (std::size_t index = 0; index < split_.size() && covered; ++index) {
        const auto bank = split_[index].bank;
        for (std::size_t i = 0; i < values; ++i) {
            const auto expert = static_cast<std::uint32_t>(route_ids[i]);
            if (!cache_->expert_slot(bank, expert) || !cache_->offset(ExpertCacheKey{bank, expert})) {
                covered = false;
                break;
            }
        }
    }
    if (covered) {
        ++gh1_static_calls_;
        ++gh1_no_miss_calls_;
        ++gh1_capt_free_calls_;
        ++gh1_capt_ready_calls_;
        coverage_.routes += values;
        coverage_.inside += values;
        last_cost_          = StepCost{};
        last_cost_.route_ms = route_ms;
        rebase_split();
        return true;
    }

    // Whether this layer may change the pool's resident set. A prefill on a pool that can hold most
    // of a layer would otherwise replace the decode set with experts decode never asks for; a pool
    // that cannot hold a layer gains nothing from protecting it, so the floor is the same one the
    // fused path uses.
    // port (sm_89): a decode step no longer rebuilds the set from scratch.
    //
    // `tokens <= 8` used to force refresh, so every token took "this step's 10 routes + the
    // decayed-score top 118" as the new set. Those 118 come from a score that only looks at the
    // last few steps, so the previous step's routes fell out of the set and had to be installed
    // again the step after - measured as 22 of 30 routes newly installed per layer, 1.32 GiB of
    // upload per token, and 6.8 tok/s. Holding the set and adding the routes is the path the
    // !refresh branch below already took during prefill; decode takes it too now.
    const bool refresh = !set_initialised_ || !pool_covers_layer(narrowest);
    // A decode step still has to install the routes it just saw; it just does not discard the rest.
    const bool install = refresh || tokens <= kSplitSetUpdateTokens;

    // ---- P3: the layer's CPU share, decided here and acted on in exactly two places ------------
    //
    // The ratio is a result. One layer asks the cost model (cpu_path_policy.h decision 3) with the
    // three facts only this scope has - what a route costs over the link (three entry images), what
    // it costs the CPU half (the run geometry), and what this layer may spend on the CPU (its own
    // wall and the Device critical path) - and gets a COUNT back. Nothing in its path reads a
    // fraction from the environment.
    //
    // The experts it names are the coldest of this layer's routes, because the choice is not "may
    // this expert wait" but "which of the ten is least likely to be wanted again next step": the
    // decayed score is the same ranking the pool itself fills its remaining seats from, so a route
    // it prefers to keep is a route sent to the pool.
    //
    // The membership then does exactly two things, both applied once below: the route is left off
    // the upload list (no PCIe traffic for it), and publish_split_run keeps it out of the placement
    // word, so neither the pool nor the pinned tier offers the Device an address for it. The CPU
    // half reads the same entry images from the tier - which is why the tier must still hold them,
    // and why the fill below is unchanged: it is asked for the complement either way.
    //
    // With the switch off nothing here runs, `split_cpu_only_` stays empty, every branch that reads
    // it degenerates to "not a member", and the layer is bit for bit the pre-P3 layer.
    std::vector<std::int32_t> distinct; // this layer's routes, deduplicated and sorted
    if (host_ != nullptr || cpu_path_switch() == CpuPathSwitch::On) {
        distinct.assign(route_ids, route_ids + values);
        std::sort(distinct.begin(), distinct.end());
        distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
    }
    // One layer's wall, measured the way the [ctl] decode estimator measures it: the interval
    // between this call and this stream's previous one. Nothing else in this path stamps a whole
    // layer, and both queue-depth trims are ratios against exactly this interval.
    static thread_local std::chrono::steady_clock::time_point last_ensure_at{};
    double layer_wall_ms = 0.0;
    {
        const auto now = std::chrono::steady_clock::now();
        if (last_ensure_at.time_since_epoch().count() != 0) {
            layer_wall_ms =
                std::chrono::duration<double, std::milli>(now - last_ensure_at).count();
        }
        last_ensure_at = now;
    }
    int cpu_routes_this_layer = 0;
    for (auto& set : split_cpu_only_) { set.clear(); }
    if (cpu_path_switch() == CpuPathSwitch::On && host_ != nullptr && !distinct.empty()) {
        auto& control = policy_controller();
        if (!(control.link_gbps > 0.0)) {
            // One 32 MiB mapped copy, best of three: the link this decision divides by. Probed here
            // because the split form is its own call path and the controller's probe otherwise
            // lives in the fused one.
            control.probe_link(static_cast<std::uint64_t>(2U << 20));
        }
        CpuLayerQuery query;
        query.routes = static_cast<int>(distinct.size());
        for (const auto& run : split_) {
            query.bytes_per_expert += static_cast<double>(run.entry.entry_bytes);
        }
        // The half's arithmetic for one route: gate + up + down, two FLOPs per MAC (the same
        // convention the benchmark's 778 GFLOPS is quoted in), times the columns this layer runs.
        // The column factor is not optional: a prefill tile is 64-128 rows per route, so a cost that
        // ignored it would price a prefill route at a decode route's price and buy a 128x mistake.
        double macs = 0.0;
        for (const auto& run : split_) { macs += static_cast<double>(run.expert.elements); }
        query.flops_per_expert    = 2.0 * macs * static_cast<double>(tokens);
        // Decision 4's shape term: the columns this layer verifies. A decode layer passes 1; a verify
        // pass passes its T, which is what makes the read amortise and the arithmetic not.
        query.tokens              = tokens;
        query.link_gbps           = control.link_gbps;
        query.upload_us_per_entry = control.signals.upload_us_per_expert;
        query.cold_read_us        = control.signals.cold_read_us_per_expert;
        // The critical path this layer's CPU half has to fit under. On this form the copy IS the
        // critical path - the layer waits on the pool's fill - so the measured copy time is the
        // number the Device path needs anyway, and the CPU half may take as long as it does.
        query.gpu_layer_ms        = last_cost_.upload_ms;
        query.upload_ms           = last_cost_.upload_ms;
        query.cpu_ms              = cpu_path_ledger().last_layer_cpu_ms;
        query.wall_ms             = layer_wall_ms;
        query.lanes               = static_cast<int>(std::max<std::uint32_t>(1U, control.signals.lanes_busy));
        // How much of the tier the CPU half would find warm: the residency's own fill counters,
        // differenced per layer. Unknown on the first layer, which the model reads as "cold".
        {
            const auto& stats = host_->stats();
            static thread_local std::uint64_t seen_hits = 0, seen_misses = 0;
            const std::uint64_t hits   = stats.hits - seen_hits;
            const std::uint64_t misses = stats.misses - seen_misses;
            seen_hits   = stats.hits;
            seen_misses = stats.misses;
            if (hits + misses > 0) {
                query.tier_warm = static_cast<double>(hits) /
                                  static_cast<double>(hits + misses);
            }
        }
        const auto budget = decide_cpu_layer_budget(query);
        cpu_path_ledger().note_budget(budget.cost_gpu_us, budget.cost_cpu_us, budget.budget_us);
        cpu_path_ledger().note_layer_routes(static_cast<int>(distinct.size()));
        cpu_path_ledger().last_flops_per_expert = query.flops_per_expert;
        // Does the tier already hold this expert on every run the CPU half will read? A route it
        // holds costs a memcpy out of the arena; one it does not costs a physical read, measured at
        // 4.5 ms an entry on this box against the pool's 42 us when the pages are already warm. The
        // ratio is after one saving either way - the upload the Device is spared - so among routes
        // equally cold to the pool the resident ones are free and the others are not.
        const auto tier_holds = [this](std::int32_t expert) {
            if (split_.empty()) { return false; }
            const auto id = static_cast<std::uint32_t>(expert);
            for (const auto& run : split_) {
                if (!host_->holds(run.bank, id)) { return false; }
            }
            return true;
        };
        if (budget.count > 0) {
            std::vector<std::int32_t> picked = distinct;
            std::stable_sort(picked.begin(), picked.end(), [&](std::int32_t a, std::int32_t b) {
                const bool held_a = tier_holds(a);
                const bool held_b = tier_holds(b);
                if (held_a != held_b) { return held_a; }
                const auto sa = hot_score_[static_cast<std::size_t>(a)];
                const auto sb = hot_score_[static_cast<std::size_t>(b)];
                if (sa != sb) { return sa < sb; }
                return a < b;
            });
            picked.resize(std::min<std::size_t>(static_cast<std::size_t>(budget.count),
                                                picked.size()));
            for (auto& set : split_cpu_only_) { set.assign(count_, 0); }
            for (const auto expert : picked) {
                for (auto& set : split_cpu_only_) {
                    set[static_cast<std::size_t>(expert)] = 1;
                }
            }
            cpu_routes_this_layer = static_cast<int>(picked.size());
        }
        // Per layer, as the ratio is taken: what the model charged on each side, the cap it was cut
        // to, why, and the two rates the count was divided by (span = tier fill + arithmetic,
        // compute = the arithmetic alone). `held` is the residency the CPU half would have found:
        // the read the model prices at `cold_read_us` is only paid for the routes outside it, so the
        // ratio cannot reach its own steady state until this number is high - a ratio priced on a
        // tier that holds nothing is priced on a physical read per route.
        std::size_t held_candidates = 0;
        for (const auto expert : distinct) {
            if (tier_holds(expert)) { ++held_candidates; }
        }
        std::fprintf(stderr,
                     "[cpu-path] layer=%llu routes=%zu cpu=%d (%.0f%%) held=%zu/%zu | est "
                     "copy=%.0fus cpu=%.0fus "
                     "cap=%.0fus why=%s | link=%.1fGB/s span=%.1fus compute=%.1fus per exp "
                     "(%.0f GFLOPS)\n",
                     static_cast<unsigned long long>(control.layers + 1), distinct.size(),
                     cpu_routes_this_layer,
                     distinct.empty() ? 0.0
                                      : 100.0 * cpu_routes_this_layer /
                                            static_cast<double>(distinct.size()),
                     held_candidates, distinct.size(),
                     budget.cost_gpu_us, budget.cost_cpu_us, budget.budget_us, budget.why,
                     control.link_gbps,
                     cpu_half_rate().span_us_per_expert < 0.0
                         ? 0.0
                         : cpu_half_rate().span_us_per_expert,
                     cpu_half_rate().compute_us_per_expert < 0.0
                         ? 0.0
                         : cpu_half_rate().compute_us_per_expert,
                     cpu_half_rate().cpu_gflops);
        std::fflush(stderr);
    }
    split_cpu_routes_ = static_cast<std::uint32_t>(cpu_routes_this_layer);

    std::array<std::vector<std::uint32_t>, 3> wanted;
    std::vector<std::uint32_t> ranked; // built once: the ranking is per expert, not per run
    std::vector<bool> chosen(count_, false);
    std::uint32_t inside_gate = 0;
    for (std::size_t index = 0; index < split_.size(); ++index) {
        auto& set = wanted[index];
        set.reserve(capacity[index]);
        if (frozen_ && !split_frozen_[index].empty()) {
            set = split_frozen_[index];
            continue;
        }
        std::fill(chosen.begin(), chosen.end(), false);
        // Routes first, then what the pool already holds. The order is the difference between a
        // set that converges and one that thrashes: filling the held set first let 128 residents
        // take every seat, so this step's own routes did not fit and fell to the CPU half
        // (host_routes 7 of 10). Routes first keeps them addressable and still retains the rest.
        for (std::size_t i = 0; i < values && set.size() < capacity[index]; ++i) {
            const auto expert = static_cast<std::uint32_t>(route_ids[i]);
            if (chosen[expert]) { continue; }
            chosen[expert] = true;
            // P3: a route the CPU half owns keeps no pool seat. It is marked chosen so neither the
            // held set below nor the ranking hands it a seat back; the next-best expert takes the
            // seat instead, and nothing is staged for it (only the routes above are uploaded), so
            // the seat costs no traffic - it is a placement, not a copy.
            if (!split_cpu_only_[index].empty() && split_cpu_only_[index][expert] != 0) {
                continue;
            }
            set.push_back(expert);
        }
        if (ranked.empty() && refresh) {
            ranked.resize(count_);
            std::iota(ranked.begin(), ranked.end(), 0U);
            std::stable_sort(ranked.begin(), ranked.end(), [&](std::uint32_t a, std::uint32_t b) {
                return hot_score_[a] > hot_score_[b];
            });
        }
        if (!refresh) {
            // port (sm_89): last layer's set, in place. Rescanning the cache's 512 slots and
            // re-sorting the score every layer was what was left in the layout window after the
            // upload fell to 0.4 ms; a decode step only changes by the routes that arrived.
            for (const auto expert : split_prev_[index]) {
                if (set.size() >= capacity[index]) { break; }
                if (expert >= chosen.size() || chosen[expert]) { continue; }
                chosen[expert] = true;
                if (!split_cpu_only_[index].empty() &&
                    split_cpu_only_[index][static_cast<std::size_t>(expert)] != 0) {
                    continue; // P3: the CPU half's expert holds no seat, see the routes loop
                }
                set.push_back(expert);
            }
        }
        if (set.size() < capacity[index]) {
            if (ranked.empty()) {
                ranked.resize(count_);
                std::iota(ranked.begin(), ranked.end(), 0U);
                std::stable_sort(ranked.begin(), ranked.end(), [&](std::uint32_t a, std::uint32_t b) {
                    return hot_score_[a] > hot_score_[b];
                });
            }
            for (const auto expert : ranked) {
                if (set.size() >= capacity[index]) { break; }
                if (chosen[expert]) { continue; }
                chosen[expert] = true;
                if (!split_cpu_only_[index].empty() && split_cpu_only_[index][expert] != 0) {
                    continue; // P3: the CPU half's expert holds no seat, see the routes loop
                }
                set.push_back(expert);
            }
        }
        split_prev_[index] = wanted[index];
        if (index == 0) {
            for (std::size_t i = 0; i < values; ++i) {
                const auto expert = static_cast<std::size_t>(route_ids[i]);
                // P3: a route the CPU half owns is not "inside" anything the pool covers, so it is
                // not counted towards the coverage the panel reports.
                if (!split_cpu_only_[index].empty() && split_cpu_only_[index][expert] != 0) {
                    continue;
                }
                if (chosen[expert]) { ++inside_gate; }
            }
        }
    }

    // Q20 second leg, filled BEFORE the tables are built: a slot claimed after them still holds the
    // previous expert. Per run: each run's pool set is different, so its complement is too.
    if (host_ != nullptr && std::getenv("NINFER_SKIP_HOST_FILL") == nullptr) {
        static thread_local std::vector<std::uint8_t> in_wanted;
        if (in_wanted.size() != count_) { in_wanted.assign(count_, 0); }
        for (std::size_t index = 0; index < split_.size(); ++index) {
            auto& targets = split_host_targets_[index];
            targets.clear();
            std::fill(in_wanted.begin(), in_wanted.end(), std::uint8_t{0});
            for (const auto expert : wanted[index]) {
                if (expert < in_wanted.size()) { in_wanted[expert] = 1; }
            }
            for (const auto expert : distinct) {
                const auto id = static_cast<std::uint32_t>(expert);
                const bool pool_serves =
                    refresh ? id < in_wanted.size() && in_wanted[id] != 0
                            : cache_->expert_slot(split_[index].bank, id).has_value();
                if (!pool_serves) { targets.push_back(expert); }
            }
            // A run's complement can be wider than the tier's dynamic range: three runs price their
            // experts differently, so one quota cannot cover all three complements (that is why the
            // static stripe stays on the fused form). Asking the tier for more than it holds is
            // refused inside the fill, and a refused fill would abort the layer; the documented
            // answer is that the tier keeps what fits and the CPU half serves the rest, so the
            // target set is cut to the range, hottest first, and the rest stays a CPU route.
            if (const auto room = host_->routed_capacity(); targets.size() > room) {
                std::stable_sort(targets.begin(), targets.end(),
                                 [&](std::int32_t a, std::int32_t b) {
                                     const auto sa = hot_score_[static_cast<std::size_t>(a)];
                                     const auto sb = hot_score_[static_cast<std::size_t>(b)];
                                     if (sa != sb) { return sa > sb; }
                                     return a < b;
                                 });
                targets.resize(room);
                std::sort(targets.begin(), targets.end());
            }
            if (!targets.empty()) {
                host_->ensure(ExpertBankId{split_[index].bank},
                              std::span<const std::int32_t>(targets), nullptr);
            }
        }
        host_->begin_layer();
    } else {
        for (auto& targets : split_host_targets_) { targets.clear(); }
    }

    // Install each run's set, then publish all three.
    demoted_.clear();
    for (std::size_t index = 0; index < split_.size() && install; ++index) {
        (void)cache_->set_resident_experts(split_[index].bank,
                                           std::span<const std::uint32_t>(wanted[index]),
                                           static_cast<std::uint32_t>(wanted[index].size()));
    }
    std::uint32_t host_routes_gate = 0;
    for (std::size_t index = 0; index < split_.size(); ++index) {
        publish_split_run(index, stream, placement_mode);
        std::uint32_t host_hits = 0;
        for (std::size_t i = 0; i < values; ++i) {
            const auto expert = static_cast<std::size_t>(route_ids[i]);
            if (expert < split_[index].host_flags.size() &&
                split_[index].host_flags[expert] != 0) {
                ++host_hits;
            }
        }
        split_[index].host_routes = host_hits;
        if (index == 0) { host_routes_gate = host_hits; }
    }

    // One upload turn for all three runs: the routed keys of each run, deduplicated per bank.
    //
    // P3: a route the CPU half owns is left off this list, and that is the whole bandwidth saving -
    // the link no longer carries an entry image the CPU is about to compute over. The CPU reads the
    // same image from the pinned tier instead (one read, no DMA), which is what the compute side of
    // the cost model prices.
    required_.clear();
    required_.reserve(values * split_.size());
    for (std::size_t index = 0; index < split_.size(); ++index) {
        const auto& run = split_[index];
        for (std::size_t i = 0; i < values; ++i) {
            const auto expert = static_cast<std::uint32_t>(route_ids[i]);
            if (!split_cpu_only_[index].empty() && split_cpu_only_[index][expert] != 0) {
                continue;
            }
            required_.push_back(ExpertCacheKey{run.bank, expert});
        }
    }
    std::sort(required_.begin(), required_.end(), [](const ExpertCacheKey& a, const ExpertCacheKey& b) {
        return a.bank != b.bank ? a.bank < b.bank : a.expert < b.expert;
    });
    required_.erase(std::unique(required_.begin(), required_.end(),
                                [](const ExpertCacheKey& a, const ExpertCacheKey& b) {
                                    return a.bank == b.bank && a.expert == b.expert;
                                }),
                    required_.end());
    // The bytes this layer will actually put on the link: an entry only moves when its image is not
    // already ready. Counted in both arms - a counter that only runs when the policy is on cannot
    // say what the policy saved, which is the one comparison it exists for.
    std::uint64_t upload_bytes = 0;
    for (const auto& key : required_) {
        if (cache_->offset(key)) { continue; }
        for (const auto& run : split_) {
            if (run.bank != key.bank) { continue; }
            upload_bytes += run.entry.entry_bytes;
            break;
        }
    }

    // What this layer will actually put on the link: an entry only moves when its image is not
    // already ready. Counted before the copy, priced with each run's own entry image.
    const auto upload_begin = std::chrono::steady_clock::now();
    cache_->prepare(required_, stream);
    try {
        cache_->wait_for_progress(stream);
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string("pool: ") + error.what());
    }
    rebase_split();

    last_cost_.route_ms = route_ms;
    last_cost_.layout_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - layout_begin)
            .count();
    last_cost_.upload_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - upload_begin)
            .count();
    last_cost_.uploaded = cache_->last_uploaded();
    last_cost_.upload_bytes = upload_bytes;

    // P3: the split form is its own call path, so it feeds the controller nodes the cost model reads
    // (the fused path feeds them from streamed_ffn). Gated on the switch: with the policy down none
    // of this runs, no probe is taken, and the panel is exactly what it was.
    if (cpu_path_switch() == CpuPathSwitch::On) {
        auto& control = policy_controller();
        control.observe_upload(last_cost_.upload_ms * 1000.0, last_cost_.uploaded);
        control.observe_misses(last_cost_.uploaded);
        // The Device critical path is not measured separately on this path; passing the current
        // EWMA back into observe_layer leaves it untouched and still records the layer's wall.
        control.observe_layer(last_cost_.route_ms + last_cost_.layout_ms + last_cost_.upload_ms,
                              control.signals.gpu_ms, layer_wall_ms);
        control.observe_prefill(tokens > 8);
        const auto sources = cache_->upload_sources();
        if (sources.file_experts > 0) {
            const double cold = sources.read_us / static_cast<double>(sources.file_experts);
            control.signals.cold_read_us_per_expert +=
                (cold - control.signals.cold_read_us_per_expert) / 16.0;
        }
        if (const auto total = sources.arena_experts + sources.file_experts; total > 0) {
            const double ratio =
                static_cast<double>(sources.arena_experts) / static_cast<double>(total);
            control.signals.mirror_hit_ratio +=
                (ratio - control.signals.mirror_hit_ratio) / 16.0;
        }
    }

    // A3b per run: the tables the kernels are about to read must be the ones this run's set now is.
    for (const auto& run : split_) {
        if (run.table_step != cache_->step_generation(run.bank) ||
            run.table_hot != cache_->hot_generation(run.bank) ||
            run.table_prefetch != cache_->prefetch_generation(run.bank)) {
            throw std::runtime_error(
                "expert table is stale (bank " + std::to_string(run.bank) +
                " of the gate/up/down form): device holds hot/step/pred " +
                std::to_string(run.table_hot) + "/" + std::to_string(run.table_step) + "/" +
                std::to_string(run.table_prefetch) + ", cache is on " +
                std::to_string(cache_->hot_generation(run.bank)) + "/" +
                std::to_string(cache_->step_generation(run.bank)) + "/" +
                std::to_string(cache_->prefetch_generation(run.bank)));
        }
    }

    // Coverage: the router's own run is the one the ranking is about, so the reported number is the
    // gate run's. Summing three runs would count every route three times.
    coverage_.routes += values;
    coverage_.inside += inside_gate;
    if (last_cost_.uploaded == 0) { ++gh1_no_miss_calls_; }
    // "Every route has an address" is a per-run question here: a route the down run cannot hold has
    // no address for the down GEMM, so the layer is not capturable even if gate and up are.
    bool all_addressable = true;
    for (const auto& run : split_) {
        for (std::size_t i = 0; i < values && all_addressable; ++i) {
            const auto expert = static_cast<std::uint32_t>(route_ids[i]);
            if (expert >= run.mask.size() || run.mask[expert] == 0) { all_addressable = false; }
        }
    }
    const bool host_bound = last_cost_.uploaded != 0;
    if (!host_bound && all_addressable) {
        ++gh1_capt_free_calls_;
        ++gh1_capt_ready_calls_;
    } else {
        if (!all_addressable) { ++gh1_captfail_unaddressable_; }
        if (last_cost_.uploaded != 0) { ++gh1_captfail_upload_; }
    }
    last_host_routes_    = host_routes_gate;
    split_host_routes_   = host_routes_gate;
    set_initialised_     = true;
    gh1_report(stream);
    // S2-5: the tables this layer is about to read were just staged into the Device mirror, so this is
    // the one place where "the ids of this layer, resolved against the tables of the same generation"
    // is answerable without a second staging pass. No-op unless NINFER_SLOT_LOOKUP is set.
    lookup_routed_slots(device_ids, static_cast<std::int32_t>(values), stream,
                        ops::RouteTableKind::Placement);
    if (timing) {
        static std::uint64_t calls = 0;
        static double route_total = 0.0, layout_total = 0.0;
        route_total += route_ms;
        layout_total += last_cost_.layout_ms;
        if (++calls % 40 == 0) {
            std::fprintf(stderr,
                         "[split] gate/up/down per-layer route=%.3fms layout=%.3fms host_routes=%.2f "
                         "of %zu (of 40 layers)\n",
                         route_total / 40.0, layout_total / 40.0,
                         static_cast<double>(split_host_routes_) , values);
            route_total = layout_total = 0.0;
        }
    }
    if (tokens <= kSplitSetUpdateTokens) {
        prev_routes_.assign(route_ids, route_ids + values);
        std::sort(prev_routes_.begin(), prev_routes_.end());
        prev_routes_.erase(std::unique(prev_routes_.begin(), prev_routes_.end()), prev_routes_.end());
    }
    return true;
}

bool ExpertStream::ensure(const void* device_ids, std::int32_t top_k, std::int32_t tokens,
                          cudaStream_t stream, const std::int32_t* host_ids, bool assume_covered) {    if (cache_ == nullptr) { throw std::logic_error("expert stream is not attached"); }
    // The model's routing width, for the prefetch port: the fused path never enters ensure_split(),
    // so it has to record it here or the port would price the region at zero width.
    if (top_k > 0) { top_k_ = top_k; }
    // port (sm_89): the gate/up/down form is a different placement problem (three runs, three
    // tables, three masks), so it has its own body. The fused path below is not touched by it: a
    // fused stream never enters ensure_split() and the three-run stream never falls through here.
    //
    // `assume_covered` (O1's arm) is NOT honoured on the three-run path: it says "every route of
    // THIS bank is resident", and a layer here has three sets whose answers can differ, so the claim
    // would have to be made per run. The routes are therefore always taken there; the arm only ever
    // removed a read-back, so ignoring it costs time, never correctness.
    if (split_banks_) { return ensure_split(device_ids, top_k, tokens, stream, host_ids); }
    // port (sm_89): Q20's measurement. Three separate booleans rather than one "did the host do
    // anything": when capt_free stays at zero the fix depends on WHICH of the three fired - a route
    // read-back is removed by device-side coverage, a publication by a static table, an upload by a
    // second addressing leg. One boolean would say "not capturable" and name none of them.
    bool host_fetch = false, host_publish = false, host_upload = false;
    // Q20: the second addressing leg. Read once so the fill site and the tier-fill site cannot
    // disagree about whether the layer is meant to run without an upload. The freeze mode implies
    // both: its placement only exists if the tier is filled before the tables and the Device can
    // address it, so a run that sets only NINFER_HOST_FREEZE is not a half-configured streaming run.
    static const bool host_direct = std::getenv("NINFER_HOST_DIRECT") != nullptr ||
                                    std::getenv("NINFER_HOST_FREEZE") != nullptr;
    // Declared here as well as used at the fill site, because Q20's second leg needs the tier filled
    // BEFORE the Device tables are built - the tables name slots, and a slot filled afterwards names
    // the previous layer's expert.
    static const bool host_tier = std::getenv("NINFER_HOST_TIER") != nullptr ||
                                  std::getenv("NINFER_HOST_FREEZE") != nullptr;
    // P0/A: the epoch clock. Everything below the covered fast return can be skipped when the Device
    // pool already holds the layer, and that is exactly the state a high-coverage run is in - so the
    // refresh installed the static set once and never again (measured: asked=1 over 192 tokens,
    // static_present=0 afterwards). Count every call here, one per bank per token, so the epoch
    // arrives regardless of coverage. The probe-only arm measures how long a refreshed set survives;
    // the S/D split (NINFER_WARM_STATIC_RESERVE) keeps this same clock because a refresh that only
    // touches the static stripe cannot cost the layer its working set.
    static const bool static_reserve = std::getenv("NINFER_WARM_STATIC_RESERVE") != nullptr;
    if (static_warm_ && epoch_steps_ > 0 &&
        (static_reserve || std::getenv("NINFER_STATIC_PROBE") != nullptr)) {
        if (++static_refresh_clock_ >= epoch_steps_) {
            static_refresh_clock_ = 0;
            refresh_static_candidates();
        }
    }
    if (host_direct && bank_ == 0 && std::getenv("NINFER_HOST_PROBE") != nullptr) {
        static bool announced = false;
        if (!announced) {
            announced = true;
            std::fprintf(stderr, "[hostd] enabled bank=%u host=%p\n", bank_,
                         static_cast<const void*>(host_));
            std::fflush(stderr);
        }
    }
    // port (sm_89): measurement switch. Claiming the layer cannot be held makes the caller take the
    // host tier for the whole routed term, which is the CPU reference the device kernels are scored
    // against. Nothing else changes: same fetch, same FP32 fold, same shared expert.
    if (std::getenv("NINFER_MOE_HOST_ONLY")) { return false; }
    // port (sm_89): every layer is resident, so the window cannot move and the routers do not need
    // to be read back. Rebasing is the whole per-layer work, and it happens during capture so the
    // frozen kernel arguments already point at the right window.
    if (pinned_) {
        if (cache_->window_resident(bank_)) {
            rebase_window();
            // A pinned layer pays none of the bookkeeping below, so its cost is zero everywhere.
            last_cost_ = StepCost{};
            // GH1 on the fully resident path. This early return skips every counter below, which is
            // why the panel reported nothing at all once the pool held all 10,240 slots: the healthy
            // configuration was the one it could not describe. A pinned layer is by definition
            // capturable, its table does not move and it cannot miss.
            ++gh1_calls_;
            ++gh1_no_miss_calls_;
            ++gh1_static_calls_;
            // A pinned layer re-points host-side pointers and nothing else: no read-back, no table,
            // no copy. It is the one path where both rates are honest 100%.
            ++gh1_capt_free_calls_;
            ++gh1_capt_ready_calls_;
            gh1_report(stream);
            return true;
        }
        // A resize dropped this window; fall through so the normal path allocates a new one.
    }
    // One window holds a whole layer when the pool is large enough; otherwise the window shrinks to
    // the pool and the caller splits the layer between the Device pool and the host tier.
    // The window belongs to this bank alone, so it is sized from this bank's seats.
    const auto capacity = cache_->resident_capacity(bank_);
    // With only a couple of slots per bank the split path costs more than it saves: it still pays
    // the routing read-back, the per-expert mask and the merge, while almost every expert lands on
    // the host anyway. Below this floor the whole layer goes to the host tier instead - measured
    // 14.4 tok/s on the split path against a host-only layer at the same pool size.
    if (capacity < 4) { return false; }
    const bool fully = capacity >= count_;
    // P1: the per-layer freeze clock. It advances on every call, not on one policy branch: the old
    // `steps_` lived inside the streaming branch, so a layer whose routes were all resident returned
    // before it ever reached the freeze and the set was installed on the first token, from an empty
    // ranking, and never corrected.
    if (freeze_mode_) {
        ++freeze_clock_;
        if (frozen_ && freeze_refresh_ != 0 && freeze_clock_ >= freeze_refresh_) {
            // Re-select. The set was chosen from the traffic up to the freeze boundary; once the
            // request has moved on (prefill into decode) that set is a profile of the wrong phase.
            // Following the router again for `freeze_after_` calls and then re-freezing replaces it
            // with one measured on the current traffic. Capture is off by definition here: the
            // table moves during the re-warm.
            frozen_        = false;
            static_frozen_ = false;
            if (host_ != nullptr) { host_->set_static_frozen(bank_, false); }
            freeze_clock_ = 0;
            if (bank_ == 0) {
                std::fprintf(stderr, "[freeze] bank=0 re-select after %u calls\n", freeze_refresh_);
                std::fflush(stderr);
            }
        } else if (!frozen_ && freeze_after_ != 0 && freeze_clock_ >= freeze_after_) {
            freeze_now(tokens, stream);
        }
    }
    // Q10 probe (see static_probe): why this layer was not static. Filled in the split branch, printed
    // at its end where the upload count is known. Zero cost when the switch is off.
    std::uint32_t probe_absent = 0, probe_unready = 0;
    std::uint64_t probe_moved = 0, probe_ins = 0;
    // GH1: the three coverage numbers the plan gates on, all with ONE denominator - the number of
    // ensure() calls, i.e. layers. Counted here rather than only inside the split branch, because on
    // a fully resident build the split branch never runs and the panel would report 0% for
    // everything, which is the opposite of the truth.
    //
    //   static     : the R1 table did not move for this layer
    //   no_miss    : every route this layer published was already resident
    //   capturable : the whole expert set fitted the pool, so a router -> GEMM capture needs no
    //                host intervention (this is GH2/G2's precondition, not a promise that a graph
    //                exists yet)
    ++gh1_calls_;
    if (!fully) {
        ++gh1_split_calls_;
    } else {
        // A resident layer cannot miss and its table cannot move.
        ++gh1_no_miss_calls_;
        ++gh1_static_calls_;
    }
    gh1_report(stream);
    // Q11/Q12: the window the freeze rule reads. Counted here so a layer that returns through the
    // covered fast path is still part of the window - that is exactly the layer the rule is looking
    // for, and counting only the split path would make "32 clean layers" unreachable by construction.
    const bool freeze_watch = !freeze_mode_ && freeze_rule_enabled();
    if (freeze_watch) { ++freeze_window_layers_; }
    // port (sm_89): O1's arm (see the declaration). Taken before anything needs the ids, because
    // needing the ids is exactly what it removes: the branch below copies 32 bytes back and waits on
    // the stream for them, and that wait - not the copy - is 0.06 ms of every layer.
    //
    // What it also skips is this layer's placement, and at 12 GiB that is the part that matters: the
    // pool holds 47 of a bank's 256 experts and its set moves every step, so the mask the step runs
    // under stops covering the routes within a layer or two. Measured: every armed step came back
    // dirty (3-44 routes), every token was replayed, and decode went from 6.4/7.5 to 12.0/11.6 ms a
    // token. Nothing here is silent about it - the step gate counts the routes this skips - but the
    // numbers say the claim is false, not that the detection is weak.
    // Note: the ~0.5 ms fetch is NOT in this function - ffn reads the ids to pack the split and
    // decide all_resident. This branch only removes ensure's own work; it is taken when the caller
    // already knows coverage (O1's device verdict) or when the frozen table covers the whole bank.
    if ((assume_covered ||
         (std::getenv("NINFER_STATIC_FAST_OFF") == nullptr &&
          std::getenv("NINFER_STATIC_FAST_PARTIAL") == nullptr && static_covered())) &&
        !fully) {
        ++gh1_static_calls_;
        ++gh1_no_miss_calls_;
        // Same reason as the pinned path: coverage was established on the Device and this call only
        // rebases host pointers. Counting it is what makes capt_free a rate rather than a restatement
        // of the pool size.
        ++gh1_capt_free_calls_;
        ++gh1_capt_ready_calls_;
        last_cost_ = StepCost{};
        // Re-assert the frozen pool set. It is idempotent when nothing moved, and it is the one part
        // of the skipped path that can differ: an expert that could not get a pool slot at the
        // freeze is not in S either, so it would be uncovered for every later layer.
        if (cache_ != nullptr && !frozen_wanted_.empty()) {
            (void)cache_->set_resident_experts(bank_, frozen_wanted_,
                                               static_cast<std::uint32_t>(frozen_wanted_.size()));
        }
        // Settle the stream the way the full path does. The copy pool publishes and settles in one
        // place; a layer that returns without it can leave the uploader's staging ownership open.
        if (cache_ != nullptr) { cache_->wait_for_progress(stream); }
        // The pool upload runs on the cache's own stream. A stream-local sync cannot order it
        // against this layer's kernel; a device-wide one can. The read-back the full path pays for
        // ends in cudaStreamSynchronize, so nobody ever noticed which of the two it was covering.
        if (std::getenv("NINFER_STATIC_FAST_DEV_SYNC") != nullptr) { (void)cudaDeviceSynchronize(); }
        // Timing probe: the full path synchronises the stream because it reads the routes back. The
        // fast path removes that barrier; if anything wrote the tier asynchronously just before this
        // layer, the kernel could read it before it lands. One sync tells the difference.
        if (std::getenv("NINFER_STATIC_FAST_SYNC") != nullptr) {
            (void)cudaStreamSynchronize(stream);
        }
        rebase_window();
        // The layer skipped its placement pass, so the holds that pass would rotate must still
        // happen once per layer - otherwise a later re-freeze (which refuses to reset held slots)
        // could never proceed.
        if (host_ != nullptr) { host_->begin_layer(); }
        probe_tables("fast");
        if (std::getenv("NINFER_READY_PROBE") != nullptr && bank_ == 0) {
            static std::uint32_t shown = 0;
            if (shown++ < 48) {
                const auto unready = cache_ == nullptr ? 0U : cache_->unready_count(bank_, frozen_wanted_);
                std::fprintf(stderr, "[ready] seq=%u where=fast mapped=%zu unready=%u\n",
                             shown - 1, frozen_wanted_.size(), unready);
                std::fflush(stderr);
            }
        }
        probe_device_tables("fast", device_ids,
                            static_cast<std::size_t>(top_k) * static_cast<std::size_t>(tokens));
        return true;
    }
    const auto values = static_cast<std::size_t>(top_k) * static_cast<std::size_t>(tokens);
    const auto bytes  = values * sizeof(std::int32_t);
    // port (sm_89): the host must know the routed experts before it can publish them, so one
    // pinned staging buffer carries the route IDs back and the caller then waits on the stream.
    // port (sm_89): the caller may already have the routes on the host - the split path reads back
    // the layer's inputs in one batch, and the 32 route ids are part of it. Taking them from there
    // drops one full stream synchronise and one duplicate copy per layer, which was 40 x ~0.3 ms of
    // the split token.
    if (host_ids == nullptr && (!route_staging_ || route_staging_->size() < bytes)) {
        route_staging_ = std::make_unique<PinnedHostBuffer>(bytes);
    }
    // port (sm_89): NINFER_SPLIT_TIMING separates the per-layer cost into the routing read-back,
    // the layout update (resident set + mask/offset tables) and the upload, because the split
    // path's fixed cost is what decides whether holding experts at all pays.
    const bool timing = std::getenv("NINFER_SPLIT_TIMING") != nullptr;
    double route_ms   = 0.0;
    const std::int32_t* route_ids = host_ids;
    if (route_ids == nullptr) {
        const auto stage_begin = std::chrono::steady_clock::now();
        const auto status     = cudaMemcpyAsync(route_staging_->data(), device_ids, bytes,
                                                cudaMemcpyDeviceToHost, stream);
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("route copy: ") + cudaGetErrorString(status));
        }
        const auto sync = cudaStreamSynchronize(stream);
        if (sync != cudaSuccess) {
            throw std::runtime_error(std::string("route sync: ") + cudaGetErrorString(sync));
        }
        route_ms  = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                             stage_begin).count();
        route_ids = static_cast<const std::int32_t*>(route_staging_->data());
    }
    const auto layout_begin = std::chrono::steady_clock::now();

    // Port (sm_89): every routed expert must be resident before the consumer runs, so the
    // required set is deduplicated and published in one batch.
    for (std::size_t i = 0; i < values; ++i) {
        const auto expert = route_ids[i];
        if (expert < 0 || static_cast<std::uint32_t>(expert) >= count_) {
            throw std::out_of_range("expert stream received a route outside its bank");
        }
    }
    // Q30/P6: how much of this layer the prediction region actually served. A placement is work; this
    // is the only number that says whether the region is worth its slots. The denominator is the same
    // population as the numerator (S3-4: it used to be the warm tier's ask count, so the printed rate
    // was 0/0 on every run without NINFER_PREFETCH).
    if (cache_ != nullptr && cache_->prefetch_slots_per_bank() != 0) {
        for (std::size_t i = 0; i < values; ++i) {
            ++prefetch_route_samples_;
            const auto slot = cache_->expert_slot(bank_, static_cast<std::uint32_t>(route_ids[i]));
            if (slot && cache_->prefetch_slot(*slot)) { ++prefetch_hits_; }
        }
    }
    // port (sm_89): a pool that cannot hold a whole layer keeps a hot set per bank. This step's
    // routes are placed first - they must run on the Device - and the remaining slots go to the
    // experts with the highest decayed score, so the next step has a chance to hit. The cache only
    // uploads the experts that actually changed, and experts that stay keep their slot.
    if (!fully) {
        // (the split branch itself is counted at the top of ensure)
        if (tokens <= kHotSetUpdateTokens) {
            for (std::size_t i = 0; i < values; ++i) {
                hot_score_[static_cast<std::size_t>(route_ids[i])] += 1.0F;
            }
            const auto decay = hot_decay_period();
            if (decay != 0 && ++since_decay_ >= decay) {
                for (auto& score : hot_score_) { score *= 0.5F; }
                since_decay_ = 0;
            }
        }
        // port (sm_89): a step whose routes are all already in the pool has nothing to publish.
        //
        // The set is a means to an end, and the policies below re-rank the fill every step: at 16 GiB
        // that moves 1.77 experts a layer even though 91% of layers miss nothing, which costs a table
        // publish per layer (static 0%) and is what keeps a whole-layer graph uncapturable. Asking the
        // cache whether each route is resident *and ready* is the honest test - the mask cannot answer
        // it, because the mask is written from the placement that this skip avoids making.
        bool covered = true;
        // port (sm_89): Q10 asks why a third of the layers move the table when the mask says every
        // route is resident. The answer has to separate the two ways `offset()` says no - the expert
        // has no slot at all, or it has one whose image has not landed yet - because they call for
        // different fixes (placement policy vs upload pacing). Counted here, printed by [st].
        std::uint32_t absent = 0, unready = 0;
        for (std::size_t i = 0; i < values; ++i) {
            const auto expert = static_cast<std::size_t>(route_ids[i]);
            if (expert >= keys_.size() || !cache_->expert_slot(bank_, expert)) {
                ++absent;
                covered = false;
                continue;
            }
            if (!cache_->offset(keys_[expert])) {
                ++unready;
                covered = false;
            }
        }
        probe_absent  = absent;
        probe_unready = unready;
        if (last_covered_) { ++predict_trials_; if (covered) { ++predict_hits_; } }
        last_covered_ = covered;
        if (covered) {
            ++gh1_static_calls_;
            ++gh1_no_miss_calls_;
            last_cost_          = StepCost{};
            last_cost_.route_ms = route_ms;
            rebase_window();
            // Nothing is published or uploaded on a covered layer - this return is above both. The
            // only thing that could make it host-bound is a read-back, and a covered layer does not
            // need one to compute: every route already has a pool address. Under Q20's definition
            // that is a host-free layer, which is what the rate is supposed to count.
            ++gh1_capt_free_calls_;
            ++gh1_capt_ready_calls_;
            if (freeze_watch) { ++freeze_window_clean_; }
            last_host_routes_ = 0;
            static_probe(bank_, tokens, true, probe_absent, probe_unready, 0, 0, 0, true);
            return true;
        }
        // port (sm_89): two policies, because they disagree on what the set is for.
        //
        // Streaming (default): this step's routes are placed first, so the Device computes them
        // and the host only sees the experts that did not fit. The set follows the router, which
        // keeps hits high while the pool can hold a useful fraction of a layer.
        //
        // Static (NINFER_EXPERT_STATIC=1): the set is the top-N of the decayed route score alone,
        // which is what Strata's profile does (h = 0.6447 over 4,105 of 24,576 slots against 0.4864
        // for compulsory fill). It never chases a single step, so the layout changes rarely, but it
        // only pays when the pool holds enough slots for the popular experts to cover most routes.
        // port (sm_89): the set is rebuilt once per generation and held in between. 0 restores the
        // old per-step behaviour.
        static const std::int32_t generation_steps = [] {
            const char* value = std::getenv("NINFER_EXPERT_GEN_STEPS");
            // Measured worse at both 8 and 12 GiB: rebuilding from a 64-step window churns half the
            // set per generation and lands at 49% coverage against 78.5% for a converged profile, so
            // the per-step streaming policy stays the default until the histogram is converged by
            // other means. Set it to a step count to experiment.
            return value != nullptr && *value != '\0' ? std::atoi(value) : 0;
        }();
        const bool generation_mode = generation_steps > 0;
        static const bool static_set = std::getenv("NINFER_EXPERT_STATIC") != nullptr;
        if (generation_mode) {
            if (steps_in_gen_ >= static_cast<std::uint32_t>(generation_steps)) {
                if (generation_ > 0) {
                    std::fprintf(stderr, "[gen] n=%u set=%zu churn=%zu\n", generation_,
                                 previous_wanted_.size(),
                                 static_cast<std::size_t>(std::count_if(
                                     frozen_wanted_.begin(), frozen_wanted_.end(),
                                     [&](std::uint32_t expert) {
                                         return std::find(previous_wanted_.begin(),
                                                          previous_wanted_.end(),
                                                          expert) == previous_wanted_.end();
                                     })));
                }
                previous_wanted_ = frozen_wanted_;
                frozen_          = false;
                steps_in_gen_    = 0;
                ++generation_;
            }
            ++steps_in_gen_;
        }
        // port (sm_89): a static set only pays once it stops moving. Ranking by decayed score on
        // every step keeps reshuffling the tail while the scores are still being learned, so the
        // set is frozen after a warm-up window: the first N tokens decide it and the rest of the
        // request keeps it, sending a miss to the host tier instead of triggering an upload.
        // This is the fused form's half of freeze_after_for_form(): the split form's default is a
        // window, the fused form's is none, and NINFER_EXPERT_FREEZE=0 keeps the always-rerank
        // behaviour here.
        static const int freeze_after =
            static_cast<int>(freeze_after_for_form(/*split=*/false));
        std::vector<std::uint32_t> wanted;
        std::vector<bool> chosen(count_, false);
        wanted.reserve(capacity);
        if (!frozen_) {
            // A profile names the set before the first token, so a run can be repeatable: the same
            // pool gives the same experts in the same slots instead of whatever the first prompt
            // happened to route to.
            auto profiled = profile_top_experts(bank_, count_, capacity);
            if (!profiled.empty()) {
                frozen_wanted_ = std::move(profiled);
                frozen_        = true;
            }
        }
        // The pool is two regions, and they never trade places. The hot region keeps the experts
        // worth holding across steps; the spill region is a small ring that only ever holds the
        // experts the current step misses, so a hot decision can never push a spill entry out and a
        // miss can never shrink the hot set. Measured: eight spill slots turn four concurrent lanes
        // from 32-35 tok/s into 82-86 at twelve gigabytes (plan section forty-one).
        // The ring has to hold one step, not one token: eight lanes miss eight times as many
        // experts at once, and a ring sized for a single lane sends the rest to the host tier -
        // measured 5.2 tok/s at eight lanes against 50 for the same structure with the lanes
        // counted. It is sized from the widest step seen so far, so it does not resize per step
        // (a region that changes size republishes the table, see section forty-one).
        // port (sm_89): only a decode-sized step is a lane count. A 512-token prefill is one lane
        // with 512 rows; counting it here latched lanes_seen_ at 512 for the rest of the run, which
        // put the pool on the pure-SLRU branch (`slru_active`) and let the prefill's 512 experts
        // evict the decode hot set - the 1440 experts a token needs all live in a 18528-seat pool,
        // yet decode measured 3.6 tok/s. The widest step still sizes the spill ring, so it keeps
        // being tracked, but only from steps that can actually be concurrent lanes.
        constexpr std::int32_t kLaneSampleTokens = 8;
        if (tokens > 0 && tokens <= kLaneSampleTokens) {
            lanes_seen_ = std::max<std::uint32_t>(lanes_seen_, static_cast<std::uint32_t>(tokens));
        }
        // Width-based policy: from six lanes up the whole-pool 50/50 SLRU measured 52 tok/s at
        // twelve gigabytes against 36 for the route-first default, and below that the default wins
        // (105/86 at one/four lanes). The switch is therefore made on the widest step seen.
        const bool slru_active = slru_policy_ || (auto_slru_ && lanes_seen_ >= 6);
        if (bank_ == 0 && std::getenv("NINFER_LANES_DEBUG") != nullptr) {
            static int shown = 0;
            if (shown++ < 6) { std::fprintf(stderr, "[lanes] tokens=%d seen=%u auto=%d slru=%d\n", tokens, lanes_seen_, (int)auto_slru_, (int)slru_active); }
        }
        // A pure SLRU run owns the whole pool: the ring, the seeding and the promotion below are
        // for the region split, and mixing them into the same path is what left it half converted.
        const bool pure_slru = slru_active && !regions_ && epoch_steps_ == 0;
        const auto spill_cap = [&]() -> std::size_t {
            // P1: a frozen placement has no ring. A ring member is a slot that moves every step,
            // which is exactly what a table that a captured layer replays cannot contain.
            if (freeze_mode_ && frozen_) { return 0; }
            if (slru_active) { return 0; }
            if (!regions_) {
                // default policy: the eight fixed slots the report measured
                if (const char* value = std::getenv("NINFER_EXPERT_SPILL_PCT")) {
                    return static_cast<std::size_t>(capacity) *
                           std::min<std::uint64_t>(std::strtoul(value, nullptr, 10), 90) / 100;
                }
                if (const char* value = std::getenv("NINFER_EXPERT_SPILL_SLOTS")) {
                    return std::min<std::size_t>(std::strtoul(value, nullptr, 10),
                                                 static_cast<std::size_t>(capacity) / 2);
                }
                return std::min<std::size_t>(spill_slots(capacity),
                                             static_cast<std::size_t>(capacity) / 2);
            }
            if (const char* value = std::getenv("NINFER_EXPERT_SPILL_PCT")) {
                return static_cast<std::size_t>(capacity) *
                       std::min<std::uint64_t>(std::strtoul(value, nullptr, 10), 90) / 100;
            }
            if (const char* value = std::getenv("NINFER_EXPERT_SPILL_SLOTS")) {
                return std::min<std::size_t>(std::strtoul(value, nullptr, 10),
                                             static_cast<std::size_t>(capacity) / 2);
            }
            const auto wanted_slots = std::min<std::size_t>(
                static_cast<std::size_t>(8) * std::max<std::uint32_t>(1U, lanes_seen_),
                static_cast<std::size_t>(capacity) / 3);
            return std::min<std::size_t>(std::max<std::size_t>(spill_slots(capacity), wanted_slots),
                                         static_cast<std::size_t>(capacity) / 3);
        }();
        const auto hot_cap = static_cast<std::size_t>(capacity) - spill_cap;
        // Kept for the attribution pass at the tail of ensure(), which runs outside this scope and has
        // to know where the hot set ends inside `resident_order_`.
        hot_cap_ = hot_cap;

        if (frozen_) {
            // P1: the placement is the set. No route is appended and no slot is borrowed for this
            // step's demand: the Device table, the host static table and the offset tables are all
            // functions of `frozen_wanted_` alone, which is what lets a layer be captured.
            //
            // The old branch held 15% of the pool back for outside routes and appended them here,
            // because a full set sent every outsider to the CPU at 0.5 ms. That trade is now made on
            // the host side: the complement lives in the pinned tier, and a route outside the Device
            // set is a host hit rather than a CPU miss. With a host tier smaller than the complement
            // the old behaviour is still available by not freezing.
            const auto keep = std::min<std::size_t>(capacity, frozen_wanted_.size());
            wanted.assign(frozen_wanted_.begin(), frozen_wanted_.begin() + keep);
            for (const auto expert : wanted) { chosen[expert] = true; }
            hot_cap_ = keep;
        } else if (slru_active) {
            // llama.cpp's 50/50 SLRU (src/llama-moe-cache.cpp:16-121), over slot ids so a slot that
            // keeps its expert keeps its number: hits are promoted to the protected half, misses
            // take the probation head while skipping the experts this step needs, and a full set
            // that holds them all sends the rest to the host tier. The scoring fill below is not
            // consulted - that is the point of the comparison.
            constexpr std::uint32_t kEmpty = 0xFFFFFFFFU;
            // SLRU owns the hot region only: the spill ring below is not its to evict.
            const auto n_slots = pure_slru ? static_cast<std::size_t>(capacity) : hot_cap;
            if (slru_expert_.size() != n_slots) {
                slru_expert_.assign(n_slots, kEmpty);
                slru_of_expert_.assign(count_, -1);
                slru_prev_.assign(n_slots, -1);
                slru_next_.assign(n_slots, -1);
                slru_protected_.assign(n_slots, 0);
                slru_head_[0] = slru_head_[1] = 0;
                slru_tail_[0] = static_cast<std::int32_t>(n_slots) - 1;
                for (std::size_t s = 0; s < n_slots; ++s) {
                    slru_prev_[s] = static_cast<std::int32_t>(s) - 1;
                    slru_next_[s] = (s + 1 < n_slots) ? static_cast<std::int32_t>(s + 1) : -1;
                }
                slru_head_[1] = slru_tail_[1] = -1;
                slru_n_protected_ = 0;
            }
            const auto unlink = [&](std::int32_t s) {
                const auto seg = slru_protected_[s] ? 1 : 0;
                if (slru_prev_[s] >= 0) { slru_next_[slru_prev_[s]] = slru_next_[s]; }
                else { slru_head_[seg] = slru_next_[s]; }
                if (slru_next_[s] >= 0) { slru_prev_[slru_next_[s]] = slru_prev_[s]; }
                else { slru_tail_[seg] = slru_prev_[s]; }
            };
            const auto push_back = [&](std::int32_t s, bool protect) {
                const auto seg = protect ? 1 : 0;
                slru_protected_[s] = protect ? 1 : 0;
                slru_prev_[s] = slru_tail_[seg];
                slru_next_[s] = -1;
                if (slru_tail_[seg] >= 0) { slru_next_[slru_tail_[seg]] = s; }
                else { slru_head_[seg] = s; }
                slru_tail_[seg] = s;
            };
            const auto touch = [&](std::int32_t s) {
                const bool was = slru_protected_[s] != 0;
                unlink(s);
                push_back(s, true);
                if (!was) { ++slru_n_protected_; }
                if (slru_n_protected_ > n_slots / 2) {
                    const auto demote = slru_head_[1];
                    if (demote >= 0) {
                        unlink(demote);
                        push_back(demote, false);
                        --slru_n_protected_;
                    }
                }
            };
            // this step's distinct experts, in route order
            std::vector<std::uint32_t> need;
            need.reserve(values);
            std::vector<bool> in_need(count_, false);
            for (std::size_t i = 0; i < values; ++i) {
                const auto expert = static_cast<std::uint32_t>(route_ids[i]);
                if (in_need[expert]) { continue; }
                in_need[expert] = true;
                need.push_back(expert);
            }
            for (const auto expert : need) {
                const auto slot = slru_of_expert_[expert];
                if (slot >= 0) { touch(slot); }
            }
            // Demotion order: the hot entry with the lowest decayed score goes first, not the one
            // that was touched longest ago. A score that decays is what makes "hot to warm" a
            // decision about the workload rather than about the clock, and it is the same number
            // the RAM tier's journal uses to decide what to keep.
            const auto coldest = [&](bool protect_need) -> std::int32_t {
                std::int32_t victim = -1;
                float best = 0.0F;
                for (std::size_t s = 0; s < n_slots; ++s) {
                    const auto held = slru_expert_[s];
                    if (held != kEmpty && protect_need && in_need[held]) { continue; }
                    const auto score = held == kEmpty ? -1.0F : hot_score_[held];
                    if (victim < 0 || score < best) { victim = static_cast<std::int32_t>(s); best = score; }
                }
                return victim;
            };
            const auto install = [&](std::int32_t slot, std::uint32_t expert, bool protect) {
                const auto held = slru_expert_[slot];
                if (held != kEmpty) {
                    slru_of_expert_[held] = -1;
                    warm_journal_.push_back(held);
                    if (warm_journal_.size() > 512) { warm_journal_.erase(warm_journal_.begin()); }
                }
                const bool was = slru_protected_[slot] != 0;
                unlink(slot);
                if (was) { --slru_n_protected_; }
                slru_expert_[slot] = expert;
                slru_of_expert_[expert] = slot;
                push_back(slot, protect);
            };
            // Demand is an EWMA of "this expert was needed", so an expert that is needed in three
            // steps out of ten reads the same as one needed twice in a row - which is the point:
            // what matters is the probability it is needed again, not the streak.
            if (demand_.size() != count_) { demand_.assign(count_, 0.0F); }
            for (auto& value : demand_) { value *= 0.9F; }
            for (const auto expert : need) { demand_[expert] += 1.0F; }
            if (!pure_slru && !hot_seeded_) {
                // Seed from the first step's routes: the hot region then starts where the routing
                // is, instead of where an un-converged score says it should be.
                for (const auto expert : need) {
                    if (slru_filled_ >= hot_cap) { break; }
                    // an empty slot first: every slot starts at score zero, so picking by score
                    // here would keep overwriting the expert just installed
                    std::int32_t slot = -1;
                    for (std::size_t s = 0; s < n_slots; ++s) {
                        if (slru_expert_[s] == kEmpty) { slot = static_cast<std::int32_t>(s); break; }
                    }
                    if (slot < 0) { slot = coldest(false); }
                    if (slot < 0) { break; }
                    install(slot, expert, true);
                    ++slru_filled_;
                }
                hot_seeded_ = slru_filled_ >= hot_cap;
                hot_seeded_ = hot_seeded_ || slru_filled_ >= hot_cap;
            }
            // Promotion: expected demand above the cheap threshold, and above the coldest hot entry
            // by a margin - a swap that does not beat what it displaces is churn.
            //
            // port (sm_89): the two install sites below are counted apart, because "upload costs
            // 1.5-2.5 ms per token while nothing misses" only says something once you know whether
            // the copies come from this preparatory admission or from a route that was genuinely
            // absent. The answer decides whether O1's next lever is the ranking or the transfer.
            std::uint64_t promoted = 0, absent = 0;
            // port (sm_89): the pool already serving every expert this step needs is the signal that
            // the ranking has nothing left to buy. measured at 12 GiB, 512 tokens: promoted 0.88-1.05
            // experts per layer against absent 0.40-0.53, and 1.3-1.5 experts a layer is what the
            // 1.5-2.5 ms of upload per token is. NINFER_LAZY_PROMOTE stops the preparatory half when
            // the step is fully served; the fallback loop below still installs anything genuinely
            // absent, so a real miss can never be swallowed.
            static const bool lazy_promote = std::getenv("NINFER_LAZY_PROMOTE") != nullptr;
            bool all_served                 = true;
            if (lazy_promote) {
                for (const auto expert : need) {
                    if (slru_of_expert_[expert] < 0) {
                        all_served = false;
                        break;
                    }
                }
            }
            for (const auto expert : need) {
                if (all_served) { break; }
                if (slru_of_expert_[expert] >= 0) { continue; }
                const auto slot = coldest(true);
                if (slot < 0) { break; }
                const auto held = slru_expert_[slot];
                const auto incoming = demand_[expert];
                const auto displaced = held == kEmpty ? 0.0F : demand_[held];
                if (incoming < 0.25F || incoming < displaced + 0.1F) { continue; }
                install(slot, expert, false);
                ++promoted;
            }
            for (const auto expert : need) {
                if (slru_of_expert_[expert] >= 0) { continue; }
                const auto slot = coldest(true);
                if (slot < 0) { break; }
                install(slot, expert, false);
                ++absent;
            }
            if (std::getenv("NINFER_UPLOAD_TIMING") != nullptr) {
                static std::uint64_t layers = 0, promoted_total = 0, absent_total = 0;
                promoted_total += promoted;
                absent_total += absent;
                if (++layers % 40 == 0) {
                    std::fprintf(stderr, "[install] per layer: promoted=%.2f absent=%.2f\n",
                                 static_cast<double>(promoted_total) / 40.0,
                                 static_cast<double>(absent_total) / 40.0);
                    std::fflush(stderr);
                    promoted_total = 0;
                    absent_total = 0;
                }
            }
            wanted.clear();
            wanted.reserve(capacity);
            for (std::size_t s = 0; s < n_slots; ++s) {
                if (slru_expert_[s] == kEmpty) { break; }   // dense: never a hole
                wanted.push_back(slru_expert_[s]);
                chosen[slru_expert_[s]] = true;
            }
            // the spill ring: this step's misses, appended after the hot region so they take the
            // slots the hot set does not own and cannot displace it
            for (std::size_t i = 0; i < values && wanted.size() < capacity; ++i) {
                const auto expert = static_cast<std::uint32_t>(route_ids[i]);
                if (chosen[expert]) { continue; }
                chosen[expert] = true;
                wanted.push_back(expert);
            }

        } else if (!regions_) {
            // Until the region split is measured against it, the proven streaming policy stays the
            // default: this step's routes take slots first, the decayed score fills the rest. That
            // is what every number in the report was taken with.
            for (std::size_t i = 0; i < values && wanted.size() < capacity; ++i) {
                const auto expert = static_cast<std::uint32_t>(route_ids[i]);
                if (chosen[expert]) { continue; }
                chosen[expert] = true;
                wanted.push_back(expert);
            }
            if (wanted.size() < capacity) {
                std::vector<std::uint32_t> ranked(count_);
                std::iota(ranked.begin(), ranked.end(), 0U);
                std::stable_sort(ranked.begin(), ranked.end(),
                                 [&](std::uint32_t a, std::uint32_t b) {
                                     return hot_score_[a] > hot_score_[b];
                                 });
                for (const auto expert : ranked) {
                    if (wanted.size() >= capacity) { break; }
                    if (chosen[expert]) { continue; }
                    chosen[expert] = true;
                    wanted.push_back(expert);
                }
            }
            if (generation_mode) { frozen_ = true; frozen_wanted_ = wanted; }
            if (static_set && freeze_after > 0 && ++steps_ >= static_cast<std::uint64_t>(freeze_after)) {
                frozen_ = true; frozen_wanted_ = wanted;
            }
        } else if (epoch_steps_ > 0) {
            // Epoch-static hot region. The hot list is rebuilt only at an epoch boundary from the
            // load window, so within an epoch the hot rows of the table never move; the spill ring
            // absorbs whatever the epoch misses. Routing every step, placement every N steps, and
            // nothing in between - the arrangement vLLM (eplb_state.py:647-693) and DeepInfer
            // (eplb/__init__.py:146-165) both use.
            if (load_window_.size() != count_) { load_window_.assign(count_, 0.0F); }
            if (hot_list_.empty()) {
                std::vector<bool> seen(count_, false);
                for (std::size_t i = 0; i < values && hot_list_.size() < hot_cap; ++i) {
                    const auto expert = static_cast<std::uint32_t>(route_ids[i]);
                    if (seen[expert]) { continue; }
                    seen[expert] = true;
                    hot_list_.push_back(expert);
                }
                if (static_warm_) { refresh_static_candidates(); }
            }
            for (std::size_t i = 0; i < values; ++i) {
                load_window_[static_cast<std::size_t>(route_ids[i])] += 1.0F;
            }
            if (++step_in_epoch_ >= epoch_steps_) {
                std::vector<std::uint32_t> ranked(count_);
                std::iota(ranked.begin(), ranked.end(), 0U);
                std::stable_sort(ranked.begin(), ranked.end(), [&](std::uint32_t a, std::uint32_t b) {
                    return load_window_[a] > load_window_[b];
                });
                hot_list_.assign(ranked.begin(), ranked.begin() + std::min<std::size_t>(hot_cap, ranked.size()));
                std::fill(load_window_.begin(), load_window_.end(), 0.0F);
                step_in_epoch_ = 0;
                ++epochs_;
                if (static_warm_) { refresh_static_candidates(); }
            }
            wanted.assign(hot_list_.begin(), hot_list_.end());
            for (const auto expert : wanted) { chosen[expert] = true; }
            for (std::size_t i = 0; i < values && wanted.size() < capacity; ++i) {
                const auto expert = static_cast<std::uint32_t>(route_ids[i]);
                if (chosen[expert]) { continue; }
                chosen[expert] = true;
                wanted.push_back(expert);
            }
        } else {
            // Region split (NINFER_EXPERT_REGIONS=1): the hot region is the top of the decayed
            // score, the spill ring is this step's misses, and the two never trade places.
            // is this step's misses. The routes no longer choose the hot set - that was the
            // streaming policy, and it made the hot region a per-step artefact with no slot that
            // survived a change of routing.
            if (!static_set && !generation_mode) {
                if (wanted.size() < hot_cap) {
                    std::vector<std::uint32_t> ranked(count_);
                    std::iota(ranked.begin(), ranked.end(), 0U);
                    std::stable_sort(ranked.begin(), ranked.end(),
                                     [&](std::uint32_t a, std::uint32_t b) {
                                         return hot_score_[a] > hot_score_[b];
                                     });
                    for (const auto expert : ranked) {
                        if (wanted.size() >= hot_cap) { break; }
                        if (chosen[expert]) { continue; }
                        chosen[expert] = true;
                        wanted.push_back(expert);
                    }
                }
                for (std::size_t i = 0; i < values && wanted.size() < capacity; ++i) {
                    const auto expert = static_cast<std::uint32_t>(route_ids[i]);
                    if (chosen[expert]) { continue; }
                    chosen[expert] = true;
                    wanted.push_back(expert);
                }
            }
            if (generation_mode) {
                // Hold it for the generation: a set that changes per step cannot back a graph and
                // makes every layer republish its offset table.
                frozen_        = true;
                frozen_wanted_ = wanted;
            }
            if (static_set && freeze_after > 0 &&
                ++steps_ >= static_cast<std::uint64_t>(freeze_after)) {
                frozen_        = true;
                frozen_wanted_ = wanted;
            }
        }
        // port (sm_89): Q11/Q12 - the freeze rule as a measurement, not as a switch.
        //
        // The old rule read "no_miss >= 99%" for the whole run, which is a percentage nobody could
        // satisfy on a pool that holds part of a layer, and it said nothing about whether freezing
        // helped. The two questions are separate:
        //
        //   freeze  when the last 32 layers were all clean. Every route was already resident, so
        //           following the router cannot buy anything; holding the set still is free.
        //   keep    only if the following 32 layers show the freeze raised the static rate or did
        //           not lower throughput. If neither, the freeze bought nothing and is undone.
        //
        // `wanted` is the set the freeze is about, so the decision lives here rather than at the top
        // of ensure(). The window above counts every layer, including the covered fast returns.
        if (freeze_watch) {
            if (frozen_ && freeze_by_rule_) {
                ++frozen_layers_;
                if (frozen_layers_ >= kFreezeWindowCalls) {
                    const double static_now = policy_controller().signals.static_pct;
                    const double tps_now    = policy_controller().signals.decode_tps;
                    const bool static_up =
                        freeze_static_baseline_ >= 0.0 && static_now >= 0.0 &&
                        static_now > freeze_static_baseline_;
                    const bool tps_ok =
                        freeze_tps_baseline_ <= 0.0 || tps_now <= 0.0 ||
                        tps_now >= freeze_tps_baseline_ * 0.97;
                    if (static_up || tps_ok) {
                        if (bank_ == 0) {
                            std::fprintf(stderr,
                                         "[freeze] kept: static %.1f%% (was %.1f%%), tps %.1f "
                                         "(was %.1f)\n",
                                         static_now, freeze_static_baseline_, tps_now,
                                         freeze_tps_baseline_);
                            std::fflush(stderr);
                        }
                    } else {
                        frozen_         = false;
                        freeze_by_rule_ = false;
                        if (bank_ == 0) {
                            std::fprintf(stderr,
                                         "[freeze] thawed: static %.1f%% (was %.1f%%) and tps %.1f "
                                         "(was %.1f) - the freeze bought nothing\n",
                                         static_now, freeze_static_baseline_, tps_now,
                                         freeze_tps_baseline_);
                            std::fflush(stderr);
                        }
                    }
                    frozen_layers_          = 0;
                    freeze_static_baseline_ = static_now;
                    freeze_tps_baseline_    = tps_now;
                }
            } else if (!frozen_ && freeze_window_layers_ >= kFreezeWindowCalls &&
                       freeze_window_clean_ >= freeze_window_layers_) {
                frozen_                 = true;
                freeze_by_rule_         = true;
                frozen_wanted_          = wanted;
                frozen_layers_          = 0;
                freeze_static_baseline_ = policy_controller().signals.static_pct;
                freeze_tps_baseline_    = policy_controller().signals.decode_tps;
                freeze_window_layers_   = 0;
                freeze_window_clean_    = 0;
                if (bank_ == 0) {
                    std::fprintf(stderr,
                                 "[freeze] frozen: %u clean layers, static=%.1f%% tps=%.1f\n",
                                 kFreezeWindowCalls, freeze_static_baseline_,
                                 freeze_tps_baseline_);
                    std::fflush(stderr);
                }
            }
            if (freeze_window_layers_ >= kFreezeWindowCalls) {
                freeze_window_layers_ = 0;
                freeze_window_clean_  = 0;
            }
        }
        // port (sm_89): the router does not repeat a step, so the set is refreshed every step. A
        // periodic refresh looked cheaper on paper (two device copies saved) but measured 11.1
        // against 24.3 tok/s: a stale set sends far more experts down the host tier than the two
        // copies ever cost.
        // Coverage of the fixed set: how many of this step's routes the pool already holds. A
        // captured decode graph is only correct when this is 100%, so it is the number the graph
        // work has to reach before the gate may open.
        {
            // The tally is per stream and readable from outside: a set decision, a prefetch target
            // and (once a table is baked into a capture) a capture gate all need this number, and
            // computing it here is free - the routes are already on the host.
            std::uint64_t inside = 0;
            for (std::size_t i = 0; i < values; ++i) {
                if (chosen[static_cast<std::size_t>(route_ids[i])]) { ++inside; }
            }
            // The profile is written from here rather than at exit: the streams are already gone by
            // the time an atexit handler runs, and a snapshot every few steps is enough to rank a
            // bank. 40 banks x 256 floats is 40 KiB.
            if (std::getenv("NINFER_EXPERT_PROFILE_DUMP") != nullptr && ++coverage_dumps_ % 8 == 0) {
                dump_profiles();
            }
            // port (sm_89): route trace for offline analysis (NINFER_EXPERT_ROUTE_DUMP=<path>). One
            // line per (layer, token): the experts that token routed to. Nothing reads it at run
            // time; it exists because every cache decision - layer-to-layer reuse, step-to-step
            // reuse, how far a prediction is worth trusting - is a claim about this data, and the
            // numbers quoted for those claims have to come from a run rather than an estimate.
            if (const char* dump = std::getenv("NINFER_EXPERT_ROUTE_DUMP")) {
                static std::FILE* trace = std::fopen(dump, "wb");
                if (trace != nullptr) {
                    const auto top = static_cast<std::size_t>(top_k <= 0 ? 1 : top_k);
                    for (std::size_t token = 0; token < static_cast<std::size_t>(tokens); ++token) {
                        std::fprintf(trace, "%u %zu", bank_, token);
                        for (std::size_t k = 0; k < top; ++k) {
                            std::fprintf(trace, " %d", route_ids[k * static_cast<std::size_t>(tokens) + token]);
                        }
                        std::fputc('\n', trace);
                    }
                    std::fflush(trace);
                }
            }
            // One sample per decode token: the spill region has to hold what a token misses, so
            // the tail of this distribution - not its mean - is the number of slots it needs.
            if (tokens <= 8 && capacity < count_) {
                if (miss_last_owner_.size() != count_) { miss_last_owner_.assign(count_, 0); }
                // Distinct routes the previous set did not hold. `chosen` cannot answer this: it
                // already contains every route of this step by construction, which is exactly why the
                // histogram read zero for the whole run before.
                std::uint32_t miss = 0;
                for (std::size_t i = 0; i < values; ++i) {
                    const auto expert = static_cast<std::size_t>(route_ids[i]);
                    if (miss_last_owner_[expert] != 0) { continue; }
                    miss_last_owner_[expert] = 1;
                    ++miss;
                }
                miss_hist_[std::min<std::uint32_t>(miss, 32)] += 1;
                refresh_spill_measure(capacity);
                // This step's set becomes the reference for the next one.
                std::fill(miss_last_owner_.begin(), miss_last_owner_.end(), std::uint8_t{0});
                for (const auto expert : wanted) {
                    if (expert < miss_last_owner_.size()) { miss_last_owner_[expert] = 1; }
                }
                if (++miss_samples_ >= 256) {
                    for (auto& bucket : miss_hist_) { bucket /= 2; }
                    miss_samples_ = 0;
                }
            }
            // "no miss" is not `inside == values`: `chosen` is this step's own placement, so that
            // comparison is true by construction. The honest count is taken at the tail of ensure,
            // from what actually had to be moved into the pool.
            coverage_.routes += values;
            coverage_.inside += inside;
            // Q50: the live point. The split is decided once at startup from a baked curve; this is
            // how the run's own measurement replaces that prior, so the next decision (the runtime
            // re-split, which is the controller's) reads this machine's numbers and not a comment.
            // Bank 0 only: every bank sees the same pool size, and forty copies of one observation
            // would just be forty writes to the same 512 MiB bucket.
            static const bool route_all = std::getenv("NINFER_ROUTE_ALL") != nullptr;
            const bool curve_tick = (bank_ == 0 || route_all) && ++coverage_calls_ % 40 == 0;
            if (curve_tick) {
                const auto routes = static_cast<double>(coverage_.routes == 0 ? 1 : coverage_.routes);
                const auto misses_per_token =
                    top_k == 0 ? 0.0
                               : static_cast<double>(coverage_.routes - coverage_.inside) *
                                     static_cast<double>(top_k) / routes;
                ninfer::core::observe_pool_value(
                    cache_->capacity_bytes(), misses_per_token,
                    policy_controller().signals.decode_tps);
            }
            if (curve_tick && (std::getenv("NINFER_SPLIT_TIMING") != nullptr ||
                 std::getenv("NINFER_EXPERT_TIMING") != nullptr)) {
                const auto routes = static_cast<double>(coverage_.routes == 0 ? 1 : coverage_.routes);
                const auto covered = 100.0 * static_cast<double>(coverage_.inside) / routes;
                // One token publishes top_k entries per layer, so the miss rate scales back to a
                // per-token count without knowing how many layers this stream has seen.
                const auto misses_per_token =
                    top_k == 0 ? 0.0
                               : static_cast<double>(coverage_.routes - coverage_.inside) *
                                     static_cast<double>(top_k) / routes;
                const auto bytes_per_step =
                    publish_calls_ == 0 ? 0.0
                                        : static_cast<double>(publish_bytes_) /
                                              static_cast<double>(publish_calls_);
                std::fprintf(stderr,
                             "[set] bank=%u coverage=%llu/%llu (%.1f%%) misses/token=%.2f slots=%u "
                             "publish=%.0fB/call max=%lluB step_max=%lluB (A1 128B/step: %s)\n",
                             bank_, static_cast<unsigned long long>(coverage_.inside),
                             static_cast<unsigned long long>(coverage_.routes), covered,
                             misses_per_token, capacity, bytes_per_step,
                             (unsigned long long)publish_bytes_max_,
                             (unsigned long long)publish_bytes_step_max_,
                             publish_bytes_step_max_ <= 128 ? "step part fits"
                                                            : "step part over");
                // P6/step 5: where the routes actually came from. Percentages because the absolute
                // counts differ per bank and the question is "which structure is carrying this layer".
                const auto total = static_cast<double>(route_hot_ + route_ring_ + route_prefetch_ +
                                                       route_host_ + route_cpu_);
                const auto pct = [&](std::uint64_t part) {
                    return total == 0 ? 0.0 : 100.0 * static_cast<double>(part) / total;
                };
                std::fprintf(stderr,
                             "[route] bank=%u hot=%.1f%% ring=%.1f%% pred=%.1f%% host=%.1f%% "
                             "cpu=%.1f%% (n=%llu)\n",
                             bank_, pct(route_hot_), pct(route_ring_), pct(route_prefetch_),
                             pct(route_host_), pct(route_cpu_),
                             (unsigned long long)(route_hot_ + route_ring_ + route_prefetch_ +
                                                  route_host_ + route_cpu_));
                const auto entered_per_step = entered_steps_ == 0
                    ? 0.0
                    : static_cast<double>(entered_) / static_cast<double>(entered_steps_);
                if (bank_ == 0) {
                    std::fprintf(stderr, "[set] bank=0 entered=%.2f/step\n", entered_per_step);
                    if (host_ != nullptr) {
                        const auto& warm = host_->stats();
                        std::fprintf(stderr,
                                     "[warm] second-chance: admitted=%llu declined=%llu hits=%llu "
                                     "promoted=%llu used=%u | warm hits=%llu misses=%llu "
                                     "evictions=%llu\n",
                                     (unsigned long long)warm.second_chance_admissions,
                                     (unsigned long long)warm.second_chance_declines,
                                     (unsigned long long)warm.second_chance_hits,
                                     (unsigned long long)warm.second_chance_promotions,
                                     host_->second_chance_used(),
                                     (unsigned long long)warm.hits,
                                     (unsigned long long)warm.misses,
                                     (unsigned long long)warm.evictions);
                        if (static_warm_) {
                            std::fprintf(stderr,
                                         "[warm-static] table refreshes=%llu candidates=%u "
                                         "set=%u/bank\n",
                                         (unsigned long long)static_table_refreshes_,
                                         static_candidates_count_, hot_cap_);
                        }
                    }
                }
            }
        }
        // port (sm_26): a prefill routes (nearly) every expert of a layer, so letting it choose the
        // resident set evicts exactly the hot set decode needs. llama.cpp's MoE cache draws the same
        // line for the same reason - large batches only read the cache, never fill it. Above the
        // decode batch size the set stays as decode left it, and routes outside it take the host
        // tier, which prefill would pay to read either way.
        //
        // The first call still builds the set: with nothing resident every route would fall to the
        // host tier and a cold prefill would never establish one.
        //
        // A pool that cannot hold a layer's prefill routes gains nothing from protecting the set:
        // the prefill fills what little there is with experts decode will not ask for, and decode
        // then pays a host read for the one or two it misses. Measured at a 512 MiB pool: 32.1 ->
        // 24.3 tok/s and prefill 3.4 -> 5.0 s. Above roughly a hundred slots a layer the prefill
        // set is broad enough to be a good decode set, and the same change measured 59-62 -> 72.5
        // at 12 GiB.
        constexpr std::int32_t kSetUpdateTokens = 8;
        // port (sm_89): Q20a - the pool keeps a stable hot set, and this layer's misses become the
        // overflow's job.
        //
        // The streaming policy below places the step's own routes first, which is why the pool evicts
        // every layer and the offset table moves every layer. That is fine when the host tier is a
        // CPU fallback, and fatal for a captured layer: a table that moves cannot be baked, and a
        // route that is force-placed never needs the second leg - measured as miss=0 with the leg
        // wired, so the overflow was unreachable by construction.
        //
        // Under NINFER_HOST_DIRECT the set is chosen by demand only. A route outside it is served
        // from the pinned tier if it is there, and by the CPU half if it is not, which is the split
        // Q20 is about.
        // Q20: device-side admission uses these even when it finds nothing; zero means the ring is
        // host-filled (the measured default).
        std::uint32_t device_ring_slots = device_ring_slots_;
        if (host_direct && !frozen_) {
            std::vector<std::uint32_t> ranked(count_);
            std::iota(ranked.begin(), ranked.end(), 0U);
            std::stable_sort(ranked.begin(), ranked.end(), [&](std::uint32_t a, std::uint32_t b) {
                return hot_score_[a] > hot_score_[b];
            });
            // Q20: the fixed S ring, finally active on the path that has a device-side second leg.
            //
            // Under host-direct the set used to be the top `capacity` experts by score, so the ring
            // was overwritten here and `[route] ring` read 0.0% even with an epoch set: every route
            // outside the hot set went to the pinned tier. That is safe but it is not the structure
            // GC2 needs - R2 has to be a fixed-S region whose members change without the grid
            // changing. S is taken from the same place the split policy takes it (a fixed few slots,
            // one token's routing), so a layer that misses k <= S experts serves them from the pool
            // and only the rest fall to the host leg.
            std::uint32_t ring = device_ring_slots != 0 ? device_ring_slots : 0;
            if (ring == 0 && std::getenv("NINFER_EXPERT_RING_SLOTS") != nullptr) {
                const char* value = std::getenv("NINFER_EXPERT_RING_SLOTS");
                ring = static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10));
            } else if (ring == 0 && regions_) {
                ring = spill_slots(capacity);
            }
            ring = std::min<std::uint32_t>(ring, static_cast<std::uint32_t>(capacity / 2));
            const auto keep = static_cast<std::size_t>(capacity - ring);
            wanted.assign(ranked.begin(),
                          ranked.begin() + static_cast<std::ptrdiff_t>(
                                               std::min<std::size_t>(keep, ranked.size())));
            std::fill(chosen.begin(), chosen.end(), false);
            for (const auto expert : wanted) { chosen[expert] = true; }
            // Q20: when the Device owns admission, the host stops naming this step's ring members.
            // Leaving the slots empty is deliberate - the kernel below fills them from the pinned
            // candidate set, and only the experts it names are copied.
            static const bool ring_device_admit = [] {
                const char* value = std::getenv("NINFER_RING_DEVICE_ADMIT");
                return value != nullptr && *value != '\0' && std::strtol(value, nullptr, 10) != 0;
            }();
            if (!ring_device_admit) {
                for (std::size_t i = 0; i < values && wanted.size() < capacity; ++i) {
                    const auto expert = static_cast<std::uint32_t>(route_ids[i]);
                    if (chosen[expert]) { continue; }
                    chosen[expert] = true;
                    wanted.push_back(expert);
                }
            }
            // The attribution pass and set_resident() both read this: everything before `keep` is the
            // hot regime, everything after it is the ring.
            hot_cap_ = keep;
        }
        // port (sm_89): Q20 second leg, filled BEFORE the Device tables are built.
        //
        // The tables name slots, so a slot claimed after they are built still holds the previous
        // expert: measured as host_experts=5.0 route_hits=0.0, a leg that looked wired and served
        // nothing. This is the same fill the tier always wanted; only its position changed.
        // Whether this layer is allowed to change the pool's resident set. Computed here because
        // the host fill has to know which pool it is complementing: a fill against `wanted` is right
        // only if `wanted` is about to be installed, and a fill against the current residency is
        // right only if the set is held. Getting this backwards leaves routes with no address on
        // either leg (measured: ~190 routes a layer on a prefill).
        const bool refresh = !set_initialised_ || tokens <= kSetUpdateTokens ||
                             !pool_covers_layer(static_cast<std::uint32_t>(capacity));
        if (host_tier && host_ != nullptr && route_ids != nullptr &&
            std::getenv("NINFER_SKIP_HOST_FILL") == nullptr) {
            static thread_local std::vector<std::int32_t> host_distinct;
            host_distinct.assign(route_ids, route_ids + values);
            std::sort(host_distinct.begin(), host_distinct.end());
            host_distinct.erase(std::unique(host_distinct.begin(), host_distinct.end()),
                                host_distinct.end());
            // P0: the tier serves the routes the pool will NOT hold this layer. Asking the current
            // residency is one layer stale: `set_resident_experts` runs after this fill and installs
            // `wanted`, so an expert that is resident now but not in `wanted` would be skipped here
            // and then dropped from the pool - no address on either leg, and the CPU half computes
            // it. That gap was ~13.5 routes a layer on the streaming path.
            static thread_local std::vector<std::int32_t> host_targets;
            host_targets.clear();
            host_targets.reserve(host_distinct.size());
            {
                static thread_local std::vector<std::uint8_t> in_wanted;
                if (in_wanted.size() != count_) { in_wanted.assign(count_, 0); }
                std::fill(in_wanted.begin(), in_wanted.end(), std::uint8_t{0});
                for (const auto expert : wanted) {
                    if (expert < in_wanted.size()) { in_wanted[expert] = 1; }
                }
                for (const auto expert : host_distinct) {
                    const auto id = static_cast<std::uint32_t>(expert);
                    const bool pool_serves =
                        refresh ? (id < in_wanted.size() && in_wanted[id] != 0)
                                : cache_->expert_slot(bank_, id).has_value();
                    if (!pool_serves) { host_targets.push_back(expert); }
                }
            }
            // P0: the tier's own hit counter cannot answer "did the static set survive" - the static
            // path asks only for the experts that are missing, so a perfect static set still reports
            // hits=0. Count per expert against the membership list instead, and count what the layer's
            // own routes found, which is the coverage the arrangement actually buys.
            static const bool static_probe = std::getenv("NINFER_STATIC_PROBE") != nullptr;
            if (static_probe && static_warm_) {
                if (static_probe_slots_.size() != count_) {
                    static_probe_slots_.assign(count_, HostExpertResidency::kAbsentSlot);
                }
                host_->snapshot_slots(bank_, static_probe_slots_);
                std::uint32_t static_present = 0, dyn_present = 0;
                for (std::uint32_t expert = 0; expert < count_; ++expert) {
                    if (static_probe_slots_[expert] == HostExpertResidency::kAbsentSlot) { continue; }
                    const bool member = expert < static_member_.size() && static_member_[expert] != 0;
                    if (member) { ++static_present; } else { ++dyn_present; }
                }
                std::uint32_t route_static = 0, route_dyn = 0, route_absent = 0;
                for (const auto expert : host_distinct) {
                    const auto id = static_cast<std::uint32_t>(expert);
                    if (id >= count_) { continue; }
                    if (static_probe_slots_[id] == HostExpertResidency::kAbsentSlot) {
                        ++route_absent;
                    } else if (id < static_member_.size() && static_member_[id] != 0) {
                        ++route_static;
                    } else {
                        ++route_dyn;
                    }
                }
                static_probe_static_present_ += static_present;
                static_probe_dyn_present_ += dyn_present;
                static_probe_route_static_ += route_static;
                static_probe_route_dyn_ += route_dyn;
                static_probe_route_absent_ += route_absent;
                if (bank_ == 0 && ++static_probe_layers_ % 1 == 0) {
                    constexpr double kWindow = 1.0;
                    std::fprintf(stderr,
                                 "[static-probe] bank=0 static_present=%.1f dyn_present=%.1f "
                                 "route_static=%.2f route_dyn=%.2f route_absent=%.2f q=%u\n",
                                 static_cast<double>(static_probe_static_present_) / kWindow,
                                 static_cast<double>(static_probe_dyn_present_) / kWindow,
                                 static_cast<double>(static_probe_route_static_) / kWindow,
                                 static_cast<double>(static_probe_route_dyn_) / kWindow,
                                 static_cast<double>(static_probe_route_absent_) / kWindow,
                                 static_cast<std::uint32_t>(static_member_.size() == 0
                                                                ? 0
                                                                : std::count(static_member_.begin(),
                                                                             static_member_.end(),
                                                                             std::uint8_t{1})));
                    std::fflush(stderr);
                    static_probe_static_present_ = 0;
                    static_probe_dyn_present_    = 0;
                    static_probe_route_static_   = 0;
                    static_probe_route_dyn_      = 0;
                    static_probe_route_absent_   = 0;
                }
            }
            if (freeze_mode_ && !static_frozen_) {
                // P1 warm-up. Before the set is chosen the static stripe is an ordinary working set:
                // every route is filled there (overflowing into the dynamic stripe), so the layer
                // runs with cpu=0 while the scores are still being learned. The freeze replaces the
                // contents wholesale, so churn here costs nothing that survives.
                if (freeze_trace()) {
                    std::fprintf(stderr,
                                 "[freeze] bank=%u warm targets=%zu s=%u d=%u used=%u clock=%llu\n",
                                 bank_, host_targets.size(), host_->static_slots_per_bank(),
                                 host_->dynamic_slots_per_bank(), 0U,
                                 static_cast<unsigned long long>(freeze_clock_));
                    std::fflush(stderr);
                }
                host_->ensure_scored(ExpertBankId{bank_},
                                     std::span<const std::int32_t>(host_targets),
                                     hot_score_.data(), /*static_fill=*/true);
                if (freeze_trace()) {
                    std::fprintf(stderr, "[freeze] bank=%u warm fill done\n", bank_);
                    std::fflush(stderr);
                }
                publish_static_table();
                if (freeze_trace()) {
                    std::fprintf(stderr, "[freeze] bank=%u warm publish done\n", bank_);
                    std::fflush(stderr);
                }
            } else if (static_warm_ && static_candidates_ready_) {
                // Fill the whole target set, not only the ones absent from the last snapshot. The
                // snapshot says what was present BEFORE this fill, and the fill itself can evict a
                // D entry that is one of this layer's targets - which then has no address and falls
                // to the CPU half. Filling every target is a hash lookup for the resident ones and
                // makes the fill's own protection cover the layer.
                if (host_->bank_slot_generation(bank_) != static_table_generation_) {
                    publish_static_table();
                }
                if (!host_targets.empty()) {
                    host_->ensure_scored(ExpertBankId{bank_},
                                         std::span<const std::int32_t>(host_targets),
                                         hot_score_.data());
                    if (host_->bank_slot_generation(bank_) != static_table_generation_) {
    publish_static_table();
    // Upload the whole frozen pool set now. `set_resident_experts` only names the slots; the images
    // arrive lazily through prepare() on the layer that routes to them. A later layer that skips
    // ensure() would otherwise read another expert's bytes for any member whose image never landed -
    // measured as a wrong hash with the fast path on and correct with the full path.
    if (cache_ != nullptr && !frozen_wanted_.empty()) {
        std::vector<ExpertCacheKey> preload;
        preload.reserve(frozen_wanted_.size());
        for (const auto expert : frozen_wanted_) {
            if (expert < keys_.size()) { preload.push_back(keys_[expert]); }
        }
        cache_->prepare(std::span<const ExpertCacheKey>(preload), stream);
        cache_->wait_for_progress(stream);
        // Did the preload actually land? `offset()` answers only for a ready image, so this is the
        // difference between "the set is named" and "the set is on the Device".
        std::uint32_t unready = 0;
        for (const auto expert : frozen_wanted_) {
            if (!cache_->offset(keys_[expert])) { ++unready; }
        }
        if (unready != 0) {
            std::fprintf(stderr, "[freeze] bank=%u preload unready=%u of %zu\n", bank_, unready,
                         frozen_wanted_.size());
            std::fflush(stderr);
        }
    }
                    }
                }
            } else {
                host_->ensure_scored(ExpertBankId{bank_},
                                     std::span<const std::int32_t>(host_targets), hot_score_.data());
            }
        }
        if (freeze_mode_) {
            // The freeze installs and fills the set itself; there is nothing to refresh, and
            // refreshing before the freeze would install the first token's routing as "the set".
        } else if (static_warm_ && !static_candidates_ready_) {
            refresh_static_candidates();
        }
        if (host_direct && device_ring_slots != 0 && host_ != nullptr && device_ids != nullptr) {
            const auto admitted =
                device_admit_ring(device_ids, static_cast<std::int32_t>(values), 0,
                                  device_ring_slots, stream, wanted, chosen);
            if (admitted != 0 && bank_ == 0 && std::getenv("NINFER_RING_PROBE") != nullptr) {
                static int shown = 0;
                if (shown++ < 8) {
                    std::fprintf(stderr, "[ring] device admitted %zu expert(s)/layer (S=%u)\n",
                                 admitted, device_ring_slots);
                    std::fflush(stderr);
                }
            }
        }
        if (host_ != nullptr) { host_->begin_layer(); }
        if (freeze_trace()) {
            std::fprintf(stderr, "[freeze] bank=%u begin_layer done wanted=%zu\n", bank_,
                         wanted.size());
            std::fflush(stderr);
        }
        // Q20: rotate the Device-hold generations once per layer, whether or not the table moved.
        // Skipping it on an unchanged layer would leave a slot pinned for a layer that never reads
        // it, which is a leak the tier cannot see.
        // Q30/P6: the prediction is placed by stage_predicted(), at the end of the layer that made it.
        // Placing it here was tried first and is structurally useless: by the time ensure() runs, this
        // layer's routes are known, so "predicted and not resident" is no longer a prediction - it is
        // the demand set, and the demand path fills it anyway.
        // A prediction is a placement, so the table has to be rebuilt even when the resident set did
        // not move - otherwise the rows the region just took never reach the Device.
        const bool set_moved =
            refresh && cache_->set_resident_experts(bank_, wanted, static_cast<std::uint32_t>(hot_cap));
        if (freeze_trace()) {
            std::fprintf(stderr, "[freeze] bank=%u resident done moved=%d\n", bank_, int(set_moved));
            std::fflush(stderr);
        }
        // A section's publish is needed when either the residency changed or a segment moved since the
        // last publication. The second half matters now that the prediction region is staged at the
        // end of the previous layer's ffn: that placement happens outside ensure(), so `set_moved`
        // does not see it and the rows would otherwise never reach the Device.
        const auto hot_gen      = cache_->hot_generation(bank_);
        const auto step_gen     = cache_->step_generation(bank_);
        const auto prefetch_gen = cache_->prefetch_generation(bank_);
        const bool need_publish = !published_once_ || pub_hot_ != hot_gen || pub_step_ != step_gen ||
                                 pub_prefetch_ != prefetch_gen;
        // port (sm_89): GH3 - the Device derives its own tables from one placement word per expert.
        //
        // This is where the NaN under NINFER_HOST_DIRECT came from, and it is worth naming because the
        // shape of the fix is the answer to it. The tier holds each expert for a while and then drops
        // it; a Device table that names a slot is only true while that slot still holds that expert.
        // The old publisher tied the tier's table to the POOL's generations - so on the layers where
        // the pool happened not to move (which is most of them once the hot set is static), the tier
        // table went stale while the arena kept churning underneath it: the hold it had taken expired
        // after two layers, the slot was reused for another expert, and the GEMM addressed it. That is
        // reading another expert's bytes, and half a bf16 stream of another expert overflows to inf
        // within a few layers - the router's coefficients came back NaN and the CPU half refused them.
        //
        // The invariant that removes it: the tier's contribution to the Device tables is re-derived
        // EVERY layer, from one snapshot, and exactly the slots it names are held for that layer. The
        // placement word is that contribution, so `placement_mode` also has to rebuild the block below
        // even when nothing in the pool moved.
        const bool placement_mode =
            host_direct && host_ != nullptr && host_->arena_device_base() != nullptr &&
            std::getenv("NINFER_SKIP_PLACEMENT") == nullptr;
        if (set_moved || need_publish || placement_mode) {
            set_initialised_ = true;
            if (freeze_trace()) {
                std::fprintf(stderr, "[fz] bank=%u block moved=%d publish=%d\n", bank_, int(set_moved),
                             int(need_publish));
                std::fflush(stderr);
            }
            // One pass over the pool. The slot each expert owns was read twice here (once for the
            // offset table, once for the diff) and GH3 needs the same answer a third time for the
            // placement word, so it is taken once and shared.
            if (pool_slot_.size() != count_) { pool_slot_.assign(count_, -1); }
            for (std::uint32_t expert = 0; expert < count_; ++expert) {
                const auto slot    = cache_->expert_slot(bank_, expert);
                pool_slot_[expert] = slot ? static_cast<std::int32_t>(*slot) : -1;
            }
            if (offsets_.size() != count_) { offsets_.assign(count_, 0); }
            std::fill(offsets_.begin(), offsets_.end(), 0);
            mask_host_.assign(count_, 0);
            for (std::uint32_t expert = 0; expert < count_; ++expert) {
                if (pool_slot_[expert] < 0) { continue; }
                offsets_[expert]   = cache_->slot_offset(
                    static_cast<std::uint32_t>(pool_slot_[expert]));
                mask_host_[expert] = 1;
            }
            resident_order_ = wanted;
            const auto table_bytes = offsets_.size() * sizeof(std::uint64_t);
            const auto mask_bytes  = mask_host_.size() * sizeof(std::int32_t);
            // Only the rows that moved are copied. A step usually changes a handful of experts
            // (one lane misses well under one expert a layer at 163 slots), while the full table is
            // 2 KiB of offsets plus 1 KiB of mask copied forty times a token, every step, whether
            // anything changed or not. The full copy stays as the fallback when most rows move.
            if (prev_slot_.size() != count_) { prev_slot_.assign(count_, -1); }
            std::uint32_t entered_this_step = 0;
            demoted_.clear();
            std::vector<std::uint32_t> moved;
            moved.reserve(count_);
            for (std::uint32_t expert = 0; expert < count_; ++expert) {
                const auto now = pool_slot_[expert];
                if (now != prev_slot_[expert]) {
                    moved.push_back(expert);
                    // Leaving the Device pool is the one signal that says "this expert was hot and is
                    // not hot enough any more" - the warm tier's admission criterion (1y.7 step 3b).
                    if (prev_slot_[expert] >= 0 && now < 0) {
                        demoted_.push_back(static_cast<std::int32_t>(expert));
                    }
                    // An expert that was not resident at all is a row that has to be uploaded, not
                    // just a table entry that moved: that is the number the transfer cost follows.
                    if (prev_slot_[expert] < 0 && now >= 0) { ++entered_this_step; }
                }
            }
            // port (sm_89): Q20's second leg, and GH3's placement word.
            //
            // Only experts ALREADY in the pinned tier qualify. Filling a cold one here would be a
            // synchronous artifact read inside the layer, which is exactly the host round trip this
            // removes - a cold expert stays absent and is served by the CPU half as before.
            //
            // The tier's side of the table is rebuilt here on EVERY layer, not only when the pool
            // moved. That is the fix discussed above: the arena keeps churning while the pool sits
            // still, so "the pool did not move" says nothing about whether this table is still true.
            bool placement_changed = false;
            // Which region a changed row came from. A rate says the table moved; this says whose
            // lifetime did, which is the difference between "S is not frozen" and "D churned".
            std::uint32_t moved_pool = 0, moved_s = 0, moved_d = 0, moved_c = 0, moved_p = 0,
                          moved_absent = 0;
            if (placement_mode) {
                std::uint64_t gu_plane[3] = {}, dn_plane[3] = {};
                std::uint64_t gu_high_bytes = 0, dn_high_bytes = 0;
                const auto* host_base = host_->arena_device_base();
                if (placement_.size() != count_) {
                    placement_.assign(count_, ops::kExpertPlacementAbsent);
                    placement_published_.assign(count_, ops::kExpertPlacementAbsent);
                    host_slots_.assign(count_, HostExpertResidency::kAbsentSlot);
                    host_flags_.assign(count_, 0);
                }
                if (host_base != nullptr &&
                    host_->plane_offsets(bank_, gu_plane, dn_plane, gu_high_bytes, dn_high_bytes)) {
                    for (std::size_t k = 0; k < 3; ++k) {
                        host_gate_up_plane_[k] = gu_plane[k];
                        host_down_plane_[k]    = dn_plane[k];
                    }
                    // One consistent answer for all 256 experts, under the tier's own lock.
                    host_->snapshot_slots(bank_, host_slots_);
                    placed_slots_.clear();
                    for (std::uint32_t expert = 0; expert < count_; ++expert) {
                        // The pool wins a tie: an expert resident in both is addressed in video memory,
                        // which is the copy the Device reads faster and the one the ring orders.
                        std::uint32_t placement = ops::kExpertPlacementAbsent;
                        if (pool_slot_[expert] >= 0) {
                            placement = static_cast<std::uint32_t>(pool_slot_[expert]);
                        } else if (host_slots_[expert] != HostExpertResidency::kAbsentSlot) {
                            placement = ops::kExpertPlacementHost | host_slots_[expert];
                        }
                        if (placement != placement_published_[expert]) {
                            placement_changed = true;
                            if (placement == ops::kExpertPlacementAbsent) {
                                ++moved_absent;
                            } else if ((placement & ops::kExpertPlacementHost) == 0) {
                                ++moved_pool;
                            } else {
                                switch (host_->slot_region(placement & ~ops::kExpertPlacementHost)) {
                                    case HostExpertResidency::SlotRegion::Static: ++moved_s; break;
                                    case HostExpertResidency::SlotRegion::SecondChance: ++moved_c; break;
                                    case HostExpertResidency::SlotRegion::Prefetch: ++moved_p; break;
                                    default: ++moved_d; break;
                                }
                            }
                        }
                        placement_[expert]  = placement;
                        const bool absent   = placement == ops::kExpertPlacementAbsent;
                        const bool on_host  = !absent && (placement & ops::kExpertPlacementHost) != 0;
                        host_flags_[expert] = on_host ? 1 : 0;
                        // The mask is the same rule the CPU half's complement uses: an expert has an
                        // address or it does not, whichever leg it lives on.
                        mask_host_[expert]  = absent ? 0 : 1;
                        if (on_host) {
                            placed_slots_.push_back(placement & ~ops::kExpertPlacementHost);
                        }
                    }
                    // The first layer has to publish even if the word happens to match the empty table
                    // the bank was attached with: "nothing has been published" is not "nothing moved".
                    if (!published_once_) { placement_changed = true; }
                    // Held for exactly as long as the table names them. This used to run only when a
                    // publication happened, which is precisely why a run of unchanged-pool layers
                    // could drop a slot the Device still addressed.
                    host_->hold_slots(placed_slots_);
                    probe_tables("full");
                    probe_device_tables("full", device_ids, values);
                    if (std::getenv("NINFER_READY_PROBE") != nullptr && bank_ == 0) {
                        static std::uint32_t shown = 0;
                        if (shown++ < 48) {
                            const auto unready =
                                cache_ == nullptr ? 0U : cache_->unready_count(bank_, frozen_wanted_);
                            std::fprintf(stderr, "[ready] seq=%u where=full mapped=%zu unready=%u\n",
                                         shown - 1, frozen_wanted_.size(), unready);
                            std::fflush(stderr);
                        }
                    }
                    if (std::getenv("NINFER_PUBLISH_PROBE") != nullptr &&
                        (moved_pool + moved_s + moved_d + moved_c + moved_p + moved_absent) != 0) {
                        static std::uint32_t shown = 0;
                        if (shown++ < 200) {
                            std::fprintf(stderr,
                                         "[publish] bank=%u pool=%u s=%u d=%u c=%u p=%u absent=%u\n",
                                         bank_, moved_pool, moved_s, moved_d, moved_c, moved_p,
                                         moved_absent);
                            std::fflush(stderr);
                        }
                    }
                    if (freeze_trace()) {
                        std::fprintf(stderr, "[fz] bank=%u held=%zu changed=%d\n", bank_,
                                     placed_slots_.size(), int(placement_changed));
                        std::fflush(stderr);
                    }
                    if (std::getenv("NINFER_STATIC_PROBE") != nullptr) {                         if (host_check_.size() != count_) { host_check_.assign(count_, HostExpertResidency::kAbsentSlot); }                         host_->snapshot_slots(bank_, host_check_);                         std::uint32_t moved = 0;                         for (std::uint32_t e = 0; e < count_; ++e) {                             if (host_slots_[e] != host_check_[e]) { ++moved; }                         }                         if (moved != 0) {                             static std::uint32_t reported = 0;                             if (bank_ == 0 && reported++ < 8) {                                 std::fprintf(stderr, "[static-probe] mapping moved after publish: bank=%u experts=%u\n", bank_, moved);                                 std::fflush(stderr);                             }                         }                     }
                    // The kernel reads these bases and the two tables, so they are handed to the view
                    // here rather than at rebase time - rebase only re-points the pool planes.
                    auto& gu = experts_.bank.gate_up;
                    auto& dn = experts_.bank.down;
                    gu.host_codes  = host_base + host_gate_up_plane_[0];
                    gu.host_high   = gu_high_bytes != 0 ? host_base + host_gate_up_plane_[1] : nullptr;
                    gu.host_scales = host_base + host_gate_up_plane_[2];
                    dn.host_codes  = host_base + host_down_plane_[0];
                    dn.host_high   = dn_high_bytes != 0 ? host_base + host_down_plane_[1] : nullptr;
                    dn.host_scales = host_base + host_down_plane_[2];
                    if (host_offsets_device_ == nullptr) {
                        host_offsets_device_ =
                            std::make_unique<DeviceBuffer>(count_ * sizeof(std::uint64_t));
                        host_flags_device_ =
                            std::make_unique<DeviceBuffer>(count_ * sizeof(std::int32_t));
                        placement_device_ =
                            std::make_unique<DeviceBuffer>(count_ * sizeof(std::uint32_t));
                        placement_stage_[0] =
                            std::make_unique<PinnedHostBuffer>(count_ * sizeof(std::uint32_t));
                        placement_stage_[1] =
                            std::make_unique<PinnedHostBuffer>(count_ * sizeof(std::uint32_t));
                    }
                    gu.host_offsets = static_cast<const std::uint64_t*>(host_offsets_device_->p);
                    dn.host_offsets = gu.host_offsets;
                    gu.host_flags   = static_cast<const std::int32_t*>(host_flags_device_->p);
                    dn.host_flags   = gu.host_flags;
                    if (placement_changed) {
                        // The one word per expert is the whole publication: 4 bytes per expert that
                        // moved, and the Device derives the three tables the GEMM reads.
                        //
                        // The batch reads from a pinned copy, and it alternates between two of them:
                        // the sources are consumed in stream order, so the copy this layer hands over
                        // must not be the one the layer before it already handed over.
                        auto& stage = placement_stage_[placement_stage_next_];
                        placement_stage_next_ ^= 1;
                        std::memcpy(stage->data(), placement_.data(), count_ * sizeof(std::uint32_t));
                        std::vector<void*>       dsts;
                        std::vector<const void*> srcs;
                        std::vector<std::size_t> sizes;
                        dsts.reserve(count_);
                        srcs.reserve(count_);
                        sizes.reserve(count_);
                        for (std::uint32_t expert = 0; expert < count_; ++expert) {
                            if (placement_[expert] == placement_published_[expert]) { continue; }
                            dsts.push_back(static_cast<char*>(placement_device_->p) +
                                           static_cast<std::size_t>(expert) * sizeof(std::uint32_t));
                            srcs.push_back(static_cast<const char*>(stage->data()) +
                                           static_cast<std::size_t>(expert) * sizeof(std::uint32_t));
                            sizes.push_back(sizeof(std::uint32_t));
                        }
                        cudaMemcpyAttributes attributes{};
                        attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
                        attributes.flags          = 0;
                        std::size_t attribute_index = 0;
                        if (freeze_trace()) {
                            std::fprintf(stderr, "[fz] bank=%u batch n=%zu\n", bank_, sizes.size());
                            std::fflush(stderr);
                        }
                        const auto submitted        = cudaMemcpyBatchAsync(
                            dsts.data(), srcs.data(), sizes.data(), sizes.size(), &attributes,
                            &attribute_index, 1, stream);
                        if (freeze_trace()) {
                            std::fprintf(stderr, "[fz] bank=%u batch done status=%d\n", bank_,
                                         int(submitted));
                            std::fflush(stderr);
                        }
                        publish_bytes_ += sizes.size() * sizeof(std::uint32_t);
                        publish_bytes_max_ =
                            std::max(publish_bytes_max_, sizes.size() * sizeof(std::uint32_t));
                        // A1's step judgement, on the same rule the pool's publication uses: a step
                        // that moves a handful of rows is the one a captured layer has to absorb. A
                        // tier that turned over half its table in one layer is a budget event, not the
                        // steady state, and averaging the two would hide both.
                        if (sizes.size() <= 4) {
                            publish_bytes_step_max_ = std::max<std::uint64_t>(
                                publish_bytes_step_max_, sizes.size() * sizeof(std::uint32_t));
                        }
                        if (submitted != cudaSuccess) {
                            // A driver that rejects the batch still has to get the word across, or the
                            // layer would run on the previous placement - the exact failure this
                            // rewrite exists to make impossible. The full copy is 1 KiB.
                            (void)cudaMemcpyAsync(placement_device_->p, stage->data(),
                                                  count_ * sizeof(std::uint32_t),
                                                  cudaMemcpyHostToDevice, stream);
                        }
                        ops::moe_expand_expert_placements(
                            static_cast<const std::uint32_t*>(placement_device_->p),
                            static_cast<int>(count_), cache_->slot_stride(bank_),
                            cache_->slot_bias(bank_), host_->slot_bytes(),
                            static_cast<std::uint64_t*>(offsets_device_->p),
                            static_cast<std::int32_t*>(mask_device_->p),
                            static_cast<std::uint64_t*>(host_offsets_device_->p),
                            static_cast<std::int32_t*>(host_flags_device_->p), stream);
                        if (freeze_trace()) {
                            std::fprintf(stderr, "[fz] bank=%u expand done\n", bank_);
                            std::fflush(stderr);
                        }
                        placement_published_ = placement_;
                        if (frozen_ && static_frozen_) { freeze_table_published_ = true; }
                        placement_dirty_     = false;
                        ++publish_calls_;
                    }
                    if (bank_ == 0 && std::getenv("NINFER_EXPERT_TIMING")) {
                        static std::uint64_t layers = 0, placed_total = 0, route_total = 0;
                        std::uint32_t host_hits = 0;
                        for (std::size_t i = 0; i < values; ++i) {
                            const auto expert = static_cast<std::uint32_t>(route_ids[i]);
                            if (expert < host_flags_.size() && host_flags_[expert] != 0) { ++host_hits; }
                        }
                        placed_total += placed_slots_.size();
                        route_total += host_hits;
                        if (++layers % 8 == 0) {
                            std::fprintf(stderr,
                                         "[hostd] per layer: host_experts=%.2f route_hits=%.2f "
                                         "of %llu\n",
                                         static_cast<double>(placed_total) / 8.0,
                                         static_cast<double>(route_total) / 8.0,
                                         static_cast<unsigned long long>(values));
                            std::fflush(stderr);
                            placed_total = 0;
                            route_total  = 0;
                        }
                    }
                } else {
                    // The tier cannot be addressed, so nothing it holds is offerable: every route the
                    // pool does not hold goes to the CPU half. Drop the addresses rather than leaving
                    // last layer's in place, and leave `placement_dirty_` set so the A3b check below
                    // reports it instead of computing with them.
                    placement_dirty_ = true;
                    std::fill(host_flags_.begin(), host_flags_.end(), 0);
                    experts_.bank.gate_up.host_flags   = nullptr;
                    experts_.bank.down.host_flags      = nullptr;
                    experts_.bank.gate_up.host_offsets = nullptr;
                    experts_.bank.down.host_offsets    = nullptr;
                }
            } else if (!host_flags_.empty()) {
                // Switching it off must leave nothing behind: a stale flags table would keep
                // addressing the tier for a route the pool expects to own.
                host_flags_.clear();
                experts_.bank.gate_up.host_flags   = nullptr;
                experts_.bank.down.host_flags      = nullptr;
                experts_.bank.gate_up.host_offsets = nullptr;
                experts_.bank.down.host_offsets    = nullptr;
            }
            const auto delta_bytes = moved.size() * (sizeof(std::uint64_t) + sizeof(std::int32_t));            static const bool delta_enabled = [] {
                const char* value = std::getenv("NINFER_PUBLISH_DELTA");
                return value == nullptr || value[0] != '0';
            }();
            // 13.5 (step 5): the delta path assumes the Device tables still hold the rows this
            // publisher last wrote. A resize breaks that baseline from outside - the cache moved
            // slots this stream never published, and the generation bump that follows is exactly
            // what `need_publish` reports. Measured on the e2 artifact: after a pool shrink the
            // first republish of the cleared banks submitted its moved rows through the batch
            // copy, the call returned success, and the Device table still held the pre-shrink
            // high offsets - the grouped kernel then read a released VMM segment and the layer
            // died as cudaErrorIllegalAddress. A publication that follows a generation change
            // therefore carries the whole table: the fallback path writes both buffers in one
            // submission and was verified to land (bank 0-3 of the same batch, moved=0, stayed
            // correct while every delta bank stayed stale).
            const bool delta = delta_enabled && !moved.empty() && !need_publish &&
                               delta_bytes * 2 < table_bytes + mask_bytes;
            // Async on the consumer stream: the kernels that read these tables are queued behind
            // them there, while a blocking copy would stall the host twice per layer.
            // port (sm_89): both tables change together and neither is ordered against the other,
            // so one batch submission replaces two. At 2 KiB and 1 KiB the copies are entirely
            // submission-bound (a 4-slice batch of this size measured 2.1x the loop form), and a
            // layer pays for them forty times a token. The sources are host vectors rewritten on
            // the next layer, so they keep stream ordering: the driver must not read them later.
            // The mask must be the member buffer, not a local: an async copy out of a local vector
            // outlives the scope that owns it, which is a use-after-free that only shows up as a
            // resident set that is occasionally one layer stale.
            // A layer whose set did not move has nothing to publish: the Device tables already say
            // what this layer needs. Skipping the submission is worth more than making it cheaper -
            // at 16 GiB most layers move nothing, and the layer's cost is a chain of submissions that
            // are not overlapped (1x.4s), not one expensive call.
            // "The table did not move" is a claim about the cache, so the cache has to confirm it: any
            // set_resident() that changed a bank bumps its generation, and a generation the publisher
            // has not seen yet forces the publication even when this step's `moved` is empty. That is
            // A3's check, applied where it now matters - the skip below would otherwise hide a stale
            // table behind a correct-looking diff.
            // A1: three segments, three lifetimes, so three generations on each side. The hot segment
            // may only move at an epoch boundary (so its generation does not change inside one), the
            // step segment moves with the ring, and the prediction region moves on its own placements.
            //
            // These are PER INSTANCE, not `static thread_local`: a thread-local counter here is shared
            // by all forty banks, so one bank's publish marked every other bank as up to date and the
            // next bank skipped its own - which the A3b check then caught as "device holds 23, cache
            // is on 24". The old single counter had the same shape; it only survived because every
            // bank happened to publish every layer.
            const bool stale = pub_hot_ != hot_gen || pub_step_ != step_gen ||
                              pub_prefetch_ != prefetch_gen;
            const bool unchanged = moved.empty() && published_once_ && !stale;
            if (placement_mode) {
                // GH3: the tables this layer reads were written by the Device from this layer's own
                // placement, so there is nothing to copy and no row to push. Only the bookkeeping has
                // to say they are current - A3b below compares those numbers, and skipping them would
                // make a correct table look stale.
                published_once_ = true;
                pub_hot_        = hot_gen;
                pub_step_       = step_gen;
                pub_prefetch_   = prefetch_gen;
                table_hot_      = hot_gen;
                table_step_     = step_gen;
                table_prefetch_ = prefetch_gen;
                // A layer whose placement moved did touch the host, so it is not host-free: the
                // question `capt_free` asks is whether a captured layer could replay the tables it
                // was captured with, and a row that changed answers that with "no". What got smaller
                // is the price of the answer (4 bytes a row, not 12), not the answer.
                if (placement_changed) { host_publish = true; }
            } else if (!unchanged && table_bytes != 0 && mask_bytes != 0) {
                host_publish    = true;
                published_once_ = true;
                pub_hot_        = hot_gen;
                pub_step_       = step_gen;
                pub_prefetch_   = prefetch_gen;
                cudaMemcpyAttributes attributes{};
                attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
                attributes.flags          = 0;
                std::size_t attribute_index = 0;
                cudaError_t submitted       = cudaErrorInvalidValue;
                if (delta) {
                    // Two slices per moved expert: its offset row and its mask word.
                    std::vector<void*>       dsts;
                    std::vector<const void*> srcs;
                    std::vector<std::size_t> sizes;
                    dsts.reserve(moved.size() * 2);
                    srcs.reserve(moved.size() * 2);
                    sizes.reserve(moved.size() * 2);
                    for (const auto expert : moved) {
                        dsts.push_back(static_cast<char*>(offsets_device_->p) +
                                       static_cast<std::size_t>(expert) * sizeof(std::uint64_t));
                        srcs.push_back(offsets_.data() + expert);
                        sizes.push_back(sizeof(std::uint64_t));
                        dsts.push_back(static_cast<char*>(mask_device_->p) +
                                       static_cast<std::size_t>(expert) * sizeof(std::int32_t));
                        srcs.push_back(mask_host_.data() + expert);
                        sizes.push_back(sizeof(std::int32_t));
                    }
                    submitted = cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(),
                                                     sizes.size(), &attributes, &attribute_index, 1,
                                                     stream);
                    if (submitted == cudaSuccess) {
                        publish_bytes_ += delta_bytes;
                        publish_bytes_max_ = std::max(publish_bytes_max_, delta_bytes);
                        if (moved.size() <= 4) {
                            publish_bytes_step_max_ =
                                std::max(publish_bytes_step_max_, delta_bytes);
                        }
                    }
                }
                if (!delta || submitted != cudaSuccess) {
                    void* const       destinations[2] = {offsets_device_->p, mask_device_->p};
                    const void* const sources[2]      = {offsets_.data(), mask_host_.data()};
                    const std::size_t sizes[2]        = {table_bytes, mask_bytes};
                    submitted = cudaMemcpyBatchAsync(
                        destinations, sources, sizes, 2, &attributes, &attribute_index, 1, stream);
                    // A driver that rejects the batch still has to get the tables across: fall back
                    // to the two copies rather than leaving the layer with a stale resident set.
                    if (submitted != cudaSuccess) {
                        (void)cudaMemcpyAsync(offsets_device_->p, offsets_.data(), table_bytes,
                                              cudaMemcpyHostToDevice, stream);
                        (void)cudaMemcpyAsync(mask_device_->p, mask_host_.data(), mask_bytes,
                                              cudaMemcpyHostToDevice, stream);
                    }
                    publish_bytes_ += static_cast<std::uint64_t>(table_bytes + mask_bytes);
                    publish_bytes_max_ = std::max(publish_bytes_max_,
                                                  static_cast<std::uint64_t>(table_bytes + mask_bytes));
                    if (moved.size() <= 4) {
                        publish_bytes_step_max_ = std::max(
                            publish_bytes_step_max_,
                            static_cast<std::uint64_t>(table_bytes + mask_bytes));
                    }
                }
                // A3b: what this bank's Device tables now hold, as generations that can be compared.
                // Only a submission the driver accepted counts - "we tried to publish" is exactly the
                // state that used to look fine and compute with the previous layer's addresses.
                if (submitted == cudaSuccess) {
                    table_step_     = step_gen;
                    table_hot_      = hot_gen;
                    table_prefetch_ = prefetch_gen;
                } else {
                    table_step_ = table_hot_ = table_prefetch_ = kTableUnknown;
                }
                ++publish_calls_;
            }
            if (moved.empty()) { ++gh1_static_calls_; }
            probe_moved = moved.size();
            probe_ins   = entered_this_step;
            entered_ += entered_this_step;
            ++entered_steps_;
            for (const auto expert : moved) {
                const auto slot = cache_->expert_slot(bank_, expert);
                prev_slot_[expert] = slot ? static_cast<std::int32_t>(*slot) : -1;
            }
        }
    }
    const auto layout_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - layout_begin).count();
    required_.clear();
    for (std::size_t i = 0; i < values; ++i) {
        const auto expert = route_ids[i];
        required_.push_back(keys_[static_cast<std::size_t>(expert)]);
    }
    std::sort(required_.begin(), required_.end(), [](const ExpertCacheKey& a, const ExpertCacheKey& b) {
        return a.expert < b.expert;
    });
    required_.erase(std::unique(required_.begin(), required_.end(),
                                [](const ExpertCacheKey& a, const ExpertCacheKey& b) {
                                    return a.expert == b.expert;
                                }),
                    required_.end());
    // port (sm_89): fill the host tier before the Device asks for these experts.
    //
    // The warm tier existed and nothing filled it. HostExpertResidency::ensure() has been able to read
    // a slot image out of the artifact since it was written, but only the whole-layer CPU fallback ever
    // called it - so in the split path the arena stayed empty and every Device upload read the pageable
    // artifact mapping instead of a pinned slot. That is the "arena = 0.0 experts per layer" the plan
    // measured (1x.6 O5, root cause 2). This is the consumer, not an optimisation: without it the tier
    // is unreachable from the path that needs it.
    //
    // Off by default, per v6's rule for a new mechanism, so the current behaviour stays the reference
    // this switch is measured against.
    // port (sm_89): 2026-10-31 - NINFER_HOST_TIER is verified on the prefill/scoring path (PPL
    // 7.785666 bit-identical, arena hit 19.9%) and NOT on decode: a decode request with the switch on
    // fails to return (no exception in the log, the connection is reset). The tier is therefore
    // reachable only from the path that was measured; turning it on for serving needs that fixed
    // first, and nothing else in the tree depends on it.
    // port (sm_89): Q20 second leg needs this fill to have happened: the Device can only read what
    // is already in the tier, and `host_direct` deliberately never fills on demand inside a layer.
    if (host_tier && host_ != nullptr) {
        // port (sm_89): D4's numerator. An expert that was served from a slot the read-ahead filled
        // is the only thing that makes a prefetch "useful", and the tier is the only place that
        // knows it: it marks the slots it fills and reports the hits here. Reported in bytes so the
        // gate compares against what the queue asked for.
        if (const auto hits = host_->take_prefetch_hits(); hits != 0) {
            ExpertPrefetchQueue::instance().note_used(static_cast<std::size_t>(hits) *
                                                      host_->slot_bytes());
        }
        // Q69: a warmed expert that a routed fill later loaded is the same evidence as a filled
        // slot that was routed to - it is the read-ahead having paid. Reported in bytes so the gate
        // compares like with like.
        if (const auto warm_hits = host_->take_warm_hits(); warm_hits != 0) {
            ExpertPrefetchQueue::instance().note_used(static_cast<std::size_t>(warm_hits) *
                                                      host_->slot_bytes());
        }
        if (timing) {
            static std::uint64_t shown = 0;
            const auto& filled = host_->stats();
            if (++shown % 40 == 0) {
                const auto total = filled.hits + filled.misses;
                std::fprintf(stderr, "[host] arena hits=%llu misses=%llu evict=%llu hit=%.1f%%\n",
                             static_cast<unsigned long long>(filled.hits),
                             static_cast<unsigned long long>(filled.misses),
                             static_cast<unsigned long long>(filled.evictions),
                             total != 0 ? 100.0 * static_cast<double>(filled.hits) /
                                              static_cast<double>(total)
                                        : 0.0);
                std::fflush(stderr);
            }
        }
    }
    const auto upload_begin = std::chrono::steady_clock::now();
    cache_->prepare(required_, stream);
    if (freeze_trace()) {
        std::fprintf(stderr, "[fz] bank=%u prepared\n", bank_);
        std::fflush(stderr);
    }
    try {
        cache_->wait_for_progress(stream);
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string("pool: ") + error.what());
    }
    if (freeze_trace()) {
        std::fprintf(stderr, "[fz] bank=%u uploaded\n", bank_);
        std::fflush(stderr);
    }
    if (timing) {
        static std::uint64_t calls = 0;
        static double route_total = 0.0, layout_total = 0.0;
        route_total += route_ms;
        layout_total += layout_ms;
        if (++calls % 40 == 0) {
            std::fprintf(stderr,
                         "[split] per-layer route=%.3fms layout=%.3fms (of 40 layers)\n",
                         route_total / 40.0, layout_total / 40.0);
            route_total = 0.0;
            layout_total = 0.0;
        }
        // How many distinct experts this bank has ever routed to: the number a pool has to hold
        // before a whole step can run on the Device with no host tier at all.
        std::size_t touched = 0;
        for (const auto score : hot_score_) {
            if (score > 0.0F) { ++touched; }
        }
        static std::size_t touched_max = 0;
        if (touched > touched_max) {
            touched_max = touched;
            std::fprintf(stderr, "[reach] bank=%u distinct=%zu of %u\n", bank_, touched, count_);
        }
    }
    if (std::getenv("NINFER_EXPERT_TIMING")) {
        static std::uint64_t calls = 0;
        static double seconds = 0.0;
        seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - upload_begin).count();
        if (++calls % 40 == 0) {
            std::fprintf(stderr, "[experts] calls=%llu upload=%.3fs avg=%.2fms hits=%llu\n",
                         (unsigned long long)calls, seconds, seconds * 1000.0 / 40.0,
                         (unsigned long long)cache_->hits());
            seconds = 0.0;
        }
    }
    // port (sm_89): the bank's window can move between layers, so the plane bases are
    // rebased onto the window that is resident now. A fixed base would address window 0.
    rebase_window();
    last_cost_.route_ms  = route_ms;
    last_cost_.layout_ms = layout_ms;
    last_cost_.upload_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - upload_begin)
            .count();
    // The honest miss count for the path controller: the mask cannot answer this question, because
    // it is written from the same decision that places the step's routes.
    last_cost_.uploaded = cache_->last_uploaded();
    // An upload only counts as host intervention when bytes actually moved. `prepare()` is called on
    // every split layer, but a layer whose required experts are already resident submits nothing -
    // counting the call instead of the transfer would report the split path as host-bound even when
    // it was not, which is the same mistake F3 made in the other direction.
    host_upload = last_cost_.uploaded != 0;
    // port (sm_89): Q20 - a route the second leg resolves is not a host-bound route.
    //
    // `capt_free` used to fail a layer whenever the ids had been read back, and the ids are read
    // back on nearly every split layer just to maintain the tables. That made the rate track the
    // bookkeeping rather than the data path: a layer whose every expert came from the pool or the
    // pinned tier still counted as host-bound, even though nothing was moved and nothing waited.
    //
    // The read-back only counts against the layer when it is there to find a route the Device
    // cannot address. If every route already has an address, the bookkeeping copy is not a data
    // dependency - and the number this rate exists to describe is whether the layer needed the host
    // to compute.
    std::uint32_t host_resolved = 0, pool_resolved = 0;
    for (std::size_t i = 0; i < values; ++i) {
        const auto expert = static_cast<std::uint32_t>(route_ids[i]);
        if (expert < host_flags_.size() && host_flags_[expert] != 0) {
            ++host_resolved;
        } else if (cache_->expert_slot(bank_, expert).has_value()) {
            ++pool_resolved;
        }
    }
    last_host_routes_ = host_resolved;
    // C5/D4: the read-ahead's numerator on the SPLIT path. The tier is the only place that knows a
    // route was served from a slot the read-ahead filled, and until now only the fused path asked it -
    // so on a gate/up/down artifact the queue could fetch bytes nobody ever reported using, which
    // reads as a mechanism that does not pay when it is really a counter that was never read.
    if (host_ != nullptr) {
        if (const auto hits = host_->take_prefetch_hits(); hits != 0) {
            ExpertPrefetchQueue::instance().note_used(static_cast<std::size_t>(hits) *
                                                       host_->slot_bytes());
        }
        if (const auto warm_hits = host_->take_warm_hits(); warm_hits != 0) {
            ExpertPrefetchQueue::instance().note_used(static_cast<std::size_t>(warm_hits) *
                                                       host_->slot_bytes());
        }
    }
    const bool all_addressable = host_resolved + pool_resolved >= values;
    const bool host_bound = host_upload || host_publish || (host_fetch && !all_addressable);
    if (!host_bound) { ++gh1_capt_free_calls_; }
    // GC2's precondition is the same question plus coverage: every route has an address AND none
    // is outside the two tiers. A layer that satisfies both would replay a table it was captured
    // with.
    if (!host_bound && all_addressable) { ++gh1_capt_ready_calls_; }
    // The attribution of the other case, counted at the decision so a rate never has to be read
    // as a cause (F3's mistake, in the other direction). One layer can fire several reasons, which
    // is exactly the point: "not ready" is not one problem.
    if (host_bound || !all_addressable) {
        if (!all_addressable) { ++gh1_captfail_unaddressable_; }
        if (host_publish) { ++gh1_captfail_publish_; }
        if (host_upload) { ++gh1_captfail_upload_; }
    }
    // P6/step 5: which structure served each route. A hit is a joint outcome of the hot region, the
    // spill ring, the prediction region and the host leg, and until this existed the panel could only
    // say "the pool served it" - which is exactly the number a controller cannot act on. Classified
    // by the slot the route resolves to, so the total always equals `values`.
    {
        // `wanted` is built hot-first, so the first `hot_cap` entries are the epoch's hot set; the
        // rest are the ring. That is the distinction the controller needs (a ring hit is cheap but
        // not repeatable, a hot hit is the set it chose), and it is not recoverable from the slot.
        const auto hot_end = std::min<std::size_t>(resident_order_.size(), hot_cap_);
        std::vector<char> is_hot(count_, 0);
        for (std::size_t i = 0; i < hot_end; ++i) {
            const auto expert = resident_order_[i];
            if (expert < count_) { is_hot[expert] = 1; }
        }
        for (std::size_t i = 0; i < values; ++i) {
            const auto expert = static_cast<std::uint32_t>(route_ids[i]);
            if (expert >= count_) { continue; }
            const auto slot = cache_->expert_slot(bank_, expert);
            if (slot && cache_->prefetch_slot(*slot)) {
                ++route_prefetch_;
            } else if (slot) {
                // Hot and ring share one slot space; the difference is whether the expert is in the
                // set the epoch chose, which is what the warm/cold decision below is about.
                if (is_hot[expert]) { ++route_hot_; } else { ++route_ring_; }
            } else if (host_ != nullptr && expert < host_flags_.size() && host_flags_[expert] != 0) {
                ++route_host_;
            } else {
                ++route_cpu_;
            }
        }
    }
    if (bank_ == 0 && host_direct && std::getenv("NINFER_EXPERT_TIMING")) {
        static std::uint64_t layers2 = 0, host_routes = 0, pool_routes = 0;
        host_routes += host_resolved;
        pool_routes += pool_resolved;
        if (++layers2 % 8 == 0) {
            std::fprintf(stderr, "[hostd] routes/layer: pool=%.2f host=%.2f of %.0f\n",
                         static_cast<double>(pool_routes) / 8.0,
                         static_cast<double>(host_routes) / 8.0, static_cast<double>(values));
            std::fflush(stderr);
            host_routes = 0;
            pool_routes = 0;
        }
    }
    static_probe(bank_, tokens, false, probe_absent, probe_unready, probe_moved, probe_ins,
                 last_cost_.uploaded, !host_bound);
    if (freeze_mode_ && bank_ == 0 && (++freeze_report_layers_ % 10) == 0) {
        const auto& host_stats = host_ != nullptr ? host_->stats() : HostExpertResidency::Stats{};
        std::fprintf(stderr,
                     "[freeze] summary layers=%llu frozen=%d members=%u present=%u moves=%llu "
                     "route_cpu=%llu static_drops=%llu evictions=%llu pool_missing=%u\n",
                     static_cast<unsigned long long>(freeze_report_layers_), int(frozen_),
                     freeze_members_, freeze_present_,
                     static_cast<unsigned long long>(freeze_membership_moves_),
                     static_cast<unsigned long long>(route_cpu_),
                     static_cast<unsigned long long>(host_stats.static_drops),
                     static_cast<unsigned long long>(host_stats.evictions),
                     freeze_pool_missing_);
        std::fflush(stderr);
    }
    // port (sm_89): D1 read-ahead, in three pieces that are now separate on purpose.
    //
    // 1. The selection (below) is unchanged: NINFER_PREFETCH_PER_LAYER (default 8) experts, taken
    //    from the top of the decayed route score among the ones the pool does not hold. Batching
    //    matters twice - a batch of pinned slices submits far cheaper per slice than the same slices
    //    one at a time, and the arena fill is a memcpy per expert either way.
    // 2. NINFER_PREFETCH_SYNC keeps the original fill on the calling thread. It is the A/B
    //    reference for the queued path and nothing else, so it is a separate switch rather than
    //    the same one wearing two meanings.
    // 3. NINFER_PREFETCH offers the same batch to the process-wide queue. The queue takes the bank
    //    and the expert list with the job and holds the residency weakly, so forty banks share one
    //    worker and a job cannot outlive the model it fills. The job body is still a placeholder:
    //    it counts itself and stages nothing (that is item 4), so turning this on today changes
    //    nothing about what is resident.
    static const bool prefetch_sync  = std::getenv("NINFER_PREFETCH_SYNC") != nullptr;
    // NINFER_PREFETCH is the STARTUP half of the switch only: it decides whether a residency
    // attaches the process-wide queue at all (model_instance.cpp retain), because the workers'
    // lifetime is bound to the residency. The RUNTIME half is the queue's gate, which the
    // controller owns through the parameter bus (the prefetch_on executor pushes set_enabled
    // down into every worker). Asking the gate here means a closed read-ahead stops the offer
    // itself - the selection and the sort below cost host time a closed gate has no use for -
    // and a run without NINFER_PREFETCH still sees users()==0, hence gate_open()==false, hence
    // the same zero offers as before (the panel's usefulness simply never starts measuring).
    const bool prefetch_queue = ExpertPrefetchQueue::instance().gate_open();
    if ((prefetch_sync || prefetch_queue) && host_ != nullptr) {
        // Q30/P6: how many experts one read-ahead asks for.
        //
        // It used to be a fixed eight, which is neither the model's top-k nor the problem's size: the
        // prediction already filters to "predicted AND not in the pool", and on this artifact that
        // delta is about one expert a layer - so seven of every eight were work the filter threw away,
        // and a job is billed by its bytes (8 x 1.92 MiB = 15.4 MiB against a 256 MiB budget). The
        // count now IS the delta, with the model's own top-k as the ceiling; NINFER_PREFETCH_PER_LAYER
        // stays as an override (0 = derive), so an A/B can still pin it.
        static const std::uint32_t per_layer_override = [] {
            const char* value = std::getenv("NINFER_PREFETCH_PER_LAYER");
            return value != nullptr && *value != '\0'
                       ? static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10))
                       : 0U;
        }();
        std::uint32_t predicted_absent = 0;
        {
            std::vector<std::uint32_t> distinct;
            distinct.reserve(predicted_.size());
            for (const auto expert : predicted_) {
                if (expert < 0 || static_cast<std::uint32_t>(expert) >= count_) { continue; }
                const auto id = static_cast<std::uint32_t>(expert);
                if (cache_->expert_slot(bank_, id)) { continue; } // already resident: not a prediction
                if (std::find(distinct.begin(), distinct.end(), id) != distinct.end()) { continue; }
                distinct.push_back(id);
            }
            predicted_absent = static_cast<std::uint32_t>(distinct.size());
        }
        const std::size_t per_layer =
            per_layer_override != 0
                ? static_cast<std::size_t>(per_layer_override)
                : static_cast<std::size_t>(std::max(1U, predicted_absent));
        prefetch_target_ += predicted_absent;
        // Partial selection: recency first, then the decayed score. The score ranks experts this
        // bank has been using, which is exactly the set the pool already holds, so it can only ever
        // fill the tail; the head is what this bank routed on its previous step, which is what the
        // conversation is most likely to ask for again.
        std::vector<std::pair<float, std::int32_t>> top;
        top.reserve(per_layer);
        auto take = [&](std::int32_t expert) {
            if (expert < 0 || static_cast<std::uint32_t>(expert) >= count_) { return; }
            if (top.size() == per_layer) { return; }
            if (cache_->expert_slot(bank_, expert)) { return; }
            // P1: an expert the host tier already holds is not a prediction either. Without this the
            // read-ahead copied the static set into the read-ahead stripe, one slot at a time, and
            // evicted the previous prediction to do it.
            if (host_->holds(bank_, static_cast<std::uint32_t>(expert))) { return; }
            for (const auto& taken : top) {
                if (taken.second == expert) { return; }
            }
            top.emplace_back(hot_score_[expert], expert);
        };
        const std::size_t before = top.size();
        // Q68b: the prediction comes first now. It is what this layer's router would pick from the
        // previous layer's residual stream - measured at 74% of the real routes, 25x the chance
        // level - so it is a better target than either of the two history signals below.
        const std::size_t predicted_before = top.size();
        for (const auto expert : predicted_) { take(expert); }
        prefetch_predicted_ += top.size() - predicted_before;
        // Then the previous step of this same bank (7% of the time it survives residency), and the
        // decayed score only fills whatever is left.
        for (const auto expert : prev_routes_) { take(expert); }
        // How many of the wanted experts came from the previous step rather than from the score.
        // This is the kill-switch for the idea: if it is zero the recency signal never applies (the
        // previous step's routes are still resident) and the variant is the old one wearing a new
        // comment.
        prefetch_recent_ += top.size() - before;
        for (std::uint32_t expert = 0; expert < count_; ++expert) {
            const float score = hot_score_[expert];
            if (score <= 0.0F) { continue; }
            if (cache_->expert_slot(bank_, expert)) { continue; }
            // Already taken by recency in the head of the list.
            bool taken = false;
            for (const auto& entry : top) {
                if (entry.second == static_cast<std::int32_t>(expert)) { taken = true; break; }
            }
            if (taken) { continue; }
            if (top.size() == per_layer && score <= top.back().first) { continue; }
            const std::pair<float, std::int32_t> entry{score, static_cast<std::int32_t>(expert)};
            const auto at = std::lower_bound(top.begin(), top.end(), entry,
                                             [](const auto& l, const auto& r) { return l.first > r.first; });
            top.insert(at, entry);
            if (top.size() > per_layer) { top.pop_back(); }
        }
        if (!top.empty()) {
            std::vector<std::int32_t> wanted(top.size());
            for (std::size_t i = 0; i < top.size(); ++i) { wanted[i] = top[i].second; }
            if (prefetch_sync) {
                // Q69: the synchronous path is the A/B reference for the queued one, so it takes the
                // same target the queued path would - otherwise the two arms would differ by two
                // things instead of one.
                static const bool warm_mode = [] {
                    const char* value = std::getenv("NINFER_PREFETCH_MODE");
                    return value != nullptr && std::strcmp(value, "warm") == 0;
                }();
                if (warm_mode) {
                    host_->warm(ExpertBankId{bank_}, std::span<const std::int32_t>(wanted));
                } else {
                    // The queued path fills the read-ahead stripe (expert_prefetch.cpp calls
                    // fill_prefetch). The reference path must do the same, or the two arms differ by
                    // where the bytes land as well as by who reads them - the old loop called
                    // ensure(), which fills the dynamic stripe and evicts the working set.
                    host_->fill_prefetch(ExpertBankId{bank_},
                                         std::span<const std::int32_t>(wanted));
                }
            } else {
                // D3: the deadline is measured in layer batches, so the clock is one tick per offer
                // and it is process-wide - the forty banks are forty layers of the same step, and a
                // per-bank clock would let one bank's backlog expire against another's progress.
                // The window is NINFER_PREFETCH_WINDOW batches (default 40, one token's worth on
                // this model) and advance() is what drops the jobs that could not start inside it.
                // C5: the same clock the prediction's fill uses (prefetch_layer_tick), so the two
                // offers share one window instead of each owning half of it.
                auto& queue     = ExpertPrefetchQueue::instance();
                const auto tick = prefetch_layer_tick().fetch_add(1) + 1;
                const auto window = prefetch_window();
                if (window != 0 && (tick % window) == 0) { queue.advance(tick); }
                // One job per expert, not one job per bank. A job is a disk read and the buffered
                // reader serialises concurrent reads on one handle by design (a single file pointer),
                // so the only parallelism available is across jobs - and one bank's two experts were
                // a single job, i.e. serial by construction. Per expert, the workers split them.
                //
                // The budget is charged the same total bytes; only the granularity changed, so a job
                // that is refused now costs one expert's worth instead of the whole batch's.
                for (const auto expert : wanted) {
                    const std::int32_t one[1] = {expert};
                    (void)queue.enqueue(host_weak_, bank_, std::span<const std::int32_t>(one, 1),
                                        static_cast<std::size_t>(host_->slot_bytes()), tick + window,
                                        tick);
                }
            }
            prefetch_calls_ += wanted.size();
            if ((++prefetch_batches_ % 40) == 0) {
                const auto& st    = host_->stats();
                const auto total  = st.hits + st.misses;
                const auto queued = ExpertPrefetchQueue::instance().stats();
                std::fprintf(stderr,
                             "[pf] experts=%llu pred=%llu recent=%llu batches=%llu hit=%.1f%% evict=%llu | q enq=%llu "
                             "drop(budget=%llu off=%llu deadline=%llu backlog=%llu stopped=%llu) "
                             "ran=%llu expired=%llu pending=%llu/%llu useful=%.2f | "
                             "warm=%llu MiB memcpy_hits=%llu skip=%llu\n",
                             (unsigned long long)prefetch_calls_,
                             (unsigned long long)prefetch_predicted_,
                             (unsigned long long)prefetch_recent_,
                             (unsigned long long)prefetch_batches_,
                             total ? 100.0 * (double)st.hits / (double)total : 0.0,
                             (unsigned long long)st.evictions,
                             (unsigned long long)queued.queue.enqueued,
                             (unsigned long long)queued.queue.dropped_budget,
                             (unsigned long long)queued.queue.dropped_gate,
                             (unsigned long long)queued.queue.dropped_deadline,
                             (unsigned long long)queued.queue.dropped_backlog,
                             (unsigned long long)queued.queue.dropped_stopped,
                             (unsigned long long)queued.ran, (unsigned long long)queued.expired,
                             (unsigned long long)queued.pending,
                             (unsigned long long)queued.max_pending,
                             queued.queue.usefulness(),
                             (unsigned long long)(st.warmed_bytes >> 20),
                             (unsigned long long)st.memcpy_hits,
                             (unsigned long long)st.warm_skips);
                std::fflush(stderr);
            }
        }
    }
    // Record this step's routes for the next step's read-ahead. Small batches only: a prefill chunk
    // touches most of the bank, and a read-ahead that copied it would be copying the prefill.
    if (tokens <= kHotSetUpdateTokens) {
        prev_routes_.assign(route_ids, route_ids + values);
        std::sort(prev_routes_.begin(), prev_routes_.end());
        prev_routes_.erase(std::unique(prev_routes_.begin(), prev_routes_.end()),
                           prev_routes_.end());
        // Q68a: the same set, published for the neighbouring layer to be measured against. The
        // probe reads bank-1's copy, which was written earlier in this step because the layers run
        // in order, so it compares this layer with the layer below it on the same token.
        if (std::getenv("NINFER_LAYER_PROBE") != nullptr) {
            auto& trace = layer_trace();
            trace.sets[bank_]  = prev_routes_;
            trace.valid[bank_] = true;
            if (bank_ > 0 && trace.valid[bank_ - 1] && !prev_routes_.empty()) {
                const auto& below = trace.sets[bank_ - 1];
                std::size_t shared = 0, not_resident = 0;
                for (const auto expert : prev_routes_) {
                    if (!std::binary_search(below.begin(), below.end(), expert)) { continue; }
                    ++shared;
                    if (!cache_->expert_slot(bank_, expert)) { ++not_resident; }
                }
                static double recall_sum = 0.0, absent_sum = 0.0;
                static std::uint64_t probes = 0;
                recall_sum += static_cast<double>(shared) / static_cast<double>(prev_routes_.size());
                absent_sum += shared == 0 ? 0.0
                                          : static_cast<double>(not_resident) /
                                                static_cast<double>(shared);
                if (++probes % 40 == 0) {
                    std::fprintf(stderr, "[probe] recall=%.3f absent=%.3f layers=%llu\n",
                                 recall_sum / 40.0, absent_sum / 40.0,
                                 static_cast<unsigned long long>(probes));
                    std::fflush(stderr);
                    recall_sum = absent_sum = 0.0;
                }
            }
        }
    }
    // port (sm_89): A3b, on the per-bank generation this time. A layer is about to hand these two
    // tables to kernels that read them as addresses; if what this bank's Device tables hold is not
    // what this bank's set now is, a publish did not land and the kernels would compute with another
    // set's offsets - a silent wrong token, the one outcome the plan never accepts.
    //
    // The first version compared against layout_generation() and fired on the default path with
    // `1 vs 40`: the pool's counter had moved 39 times for other banks. That is why the number had to
    // become per-bank before the check could be one.
    if (std::getenv("NINFER_TABLE_STALE") != nullptr) {
        table_step_ = table_hot_ = table_prefetch_ = kTableUnknown;
    }
    if (table_step_ != cache_->step_generation(bank_) || table_hot_ != cache_->hot_generation(bank_) ||
        table_prefetch_ != cache_->prefetch_generation(bank_)) {
        throw std::runtime_error("expert table is stale (bank " + std::to_string(bank_) +
                                 "): device holds hot/step/pred " + std::to_string(table_hot_) + "/" +
                                 std::to_string(table_step_) + "/" +
                                 std::to_string(table_prefetch_) + ", cache is on " +
                                 std::to_string(cache_->hot_generation(bank_)) + "/" +
                                 std::to_string(cache_->step_generation(bank_)) + "/" +
                                 std::to_string(cache_->prefetch_generation(bank_)));
    }
    // GH3's half of the same check, and the one that would have caught the NINFER_HOST_DIRECT NaN on
    // the first layer it happened instead of forty layers later: if the placement this layer derived
    // did not reach the Device, the GEMM is about to address slots that no longer hold those experts.
    // Better a refusal than another expert's bytes.
    if (placement_dirty_) {
        throw std::runtime_error("host placement is stale (bank " + std::to_string(bank_) +
                                 "): the Device is running on an earlier snapshot of the tier");
    }
    // port (sm_89): D1-D4 prefetch, in the only form that is safe today.
    //
    // The queue in core/prefetch.h is shaped for a worker thread. HostExpertResidency::ensure()
    // mutates the arena's slot tables and its expert offsets, and the main thread calls it every
    // layer, so running it on a worker without a lock would be a data race - and a race in the tier is
    // silent corruption, not a slowdown. The mechanism is therefore wired here, on the calling thread,
    // with what D1 asks for: the experts the pool does NOT hold but the decayed route score says are
    // hottest get read into pinned RAM now, so that when a later step routes to one, the Device upload
    // is a pinned copy instead of a pageable artifact read.
    //
    // Budget (D2) is the slot cap; the deadline is one step (anything not used by then is dropped by
    // the goal's own wording); the usefulness gate (D4) is the arena hit rate, which [host] reports -
    // if it does not clear 10%, the switch stays off.
    // port (sm_89): D1-D4 prefetch is NOT wired here, and this is the reason.
    //
    // A first version read the hottest non-resident expert into the warm tier at the end of each
    // layer. It crashed the server with a reset connection, and the mechanism is clear enough: the
    // Device uploads are asynchronous on the stream and read their bytes out of the arena through
    // arena_image(); the tier's claim() evicts coldest-first, so a read-ahead issued after
    // wait_for_progress() can free the very slot a copy still in flight is reading. That is not a
    // slowdown, it is reading another expert's bytes.
    //
    // Wiring it therefore needs one of two things first, and neither exists yet: a staging pool the
    // uploader owns (so eviction cannot touch what is in flight), or a lock plus a completion signal
    // between the tier and the uploader. Recorded as the prerequisite for D1 rather than worked around
    // - the switch does not exist, so nothing here can be turned on by accident.

    // GH1's no_miss, defined so it can be false: the pool already held every route this step.
    if (last_cost_.uploaded == 0) { ++gh1_no_miss_calls_; }
    // The uploads were queued on the consumer stream, so the kernels that read them are already
    // ordered after the copies; only the artifact path had to wait, and it did so inside
    // progress(). A device-wide synchronise here used to cost most of the layer's time.
    return true;
}

std::uint32_t* ExpertStream::missed_routes_device() {
    if (!miss_count_checked_) {
        miss_count_checked_ = true;
        if (std::getenv("NINFER_O1_MISS_COUNT") != nullptr) {
            // One word, allocated on first use: the Device context is certainly up by the time a
            // layer asks for it, and a run that never enables the counter never pays for it.
            miss_count_      = std::make_unique<PinnedHostBuffer>(sizeof(std::uint32_t), true);
            miss_count_host_ = static_cast<std::uint32_t*>(miss_count_->data());
            *miss_count_host_ = 0;
            const cudaError_t err =
                cudaHostGetDevicePointer(reinterpret_cast<void**>(&miss_count_device_),
                                         miss_count_->data(), 0);
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("cudaHostGetDevicePointer failed: ") +
                                         cudaGetErrorString(err));
            }
            std::fprintf(stderr, "[o1] miss counter on (mapped) bank=%u\n", bank_);
            std::fflush(stderr);
        }
    }
    return miss_count_device_;
}

bool ExpertStream::miss_count_enabled() { return missed_routes_device() != nullptr; }

std::uint32_t* ExpertStream::route_coverage_device() {
    if (!coverage_word_checked_) {
        coverage_word_checked_ = true;
        // Same shape as the miss counter: one mapped word, built on first use, so a run that never
        // asks never pays for it.
        coverage_word_ = std::make_unique<PinnedHostBuffer>(sizeof(std::uint32_t), true);
        coverage_word_host_ = static_cast<std::uint32_t*>(coverage_word_->data());
        *coverage_word_host_ = 0;
        const cudaError_t err =
            cudaHostGetDevicePointer(reinterpret_cast<void**>(&coverage_word_device_),
                                     coverage_word_->data(), 0);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaHostGetDevicePointer failed: ") +
                                     cudaGetErrorString(err));
        }
        std::fprintf(stderr, "[o1v] coverage word on (mapped) bank=%u\n", bank_);
        std::fflush(stderr);
    }
    return coverage_word_device_;
}

std::uint32_t ExpertStream::route_coverage_read() {
    if (coverage_word_host_ == nullptr) { return 0; }
    return *static_cast<volatile std::uint32_t*>(coverage_word_host_);
}

std::uint32_t ExpertStream::missed_routes_total() {
    if (miss_count_host_ == nullptr) { return 0; }
    // Volatile, because the word is written on the other side of the bus: a cached read would keep
    // returning whatever the host happened to see first.
    return *static_cast<volatile std::uint32_t*>(miss_count_host_);
}

void ExpertStream::observe_missed(std::uint32_t reading, std::uint64_t host_layer, bool decode) {
    // The counter is never reset, so a reading is a running total. Its delta against the last reading
    // is the layer that produced the previous call: the read sits right after the fetch that
    // synchronised, and that fetch only covers what was enqueued before this layer's Device half.
    // Deltas rather than totals, because one divergence would otherwise offset every later reading.
    const std::uint32_t delta = reading - miss_last_reading_;
    miss_last_reading_        = reading;
    miss_tally_.device += delta;
    miss_tally_.host += host_layer;
    if (decode) {
        if (delta != miss_prev_host_) { ++miss_tally_.mismatch; }
        ++miss_tally_.compared;
    } else {
        if (delta != miss_prev_host_) { ++miss_tally_.prefill_mismatch; }
        ++miss_tally_.prefill_compared;
    }
    miss_prev_host_ = host_layer;
    if (std::getenv("NINFER_O1_DEBUG") != nullptr) {
        static std::uint64_t shown = 0;
        if (bank_ == 0 && shown++ < 400) {
            std::fprintf(stderr, "[o1d] n=%llu delta=%u host_layer=%llu prev=%llu\n",
                         static_cast<unsigned long long>(shown), delta,
                         static_cast<unsigned long long>(host_layer),
                         static_cast<unsigned long long>(miss_prev_host_));
            std::fflush(stderr);
        }
    }
}

namespace {
// O1's step verdict storage. One mapped word for the whole model, because a replay asks whether ANY
// layer of the step dropped a route, not which one - and it has to be read where the step already
// synchronises, the sampling wait in the decode batch. Built on first use and only under the switch.
struct StepGate {
    std::unique_ptr<PinnedHostBuffer> buffer;
    std::uint32_t* host   = nullptr;
    std::uint32_t* device = nullptr;
    bool enabled          = false;
    bool built            = false;
};

StepGate& step_gate() {
    static StepGate gate;
    if (!gate.built) {
        gate.built = true;
        // The verdict is not optional bookkeeping: both ways of running a step without a read-back
        // need it to notice a wrong prediction, so asking for either turns it on.
        gate.enabled = std::getenv("NINFER_O1_STEP_GATE") != nullptr ||
                       std::getenv("NINFER_O1_SKIP") != nullptr ||
                       std::getenv("NINFER_O1_DEVICE") != nullptr;
        if (gate.enabled) {
            gate.buffer = std::make_unique<PinnedHostBuffer>(sizeof(std::uint32_t), true);
            gate.host   = static_cast<std::uint32_t*>(gate.buffer->data());
            *gate.host  = 0;
            const cudaError_t err =
                cudaHostGetDevicePointer(reinterpret_cast<void**>(&gate.device), gate.buffer->data(),
                                         0);
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("cudaHostGetDevicePointer failed: ") +
                                         cudaGetErrorString(err));
            }
            std::fprintf(stderr, "[o1t] step gate on (mapped %p)\n", static_cast<void*>(gate.device));
            std::fflush(stderr);
        }
    }
    return gate;
}
} // namespace

std::uint32_t* step_miss_device() { return step_gate().device; }

std::uint32_t step_miss_read() {
    StepGate& gate = step_gate();
    if (!gate.enabled) { return 0; }
    // Volatile: the word is written from the other side of the bus.
    return *static_cast<volatile std::uint32_t*>(gate.host);
}

void step_miss_reset() {
    StepGate& gate = step_gate();
    if (!gate.enabled) { return; }
    *gate.host = 0;
}

namespace {
std::atomic<bool>& skip_armed() {
    static std::atomic<bool> flag{false};
    return flag;
}
std::atomic<bool>& force_fetch() {
    static std::atomic<bool> flag{false};
    return flag;
}
} // namespace

bool o1_skip_armed() {
    // Both switches are needed: the env var says the feature exists, the arm says decode has
    // promised to replay this step if it comes back dirty.
    return skip_armed().load(std::memory_order_relaxed) &&
           !force_fetch().load(std::memory_order_relaxed);
}

void o1_arm_skip(bool armed) { skip_armed().store(armed, std::memory_order_relaxed); }

bool o1_force_fetch() { return force_fetch().load(std::memory_order_relaxed); }

void o1_force_fetch(bool force) { force_fetch().store(force, std::memory_order_relaxed); }

bool o1_skip_requested() {
    static const bool requested = std::getenv("NINFER_O1_SKIP") != nullptr ||
                                  std::getenv("NINFER_O1_DEVICE") != nullptr;
    return requested;
}

} // namespace ninfer::models::qwen3_5::execution






