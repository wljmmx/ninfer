#include "models/qwen3_5/execution/expert_stream.h"
#include "models/qwen3_5/execution/expert_prefetch.h"
#include "models/qwen3_5/execution/expert_path_control.h"
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
    static std::map<std::uint32_t, const std::vector<float>*> registry;
    return registry;
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

} // namespace

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
        if (value != nullptr && *value != '\0') {
            return static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10));
        }
        // Four ensure() calls: one layer has decided nothing yet, and a single decode token routes
        // eight experts. A gate pins the number explicitly; a serving run that wants a converged set
        // should raise it.
        return std::uint32_t{4};
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

void ExpertStream::attach_host(std::shared_ptr<HostExpertResidency> host) noexcept {
    host_      = host.get();
    host_weak_ = std::move(host);
    // Q53: the second-chance band's admission compares this bank's score against the occupant it would
    // displace, so the tier needs the ranking. The vector is allocated once in the constructor, so
    // the pointer stays valid for the stream's life.
    if (host_ != nullptr) { host_->attach_scores(bank_, hot_score_.data(), count_); }
    // Q20/B: the static stripe must exist before the first dynamic fill - configuring it later moves
    // the dynamic bump pointer under a pointer that has already advanced, and the next eviction then
    // scans untouched slots and reports "nothing to evict" (the tier now refuses that ordering
    // outright). The quota is a process-wide decision, so the first stream installs it and the rest
    // are no-ops. The freeze route always installs it: its S stripe is the placement, not an option.
    if (host_ != nullptr && static_warm_ &&
        (freeze_mode_ || std::getenv("NINFER_WARM_STATIC_RESERVE") != nullptr)) {
        host_->configure_static_slots(static_quota());
    }
}

void ExpertStream::attach(ExpertCache& cache, std::uint32_t bank, const ExpertBankLayout& entry) {
    register_profile_dump(bank, &hot_score_);
    if (bank > 0xffffU || cache.capacity_bytes() < entry.entry_bytes) {
        throw std::invalid_argument("expert stream cannot attach to this cache");
    }
    cache_ = &cache;
    bank_  = bank;
    entry_ = entry;
    if (host_ != nullptr) { host_->attach_scores(bank_, hot_score_.data(), count_); }
    for (auto& key : keys_) { key.bank = bank; }
    

    const auto pool = cache.pool();
    const auto base = static_cast<const std::byte*>(pool.data);
    auto& gate_up_view = experts_.bank.gate_up;
    gate_up_view.codes = base + entry.codes.offset;
    gate_up_view.high  = entry.high.bytes ? base + entry.high.offset : nullptr;
    gate_up_view.scales = base + entry.scales.offset;
    gate_up_view.code_stride  = cache.slot_bytes();
    gate_up_view.high_stride  = entry.high.bytes ? cache.slot_bytes() : 0;
    gate_up_view.scale_stride = cache.slot_bytes();
    auto& down_view   = experts_.bank.down;
    down_view         = gate_up_view;
    down_view.codes   = base + entry.down_codes.offset;
    down_view.high    = entry.down_high.bytes ? base + entry.down_high.offset : nullptr;
    down_view.scales  = base + entry.down_scales.offset;
    down_view.code_stride  = cache.slot_bytes();
    down_view.high_stride  = entry.down_high.bytes ? cache.slot_bytes() : 0;
    down_view.scale_stride = cache.slot_bytes();
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
        offsets_[expert] = static_cast<std::uint64_t>(*slot) * cache.slot_bytes();
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

std::uint32_t ExpertStream::spill_slots(std::uint32_t capacity) const noexcept {
    // A fixed eight, not a measured p99. Under four concurrent lanes the measured p99 of a step's
    // misses is six at 163 slots, and sizing to that measured 63-68 tok/s against 82-86 for eight:
    // a region that changes size moves the resident set every step, which republishes the offset
    // table and re-uploads slots the previous size had just placed. Eight is also one full token's
    // routing, which is the granularity the tail actually arrives in.
    // The miss histogram is still sampled: it is reported as diagnostics, and it is what a future
    // per-generation (not per-step) decision would read.
    return std::min<std::uint32_t>(8U, std::max<std::uint32_t>(2U, capacity / 3));
}
bool ExpertStream::fully_resident() const noexcept {
    return cache_ != nullptr && cache_->resident_capacity() >= count_;
}

void ExpertStream::gh1_report() {
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
    std::fprintf(stderr,
                 "[gh1] bank=%u calls=%llu static=%.1f%% no_miss=%.1f%% capturable=%.1f%% "
                 "capt_free=%.1f%% capt_ready=%.1f%%\n",
                 bank_, static_cast<unsigned long long>(gh1_calls_),
                 pct(gh1_static_calls_, gh1_calls_), pct(gh1_no_miss_calls_, gh1_calls_),
                 pct(gh1_calls_ - gh1_split_calls_, gh1_calls_),
                 pct(gh1_capt_free_calls_, gh1_calls_),
                 pct(gh1_capt_ready_calls_, gh1_calls_));
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

void ExpertStream::stage_predicted(cudaStream_t stream) {
    static const bool vram = [] {
        const char* value = std::getenv("NINFER_PREFETCH_MODE");
        return value != nullptr && std::strcmp(value, "vram") == 0;
    }();
    // One line for the first few calls, whatever the reason: "no placement happened" and "the region
    // is off" look identical in the counters, and that ambiguity already cost a measurement round.
    static std::uint64_t calls = 0;
    const bool trace =
        bank_ < 4 && std::getenv("NINFER_EXPERT_TIMING") != nullptr && ++calls <= 6;
    const auto say = [&](const char* why) {
        if (!trace) { return; }
        std::fprintf(stderr, "[pfv] skip bank=0: %s (vram=%d cache=%d predicted=%zu region=%u)\n", why,
                     vram ? 1 : 0, cache_ != nullptr ? 1 : 0, predicted_.size(),
                     cache_ != nullptr ? cache_->prefetch_slots_per_bank() : 0U);
        std::fflush(stderr);
    };
    if (!vram) { say("mode is not vram"); return; }
    if (cache_ == nullptr) { say("no cache"); return; }
    if (predicted_.empty()) { say("predicted_ is empty"); return; }
    if (cache_->prefetch_slots_per_bank() == 0) { say("no region"); return; }
    std::vector<std::uint32_t> absent;
    absent.reserve(predicted_.size());
    for (const auto expert : predicted_) {
        if (expert < 0 || static_cast<std::uint32_t>(expert) >= count_) { continue; }
        const auto id = static_cast<std::uint32_t>(expert);
        if (cache_->expert_slot(bank_, id)) { continue; } // already resident: not a prediction
        absent.push_back(id);
    }
    prefetch_delta_ += absent.size();
    std::vector<ExpertCacheKey> placed;
    if (!absent.empty() && cache_->set_prefetch_experts(bank_, absent, placed) != 0) {
        cache_->prepare(std::span<const ExpertCacheKey>(placed), stream);
        (void)cache_->progress(stream);
        prefetch_placed_ += placed.size();
    }
    static std::uint64_t reports = 0;
    if (bank_ < 4 && (++reports % 40) == 0) {
        std::fprintf(stderr, "[pfv] bank=%u delta=%llu placed=%llu hits=%llu region=%u/bank\n", bank_,
                     (unsigned long long)prefetch_delta_,
                     (unsigned long long)prefetch_placed_,
                     (unsigned long long)prefetch_hits_,
                     cache_->prefetch_slots_per_bank());
        std::fflush(stderr);
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
        const auto device = std::min(cache_->resident_capacity(), count_);
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
        offset = std::min<std::size_t>(cache_->resident_capacity(), ranked.size());
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
void ExpertStream::freeze_now(std::uint32_t /*tokens*/) {
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
    const auto capacity = std::min<std::size_t>(cache_->resident_capacity(), ranked.size());
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
    freeze_members_ = static_cast<std::uint32_t>(candidates.size());
    if (host_->static_slots_per_bank() != 0) { host_->set_static_frozen(bank_, false); }
    if (!candidates.empty()) {
        host_->ensure_scored(ExpertBankId{bank_}, candidates, hot_score_.data(),
                             /*static_fill=*/true);
    }
    if (host_->static_slots_per_bank() != 0) { host_->set_static_frozen(bank_, true); }
    freeze_present_ = host_->static_present(bank_, candidates);
    publish_static_table();
    std::fprintf(stderr,
                 "[freeze] bank=%u state=frozen device=%zu host_static=%u present=%u d=%u c=%u "
                 "p=%u clock=%llu\n",
                 bank_, frozen_wanted_.size(), freeze_members_, freeze_present_,
                 host_->dynamic_slots_per_bank(), host_->second_chance_per_bank(),
                 host_->prefetch_slots_per_bank(),
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

void ExpertStream::note_request_begin() {
    if (thaw_on_request_) { thaw(); }
}

void ExpertStream::thaw() {
    if (!freeze_mode_ || (!frozen_ && !static_frozen_)) { return; }
    frozen_         = false;
    static_frozen_  = false;
    freeze_clock_   = 0;
    freeze_present_ = 0;
    frozen_wanted_.clear();
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
            std::make_unique<DeviceBuffer>(static_cast<std::size_t>(cache_->resident_capacity()) *
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
    const auto capacity = std::min(cache_->resident_capacity(), ring_slots);
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
            static_cast<std::size_t>(ring_slot) * slot_bytes;
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

bool ExpertStream::ensure(const void* device_ids, std::int32_t top_k, std::int32_t tokens,
                          cudaStream_t stream, const std::int32_t* host_ids, bool assume_covered) {    if (cache_ == nullptr) { throw std::logic_error("expert stream is not attached"); }
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
            gh1_report();
            return true;
        }
        // A resize dropped this window; fall through so the normal path allocates a new one.
    }
    // One window holds a whole layer when the pool is large enough; otherwise the window shrinks to
    // the pool and the caller splits the layer between the Device pool and the host tier.
    const auto capacity = cache_->resident_capacity();
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
            freeze_now(tokens);
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
    gh1_report();
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
    if (assume_covered && !fully) {
        ++gh1_static_calls_;
        ++gh1_no_miss_calls_;
        // Same reason as the pinned path: coverage was established on the Device and this call only
        // rebases host pointers. Counting it is what makes capt_free a rate rather than a restatement
        // of the pool size.
        ++gh1_capt_free_calls_;
        ++gh1_capt_ready_calls_;
        last_cost_ = StepCost{};
        rebase_window();
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
        host_fetch = true;
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
    // is the only number that says whether the region is worth its slots.
    if (cache_ != nullptr && cache_->prefetch_slots_per_bank() != 0) {
        for (std::size_t i = 0; i < values; ++i) {
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
        // NINFER_EXPERT_FREEZE=0 keeps the always-rerank behaviour.
        static const int freeze_after = [] {
            const char* value = std::getenv("NINFER_EXPERT_FREEZE");
            return value != nullptr && *value ? std::atoi(value) : 0;
        }();
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
        if (tokens > 0) {
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
                if (miss_scratch_.size() != count_) { miss_scratch_.assign(count_, 0); }
                if (++miss_generation_ == 0) {
                    std::fill(miss_scratch_.begin(), miss_scratch_.end(), 0);
                    miss_generation_ = 1;
                }
                // Distinct experts this step needs that the set does not hold. Counted per step,
                // not per lane: the spill region has to serve every lane at once, so a four-lane
                // step misses about twice as many experts as a single one (measured).
                std::uint32_t miss = 0;
                for (std::size_t i = 0; i < values; ++i) {
                    const auto expert = static_cast<std::size_t>(route_ids[i]);
                    if (chosen[expert] || miss_scratch_[expert] == miss_generation_) { continue; }
                    miss_scratch_[expert] = miss_generation_;
                    ++miss;
                }
                miss_hist_[std::min<std::uint32_t>(miss, 32)] += 1;
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
        constexpr std::int32_t kSetUpdateTokens   = 8;
        constexpr std::uint32_t kPrefillProtectMin = 96;
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
                             capacity < kPrefillProtectMin;
        if (host_tier && host_ != nullptr && route_ids != nullptr) {
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
            host_direct && host_ != nullptr && host_->arena_device_base() != nullptr;
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
                offsets_[expert]   = static_cast<std::uint64_t>(pool_slot_[expert]) * cache_->slot_bytes();
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
                            static_cast<int>(count_), cache_->slot_bytes(), host_->slot_bytes(),
                            static_cast<std::uint64_t*>(offsets_device_->p),
                            static_cast<std::int32_t*>(mask_device_->p),
                            static_cast<std::uint64_t*>(host_offsets_device_->p),
                            static_cast<std::int32_t*>(host_flags_device_->p), stream);
                        if (freeze_trace()) {
                            std::fprintf(stderr, "[fz] bank=%u expand done\n", bank_);
                            std::fflush(stderr);
                        }
                        placement_published_ = placement_;
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
            const bool delta = delta_enabled && !moved.empty() &&
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
    const bool all_addressable = host_resolved + pool_resolved >= values;
    const bool host_bound = host_upload || host_publish || (host_fetch && !all_addressable);
    if (!host_bound) { ++gh1_capt_free_calls_; }
    // GC2's precondition is the same question plus coverage: every route has an address AND none
    // is outside the two tiers. A layer that satisfies both would replay a table it was captured
    // with.
    if (!host_bound && all_addressable) { ++gh1_capt_ready_calls_; }
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
                     "route_cpu=%llu static_drops=%llu evictions=%llu\n",
                     static_cast<unsigned long long>(freeze_report_layers_), int(frozen_),
                     freeze_members_, freeze_present_,
                     static_cast<unsigned long long>(freeze_membership_moves_),
                     static_cast<unsigned long long>(route_cpu_),
                     static_cast<unsigned long long>(host_stats.static_drops),
                     static_cast<unsigned long long>(host_stats.evictions));
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
    static const bool prefetch_queue = std::getenv("NINFER_PREFETCH") != nullptr;
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
                static std::atomic<std::uint64_t> layer_tick{0};
                static const std::uint64_t window = [] {
                    const char* value = std::getenv("NINFER_PREFETCH_WINDOW");
                    if (value != nullptr && *value != '\0') {
                        return static_cast<std::uint64_t>(std::strtoull(value, nullptr, 10));
                    }
                    return std::uint64_t{40};
                }();
                auto& queue     = ExpertPrefetchQueue::instance();
                const auto tick = ++layer_tick;
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





