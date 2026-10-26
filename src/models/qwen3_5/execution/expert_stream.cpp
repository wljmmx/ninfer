#include "models/qwen3_5/execution/expert_stream.h"

#include <cuda_runtime.h>

#include <algorithm>
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
    const std::array<std::uint64_t, 2> gate_up_shape{gate_up_.region.rows, gate_up_.columns};
    const std::array<std::uint64_t, 2> down_shape{down_.region.rows, down_.columns};
    gate_up_expert_ = weight_geometry(gate_up_.format, gate_up_.layout, gate_up_shape);
    down_expert_    = weight_geometry(down_.format, down_.layout, down_shape);
    keys_.reserve(count_);
    for (std::uint32_t expert = 0; expert < count_; ++expert) {
        keys_.push_back(ExpertCacheKey{0, expert});
    }
    hot_score_.assign(count_, 0.0F);
}

void ExpertStream::attach(ExpertCache& cache, std::uint32_t bank, const ExpertBankLayout& entry) {
    register_profile_dump(bank, &hot_score_);
    if (bank > 0xffffU || cache.capacity_bytes() < entry.entry_bytes) {
        throw std::invalid_argument("expert stream cannot attach to this cache");
    }
    cache_ = &cache;
    bank_  = bank;
    entry_ = entry;
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

bool ExpertStream::ensure(const void* device_ids, std::int32_t top_k, std::int32_t tokens,
                          cudaStream_t stream, const std::int32_t* host_ids) {
    if (cache_ == nullptr) { throw std::logic_error("expert stream is not attached"); }
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
    // port (sm_89): a pool that cannot hold a whole layer keeps a hot set per bank. This step's
    // routes are placed first - they must run on the Device - and the remaining slots go to the
    // experts with the highest decayed score, so the next step has a chance to hit. The cache only
    // uploads the experts that actually changed, and experts that stay keep their slot.
    if (!fully) {
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
        if (frozen_) {
            // port (sm_89): a frozen set used to take every slot, so a route outside it had nowhere
            // to go and was forced onto the CPU - measured 121 ms of CPU per token at a 4096 MiB
            // pool against 0.00 ms of upload, for about 3.2 experts a layer. Hold the set back to
            // `capacity - spill` and let this step's outside routes take the rest, where an upload
            // costs 0.12 ms against the CPU's 0.5.
            // Measured across 4-12 GiB: holding 15% back is worth 5.4x at 4 GiB (a frozen set that
            // takes every slot has nowhere to put the routes outside it, so they fall to the CPU at
            // 0.5 ms each - 121 ms per token measured), and it is still the best policy up to 10 GiB.
            // At 12 GiB the set covers 95.7% and the outside routes cost only 0.8 ms on the CPU, so
            // filling the pool wins there (100.5 against 94.6 tok/s) - and a full, static table is
            // what a per-layer graph needs anyway.
            // port (sm_89): the size of the spill region is the number of experts this layer's
            // tokens actually miss, not a percentage of the pool. Measured on 13k tokens of real
            // conversation: 0.57 misses a token at 163 slots a layer (p99 = 3, max = 4), 0.91 at
            // 109 (p99 = 4), 2.17 at 54 (p99 = 5). Leaving 15% free is right at 54 slots (8) and
            // six times too generous at 163 (24 for a p99 of 3), so the slots the hot set loses
            // buy nothing. NINFER_EXPERT_SPILL_PCT keeps the old rule, NINFER_EXPERT_SPILL_SLOTS
            // pins a count, and both exist to A/B against the estimate.
            const auto spill = [&]() -> std::size_t {
                if (const char* value = std::getenv("NINFER_EXPERT_SPILL_PCT")) {
                    return static_cast<std::size_t>(capacity) *
                           std::min<std::uint64_t>(std::strtoul(value, nullptr, 10), 90) / 100;
                }
                if (const char* value = std::getenv("NINFER_EXPERT_SPILL_SLOTS")) {
                    return std::min<std::size_t>(std::strtoul(value, nullptr, 10),
                                                 static_cast<std::size_t>(capacity) / 2);
                }
                return std::min<std::size_t>(spill_slots(capacity), static_cast<std::size_t>(capacity) / 2);
            }();
            const auto pinned = capacity > spill ? capacity - spill : capacity;
            const auto keep   = std::min<std::size_t>(pinned, frozen_wanted_.size());
            wanted.assign(frozen_wanted_.begin(), frozen_wanted_.begin() + keep);
            for (const auto expert : wanted) { chosen[expert] = true; }
            for (std::size_t i = 0; i < values && wanted.size() < capacity; ++i) {
                const auto expert = static_cast<std::uint32_t>(route_ids[i]);
                if (chosen[expert]) { continue; }
                chosen[expert] = true;
                wanted.push_back(expert);
            }
        } else {
            if (!static_set && !generation_mode) {
                for (std::size_t i = 0; i < values && wanted.size() < capacity; ++i) {
                    const auto expert = static_cast<std::uint32_t>(route_ids[i]);
                    if (chosen[expert]) { continue; }
                    chosen[expert] = true;
                    wanted.push_back(expert);
                }
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
            coverage_.routes += values;
            coverage_.inside += inside;
            if ((std::getenv("NINFER_SPLIT_TIMING") != nullptr ||
                 std::getenv("NINFER_EXPERT_TIMING") != nullptr) &&
                ++coverage_calls_ % 40 == 0) {
                const auto routes = static_cast<double>(coverage_.routes == 0 ? 1 : coverage_.routes);
                const auto covered = 100.0 * static_cast<double>(coverage_.inside) / routes;
                // One token publishes top_k entries per layer, so the miss rate scales back to a
                // per-token count without knowing how many layers this stream has seen.
                const auto misses_per_token =
                    top_k == 0 ? 0.0
                               : static_cast<double>(coverage_.routes - coverage_.inside) *
                                     static_cast<double>(top_k) / routes;
                std::fprintf(stderr,
                             "[set] bank=%u coverage=%llu/%llu (%.1f%%) misses/token=%.2f slots=%u\n",
                             bank_, static_cast<unsigned long long>(coverage_.inside),
                             static_cast<unsigned long long>(coverage_.routes), covered,
                             misses_per_token, capacity);
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
        const bool refresh = !set_initialised_ || tokens <= kSetUpdateTokens ||
                             capacity < kPrefillProtectMin;
        if (refresh && cache_->set_resident_experts(bank_, wanted)) {
            set_initialised_ = true;
            std::fill(offsets_.begin(), offsets_.end(), 0);
            mask_host_.assign(count_, 0);
            for (std::uint32_t expert = 0; expert < count_; ++expert) {
                const auto slot = cache_->expert_slot(bank_, expert);
                if (!slot) { continue; }
                offsets_[expert]         = static_cast<std::uint64_t>(*slot) * cache_->slot_bytes();
                mask_host_[expert]       = 1;
            }
            resident_order_ = wanted;
            const auto table_bytes = offsets_.size() * sizeof(std::uint64_t);
            const auto mask_bytes  = mask_host_.size() * sizeof(std::int32_t);
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
            if (table_bytes != 0 && mask_bytes != 0) {
                void* const       destinations[2] = {offsets_device_->p, mask_device_->p};
                const void* const sources[2]      = {offsets_.data(), mask_host_.data()};
                const std::size_t sizes[2]        = {table_bytes, mask_bytes};
                cudaMemcpyAttributes attributes{};
                attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
                attributes.flags          = 0;
                std::size_t attribute_index = 0;
                const auto submitted        = cudaMemcpyBatchAsync(
                    destinations, sources, sizes, 2, &attributes, &attribute_index, 1, stream);
                // A driver that rejects the batch still has to get the tables across: fall back to
                // the two copies rather than leaving the layer with a stale resident set.
                if (submitted != cudaSuccess) {
                    (void)cudaMemcpyAsync(offsets_device_->p, offsets_.data(), table_bytes,
                                          cudaMemcpyHostToDevice, stream);
                    (void)cudaMemcpyAsync(mask_device_->p, mask_host_.data(), mask_bytes,
                                          cudaMemcpyHostToDevice, stream);
                }
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
    const auto upload_begin = std::chrono::steady_clock::now();
    cache_->prepare(required_, stream);
    try {
        cache_->wait_for_progress(stream);
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string("pool: ") + error.what());
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
    // The uploads were queued on the consumer stream, so the kernels that read them are already
    // ordered after the copies; only the artifact path had to wait, and it did so inside
    // progress(). A device-wide synchronise here used to cost most of the layer's time.
    return true;
}

} // namespace ninfer::models::qwen3_5::execution


