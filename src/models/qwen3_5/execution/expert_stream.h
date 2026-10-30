#pragma once

#include "core/arena.h"
#include "core/expert_cache.h"
#include "core/expert_file.h"
#include "core/weight_bank.h"
#include "ninfer/ops/moe.h"

#include <array>
#include <cstdint>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

class HostExpertResidency;

// port (sm_89): one routed-expert bank whose weights stay in the artifact and stream into a
// shared Device pool. The bank description is fixed at startup; only the per-expert offset table
// is republished as the resident set changes.
class ExpertStream {
public:
    ExpertStream(std::uint32_t count, WeightBankLayout gate_up, WeightBankLayout down,
                 WeightGeometry gate_up_parent, WeightGeometry down_parent,
                 std::uint64_t gate_up_object, std::uint64_t down_object);
    // port (sm_89): an artifact may store one object per expert instead of one object per bank. The
    // loader then needs a handle per expert, not just the bank's first. Both forms are described by
    // the same per-bank layout, so nothing below changes shape - only how many objects it spans.
    ExpertStream(std::uint32_t count, WeightBankLayout gate_up, WeightBankLayout down,
                 WeightGeometry gate_up_parent, WeightGeometry down_parent,
                 std::vector<std::uint64_t> gate_up_objects, std::vector<std::uint64_t> down_objects);

    // Startup-only. `entry` is the cache's per-entry plane layout for this bank.
    void attach(ExpertCache& cache, std::uint32_t bank, const ExpertBankLayout& entry);

    // Publishes the experts the next call routes to, then republishes the offset table. Returns
    // false when the pool cannot hold one whole layer at all; the caller then owns the decision
    // (today it raises, later it can fall back to the host tier).
    bool ensure(const void* device_ids, std::int32_t top_k, std::int32_t tokens,
                cudaStream_t stream, const std::int32_t* host_ids = nullptr);

    [[nodiscard]] const ops::CudaSwiGLUExperts& experts() const noexcept { return experts_; }
    [[nodiscard]] std::uint32_t bank_index() const noexcept { return bank_; }
    // Device-side int32 array, one entry per expert: 1 when the pool holds it.
    [[nodiscard]] const void* resident_mask() const noexcept {
        return mask_device_ ? mask_device_->p : nullptr;
    }
    // Host copy of the same mask, for the CPU half of a split layer.
    [[nodiscard]] std::span<const std::int32_t> resident_mask_host() const noexcept {
        return mask_host_;
    }
    [[nodiscard]] bool fully_resident() const noexcept;
    [[nodiscard]] bool attached() const noexcept { return cache_ != nullptr; }
    // Host half of a split tier; null when the Device pool holds every expert.
    void attach_host(HostExpertResidency& host) noexcept { host_ = &host; }
    [[nodiscard]] HostExpertResidency* host() const noexcept { return host_; }

    [[nodiscard]] std::uint32_t count() const noexcept { return count_; }
    [[nodiscard]] std::uint64_t gate_up_object() const noexcept { return gate_up_objects_.front(); }
    [[nodiscard]] std::uint64_t down_object() const noexcept { return down_objects_.front(); }
    // True when the artifact gives every expert its own object. The bank is then described expert
    // by expert; the fused form describes one object and addresses experts by row range.
    [[nodiscard]] bool per_expert_objects() const noexcept { return per_expert_objects_; }
    [[nodiscard]] std::span<const std::uint64_t> gate_up_objects() const noexcept {
        return gate_up_objects_;
    }
    [[nodiscard]] std::span<const std::uint64_t> down_objects() const noexcept {
        return down_objects_;
    }
    [[nodiscard]] const WeightGeometry& gate_up_parent() const noexcept { return gate_up_parent_; }
    [[nodiscard]] const WeightGeometry& down_parent() const noexcept { return down_parent_; }
    [[nodiscard]] const WeightBankLayout& gate_up_layout() const noexcept { return gate_up_; }
    [[nodiscard]] const WeightBankLayout& down_layout() const noexcept { return down_; }
    [[nodiscard]] const WeightGeometry& gate_up_expert_geometry() const noexcept {
        return gate_up_expert_;
    }
    [[nodiscard]] const WeightGeometry& down_expert_geometry() const noexcept {
        return down_expert_;
    }

    // port (sm_89): how many of the routed entries this bank published were already resident.
    // Observation only - nothing schedules on it yet - but it is the number a set, a prefetch
    // target and any future capture gate has to be judged by. It costs a handful of comparisons per
    // layer: the routes are already on the host for the residency decision.
    struct Coverage {
        std::uint64_t routes = 0;
        std::uint64_t inside = 0;
    };
    [[nodiscard]] Coverage coverage() const noexcept { return coverage_; }
    // port (sm_89): how many slots a frozen set must leave free. Measured on real text, a token
    // misses 0.57 experts a layer at 163 slots (p99 = 3) and 0.91 at 109 (p99 = 4), so a percentage
    // of the pool over-reserves badly at the large end - 15% of 163 is 24 slots for a p99 of 3 -
    // while a fixed few is enough everywhere. This is the p99 of the tracked distribution.
    [[nodiscard]] std::uint32_t spill_slots(std::uint32_t capacity) const noexcept;
    void reset_coverage() noexcept { coverage_ = Coverage{}; }
    // Reserved for a captured expert table: a capture cannot read a host tally, so that design has
    // to count its misses on the device. Null until such a capture exists.
    [[nodiscard]] const std::uint32_t* device_miss_counter() const noexcept { return nullptr; }

private:
    std::uint32_t count_          = 0;
    WeightBankLayout gate_up_;
    WeightBankLayout down_;
    WeightGeometry gate_up_parent_;
    WeightGeometry down_parent_;
    WeightGeometry gate_up_expert_;
    WeightGeometry down_expert_;
    std::vector<std::uint64_t> gate_up_objects_;
    std::vector<std::uint64_t> down_objects_;
    bool per_expert_objects_ = false;

    ExpertCache* cache_ = nullptr;
    HostExpertResidency* host_ = nullptr;
    std::uint32_t bank_ = 0;
    ExpertBankLayout entry_{};
    // port (sm_89): the pool holds every layer at once, so this bank's window is fixed and no
    // per-step routing decision - and therefore no host round trip - is needed.
    bool pinned_ = false;
    void rebase_window();
    // port (sm_89): decayed per-expert route counts. A pool that cannot hold a whole layer keeps
    // the top entries of this instead of whatever the current step happened to route to, which
    // used to replace all eight experts on every token and make the hot set useless.
    std::vector<float> hot_score_;
    // llama.cpp-style SLRU over slot ids (src/llama-moe-cache.cpp:16-121): a probation list and a
    // protected half. A hit promotes the slot, a miss takes the probation head while skipping the
    // experts this step needs, and a slot that keeps its expert keeps its number, so republishing
    // the table does not move it. The policy is opt-in (NINFER_EXPERT_POLICY=slru) because the
    // score fill it replaces is what the measured streaming numbers were taken with.
    bool slru_policy_ = false;
    // switch to the whole-pool SLRU once a step is six lanes wide; NINFER_EXPERT_AUTO_SLRU=0 pins
    // the default policy at every width
    bool auto_slru_ = false;
    // opt-in region split (hot + dedicated spill ring); the default policy is unchanged until it
    // is measured against the report's numbers
    bool regions_ = false;
    // Epoch-static hot region (NINFER_EXPERT_EPOCH=<steps>): the hot list is rebuilt only at an
    // epoch boundary from the load window, so the table cannot churn per step.
    std::uint32_t epoch_steps_ = 0;
    std::uint32_t step_in_epoch_ = 0;
    std::uint64_t epochs_ = 0;
    std::vector<float> load_window_;
    std::vector<std::uint32_t> hot_list_;
    std::vector<std::uint32_t> slru_expert_;   // hot region, kEmpty when free
    std::vector<std::int32_t> slru_of_expert_; // count_, -1 when not resident
    std::vector<std::uint8_t> slru_protected_;
    std::uint32_t slru_n_protected_ = 0;
    // Promotion and demotion between the regions: an expert that misses twice in a row is
    // promoted into the hot region; the hot entry with the lowest decayed score leaves for the
    // warm tier, and its id is kept in a short journal the RAM side and the prefetch read.
    std::vector<std::uint8_t> prev_spill_;
    std::vector<std::uint8_t> spill_hits_;
    std::vector<std::uint32_t> warm_journal_;
    // Expected next-step demand per expert: an EWMA of "needed this step" (decay 0.9, about ten
    // steps of memory). Promotion and demotion are decided by comparing these numbers, so an
    // expert is hot because it keeps being needed, not because it happened to be needed twice in
    // a row.
    std::vector<float> demand_;
    bool hot_seeded_ = false;
    std::uint32_t slru_filled_ = 0;
    std::vector<std::int32_t> slru_prev_, slru_next_;
    std::int32_t slru_head_[2]{-1, -1};
    std::int32_t slru_tail_[2]{-1, -1};
    std::vector<std::uint32_t> resident_order_;
    // Last published slot per expert, so a step only copies the rows that actually moved.
    std::vector<std::int32_t> prev_slot_;
    std::uint64_t publish_bytes_ = 0;
    std::uint64_t publish_calls_ = 0;
    std::uint64_t entered_ = 0;
    // GH1: one counter each for the three coverage numbers the plan gates on - steps where the
    // table did not move (static rate), steps with no missing expert (whole-layer capturable),
    // and the denominator.
    std::uint64_t gh1_steps_ = 0;
    std::uint64_t gh1_static_steps_ = 0;
    std::uint64_t gh1_no_miss_steps_ = 0;
    std::uint64_t gh1_calls_        = 0;
    std::uint64_t gh1_split_calls_  = 0;
    std::uint64_t gh1_no_miss_calls_ = 0;
    std::uint64_t gh1_static_calls_  = 0;
    // Prints the GH1 panel; one place so the pinned fast path and the normal path cannot drift.
    void gh1_report();
    std::uint64_t entered_steps_ = 0;
    // port (sm_26): whether this bank's resident set has ever been built. A prefill is not allowed to
    // choose the set, but the first call has to, or a cold start would route everything to the host
    // tier and never establish one.
    bool set_initialised_ = false;
    std::uint32_t since_decay_   = 0;
    std::uint32_t since_reorder_ = 0;
    // port (sm_89): a warm-up then a frozen set, so a static policy stops re-ranking per step.
    bool frozen_ = false;
    std::uint64_t steps_ = 0;
    // Routed entries published so far and how many were already resident (see Coverage).
    Coverage coverage_{};
    // Misses per step, histogram 0..32, decayed so a new workload is not held hostage. One
    // sample per step: the spill region holds every lane's miss at once (8 lanes x 8 experts).
    std::array<std::uint32_t, 33> miss_hist_{};
    std::uint32_t miss_samples_ = 0;
    std::vector<std::uint32_t> miss_scratch_;
    std::uint32_t miss_generation_ = 0;
    // widest step this stream has served, so the spill ring can hold one of them
    std::uint32_t lanes_seen_ = 0;
    std::uint64_t coverage_calls_ = 0;
    std::uint64_t coverage_dumps_ = 0;
    std::vector<std::uint32_t> frozen_wanted_;
    // port (sm_89): a generation is a run of steps that shares one resident set. The set is rebuilt
    // at a boundary (NINFER_EXPERT_GEN_STEPS, default 64) and held in between, which is what keeps
    // the offset table still enough for a captured graph; coverage and churn are reported per
    // generation so the steady state can be measured instead of assumed.
    std::uint32_t steps_in_gen_ = 0;
    std::uint32_t generation_   = 0;
    std::vector<std::uint32_t> previous_wanted_;
    std::vector<ExpertCacheKey> keys_;
    std::vector<ExpertCacheKey> required_;
    // Per-expert byte offsets the kernel adds to each plane base. Uniform today, but the kernel
    // already addresses through this table, so a partial resident set only has to rewrite entries.
    std::vector<std::uint64_t> offsets_;
    std::unique_ptr<DeviceBuffer> offsets_device_;
    // One entry per expert: 1 when the pool holds it. A partial resident set drives the GPU
    // through ExpertSubsetKind::MaskOnes and leaves the rest to the host tier.
    std::unique_ptr<DeviceBuffer> mask_device_;
    std::vector<std::int32_t> mask_host_;
    std::unique_ptr<PinnedHostBuffer> route_staging_;
    ops::CudaSwiGLUExperts experts_{};
};

} // namespace ninfer::models::qwen3_5::execution

