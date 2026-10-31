#include <algorithm>
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
    //
    // `assume_covered` is O1's arm: the caller has decided, from a verdict it already read, that
    // every route of this layer is resident, so the one thing ensure() needs the ids for - the
    // placement - has nothing to do. The layer then pays neither the copy nor the stream
    // synchronise that waits for it (measured at 0.06 ms a layer, 2.4 ms a token).
    //
    // Measured at 12 GiB and NOT usable as a prediction: the pool there holds 47 of a bank's 256
    // experts and its set moves, so "the previous step was covered" is false for 3-44 routes of the
    // next step. Every armed step came back dirty, every token was replayed, and decode fell from
    // 6.4/7.5 to 12.0/11.6 ms a token (155/133 -> 84/86 tok/s). Without this claim the same skip is
    // clean and gains nothing (1x.4u), so the arm is kept for the machinery, defaulted off, and the
    // work that would make it pay is a device-side residency decision, not a step-level guess.
    bool ensure(const void* device_ids, std::int32_t top_k, std::int32_t tokens,
                cudaStream_t stream, const std::int32_t* host_ids = nullptr,
                bool assume_covered = false);

    // port (sm_89): O1's device verdict word (see ops::route_coverage). Mapped host memory written by
    // the coverage kernel on the stream, so the host reads a layer's answer at its next synchronise
    // without a read-back of the ids. Allocated on first use and only when a caller asks for it.
    [[nodiscard]] std::uint32_t* route_coverage_device();
    // Valid once the stream has synchronised past the layer that wrote it.
    [[nodiscard]] std::uint32_t route_coverage_read();

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
    // Experts worth keeping in the warm tier: routed more than once and not in the Device set.
    //
    // The warm tier is a partition by probability, not a copy of everything the router touched: an
    // expert that has been routed once is a cold miss and belongs on disk, while one that keeps
    // coming back but does not fit the Device pool is exactly what RAM is for. `min_hits` is a count
    // of routes seen so far, not a byte threshold - it says "this one has proven it comes back".
    // Experts the pool dropped on the last ensure(): the warm tier admits from here rather than from
    // the routes, because these are the ones with a demonstrated short reuse distance.
    [[nodiscard]] std::span<const std::int32_t> demoted_this_step() const noexcept { return demoted_; }

    // port (sm_89): O1's miss counter. A predicted skip runs a layer without reading the ids back, so
    // "did the resident set really cover it?" cannot be answered from the mask - the mask is published
    // by the same decision that placed the routes, so asking it always answers yes (GH1's `no_miss`
    // read 100% for exactly that reason). The kernels add one per route they skip, into mapped host
    // memory, and the host reads the running total where it already synchronises. Off unless
    // NINFER_O1_MISS_COUNT is set: today it is an instrument that has to agree with the host's own
    // count, and it becomes a fast path only once the numbers say a prediction pays.
    [[nodiscard]] bool miss_count_enabled();
    [[nodiscard]] std::uint32_t* missed_routes_device();
    // The running total as the host reads it. Callers must have synchronised the stream first; the
    // value is a plain word in mapped memory, so the read is volatile rather than cached.
    [[nodiscard]] std::uint32_t missed_routes_total();
    // `reading` is the counter's running total; `host_layer` is what this layer's fetched ids said the
    // mask dropped. A reading's delta against the previous one is the previous call's layer, because
    // the read sits right after a fetch that only covered the layers before this one.
    void observe_missed(std::uint32_t reading, std::uint64_t host_layer, bool decode);
    struct MissTally {
        std::uint64_t device   = 0; // routes the Device half skipped, as read so far
        std::uint64_t host     = 0; // the same routes counted from the fetched ids
        std::uint64_t compared = 0; // decode layers, where the comparison is a gate
        std::uint64_t mismatch = 0; // readings where the two disagreed
        // A prefill layer can republish the resident set between the host's count and the kernel's
        // run, so the two are then not looking at the same mask. Reported, not gated: O1 replays a
        // decode step, and there the mask does not move underneath the comparison.
        std::uint64_t prefill_compared = 0;
        std::uint64_t prefill_mismatch = 0;
    };
    [[nodiscard]] const MissTally& miss_tally() const noexcept { return miss_tally_; }

    // O1 step 2's predictor, one bit per bank: whether this bank's layer covered its routes on the
    // previous step. Kept here because a bank sees exactly one layer per step, so "the same layer
    // last step" is the only history a bank has.
    void note_layer_covered(bool covered) noexcept { prev_covered_ = covered; }
    [[nodiscard]] bool predicted_covered() const noexcept { return prev_covered_; }

    [[nodiscard]] ExpertCache::UploadSourceStats upload_source_stats() const {
        return cache_->upload_sources();
    }
    void warm_candidates(std::span<const std::int32_t> routes, std::vector<std::int32_t>& out,
                         float min_hits) const {
        out.clear();
        if (hot_score_.size() != static_cast<std::size_t>(count_)) { return; }
        for (const auto route : routes) {
            const auto expert = static_cast<std::size_t>(route);
            if (expert >= hot_score_.size()) { continue; }
            if (hot_score_[expert] < min_hits) { continue; }
            out.push_back(route);
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
    }
    // GH1 rates, for the policy controller: static = the set did not move this step, no_miss = every
    // route was already in the pool, capturable = the step did not take the split path at all.
    // Denominator is gh1_calls_. Negative means "not measured yet", never zero.
    struct Gh1Rates {
        double static_pct     = -1.0;
        double no_miss_pct    = -1.0;
        double capturable_pct = -1.0;
    };
    // Share of layers that are covered when the previous layer was covered - the speculative skip's
    // expected hit rate. Negative when there is no sample yet.
    [[nodiscard]] double covered_predictor_rate() const noexcept {
        if (predict_trials_ == 0) { return -1.0; }
        return static_cast<double>(predict_hits_) / static_cast<double>(predict_trials_);
    }
    [[nodiscard]] Gh1Rates gh1_rates() const noexcept {
        Gh1Rates rates;
        if (gh1_calls_ == 0) { return rates; }
        const auto denom     = static_cast<double>(gh1_calls_);
        rates.static_pct     = 100.0 * static_cast<double>(gh1_static_calls_) / denom;
        rates.no_miss_pct    = 100.0 * static_cast<double>(gh1_no_miss_calls_) / denom;
        rates.capturable_pct = 100.0 * static_cast<double>(gh1_calls_ - gh1_split_calls_) / denom;
        return rates;
    }
    // port (sm_89): what the last ensure() call spent, split by node, so the layer timeline can say
    // where a split token goes instead of folding routing, layout and upload into one number. The
    // clocks are taken unconditionally - three steady_clock reads per layer - and only the reporting
    // is gated on NINFER_SPLIT_TIMING.
    struct StepCost {
        double route_ms  = 0.0; // the route ids, only when the caller has none to hand over
        double layout_ms = 0.0; // hot set, slot publish, mask and offset tables
        double upload_ms = 0.0; // prepare + wait_for_progress
        std::uint32_t uploaded = 0; // experts actually moved into the pool on this layer
    };
    [[nodiscard]] StepCost last_cost() const noexcept { return last_cost_; }
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
    std::vector<std::int32_t> demoted_;
    // Predictor scoreboard for O1's speculative skip: if the previous layer was fully covered, how
    // often is this one too? That ratio is the repair rate a speculative design would pay, and it is
    // the number that decides whether skipping the read-back is worth its complexity.
    bool last_covered_             = false;
    std::uint64_t predict_trials_  = 0;
    std::uint64_t predict_hits_    = 0;
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
    StepCost last_cost_{};
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
    std::unique_ptr<PinnedHostBuffer> route_staging_;    // O1's device verdict: one mapped word, stored by the coverage kernel and read by the host.
    std::unique_ptr<PinnedHostBuffer> coverage_word_;
    std::uint32_t* coverage_word_host_   = nullptr;
    std::uint32_t* coverage_word_device_ = nullptr;
    bool coverage_word_checked_          = false;
    // O1's counter: mapped, so the skip is counted where it happens. `miss_lag_` is the host total
    // through the previous layer, which is what a reading can actually be compared with.
    std::unique_ptr<PinnedHostBuffer> miss_count_;
    std::uint32_t* miss_count_host_   = nullptr;
    std::uint32_t* miss_count_device_ = nullptr;
    bool miss_count_checked_          = false;
    MissTally miss_tally_{};
    std::uint32_t miss_last_reading_ = 0;
    std::uint64_t miss_prev_host_    = 0;
    bool prev_covered_               = false;
    ops::CudaSwiGLUExperts experts_{};
};

} // namespace ninfer::models::qwen3_5::execution


