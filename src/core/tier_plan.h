#pragma once

// One table that divides each tier among the consumers that exist, so a budget has exactly one home.
//
// Why this and not the ledger: `memory_ledger.h` accumulates whatever each module asks for, in the
// order the modules happen to run. That answers "who took the bytes", never "what is the plan" - and
// it cannot see a consumer that was never asked (the prediction region, the read-ahead band, the KV
// host tier), which is how two sizers ended up each taking "a share of free" and one of them landing
// on the other's memory. This table is decided FIRST, at construction, and the ledger is then told
// exactly these numbers. One decision, one projection.
//
// The rules the table encodes:
//   * Device: `expert_pool` is the routed set. `prefetch` and `slru_band` are NOT extra memory - they
//     are slices of that pool (one carved per bank by the cache, the other a policy segmentation done
//     per layer by the stream), so they are recorded as carved lines and do not add to the tier.
//     `startup` is the planning allowance the cap itself carries; without it the table would not add
//     up to the cap, which is why it is a line rather than a footnote.
//   * Host: `expert_arena` (the warm tier), `readahead` (the second-chance band's sibling), 
//     `second_chance`, and `kv_host` - the KV payload a tiered context keeps in RAM. The last one is
//     owned by the context planner, not by the expert tier, and it is in this table precisely so the
//     two cannot each take a share and meet in the middle.
//   * SSD: NO expert budget. The cold range is an outcome - what the two warm tiers did not hold -
//     not a knob, so there is nothing to divide. `kv_spill` has no size either: the tier is judged by
//     its RE-READ count (a page read more than once is the spill failing), not by a cap, because a
//     spill cap only decides which pages are read cold twice and calls it a policy.
//
// A consumer that is not in this table cannot be granted bytes: the list is closed on purpose. That
// is the whole failure mode this file exists to prevent - an unregistered sizer.

#include "core/memory_ledger.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>

namespace ninfer::core {

enum class Budget : std::uint8_t {
    DeviceExpertPool = 0,
    DeviceKvPages,
    DevicePrefetch,
    DeviceSlruBand,
    DeviceStartup,
    HostExpertArena,
    HostReadahead,
    HostSecondChance,
    HostKvHost,
    DiskKvSpill,
    Count,
};

inline constexpr std::size_t kBudgetCount = static_cast<std::size_t>(Budget::Count);

// One row of the closed list. `carved_from` marks a slice of another line (the tier total ignores it);
// `planned` marks a figure this table does not claim - a planning input whose tier owner is another
// module and whose admission happens when that module activates. Both are shown because the whole
// point is to see them next to the claims, and neither is added to `claimed`, so the arithmetic stays
// true. `note` is printed after a zero line, so "zero because the policy takes nothing" and "zero
// because nobody asked" cannot be read the same way.
struct BudgetSlot {
    Budget      id;
    Tier        tier;
    const char* name;
    Budget      carved_from; // == id when the line is a claim of its own
    const char* note;        // printed when the line is zero; may be nullptr
    bool        planned;     // shown, not claimed
};

// What the SSD tier reports about KV spill: the spec's triple, because there is nothing here to size.
// No expert budget lives on this tier (the cold range is an outcome), and the spill is NOT judged by
// a cap - a cap only decides which pages are read cold twice and calls it a policy - so it is judged
// by its RE-READ count. `cap` says "none" when no file exists, and names the file's extent capacity
// when one does (a capacity, not a plan claim). `reread` says "none" when no per-page re-read counter
// exists; the nearest real counters are kv_tier_stats()' spilled/fetched pages, which the `kv tiers`
// panel prints at the end of a run.
struct KvSpillView {
    bool        enabled = false;
    std::string cap     = "none";
    std::string reread  = "none";
};

// Does anything consume a Device prefetch region? The Device form of `ExpertStream::stage_predicted`
// resolves the predicted ids against the pool's own mirror and counts them, and states in its own
// panel that it placed nothing: `[pf-pred] Device form: ... fill=host-only (0 placed by the Device)`
// (expert_stream.cpp). The placement half - the words set_prefetch_experts() writes and the bytes
// that then have to leave the artifact - only exists on the host form. So the region has no Device
// consumer today, and the line is zero BY RULE rather than because a controller computed zero:
// before this, `[ctl] prefetch port ... -> 0 slot(s)/bank` made "nobody is connected" look like "the
// policy chose nothing", and `NINFER_PREFETCH_SLOTS` could still carve a region no one filled
// (measured: 20 slots/bank = 3.6 GiB = 30% of the pool, 1.54 tok/s against 8.28 without).
inline constexpr bool device_prefetch_consumer_present() noexcept { return false; }

// The reason string the Device prefetch line carries when it is zero, so "the policy took nothing"
// and "there is nobody to take it" cannot be read the same way. Kept beside the predicate it explains.
inline constexpr const char* kDevicePrefetchNoConsumer =
    "no consumer: device-side placement absent (stage_predicted resolves and counts, fill=host-only)";

inline constexpr std::array<BudgetSlot, kBudgetCount> kBudgetSlots{{
    // Name and order follow the capacity spec: device expert_pool_mb | kv_pages_mb | prefetch_mb |
    // slru_band_mb, then the startup allowance the cap carries (not part of the spec's four, so it
    // sits last and keeps its own line).
    {Budget::DeviceExpertPool, Tier::Device, "expert_pool_mb", Budget::DeviceExpertPool, nullptr, false},
    {Budget::DeviceKvPages, Tier::Device, "kv_pages_mb", Budget::DeviceKvPages, nullptr, false},
    {Budget::DevicePrefetch, Tier::Device, "prefetch_mb", Budget::DeviceExpertPool,
     kDevicePrefetchNoConsumer, false},
    {Budget::DeviceSlruBand, Tier::Device, "slru_band_mb", Budget::DeviceExpertPool,
     "no claim of its own - SLRU segments expert_pool_mb at policy time", false},
    {Budget::DeviceStartup, Tier::Device, "staging_mb", Budget::DeviceStartup,
     "the planning allowance the cap carries", false},
    // host expert_arena_mb | kv_host_mb | prefetch_readahead_mb | second_chance_mb, in that order.
    {Budget::HostExpertArena, Tier::Host, "expert_arena_mb", Budget::HostExpertArena, nullptr, false},
    {Budget::HostKvHost, Tier::Host, "kv_host_mb", Budget::HostKvHost,
     "planned here; the context planner claims it when the tiered KV mode activates", true},
    {Budget::HostReadahead, Tier::Host, "prefetch_readahead_mb", Budget::HostReadahead, nullptr, false},
    {Budget::HostSecondChance, Tier::Host, "second_chance_mb", Budget::HostSecondChance, nullptr, false},
    // ssd: no expert budget, no size knob. Judged by its re-read count, see the footer.
    {Budget::DiskKvSpill, Tier::Disk, "kv_spill", Budget::DiskKvSpill,
     "no size knob - the tier is judged by its re-read count", false},
}};

class TierPlan {
public:
    static constexpr std::uint64_t kNoCap = MemoryLedger::kNoCap;

    [[nodiscard]] static const BudgetSlot& slot(Budget id) noexcept {
        return kBudgetSlots[static_cast<std::size_t>(id)];
    }

    void set_cap(Tier tier, std::uint64_t bytes) noexcept { caps_[index(tier)] = bytes; }
    [[nodiscard]] std::uint64_t cap(Tier tier) const noexcept { return caps_[index(tier)]; }

    // Record one consumer's grant. A line whose tier does not match its row is refused outright: a
    // host budget filed under Device is exactly the mis-filing this table is meant to make visible.
    // A claim that does not fit its tier is refused and nothing is recorded - the caller lowers the
    // ask, the same contract as the ledger's.
    bool set(Budget id, std::uint64_t bytes) noexcept {
        // A line whose consumer does not exist is zero, whatever the caller computed: the table owns
        // that rule, so no controller number and no override can put bytes on it again.
        if (id == Budget::DevicePrefetch && !device_prefetch_consumer_present()) { bytes = 0; }
        const auto& row = slot(id);
        if (row.carved_from != id) {
            const auto parent = bytes_of(row.carved_from);
            if (bytes > parent) { return false; }
            bytes_[static_cast<std::size_t>(id)] = bytes;
            return true;
        }
        if (row.planned) {
            bytes_[static_cast<std::size_t>(id)] = bytes;
            return true;
        }
        const auto tier_index = index(row.tier);
        const auto rest       = caps_[tier_index] == kNoCap
                                    ? kNoCap
                                    : caps_[tier_index] - (total(row.tier) - bytes_of(id));
        if (caps_[tier_index] != kNoCap && bytes > rest) { return false; }
        bytes_[static_cast<std::size_t>(id)] = bytes;
        return true;
    }

    [[nodiscard]] std::uint64_t bytes_of(Budget id) const noexcept {
        return bytes_[static_cast<std::size_t>(id)];
    }

    // The SSD tier's non-byte facts: the spill's own triple, recorded by the owner that sees the
    // switch, and printed on the ssd line because that is where the tier lives.
    void set_kv_spill(const KvSpillView& view) noexcept { kv_spill_ = view; }
    [[nodiscard]] const KvSpillView& kv_spill() const noexcept { return kv_spill_; }

    // Bytes this tier has actually claimed: carved lines are excluded, because they are slices of a
    // line already counted.
    [[nodiscard]] std::uint64_t total(Tier tier) const noexcept {
        std::uint64_t sum = 0;
        for (const auto& row : kBudgetSlots) {
            if (row.tier != tier || row.carved_from != row.id || row.planned) { continue; }
            sum += bytes_of(row.id);
        }
        return sum;
    }

    // One block, one line per tier: every name in the closed list appears, so an absent consumer reads
    // as a zero with its reason instead of as a silence.
    void report() const {
        for (const auto tier : {Tier::Device, Tier::Host, Tier::Disk}) {
            std::fprintf(stderr, "[plan] %-6s cap=", tier == Tier::Device ? "device"
                                            : tier == Tier::Host ? "host"
                                                                 : "ssd");
            if (cap(tier) == kNoCap) {
                std::fprintf(stderr, "none");
            } else {
                std::fprintf(stderr, "%llu MiB", static_cast<unsigned long long>(cap(tier) >> 20));
            }
            for (const auto& row : kBudgetSlots) {
                if (row.tier != tier) { continue; }
                if (row.id == Budget::DiskKvSpill) {
                    // The spec's triple, not a byte count: the tier has no budget to divide, and a
                    // `=0` here would read as "measured and empty" rather than "a shape".
                    std::fprintf(stderr, " | %s={enabled=%d, cap=%s, reread=%s}", row.name,
                                 kv_spill_.enabled ? 1 : 0, kv_spill_.cap.c_str(),
                                 kv_spill_.reread.c_str());
                    continue;
                }
                const auto bytes = bytes_of(row.id);
                std::fprintf(stderr, " | %s=%llu", row.name,
                             static_cast<unsigned long long>(bytes >> 20));
                if (row.carved_from != row.id) { std::fputc('*', stderr); }
                if (row.planned) { std::fputc('~', stderr); }
                if (bytes == 0 && row.note != nullptr) {
                    std::fprintf(stderr, "(%s)", row.note);
                }
            }
            if (cap(tier) != kNoCap) {
                std::fprintf(stderr, " | claimed=%llu MiB",
                             static_cast<unsigned long long>(total(tier) >> 20));
            }
            std::fputc('\n', stderr);
        }
        std::fprintf(stderr,
                     "[plan] * = a slice of another line, not a claim; ~ = planned here, claimed by "
                     "its own owner; ssd holds no expert budget (the cold range is what the warm "
                     "tiers did not keep, not a knob) and its kv_spill is judged by re-reads, not by "
                     "a cap\n");
        std::fflush(stderr);
    }

private:
    static constexpr std::size_t index(Tier tier) noexcept {
        return tier == Tier::Device ? 0U : tier == Tier::Host ? 1U : 2U;
    }

    std::array<std::uint64_t, 3>            caps_{0, 0, kNoCap};
    std::array<std::uint64_t, kBudgetCount> bytes_{};
    KvSpillView                             kv_spill_{};
};

} // namespace ninfer::core
