#pragma once

// P7.1 (v10 8.4, V10-55): a capacity refusal is a bet, not a verdict.
//
// The 24 GB card has three consumers that compete for the same bytes when a run is tight: the
// expert pool, the KV pages every decode step reads, and the Host leg that can take either one's
// overflow. Today a run that cannot seat both ends dies at `attach_expert_cache()` with "expert
// cache holds less than one expert window", and a KV ask that misses its budget simply throws.
// Neither answer is the useful one: the question is WHICH consumer gets the Device, and that is a
// controller decision.
//
// The three bets (V10-55's A/B/C):
//   A  ExpertResident  the pool takes what it asks; KV takes the remainder (possibly zero, i.e. the
//                      whole KV leg moves to Host). Right when residency is what the run is short of.
//   B  KvFirst         the KV target is met first; the pool takes the remainder (it may then hold a
//                      partial layer, which the Host tier already covers). Right when locality is.
//   C  BothGive        neither gets its whole ask: the bytes above a floor are shared in proportion
//                      to the two remaining asks. Both sides miss a little; neither is starved.
//
// WHAT THE POOL ASKS FOR. "Experts fully resident" (A) means every bank's window, i.e. the whole
// routed set - not one window. One window is the pool's FLOOR (below it no routed layer can be
// served from the Device at all), and the two are separate fields for that reason: a request that
// named only the widest bank would report "both asks fit" on a card that is a gigabyte short of the
// expert set, which is exactly the run this decision exists for. `expert_window_slots` is the floor
// (and the alignment); `expert_full_slots` is the ask. A caller that only knows the window may
// leave the full count zero, and the ask then falls back to one window - degenerate, never wrong.
//
// `choice` names the PRIORITY that was applied, not the shape that fell out. When both asks fit,
// every mode returns A (experts fully resident) with `bet_exercised == false`, because there is no
// bet to take - that case is reported, not silently relabelled.
//
// THE ARENA'S FLOOR (V10-78). C's "both give a little" takes pages away from the KV leg, and a plan
// that has to serve the whole context resident cannot build that either: the arena's own minimum is
// priced from `page_count(max_context)` (program/planning/startup.cpp:1291), so a KV figure under
// it is refused at startup and the bytes the bet handed back are never released
// (model_instance.cpp:357). On 35B e2 at `--max-context 65536` the pinned arena is 1024 pages
// (1436 MiB) and that IS the floor, so C printed 1145 MiB while the run kept all 1024 pages. A
// letter that names a priority the engine does not apply is the same defect A had, so the KV side
// is never priced below `kv_floor_bytes` when no resident-KV leg exists: the pages keep the floor,
// the pool takes the remainder, and `choice` follows those bytes - B's shape, because that is the
// priority that was applied. The rule's own figure stays in `kv_target_bytes`, so a caller prints
// the target beside the realised split. A tiered leg names a floor of zero: there pages may leave
// the Device, and C remains a distinct arrangement.
//
// A'S PRECONDITION (V10-74). A pays for the pool out of the KV leg, so the pages the remainder does
// not cover have to live on Host and be read back when a step asks for them - and a step asks for
// every page of the history. That capability, a resident-KV leg, does not exist today: the Host KV
// tier holds demoted extents that no attention path reads back
// (models/qwen3_5/execution/kv_host_split_mirror.h:11), the one Host -> Device fetch lives inside
// the eviction-window contract (program/storage/sequence.cpp:1226), and a KV arena smaller than the
// context is refused outright for a model with full-attention layers
// (program/planning/startup.cpp:1251). `resident_kv_leg` is therefore FALSE by default, A is
// reported as not executable for that reason rather than as "the KV ask is below its floor", and
// the bet returns the arrangement the engine can actually run. A plan may set the flag to price a
// leg that does not exist yet (the tests do, with a leg they own); a run must not.
//
// The three modes are three different rules, not one formula renamed:
//   performance  experts first: pool = min(ask, budget); KV takes what is left - and since that
//                "left" is below the KV ask exactly when the bet is taken, this rule is only
//                offered when `resident_kv_leg` is set. Without it the pool takes what is left
//                after the KV ask, which is B's shape; the refusal names the leg.
//   stable       KV first:      kv = min(ask, budget - one slot); pool takes what is left.
//                (one slot, not one window: a partial pool is legal here - the Host tier covers the
//                rest of the layer, and refusing it would turn a workable run into a startup fault)
//   auto         KV first, gated by a coverage floor: KV gets its target only if the pool would
//                still reach `expert_usable_ratio` of its ask; otherwise the bytes above that
//                floor are shared in proportion to the two remaining asks. Measured evidence
//                outranks the structural rule and only when it exists: a high miss rate ranks
//                residency first (the performance rule), a high p99 ranks KV first (the stable one).
//
// Every rule is a function of the budget and the fixed asks, and every rule is monotone in the
// budget: a bigger Device never shrinks either side. That is a property of the arithmetic, not of
// the caller - `tests/runtime/test_capacity_bet.cpp` sweeps it. A mode CHANGE may move bytes,
// because that is what changing the bet means.
//
// Thresholds. `kDefaultCapacityBetThresholds` is a DEFAULT, not a measurement of this build: the
// numbers below are conservative round values chosen so the default bet behaves like the old
// KV-first split wherever the two overlap. They live in one struct so a run can replace all of them
// from one place, and so no branch below carries a number of its own.
//
// This header is intentionally pure: no CUDA, no allocation, no logging, no globals. The caller
// owns the Device query, the mode, the observed signals and the printing.

#include "core/controller_mode.h"

#include <algorithm>
#include <cstdint>

namespace ninfer::core {

enum class CapacityBet : std::uint8_t {
    ExpertResident = 0, // A: the pool wins, the KV leg gives way (possibly all the way to Host)
    KvFirst        = 1, // B: the KV target wins, the pool takes the remainder
    BothGive       = 2, // C: neither gets its whole ask; the surplus above a floor is shared
};

[[nodiscard]] inline constexpr const char* capacity_bet_name(CapacityBet bet) {
    switch (bet) {
    case CapacityBet::ExpertResident: return "A";
    case CapacityBet::KvFirst: return "B";
    case CapacityBet::BothGive: break;
    }
    return "C";
}

// The word the log uses next to the letter, so a reader does not have to hold the legend in mind.
[[nodiscard]] inline constexpr const char* capacity_bet_label(CapacityBet bet) {
    switch (bet) {
    case CapacityBet::ExpertResident: return "expert-resident";
    case CapacityBet::KvFirst: return "kv-first";
    case CapacityBet::BothGive: break;
    }
    return "both-give";
}

// What the controller has already measured. Negative means "not measured on this path" for every
// field - never zero, which would read as a measurement of failure (the same convention
// PolicySignals uses in models/qwen3_5/execution/expert_path_control.h).
struct CapacitySignals {
    double decode_tps    = -1.0;
    // Carried for the caller's log line. The ranking below uses the two costs a bet can move;
    // tps is the OUTCOME both of them move, so ranking on it would only restate the winner.
    double lambda_p99_ms = -1.0;
    double miss_rate     = -1.0;

    [[nodiscard]] bool has_miss_rate() const noexcept { return miss_rate >= 0.0; }
    [[nodiscard]] bool has_latency() const noexcept { return lambda_p99_ms >= 0.0; }
};

// The whole tuning surface of the bet. DEFAULT VALUES - see the header comment: conservative round
// numbers, not measurements of this build. `expert_usable_ratio` is the coverage below which a pool
// is not worth its bytes (a pool that misses most routes pays the PCIe leg anyway); 0.90 keeps the
// auto default close to the old KV-first split. The two evidence thresholds are the points where a
// measurement outranks the structural rule.
struct CapacityBetThresholds {
    // Share of one layer window the pool must reach for `auto` to accept a KV-first split.
    double expert_usable_ratio = 0.90;
    // Route-miss rate (0..1) above which residency is the binding constraint for `auto`.
    double auto_miss_rate_high = 0.20;
    // Decode p99 (ms) above which KV locality is the binding constraint for `auto`.
    double auto_latency_high_ms = 60.0;
};
inline constexpr CapacityBetThresholds kDefaultCapacityBetThresholds{};

// Everything the decision is allowed to read. Byte counts, not MiB: the caller already has bytes and
// a second unit is a second chance to be wrong by 1024.
struct CapacityBetRequest {
    // What the Device can give the pool and the KV pages TOGETHER, before the startup allowance.
    std::uint64_t device_budget_bytes = 0;
    // The planning allowance the device cap carries (Graph capture, weight staging). Deducted first,
    // exactly as `plan_device_split()` deducts it, so the two cannot disagree about the ceiling.
    std::uint64_t startup_headroom_bytes = 0;
    // The arena's own floor: the Device bytes a plan must keep for KV to serve the whole context.
    // A figure below it cannot be built without a resident-KV leg (THE ARENA'S FLOOR), so the KV
    // side is never priced below it. Zero means "no floor" - a tiered leg may move pages to Host.
    std::uint64_t kv_floor_bytes = 0;
    // One expert slot: the pool holds whole slots, and this is also the pool's alignment.
    std::uint64_t expert_slot_bytes = 0;
    // Slots one bank's window needs (the widest bank). This is the pool's FLOOR, not its ask: a
    // Device that cannot hold one window cannot serve a routed layer from video memory at all.
    std::uint64_t expert_window_slots = 0;
    // Slots the whole routed set needs, i.e. every bank's window - the pool's ASK, and what
    // "experts fully resident" means. Zero means "no set shape known": the ask is then one window.
    std::uint64_t expert_full_slots = 0;
    // The KV price the arena already resolved, expressed per token and per target.
    std::uint64_t kv_bytes_per_token = 0;
    std::uint64_t kv_target_tokens   = 0;
    // KV page granularity, for the KV figure's alignment. Zero means "do not align".
    std::uint64_t kv_alignment_bytes = 0;
    // Whether a resident-KV leg exists: attention runs with FEWER Device pages than the context
    // needs, and a page the Device does not hold is read back from Host when a step asks for it.
    // False is the engine as it is (see A'S PRECONDITION above), and it is what makes A a refusal
    // instead of a plan.
    bool            resident_kv_leg = false;
    ControllerMode  mode            = ControllerMode::Auto;
    CapacitySignals signals{};

    [[nodiscard]] std::uint64_t expert_ask_bytes() const noexcept {
        // The whole set when it is known, one window when it is not, one slot when neither is: each
        // fallback is the smallest claim that still lets the decision be taken.
        const std::uint64_t slots = expert_full_slots != 0     ? expert_full_slots
                                    : expert_window_slots != 0 ? expert_window_slots
                                                               : 1;
        return expert_slot_bytes * slots;
    }
    [[nodiscard]] std::uint64_t kv_ask_bytes() const noexcept {
        return kv_bytes_per_token * kv_target_tokens;
    }
};

// The sentence a caller prints when A was the answer the rule reached and could not be taken. It
// names the missing capability rather than a byte count, because the byte count was never the
// problem: the pages it priced away have no reader once they leave the Device.
inline constexpr const char* kExpertResidentNeedsKvLeg =
    "A not executable: no resident-KV leg (evicted pages are not read back)";

struct CapacityBetPlan {
    std::uint64_t expert_pool_bytes = 0;
    std::uint64_t kv_pages_bytes    = 0;
    CapacityBet   choice            = CapacityBet::ExpertResident;
    // False only when the Device cannot hold even one expert slot after the startup allowance.
    // Everything else is answerable: a bet is exactly the statement that a tight budget has an
    // answer that is not "refuse".
    bool        feasible      = false;
    // True when the budget could not seat both asks, i.e. when a bet was actually taken. A caller
    // that only wants to log a tight run can gate on this instead of re-deriving it.
    bool        bet_exercised = false;
    const char* reason        = "no request";

    // The KV figure the rule reached before the arena's floor raised it, and whether it did. False
    // (and the target equal to `kv_pages_bytes`) when the floor did not bind. A caller prints the
    // pair, never the target alone, so the difference stays visible (THE ARENA'S FLOOR).
    std::uint64_t kv_target_bytes  = 0;
    bool          kv_floor_applied = false;

    // Echo of the request, so a caller logs one object instead of two, and - when A was on the
    // table and could not be taken - the sentence that says why, ready to print. A null refusal
    // means A was never refused, not that it was taken.
    bool        resident_kv_leg         = false;
    const char* expert_resident_refusal = nullptr;

    [[nodiscard]] std::uint64_t device_total_bytes() const noexcept {
        return expert_pool_bytes + kv_pages_bytes;
    }
};

namespace detail {

// (a * b) / d with the guarantee b <= d, which is all this decision needs: the split shares are
// both at most the sum they are divided by. Written this way because the 64-bit product would
// overflow long before the quotient does (24 GiB x 24 GiB), and because a double round-trip would
// make a one-byte jitter visible in the monotonicity sweep.
[[nodiscard]] inline std::uint64_t muldiv_le(std::uint64_t a, std::uint64_t b,
                                             std::uint64_t d) noexcept {
    if (d == 0) { return 0; }
    return (a / d) * b + ((a % d) * b) / d;
}

[[nodiscard]] inline std::uint64_t round_down(std::uint64_t bytes, std::uint64_t unit) noexcept {
    return unit == 0 ? bytes : bytes / unit * unit;
}

// Ratios are quantised to 1e-4 so the decision is integer-exact: `0.90 * bytes` through a double is
// one ulp away from a different byte count, and the tests assert byte counts.
[[nodiscard]] inline std::uint64_t ratio_bp(double ratio) noexcept {
    const double clamped = std::min(1.0, std::max(0.0, ratio));
    return static_cast<std::uint64_t>(clamped * 10000.0 + 0.5);
}

} // namespace detail

// The decision. Pure, monotone in the budget, and the only place the three rules exist.
[[nodiscard]] inline CapacityBetPlan
plan_capacity_bet(const CapacityBetRequest& request,
                  const CapacityBetThresholds& thresholds = kDefaultCapacityBetThresholds) {
    CapacityBetPlan plan;
    if (request.expert_slot_bytes == 0) {
        plan.reason = "no expert slot price: the pool cannot be sized";
        return plan;
    }
    const std::uint64_t headroom =
        std::min(request.startup_headroom_bytes, request.device_budget_bytes);
    const std::uint64_t budget  = request.device_budget_bytes - headroom;
    const std::uint64_t experts = request.expert_ask_bytes();
    const std::uint64_t kv      = request.kv_ask_bytes();
    if (budget < request.expert_slot_bytes) {
        plan.reason = "the Device cannot hold one expert slot after the startup allowance";
        return plan;
    }
    plan.feasible        = true;
    plan.resident_kv_leg = request.resident_kv_leg;
    if (experts + kv <= budget) {
        // Nothing is scarce, so nothing is bet. Reported as A because "the experts are resident" is
        // exactly what A names, and because returning a mode-dependent answer here would turn a
        // tautology into a preference.
        plan.expert_pool_bytes = experts;
        plan.kv_pages_bytes    = kv;
        plan.choice            = CapacityBet::ExpertResident;
        plan.bet_exercised     = false;
        plan.reason            = "the whole ask fits: experts and KV are both resident, nothing"
                                 " spills";
        plan.expert_pool_bytes =
            detail::round_down(plan.expert_pool_bytes, request.expert_slot_bytes);
        plan.kv_pages_bytes =
            detail::round_down(plan.kv_pages_bytes, request.kv_alignment_bytes);
        return plan;
    }
    plan.bet_exercised = true;

    const auto kv_first = [&] {
        // stable: the KV target is met first, leaving one expert slot - a pool of one slot is still
        // a pool (the Host tier serves the rest of the layer), and refusing it would be the startup
        // fault this decision exists to avoid.
        const std::uint64_t reserve = std::min(budget, request.expert_slot_bytes);
        plan.kv_pages_bytes        = std::min(kv, budget - reserve);
        plan.expert_pool_bytes     = std::min(experts, budget - plan.kv_pages_bytes);
        plan.choice = plan.kv_pages_bytes >= kv ? CapacityBet::KvFirst : CapacityBet::BothGive;
        plan.reason = plan.choice == CapacityBet::KvFirst
                          ? "KV first: the pages keep their target, the pool streams the remainder"
                          : "KV first: even one expert slot is taken from the KV target";
    };
    const auto experts_first = [&] {
        // performance: the pool takes its ask, and KV takes the remainder - which, for every budget
        // this rule is reached with, is less than the KV ask. That answer is A, and A is only
        // runnable with a resident-KV leg: the pages it prices away have no reader once they leave
        // the Device. Without one the pool takes what is left after the KV ask (B's shape) and the
        // refusal says which capability is missing, instead of pricing a leg that is not there.
        if (!request.resident_kv_leg) {
            kv_first();
            plan.reason =
                plan.choice == CapacityBet::KvFirst
                    ? "experts first, but A needs a resident-KV leg: the KV leg keeps its target "
                      "and the pool takes the remainder"
                    : "experts first, but A needs a resident-KV leg: not even the KV target fits "
                      "beside one expert slot";
            plan.expert_resident_refusal = kExpertResidentNeedsKvLeg;
            return;
        }
        // The pool takes its ask if it fits, and KV takes the remainder. When the ask does not fit,
        // KV keeps nothing but the bytes the pool cannot use - which is the same rule, applied to a
        // smaller pool.
        plan.expert_pool_bytes = std::min(experts, budget);
        plan.kv_pages_bytes    = std::min(kv, budget - plan.expert_pool_bytes);
        plan.choice = plan.expert_pool_bytes >= experts ? CapacityBet::ExpertResident
                                                        : CapacityBet::BothGive;
        plan.reason = plan.choice == CapacityBet::ExpertResident
                          ? "experts first: the pool seats its whole window, KV takes the remainder"
                          : "experts first: the window does not fit, so KV keeps only the leftovers";
    };

    switch (request.mode) {
    case ControllerMode::Performance:
        experts_first();
        break;
    case ControllerMode::Stable:
        kv_first();
        break;
    case ControllerMode::Auto: {
        // Evidence outranks the structural rule, and only when it exists. Both measured and both
        // high: the larger relative excess wins, because that is the one more likely to be the
        // actual limit; a tie goes to latency, which is the SLO.
        const bool miss_high = request.signals.has_miss_rate() &&
                               request.signals.miss_rate >= thresholds.auto_miss_rate_high;
        const bool latency_high = request.signals.has_latency() &&
                                  request.signals.lambda_p99_ms >= thresholds.auto_latency_high_ms;
        if (miss_high && (!latency_high || request.signals.miss_rate / thresholds.auto_miss_rate_high >=
                                               request.signals.lambda_p99_ms /
                                                   thresholds.auto_latency_high_ms)) {
            experts_first();
            // The refusal already carries the sentence a reader needs; the mode's own wording would
            // bury it, so it is only written when A was actually taken.
            if (request.resident_kv_leg) {
                plan.reason = "auto: the measured miss rate ranks residency first, so the pool "
                              "takes its window";
            }
            break;
        }
        if (latency_high) {
            kv_first();
            plan.reason = "auto: the measured p99 ranks KV locality first, so the pages keep their "
                          "target";
            break;
        }
        // The structural rule: KV may have its target only if what is left still reaches the usable
        // expert floor. Below that floor the bytes above it are shared in proportion to the two
        // remaining asks - which is what makes "both give a little" a rule and not a slogan.
        const std::uint64_t usable =
            detail::muldiv_le(experts, detail::ratio_bp(thresholds.expert_usable_ratio), 10000);
        if (budget <= usable) {
            plan.expert_pool_bytes = std::min(experts, budget);
            plan.kv_pages_bytes    = budget - plan.expert_pool_bytes;
            plan.choice =
                plan.kv_pages_bytes == 0 ? CapacityBet::ExpertResident : CapacityBet::BothGive;
            // Zero is its own answer, not a small number: a KV leg with no Device bytes has to live
            // on the Host, and the reason says so instead of claiming it "takes the rest".
            plan.reason =
                plan.kv_pages_bytes == 0
                    ? "auto: the whole budget is under the pool's usable floor, so the pool "
                      "takes it all and KV is left nothing - only a Host-resident KV leg "
                      "could take this answer"
                    : "auto: the KV target cannot be met without dropping the pool below "
                      "the usable floor, so the pool keeps what it can and KV takes the "
                      "rest";
            break;
        }
        const std::uint64_t surplus       = budget - usable;
        const std::uint64_t expert_remain = experts > usable ? experts - usable : 0;
        const std::uint64_t share_total   = expert_remain + kv;
        const std::uint64_t pool_share =
            detail::muldiv_le(surplus, expert_remain, share_total);
        plan.expert_pool_bytes = usable + pool_share;
        plan.kv_pages_bytes    = budget - plan.expert_pool_bytes;
        plan.choice = plan.kv_pages_bytes == 0   ? CapacityBet::ExpertResident
                      : plan.kv_pages_bytes >= kv ? CapacityBet::KvFirst
                                                  : CapacityBet::BothGive;
        plan.reason = plan.choice == CapacityBet::KvFirst
                          ? "auto: the pool clears its usable floor, so KV keeps its target"
                          : "auto: each side keeps its floor and the surplus is shared in proportion";
        break;
    }
    }
    // A is the one answer that costs attention its pages: either "the pool seats its whole ask and
    // the KV leg takes the remainder" (the experts-first rule above, which refuses to price that
    // without the leg) or "the KV leg is left no Device bytes at all" (the structural rule, when
    // the budget is under the pool's usable floor). Both need a page the Device does not hold to
    // be read back for attention, so without a resident-KV leg A is not executable: the shape is
    // reported as what it is - C, neither side whole - with the refusal beside it. The bytes are
    // left as the rule computed them; a re-solve would not give the KV leg pages it cannot read
    // back either.
    if (plan.bet_exercised && plan.choice == CapacityBet::ExpertResident &&
        !request.resident_kv_leg) {
        plan.choice                  = CapacityBet::BothGive;
        plan.expert_resident_refusal = kExpertResidentNeedsKvLeg;
    }
    plan.expert_pool_bytes = std::min(experts, detail::round_down(plan.expert_pool_bytes,
                                                                 request.expert_slot_bytes));
    plan.kv_pages_bytes    = std::min(kv, detail::round_down(plan.kv_pages_bytes,
                                                          request.kv_alignment_bytes));
    // Rounding must not take the pool below one slot when the Device can hold one: the caller turns
    // "less than a slot" into a startup failure, and that failure is what this decision is for.
    if (plan.expert_pool_bytes < request.expert_slot_bytes &&
        budget >= request.expert_slot_bytes) {
        plan.expert_pool_bytes = request.expert_slot_bytes;
        plan.kv_pages_bytes    = budget > request.expert_slot_bytes ? budget - request.expert_slot_bytes
                                                                    : 0;
    }
    // THE ARENA'S FLOOR: a KV figure under it is not a smaller KV leg, it is a plan the arena can
    // not build (startup prices the minimum from max_context), so those bytes are never released
    // and the printed split would describe a run that does not exist. The rule's target is kept;
    // the bytes become the arrangement that runs - the pages keep the floor, the pool takes the
    // rest - and the letter follows those bytes, because the line must name the priority applied.
    // A leg that reads pages back from Host has no floor here: the caller names zero for it.
    const std::uint64_t kv_floor = std::min(request.kv_floor_bytes, kv);
    if (plan.bet_exercised && !request.resident_kv_leg && plan.kv_pages_bytes < kv_floor) {
        const std::uint64_t reserve = std::min(budget, request.expert_slot_bytes);
        plan.kv_target_bytes        = plan.kv_pages_bytes;
        plan.kv_floor_applied       = true;
        // The floor may not eat the pool's last slot, but a Device that cannot hold both is a
        // startup fault in its own right - the caller fails loud there, it does not run this plan.
        plan.kv_pages_bytes    = std::min(kv_floor, budget - reserve);
        plan.expert_pool_bytes = std::min(experts, detail::round_down(budget - plan.kv_pages_bytes,
                                                                      request.expert_slot_bytes));
        plan.choice = plan.kv_pages_bytes >= kv ? CapacityBet::KvFirst : CapacityBet::BothGive;
        plan.reason = "the KV target is below the arena's floor (the pages a resident context "
                      "needs): the pages keep their floor and the pool takes the remainder";
    }
    // After rounding the two claims may leave a byte or two unclaimed (an eight-byte alignment).
    // That is reported as a smaller total, never silently handed to the other consumer: both lines
    // are printed, so the difference stays visible.
    return plan;
}

} // namespace ninfer::core
