// P7.1 (v10 8.4, V10-55): the capacity bet, without a Device.
//
// The decision is pure, so this test is its whole contract: which bet each mode takes, that no bet
// ever claims more than the Device has, and that a bigger Device never shrinks either side. The
// numbers are round (1 MiB slots, a 20 GiB window, a 6 GiB KV target) because the assertions are
// about the RULES; a run's own geometry arrives through the same three fields.

#include "core/capacity_bet.h"

#include <cstdint>
#include <cstdio>

namespace {

using ninfer::ControllerMode;
using ninfer::core::CapacityBet;
using ninfer::core::CapacityBetPlan;
using ninfer::core::CapacityBetRequest;
using ninfer::core::plan_capacity_bet;

constexpr std::uint64_t kMib = 1024ULL * 1024ULL;
constexpr std::uint64_t kGib = 1024ULL * kMib;

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

// 20 GiB of expert window (20480 slots of 1 MiB) and a 6 GiB KV target.
CapacityBetRequest canonical(std::uint64_t budget, ControllerMode mode) {
    CapacityBetRequest request;
    request.device_budget_bytes    = budget;
    request.startup_headroom_bytes = 0;
    request.expert_slot_bytes      = kMib;
    request.expert_window_slots    = 20 * 1024; // 20 GiB
    request.kv_bytes_per_token     = 6 * kMib;
    request.kv_target_tokens       = 1024; // 6 GiB
    request.mode                   = mode;
    return request;
}

void test_a_slack_device_takes_no_bet() {
    // 20 + 6 GiB of ask against 32 GiB: every mode answers the same, and says no bet was taken.
    for (const auto mode :
         {ControllerMode::Auto, ControllerMode::Stable, ControllerMode::Performance}) {
        const auto plan = plan_capacity_bet(canonical(32 * kGib, mode));
        check(plan.feasible, "a slack Device is feasible");
        check(!plan.bet_exercised, "a slack Device exercises no bet");
        check(plan.expert_pool_bytes == 20 * kGib, "the pool gets its whole window");
        check(plan.kv_pages_bytes == 6 * kGib, "KV gets its whole target");
        check(plan.choice == CapacityBet::ExpertResident,
              "a slack Device reports A (both resident)");
    }
}

void test_tight_device_picks_three_different_bets() {
    // 21 GiB against a 26 GiB ask: the one Device for which the three modes must disagree.
    const auto performance = plan_capacity_bet(canonical(21 * kGib, ControllerMode::Performance));
    check(performance.feasible && performance.bet_exercised, "the tight case takes a bet");
    check(performance.choice == CapacityBet::ExpertResident, "performance picks A: experts first");
    check(performance.expert_pool_bytes == 20 * kGib, "A seats the whole window");
    check(performance.kv_pages_bytes == 1 * kGib, "A leaves KV the remainder");

    const auto stable = plan_capacity_bet(canonical(21 * kGib, ControllerMode::Stable));
    check(stable.choice == CapacityBet::KvFirst, "stable picks B: KV first");
    check(stable.kv_pages_bytes == 6 * kGib, "B keeps the KV target");
    check(stable.expert_pool_bytes == 15 * kGib, "B leaves the pool the remainder");

    const auto automatic = plan_capacity_bet(canonical(21 * kGib, ControllerMode::Auto));
    check(automatic.choice == CapacityBet::BothGive, "auto picks C: both give a little");
    // 90% floor = 18 GiB; the 3 GiB above it is shared 2:6 between the pool's remaining 2 GiB of
    // ask and the KV target, so the pool ends at 18.75 GiB and KV at 2.25 GiB.
    check(automatic.expert_pool_bytes == 20 * kGib - 5 * kGib / 4,
          "C gives the pool its floor plus a proportional share");
    check(automatic.kv_pages_bytes == 21 * kGib - automatic.expert_pool_bytes,
          "C gives KV everything the pool did not take");
    check(automatic.expert_pool_bytes + automatic.kv_pages_bytes == 21 * kGib,
          "C splits the whole budget between the two consumers");

    check(performance.expert_pool_bytes != stable.expert_pool_bytes &&
              stable.expert_pool_bytes != automatic.expert_pool_bytes,
          "the three modes do not collapse to one number");
}

void test_no_bet_claims_more_than_the_device_has() {
    for (const auto mode :
         {ControllerMode::Auto, ControllerMode::Stable, ControllerMode::Performance}) {
        for (std::uint64_t budget = kMib; budget <= 30 * kGib; budget += 256 * kMib) {
            const auto request = canonical(budget, mode);
            const auto plan    = plan_capacity_bet(request);
            if (!plan.feasible) { continue; }
            check(plan.device_total_bytes() + request.startup_headroom_bytes <= budget,
                  "a bet never claims more than the Device budget");
            check(plan.expert_pool_bytes <= request.expert_ask_bytes(),
                  "a bet never claims more than the pool asked for");
            check(plan.kv_pages_bytes <= request.kv_ask_bytes(),
                  "a bet never claims more than KV asked for");
        }
    }
}

void test_sizes_are_monotone_in_the_budget() {
    // The property the three rules were chosen for: a bigger Device never shrinks either side.
    // It is what lets the same decision be re-taken after a measurement without the plan moving
    // backwards, and it is asserted per mode because each rule must hold on its own.
    for (const auto mode :
         {ControllerMode::Auto, ControllerMode::Stable, ControllerMode::Performance}) {
        CapacityBetPlan previous = plan_capacity_bet(canonical(kMib, mode));
        for (std::uint64_t budget = kMib + 128 * kMib; budget <= 32 * kGib; budget += 128 * kMib) {
            const auto plan = plan_capacity_bet(canonical(budget, mode));
            check(plan.expert_pool_bytes >= previous.expert_pool_bytes,
                  "the expert pool never shrinks as the Device grows");
            check(plan.kv_pages_bytes >= previous.kv_pages_bytes,
                  "the KV target never shrinks as the Device grows");
            previous = plan;
        }
    }
}

void test_headroom_is_deducted_before_the_bet() {
    auto request                   = canonical(21 * kGib, ControllerMode::Stable);
    request.startup_headroom_bytes = 1 * kGib;
    const auto plan                = plan_capacity_bet(request);
    check(plan.feasible, "a run with headroom is still feasible");
    check(plan.device_total_bytes() + request.startup_headroom_bytes <=
              request.device_budget_bytes,
          "the startup allowance comes out of the same budget");
    check(plan.device_total_bytes() <= 20 * kGib, "headroom shrinks what is split");
}

void test_a_device_that_cannot_hold_one_slot_is_infeasible() {
    auto request = canonical(0, ControllerMode::Auto);
    check(!plan_capacity_bet(request).feasible, "a zero budget holds no slot");

    request.device_budget_bytes = kMib - 1;
    const auto plan             = plan_capacity_bet(request);
    check(!plan.feasible, "less than one slot is infeasible");
    check(plan.expert_pool_bytes == 0 && plan.kv_pages_bytes == 0,
          "an infeasible bet claims nothing");

    request.device_budget_bytes = kMib;
    const auto one              = plan_capacity_bet(request);
    check(one.feasible && one.expert_pool_bytes == kMib, "exactly one slot is the floor");
    check(one.kv_pages_bytes == 0, "at the floor KV gets nothing");
}

void test_the_ask_is_the_whole_set_not_one_window() {
    // A routed set of 20480 slots (40 banks) behind a 256-slot window: the whole set is 20 GiB, the
    // window is 256 MiB. Naming only the window makes a 21 GiB Device look slack, which is the
    // reading V10-55 exists to correct - "experts fully resident" is the set, not one bank.
    auto request                = canonical(21 * kGib, ControllerMode::Stable);
    request.expert_window_slots = 256;
    request.expert_full_slots   = 20 * 1024;
    const auto whole            = plan_capacity_bet(request);
    check(whole.bet_exercised, "the whole routed set does not fit even though one window would");
    check(whole.kv_pages_bytes == 6 * kGib, "stable keeps the KV target");
    check(whole.expert_pool_bytes == 15 * kGib, "the pool streams the rest of the set");

    request.expert_full_slots = 0;
    check(!plan_capacity_bet(request).bet_exercised, "one window alone is not the ask");

    // The same shape under the three modes: the bet still has to separate them.
    auto full              = request;
    full.expert_full_slots = 20 * 1024;
    full.mode              = ControllerMode::Performance;
    check(plan_capacity_bet(full).choice == CapacityBet::ExpertResident,
          "performance seats the whole set");
    full.mode = ControllerMode::Stable;
    check(plan_capacity_bet(full).choice == CapacityBet::KvFirst, "stable keeps the pages");
    full.mode = ControllerMode::Auto;
    check(plan_capacity_bet(full).choice == CapacityBet::BothGive, "auto splits the surplus");
}

void test_measured_evidence_steers_auto() {
    // Nothing measured: the structural rule (C at 21 GiB).
    const auto unknown = plan_capacity_bet(canonical(21 * kGib, ControllerMode::Auto));
    check(unknown.choice == CapacityBet::BothGive,
          "no evidence leaves auto on its structural rule");

    // A high miss rate says residency is the limit: auto takes the performance rule.
    auto misses              = canonical(21 * kGib, ControllerMode::Auto);
    misses.signals.miss_rate = 0.35;
    const auto residency     = plan_capacity_bet(misses);
    check(residency.choice == CapacityBet::ExpertResident, "a high miss rate ranks residency first");
    check(residency.expert_pool_bytes == 20 * kGib, "residency-first seats the whole window");

    // A high p99 says locality is the limit: auto takes the stable rule.
    auto latency                  = canonical(21 * kGib, ControllerMode::Auto);
    latency.signals.lambda_p99_ms = 90.0;
    const auto locality           = plan_capacity_bet(latency);
    check(locality.choice == CapacityBet::KvFirst, "a high p99 ranks KV locality first");
    check(locality.kv_pages_bytes == 6 * kGib, "locality-first keeps the KV target");

    // Both high: the larger relative excess wins (0.40 / 0.20 beats 70 / 60).
    auto both                  = canonical(21 * kGib, ControllerMode::Auto);
    both.signals.miss_rate     = 0.40;
    both.signals.lambda_p99_ms = 70.0;
    check(plan_capacity_bet(both).choice == CapacityBet::ExpertResident,
          "the larger relative excess breaks the tie between the two criteria");

    // A measured but unremarkable pair leaves the structural rule in force.
    auto mild                  = canonical(21 * kGib, ControllerMode::Auto);
    mild.signals.miss_rate     = 0.05;
    mild.signals.lambda_p99_ms = 20.0;
    check(plan_capacity_bet(mild).choice == CapacityBet::BothGive,
          "measurements below both thresholds do not override the rule");
}

void test_the_pool_stays_slot_aligned() {
    for (const auto mode :
         {ControllerMode::Auto, ControllerMode::Stable, ControllerMode::Performance}) {
        for (std::uint64_t budget = kMib; budget <= 26 * kGib; budget += 64 * kMib) {
            auto request = canonical(budget, mode);
            // A slot that does not divide the budget evenly: the pool must still hold whole slots.
            request.expert_slot_bytes = 1536 * 1024;
            const auto plan           = plan_capacity_bet(request);
            if (!plan.feasible) { continue; }
            check(plan.expert_pool_bytes % request.expert_slot_bytes == 0,
                  "the pool holds whole slots");
            check(plan.expert_pool_bytes >= request.expert_slot_bytes,
                  "a feasible bet always seats at least one slot");
        }
    }
}

void test_kv_alignment_when_a_page_size_is_named() {
    auto request               = canonical(21 * kGib, ControllerMode::Stable);
    request.kv_alignment_bytes = 2 * kMib;
    const auto plan            = plan_capacity_bet(request);
    check(plan.kv_pages_bytes % request.kv_alignment_bytes == 0,
          "the KV figure is a whole number of pages");
}

} // namespace

int main() {
    test_a_slack_device_takes_no_bet();
    test_tight_device_picks_three_different_bets();
    test_no_bet_claims_more_than_the_device_has();
    test_sizes_are_monotone_in_the_budget();
    test_headroom_is_deducted_before_the_bet();
    test_a_device_that_cannot_hold_one_slot_is_infeasible();
    test_the_ask_is_the_whole_set_not_one_window();
    test_measured_evidence_steers_auto();
    test_the_pool_stays_slot_aligned();
    test_kv_alignment_when_a_page_size_is_named();
    if (failures == 0) { std::printf("ok\n"); }
    return failures == 0 ? 0 : 1;
}
