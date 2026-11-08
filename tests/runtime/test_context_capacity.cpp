// T156 (upstream 049d5fb4): joint KV / StateImage / workspace capacity planning.

#include "runtime/engine/context_capacity.h"

#include <cstdio>
#include <limits>

namespace {

using ninfer::ContextCacheOptions;
using ninfer::KvCapacityMode;
using ninfer::KvCapacityPolicy;
using ninfer::runtime::ContextCapacity;
using ninfer::runtime::ContextCapacityCurve;
using ninfer::runtime::ContextCapacityPlan;
using ninfer::runtime::SequenceCapacityCurve;

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

template <class Call>
bool throws_invalid(Call&& call) {
    try {
        call();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

ContextCapacityCurve curve() {
    ContextCapacityCurve out;
    out.kv = SequenceCapacityCurve{
        .main_page_tokens                     = 128,
        .minimum_main_page_groups             = 16,
        .maximum_main_page_groups             = 1000,
        .minimum_device_reservation_bytes     = 1'000'000,
        .bytes_per_additional_main_page_group = 10'000,
    };
    out.minimum_state_slots           = 8;
    out.bytes_per_additional_state_slot = 50'000;
    out.workspace_bytes               = 100'000;
    out.control_bytes_per_page_group  = 64;
    out.control_bytes_per_state_slot  = 512;
    return out;
}

void test_breakdown_is_exact() {
    const auto c = curve();
    const ContextCapacity capacity{16, 24};
    // KV anchor 1,000,000 + 16 extra images x 50,000 + workspace 100,000 + 16 x 512 control.
    check(c.reservation_bytes(capacity) == 1'800'000, "KV anchor plus StateImage strides");
    check(c.control_bytes(capacity) == 8192, "control grows with slots beyond the lanes");
    check(c.total_bytes(capacity) == 1'908'192, "workspace is charged once");
}

void test_control_and_workspace_are_charged() {
    const auto c = curve();
    const ContextCapacity capacity{20, 8};
    check(c.reservation_bytes(capacity) == 1'040'000, "page growth uses the KV stride");
    check(c.control_bytes(capacity) == 256, "control grows with page groups");
    check(c.total_bytes(capacity) == 1'140'256, "workspace is not free");
}

void test_explicit_capacity_is_pinned() {
    ContextCacheOptions cache;
    cache.enabled            = true;
    cache.device_state_slots = 8;
    const auto plan = plan_context_capacity(KvCapacityPolicy::explicit_capacity(2048), cache,
                                            curve(), 2'000'000, 2048);
    check(plan.capacity.primary_page_groups == 16, "explicit tokens pin the page count");
    check(plan.capacity.device_state_slots == 16, "explicit extra slots add to the lanes");
    check(plan.resolved_tokens == 2048, "resolved tokens follow the page count");
    check(!plan.checkpoints_targeted, "explicit capacity never grows the checkpoint set");
    check(plan.total_reservation_bytes == 1'504'096, "explicit total is the exact breakdown");
    check(plan.planned_slack_bytes == 495'904, "slack is what the budget has left");
}

void test_automatic_spends_on_slots_then_pages() {
    ContextCacheOptions cache;
    cache.enabled            = true;
    cache.device_state_slots = std::nullopt;
    const auto plan = plan_context_capacity(KvCapacityPolicy::automatic(0), cache, curve(),
                                            20'000'000, 640);
    check(plan.checkpoints_targeted, "automatic sizing targets the checkpoint density");
    // 2 x 8 lanes + ceil(resolved tokens / 640).
    const std::uint32_t expected_target =
        16U + 1U + (plan.resolved_tokens - 1U) / 640U;
    check(plan.target_slots == expected_target,
          "checkpoint target is two lane sets plus one per interval");
    check(plan.capacity.device_state_slots >= 20, "the target slots fit");
    check(plan.capacity.primary_page_groups > 16, "leftover bytes become page groups");
    check(plan.total_reservation_bytes <= 20'000'000, "the plan fits the budget");
    check(plan.planned_slack_bytes < 10'100, "the plan does not strand a whole page stride");
}

void test_explicit_state_slots_suppress_the_target() {
    ContextCacheOptions cache;
    cache.enabled            = true;
    cache.device_state_slots = 2;
    const auto plan = plan_context_capacity(KvCapacityPolicy::automatic(0), cache, curve(),
                                            20'000'000, 640);
    check(!plan.checkpoints_targeted, "an explicit slot count wins over the heuristic");
    check(plan.capacity.device_state_slots == 10, "lanes plus the requested extra");
}

void test_interval_zero_disables_the_target() {
    ContextCacheOptions cache;
    cache.enabled = true;
    const auto plan = plan_context_capacity(KvCapacityPolicy::automatic(0), cache, curve(),
                                            20'000'000, 0);
    check(!plan.checkpoints_targeted, "no interval, no checkpoint slots");
    check(plan.capacity.device_state_slots == 8, "only the execution lanes are planned");
}

void test_infeasible_and_invalid() {
    ContextCacheOptions cache;
    check(throws_invalid([&] {
              (void)plan_context_capacity(KvCapacityPolicy::explicit_capacity(2048), cache,
                                          curve(), 1'000'000, 640);
          }),
          "a plan beyond the budget throws with the numbers");
    auto no_stride = curve();
    no_stride.bytes_per_additional_state_slot = 0;
    check(throws_invalid([&] {
              (void)plan_context_capacity(KvCapacityPolicy::explicit_capacity(2048), cache,
                                          no_stride, 2'000'000, 640);
          }),
          "a StateImage without a stride is invalid");
    auto no_lanes = curve();
    no_lanes.minimum_state_slots = 0;
    check(throws_invalid([&] { (void)no_lanes.total_bytes(ContextCapacity{16, 0}); }),
          "zero lanes is not a domain");

    ContextCapacityCurve c = curve();
    check(throws_invalid([&] { (void)c.reservation_bytes(ContextCapacity{15, 8}); }),
          "pages below the curve minimum are refused");
    check(throws_invalid([&] { (void)c.reservation_bytes(ContextCapacity{16, 7}); }),
          "slots below the lane count are refused");
    check(throws_invalid([&] { (void)c.reservation_bytes(ContextCapacity{16, INT32_MAX + 1U}); }),
          "slots beyond int32 are refused");
}

} // namespace

int main() {
    test_breakdown_is_exact();
    test_control_and_workspace_are_charged();
    test_explicit_capacity_is_pinned();
    test_automatic_spends_on_slots_then_pages();
    test_explicit_state_slots_suppress_the_target();
    test_interval_zero_disables_the_target();
    test_infeasible_and_invalid();
    if (failures == 0) { std::printf("ok\n"); }
    return failures == 0 ? 0 : 1;
}
