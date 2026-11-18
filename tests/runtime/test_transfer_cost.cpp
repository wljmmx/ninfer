// V10-82: the per-step cost model behind the capacity bet, without a Device and without a clock.
// V10-83: and the two terms the streaming leg turned out to have.
//
// The model is the whole of what the bet compares, so what this test pins is its shape, not a
// number: a step that owes nothing costs nothing, more traffic never costs less, and each leg is
// priced with its OWN rate (a rate that belongs to the other leg must not move it). The arithmetic
// is checked once against a hand-computed millisecond figure, because the second failure mode of a
// model like this is a unit slip that every monotonicity assertion happily passes. The V10-83 terms
// are pinned the same way: the disk and the link combine in series (one byte pays both), the
// per-layer round trip is charged only to layers that upload, and an unmeasured disk leg leaves the
// pre-V10-83 single-rate pipeline rather than a free read.

#include "core/transfer_cost.h"

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

using ninfer::core::effective_rate_bytes_per_s;
using ninfer::core::kStreamLayerFixedMs;
using ninfer::core::price_step_transfers;
using ninfer::core::StepTransferGeometry;
using ninfer::core::TransferRates;

constexpr std::uint64_t kMib = 1024ULL * 1024ULL;

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

void check_close(double actual, double expected, double tolerance, const char* what) {
    if (!(std::fabs(actual - expected) <= tolerance)) {
        std::fprintf(stderr, "FAIL: %s (got %.6f, want %.6f)\n", what, actual, expected);
        ++failures;
    }
}

// 1 MiB pages, 1 MiB slots, 20 and 10 GB/s - round numbers, one per leg, so a leg priced with the
// wrong rate is visible as a factor of two.
TransferRates rates() {
    TransferRates out;
    out.page_in_bytes_per_s = 20e9;
    out.expert_bytes_per_s  = 10e9;
    return out;
}

StepTransferGeometry geometry() {
    StepTransferGeometry out;
    out.page_bytes            = kMib;
    out.pages_addressed       = 5120;
    out.pages_resident        = 1024;
    out.slot_bytes            = kMib;
    out.experts_routed        = 512;
    out.expert_slots_resident = 128;
    return out;
}

void test_the_two_bills_are_pages_over_page_rate_and_slots_over_slot_rate() {
    const auto cost = price_step_transfers(geometry(), rates());
    check(cost.pages_fetched == 4096, "the page bill counts the pages the Device does not hold");
    check(cost.expert_misses == 384, "the slot bill counts the routes the pool does not hold");
    // 4096 MiB at 20 GB/s = 0.2147 s = 214.75 ms; 384 MiB at 10 GB/s = 40.27 ms.
    check_close(cost.page_in_ms, 4096.0 * static_cast<double>(kMib) / 20e9 * 1000.0, 1e-6,
                "the page bill is bytes over the page-in rate, in milliseconds");
    check_close(cost.expert_ms, 384.0 * static_cast<double>(kMib) / 10e9 * 1000.0, 1e-6,
                "the slot bill is bytes over the expert rate, in milliseconds");
    check_close(cost.total_ms(), cost.page_in_ms + cost.expert_ms, 1e-9,
                "the step owes the sum of the two legs");
}

void test_a_step_that_owes_nothing_costs_nothing() {
    auto full                  = geometry();
    full.pages_resident        = full.pages_addressed;
    full.expert_slots_resident = full.experts_routed;
    // 40 routed layers and still nothing owed: the fixed term belongs to layers that UPLOAD, so a
    // pool that seats every route must not be charged for the round trips it never makes.
    full.expert_layers         = 40;
    const auto cost            = price_step_transfers(full, rates());
    check(cost.pages_fetched == 0 && cost.expert_misses == 0,
          "a fully resident split fetches and streams nothing");
    check(cost.page_in_ms == 0.0 && cost.expert_ms == 0.0 && cost.expert_fixed_ms == 0.0,
          "nothing owed is priced at exactly zero, fixed term included");

    // A split that seats MORE than the step reads is also free, never negative: the arrangement can
    // over-provision (a pool bigger than one step's routes) and the bill clamps at zero.
    auto over                  = geometry();
    over.pages_resident        = 100000;
    over.expert_slots_resident = 100000;
    const auto over_cost       = price_step_transfers(over, rates());
    check(over_cost.page_in_ms == 0.0 && over_cost.expert_ms == 0.0,
          "an over-provisioned split is priced at zero, not at a negative");
}

void test_more_traffic_never_costs_less() {
    const auto base = price_step_transfers(geometry(), rates());
    double previous = 0.0;
    for (std::uint64_t addressed = 1024; addressed <= 4096; addressed += 512) {
        auto more            = geometry();
        more.pages_addressed = addressed;
        const auto cost      = price_step_transfers(more, rates());
        check(cost.page_in_ms >= previous,
              "a step that reads fewer pages cannot cost more than one that reads more");
        check(cost.expert_misses == base.expert_misses,
              "the page count does not move the slot bill");
        previous = cost.page_in_ms;
    }
    // More resident pages: the bill must not rise as the split covers more of the same step.
    bool first = true;
    for (std::uint64_t resident = 0; resident <= 5120; resident += 512) {
        auto fewer           = geometry();
        fewer.pages_resident = resident;
        const auto cost      = price_step_transfers(fewer, rates());
        if (!first) { check(cost.page_in_ms <= previous, "more resident pages never cost more"); }
        first    = false;
        previous = cost.page_in_ms;
    }
    previous = 0.0;
    for (std::uint64_t routed = 128; routed <= 2048; routed += 128) {
        auto more           = geometry();
        more.experts_routed = routed;
        const auto cost     = price_step_transfers(more, rates());
        check(cost.expert_ms >= previous, "a step that routes more experts cannot cost less");
        previous = cost.expert_ms;
    }
    check(base.expert_ms > 0.0, "the sweep priced a nonzero slot bill");
}

void test_each_leg_honours_its_own_rate() {
    const auto base   = price_step_transfers(geometry(), rates());
    auto faster_pages = rates();
    faster_pages.page_in_bytes_per_s *= 2.0;
    const auto page_doubled = price_step_transfers(geometry(), faster_pages);
    check_close(page_doubled.page_in_ms, base.page_in_ms / 2.0, 1e-9,
                "twice the page-in rate halves the page bill");
    check_close(page_doubled.expert_ms, base.expert_ms, 1e-9,
                "the page-in rate does not price the slot bill");

    auto faster_slots = rates();
    faster_slots.expert_bytes_per_s *= 2.0;
    const auto slot_doubled = price_step_transfers(geometry(), faster_slots);
    check_close(slot_doubled.expert_ms, base.expert_ms / 2.0, 1e-9,
                "twice the expert rate halves the slot bill");
    check_close(slot_doubled.page_in_ms, base.page_in_ms, 1e-9,
                "the expert rate does not price the page bill");
}

void test_an_unmeasured_rate_is_not_a_fast_leg() {
    // A zero rate is "nothing was measured": the model prices that leg at zero, and `measured()` is
    // what stops a caller from reading that as free traffic. This is the documented contract the
    // bet gates on, so it is asserted rather than assumed.
    TransferRates none;
    const auto cost = price_step_transfers(geometry(), none);
    check(!none.measured(), "an all-zero pair is not a measurement");
    check(cost.page_in_ms == 0.0 && cost.expert_ms == 0.0,
          "with no rates the model prices nothing (it does not divide by zero)");

    TransferRates half;
    half.page_in_bytes_per_s = 20e9;
    check(!half.measured(), "one measured leg is not a pair");
}

// V10-83: one expert byte is read out of the artifact and then copied to the Device, in series.
void test_the_streaming_leg_is_the_disk_and_the_link_in_series() {
    // 10 GB/s on each leg is 5 GB/s serialized - half, not the 10 GB/s the pre-V10-83 model used.
    check_close(effective_rate_bytes_per_s(10e9, 10e9), 5e9, 1e-3,
                "two equal legs in series halve the rate");
    check_close(effective_rate_bytes_per_s(6.83e9, 23.2e9), 5.2766e9, 1e7,
                "the measured pair on this machine combines as 1/(1/6.83 + 1/23.2)");
    // A missing disk measurement is the single-rate pipeline (a retained Host replica), not "free".
    check_close(effective_rate_bytes_per_s(0.0, 23.2e9), 23.2e9, 1e3,
                "no disk leg leaves the copy rate alone");
    check_close(effective_rate_bytes_per_s(6.83e9, 0.0), 0.0, 0.0,
                "no copy rate is no copy to price");

    auto routed                = geometry();
    routed.expert_layers       = 0; // the fixed term is priced by its own test
    auto with_disk             = rates();
    with_disk.disk_bytes_per_s = 10e9;
    const auto cost            = price_step_transfers(routed, with_disk);
    check_close(cost.expert_bytes_ms, 384.0 * static_cast<double>(kMib) / 5e9 * 1000.0, 1e-6,
                "the slot bill is the same bytes over the serialized rate");
    // The page leg is Host memory to Device: a disk rate must not touch it.
    check_close(cost.page_in_ms, 4096.0 * static_cast<double>(kMib) / 20e9 * 1000.0, 1e-6,
                "the page bill keeps its one rate");
}

void test_a_layer_that_streams_pays_its_round_trip() {
    auto routed          = geometry();
    routed.expert_layers = 40;
    const auto cost      = price_step_transfers(routed, rates());
    check(cost.stream_layers == 40, "every routed layer streams when the pool misses");
    check_close(cost.expert_fixed_ms, 40.0 * kStreamLayerFixedMs, 1e-9,
                "the fixed term is the per-layer round trip times the streaming layers");
    check_close(cost.expert_ms, cost.expert_bytes_ms + cost.expert_fixed_ms, 1e-9,
                "the slot leg is its bytes plus its round trips");

    // No layer count, no fixed term: a caller that cannot say how many layers route prices bytes.
    auto anonymous          = geometry();
    anonymous.expert_layers = 0;
    const auto bytes_only   = price_step_transfers(anonymous, rates());
    check(bytes_only.expert_fixed_ms == 0.0, "an unknown layer count prices no round trips");
    check_close(bytes_only.expert_ms, cost.expert_ms - 40.0 * kStreamLayerFixedMs, 1e-9,
                "and the difference is exactly the fixed term");
}

} // namespace

int main() {
    test_the_two_bills_are_pages_over_page_rate_and_slots_over_slot_rate();
    test_a_step_that_owes_nothing_costs_nothing();
    test_more_traffic_never_costs_less();
    test_each_leg_honours_its_own_rate();
    test_an_unmeasured_rate_is_not_a_fast_leg();
    test_the_streaming_leg_is_the_disk_and_the_link_in_series();
    test_a_layer_that_streams_pays_its_round_trip();
    if (failures == 0) { std::printf("ok\n"); }
    return failures == 0 ? 0 : 1;
}
