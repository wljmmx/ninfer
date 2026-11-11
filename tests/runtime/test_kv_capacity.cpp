#include "runtime/engine/kv_capacity.h"

#include <iostream>
#include <stdexcept>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;
    const ninfer::runtime::SequenceCapacityCurve curve{
        .main_page_tokens                     = 64,
        .minimum_main_page_groups             = 2,
        .maximum_main_page_groups             = 6,
        .minimum_device_reservation_bytes     = 1000,
        .bytes_per_additional_main_page_group = 128,
    };

    const auto automatic =
        ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve, 1360);
    failures +=
        check(automatic.main_page_groups == 4 && automatic.resolved_tokens == 256 &&
                  automatic.runtime_reservation_bytes == 1256 &&
                  automatic.automatic_headroom_bytes == 50 && automatic.planned_slack_bytes == 104,
              "automatic KV capacity did not select the largest fitting page count");

    const auto capped =
        ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve, 10000);
    failures += check(capped.main_page_groups == 6 && capped.resolved_tokens == 384,
                      "automatic KV capacity exceeded or missed the target maximum");

    const auto explicit_capacity = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::explicit_capacity(129), curve, 1200);
    failures +=
        check(explicit_capacity.main_page_groups == 3 && explicit_capacity.resolved_tokens == 192 &&
                  explicit_capacity.runtime_reservation_bytes == 1128,
              "explicit KV capacity did not use page-aligned token semantics");

    bool insufficient_rejected = false;
    try {
        (void)ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve,
                                                   1049);
    } catch (const std::invalid_argument&) { insufficient_rejected = true; }
    failures += check(insufficient_rejected,
                      "automatic KV capacity accepted less than the minimum reservation");

    // --- port (sm_89): the degradation floor is the curve minimum ----------------------------
    // A request that misses the budget degrades to the largest feasible page count instead of
    // aborting startup. On the real curves the minimum IS one full max_context (startup.cpp sizes
    // it as max(page_count(max_context), max_concurrency)), so pinning "degradation never crosses
    // the curve minimum" here is pinning "the arena always holds at least one full-length
    // request" - the invariant request admission and the output clamp rely on.
    const auto degraded_mid = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::explicit_capacity(384), curve, 1360);
    failures += check(degraded_mid.main_page_groups == 4 &&
                          degraded_mid.resolved_tokens == 256 &&
                          degraded_mid.runtime_reservation_bytes == 1256,
                      "KV capacity degradation did not keep the largest fitting page count");

    const auto degraded_floor = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::explicit_capacity(384), curve, 1001);
    failures += check(degraded_floor.main_page_groups == 2 &&
                          degraded_floor.resolved_tokens == 128 &&
                          degraded_floor.runtime_reservation_bytes == 1000,
                      "KV capacity degradation dropped below the curve minimum");

    const auto automatic_floor = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::automatic(0), curve, 1105);
    failures += check(automatic_floor.main_page_groups == 2 &&
                          automatic_floor.resolved_tokens == 128,
                      "automatic KV capacity expanded past its own budget");

    // --- Residency split of the per-page device cost (appendix BP) ----------------------
    // Measured on the RTX 4090 / qwen3.8-27b / rk4v4-e8 configuration:
    //   reservation(P) = 683,493,888 + 983,044*P,  i.e. 746,408,704 at the 64-page anchor.
    const ninfer::runtime::SequenceCapacityCurve measured_curve{
        .main_page_tokens                     = 64,
        .minimum_main_page_groups             = 64,
        .maximum_main_page_groups             = 15625,
        .minimum_device_reservation_bytes     = 746408704,
        .bytes_per_additional_main_page_group = 983044,
    };
    const ninfer::runtime::KvResidencyStrides measured_strides{
        .payload_bytes_per_resident_page_group = 983040,
        .ledger_bytes_per_logical_page_group   = 4,
    };
    constexpr std::size_t loose_budget = 64ULL * 1024 * 1024 * 1024;

    // A fully resident residency model must reproduce the measured rows byte for byte; this is
    // simultaneously the "zero behaviour change" gate and the check that the stride decomposes.
    struct Row {
        std::uint32_t pages;
        std::size_t   measured;
    };
    for (const Row row : {Row{64, 746408704}, Row{128, 809323520}, Row{256, 935153152},
                          Row{512, 1186812416}}) {
        const auto resolution = ninfer::runtime::resolve_kv_capacity_with_residency(
            ninfer::KvCapacityPolicy::explicit_capacity(row.pages * 64), measured_curve,
            measured_strides, loose_budget);
        failures += check(resolution.residency_active &&
                              resolution.logical_page_groups == row.pages &&
                              resolution.resident_page_groups == row.pages &&
                              resolution.device_reservation_bytes == row.measured &&
                              resolution.fully_resident_reservation_bytes == row.measured &&
                              resolution.host_payload_bytes == 0,
                          "residency model did not reproduce a measured fully resident row");
    }

    // 1M logical tokens against the measured 7.06 GB post-weights budget (appendix BP.2).
    const auto tiered = ninfer::runtime::resolve_kv_capacity_with_residency(
        ninfer::KvCapacityPolicy::explicit_capacity(1000000), measured_curve, measured_strides,
        7060000000ULL);
    failures += check(tiered.residency_active && tiered.logical_page_groups == 15625 &&
                          tiered.resident_page_groups == 6486 &&
                          tiered.device_reservation_bytes == 7059553828ULL &&
                          tiered.host_payload_bytes == 8984002560ULL &&
                          tiered.fully_resident_reservation_bytes == 16043556388ULL,
                      "residency resolution did not match the appendix BP numbers");

    // The fully resident model degrades instead of throwing (kv_capacity.cpp: the port rule that
    // replaced "over budget must abort"): 1M tokens price at 16,043,556,388 bytes, and the
    // 7.06 GB budget cuts that to the largest feasible count - 6486 page groups at
    // 7,059,517,272 bytes. That is the same resident page count the residency split picked two
    // cases above, because both walks price the same per-group stride against the same budget.
    // The degradation never crosses the curve minimum; here the budget holds 6486 groups, far
    // above the 64-group floor, and the slack pins that the choice was budget-exact.
    const auto fully_resident_degraded = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::explicit_capacity(1000000), measured_curve, 7060000000ULL);
    failures += check(fully_resident_degraded.main_page_groups == 6486 &&
                          fully_resident_degraded.resolved_tokens == 415104 &&
                          fully_resident_degraded.runtime_reservation_bytes == 7059517272ULL &&
                          fully_resident_degraded.planned_slack_bytes == 482728ULL,
                      "the fully resident model did not degrade to the largest feasible page count");

    // Unset strides must leave resolution untouched, including under the automatic policy.
    const auto inactive = ninfer::runtime::resolve_kv_capacity_with_residency(
        ninfer::KvCapacityPolicy::automatic(50), curve, {}, 1360);
    failures += check(!inactive.residency_active && inactive.logical_page_groups == 4 &&
                          inactive.resident_page_groups == 4 &&
                          inactive.device_reservation_bytes == 1256 &&
                          inactive.host_payload_bytes == 0 &&
                          inactive.fully_resident_reservation_bytes == 1256,
                      "an unset residency model changed the resolution");

    bool strides_rejected = false;
    try {
        (void)ninfer::runtime::resolve_kv_capacity_with_residency(
            ninfer::KvCapacityPolicy::explicit_capacity(128), curve,
            {.payload_bytes_per_resident_page_group = 128, .ledger_bytes_per_logical_page_group = 8},
            4096);
    } catch (const std::invalid_argument&) { strides_rejected = true; }
    failures += check(strides_rejected,
                      "residency strides inconsistent with the capacity stride were accepted");

    // --- port (sm_89): tiering the anchor (the 256K dense case) --------------------------------
    // The original model can only tier the pages ABOVE the curve minimum, and on this family the
    // curve minimum IS one full max_context - so a 262,144-token request (9.4 GB of KV on a 24 GB
    // card) had no plan at all. `tier_minimum_residency` prices the anchor's payload per resident
    // page as well: the anchor's fixed part (workspace, graph, the non-KV persistent layout) plus
    // payload per resident page group plus the ledger of every logical page group.
    const ninfer::runtime::KvResidencyStrides tiering_strides{
        .payload_bytes_per_resident_page_group = 983040,
        .ledger_bytes_per_logical_page_group   = 4,
        .tier_minimum_residency                = true,
    };

    // A budget above the anchor must produce the SAME plan as before: enabling the switch cannot
    // move a configuration that already fit.
    const auto tiered_above = ninfer::runtime::resolve_kv_capacity_with_residency(
        ninfer::KvCapacityPolicy::explicit_capacity(1000000), measured_curve, tiering_strides,
        7060000000ULL);
    failures += check(tiered_above.resident_page_groups == 6486 &&
                          tiered_above.device_reservation_bytes == 7059553828ULL &&
                          tiered_above.host_payload_bytes == 8984002560ULL,
                      "anchor tiering moved a plan whose fully resident footprint already fit");

    // The real 256K shape: one page group per 64 tokens and the curve minimum at max_context, so
    // the fully resident anchor alone is 4.71 GB - more than the whole post-weight budget.
    const ninfer::runtime::SequenceCapacityCurve dense_256k_curve{
        .main_page_tokens                     = 64,
        .minimum_main_page_groups             = 4096,
        .maximum_main_page_groups             = 4096,
        .minimum_device_reservation_bytes     = 4710042112ULL,
        .bytes_per_additional_main_page_group = 983044,
    };
    bool strict_rejected = false;
    try {
        (void)ninfer::runtime::resolve_kv_capacity_with_residency(
            ninfer::KvCapacityPolicy::explicit_capacity(262144), dense_256k_curve, measured_strides,
            3000000000ULL);
    } catch (const std::invalid_argument&) { strict_rejected = true; }
    failures += check(strict_rejected,
                      "the strict residency model accepted a plan below its fully resident anchor");

    const auto anchor_tiered = ninfer::runtime::resolve_kv_capacity_with_residency(
        ninfer::KvCapacityPolicy::explicit_capacity(262144), dense_256k_curve, tiering_strides,
        3000000000ULL);
    // fixed = anchor - payload*4096 = 683,510,272; resident = (3 GB - fixed)/983,040 = 2356.
    constexpr std::size_t kFixed = 683510272ULL;
    failures += check(anchor_tiered.logical_page_groups == 4096 &&
                          anchor_tiered.resident_page_groups == 2356 &&
                          anchor_tiered.device_reservation_bytes ==
                              kFixed + 983040ULL * 2356 + 4ULL * (4096 - 4096) &&
                          anchor_tiered.host_payload_bytes == 983040ULL * (4096 - 2356) &&
                          anchor_tiered.fully_resident_reservation_bytes == 4710042112ULL,
                      "anchor tiering did not split 256K across Device and Host");

    // A budget that cannot hold even one reseated page group is a hard failure, not a zero-page plan.
    bool empty_rejected = false;
    try {
        (void)ninfer::runtime::resolve_kv_capacity_with_residency(
            ninfer::KvCapacityPolicy::explicit_capacity(262144), dense_256k_curve, tiering_strides,
            683510271ULL);
    } catch (const std::invalid_argument&) { empty_rejected = true; }
    failures += check(empty_rejected, "anchor tiering accepted a plan with no resident page group");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
