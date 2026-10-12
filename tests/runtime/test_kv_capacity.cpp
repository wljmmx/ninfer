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

    bool fully_resident_rejected = false;
    try {
        (void)ninfer::runtime::resolve_kv_capacity(
            ninfer::KvCapacityPolicy::explicit_capacity(1000000), measured_curve, 7060000000ULL);
    } catch (const std::invalid_argument&) { fully_resident_rejected = true; }
    failures += check(fully_resident_rejected,
                      "the fully resident model accepted a capacity the device cannot hold");

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

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
