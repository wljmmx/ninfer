#include "runtime/engine/kv_capacity.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::runtime {
namespace {

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) { throw std::overflow_error(label); }
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(label);
    }
    return a * b;
}

void validate_curve(const SequenceCapacityCurve& curve) {
    if (curve.main_page_tokens == 0 || curve.minimum_main_page_groups == 0 ||
        curve.minimum_main_page_groups > curve.maximum_main_page_groups ||
        curve.minimum_device_reservation_bytes == 0) {
        throw std::invalid_argument("sequence capacity curve is invalid");
    }
    if (curve.minimum_main_page_groups < curve.maximum_main_page_groups &&
        curve.bytes_per_additional_main_page_group == 0) {
        throw std::invalid_argument("expandable sequence capacity curve has zero byte stride");
    }
}

std::uint32_t explicit_page_groups(const KvCapacityPolicy& policy,
                                   const SequenceCapacityCurve& curve) {
    if (policy.explicit_tokens == 0) {
        throw std::invalid_argument("explicit KV capacity must be positive");
    }
    const std::uint64_t pages =
        1ULL + (static_cast<std::uint64_t>(policy.explicit_tokens) - 1ULL) / curve.main_page_tokens;
    if (pages > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("explicit KV page count exceeds uint32");
    }
    return static_cast<std::uint32_t>(pages);
}

bool residency_configured(const KvResidencyStrides& strides) {
    return strides.payload_bytes_per_resident_page_group != 0 ||
           strides.ledger_bytes_per_logical_page_group != 0;
}

// Anchor + payload per resident page + ledger per logical page. Both page counts are expressed
// relative to the curve minimum, where the anchor reservation is exact.
std::size_t residency_reservation_bytes(const SequenceCapacityCurve& curve,
                                        const KvResidencyStrides& strides, std::uint32_t logical,
                                        std::uint32_t resident) {
    const std::size_t payload = checked_mul(
        static_cast<std::size_t>(resident - curve.minimum_main_page_groups),
        strides.payload_bytes_per_resident_page_group, "KV residency payload overflows size_t");
    const std::size_t ledger = checked_mul(
        static_cast<std::size_t>(logical - curve.minimum_main_page_groups),
        strides.ledger_bytes_per_logical_page_group, "KV residency ledger overflows size_t");
    return checked_add(checked_add(curve.minimum_device_reservation_bytes, payload,
                                   "KV residency reservation overflows size_t"),
                       ledger, "KV residency reservation overflows size_t");
}

} // namespace

std::size_t SequenceCapacityCurve::reservation_bytes(std::uint32_t main_page_groups) const {
    validate_curve(*this);
    if (main_page_groups < minimum_main_page_groups ||
        main_page_groups > maximum_main_page_groups) {
        throw std::invalid_argument("Main KV page count is outside the target capacity curve");
    }
    const std::size_t additional =
        static_cast<std::size_t>(main_page_groups - minimum_main_page_groups);
    return checked_add(minimum_device_reservation_bytes,
                       checked_mul(additional, bytes_per_additional_main_page_group,
                                   "sequence capacity increment overflows size_t"),
                       "sequence capacity reservation overflows size_t");
}

std::uint32_t SequenceCapacityCurve::resolved_tokens(std::uint32_t main_page_groups) const {
    validate_curve(*this);
    if (main_page_groups < minimum_main_page_groups ||
        main_page_groups > maximum_main_page_groups) {
        throw std::invalid_argument("Main KV page count is outside the target capacity curve");
    }
    const std::uint64_t tokens = static_cast<std::uint64_t>(main_page_groups) * main_page_tokens;
    if (tokens > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("resolved KV token capacity exceeds uint32");
    }
    return static_cast<std::uint32_t>(tokens);
}

KvCapacityResolution resolve_kv_capacity(const KvCapacityPolicy& policy,
                                         const SequenceCapacityCurve& curve,
                                         std::size_t available_runtime_bytes) {
    validate_curve(curve);

    std::uint32_t pages         = curve.minimum_main_page_groups;
    std::size_t capacity_budget = available_runtime_bytes;
    switch (policy.mode) {
    case KvCapacityMode::Explicit:
        if (policy.automatic_headroom_bytes != 0) {
            throw std::invalid_argument("explicit KV capacity must not carry automatic headroom");
        }
        pages = explicit_page_groups(policy, curve);
        break;
    case KvCapacityMode::Automatic:
        if (available_runtime_bytes < policy.automatic_headroom_bytes) {
            throw std::invalid_argument(
                "automatic KV headroom requires " +
                std::to_string(policy.automatic_headroom_bytes) + " bytes, but only " +
                std::to_string(available_runtime_bytes) + " bytes are available after weights");
        }
        capacity_budget -= policy.automatic_headroom_bytes;
        if (capacity_budget < curve.minimum_device_reservation_bytes) {
            throw std::invalid_argument(
                "minimum Engine runtime reservation requires " +
                std::to_string(curve.minimum_device_reservation_bytes) + " bytes in addition to " +
                std::to_string(policy.automatic_headroom_bytes) +
                " bytes of automatic headroom, but only " +
                std::to_string(available_runtime_bytes) + " bytes are available after weights");
        }
        if (curve.minimum_main_page_groups < curve.maximum_main_page_groups) {
            const std::size_t additional =
                (capacity_budget - curve.minimum_device_reservation_bytes) /
                curve.bytes_per_additional_main_page_group;
            const std::uint64_t candidate =
                static_cast<std::uint64_t>(curve.minimum_main_page_groups) + additional;
            pages = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(candidate, curve.maximum_main_page_groups));
        }
        break;
    default:
        throw std::invalid_argument("unknown KV capacity policy");
    }

    // port (sm_89): a request that misses the budget degrades to the largest feasible page
    // count instead of aborting startup. Only the minimum device reservation is a hard floor;
    // when even that does not fit the configuration is physically infeasible.
    std::size_t reservation = curve.reservation_bytes(pages);
    if (reservation > capacity_budget) {
        if (curve.minimum_device_reservation_bytes > capacity_budget) {
            throw std::invalid_argument("minimum Engine runtime reservation requires " +
                                        std::to_string(curve.minimum_device_reservation_bytes) +
                                        " bytes, but only " + std::to_string(capacity_budget) +
                                        " bytes are available for runtime capacity");
        }
        const auto per_group = curve.bytes_per_additional_main_page_group;
        const auto excess    = reservation - capacity_budget;
        const auto drop      = (excess + per_group - 1) / per_group;
        const auto floor     = curve.minimum_main_page_groups;
        std::uint32_t degraded =
            drop >= pages - floor ? floor : static_cast<std::uint32_t>(pages - drop);
        while (degraded > floor && curve.reservation_bytes(degraded) > capacity_budget) {
            --degraded;
        }
        reservation = curve.reservation_bytes(degraded);
        if (reservation > capacity_budget) {
            throw std::invalid_argument("requested Engine runtime reservation requires " +
                                        std::to_string(reservation) + " bytes, but only " +
                                        std::to_string(capacity_budget) +
                                        " bytes are available for runtime capacity");
        }
        std::fprintf(stderr,
                     "[kv] capacity degraded: %u -> %u page groups (%zu -> %zu bytes, "
                     "%zu bytes available)\n",
                     pages, degraded, curve.reservation_bytes(pages), reservation,
                     capacity_budget);
        std::fflush(stderr);
        pages = degraded;
    }

    return KvCapacityResolution{
        .mode                                 = policy.mode,
        .main_page_groups                     = pages,
        .maximum_main_page_groups             = curve.maximum_main_page_groups,
        .resolved_tokens                      = curve.resolved_tokens(pages),
        .minimum_runtime_reservation_bytes    = curve.minimum_device_reservation_bytes,
        .bytes_per_additional_main_page_group = curve.bytes_per_additional_main_page_group,
        .runtime_reservation_bytes            = reservation,
        .available_after_weights_bytes        = available_runtime_bytes,
        .automatic_headroom_bytes             = policy.automatic_headroom_bytes,
        .planned_slack_bytes                  = available_runtime_bytes - reservation,
    };
}

KvResidencyResolution resolve_kv_capacity_with_residency(const KvCapacityPolicy& policy,
                                                        const SequenceCapacityCurve& curve,
                                                        const KvResidencyStrides& strides,
                                                        std::size_t available_runtime_bytes) {
    validate_curve(curve);

    KvResidencyResolution resolution;
    if (!residency_configured(strides)) {
        // Residency model inactive: report exactly what the fully resident resolution does.
        const auto base = resolve_kv_capacity(policy, curve, available_runtime_bytes);
        resolution.logical_page_groups               = base.main_page_groups;
        resolution.resident_page_groups              = base.main_page_groups;
        resolution.device_reservation_bytes          = base.runtime_reservation_bytes;
        resolution.fully_resident_reservation_bytes  = base.runtime_reservation_bytes;
        return resolution;
    }

    if (checked_add(strides.payload_bytes_per_resident_page_group,
                    strides.ledger_bytes_per_logical_page_group,
                    "KV residency strides overflow size_t") !=
        curve.bytes_per_additional_main_page_group) {
        throw std::invalid_argument(
            "KV residency strides do not add up to the capacity curve stride");
    }
    if (policy.mode != KvCapacityMode::Explicit) {
        throw std::invalid_argument(
            "KV residency resolution requires an explicit logical capacity");
    }
    if (policy.automatic_headroom_bytes != 0) {
        throw std::invalid_argument("explicit KV capacity must not carry automatic headroom");
    }

    const std::uint32_t logical = explicit_page_groups(policy, curve);
    if (logical < curve.minimum_main_page_groups || logical > curve.maximum_main_page_groups) {
        throw std::invalid_argument("Explicit KV capacity is outside the target capacity curve");
    }
    resolution.residency_active                 = true;
    resolution.logical_page_groups              = logical;
    resolution.fully_resident_reservation_bytes = curve.reservation_bytes(logical);

    // Ledger for every logical page exists regardless of residency; the largest residency that
    // fits the budget then follows from the payload stride.
    const std::size_t with_ledger =
        checked_add(curve.minimum_device_reservation_bytes,
                    checked_mul(static_cast<std::size_t>(logical - curve.minimum_main_page_groups),
                                strides.ledger_bytes_per_logical_page_group,
                                "KV residency ledger overflows size_t"),
                    "KV residency reservation overflows size_t");
    if (available_runtime_bytes < with_ledger) {
        throw std::invalid_argument(
            "minimum Engine runtime reservation requires " + std::to_string(with_ledger) +
            " bytes for " + std::to_string(logical) +
            " logical pages, but only " + std::to_string(available_runtime_bytes) +
            " bytes are available for runtime capacity");
    }

    std::uint32_t resident = logical;
    if (strides.payload_bytes_per_resident_page_group != 0) {
        const std::size_t affordable =
            (available_runtime_bytes - with_ledger) / strides.payload_bytes_per_resident_page_group;
        const std::uint64_t candidate =
            static_cast<std::uint64_t>(curve.minimum_main_page_groups) + affordable;
        resident = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(candidate, static_cast<std::uint64_t>(logical)));
    }

    resolution.resident_page_groups      = resident;
    resolution.device_reservation_bytes  = residency_reservation_bytes(curve, strides, logical, resident);
    resolution.host_payload_bytes        = checked_mul(
        static_cast<std::size_t>(logical - resident),
        strides.payload_bytes_per_resident_page_group, "KV residency host payload overflows size_t");
    return resolution;
}

} // namespace ninfer::runtime
