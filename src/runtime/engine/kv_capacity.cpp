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

std::size_t checked_sub(std::size_t a, std::size_t b, const char* label) {
    if (b > a) { throw std::overflow_error(label); }
    return a - b;
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

// The anchor's non-payload part: everything the fully resident plan reserves at the curve minimum
// except the KV payload itself (workspace, graph allowance, the non-KV persistent layout).
std::size_t anchor_fixed_bytes(const SequenceCapacityCurve& curve,
                               const KvResidencyStrides& strides) {
    const std::size_t anchor_payload = checked_mul(
        static_cast<std::size_t>(curve.minimum_main_page_groups),
        strides.payload_bytes_per_resident_page_group, "KV residency anchor overflows size_t");
    return checked_sub(curve.minimum_device_reservation_bytes, anchor_payload,
                       "KV residency payload stride exceeds the fully resident anchor");
}

} // namespace

std::size_t residency_device_bytes(const SequenceCapacityCurve& curve,
                                   const KvResidencyStrides& strides, std::uint32_t logical,
                                   std::uint32_t resident) {
    if (logical < resident) {
        throw std::invalid_argument("KV residency logical pages are below the resident set");
    }
    if (logical < curve.minimum_main_page_groups) {
        throw std::invalid_argument("KV residency logical pages are below the capacity curve");
    }
    const std::size_t payload_stride = strides.payload_bytes_per_resident_page_group;
    const std::size_t ledger         = checked_mul(
        static_cast<std::size_t>(logical - curve.minimum_main_page_groups),
        strides.ledger_bytes_per_logical_page_group, "KV residency ledger overflows size_t");
    if (resident < curve.minimum_main_page_groups) {
        if (!strides.tier_minimum_residency) {
            throw std::invalid_argument(
                "KV residency cannot place fewer pages than the curve minimum without tiering the "
                "anchor");
        }
        const std::size_t payload = checked_mul(static_cast<std::size_t>(resident), payload_stride,
                                                "KV residency payload overflows size_t");
        return checked_add(checked_add(anchor_fixed_bytes(curve, strides), payload,
                                       "KV residency reservation overflows size_t"),
                           ledger, "KV residency reservation overflows size_t");
    }
    const std::size_t payload = checked_mul(
        static_cast<std::size_t>(resident - curve.minimum_main_page_groups), payload_stride,
        "KV residency payload overflows size_t");
    return checked_add(checked_add(curve.minimum_device_reservation_bytes, payload,
                                   "KV residency reservation overflows size_t"),
                       ledger, "KV residency reservation overflows size_t");
}

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
                                                        std::size_t available_runtime_bytes,
                                                        std::size_t available_host_bytes) {
    validate_curve(curve);

    KvResidencyResolution resolution;
    if (!residency_configured(strides)) {
        // Residency model inactive: report exactly what the fully resident resolution does.
        const auto base = resolve_kv_capacity(policy, curve, available_runtime_bytes);
        resolution.declared_page_groups             = base.main_page_groups;
        resolution.logical_page_groups               = base.main_page_groups;
        resolution.resident_page_groups              = base.main_page_groups;
        resolution.device_reservation_bytes          = base.runtime_reservation_bytes;
        resolution.fully_resident_reservation_bytes  = base.runtime_reservation_bytes;
        return resolution;
    }

    // The two strides are the two halves of the curve's own step, so they have to reconstruct it.
    // A FLAT curve (minimum == maximum, i.e. one lane at exactly max_context) has no step at all and
    // reports zero - while the same strides still describe how its anchor decomposes, which is
    // exactly what anchor tiering needs. So the identity is only required when there is a stride.
    if (curve.bytes_per_additional_main_page_group != 0 &&
        checked_add(strides.payload_bytes_per_resident_page_group,
                    strides.ledger_bytes_per_logical_page_group,
                    "KV residency strides overflow size_t") !=
            curve.bytes_per_additional_main_page_group) {
        throw std::invalid_argument(
            "KV residency strides do not add up to the capacity curve stride");
    }
    if (policy.mode != KvCapacityMode::Explicit && policy.mode != KvCapacityMode::Automatic) {
        throw std::invalid_argument("unknown KV capacity policy");
    }
    if (policy.mode == KvCapacityMode::Explicit) {
        if (policy.automatic_headroom_bytes != 0) {
            throw std::invalid_argument("explicit KV capacity must not carry automatic headroom");
        }
    } else if (policy.explicit_tokens != 0) {
        throw std::invalid_argument("inconsistent automatic KV capacity policy");
    }

    resolution.host_budget_bytes    = available_host_bytes;
    resolution.residency_active     = true;
    // The capacity the caller DECLARED. An explicit policy names it; an automatic one is derived
    // from what was declared alongside it - max_concurrency lanes of max_context - which is exactly
    // the curve's maximum. That is the fix for a plan that used to price one lane's 204800 tokens
    // for a command asking for eight.
    const std::uint32_t declared = policy.mode == KvCapacityMode::Explicit
                                       ? explicit_page_groups(policy, curve)
                                       : curve.maximum_main_page_groups;
    if (declared < curve.minimum_main_page_groups || declared > curve.maximum_main_page_groups) {
        throw std::invalid_argument(
            policy.mode == KvCapacityMode::Explicit
                ? "Explicit KV capacity is outside the target capacity curve"
                : "the automatic KV capacity is outside the target capacity curve");
    }
    resolution.declared_page_groups = declared;

    const std::size_t payload       = strides.payload_bytes_per_resident_page_group;
    const std::size_t ledger_stride = strides.ledger_bytes_per_logical_page_group;
    const std::uint32_t minimum     = curve.minimum_main_page_groups;
    // Pages every plan keeps on the Device whatever the budget says: the curve minimum, unless the
    // anchor itself is tiered (then none - the resident set is whatever the payload stride affords).
    const std::uint32_t floor_pages = strides.tier_minimum_residency ? 0U : minimum;
    // Everything the plan costs that is not KV payload of a resident page group: workspace, graph
    // allowance, the non-KV persistent layout, and - when the anchor is not tiered - the payload of
    // the floor pages.
    const std::size_t fixed_bytes = strides.tier_minimum_residency
                                        ? anchor_fixed_bytes(curve, strides)
                                        : curve.minimum_device_reservation_bytes;
    const auto ledger_for = [&](std::uint32_t logical) {
        return checked_mul(static_cast<std::size_t>(logical - minimum), ledger_stride,
                           "KV residency ledger overflows size_t");
    };
    // Ledger for every logical page exists regardless of residency; the largest residency that
    // fits the budget then follows from the payload stride. Tiering the anchor also withdraws the
    // anchor's payload, so the floor becomes the anchor's fixed part plus the whole ledger.
    const std::size_t with_ledger = checked_add(fixed_bytes, ledger_for(declared),
                                               "KV residency reservation overflows size_t");
    if (available_runtime_bytes < with_ledger) {
        // Same split as the success path, printed on the way out: a refusal has to name the bytes
        // that did not fit, not just their sum.
        std::fprintf(stderr,
                     "[kv-tier] REFUSED logical=%u minimum=%u | payload_stride=%zu ledger_stride=%zu "
                     "ledger=%zu anchor_fixed=%zu | needs=%zu available=%zu\n",
                     declared, minimum, payload, ledger_stride, ledger_for(declared), fixed_bytes,
                     with_ledger, available_runtime_bytes);
        std::fflush(stderr);
        throw std::invalid_argument(
            "minimum Engine runtime reservation requires " + std::to_string(with_ledger) +
            " bytes for " + std::to_string(declared) +
            " logical pages, but only " + std::to_string(available_runtime_bytes) +
            " bytes are available for runtime capacity");
    }

    // --- Host pages come from the Host arena budget ----------------------------------------------
    // A logical page that the Device does not keep resident is payload on the Host side, so the
    // Host budget caps how many logical pages the plan can back at all. Solving
    //     fixed + ledger(L) + payload * resident(L) = device budget,   hosted(L) = L - resident(L)
    // for `hosted(L) <= host_pages` is linear in L and closes in one step:
    //     anchored:  L <= (payload*host_pages + device - fixed + ledger_stride*minimum) / stride
    //     not:       L <= minimum + (payload*host_pages + device - fixed) / stride
    // Only the split adapts while the Host tier can hold the difference; when it cannot, the
    // logical capacity itself has to come down and `degraded` says so out loud (the caller prints
    // it). An explicit capacity is never silently rewritten - it is refused by the branch below.
    std::uint32_t logical = declared;
    if (payload == 0) {
        // No payload stride means no page can be moved off the Device; nothing to tier.
        logical = std::min(declared, curve.maximum_main_page_groups);
    } else if (available_host_bytes != static_cast<std::size_t>(-1)) {
        const std::size_t stride = payload + ledger_stride;
        const std::size_t host_pages = available_host_bytes / payload;
        const std::size_t device_spare = available_runtime_bytes - fixed_bytes;
        const std::uint64_t numerator = checked_add(
            checked_mul(host_pages, payload, "KV residency host budget overflows size_t"),
            strides.tier_minimum_residency
                ? checked_add(device_spare,
                              checked_mul(static_cast<std::size_t>(minimum), ledger_stride,
                                          "KV residency floor ledger overflows size_t"),
                              "KV residency host solve overflows size_t")
                : device_spare,
            "KV residency host solve overflows size_t");
        if (stride == 0) {
            throw std::invalid_argument("KV residency has no per-page stride to solve against");
        }
        const std::uint64_t backed = static_cast<std::uint64_t>(floor_pages) + numerator / stride;
        if (backed < logical) {
            // An automatic policy may only serve what the two tiers can back, and it says so; an
            // explicit capacity is the caller's number and is never rewritten - it is refused
            // here with the bytes that are missing.
            if (policy.mode == KvCapacityMode::Automatic) {
                logical = static_cast<std::uint32_t>(backed);
            } else {
                const std::size_t hosted_pages =
                    payload == 0 ? 0 : available_host_bytes / payload;
                const std::size_t resident_cap =
                    floor_pages + (available_runtime_bytes - fixed_bytes) / payload;
                std::fprintf(stderr,
                             "[kv-tier] REFUSED capacity=%u backed=%llu | device_resident=%zu "
                             "host_pages=%zu payload_stride=%zu | device=%zu host_budget=%zu\n",
                             declared, static_cast<unsigned long long>(backed), resident_cap,
                             hosted_pages, payload, available_runtime_bytes,
                             available_host_bytes);
                std::fflush(stderr);
                throw std::invalid_argument(
                    "KV capacity of " + std::to_string(declared) + " logical page groups (" +
                    std::to_string(static_cast<unsigned long long>(declared) *
                                   curve.main_page_tokens) +
                    " tokens) does not fit: the Device budget holds " +
                    std::to_string((available_runtime_bytes - fixed_bytes) / payload) +
                    " resident page group(s) after the residency slack, and the Host budget (" +
                    std::to_string(available_host_bytes) + " bytes) " +
                    std::to_string(hosted_pages) +
                    " more, which totals " + std::to_string(backed) +
                    " page groups. Raise --host-context-mib by " +
                    std::to_string((static_cast<std::size_t>(declared) -
                                    static_cast<std::size_t>(backed)) *
                                   payload) +
                    " bytes, lower --kv-capacity, or use --kv-capacity auto to accept the "
                    "capacity this box can actually back.");
            }
        }
    }
    if (logical < minimum) {
        const std::size_t hosted_pages =
            payload == 0 ? 0 : available_host_bytes / payload;
        throw std::invalid_argument(
            "KV capacity " + std::to_string(declared) + " logical page groups (" +
            std::to_string(static_cast<unsigned long long>(declared) * curve.main_page_tokens) +
            " tokens) cannot be backed by this engine: the Device budget holds " +
            std::to_string((available_runtime_bytes - fixed_bytes) / (payload == 0 ? 1 : payload)) +
            " resident page group(s) and the Host budget (" + std::to_string(available_host_bytes) +
            " bytes) another " + std::to_string(hosted_pages) +
            ", short of the " + std::to_string(minimum) +
            " page groups one max_context request needs");
    }
    resolution.degraded = logical != declared;

    std::uint32_t resident = logical;
    if (payload != 0) {
        const std::size_t spare =
            available_runtime_bytes - fixed_bytes - ledger_for(logical);
        const std::size_t affordable = spare / payload;
        const std::uint64_t candidate = static_cast<std::uint64_t>(floor_pages) + affordable;
        resident = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(candidate, static_cast<std::uint64_t>(logical)));
    }
    if (resident == 0) {
        throw std::invalid_argument(
            "KV residency budget holds no page group: the anchor needs " +
            std::to_string(with_ledger) + " bytes plus " +
            std::to_string(payload) + " bytes per resident page group, but only " +
            std::to_string(available_runtime_bytes) + " bytes are available");
    }

    resolution.logical_page_groups              = logical;
    resolution.fully_resident_reservation_bytes = curve.reservation_bytes(declared);
    resolution.resident_page_groups             = resident;
    resolution.device_reservation_bytes =
        residency_device_bytes(curve, strides, logical, resident);
    resolution.host_payload_bytes = checked_mul(
        static_cast<std::size_t>(logical - resident), payload,
        "KV residency host payload overflows size_t");
    // P0 (KV tier plan): the split the failure message only gave as one number. Every byte of the
    // device reservation is named here, so a plan that does not fit is attributable instead of
    // argued about: the ledger exists per LOGICAL page, the payload only per RESIDENT one. P1c adds
    // the declared count and the Host budget, so "why is the logical capacity not what I asked for"
    // is answered by the same line.
    std::fprintf(stderr,
                 "[kv-tier] declared=%u logical=%u minimum=%u resident=%u | payload_stride=%zu "
                 "ledger_stride=%zu ledger=%zu anchor_fixed=%zu state_image=%zu | device=%zu "
                 "host_payload=%zu host_budget=%zu available=%zu\n",
                 declared, logical, minimum, resident, payload, ledger_stride, ledger_for(logical),
                 fixed_bytes, with_ledger - ledger_for(declared), resolution.device_reservation_bytes,
                 resolution.host_payload_bytes, available_host_bytes, available_runtime_bytes);
    std::fflush(stderr);
    if (resolution.degraded) {
        std::fprintf(stderr,
                     "[kv-tier] logical capacity: declared %u page groups (%llu tokens), the Device "
                     "holds %u and the Host budget backs %u -> %u page groups (%llu tokens)\n",
                     declared,
                     static_cast<unsigned long long>(declared) * curve.main_page_tokens,
                     resident, logical - resident, logical,
                     static_cast<unsigned long long>(logical) * curve.main_page_tokens);
        std::fflush(stderr);
    }
    return resolution;
}

bool kv_residency_share_fits(const KvResidencyShare& share) noexcept {
    // 0 is the "no residency contract" answer: every logical page is resident, so no aggregate bound
    // exists and every request is judged by its own length alone (the pre-existing single-lane rule).
    if (share.capacity_tokens == 0) { return true; }
    if (share.request_tokens > share.capacity_tokens) { return false; }
    return share.live_tokens <= share.capacity_tokens - share.request_tokens;
}

} // namespace ninfer::runtime
