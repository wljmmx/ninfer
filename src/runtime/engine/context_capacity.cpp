#include "runtime/engine/context_capacity.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::runtime {
namespace {

std::size_t checked_add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a) {
        throw std::overflow_error("context capacity reservation overflows size_t");
    }
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error("context capacity reservation overflows size_t");
    }
    return a * b;
}

void validate_curve(const ContextCapacityCurve& curve) {
    // `SequenceCapacityCurve::reservation_bytes` validates the page half on every call; the state
    // half has its own domain rules.
    if (curve.minimum_state_slots == 0 || curve.bytes_per_additional_state_slot == 0) {
        throw std::invalid_argument("context capacity curve has no StateImage stride");
    }
}

std::uint32_t explicit_pages(const KvCapacityPolicy& policy, const SequenceCapacityCurve& curve) {
    const std::uint64_t pages =
        1ULL + (static_cast<std::uint64_t>(policy.explicit_tokens) - 1ULL) / curve.main_page_tokens;
    if (pages > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("explicit KV page count exceeds uint32");
    }
    return static_cast<std::uint32_t>(pages);
}

std::size_t breakdown_state(const ContextCapacityCurve& curve, ContextCapacity capacity) {
    return checked_mul(static_cast<std::size_t>(capacity.device_state_slots -
                                                curve.minimum_state_slots),
                       curve.bytes_per_additional_state_slot);
}

} // namespace

std::size_t ContextCapacityCurve::reservation_bytes(ContextCapacity capacity) const {
    validate_curve(*this);
    if (capacity.primary_page_groups < kv.minimum_main_page_groups ||
        capacity.primary_page_groups > kv.maximum_main_page_groups) {
        throw std::invalid_argument("context capacity page groups are outside the storage domain");
    }
    if (capacity.device_state_slots < minimum_state_slots ||
        capacity.device_state_slots > INT32_MAX) {
        throw std::invalid_argument("context capacity StateImage slots are outside the domain");
    }
    return checked_add(kv.reservation_bytes(capacity.primary_page_groups),
                       breakdown_state(*this, capacity));
}

std::size_t ContextCapacityCurve::control_bytes(ContextCapacity capacity) const {
    return checked_add(
        checked_mul(static_cast<std::size_t>(capacity.primary_page_groups -
                                             kv.minimum_main_page_groups),
                    control_bytes_per_page_group),
        checked_mul(static_cast<std::size_t>(capacity.device_state_slots - minimum_state_slots),
                    control_bytes_per_state_slot));
}

std::size_t ContextCapacityCurve::total_bytes(ContextCapacity capacity) const {
    return checked_add(checked_add(reservation_bytes(capacity), control_bytes(capacity)),
                       workspace_bytes);
}

std::uint32_t ContextCapacityCurve::resolved_tokens(std::uint32_t pages) const {
    return kv.resolved_tokens(pages);
}

ContextCapacityPlan plan_context_capacity(const KvCapacityPolicy& policy,
                                          const ContextCacheOptions& cache,
                                          const ContextCapacityCurve& curve,
                                          std::size_t available_runtime_bytes,
                                          std::uint32_t checkpoint_interval) {
    validate_curve(curve);
    if (policy.mode != KvCapacityMode::Explicit && policy.mode != KvCapacityMode::Automatic) {
        throw std::invalid_argument("unknown KV capacity policy");
    }
    if (policy.mode == KvCapacityMode::Explicit) {
        if (policy.explicit_tokens == 0 || policy.automatic_headroom_bytes != 0) {
            throw std::invalid_argument("inconsistent explicit KV capacity policy");
        }
    } else if (policy.explicit_tokens != 0) {
        throw std::invalid_argument("inconsistent automatic KV capacity policy");
    }
    // `checkpoint_interval == 0` is a valid deployment: no checkpoint target is added.

    const std::size_t headroom = policy.mode == KvCapacityMode::Automatic
                                     ? policy.automatic_headroom_bytes
                                     : std::size_t{0};
    if (available_runtime_bytes < headroom) {
        throw std::invalid_argument(
            "automatic context headroom requires " + std::to_string(headroom) +
            " bytes, but only " + std::to_string(available_runtime_bytes) + " are available");
    }
    const std::size_t budget = available_runtime_bytes - headroom;

    const auto fits = [&](ContextCapacity capacity) {
        try {
            return curve.total_bytes(capacity) <= budget;
        } catch (const std::overflow_error&) {
            return false;
        }
    };
    const auto require = [&](ContextCapacity capacity) {
        if (fits(capacity)) { return; }
        std::size_t total = 0;
        try {
            total = curve.total_bytes(capacity);
        } catch (const std::overflow_error&) { total = std::numeric_limits<std::size_t>::max(); }
        throw std::invalid_argument(
            "context capacity requires " + std::to_string(total) + " bytes (" +
            std::to_string(capacity.primary_page_groups) + " page groups + " +
            std::to_string(capacity.device_state_slots) + " StateImage slots), but only " +
            std::to_string(budget) + " bytes are available after weights and headroom");
    };
    const auto clamp_slots = [](std::uint64_t slots) {
        return static_cast<std::uint32_t>(std::min<std::uint64_t>(slots, INT32_MAX));
    };
    // Largest value in [low, high] for which `fits` holds. `low` itself is assumed to fit.
    const auto largest_pages = [&](std::uint32_t low, std::uint32_t high,
                                   std::uint32_t slots) {
        while (low < high) {
            const auto middle = static_cast<std::uint32_t>(
                low + (static_cast<std::uint64_t>(high) - low + 1U) / 2U);
            if (fits(ContextCapacity{middle, slots})) {
                low = middle;
            } else {
                high = middle - 1U;
            }
        }
        return low;
    };
    const auto largest_slots = [&](std::uint32_t low, std::uint32_t high,
                                   std::uint32_t pages) {
        while (low < high) {
            const auto middle = static_cast<std::uint32_t>(
                low + (static_cast<std::uint64_t>(high) - low + 1U) / 2U);
            if (fits(ContextCapacity{pages, middle})) {
                low = middle;
            } else {
                high = middle - 1U;
            }
        }
        return low;
    };

    ContextCapacity selected{
        .primary_page_groups = curve.kv.minimum_main_page_groups,
        .device_state_slots  = curve.minimum_state_slots,
    };
    const std::uint32_t requested_extra = cache.device_state_slots.value_or(0);
    selected.device_state_slots = clamp_slots(static_cast<std::uint64_t>(
        selected.device_state_slots) + requested_extra);
    require(selected);

    const bool target_checkpoints = cache.enabled && checkpoint_interval != 0;
    const auto checkpoint_target = [&](std::uint32_t pages) {
        const std::uint64_t tokens =
            static_cast<std::uint64_t>(pages) * curve.kv.main_page_tokens;
        const std::uint64_t spaced = 1ULL + (tokens - 1ULL) / checkpoint_interval;
        return clamp_slots(2ULL * curve.minimum_state_slots + spaced);
    };
    // A caller that pinned the slot count keeps it; only the checkpoint heuristic may move it.
    const auto slot_target = [&](std::uint32_t pages, std::uint32_t current) {
        if (!target_checkpoints || cache.device_state_slots.has_value()) { return current; }
        return std::max(current, checkpoint_target(pages));
    };

    ContextCapacityPlan plan;
    plan.checkpoints_targeted = policy.mode == KvCapacityMode::Automatic && target_checkpoints &&
                               !cache.device_state_slots.has_value();
    plan.target_slots =
        plan.checkpoints_targeted
            ? slot_target(curve.kv.minimum_main_page_groups, selected.device_state_slots)
            : selected.device_state_slots;

    switch (policy.mode) {
    case KvCapacityMode::Explicit:
        selected.primary_page_groups = explicit_pages(policy, curve.kv);
        require(selected);
        break;
    case KvCapacityMode::Automatic: {
        // Checkpoint slots first: they are discrete and small, so reserving them up front avoids
        // filling the budget with pages that the slot rounding would then have to take back.
        if (plan.checkpoints_targeted) {
            selected.device_state_slots =
                largest_slots(selected.device_state_slots,
                              std::max(selected.device_state_slots, plan.target_slots),
                              selected.primary_page_groups);
        }
        selected.primary_page_groups =
            largest_pages(selected.primary_page_groups, curve.kv.maximum_main_page_groups,
                          selected.device_state_slots);
        if (plan.checkpoints_targeted) {
            // The page count moved the token target; re-grow slots, then spend the remainder on
            // pages again. State slots are discrete, so the second pass is where leftover bytes
            // become page groups instead of a single stranded image.
            plan.target_slots =
                checkpoint_target(selected.primary_page_groups);
            selected.device_state_slots =
                largest_slots(selected.device_state_slots,
                              std::max(selected.device_state_slots, plan.target_slots),
                              selected.primary_page_groups);
            selected.primary_page_groups =
                largest_pages(selected.primary_page_groups, curve.kv.maximum_main_page_groups,
                              selected.device_state_slots);
        }
        break;
    }
    default:
        throw std::invalid_argument("unknown KV capacity policy");
    }

    plan.capacity                   = selected;
    plan.maximum_page_groups        = curve.kv.maximum_main_page_groups;
    plan.resolved_tokens            = curve.resolved_tokens(selected.primary_page_groups);
    plan.kv_reservation_bytes       = curve.kv.reservation_bytes(selected.primary_page_groups);
    plan.state_reservation_bytes    = breakdown_state(curve, selected);
    plan.workspace_reservation_bytes = curve.workspace_bytes;
    plan.control_reservation_bytes  = curve.control_bytes(selected);
    plan.total_reservation_bytes    = curve.total_bytes(selected);
    plan.available_bytes            = available_runtime_bytes;
    plan.planned_slack_bytes        = budget - plan.total_reservation_bytes;
    return plan;
}

} // namespace ninfer::runtime
