#pragma once

// V10-82: what each Host -> Device leg costs a decode step.
//
// The capacity bet (core/capacity_bet.h) is a choice between two arrangements: A keeps the experts
// on the Device and pays for it out of the KV pages, B keeps the pages and pays for it out of the
// pool. Until now that choice was a PRIORITY ("experts first", "KV first") because nothing priced
// the two legs; the measured A-versus-B decode rates at one context were taken with the same arena,
// so they could not rank the arrangements either.
//
// The two bills are both PCIe traffic and both are linear in the bytes a step owes:
//   A  the pages a step reads that the Device does not hold: (addressed - resident) x page bytes;
//   B  the experts a step routes that the pool does not hold:  (routed - resident) x slot bytes.
// A caller that knows the rates turns either into milliseconds and the bet can compare them. The
// function below is the whole model: no CUDA, no allocation, no clock, no logging - the same three
// arguments always give the same two numbers, so a sweep can assert what the model claims.
//
// WHAT THIS MODEL DOES NOT PRICE, stated here because the measured error in the report is mostly
// these terms: a fetch that cannot take a free slot also DEMOTES a page first (a second copy of the
// same size), the copy is issued per page rather than as one bulk transfer, and the work is not
// free of the kernels around it. The rates are therefore an optimistic lower bound, and the caller
// prints them beside the prediction so a reader can see what the number was built from.
//
// Measured on 35B e2 / RTX 4090 (V10-82, report .temp/v10-82/report.md): for the expert leg the
// byte-proportional term is right - 0.0449 ms/MiB measured against 0.0455 predicted (1.1%) - and
// the error is a fixed 22.9 ms per step that appears at the first miss, because a step that streams
// uploads and synchronizes per layer (40 of them) before the bytes matter. Any comparison built on
// the total below is therefore exact only when a leg owes nothing.

#include <cstdint>

namespace ninfer::core {

// Bytes per second, Host -> Device. Zero means "never measured on this machine": both this header
// and the bet treat a zero as unpriced rather than as free, which is why `measured()` asks for
// both.
struct TransferRates {
    double page_in_bytes_per_s = 0.0;
    double expert_bytes_per_s  = 0.0;

    [[nodiscard]] bool measured() const noexcept {
        return page_in_bytes_per_s > 0.0 && expert_bytes_per_s > 0.0;
    }
};

// One decode step's geometry, in the bet's own units. `pages_addressed` is what the step READS
// (the whole live history for a full-attention layer, the window's reach when one is installed) and
// `pages_resident` is what the split leaves on the Device; `experts_routed` is the step's routed
// slots and `expert_slots_resident` what the pool holds. Either pair may name a zero rate's leg as
// fully covered - the difference is clamped at zero, never negative.
struct StepTransferGeometry {
    std::uint64_t page_bytes            = 0;
    std::uint64_t pages_addressed       = 0;
    std::uint64_t pages_resident        = 0;
    std::uint64_t slot_bytes            = 0;
    std::uint64_t experts_routed        = 0;
    std::uint64_t expert_slots_resident = 0;
};

struct StepTransferCost {
    std::uint64_t pages_fetched = 0;
    std::uint64_t expert_misses = 0;
    double page_in_ms           = 0.0;
    double expert_ms            = 0.0;

    [[nodiscard]] double total_ms() const noexcept { return page_in_ms + expert_ms; }
};

namespace detail {

[[nodiscard]] inline std::uint64_t excess_of(std::uint64_t want, std::uint64_t have) noexcept {
    return want > have ? want - have : 0;
}

// Bytes over a rate, in milliseconds. A zero rate prices its leg at zero and is the caller's signal
// that nothing was measured - `TransferRates::measured()` is how a caller avoids reading that as
// free traffic.
[[nodiscard]] inline double transfer_ms(std::uint64_t bytes, double bytes_per_s) noexcept {
    if (bytes == 0 || bytes_per_s <= 0.0) { return 0.0; }
    return static_cast<double>(bytes) / bytes_per_s * 1000.0;
}

} // namespace detail

// The per-step bill of one arrangement. Monotone in every input the arrangement can move: more
// addressed pages or routed experts never costs less, more resident pages or slots never costs
// more, and a step that owes nothing (every page resident, every route seated) costs exactly zero.
[[nodiscard]] inline StepTransferCost price_step_transfers(const StepTransferGeometry& geometry,
                                                           const TransferRates& rates) noexcept {
    StepTransferCost cost;
    cost.pages_fetched = detail::excess_of(geometry.pages_addressed, geometry.pages_resident);
    cost.expert_misses = detail::excess_of(geometry.experts_routed, geometry.expert_slots_resident);
    cost.page_in_ms =
        detail::transfer_ms(cost.pages_fetched * geometry.page_bytes, rates.page_in_bytes_per_s);
    cost.expert_ms =
        detail::transfer_ms(cost.expert_misses * geometry.slot_bytes, rates.expert_bytes_per_s);
    return cost;
}

} // namespace ninfer::core
