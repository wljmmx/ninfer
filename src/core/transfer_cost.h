#pragma once

// V10-82: what each leg costs a decode step. V10-83: what the streaming leg is made of.
//
// The capacity bet (core/capacity_bet.h) is a choice between two arrangements: A keeps the experts
// on the Device and pays for it out of the KV pages, B keeps the pages and pays for it out of the
// pool. Until V10-82 that choice was a PRIORITY ("experts first", "KV first") because nothing
// priced the two legs; the measured A-versus-B decode rates at one context were taken with the same
// arena, so they could not rank the arrangements either.
//
// The two bills are both linear in the bytes a step owes, plus one fixed term each:
//   A  the pages a step reads that the Device does not hold: (addressed - resident) x page bytes;
//   B  the experts a step routes that the pool does not hold:  (routed - resident) x slot bytes.
// A caller that knows the rates turns either into milliseconds and the bet can compare them.
//
// V10-83: B's bytes are not a single copy. The engine reads an expert out of the artifact
// (buffered, src/artifact/file_io.cpp:78-107), and only then copies it to the Device - the layer
// blocks in host_->ensure() (expert_stream.cpp:3989-3992), uploads, and drains the upload with
// wait_for_progress() (4083) before its GEMM. One byte therefore pays BOTH rates, in series:
//
//     effective = 1 / (1 / disk + 1 / pcie)
//
// Measured on this machine over the 35B e2 artifact: 6.83/7.15 GB/s buffered sequential (1 GiB
// reads) against 23.2 GB/s on the link, i.e. 5.3 GB/s serialized - 4.4x the PCIe-only figure V10-82
// used. Pricing the link alone is what made V10-82's B bill small next to the measurement. The
// page-in leg is Host memory to Device (the Host KV arena already holds the payload), so it keeps
// its one rate.
//
// V10-83: and a layer that streams pays a round trip the byte rate does not describe - the host
// tier fills the layer's experts, the uploader's prepare() is enqueued, and wait_for_progress()
// drains it before that layer's GEMM can start. V10-82's regression on 35B e2 fixed it at 22.94 ms
// per step over 40 streaming layers: 0.5735 ms per layer, byte-independent (a 129 MiB step and a
// 474 MiB step both paid it). It is charged per layer that has any miss, never per slot.
//
// WHAT THIS MODEL STILL DOES NOT PRICE, stated because the residual error in the report is these
// terms: a fetch that cannot take a free slot also DEMOTES a page first (a second copy of the same
// size), the host tier's own hits are not modelled (a resident expert is free, a page-cache hit is
// not faster than the probed read), and the kernels around the copy are not free either. The rates
// are an optimistic lower bound, and the caller prints all three beside the prediction so a reader
// can see what the number was built from.

#include <cstdint>

namespace ninfer::core {

// Two transfers in series, one byte. A zero on either side means that leg was not measured on this
// path: the other rate is then the whole pipeline (the pre-V10-83 model), never a division by zero
// and never "free". A zero copy rate costs nothing because there is no copy to pay for; a zero disk
// rate is the documented "read through a source that is not the artifact" (a retained Host
// replica), which is exactly a pipeline with one leg.
[[nodiscard]] inline double effective_rate_bytes_per_s(double disk_bytes_per_s,
                                                       double copy_bytes_per_s) noexcept {
    if (copy_bytes_per_s <= 0.0) { return 0.0; }
    if (disk_bytes_per_s <= 0.0) { return copy_bytes_per_s; }
    return 1.0 / (1.0 / disk_bytes_per_s + 1.0 / copy_bytes_per_s);
}

// Bytes per second. Zero means "never measured on this machine": both this header and the bet treat
// a zero as unpriced rather than as free, which is why `measured()` asks for the two Host -> Device
// legs. `disk_bytes_per_s` is the artifact read leg and is optional by construction: a run whose
// experts come from a retained Host replica has no file to read.
struct TransferRates {
    double page_in_bytes_per_s = 0.0;
    double expert_bytes_per_s  = 0.0;
    double disk_bytes_per_s    = 0.0;

    [[nodiscard]] bool measured() const noexcept {
        return page_in_bytes_per_s > 0.0 && expert_bytes_per_s > 0.0;
    }

    // What one expert byte actually costs this pipeline: the artifact read and the Device copy, in
    // series. Equals `expert_bytes_per_s` when no disk leg was measured.
    [[nodiscard]] double expert_effective_bytes_per_s() const noexcept {
        return effective_rate_bytes_per_s(disk_bytes_per_s, expert_bytes_per_s);
    }
};

// The per-layer round trip a streaming layer pays on top of its bytes (see the header comment).
// From V10-82's regression on 35B e2: 22.94 ms/step over 40 layers, slope-removed.
inline constexpr double kStreamLayerFixedMs = 0.5735;

// One decode step's geometry, in the bet's own units. `pages_addressed` is what the step READS
// (the whole live history for a full-attention layer, the window's reach when one is installed) and
// `pages_resident` is what the split leaves on the Device; `experts_routed` is the step's routed
// slots and `expert_slots_resident` what the pool holds. Either pair may name a zero rate's leg as
// fully covered - the difference is clamped at zero, never negative. `expert_layers` is how many
// layers route experts at all: it converts the fixed term above into a per-step figure, and a step
// with no miss prices no layer, because a layer that uploads nothing pays no round trip.
struct StepTransferGeometry {
    std::uint64_t page_bytes            = 0;
    std::uint64_t pages_addressed       = 0;
    std::uint64_t pages_resident        = 0;
    std::uint64_t slot_bytes            = 0;
    std::uint64_t experts_routed        = 0;
    std::uint64_t expert_slots_resident = 0;
    std::uint64_t expert_layers         = 0;
};

struct StepTransferCost {
    std::uint64_t pages_fetched = 0;
    std::uint64_t expert_misses = 0;
    std::uint64_t stream_layers = 0;
    double page_in_ms           = 0.0;
    double expert_bytes_ms      = 0.0;
    double expert_fixed_ms      = 0.0;
    // The whole slot leg: bytes through the serial pipeline plus the per-layer round trips.
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
// more, and a step that owes nothing (every page resident, every route seated) costs exactly zero -
// including the fixed term, which is charged only to layers that actually upload.
[[nodiscard]] inline StepTransferCost price_step_transfers(const StepTransferGeometry& geometry,
                                                           const TransferRates& rates) noexcept {
    StepTransferCost cost;
    cost.pages_fetched = detail::excess_of(geometry.pages_addressed, geometry.pages_resident);
    cost.expert_misses = detail::excess_of(geometry.experts_routed, geometry.expert_slots_resident);
    cost.stream_layers = cost.expert_misses != 0 ? geometry.expert_layers : 0;
    cost.page_in_ms =
        detail::transfer_ms(cost.pages_fetched * geometry.page_bytes, rates.page_in_bytes_per_s);
    cost.expert_bytes_ms = detail::transfer_ms(cost.expert_misses * geometry.slot_bytes,
                                               rates.expert_effective_bytes_per_s());
    cost.expert_fixed_ms =
        static_cast<double>(cost.stream_layers) * kStreamLayerFixedMs;
    cost.expert_ms = cost.expert_bytes_ms + cost.expert_fixed_ms;
    return cost;
}

} // namespace ninfer::core
