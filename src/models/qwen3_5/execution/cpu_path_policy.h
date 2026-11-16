#pragma once

// port (sm_89), P2: the CPU in-place path's costs and call-path decision hooks, wired into the
// v8 controller / parameter bus as decision INPUTS (v8 12.1: the bus is the single registry of
// record - every rate a decision reads is a row here, with its provenance, before anything may
// consult it).
//
// The substrate underneath this header is landed and measured, but nothing in the tree calls it
// from a policy yet:
//   * MoE expert CPU half (ops/moe/kernels/cpu + routed_swiglu/cpu): int8/int16/float tiers
//     behind NINFER_MOE_CPU_INT8, measured 0.0484 ms/expert int8 (the default tier) and 0.0504
//     int16 at Columns=1 on this box (8d07afe3, 2026-11-07). It runs today only as the
//     pool-lack fallback (ffn.cpp's host_routed_ffn), never by a cost decision.
//   * KV partition partial (ops/softmax_attention/cpu/partial.h): one split's attention partial
//     over host-resident rk4v4/rk8v4 pages, in the exact numeric form the GPU grouped instance
//     produces, written into the merge's partial buffers; cost.h carries its CALIBRATED rates
//     (i9-14900K AVX2, scalar mirror single-threaded, min/200, 96 MiB flush, cold-page rotation,
//     2026-11-07) and names this seam's caller as the later KV tiering batch.
//
// The rule both decision functions implement (the port's own statement of it):
//
//   the data is NOT already device-addressable, AND copying it in costs more than computing it
//   in place -> compute on the CPU and hand the result back for the GPU-side merge.
//
// The copy side of every comparison is the link the controller already probes live
// (PolicyController::link_gbps, probe_link: one 32 MiB mapped host-to-device copy, best of 3,
// expert_path_control.h). No placeholder bandwidth row is registered beside it: inventing a
// second number next to a live probe is exactly the two-places-one-value bug the bus exists to
// remove, so a decision asked with no probed bandwidth answers NoData, never a guess.
//
// DEFAULT OFF - NINFER_CPU_PATH_POLICY (off|on, "1" accepted), the forced-switch pattern of
// NINFER_CONTROLLER_SERIAL_OFF / NINFER_CONTROLLER_PREFETCH_OFF. Unset, "off", "0" and anything
// the parser does not recognise (typos included) hold the existing legs: every answerable query
// reads DeviceCopy / PoolCopy, so nothing about how a layer runs changes. A typo may silently
// disable a default-off path, but must never silently arm one. "on" arms the cost model only:
// the two runtime legs exist now (ExpertStream::ensure leaving a miss for the CPU half, behind
// NINFER_CPU_LEG_ACT=on; the KV host-partial gate, behind NINFER_KV_CPU_PARTITION=<split>), and
// both are off by default, so a verdict changes nothing until one of those is named. The shadow
// rule still governs what ships: a leg that has not been measured as a win stays unarmed.
//
// The three deployment modes (v8 12.4) are untouched: these hooks neither read nor move the
// mode, and a verdict costs the same in auto, stable and performance.

#include "models/qwen3_5/execution/param_bus.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ninfer::models::qwen3_5::execution {

// ---- the rows this header owns (names shared by the registration and every reader) ---------
inline constexpr char kCpuKvPartialH24Kv4Row[] = "cpu_kv_partial_h24kv4_ns_per_tok";
inline constexpr char kCpuKvPartialH16Kv2Row[] = "cpu_kv_partial_h16kv2_ns_per_tok";
inline constexpr char kCpuMoeExpertRow[]       = "cpu_moe_expert_ms";
inline constexpr char kCpuPartialUploadRow[]   = "cpu_partial_upload_b_per_tok";
inline constexpr char kCpuPathOnRow[]          = "cpu_path_on";

// CALIBRATED per-token anchors: ns per KV key token at decode width=1, derived from the rates
// ops/softmax_attention/cpu/cost.h calibrated on this machine (qk 1.70e9 int8 MAC/s, pv
// 1.40e9 fp16 MAC/s, pages 2.38e9 B/s; per key = 256 MACs per query head per stage + exp/
// rescale terms + kv_heads * 264 B of page planes):
//   D256H24Kv4, rk4v4 packed K: 24*(256/1.70e9 + 256/1.40e9) + exp/rescale + 4*264/2.38e9
//                            = 8598.5 ns; measured anchor 9.15-9.24 us/key (model reads 6.8%
//                              low on this, its heaviest decode geometry)
//   D256H16Kv2, rk4v4 packed K: 5658.4 ns; the measured anchor sits 0.6% under the model.
// The anchors are codec-blind by cost.h's own design (int8-K adds kv_heads*128 B/key, 2.5% on
// H24 - inside the model's stated ~7% band); `packed_k` stays in the query struct so the
// per-codec recalibration cost.h names has its input ready when that is wired.
inline constexpr double kCpuKvPartialH24Kv4NsPerTok = 8598.5;
inline constexpr double kCpuKvPartialH16Kv2NsPerTok = 5658.4;
// The partial-upload formula's constant part: one split's partial is {256 acc + maximum + sum}
// = 258 FP32 words per query head per token (causal_operands.h allocate_causal_partials), so
// 258 * 4 B per (head, token) - the call multiplies by q_heads * width (cpu_partial_upload_bytes).
inline constexpr double kCpuPartialUploadBPerTok = 1032.0;
// The CPU half's per-expert compute, measured on this box at Columns=1. 2026-11-07 (8d07afe3):
// int8 tier 0.0484 ms, int16 tier 0.0504 ms. Re-measured 2026-11-16 (moe_cpu_bench, 200 iterations,
// 96 MiB flush, six interleaved rounds, min ms/expert at misses=1) and the ORDER differs by form and
// by shape, so a single pair of numbers cannot be right for both:
//   fused form, int8 gate/up 0.0569 vs int16 gate/up 0.0452 - int16 is the faster tier here;
//   split form, one row at a time, 0.395 ms int8 against 0.451 ms int16 at tokens=1 - int8 wins;
//   the same split form at tokens=512 reverses by ~4%.
// These rows still feed decision 2 only, and decision 2 does not actuate (see its comment), so the
// numbers below are a recorded prior rather than a live input. Do not quote them as "the CPU half
// costs 48 us" without the form and the column count beside them.
inline constexpr double kCpuMoeExpertInt8Ms  = 0.0569;
inline constexpr double kCpuMoeExpertInt16Ms = 0.0452;

// What one clean layer saves: the route read-back ffn skips when the placement covers every route
// (expert_stream.h's static_covered() - "the only place the ~0.5 ms read-back can be removed"). It is
// a LAYER cost, not a per-route one: a single route fetched from the host forfeits all of it. That
// asymmetry is the whole of decision 2's layer form below, and it is why the per-expert comparison
// could never see the crossing.
inline constexpr double kCpuReadbackUs = 500.0;

// ---- the forced switch --------------------------------------------------------------------
enum class CpuPathSwitch : std::uint8_t { Off = 0, On = 1 };

// Read once per process; the bus row mirrors the same value for the panel. Only an exact
// "on"/"1" arms the policy (see the header comment for why a typo must hold, not arm).
[[nodiscard]] inline CpuPathSwitch cpu_path_switch() {
    static const CpuPathSwitch value = [] {
        const char* env = std::getenv("NINFER_CPU_PATH_POLICY");
        return env != nullptr && (std::strcmp(env, "on") == 0 || std::strcmp(env, "1") == 0)
                   ? CpuPathSwitch::On
                   : CpuPathSwitch::Off;
    }();
    return value;
}

// Registered from ParamBus::register_startup_params (the declaration sits in param_bus.h; the
// definitions live here so the decision code that reads the rows owns their names and their
// provenance). The four cost rows are L0 frozen: calibrated machine constants, readable and
// printable, never writable by anyone - exactly the tier the bus reserves for geometry. The
// policy row is an L2 mirror of the forced switch with no executor: a forced switch nobody may
// flip at runtime is the point of "forced" (set() refuses it as NoExecutor, like graph_on).
inline void register_cpu_path_params(ParamBus& bus) {
    bus.register_spec({.name     = kCpuKvPartialH24Kv4Row,
                       .initial  = kCpuKvPartialH24Kv4NsPerTok,
                       .lo       = kCpuKvPartialH24Kv4NsPerTok, // the model; the truth is at or
                       .hi       = 9240.0,                      // above it here (measured top)
                       .tier     = ParamTier::L0,
                       .cost_ms  = 0.0,
                       .observe  = "kv copy-vs-partial",
                       .rollback = kCpuKvPartialH24Kv4NsPerTok,
                       .source   = "derived from ops/softmax_attention/cpu/cost.h rates "
                                   "(i9-14900K AVX2 scalar mirror, single-thread, min/200, "
                                   "96 MiB flush, cold-page rotation, 2026-11-07); D256H24Kv4 "
                                   "rk4v4 packed-int4 decode, keys=8192 anchor; measured "
                                   "9.15-9.24 us/key (model -6.8% on this geometry)"});
    bus.register_spec({.name     = kCpuKvPartialH16Kv2Row,
                       .initial  = kCpuKvPartialH16Kv2NsPerTok,
                       .lo       = 5625.0,                       // the measured anchor (model is
                       .hi       = kCpuKvPartialH16Kv2NsPerTok,  // 0.6% high on this geometry)
                       .tier     = ParamTier::L0,
                       .cost_ms  = 0.0,
                       .observe  = "kv copy-vs-partial",
                       .rollback = kCpuKvPartialH16Kv2NsPerTok,
                       .source   = "derived from ops/softmax_attention/cpu/cost.h rates "
                                   "(same calibration as the H24Kv4 row); D256H16Kv2 rk4v4 "
                                   "packed-int4 decode, keys=8192 anchor; measured anchor "
                                   "reads 0.6% under the model"});
    bus.register_spec({.name     = kCpuMoeExpertRow,
                       .initial  = kCpuMoeExpertInt8Ms,
                       .lo       = kCpuMoeExpertInt8Ms,  // the two measured tiers span the
                       .hi       = kCpuMoeExpertInt16Ms, // row: int8 (default) .. int16
                       .tier     = ParamTier::L0,
                       .cost_ms  = 0.0,
                       .observe  = "expert copy-vs-half",
                       .rollback = kCpuMoeExpertInt8Ms,
                       .source   = "measured 2026-11-07 (8d07afe3), the routed swiglu CPU half "
                                   "(ops/moe kernels/cpu groupwise_int), one expert, Columns=1, "
                                   "this box; int8 0.0484 ms (NINFER_MOE_CPU_INT8 default), "
                                   "int16 0.0504 ms (the =16 tier, 96.0%/94.1% of int8 A/B)"});
    bus.register_spec({.name     = kCpuPartialUploadRow,
                       .initial  = kCpuPartialUploadBPerTok,
                       .lo       = kCpuPartialUploadBPerTok, // a structural layout constant,
                       .hi       = kCpuPartialUploadBPerTok, // not a measurement: exact
                       .tier     = ParamTier::L0,
                       .cost_ms  = 0.0,
                       .observe  = "partial upload bytes",
                       .rollback = kCpuPartialUploadBPerTok,
                       .source   = "258 fp32 words (256 acc + maximum + sum, causal_operands.h "
                                   "allocate_causal_partials) x 4 B per query head per token "
                                   "per split; x q_heads x width at call time "
                                   "(cpu_partial_upload_bytes)"});
    bus.register_spec({.name     = kCpuPathOnRow,
                       .initial  = cpu_path_switch() == CpuPathSwitch::On ? 1.0 : 0.0,
                       .lo       = 0.0,
                       .hi       = 1.0,
                       .tier     = ParamTier::L2,
                       .cost_ms  = 0.0,
                       .observe  = "[ctl] cpu verdicts",
                       .rollback = 0.0,
                       .source   = "NINFER_CPU_PATH_POLICY (off|on) - the forced switch; off "
                                   "holds the existing legs, and no executor exists because a "
                                   "forced switch is not flippable at runtime"});
}

// ---- decision 1: one KV split, retrieve-copy vs CPU partial --------------------------------
//
// The call site existed all along and is NOT the missing piece: `rk_kv_cpu_split_gate`
// (ops/softmax_attention/dense/causal_cache/int8/host_partial.h:269) runs between the grouped MMA
// and the merge on every int8 attention call (int8/launch.cu:66,100), and it already takes a split
// index from a switch (NINFER_KV_CPU_PARTITION) and hands that split to the CPU leg. What it cannot
// do is DECIDE, and the reason is not this function: its inputs have no producer.
//
//   * the gate's first precondition is `rk_host_partial_active() != nullptr`, and
//     `rk_host_partial_install()` has NO caller anywhere in the tree (checked 2026-11-16, src and
//     apps) - so every call falls back at `kNoView` and the seam is unreachable, not merely off;
//   * the window the split geometry is priced over is `view->positions[width - 1] + 1`, i.e. it is
//     read FROM the host mirror. Without a mirror there is no per-split key range, and
//     `partition.active(window)` cannot be evaluated;
//   * a split's copy-in bytes are that range's plane bytes (`rk_partial_count_ops().kv_bytes`),
//     which is again a function of the range.
//
// And the engine has nothing that could be mirrored: its Host KV tier holds demoted extents that no
// attention path ever reads back (demote -> spill -> ensure_host_resident in
// program/storage/sequence.cpp and host_kv_store.h), so there are no "host-resident split pages"
// whose copy-in this decision would replace. The host partial op itself is exercised only by
// apps/diag-cpu-partial.
//
// So this seam needs the KV-tiering batch to build the mirror and the split ownership first - the
// gate, the op, the merge contract and this decision are all waiting for that, and wiring the
// decision anywhere today would feed it a range it invented. READ THE PANEL ACCORDINGLY:
// `[ctl] cpu: kv: cpu-partial=0/0` is the absence of a producer, not a measured zero.
struct CpuKvSplitQuery {
    std::uint64_t partition_bytes = 0; // host-resident page bytes the split would copy in
    int keys                      = 0; // the split's key tokens (range.end - range.begin)
    int query_heads               = 0; // production decode: 24 (H24Kv4) or 16 (H16Kv2)
    int kv_heads                  = 0; // 4 or 2
    int width                     = 1; // the query tile's tokens (decode: 1)
    bool packed_k                  = true; // rk4v4 int4 K; false = rk8v4 int8 K (see the anchors)
    bool device_address           = false; // the split's pages already have a device address
};

enum class CpuKvSplitLeg : std::uint8_t {
    DeviceCopy  = 0, // copy the pages in; the GPU instance computes the split
    HostPartial = 1, // CPU partial over the host pages; the partial goes back for the merge
    NoData      = 2, // the model cannot answer (unprobed link, unregistered geometry): hold
};

// One split's partial-upload bytes: the merge buffers this split would have to hand to the GPU
// if it ran as a host partial. Reads the registered constant so the formula has exactly one
// home and every caller - the decision below included - agrees with the panel.
[[nodiscard]] inline double cpu_partial_upload_bytes(int query_heads, int width) noexcept {
    const double per_tok = param_bus().get(kCpuPartialUploadRow);
    if (!(per_tok > 0.0) || query_heads <= 0 || width <= 0) { return -1.0; }
    return per_tok * static_cast<double>(query_heads) * static_cast<double>(width);
}

// Pure cost comparison; `copy_gbps` is the caller's live probed link (the controller's
// link_gbps - this header deliberately holds no CUDA of its own). The optional out-params
// carry the model's two sides in ms (copy-in, and compute+upload for the host leg) so the
// observation line can print what the model would say even while the switch holds the
// existing legs - the framework's shadow view. Estimates fill whenever the model can compute
// them; -1.0 means "no estimate for this query".
[[nodiscard]] inline CpuKvSplitLeg decide_cpu_kv_split(const CpuKvSplitQuery& query,
                                                       double copy_gbps,
                                                       double* est_copy_ms = nullptr,
                                                       double* est_cpu_ms  = nullptr) noexcept {
    if (est_copy_ms != nullptr) { *est_copy_ms = -1.0; }
    if (est_cpu_ms != nullptr) { *est_cpu_ms = -1.0; }
    if (query.device_address) {
        return CpuKvSplitLeg::DeviceCopy; // already device-readable: no leg to choose
    }
    if (query.keys <= 0 || query.partition_bytes == 0) { return CpuKvSplitLeg::NoData; }
    const bool h24kv4 = query.query_heads == 24 && query.kv_heads == 4;
    const bool h16kv2 = query.query_heads == 16 && query.kv_heads == 2;
    if (!h24kv4 && !h16kv2) {
        return CpuKvSplitLeg::NoData; // no calibrated anchor for this geometry: refuse to guess
    }
    const ParamBus& bus = param_bus();
    const double ns_per_tok = bus.get(h24kv4 ? kCpuKvPartialH24Kv4Row : kCpuKvPartialH16Kv2Row);
    const double upload_b   = cpu_partial_upload_bytes(query.query_heads, query.width);
    if (!(ns_per_tok > 0.0) || !(upload_b > 0.0) || !(copy_gbps > 0.0)) {
        return CpuKvSplitLeg::NoData; // no probe, no anchor: never a guess
    }
    const double copy_s    = static_cast<double>(query.partition_bytes) / (copy_gbps * 1e9);
    const double compute_s = ns_per_tok * static_cast<double>(query.keys) * 1e-9;
    const double upload_s  = upload_b / (copy_gbps * 1e9);
    if (est_copy_ms != nullptr) { *est_copy_ms = copy_s * 1000.0; }
    if (est_cpu_ms != nullptr) { *est_cpu_ms = (compute_s + upload_s) * 1000.0; }
    if (cpu_path_switch() == CpuPathSwitch::Off) { return CpuKvSplitLeg::DeviceCopy; }
    return copy_s > compute_s + upload_s ? CpuKvSplitLeg::HostPartial
                                         : CpuKvSplitLeg::DeviceCopy;
}

// ---- decision 2: one missing expert, copy into the pool vs the CPU half ---------------------
//
// Asked only about experts OUTSIDE the hot region - the caller's precondition: the pool's
// placement has already decided they are not resident. The copy side is the measured upload
// (PolicyController::signals.upload_us_per_expert, the EWMA of the read-plus-DMA the pool
// actually pays); the compute side is the calibrated CPU half rate. The runtime leg exists:
// ExpertStream::ensure_split asks this question BEFORE the copy and puts the misses on the same
// CPU mask decision 3 uses, so an expert that would have been fetched is computed on the half
// instead. It is armed by NINFER_CPU_LEG_ACT=on and off by default, because at today's rates it
// measured no win (see the switch's own note) - a leg that is wired is not the same as a leg that
// pays, and only the second one may ship armed.
enum class CpuExpertLeg : std::uint8_t {
    PoolCopy = 0, // upload the expert into the pool (the in-force behaviour)
    CpuHalf  = 1, // compute it in the CPU half, merge the term on the GPU
    NoData   = 2,
};

// Pure; `copy_us_per_expert` is the caller's measured upload. `est_cpu_us` (optional) carries
// the calibrated CPU-side rate so the observation line can print both sides.
//
// SUPERSEDED by the layer form below, and kept only because the panel's per-expert counters name it.
// The comparison here charges one transfer against one route and concludes "copy" at 42 us against
// 45 us - which is right about those two numbers and wrong about the layer, because the copy also
// forfeits kCpuReadbackUs for a layer that would otherwise have been clean. Do not wire this one.
[[nodiscard]] inline CpuExpertLeg decide_cpu_expert_half(double copy_us_per_expert,
                                                         double* est_cpu_us = nullptr) noexcept {
    const double cpu_ms = param_bus().get(kCpuMoeExpertRow);
    if (est_cpu_us != nullptr) { *est_cpu_us = cpu_ms > 0.0 ? cpu_ms * 1000.0 : -1.0; }
    if (!(copy_us_per_expert > 0.0) || !(cpu_ms > 0.0)) { return CpuExpertLeg::NoData; }
    if (cpu_path_switch() == CpuPathSwitch::Off) { return CpuExpertLeg::PoolCopy; }
    return copy_us_per_expert > cpu_ms * 1000.0 ? CpuExpertLeg::CpuHalf : CpuExpertLeg::PoolCopy;
}

// ---- decision 2, layer form: this layer's misses, copy them in or answer them on the half --------
//
// The question is per LAYER, not per expert, because everything the copy costs that is not the
// transfer is layer-shaped:
//
//   copy_us(layer, m) = m * transfer_us + (m > 0 ? kCpuReadbackUs : 0)
//                       + the spill slots the frozen set reserved (p99 3-4, expert_stream.h:374-380)
//                       + a vote against the freeze rule (it counts clean layers)
//                       + a captured expert table that can no longer be captured ("a capture cannot
//                         read a host tally", expert_stream.h:~415)
//   cpu_us(layer, m)  = m * span_us          (no pool state is touched at all)
//
// The first line is the one the arithmetic can settle, and it crosses where
//     m* = kCpuReadbackUs / (span_us - transfer_us)
// With the split form's measured span (~147 us: 71 gate/up + 76 down) and a 42 us upload this is
// m* = 4.8 - the p99 of the miss histogram, i.e. exactly the few misses a frozen layer already
// reserves slots for. With the fused form's ~45 us it is ~167, which is why the fused form should
// keep copying and why nothing was ever wired to this seam before.
//
// The terms below the first line are real but not additive in microseconds here; they are the reason
// m* is an UPPER bound on where the half starts to pay rather than a break-even, and they are
// observable on the stream's own counters (gh1 no_miss/capt_free, miss_tally), not invented here.
struct CpuLayerLegQuery {
    int    misses      = 0;   // routes this layer would otherwise fetch
    double copy_us     = 0.0; // one entry's transfer, as the pool measures it
    double cpu_us      = -1.0; // one route on the CPU half (span, tier fill included)
    double readback_us = kCpuReadbackUs;
};

struct CpuLayerLegVerdict {
    CpuExpertLeg leg  = CpuExpertLeg::NoData;
    double threshold  = -1.0; // m*; < 0 means "no crossing at this span"
    double copy_us    = -1.0; // the layer's copy bill at this miss count
    double cpu_us     = -1.0; // the layer's half bill at the same count
    const char* why   = "no data";
};

[[nodiscard]] inline CpuLayerLegVerdict decide_cpu_layer_leg(const CpuLayerLegQuery& query) noexcept {
    CpuLayerLegVerdict out;
    if (query.misses <= 0 || !(query.copy_us > 0.0) || !(query.cpu_us > 0.0)) { return out; }
    const double m = static_cast<double>(query.misses);
    out.copy_us    = m * query.copy_us + (query.readback_us > 0.0 ? query.readback_us : 0.0);
    out.cpu_us     = m * query.cpu_us;
    if (query.cpu_us > query.copy_us && query.readback_us > 0.0) {
        out.threshold = query.readback_us / (query.cpu_us - query.copy_us);
    }
    if (cpu_path_switch() == CpuPathSwitch::Off) {
        out.leg = CpuExpertLeg::PoolCopy;
        out.why = "policy off";
        return out;
    }
    out.leg = out.cpu_us < out.copy_us ? CpuExpertLeg::CpuHalf : CpuExpertLeg::PoolCopy;
    out.why = out.leg == CpuExpertLeg::CpuHalf ? "half cheaper than copy + read-back" : "copy cheaper";
    return out;
}

// ---- decision 3: one layer's CPU share - the ratio, which is a RESULT ---------------------------
//
// The two decisions above answer "may this seam move at all". This one answers "how much of one
// layer", and it is the only place the port's ratio exists: nothing here writes a percentage and
// nothing reads a fraction from the environment. A layer asks with the facts that have been
// measured and gets a COUNT back.
//
//   cost_gpu(e) = bytes(e) / link_gbps + slack
//   cost_cpu(e) = flops(e) / cpu_gflops + (1 - tier_warm) * cold_read_us
//
// `slack` is not invented. The controller already measures what one expert upload costs end to end
// (PolicyController::signals.upload_us_per_expert: read, staging and DMA as the pool actually pays
// them), so the copy side is that measurement when it exists and the bare link time when it does
// not - a route moves three entries, hence the x3. The compute side is the calibrated 778 GFLOPS
// (bench/ops/moe_cpu_expert_bench.cpp, i9-14900K AVX2, 12.6 us per expert at the L3-hot tier) until
// the [host] line has timed this box's own half; after that it is that measurement
// (note_cpu_half), so the rate follows the machine instead of a constant. The tier read is charged
// only for the share of experts the pinned tier does not already hold: a warm slot costs the CPU
// nothing, a cold one costs a measured read.
//
// The count is then capped and trimmed by the two queues this box actually has:
//   * one layer may spend at most max(the Device critical path, kCpuLayerFloorUs) on the CPU, and
//     lanes share that budget - two lanes buy half each, because the team is the same team;
//   * a link with nothing queued has nothing to take back: the previous layer's upload is compared
//     against the wall it happened in, and a copy that took under kLinkBusyShare of it is idle;
//   * a CPU half that cost more than kCpuBusyShare of that same wall last layer IS the critical
//     path, so the count halves instead of growing.
//
// Every term is either calibrated-and-named or measured; none is a guess wearing a constant's
// clothes. The result is reported per layer ([cpu-path]) and per token ([host] share).
//
// ---- decision 4: one route of a batch, are the shared costs spread or kept whole? ----------------
//
// NOT IMPLEMENTED, AND IT IS THE ONE DECISION THE MODEL CURRENTLY GETS WRONG BY CONSTRUCTION.
//
// Both sides of the comparison are per ROUTE while one layer's work is per TOKEN, and the two sides do
// not repeat the same way:
//   * the tier read (and the upload the CPU half spares the Device) happens once per expert and serves
//     every token of the batch - so per token it is 1/T of the bill;
//   * the arithmetic repeats per token - so per token it is flat.
// The model holds a single scalar, `rate.compute_us_per_expert`, and once note_cpu_half has filled it
// the per-token `flops_per_expert` term is not consulted at all: whichever shape the rate was last
// measured at becomes the price of every shape.
//
// THERE IS A CROSSOVER, AND IT IS WHY "just amortise" IS THE WRONG ANSWER. What is shared is not only
// the tier read; it is every fixed cost of serving the route at all - the per-task quantisation, the
// barrier round, the tier fill - and those are shared only while there are few enough routes to share
// them. The measurements bracket both regimes:
//   * amortised: moe_cpu_expert_bench, 256 experts, one shape - 6.2 us/expert-token at tokens=1
//     against 4.0 us at tokens=512, i.e. 35% cheaper per token once one task serves a real batch;
//   * contended: 8-way concurrency, gate/up per expert 92.8 us at one forced route against 175.4 us
//     at ten (+89%), i.e. per-route cost RISES once the routes compete for the same team.
// So the price is not one of the two forms but a switch between them, and the switch is a threshold
// this header does not have yet:
//     while the half is under its layer budget:  cost_cpu_per_token(T) = read/T + compute
//     once it saturates (kCpuBusyShare already trims the count): cost_cpu_per_route = the whole bill
// The two-term fit reproduces the amortised end alone (compute 4.0 us, read 2.2 us: 2.2 + 4.0 = 6.2 at
// T=1, 4.0 + ~0 at T=512); the whole-bill form is what `span_us_per_expert` measures today, and it is
// right below the crossover - which is most of the decode traffic this port actually serves.
//
// Deliberately not wired now: the crossover depends on T and on how many routes the layer gives the
// half, and structured speculation changes both (a verify pass is a batch of tree nodes). A threshold
// fitted on today's shapes would be re-fitted the moment the tree lands. This decision is upstream of
// that work, not beside it.

// ---- decision 5: how many prediction slots the region gets, if it exists at all ------------------
//
// The size is a MODEL fact times a MEASURED fact, never a constant: it scales with top_k, with the
// number of runs a split layer has (three), and with how much of the layer is actually cold. Measured
// on the 125B decode path, 2026-11-16:
//
//   residency 57.5% of routed experts held by every run  ->  cold share 42.5%   ([gh1] residency)
//   [pf-region] hits=0 calls=0 with PREFETCH, with PREFETCH_PREDICT, and with both: the split decode
//   path never calls set_predicted/stage_predicted - those live on the fused path (ffn.cpp:967,
//   ffn.cpp:1564, text.cpp:2491) - so a carved region pays its residency cost and stages nothing
//   (1.5 tok/s against 7.5 with the region off).
//
// So the rules encoded here, in order:
//   * nothing stages on this path -> 0 slots, whatever anyone asks for;
//   * staging present, no rate yet -> the historical prior (2 x top_k), because a region that has
//     never been measured is not worth more than the one that was;
//   * cold share known -> scale the prior by it (a layer that is mostly resident wants a small
//     region);
//   * hit rate known -> scale by cold_share x hit_rate, which is the slots that will actually be
//     used. The -28% per-node target needs 28/42.5 ~= 66% of the cold set caught, so a poor rate
//     shrinks the region instead of paying for slots nobody reads.
//   * the pool clamps it to a quarter of a bank's window, and so does this.
struct PrefetchSizingQuery {
    int    top_k               = 0;
    int    pool_slots_per_bank = 0;
    double cold_share          = -1.0; // 1 - residency; <0 unknown
    double hit_rate            = -1.0; // [pf-region] hits/calls; <0 unknown
    bool   staging_reachable   = false;
};

struct PrefetchSizing {
    int slots_per_bank = 0;
    int reason         = 0; // 0 no staging, 1 prior, 2 cold-scaled, 3 rate-scaled
};

[[nodiscard]] inline PrefetchSizing decide_prefetch_slots(const PrefetchSizingQuery& query) noexcept {
    PrefetchSizing out;
    if (!query.staging_reachable || query.top_k <= 0 || query.pool_slots_per_bank <= 0) { return out; }
    const int ceiling = std::max(1, query.pool_slots_per_bank / 4);
    int want          = 2 * query.top_k;
    out.reason        = 1;
    if (query.cold_share >= 0.0) {
        want       = static_cast<int>(query.cold_share * static_cast<double>(want) + 0.5);
        out.reason = 2;
    }
    if (query.hit_rate >= 0.0) {
        // Only the slots that the measured hit rate will actually serve are worth holding.
        want       = static_cast<int>(query.cold_share > 0.0
                                          ? query.hit_rate * query.cold_share *
                                                static_cast<double>(2 * query.top_k) + 0.5
                                          : static_cast<double>(want));
        out.reason = 3;
    }
    out.slots_per_bank = std::min(std::max(want, 0), ceiling);
    return out;
}

// ---- the prefetch port: what the controller needs to size a prediction region -------------------
//
// The strategies all live here, in the controller, and the prediction side only REPORTS. That is the
// same shape as the budget question for KV and the expert pool: every consumer states what it asks
// for and what was observed, one place divides. Nothing on the prediction side decides its own size.
//
// Signals come from code that already counts them:
//   cold_share  [gh1] residency   (residency_inside_/residency_routes_, dynamic per layer)
//   hit_rate    [pf-region] hits/calls
//   cpu_share   [ctl] cpu share   (a route the half took never needs a prediction)
//   freeze      the stream's own frozen state (a pinned set shrinks what prefetch can add)
//   lanes       control.signals.lanes_busy (>= 6 switches the replacement policy wholesale)
// and the geometry (top_k, runs_per_layer, slot bytes) is read from the artifact layout, never from
// a name: the same model packaged two ways has the same geometry.
struct PrefetchSignals {
    double cold_share          = -1.0;
    double hit_rate            = -1.0;
    double cpu_share           = -1.0;
    bool   staging_reachable   = false;
    bool   freeze_active       = false;
    int    lanes               = 1;
    int    top_k               = 0;
    int    runs_per_layer      = 1;
    int    pool_slots_per_bank = 0;
    // C3: the model's own depth, so the cutoff below can be derived instead of chosen. `layers` is how
    // many layers the pool holds (banks / runs_per_layer, i.e. the length of a step on this model);
    // `ring_depth` is the prediction ring's depth when one is allocated (0 = none), because a guess
    // older than the ring can never arrive; `max_layers_ahead` is the controller's own cap, 0 meaning
    // "derive it from the two above".
    int    layers              = 0;
    int    ring_depth          = 0;
    int    max_layers_ahead    = 0;
    // C4: the region the pool actually carved (0 = none). The plan cannot ask for more slots than
    // exist: a decision that outvoted the carve would arm fills into a ring that has no room, and the
    // pool's own set_prefetch_experts() would drop the difference silently.
    int    region_slots        = 0;
};

inline PrefetchSignals& prefetch_signals() {
    static thread_local PrefetchSignals signals;
    return signals;
}

inline void note_prefetch_signals(const PrefetchSignals& signals) { prefetch_signals() = signals; }

// The port itself: no arguments, because the numbers must be the ones that were measured, not the
// ones a caller would like to state. A route the CPU half took is removed from the cold set first -
// prefetching for it would be paying twice.
[[nodiscard]] inline PrefetchSizing decide_prefetch_slots() noexcept {
    const auto& signals = prefetch_signals();
    PrefetchSizingQuery query;
    query.top_k               = signals.top_k;
    query.pool_slots_per_bank = signals.pool_slots_per_bank;
    query.cold_share = signals.cold_share >= 0.0
                           ? signals.cold_share * std::max(0.0, 1.0 - std::max(0.0, signals.cpu_share))
                           : -1.0;
    query.hit_rate          = signals.hit_rate;
    query.staging_reachable = signals.staging_reachable;
    return decide_prefetch_slots(query);
}

// ---- C3: how far ahead the region may fill -------------------------------------------------------
//
// Decision 5 answers "how many slots". This answers "how many LAYERS those slots may be spent on", and
// it is the same kind of question: a cost divided by a demand, with both sides read from the model and
// from a measurement.
//
//   demand(layer) = cold_share x top_k          (one bank's expected fill for one layer: the routes
//                                                that are cold, which is exactly what a prediction
//                                                of that layer would add to the region)
//   affordable    = slots / max(1, demand)      (how many layers the region can hold at once)
//   budget        = the controller's cap, else the ring's depth when one exists, else the model's own
//                   layer count (a ring cannot deliver a guess older than its depth, and a fill for a
//                   layer past the end of a step is a guess about the next token)
//   cutoff        = min(affordable, budget, layers)
//
// A cutoff of 0 means the region cannot hold even one layer's cold delta: nothing may be filled, and
// that is a RESULT of the arithmetic, not a switch. This is what makes "armed" a number rather than a
// policy flag - a 4-slot region on a 42.5%-cold, top_k=8 layer has demand 4, so it may run one layer
// ahead; a 2-slot region has cutoff 0 and stays shut.
struct PrefetchPlan {
    int         slots_per_bank  = 0;  // decision 5's answer
    int         delta_per_layer = 0;  // one bank's expected cold delta (demand above)
    int         cutoff          = 0;  // layers ahead the region may be filled for
    bool        armed           = false;
    int         reason          = 0;  // PrefetchSizing::reason, or the C3 refusal below
    const char* why             = "no staging";
};

// The experiment override, read the same way NINFER_CPU_PATH_FORCE is: an A/B that had to beat the
// measured rate before it could measure anything would measure the rate, not the mechanism. 0/unset
// means "no override" - the controller's decision governs. The value still cannot exceed the region
// that was carved, so an override cannot name slots that do not exist.
[[nodiscard]] inline int prefetch_slots_forced() {
    static const int value = [] {
        const char* env = std::getenv("NINFER_PREFETCH_SLOTS");
        if (env == nullptr || *env == '\0') { return 0; }
        const long asked = std::strtol(env, nullptr, 10);
        return asked <= 0 ? 0 : static_cast<int>(asked);
    }();
    return value;
}

[[nodiscard]] inline PrefetchPlan decide_prefetch_plan() noexcept {
    const auto& signals = prefetch_signals();
    const auto  sizing  = decide_prefetch_slots();
    PrefetchPlan plan;
    plan.slots_per_bank = sizing.slots_per_bank;
    plan.reason         = sizing.reason;
    const int forced    = prefetch_slots_forced();
    if (forced > 0) {
        plan.slots_per_bank = forced;
    }
    if (signals.region_slots > 0) {
        // The carve is the physical ceiling. A decision above it would name slots the ring does not
        // have; the pool would place what fits and the rest would silently be a demand miss.
        plan.slots_per_bank = std::min(plan.slots_per_bank, signals.region_slots);
    }
    if (!signals.staging_reachable) {
        plan.why = "staging not reachable";
        return plan;
    }
    if (signals.top_k <= 0) {
        plan.why = "no routing width";
        return plan;
    }
    if (plan.slots_per_bank <= 0) {
        plan.why = "no slots";
        return plan;
    }
    const double cold = signals.cold_share >= 0.0 ? signals.cold_share : 1.0;
    plan.delta_per_layer =
        std::max(1, static_cast<int>(cold * static_cast<double>(signals.top_k) + 0.5));
    const int affordable = plan.slots_per_bank / plan.delta_per_layer;
    int budget = signals.max_layers_ahead > 0
                     ? signals.max_layers_ahead
                     : (signals.ring_depth > 0 ? signals.ring_depth : signals.layers);
    if (signals.layers > 0 && budget > signals.layers) { budget = signals.layers; }
    plan.cutoff = std::min(std::max(0, affordable), std::max(0, budget));
    plan.armed  = plan.cutoff > 0;
    plan.why    = plan.armed
                      ? (forced > 0 ? "env override"
                                    : (sizing.reason == 3 ? "rate-scaled" : "cost model"))
                      : "depth";
    return plan;
}

inline constexpr double kCpuGflopsCalibrated = 778.0; // the benchmark's own effective rate
inline constexpr double kCpuLayerFloorUs     = 250.0; // a layer shorter than this still gets a floor
inline constexpr double kLinkBusyShare       = 0.10;  // upload/wall under this: the link is idle
inline constexpr double kCpuBusyShare        = 0.50;  // cpu/wall over this: the half is the path
inline constexpr int    kCpuRampStep         = 2;     // routes the count may grow per layer
inline constexpr int    kCpuReprobeLayers    = 40;    // idle layers before one route is re-measured

struct CpuLayerQuery {
    int    routes              = 0; // distinct routes this layer would otherwise upload
    int    tokens              = 1; // columns this layer verifies (decode: 1; a verify pass: T)
    double bytes_per_expert    = 0.0; // every run of the layer summed: one route's link traffic
    double flops_per_expert    = 0.0; // the CPU half's arithmetic for one route
    double link_gbps           = 0.0; // PolicyController::link_gbps (probed, never assumed)
    double upload_us_per_entry = 0.0; // PolicyController::signals.upload_us_per_expert
    double cold_read_us        = 0.0; // measured read of one cold expert (0: none measured yet)
    double tier_warm           = -1.0; // share of tier fills answered without a read; <0 unknown
    double gpu_layer_ms        = 0.0; // the Device critical path (0: unmeasured)
    double upload_ms           = 0.0; // the previous layer's upload (the link's queue depth)
    double cpu_ms              = 0.0; // the previous layer's CPU half
    double wall_ms             = 0.0; // ... and both were measured against this wall
    int    lanes               = 1;
};

struct CpuLayerBudget {
    int    count       = 0;
    double cost_gpu_us = -1.0;
    double cost_cpu_us = -1.0; // compute + the cold read the tier would charge
    double budget_us   = -1.0;
    double cpu_gflops  = 0.0;
    const char* why    = "off";
    // The link was measured idle this layer (`why` says so in words). Exposed as a flag because
    // decision 2 has to agree with it: its copy side prices one marginal transfer, and when the link
    // has spare capacity that transfer is already paid for. A second decision that moved routes off
    // an idle link would save nothing and buy CPU time - which is exactly what it measured when it
    // was first allowed to act (all ten routes to the half, 1.5 tok/s).
    bool link_idle = false;
};

// The CPU rate this box has measured. Thread-local like the bus and the ledger: the half runs on
// the engine worker, so its rate belongs to that worker's measurement, not to the process.
struct CpuHalfRate {
    // Two rates, because the two questions are different:
    //   compute  - what one route costs the arithmetic alone (the marginal side of the comparison);
    //   span     - what one route costs the CPU side of the layer end to end, tier fill included
    //              (the number the layer budget is divided by).
    // The span is the BEST observed, not an average: the first fill of a process pays arena and
    // page first-touch that never comes back, and a policy that priced its steady state on that
    // would never run again. Re-probes (kCpuReprobeLayers) keep the estimate from going stale.
    double compute_us_per_expert = -1.0;
    double span_us_per_expert    = -1.0;
    // The two rates above are per-expert figures measured at SOME route count, and the two were not
    // the same one: `span` kept the best (smallest) observation while `compute` ran an EWMA over
    // every observation, so on 2026-11-16 a run read span=57.8 us against compute=1401 us - a 24x
    // disagreement between two numbers that describe the same kernel. Any crossing derived from them
    // (m* = readback/(span - copy)) was therefore arithmetic on mismatched sizes, which is how a
    // 12-miss threshold landed on layers that only ever had 8. The pair is now kept PER SIZE, both
    // halves from the same layer observation, and the decisions price a layer at the size it has.
    struct Sample {
        int    count                = 0;   // routes the CPU half actually computed
        int    tokens               = 0;   // columns that layer verified
        double span_us_per_expert   = -1.0;
        double compute_us_per_expert = -1.0;
    };
    // route counts 1, 2, 3-4, 5-8, 9-16, 17-32, 33-64, 65+ (bucket = ceil(log2(count))).
    static constexpr int kSampleBins = 8;
    std::array<Sample, kSampleBins> samples{};
    int    span_count            = 0;  // the size `span_us_per_expert` was measured at
    int    compute_count         = 0;
    // Decision 4's two terms. `compute_us_per_token` is the arithmetic per token (what repeats for a
    // batch) and `measured_tokens` is the column count it was measured at, so a rate measured at one
    // shape is not silently reused as the price of another. Both stay -1 until a batch has actually
    // run through the half.
    double compute_us_per_token  = -1.0;
    int    measured_tokens       = 0;
    double cpu_gflops            = kCpuGflopsCalibrated;
    int    last_count            = 0;  // the count the previous layer ran with (the ramp's anchor)
    int    idle_layers           = 0;  // layers since the CPU half last took a route (re-probe clock)
};

inline CpuHalfRate& cpu_half_rate() {
    static thread_local CpuHalfRate rate;
    return rate;
}

// The bucket a route count falls in. Both halves of one observation are stored under the same
// bucket, which is what makes a sampled pair a same-size pair by construction.
[[nodiscard]] inline int cpu_sample_bin(int count) noexcept {
    int bin = 0;
    while ((1 << bin) < count && bin + 1 < CpuHalfRate::kSampleBins) { ++bin; }
    return bin;
}

// The observation that prices `count` routes: the same bucket when this box has run that size,
// otherwise the nearest SMALLER one. Smaller is the conservative direction - it amortises the fixed
// part over fewer routes, so its per-expert figure is the larger one - and "nearest larger" is only
// used when nothing small exists. -1 means this box has never timed a comparable size, and a caller
// must then say NoData rather than reuse a rate measured somewhere else.
[[nodiscard]] inline const CpuHalfRate::Sample* cpu_sample_for(int count) noexcept {
    if (count <= 0) { return nullptr; }
    const auto& rate = cpu_half_rate();
    const int   bin  = cpu_sample_bin(count);
    for (int i = bin; i >= 0; --i) {
        if (rate.samples[static_cast<std::size_t>(i)].count > 0) { return &rate.samples[static_cast<std::size_t>(i)]; }
    }
    for (int i = bin + 1; i < CpuHalfRate::kSampleBins; ++i) {
        if (rate.samples[static_cast<std::size_t>(i)].count > 0) { return &rate.samples[static_cast<std::size_t>(i)]; }
    }
    return nullptr;
}

// One route's span at this layer's size, -1 when the box has not run a comparable size yet.
[[nodiscard]] inline double cpu_route_us_for(int count) noexcept {
    if (const auto* sample = cpu_sample_for(count); sample != nullptr) {
        return sample->span_us_per_expert;
    }
    return -1.0;
}

// One route's arithmetic per token at this layer's size, read off the same observation as the span
// above. Priced per token because a verify pass repeats the arithmetic T times and pays the tier
// read once - the decision-4 shape term, now with a size attached.
[[nodiscard]] inline double cpu_route_us_per_token_for(int count) noexcept {
    const auto* sample = cpu_sample_for(count);
    if (sample == nullptr || sample->compute_us_per_expert <= 0.0 || sample->tokens <= 0) {
        return -1.0;
    }
    return sample->compute_us_per_expert / static_cast<double>(sample->tokens);
}

// One measured CPU half: `experts` routes in `cpu_ms`. The effective per-route time is what the
// next budget is priced with; the GFLOPS figure is printed beside it so the calibration can be
// checked against the benchmark's 778 rather than trusted.
// One measured CPU half: `experts` routes, `span_ms` from the tier fill to the end of the
// arithmetic, `compute_ms` the arithmetic alone. The compute rate is an EWMA; the span is the best
// observed, and the printed GFLOPS figure is read off the compute rate so the calibration can be
// checked against the benchmark's 778 rather than trusted.
inline void note_cpu_half(int experts, int tokens, double span_ms, double compute_ms,
                          double flops_per_expert) noexcept {
    if (experts <= 0) { return; }
    auto& rate = cpu_half_rate();
    const double span = span_ms * 1000.0 / static_cast<double>(experts);
    if (span > 0.0 && (rate.span_us_per_expert < 0.0 || span < rate.span_us_per_expert)) {
        rate.span_us_per_expert = span;
        rate.span_count         = experts;
    }
    if (compute_ms > 0.0) {
        const double per = compute_ms * 1000.0 / static_cast<double>(experts);
        // The same-size record: this observation's span AND its compute, filed under the count that
        // produced both, and only replaced by a BETTER observation at the same size. Min-per-size
        // keeps the original intent (the rate a warmed-up half reaches, not the first touch of an
        // arena) while removing the thing that made the two printed figures disagree: an average over
        // 1-route and 64-route layers is a rate no layer ever ran at.
        auto& sample = rate.samples[static_cast<std::size_t>(cpu_sample_bin(experts))];
        if (span > 0.0 && (sample.count == 0 || span < sample.span_us_per_expert)) {
            sample.count                 = experts;
            sample.tokens                = tokens > 0 ? tokens : 1;
            sample.span_us_per_expert    = span;
            sample.compute_us_per_expert = compute_ms > 0.0 ? per : -1.0;
        }
        rate.compute_us_per_expert = rate.compute_us_per_expert > 0.0
                                         ? rate.compute_us_per_expert + (per - rate.compute_us_per_expert) / 8.0
                                         : per;
        rate.compute_count = experts;
        // Decision 4: the same measurement read per TOKEN. A layer that verified T columns paid for T
        // tokens of arithmetic, so dividing by T is what makes two shapes comparable; without it the
        // measured per-route rate becomes the price of every shape (the bug decision 4 names).
        if (tokens > 0) {
            const double per_token = per / static_cast<double>(tokens);
            rate.compute_us_per_token = rate.compute_us_per_token > 0.0
                                            ? rate.compute_us_per_token + (per_token - rate.compute_us_per_token) / 8.0
                                            : per_token;
            rate.measured_tokens = tokens;
        }
        if (flops_per_expert > 0.0) {
            rate.cpu_gflops = flops_per_expert / (rate.compute_us_per_expert * 1e-6) / 1e9;
        }
    }
}

// One route's price on the CPU half, for a caller that has no layer query to hand (decision 2's layer
// form). The measured span when this box has one - it includes the tier fill, which a layer-level
// comparison has to pay - and the recorded prior otherwise.
// One route's price on the CPU half, for a caller that has no layer query to hand (decision 2's layer
// form). The measured span when this box has one - it includes the tier fill, which a layer-level
// comparison has to pay - and the recorded prior otherwise. `count` asks for the size-scoped sample
// when the box has timed that size (P3.1: legitimate now that both sides of every comparison carry the
// same tier read; a rate fitted at one size is not the price of another).
[[nodiscard]] inline double cpu_route_us_estimate(int count = 0) noexcept {
    if (count > 0) {
        if (const auto* sample = cpu_sample_for(count);
            sample != nullptr && sample->span_us_per_expert > 0.0) {
            return sample->span_us_per_expert;
        }
    }
    const auto& rate = cpu_half_rate();
    if (rate.span_us_per_expert > 0.0) { return rate.span_us_per_expert; }
    const double ms = param_bus().get(kCpuMoeExpertRow);
    return ms > 0.0 ? ms * 1000.0 : -1.0;
}

// Probe (default off, NINFER_CPU_PATH_FORCE=N): pin the ratio instead of letting the cost model
// decide it, so the tier fill and the two arms can be compared at a fixed count. A measurement
// switch, not a policy: the default path never reads it, and no value of it is a default. Without
// it the half never runs at all at this pool size (the copy wins), which means a change to its
// kernel shape has no way to be measured.
[[nodiscard]] inline int cpu_path_forced_count() {
    static const int value = [] {
        const char* env = std::getenv("NINFER_CPU_PATH_FORCE");
        if (env == nullptr || *env == '\0') { return -1; }
        const long count = std::strtol(env, nullptr, 10);
        return count < 0 ? -1 : static_cast<int>(count);
    }();
    return value;
}

[[nodiscard]] inline CpuLayerBudget decide_cpu_layer_budget(const CpuLayerQuery& query) noexcept {
    CpuLayerBudget out;
    if (cpu_path_switch() == CpuPathSwitch::Off) {
        out.why = "policy off";
        return out;
    }
    if (query.routes <= 0) {
        out.why = "no routes";
        return out;
    }
    // The probe bypasses the model but not the switch: it exists to measure a fixed ratio.
    if (const int forced = cpu_path_forced_count(); forced > 0) {
        out.count       = std::min(forced, query.routes);
        out.why         = "forced";
        out.cost_gpu_us = -1.0;
        out.cost_cpu_us = -1.0;
        out.budget_us   = -1.0;
        cpu_half_rate().last_count  = out.count;
        cpu_half_rate().idle_layers = 0;
        return out;
    }
    if (!(query.link_gbps > 0.0)) {
        out.why = "no link probe";
        return out;
    }
    auto& rate         = cpu_half_rate();
    out.cpu_gflops     = rate.cpu_gflops > 0.0 ? rate.cpu_gflops : kCpuGflopsCalibrated;
    // The two sides are MARGINAL, and BOTH pay one trip through the tier's entry image: the CPU half
    // fills the entry it computes over, and the pool reads the same bytes to fill the seat it copies
    // into. So the read is charged to both and cancels in the difference, which is what makes the
    // comparison the port is built on: one PCIe trip versus one CPU compute.
    //
    // P3.1: the copy side used to drop that read whenever an upload measurement existed (the 3x form
    // below), while the CPU side's span - the number it is divided by - measures the fill from end to
    // end. Two scopes, one comparison: on a cold tier the span read 8186 us/route while the copy side
    // read 12.6 us/route, and the crossing m* = readback/(span - copy) moved with the tier's
    // temperature instead of with the work. Both sides now carry the same read, from the same
    // measurement (`cold_read_us`, the uploader's own observed per-expert read scaled by the cold
    // share), so what is left in the difference is compute - transfer, which is a property of the box.
    const double link_us = query.bytes_per_expert / (query.link_gbps * 1e9) * 1e6;
    const double cold_share =
        query.tier_warm < 0.0 ? 1.0 : std::max(0.0, 1.0 - query.tier_warm);
    const double read_us = cold_share * query.cold_read_us;
    // The copy side: the transfer, the read the fill needs, and the slack a measured upload shows
    // above the bare transfer. The second form is a MAX and not a sum because the measured upload may
    // already contain the read (it timed a real copy out of whatever the tier held); adding the model
    // term on top of it would charge the same bytes twice, and the max keeps the estimate honest on
    // both regimes - cold (the model term dominates) and warm (the measured copy dominates).
    const double gpu_us  = std::max(link_us + read_us,
                                    query.upload_us_per_entry > 0.0
                                        ? 3.0 * query.upload_us_per_entry
                                        : 0.0);
    // The compute side, priced for the shape this layer actually runs (decision 4). The tier read is
    // paid once per expert and serves every token of the layer, the arithmetic repeats per token, so
    // the read is charged per token as read/T while the compute is charged as T * per-token. Without
    // the T the measured per-route scalar silently became the price of every shape - a rate fitted at
    // decode would overprice a verify pass by T, and one fitted at a batch would underprice decode.
    // At T == 1 this reduces to the whole-bill form (compute + read), which is what the decode path
    // has always been priced with.
    const double tokens   = static_cast<double>(std::max(1, query.tokens));
    const double compute_us =
        rate.compute_us_per_token > 0.0
            ? tokens * rate.compute_us_per_token
            : (rate.compute_us_per_expert > 0.0
                   ? rate.compute_us_per_expert
                   : query.flops_per_expert / (out.cpu_gflops * 1e9) * 1e6);
    const double cpu_us     = compute_us + read_us / tokens;
    out.cost_gpu_us         = gpu_us;
    out.cost_cpu_us         = cpu_us;
    if (!(gpu_us > cpu_us)) {
        out.why = "copy already cheaper";
        return out;
    }
    double budget_us = std::max(query.gpu_layer_ms > 0.0 ? query.gpu_layer_ms * 1000.0 : 0.0,
                                kCpuLayerFloorUs);
    if (query.lanes > 1) { budget_us /= static_cast<double>(query.lanes); }
    out.budget_us = budget_us;
    // How many routes fit: the CPU half's own measured span per route when this box has timed one -
    // that span includes the tier fill, which is serial on this thread and is precisely what makes
    // the half the critical path when it grows past the copy it is meant to overlap.
    //
    // P3.1: the SIZE-SCOPED rate is used again, which is why the note that used to stand here is gone.
    // It was withdrawn because the span it measures charged the tier read to the CPU side while the
    // copy side cancelled that same read - so a size whose fill was cold was divided by a cold span
    // and compared against a warm copy. Both sides now carry the read (see gpu_us above), so the
    // per-size sample is the right divisor and the fallback is the best-observed span, not a rate
    // fitted at another size.
    const auto* scoped_sample = cpu_sample_for(query.routes);
    const double scoped_us    = scoped_sample != nullptr ? scoped_sample->span_us_per_expert : -1.0;
    const double per_route_us = scoped_us > 0.0
                                    ? scoped_us
                                    : (rate.span_us_per_expert > 0.0 ? rate.span_us_per_expert
                                                                     : cpu_us);
    int count = per_route_us > 0.0 ? static_cast<int>(budget_us / per_route_us) : 0;
    // Queue depth, both sides, from the previous layer of this same stream. An upload nobody has
    // timed yet (-1/0 ms) is not an idle link: the trim only fires on a measurement.
    if (query.wall_ms > 0.0 && query.upload_ms > 0.0 &&
        query.upload_ms < kLinkBusyShare * query.wall_ms) {
        out.why = "link idle";
        out.link_idle = true;
        count   = 0;
    } else if (query.wall_ms > 0.0 && query.cpu_ms > kCpuBusyShare * query.wall_ms) {
        out.why = "cpu half busy";
        count   = count / 2;
    } else {
        out.why = "cost model";
    }
    out.count = std::min(count, query.routes);
    // The re-probe. A count of zero ends the measurement, and a measurement that can never be
    // retaken is a latch, not a policy: after kCpuReprobeLayers idle layers the actuator takes ONE
    // route back, so a regime that was priced on a cold start gets re-measured at its own cost. One
    // route in forty is the price of not being wrong forever.
    if (out.count == 0) {
        if (++rate.idle_layers >= kCpuReprobeLayers) {
            rate.idle_layers = 0;
            out.count        = std::min(1, query.routes);
            out.why          = "re-probe";
        }
    } else {
        rate.idle_layers = 0;
    }
    // The ramp. A count is a result, but it is reached one layer at a time: a model that has only
    // ever seen a small half cannot say what a large one costs, and the first layer that guesses
    // "all of them" pays the guess in full. Additive here, multiplicative in the trims above -
    // classic AIMD, and the reason a bad first estimate costs a couple of layers instead of a run.
    out.count = std::min(out.count, rate.last_count + kCpuRampStep);
    rate.last_count = out.count;
    return out;
}

// ---- read-only observation: the [ctl] panel's cpu line ----------------------------------------
//
// The counts ARE the hit/hit-rate observables: kv_host_partial/expert_cpu_half are the hits,
// the answered denominators give the rates, and the last estimates are the model's two sides
// for the most recent query (what it would say - the shadow view the switch may be holding
// back). Thread-local like the bus and the controller: every seam lives on the engine worker.
struct CpuPathLedger {
    std::uint64_t kv_answered     = 0;
    std::uint64_t kv_host_partial = 0;
    std::uint64_t kv_device_copy  = 0;
    std::uint64_t kv_nodata       = 0;
    std::uint64_t expert_answered  = 0;
    std::uint64_t expert_cpu_half  = 0;
    std::uint64_t expert_pool_copy  = 0;
    std::uint64_t expert_nodata     = 0;
    double last_kv_copy_ms    = -1.0;
    double last_kv_cpu_ms     = -1.0;
    double last_expert_copy_us = -1.0;
    double last_expert_cpu_us  = -1.0;
    // Decision 3's ledger: the ratio the actuator actually applied. `cpu_routes` is the numerator
    // (routes the CPU half computed), `routes_seen` the denominator (routes the layers had), and the
    // two are counted at different points of one layer - routes in ensure(), cpu routes in the CPU
    // half - so `note_layer_cpu` is what closes a layer and prints it.
    std::uint64_t cpu_routes    = 0;
    std::uint64_t routes_seen   = 0;
    std::uint64_t layer_cpu     = 0;
    double last_layer_cpu_ms    = 0.0;
    double last_cost_gpu_us     = -1.0;
    double last_cost_cpu_us     = -1.0;
    double last_budget_us       = -1.0;
    // Decision 2's layer form: how often a layer was offered the choice and what it said, plus the
    // crossing the model derived for the last layer's span.
    std::uint64_t leg_answered  = 0;
    std::uint64_t leg_cpu_half  = 0;
    std::uint64_t leg_pool_copy = 0;
    std::uint64_t leg_nodata    = 0;
    // ... and what the actuator DID about it. Separate from the three above on purpose: the verdict
    // counters say what the model answered, these say how often the answer was acted on and over how
    // many routes. A verdict that is answered but never acted on is the failure the shadow rule
    // allows and this pair is what makes it visible.
    std::uint64_t leg_acted_layers = 0;
    std::uint64_t leg_acted_routes = 0;
    int    last_leg_misses     = 0;
    double last_leg_threshold  = -1.0;
    double last_leg_copy_us    = -1.0;
    double last_leg_cpu_us     = -1.0;

    void note_kv(CpuKvSplitLeg leg, double copy_ms, double cpu_ms) noexcept {
        last_kv_copy_ms = copy_ms;
        last_kv_cpu_ms  = cpu_ms;
        switch (leg) {
        case CpuKvSplitLeg::HostPartial: ++kv_answered; ++kv_host_partial; return;
        case CpuKvSplitLeg::DeviceCopy: ++kv_answered; ++kv_device_copy; return;
        case CpuKvSplitLeg::NoData: ++kv_nodata; return;
        }
    }

    void note_expert(CpuExpertLeg leg, double copy_us, double cpu_us) noexcept {
        last_expert_copy_us = copy_us;
        last_expert_cpu_us  = cpu_us;
        switch (leg) {
        case CpuExpertLeg::CpuHalf: ++expert_answered; ++expert_cpu_half; return;
        case CpuExpertLeg::PoolCopy: ++expert_answered; ++expert_pool_copy; return;
        case CpuExpertLeg::NoData: ++expert_nodata; return;
        }
    }

    // One closed layer: what the ratio cost and what it bought. The two halves are counted where
    // each exists - routes at the decision (ensure_split), experts and ms at the CPU half - which is
    // why there are two notes instead of one.
    void note_layer_routes(int routes) noexcept {
        if (routes > 0) { routes_seen += static_cast<std::uint64_t>(routes); }
    }
    void note_layer_cpu(int experts, int tokens, double span_ms, double compute_ms) noexcept {
        layer_cpu = static_cast<std::uint64_t>(experts < 0 ? 0 : experts);
        cpu_routes += layer_cpu;
        last_layer_cpu_ms = span_ms;
        if (layer_cpu != 0) {
            note_cpu_half(experts, tokens, span_ms, compute_ms, last_flops_per_expert);
        }
    }
    // Decision 2's layer form. Counted beside the per-expert counters rather than replacing them:
    // the per-expert line says what one route compares like, this one says what the layer costs once
    // the read-back is in it, and the two disagreeing is the finding, not a bug.
    void note_layer_leg(const CpuLayerLegVerdict& verdict, int misses) noexcept {
        last_leg_misses    = misses;
        last_leg_threshold = verdict.threshold;
        last_leg_copy_us   = verdict.copy_us;
        last_leg_cpu_us    = verdict.cpu_us;
        switch (verdict.leg) {
        case CpuExpertLeg::CpuHalf: ++leg_answered; ++leg_cpu_half; return;
        case CpuExpertLeg::PoolCopy: ++leg_answered; ++leg_pool_copy; return;
        case CpuExpertLeg::NoData: ++leg_nodata; return;
        }
    }
    // The layer leg was not only answered but applied: these routes left the upload list and the
    // placement word. `routes` is the count that moved, not the count that was asked about.
    void note_layer_leg_acted(int routes) noexcept {
        if (routes <= 0) { return; }
        ++leg_acted_layers;
        leg_acted_routes += static_cast<std::uint64_t>(routes);
    }
    // The cost model's own two sides for the decision in force, set where it was taken.
    void note_budget(double gpu_us, double cpu_us, double budget_us) noexcept {
        last_cost_gpu_us = gpu_us;
        last_cost_cpu_us = cpu_us;
        last_budget_us   = budget_us;
    }
    double last_flops_per_expert = 0.0;

    // Printed by PolicyController::report() with the rest of the panel (same cadence, same
    // NINFER_SPLIT_TIMING gate). Read-only: with no actuator wired this line IS the whole
    // mechanism's state today.
    void report() const {
        const char* policy = cpu_path_switch() == CpuPathSwitch::On ? "on" : "off";
        std::fprintf(stderr,
                     "[ctl] cpu: policy=%s | kv: cpu-partial=%llu/%llu (copy=%llu nodata=%llu",
                     policy, static_cast<unsigned long long>(kv_host_partial),
                     static_cast<unsigned long long>(kv_answered),
                     static_cast<unsigned long long>(kv_device_copy),
                     static_cast<unsigned long long>(kv_nodata));
        if (last_kv_copy_ms >= 0.0) {
            std::fprintf(stderr, "; last est copy=%.3f ms vs cpu %.3f ms)", last_kv_copy_ms,
                         last_kv_cpu_ms);
        } else {
            std::fprintf(stderr, "; no est)");
        }
        std::fprintf(stderr, " | expert (superseded by the leg line): cpu-half=%llu/%llu (pool-copy=%llu nodata=%llu",
                     static_cast<unsigned long long>(expert_cpu_half),
                     static_cast<unsigned long long>(expert_answered),
                     static_cast<unsigned long long>(expert_pool_copy),
                     static_cast<unsigned long long>(expert_nodata));
        if (last_expert_copy_us >= 0.0) {
            std::fprintf(stderr, "; last copy=%.0f us/exp vs cpu %.1f us/exp)\n",
                         last_expert_copy_us, last_expert_cpu_us);
        } else {
            std::fprintf(stderr, "; no est)\n");
        }
        // Decision 3's share: the routes the CPU half took over the routes the layers had, with the
        // rate the model priced them at. Printed even at zero, because "the actuator never fired"
        // and "the actuator fired and did nothing" are different statements.
        std::fprintf(stderr,
                     "[ctl] cpu share: routes cpu=%llu/%llu (%.1f%%) last=%llu experts | "
                     "rate span=%.1fus@%d compute=%.1fus@%d per exp (%.0f GFLOPS) | est copy=%.0f us vs "
                     "cpu=%.0f us, budget=%.0f us\n",
                     static_cast<unsigned long long>(cpu_routes),
                     static_cast<unsigned long long>(routes_seen),
                     routes_seen == 0 ? 0.0
                                      : 100.0 * static_cast<double>(cpu_routes) /
                                            static_cast<double>(routes_seen),
                     static_cast<unsigned long long>(layer_cpu),
                     cpu_half_rate().span_us_per_expert < 0.0
                         ? 0.0
                         : cpu_half_rate().span_us_per_expert,
                     cpu_half_rate().span_count,
                     cpu_half_rate().compute_us_per_expert < 0.0
                         ? 0.0
                         : cpu_half_rate().compute_us_per_expert,
                     cpu_half_rate().compute_count,
                     cpu_half_rate().cpu_gflops, last_cost_gpu_us, last_cost_cpu_us,
                     last_budget_us);
        // Decision 2's layer form, printed next to the per-expert line on purpose: the two disagree
        // whenever the read-back matters, and that disagreement is the whole reason the layer form
        // exists. `m*` is the miss count at which the half starts to pay at this span; -1 means the
        // span is under the transfer and there is no crossing at any count.
        std::fprintf(stderr,
                     "[ctl] cpu leg: layer-level=%llu/%llu (half=%llu copy=%llu nodata=%llu) | "
                     "acted=%llu layer(s)/%llu route(s) | "
                     "last m=%d m*=%.1f t=%d | copy=%.0f us half=%.0f us (readback %.0f us)\n",
                     static_cast<unsigned long long>(leg_cpu_half),
                     static_cast<unsigned long long>(leg_answered),
                     static_cast<unsigned long long>(leg_cpu_half),
                     static_cast<unsigned long long>(leg_pool_copy),
                     static_cast<unsigned long long>(leg_nodata),
                     static_cast<unsigned long long>(leg_acted_layers),
                     static_cast<unsigned long long>(leg_acted_routes), last_leg_misses,
                     last_leg_threshold, cpu_half_rate().measured_tokens, last_leg_copy_us,
                     last_leg_cpu_us, kCpuReadbackUs);
        std::fflush(stderr);
    }
};

inline CpuPathLedger& cpu_path_ledger() {
    static thread_local CpuPathLedger ledger;
    return ledger;
}

} // namespace ninfer::models::qwen3_5::execution
