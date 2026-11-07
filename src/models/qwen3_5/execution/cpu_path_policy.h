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
// the runtime legs a verdict names (ExpertStream::ensure leaving an expert for the CPU half;
// the KV tiering controller dispatching a split as a host partial) are later batches - verdicts
// are recorded and printed, not acted on, which is the framework's own shadow rule ("print what
// it would do and why, and change nothing").
//
// The three deployment modes (v8 12.4) are untouched: these hooks neither read nor move the
// mode, and a verdict costs the same in auto, stable and performance.

#include "models/qwen3_5/execution/param_bus.h"

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
// The CPU half's per-expert compute, measured on this box at Columns=1 (8d07afe3, 2026-11-07):
// int8 tier 0.0484 ms (NINFER_MOE_CPU_INT8 default), int16 tier 0.0504 ms (the =16 gate, A/B
// 96.0%/94.1% of int8 - the default tier stays int8).
inline constexpr double kCpuMoeExpertInt8Ms  = 0.0484;
inline constexpr double kCpuMoeExpertInt16Ms = 0.0504;

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
// The call site is the KV tiering controller's split dispatch (the later batch cost.h names):
// a split whose pages are host-resident either gets its pages copied in - the GPU grouped
// instance then reads them like any device page - or runs as a CPU partial over the host pages,
// handing its (acc, maximum, sum) buffers back for the GPU merge (partial.h's contract). No
// caller exists in the tree yet, so this seam's counters stay at zero until it lands; the
// function, its registered rates and the observation below are what that batch will call.
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
// actually pays, already observed and printed by the controller); the compute side is the
// calibrated CPU half rate. The runtime leg a cpu-half verdict names - ensure() leaving the
// expert for host_routed_ffn's CPU half instead of uploading it - is the actuator a later batch
// wires; today ffn.cpp records the verdict at the decision point that already feeds the
// controller, and the [ctl] cpu line prints it.
enum class CpuExpertLeg : std::uint8_t {
    PoolCopy = 0, // upload the expert into the pool (the in-force behaviour)
    CpuHalf  = 1, // compute it in the CPU half, merge the term on the GPU
    NoData   = 2,
};

// Pure; `copy_us_per_expert` is the caller's measured upload. `est_cpu_us` (optional) carries
// the calibrated CPU-side rate so the observation line can print both sides.
[[nodiscard]] inline CpuExpertLeg decide_cpu_expert_half(double copy_us_per_expert,
                                                         double* est_cpu_us = nullptr) noexcept {
    const double cpu_ms = param_bus().get(kCpuMoeExpertRow);
    if (est_cpu_us != nullptr) { *est_cpu_us = cpu_ms > 0.0 ? cpu_ms * 1000.0 : -1.0; }
    if (!(copy_us_per_expert > 0.0) || !(cpu_ms > 0.0)) { return CpuExpertLeg::NoData; }
    if (cpu_path_switch() == CpuPathSwitch::Off) { return CpuExpertLeg::PoolCopy; }
    return copy_us_per_expert > cpu_ms * 1000.0 ? CpuExpertLeg::CpuHalf : CpuExpertLeg::PoolCopy;
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
        std::fprintf(stderr, " | expert: cpu-half=%llu/%llu (pool-copy=%llu nodata=%llu",
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
        std::fflush(stderr);
    }
};

inline CpuPathLedger& cpu_path_ledger() {
    static thread_local CpuPathLedger ledger;
    return ledger;
}

} // namespace ninfer::models::qwen3_5::execution
