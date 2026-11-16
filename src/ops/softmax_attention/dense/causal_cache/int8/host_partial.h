#pragma once

// port (sm_89), engine-integration P3 Phase C: the KV-partition CPU-partial seam (default off).
//
// Scope of this header: export ONE small interface that lets a caller run a single split's
// attention partial on the CPU over HOST-RESIDENT rk4v4 / rk8v4 KV pages and land the result in
// the device partial buffers the production merge kernel consumes, plus the knob-gated branch
// int8/launch.cu installs between the grouped MMA launch and the merge launch.
//
// The seam is deliberately narrow:
//   * The host view (page planes, block table, Q, positions mirrors) is SUPPLIED BY THE CALLER.
//     Nothing in this tree supplies it yet - the later KV-tiering batch owns its construction,
//     exactly as ops/softmax_attention/cpu/partial.h and cost.h name it. Production signatures
//     are unchanged (the exported attention entries take no extra parameter); a thread-local
//     installed view is the signature-stable supply channel.
//   * The knob is NINFER_KV_CPU_PARTITION=<split index>. Unset / negative / unparseable means
//     OFF (the repo's forced-switch rule: a typo may disable a default-off path, never arm one).
//     When armed, the named split's partial comes from the CPU; every precondition miss falls
//     back to the GPU with ONE log line per reason per process. Unarmed calls pay one integer
//     compare and return - the production hot path is untouched.
//   * The GPU kernel still computes the slot (one grid.y block); the CPU result overwrites it
//     via an async H2D enqueued after the MMA kernel and before the merge launch on the same
//     stream, so stream order alone guarantees the merge reads the CPU values. Discarding the
//     GPU slot is the validation-seam shape: the later tiering batch that actually owns the
//     split (host pages, no device copy) replaces the whole leg.
//
// Preconditions the gate enforces (each miss = GPU fallback + one log line): view installed;
// unmasked single-batch call (batch==1, no valid_columns); view geometry matches the dispatched
// G; view dtype matches the dispatched codec (PackedK); window from the host positions mirror
// fits the call's visible capacity; knob split < active splits; the stream is not capturing.
// The first engage also logs one line; the thread-local engage counter lets a harness assert
// the seam actually fired instead of silently falling back to two identical GPU runs.
//
// Numerical contract: identical to ops/softmax_attention/cpu/partial.h (the Phase B substrate
// this calls) - the CPU partial reproduces the grouped MMA instance's arithmetic to within the
// verified deviation classes (expf-vs-ex2.approx residual ~3e-7 relative; single FP16-quantum
// flips of one key's contribution), which is what the P3 harness's 1e-5 bar audits.

#include "core/device.h"
#include "ops/softmax_attention/common/causal_operands.h"
#include "ops/softmax_attention/common/causal_partition.h"
#include "ops/softmax_attention/cpu/partial.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace ninfer::ops::detail {

// --- the caller-supplied host view -----------------------------------------------------------

// K codec identity of the mirrored planes. Rk4v4 = packed int4 K (the PackedK=true grouped
// instance), Rk8v4 = plain int8 K (PackedK=false). V is int4-packed with FP16 G64 scales in
// both, which is why partial.h covers exactly these two dtypes.
enum class RkHostKvDType : std::uint8_t { Rk4v4 = 0, Rk8v4 = 1 };

// One layer's host-resident mirror the later KV-tiering batch will own: the page code/scale
// planes and the row's block table (physical ids, negative = absent page), plus THIS call's
// host mirrors of Q (BF16 bits, causal_q_index order: d + 256*(head + QHeads*token)) and
// positions. The mirror must track the call it is asked about: the caller updates Q/positions
// between calls and guarantees the mirror matches the device operands of the next attention
// call on this thread. `pages.kv_heads` must equal the dispatched geometry's KVHeads.
struct RkHostPartialView {
    RkHostKvDType dtype = RkHostKvDType::Rk4v4;
    int query_heads       = 0;  // must equal the dispatched geometry's QHeads
    softmax_cpu::RkPartialKvPages pages;
    const std::uint16_t* q_bf16   = nullptr;
    const std::int32_t* positions = nullptr;
    // APPEND FRONTIER. The production decode call fuses the current token's append into the same
    // kernel that scores it, so a producer that mirrors at the call boundary cannot hold the newest
    // token's codes. `page_filled_keys` states how many leading positions the mirrored pages are
    // authoritative for; the gate refuses a split whose range reaches past it. 0 means "no bound" -
    // a producer whose pages are authoritative everywhere (a real Host tier that owns them).
    int page_filled_keys = 0;
};

namespace rk_host_partial_detail {

// The installed view lives in one thread-local slot so install/active/engage all agree.
[[nodiscard]] inline const RkHostPartialView*& view_slot() {
    static thread_local const RkHostPartialView* installed = nullptr;
    return installed;
}

// The thread-local engage counter (assertable by a harness; see rk_host_partial_engagements).
[[nodiscard]] inline std::uint64_t& engage_slot() {
    static thread_local std::uint64_t count = 0;
    return count;
}

} // namespace rk_host_partial_detail

// Install / clear the view for THIS thread (the engine worker that runs attention). Passing
// nullptr clears. The pointer must outlive every call that consults it.
inline void rk_host_partial_install(const RkHostPartialView* view) {
    rk_host_partial_detail::view_slot() = view;
}

// The view the gate would consult right now (nullptr when none is installed).
[[nodiscard]] inline const RkHostPartialView* rk_host_partial_active() {
    return rk_host_partial_detail::view_slot();
}

// Engagements of the CPU leg on this thread: a harness asserts this instead of trusting
// silence (a silently disarmed seam would make an A/B compare two identical GPU runs).
[[nodiscard]] inline std::uint64_t rk_host_partial_engagements() {
    return rk_host_partial_detail::engage_slot();
}

// --- the knob --------------------------------------------------------------------------------

// NINFER_KV_CPU_PARTITION=<split index>, read once per process; -1 = off. Only ">= 0" parses;
// per-call bound checks (split < active) decide the rest, so a huge value logs one fallback
// line instead of arming anything.
[[nodiscard]] inline int rk_kv_cpu_partition_split() {
    static const int split = [] {
        const char* env = std::getenv("NINFER_KV_CPU_PARTITION");
        if (env == nullptr) { return -1; }
        const long parsed = std::strtol(env, nullptr, 10);
        return parsed >= 0 ? static_cast<int>(parsed) : -1;
    }();
    return split;
}

namespace rk_host_partial_detail {

// One line per fallback reason per process: the knob is a diagnostic switch, and a decode loop
// must not spray 512 copies of the same line.
enum Fallback : unsigned {
    kNoView = 0,
    kMasked,
    kMultiBatch,
    kGeometry,
    kDType,
    kCapacity,
    kSplitRange,
    kCapture,
    // A view that names no page planes or no block table would fault inside the operator instead
    // of falling back, so it is refused here like every other precondition.
    kNoPages,
    // The call's token tile is wider than the pinned staging this seam was built for.
    kWidth,
    // The producer's pages are authoritative only below `page_filled_keys`, and this split's key
    // range reaches past that (the fused append writes the tokens being scored).
    kAppendLag,
    kFallbackCount
};

inline void log_fallback(Fallback reason, const char* what) {
    static std::atomic<unsigned> logged{0};
    const unsigned bit = 1u << reason;
    if ((logged.fetch_or(bit) & bit) != 0u) { return; }
    std::fprintf(stderr, "[kv-cpu] NINFER_KV_CPU_PARTITION=%d: %s -> split stays on the GPU\n",
                 rk_kv_cpu_partition_split(), what);
    std::fflush(stderr);
}

// Production grouped-family staging bound: 24 query heads x the grouped token tile 8
// (Int8KvCausalPlan::kTokenTile). Both production geometries fit; a wider grouped call would
// need a larger staging block and is refused loudly by the executor below.
inline constexpr int kStageQHeads = 24;
inline constexpr int kStageWidth  = 8;

// Pinned staging for one split slot in compact (split-0) layout, plus the event that fences the
// previous call's in-flight H2D before the host overwrites the staging again. Deliberately
// never freed: cudaFreeHost at thread-exit races CUDA runtime teardown on engine workers, and
// the ~200 KiB is reclaimed with the process. Allocated on first engage only.
struct Staging {
    float* acc     = nullptr;  // [256 * kStageQHeads * kStageWidth]
    float* maximum = nullptr;  // [kStageQHeads * kStageWidth]
    float* sum     = nullptr;  // [kStageQHeads * kStageWidth]
    cudaEvent_t copied = nullptr;
    bool copied_valid  = false;
};

inline Staging& staging() {
    static thread_local Staging value = [] {
        constexpr std::size_t acc_floats =
            static_cast<std::size_t>(256) * kStageQHeads * kStageWidth;
        constexpr std::size_t stat_floats = static_cast<std::size_t>(kStageQHeads) * kStageWidth;
        constexpr std::size_t bytes       = (acc_floats + 2 * stat_floats) * sizeof(float);
        Staging s;
        void* host = nullptr;
        CUDA_CHECK(cudaHostAlloc(&host, bytes, cudaHostAllocDefault));
        s.acc     = static_cast<float*>(host);
        s.maximum = s.acc + acc_floats;
        s.sum     = s.maximum + stat_floats;
        CUDA_CHECK(cudaEventCreateWithFlags(&s.copied, cudaEventDisableTiming));
        return s;
    }();
    return value;
}

// NINFER_KV_CPU_PARTIAL_CHECK=1: report the CPU partial's deviation from the GPU partial the grouped
// MMA just wrote for the SAME split over the SAME pages. This is the one moment both values exist:
// the CPU leg overwrites the slot, so without this the only observable is the final output. The
// comparison is the operator's own contract audited directly (acc / maximum / sum, per element),
// measured rather than inferred from a PPL or a token match.
inline bool partial_check_enabled() {
    static const bool on = [] {
        const char* value = std::getenv("NINFER_KV_CPU_PARTIAL_CHECK");
        return value != nullptr && *value != '\0' && std::strtol(value, nullptr, 10) != 0;
    }();
    return on;
}

template <int QueryHeads>
inline void compare_with_device(int width, int split, softmax_cpu::RkPartialTokenRange range,
                                CausalPartialView partial, const Staging& st,
                                cudaStream_t stream) {
    static bool reported = false;
    if (reported) { return; }
    reported = true;
    const std::size_t acc_n  = static_cast<std::size_t>(256) * QueryHeads * width;
    const std::size_t stat_n = static_cast<std::size_t>(QueryHeads) * width;
    const std::int64_t slot_acc  = static_cast<std::int64_t>(acc_n) * split;
    const std::int64_t slot_stat = static_cast<std::int64_t>(stat_n) * split;
    std::vector<float> gpu_acc(acc_n), gpu_max(stat_n), gpu_sum(stat_n);
    if (cudaMemcpyAsync(gpu_acc.data(), partial.acc + slot_acc, acc_n * sizeof(float),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaMemcpyAsync(gpu_max.data(), partial.maximum + slot_stat, stat_n * sizeof(float),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaMemcpyAsync(gpu_sum.data(), partial.sum + slot_stat, stat_n * sizeof(float),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
        (void)cudaGetLastError();
        std::fprintf(stderr, "[kv-cpu] partial check: the GPU slot could not be read back\n");
        std::fflush(stderr);
        return;
    }
    double acc_max_abs = 0.0, acc_max_rel = 0.0, acc_magnitude = 0.0;
    std::size_t over = 0;
    for (std::size_t i = 0; i < acc_n; ++i) {
        const double a = st.acc[i];
        const double b = gpu_acc[i];
        const double d = std::fabs(a - b);
        const double scale = std::max(std::fabs(a), std::fabs(b));
        acc_magnitude = std::max(acc_magnitude, scale);
        acc_max_abs   = std::max(acc_max_abs, d);
        if (scale > 0.0) { acc_max_rel = std::max(acc_max_rel, d / scale); }
        if (scale > 0.0 && d / scale > 1e-5) { ++over; }
    }
    // The operator's own bar (partial.h): 1e-5 relative, OR inside one FP16 quantum of the
    // accumulator's magnitude (2^-11), because acc is built from FP16-rounded p and V.
    const double quantum = std::ldexp(1.0, -11) * acc_magnitude;
    std::size_t outside = 0;
    for (std::size_t i = 0; i < acc_n; ++i) {
        const double a = st.acc[i];
        const double b = gpu_acc[i];
        const double d = std::fabs(a - b);
        const double scale = std::max(std::fabs(a), std::fabs(b));
        if (d > quantum && (scale == 0.0 || d / scale > 1e-5)) { ++outside; }
    }
    double stat_max_abs[2] = {0.0, 0.0};
    for (std::size_t i = 0; i < stat_n; ++i) {
        stat_max_abs[0] = std::max(stat_max_abs[0], std::fabs(static_cast<double>(st.maximum[i]) - gpu_max[i]));
        stat_max_abs[1] = std::max(stat_max_abs[1], std::fabs(static_cast<double>(st.sum[i]) - gpu_sum[i]));
    }
    std::fprintf(stderr,
                 "[kv-cpu] partial check split %d: range=[%d,%d) width=%d | "
                 "acc max|d|=%.3e max|acc|=%.3e (quantum %.3e) rel=%.3e | "
                 "over 1e-5: %zu/%zu | outside the quantum bar: %zu | "
                 "maximum max|d|=%.3e | sum max|d|=%.3e\n",
                 split, range.begin, range.end, width, acc_max_abs, acc_magnitude, quantum,
                 acc_max_rel, over, acc_n, outside, stat_max_abs[0], stat_max_abs[1]);
    std::fflush(stderr);
}

inline void log_engage(const RkHostPartialView& view, int split, int window, int active,
                       softmax_cpu::RkPartialTokenRange range, int key_tile, int width) {
    static std::atomic<int> logged{0};
    if (logged.exchange(1) != 0) { return; }
    std::fprintf(stderr,
                 "[kv-cpu] split %d -> CPU partial (dtype=%s q_heads=%d kv_heads=%d width=%d "
                 "window=%d active=%d range=[%d,%d) key_tile=%d); the GPU slot is overwritten "
                 "before the merge\n",
                 split, view.dtype == RkHostKvDType::Rk4v4 ? "rk4v4" : "rk8v4", view.query_heads,
                 view.pages.kv_heads, width, window, active, range.begin, range.end, key_tile);
    std::fflush(stderr);
}

} // namespace rk_host_partial_detail

// --- the exported executor ----------------------------------------------------------------------

// One split's CPU partial over the caller's host pages, landed in the caller's device partial
// slot through pinned staging + an async H2D on `stream` (enqueue order alone puts it before
// whatever the caller launches next, i.e. the merge). `partial` is the batch-0 base view of
// the production partial storage; `split` selects the destination slot.
//
// The CPU kernel writes the compact staging as virtual slot 0; the H2D copies then translate
// staging -> slot `split`, which is a pure base-pointer shift because the slot layout
// (d + 256*(head + QHeads*(token + width*split))) is translation-invariant in `split`.
// `width`/`tokens` follow partial.h's unmasked-decode contract (tokens == width here);
// `range`/`key_tile` must mirror the dispatched instance (rk_partial_split_range and
// Schedule::kKeyRows).
template <int QueryHeads, int KVHeads, bool PackedK>
void rk_host_partial_stage(const RkHostPartialView& view, float scale, int width, int split,
                           softmax_cpu::RkPartialTokenRange range, int key_tile,
                           CausalPartialView partial, cudaStream_t stream) {
    static_assert(QueryHeads > 0 && KVHeads > 0 && QueryHeads % KVHeads == 0);
    if (split < 0 || width < 1 || key_tile < 1) {
        throw std::invalid_argument("host partial: invalid split/width/key tile");
    }
    if (QueryHeads > rk_host_partial_detail::kStageQHeads ||
        width > rk_host_partial_detail::kStageWidth) {
        throw std::logic_error("host partial: geometry exceeds the pinned staging bound");
    }
    auto& st = rk_host_partial_detail::staging();
    // Never overwrite staging a previous call's in-flight H2D may still be reading.
    if (st.copied_valid) { CUDA_CHECK(cudaEventSynchronize(st.copied)); }

    softmax_cpu::RkPartialRequest request;
    request.q_bf16    = view.q_bf16;
    request.positions = view.positions;
    request.scale     = scale;
    request.width    = width;
    request.tokens   = width;  // the unmasked single-batch decode contract
    request.split    = 0;      // staging holds one compact slot
    request.range    = range;
    request.key_tile = key_tile;
    const softmax_cpu::RkPartialOutput out{st.acc, st.maximum, st.sum};
    // P4 codec axis: the partial's bool template form was removed (an enum spelling silently
    // bound to the bool overload), so the seam spells the codec explicitly per PackedK.
    if constexpr (PackedK) {
        softmax_cpu::rk_attention_partial<QueryHeads, KVHeads, softmax_cpu::RkKCodec::Int4>(
            view.pages, request, out);
    } else {
        softmax_cpu::rk_attention_partial<QueryHeads, KVHeads, softmax_cpu::RkKCodec::Int8>(
            view.pages, request, out);
    }

    const auto slot_acc  = static_cast<std::int64_t>(256) * QueryHeads * width * split;
    const auto slot_stat = static_cast<std::int64_t>(QueryHeads) * width * split;
    // Before the overwrite: the GPU's own value for this split is still in the slot.
    if (rk_host_partial_detail::partial_check_enabled()) {
        rk_host_partial_detail::compare_with_device<QueryHeads>(width, split, range, partial, st,
                                                               stream);
    }
    const std::size_t acc_bytes =
        static_cast<std::size_t>(256) * QueryHeads * width * sizeof(float);
    const std::size_t stat_bytes = static_cast<std::size_t>(QueryHeads) * width * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(partial.acc + slot_acc, st.acc, acc_bytes, cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(partial.maximum + slot_stat, st.maximum, stat_bytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(partial.sum + slot_stat, st.sum, stat_bytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaEventRecord(st.copied, stream));
    st.copied_valid = true;
}

// --- the launch-side gate ------------------------------------------------------------------------

// Called by int8/launch.cu between the grouped MMA launch and the merge launch. Default (knob
// unset) returns after one compare. Armed: resolves the split's key range from the host
// positions mirror, revalidates every precondition against the dispatched instance, and hands
// the CPU leg its inputs. The discarded-GPU-slot and page-score-counter caveats are inherent
// to the validation-seam shape (the counters still record the GPU block's pass over the split).
template <class G, bool Masked, bool PackedK>
void rk_kv_cpu_split_gate(const CausalAttentionOperands& p, CausalKvPartition partition,
                          CausalPartialView partial, int key_tile, cudaStream_t stream) {
    const int split = rk_kv_cpu_partition_split();
    if (split < 0) { return; }  // default off: the whole seam is one compare on this path
    namespace detail = rk_host_partial_detail;
    using detail::log_fallback;

    const RkHostPartialView* view = rk_host_partial_active();
    if (view == nullptr) {
        log_fallback(detail::kNoView, "no host view installed");
        return;
    }
    // Before anything dereferences the view: a view that names no planes or no block table would
    // fault inside the operator (partial.h's own operand check is an exception, but on this path
    // that is a crash inside a CUDA launch sequence, not a fallback).
    if (view->pages.k_codes == nullptr || view->pages.v_codes == nullptr ||
        view->pages.k_scales == nullptr || view->pages.v_scales == nullptr ||
        view->pages.block_table == nullptr || view->pages.kv_heads <= 0 ||
        view->query_heads <= 0) {
        log_fallback(detail::kNoPages, "the host view names no page planes or block table");
        return;
    }
    if constexpr (Masked) {
        log_fallback(detail::kMasked, "masked call (valid_columns) is out of the seam's v1 scope");
        return;
    }
    if (p.batch != 1) {
        log_fallback(detail::kMultiBatch, "batch > 1 is out of the seam's v1 scope");
        return;
    }
    if (p.width > detail::kStageWidth) {
        log_fallback(detail::kWidth, "the token tile is wider than the pinned staging");
        return;
    }
    if (view->query_heads != G::QHeads || view->pages.kv_heads != G::KVHeads) {
        log_fallback(detail::kGeometry, "host view geometry does not match the dispatched G");
        return;
    }
    if ((view->dtype == RkHostKvDType::Rk4v4) != PackedK) {
        log_fallback(detail::kDType, "host view dtype does not match the dispatched codec");
        return;
    }
    if (view->positions == nullptr || view->q_bf16 == nullptr) {
        log_fallback(detail::kNoView, "host view has no Q/positions mirror");
        return;
    }
    const int width  = p.width;
    const int window = static_cast<int>(view->positions[width - 1]) + 1;
    if (window < 1 || window > p.visible_capacity) {
        log_fallback(detail::kCapacity,
                     "host-mirror window exceeds the call's visible capacity");
        return;
    }
    const int active = partition.active(window);
    if (split >= active) {
        log_fallback(detail::kSplitRange, "knob split is out of range for this window");
        return;
    }
    // Capture guard: an enqueued-from-host H2D with fixed staging would freeze stale bytes into
    // the graph, and event waits are illegal under capture - captured graphs always take the
    // GPU leg (replays re-execute the recorded GPU ops and never re-run host gates).
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    const cudaError_t probe         = cudaStreamIsCapturing(stream, &capture);
    if (probe != cudaSuccess) {
        (void)cudaGetLastError();  // the probe's own error must not poison the merge's check
    }
    if (probe != cudaSuccess || capture != cudaStreamCaptureStatusNone) {
        log_fallback(detail::kCapture, "stream is capturing");
        return;
    }

    const auto range = softmax_cpu::rk_partial_split_range(window, split, active, key_tile);
    // The producer states how far its pages are authoritative (see page_filled_keys): a split whose
    // range reaches into the tokens this call is still appending would be scored against stale
    // bytes, which looks like a numeric bug rather than the logical one it is.
    if (view->page_filled_keys > 0 && range.end > view->page_filled_keys) {
        log_fallback(detail::kAppendLag,
                     "the split's range reaches past the host pages' append frontier");
        return;
    }
    detail::log_engage(*view, split, window, active, range, key_tile, width);
    ++detail::engage_slot();
    rk_host_partial_stage<G::QHeads, G::KVHeads, PackedK>(*view, p.scale, width, split, range,
                                                          key_tile, partial, stream);
}

} // namespace ninfer::ops::detail
