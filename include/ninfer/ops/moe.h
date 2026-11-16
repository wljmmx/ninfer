#pragma once

#include "core/arena.h"
#include "core/cpu/executor.h"
#include "core/device.h"
#include "core/weight_bank.h"
// port (sm_89): master keeps the projection weight forms in weight_input.h.
#include "ninfer/ops/weight_input.h"

#include <optional>

namespace ninfer::ops {

// Every SwiGLU uses gate_up [2I,H] with gate rows first, then up rows; down is [H,I].
struct SwiGLUProfile {
    std::int32_t hidden = 0, intermediate = 0;
    QType gate_up_type = QType::BF16, down_type = QType::BF16;
    QuantLayout gate_up_layout = QuantLayout::Contiguous, down_layout = QuantLayout::Contiguous;
    LinearPolicy gate_up_policy = LinearPolicy::A16Only, down_policy = LinearPolicy::A16Only;
};

struct RouteProfile {
    std::int32_t hidden = 0, experts = 0, top_k = 0;
};

struct RoutedSwiGLUProfile {
    SwiGLUProfile ffn;
    std::int32_t experts = 0, top_k = 0;
};

// NVFP4 A4 calibration divisors for x and the FP32 SiLU(gate)*up product, respectively.
// Matrix divisors remain independently addressed by logical expert in the bank.
// Block-FP8 weights and their dynamic activation groups have no tensor divisors.
struct ExpertActivationDivisors {
    std::optional<float> gate_up, down;
};

// Combines activation permissions and calibration across Uses evaluated by one expert call.
// NVFP4 A4 requires the same positive divisor for every included input.
class ExpertInputUsage {
public:
    void include(const WeightInput&);

    [[nodiscard]] LinearPolicy policy() const noexcept { return policy_; }

    [[nodiscard]] std::optional<float> activation_divisor(QType format) const;

private:
    LinearPolicy policy_ = LinearPolicy::AllowA4;
    std::optional<float> divisor_;
    bool missing_divisor_ = false, unequal_divisors_ = false;
};

struct CudaSwiGLUExperts {
    ExpertBankDeviceView bank;
    ExpertActivationDivisors activation_divisors;
    // Qwen4Exp stores an expert's gate and up as two separate objects instead of one fused
    // [2I,H] parent, so the fused row map - "expert * 2I, then gate rows, then up rows" - cannot
    // describe them: row I of the gate object is the *next* expert's first gate row. These two
    // views therefore carry one image of `intermediate` rows per expert each, and `bank.gate_up`
    // must stay empty while they are set; a call that supplies both is refused rather than
    // resolved by precedence. Both are null on the fused layout.
    ExpertMatrixDeviceView gate, up;
};

struct CpuSwiGLUExperts {
    WeightBankView gate_up, down;
    // port (sm_89): the gate/up/down form on the host side. Qwen4Exp stores an expert's gate and up
    // as two objects, so the fused [2I,H] row map cannot describe them (row I of the gate object is
    // the next expert's first gate row). Each view names one image of `intermediate` rows per
    // expert; `gate_up` must stay default-constructed while they are set, and the CPU op refuses a
    // bank that names both forms.
    WeightBankView gate, up;
};

struct SwiGLUWeights {
    SingleProjectionWeight gate_up, down;
};

struct ScaledSwiGLUTerm {
    SwiGLUWeights weights;
    Tensor coefficients; // Contiguous FP32 [T].
};

struct ExpertRoutes {
    // port (sm_89): the tensor is declared [K,T], but its *storage* is token-major: assignment
    // (j,t) lives at index t*K + j, which is what topk_softmax_route writes
    // (`ids + token * top_k`) and what the routed kernels read (`assignment = token * K + j`).
    // Both halves agree; only the row/column labels of the declared shape are the transpose.
    Tensor ids;          // Contiguous I32 [K,T]; distinct logical IDs within each token.
    Tensor coefficients; // Contiguous FP32 [K,T]; consumed without renormalization.
};

enum class ExpertSubsetKind : std::uint8_t { All, None, MaskOnes, MaskZeros };

struct ExpertSubset {
    ExpertSubsetKind kind = ExpertSubsetKind::All;
    Tensor mask; // I32 [E], 0/1, on the execution device; unused by All and None.
};

enum class MoeEpilogue : std::uint8_t { Write, AddResidual };

struct MoeProfile {
    RoutedSwiGLUProfile routed;
    SwiGLUProfile shared;
};

struct MoeWeights {
    Weight control; // BF16 [E+1,H]: routed logits, then one independent shared logit.
    CudaSwiGLUExperts routed;
    SwiGLUWeights shared;
};

// Pure L2 read-ahead; the referenced CUDA storage outlives the submitted device work.
struct MoeExecutionHints {
    const void* read_ahead       = nullptr;
    std::size_t read_ahead_bytes = 0;
};

[[nodiscard]] SwiGLUProfile swiglu_profile(const SwiGLUWeights& weights);
[[nodiscard]] SwiGLUProfile swiglu_profile(LinearProfile gate_up, LinearProfile down);
[[nodiscard]] RoutedSwiGLUProfile
routed_swiglu_profile(const WeightBankLayout& gate_up, const WeightBankLayout& down,
                      std::int32_t top_k, LinearPolicy gate_up_policy, LinearPolicy down_policy);
[[nodiscard]] RoutedSwiGLUProfile
routed_swiglu_profile(const WeightBankView& gate_up, const WeightBankView& down, std::int32_t top_k,
                      LinearPolicy gate_up_policy, LinearPolicy down_policy);
// port (sm_89): the split form Qwen4Exp stores - gate and up in two banks of I rows each, down in
// a third of H rows. The routed profile carries one gate-side format and layout, so the two banks
// must agree on both; a pair that does not is refused instead of being reported as one of them.
[[nodiscard]] RoutedSwiGLUProfile routed_swiglu_profile(
    const WeightBankLayout& gate, const WeightBankLayout& up, const WeightBankLayout& down,
    std::int32_t top_k, LinearPolicy gate_up_policy, LinearPolicy down_policy);

// Execution admits registered geometries and format pairs; unsupported profiles are rejected.
// For each BF16 x[:,t], l=control*x; select K greatest l[0:E], descending, ties by lower ID.
// Publish their IDs and softmax coefficients, and sigmoid(l[E]) separately. Shapes are
// x [H,T], routes [K,T], shared_coefficients [T]; all contiguous, T>0. x/routes and encoded
// matrix planes are 16-byte aligned; shared coefficients are 4-byte aligned. Every output is
// written. Outputs and live scratch are mutually disjoint and overlap no input. Enqueues on
// execution without host synchronization; CUDA Graph capture/replay is supported.
[[nodiscard]] std::size_t topk_softmax_route_workspace_capacity_bytes(const RouteProfile& profile,
                                                                      std::int32_t min_tokens,
                                                                      std::int32_t max_tokens);
void topk_softmax_route(const Tensor& x, const RouteProfile& profile, const Weight& control,
                        ExpertRoutes& routes, Tensor& shared_coefficients, MemoryArena& workspace,
                        CudaExecutionView execution);

// F_e(x)=down_e*(SiLU(gate_e*x)*(up_e*x)). Sum coefficient[j,t]*F_ids[j,t](x[:,t]) over subset.
// The shared overload adds coefficients[t]*F_shared(x[:,t]) exactly once, including for None.
// x is contiguous BF16 [H,T], destination is contiguous FP32 or BF16 [H,T], T>0. Both,
// routes and encoded planes are 16-byte aligned; mask/shared coefficients are 4-byte aligned.
// Write reads no old destination; AddResidual requires BF16 and adds its old value before
// final rounding. None writes zero without accessing routed weights/routes (or computes shared).
// Route IDs are in [0,E); unselected weights and index entries are never read. Callers publish
// and retain every selected weight through execution. All views belong to the execution GPU.
// Destination and live scratch are mutually disjoint and overlap no input. Intermediate precision
// is private; there is no semantic staging cast. No hidden allocation, transfer or host readback
// is performed.
// Fixed-profile Graph replay may change route/mask/index contents after prior consumers finish.
//
// port (sm_89): `missed_routes`, when not null, receives one for every route the mask excluded, added
// on the device where the route is skipped. It is the only honest answer to "did the resident set
// cover this layer?" for a caller that deliberately never read the ids back: the mask is published
// from the same decision that places the routes, so asking the mask always says it covered them.
// The address must be mapped host memory or Device memory that outlives the enqueued work, and the
// count is not reset here. Only the groupwise-int paths count; a block-scaled layer rejects a
// non-null pointer rather than report a zero it did not measure.
[[nodiscard]] std::size_t
routed_swiglu_workspace_capacity_bytes(const RoutedSwiGLUProfile& profile, ExpertSubsetKind subset,
                                       DType output_type, MoeEpilogue epilogue,
                                       std::int32_t min_tokens, std::int32_t max_tokens);
[[nodiscard]] std::size_t routed_swiglu_workspace_capacity_bytes(
    const RoutedSwiGLUProfile& profile, const SwiGLUProfile& shared, ExpertSubsetKind subset,
    DType output_type, MoeEpilogue epilogue, std::int32_t min_tokens, std::int32_t max_tokens);
void routed_swiglu(const Tensor& x, const RoutedSwiGLUProfile& profile,
                   const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                   ExpertSubset subset, MoeEpilogue epilogue, Tensor& destination,
                   MemoryArena& workspace, CudaExecutionView execution,
                   std::uint32_t* missed_routes = nullptr);
void routed_swiglu(const Tensor& x, const RoutedSwiGLUProfile& profile,
                   const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                   ExpertSubset subset, const ScaledSwiGLUTerm& shared, MoeEpilogue epilogue,
                   Tensor& destination, MemoryArena& workspace, CudaExecutionView execution,
                   std::uint32_t* missed_routes = nullptr);

// CPU contribution of the same routed formula. x is BF16 [H,T], destination is FP32 [H,T];
// all tensors are contiguous CPU views with element alignment. Encoded bank planes have
// 16-byte alignment. Route coefficients are finite; masks contain only 0/1. None writes zero
// without accessing routes or banks. Other subsets read only their selected expert payloads.
// Destination is overwritten and, together with live scratch, overlaps no input or each other.
// Intermediate precision is private; no activation quantization or semantic staging cast is
// introduced. The caller retains banks, tensors and scratch until synchronous return. No hidden
// allocation, transfer or thread creation occurs. Shared experts and residual addition are
// separate computations. Requires an AVX2/FMA/F16C worker team for nonempty contributions.
// NVFP4 consumes the native swizzled scales and each expert's matrix divisor directly.
// Capacity covers the inclusive token interval from a 256-byte-aligned local cursor; it is
// independent of worker count and excludes destination storage. None requires no scratch.
[[nodiscard]] std::size_t routed_swiglu_workspace_capacity_bytes(const RoutedSwiGLUProfile& profile,
                                                                 ExpertSubsetKind subset,
                                                                 std::int32_t min_tokens,
                                                                 std::int32_t max_tokens);
// port (sm_89): the CPU half of the gate/up/down form. Same contract as the fused query above, but
// the scratch is the activation plane of that form ([intermediate, tokens] FP32), because the three
// matrices are addressed per expert instead of through one [2I,H] parent.
[[nodiscard]] std::size_t
split_routed_swiglu_workspace_capacity_bytes(const RoutedSwiGLUProfile& profile,
                                            ExpertSubsetKind subset, std::int32_t min_tokens,
                                            std::int32_t max_tokens);
// `missed_routes` is accepted for signature parity and ignored: the CPU half of a split layer runs
// exactly the routes the mask kept, so it never skips one.
void routed_swiglu(const Tensor& x, const RoutedSwiGLUProfile& profile,
                   const CpuSwiGLUExperts& experts, const ExpertRoutes& routes, ExpertSubset subset,
                   Tensor& destination, MemoryArena& workspace, CpuExecutionView execution,
                   std::uint32_t* missed_routes = nullptr);

// Complete top-k-softmax routed SwiGLU plus sigmoid-gated shared SwiGLU, with the same routing,
// expert and residual formulas above. Destination is BF16 [H,T]. Every routed expert is readable.
// Only destination and caller scratch are mutated; they are mutually disjoint and overlap no
// input. Hints affect cache warming only. CUDA Graph capture/replay is supported.
// Capacity queries cover every positive T in their inclusive interval, from an aligned local
// workspace cursor, with no dependence on route values. Output storage is excluded from scratch.
[[nodiscard]] std::size_t
topk_softmax_swiglu_moe_workspace_capacity_bytes(const MoeProfile& profile, MoeEpilogue epilogue,
                                                 std::int32_t min_tokens, std::int32_t max_tokens);
void topk_softmax_swiglu_moe(const Tensor& x, const MoeProfile& profile, const MoeWeights& weights,
                             MoeEpilogue epilogue, Tensor& destination, MemoryArena& workspace,
                             CudaExecutionView execution, const MoeExecutionHints& hints = {});

// port (sm_89): O1's device-side coverage test. Of the K*T assignments in `ids`, how many does `mask`
// exclude? An assignment is covered when its id lies in [0, mask.ne[0]) and mask[id] == `member`
// (`member` is 1 for the resident set and 0 for its complement), which is the same rule a host scan
// of the same two arrays applies.
//
// It exists because the honest question - did the resident set cover this layer? - was only
// answerable by reading the ids back to the host. The route-skipping kernels can count the same
// thing while they run (`missed_routes`), but only after the placement has published a mask that
// includes this layer's own routes, which is the answer "yes" by construction. This asks the mask
// that is already published, so it answers the placement that the layer will actually run under.
//
// Only ids and mask are read - no expert payload, no quantised format - so unlike `missed_routes` it
// answers on every expert path. `excluded` receives the count as a store, not an addition: it must be
// mapped host memory or Device memory that outlives the enqueued work, and a caller that never reads
// the ids back can still tell a covered layer from an uncovered one at its next synchronise.
// Enqueues on `execution` without host synchronisation; CUDA Graph capture/replay is supported.
void route_coverage(const Tensor& ids, const Tensor& mask, std::int32_t member,
                    std::uint32_t* excluded, CudaExecutionView execution);

// port (sm_89): the split layer's host hand-off (kernels/cuda/host_handoff.cu).
//
// The four buffers a split layer reads back are packed into one block so the layer pays one
// submission instead of four, and the merge the host used to do is done on the device with the same
// accumulation order and rounding, so no read-back and no extra synchronise are needed.
std::size_t moe_pack_host_fetch_bytes(std::size_t x_bytes, std::size_t residual_bytes,
                                      std::size_t ids_bytes, std::size_t coefficient_bytes);
void moe_pack_host_fetch(const void* x, const void* residual, const void* ids,
                         const void* coefficients, int x_count, int ids_count, void* packed,
                         cudaStream_t stream);
void moe_merge_host_term(const void* host_term, const void* gpu_term, void* residual, int count,
                         cudaStream_t stream);

// port (sm_89): GH3 - the Device writes the expert tables (kernels/cuda/expert_table.cu).
//
// The publisher used to keep three parallel per-expert tables and merge them on the host: a pool
// offset, a host-tier offset, and a mask that meant "either one has an address". Every publish then
// pushed those rows to the Device. The mask in particular was *computed* on the host, which is the
// one thing a captured layer cannot absorb - and the host's per-row work grew with every structure
// added to the placement (the prediction region was the third).
//
// GH3 collapses the source of truth to one word per expert:
//
//   <slot>                        the pool holds this expert at `slot`
//   kExpertPlacementHost | <slot> the pinned tier holds it at `slot`
//   kExpertPlacementAbsent        neither, so the CPU half serves it
//
// and this kernel expands that word into the four tables the GEMM reads. The host's publish is now
// 4 bytes per expert that moved instead of 12, and `mask[e] = 1` - "this route resolves to an
// address" - is written where it is consumed.
inline constexpr std::uint32_t kExpertPlacementAbsent = 0xffffffffu;
inline constexpr std::uint32_t kExpertPlacementHost   = 0x80000000u;

// `count` is the logical expert count of one bank. The two slot strides are the byte distance between
// consecutive slots of the pool and of the pinned tier; they differ, which is why the tables are
// derived here rather than handed over finished. `pool_bias_bytes` is added after the multiply: a
// bank whose experts are wider than its neighbours owns a run that starts at its own byte base, so
// `slot * stride` alone would land outside it (zero when every bank shares one stride). Any of the
// four destinations may be null, and a table the caller does not keep is not written. Enqueues on
// `stream`; no host synchronisation.
void moe_expand_expert_placements(const std::uint32_t* placements, int count,
                                  std::uint64_t pool_slot_bytes, std::uint64_t pool_bias_bytes,
                                  std::uint64_t host_slot_bytes,
                                  std::uint64_t* pool_offsets, std::int32_t* pool_mask,
                                  std::uint64_t* host_offsets, std::int32_t* host_flags,
                                  cudaStream_t stream);

// port (sm_89): S2 - expert id -> pool slot, on the Device (kernels/cuda/route_slots.cu).
//
// `slot_table` is one bank's exported table (ExpertCache::export_slots) as it sits in the Device
// mirror; `experts` is that bank's entry count, i.e. the table's length. For each of the `count` ids
// this writes
//
//   slots[i] = slot_table[ids[i]]              when 0 <= ids[i] < experts and the slot is live
//   slots[i] = kAbsentSlot                     otherwise (id out of range, or the bank holds no slot)
//
// and adds one to `*misses` (atomically, Device memory) for every i that lands on the second line.
// kAbsentSlot is the pool's own marker (core/expert_cache.h) - the same word export_slots() writes -
// so a caller can compare against the table it staged without translating between two encodings.
// `misses` may be null, in which case nothing is counted. Enqueues on `stream`; no host
// synchronisation.
void moe_route_slots(const std::int32_t* ids, int count, const std::uint32_t* slot_table,
                     std::uint32_t experts, std::uint32_t* slots, std::uint32_t* misses,
                     cudaStream_t stream);

// C2: WHICH table an id->slot query is against. The two are different spaces and must never be read
// through one another: a placement slot is a pool slot (the address of an expert's bytes), while a
// region index is a position inside one bank's prediction ring (0..prefetch_slots_per_bank-1). The
// kernel takes the kind explicitly so a caller cannot pick the wrong one by passing a table it has.
enum class RouteTableKind : std::uint32_t {
    Placement        = 0, // table[expert] = pool slot   (export_slots)
    PrefetchRegion   = 1, // table[region] = expert      (export_prefetch_owners), searched backwards
};

// The region form of the same query (kernels/cuda/route_slots.cu). `table` here is the region's
// OWNER table - `table[region] = expert`, kAbsentSlot for a hole - and the result is the REGION INDEX
// that holds `id`, or kAbsentSlot when the region does not hold it. It is deliberately not the same
// arithmetic as the placement form: the region is indexed by slot, so the answer is a search over it.
//
// A mixed-up caller is refused rather than served a plausible number: `regions` (the region's length)
// and `experts` (the bank's width) are different numbers on every bank with a region, and the host-side
// caller checks the answer against prefetch_region_slot() before it is used as an address.
void moe_route_region_slots(const std::int32_t* ids, int count,
                            const std::uint32_t* region_owners, std::uint32_t regions,
                            std::uint32_t* region_indices, std::uint32_t* misses,
                            cudaStream_t stream);

// One entry point for a caller that already has the kind in hand: dispatches to the two above. Passing
// Placement with a region owner table (or the reverse) is caught downstream, not here - the kind is the
// caller's statement about the table it is holding.
void moe_route_slots(const std::int32_t* ids, int count, const std::uint32_t* table,
                     std::uint32_t entries, RouteTableKind kind, std::uint32_t* slots,
                     std::uint32_t* misses, cudaStream_t stream);

// port (sm_89): P1 - the prediction hand-off as a ring the Device writes and the host drains
// (kernels/cuda/pred_ring.cu).
//
// S3 hands a prediction over as one Device buffer that the next layer overwrites. That is enough for
// a single reader that always consumes what the last producer wrote, and it is not enough for a
// pipeline: the host has no way to tell "the entry I read is the one produced for the layer I am
// placing" from "the producer has already moved on", and it has no way to tell that it missed one.
// The ring makes both answerable with one monotonic counter and no host write into Device memory.
//
// One entry is a header plus `id_capacity` ids. `head` is a single Device word the producer bumps;
// the entry lands in slot `index % depth`, so a consumer that has fallen behind reads the newest
// entries rather than a mixture of two. `truncated` is written instead of being assumed away: a
// caller that hands over more ids than an entry holds gets the count it asked for recorded, so the
// consumer can refuse the entry instead of placing half a prediction.
//
// Nothing here synchronises. The producer runs on `stream` after the router that produced `ids`, and
// the consumer's read-back is ordered behind it by the same stream (see ExpertStream::
// pred_ring_begin_fetch, which issues its copy before the layer's own existing synchronise).
struct PredRingHeader {
    std::uint32_t source_bank = 0; // the layer that made the guess
    std::uint32_t target_bank = 0; // the layer that consumes it
    std::uint32_t layer       = 0; // source layer ordinal, for attribution
    std::uint32_t count       = 0; // ids written, <= capacity
    std::uint32_t capacity    = 0; // the entry's id capacity, echoed so a reader can refuse
    std::uint32_t truncated   = 0; // 1 when `count` exceeded `capacity` (no ids were written)
    std::uint64_t sequence    = 0; // the head value this entry was written under
};

void moe_pred_ring_write(const std::int32_t* ids, int count, std::uint32_t source_bank,
                         std::uint32_t target_bank, std::uint32_t layer,
                         PredRingHeader* headers, std::int32_t* entry_ids,
                         std::uint32_t id_capacity, std::uint32_t depth,
                         unsigned long long* head, cudaStream_t stream);

} // namespace ninfer::ops
