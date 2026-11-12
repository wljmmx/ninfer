#pragma once
#include "ninfer/types.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/resources.h"
#include "runtime/engine/kv_capacity.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/program/prefill_row.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace ninfer {
struct DeviceContext;
}

namespace ninfer::models::qwen3_5 {
namespace execution {
class Parameters;
}

namespace detail {
struct SequencePlanImpl;
struct SequencePlannerImpl;
struct RequestBasePlanImpl;
struct ResumeStateImpl;
struct DemotionPlan;
struct ReleasePlan;
struct ReclaimPlan;
struct DemotionBatch;
class ProgramImpl;
struct ContractAccess;
// Only ever held by shared_ptr in this header (restore identity); defined in the implementation.
struct PreparedCaptureBacking;
} // namespace detail
class SequencePlanner;
class Program;

struct GraphExecutionProfile {
    std::uint32_t min            = 0;
    std::uint32_t max            = 0;
    std::uint32_t topology_class = 0;
};
enum class TextPhase { Prefill, Verify };

struct PrefixShortlistKey {
    std::array<std::uint64_t, 2> digests{};
    std::uint32_t frontier                                                  = 0;
    std::uint32_t identity_tag                                              = 0;
    friend bool operator==(PrefixShortlistKey, PrefixShortlistKey) noexcept = default;
};

class SequencePlan {
public:
    SequencePlan(SequencePlan&&) noexcept;
    SequencePlan& operator=(SequencePlan&&) noexcept;
    ~SequencePlan();

    SequencePlan(const SequencePlan&)            = delete;
    SequencePlan& operator=(const SequencePlan&) = delete;

    [[nodiscard]] std::uint32_t capacity() const noexcept;
    [[nodiscard]] std::uint32_t kv_capacity() const noexcept;
    [[nodiscard]] std::uint32_t max_concurrency() const noexcept;
    [[nodiscard]] std::size_t device_reservation_bytes() const noexcept;
    // port (sm_89): the KV payload the plan actually lays out on the Device (K/V planes of the
    // resident page groups plus the draft cache's own pages). The resolution's stride-based estimate
    // cannot know the draft's extra page, so this is what the ledger and the summary report.
    [[nodiscard]] std::size_t kv_payload_bytes() const noexcept;
    [[nodiscard]] std::size_t workspace_capacity_bytes() const noexcept;
    [[nodiscard]] std::size_t host_capacity_bytes() const noexcept;

private:
    explicit SequencePlan(std::unique_ptr<detail::SequencePlanImpl> impl) noexcept;
    std::unique_ptr<detail::SequencePlanImpl> impl_;

    friend class SequencePlanner;

    friend std::unique_ptr<Program> create_program(const execution::Parameters&, SequencePlan&&,
                                                   DeviceContext&, const StartupObserver&);
};

class SequencePlanner {
public:
    SequencePlanner(SequencePlanner&&) noexcept;
    SequencePlanner& operator=(SequencePlanner&&) noexcept;
    ~SequencePlanner();

    SequencePlanner(const SequencePlanner&)            = delete;
    SequencePlanner& operator=(const SequencePlanner&) = delete;

    [[nodiscard]] const runtime::SequenceCapacityCurve& capacity_curve() const noexcept;
    // port (sm_89): the measured split of the curve stride into "payload of a resident Device page
    // group" and "ledger of a logical page group". Both zero when the model keeps no full-attention
    // pages, and then the residency model stays inactive.
    [[nodiscard]] const runtime::KvResidencyStrides& residency_strides() const noexcept;
    // port (sm_89): a captured decode graph is only valid while the expert resident set is fixed,
    // and that is known only after the pool is attached. Must be called before finalize().
    void set_use_cuda_graph(bool enabled) noexcept;
    // `resident_page_groups` (0 = fully resident) is the Device page pool the plan is built with, and
    // `host_kv_payload_bytes` the payload the Host tier must be able to hold. Both come straight from
    // the capacity resolution; passing the defaults reproduces the pre-residency plan exactly.
    [[nodiscard]] SequencePlan finalize(std::uint32_t main_page_groups,
                                        std::uint32_t resident_page_groups = 0,
                                        std::size_t host_kv_payload_bytes  = 0) &&;

private:
    explicit SequencePlanner(std::unique_ptr<detail::SequencePlannerImpl> impl) noexcept;
    std::unique_ptr<detail::SequencePlannerImpl> impl_;

    friend SequencePlanner make_sequence_planner(const execution::Parameters&, DeviceContext&,
                                                 const EngineOptions&);
};

class RequestBasePlan {
public:
    RequestBasePlan(RequestBasePlan&&) noexcept;
    RequestBasePlan& operator=(RequestBasePlan&&) noexcept;
    ~RequestBasePlan();

    RequestBasePlan(const RequestBasePlan&)            = delete;
    RequestBasePlan& operator=(const RequestBasePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;
    [[nodiscard]] const PreparedContextCache& context_cache() const noexcept;
    // The prompt as the model will see it. Read-only, and the only identity an external cache can be
    // keyed by: a checkpoint is addressed by its token sequence, nothing else.
    [[nodiscard]] std::span<const TokenId> prompt_tokens() const noexcept;
    // What a restored checkpoint must carry to be findable later. Without it the record is readable
    // and checkpoint_matches() never accepts it -- the comparison is on identity, not on bytes.
    [[nodiscard]] std::shared_ptr<const detail::PreparedCaptureBacking> capture_backing() const noexcept;
    [[nodiscard]] std::vector<std::uint32_t> capture_frontiers() const;
    [[nodiscard]] std::optional<PrefixShortlistKey>
    prefix_shortlist_key(std::uint32_t frontier) const noexcept;

private:
    explicit RequestBasePlan(std::shared_ptr<detail::RequestBasePlanImpl> impl) noexcept;
    std::shared_ptr<detail::RequestBasePlanImpl> impl_;

    friend class detail::ProgramImpl;
};

class SequenceHandle {
public:
    SequenceHandle() noexcept                                       = default;
    SequenceHandle(const SequenceHandle&) noexcept                  = default;
    SequenceHandle& operator=(const SequenceHandle&) noexcept       = default;
    friend bool operator==(SequenceHandle, SequenceHandle) noexcept = default;

private:
    const void* owner_ = nullptr;
    runtime::LaneId lane_{};
    std::uint64_t epoch_ = 0;

    friend struct detail::ContractAccess;
};

// One complete immutable state/KV coverage record. Generation protects reused descriptors.
struct CheckpointHandle {
    const void* owner                                                   = nullptr;
    std::uint32_t index                                                 = 0;
    std::uint64_t generation                                            = 0;
    friend bool operator==(CheckpointHandle, CheckpointHandle) noexcept = default;
};

struct CheckpointMetadata {
    std::uint32_t frontier       = 0;
    runtime::CheckpointRole role = runtime::CheckpointRole::Continuation;
    bool leased                  = false;
};

struct CheckpointSummary {
    PrefixShortlistKey key;
    std::uint32_t frontier       = 0;
    runtime::CheckpointRole role = runtime::CheckpointRole::Continuation;
    bool leased                  = false;
    // Resources relevant to eviction; shared optional aliases are counted per record.
    runtime::ContextResourceUsage evictable_resources;
};

// A checkpoint's KV prefix as raw Host bytes, page by page. State images are not part of it yet:
// restoring KV without the recurrent state would resume from a state that never saw the prefix.
// What a state image export hands back. The layout numbers travel with the bytes on purpose: a
// restore that memcpy'd one model's state into another would not fail, it would just produce wrong
// output, so the import side has to be able to refuse on numbers rather than on faith.
struct CheckpointStateExport {
    std::uint64_t image_bytes    = 0;  // u64: a u32 would cap the image at 4 GiB
    // L6-b: the tokens the state was captured after. The KV run is stored in whole pages, so
    // page_count * page_tokens is a bound and not the length; the tier's state header carries this
    // so a reader can tell where the real prefix ends (and, later, pad the tail to a whole page
    // without claiming the padding).
    std::uint32_t valid_tokens   = 0;
    std::uint32_t layers         = 0;   // linear-attention layers that own this image
    std::int32_t  conv_channels  = 0;
    std::int32_t  value_heads    = 0;
    std::int32_t  value_head_dim = 0;
    std::int32_t  key_head_dim   = 0;
    std::int32_t  hidden         = 0;
    bool has_dflash_local        = false;  // speculative draft regions present
    bool materialised_host_copy  = false;  // false = a Host replica already existed
    std::vector<std::byte> bytes;
};
struct CheckpointExport {
    std::uint32_t frontier   = 0;
    std::uint32_t page_count = 0;
    std::size_t page_bytes   = 0;
    std::vector<std::uint32_t> epochs;
    std::vector<std::byte> kv_bytes;
};

// The sizes LMCache has to be told before anything can be stored, read from the live layout rather
// than from a model constant: the KV page stride and the state image size are what the transport
// validates every payload against, and they are only known once the pools exist.
struct CheckpointTierGeometry {
    std::size_t page_bytes         = 0;
    // Tokens per page, from the live geometry. Not kPagedKVPageSize: a caller that hard-codes the
    // page size is hard-coding this engine's, and the cache chunk is expressed in pages.
    std::int64_t page_tokens       = 0;
    std::size_t state_image_bytes  = 0;
    // The KV format itself, not just how big a page is. Two layouts can land on the same page byte
    // count -- a 4-bit format at one head width and an 8-bit one at half of it both give 136 B per
    // vector -- so a cache keyed only on sizes would let one answer for the other. The tag is the
    // wire image of KvCacheStorage (same numbering), 0 meaning BFloat16.
    std::uint32_t kv_dtype_tag     = 0;
    // 2 = K and V occupy separate vectors, 1 = fused (MLA / indexer).
    std::uint32_t kv_planes        = 0;
};

// How much room the KV page pool has right now. A restore takes Device pages for the whole life of
// the checkpoint it creates, so "there is room" is not the same question as "it fits": a restore
// that takes the last free page leaves the next request with nowhere to put its prompt.
//
// Deliberately not part of CheckpointTierGeometry: that describes the format, this describes the
// moment. Mixing them would make a snapshot look like a fact about the layout.
struct CheckpointPoolHeadroom {
    std::uint32_t capacity_pages  = 0;
    std::uint32_t available_pages = 0;
};

struct SourceCandidate {
    std::optional<CheckpointHandle> checkpoint;
    std::uint32_t reused_tokens = 0;
    runtime::PrefillWork remaining_work;
    std::vector<runtime::ContextTransferRequirement> transfers;
    bool consume_source = false;
    bool take_private   = false;
    bool move_state     = false;
    bool move_history   = false;
    bool split_state    = false;
    bool backup_state   = false;
    // Compatible carry intent. Binding may consume the selected point if its State slot
    // is necessary for execution and no preservation destination can be obtained.
    std::vector<CheckpointHandle> private_points;
    // Authorized retirement, applied only after binding capacity has been checked.
    std::vector<CheckpointHandle> retired_points;
};

struct BindingReservation {
    bool reserved          = false;
    bool source_valid      = true;
    bool capacity_possible = true;
    runtime::ContextResourceUsage shortage;
    std::vector<CheckpointHandle> retired_points;
    std::optional<CheckpointHandle> consumed_source;
    // What this binding actually reuses, when the caller cannot know it from the candidate. A
    // third-tier import is found inside binding (there is no handle to pass in and none to read the
    // frontier from), so the number has to travel back out or the request would report itself as a
    // root prefill after reusing most of its prompt.
    std::uint32_t reused_tokens = 0;

    explicit operator bool() const noexcept { return reserved; }
};

struct ContextDemotion {
    std::vector<CheckpointHandle> sources;
    std::size_t host_bytes = 0;
    runtime::ContextResourceUsage released;
    std::shared_ptr<const detail::DemotionPlan> impl;
};

struct ContextRelease {
    std::vector<CheckpointHandle> sources;
    // Guaranteed release in the queried shortage pool; other quantities are not inventoried.
    runtime::ContextResourceUsage released;
    std::shared_ptr<const detail::ReleasePlan> impl;
};

class ContextDemotionBatch {
public:
    ContextDemotionBatch(ContextDemotionBatch&&) noexcept;
    ContextDemotionBatch& operator=(ContextDemotionBatch&&) noexcept;
    ~ContextDemotionBatch();
    ContextDemotionBatch(const ContextDemotionBatch&)            = delete;
    ContextDemotionBatch& operator=(const ContextDemotionBatch&) = delete;

    [[nodiscard]] bool append(const ContextDemotion&);
    [[nodiscard]] bool covers(const ContextRelease&) const;
    [[nodiscard]] std::optional<ContextDemotion> finish() const;

private:
    explicit ContextDemotionBatch(std::unique_ptr<detail::DemotionBatch>);
    std::unique_ptr<detail::DemotionBatch> impl_;
    friend struct ContextReclaimPlan;
};

// Read-only facts for one synchronous evaluation. Discard before any Native mutation or yield.
// start_demote independently revalidates the selected quote before reserving destinations.
struct ContextReclaimPlan {
    std::vector<ContextDemotion> demotions;
    std::vector<ContextRelease> releases;

    [[nodiscard]] ContextDemotionBatch begin_kv_batch(runtime::ContextResourceUsage shortage) const;
    [[nodiscard]] std::uint64_t recovery_loss(std::span<const CheckpointHandle> removed,
                                              std::span<const CheckpointHandle> surviving) const;

private:
    std::shared_ptr<detail::ReclaimPlan> impl_;
    friend class detail::ProgramImpl;
};

class ResumeState {
public:
    ResumeState(ResumeState&&) noexcept;
    ResumeState& operator=(ResumeState&&) noexcept;
    ~ResumeState();
    ResumeState(const ResumeState&)            = delete;
    ResumeState& operator=(const ResumeState&) = delete;
    [[nodiscard]] bool has_snapshot() const noexcept;
    [[nodiscard]] std::optional<CheckpointHandle> snapshot_handle() const noexcept;
    [[nodiscard]] std::uint32_t frontier() const noexcept;

private:
    explicit ResumeState(std::unique_ptr<detail::ResumeStateImpl>) noexcept;
    std::unique_ptr<detail::ResumeStateImpl> impl_;

    friend class detail::ProgramImpl;
};
enum class ExecutionUnitKind : std::uint8_t { Prefill, Replay, Decode, Control, Normalize };

struct ExecutionUnit {
    SequenceHandle sequence;
    ExecutionUnitKind kind = ExecutionUnitKind::Decode;
    // Decode: remaining output budget. Control: exact forced span. Other kinds: zero.
    std::uint32_t tokens = 0;
};

class PendingBatch {
public:
    PendingBatch() noexcept = default;
    ~PendingBatch()         = default;

    PendingBatch(PendingBatch&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)),
          transaction_(std::exchange(other.transaction_, 0)), rows_(other.rows_),
          row_count_(std::exchange(other.row_count_, 0)), tokens_(other.tokens_),
          row_counts_(other.row_counts_), row_stride_(other.row_stride_), timing_(other.timing_),
          constraint_failed_(other.constraint_failed_) {
        other.tokens_     = {};
        other.row_counts_ = {};
        other.row_stride_ = 0;
        other.timing_     = {};
    }

    PendingBatch& operator=(PendingBatch&&)      = delete;
    PendingBatch(const PendingBatch&)            = delete;
    PendingBatch& operator=(const PendingBatch&) = delete;

    [[nodiscard]] std::size_t row_count() const noexcept { return row_count_; }

    [[nodiscard]] std::span<const TokenId> tokens() const noexcept { return tokens_; }

    [[nodiscard]] std::span<const std::int32_t> row_counts() const noexcept { return row_counts_; }

    [[nodiscard]] std::uint32_t row_stride() const noexcept { return row_stride_; }

    [[nodiscard]] bool constraint_failed(std::size_t row) const {
        return constraint_failed_.at(row);
    }

    [[nodiscard]] runtime::ExecutionTiming execution_timing() const noexcept { return timing_; }

private:
    const void* owner_         = nullptr;
    std::uint64_t transaction_ = 0;
    std::array<SequenceHandle, kMaximumConcurrency> rows_{};
    std::size_t row_count_ = 0;
    std::span<const TokenId> tokens_;
    std::span<const std::int32_t> row_counts_;
    std::uint32_t row_stride_ = 0;
    runtime::ExecutionTiming timing_;
    std::array<bool, kMaximumConcurrency> constraint_failed_{};

    friend struct detail::ContractAccess;
};

struct PrefillProgress {
    runtime::BeginSummary summary;
    std::uint32_t processed_prompt_tokens = 0;
    bool complete                         = false;
    runtime::ExecutionTiming timing;
    std::optional<PendingBatch> pending;
    bool capture_ready = false;
};

struct CommitRowResult {
    runtime::CommitDisposition disposition = runtime::CommitDisposition::Active;
    GenerationTimings timings;
    SpeculativeStats speculative;

    // Fixed-size cumulative observation, including active rows without copying per-position data.
    struct SpeculativeCounters {
        std::uint64_t rounds          = 0;
        std::uint64_t drafted_tokens  = 0;
        std::uint64_t accepted_tokens = 0;
        std::uint64_t fallback_steps  = 0;
    } speculative_counters;
};

struct CommitResult {
    std::array<CommitRowResult, kMaximumConcurrency> rows{};
    std::array<bool, kMaximumConcurrency> capture_ready{};
    std::size_t row_count = 0;
    runtime::ExecutionTiming timing;
};

struct DiscardResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    std::size_t row_count         = 0;
};

struct FinishResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    GenerationTimings timings;
    SpeculativeStats speculative;
    std::optional<CheckpointHandle> checkpoint;
};

struct AbortResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    GenerationTimings timings;
    SpeculativeStats speculative;
};

struct ReplayProgress {
    bool capture_ready             = false;
    std::uint32_t processed_tokens = 0;
    bool complete                  = false;
    runtime::ExecutionTiming timing;
};

struct CapturePreparation {
    std::uint32_t frontier = 0;
    bool reserved          = false;
    runtime::ContextResourceUsage shortage;
    std::size_t host_bytes = 0;
};
enum class ContextOperationKind : std::uint8_t { Bind, Capture, Demote, Pause };

struct ContextProgress {
    ContextOperationKind kind = ContextOperationKind::Bind;
    bool advanced             = false;
    bool complete             = false;
    bool published            = false;
    std::optional<SequenceHandle> sequence;
    std::optional<ResumeState> paused;
    bool replaying = false;
    std::vector<CheckpointHandle> private_points;
    std::vector<CheckpointHandle> retired_checkpoints;
    std::vector<CheckpointHandle> captured_checkpoints;
    std::vector<runtime::ContextTransferObservation> transfers;
    runtime::ContextOperationCounts operations;
    std::optional<GenerationTimings> request_timings;
    SpeculativeStats request_speculative;
};

struct PhysicalUsageSnapshot {
    runtime::ContextResourceUsage occupied;
    runtime::ContextResourceUsage capacity;
    std::size_t host_reserved_bytes      = 0;
    std::size_t host_peak_occupied_bytes = 0;
    std::uint32_t host_state_slots       = 0;
    std::size_t host_kv_bytes            = 0;
};

class Program {
public:
    ~Program() noexcept;
    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&)                 = delete;
    Program& operator=(Program&&)      = delete;
    [[nodiscard]] RequestBasePlan plan_request(PreparedPrompt&& prompt,
                                               const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] std::vector<float> causal_score(PreparedPrompt&& prompt,
                                                  std::uint32_t first_target);
    [[nodiscard]] std::optional<SourceCandidate>
    inspect_source(const RequestBasePlan& base, std::optional<CheckpointHandle> checkpoint,
                   bool consume_source                              = false,
                   std::span<const CheckpointHandle> private_points = {},
                   std::span<const CheckpointHandle> retired_points = {}) const;
    [[nodiscard]] PrefixShortlistKey checkpoint_key(CheckpointHandle, std::uint32_t frontier) const;
    [[nodiscard]] runtime::ContextResourceUsage
        checkpoint_footprint(std::span<const CheckpointHandle>) const;
    [[nodiscard]] CheckpointSummary checkpoint_summary(CheckpointHandle checkpoint) const;
    [[nodiscard]] std::optional<CheckpointExport> export_checkpoint(CheckpointHandle checkpoint);
    [[nodiscard]] std::optional<CheckpointStateExport> export_state(CheckpointHandle checkpoint);
    [[nodiscard]] CheckpointTierGeometry checkpoint_geometry() const;
    // Live room in the Device page pool. A caller that is about to import a checkpoint asks this
    // first: taking the last free page turns the next request's allocation into a failure.
    [[nodiscard]] CheckpointPoolHeadroom checkpoint_pool_headroom() const;
    // Restores a checkpoint from bytes an external cache handed back, pricing the fingerprint from
    // the live layout. Returns nullopt when the bytes do not fit this program; error says why.
    [[nodiscard]] std::optional<CheckpointHandle> import_checkpoint_current_layout(
        const CheckpointExport& kv, const std::vector<std::string>& state_shards,
        std::shared_ptr<const detail::PreparedCaptureBacking> identity = nullptr,
        const PrefixShortlistKey* key = nullptr, std::string* error = nullptr);
    [[nodiscard]] CheckpointMetadata checkpoint_metadata(CheckpointHandle checkpoint) const;
    [[nodiscard]] bool checkpoint_matches(CheckpointHandle checkpoint,
                                          const RequestBasePlan& base) const;
    [[nodiscard]] std::uint32_t checkpoint_recovery_frontier(CheckpointHandle retained,
                                                             const RequestBasePlan& base,
                                                             std::uint32_t target) const;
    [[nodiscard]] std::uint64_t
    checkpoint_recovery_loss(std::span<const CheckpointHandle> removed,
                             std::span<const CheckpointHandle> surviving) const;
    [[nodiscard]] bool valid_checkpoint(CheckpointHandle checkpoint) const noexcept;
    [[nodiscard]] bool release_checkpoint(CheckpointHandle checkpoint) noexcept;
    [[nodiscard]] bool revoke_snapshot(ResumeState& paused) noexcept;
    [[nodiscard]] runtime::ContextResourceUsage snapshot_resources(const ResumeState& paused) const;
    // Exact Host bytes freed by deleting this fixed set, with physical sharing counted once.
    [[nodiscard]] std::size_t
    host_bytes_released(std::span<const CheckpointHandle> checkpoints) const;
    [[nodiscard]] std::optional<std::size_t> pause_host_bytes(SequenceHandle sequence) const;
    [[nodiscard]] std::size_t
    release_redundant_host(std::span<const CheckpointHandle> excluded,
                           std::optional<SequenceHandle> pending_backup = std::nullopt);
    // Reservations stay owned by their lane through commit. Failure leaves prior permits intact.
    [[nodiscard]] runtime::ResourceReservation reserve_units(std::span<const ExecutionUnit> units);
    // 13.4: runtime shrink of the KV half. The caller is the engine's idle-window check; what this
    // adds is the ownership check - a retained checkpoint whose pages sit above the new edge is an
    // owner, so the shrink is refused instead of silently invalidating it. Returns the capacity in
    // effect afterwards (0 when the model has no main KV pool).
    std::uint32_t resize_kv_capacity(std::uint32_t target_pages);
    // Q67/S1: prompt tokens this lane still has to prefill (0 when it is not prefilling). The
    // cursor lives in the Program, not in the engine's request record, so the batch container asks
    // here for what a merged pass would be padded to.
    [[nodiscard]] std::uint32_t prefill_remaining(std::uint32_t lane) const;
    // port (sm_89): the live edge of the KV residency contract, and how far one lane's KV has grown.
    // Under residency the Device pool deliberately holds only a subset of the logical pages (the
    // Host tier carries the rest), and the full-attention layers read the whole history every step,
    // so a request whose live history exceeds the pool cannot be served at all - not by waiting and
    // not by swapping. `kv_residency_token_limit()` is that bound in tokens as the LIVE pool edge
    // (the control plane can shrink the pool at runtime), and 0 means "no residency contract: every
    // logical page is resident". The Runtime asks the pair before it hands a lane an execution unit,
    // so a request that can never fit is refused with its numbers instead of being paused and
    // replayed against a bound it can never satisfy.
    [[nodiscard]] std::uint32_t kv_residency_token_limit() const noexcept;
    [[nodiscard]] std::uint32_t sequence_execution_frontier(SequenceHandle sequence) const noexcept;
    // The most a single Decode step adds to that frontier (its token plus the draft window when a
    // speculative backend runs). The Runtime prices a step's growth with it.
    [[nodiscard]] std::uint32_t kv_residency_step_growth() const noexcept;
    // port (sm_89): the pool is shared, so admission cannot price a request alone. This is what the
    // lanes already resident hold: one entry per live lane, each rounded up to whole pages (a lane's
    // address owns whole pages of the one physical pool) and already including one Decode step of
    // growth, because that is the part every resident lane is guaranteed. `live_tokens` is their sum
    // and `capacity_tokens` the same pool edge `kv_residency_token_limit()` reports (0 = no contract).
    struct KvResidencyBudget {
        std::uint32_t capacity_tokens = 0;
        std::uint32_t live_tokens     = 0;
        std::uint32_t lane_count      = 0;
        std::array<std::uint32_t, kMaximumConcurrency> lane_tokens{};
        // The same pricing in the pool's own unit - whole resident page groups - so a refusal can say
        // what the growth asked for and what the pool holds. `lane_pages` is each lane's demand after
        // one Decode step of growth, the quantity the Runtime compares with `capacity_pages`.
        std::uint32_t capacity_pages = 0;
        std::uint32_t live_pages     = 0;
        std::array<std::uint32_t, kMaximumConcurrency> lane_pages{};
    };
    [[nodiscard]] KvResidencyBudget kv_residency_budget() const noexcept;
    // The same pricing for a request that is not live yet: its prompt plus one step of growth,
    // rounded up to whole pages. 0 when there is no residency contract.
    [[nodiscard]] std::uint32_t kv_residency_request_tokens(std::uint32_t prompt_tokens) const noexcept;
    // Q67/S1: name each lane's place in a merged prefill pass (build only; nothing executes).
    std::size_t prefill_batch_rows(std::span<const std::uint32_t> lanes,
                                   std::span<execution::PrefillRow> out);
    void report_prefill_rows(std::span<const std::uint32_t> lanes);
    // Q67/S1: what one merged prefill pass produced. `lanes`/`summaries` are the per-row bookkeeping
    // the engine needs, because commit_pending takes N lanes exactly like a decode round.
    struct MergedPrefillResult {
        PrefillProgress progress;
        std::array<std::uint32_t, kMaximumConcurrency> lanes{};
        std::array<runtime::BeginSummary, kMaximumConcurrency> summaries{};
        std::size_t count = 0;
    };
    [[nodiscard]] std::optional<MergedPrefillResult> advance_prefill_batch();
    [[nodiscard]] bool reclaim_capture_reservation(runtime::ContextResourceUsage shortage);
    void release_units(std::span<const SequenceHandle> sequences) noexcept;
    [[nodiscard]] BindingReservation
    start_binding(const RequestBasePlan& base, runtime::LaneId lane, const SourceCandidate& source,
                  ResumeState* resume           = nullptr,
                  ExecutionUnitKind resume_kind = ExecutionUnitKind::Decode,
                  std::uint32_t resume_tokens   = 1);
    [[nodiscard]] bool start_capture(SequenceHandle sequence);
    [[nodiscard]] bool capture_is_input(SequenceHandle sequence) const;
    [[nodiscard]] std::optional<CapturePreparation> prepare_capture(SequenceHandle sequence);
    void skip_capture(SequenceHandle sequence);
    // Finite physical actions, each including every optional holder needed for its release.
    [[nodiscard]] ContextReclaimPlan plan_reclaim(std::span<const CheckpointHandle> allowed,
                                                  std::span<const CheckpointHandle> excluded,
                                                  runtime::ContextResourceUsage shortage) const;
    [[nodiscard]] std::vector<ContextRelease>
    plan_releases(std::span<const CheckpointHandle> allowed,
                  std::span<const CheckpointHandle> excluded,
                  runtime::ContextResourceUsage shortage) const;
    [[nodiscard]] bool start_demote(const ContextDemotion& plan);
    [[nodiscard]] bool start_pause(SequenceHandle sequence, bool save_snapshot,
                                   runtime::ExecutionTiming* timing = nullptr);
    [[nodiscard]] ContextProgress poll_context(runtime::CancellationFlagView cancellation);
    [[nodiscard]] bool has_context_transaction() const noexcept;
    [[nodiscard]] bool context_blocks(SequenceHandle sequence) const noexcept;
    // A resumed binding retains its complete recovery capacity until committed new progress.
    [[nodiscard]] bool recovery_pending(SequenceHandle sequence) const noexcept;
    [[nodiscard]] PrefillProgress advance_prefill(SequenceHandle sequence,
                                                  runtime::ExecutionTiming* failed_timing = nullptr,
                                                  runtime::TokenMaskProvider* masks = nullptr);
    [[nodiscard]] ReplayProgress advance_replay(SequenceHandle sequence,
                                                runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] PendingBatch decode(std::span<const SequenceHandle> sequences,
                                      std::span<const runtime::RoundBudget> budgets,
                                      runtime::ExecutionTiming* failed_timing = nullptr,
                                      runtime::TokenMaskProvider* masks       = nullptr);
    // Forced control contributes to counts once. Replay does not call this operation.
    [[nodiscard]] runtime::ExecutionTiming
    append_forced_tokens(std::span<const SequenceHandle> sequences,
                         std::span<const TokenId> row_major_tokens, std::uint32_t row_stride,
                         std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
                         runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] CommitResult
    commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions,
           runtime::CommitObservation observation  = runtime::CommitObservation::AllRows,
           runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] DiscardResult abort_pending(PendingBatch&& pending) noexcept;
    [[nodiscard]] FinishResult finish(SequenceHandle sequence) noexcept;
    [[nodiscard]] AbortResult abort(SequenceHandle sequence) noexcept;
    void fail_all_cleanup() noexcept;
    [[nodiscard]] PhysicalUsageSnapshot physical_usage() const noexcept;
    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    void reset_memory_peaks() noexcept;
private:
    explicit Program(std::unique_ptr<detail::ProgramImpl> impl) noexcept;
    std::unique_ptr<detail::ProgramImpl> impl_;
    friend std::unique_ptr<Program> create_program(const execution::Parameters&, SequencePlan&&,
                                                   DeviceContext&, const StartupObserver&);
};

[[nodiscard]] SequencePlanner make_sequence_planner(const execution::Parameters&, DeviceContext&,
                                                    const EngineOptions&);
// appendix IK/IL: the eviction contract is decided once at startup; this installs that decision
// so the per-round application reads it instead of re-reading the environment (the environment
// still wins, which keeps every recorded experiment reproducible). The per-dtype capability
// predicate is exported too, because the decision must respect it: NVFP4 has no absent-page
// path, so a window there would be a silent no-op (sequence.cpp:392-410).
void install_kv_eviction_window(std::uint32_t sink_pages, std::uint32_t window_pages);
[[nodiscard]] bool kv_absent_pages_supported_for(KvCacheStorage storage) noexcept;

// True when the environment already names a window. The startup decision must know this: with
// the CLI at off, an environment-named contract still wins, and reporting it as off would be
// exactly the silent skip this work removes.
[[nodiscard]] bool kv_eviction_window_from_environment_enabled() noexcept;

[[nodiscard]] std::unique_ptr<Program> create_program(const execution::Parameters&, SequencePlan&&,
                                                      DeviceContext&, const StartupObserver&);
} // namespace ninfer::models::qwen3_5

