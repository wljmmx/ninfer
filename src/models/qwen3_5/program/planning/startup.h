#pragma once
#include "models/qwen3_5/program/internal.h"
#include "runtime/engine/kv_capacity.h"

#include "core/cyclic_kv_cache.h"
#include "core/controller_mode.h"
#include "core/dtype.h"
#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/tensor.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "models/qwen3_5/program/round_buffers.h"
#include "models/qwen3_5/state/state_image.h"
#include "models/load_options.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>

namespace ninfer::models::qwen3_5::detail {

using TensorLayout                              = TensorRegion;
inline constexpr std::uint32_t kCausalScoreTile = 1024;

// P1a: a lane's StateImage is a complete mirror of its decode state (~196 MiB at 204800 tokens), so
// allocating one Device image per lane plus every declared extra snapshot priced 3.13 GB at eight
// lanes - the term that kept the 8-lane dflash2 configuration from starting on a 24 GB card. Only
// the images an actively running lane reads have to be Device-resident: StateImageStore already
// keeps a logical identity alive with a Host replica instead of a Device slot
// (StateImageStore::reserve_device_to_host / begin_host_to_device), and the engine reclaims a state
// slot by demoting a retained checkpoint's image before it admits another lane. The layout therefore
// allocates this many physical Device images while the store is given the logical capacity
// separately.
//
// P1c: the count is the controller's decision (core/controller_mode.h: auto/stable 2, performance 4),
// not an environment switch a caller has to remember. NINFER_STATE_IMAGE_DEVICE_SLOTS still wins when
// it is set - reproducing a recorded arm is a contract - and it is clamped to the logical count.
inline constexpr std::uint32_t kDefaultDeviceStateImages = 2;

[[nodiscard]] inline std::uint32_t device_state_image_slots(std::uint32_t logical_slots) noexcept {
    const auto policy = ninfer::controller_residency_policy(ninfer::controller_mode());
    std::uint32_t physical =
        policy.all_device_images ? logical_slots : policy.device_state_images;
    if (const char* value = std::getenv("NINFER_STATE_IMAGE_DEVICE_SLOTS");
        value != nullptr && *value != '\0') {
        const long requested = std::strtol(value, nullptr, 10);
        if (requested > 0) { physical = static_cast<std::uint32_t>(requested); }
    }
    return std::min(physical, logical_slots);
}

struct DFlashPersistentLayout {
    std::optional<qwen3_5::PagedKVCacheLayout> full;
    TensorLayout prefill_features;
    TensorLayout prefill_positions;
    TensorLayout pending_features;

    [[nodiscard]] std::size_t kv_payload_bytes() const noexcept {
        return full ? full->payload_bytes() : 0;
    }
};

struct PersistentLayout {
    qwen3_5::DecoderStateLayout decoder;
    qwen3_5::StateImageDeviceLayout state_images;
    std::optional<GdnReplayRecordLayout> replay_records;
    std::optional<DFlashPersistentLayout> dflash;
    qwen3_5::RoundStateLayout round;
    TensorLayout prefill_hidden;
    std::optional<TensorLayout> score_hidden;
    // A four-stream model's prefill frame is count*hidden wide, so the scoring pass cannot read its
    // per-position block output out of it; this is the head-wide outlet it reads instead. Absent on a
    // single-stream model, where the frame already is the sample.
    std::optional<TensorLayout> score_sample;
    std::optional<TensorLayout> token_counts;
    std::optional<TensorLayout> sampling_config;
    std::optional<TensorLayout> grammar_masks;
    std::size_t bytes            = 0;
    std::size_t kv_payload_bytes = 0;
};

struct VisionWorkspacePlan {
    std::int32_t output_hidden         = 0;
    std::uint32_t max_merged_tokens    = 0;
    std::size_t general_capacity_bytes = 0;
    std::size_t encode_peak_bytes      = 0;
    std::size_t handoff_offset_bytes   = 0;
    std::size_t handoff_capacity_bytes = 0;
    std::size_t capacity_bytes         = 0;
};

struct WorkspacePlan {
    std::size_t text_prefill     = 0;
    std::size_t ordinary_round   = 0;
    std::size_t mtp_prefill      = 0;
    std::size_t mtp_round        = 0;
    std::size_t dflash_context   = 0;
    std::size_t dflash_round     = 0;
    std::size_t causal_score     = 0;
    std::size_t general_capacity = 0;
    std::optional<VisionWorkspacePlan> vision;
    std::size_t capacity = 0;
};

struct SequencePlanningInputs {
    const execution::Parameters* parameters = nullptr;
    std::uint32_t capacity                  = 0;
    std::uint32_t max_concurrency           = 1;
    std::uint32_t prefill_chunk             = 0;
    std::uint32_t draft_window              = 0;
    SpeculativeBackend speculative_backend  = SpeculativeBackend::None;
    KvCacheStorage kv_storage               = KvCacheStorage::BFloat16;
    ProposalHead proposal_head              = ProposalHead::Full;
    float draft_conf                        = 0.0F;
    models::LoadOptions features;
    bool use_cuda_graph               = true;
    bool causal_scoring               = false;
    int device                        = 0;
    std::int32_t multiprocessor_count = 0;
    ContextCacheOptions context_cache;
    // port (sm_89): the Device page pool this plan is built with. Zero means "fully resident": the
    // pool follows the logical page count, exactly as every pre-residency caller expects. A nonzero
    // value builds the pool at that size while the execution tables stay logical, which is what lets
    // logical capacity exceed what the Device can hold resident.
    std::uint32_t resident_page_groups = 0;
};

struct SequencePlanImpl {
    const execution::Parameters* parameters = nullptr;
    std::uint32_t capacity                  = 0;
    std::uint32_t kv_capacity               = 0;
    std::uint32_t main_page_groups          = 0;
    std::uint32_t max_concurrency           = 1;
    std::uint32_t prefill_chunk             = 0;
    std::uint32_t draft_window              = 0;
    SpeculativeBackend speculative_backend  = SpeculativeBackend::None;
    KvCacheStorage kv_storage               = KvCacheStorage::BFloat16;
    ProposalHead proposal_head              = ProposalHead::Full;
    float draft_conf                        = 0.0F;
    models::LoadOptions features;
    bool use_cuda_graph               = true;
    bool causal_scoring               = false;
    int device                        = 0;
    std::int32_t multiprocessor_count = 0;
    ContextCacheOptions context_cache;
    PersistentLayout persistent;
    WorkspacePlan workspace;
    std::size_t graph_allowance_bytes    = 0;
    std::size_t device_reservation_bytes = 0;
    // port (sm_89): logical (execution table) vs resident (Device page pool) page groups, and the
    // payload the plan leaves on the Host side. Equal page counts and a zero payload when the plan
    // is fully resident, which is the only shape a plan had before residency existed.
    std::uint32_t logical_page_groups  = 0;
    std::uint32_t resident_page_groups = 0;
    std::size_t host_kv_payload_bytes  = 0;
};

struct SequencePlannerImpl {
    SequencePlanningInputs inputs;
    runtime::SequenceCapacityCurve curve;
    // port (sm_89): measured from the real layout, not assumed - the payload a resident page group
    // costs and the ledger every logical page group costs on top. The two must add up to the curve
    // stride, and kv_capacity.cpp refuses the plan if they do not.
    runtime::KvResidencyStrides residency_strides;
    std::unique_ptr<SequencePlanImpl> minimum;
};

// appendix FR/FS (S-A): the masked draft has no positional SCALING - DraftConfig carries a
// rope_theta but no factor, and the draft's RoPE angle is computed from the absolute position, with
// the fused materialization contract pinning theta at 1e7 - while the text model is allowed past
// its trained range through YaRN. So a context longer than the draft's max_position_embeddings
// cannot be served by the draft. Measured past that range (one lane, max_tokens=32): acceptance
// 72.4% -> 54.3% and decode 43.6 tok/s, i.e. the no-speculation baseline (43.5-46.4) - the draft's
// work is simply wasted there, which is why such a batch is served by the ordinary round.
// NINFER_DRAFT_ALLOW_POSITION_OVERRUN=<nonzero> forces the draft past its range anyway (FR.3's
// measurement mode) and thereby disables this fallback.
[[nodiscard]] inline bool draft_position_overrun_allowed() noexcept {
    const char* value = std::getenv("NINFER_DRAFT_ALLOW_POSITION_OVERRUN");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

// True when the configured draft cannot serve a context of `capacity` tokens, so that batches
// which reach past the draft's range take the ordinary round. Engaged exactly for a masked draft:
// config.cpp parses Config::draft only for DFlash/DFlash2 and load.cpp binds the draft weights off
// that same member, so `!draft` also excludes Mtp - the backend the original hard guard never fired
// for.
[[nodiscard]] inline bool draft_needs_ordinary_fallback(const execution::Parameters& parameters,
                                                       std::uint32_t capacity) noexcept {
    const auto& draft = parameters.model.config().draft;
    if (!draft) { return false; }
    if (draft_position_overrun_allowed()) { return false; }
    return capacity > draft->max_position_embeddings;
}

[[nodiscard]] std::unique_ptr<qwen3_5::detail::SequencePlannerImpl>
make_sequence_planner_impl(const execution::Parameters& parameters, DeviceContext& device,
                           const EngineOptions& options);
[[nodiscard]] std::unique_ptr<SequencePlanImpl>
finalize_sequence_plan_impl(std::unique_ptr<qwen3_5::detail::SequencePlannerImpl> planner,
                            std::uint32_t main_page_groups, std::uint32_t resident_page_groups = 0,
                            std::size_t host_kv_payload_bytes = 0);

} // namespace ninfer::models::qwen3_5::detail
