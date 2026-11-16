#pragma once

#include "ninfer/types.h"

#include <string_view>

namespace ninfer::models {

struct LoadOptions {
    EnginePurpose purpose          = EnginePurpose::Generation;
    bool vision                    = false;
    SpeculativeBackend speculative = SpeculativeBackend::None;
    ProposalHead proposal_head     = ProposalHead::Full;
    // port (sm_89): routed experts are not materialized; they stream from the artifact.
    bool stream_experts            = false;
    // port (sm_89): keep a Host replica as the streamed tier's source.
    bool expert_host_copy          = false;
    // V10-14: explicit cold-start expert-priority override, a JSON file with the artifact table's
    // shape (see models/qwen3_5/expert_priority_source.h). It wins over the artifact's own table,
    // row by row. Left empty, an override can still be named through NINFER_EXPERT_PRIORITY_FILE -
    // the load resolves that fallback itself, so this field stays empty in the options a caller
    // compares against, and the two never disagree about what was loaded.
    std::filesystem::path expert_priority_file;

    bool operator==(const LoadOptions&) const = default;

    [[nodiscard]] bool speculative_enabled() const noexcept {
        return speculative != SpeculativeBackend::None;
    }

    [[nodiscard]] bool mtp() const noexcept { return speculative == SpeculativeBackend::Mtp; }

    [[nodiscard]] bool dflash() const noexcept { return speculative == SpeculativeBackend::DFlash; }

    [[nodiscard]] bool dflash2() const noexcept {
        return speculative == SpeculativeBackend::DFlash2;
    }

    [[nodiscard]] bool masked_draft() const noexcept { return dflash() || dflash2(); }

    [[nodiscard]] bool proposal_enabled() const noexcept {
        return purpose == EnginePurpose::Generation && speculative != SpeculativeBackend::None &&
               proposal_head == ProposalHead::Optimized;
    }

    [[nodiscard]] std::string_view speculative_component() const noexcept {
        switch (speculative) {
        case SpeculativeBackend::None:
            return {};
        case SpeculativeBackend::Mtp:
            return "mtp";
        case SpeculativeBackend::DFlash:
            return "dflash";
        case SpeculativeBackend::DFlash2:
            return "dflash2";
        }
        return {};
    }
};

[[nodiscard]] constexpr bool is_masked_draft_backend(SpeculativeBackend backend) noexcept {
    return backend == SpeculativeBackend::DFlash || backend == SpeculativeBackend::DFlash2;
}

[[nodiscard]] inline LoadOptions load_options(const EngineOptions& options) noexcept {
    return {.purpose       = options.purpose,
            .vision        = options.enable_vision,
            .speculative   = options.speculative.backend,
            .proposal_head = options.speculative.proposal_head,
            .stream_experts   = options.expert_cache_bytes != 0,
            .expert_host_copy = options.expert_host_copy};
}

} // namespace ninfer::models
