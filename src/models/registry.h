#pragma once

#include <string_view>

namespace ninfer::models::qwen3_5 {
struct TextConfig;
} // namespace ninfer::models::qwen3_5

namespace ninfer::models {

enum class Architecture { Qwen3_5, Qwen3_5Moe, Qwen4Exp };

[[nodiscard]] Architecture resolve_architecture(std::string_view architecture,
                                                std::string_view model_type);
[[nodiscard]] std::string_view architecture_name(Architecture architecture) noexcept;

// What a model family states about itself. Parsing and loading read this record instead of asking
// which architecture is running, so the identity of a family appears in one place: the registry
// table below, whose Qwen4Exp row is the one `models/qwen4_exp` owns.
struct Identity {
    // The architecture the MTP companion must declare: the target's own name with the `ForCausalLM`
    // suffix replaced by `MTP`.
    std::string_view mtp_architecture;
    // A routed-expert FFN instead of a dense one.
    bool moe = false;
    // The artifact packs every routed expert of a bank into one binding per matrix (gate, up, down),
    // so a per-expert parameter is a row block of that bank.
    bool banked_experts = false;
    // The artifact names the activation each projection consumes (`attention_input`, `moe_input`)
    // rather than the engine's logical name.
    bool aliased_activations = false;
    // The residual is `hc_count` streams, mixed through a rank-`hc_lowrank` pair.
    bool four_stream_residual = false;
    // The GDN output-gate activation the family's artifacts declare; empty leaves it to the artifact.
    std::string_view output_gate;
    // The family's own structural requirements, run after the generic text parse. One explicit
    // entry per family; there is no discovery and no per-checkpoint registration.
    void (*complete)(qwen3_5::TextConfig&) = nullptr;
};

[[nodiscard]] Identity identity(Architecture architecture) noexcept;

} // namespace ninfer::models
