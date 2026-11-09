#pragma once

#include "models/qwen3_5/model.h"
#include "ninfer/ops/moe.h"
#include "ninfer/ops/weight_input.h"

#include <array>
#include <memory>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

using LinearParameters = ops::SingleProjectionWeight;

[[nodiscard]] inline std::int32_t dimension(std::uint64_t value) {
    if (value > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("model dimension exceeds the Tensor integer domain");
    }
    return static_cast<std::int32_t>(value);
}

// Which matrix of a routed layer a bank names.
enum class ExpertBankRole { GateUp, Gate, Up, Down };

// The routed experts of one layer as their artifact stores them.
//
// qwen3_5 keeps gate and up adjacent inside one parent, so expert e owns rows [e*2I, (e+1)*2I) of
// that matrix (gate rows first, then up rows). Qwen4Exp keeps three independent banks, so expert e
// owns rows [e*I, (e+1)*I) of gate, the same block of up, and rows [e*H, (e+1)*H) of down.
//
// The form is read off the parents, never assumed: a bank whose rows match neither form is refused
// under its logical parameter name. `block()` is the only way to address one expert, so a caller
// can never quietly read the neighbouring expert's rows.
class ExpertBankDescription {
public:
    [[nodiscard]] static ExpertBankDescription
    from_bindings(const WeightView& gate, const WeightView& up, const WeightView& down,
                  std::uint32_t count, std::uint64_t intermediate, std::string parameter);
    [[nodiscard]] static ExpertBankDescription fused(const WeightView& gate, const WeightView& up,
                                                     const WeightView& down, std::uint32_t count,
                                                     std::uint64_t intermediate,
                                                     std::string parameter);
    [[nodiscard]] static ExpertBankDescription separate(const WeightView& gate, const WeightView& up,
                                                        const WeightView& down, std::uint32_t count,
                                                        std::uint64_t intermediate,
                                                        std::string parameter);

    [[nodiscard]] bool fused_form() const noexcept { return fused_; }
    // False when one object holds one expert: the same rows, but no bank-wide parent to view.
    [[nodiscard]] bool whole_bank() const noexcept { return whole_bank_; }
    [[nodiscard]] std::uint32_t count() const noexcept { return count_; }
    [[nodiscard]] std::uint64_t intermediate() const noexcept { return intermediate_; }
    // Per-expert rows of one bank. GateUp is the fused block width (2I), which no separate bank
    // holds; asking a separate description for that bank is refused by `bank()`.
    [[nodiscard]] std::uint64_t rows(ExpertBankRole role) const;

    // The bank-wide view, one entry per expert in expert order. A role the description does not
    // carry, or a bank whose parent does not hold every expert, is refused by parameter name.
    [[nodiscard]] const WeightBankView& bank(ExpertBankRole role) const;
    // The pointer-free layout of one bank; available even when the parent holds one expert.
    [[nodiscard]] const WeightBankLayout& layout(ExpertBankRole role) const;
    // The row block expert `expert` occupies in `role`.
    [[nodiscard]] WeightBankRegion block(ExpertBankRole role, std::uint32_t expert) const;
    [[nodiscard]] const std::string& parameter() const noexcept { return parameter_; }

private:
    bool fused_          = true;
    bool whole_bank_     = true;
    std::uint32_t count_ = 0;
    std::uint64_t intermediate_ = 0;
    std::uint64_t hidden_       = 0;
    // Where expert 0's block starts in each bank's parent, so a parent may carry a prefix.
    std::uint64_t gate_up_begin_ = 0;
    std::uint64_t gate_begin_    = 0;
    std::uint64_t up_begin_      = 0;
    std::uint64_t down_begin_    = 0;
    std::string parameter_;
    std::optional<WeightBankView> gate_up_, gate_, up_, down_;
};

struct DenseParameters {
    LinearParameters gate_up;
    LinearParameters down;
};

class ExpertStream;

// Qwen4Exp keeps the routed control rows, the shared expert and the three expert banks in their
// own parents instead of the fused matrices qwen3_5 carries. Every field is the artifact's own
// matrix, so nothing here is a re-slice of a neighbour's rows.
struct MoeSplitParameters {
    std::string parameter;               // the expert bank this layer describes, for refusals
    Weight router, shared_score;         // BF16 [E,H] and [1,H]
    LinearParameters shared_gate, shared_up, shared_down;
    ExpertBankDescription experts;
};

// port (sm_89): the bench routed-MoE view of the same artifact tensors. Experts form one affine
// bank per layer (experts/e/gate, experts/e/up interleaved), so the routed kernel reads them
// through a WeightBankView instead of one parent per expert.
struct RoutedMoeParameters {
    ops::MoeProfile profile;
    Weight control;            // BF16 [E+1,H]: router rows, then the shared logit
    ops::SwiGLUWeights shared; // fused gate_up [2I,H] plus down [H,I]
    ops::CudaSwiGLUExperts routed;
    // port (sm_89): non-null when the experts stream from the artifact instead of being resident.
    std::shared_ptr<ExpertStream> stream;
    // port (sm_89): keeps the split form's control rows alive. Qwen4Exp stores the router [E,H]
    // and the shared logit [1,H] in two parents, while the routing op reads one [E+1,H] matrix;
    // preparation copies the two into this device allocation and `control` points into it. Null on
    // the fused form, whose control rows are one contiguous region of the artifact.
    std::shared_ptr<void> control_rows;
    // Non-null when the artifact keeps those matrices apart (Qwen4Exp). The fused members above
    // then hold nothing, and a caller that needs a fused form asks the accessors below and is
    // refused by name instead of multiplying by an empty matrix.
    std::optional<MoeSplitParameters> split;

    [[nodiscard]] const Weight& fused_control() const;
    [[nodiscard]] const ops::SwiGLUWeights& fused_shared() const;
};

using FfnParameters =
    std::variant<DenseParameters, ops::SparseMoeWeights, RoutedMoeParameters>;

struct AttentionParameters {
    ops::ProjectionWeights projection;
    Tensor query_norm, key_norm;
    LinearParameters output;
};

struct GdnParameters {
    ops::ProjectionWeights projection;
    ops::ProjectionWeights control;
    Tensor a_log, dt_bias, convolution, norm;
    LinearParameters output;
};

struct BlockParameters {
    Tensor input_norm, post_attention_norm;
    std::variant<AttentionParameters, GdnParameters> mixer;
    FfnParameters ffn;
    ops::SparseMoeHints projection_prefetch;
};

struct TextParameters {
    Weight token_embedding;
    LinearParameters output_head;
    Tensor final_norm;
    std::vector<BlockParameters> layers;
};

struct MtpProjectionParameters {
    LinearParameters packed;
    // Dense MTP projects K/V and Q/gate independently in its incremental path.
    // MoE MTP uses its existing complete-parent Attention projection.
    std::optional<std::array<LinearParameters, 4>> rows;
};

struct MtpParameters {
    LinearParameters input_projection;
    Tensor embedding_norm, hidden_norm, input_norm, post_attention_norm, final_norm;
    MtpProjectionParameters projection;
    Tensor query_norm, key_norm;
    LinearParameters output;
    FfnParameters ffn;
    LinearParameters output_head;
};

struct NormParameters {
    Tensor weight, bias;
};

struct VisionBlockParameters {
    NormParameters norm1, norm2;
    LinearParameters qkv;
    Tensor qkv_bias;
    LinearParameters output, fc1, fc2;
    Tensor output_bias, fc1_bias, fc2_bias;
};

struct VisionParameters {
    LinearParameters patch_embedding;
    Tensor patch_embedding_bias, position_embedding;
    std::vector<VisionBlockParameters> layers;
    NormParameters merger_norm;
    LinearParameters merger_fc1, merger_fc2;
    Tensor merger_fc1_bias, merger_fc2_bias;
};

struct DynamicConvParameters {
    Tensor base_kernel;
    LinearParameters kernel_projection;
};

struct DraftBlockParameters {
    Tensor input_norm, post_attention_norm;
    LinearParameters query_key_value, context_key, context_value;
    Tensor query_norm, key_norm;
    LinearParameters output;
    DenseParameters mlp;
    std::optional<DynamicConvParameters> attention_conv, mlp_conv;
};

struct SelectorParameters {
    LinearParameters hidden_projection;
    Tensor predecessor_codebook, successor_codebook;
};

struct DraftParameters {
    LinearParameters feature_projection;
    Tensor context_norm, final_norm;
    std::vector<DraftBlockParameters> layers;
    std::optional<SelectorParameters> selector;
    LinearParameters output_head;
};

struct ProposalParameters {
    LinearParameters head;
    std::optional<Tensor> token_ids;
    std::uint32_t rows = 0;
};

// Cold native preparation for the fixed model implementation. This owner is stable before
// startup sizing, execution, or Graph capture; all weight addresses borrow the source Model.
// Shape-dependent kernel selection and scratch remain with the calling implementation and Op.
class Parameters {
public:
    explicit Parameters(const Model& source);
    Parameters(const Parameters&)            = delete;
    Parameters& operator=(const Parameters&) = delete;
    Parameters(Parameters&&)                 = delete;
    Parameters& operator=(Parameters&&)      = delete;

    const Model& model;
    TextParameters text;
    std::optional<MtpParameters> mtp;
    std::optional<VisionParameters> vision;
    std::optional<DraftParameters> draft;
    std::optional<ProposalParameters> proposal;
};

} // namespace ninfer::models::qwen3_5::execution
