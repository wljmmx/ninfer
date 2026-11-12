#include "models/qwen3_5/load/bindings.h"

namespace ninfer::models::qwen3_5::loading {
namespace {

// Qwen4Exp normalizes the HC-expanded residual, so its stream norms cover every HC stream: the
// pre-FC hidden norm is 10240 wide, and the head's final correction is the block-correction pair
// without an injection stream. The single-stream form keeps the Qwen3.5 widths untouched.
std::uint64_t norm_width(const TextConfig& config) {
    return config.hc ? config.hc->stream_width(config.hidden_size)
                     : std::uint64_t(config.hidden_size);
}

} // namespace

MtpWeights bind_mtp(Bindings& b, const TextConfig& config, const TextWeights& target) {
    const auto h = config.hidden_size;
    MtpWeights out;
    // The artifact states which stem it carries: Qwen4Exp keeps the HF pair apart (and the fused
    // [h,2h] geometry has no registered kernel route at h=2560), the Qwen3.5 form ships one matrix.
    if (b.artifact_has("mtp/fc_embedding") && b.artifact_has("mtp/fc_hidden")) {
        out.fc_embedding = b.parameter("mtp/fc_embedding", {h, h}, {"mtp/normalized_embedding"});
        out.fc_hidden    = b.parameter("mtp/fc_hidden", {h, h}, {"mtp/normalized_hidden"});
    } else {
        out.input_projection =
            b.parameter("mtp/input_projection", {h, 2ULL * h}, {"mtp/stem_input"});
    }
    out.embedding_norm   = b.direct("mtp/embedding_norm", {h});
    out.hidden_norm      = b.direct("mtp/hidden_norm", {norm_width(config)});
    out.final_norm =
        config.hc ? b.aliased_direct("mtp/final_norm", "mtp/final_hc/norm", {norm_width(config)})
                  : b.direct("mtp/final_norm", {h});
    // Qwen4Exp's draft layer is an indexed (QSA) block in the source. The engine binds its four
    // indexer matrices and the output-gate alias so the artifact is read faithfully, but the draft
    // mixer runs DENSE full attention: QSA block selection is not wired on this layer, which is a
    // declared capability downgrade (docs/weight-conversion.md), not a silent one.
    const bool indexed = config.hc.has_value() && config.indexer.has_value();
    out.layer          = bind_block(b, config, "mtp/layers/0/",
                                    indexed ? MixerKind::IndexedAttention : MixerKind::FullAttention);
    if (config.hc) { out.final_hc = bind_hc(b, config, "mtp/final_hc/", false); }
    out.token_embedding = target.token_embedding;
    out.output_head     = target.output_head;
    return out;
}

} // namespace ninfer::models::qwen3_5::loading
