#include "models/qwen3_5/load/bindings.h"

#include "models/load_rules.h"

namespace ninfer::models::qwen3_5::loading {
namespace {

// The width a norm covers. A streamed residual normalizes every stream group at once, so its norms
// are `count * hidden` wide; the single-stream form is the hidden width.
std::uint64_t norm_width(const TextConfig& config) {
    return config.hc ? config.hc->stream_width(config.hidden_size)
                     : std::uint64_t(config.hidden_size);
}

} // namespace

MtpWeights bind_mtp(Bindings& b, const TextConfig& config, const TextWeights& target) {
    const auto h = config.hidden_size;
    MtpWeights out;
    // The artifact states which stem it carries: the split pair keeps `fc_embedding` and `fc_hidden`
    // apart (the fused [h,2h] geometry has no registered kernel route at h=2560), while the
    // single-matrix form ships one projection.
    if (b.artifact_has("mtp/fc_embedding") && b.artifact_has("mtp/fc_hidden")) {
        out.fc_embedding = b.parameter("mtp/fc_embedding", {h, h}, {"mtp/normalized_embedding"});
        out.fc_hidden    = b.parameter("mtp/fc_hidden", {h, h}, {"mtp/normalized_hidden"});
    } else {
        out.input_projection =
            b.parameter("mtp/input_projection", {h, 2ULL * h}, {"mtp/stem_input"});
    }
    out.embedding_norm   = b.direct("mtp/embedding_norm", {h});
    out.hidden_norm      = b.direct("mtp/hidden_norm", {norm_width(config)});
    if (config.hc) {
        // The draft head's norm lives inside its final correction: the family's own spelling.
        const auto* rules = ::ninfer::models::text_binding_rules(config.architecture);
        if (rules == nullptr || rules->mtp_norm == nullptr ||
            rules->mtp_final_correction == nullptr) {
            throw artifact::ArtifactError(
                "the family's rules do not bind the draft head's final correction");
        }
        rules->mtp_norm(b, config, out);
        // The draft layer is an indexed (QSA) block in the source artifact. The engine binds its
        // four indexer matrices and the output-gate alias so the artifact is read faithfully, but
        // the draft mixer runs DENSE full attention: QSA block selection is not wired on this
        // layer, which is a declared capability downgrade (docs/weight-conversion.md), not a
        // silent one.
        const bool indexed = config.indexer.has_value();
        out.layer          = bind_block(b, config, "mtp/layers/0/",
                                        indexed ? MixerKind::IndexedAttention
                                                : MixerKind::FullAttention);
        rules->mtp_final_correction(b, config, out);
    } else {
        out.final_norm = b.direct("mtp/final_norm", {h});
        out.layer      = bind_block(b, config, "mtp/layers/0/", MixerKind::FullAttention);
    }
    out.token_embedding = target.token_embedding;
    out.output_head     = target.output_head;
    return out;
}

} // namespace ninfer::models::qwen3_5::loading
