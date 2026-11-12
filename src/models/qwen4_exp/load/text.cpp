// This family's binding vocabulary: the names and shapes its artifacts use where the shared text
// binder has its own. A Qwen3.5 artifact never reaches any of these; the registry identity states
// that the residual is four streams, and that is what makes the shared binder ask for them.
#include "models/qwen4_exp/load/text.h"

#include <algorithm>
#include <cstdint>

namespace ninfer::models::qwen4_exp::loading {
namespace {

using qwen3_5::BlockWeights;
using qwen3_5::TextConfig;
using qwen3_5::TextWeights;
using qwen3_5::loading::Bindings;

// The residual is four streams, so its norms cover every stream group: the artifact's `hnorm`
// reduction spans `count * hidden` entries at once, not one reduction per stream.
std::uint64_t stream_norm_width(const TextConfig& config) {
    return config.hc ? config.hc->stream_width(config.hidden_size)
                     : std::uint64_t(config.hidden_size);
}

// The block's two norms. They are not standalone tensors in this family's artifacts: each sits
// inside the correction that consumes it, so the logical name is bound through that spelling.
void bind_block_norms(Bindings& b, const TextConfig& config, const std::string& prefix,
                      BlockWeights& out) {
    const auto width = stream_norm_width(config);
    out.input_norm =
        b.aliased_direct(prefix + "input_norm", prefix + "attention_hc/norm", {width});
    out.post_attention_norm =
        b.aliased_direct(prefix + "post_attention_norm", prefix + "moe_hc/norm", {width});
}

void bind_block_corrections(Bindings& b, const TextConfig& config, const std::string& prefix,
                            BlockWeights& out) {
    out.input_hc          = bind_hc(b, config, prefix + "attention_hc/", true);
    out.post_attention_hc = bind_hc(b, config, prefix + "moe_hc/", true);
}

// The head's fold: the artifact keeps `text/final_hc/norm` inside the correction, and the correction
// itself has no injection stream because it feeds the head rather than writing the residual back.
void bind_final_correction(Bindings& b, const TextConfig& config, TextWeights& out) {
    out.final_norm = b.aliased_direct("text/final_norm", "text/final_hc/norm",
                                      {stream_norm_width(config)});
    out.final_hc   = bind_hc(b, config, "text/final_hc/", false);
}

// The n-gram embedding. Its table is paged from disk and never resident, so its binding needs no
// device bytes; the two row blocks of one parent are the hash keys and the value.
void bind_block_ple(Bindings& b, const TextConfig& config, const std::string& prefix,
                    BlockWeights& out) {
    const auto& ple    = config.ple.value();
    const auto p       = prefix + "ple/";
    const auto streams = std::uint64_t(config.hc ? config.hc->count : 1) * ple.embed_dim;
    qwen3_5::PleWeights weights;
    weights.key         = b.parameter(p + "key", {streams, ple.embed_dim}, {p + "embedding"});
    weights.value       = b.parameter(p + "value", {ple.embed_dim, ple.embed_dim}, {p + "embedding"});
    weights.norm_key    = b.direct(p + "norm_key", {streams});
    weights.norm_query  = b.direct(p + "norm_query", {streams});
    weights.norm_conv   = b.direct(p + "norm_conv", {streams});
    weights.convolution = b.direct(p + "convolution", {ple.conv_kernel_size, streams});
    weights.table       = b.parameter(p + "table", {ple.table_rows(), ple.table_width()}, {},
                                      std::nullopt, artifact::Residency::Streamed);
    out.ple             = weights;
}

// The draft head's own fold. Both stem terms are the artifact's, so the norm widths follow the
// streamed residual the same way the text blocks' do.
void bind_mtp_norm(Bindings& b, const TextConfig& config, qwen3_5::MtpWeights& out) {
    out.final_norm = b.aliased_direct("mtp/final_norm", "mtp/final_hc/norm",
                                      {stream_norm_width(config)});
}

void bind_mtp_correction(Bindings& b, const TextConfig& config, qwen3_5::MtpWeights& out) {
    out.final_hc = bind_hc(b, config, "mtp/final_hc/", false);
}

} // namespace

qwen3_5::HcWeights bind_hc(Bindings& b, const TextConfig& config, const std::string& prefix,
                           bool inject) {
    const auto& hc   = config.hc.value();
    const auto width = hc.stream_width(config.hidden_size);
    qwen3_5::HcWeights out;
    out.down = b.parameter(prefix + "down", {hc.lowrank, width}, {prefix + "normalized_streams"});
    out.up   = b.parameter(prefix + "up", {width, hc.lowrank}, {prefix + "lowrank_activation"});
    if (inject) {
        out.inject = b.parameter(prefix + "inject", {hc.count, width},
                                 {prefix + "normalized_streams"});
    }
    return out;
}

const TextBindingRules& binding_rules() noexcept {
    static const TextBindingRules rules{
        .block_norms            = &bind_block_norms,
        .block_corrections      = &bind_block_corrections,
        .final_correction       = &bind_final_correction,
        .block_ple              = &bind_block_ple,
        .mtp_norm               = &bind_mtp_norm,
        .mtp_final_correction   = &bind_mtp_correction,
    };
    return rules;
}

} // namespace ninfer::models::qwen4_exp::loading
