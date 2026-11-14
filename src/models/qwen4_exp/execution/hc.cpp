// The four-stream residual: expand the embedding once, fold each block's normalized view to a
// single stream, mix through the rank-`lowrank` pair and inject the block's product back into the
// four streams. This is the family's own execution path - a Qwen3.5 artifact has one residual
// stream and never enters it - so it lives beside the family's identity rather than in the shared
// text engine. The declarations stay on `TextContext` because every pass publishes these roots;
// only the bodies are the family's.
#include "models/qwen3_5/execution/text.h"

#include "models/qwen3_5/execution/diagnostics.h"
#include "models/qwen3_5/execution/workspace.h"
#include "ninfer/ops/gated_residual.h"
#include "ninfer/ops/rmsnorm.h"

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_5::execution {

void TextContext::hc_stem(Tensor& embedding, std::int32_t tokens) {
    const auto& hc = config_.hc.value();
    // The stream width the artifact states, so a plan built from a different one is refused here
    // rather than by a silent shape mismatch three ops later.
    if (hc.count != 4 || hc.stream_width(config_.hidden_size) != 4 * config_.hidden_size) {
        throw std::invalid_argument("Qwen4Exp hyper-connection requires four residual streams of "
                                    "the hidden width");
    }
    const auto roots = workspace::text_hc_roots(work_, config_, tokens);
    hc_              = roots;
    const auto width = dimension(config_.hidden_size), streams = dimension(hc.count);
    Tensor weight    = parameters_.text.layers.front().input_norm.view({width, streams});
    Tensor residual  = hc_.residual.view({width, streams, tokens});
    Tensor normed    = hc_.normalized.view({width, streams, tokens});
    // The residual starts as the embedding, stream-replicated and stream-normed exactly once.
    ops::hc_expand_norm(embedding, weight, config_.rms_norm_eps, residual, normed,
                        ctx_.execution_view());
}

void TextContext::hc_mix_into(const HcParameters& weights, std::int32_t tokens) {
    const auto normalized = hc_.normalized.view({dimension(config_.hidden_size),
                                                 dimension(config_.hc->count), tokens});
    // Produces the block's single-stream input and the four coefficients its correction applies.
    nan_probe_tensor("hc-mix-pre", hc_.input, 300, ctx_.stream);
    nan_probe_tensor("hc-normed", hc_.normalized, 400, ctx_.stream);
    ops::hc_mix(normalized, weights.block_form(), work_, hc_.input, hc_.injection,
                ctx_.execution_view());
    nan_probe_tensor("hc-mix-out", hc_.input, 200, ctx_.stream);
}

void TextContext::hc_combine(const Tensor& product, const Tensor& norm_weight,
                             std::int32_t tokens) {
    const auto width   = dimension(config_.hidden_size);
    const auto streams = dimension(config_.hc->count);
    Tensor residual    = hc_.residual.view({width, streams, tokens});
    Tensor normed      = hc_.normalized.view({width, streams, tokens});
    const auto weight  = norm_weight.view({width, streams});
    // residual += injection * product, then the grouped norm of the result for the next block.
    // The residual aliases itself in place, which the Op admits exactly.
    ops::hc_combine_norm(residual, product, hc_.injection, weight, config_.rms_norm_eps, residual,
                         normed, ctx_.execution_view());
}

void TextContext::hc_tap_publish(std::size_t layer, std::int32_t tokens) {
    const auto width   = dimension(config_.hidden_size);
    const auto streams = dimension(config_.hc->count);
    // What a one-stream target hands its drafter is the block's raw residual, not the normalized
    // view its own attention consumes, and not the same view taken before the FFN add. The
    // four-stream family has no single residual, so the fold stands in for it: a block's rank pair
    // over the post-FFN residual, unmixed with any norm. Measured on v5_e2, folding the
    // normalized view instead costs the drafter's first-token agreement 43.5 of 100 rounds against
    // this view's 69.2 (k=4), and the pre-FFN view lands at 52.3.
    //
    // Which of the block's two pairs folds is not decided by the model: the pair names which
    // branch's input the fold is for, and the tap is neither branch. NINFER_DFLASH2_TAP_PAIR picks
    // between the block's own post-attention pair (default), its attention pair, and the next
    // block's attention pair, so one artifact can A/B them.
    const auto& block = parameters_.text.layers[layer];
    const HcParameters* tap_weights = &*block.post_attention_hc;
    if (const char* pair = std::getenv("NINFER_DFLASH2_TAP_PAIR"); pair != nullptr) {
        const std::string named(pair);
        if (named == "input") {
            tap_weights = &*block.input_hc;
        } else if (named == "next" && layer + 1 < parameters_.text.layers.size()) {
            tap_weights = &*parameters_.text.layers[layer + 1].input_hc;
        }
    }
    ops::hc_mix(hc_.residual.view({width, streams, tokens}), tap_weights->block_form(), work_,
                hc_.input, hc_.injection, ctx_.execution_view());
}

void TextContext::final_hidden(Tensor& x, Tensor& out, std::int32_t tokens) {
    if (!hc_enabled()) {
        ops::rmsnorm(x, *final_norm_, config_.rms_norm_eps, true, out, ctx_.stream);
        return;
    }
    const auto& weights = *final_hc_;
    ops::hc_mix(hc_.normalized.view({dimension(config_.hidden_size),
                                     dimension(config_.hc->count), tokens}),
                weights.final_form(), work_, out, ctx_.execution_view());
}

} // namespace ninfer::models::qwen3_5::execution
