#include "models/qwen3_5/load/bindings.h"

#include "models/load_rules.h"

#include <algorithm>
#include <limits>
#include <set>
#include <string>

namespace ninfer::models::qwen3_5::loading {
namespace {

// The family's binding table, or a refusal: a streamed residual with no rules would bind names the
// artifact does not carry, and the failure would surface as a missing parameter three steps later.
const TextBindingRules& family_rules(const TextConfig& config) {
    const auto* rules = ::ninfer::models::text_binding_rules(config.architecture);
    if (rules == nullptr) {
        throw artifact::ArtifactError(
            std::string("no text binding rules for architecture ") +
            std::string(::ninfer::models::architecture_name(config.architecture)));
    }
    return *rules;
}

} // namespace

AttentionWeights bind_attention(Bindings& b, const TextConfig& config, const std::string& p,
                                bool indexed) {
    const auto& a = config.attention.value();
    const auto h  = config.hidden_size;
    const auto q  = a.query_width();
    const auto k  = a.key_width();
    AttentionWeights out;
    out.query = b.parameter(p + "attention/query", {q, h}, {p + "mixer_input"});
    out.key   = b.parameter(p + "attention/key", {k, h}, {p + "mixer_input"});
    // The output gate sits beside the query inside one fused parent under the family's own name.
    out.gate       = indexed ? b.aliased(p + "attention/gate", p + "attention/output_gate", {q, h},
                                         {p + "mixer_input"})
                             : b.parameter(p + "attention/gate", {q, h}, {p + "mixer_input"});
    out.value      = b.parameter(p + "attention/value", {k, h}, {p + "mixer_input"});
    out.query_norm = b.direct(p + "attention/query_norm", {a.head_dim});
    out.key_norm   = b.direct(p + "attention/key_norm", {a.head_dim});
    out.output     = b.parameter(p + "attention/output", {h, q}, {p + "attention/gated_output"});
    if (indexed) {
        const auto& x = config.indexer.value();
        IndexerWeights index;
        index.query =
            b.parameter(p + "attention/index_query", {x.query_width(), h}, {p + "mixer_input"});
        index.key = b.parameter(p + "attention/index_key", {x.key_width(), h}, {p + "mixer_input"});
        index.query_norm = b.direct(p + "attention/index_query_norm", {x.head_dim});
        index.key_norm   = b.direct(p + "attention/index_key_norm", {x.head_dim});
        out.indexer      = index;
    }
    return out;
}

DenseWeights bind_dense(Bindings& b, std::uint64_t h, std::uint64_t intermediate,
                        const std::string& p, bool draft) {
    const auto input = p + (draft ? "mlp_input" : "ffn_input");
    return {b.parameter(p + "mlp/gate", {intermediate, h}, {input}),
            b.parameter(p + "mlp/up", {intermediate, h}, {input}),
            b.parameter(p + "mlp/down", {h, intermediate},
                        {p + (draft ? "mlp_product" : "mlp/product")})};
}

namespace {

GdnWeights bind_gdn(Bindings& b, const TextConfig& text, const std::string& p) {
    const auto& g    = text.gdn.value();
    const auto h     = text.hidden_size;
    const auto k     = g.key_width();
    const auto v     = g.value_width();
    const auto heads = g.linear_num_value_heads;
    GdnWeights out;
    out.query        = b.parameter(p + "gdn/query", {k, h}, {p + "mixer_input"});
    out.key          = b.parameter(p + "gdn/key", {k, h}, {p + "mixer_input"});
    out.value        = b.parameter(p + "gdn/value", {v, h}, {p + "mixer_input"});
    out.z            = b.parameter(p + "gdn/z", {v, h}, {p + "mixer_input"});
    out.a_projection = b.parameter(p + "gdn/a_projection", {heads, h}, {p + "mixer_input"});
    out.b_projection = b.parameter(p + "gdn/b_projection", {heads, h}, {p + "mixer_input"});
    out.a_log        = b.direct(p + "gdn/a_log", {heads}, QType::FP32);
    out.dt_bias      = b.direct(p + "gdn/dt_bias", {heads}, QType::FP32);
    out.convolution =
        b.direct(p + "gdn/convolution", {g.linear_conv_kernel_dim, g.conv_channels()});
    out.norm   = b.direct(p + "gdn/norm", {g.linear_value_head_dim});
    out.output = b.parameter(p + "gdn/output", {h, v}, {p + "gdn/gated_output"});
    return out;
}

MoeWeights bind_moe(Bindings& b, const TextConfig& config, const std::string& prefix) {
    const auto& moe   = std::get<MoeConfig>(config.ffn);
    const auto p      = prefix + "moe/";
    const auto input  = prefix + "ffn_input";
    const auto h      = config.hidden_size;
    const auto ir     = moe.moe_intermediate_size;
    const auto shared = moe.shared_expert_intermediate_size;
    // A family that packs every expert of a bank into one binding states that in its registry
    // identity, so a per-expert parameter is a row block of that bank; the plan keeps the
    // per-expert shape the execution path runs with.
    const bool banked = ::ninfer::models::identity(config.architecture).banked_experts;
    MoeWeights out;
    out.router       = b.parameter(p + "router", {moe.num_experts, h}, {input});
    out.shared_score = b.parameter(p + "shared_score", {1, h}, {input});
    out.experts.reserve(moe.num_experts);
    // port (sm_89): when experts stream, they are declared without being read or uploaded.
    const auto residency =
        b.stream_experts ? (b.expert_host_copy ? artifact::Residency::Host
                                               : artifact::Residency::Streamed)
                         : artifact::Residency::Device;
    const auto product = banked ? p + "experts/product" : std::string{};
    for (std::uint32_t e = 0; e < moe.num_experts; ++e) {
        const auto ep = p + "experts/" + std::to_string(e) + "/";
        if (banked) {
            out.experts.push_back({b.banked(ep + "gate", p + "experts/gate", {ir, h},
                                            std::uint64_t(e) * ir, ir, {input}, residency),
                                   b.banked(ep + "up", p + "experts/up", {ir, h},
                                            std::uint64_t(e) * ir, ir, {input}, residency),
                                   b.banked(ep + "down", p + "experts/down", {h, ir},
                                            std::uint64_t(e) * h, h, {product}, residency)});
        } else {
            out.experts.push_back(
                {b.parameter(ep + "gate", {ir, h}, {input}, std::nullopt, residency),
                 b.parameter(ep + "up", {ir, h}, {input}, std::nullopt, residency),
                 b.parameter(ep + "down", {h, ir}, {ep + "product"}, std::nullopt, residency)});
        }
    }
    out.shared = {b.parameter(p + "shared/gate", {shared, h}, {input}),
                  b.parameter(p + "shared/up", {shared, h}, {input}),
                  b.parameter(p + "shared/down", {h, shared}, {p + "shared/product"})};
    return out;
}

} // namespace

BlockWeights bind_block(Bindings& b, const TextConfig& config, const std::string& p,
                        MixerKind mixer) {
    BlockWeights out;
    if (::ninfer::models::identity(config.architecture).aliased_activations) {
        // Such a family names the activation the mixer and the MoE each read: `attention_input`
        // for the mixer and `moe_input` for the FFN. The logical names stay the engine's; both spellings
        // resolve, and a Use that matches both or neither is refused under the logical parameter.
        b.alias_input(p + "mixer_input", p + "attention_input");
        b.alias_input(p + "ffn_input", p + "moe_input");
    }
    // A streamed residual folds its grouped norm into the correction that consumes it, so the two
    // norms and their corrections are the family's to name.
    if (config.hc) {
        const auto& rules = family_rules(config);
        if (rules.block_norms == nullptr || rules.block_corrections == nullptr) {
            throw artifact::ArtifactError(
                "the family's rules do not bind a streamed residual's norms");
        }
        rules.block_norms(b, config, p, out);
        if (attention_mixer(mixer)) {
            out.mixer = bind_attention(b, config, p, mixer == MixerKind::IndexedAttention);
        } else {
            out.mixer = bind_gdn(b, config, p);
        }
        rules.block_corrections(b, config, p, out);
    } else {
        out.input_norm          = b.direct(p + "input_norm", {config.hidden_size});
        out.post_attention_norm = b.direct(p + "post_attention_norm", {config.hidden_size});
        if (attention_mixer(mixer)) {
            out.mixer = bind_attention(b, config, p, mixer == MixerKind::IndexedAttention);
        } else {
            out.mixer = bind_gdn(b, config, p);
        }
    }
    if (const auto* dense = std::get_if<DenseConfig>(&config.ffn)) {
        out.ffn = bind_dense(b, config.hidden_size, dense->intermediate_size, p);
    } else {
        out.ffn = bind_moe(b, config, p);
    }
    return out;
}

TextWeights bind_text(Bindings& b, const TextConfig& config, const LoadOptions& options) {
    TextWeights out;
    out.token_embedding =
        b.parameter("text/token_embedding", {config.vocab_size, config.hidden_size});
    std::vector<std::string> head_inputs{"text/final_hidden"};
    if (options.speculative != SpeculativeBackend::None && !options.proposal_enabled()) {
        head_inputs.push_back(std::string(options.speculative_component()) + "/final_hidden");
    }
    out.output_head = b.parameter("text/output_head", {config.vocab_size, config.hidden_size},
                                  std::move(head_inputs));
    if (config.hc) {
        const auto& rules = family_rules(config);
        if (rules.final_correction == nullptr) {
            throw artifact::ArtifactError(
                "the family's rules do not bind a streamed residual's final correction");
        }
        rules.final_correction(b, config, out);
    } else {
        out.final_norm = b.direct("text/final_norm", {config.hidden_size});
    }
    out.layers.reserve(config.num_hidden_layers);
    for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
        const auto p = "text/layers/" + std::to_string(i) + "/";
        out.layers.push_back(bind_block(b, config, p, config.layer_types[i]));
        // A block the config names carries the n-gram embedding, in the family's own spelling.
        if (config.ple && std::find(config.ple->layer_ids.begin(), config.ple->layer_ids.end(),
                                    i) != config.ple->layer_ids.end()) {
            const auto& rules = family_rules(config);
            if (rules.block_ple == nullptr) {
                throw artifact::ArtifactError(
                    "the family's rules do not bind the n-gram embedding");
            }
            rules.block_ple(b, config, p, out.layers.back());
        }
    }
    return out;
}

ProposalWeights bind_proposal(Bindings& b, const artifact::Proposal& proposal,
                              const TextConfig& target, const LoadOptions& options,
                              std::uint32_t public_tokens) {
    const auto rows = proposal.indexed ? proposal.rows : target.vocab_size;
    if (rows > std::numeric_limits<std::uint32_t>::max() ||
        (proposal.indexed && rows > public_tokens)) {
        throw artifact::ArtifactError("proposal rows exceed the output domain");
    }
    ProposalWeights out;
    out.rows = static_cast<std::uint32_t>(rows);
    out.head = b.parameter("proposal/head", {rows, target.hidden_size},
                           {std::string(options.speculative_component()) + "/final_hidden"});
    if (proposal.indexed) {
        out.token_ids = b.direct("proposal/token_ids", {rows}, QType::INT32);
        out.global_token_ids =
            b.binder.values(b.at(*out.token_ids).reference.binding, QType::INT32).integers();
        std::set<std::int32_t> unique;
        for (const auto id : out.global_token_ids) {
            if (id < 0 || std::uint32_t(id) >= public_tokens || !unique.insert(id).second) {
                throw artifact::ArtifactError(
                    "proposal token IDs must be unique and in the public domain");
            }
        }
    }
    return out;
}

} // namespace ninfer::models::qwen3_5::loading
