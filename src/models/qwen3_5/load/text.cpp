#include "models/qwen3_5/load/bindings.h"

#include "models/qwen4_exp/ple/table.h"

#include <algorithm>
#include <limits>
#include <set>
#include <string>

namespace ninfer::models::qwen3_5::loading {
namespace {

// Qwen4Exp normalizes the HC-expanded residual, so its stream norms cover every HC stream.
std::uint64_t norm_width(const TextConfig& config) {
    return config.hc ? config.hc->stream_width(config.hidden_size)
                     : std::uint64_t(config.hidden_size);
}

// The n-gram table is paged from disk and never resident, so its binding needs no device bytes.
PleWeights bind_ple(Bindings& b, const TextConfig& config, const std::string& p) {
    const auto& ple    = config.ple.value();
    const auto streams = std::uint64_t(config.hc ? config.hc->count : 1) * ple.embed_dim;
    PleWeights out;
    out.key         = b.parameter(p + "key", {streams, ple.embed_dim}, {p + "embedding"});
    out.value       = b.parameter(p + "value", {ple.embed_dim, ple.embed_dim}, {p + "embedding"});
    out.norm_key    = b.direct(p + "norm_key", {streams});
    out.norm_query  = b.direct(p + "norm_query", {streams});
    out.norm_conv   = b.direct(p + "norm_conv", {streams});
    out.convolution = b.direct(p + "convolution", {ple.conv_kernel_size, streams});
    out.table = b.parameter(p + "table", {ple.table_rows(), ple.table_width()}, {}, std::nullopt,
                            artifact::Residency::Streamed);
    return out;
}

} // namespace

// The low-rank correction around one grouped norm. `inject` writes the streams back, which the
// final correction does without: it feeds the head instead.
HcWeights bind_hc(Bindings& b, const TextConfig& config, const std::string& p, bool inject) {
    const auto& hc   = config.hc.value();
    const auto width = hc.stream_width(config.hidden_size);
    HcWeights out;
    out.down = b.parameter(p + "down", {hc.lowrank, width}, {p + "normalized_streams"});
    out.up   = b.parameter(p + "up", {width, hc.lowrank}, {p + "lowrank_activation"});
    if (inject) {
        out.inject = b.parameter(p + "inject", {hc.count, width}, {p + "normalized_streams"});
    }
    return out;
}

AttentionWeights bind_attention(Bindings& b, const TextConfig& config, const std::string& p,
                                bool indexed) {
    const auto& a = config.attention.value();
    const auto h  = config.hidden_size;
    const auto q  = a.query_width();
    const auto k  = a.key_width();
    AttentionWeights out;
    out.query = b.parameter(p + "attention/query", {q, h}, {p + "mixer_input"});
    out.key   = b.parameter(p + "attention/key", {k, h}, {p + "mixer_input"});
    // Qwen4Exp keeps the output gate beside the query inside one fused parent and names it apart.
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
    // Qwen4Exp packs every expert of a bank into one binding, so a per-expert parameter is a row
    // block of that bank; the plan keeps the per-expert shape the execution path runs with.
    const bool banked = config.architecture == Architecture::Qwen4Exp;
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
    if (config.architecture == Architecture::Qwen4Exp) {
        // Qwen4Exp names the activation the mixer and the MoE each read: `attention_input` for the
        // mixer and `moe_input` for the FFN. The logical names stay the engine's; both spellings
        // resolve, and a Use that matches both or neither is refused under the logical parameter.
        b.alias_input(p + "mixer_input", p + "attention_input");
        b.alias_input(p + "ffn_input", p + "moe_input");
    }
    // Qwen4Exp folds the grouped stream norm into the HC binding that consumes it.
    if (config.hc) {
        out.input_norm =
            b.aliased_direct(p + "input_norm", p + "attention_hc/norm", {norm_width(config)});
        out.post_attention_norm =
            b.aliased_direct(p + "post_attention_norm", p + "moe_hc/norm", {norm_width(config)});
    } else {
        out.input_norm          = b.direct(p + "input_norm", {config.hidden_size});
        out.post_attention_norm = b.direct(p + "post_attention_norm", {config.hidden_size});
    }
    if (attention_mixer(mixer)) {
        out.mixer = bind_attention(b, config, p, mixer == MixerKind::IndexedAttention);
    } else {
        out.mixer = bind_gdn(b, config, p);
    }
    if (config.hc) {
        out.input_hc          = bind_hc(b, config, p + "attention_hc/", true);
        out.post_attention_hc = bind_hc(b, config, p + "moe_hc/", true);
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
    out.final_norm =
        config.hc ? b.aliased_direct("text/final_norm", "text/final_hc/norm", {norm_width(config)})
                  : b.direct("text/final_norm", {config.hidden_size});
    if (config.hc) { out.final_hc = bind_hc(b, config, "text/final_hc/", false); }
    out.layers.reserve(config.num_hidden_layers);
    for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
        const auto p = "text/layers/" + std::to_string(i) + "/";
        out.layers.push_back(bind_block(b, config, p, config.layer_types[i]));
        // Qwen4Exp embeds n-grams on named blocks only.
        if (config.ple && std::find(config.ple->layer_ids.begin(), config.ple->layer_ids.end(),
                                    i) != config.ple->layer_ids.end()) {
            out.layers.back().ple = bind_ple(b, config, p + "ple/");
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

std::shared_ptr<const qwen4_exp::PleTableSource>
open_ple_table(const artifact::Reader& reader, artifact::Binder& binder,
               std::span<const PendingWeight> pending, const TextWeights& weights,
               const TextConfig& config) {
    if (!config.ple) throw artifact::ArtifactError("PLE table requested for a model without PLE");
    const auto& ple = *config.ple;
    if (ple.layer_ids.empty()) {
        throw artifact::ArtifactError("PLE declares no layer ids");
    }
    // Every PLE block gathers from the same n-gram bank, so its bound object ranges must be the
    // same object set; two different tables would otherwise be read as one table's row space.
    std::vector<artifact::ObjectHandle> shards;
    const auto collect = [&](const PleWeights& w) {
        if (std::size_t(w.table.index) >= pending.size())
            throw artifact::ArtifactError("PLE table binding is not in the load plan");
        const auto& reference = pending[std::size_t(w.table.index)].reference;
        if (reference.binding.parts.empty())
            throw artifact::ArtifactError(reference.name + ": PLE table has no object parts");
        for (const auto& part : reference.binding.parts) {
            if (std::find(shards.begin(), shards.end(), part.object) == shards.end())
                shards.push_back(part.object);
        }
    };
    collect(weights.layers.at(ple.layer_ids.front()).ple.value());
    const auto first = shards;
    for (const auto layer : ple.layer_ids) {
        shards.clear();
        collect(weights.layers.at(layer).ple.value());
        if (shards != first) {
            throw artifact::ArtifactError(
                "PLE layer " + std::to_string(layer) +
                " reads a different table object set than layer " +
                std::to_string(ple.layer_ids.front()) +
                "; one n-gram stage cannot gather from two banks");
        }
    }
    shards = first;
    // The controls are a resource of the text component; the hash cannot be evaluated without
    // them, so a PLE artifact that omits them fails here, naming the resource.
    const auto controls = binder.resource("text", "ple_hash_controls");
    return std::make_shared<qwen4_exp::PleTableSource>(reader, shards, controls);
}

std::int32_t ple_eos_token(const artifact::Directory& directory) {
    const auto& config = directory.component("text").config;
    const auto found   = config.find("eos_token_id");
    if (found == config.end() || !found->is_number_integer()) return -1;
    const auto id = found->get<std::int64_t>();
    if (id < 0 || id > std::numeric_limits<std::int32_t>::max()) return -1;
    return static_cast<std::int32_t>(id);
}

} // namespace ninfer::models::qwen3_5::loading
