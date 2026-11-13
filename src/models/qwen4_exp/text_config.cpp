#include "models/qwen4_exp/text_config.h"

#include "models/config_json.h"
#include "models/qwen3_5/text_config.h"
#include "models/qwen4_exp/config.h"
#include "models/text_core.h"

#include <string>

namespace ninfer::models::qwen4_exp {
namespace {

using artifact::ArtifactError;
using config_json::dimension;
using config_json::integer;
using qwen3_5::GdnGateKind;
using qwen3_5::HcConfig;
using qwen3_5::IndexerConfig;
using qwen3_5::MultimodalTokenIds;
using qwen3_5::PleConfig;

// The artifact names the activation its linear attention's output gate applies. This family's
// mathematics is the sigmoid gate, and its registry identity states that; an artifact that names
// another one belongs to a different architecture.
GdnGateKind output_gate(const std::string& declared) {
    const std::string_view family = identity().output_gate;
    if (!declared.empty() && declared != family) {
        throw ArtifactError("output_gate_type '" + declared + "' is not the '" +
                            std::string(family) + "' gate this architecture declares");
    }
    if (family != "sigmoid") {
        throw ArtifactError("unsupported output_gate_type '" + std::string(family) + "'");
    }
    return GdnGateKind::Sigmoid;
}

} // namespace

qwen3_5::TextConfig parse_text_config(Architecture architecture, const artifact::Json& value,
                                      bool mtp) {
    if (architecture != Architecture::Qwen4Exp) {
        throw ArtifactError("Qwen4Exp text parser was given " +
                            std::string(architecture_name(architecture)));
    }
    qwen3_5::TextConfig out = text_core::read(
        value, architecture, mtp,
        {"hc_count",
         "hc_lowrank",
         "indexer_n_heads",
         "indexer_kv_heads",
         "indexer_head_dim",
         "indexer_compress_ratio",
         "indexer_budget",
         "ple_embed_dim",
         "ngram_size",
         "heads_per_ngram",
         "split_ngram_parts",
         "ple_conv_kernel_size",
         "ple_conv_dilation",
         "ple_layer_ids",
         "ple_shard_shape",
         "output_gate_type",
         "image_token_id",
         "video_token_id",
         "vision_start_token_id",
         "vision_end_token_id"});

    out.gdn_gate = output_gate(config_json::name(value, "output_gate_type"));

    if (value.contains("hc_count") || value.contains("hc_lowrank")) {
        HcConfig hc{dimension(value, "hc_count"), dimension(value, "hc_lowrank")};
        (void)hc.stream_width(out.hidden_size);
        out.hc = hc;
    }
    // The residual is four streams: the embedding is expanded once, every block folds its grouped
    // normalized view to a single stream, and the correction splits the product back out. A config
    // without that structure is not this model's text block.
    if (!out.hc) {
        throw ArtifactError("hc_count/hc_lowrank are required (the residual is four streams)");
    }
    if (out.hc->count != 4) {
        throw ArtifactError("hc_count must be four, got " + std::to_string(out.hc->count));
    }

    if (value.contains("indexer_n_heads") || value.contains("indexer_kv_heads") ||
        value.contains("indexer_head_dim") || value.contains("indexer_compress_ratio") ||
        value.contains("indexer_budget")) {
        IndexerConfig indexer{
            dimension(value, "indexer_n_heads"), dimension(value, "indexer_kv_heads"),
            dimension(value, "indexer_head_dim"), dimension(value, "indexer_compress_ratio"),
            dimension(value, "indexer_budget")};
        if (indexer.num_heads % indexer.num_kv_heads) {
            throw ArtifactError("indexer heads must be divisible by KV heads");
        }
        if (indexer.compress_ratio == 0 || indexer.budget % indexer.compress_ratio) {
            throw ArtifactError("indexer budget must be a positive multiple of the compress ratio");
        }
        if (indexer.selection_blocks() == 0) {
            throw ArtifactError("indexer budget is smaller than one pooled block");
        }
        out.indexer = indexer;
    }
    if (out.indexed_attention_layers && !out.indexer) {
        throw ArtifactError("indexed layer_type requires the indexer geometry");
    }

    if (value.contains("ple_embed_dim") || value.contains("ple_shard_shape")) {
        PleConfig ple{
            dimension(value, "ple_embed_dim"),        dimension(value, "ngram_size"),
            dimension(value, "heads_per_ngram"),      dimension(value, "split_ngram_parts"),
            dimension(value, "ple_conv_kernel_size"), dimension(value, "ple_conv_dilation")};
        const auto& ids = value.at("ple_layer_ids");
        if (!ids.is_array() || ids.empty()) {
            throw ArtifactError("ple_layer_ids must name at least one text block");
        }
        for (const auto& id : ids) {
            const auto layer = integer(id, "ple layer index", false);
            if (layer >= out.num_hidden_layers) {
                throw ArtifactError("ple_layer_ids index exceeds the text blocks");
            }
            ple.layer_ids.push_back(layer);
        }
        const auto& shard = value.at("ple_shard_shape");
        if (!shard.is_array() || shard.size() != 2) {
            throw ArtifactError("ple_shard_shape must hold rows and width");
        }
        ple.shard_shape[0] = integer(shard[0], "ple shard rows");
        ple.shard_shape[1] = integer(shard[1], "ple shard width");
        (void)ple.table_rows();
        out.ple = ple;
    }

    if (value.contains("image_token_id")) {
        out.multimodal_token_ids =
            MultimodalTokenIds{dimension(value, "image_token_id"),
                               dimension(value, "video_token_id"),
                               dimension(value, "vision_start_token_id"),
                               dimension(value, "vision_end_token_id")};
    }
    return out;
}

} // namespace ninfer::models::qwen4_exp
