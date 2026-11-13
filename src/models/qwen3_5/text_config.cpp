#include "models/qwen3_5/text_config.h"

#include "models/config_json.h"
#include "models/text_core.h"

#include <string>

namespace ninfer::models::qwen3_5 {

using artifact::ArtifactError;

TextConfig parse_text_config(Architecture architecture, const artifact::Json& value, bool mtp) {
    if (architecture != Architecture::Qwen3_5 && architecture != Architecture::Qwen3_5Moe) {
        throw ArtifactError("Qwen3.5 text parser was given " +
                            std::string(architecture_name(architecture)));
    }
    // Every key below describes Qwen4Exp's text block and none of them is declared by a Qwen3.5
    // artifact. Accepting one would run a Qwen4Exp payload through the dense/GDN/MoE mathematics
    // with its extra structure silently dropped.
    config_json::refuse_keys(value,
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
                              "vision_end_token_id"},
                             architecture_name(architecture));
    TextConfig out = text_core::read(value, architecture, mtp, {});
    if (out.indexed_attention_layers) {
        throw ArtifactError("layer_type 'indexed_attention' does not belong to " +
                            std::string(architecture_name(architecture)));
    }
    // Qwen3.5 declares no output-gate name; its linear attention runs the rectified gate this
    // engine has always used, which is the record's default.
    return out;
}

} // namespace ninfer::models::qwen3_5
