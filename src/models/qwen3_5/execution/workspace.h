#pragma once

// Phase allocations shared by actual execution and startup sizing.

#include "core/arena.h"
#include "core/layout.h"
#include "models/qwen3_5/execution/parameters.h"

#include <cstdint>

namespace ninfer::models::qwen3_5::execution::workspace {

template <class Allocator>
Tensor matrix(Allocator& allocator, DType dtype, std::int32_t rows, std::int32_t tokens) {
    return allocator.alloc(dtype, {rows, tokens});
}

template <class Allocator>
Tensor vector(Allocator& allocator, DType dtype, std::int32_t elements) {
    return allocator.alloc(dtype, {elements});
}

struct TextPrefillRoots {
    Tensor ids;
    Tensor positions;
    Tensor rope_positions;
    Tensor residual;
    Tensor scatter_indices;
};

template <class Allocator>
TextPrefillRoots text_prefill_roots(Allocator& allocator, const TextConfig& config,
                                    std::int32_t tokens, std::int32_t rope_axes,
                                    std::int32_t scatter_tokens) {
    TextPrefillRoots out;
    out.ids       = vector(allocator, DType::I32, tokens);
    out.positions = vector(allocator, DType::I32, tokens);
    if (rope_axes != 0) { out.rope_positions = matrix(allocator, DType::I32, tokens, rope_axes); }
    out.residual = matrix(allocator, DType::BF16, dimension(config.hidden_size), tokens);
    if (scatter_tokens != 0) {
        out.scatter_indices = vector(allocator, DType::I32, scatter_tokens);
    }
    return out;
}

template <class Allocator>
Tensor visual_scatter_indices(Allocator& allocator, std::int32_t tokens) {
    return vector(allocator, DType::I32, tokens);
}

struct TextAttentionProjectionRoots {
    Tensor hidden;
    Tensor query;
    Tensor gate;
    Tensor key;
    Tensor value;
};

template <class Allocator>
TextAttentionProjectionRoots
text_attention_projection(Allocator& allocator, const TextConfig& config, std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, dimension(config.hidden_size), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention->query_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention->query_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention->key_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention->key_width()), tokens),
    };
}

struct TextAttentionResultRoots {
    Tensor normalized_query;
    Tensor normalized_key;
    Tensor attention;
};

template <class Allocator>
TextAttentionResultRoots text_attention_results(Allocator& allocator, const TextConfig& config,
                                                std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, dimension(config.attention->query_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention->key_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention->query_width()), tokens),
    };
}

struct QsaPassBindings {
    // Live columns per row, the record column base of the call's first token, and the record
    // offset prepare writes from. All three are device metadata of the current pass: the first is
    // the row's live prefix, the other two are `positions[0,b]` and zero.
    Tensor columns;
    Tensor base_positions;
    Tensor offsets;
};

template <class Allocator>
QsaPassBindings qsa_pass_bindings(Allocator& allocator) {
    return {vector(allocator, DType::I32, 8), vector(allocator, DType::I32, 8),
            vector(allocator, DType::I32, 8)};
}

// One indexed layer's transient side. The projections take the same shape they do in the dense
// mixer (`[row_width, tokens]`, tokens = T*B), and `query`/`index_query` are normalized and rotated
// in place by prepare, which the Op admits exactly.
struct QsaRoots {
    Tensor index_query;    // BF16 [4*128, tokens], prepared in place
    Tensor index_key;      // BF16 [128, tokens], raw projection words
    Tensor attention;      // BF16 [query_width, tokens], post-gate attention output
    Tensor block_ids;      // I32 [512, tokens]
    Tensor block_counts;   // I32 [tokens]
    Tensor record_keys;    // BF16 [128, tokens]
    Tensor record_rope;    // I32 [3, tokens]
    // Diagnosis only (NINFER_QSA_DENSE_READ): the read-only dense consumer normalizes its query
    // into its own buffer, which the sparse path never needs.
    Tensor normalized_query;
};

template <class Allocator>
QsaRoots qsa_roots(Allocator& allocator, const TextConfig& config, std::int32_t tokens) {
    const auto& indexer = config.indexer.value();
    const auto query_width = dimension(config.attention->query_width());
    QsaRoots out;
    out.index_query = matrix(allocator, DType::BF16, dimension(indexer.query_width()), tokens);
    out.index_key   = matrix(allocator, DType::BF16, dimension(indexer.key_width()), tokens);
    out.attention   = matrix(allocator, DType::BF16, query_width, tokens);
    out.block_ids   = matrix(allocator, DType::I32, 512, tokens);
    out.block_counts = vector(allocator, DType::I32, tokens);
    out.record_keys  = matrix(allocator, DType::BF16, dimension(indexer.head_dim), tokens);
    out.record_rope  = matrix(allocator, DType::I32, 3, tokens);
    return out;
}

template <class Allocator>
Tensor qsa_dense_query(Allocator& allocator, const TextConfig& config, std::int32_t tokens) {
    return matrix(allocator, DType::BF16, dimension(config.attention->query_width()), tokens);
}

struct GdnControlRoots {
    Tensor hidden;
    Tensor g;
    Tensor beta;
};

template <class Allocator>
GdnControlRoots gdn_control(Allocator& allocator, const TextConfig& config, std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, dimension(config.hidden_size), tokens),
        matrix(allocator, DType::FP32, dimension(config.gdn->linear_num_value_heads), tokens),
        matrix(allocator, DType::FP32, dimension(config.gdn->linear_num_value_heads), tokens),
    };
}

struct GdnProjectionRoots {
    Tensor output_gate;
    Tensor query;
    Tensor key;
    Tensor value;
};

template <class Allocator>
GdnProjectionRoots gdn_projection(Allocator& allocator, const TextConfig& config,
                                  std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, dimension(config.gdn->value_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.gdn->key_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.gdn->key_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.gdn->value_width()), tokens),
    };
}

template <class Allocator>
Tensor gdn_prefill_conv(Allocator& allocator, const TextConfig& config, std::int32_t tokens) {
    return matrix(allocator, DType::BF16, dimension(config.gdn->conv_channels()), tokens);
}

template <class Allocator>
Tensor gdn_recurrent_output(Allocator& allocator, const TextConfig& config, std::int32_t tokens) {
    return matrix(allocator, DType::BF16, dimension(config.gdn->value_width()), tokens);
}

template <class Allocator>
Tensor gdn_normalized_output(Allocator& allocator, const TextConfig& config, std::int32_t tokens) {
    return matrix(allocator, DType::BF16, dimension(config.gdn->value_width()), tokens);
}

template <class Allocator>
Tensor post_mixer_hidden(Allocator& allocator, const TextConfig& config, std::int32_t tokens) {
    return matrix(allocator, DType::BF16, dimension(config.hidden_size), tokens);
}

// Qwen4Exp's four-stream residual. The expanded streams and their grouped normalized view stay live
// across every layer, so they are roots of the call rather than a stage: `residual` is the stream
// the corrections accumulate into, `normalized` is what the next mixer reads, and `input`/`output`
// are the single-stream pair a mixer or the FFN consumes and produces. `injection` is the four
// coefficients the pending correction applies to that product. Every tensor is a view of the same
// [hidden*streams, tokens] / [hidden, tokens] pair the dense form uses, so a plan built from this
// recipe and the execution that calls it reserve identical bytes in the same order.
struct TextHcRoots {
    Tensor residual;
    Tensor normalized;
    Tensor input;
    Tensor injection;
    Tensor output;
};

template <class Allocator>
TextHcRoots text_hc_roots(Allocator& allocator, const TextConfig& config, std::int32_t tokens) {
    const auto& hc   = config.hc.value();
    const auto width = dimension(hc.stream_width(config.hidden_size));
    return {matrix(allocator, DType::BF16, width, tokens),
            matrix(allocator, DType::BF16, width, tokens),
            matrix(allocator, DType::BF16, dimension(config.hidden_size), tokens),
            matrix(allocator, DType::FP32, dimension(hc.count), tokens),
            matrix(allocator, DType::BF16, dimension(config.hidden_size), tokens)};
}

struct MtpStemRoots {
    Tensor embedding;
    Tensor normalized_embedding;
    Tensor normalized_hidden;
    Tensor packed_input;
    Tensor residual;
    Tensor attention_hidden;
};

template <class Allocator>
MtpStemRoots mtp_stem(Allocator& allocator, const TextConfig& config, std::int32_t tokens,
                      bool allocate_embedding) {
    MtpStemRoots out;
    if (allocate_embedding) {
        out.embedding = matrix(allocator, DType::BF16, dimension(config.hidden_size), tokens);
    }
    out.normalized_embedding =
        matrix(allocator, DType::BF16, dimension(config.hidden_size), tokens);
    out.normalized_hidden = matrix(allocator, DType::BF16, dimension(config.hidden_size), tokens);
    out.packed_input = matrix(allocator, DType::BF16, dimension(2ULL * config.hidden_size), tokens);
    out.residual     = matrix(allocator, DType::BF16, dimension(config.hidden_size), tokens);
    out.attention_hidden = matrix(allocator, DType::BF16, dimension(config.hidden_size), tokens);
    return out;
}

struct MtpAttentionProjectionRoots {
    Tensor query;
    Tensor key;
    Tensor gate;
    Tensor value;
};

template <class Allocator>
MtpAttentionProjectionRoots mtp_attention_projection(Allocator& allocator, const TextConfig& config,
                                                     std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, dimension(config.attention->query_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention->key_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention->query_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention->key_width()), tokens),
    };
}

struct MtpAttentionResultRoots {
    Tensor normalized_query;
    Tensor normalized_key;
    Tensor attention;
};

template <class Allocator>
MtpAttentionResultRoots mtp_attention_results(Allocator& allocator, const TextConfig& config,
                                              std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, dimension(config.attention->query_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention->key_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention->query_width()), tokens),
    };
}

struct MtpPostAttentionRoots {
    Tensor output;
    Tensor post_mixer_hidden;
};

template <class Allocator>
MtpPostAttentionRoots mtp_post_attention(Allocator& allocator, const TextConfig& config,
                                         std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, dimension(config.hidden_size), tokens),
        matrix(allocator, DType::BF16, dimension(config.hidden_size), tokens),
    };
}

struct DFlashContextRoots {
    Tensor projected;
    Tensor normalized;
};

template <class Allocator>
DFlashContextRoots dflash_context(Allocator& allocator, const TextConfig& target,
                                  const DraftConfig& config, std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, dimension(target.hidden_size), tokens),
        matrix(allocator, DType::BF16, dimension(target.hidden_size), tokens),
    };
}

struct DFlashContextLayerRoots {
    Tensor key_raw;
    Tensor value;
    Tensor key;
};

template <class Allocator>
DFlashContextLayerRoots dflash_context_layer(Allocator& allocator, const TextConfig& target,
                                             const DraftConfig& config, std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, dimension(config.attention.key_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention.key_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention.key_width()), tokens),
    };
}

struct DFlashProposalRoots {
    Tensor ids;
    Tensor positions;
    Tensor residual;
};

template <class Allocator>
DFlashProposalRoots dflash_proposal(Allocator& allocator, const TextConfig& target,
                                    const DraftConfig& config, std::int32_t tokens) {
    return {
        vector(allocator, DType::I32, tokens),
        vector(allocator, DType::I32, tokens),
        matrix(allocator, DType::BF16, dimension(target.hidden_size), tokens),
    };
}

struct DFlashAttentionRoots {
    Tensor hidden;
    Tensor query_raw;
    Tensor key_raw;
    Tensor value;
    Tensor query;
    Tensor key;
    Tensor attention;
};

template <class Allocator>
DFlashAttentionRoots dflash_attention(Allocator& allocator, const TextConfig& target,
                                      const DraftConfig& config, std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, dimension(target.hidden_size), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention.query_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention.key_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention.key_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention.query_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention.key_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.attention.query_width()), tokens),
    };
}

// Dynamic-convolution outputs remain live through the corresponding attention/MLP branch.
struct DFlash2BranchRoots {
    Tensor prepared;
    Tensor finish_delta;
};

template <class Allocator>
DFlash2BranchRoots dflash2_branch(Allocator& allocator, const TextConfig& target,
                                  const DraftConfig& config, std::int32_t width,
                                  std::int32_t batch) {
    return {allocator.alloc(DType::BF16, {dimension(target.hidden_size), width, batch}),
            allocator.alloc(DType::BF16,
                            {dimension(target.hidden_size / config.dflash2->conv_group_size), 2,
                             width, batch})};
}

struct DFlashMlpRoots {
    Tensor hidden;
    Tensor intermediate;
};

template <class Allocator>
DFlashMlpRoots dflash_mlp(Allocator& allocator, const TextConfig& target, const DraftConfig& config,
                          std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, dimension(target.hidden_size), tokens),
        matrix(allocator, DType::BF16, dimension(config.intermediate_size), tokens),
    };
}

} // namespace ninfer::models::qwen3_5::execution::workspace
