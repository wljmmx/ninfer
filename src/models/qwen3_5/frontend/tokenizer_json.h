#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::frontend::detail {

// One entry of model.merges. The file may spell an entry as "left right" or as ["left", "right"];
// both forms are kept distinct so the rank order and the pair boundaries survive unchanged.
struct MergeEntry {
    bool pair_form = false;
    std::string left;
    std::string right;
};

// One entry of added_tokens. `present` records which fields the document actually carried so the
// caller can keep applying the same required-field checks it applied to the DOM.
struct RawAddedToken {
    static constexpr unsigned kId         = 1U << 0;
    static constexpr unsigned kContent    = 1U << 1;
    static constexpr unsigned kSingleWord = 1U << 2;
    static constexpr unsigned kLstrip     = 1U << 3;
    static constexpr unsigned kRstrip     = 1U << 4;
    static constexpr unsigned kNormalized = 1U << 5;
    static constexpr unsigned kSpecial    = 1U << 6;

    std::int64_t id   = -1;
    std::string content;
    bool single_word  = false;
    bool lstrip       = false;
    bool rstrip       = false;
    bool normalized   = false;
    bool special      = false;
    unsigned present  = 0;
};

// Everything the tokenizer needs from tokenizer.json, collected while the document is streamed.
// Building this directly avoids the DOM's vector<basic_json> growth, which is the path that faults
// on the 12 MB vocabulary document.
struct TokenizerJson {
    bool has_model = false;
    bool has_vocab = false;
    std::string model_type;
    std::vector<std::pair<std::string, std::int64_t>> vocab;
    std::vector<MergeEntry> merges;
    std::vector<RawAddedToken> added_tokens;

    bool has_normalizer = false;
    std::string normalizer_type;
    bool has_pre_tokenizer = false;
    std::vector<std::string> pre_tokenizer_types;
    bool has_decoder = false;
    std::string decoder_type;
    bool has_post_processor = false;
    std::string post_processor_type;

    bool has_dropout     = false;
    std::int64_t dropout = 0;
    bool has_ignore_merges = false;
    bool ignore_merges     = false;
    bool has_continuing_subword_prefix = false;
    std::string continuing_subword_prefix;
    bool has_end_of_word_suffix = false;
    std::string end_of_word_suffix;
};

// Streams tokenizer.json through the nlohmann SAX interface and fills `out`. Returns false and
// fills `error` on malformed input or an unsupported shape.
bool parse_tokenizer_json(std::string_view text, TokenizerJson& out, std::string& error);

} // namespace ninfer::models::qwen3_5::frontend::detail
