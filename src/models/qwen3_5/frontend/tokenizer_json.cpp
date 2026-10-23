#include "models/qwen3_5/frontend/tokenizer_json.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <initializer_list>
#include <limits>
#include <string>

namespace ninfer::models::qwen3_5::frontend::detail {
namespace {

using Json = nlohmann::json;

bool at_path(const std::vector<std::string>& path, std::initializer_list<const char*> want) {
    if (path.size() != want.size()) { return false; }
    std::size_t index = 0;
    for (const char* part : want) {
        if (path[index] != part) { return false; }
        ++index;
    }
    return true;
}

bool under_path(const std::vector<std::string>& path, std::initializer_list<const char*> prefix) {
    if (path.size() < prefix.size()) { return false; }
    std::size_t index = 0;
    for (const char* part : prefix) {
        if (path[index] != part) { return false; }
        ++index;
    }
    return true;
}

// Streams tokenizer.json through the SAX interface and keeps only the fields the tokenizer consumes.
// No DOM is built, so the vector<basic_json> growth that faults on this document cannot happen.
class TokenizerJsonReader final : public nlohmann::json_sax<Json> {
public:
    explicit TokenizerJsonReader(TokenizerJson& out) : out_(out) {}

    bool null() override {
        if (model_member("dropout")) {
            out_.has_dropout = false;
        } else if (model_member("continuing_subword_prefix")) {
            out_.has_continuing_subword_prefix = false;
        } else if (model_member("end_of_word_suffix")) {
            out_.has_end_of_word_suffix = false;
        } else if (root_member("post_processor")) {
            out_.has_post_processor = false;
        }
        return true;
    }

    bool boolean(bool value) override {
        if (added_element()) { record_added_bool(value); }
        return true;
    }

    bool number_integer(number_integer_t value) override { return number(value); }
    bool number_unsigned(number_unsigned_t value) override {
        if (value > static_cast<number_unsigned_t>(std::numeric_limits<std::int64_t>::max())) {
            return fail("tokenizer.json number does not fit in int64");
        }
        return number(static_cast<number_integer_t>(value));
    }
    bool number_float(number_float_t value, const string_t&) override {
        if (model_member("dropout")) {
            const auto rounded = static_cast<std::int64_t>(value);
            if (static_cast<number_float_t>(rounded) != value) {
                return fail("model.dropout is not an integer");
            }
            out_.has_dropout = true;
            out_.dropout     = rounded;
        }
        return true;
    }

    bool string(string_t& value) override {
        if (model_member("type")) {
            out_.model_type = value;
        } else if (at_path(path_, {"", "normalizer"}) && pending_key_ == "type") {
            out_.normalizer_type = value;
        } else if (at_path(path_, {"", "decoder"}) && pending_key_ == "type") {
            out_.decoder_type = value;
        } else if (at_path(path_, {"", "post_processor"}) && pending_key_ == "type") {
            out_.post_processor_type = value;
        } else if (under_path(path_, {"", "pre_tokenizer"}) && pending_key_ == "type") {
            out_.pre_tokenizer_types.push_back(value);
        } else if (merge_element()) {
            MergeEntry entry;
            entry.pair_form = false;
            entry.left      = value;
            out_.merges.push_back(std::move(entry));
        } else if (merge_pair_element()) {
            if (merge_slot_ == 0) {
                current_merge_.left = value;
            } else if (merge_slot_ == 1) {
                current_merge_.right = value;
            } else {
                return fail("model.merges entry has more than two symbols");
            }
            ++merge_slot_;
        } else if (added_element() && pending_key_ == "content") {
            current_added_.content = value;
            current_added_.present |= RawAddedToken::kContent;
        }
        return true;
    }

    bool binary(binary_t&) override { return fail("tokenizer.json contains binary data"); }

    bool start_object(std::size_t) override {
        path_.push_back(pending_key_);
        pending_key_.clear();
        if (at_path(path_, {"", "model"})) { out_.has_model = true; }
        if (at_path(path_, {"", "model", "vocab"})) { out_.has_vocab = true; }
        if (at_path(path_, {"", "normalizer"})) { out_.has_normalizer = true; }
        if (at_path(path_, {"", "pre_tokenizer"})) { out_.has_pre_tokenizer = true; }
        if (at_path(path_, {"", "decoder"})) { out_.has_decoder = true; }
        if (at_path(path_, {"", "post_processor"})) { out_.has_post_processor = true; }
        if (added_element()) { current_added_ = RawAddedToken{}; }
        return true;
    }

    bool key(string_t& value) override {
        pending_key_ = value;
        if (at_path(path_, {"", "model", "vocab"})) { current_key_ = value; }
        return true;
    }

    bool end_object() override { return leave(); }

    bool start_array(std::size_t) override {
        path_.push_back(pending_key_);
        pending_key_.clear();
        if (merge_pair_container()) {
            current_merge_            = MergeEntry{};
            current_merge_.pair_form  = true;
            merge_slot_               = 0;
        }
        return true;
    }

    bool end_array() override {
        if (merge_pair_container()) {
            if (merge_slot_ != 2) { return fail("model.merges entry does not have two symbols"); }
            out_.merges.push_back(std::move(current_merge_));
        }
        return leave();
    }

    bool parse_error(std::size_t position, const std::string&,
                     const nlohmann::detail::exception&) override {
        error_ = "malformed tokenizer.json at byte " + std::to_string(position);
        return false;
    }

    [[nodiscard]] const std::string& error() const noexcept { return error_; }

private:
    // A member of the document root, e.g. {"decoder": {...}} is read while path_ == [""].
    [[nodiscard]] bool root_member(const char* name) const {
        return path_.size() == 1 && path_[0].empty() && pending_key_ == name;
    }

    // A member of the "model" object, e.g. {"model": {"type": "BPE"}}.
    [[nodiscard]] bool model_member(const char* name) const {
        return at_path(path_, {"", "model"}) && pending_key_ == name;
    }

    [[nodiscard]] bool added_element() const {
        return path_.size() == 3 && path_[0].empty() && path_[1] == "added_tokens";
    }

    // A merge spelled as one string, read while path_ is the merges array itself.
    [[nodiscard]] bool merge_element() const { return at_path(path_, {"", "model", "merges"}); }

    // A merge spelled as ["left", "right"]: the inner array, then its two strings.
    [[nodiscard]] bool merge_pair_container() const {
        return path_.size() == 4 && path_[0].empty() && path_[1] == "model" &&
               path_[2] == "merges" && path_[3].empty();
    }
    [[nodiscard]] bool merge_pair_element() const {
        return path_.size() == 5 && path_[0].empty() && path_[1] == "model" &&
               path_[2] == "merges" && path_[3].empty() && path_[4].empty();
    }

    bool leave() {
        if (added_element()) {
            if ((current_added_.present & RawAddedToken::kId) == 0) {
                return fail("missing field added_tokens.id");
            }
            for (const unsigned required :
                 {RawAddedToken::kContent, RawAddedToken::kSingleWord, RawAddedToken::kLstrip,
                  RawAddedToken::kRstrip, RawAddedToken::kNormalized, RawAddedToken::kSpecial}) {
                if ((current_added_.present & required) == 0) {
                    return fail("added_tokens entry is missing a required field");
                }
            }
            out_.added_tokens.push_back(std::move(current_added_));
        }
        path_.pop_back();
        return true;
    }

    void record_added_bool(bool value) {
        if (pending_key_ == "single_word") {
            current_added_.single_word = value;
            current_added_.present |= RawAddedToken::kSingleWord;
        } else if (pending_key_ == "lstrip") {
            current_added_.lstrip = value;
            current_added_.present |= RawAddedToken::kLstrip;
        } else if (pending_key_ == "rstrip") {
            current_added_.rstrip = value;
            current_added_.present |= RawAddedToken::kRstrip;
        } else if (pending_key_ == "normalized") {
            current_added_.normalized = value;
            current_added_.present |= RawAddedToken::kNormalized;
        } else if (pending_key_ == "special") {
            current_added_.special = value;
            current_added_.present |= RawAddedToken::kSpecial;
        }
    }

    bool number(std::int64_t value) {
        if (at_path(path_, {"", "model", "vocab"})) {
            out_.vocab.emplace_back(current_key_, value);
        } else if (added_element() && pending_key_ == "id") {
            current_added_.id = value;
            current_added_.present |= RawAddedToken::kId;
        } else if (model_member("dropout")) {
            out_.has_dropout = true;
            out_.dropout     = value;
        }
        return true;
    }

    bool fail(const char* message) {
        error_ = message;
        return false;
    }

    TokenizerJson& out_;
    std::vector<std::string> path_;
    std::string pending_key_;
    std::string current_key_;
    RawAddedToken current_added_;
    MergeEntry current_merge_;
    int merge_slot_ = 0;
    std::string error_;
};

} // namespace

bool parse_tokenizer_json(std::string_view text, TokenizerJson& out, std::string& error) {
    out = TokenizerJson{};
    TokenizerJsonReader reader(out);
    const bool ok = Json::sax_parse(text, &reader);
    if (!ok) {
        error = reader.error().empty() ? "malformed tokenizer.json" : reader.error();
        return false;
    }
    error.clear();
    return true;
}

} // namespace ninfer::models::qwen3_5::frontend::detail
