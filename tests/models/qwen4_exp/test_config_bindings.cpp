// Stage M1 of the Qwen4Exp port: the text config of the real artifact parses, and every binding it
// carries resolves through the aliases the model layer declares. Execution is not exercised here.
//
// The declared geometry of this model is 113 GiB. A plan never reads device bytes, so the fixture
// payload is a sparse file: the manifest is truthful about shapes and encoded sizes, and the
// unread tail stays unallocated. Entries are laid out in the artifact's own order.

#include "artifact/formats.h"
#include "artifact/fixture.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "core/weight_view.h"
#include "models/qwen3_5/config.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/load/bindings.h"

#include <nlohmann/json.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <winioctl.h>
#endif

namespace {

using namespace ninfer;
using namespace ninfer::test::artifact_fixture;
namespace qwen = ninfer::models::qwen3_5;
using models::Architecture;

constexpr std::uint32_t kHidden           = 2560;
constexpr std::uint32_t kLayers           = 48;
constexpr std::uint32_t kAttentionHeads   = 24;
constexpr std::uint32_t kKvHeads          = 2;
constexpr std::uint32_t kHeadDim          = 256;
constexpr std::uint32_t kLinearKeyHeads   = 16;
constexpr std::uint32_t kLinearValueHeads = 48;
constexpr std::uint32_t kLinearHeadDim    = 128;
constexpr std::uint32_t kConvKernel       = 4;
constexpr std::uint32_t kExperts          = 512;
constexpr std::uint32_t kExpertWidth      = 640;
constexpr std::uint32_t kHcCount          = 4;
constexpr std::uint32_t kHcLowrank        = 320;
constexpr std::uint32_t kVocab            = 248320;
constexpr std::uint32_t kPleEmbed         = 2560;
constexpr std::uint32_t kPleParts         = 128;
constexpr std::uint32_t kPleShardRows     = 2500012;
constexpr std::uint32_t kPleRowWidth      = 160;
constexpr std::uint32_t kStreams          = kHcCount * kHidden;
constexpr std::uint32_t kIndexerHeads     = 4;
constexpr std::uint32_t kIndexerKvHeads   = 1;
constexpr std::uint32_t kIndexerHeadDim   = 128;

constexpr std::uint32_t kAttentionLayers = kLayers / 4;
constexpr std::uint32_t kLinearLayers    = kLayers - kAttentionLayers;
constexpr std::uint32_t kBindings        = 1308;

int failures = 0;

void check(bool condition, const std::string& label) {
    if (condition) { return; }
    std::cerr << label << ": failed\n";
    ++failures;
}

// The real text config of the artifact (its key set, not a paraphrase).
Json text_config() {
    Json layers = Json::array();
    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
        layers.push_back(layer % 4 == 3 ? "indexed_attention" : "linear_attention");
    }
    return {{"hidden_size", kHidden},
            {"vocab_size", kVocab},
            {"num_hidden_layers", kLayers},
            {"max_position_embeddings", 262144},
            {"hc_count", kHcCount},
            {"hc_lowrank", kHcLowrank},
            {"num_attention_heads", kAttentionHeads},
            {"num_key_value_heads", kKvHeads},
            {"head_dim", kHeadDim},
            {"linear_num_key_heads", kLinearKeyHeads},
            {"linear_num_value_heads", kLinearValueHeads},
            {"linear_key_head_dim", kLinearHeadDim},
            {"linear_value_head_dim", kLinearHeadDim},
            {"linear_conv_kernel_dim", kConvKernel},
            {"indexer_n_heads", kIndexerHeads},
            {"indexer_kv_heads", kIndexerKvHeads},
            {"indexer_head_dim", kIndexerHeadDim},
            {"indexer_compress_ratio", 4},
            {"indexer_budget", 2048},
            {"num_experts", kExperts},
            {"num_experts_per_tok", 10},
            {"moe_intermediate_size", kExpertWidth},
            {"shared_expert_intermediate_size", kExpertWidth},
            {"ple_embed_dim", kPleEmbed},
            {"ngram_size", 3},
            {"heads_per_ngram", 8},
            {"split_ngram_parts", kPleParts},
            {"ple_conv_kernel_size", kConvKernel},
            {"architectures", {"Qwen4ExpForCausalLM"}},
            {"model_type", "qwen4_exp_text"},
            {"hidden_act", "silu"},
            {"output_gate_type", "sigmoid"},
            {"tie_word_embeddings", false},
            {"attention_bias", false},
            {"mamba_ssm_dtype", "float32"},
            {"rms_norm_eps", 1e-06},
            {"norm_topk_prob", true},
            {"layer_types", layers},
            {"ple_layer_ids", {1}},
            {"ple_conv_dilation", 3},
            {"eos_token_id", 248044},
            {"rope_parameters",
             {{"mrope_interleaved", true},
              {"mrope_section", {11, 11, 10}},
              {"partial_rotary_factor", 0.25},
              {"rope_theta", 10000000},
              {"rope_type", "default"}}},
            {"image_token_id", 248056},
            {"video_token_id", 248057},
            {"vision_start_token_id", 248053},
            {"vision_end_token_id", 248054},
            {"ple_shard_shape", {kPleShardRows, kPleRowWidth}}};
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

std::uint64_t allocated_bytes(const std::filesystem::path& path) {
#ifdef _WIN32
    DWORD high     = 0;
    const auto low = GetCompressedFileSizeW(path.c_str(), &high);
    return (std::uint64_t(high) << 32) | low;
#else
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    return error ? 0 : size;
#endif
}

void extend_sparsely(const std::filesystem::path& path, std::uint64_t bytes) {
#ifdef _WIN32
    const auto handle = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                                    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("fixture: cannot reopen the declared payload");
    }
    DWORD returned = 0;
    const BOOL sparse =
        DeviceIoControl(handle, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &returned, nullptr);
    LARGE_INTEGER size{};
    size.QuadPart    = static_cast<LONGLONG>(bytes);
    const BOOL moved = SetFilePointerEx(handle, size, nullptr, FILE_BEGIN);
    const BOOL ended = moved && SetEndOfFile(handle);
    CloseHandle(handle);
    if (!sparse || !ended) {
        throw std::runtime_error(
            "fixture: the declared payload needs a sparse extension (NTFS/FSCTL_SET_SPARSE)");
    }
    // Only the manifest and the resources may be allocated; a real allocation here would consume
    // the declared payload on the developer's volume.
    if (allocated_bytes(path) > bytes / 2) {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
        throw std::runtime_error("fixture: the declared payload was not allocated sparsely");
    }
#else
    std::filesystem::resize_file(path, bytes);
#endif
}

struct ModelFile {
    std::filesystem::path directory;
    std::filesystem::path entry;
    Json root;
    std::vector<std::byte> resources;
    std::uint64_t cursor = 0;

    ModelFile() : directory(make_unique_directory()), entry(directory / "model.ninfer") {
        root = {
            {"components", {{"text", {{"config", text_config()}, {"resources", Json::object()}}}}},
            {"objects", Json::array()},
            {"bindings", Json::object()},
            {"uses", Json::array()},
            {"files", Json::array({{{"path", nullptr}, {"payload_bytes", 0}}})},
            {"metadata", {{"name", "qwen4exp-binding-fixture"}}}};
    }

    ~ModelFile() {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    }

    ModelFile(const ModelFile&)            = delete;
    ModelFile& operator=(const ModelFile&) = delete;

    void resource(const std::string& role, const std::string& bytes) {
        const auto offset = resources.size();
        for (const auto byte : bytes) { resources.push_back(std::byte(byte)); }
        root["objects"].push_back({{"id", role},
                                   {"kind", "resource"},
                                   {"encoding", "raw_bytes_v1"},
                                   {"offset", offset},
                                   {"bytes", bytes.size()}});
        root["components"]["text"]["resources"][role] = role;
        cursor = std::max(cursor, static_cast<std::uint64_t>(resources.size()));
    }

    void tensor(const std::string& id, std::vector<std::uint64_t> shape, QType format,
                QuantLayout layout) {
        const auto geometry = weight_geometry(format, layout, shape);
        const auto offset   = align_up(cursor, std::max<std::uint64_t>(256, geometry.alignment));
        root["objects"].push_back({{"id", id},
                                   {"kind", "tensor"},
                                   {"shape", shape},
                                   {"format", std::string(artifact::format_name(format))},
                                   {"layout", std::string(artifact::layout_name(layout))},
                                   {"offset", offset},
                                   {"bytes", geometry.bytes}});
        cursor = offset + geometry.bytes;
    }

    void whole(const std::string& name, const std::string& object) {
        root["bindings"][name] = {{"object", object}};
    }

    void part(const std::string& name, const std::string& object, std::uint64_t begin,
              std::uint64_t end) {
        root["bindings"][name] = {
            {"parts", Json::array({{{"object", object}, {"range", {begin, end}}}})}};
    }

    void use(const std::string& parameter, const std::string& input) {
        root["uses"].push_back(
            {{"parameter", parameter}, {"input", input}, {"activation_policy", "A16Only"}});
    }

    // Declares one tensor and binds the parameter of the same name to it.
    void parameter(const std::string& name, std::vector<std::uint64_t> shape, QType format,
                   QuantLayout layout, const std::string& input = {}) {
        tensor(name, shape, format, layout);
        whole(name, name);
        if (!input.empty()) { use(name, input); }
    }

    void write() const {
        auto manifest                         = root;
        manifest["files"][0]["payload_bytes"] = cursor;
        const auto text                       = manifest.dump();
        const auto start                      = align_up(32 + text.size(), 4096);
        std::ofstream file(entry, std::ios::binary | std::ios::trunc);
        file.exceptions(std::ios::badbit | std::ios::failbit);
        std::array<std::byte, 32> header{};
        const std::array<unsigned char, 8> magic{'N', 'I', 'N', 'F', 'E', 'R', 0, 3};
        for (std::size_t i = 0; i < magic.size(); ++i) { header[i] = std::byte(magic[i]); }
        put_word(header, 8, text.size(), 8);
        header[16] = std::byte{0x71};
        file.write(reinterpret_cast<const char*>(header.data()), header.size());
        file.write(text.data(), static_cast<std::streamsize>(text.size()));
        const std::string padding(start - 32 - text.size(), '\0');
        file.write(padding.data(), static_cast<std::streamsize>(padding.size()));
        file.write(reinterpret_cast<const char*>(resources.data()),
                   static_cast<std::streamsize>(resources.size()));
        file.close();
        extend_sparsely(entry, start + cursor);
    }
};

// A tiny public domain is enough: the loader only requires the tokenizer's ids to be contiguous
// and inside the embedding rows.
void add_resources(ModelFile& file) {
    const std::string tokenizer = Json{
        {"model",
         {{"type", "BPE"}, {"vocab", {{"a", 0}, {"b", 1}, {"c", 2}}}, {"merges", Json::array()}}},
        {"added_tokens",
         Json::array()}}.dump();
    file.resource("tokenizer.json", tokenizer);
    file.resource("tokenizer_config.json", Json{{"added_tokens_decoder", Json::object()}}.dump());
    file.resource("generation_config.json", Json{{"eos_token_id", 1}}.dump());
    file.resource("chat_template.jinja", "{{ prompt }}");
}

void build(ModelFile& file) {
    add_resources(file);
    const auto layer_prefix = [](std::uint32_t layer) {
        return "text/layers/" + std::to_string(layer) + "/";
    };

    file.parameter("text/token_embedding", {kVocab, kHidden}, QType::Q8_G32_FP16,
                   QuantLayout::RowSplit);
    file.parameter("text/output_head", {kVocab, kHidden}, QType::Q5_G64_FP16, QuantLayout::RowSplit,
                   "text/final_hidden");
    // The final grouped norm and its low-rank correction; there is no injection stream.
    file.parameter("text/final_hc/norm", {kStreams}, QType::BF16, QuantLayout::Contiguous);
    file.parameter("text/final_hc/down", {kHcLowrank, kStreams}, QType::Q5_G64_FP16,
                   QuantLayout::RowSplit, "text/final_hc/normalized_streams");
    file.parameter("text/final_hc/up", {kStreams, kHcLowrank}, QType::Q5_G64_FP16,
                   QuantLayout::RowSplit, "text/final_hc/lowrank_activation");

    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
        const auto p       = layer_prefix(layer);
        const bool indexed = layer % 4 == 3;
        // The stream corrections. down and inject share one fused parent, as in the artifact.
        for (const auto* hc : {"attention_hc/", "moe_hc/"}) {
            const auto prefix = p + hc;
            file.parameter(prefix + "norm", {kStreams}, QType::BF16, QuantLayout::Contiguous);
            const auto parent = prefix + "down_inject";
            file.tensor(parent, {kHcLowrank + kHcCount, kStreams}, QType::Q5_G64_FP16,
                        QuantLayout::RowSplit);
            file.part(prefix + "down", parent, 0, std::uint64_t(kHcLowrank) * kStreams);
            file.part(prefix + "inject", parent, std::uint64_t(kHcLowrank) * kStreams,
                      std::uint64_t(kHcLowrank + kHcCount) * kStreams);
            file.use(prefix + "down", prefix + "normalized_streams");
            file.use(prefix + "inject", prefix + "normalized_streams");
            file.parameter(prefix + "up", {kStreams, kHcLowrank}, QType::Q5_G64_FP16,
                           QuantLayout::RowSplit, prefix + "lowrank_activation");
        }
        // The MoE tail.
        file.parameter(p + "moe/router", {kExperts, kHidden}, QType::BF16, QuantLayout::Contiguous,
                       p + "ffn_input");
        file.parameter(p + "moe/shared_score", {1, kHidden}, QType::BF16, QuantLayout::Contiguous,
                       p + "ffn_input");
        file.parameter(p + "moe/shared/gate", {kExpertWidth, kHidden}, QType::Q5_G64_FP16,
                       QuantLayout::RowSplit, p + "ffn_input");
        file.parameter(p + "moe/shared/up", {kExpertWidth, kHidden}, QType::Q5_G64_FP16,
                       QuantLayout::RowSplit, p + "ffn_input");
        file.parameter(p + "moe/shared/down", {kHidden, kExpertWidth}, QType::Q5_G64_FP16,
                       QuantLayout::RowSplit, p + "moe/shared/product");
        // One bank object per expert role; a per-expert parameter is a row block of it.
        file.parameter(p + "moe/experts/gate", {std::uint64_t(kExperts) * kExpertWidth, kHidden},
                       QType::Q4_G64_FP16, QuantLayout::RowSplit, p + "ffn_input");
        file.parameter(p + "moe/experts/up", {std::uint64_t(kExperts) * kExpertWidth, kHidden},
                       QType::Q4_G64_FP16, QuantLayout::RowSplit, p + "ffn_input");
        file.parameter(p + "moe/experts/down", {std::uint64_t(kExperts) * kHidden, kExpertWidth},
                       QType::Q6_G64_FP16, QuantLayout::RowSplit, p + "moe/experts/product");

        if (indexed) {
            const std::uint64_t q = std::uint64_t(kAttentionHeads) * kHeadDim;
            const std::uint64_t k = std::uint64_t(kKvHeads) * kHeadDim;
            // query and the output gate share one fused parent.
            const auto parent = p + "attention/query_gate";
            file.tensor(parent, {2 * q, kHidden}, QType::Q4_G64_FP16, QuantLayout::RowSplit);
            file.part(p + "attention/query", parent, 0, q * kHidden);
            file.part(p + "attention/output_gate", parent, q * kHidden, 2 * q * kHidden);
            file.use(p + "attention/query", p + "mixer_input");
            file.use(p + "attention/output_gate", p + "mixer_input");
            file.parameter(p + "attention/key", {k, kHidden}, QType::Q4_G64_FP16,
                           QuantLayout::RowSplit, p + "mixer_input");
            file.parameter(p + "attention/value", {k, kHidden}, QType::Q5_G64_FP16,
                           QuantLayout::RowSplit, p + "mixer_input");
            file.parameter(p + "attention/output", {kHidden, q}, QType::Q5_G64_FP16,
                           QuantLayout::RowSplit, p + "attention/gated_output");
            file.parameter(p + "attention/query_norm", {kHeadDim}, QType::BF16,
                           QuantLayout::Contiguous);
            file.parameter(p + "attention/key_norm", {kHeadDim}, QType::BF16,
                           QuantLayout::Contiguous);
            file.parameter(p + "attention/index_query",
                           {std::uint64_t(kIndexerHeads) * kIndexerHeadDim, kHidden},
                           QType::Q4_G64_FP16, QuantLayout::RowSplit, p + "mixer_input");
            file.parameter(p + "attention/index_key",
                           {std::uint64_t(kIndexerKvHeads) * kIndexerHeadDim, kHidden},
                           QType::Q4_G64_FP16, QuantLayout::RowSplit, p + "mixer_input");
            file.parameter(p + "attention/index_query_norm", {kIndexerHeadDim}, QType::BF16,
                           QuantLayout::Contiguous);
            file.parameter(p + "attention/index_key_norm", {kIndexerHeadDim}, QType::BF16,
                           QuantLayout::Contiguous);
        } else {
            const std::uint64_t k     = std::uint64_t(kLinearKeyHeads) * kLinearHeadDim;
            const std::uint64_t v     = std::uint64_t(kLinearValueHeads) * kLinearHeadDim;
            const std::uint64_t conv  = 2 * k + v;
            const std::uint64_t heads = kLinearValueHeads;
            file.parameter(p + "gdn/query", {k, kHidden}, QType::Q4_G64_FP16, QuantLayout::RowSplit,
                           p + "mixer_input");
            file.parameter(p + "gdn/key", {k, kHidden}, QType::Q4_G64_FP16, QuantLayout::RowSplit,
                           p + "mixer_input");
            file.parameter(p + "gdn/value", {v, kHidden}, QType::Q5_G64_FP16, QuantLayout::RowSplit,
                           p + "mixer_input");
            file.parameter(p + "gdn/z", {v, kHidden}, QType::Q4_G64_FP16, QuantLayout::RowSplit,
                           p + "mixer_input");
            const auto parent = p + "gdn/ab_parent";
            file.tensor(parent, {2 * heads, kHidden}, QType::BF16, QuantLayout::Contiguous);
            file.part(p + "gdn/a_projection", parent, 0, heads * kHidden);
            file.part(p + "gdn/b_projection", parent, heads * kHidden, 2 * heads * kHidden);
            file.use(p + "gdn/a_projection", p + "mixer_input");
            file.use(p + "gdn/b_projection", p + "mixer_input");
            file.parameter(p + "gdn/a_log", {heads}, QType::FP32, QuantLayout::Contiguous);
            file.parameter(p + "gdn/dt_bias", {heads}, QType::FP32, QuantLayout::Contiguous);
            file.parameter(p + "gdn/convolution", {kConvKernel, conv}, QType::BF16,
                           QuantLayout::Contiguous);
            file.parameter(p + "gdn/norm", {kLinearHeadDim}, QType::BF16, QuantLayout::Contiguous);
            file.parameter(p + "gdn/output", {kHidden, v}, QType::Q5_G64_FP16,
                           QuantLayout::RowSplit, p + "gdn/gated_output");
        }

        if (layer == 1) {
            // key and value share one fused parent: [streams + embed, embed].
            const std::uint64_t key_rows = std::uint64_t(kHcCount) * kPleEmbed;
            const auto parent            = p + "ple/key_value";
            file.tensor(parent, {key_rows + kPleEmbed, kPleEmbed}, QType::Q5_G64_FP16,
                        QuantLayout::RowSplit);
            file.part(p + "ple/key", parent, 0, key_rows * kPleEmbed);
            file.part(p + "ple/value", parent, key_rows * kPleEmbed,
                      (key_rows + kPleEmbed) * kPleEmbed);
            file.use(p + "ple/key", p + "ple/embedding");
            file.use(p + "ple/value", p + "ple/embedding");
            file.parameter(p + "ple/norm_key", {key_rows}, QType::BF16, QuantLayout::Contiguous);
            file.parameter(p + "ple/norm_query", {key_rows}, QType::BF16, QuantLayout::Contiguous);
            file.parameter(p + "ple/norm_conv", {key_rows}, QType::BF16, QuantLayout::Contiguous);
            file.parameter(p + "ple/convolution", {kConvKernel, key_rows}, QType::BF16,
                           QuantLayout::Contiguous);
            file.parameter(p + "ple/table",
                           {std::uint64_t(kPleShardRows) * kPleParts, kPleRowWidth},
                           QType::Q4_G64_FP16, QuantLayout::RowSplit);
        }
    }
}

std::set<std::string> declared_bindings(const artifact::Directory& directory) {
    std::set<std::string> out;
    for (const auto& [name, binding] : directory.bindings) { out.insert(name); }
    return out;
}

void check_config_and_bindings() {
    ModelFile file;
    build(file);
    file.write();

    artifact::Reader reader(file.entry);
    auto plan        = qwen::plan_load(reader);
    const auto& text = plan.config().text;

    check(text.architecture == Architecture::Qwen4Exp, "the text config keeps its architecture");
    check(text.layer_types.size() == kLayers && text.compact_layer_indices.size() == kLayers,
          "every layer maps onto a mixer");
    check(text.linear_attention_layers == 36 && text.full_attention_layers == 12 &&
              text.indexed_attention_layers == 12,
          "36 linear and 12 full layers, all of the latter indexed");
    check(text.layer_types.at(3) == qwen::MixerKind::IndexedAttention &&
              text.layer_types.at(0) == qwen::MixerKind::LinearAttention,
          "layer 3 is the indexed mixer and layer 0 the GDN one");
    check(qwen::attention_mixer(text.layer_types.at(3)) &&
              !qwen::attention_mixer(text.layer_types.at(0)),
          "the indexed mixer counts as a full-attention mixer");
    check(text.hc && text.hc->count == 4 && text.hc->lowrank == 320,
          "the HC stream structure is stored");
    check(text.indexer && text.indexer->num_heads == 4 && text.indexer->num_kv_heads == 1 &&
              text.indexer->head_dim == 128 && text.indexer->compress_ratio == 4 &&
              text.indexer->budget == 2048,
          "the indexer geometry is stored");
    check(text.ple && text.ple->embed_dim == 2560 && text.ple->ngram_size == 3 &&
              text.ple->heads_per_ngram == 8 && text.ple->split_parts == 128 &&
              text.ple->conv_kernel_size == 4 && text.ple->conv_dilation == 3 &&
              text.ple->layer_ids == std::vector<std::uint32_t>{1} &&
              text.ple->table_rows() == 320001536ULL && text.ple->table_width() == 160,
          "the per-layer embedding geometry is stored");
    check(text.rope_parameters && text.rope_parameters->mrope_interleaved &&
              !text.rope_parameters->yarn && text.rope_parameters->rotary_dim == 64,
          "rope_type default keeps the trained frequencies");
    check(!text.attention_bias && text.norm_topk_prob && text.output_gate_type == "sigmoid" &&
              text.mamba_ssm_dtype == "float32",
          "the flags the artifact spells out are stored");
    check(text.multimodal_token_ids && text.multimodal_token_ids->image == 248056 &&
              text.multimodal_token_ids->video == 248057 &&
              text.multimodal_token_ids->vision_start == 248053 &&
              text.multimodal_token_ids->vision_end == 248054,
          "the sentinel token ids are stored");
    const auto& moe = std::get<qwen::MoeConfig>(text.ffn);
    check(moe.num_experts == kExperts && moe.num_experts_per_tok == 10 &&
              moe.moe_intermediate_size == kExpertWidth &&
              moe.shared_expert_intermediate_size == kExpertWidth,
          "the MoE geometry is stored");

    const auto& weights = plan.weights().text;
    check(weights.layers.size() == kLayers, "the plan binds every layer");
    check(weights.final_hc && weights.final_hc->inject == std::nullopt,
          "the final correction has no injection stream");

    const auto object_of = [&](qwen::WeightId id) {
        return artifact::object_id(
            reader.directory().object(plan.parameter(id).binding.parts.at(0).object));
    };
    // The aliased names resolve to the artifact's own bindings.
    check(plan.parameter(weights.final_norm).name == "text/final_norm" &&
              plan.parameter(weights.final_norm).shape == std::vector<std::uint64_t>{kStreams} &&
              object_of(weights.final_norm) == "text/final_hc/norm",
          "final_norm resolves to final_hc/norm");
    const auto& first = weights.layers.at(0);
    check(plan.parameter(first.input_norm).name == "text/layers/0/input_norm" &&
              object_of(first.input_norm) == "text/layers/0/attention_hc/norm",
          "input_norm resolves to attention_hc/norm");
    check(object_of(first.post_attention_norm) == "text/layers/0/moe_hc/norm",
          "post_attention_norm resolves to moe_hc/norm");
    check(first.input_hc && first.input_hc->inject.has_value() &&
              plan.parameter(*first.input_hc->inject).shape ==
                  std::vector<std::uint64_t>{kHcCount, kStreams},
          "the per-mixer injection stream is bound");
    const auto& third     = weights.layers.at(3);
    const auto& attention = std::get<qwen::AttentionWeights>(third.mixer);
    check(attention.indexer && plan.parameter(attention.indexer->query).shape ==
                                   std::vector<std::uint64_t>{512, kHidden},
          "the indexed layer binds its indexer heads");
    check(object_of(attention.gate) == object_of(attention.query) &&
              plan.parameter(attention.gate).binding.parts.at(0).begin ==
                  std::uint64_t(kAttentionHeads) * kHeadDim * kHidden,
          "attention/gate resolves to the fused parent's output_gate rows");
    const auto& moe_weights = std::get<qwen::MoeWeights>(third.ffn);
    check(moe_weights.experts.size() == kExperts, "every expert is bound");
    const auto& expert = moe_weights.experts.at(7).gate;
    check(plan.parameter(expert).name == "text/layers/3/moe/experts/7/gate" &&
              plan.parameter(expert).shape == std::vector<std::uint64_t>{kExpertWidth, kHidden} &&
              plan.parameter(expert).binding.parts.at(0).begin ==
                  7 * std::uint64_t(kExpertWidth) * kHidden &&
              object_of(expert) == "text/layers/3/moe/experts/gate",
          "expert 7 is the row block its index selects out of the bank");
    check(std::get<qwen::MoeWeights>(weights.layers.at(0).ffn).experts.at(kExperts - 1).down !=
              expert,
          "the last expert of a layer is a distinct parameter");
    check(weights.layers.at(1).ple && plan.parameter(weights.layers.at(1).ple->table).shape ==
                                          std::vector<std::uint64_t>{320001536, 160},
          "the paged table is bound on its own layer");
    check(!weights.layers.at(0).ple && !weights.layers.at(2).ple,
          "layers outside ple_layer_ids carry no embedding table");

    // The binder ignores unreferenced bindings, so only the claimed set shows that every one of
    // the artifact's bindings reached a logical parameter.
    artifact::Binder binder(reader);
    qwen::loading::Bindings bindings(binder);
    (void)qwen::loading::bind_text(bindings, text, {});
    std::set<std::string> claimed;
    for (const auto& pending : bindings.weights) { claimed.insert(pending.binding_name); }
    check(claimed == declared_bindings(reader.directory()),
          "every artifact binding is claimed by exactly one parameter name");
    check(claimed.size() == kBindings, "the artifact carries 1308 bindings");
    const auto banked = std::uint64_t(kLayers) * 3 * (kExperts - 1);
    check(bindings.weights.size() == kBindings + banked,
          "each expert bank yields one parameter per expert");

    std::cout << "declared payload " << file.cursor << " B, " << claimed.size() << " bindings, "
              << bindings.weights.size() << " logical parameters\n";
}

void check_misspelled_binding() {
    ModelFile file;
    build(file);
    const auto renamed = file.root["bindings"]["text/layers/0/attention_hc/norm"];
    file.root["bindings"].erase("text/layers/0/attention_hc/norm");
    file.root["bindings"]["text/layers/0/attention_hc/norm_typo"] = renamed;
    file.write();

    std::string message;
    try {
        artifact::Reader reader(file.entry);
        (void)qwen::plan_load(reader);
    } catch (const artifact::ArtifactError& error) { message = error.what(); }
    check(message.find("text/layers/0/input_norm") != std::string::npos,
          "a misspelled binding is refused under the logical parameter name");
    check(message.find("text/layers/0/attention_hc/norm") != std::string::npos,
          "the refusal names the binding the artifact was expected to carry");
}

// The aliases are opt-in: a qwen3_5 artifact keeps binding under the names the engine already
// declares, and no HC/indexer/PLE structure appears in its config.
void check_qwen3_5_names_unchanged() {
    ModelFile file;
    add_resources(file);
    constexpr std::uint32_t hidden            = 256;
    constexpr std::uint32_t heads             = 4;
    constexpr std::uint32_t head              = 64;
    file.root["components"]["text"]["config"] = Json{{"architectures", {"Qwen3_5ForCausalLM"}},
                                                     {"model_type", "qwen3_5_text"},
                                                     {"hidden_size", hidden},
                                                     {"vocab_size", 64},
                                                     {"num_hidden_layers", 1},
                                                     {"max_position_embeddings", 128},
                                                     {"tie_word_embeddings", false},
                                                     {"rms_norm_eps", 1e-06},
                                                     {"layer_types", {"full_attention"}},
                                                     {"num_attention_heads", heads},
                                                     {"num_key_value_heads", 1},
                                                     {"head_dim", head},
                                                     {"intermediate_size", 128},
                                                     {"rope_parameters",
                                                      {{"rope_theta", 10000},
                                                       {"partial_rotary_factor", 1.0},
                                                       {"mrope_section", {head / 2, 0, 0}}}}};
    const std::string p                       = "text/layers/0/";
    file.parameter("text/token_embedding", {64, hidden}, QType::Q8_G32_FP16, QuantLayout::RowSplit);
    file.parameter("text/output_head", {64, hidden}, QType::Q5_G64_FP16, QuantLayout::RowSplit,
                   "text/final_hidden");
    file.parameter("text/final_norm", {hidden}, QType::BF16, QuantLayout::Contiguous);
    file.parameter(p + "input_norm", {hidden}, QType::BF16, QuantLayout::Contiguous);
    file.parameter(p + "post_attention_norm", {hidden}, QType::BF16, QuantLayout::Contiguous);
    file.parameter(p + "attention/query", {heads * head, hidden}, QType::Q4_G64_FP16,
                   QuantLayout::RowSplit, p + "mixer_input");
    file.parameter(p + "attention/key", {head, hidden}, QType::Q4_G64_FP16, QuantLayout::RowSplit,
                   p + "mixer_input");
    file.parameter(p + "attention/gate", {heads * head, hidden}, QType::Q4_G64_FP16,
                   QuantLayout::RowSplit, p + "mixer_input");
    file.parameter(p + "attention/value", {head, hidden}, QType::Q5_G64_FP16, QuantLayout::RowSplit,
                   p + "mixer_input");
    file.parameter(p + "attention/query_norm", {head}, QType::BF16, QuantLayout::Contiguous);
    file.parameter(p + "attention/key_norm", {head}, QType::BF16, QuantLayout::Contiguous);
    file.parameter(p + "attention/output", {hidden, heads * head}, QType::Q5_G64_FP16,
                   QuantLayout::RowSplit, p + "attention/gated_output");
    file.parameter(p + "mlp/gate", {128, hidden}, QType::Q4_G64_FP16, QuantLayout::RowSplit,
                   p + "ffn_input");
    file.parameter(p + "mlp/up", {128, hidden}, QType::Q4_G64_FP16, QuantLayout::RowSplit,
                   p + "ffn_input");
    file.parameter(p + "mlp/down", {hidden, 128}, QType::Q6_G64_FP16, QuantLayout::RowSplit,
                   p + "mlp/product");
    file.write();

    artifact::Reader reader(file.entry);
    auto plan        = qwen::plan_load(reader);
    const auto& text = plan.config().text;
    check(text.architecture == Architecture::Qwen3_5 && !text.hc && !text.indexer && !text.ple &&
              !text.multimodal_token_ids && text.full_attention_layers == 1,
          "a qwen3_5 config keeps the fields the Qwen4Exp payload carries absent");
    const auto& weights  = plan.weights().text;
    const auto object_of = [&](qwen::WeightId id) {
        return artifact::object_id(
            reader.directory().object(plan.parameter(id).binding.parts.at(0).object));
    };
    check(object_of(weights.final_norm) == "text/final_norm" &&
              !plan.parameter(weights.final_norm).shape.empty() &&
              plan.parameter(weights.final_norm).shape.at(0) == hidden && !weights.final_hc,
          "final_norm keeps its own binding when no HC structure is declared");
    const auto& block = weights.layers.at(0);
    check(object_of(block.input_norm) == "text/layers/0/input_norm" &&
              object_of(block.post_attention_norm) == "text/layers/0/post_attention_norm" &&
              !block.input_hc && !block.post_attention_hc && !block.ple,
          "the block norms keep their own bindings");
    const auto& attention = std::get<qwen::AttentionWeights>(block.mixer);
    check(!attention.indexer && object_of(attention.gate) == "text/layers/0/attention/gate",
          "the output gate keeps its own binding");
}

void check_ambiguous_binding() {
    ModelFile file;
    build(file);
    // The artifact now carries both the canonical gate and the aliased output_gate.
    file.root["bindings"]["text/layers/3/attention/gate"] =
        file.root["bindings"]["text/layers/3/attention/output_gate"];
    file.root["uses"].push_back({{"parameter", "text/layers/3/attention/gate"},
                                 {"input", "text/layers/3/mixer_input"},
                                 {"activation_policy", "A16Only"}});
    file.write();

    std::string message;
    try {
        artifact::Reader reader(file.entry);
        (void)qwen::plan_load(reader);
    } catch (const artifact::ArtifactError& error) { message = error.what(); }
    check(message.find("text/layers/3/attention/gate") != std::string::npos,
          "a binding that exists under both names is refused under the logical name");
}

// Qwen4Exp names the activation its projections consume; the engine's logical name is an alias.
// Both spellings resolve, and a Use the artifact carries under both is refused by logical name.
void check_ambiguous_use() {
    ModelFile file;
    build(file);
    file.root["uses"].push_back({{"parameter", "text/layers/0/gdn/query"},
                                 {"input", "text/layers/0/attention_input"},
                                 {"activation_policy", "A16Only"}});
    file.write();

    std::string message;
    try {
        artifact::Reader reader(file.entry);
        (void)qwen::plan_load(reader);
    } catch (const artifact::ArtifactError& error) { message = error.what(); }
    check(message.find("text/layers/0/gdn/query") != std::string::npos &&
              message.find("ambiguous") != std::string::npos,
          "a Use that exists under both spellings is refused under the logical name");
}

// The alias is the only thing that makes the artifact's spelling bind: without it the refusal names
// the logical parameter and both spellings it accepted.
void check_missing_use() {
    ModelFile file;
    build(file);
    Json kept;
    for (auto& use : file.root["uses"]) {
        if (use.at("parameter") == "text/layers/0/gdn/query") { kept = use; }
    }
    std::vector<Json> remaining;
    for (auto& use : file.root["uses"]) {
        if (use.at("parameter") == "text/layers/0/gdn/query") { continue; }
        remaining.push_back(use);
    }
    file.root["uses"] = remaining;
    file.write();

    std::string message;
    try {
        artifact::Reader reader(file.entry);
        (void)qwen::plan_load(reader);
    } catch (const artifact::ArtifactError& error) { message = error.what(); }
    check(message.find("text/layers/0/gdn/query") != std::string::npos &&
              message.find("missing Use") != std::string::npos,
          "a Use the artifact does not carry is refused under the logical parameter name");
}

// Stage M2. The plan is not the whole logical model: execution::Parameters walks it layer by layer
// with the config's geometry, which is where a shape or format the ops do not implement is refused.
// Experts stream and the embedding table stays in the artifact, so only the resident part (about
// 3.2 GiB of the 113 GiB payload) is uploaded.
void check_real_artifact_parameters(const std::filesystem::path& path) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP real artifact parameter model (no CUDA device)\n";
        return;
    }
    artifact::Reader reader{path};
    models::LoadOptions options;
    options.stream_experts = true;
    try {
        DeviceContext device;
        auto model = qwen::materialize_model(qwen::plan_load(reader, options), device);
        qwen::execution::Parameters parameters(*model);
        std::cout << "real artifact: the logical parameter model covers "
                  << parameters.text.layers.size() << " layers\n";
    } catch (const std::exception& error) {
        const std::string message = error.what();
        std::cout << "real artifact stopped at: " << message << '\n';
        check(message.find("binding") == std::string::npos &&
                  message.find("Use") == std::string::npos,
              "the parameter model of the real artifact does not stop on a binding or a Use");
    }
}

// The conversion output of the real model. Binding and Use names are what this stage owns, and the
// logical parameter model is the next gate; both are reported verbatim.
void check_real_artifact() {
    const char* path = std::getenv("NINFER_QWEN4EXP_ARTIFACT");
    if (path == nullptr) {
        std::cout << "SKIP real artifact (NINFER_QWEN4EXP_ARTIFACT not set)\n";
        return;
    }
    const std::filesystem::path artifact_path(path);
    {
        artifact::Reader reader{artifact_path};
        try {
            auto plan = qwen::plan_load(reader);
            std::cout << "real artifact: " << plan.weights().text.layers.size()
                      << " layers bound\n";
        } catch (const std::exception& error) {
            const std::string message = error.what();
            std::cout << "real artifact stopped at: " << message << '\n';
            check(message.find("binding") == std::string::npos &&
                      message.find("Use") == std::string::npos,
                  "every binding and Use of the real artifact resolves under its aliased name");
            return;
        }
    }
    check_real_artifact_parameters(artifact_path);
}

} // namespace

int main() {
    try {
        check_config_and_bindings();
        check_qwen3_5_names_unchanged();
        check_misspelled_binding();
        check_ambiguous_binding();
        check_ambiguous_use();
        check_missing_use();
        check_real_artifact();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    if (failures) { return 1; }
    std::cout << "OK Qwen4Exp text config parse, layer mapping, aliases and bank bindings\n";
    return 0;
}
