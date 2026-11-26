#include "artifact/formats.h"

#include "artifact/schema.h"

#include <array>

namespace ninfer::artifact {
namespace {

constexpr std::array kFormats = {
    std::pair{QType::BF16, std::string_view{"bf16"}},
    std::pair{QType::FP32, std::string_view{"fp32"}},
    std::pair{QType::INT32, std::string_view{"int32"}},
    std::pair{QType::Q4_G64_FP16, std::string_view{"q4_g64_fp16"}},
    std::pair{QType::Q5_G64_FP16, std::string_view{"q5_g64_fp16"}},
    std::pair{QType::Q6_G64_FP16, std::string_view{"q6_g64_fp16"}},
    std::pair{QType::Q8_G32_FP16, std::string_view{"q8_g32_fp16"}},
    std::pair{QType::NVFP4, std::string_view{"nvfp4"}},
    std::pair{QType::FP8_E4M3FN_ROW_BF16, std::string_view{"fp8_e4m3fn_row_bf16"}},
    std::pair{QType::NF4_G16_E4M3FN, std::string_view{"nf4_g16_e4m3fn"}},
    // port (sm_89) T117: the block-scaled FP8 weights the expert path already encodes. Without a
    // name a container cannot carry them at all, and the 64x64 form T116 added would stay
    // unreachable from an artifact.
    std::pair{QType::FP8_E4M3FN_BLOCK128X128_BF16,
              std::string_view{"fp8_e4m3fn_block128x128_bf16"}},
    std::pair{QType::FP8_E4M3FN_BLOCK64X64_BF16,
              std::string_view{"fp8_e4m3fn_block64x64_bf16"}},
    // port (sm_89): GGML block encodings, carried verbatim from GGUF. BF16/F32/INT32 GGML tensors
    // use the names above; these are the ones with no native equivalent.
    std::pair{QType::GGML_F16, std::string_view{"ggml_f16"}},
    std::pair{QType::GGML_Q4_K, std::string_view{"ggml_q4_k"}},
    std::pair{QType::GGML_Q5_K, std::string_view{"ggml_q5_k"}},
    std::pair{QType::GGML_Q6_K, std::string_view{"ggml_q6_k"}},
    std::pair{QType::GGML_IQ4_NL, std::string_view{"ggml_iq4_nl"}},
    std::pair{QType::GGML_IQ4_XS, std::string_view{"ggml_iq4_xs"}},
    std::pair{QType::GGML_IQ3_S, std::string_view{"ggml_iq3_s"}},
    std::pair{QType::GGML_IQ3_XXS, std::string_view{"ggml_iq3_xxs"}},
    std::pair{QType::GGML_IQ2_XXS, std::string_view{"ggml_iq2_xxs"}},
    std::pair{QType::GGML_IQ2_XS, std::string_view{"ggml_iq2_xs"}},
    std::pair{QType::GGML_IQ2_S, std::string_view{"ggml_iq2_s"}},
    std::pair{QType::GGML_Q2_0, std::string_view{"ggml_q2_0"}},
};
constexpr std::array kLayouts = {
    std::pair{QuantLayout::Contiguous, std::string_view{"contiguous_le_v1"}},
    std::pair{QuantLayout::RowSplit, std::string_view{"row_split_k128_v1"}},
    std::pair{QuantLayout::RowScale, std::string_view{"row_scale_v1"}},
    std::pair{QuantLayout::RowPagedG16_4K, std::string_view{"row_paged_g16_4k_v1"}},
    std::pair{QuantLayout::BlockScaleK16M128x4, std::string_view{"block_scale_k16_m128x4_v1"}},
    std::pair{QuantLayout::BatchedBlockScaleK16M128x4,
              std::string_view{"batched_block_scale_k16_m128x4_v1"}},
    std::pair{QuantLayout::BlockScaleM128K128, std::string_view{"block_scale_m128_k128_v1"}},
    std::pair{QuantLayout::BlockScaleM64K64, std::string_view{"block_scale_m64_k64_v1"}},
    std::pair{QuantLayout::GgmlBlock, std::string_view{"ggml_block_v1"}},
};

} // namespace

QType parse_format(std::string_view name) {
    for (const auto& [format, spelling] : kFormats) {
        if (spelling == name) { return format; }
    }
    throw ArtifactError("unknown tensor format: " + std::string(name));
}

QuantLayout parse_layout(std::string_view name) {
    for (const auto& [layout, spelling] : kLayouts) {
        if (spelling == name) { return layout; }
    }
    throw ArtifactError("unknown tensor layout: " + std::string(name));
}

std::string_view format_name(QType format) noexcept {
    for (const auto& [value, spelling] : kFormats) {
        if (value == format) { return spelling; }
    }
    return {};
}

std::string_view layout_name(QuantLayout layout) noexcept {
    for (const auto& [value, spelling] : kLayouts) {
        if (value == layout) { return spelling; }
    }
    return {};
}

} // namespace ninfer::artifact
