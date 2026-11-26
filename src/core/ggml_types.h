#pragma once

// The GGML block types this container carries: what one encoded block costs and how many values it
// holds. Split from the decoders (core/ggml_dequant.h) so a translation unit that only needs the
// geometry does not pull the codebooks in with it.
//
// The container's `ggml_block_v1` geometry and the kernels both read this table, so a block size
// cannot disagree between the two.

#include "core/weight.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::core {

struct GgmlBlockGeometry {
    std::uint32_t block;      // values per encoded block
    std::uint32_t type_bytes; // encoded bytes per block
};

[[nodiscard]] inline GgmlBlockGeometry ggml_block_geometry(QType format) {
    switch (format) {
    case QType::GGML_F16: return {1, 2};
    case QType::GGML_Q4_K: return {256, 144};
    case QType::GGML_Q5_K: return {256, 176};
    case QType::GGML_Q6_K: return {256, 210};
    case QType::GGML_IQ4_NL: return {32, 18};
    case QType::GGML_IQ4_XS: return {256, 136};
    case QType::GGML_IQ3_S: return {256, 110};
    case QType::GGML_IQ3_XXS: return {256, 98};
    case QType::GGML_IQ2_XXS: return {256, 66};
    case QType::GGML_IQ2_XS: return {256, 74};
    case QType::GGML_IQ2_S: return {256, 82};
    case QType::GGML_Q2_0: return {64, 18};
    default: throw std::invalid_argument("weight is not a GGML block type");
    }
}

[[nodiscard]] inline bool is_ggml_block(QType format) noexcept {
    switch (format) {
    case QType::GGML_F16:
    case QType::GGML_Q4_K:
    case QType::GGML_Q5_K:
    case QType::GGML_Q6_K:
    case QType::GGML_IQ4_NL:
    case QType::GGML_IQ4_XS:
    case QType::GGML_IQ3_S:
    case QType::GGML_IQ3_XXS:
    case QType::GGML_IQ2_XXS:
    case QType::GGML_IQ2_XS:
    case QType::GGML_IQ2_S:
    case QType::GGML_Q2_0:
        return true;
    default:
        return false;
    }
}

} // namespace ninfer::core
