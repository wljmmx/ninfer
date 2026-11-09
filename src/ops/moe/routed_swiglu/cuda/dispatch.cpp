#include "ops/moe/routed_swiglu/cuda/dispatch.h"
#include "ops/moe/operands.h"

namespace ninfer::ops::detail::moe_routed {

const ShapePolicy& shape_dispatch(const RoutedSwiGLUProfile& profile) {
    validate_expert_profile(profile);
    // port (sm_89): the q4/q5 and q4/q6 families now name two geometries. validate_expert_profile
    // has already refused anything that is not one of the two rows, so the hidden size alone picks
    // the policy; a third q4 row would add its case here.
    const bool wide = profile.ffn.hidden == 2560;
    switch (profile.ffn.down_type) {
    case QType::FP8_E4M3FN_BLOCK128X128_BF16:
        return fp8_h2560_i640_e512_k10();
    case QType::NVFP4:
        return nvfp4_h2560_i640_e512_k10();
    case QType::Q5_G64_FP16:
        return wide ? q4_q5_h2560_i640_e512_k10() : q4_q5_h2048_i512_e256_k8();
    case QType::Q6_G64_FP16:
        return wide ? q4_q6_h2560_i640_e512_k10() : q4_q6_h2048_i512_e256_k8();
    case QType::Q8_G32_FP16:
        return q8_q8_h2048_i512_e256_k8();
    default:
        throw std::invalid_argument("unsupported MoE expert format");
    }
}

} // namespace ninfer::ops::detail::moe_routed
