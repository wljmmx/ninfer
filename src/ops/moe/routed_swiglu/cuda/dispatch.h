#pragma once

#include "ops/moe/kernels/cuda/groupwise_int/launch.h"
#include "ops/moe/kernels/cuda/block_scaled/launch.h"

#include <span>

namespace ninfer::ops::detail::moe_routed {

inline constexpr int kSliceTokens = 4096;

namespace kernel = moe_cuda;

struct Plan {
    kernel::ExpertImplementation implementation = kernel::ExpertImplementation::GroupwiseInt;
    kernel::groupwise_int::Plan integer;
    kernel::block_scaled::Plan block_scaled;
    // GgmlBlock: the only numbers its scratch depends on. A layer's token slice does not change
    // them - the staging area is per CTA, not per token.
    struct Ggml {
        int top_k        = 0;
        int intermediate = 0;
        bool shared      = false; // the shared-expert term is part of the call
    } ggml;
};

struct ShapePolicy {
    Plan (*select)(int tokens, bool shared);
    std::span<const int> allocation_boundaries;
};

const ShapePolicy& shape_dispatch(const RoutedSwiGLUProfile& profile);
const ShapePolicy& q4_q5_h2048_i512_e256_k8();
const ShapePolicy& q4_q6_h2048_i512_e256_k8();
const ShapePolicy& q4_q5_h2560_i640_e512_k10();
const ShapePolicy& q4_q6_h2560_i640_e512_k10();
const ShapePolicy& q8_q8_h2048_i512_e256_k8();
const ShapePolicy& nvfp4_h2560_i640_e512_k10();
const ShapePolicy& fp8_h2560_i640_e512_k10();
// The Qwen4Exp GGML row: H=2560, I=640, E=512, K=10, experts stored as GGML block streams.
const ShapePolicy& ggml_h2560_i640_e512_k10();

} // namespace ninfer::ops::detail::moe_routed
