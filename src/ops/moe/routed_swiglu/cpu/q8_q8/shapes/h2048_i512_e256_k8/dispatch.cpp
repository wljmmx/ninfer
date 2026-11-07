#include "ops/moe/routed_swiglu/cpu/dispatch.h"
#include "ops/moe/kernels/cpu/groupwise_int/work.h"

namespace ninfer::ops::detail::moe_routed_cpu {
namespace {
namespace kernel          = moe_cpu::groupwise_int;
constexpr int token_block = 8;

// port (sm_89, P4-B): the q8_q8 shape — Q8_G32_FP16 gate/up and down planes (one int8 code per
// dimension, FP16 scale per 32-value group, ops/common/quant/integer.cuh Q8RowSplitStorage) at
// the q4_q5 geometry (h2048/i512/e256/k8, RowSplit). Both planes run the float decode + FMA dot
// (dot<Simd, 8, K> in groupwise_int/compute.h): the int8/int16 activation tiers and the bf16
// gate dot are q4-gate nibble paths and never apply, so the Avx512Bf16 worker is stepped down
// to Avx512 — its dpbf16 gate has no Q8 form and the float dot is the whole gain there.
kernel::Plan select_plan(CpuFeatures features) {
    kernel::Plan plan{.token_block = token_block, .down_bits = 8, .gate_bits = 8};
    switch (select_isa(features)) {
    case Isa::Avx512Bf16:
    case Isa::Avx512:
        plan.kernel      = &kernel::avx512;
        plan.max_columns = 4;
        break;
    case Isa::Avx2:
        plan.kernel      = &kernel::avx2;
        plan.max_columns = 2;
        break;
    }
    return plan;
}

std::size_t capacity(int tokens) { return kernel::capacity(tokens, token_block); }

void launch(const moe_cpu::Invocation& args, MemoryArena& arena, CpuExecutionView execution) {
    kernel::launch(args, select_plan(execution.features()), arena, execution);
}
} // namespace

const ShapePolicy& q8_q8_h2048_i512_e256_k8() {
    static const ShapePolicy policy{.capacity = capacity, .launch = launch};
    return policy;
}
} // namespace ninfer::ops::detail::moe_routed_cpu
