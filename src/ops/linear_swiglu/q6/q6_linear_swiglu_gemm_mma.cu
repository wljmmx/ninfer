#include "core/weight.h"
#include "ops/linear_swiglu/q6/q6_linear_swiglu_kernels.h"
#include "ops/linear_swiglu/folded_swiglu_high_launch.cuh"
#include "ops/linear/q6/q6_rowsplit_storage.cuh"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

// Same block/warp tiling as the Q4 SwiGLU route: a column-tile-major CTA order over token tiles on
// grid.x so the CTAs of one wave share each weight block in L2. Q6 adds one high-bit byte per eight
// codes, which the kernel stages next to them (HB = 16).
using GateUpC8Cfg   = GemmCfg<64, 8, 64, 32, 8, 2, 1, false, true, true>;
using GateUpC48Cfg  = GemmCfg<64, 48, 64, 64, 8, 2, 1, false, true, true>;
using GateUpC128Cfg = GemmCfg<64, 128, 64, 32, 32, 2, 1, false, true, true>;

constexpr int kHighBytesPerGroup = Q6RowSplitStorage::kHighBytesPerGroup;

} // namespace

void q6_linear_swiglu_mma_c8_launch(const Tensor& x, const Weight& w, Tensor& out,
                                    cudaStream_t stream) {
    launch_folded_high_route<Q6MmaDecodeAtom, kHighBytesPerGroup, GateUpC8Cfg>(x, w, out, stream);
}

void q6_linear_swiglu_mma_c48_launch(const Tensor& x, const Weight& w, Tensor& out,
                                     cudaStream_t stream) {
    launch_folded_high_route<Q6MmaDecodeAtom, kHighBytesPerGroup, GateUpC48Cfg>(x, w, out, stream);
}

void q6_linear_swiglu_mma_c128_launch(const Tensor& x, const Weight& w, Tensor& out,
                                      cudaStream_t stream) {
    launch_folded_high_route<Q6MmaDecodeAtom, kHighBytesPerGroup, GateUpC128Cfg>(x, w, out,
                                                                                 stream);
}

} // namespace ninfer::ops::detail
