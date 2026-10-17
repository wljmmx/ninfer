#pragma once
#include "ops/linear/kernels/fp8/fp8_launch.h"
#include "ops/linear/kernels/fp8/fp8_template_launch.cuh"
#include "ops/linear/kernels/fp8/fp8_instances.cuh"
#include <algorithm>

namespace ninfer::ops::detail {
template <class Geometry, class Schedule>
void fp8_linear_a16_gemv(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_fp8_a16_gemv<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(
        fp8_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        LinearIdentityEpilogue{}, stream);
}

template <class Geometry, int Capacity, class Schedule, bool Exact = false>
void fp8_linear_a16_simt(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_fp8_a16_simt<Fp8ScheduleInstance<Schedule, Geometry::kInputRows, Capacity, Exact>>(
        fp8_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        LinearIdentityEpilogue{}, stream);
}

template <class Geometry, class Schedule>
void fp8_linear_a16_mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_fp8_a16_mma<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(
        fp8_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        LinearIdentityEpilogue{}, stream);
}

template <class Geometry, class Schedule>
void fp8_linear_a16_sliced_k(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(
        fp8_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        LinearIdentityEpilogue{}, stream);
}

#if NINFER_CUDA_HAS_BLOCK_SCALE_MMA
template <class Geometry, class Schedule>
void launch_fp8_a8(const Tensor& x, const Weight& w, Tensor& out, Fp8A8Workspace scratch,
                   cudaStream_t stream) {
    launch_fp8_a8_quantize(x, w, scratch, stream);
    launch_fp8_a8_mma<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(
        fp8_a8_operands(w, scratch, x.ne[1]),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n}, LinearIdentityEpilogue{},
        stream);
}
#else
// port (sm_89): the fp8 A8 kernels use the block-scaled fp8 MMA (.kind::mxf8f6f4), which Ada lacks, so
// this route cannot exist here. The shapes forward to their A16 selector before reaching it.
#include <stdexcept>

template <class Geometry, class Schedule>
void launch_fp8_a8(const Tensor&, const Weight&, Tensor&, Fp8A8Workspace, cudaStream_t) {
    throw std::logic_error("fp8 A8 requires a block-scaled MMA target (sm_100+)");
}
#endif

#if NINFER_CUDA_HAS_TMA
template <class Geometry, class Schedule>
void launch_fp8_a8_tma(const Tensor& x, const Weight& w, Tensor& out, Fp8A8Workspace scratch,
                       cudaStream_t stream) {
    launch_fp8_a8_quantize(x, w, scratch, stream);
    launch_fp8_a8_tma_mma<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(
        fp8_a8_operands(w, scratch, x.ne[1]),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n}, LinearIdentityEpilogue{},
        stream, scratch.partials);
}
#else
// port (sm_89): the TMA kernels need sm_90, so this route cannot exist on Ada. Every shape's launch_a8
// forwards to its A16 selector before this point, and the shapes that still name the route only need
// it to be defined.
#include <stdexcept>

template <class Geometry, class Schedule>
void launch_fp8_a8_tma(const Tensor&, const Weight&, Tensor&, Fp8A8Workspace, cudaStream_t) {
    throw std::logic_error("fp8 A8 TMA requires sm_90+; this build uses the A16 path");
}
#endif
} // namespace ninfer::ops::detail
