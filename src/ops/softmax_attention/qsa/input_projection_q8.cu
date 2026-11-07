#include "ops/softmax_attention/qsa/launch.h"
#include "ops/linear/q8/q8_gemv_launch.cuh"
#include "ops/linear/q8/q8_grouped_sliced_k_launch.cuh"
#include "ops/linear/q8/q8_mma_launch.cuh"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"
#include "ops/linear/q8/shapes/n12288_k2560.cuh"

namespace ninfer::ops::detail::qsa {
namespace {

struct Routes {
    Q8LinearOperands operands;
    LinearBf16SegmentedOutput<6144, 6144> output;
    cudaStream_t stream;

    template <class S>
    void gemv() const {
        launch_q8_a16_gemv<S>(operands, output, LinearIdentityEpilogue{}, stream);
    }

    template <class S>
    void sliced() const {
        launch_q8_a16_sliced_k_mma<S>(operands, output, LinearIdentityEpilogue{}, stream);
    }

    template <class S>
    void grouped() const {
        launch_q8_a16_grouped_sliced_k_mma<S>(operands, output, LinearIdentityEpilogue{}, stream);
    }

    template <class S>
    void mma() const {
        launch_q8_a16_mma<S>(operands, output, LinearIdentityEpilogue{}, stream);
    }
};

} // namespace

void launch_query_gate(const Tensor& x, const Weight& weight, Tensor& query, Tensor& gate,
                       cudaStream_t stream) {
    const LinearBf16SegmentedOutput<6144, 6144> output{
        {static_cast<__nv_bfloat16*>(query.data), static_cast<__nv_bfloat16*>(gate.data)}};
    q8_n12288_k2560::select(x.ne[1], Routes{q8_linear_operands(x, weight), output, stream});
}

} // namespace ninfer::ops::detail::qsa
