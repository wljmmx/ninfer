#include "ops/softmax_attention/qsa/launch.h"
#include "ops/linear/bf16/bf16_template_launch.cuh"
#include "ops/linear/bf16/shapes/n1664_k2560.cuh"
#include "ops/linear/bf16/shapes/n13952_k2560.cuh"

namespace ninfer::ops::detail::qsa {
namespace {

// The measured QSA projection ladders live with their linear shapes; this binder only
// swaps the output binding for a set of disjoint segments.
template <int... Rows>
struct SegmentedRoutes {
    Bf16A16Operands operands;
    LinearBf16SegmentedOutput<Rows...> output;
    cudaStream_t stream;

    template <class S>
    void gemv() const {
        launch_bf16_a16_gemv<S>(operands, output, LinearIdentityEpilogue{}, stream);
    }

    template <class S>
    void sliced() const {
        launch_bf16_a16_sliced_k_mma<S>(operands, output, LinearIdentityEpilogue{}, stream);
    }

    template <class S>
    void mma() const {
        launch_bf16_a16_mma<S>(operands, output, LinearIdentityEpilogue{}, stream);
    }

#if defined(NINFER_ENABLE_TMA)
    template <class S>
    void tma() const {
        launch_bf16_a16_tma_mma<S>(operands, output, LinearIdentityEpilogue{}, stream);
    }
#endif
};

} // namespace

void launch_kv_index(const Tensor& x, const Weight& weight, Tensor& key, Tensor& value,
                     Tensor& index_query, Tensor& index_key, cudaStream_t stream) {
    const LinearBf16SegmentedOutput<512, 512, 512, 128> output{
        {static_cast<__nv_bfloat16*>(key.data), static_cast<__nv_bfloat16*>(value.data),
         static_cast<__nv_bfloat16*>(index_query.data),
         static_cast<__nv_bfloat16*>(index_key.data)}};
    bf16_n1664_k2560::select(
        x.ne[1],
        SegmentedRoutes<512, 512, 512, 128>{bf16_a16_operands(x, weight), output, stream});
}

void launch_joint(const Tensor& x, const Weight& weight, Tensor& query, Tensor& key, Tensor& gate,
                  Tensor& value, Tensor& index_query, Tensor& index_key, cudaStream_t stream) {
    const LinearBf16SegmentedOutput<6144, 512, 6144, 512, 512, 128> output{
        {static_cast<__nv_bfloat16*>(query.data), static_cast<__nv_bfloat16*>(key.data),
         static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(value.data),
         static_cast<__nv_bfloat16*>(index_query.data),
         static_cast<__nv_bfloat16*>(index_key.data)}};
    bf16_n13952_k2560::select(
        x.ne[1],
        SegmentedRoutes<6144, 512, 6144, 512, 512, 128>{bf16_a16_operands(x, weight), output,
                                                        stream});
}

} // namespace ninfer::ops::detail::qsa
