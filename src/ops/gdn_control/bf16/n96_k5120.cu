#include "ops/gdn_control/bf16/small_split.cuh"
#include "ops/gdn_control/bf16/norm.cuh"
#include "ops/gdn_control/bf16/mma_launch.cuh"
#include "ops/gdn_control/bf16/gemv.cuh"

namespace ninfer::ops::detail::gdn_control {
void launch_n96_k5120(const Inputs& in, GdnControlOutputs out, Plan plan, Scratch scratch,
                      DeviceExecutionView e) {
    using Geometry = MmaGeometry<48, 5120, 128>;
    if (plan.kernel == Kernel::Gemv) {
        gemv_kernel<48, 5120><<<96, 256, 0, e.stream>>>(
            static_cast<const __nv_bfloat16*>(in.x.data), static_cast<const __nv_bfloat16*>(in.a),
            static_cast<const __nv_bfloat16*>(in.b), static_cast<const float*>(in.a_log.data),
            static_cast<const float*>(in.dt_bias.data), static_cast<float*>(out.g.data),
            static_cast<float*>(out.beta.data));
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (plan.kernel == Kernel::SmallSplit) {
        small_partial_kernel<8, 512, 4><<<dim3(24, 10, 1), 128, 0, e.stream>>>(
            static_cast<const __nv_bfloat16*>(in.x.data), static_cast<const __nv_bfloat16*>(in.a),
            static_cast<const __nv_bfloat16*>(in.b), static_cast<float*>(scratch.partial.data),
            in.x.ne[1]);
        CUDA_CHECK(cudaGetLastError());
        small_reduce_kernel<<<(48 * in.x.ne[1] + 127) / 128, 128, 0, e.stream>>>(
            static_cast<const float*>(scratch.partial.data),
            static_cast<const float*>(in.a_log.data), static_cast<const float*>(in.dt_bias.data),
            static_cast<float*>(out.g.data), static_cast<float*>(out.beta.data), in.x.ne[1]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    bool launched = false;
    switch (plan.split_k) {
    case 8:
        launched = launch_mma<Geometry, 8, 8>(in, out, scratch, e);
        break;
    case 4:
        launched = launch_mma<Geometry, 4, 16>(in, out, scratch, e);
        break;
    case 2:
        launched = launch_mma<Geometry, 2, 16>(in, out, scratch, e);
        break;
    case 1:
        launched = launch_mma<Geometry, 1, 8>(in, out, {}, e);
        break;
    default:
        throw std::logic_error("gdn_control: invalid 96x5120 plan");
    }
    if (!launched) (void)launch_mma<Geometry, 1, 8>(in, out, {}, e);
}

void launch_norm_n96_k5120(const Inputs& in, const Norm& norm, GdnControlOutputs out,
                           cudaStream_t stream) {
    const auto launch = [&]<int Tile, int Threads>() {
        norm_simt_kernel<Tile, Threads>
            <<<dim3(48, (in.x.ne[1] + Tile - 1) / Tile), Threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(in.x.data),
                static_cast<const __nv_bfloat16*>(norm.weight.data),
                static_cast<const __nv_bfloat16*>(in.a), static_cast<const __nv_bfloat16*>(in.b),
                static_cast<const float*>(in.a_log.data),
                static_cast<const float*>(in.dt_bias.data),
                static_cast<__nv_bfloat16*>(norm.normalized.data), static_cast<float*>(out.g.data),
                static_cast<float*>(out.beta.data), in.x.ne[1], norm.eps);
    };
    const int columns = in.x.ne[1];
    if (columns <= 2)
        launch.template operator()<1, 512>();
    else if (columns <= 14)
        launch.template operator()<2, 512>();
    else if (columns <= 28)
        launch.template operator()<2, 256>();
    else
        launch.template operator()<4, 256>();
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail::gdn_control
