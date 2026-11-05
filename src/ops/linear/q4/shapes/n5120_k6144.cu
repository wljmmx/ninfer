#include "ops/linear/q4/q4_shapes.h"

namespace ninfer::ops::detail {
Q4Launch select_q4_n5120_k6144(std::int32_t tokens) {
    // Selected with complete-Op cold CUDA Graph measurements on RTX 5090.
    if (tokens <= 1) return launch_q4_a16_gemv_r1_w8_k6144;
    if (tokens <= 4) return launch_q4_a16_sliced_r16_t8_capacity4;
    if (tokens <= 8) return launch_q4_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q4_a16_sliced_r16_t16_w4_s2;
#if defined(NINFER_ARCH_SM89)
    // Measured on RTX 4090 (2026-11-06, linear_bench, min of repeat=200, flush 96 MiB):
    // - (32,64]: the 5090 rung sliced_r32_t32_w4_s2 needs two passes there and costs
    //   68.7/70.7/71.7/71.7 us at T=40/48/56/64, while mma_r32_t32_k128 covers the same
    //   span in two passes at 58.4/60.3/60.5/59.4 us (-15.1%..-17.1%). T=17-24 stays on the
    //   sliced kernel (the mma tile measured +7%/+2% there) and T=32 keeps the 5090 choice
    //   (mma measured -8.3%, under the bar).
    // - (128,256]: the 5090 rung mma_r32_t64_k64_s3 costs 215.0/214.0/216.1/220.2/222.2 us
    //   at T=129/144/160/176/191 (its 3-pass predicated variant) and 206.8/208.9/208.9 us at
    //   T=193/224/256 on r64_t128, while sliced_r32_t32_w2_s2 runs the whole span at
    //   135.0/134.1/133.1/137.2/139.3 and 173.1/172.0/177.2 us (-38.4%..-15.2%).
    // - (64,96] and (96,128] keep their 5090 kernels: r32_t96_k128 measured -1.0%/+3.3%
    //   and +15.3%/+52% at T=65/80/96/97 - no win.
    if (tokens <= 32) return launch_q4_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 96) return launch_q4_a16_mma_r32_t32_k128_s2_a2;
    if (tokens <= 256) return launch_q4_a16_sliced_r32_t32_w2_s2;
#else
    if (tokens <= 64) return launch_q4_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 96) return launch_q4_a16_mma_r32_t32_k128_s2_a2;
    if (tokens <= 128) return launch_q4_a16_sliced_r32_t32_w2_s2;
    if (tokens <= 192) return launch_q4_a16_mma_r32_t64_k64_wr16_wt32_s3_a3_b2;
#endif
    return launch_q4_a16_mma_r64_t128;
}
} // namespace ninfer::ops::detail
