#include "ops/linear/q4/q4_shapes.h"

namespace ninfer::ops::detail {
Q4Launch select_q4_n6144_k5120(std::int32_t tokens) {
    // Selected with complete-Op cold CUDA Graph measurements on RTX 5090.
    if (tokens <= 1) return launch_q4_a16_gemv_r1_w8_direct;
    if (tokens <= 4) return launch_q4_a16_sliced_r16_t8_capacity4;
    if (tokens <= 8) return launch_q4_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q4_a16_sliced_k5120_t16;
    if (tokens <= 24) return launch_q4_a16_sliced_k5120_t24;
    if (tokens <= 48) return launch_q4_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 64) return launch_q4_a16_mma_r32_t32_k128_s2_a2;
    if (tokens <= 96) return launch_q4_a16_sliced_r32_t32_w4_s1;
#if defined(NINFER_ARCH_SM89)
    // Measured on RTX 4090 (2026-11-06, linear_bench, min of repeat=200, flush 96 MiB):
    // - (128,192]: the 5090 rung mma_r32_t64_k64 costs 198.7/183.3/184.3 us at T=129/160/191
    //   there (its 3-pass predicated variant; the T=192 full tile lands at 144.4), while
    //   sliced_r32_t32_w4_s1 runs the span at 126.9/130.0/148.5 us (-36.1%..-19.4%); at
    //   T=192 itself all three of sliced/r32_t96/the old kernel measure 140.3-147.5 us,
    //   i.e. within noise of each other. T=97-128 keeps the 5090 kernel: sliced measured
    //   +4.3% at T=97, r32_t96 +50%.
    // - (640,767]: the 5090 tail r64_t128 costs 509.9/490.5/472.1/474.1/476.2 us at
    //   T=641/672/704/736/767 (the 6-pass predicated variant of the ragged span), while
    //   mma_r64_t64_k128 runs 395.3/396.3/401.4/372.7/377.9 us (-22.5%..-15.0%). T=768
    //   stays on t128: its full 6-pass tile measured 383.0 us vs 406.5 us on the t64 tile.
    if (tokens <= 128) return launch_q4_a16_mma_r32_t64_k64_wr16_wt32_s2_a2_b2;
    if (tokens <= 192) return launch_q4_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 384) return launch_q4_a16_mma_r64_t128;
    if (tokens <= 767) return launch_q4_a16_mma_r64_t64_k128_s2_a1;
#else
    if (tokens <= 192) return launch_q4_a16_mma_r32_t64_k64_wr16_wt32_s2_a2_b2;
    if (tokens <= 384) return launch_q4_a16_mma_r64_t128;
    if (tokens <= 640) return launch_q4_a16_mma_r64_t64_k128_s2_a1;
#endif
    return launch_q4_a16_mma_r64_t128;
}
} // namespace ninfer::ops::detail
