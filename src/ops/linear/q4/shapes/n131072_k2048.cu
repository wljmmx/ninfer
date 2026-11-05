#include "ops/linear/q4/q4_shapes.h"

namespace ninfer::ops::detail {
Q4Launch select_q4_n131072_k2048(std::int32_t tokens) {
    // Selected with complete-Op cold CUDA Graph measurements on RTX 5090.
    if (tokens <= 1) return launch_q4_a16_simt_r4_t1_w2_g8_s2;
    if (tokens <= 4) return launch_q4_a16_sliced_k2048_t4;
    if (tokens <= 8) return launch_q4_a16_sliced_r32_t8_w4_s2;
    if (tokens <= 16) return launch_q4_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 32) return launch_q4_a16_sliced_r32_t32_w2_s2;
    if (tokens <= 64) return launch_q4_a16_sliced_r32_t64_w2_s1;
#if defined(NINFER_ARCH_SM89)
    // Measured on RTX 4090 (2026-11-06, linear_bench, min of repeat=200, flush 96 MiB): the
    // 5090 rung mma_r64_t72 over (64,72] costs 449.5/440.3/441.5/442.6 us at T=65/68/71/72
    // there, while the same span on the t80 kernel measures 371.7/367.6/370.7/370.9 us
    // (-17.3%..-16.0%); the (72,80] rung already runs t80, so sm_89 merges the whole span.
    if (tokens <= 80) return launch_q4_a16_mma_r64_t80;
#else
    if (tokens <= 72) return launch_q4_a16_mma_r64_t72;
    if (tokens <= 80) return launch_q4_a16_mma_r64_t80;
#endif
    if (tokens <= 96) return launch_q4_a16_mma_r32_t96_k128_s2_a1_b2;
    if (tokens <= 112) return launch_q4_a16_mma_r64_t112;
    if (tokens <= 120) return launch_q4_a16_mma_r64_t120;
    return launch_q4_a16_mma_r64_t128;
}
} // namespace ninfer::ops::detail
