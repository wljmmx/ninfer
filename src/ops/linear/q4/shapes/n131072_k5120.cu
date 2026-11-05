#include "ops/linear/q4/q4_shapes.h"

namespace ninfer::ops::detail {
Q4Launch select_q4_n131072_k5120(std::int32_t tokens) {
    // Selected with complete-Op cold CUDA Graph measurements on RTX 5090.
    if (tokens <= 1) return launch_q4_a16_gemv_r4_w1_direct;
    if (tokens <= 4) return launch_q4_a16_sliced_k5120_t4;
    if (tokens <= 8) return launch_q4_a16_sliced_r32_t8_w4_s2;
    if (tokens <= 16) return launch_q4_a16_sliced_r32_t16_w4_s1;
    if (tokens <= 32) return launch_q4_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 64) return launch_q4_a16_mma_r64_t64_k128_s2_a1;
#if defined(NINFER_ARCH_SM89)
    // Measured on RTX 4090 (2026-11-05, linear_bench, min of repeat=200): the 5090 rung
    // mma_r64_t80 over (64, 80] costs 1297-1305 us there (T=65/79/80); the 96-wide r32
    // tile covers the same span in one tile-tile pass: forced-candidate measurement on
    // sm_89 gives 1044.5 us at T=65, 1056.8 at T=72, 1094.7 at T=80 (-19.6%..-19.0%).
    if (tokens <= 80) return launch_q4_a16_mma_r32_t96_k128_s2_a1_b2;
#else
    if (tokens <= 80) return launch_q4_a16_mma_r64_t80;
#endif
    if (tokens <= 96) return launch_q4_a16_mma_r32_t96_k128_s2_a1_b2;
    if (tokens <= 112) return launch_q4_a16_mma_r64_t112;
    return launch_q4_a16_mma_r64_t128;
}
} // namespace ninfer::ops::detail
