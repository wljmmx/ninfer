#include "ops/linear/q4/q4_shapes.h"

namespace ninfer::ops::detail {
Q4Launch select_q4_n34816_k5120(std::int32_t tokens) {
    // Selected with complete-Op cold CUDA Graph measurements on RTX 5090.
    if (tokens <= 1) return launch_q4_a16_gemv_r1_w8_direct;
    if (tokens <= 4) return launch_q4_a16_sliced_r16_t8_capacity4;
    if (tokens <= 8) return launch_q4_a16_sliced_r32_t8_w4_s2;
    if (tokens <= 16) return launch_q4_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 63) return launch_q4_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 64) return launch_q4_a16_mma_r64_t64_k128_s2_a1;
#if defined(NINFER_ARCH_SM89)
    // Measured on RTX 4090 (2026-11-06, linear_bench, min of repeat=200, flush 96 MiB): the
    // 5090 rung mma_r64_t80 over (64,80] costs 406.5/411.6/401.4 us at T=65/72/80 there,
    // while the 96-wide r32 tile covers the same span in one pass: 307.2/311.3/325.5 us
    // (-24.4%..-18.9%), mirroring the (64,80] fix already landed for n131072_k5120.
    // (The runner-up mma_r64_t72 measured 381.9/393.2 us at T=65/72 and needs two passes at
    // T>72, so it stayed behind the t96 kernel.)
    if (tokens <= 96) return launch_q4_a16_mma_r32_t96_k128_s2_a1_b2;
#else
    if (tokens <= 80) return launch_q4_a16_mma_r64_t80;
    if (tokens <= 96) return launch_q4_a16_mma_r32_t96_k128_s2_a1_b2;
#endif
    if (tokens <= 120) return launch_q4_a16_mma_r64_t120;
    if (tokens <= 128) return launch_q4_a16_mma_r64_t128;
    if (tokens <= 192) return launch_q4_a16_mma_r32_t96_k128_s2_a1_b2;
    if (tokens <= 224) return launch_q4_a16_mma_r64_t112;
    if (tokens <= 240) return launch_q4_a16_mma_r64_t120;
    if (tokens <= 256) return launch_q4_a16_mma_r64_t128;
    if (tokens <= 288) return launch_q4_a16_mma_r32_t96_k128_s2_a1_b2;
    return launch_q4_a16_mma_r64_t128;
}
} // namespace ninfer::ops::detail
