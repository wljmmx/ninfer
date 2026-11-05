#include "ops/linear/q4/q4_shapes.h"

namespace ninfer::ops::detail {
Q4Launch select_q4_n1024_k5120(std::int32_t tokens) {
    // Selected with complete-Op cold CUDA Graph measurements on RTX 5090.
    if (tokens <= 1) return launch_q4_a16_gemv_r1_w8_direct;
    if (tokens <= 8) return launch_q4_a16_simt_r4_t4_w2_g8_s2;
    if (tokens <= 32) return launch_q4_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 64) return launch_q4_a16_sliced_r16_t32_w4_s2;
    if (tokens <= 80) return launch_q4_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 320) return launch_q4_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 576) return launch_q4_a16_sliced_r32_t32_w2_s2;
    if (tokens <= 768) return launch_q4_a16_mma_r32_t64_k64_wr16_wt32_s2_a2_b2;
    if (tokens <= 1280) return launch_q4_a16_mma_r32_t128_k64_s2_a2;
#if defined(NINFER_ARCH_SM89)
    // Measured on RTX 4090 (2026-11-05, linear_bench, min of repeat=200): the 5090 rung
    // mma_r32_t64_k64 over (1280, 1344] costs 223-224 us there (T=1281/1343), while both
    // neighbours are far ahead (T=1280 on r32_t128: 160.8 us, T=1345 on r64_t128: 174.1 us
    // and 175.1 us at T=2048), so sm_89 routes that span straight to the tail kernel.
    return launch_q4_a16_mma_r64_t128;
#else
    if (tokens <= 1344) return launch_q4_a16_mma_r32_t64_k64_wr16_wt32_s2_a2_b2;
    return launch_q4_a16_mma_r64_t128;
#endif
}
} // namespace ninfer::ops::detail
