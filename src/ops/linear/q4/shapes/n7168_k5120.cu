#include "ops/linear/q4/q4_shapes.h"

namespace ninfer::ops::detail {
Q4Launch select_q4_n7168_k5120(std::int32_t tokens) {
    // Selected with complete-Op cold CUDA Graph measurements on RTX 5090.
    if (tokens <= 1) return launch_q4_a16_gemv_r1_w8_direct;
    if (tokens <= 4) return launch_q4_a16_sliced_r16_t8_capacity4;
    if (tokens <= 8) return launch_q4_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q4_a16_sliced_r16_t16_w4_s2;
#if defined(NINFER_ARCH_SM89)
    // Measured on RTX 4090 (2026-11-06, linear_bench, min of repeat=200, flush 96 MiB):
    // - (16,24]: the 5090 rung sliced_k5120_t24 costs 51.2/53.3/57.3 us at T=17/20/24, while
    //   the t32 sliced kernel already used over (24,96] runs the same span in one pass at
    //   39.9/39.9/41.0 us (-22.0%..-28.6%), so sm_89 merges the whole (16,96] span.
    // - (112,192]: the 5090 rung mma_r64_t48 needs three 48-token passes over (112,128] and
    //   costs 144.4/145.4/145.4 us at T=113/120/128, while the t64_k128 kernel covers the
    //   same span in two passes at 97.3/98.3/99.3 us (-32.6%..-31.7%); it already serves
    //   (128,192], so sm_89 merges (112,192].
    // - (192,320]: the 5090 rungs r32_t96_k128 (192,288] and r64_t128 (288,384] cost
    //   226.3/229.4/233.5 us at T=193/224/256 and 322.6/323.6/313.3 us at T=289/304/320,
    //   while the t64_k128 kernel runs the same span at 182.3/181.2/185.3 and
    //   227.3/228.2/234.5 us (-29.5%..-19.5%). T=321-384 keeps r64_t128: the t64 tile
    //   needs a 6th pass there and the t128 kernel's 3-pass cost falls to 266.3 us at
    //   T=352, ahead of the t64 extrapolation.
    if (tokens <= 96) return launch_q4_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 112) return launch_q4_a16_mma_r32_t64_k64_wr16_wt32_s2_a2_b2;
    if (tokens <= 320) return launch_q4_a16_mma_r64_t64_k128_s2_a1;
#else
    if (tokens <= 24) return launch_q4_a16_sliced_k5120_t24;
    if (tokens <= 96) return launch_q4_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 112) return launch_q4_a16_mma_r32_t64_k64_wr16_wt32_s2_a2_b2;
    if (tokens <= 128) return launch_q4_a16_mma_r64_t48;
    if (tokens <= 192) return launch_q4_a16_mma_r64_t64_k128_s2_a1;
    if (tokens <= 288) return launch_q4_a16_mma_r32_t96_k128_s2_a1_b2;
#endif
    if (tokens <= 384) return launch_q4_a16_mma_r64_t128;
    if (tokens <= 448) return launch_q4_a16_mma_r64_t64_k128_s2_a1;
    if (tokens <= 576) return launch_q4_a16_mma_r32_t96_k128_s2_a1_b2;
    return launch_q4_a16_mma_r64_t128;
}
} // namespace ninfer::ops::detail
