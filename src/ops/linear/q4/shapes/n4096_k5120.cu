#include "ops/linear/q4/q4_shapes.h"

namespace ninfer::ops::detail {
Q4Launch select_q4_n4096_k5120(std::int32_t tokens) {
    // Selected with complete-Op cold CUDA Graph measurements on RTX 5090.
    if (tokens <= 1) return launch_q4_a16_gemv_r1_w8_direct;
    if (tokens <= 8) return launch_q4_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q4_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 64) return launch_q4_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 96) return launch_q4_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 128) return launch_q4_a16_sliced_r32_t32_w2_s2;
#if defined(NINFER_ARCH_SM89)
    // Measured on RTX 4090 (2026-11-06, linear_bench, min of repeat=200, flush 96 MiB): the
    // 5090 rung mma_r32_t32_k64_s3 costs 90.1/93.2/100.4 us at T=129/160/192 there, while the
    // 96-wide r32 tile covers (128,192] in two passes: 73.7/81.9/91.3 us (-18.2%..-9.0%).
    // The rest of the rung stays on the 5090 kernel: r64_t128 only won T=256 (-12.3%) and
    // lost T=129/160/192/224/288/319 (+5%..+30%), sliced_r32_t32_w2_s2 measured -4.1%/-0.1%
    // at T=256/288 and -6.9% at T=319 - all under the 10% bar.
    if (tokens <= 192) return launch_q4_a16_mma_r32_t96_k128_s2_a1_b2;
#endif
    if (tokens <= 320) return launch_q4_a16_mma_r32_t32_k64_wr16_wt16_s3_a3_b2;
    return launch_q4_a16_mma_r64_t128;
}
} // namespace ninfer::ops::detail
