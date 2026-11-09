#include "ops/linear/q4/q4_shapes.h"

namespace ninfer::ops::detail {
// Qwen4Exp (hidden 2560): the GDN value and z projections are [6144,2560] Q4 parents.
// K=2560 is a whole number of 64- and 128-wide K tiles but not of the 5120-wide static variants,
// so the table follows n6144_k5120 with the static-K5120 sliced instances replaced by their
// generic siblings. As with n2048_k2560 the rungs are inherited for correctness first and are not
// yet measured at K=2560 on this device.
Q4Launch select_q4_n6144_k2560(std::int32_t tokens) {
    if (tokens <= 1) return launch_q4_a16_gemv_r4_w1_direct;
    if (tokens <= 4) return launch_q4_a16_sliced_r16_t8_capacity4;
    if (tokens <= 8) return launch_q4_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q4_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 32) return launch_q4_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 96) return launch_q4_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 128) return launch_q4_a16_mma_r32_t64_k64_wr16_wt32_s2_a2_b2;
    if (tokens <= 192) return launch_q4_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 384) return launch_q4_a16_mma_r64_t128;
    if (tokens <= 767) return launch_q4_a16_mma_r64_t64_k128_s2_a1;
    return launch_q4_a16_mma_r64_t128;
}
} // namespace ninfer::ops::detail
