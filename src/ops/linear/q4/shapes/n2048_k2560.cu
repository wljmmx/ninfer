#include "ops/linear/q4/q4_shapes.h"

namespace ninfer::ops::detail {
// Qwen4Exp (hidden 2560): the GDN query and key projections are two [2048,2560] Q4 parents.
// K=2560 keeps the 128-wide and 256-wide K tiles whole, so every generic (non-static-K) relative
// route the 5120 tables use is admissible; the static-K5120 GEMV and sliced-K instances are not.
// The rungs below are inherited from the same-row 5120 tables (n1024_k5120 and n2048 siblings) as
// a correctness-first ordering. They have NOT been measured at K=2560 on this device yet; the
// token seams that were measured on RTX 4090 for K=5120 are kept, and re-tuning is outstanding.
Q4Launch select_q4_n2048_k2560(std::int32_t tokens) {
    if (tokens <= 1) return launch_q4_a16_gemv_r4_w1_direct;
    if (tokens <= 8) return launch_q4_a16_simt_r4_t4_w2_g8_s2;
    if (tokens <= 32) return launch_q4_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 64) return launch_q4_a16_sliced_r16_t32_w4_s2;
    if (tokens <= 320) return launch_q4_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 576) return launch_q4_a16_sliced_r32_t32_w2_s2;
    if (tokens <= 1280) return launch_q4_a16_mma_r32_t64_k64_wr16_wt32_s2_a2_b2;
    return launch_q4_a16_mma_r64_t128;
}
} // namespace ninfer::ops::detail
