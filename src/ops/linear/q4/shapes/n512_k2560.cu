#include "ops/linear/q4/q4_shapes.h"

namespace ninfer::ops::detail {
// Qwen4Exp (hidden 2560): the QSA layer's key projection is [512,2560] and index_query is a
// second one of the same shape. 2560 is 40 quant groups and a whole number of 128-, 256- and
// 512-wide K tiles, so every generic route the 5120 tables use is admissible; only the static-K
// GEMV/sliced instances are not. The rungs follow n2048_k2560 (same K, same activation) because
// the two differ only in how many 16/32/64-row tiles the grid spreads over: 512 rows still gives
// 16 tiles at 32 rows, so no rung drops below the seam it was measured at on RTX 4090. They are
// inherited for correctness first and are not yet measured at [512,2560] on this device.
Q4Launch select_q4_n512_k2560(std::int32_t tokens) {
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
