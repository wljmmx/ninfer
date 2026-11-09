#include "ops/linear/q4/q4_shapes.h"

namespace ninfer::ops::detail {
// Qwen4Exp (hidden 2560): the QSA layer's index_key projection is [128,2560] - eight 16-row or
// four 32-row tiles. K=2560 keeps every 64/128/256-wide K tile whole, so the generic routes apply.
// The rungs stay on 16-wide row tiles up to T=64 and cap the MMAs at 32 rows: a 64-row tile would
// leave only two CTAs for the whole projection, and the projection is small enough (184 KiB of
// codes) that the extra tiles cost nothing while they double the occupancy. Inherited for
// correctness first; not yet measured at [128,2560] on this device.
Q4Launch select_q4_n128_k2560(std::int32_t tokens) {
    if (tokens <= 1) return launch_q4_a16_gemv_r4_w1_direct;
    if (tokens <= 8) return launch_q4_a16_simt_r4_t4_w2_g8_s2;
    if (tokens <= 32) return launch_q4_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 64) return launch_q4_a16_sliced_r16_t32_w4_s2;
    if (tokens <= 128) return launch_q4_a16_mma_r32_t64_k64_wr16_wt32_s2_a2_b2;
    return launch_q4_a16_mma_r32_t128_k64_s2_a2;
}
} // namespace ninfer::ops::detail
