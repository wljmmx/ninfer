#include "ops/linear/q4/q4_shapes.h"
#include <stdexcept>

namespace ninfer::ops::detail {
Q4Launch select_q4_n4304_k1152(std::int32_t tokens) {
    if (tokens > 131072 || tokens % 4 != 0)
        throw std::invalid_argument("q4 linear: T must be a multiple of 4 in [4,131072]");
    // Selected with complete-Op cold CUDA Graph measurements on RTX 5090.
    if (tokens <= 16) return launch_q4_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 32) return launch_q4_a16_sliced_r16_t32_w4_s2;
#if defined(NINFER_ARCH_SM89)
    // Measured on RTX 4090 (2026-11-06, linear_bench, min of repeat=200, flush 96 MiB):
    // - (64,96]: the 5090 rung mma_r32_t32_k128 costs 20.5/20.5/20.7 us at T=68/80/92, while
    //   sliced_r32_t32_w2_s2 (three 32-token passes) runs the same span at
    //   16.4/17.3/17.4 us (-20.0%..-15.3%).
    // - (384,512]: the 5090 tail r64_t128 costs 79.9/78.8/70.7/72.6 us at T=388/416/448/512,
    //   while the sliced kernel's extra passes run 50.2/50.2/54.3/61.3 us
    //   (-37.2%..-15.6%). Past 512 the t128 kernel wins again (T=640: -2.7% for sliced,
    //   T=2048: +2.5%), so the sm_89 fork ends the sliced rung at 512.
    if (tokens <= 512) return launch_q4_a16_sliced_r32_t32_w2_s2;
#else
    if (tokens <= 64) return launch_q4_a16_sliced_r32_t32_w2_s2;
    if (tokens <= 96) return launch_q4_a16_mma_r32_t32_k128_s2_a2;
    if (tokens <= 384) return launch_q4_a16_sliced_r32_t32_w2_s2;
#endif
    return launch_q4_a16_mma_r64_t128;
}
} // namespace ninfer::ops::detail
