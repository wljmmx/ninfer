#include "ops/linear/q5/q5_dispatch.h"
#include "ops/linear/q5/q5_shapes.h"
#include <array>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
struct ShapeEntry {
    std::int32_t n, k;
    Q5Launch (*select)(std::int32_t);
};

constexpr std::array kShapes{
    ShapeEntry{1024, 5120, select_q5_n1024_k5120},   ShapeEntry{6144, 5120, select_q5_n6144_k5120},
    ShapeEntry{7168, 5120, select_q5_n7168_k5120},   ShapeEntry{5120, 6144, select_q5_n5120_k6144},
    ShapeEntry{5120, 17408, select_q5_n5120_k17408}, ShapeEntry{1152, 1152, select_q5_n1152_k1152},
    ShapeEntry{1152, 4304, select_q5_n1152_k4304},
};

// v2 RTX 4090 (sm_89) tuned dispatch for large-T prefill. The v3 shape
// table was tuned on RTX 5090 and its q5_a16_mma ran 13% slower per
// launch. For T > 128 (prefill chunk), route to the v2 rowsplit GEMM
// kernels which use the same column-tile-major CTA order and balanced
// warp tiles that made the FFN and attn_input ports faster.
Q5Launch select_q5_v2_prefill(std::int32_t n, std::int32_t k, std::int32_t t) {
    switch (k) {
    case 5120:
        switch (n) {
        case 1024: case 6144: case 7168:
            if (t <= 16) return launch_q5_small_t_mma;
            if (t <= 64) return launch_q5_mma_r64_c32;
            if (t <= 128) return launch_q5_mma_r64_c64;
            return launch_q5_mma_r64_c128;
        default: break;
        }
        break;
    case 6144:
        if (n == 5120) {
            if (t <= 16) return launch_q5_small_t_mma;
            if (t <= 64) return launch_q5_mma_r64_c32;
            if (t <= 128) return launch_q5_mma_r64_c64;
            return launch_q5_mma_r64_c128;
        }
        break;
    case 17408:
        if (n == 5120) {
            if (t <= 16) return launch_q5_small_t_mma;
            if (t <= 64) return launch_q5_mma_r64_c32;
            if (t <= 128) return launch_q5_mma_r64_c64;
            return launch_q5_mma_r64_c128;
        }
        break;
    case 1152:
        if (n == 1152 && t >= 4 && t <= 131072 && (t % 4) == 0) {
            if (t <= 76) return launch_q5_simt_r8_c4;
            return launch_q5_mma_r64_c128;
        }
        break;
    case 4304:
        if (n == 1152 && t >= 4 && t <= 131072 && (t % 4) == 0) {
            if (t <= 120) return launch_q5_simt_r8_c4;
            return launch_q5_mma_r64_c128;
        }
        break;
    default: break;
    }
    return nullptr;  // Fall back to v3 shape table.
}
} // namespace

Q5Launch select_q5_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) throw std::invalid_argument("q5 linear: T must be positive");
    // sm_89: consult the v2-tuned dispatch at EVERY T first. nsys on the MTP
    // decode rounds showed the v3 RTX 5090 shape-table kernels (a16 sliced-k/
    // simt) run ~10% slower per round than the v2 small_t/mma family; the v3
    // shape table only serves shapes the v2 dispatch rejects.
    if (auto v2 = select_q5_v2_prefill(n, k, t)) return v2;
    for (const auto& entry : kShapes) {
        if (entry.n == n && entry.k == k) return entry.select(t);
    }
    throw std::invalid_argument("q5 linear: unsupported shape");
}

Q5Launch select_q5_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("q5 linear: unsupported policy");
    return select_q5_a16_launch(n, k, t);
}

void q5_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    select_q5_launch(weight.n, weight.k, x.ne[1], policy)(x, weight, out, stream);
}
} // namespace ninfer::ops::detail

