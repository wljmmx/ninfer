#pragma once

#include "ops/gated_residual/mix/kernel/up_mma_mix.cuh"
#include "ops/gated_residual/mix/operands_split.h"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"

namespace ninfer::ops::detail::hc::mix {
// The same fold as up_mma_mix_kernel with the Up matrix read straight out of its Q5 RowSplit
// planes. A staged K block is one 64-wide group, so each staged row segment is one code chunk:
// four code bytes, one high-bit byte and the group's FP16 scale. Decoding in place keeps the
// quantized matrix the only resident copy; the shared tile holds bf16 because the MMA reads bf16,
// which is the form the dense BF16 route stages too.
template <int Features, int Tokens>
__device__ __forceinline__ unsigned q5_pair(float lo, float hi) {
    const __nv_bfloat162 value = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&value);
}

template <int Features, int Tokens>
__launch_bounds__(FoldedUpSchedule<Features, Tokens>::kThreads) __global__
    void up_q5_mix_kernel(SeparateOperands o, Q5Planes up, const __nv_bfloat16* activation,
                          int token_offset) {
    using S          = FoldedUpSchedule<Features, Tokens>;
    constexpr int BK = S::kBlockK, Rows = kStreams * Features, Stages = S::kStages;
    static_assert(BK == Q5RowSplitStorage::kGroupK,
                  "a staged K block must be exactly one Q5 group");
    static_assert(kWidth % Features == 0 && S::kMmaRows == 1 && S::kMmaTokens == 1);
    const int feature_tiles = kWidth / Features;
    const int d0            = (int(blockIdx.x) % feature_tiles) * Features;
    const int token0        = token_offset + (int(blockIdx.x) / feature_tiles) * Tokens;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    extern __shared__ __align__(16) unsigned char storage[];
    auto* a                                    = reinterpret_cast<__nv_bfloat16*>(storage);
    auto* b                                    = a + Stages * Rows * BK;
    float accum[S::kMmaRows][S::kMmaTokens][4] = {};
    const auto* codes                          = static_cast<const std::uint8_t*>(up.codes);
    const auto* high_bits                      = static_cast<const std::uint8_t*>(up.high);
    const auto* scales                         = static_cast<const std::uint8_t*>(up.scales);
    const auto stage_inputs                    = [&](int slot, int group) {
        auto* as = a + slot * Rows * BK;
        auto* bs = b + slot * Tokens * BK;
        for (int i = tid; i < Rows * (BK / 8); i += S::kThreads) {
            const int row = i / (BK / 8), k8 = i % (BK / 8);
            const int native_row = (row / Features) * kWidth + d0 + row % Features;
            const std::int64_t base = std::int64_t(native_row);
            const std::uint32_t packed =
                *reinterpret_cast<const std::uint32_t*>(codes + base * up.code_row_bytes +
                                                        group * Q5RowSplitStorage::kCodeBytesPerGroup +
                                                        k8 * Q5RowSplitStorage::kCodeBytesPerChunk);
            const std::uint8_t high =
                high_bits[base * up.high_row_bytes +
                          group * Q5RowSplitStorage::kHighBytesPerGroup +
                          k8 * Q5RowSplitStorage::kHighBytesPerChunk];
            const std::uint16_t scale = *reinterpret_cast<const std::uint16_t*>(
                scales + base * up.scale_row_bytes +
                group * Q5RowSplitStorage::kScaleBytesPerGroup);
            float weights[8];
            Q5SimtDecodeAtom::decode_eight(packed, high, scale, weights);
            uint4 tile;
            tile.x = q5_pair<Features, Tokens>(weights[0], weights[1]);
            tile.y = q5_pair<Features, Tokens>(weights[2], weights[3]);
            tile.z = q5_pair<Features, Tokens>(weights[4], weights[5]);
            tile.w = q5_pair<Features, Tokens>(weights[6], weights[7]);
            *reinterpret_cast<uint4*>(as + row * BK + bf16_mma_shared_col<S>(row, k8 * 8)) = tile;
        }
        for (int i = tid; i < Tokens * (BK / 8); i += S::kThreads) {
            const int token = i / (BK / 8), k8 = i % (BK / 8);
            const bool valid = token0 + token < o.columns;
            const auto source =
                valid ? std::int64_t(token0 + token) * kRank + group * BK + k8 * 8 : 0;
            cp_async_zfill<16, S::kActivationCache>(bs + token * BK +
                                                        bf16_mma_shared_col<S>(token, k8 * 8),
                                                    activation + source, valid ? 16 : 0);
        }
    };
    constexpr int groups = kRank / BK;
#pragma unroll
    for (int slot = 0; slot < Stages; ++slot) {
        stage_inputs(slot, slot);
        cp_commit();
    }
#pragma unroll
    for (int group = 0; group < groups; ++group) {
        if (group + Stages <= groups)
            cp_wait<Stages - 1>();
        else
            cp_wait<0>();
        __syncthreads();
        const int slot = group % Stages;
        bf16_mma_compute_stage<S>(a + slot * Rows * BK, b + slot * Tokens * BK, accum, warp, lane);
        __syncthreads();
        if (group + Stages < groups) {
            stage_inputs(slot, group + Stages);
            cp_commit();
        }
    }
    auto* contributions = reinterpret_cast<float*>(storage);
    const auto at       = [&](int r, int t) { return contribution_index<Rows, Tokens>(r, t); };
    const int wm = warp / S::kWarpsTokens, wn = warp % S::kWarpsTokens;
    const int row = wm * 16 + (lane >> 2), token = wn * 8 + 2 * (lane & 3);
    const auto store = [&](int r, int t, float value) {
        const int s = r / Features, d = d0 + r % Features;
        if (token0 + t < o.columns) {
            const auto z            = __bfloat162float(static_cast<const __nv_bfloat16*>(
                o.normalized)[std::int64_t(token0 + t) * kResidualWidth + s * kWidth + d]);
            contributions[at(r, t)] = sigmoid_gate(value) * z;
        }
    };
    const auto& v = accum[0][0];
    store(row, token, v[0]);
    store(row, token + 1, v[1]);
    store(row + 8, token, v[2]);
    store(row + 8, token + 1, v[3]);
    __syncthreads();
    if (wm * 16 < Features) {
        const auto finish = [&](int d, int t) {
            if (token0 + t < o.columns) {
                float sum = 0;
#pragma unroll
                for (int s = 0; s < kStreams; ++s) sum += contributions[at(s * Features + d, t)];
                static_cast<__nv_bfloat16*>(o.output)[std::int64_t(token0 + t) * kWidth + d0 + d] =
                    __float2bfloat16_rn(sum * 0.25F);
            }
        };
        finish(row, token);
        finish(row, token + 1);
        finish(row + 8, token);
        finish(row + 8, token + 1);
    }
}
} // namespace ninfer::ops::detail::hc::mix
