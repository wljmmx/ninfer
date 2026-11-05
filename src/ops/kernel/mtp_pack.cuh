#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kMtpAttnRows = 14336;
inline constexpr int kMtpQRows    = 6144;
inline constexpr int kMtpKvRows   = 1024;

// Both kernels move pure bf16 streams whose only arithmetic is address selection, so the
// memory system caps them at a fraction of DRAM while each thread touches a single 2-byte
// element. Each thread now owns a 16-byte uint4 (8 bf16):
//   - pack: the row stride (rows) and the packed output stride (2*rows) are multiples of
//     8 for every admitted hidden size, so an 8-element group never straddles an
//     alignment boundary; a scalar tail covers rows not divisible by 8.
//   - split: 14336 rows = 8 * 1792 and every segment bound (6144 / 7168 / 13312) is a
//     multiple of 8, so a uint4 always lands in one token and one segment. The row
//     decode (one div/mod per uint4 instead of per element) is the whole 14336-is-not-a-
//     power-of-two story; the divisor is a compile-time constant, so ptxas lowers it to
//     a multiply-shift.
inline constexpr int kMtpVec = 8;

__launch_bounds__(256) __global__ void
mtp_pack_fc_input_kernel(const __nv_bfloat16* __restrict__ embedding_norm,
                         const __nv_bfloat16* __restrict__ hidden_norm,
                         __nv_bfloat16* __restrict__ out, std::int32_t rows) {
    const int token = static_cast<int>(blockIdx.y);
    const int row   = (static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)) * kMtpVec;
    if (row >= rows) { return; }

    const std::int64_t in_base  = static_cast<std::int64_t>(token) * rows;
    const std::int64_t out_base = static_cast<std::int64_t>(token) * (2 * rows);
    if ((rows & (kMtpVec - 1)) == 0 && row + kMtpVec <= rows) {
        const uint4 embedding =
            *reinterpret_cast<const uint4*>(embedding_norm + in_base + row);
        const uint4 hidden = *reinterpret_cast<const uint4*>(hidden_norm + in_base + row);
        *reinterpret_cast<uint4*>(out + out_base + row)                          = embedding;
        *reinterpret_cast<uint4*>(out + out_base + rows + row)                   = hidden;
    } else {
        for (int offset = row; offset < rows; ++offset) {
            out[out_base + offset]        = embedding_norm[in_base + offset];
            out[out_base + rows + offset] = hidden_norm[in_base + offset];
        }
    }
}

__launch_bounds__(256) __global__ void
mtp_split_attn_in_kernel(const __nv_bfloat16* __restrict__ attn_in, __nv_bfloat16* q,
                         __nv_bfloat16* k, __nv_bfloat16* gate, __nv_bfloat16* v,
                         std::int32_t tokens) {
    constexpr int kVectorsPerToken = kMtpAttnRows / kMtpVec;
    const std::int64_t index       = blockIdx.x * static_cast<std::int64_t>(blockDim.x) +
                               threadIdx.x;
    const std::int64_t total = static_cast<std::int64_t>(kVectorsPerToken) * tokens;
    if (index >= total) { return; }

    const int token = static_cast<int>(index / kVectorsPerToken);
    const int row   = static_cast<int>(index % kVectorsPerToken) * kMtpVec;
    const uint4 value =
        reinterpret_cast<const uint4*>(attn_in)[static_cast<std::size_t>(index)];

    if (row < kMtpQRows) {
        *reinterpret_cast<uint4*>(q + static_cast<std::int64_t>(token) * kMtpQRows + row) =
            value;
        return;
    }
    if (row < kMtpQRows + kMtpKvRows) {
        *reinterpret_cast<uint4*>(k + static_cast<std::int64_t>(token) * kMtpKvRows +
                                  (row - kMtpQRows)) = value;
        return;
    }
    if (row < kMtpQRows + kMtpKvRows + kMtpQRows) {
        *reinterpret_cast<uint4*>(gate + static_cast<std::int64_t>(token) * kMtpQRows +
                                  (row - kMtpQRows - kMtpKvRows)) = value;
        return;
    }
    *reinterpret_cast<uint4*>(v + static_cast<std::int64_t>(token) * kMtpKvRows +
                              (row - kMtpQRows - kMtpKvRows - kMtpQRows)) = value;
}

} // namespace ninfer::ops
