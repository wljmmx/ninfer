// The GGML-encoded embedding gather: the selected table rows are decoded from their stored blocks
// straight into the BF16 output. One CTA per output row, one thread per 32-value group.

#include "ops/linear/ggml/ggml_embedding.h"

#include "core/device.h"
#include "core/ggml_types.h"
#include "ops/linear/ggml/ggml_block_decode.cuh"

#include <cuda_bf16.h>
#include <stdexcept>

namespace ninfer::ops::detail::ggml {
namespace {

__device__ __forceinline__ void group_values(int type, const std::uint8_t* block, int group,
                                             __nv_bfloat16* out) {
    switch (type) {
    case 0: decode_group<0>(block, group, out); break;
    case 1: decode_group<1>(block, group, out); break;
    case 2: decode_group<2>(block, group, out); break;
    case 3: decode_group<3>(block, group, out); break;
    case 4: decode_group<4>(block, group, out); break;
    case 5: decode_group<5>(block, group, out); break;
    case 6: decode_group<6>(block, group, out); break;
    case 7: decode_group<7>(block, group, out); break;
    case 8: decode_group<8>(block, group, out); break;
    case 9: decode_group<9>(block, group, out); break;
    case 10: decode_group<10>(block, group, out); break;
    }
}

__global__ void gather_kernel(const std::int32_t* ids, const std::uint8_t* table,
                              __nv_bfloat16* out, int k, int groups_per_block, int row_bytes,
                              int type_bytes, int type) {
    const int token        = blockIdx.x;
    const std::uint8_t* row = table + std::size_t(ids[token]) * std::size_t(row_bytes);
    __nv_bfloat16* dst      = out + std::size_t(token) * std::size_t(k);
    const int groups        = k / 32;
    for (int group = threadIdx.x; group < groups; group += blockDim.x) {
        const std::uint8_t* block =
            row + std::size_t(group / groups_per_block) * std::size_t(type_bytes);
        group_values(type, block, group % groups_per_block, dst + std::size_t(group) * 32);
    }
}

int type_index(QType format) {
    switch (format) {
    case QType::GGML_Q4_K: return 0;
    case QType::GGML_Q5_K: return 1;
    case QType::GGML_Q6_K: return 2;
    case QType::GGML_Q2_0: return 3;
    case QType::GGML_IQ4_NL: return 4;
    case QType::GGML_IQ4_XS: return 5;
    case QType::GGML_IQ2_XXS: return 6;
    case QType::GGML_IQ2_XS: return 7;
    case QType::GGML_IQ2_S: return 8;
    case QType::GGML_IQ3_XXS: return 9;
    case QType::GGML_IQ3_S: return 10;
    default: throw std::invalid_argument("embedding: not a GGML block table");
    }
}

} // namespace

void launch_embedding_gather(const Tensor& ids, const Weight& table, Tensor& out,
                             cudaStream_t stream) {
    const auto geometry = core::ggml_block_geometry(table.qtype);
    if (table.k <= 0 || table.n <= 0 || ids.ne[0] <= 0) {
        throw std::invalid_argument("embedding: empty GGML table geometry");
    }
    if (table.qdata == nullptr) {
        throw std::invalid_argument("embedding: GGML table data must be non-null");
    }
    if (std::int64_t(table.k) % std::int64_t(geometry.block) != 0 || geometry.block % 32 != 0) {
        throw std::invalid_argument("embedding: GGML table width is not whole blocks of 32 values");
    }
    const int groups_per_block = int(geometry.block) / 32;
    const int row_bytes = table.k / int(geometry.block) * int(geometry.type_bytes);
    const int groups    = table.k / 32;
    const int threads   = groups < 128 ? (groups < 32 ? 32 : 64) : 128;

    gather_kernel<<<ids.ne[0], threads, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), static_cast<const std::uint8_t*>(table.qdata),
        static_cast<__nv_bfloat16*>(out.data), table.k, groups_per_block, row_bytes,
        int(geometry.type_bytes), type_index(table.qtype));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail::ggml
