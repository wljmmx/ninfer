#include "ops/softmax_attention/dense/packed/launch.h"

#include "ops/softmax_attention/dense/packed/kernel.cuh"
#include "core/device.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// The packed tile ladder: kPackedDescriptorRows query rows per CTA, 64 keys per step (the same
// constants the former local kernel carried; the shared noncausal kernel takes them as template
// arguments). The descriptor grid and the query tiling are the same axis: one CTA consumes
// exactly one descriptor, so the prepare kernel must be instantiated with this row count.
inline constexpr int kPackedAttentionBr = kPackedDescriptorRows;
inline constexpr int kPackedAttentionBc = 64;

std::int64_t stride_elements(const Tensor& tensor, int dim) {
    return tensor.nb[dim] / static_cast<std::int64_t>(sizeof(__nv_bfloat16));
}

// T088 (upstream 6eb6056b): the noncausal kernel is head-dimension parameterized. The
// packed path keeps its existing Br/Bc tile ladder and forwards to the shared launcher
// with PaddedD = 128 and the caller's D (72 for the text heads, 64 for the vision heads).
template <int Br, int Bc, int D = 72>
void launch_flash_tuned(const Tensor& q, const Tensor& k, const Tensor& v,
                        const PackedAttentionTile* tiles, std::int32_t uniform_segment_length,
                        std::int32_t query_tiles, Tensor& out, cudaStream_t stream) {
    ::ninfer::ops::detail::launch_flash<Br, Bc, 128, 128 / Br, D>(
        q, k, v, tiles, uniform_segment_length, query_tiles, out, stream);
}

// Dispatches the tile ladder with the head dimension the caller resolved.
template <int Br, int Bc>
void launch_flash_for_dim(const Tensor& q, const Tensor& k, const Tensor& v,
                          const PackedAttentionTile* tiles, std::int32_t uniform_segment_length,
                          std::int32_t query_tiles, Tensor& out, cudaStream_t stream) {
    if (q.ne[0] == 64) {
        launch_flash_tuned<Br, Bc, 64>(q, k, v, tiles, uniform_segment_length, query_tiles, out,
                                       stream);
    } else {
        launch_flash_tuned<Br, Bc, 72>(q, k, v, tiles, uniform_segment_length, query_tiles, out,
                                       stream);
    }
}

} // namespace

void packed_attention_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                             const Tensor& cu_seqlens, Tensor* tiles, Tensor& out,
                             cudaStream_t stream) {
    const bool packed_segments = tiles != nullptr;
    const int max_tiles =
        packed_segments ? tiles->ne[1] : (q.ne[2] + kPackedAttentionBr - 1) / kPackedAttentionBr;
    if (packed_segments) {
        packed_attention_prepare_tiles_kernel<kPackedDescriptorRows><<<1, 256, 0, stream>>>(
            static_cast<const std::int32_t*>(cu_seqlens.data), cu_seqlens.ne[0] - 1,
            static_cast<PackedAttentionTile*>(tiles->data), max_tiles, q.ne[2]);
        CUDA_CHECK(cudaGetLastError());
    }

    launch_flash_for_dim<kPackedAttentionBr, kPackedAttentionBc>(
        q, k, v, packed_segments ? static_cast<const PackedAttentionTile*>(tiles->data) : nullptr,
        0, max_tiles, out, stream);
}

std::int32_t packed_attention_uniform_tile(std::int32_t segment_length) {
    // Measured issued-MMA rates are approximately 3:6:8 for the 16/32/64
    // tiles. Minimize padded tile area divided by that rate.
    constexpr std::int32_t tiles[] = {16, 32, 64};
    constexpr std::int32_t rates[] = {3, 6, 8};
    std::int32_t best_tile         = tiles[0];
    std::int64_t best_padded       = ((segment_length + best_tile - 1) / best_tile) * best_tile;
    std::int64_t best_square       = best_padded * best_padded;
    std::int32_t best_rate         = rates[0];
    for (int i = 1; i < 3; ++i) {
        const std::int64_t padded = ((segment_length + tiles[i] - 1) / tiles[i]) * tiles[i];
        const std::int64_t square = padded * padded;
        if (square * best_rate < best_square * rates[i]) {
            best_tile   = tiles[i];
            best_square = square;
            best_rate   = rates[i];
        }
    }
    return best_tile;
}

void packed_attention_uniform_launch_with_tile(const Tensor& q, const Tensor& k, const Tensor& v,
                                               std::int32_t segment_length, std::int32_t tile_size,
                                               Tensor& out, cudaStream_t stream) {
    const std::int32_t segments    = q.ne[2] / segment_length;
    const std::int32_t query_tiles = segments * ((segment_length + tile_size - 1) / tile_size);
    switch (tile_size) {
    case 16:
        launch_flash_for_dim<16, 16>(q, k, v, nullptr, segment_length, query_tiles, out, stream);
        return;
    case 32:
        launch_flash_for_dim<32, 32>(q, k, v, nullptr, segment_length, query_tiles, out, stream);
        return;
    case 64:
        launch_flash_for_dim<64, 64>(q, k, v, nullptr, segment_length, query_tiles, out, stream);
        return;
    default:
        throw std::invalid_argument("packed_softmax_attention: invalid uniform tile size");
    }
}

void packed_attention_uniform_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                     std::int32_t segment_length, Tensor& out,
                                     cudaStream_t stream) {
    packed_attention_uniform_launch_with_tile(
        q, k, v, segment_length, packed_attention_uniform_tile(segment_length), out, stream);
}

} // namespace ninfer::ops::detail
