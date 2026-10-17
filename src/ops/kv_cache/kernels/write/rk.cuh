#pragma once

// rk (rank-compressed) KV append, carried over from the earlier implementation:
//   rk8v4     int8 K                     + packed int4 V
//   rk4v4     packed int4 K              + packed int4 V
//   rk4v4-e8  packed int4 K, E8 lattice  + packed int4 V
//   rk2v4-e8  E8 cylinder K (2-bit)      + packed int4 V
//   rk4v2-e8  E8 lattice K               + E8 cylinder V
//
// port (sm_89): the kernel itself is append/rk_kernel.cuh and is unchanged. This wrapper only restates
// the launch geometry the earlier code used - one warp per (token, kv_head) row, 256 threads, 256
// floats of Hadamard scratch per warp - against this tree's Metadata type.

#include "core/device.h"
#include "ops/common/numeric/math.h"
#include "ops/kv_cache/append/rk_kernel.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail::kv {

inline constexpr int kRkBlockThreads = 256;

// `batch > 1` mirrors the earlier multi-batch path: blockIdx.z selects the batch, the kernel advances
// the metadata pointers to it, and the source rows are offset per batch. Both paths size the grid by
// one warp per row (kRkBlockThreads / 32 == 8 warps).
template <bool MultiBatch, class G, bool PackedK, bool E8Lattice, bool E8Root, bool V8Root,
          class Metadata>
void launch_rk_write(const __nv_bfloat16* k, const __nv_bfloat16* v, const std::int32_t* positions,
                     Metadata metadata, std::int8_t* cache_k, std::int8_t* cache_v,
                     __half* scale_k, __half* scale_v, int tokens, int batch,
                     cudaStream_t stream) {
    constexpr int kWarps = kRkBlockThreads / 32;
    const int smem       = kWarps * 256 * static_cast<int>(sizeof(float));
    const int rows       = tokens * G::KVHeads;
    // MultiBatch is a template parameter because the kernel advances the metadata pointers per batch;
    // DirectMetadata has no such fields, so the mode has to match the metadata type at compile time.
    if constexpr (MultiBatch) {
        const dim3 grid(common::div_up(rows, kWarps), 1, batch);
        kv_cache_append_full_rk_kernel<G, Metadata, PackedK, E8Lattice, E8Root, true, V8Root>
            <<<grid, kRkBlockThreads, smem, stream>>>(k, v, positions, metadata, cache_k, cache_v,
                                                      scale_k, scale_v, tokens);
    } else {
        const int grid = common::div_up(rows, kWarps);
        kv_cache_append_full_rk_kernel<G, Metadata, PackedK, E8Lattice, E8Root, false, V8Root>
            <<<grid, kRkBlockThreads, smem, stream>>>(k, v, positions, metadata, cache_k, cache_v,
                                                      scale_k, scale_v, tokens);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail::kv
