// rk PackedV append launch for rank-compressed KV layouts.
#include "ops/kv_cache/append/launch.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/kv_cache/append/rk_kernel.cuh"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock = 256;

template <typename Geometry, bool PackedK, typename CacheView, typename Metadata>
void launch_rk_for(const Tensor& k, const Tensor& v, const Tensor& positions,
                   CacheView cache, Metadata metadata, cudaStream_t stream) {
    const auto tokens = static_cast<std::int32_t>(k.ne[2]);
    auto* cache_k     = static_cast<std::int8_t*>(cache.k_pages.data);
    auto* cache_v     = static_cast<std::int8_t*>(cache.v_pages.data);
    auto* scale_k     = static_cast<__half*>(cache.k_scale_pages.data);
    auto* scale_v     = static_cast<__half*>(cache.v_scale_pages.data);

    const int warps    = kBlock / 32;
    const std::int64_t fill_units = static_cast<std::int64_t>(tokens) * Geometry::KVHeads;
    const int grid = static_cast<int>(div_up(fill_units, static_cast<std::int64_t>(warps)));

    // 256 floats of shared memory per warp for the Hadamard scratch.
    const int smem_bytes = (kBlock / 32) * 256 * sizeof(float);

    kv_cache_append_full_rk_kernel<Geometry, Metadata, PackedK>
        <<<grid, kBlock, smem_bytes, stream>>>(
            static_cast<const __nv_bfloat16*>(k.data),
            static_cast<const __nv_bfloat16*>(v.data),
            static_cast<const std::int32_t*>(positions.data), metadata,
            cache_k, cache_v, scale_k, scale_v, tokens);
    CUDA_CHECK(cudaGetLastError());
}

template <bool PackedK, typename CacheView, typename Metadata>
void dispatch_rk(const Tensor& k, const Tensor& v, const Tensor& positions,
                 CacheView cache, Metadata metadata, cudaStream_t stream) {
    if (k.ne[1] == KVCacheAppendD256Kv4::KVHeads) {
        launch_rk_for<KVCacheAppendD256Kv4, PackedK>(k, v, positions, cache, metadata, stream);
    } else {
        launch_rk_for<KVCacheAppendD256Kv2, PackedK>(k, v, positions, cache, metadata, stream);
    }
}

} // namespace

void kv_cache_append_rk8v4_batch_launch(const Tensor& k, const Tensor& v,
                                        const Tensor& positions, const Tensor& valid_columns,
                                        const Tensor& table_rows, PagedKVBatchLayerView cache,
                                        cudaStream_t stream) {
    const auto launch = [&]<bool Masked>() {
        const PagedKVBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        if (k.ne[3] > 1) {
            const auto append = [&]<class G>() {
                const dim3 grid(div_up(k.ne[2] * G::KVHeads, 8), 1, k.ne[3]);
                const int smem = (kBlock / 32) * 256 * sizeof(float);
                kv_cache_append_full_rk_kernel<G, PagedKVBatchMetadata<Masked>, false, true>
                    <<<grid, kBlock, smem, stream>>>(
                        static_cast<const __nv_bfloat16*>(k.data),
                        static_cast<const __nv_bfloat16*>(v.data),
                        static_cast<const std::int32_t*>(positions.data), metadata,
                        static_cast<std::int8_t*>(cache.k_pages.data),
                        static_cast<std::int8_t*>(cache.v_pages.data),
                        static_cast<__half*>(cache.k_scale_pages.data),
                        static_cast<__half*>(cache.v_scale_pages.data), k.ne[2]);
            };
            if (k.ne[1] == 4)
                append.template operator()<KVCacheAppendD256Kv4>();
            else
                append.template operator()<KVCacheAppendD256Kv2>();
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        dispatch_rk<false>(k, v, positions, cache, metadata, stream);
    };
    if (valid_columns.data == nullptr)
        launch.template operator()<false>();
    else
        launch.template operator()<true>();
}

void kv_cache_append_rk4v4_batch_launch(const Tensor& k, const Tensor& v,
                                        const Tensor& positions, const Tensor& valid_columns,
                                        const Tensor& table_rows, PagedKVBatchLayerView cache,
                                        cudaStream_t stream) {
    // Same as rk8v4 but with PackedK=true (int4 K + int4 V).
    const auto launch = [&]<bool Masked>() {
        const PagedKVBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        if (k.ne[3] > 1) {
            const auto append = [&]<class G>() {
                const dim3 grid(div_up(k.ne[2] * G::KVHeads, 8), 1, k.ne[3]);
                const int smem = (kBlock / 32) * 256 * sizeof(float);
                kv_cache_append_full_rk_kernel<G, PagedKVBatchMetadata<Masked>, true, true>
                    <<<grid, kBlock, smem, stream>>>(
                        static_cast<const __nv_bfloat16*>(k.data),
                        static_cast<const __nv_bfloat16*>(v.data),
                        static_cast<const std::int32_t*>(positions.data), metadata,
                        static_cast<std::int8_t*>(cache.k_pages.data),
                        static_cast<std::int8_t*>(cache.v_pages.data),
                        static_cast<__half*>(cache.k_scale_pages.data),
                        static_cast<__half*>(cache.v_scale_pages.data), k.ne[2]);
            };
            if (k.ne[1] == 4)
                append.template operator()<KVCacheAppendD256Kv4>();
            else
                append.template operator()<KVCacheAppendD256Kv2>();
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        dispatch_rk<true>(k, v, positions, cache, metadata, stream);
    };
    if (valid_columns.data == nullptr)
        launch.template operator()<false>();
    else
        launch.template operator()<true>();
}

// Single-token rk8v4 append (PagedKVLayerView, used by MTP warmup / single decode).
void kv_cache_append_rk8v4_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                  PagedKVLayerView cache, cudaStream_t stream) {
    const PagedKVDirectMetadata metadata{static_cast<const std::int32_t*>(cache.block_table.data)};
    dispatch_rk<false>(k, v, positions, cache, metadata, stream);
}

} // namespace ninfer::ops::detail
