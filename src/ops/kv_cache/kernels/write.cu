#include "ops/kv_cache/kernels/write.h"
#include "core/device.h"
#include "ops/common/numeric/math.h"
#include "ops/kv_cache/kernels/write/bf16.cuh"
#include "ops/kv_cache/kernels/write/fp8.cuh"
#include "ops/kv_cache/kernels/write/int8.cuh"
#include "ops/kv_cache/kernels/write/k8v4.cuh"
#include "ops/kv_cache/kernels/write/nvfp4.cuh"
// port (sm_89): the rk (rank-compressed) layouts reuse the earlier append kernel.
#include "ops/kv_cache/kernels/write/rk.cuh"

#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail::kv {
namespace {

struct WriteArguments {
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    const std::int32_t* positions;
    void* keys;
    void* values;
    void* key_scales;
    void* value_scales;
    int width;
    int batch;
};

template <class Cache>
WriteArguments arguments(const Tensor& k, const Tensor& v, const Tensor& positions, Cache cache) {
    return {static_cast<const __nv_bfloat16*>(k.data),
            static_cast<const __nv_bfloat16*>(v.data),
            static_cast<const std::int32_t*>(positions.data),
            cache.k_pages.data,
            cache.v_pages.data,
            cache.k_scale_pages.data,
            cache.v_scale_pages.data,
            k.ne[2],
            k.ne[3]};
}

template <KvCacheStorage Storage, class G, class Metadata, PagedWriteInstance Instance>
void launch_quantized(const WriteArguments& a, Metadata metadata, cudaStream_t stream) {
    constexpr bool Page   = Instance == PagedWriteInstance::Page8x256;
    constexpr bool Batch  = Instance == PagedWriteInstance::BatchRows256;
    constexpr auto kernel = [] {
        if constexpr (Storage == KvCacheStorage::Int8Group64) {
            if constexpr (Page)
                return write_i8_page_kernel<G, Metadata>;
            else
                return write_i8_kernel<G, Metadata, Batch>;
        } else if constexpr (Storage == KvCacheStorage::Fp8E4M3Row256) {
            if constexpr (Page)
                return write_fp8_page_kernel<G, Metadata>;
            else if constexpr (Batch)
                return write_fp8_batch_kernel<G, Metadata::masked>;
            else
                return write_fp8_kernel<G, Metadata>;
        } else if constexpr (Storage == KvCacheStorage::Nvfp4Group16) {
            if constexpr (Page)
                return write_nvfp4_page_kernel<G, Metadata>;
            else
                return write_nvfp4_kernel<G, Metadata, Batch>;
        } else {
            static_assert(Storage == KvCacheStorage::Fp8KeyNvfp4Value);
            if constexpr (Page)
                return write_k8v4_page_kernel<G, Metadata>;
            else
                return write_k8v4_kernel<G, Metadata, Batch>;
        }
    }();
    using Code =
        std::conditional_t<Storage == KvCacheStorage::Int8Group64, std::int8_t, std::uint8_t>;
    using KeyScale =
        std::conditional_t<Storage == KvCacheStorage::Nvfp4Group16, std::uint8_t, __half>;
    using ValueScale = std::conditional_t<Storage == KvCacheStorage::Nvfp4Group16 ||
                                              Storage == KvCacheStorage::Fp8KeyNvfp4Value,
                                          std::uint8_t, __half>;
    const dim3 grid  = Page ? dim3(common::div_up(a.width + 7, 8), G::KVHeads)
                            : dim3(common::div_up(a.width * G::KVHeads, 8), 1, Batch ? a.batch : 1);
    kernel<<<grid, 256, 0, stream>>>(
        a.k, a.v, a.positions, metadata, static_cast<Code*>(a.keys), static_cast<Code*>(a.values),
        static_cast<KeyScale*>(a.key_scales), static_cast<ValueScale*>(a.value_scales), a.width);
}

// port (sm_89): the rk write path is shared by the layouts that differ only in their codec flags.
template <class G, bool PackedK, bool E8Lattice, bool E8Root, bool V8Root, bool V8Lattice, class Metadata>
void launch_rk_storage(const WriteArguments& a, Metadata metadata, cudaStream_t stream) {
    if constexpr (requires { Metadata::masked; }) {
        launch_rk_write<Metadata::masked, G, PackedK, E8Lattice, E8Root, V8Root, V8Lattice, Metadata>(
            a.k, a.v, a.positions, metadata, static_cast<std::int8_t*>(a.keys),
            static_cast<std::int8_t*>(a.values), static_cast<__half*>(a.key_scales),
            static_cast<__half*>(a.value_scales), a.width, a.batch, stream);
    } else {
        launch_rk_write<false, G, PackedK, E8Lattice, E8Root, V8Root, V8Lattice, Metadata>(
            a.k, a.v, a.positions, metadata, static_cast<std::int8_t*>(a.keys),
            static_cast<std::int8_t*>(a.values), static_cast<__half*>(a.key_scales),
            static_cast<__half*>(a.value_scales), a.width, 1, stream);
    }
}

template <KvCacheStorage Storage, class G, class Metadata>
void launch_instance(PagedWriteInstance instance, const WriteArguments& a, Metadata metadata,
                     cudaStream_t stream) {
    if constexpr (Storage == KvCacheStorage::RotatedInt8KeyInt4ValueGroup64) {
        // rk8v4: int8 K, packed int4 V.
        return launch_rk_storage<G, false, false, false, false, false>(a, metadata, stream);
    } else if constexpr (Storage == KvCacheStorage::RotatedInt4KeyInt4ValueGroup64) {
        // rk4v4: packed int4 K and V.
        return launch_rk_storage<G, true, false, false, false, false>(a, metadata, stream);
    } else if constexpr (Storage == KvCacheStorage::RK4V4E8) {
        // rk4v4-e8: E8-lattice key carrier (96 B plane) + packed int4 V.
        return launch_rk_storage<G, true, true, false, false, false>(a, metadata, stream);
    } else if constexpr (Storage == KvCacheStorage::RK2V4E8) {
        // rk2v4-e8: E8-cylinder key carrier (64 B plane) + packed int4 V.
        return launch_rk_storage<G, false, false, true, false, false>(a, metadata, stream);
    } else if constexpr (Storage == KvCacheStorage::RK4V2E8) {
        // rk4v2-e8: E8-lattice key carrier + E8-cylinder V.
        return launch_rk_storage<G, true, true, false, true, false>(a, metadata, stream);
        } else if constexpr (Storage == KvCacheStorage::RK4V3E8) {
        return launch_rk_storage<G, true, true, false, false, true>(a, metadata, stream);
} else if constexpr (Storage == KvCacheStorage::BFloat16) {
        if (a.batch != 1 ||
            (instance != PagedWriteInstance::Copy96 && instance != PagedWriteInstance::Copy128))
            throw std::logic_error("KV write: invalid BF16 instance");
        const int threads = instance == PagedWriteInstance::Copy128 ? 128 : 96;
        const auto units  = static_cast<std::int64_t>(a.width) * G::KVHeads * (256 / 8);
        const int blocks  = common::div_up(units, static_cast<std::int64_t>(threads));
        write_bf16_kernel<G, Metadata><<<blocks, threads, 0, stream>>>(
            a.k, a.v, a.positions, metadata, static_cast<__nv_bfloat16*>(a.keys),
            static_cast<__half*>(a.values), a.width);
        return;
    } else {
        switch (instance) {
        case PagedWriteInstance::WarpRows256:
            return launch_quantized<Storage, G, Metadata, PagedWriteInstance::WarpRows256>(
                a, metadata, stream);
        case PagedWriteInstance::Page8x256:
            return launch_quantized<Storage, G, Metadata, PagedWriteInstance::Page8x256>(
                a, metadata, stream);
        case PagedWriteInstance::BatchRows256:
            if constexpr (requires { Metadata::masked; })
                return launch_quantized<Storage, G, Metadata, PagedWriteInstance::BatchRows256>(
                    a, metadata, stream);
            break;
        default:
            break;
        }
        throw std::logic_error("KV write: invalid quantized instance");
    }
}

template <class G, class Metadata>
void storage_dispatch(KvCacheStorage storage, PagedWriteInstance instance, const WriteArguments& a,
                      Metadata metadata, cudaStream_t stream) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return launch_instance<KvCacheStorage::BFloat16, G>(instance, a, metadata, stream);
    case KvCacheStorage::Int8Group64:
        return launch_instance<KvCacheStorage::Int8Group64, G>(instance, a, metadata, stream);
    case KvCacheStorage::Fp8E4M3Row256:
        return launch_instance<KvCacheStorage::Fp8E4M3Row256, G>(instance, a, metadata, stream);
    case KvCacheStorage::Nvfp4Group16:
        return launch_instance<KvCacheStorage::Nvfp4Group16, G>(instance, a, metadata, stream);
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return launch_instance<KvCacheStorage::Fp8KeyNvfp4Value, G>(instance, a, metadata, stream);
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
        return launch_instance<KvCacheStorage::RotatedInt8KeyInt4ValueGroup64, G>(instance, a,
                                                                                 metadata, stream);
    case KvCacheStorage::RK4V4E8:
        return launch_instance<KvCacheStorage::RK4V4E8, G>(instance, a, metadata, stream);
    case KvCacheStorage::RK2V4E8:
        return launch_instance<KvCacheStorage::RK2V4E8, G>(instance, a, metadata, stream);
    case KvCacheStorage::RK4V2E8:
        return launch_instance<KvCacheStorage::RK4V2E8, G>(instance, a, metadata, stream);
    case KvCacheStorage::RK4V3E8:
        return launch_instance<KvCacheStorage::RK4V3E8, G>(instance, a, metadata, stream);
    case KvCacheStorage::RotatedInt4KeyInt4ValueGroup64:
        return launch_instance<KvCacheStorage::RotatedInt4KeyInt4ValueGroup64, G>(instance, a,
                                                                                 metadata, stream);
    }
    throw std::logic_error("KV write: unsupported storage");
}

template <class Metadata>
void geometry_dispatch(int heads, KvCacheStorage storage, PagedWriteInstance instance,
                       const WriteArguments& a, Metadata metadata, cudaStream_t stream) {
    if (heads == 4)
        storage_dispatch<D256Geometry<4>>(storage, instance, a, metadata, stream);
    else
        storage_dispatch<D256Geometry<2>>(storage, instance, a, metadata, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_paged_write(PagedWriteInstance instance, const Tensor& k, const Tensor& v,
                        const Tensor& positions, PagedKVLayerView cache, cudaStream_t stream) {
    geometry_dispatch(k.ne[1], cache.storage, instance, arguments(k, v, positions, cache),
                      DirectMetadata{static_cast<const std::int32_t*>(cache.block_table.data)},
                      stream);
}

void launch_paged_write(PagedWriteInstance instance, const Tensor& k, const Tensor& v,
                        const Tensor& positions, const Tensor& valid_columns,
                        const Tensor& table_rows, PagedKVBatchLayerView cache,
                        cudaStream_t stream) {
    const auto launch = [&]<bool Masked>() {
        const BatchMetadata<Masked> metadata{
            static_cast<const std::int32_t*>(cache.block_tables.data),
            static_cast<const std::int32_t*>(valid_columns.data),
            static_cast<const std::int32_t*>(table_rows.data), cache.block_tables.ne[0]};
        geometry_dispatch(k.ne[1], cache.storage, instance, arguments(k, v, positions, cache),
                          metadata, stream);
    };
    if (valid_columns.data)
        launch.template operator()<true>();
    else
        launch.template operator()<false>();
}

// port (sm_89) diagnostic (IT.30): V-plane distribution counters for the 2-bit E8 cylinder carrier.
// The counters live in rk_kernel.cuh with internal linkage; this TU owns both the only V8Root
// instantiation and the dump kernel, so a same-TU kernel reads them without symbol lookup.
__global__ void v8_stats_dump_kernel(unsigned long long* hist, unsigned long long* groups,
                                     double* absmax_sum, int* scale_bad, int* scale_inf) {
    *scale_bad  = 0;
    *scale_inf  = 0;
    for (int i = 0; i < 16; ++i) hist[i] = kv_v8_rad_hist[i];
    *groups     = kv_v8_groups;
    *absmax_sum = kv_v8_absmax_sum;
}

__global__ void v8_stats_enable_kernel(int on) { kv_v8_stats_on = on; }

void kv_v8_stats_enable(int on) {
    v8_stats_enable_kernel<<<1, 1>>>(on ? 1 : 0);
    (void)cudaDeviceSynchronize();
}

void kv_v8_stats_dump() {
    unsigned long long* d_hist   = nullptr;
    unsigned long long* d_groups = nullptr;
    double* d_sum                = nullptr;
    cudaMalloc(&d_hist, 16 * sizeof(unsigned long long));
    cudaMalloc(&d_groups, sizeof(unsigned long long));
    cudaMalloc(&d_sum, sizeof(double));
    int* d_bad = nullptr; int* d_inf = nullptr;
    cudaMalloc(&d_bad, sizeof(int)); cudaMalloc(&d_inf, sizeof(int));
    v8_stats_dump_kernel<<<1, 1>>>(d_hist, d_groups, d_sum, d_bad, d_inf);
    (void)cudaDeviceSynchronize();

    unsigned long long hist[16] = {};
    unsigned long long groups   = 0;
    double absmax_sum           = 0.0;
    cudaMemcpy(hist, d_hist, sizeof(hist), cudaMemcpyDeviceToHost);
    cudaMemcpy(&groups, d_groups, sizeof(groups), cudaMemcpyDeviceToHost);
    cudaMemcpy(&absmax_sum, d_sum, sizeof(absmax_sum), cudaMemcpyDeviceToHost);
    int scale_bad = 0, scale_inf = 0;
    cudaMemcpy(&scale_bad, d_bad, sizeof(scale_bad), cudaMemcpyDeviceToHost);
    cudaMemcpy(&scale_inf, d_inf, sizeof(scale_inf), cudaMemcpyDeviceToHost);
    cudaFree(d_hist);
    cudaFree(d_groups);
    cudaFree(d_sum);

    unsigned long long total = 0;
    for (auto h : hist) total += h;
    std::printf("v8-stats: groups=%llu subspaces=%llu mean_absmax=%.4f\n",
                static_cast<unsigned long long>(groups), static_cast<unsigned long long>(total),
                groups ? absmax_sum / static_cast<double>(groups) : 0.0);
    std::printf("v8-stats: v_scale<=0 = %d, v_scale=inf = %d\n", scale_bad, scale_inf);
    std::printf("v8-stats: rad_idx histogram (1..15, 0 = untouched)\n");
    for (int i = 0; i < 16; ++i) {
        std::printf("  rad=%2d  %10llu  %5.2f%%\n", i,
                    static_cast<unsigned long long>(hist[i]),
                    total ? 100.0 * static_cast<double>(hist[i]) / static_cast<double>(total) : 0.0);
    }
}

} // namespace ninfer::ops::detail::kv








