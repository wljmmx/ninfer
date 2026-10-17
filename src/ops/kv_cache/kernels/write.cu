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
template <class G, bool PackedK, bool E8Lattice, bool E8Root, bool V8Root, class Metadata>
void launch_rk_storage(const WriteArguments& a, Metadata metadata, cudaStream_t stream) {
    if constexpr (requires { Metadata::masked; }) {
        launch_rk_write<Metadata::masked, G, PackedK, E8Lattice, E8Root, V8Root, Metadata>(
            a.k, a.v, a.positions, metadata, static_cast<std::int8_t*>(a.keys),
            static_cast<std::int8_t*>(a.values), static_cast<__half*>(a.key_scales),
            static_cast<__half*>(a.value_scales), a.width, a.batch, stream);
    } else {
        launch_rk_write<false, G, PackedK, E8Lattice, E8Root, V8Root, Metadata>(
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
        return launch_rk_storage<G, false, false, false, false>(a, metadata, stream);
    } else if constexpr (Storage == KvCacheStorage::RotatedInt4KeyInt4ValueGroup64) {
        // rk4v4: packed int4 K and V.
        return launch_rk_storage<G, true, false, false, false>(a, metadata, stream);
    } else if constexpr (Storage == KvCacheStorage::RK4V4E8) {
        // rk4v4-e8: E8-lattice key carrier (96 B plane) + packed int4 V.
        return launch_rk_storage<G, true, true, false, false>(a, metadata, stream);
    } else if constexpr (Storage == KvCacheStorage::RK2V4E8) {
        // rk2v4-e8: E8-cylinder key carrier (64 B plane) + packed int4 V.
        return launch_rk_storage<G, false, false, true, false>(a, metadata, stream);
    } else if constexpr (Storage == KvCacheStorage::RK4V2E8) {
        // rk4v2-e8: E8-lattice key carrier + E8-cylinder V.
        return launch_rk_storage<G, true, true, false, true>(a, metadata, stream);
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

} // namespace ninfer::ops::detail::kv
