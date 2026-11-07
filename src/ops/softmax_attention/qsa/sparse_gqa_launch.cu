#include "ops/softmax_attention/qsa/sparse_gqa_plan.h"
#include "ops/softmax_attention/qsa/sparse_gqa_prefill.cuh"
#include "ops/softmax_attention/qsa/sparse_gqa_simt.cuh"

namespace ninfer::ops::detail::qsa {
namespace {
template <KvCacheStorage Storage, bool VectorCache>
void launch(SparseGqaArguments a, SparseGqaPlan plan, int queries, DeviceExecutionView execution) {
    constexpr bool Async = kSparseQuantized<Storage> && VectorCache;
    if (plan.instance == SparseGqaInstance::FullPrefix) {
        constexpr auto load   = Async ? SparseGqaLoad::InPlace : SparseGqaLoad::Direct;
        constexpr auto kernel = sparse_gqa_all_select_kernel<Storage, 64, 4, VectorCache, load>;
        constexpr int bytes =
            (sizeof(SparseGqaStorage<Storage, 64, 4, load>) + 15) / 16 * 16 + 4 * 16 * 256 * 2;
        static_assert(bytes <= 99 * 1024);
        static const auto status =
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        CUDA_CHECK(status);
        kernel<<<(queries + 3) / 4 * 2, 128, bytes, execution.stream>>>(a);
    } else if (plan.instance == SparseGqaInstance::Simt) {
        sparse_gqa_simt_kernel<Storage, 8, VectorCache>
            <<<dim3(queries * 24, a.splits), 256, 0, execution.stream>>>(a);
    } else if constexpr (Async) {
        const auto blocks = std::int64_t(queries) * 2 * a.splits;
        const auto sms    = execution.multiprocessor_count;
        // In-place staging retains three CTAs/SM but adds expansion barriers. Use it when
        // that residency removes a CTA wave; otherwise the separate slot has lower latency.
        const bool inplace =
            (blocks + 3 * sms - 1) / (3 * sms) < (blocks + 2 * sms - 1) / (2 * sms);
        if (inplace)
            sparse_gqa_mma_kernel<Storage, 32, 2, VectorCache, SparseGqaLoad::InPlace>
                <<<dim3(queries * 2, a.splits), 64,
                   sizeof(SparseGqaStorage<Storage, 32, 2, SparseGqaLoad::InPlace>),
                   execution.stream>>>(a);
        else
            sparse_gqa_mma_kernel<Storage, 32, 2, VectorCache, SparseGqaLoad::Staged>
                <<<dim3(queries * 2, a.splits), 64,
                   sizeof(SparseGqaStorage<Storage, 32, 2, SparseGqaLoad::Staged>),
                   execution.stream>>>(a);
    } else {
        sparse_gqa_mma_kernel<Storage, 32, 2, VectorCache>
            <<<dim3(queries * 2, a.splits), 64, sizeof(SparseGqaStorage<Storage, 32, 2>),
               execution.stream>>>(a);
    }
    if (plan.partial)
        sparse_gqa_merge_kernel<Storage><<<queries * 24, 128, 0, execution.stream>>>(a);
}
} // namespace

void launch_sparse_gqa(const Tensor& query, const Tensor& gate, const PagedKVBatchLayerView& cache,
                       const QsaBatchView& batch, const QsaBlockSelection& selected,
                       SparseGqaPlan plan, const SparseGqaWorkspace& scratch, Tensor& out,
                       DeviceExecutionView execution) {
    const auto i32  = [](const Tensor& t) { return static_cast<const int*>(t.data); };
    const auto page = [](const Tensor& t) {
        return SparseGqaPage{static_cast<const std::uint8_t*>(t.data), t.nb[1], t.nb[2], t.nb[3]};
    };
    const auto aligned = [](const Tensor& t) {
        return reinterpret_cast<std::uintptr_t>(t.data) % 16 == 0 && t.nb[1] % 16 == 0 &&
               t.nb[2] % 16 == 0 && t.nb[3] % 16 == 0;
    };
    SparseGqaArguments a{static_cast<const __nv_bfloat16*>(query.data),
                         static_cast<const __nv_bfloat16*>(gate.data),
                         static_cast<__nv_bfloat16*>(out.data),
                         page(cache.k_pages),
                         page(cache.v_pages),
                         page(cache.k_scale_pages),
                         page(cache.v_scale_pages),
                         i32(batch.positions),
                         i32(batch.valid_columns),
                         i32(batch.table_rows),
                         i32(cache.block_tables),
                         i32(selected.block_ids),
                         i32(selected.block_counts),
                         cache.block_tables.nb[1] / 4,
                         static_cast<float*>(scratch.numerator.data),
                         static_cast<float*>(scratch.maximum.data),
                         static_cast<float*>(scratch.sum.data),
                         query.ne[2],
                         query.ne[3],
                         plan.splits,
                         plan.key_rows,
                         0};
    const bool vector_cache = aligned(cache.k_pages) && aligned(cache.v_pages);
    const auto queries      = std::int64_t(a.tokens) * a.batch;
    // This slab only bounds grid.x (including the 24-head merge); scratch/output indexing
    // stays global and T is not a semantic launch limit.
    constexpr int slab = 1 << 24;
    for (a.begin = 0; a.begin < queries; a.begin += slab) {
        const int n         = int(std::min<std::int64_t>(slab, queries - a.begin));
        const auto dispatch = [&]<KvCacheStorage Storage>() {
            if (vector_cache)
                launch<Storage, true>(a, plan, n, execution);
            else
                launch<Storage, false>(a, plan, n, execution);
        };
        switch (cache.storage) {
        case KvCacheStorage::BFloat16:
            dispatch.template operator()<KvCacheStorage::BFloat16>();
            break;
        case KvCacheStorage::Fp8E4M3Row256:
            dispatch.template operator()<KvCacheStorage::Fp8E4M3Row256>();
            break;
        case KvCacheStorage::Int8Group64:
            dispatch.template operator()<KvCacheStorage::Int8Group64>();
            break;
        case KvCacheStorage::Nvfp4Group16:
            dispatch.template operator()<KvCacheStorage::Nvfp4Group16>();
            break;
        case KvCacheStorage::Fp8KeyNvfp4Value:
            dispatch.template operator()<KvCacheStorage::Fp8KeyNvfp4Value>();
            break;
        default:
            break; // Validated by the public wrapper.
        }
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail::qsa
