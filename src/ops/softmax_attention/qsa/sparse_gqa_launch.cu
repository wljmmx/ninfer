#include "ops/softmax_attention/qsa/sparse_gqa_plan.h"
#include "ops/softmax_attention/qsa/sparse_gqa_prefill.cuh"
#include "ops/softmax_attention/qsa/sparse_gqa_simt.cuh"

namespace ninfer::ops::detail::qsa {
namespace {
// The output route is fixed by the plan before the launch, so it is a template argument rather than
// a run-time test: each kernel then carries one epilogue instead of both, which halves its body and
// keeps the other route out of its register allocation.
template <KvCacheStorage Storage, bool VectorCache, SparseGqaLoad Load, int Slices>
void launch_mma_slice(SparseGqaArguments a, SparseGqaPlan plan, int queries, unsigned z,
                      DeviceExecutionView execution) {
    const auto grid   = dim3(queries * a.kv_heads, a.splits, z);
    const auto shared = sizeof(SparseGqaStorage<Storage, 32, 2, Load>);
    if (plan.partial)
        sparse_gqa_mma_kernel<Storage, 32, 2, VectorCache, Load, false, Slices>
            <<<grid, 64, shared, execution.stream>>>(a);
    else
        sparse_gqa_mma_kernel<Storage, 32, 2, VectorCache, Load, true, Slices>
            <<<grid, 64, shared, execution.stream>>>(a);
}

template <KvCacheStorage Storage, bool VectorCache, SparseGqaLoad Load>
void launch_mma(SparseGqaArguments a, SparseGqaPlan plan, int queries,
                DeviceExecutionView execution) {
    if constexpr (!kSparseSlicedValue<Storage>) {
        launch_mma_slice<Storage, VectorCache, Load, 1>(a, plan, queries, 1u, execution);
        return;
    }
    // An extra slice repeats the QK pass but shortens each block's chain, so it only pays while the
    // grid is well inside the machine. The plan's split count is an upper bound; what the kernel
    // actually launches is min(plan.splits, tiles), and the selection caps the tiles.
    // Measured on sm_89 with the effective count: four slices win below one block per SM and lose
    // at it (T=64 with 128 effective blocks: 13.3 us at two slices, 15.4 at four). Eight slices
    // were measured too and bought nothing - T=1 keeps 10.2 us, and T=8 with five splits regresses
    // to 13.3 from 12.3 because the QK pass is then repeated eight times per tile.
    const auto tiles  = std::max(1, (a.selected * 4 + 3 + a.key_rows - 1) / a.key_rows);
    const auto splits = std::min(a.splits, tiles);
    const auto blocks = std::int64_t(queries) * a.kv_heads * splits;
    if (blocks < execution.multiprocessor_count)
        launch_mma_slice<Storage, VectorCache, Load, 4>(a, plan, queries, 4u, execution);
    else
        launch_mma_slice<Storage, VectorCache, Load, 2>(a, plan, queries, 2u, execution);
}

template <KvCacheStorage Storage, bool VectorCache>
void launch(SparseGqaArguments a, SparseGqaPlan plan, int queries, DeviceExecutionView execution) {
    // rk planes keep codes and scales in separate regions, so the packed-row staging routes do not
    // describe them; the direct read path decodes straight from the two planes.
    constexpr bool Async = kSparseQuantized<Storage> && VectorCache && !kSparseRk<Storage>;
    // V11-07 cuts the output row across z slices; a rotated-value block owns the whole row.
    constexpr unsigned Z = kSparseSlicedValue<Storage> ? 2u : 1u;
    if (plan.instance == SparseGqaInstance::FullPrefix) {
        constexpr auto load   = Async ? SparseGqaLoad::InPlace : SparseGqaLoad::Direct;
        constexpr auto kernel = sparse_gqa_all_select_kernel<Storage, 64, 4, VectorCache, load>;
        constexpr int bytes =
            (sizeof(SparseGqaStorage<Storage, 64, 4, load>) + 15) / 16 * 16 + 4 * 16 * 256 * 2;
        static_assert(bytes <= 99 * 1024);
        static const auto status =
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        CUDA_CHECK(status);
        kernel<<<dim3((queries + 3) / 4 * a.kv_heads, 1, Z), 128, bytes, execution.stream>>>(a);
    } else if (plan.instance == SparseGqaInstance::Simt) {
        sparse_gqa_simt_kernel<Storage, 8, VectorCache>
            <<<dim3(queries * a.query_heads, a.splits), 256, 0, execution.stream>>>(a);
    } else if constexpr (Async) {
        const auto blocks = std::int64_t(queries) * a.kv_heads * a.splits;
        const auto sms    = execution.multiprocessor_count;
        // In-place staging retains three CTAs/SM but adds expansion barriers. Use it when
        // that residency removes a CTA wave; otherwise the separate slot has lower latency.
        const bool inplace =
            (blocks + 3 * sms - 1) / (3 * sms) < (blocks + 2 * sms - 1) / (2 * sms);
        if (inplace)
            launch_mma<Storage, VectorCache, SparseGqaLoad::InPlace>(a, plan, queries, execution);
        else
            launch_mma<Storage, VectorCache, SparseGqaLoad::Staged>(a, plan, queries, execution);
    } else {
        launch_mma<Storage, VectorCache, SparseGqaLoad::Direct>(a, plan, queries, execution);
    }
    if (plan.partial)
        sparse_gqa_merge_kernel<Storage><<<queries * a.query_heads, 128, 0, execution.stream>>>(a);
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
                         0,
                          query.ne[1],
                          cache.num_kv_heads,
                          static_cast<int>(selected.block_ids.ne[0])};
    const bool vector_cache = aligned(cache.k_pages) && aligned(cache.v_pages);
    const auto queries      = std::int64_t(a.tokens) * a.batch;
    // This slab only bounds grid.x (including the per-head merge); scratch/output indexing
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
        case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
            dispatch.template operator()<KvCacheStorage::RotatedInt8KeyInt4ValueGroup64>();
            break;
        case KvCacheStorage::RotatedInt4KeyInt4ValueGroup64:
            dispatch.template operator()<KvCacheStorage::RotatedInt4KeyInt4ValueGroup64>();
            break;
        case KvCacheStorage::RK4V4E8:
            dispatch.template operator()<KvCacheStorage::RK4V4E8>();
            break;
        case KvCacheStorage::RK2V4E8:
            dispatch.template operator()<KvCacheStorage::RK2V4E8>();
            break;
        case KvCacheStorage::RK4V2E8:
            dispatch.template operator()<KvCacheStorage::RK4V2E8>();
            break;
        default:
            break; // Validated by the public wrapper.
        }
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail::qsa
