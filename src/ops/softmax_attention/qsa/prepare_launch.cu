#include "ops/softmax_attention/qsa/launch.h"
#include "ops/softmax_attention/qsa/prepare_kernel.cuh"
#include "ops/softmax_attention/qsa/storage.h"

#include <array>
#include <cmath>
#include <stdexcept>

namespace ninfer::ops::detail::qsa {
namespace {

template <class T>
PrepareColumns<T> prepare_columns(const Tensor& t) {
    return {static_cast<T*>(t.data), t.nb[1] / std::int64_t(sizeof(T)),
            t.nb[2] / std::int64_t(sizeof(T))};
}

template <class T>
PreparePage<T> prepare_page(const Tensor& t) {
    return {static_cast<T*>(t.data), t.nb[1] / std::int64_t(sizeof(T)),
            t.nb[2] / std::int64_t(sizeof(T)), t.nb[3] / std::int64_t(sizeof(T))};
}

PrepareArguments prepare_arguments(const QsaProjectionView& p, const QsaPrepareParameters& params,
                                   const QsaBatchView& batch, const Tensor& rope,
                                   const QsaTailStateView& before, QsaCacheView cache,
                                   QsaRawRecordWriteView records, Tensor& q, Tensor& iq) {
    const auto bf  = [](const Tensor& t) { return static_cast<const __nv_bfloat16*>(t.data); };
    const auto i32 = [](const Tensor& t) { return static_cast<const std::int32_t*>(t.data); };
    PrepareArguments a{bf(p.query),
                       bf(p.key),
                       bf(p.value),
                       bf(p.index_query),
                       bf(p.index_key),
                       bf(params.query_norm),
                       bf(params.key_norm),
                       bf(params.index_query_norm),
                       bf(params.index_key_norm),
                       static_cast<__nv_bfloat16*>(q.data),
                       static_cast<__nv_bfloat16*>(iq.data),
                       i32(batch.positions),
                       i32(rope),
                       i32(batch.valid_columns),
                       i32(batch.table_rows),
                       i32(before.slots),
                       i32(records.offsets),
                       i32(cache.main.block_tables),
                       cache.main.block_tables.nb[1] / 4,
                       prepare_columns<const __nv_bfloat16>(before.raw_keys),
                       prepare_columns<const std::int32_t>(before.first_rope),
                       prepare_columns<__nv_bfloat16>(records.raw_keys),
                       prepare_columns<std::int32_t>(records.rope_positions),
                       prepare_page<__nv_bfloat16>(cache.index.keys),
                       prepare_page<std::uint8_t>(cache.main.k_pages),
                       prepare_page<std::uint8_t>(cache.main.v_pages),
                       prepare_page<std::uint8_t>(cache.main.k_scale_pages),
                       prepare_page<std::uint8_t>(cache.main.v_scale_pages),
                       p.query.ne[2],
                       p.query.ne[3],
                       rope.ne[1],
                       p.key.ne[1],
                       params.rms_epsilon,
                       {}};
    // Default model frequencies are constructed once on Host, with no Device allocation.
    static const auto default_frequency = [] {
        std::array<double, 32> result{};
        for (int i = 0; i < 32; ++i) result[i] = std::pow(1e7, -double(i) / 32);
        return result;
    }();
    for (int i = 0; i < 32; ++i)
        a.frequency[i] = params.rope_theta == 1e7F
                             ? default_frequency[i]
                             : std::pow(double(params.rope_theta), -double(i) / 32);
    return a;
}

template <KvCacheStorage Storage, bool BoundedPhase, bool VectorWeights, int KvHeads>
void submit(const PrepareArguments& args, cudaStream_t stream) {
    constexpr int groups = PrepareGeometry<KvHeads>::groups;
    // Short calls distribute a token's heads across CTAs; larger calls group heads across
    // tokens for memory throughput. The short-call bound also keeps tokens*groups in x.
    if (std::int64_t(args.tokens) * args.batch <= 320)
        prepare_kernel<Storage, BoundedPhase, VectorWeights, true, KvHeads>
            <<<dim3(args.tokens * groups, args.batch), 128, 0, stream>>>(args);
    else
        prepare_kernel<Storage, BoundedPhase, VectorWeights, false, KvHeads>
            <<<dim3(args.tokens, args.batch, groups), 128, 0, stream>>>(args);
}

template <KvCacheStorage Storage, int KvHeads>
void dispatch_geometry(const PrepareArguments& args, float theta, cudaStream_t stream) {
    const bool vector_weights = reinterpret_cast<std::uintptr_t>(args.wq) % 16 == 0 &&
                                reinterpret_cast<std::uintptr_t>(args.wiq) % 8 == 0;
    // For theta>=1 every phase is bounded by the nonnegative-I32 coordinate domain.
    if (theta >= 1) {
        if (vector_weights)
            submit<Storage, true, true, KvHeads>(args, stream);
        else
            submit<Storage, true, false, KvHeads>(args, stream);
    } else {
        if (vector_weights)
            submit<Storage, false, true, KvHeads>(args, stream);
        else
            submit<Storage, false, false, KvHeads>(args, stream);
    }
}

template <KvCacheStorage Storage>
void dispatch(const PrepareArguments& args, float theta, cudaStream_t stream) {
    if (args.kv_heads == 1)
        dispatch_geometry<Storage, 1>(args, theta, stream);
    else
        dispatch_geometry<Storage, 2>(args, theta, stream);
}

} // namespace

void launch_prepare(const QsaProjectionView& projected, const QsaPrepareParameters& parameters,
                    const QsaBatchView& batch, const Tensor& rope_positions,
                    const QsaTailStateView& before, QsaCacheView cache,
                    QsaRawRecordWriteView records, Tensor& query, Tensor& index_query,
                    DeviceExecutionView execution) {
    const auto args = prepare_arguments(projected, parameters, batch, rope_positions, before, cache,
                                        records, query, index_query);
    switch (cache.main.storage) {
    case KvCacheStorage::BFloat16:
        dispatch<KvCacheStorage::BFloat16>(args, parameters.rope_theta, execution.stream);
        break;
    case KvCacheStorage::Int8Group64:
        dispatch<KvCacheStorage::Int8Group64>(args, parameters.rope_theta, execution.stream);
        break;
    case KvCacheStorage::Fp8E4M3Row256:
        dispatch<KvCacheStorage::Fp8E4M3Row256>(args, parameters.rope_theta, execution.stream);
        break;
    case KvCacheStorage::Nvfp4Group16:
        dispatch<KvCacheStorage::Nvfp4Group16>(args, parameters.rope_theta, execution.stream);
        break;
    case KvCacheStorage::Fp8KeyNvfp4Value:
        dispatch<KvCacheStorage::Fp8KeyNvfp4Value>(args, parameters.rope_theta, execution.stream);
        break;
    default:
        // Unreachable through the public entry, which refuses an unimplemented storage by name
        // before it gets here; kept so a new enumerator cannot be dispatched silently.
        throw_unsupported_storage("qsa_prepare", cache.main.storage);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail::qsa
