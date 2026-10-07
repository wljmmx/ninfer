#include "ops/softmax_attention/dense/causal_cache/int8/launch.h"
#include "ops/softmax_attention/dense/causal_cache/int8/instances.h"
#include "ops/softmax_attention/dense/causal_cache/int8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/int8/template_launch.cuh"
#include "ops/kv_cache/append/launch.h"

namespace ninfer::ops::detail {
namespace {

template <class G, int Tokens, class Input, bool Writable>
void grouped(const CausalAttentionOperands& p, Int8KvCacheView<Writable> cache, Input input,
             CausalKvPartition partition, CausalPartialView partial, cudaStream_t stream) {
    using Instance    = Int8KvGroupedInstance<G, Tokens>;
    const auto invoke = [&]<bool MultiBatch, bool Masked>() {
        launch_int8_kv_grouped_mma<G, typename Instance::Schedule, MultiBatch, Masked>(
            p, cache, input, partition, partial, stream);
        launch_causal_natural_merge<G, typename Instance::Merge, MultiBatch, Masked, false>(
            p, cache.valid_columns, partition, partial, stream);
    };
    if (p.batch == 1) {
        if (cache.valid_columns)
            invoke.template operator()<false, true>();
        else
            invoke.template operator()<false, false>();
    } else {
        if (cache.valid_columns)
            invoke.template operator()<true, true>();
        else
            invoke.template operator()<true, false>();
    }
}

// rk4v4: same scheduling as int8 but with the PackedV=true template parameter, which
// switches the append to int4 V packing and the loader to int4 unpacking.
template <class G, int Tokens, class Input, bool Writable>
void rk_grouped(const CausalAttentionOperands& p, Int8KvCacheView<Writable> cache, Input input,
                CausalKvPartition partition, CausalPartialView partial, cudaStream_t stream) {
    using Instance    = Int8KvGroupedInstance<G, Tokens>;
    const auto invoke = [&]<bool MultiBatch, bool Masked>() {
        launch_int8_kv_grouped_mma<G, typename Instance::Schedule, MultiBatch, Masked, Writable,
                                   Input, false, true, false, false>(p, cache, input, partition,
                                                                     partial, stream);
        launch_causal_natural_merge<G, typename Instance::Merge, MultiBatch, Masked, false>(
            p, cache.valid_columns, partition, partial, stream);
    };
    if (p.batch == 1) {
        if (cache.valid_columns)
            invoke.template operator()<false, true>();
        else
            invoke.template operator()<false, false>();
    } else {
        if (cache.valid_columns)
            invoke.template operator()<true, true>();
        else
            invoke.template operator()<true, false>();
    }
}

// rk4v4: PackedV=true + PackedK=true (plain int4 RTN K + int4 V).
// The grouped/fused path (width <= 8: decode rounds, small prefills) must read
// AND append K with the SAME int4-packed codec the standalone batch append
// writes — routing it to the int8-K kernel (PackedK=false) made every decode
// round read int4 nibbles as int8 codes and corrupted the whole context.
template <class G, int Tokens, class Input, bool Writable>
void rk_packedk_grouped(const CausalAttentionOperands& p, Int8KvCacheView<Writable> cache,
                        Input input, CausalKvPartition partition, CausalPartialView partial,
                        cudaStream_t stream) {
    using Instance    = Int8KvGroupedInstance<G, Tokens>;
    const auto invoke = [&]<bool MultiBatch, bool Masked>() {
        launch_int8_kv_grouped_mma<G, typename Instance::Schedule, MultiBatch, Masked, Writable,
                                   Input, false, true, true, false, false>(p, cache, input,
                                                                          partition, partial,
                                                                          stream);
        launch_causal_natural_merge<G, typename Instance::Merge, MultiBatch, Masked, false>(
            p, cache.valid_columns, partition, partial, stream);
    };
    if (p.batch == 1) {
        if (cache.valid_columns)
            invoke.template operator()<false, true>();
        else
            invoke.template operator()<false, false>();
    } else {
        if (cache.valid_columns)
            invoke.template operator()<true, true>();
        else
            invoke.template operator()<true, false>();
    }
}

// rk4v4-e8: PackedV=true + PackedK=true + E8Lattice=true
template <class G, int Tokens, class Input, bool Writable>
void rk_e8_grouped(const CausalAttentionOperands& p, Int8KvCacheView<Writable> cache, Input input,
                   CausalKvPartition partition, CausalPartialView partial, cudaStream_t stream) {
    using Instance    = Int8KvGroupedInstance<G, Tokens>;
    const auto invoke = [&]<bool MultiBatch, bool Masked>() {
        launch_int8_kv_grouped_mma<G, typename Instance::Schedule, MultiBatch, Masked, Writable,
                                   Input, false, true, true, true, false>(p, cache, input, partition,
                                                                    partial, stream);
        launch_causal_natural_merge<G, typename Instance::Merge, MultiBatch, Masked, false>(
            p, cache.valid_columns, partition, partial, stream);
    };
    if (p.batch == 1) {
        if (cache.valid_columns)
            invoke.template operator()<false, true>();
        else
            invoke.template operator()<false, false>();
    } else {
        if (cache.valid_columns)
            invoke.template operator()<true, true>();
        else
            invoke.template operator()<true, false>();
    }
}

// rk2v4-e8: PackedV=true + E8Root=true (2-bit E8 cylinder K + 4-bit V)
template <class G, int Tokens, class Input, bool Writable>
void rk_root_grouped(const CausalAttentionOperands& p, Int8KvCacheView<Writable> cache, Input input,
                     CausalKvPartition partition, CausalPartialView partial, cudaStream_t stream) {
    using Instance    = Int8KvGroupedInstance<G, Tokens>;
    const auto invoke = [&]<bool MultiBatch, bool Masked>() {
        launch_int8_kv_grouped_mma<G, typename Instance::Schedule, MultiBatch, Masked, Writable,
                                   Input, false, true, false, false, true>(p, cache, input,
                                                                            partition, partial,
                                                                            stream);
        launch_causal_natural_merge<G, typename Instance::Merge, MultiBatch, Masked, false>(
            p, cache.valid_columns, partition, partial, stream);
    };
    if (p.batch == 1) {
        if (cache.valid_columns)
            invoke.template operator()<false, true>();
        else
            invoke.template operator()<false, false>();
    } else {
        if (cache.valid_columns)
            invoke.template operator()<true, true>();
        else
            invoke.template operator()<true, false>();
    }
}

template <class G, class Input, bool Writable>
void grouped_instance(const CausalAttentionOperands& p, Int8KvCacheView<Writable> cache,
                      Input input, CausalKvPartition partition, CausalPartialView partial,
                      cudaStream_t stream) {
    switch (p.width) {
#define NINFER_INT8_GROUPED(T)                                                                     \
    case T:                                                                                        \
        return grouped<G, T>(p, cache, input, partition, partial, stream)
        NINFER_INT8_GROUPED(1);
        NINFER_INT8_GROUPED(2);
        NINFER_INT8_GROUPED(3);
        NINFER_INT8_GROUPED(4);
        NINFER_INT8_GROUPED(5);
        NINFER_INT8_GROUPED(6);
        NINFER_INT8_GROUPED(7);
        NINFER_INT8_GROUPED(8);
#undef NINFER_INT8_GROUPED
    }
    throw std::logic_error("INT8 grouped plan exceeds the selected token tile");
}

template <class G, class Input, bool Writable>
void rk_grouped_instance(const CausalAttentionOperands& p, Int8KvCacheView<Writable> cache,
                         Input input, CausalKvPartition partition, CausalPartialView partial,
                         cudaStream_t stream) {
    switch (p.width) {
#define NINFER_RK_GROUPED(T)                                                                       \
    case T:                                                                                        \
        return rk_grouped<G, T>(p, cache, input, partition, partial, stream)
        NINFER_RK_GROUPED(1);
        NINFER_RK_GROUPED(2);
        NINFER_RK_GROUPED(3);
        NINFER_RK_GROUPED(4);
        NINFER_RK_GROUPED(5);
        NINFER_RK_GROUPED(6);
        NINFER_RK_GROUPED(7);
        NINFER_RK_GROUPED(8);
#undef NINFER_RK_GROUPED
    }
    throw std::logic_error("rk grouped plan exceeds the selected token tile");
}

template <class G, class Input, bool Writable>
void rk_packedk_grouped_instance(const CausalAttentionOperands& p, Int8KvCacheView<Writable> cache,
                                 Input input, CausalKvPartition partition, CausalPartialView partial,
                                 cudaStream_t stream) {
    switch (p.width) {
#define NINFER_RK_PK_GROUPED(T)                                                                    \
    case T:                                                                                        \
        return rk_packedk_grouped<G, T>(p, cache, input, partition, partial, stream)
        NINFER_RK_PK_GROUPED(1);
        NINFER_RK_PK_GROUPED(2);
        NINFER_RK_PK_GROUPED(3);
        NINFER_RK_PK_GROUPED(4);
        NINFER_RK_PK_GROUPED(5);
        NINFER_RK_PK_GROUPED(6);
        NINFER_RK_PK_GROUPED(7);
        NINFER_RK_PK_GROUPED(8);
#undef NINFER_RK_PK_GROUPED
    }
    throw std::logic_error("rk4v4 grouped plan exceeds the selected token tile");
}

template <class G, class Input, bool Writable>
void rk_e8_grouped_instance(const CausalAttentionOperands& p, Int8KvCacheView<Writable> cache,
                            Input input, CausalKvPartition partition, CausalPartialView partial,
                            cudaStream_t stream) {
    switch (p.width) {
#define NINFER_RK_E8_GROUPED(T)                                                                    \
    case T:                                                                                        \
        return rk_e8_grouped<G, T>(p, cache, input, partition, partial, stream)
        NINFER_RK_E8_GROUPED(1);
        NINFER_RK_E8_GROUPED(2);
        NINFER_RK_E8_GROUPED(3);
        NINFER_RK_E8_GROUPED(4);
        NINFER_RK_E8_GROUPED(5);
        NINFER_RK_E8_GROUPED(6);
        NINFER_RK_E8_GROUPED(7);
        NINFER_RK_E8_GROUPED(8);
#undef NINFER_RK_E8_GROUPED
    }
    throw std::logic_error("rk-e8 grouped plan exceeds the selected token tile");
}

template <class G, class Input, bool Writable>
void rk_root_grouped_instance(const CausalAttentionOperands& p, Int8KvCacheView<Writable> cache,
                              Input input, CausalKvPartition partition, CausalPartialView partial,
                              cudaStream_t stream) {
    switch (p.width) {
#define NINFER_RK_ROOT_GROUPED(T)                                                                  \
    case T:                                                                                        \
        return rk_root_grouped<G, T>(p, cache, input, partition, partial, stream)
        NINFER_RK_ROOT_GROUPED(1);
        NINFER_RK_ROOT_GROUPED(2);
        NINFER_RK_ROOT_GROUPED(3);
        NINFER_RK_ROOT_GROUPED(4);
        NINFER_RK_ROOT_GROUPED(5);
        NINFER_RK_ROOT_GROUPED(6);
        NINFER_RK_ROOT_GROUPED(7);
        NINFER_RK_ROOT_GROUPED(8);
#undef NINFER_RK_ROOT_GROUPED
    }
    throw std::logic_error("rk-root grouped plan exceeds the selected token tile");
}

template <class Input>
void execute_grouped(const Tensor& q, const Tensor& positions, float scale,
                     PagedKVBatchLayerView cache, const Tensor* valid, const Tensor* rows,
                     Input input, const Int8KvCausalPlan& plan, WorkspaceArena& workspace,
                     Tensor& out, cudaStream_t stream) {
    const auto view =
        make_quantized_causal_cache_view<Int8KvCacheView<Input::writes_cache>>(cache, valid, rows);
    auto scope         = workspace.scope();
    const auto partial = allocate_causal_partials(workspace, plan.query_heads, plan.width,
                                                  plan.partition.capacity, plan.batch);
    const auto p = make_causal_operands(q, positions, out, scale, plan.envelope.max_visible_keys);
    if (plan.query_heads == 24)
        grouped_instance<CausalD256H24Kv4>(p, view, input, plan.partition, partial.view(), stream);
    else
        grouped_instance<CausalD256H16Kv2>(p, view, input, plan.partition, partial.view(), stream);
}

template <class G, int Tokens>
void parallel_grouped(const CausalAttentionOperands& p, Int8KvReadView cache,
                      CausalKvPartition partition, CausalPartialView partial, cudaStream_t stream) {
    using Instance    = Int8KvGroupedInstance<G, Tokens>;
    const auto invoke = [&]<bool MultiBatch, bool Masked>() {
        launch_int8_kv_grouped_mma<G, typename Instance::Schedule, MultiBatch, Masked, false,
                                   CausalCachedInput, true>(p, cache, {}, partition, partial,
                                                            stream);
        launch_causal_natural_merge<G, typename Instance::Merge, MultiBatch, Masked, false>(
            p, cache.valid_columns, partition, partial, stream);
    };
    if (p.batch == 1) {
        if (cache.valid_columns)
            invoke.template operator()<false, true>();
        else
            invoke.template operator()<false, false>();
    } else {
        if (cache.valid_columns)
            invoke.template operator()<true, true>();
        else
            invoke.template operator()<true, false>();
    }
}

void execute_parallel(const CausalAttentionOperands& p, Int8KvReadView cache,
                      const Int8KvCausalPlan& plan, WorkspaceArena& workspace,
                      cudaStream_t stream) {
    auto scope         = workspace.scope();
    const auto partial = allocate_causal_partials(workspace, plan.query_heads, plan.width,
                                                   plan.partition.capacity, plan.batch);
    if (plan.query_heads == 24)
        parallel_grouped<CausalD256H24Kv4, Int8KvCausalPlan::kTokenTile>(p, cache, plan.partition,
                                                                         partial.view(), stream);
    else
        parallel_grouped<CausalD256H16Kv2, Int8KvCausalPlan::kTokenTile>(p, cache, plan.partition,
                                                                         partial.view(), stream);
}

enum class RkVariant { Plain, PackedKOnly, E8Lattice, E8Root };

// rk parallel: same as execute_parallel but with PackedV=true (and optionally PackedK/E8Root).
template <class G, int Tokens, bool PackedK = false, bool E8Root = false>
void rk_parallel_grouped(const CausalAttentionOperands& p, Int8KvReadView cache,
                         CausalKvPartition partition, CausalPartialView partial,
                         cudaStream_t stream) {
    using Instance    = Int8KvGroupedInstance<G, Tokens>;
    const auto invoke = [&]<bool MultiBatch, bool Masked>() {
        launch_int8_kv_grouped_mma<G, typename Instance::Schedule, MultiBatch, Masked, false,
                                   CausalCachedInput, true, true, PackedK, false, E8Root>(
            p, cache, {}, partition, partial, stream);
        launch_causal_natural_merge<G, typename Instance::Merge, MultiBatch, Masked, false>(
            p, cache.valid_columns, partition, partial, stream);
    };
    if (p.batch == 1) {
        if (cache.valid_columns)
            invoke.template operator()<false, true>();
        else
            invoke.template operator()<false, false>();
    } else {
        if (cache.valid_columns)
            invoke.template operator()<true, true>();
        else
            invoke.template operator()<true, false>();
    }
}

template <RkVariant Variant>
void rk_execute_parallel(const CausalAttentionOperands& p, Int8KvReadView cache,
                         const Int8KvCausalPlan& plan, WorkspaceArena& workspace,
                         cudaStream_t stream) {
    auto scope         = workspace.scope();
    const auto partial = allocate_causal_partials(workspace, plan.query_heads, plan.width,
                                                   plan.partition.capacity, plan.batch);
    const auto dispatch = [&]<class G>() {
        if constexpr (Variant == RkVariant::E8Root)
            rk_parallel_grouped<G, Int8KvCausalPlan::kTokenTile, false, true>(p, cache,
                plan.partition, partial.view(), stream);
        else if constexpr (Variant == RkVariant::E8Lattice)
            rk_parallel_grouped<G, Int8KvCausalPlan::kTokenTile, true, false>(p, cache,
                plan.partition, partial.view(), stream);
        else if constexpr (Variant == RkVariant::PackedKOnly)
            rk_parallel_grouped<G, Int8KvCausalPlan::kTokenTile, true, false>(p, cache,
                plan.partition, partial.view(), stream);
        else // Plain (rk8v4: int8 K, no PackedK)
            rk_parallel_grouped<G, Int8KvCausalPlan::kTokenTile, false, false>(p, cache,
                plan.partition, partial.view(), stream);
    };
    if (plan.query_heads == 24)
        dispatch.template operator()<CausalD256H24Kv4>();
    else
        dispatch.template operator()<CausalD256H16Kv2>();
}

void tiled(const CausalAttentionOperands& p, Int8KvReadView cache, cudaStream_t stream) {
    if (p.query_heads == 24)
        launch_int8_kv_tiled_mma<CausalD256H24Kv4, Int8KvTiledInstance>(p, cache, stream);
    else
        launch_int8_kv_tiled_mma<CausalD256H16Kv2, Int8KvTiledInstance>(p, cache, stream);
}

} // namespace

void int8_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& positions, const Tensor& valid, const Tensor& rows,
                              float scale, PagedKVBatchLayerView cache,
                              CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                              Tensor& out, DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    const auto plan           = make_int8_kv_causal_plan(q.ne[1], q.ne[2], q.ne[3], envelope,
                                                         execution.multiprocessor_count);
    if (plan.family != Int8KvFamily::Grouped) {
        kv_cache_append_batch_launch(k, v, positions, valid, rows, cache, stream);
        const auto p = make_causal_operands(q, positions, out, scale, envelope.max_visible_keys);
        const auto view =
            make_quantized_causal_cache_view<Int8KvCacheView<false>>(cache, &valid, &rows);
        if (plan.family == Int8KvFamily::Tiled)
            tiled(p, view, stream);
        else
            execute_parallel(p, view, plan, workspace, stream);
    } else {
        execute_grouped(q, positions, scale, cache, &valid, &rows,
                        CausalAppendInput{static_cast<const __nv_bfloat16*>(k.data),
                                          static_cast<const __nv_bfloat16*>(v.data)},
                        plan, workspace, out, stream);
    }
}

void int8_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                              const PagedKVLayerView& cache,
                              CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                              Tensor& out, DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    const auto plan =
        make_int8_kv_causal_plan(q.ne[1], q.ne[2], 1, envelope, execution.multiprocessor_count);
    const auto view = single_row_paged_kv_batch_view(cache);
    if (plan.family == Int8KvFamily::Tiled)
        tiled(make_causal_operands(q, positions, out, scale, envelope.max_visible_keys),
              make_quantized_causal_cache_view<Int8KvCacheView<false>>(view), stream);
    else if (plan.family == Int8KvFamily::ParallelGrouped)
        execute_parallel(make_causal_operands(q, positions, out, scale, envelope.max_visible_keys),
                         make_quantized_causal_cache_view<Int8KvCacheView<false>>(view), plan,
                         workspace, stream);
    else
        execute_grouped(q, positions, scale, view, nullptr, nullptr, CausalCachedInput{}, plan,
                        workspace, out, stream);
}

// --- rk4v4 attention (reuses int8 scheduling with PackedV=true) -------------------

template <class Input, RkVariant Variant>
void rk_execute_grouped(const Tensor& q, const Tensor& positions, float scale,
                        PagedKVBatchLayerView cache, const Tensor* valid, const Tensor* rows,
                        Input input, const Int8KvCausalPlan& plan, WorkspaceArena& workspace,
                        Tensor& out, cudaStream_t stream) {
    const auto view =
        make_quantized_causal_cache_view<Int8KvCacheView<Input::writes_cache>>(cache, valid, rows);
    auto scope         = workspace.scope();
    const auto partial = allocate_causal_partials(workspace, plan.query_heads, plan.width,
                                                   plan.partition.capacity, plan.batch);
    const auto p = make_causal_operands(q, positions, out, scale, plan.envelope.max_visible_keys);
    const auto dispatch = [&]<class G>() {
        if constexpr (Variant == RkVariant::E8Root)
            rk_root_grouped_instance<G>(p, view, input, plan.partition, partial.view(), stream);
        else if constexpr (Variant == RkVariant::E8Lattice)
            rk_e8_grouped_instance<G>(p, view, input, plan.partition, partial.view(), stream);
        else if constexpr (Variant == RkVariant::PackedKOnly)
            rk_packedk_grouped_instance<G>(p, view, input, plan.partition, partial.view(), stream);
        else // Plain (rk8v4: int8 K + int4 V, PackedV=true only)
            rk_grouped_instance<G>(p, view, input, plan.partition, partial.view(), stream);
    };
    if (plan.query_heads == 24)
        dispatch.template operator()<CausalD256H24Kv4>();
    else
        dispatch.template operator()<CausalD256H16Kv2>();
}

void rk4v4_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                               const Tensor& positions, const Tensor& valid, const Tensor& rows,
                               float scale, PagedKVBatchLayerView cache,
                               CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                               Tensor& out, DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    auto plan           = make_int8_kv_causal_plan(q.ne[1], q.ne[2], q.ne[3], envelope,
                                                          execution.multiprocessor_count);
    if (plan.family == Int8KvFamily::Grouped) {
        // Fused append + attention for small widths (≤8).
        rk_execute_grouped<CausalAppendInput, RkVariant::PackedKOnly>(
            q, positions, scale, cache, &valid, &rows,
            CausalAppendInput{static_cast<const __nv_bfloat16*>(k.data),
                              static_cast<const __nv_bfloat16*>(v.data)},
            plan, workspace, out, stream);
    } else {
        // For prefill widths > 8: standalone rk append, then rk parallel attention.
        kv_cache_append_rk4v4_batch_launch(k, v, positions, valid, rows, cache, stream);
        const auto p = make_causal_operands(q, positions, out, scale, envelope.max_visible_keys);
        const auto view =
            make_quantized_causal_cache_view<Int8KvCacheView<false>>(cache, &valid, &rows);
        if (plan.family == Int8KvFamily::Tiled)
            throw std::invalid_argument(
                "rk4v4 attention: tiled family does not yet support PackedV; "
                "use --prefill-chunk 256 or smaller to stay on the parallel path "
                "(the engine clamps rk-family prefill chunks automatically).");
        rk_execute_parallel<RkVariant::PackedKOnly>(p, view, plan, workspace, stream);
    }
}

void rk4v4_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                               const PagedKVLayerView& cache,
                               CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                               Tensor& out, DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    const auto plan =
        make_int8_kv_causal_plan(q.ne[1], q.ne[2], 1, envelope, execution.multiprocessor_count);
    const auto view = single_row_paged_kv_batch_view(cache);
    rk_execute_grouped<CausalCachedInput, RkVariant::Plain>(q, positions, scale, view, nullptr, nullptr,
                                                  CausalCachedInput{}, plan, workspace, out, stream);
}

// --- rk8v4 attention (8-bit K + int4 V, reuses rk4v4 dispatch) --------------------

void rk8v4_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                               const Tensor& positions, const Tensor& valid, const Tensor& rows,
                               float scale, PagedKVBatchLayerView cache,
                               CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                               Tensor& out, DeviceExecutionView execution) {
    // rk8v4 = 8-bit K (no packing) + 4-bit V (PackedV=true, PackedK=false).
    const cudaStream_t stream = execution.stream;
    auto plan           = make_int8_kv_causal_plan(q.ne[1], q.ne[2], q.ne[3], envelope,
                                                          execution.multiprocessor_count);
    if (plan.family == Int8KvFamily::Grouped) {
        rk_execute_grouped<CausalAppendInput, RkVariant::Plain>(
            q, positions, scale, cache, &valid, &rows,
            CausalAppendInput{static_cast<const __nv_bfloat16*>(k.data),
                              static_cast<const __nv_bfloat16*>(v.data)},
            plan, workspace, out, stream);
    } else {
        // Standalone rk8v4 append (int8 K + int4 V), then rk parallel attention.
        kv_cache_append_rk8v4_batch_launch(k, v, positions, valid, rows, cache, stream);
        const auto p = make_causal_operands(q, positions, out, scale, envelope.max_visible_keys);
        const auto view =
            make_quantized_causal_cache_view<Int8KvCacheView<false>>(cache, &valid, &rows);
        if (plan.family == Int8KvFamily::Tiled)
            throw std::invalid_argument(
                "rk8v4 attention: tiled family does not yet support PackedV.");
        // For rk8v4 parallel: PackedV=true, PackedK=false (RkVariant::Plain with
        // PackedK=false in rk_execute_parallel).
        // RkVariant::Plain maps to PackedK=false in rk_execute_parallel. But we need
        // a separate enum value... actually RkVariant::Plain means "PackedV only".
        // Let me use a different approach: just call with the right template directly.
        auto scope = workspace.scope();
        const auto partial = allocate_causal_partials(workspace, plan.query_heads, plan.width,
                                                       plan.partition.capacity, plan.batch);
        if (plan.query_heads == 24)
            rk_parallel_grouped<CausalD256H24Kv4, Int8KvCausalPlan::kTokenTile, false, false>(
                p, view, plan.partition, partial.view(), stream);
        else
            rk_parallel_grouped<CausalD256H16Kv2, Int8KvCausalPlan::kTokenTile, false, false>(
                p, view, plan.partition, partial.view(), stream);
    }
}

void rk8v4_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                               const PagedKVLayerView& cache,
                               CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                               Tensor& out, DeviceExecutionView execution) {
    rk4v4_kv_cached_attention(q, positions, scale, cache, envelope, workspace, out, execution);
}

// --- rk4v4-e8 attention (E8 lattice K + int4 V) -----------------------------------

void rk4v4e8_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& positions, const Tensor& valid, const Tensor& rows,
                                 float scale, PagedKVBatchLayerView cache,
                                 CausalAttentionExecutionEnvelope envelope,
                                 WorkspaceArena& workspace, Tensor& out,
                                 DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    auto plan           = make_int8_kv_causal_plan(q.ne[1], q.ne[2], q.ne[3], envelope,
                                                          execution.multiprocessor_count);
    if (plan.family == Int8KvFamily::Grouped) {
        rk_execute_grouped<CausalAppendInput, RkVariant::E8Lattice>(
            q, positions, scale, cache, &valid, &rows,
            CausalAppendInput{static_cast<const __nv_bfloat16*>(k.data),
                              static_cast<const __nv_bfloat16*>(v.data)},
            plan, workspace, out, stream);
    } else {
        // Standalone rk4v4-e8 append (int4 K with E8 lattice projection + int4 V),
        // then rk parallel attention — the batch codec must match the fused path.
        kv_cache_append_rk4v4e8_batch_launch(k, v, positions, valid, rows, cache, stream);
        const auto p = make_causal_operands(q, positions, out, scale, envelope.max_visible_keys);
        const auto view =
            make_quantized_causal_cache_view<Int8KvCacheView<false>>(cache, &valid, &rows);
        if (plan.family == Int8KvFamily::Tiled)
            throw std::invalid_argument("rk4v4-e8 attention: tiled does not support PackedV.");
        rk_execute_parallel<RkVariant::E8Lattice>(p, view, plan, workspace, stream);
    }
}

void rk4v4e8_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                                 const PagedKVLayerView& cache,
                                 CausalAttentionExecutionEnvelope envelope,
                                 WorkspaceArena& workspace, Tensor& out,
                                 DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    const auto plan =
        make_int8_kv_causal_plan(q.ne[1], q.ne[2], 1, envelope, execution.multiprocessor_count);
    const auto view = single_row_paged_kv_batch_view(cache);
    rk_execute_grouped<CausalCachedInput, RkVariant::E8Lattice>(q, positions, scale, view, nullptr, nullptr,
                                                CausalCachedInput{}, plan, workspace, out, stream);
}

// --- rk2v4-e8 attention (E8 cylinder 2-bit K + int4 V) -----------------------------

void rk2v4e8_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& positions, const Tensor& valid, const Tensor& rows,
                                 float scale, PagedKVBatchLayerView cache,
                                 CausalAttentionExecutionEnvelope envelope,
                                 WorkspaceArena& workspace, Tensor& out,
                                 DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    auto plan           = make_int8_kv_causal_plan(q.ne[1], q.ne[2], q.ne[3], envelope,
                                                           execution.multiprocessor_count);
    if (plan.family == Int8KvFamily::Grouped) {
        rk_execute_grouped<CausalAppendInput, RkVariant::E8Root>(
            q, positions, scale, cache, &valid, &rows,
            CausalAppendInput{static_cast<const __nv_bfloat16*>(k.data),
                              static_cast<const __nv_bfloat16*>(v.data)},
            plan, workspace, out, stream);
    } else {
        // rk2v4-e8 prefill (width > 8): standalone append with the SAME E8
        // cylinder codec the attention read-back decodes — an int4 RTN append
        // would put nibbles where the reader expects (root, rad_axis) pairs and
        // corrupt every cached key.
        kv_cache_append_rk2v4e8_batch_launch(k, v, positions, valid, rows, cache, stream);
        const auto p = make_causal_operands(q, positions, out, scale, envelope.max_visible_keys);
        const auto view =
            make_quantized_causal_cache_view<Int8KvCacheView<false>>(cache, &valid, &rows);
        if (plan.family == Int8KvFamily::Tiled)
            throw std::invalid_argument("rk2v4-e8 attention: tiled does not support PackedV.");
        rk_execute_parallel<RkVariant::E8Root>(p, view, plan, workspace, stream);
    }
}

void rk2v4e8_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                                 const PagedKVLayerView& cache,
                                 CausalAttentionExecutionEnvelope envelope,
                                 WorkspaceArena& workspace, Tensor& out,
                                 DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    const auto plan =
        make_int8_kv_causal_plan(q.ne[1], q.ne[2], 1, envelope, execution.multiprocessor_count);
    const auto view = single_row_paged_kv_batch_view(cache);
    rk_execute_grouped<CausalCachedInput, RkVariant::E8Root>(q, positions, scale, view, nullptr,
                                                              nullptr, CausalCachedInput{}, plan,
                                                              workspace, out, stream);
}

} // namespace ninfer::ops::detail
