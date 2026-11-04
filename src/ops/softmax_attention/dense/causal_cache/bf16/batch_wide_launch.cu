#include "ops/softmax_attention/dense/causal_cache/bf16/schedule.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/batch_wide_mma.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/batch_wide_launch.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/instances.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/template_launch.cuh"



namespace ninfer::ops::detail {
namespace {

constexpr int div_up(int a, int b) { return (a + b - 1) / b; }

template <class G, class S>
void launch_wide(const CausalAttentionOperands& p, Bf16KvCacheView<false> cache, const Tensor& table_rows,
                 cudaStream_t stream) {
    constexpr auto kernel = bf16_kv_batch_wide_mma_kernel<G, S>;
    const int bytes       = bf16_kv_dynamic_shared<bf16_kv_tiled_shared_bytes<G, S>, kernel>();
    const dim3 grid(div_up(p.width, S::kQueryRows), G::QHeads, p.batch);
    kernel<<<grid, S::kThreads, bytes, stream>>>(
        p.q, cache.keys, cache.values, cache.valid_columns,
        cache.table_rows,
        static_cast<const std::int32_t*>(cache.tables), cache.table_stride, p.positions, p.scale, p.out,
        p.width, p.batch, cache.page_scores, cache.page_scores_peak, cache.page_scores_recent,
        kv_page_score_row_span(), kv_page_score_tolerance(), kv_page_score_mass());
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void bf16_kv_batch_wide_attention(const Tensor& q, const Tensor& positions, const Tensor& valid_columns,
                                  const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
                                  CausalAttentionExecutionEnvelope envelope, Tensor& out,
                                  DeviceExecutionView execution) {
    const auto p = make_causal_operands(q, positions, out, scale, envelope.max_visible_keys);
    const auto view = bf16_kv_cache_view<false>(cache, &valid_columns, &table_rows);
    if (q.ne[1] == 24)
        launch_wide<CausalD256H24Kv4, typename Bf16KvInstanceTraits<Bf16KvInstance::Tiled64>::Schedule>(
            p, view, table_rows, execution.stream);
    else
        launch_wide<CausalD256H16Kv2, typename Bf16KvInstanceTraits<Bf16KvInstance::Tiled64>::Schedule>(
            p, view, table_rows, execution.stream);
}

} // namespace ninfer::ops::detail


