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
                 Tensor* page_score, cudaStream_t stream) {
    constexpr auto kernel = bf16_kv_batch_wide_mma_kernel<G, S>;
    const int bytes       = bf16_kv_dynamic_shared<bf16_kv_tiled_shared_bytes<G, S>, kernel>();
    const dim3 grid(div_up(p.width, S::kQueryRows), G::QHeads, p.batch);
    unsigned* const override_scores =
        page_score != nullptr ? static_cast<unsigned*>(page_score->data) : nullptr;
    kernel<<<grid, S::kThreads, bytes, stream>>>(
        p.q, cache.keys, cache.values, cache.valid_columns,
        cache.table_rows,
        static_cast<const std::int32_t*>(cache.tables), cache.table_stride, p.positions, p.scale, p.out,
        p.width, p.batch, override_scores, cache.page_scores_peak, cache.page_scores_recent,
        kv_page_score_row_span(), kv_page_score_tolerance(), kv_page_score_mass());
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void bf16_kv_batch_wide_attention(const Tensor& q, const Tensor& positions, const Tensor& valid_columns,
                                  const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
                                  CausalAttentionExecutionEnvelope envelope, Tensor& out,
                                  DeviceExecutionView execution, Tensor* page_score) {
    const auto p = make_causal_operands(q, positions, out, scale, envelope.max_visible_keys);
    const auto view = bf16_kv_cache_view<false>(cache, &valid_columns, &table_rows);
    // One CTA per (query tile, query head) stages its own KV head's tile, so the template is keyed by
    // the (query heads, KV heads) pair: 24 heads run with 4 KV heads in Qwen3.5 and with 2 in
    // Qwen4Exp's indexed layers. Tiled64 (Br=64, Bc=64) is the shape this route always launches and
    // it fits 96 KiB of smem on sm_89 for either pairing - the tile sizes do not depend on the group
    // size, only the number of CTAs that re-read the same KV head does.
    if (q.ne[1] == 24 && view.kv_heads == 2)
        launch_wide<CausalD256H24Kv2, typename Bf16KvInstanceTraits<Bf16KvInstance::Tiled64>::Schedule>(
            p, view, table_rows, page_score, execution.stream);
    else if (q.ne[1] == 24)
        launch_wide<CausalD256H24Kv4, typename Bf16KvInstanceTraits<Bf16KvInstance::Tiled64>::Schedule>(
            p, view, table_rows, page_score, execution.stream);
    else
        launch_wide<CausalD256H16Kv2, typename Bf16KvInstanceTraits<Bf16KvInstance::Tiled64>::Schedule>(
            p, view, table_rows, page_score, execution.stream);
}

} // namespace ninfer::ops::detail


