// Non-RDC compilation is required for the producer/consumer register redistribution.
#include "ops/softmax_attention/dense/causal_cache/nvfp4/tiled_launch.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/instances.h"
#include "ops/softmax_attention/kernels/causal/nvfp4/tiled_launch.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
#if NINFER_CUDA_HAS_TMA && NINFER_CUDA_HAS_BLOCK_SCALE_MMA
void nvfp4_kv_tiled_attention(const CausalAttentionOperands& p, Nvfp4KvReadView cache,
                              cudaStream_t stream) {
    if (p.query_heads == 24)
        launch_nvfp4_kv_tiled_mma<CausalD256H24Kv4, Nvfp4KvTiledInstance>(p, cache, stream);
    else
        launch_nvfp4_kv_tiled_mma<CausalD256H16Kv2, Nvfp4KvTiledInstance>(p, cache, stream);
}
#else
// port (sm_89): the NVFP4 KV cache tiled attention needs warp-specialised TMA and a block-scaled MMA target. This route is only selected when the caller asks for that
// storage format or expert format, so an Ada build fails loudly here rather than silently
// computing something else. See nvfp4_a4.cu for the same treatment on the linear side.

void nvfp4_kv_tiled_attention(const CausalAttentionOperands& p, Nvfp4KvReadView cache,
                              cudaStream_t stream) {
    throw std::logic_error("the NVFP4 KV cache tiled attention needs warp-specialised TMA and a block-scaled MMA target");
}
#endif
} // namespace ninfer::ops::detail
