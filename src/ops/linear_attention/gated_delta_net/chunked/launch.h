#pragma once

#include "core/device.h"
#include "core/layout.h"
#include "ops/common/math.h"
#include "ops/linear_attention/gated_delta_net/common.h"

#include <cuda_bf16.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail::gated_delta_net::chunked {

inline constexpr int kChunkSize = 16;
inline constexpr int kMinTokens = 16;
inline constexpr float kNormEps = 1.0e-6F;

struct Arguments {
    const __nv_bfloat16 *q, *k, *v;
    const float *g, *beta, *state_in;
    float* state_out;
    __nv_bfloat16* out;
    int qk_heads, value_heads, tokens;
    // Batch form: the tensors above address a [.., W, B] rectangle batch-major (row b's token t
    // at flat column b*W + t), `tokens` is every row's dense width, and `valid_columns` gives the
    // live prefix of each row. The single-lane callers pass width == tokens, batch == 1, a null
    // mask and zero state stride, for which every row offset below is zero - that is what keeps a
    // batched launch bit-identical, per row, to the single-lane launch of the same data.
    int width;
    int batch;
    const std::int32_t* valid_columns;
    const std::int32_t* source_state_slots;
    const std::int32_t* destination_state_slots;
    std::int64_t state_slot_stride;
    float scale;
};

// Q/K are independent of value-head gates. One packet is shared by the entire Q/K group.
struct alignas(256) QkChunk {
    __nv_bfloat16 q[kChunkSize * kStateDim];
    __nv_bfloat16 k[kChunkSize * kStateDim];
};

struct alignas(256) ControlChunk {
    // With G_i = sum_{t<=i} g_t and L_ij = beta_i exp(G_i-G_j) dot(K_i,K_j), j<i:
    // solve = (I+L)^-1 diag(beta), mqk_ij = exp(G_i-G_j) dot(Q_i,K_j), j<=i.
    float solve[kChunkSize * kChunkSize];
    float mqk[kChunkSize * kChunkSize];
    float prefix[kChunkSize]; // exp(G_i)
    float suffix[kChunkSize]; // exp(G_last-G_i)
    std::uint32_t padding[32];
};

static_assert(kChunkSize == 16);
static_assert(sizeof(QkChunk) == 8192);
static_assert(sizeof(ControlChunk) == 2304);

__host__ __device__ inline int chunk_count(int tokens) { return div_up(tokens, kChunkSize); }

__host__ __device__ constexpr int vector_index(int row, int col) {
    return row * kStateDim + (col ^ (row * 8));
}

// Resolve one row's view of the batched arguments: rebased input/output pointers into row `row`'s
// slice of the [.., width, batch] rectangle, that row's live token count, and (when the call uses
// a state pool) the row's source/destination state slots. The single-lane callers pass batch=1
// with dense width and zero state stride, for which every offset is zero and the resolved view IS
// the call's arguments - that identity is what makes the batched launch bit-identical, per row,
// to a single-lane launch of the same data.
__host__ __device__ inline Arguments resolve_row_arguments(const Arguments& args, int row) {
    Arguments local      = args;
    const std::int64_t column_base = static_cast<std::int64_t>(row) * args.width;
    local.q    = args.q + column_base * args.qk_heads * kStateDim;
    local.k    = args.k + column_base * args.qk_heads * kStateDim;
    local.v    = args.v + column_base * args.value_heads * kStateDim;
    local.g    = args.g + column_base * args.value_heads;
    local.beta = args.beta + column_base * args.value_heads;
    local.out  = args.out + column_base * args.value_heads * kStateDim;
    if (args.state_slot_stride != 0) {
        local.state_in = args.state_in +
                         static_cast<std::int64_t>(args.source_state_slots[row]) *
                             args.state_slot_stride;
        local.state_out = args.state_out +
                          static_cast<std::int64_t>(args.destination_state_slots[row]) *
                              args.state_slot_stride;
    }
    local.tokens = args.valid_columns != nullptr ? args.valid_columns[row] : args.tokens;
    return local;
}

__host__ __device__ constexpr int square_index(int row, int col) {
    return row * kChunkSize + (col ^ (((row >> 1) & 3) * 4));
}

struct WorkspaceLayout {
    TensorRegion qk;
    TensorRegion control;
    std::size_t total_bytes;
};

inline WorkspaceLayout workspace_layout(int qk_heads, int value_heads, int tokens, int batch = 1) {
    if (batch <= 0) {
        throw std::invalid_argument("GDN chunk workspace: batch must be positive");
    }
    LayoutBuilder builder;
    const int chunks = chunk_count(tokens);
    WorkspaceLayout layout;
    layout.qk      = builder.add_tensor(DType::U8, {int(sizeof(QkChunk)), chunks, qk_heads, batch},
                                        256, "gdn.qk");
    layout.control = builder.add_tensor(
        DType::U8, {int(sizeof(ControlChunk)), chunks, value_heads, batch}, 256, "gdn.control");
    layout.total_bytes = builder.finish(256, "GDN chunk workspace");
    return layout;
}

void launch_prepare(const Arguments& args, QkChunk* qk, ControlChunk* control, bool normalize_qk,
                    cudaStream_t stream);
void launch_recurrence(const Arguments& args, const QkChunk* qk, const ControlChunk* control,
                       DeviceExecutionView execution);
int value_tile(int value_heads, int multiprocessor_count);

} // namespace ninfer::ops::detail::gated_delta_net::chunked
