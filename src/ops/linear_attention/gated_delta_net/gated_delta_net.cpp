#include "ninfer/ops/gated_delta_net.h"

#include "core/device.h"
#include "ops/linear_attention/gated_delta_net/chunked/launch.h"
#include "ops/linear_attention/gated_delta_net/common.h"
#include "ops/linear_attention/gated_delta_net/launch.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

struct Geometry {
    std::int32_t qk_heads;
    std::int32_t value_heads;
    std::int32_t tokens;
};

void require_dtype(const Tensor& t, DType dtype, const char* name) {
    if (t.dtype != dtype) { throw std::invalid_argument(std::string("gated_delta_net: ") + name); }
}

void require_shape(const Tensor& t, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   std::int32_t n3, const char* name) {
    if (t.ne[0] != n0 || t.ne[1] != n1 || t.ne[2] != n2 || t.ne[3] != n3) {
        throw std::invalid_argument(std::string("gated_delta_net: invalid shape for ") + name);
    }
}

void require_contiguous_nonnull(const Tensor& t, const char* name) {
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string("gated_delta_net: ") + name +
                                    " must be contiguous");
    }
    if (t.data == nullptr) {
        throw std::invalid_argument(std::string("gated_delta_net: ") + name +
                                    " data must be non-null");
    }
}

Geometry require_geometry(const Tensor& q, const Tensor& v) {
    const Geometry geometry{q.ne[1], v.ne[1], q.ne[2]};
    if (q.ne[0] != detail::gated_delta_net::kStateDim) {
        throw std::invalid_argument("gated_delta_net: state/head dimension must be 128");
    }
    if (!detail::gated_delta_net::are_head_counts_valid(geometry.qk_heads, geometry.value_heads)) {
        throw std::invalid_argument(
            "gated_delta_net: value heads must be at least q/k heads and divisible by them");
    }
    if (geometry.tokens <= 0) {
        throw std::invalid_argument("gated_delta_net: T must be positive");
    }
    return geometry;
}

void require_scale(float scale) {
    const float expected_scale =
        1.0f / std::sqrt(static_cast<float>(detail::gated_delta_net::kStateDim));
    if (!std::isfinite(scale) || scale <= 0.0f || std::abs(scale - expected_scale) > 1.0e-6f) {
        throw std::invalid_argument("gated_delta_net: scale must be 1/sqrt(128)");
    }
}

Geometry validate_recurrent(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                            const Tensor& beta, float scale, const Tensor& ssm_state,
                            const Tensor& out) {
    require_dtype(q, DType::BF16, "q must be BF16");
    require_dtype(k, DType::BF16, "k must be BF16");
    require_dtype(v, DType::BF16, "v must be BF16");
    require_dtype(out, DType::BF16, "out must be BF16");
    require_dtype(g, DType::FP32, "g must be FP32");
    require_dtype(beta, DType::FP32, "beta must be FP32");
    require_dtype(ssm_state, DType::FP32, "ssm_state must be FP32");

    const Geometry geometry = require_geometry(q, v);
    require_shape(q, detail::gated_delta_net::kStateDim, geometry.qk_heads, geometry.tokens, 1,
                  "q");
    require_shape(k, detail::gated_delta_net::kStateDim, geometry.qk_heads, geometry.tokens, 1,
                  "k");
    require_shape(v, detail::gated_delta_net::kStateDim, geometry.value_heads, geometry.tokens, 1,
                  "v");
    require_shape(out, detail::gated_delta_net::kStateDim, geometry.value_heads, geometry.tokens, 1,
                  "out");
    require_shape(g, geometry.value_heads, geometry.tokens, 1, 1, "g");
    require_shape(beta, geometry.value_heads, geometry.tokens, 1, 1, "beta");
    require_shape(ssm_state, detail::gated_delta_net::kStateDim, detail::gated_delta_net::kStateDim,
                  geometry.value_heads, 1, "ssm_state");

    require_contiguous_nonnull(q, "q");
    require_contiguous_nonnull(k, "k");
    require_contiguous_nonnull(v, "v");
    require_contiguous_nonnull(g, "g");
    require_contiguous_nonnull(beta, "beta");
    require_contiguous_nonnull(ssm_state, "ssm_state");
    require_contiguous_nonnull(out, "out");

    require_scale(scale);
    return geometry;
}

Geometry validate_recurrent_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                         const Tensor& g, const Tensor& beta, float scale,
                                         const Tensor& ssm_states, const Tensor& source_state_slots,
                                         const Tensor& destination_state_slots, const Tensor& out) {
    constexpr std::int32_t kMaximumBatch = 8;
    require_dtype(q, DType::BF16, "q must be BF16");
    require_dtype(k, DType::BF16, "k must be BF16");
    require_dtype(v, DType::BF16, "v must be BF16");
    require_dtype(out, DType::BF16, "out must be BF16");
    require_dtype(g, DType::FP32, "g must be FP32");
    require_dtype(beta, DType::FP32, "beta must be FP32");
    require_dtype(ssm_states, DType::FP32, "ssm_states must be FP32");
    require_dtype(source_state_slots, DType::I32, "source_state_slots must be I32");
    require_dtype(destination_state_slots, DType::I32, "destination_state_slots must be I32");

    const Geometry geometry  = require_geometry(q, v);
    const std::int32_t batch = q.ne[3];
    if (batch <= 0 || batch > kMaximumBatch || geometry.tokens != 1) {
        throw std::invalid_argument("gated_delta_net: batch update requires B=1..8 and W=1");
    }
    require_shape(q, detail::gated_delta_net::kStateDim, geometry.qk_heads, geometry.tokens, batch,
                  "q");
    require_shape(k, detail::gated_delta_net::kStateDim, geometry.qk_heads, geometry.tokens, batch,
                  "k");
    require_shape(v, detail::gated_delta_net::kStateDim, geometry.value_heads, geometry.tokens,
                  batch, "v");
    require_shape(out, detail::gated_delta_net::kStateDim, geometry.value_heads, geometry.tokens,
                  batch, "out");
    require_shape(g, geometry.value_heads, geometry.tokens, batch, 1, "g");
    require_shape(beta, geometry.value_heads, geometry.tokens, batch, 1, "beta");
    if (ssm_states.ne[0] != detail::gated_delta_net::kStateDim ||
        ssm_states.ne[1] != detail::gated_delta_net::kStateDim ||
        ssm_states.ne[2] != geometry.value_heads || ssm_states.ne[3] <= 0) {
        throw std::invalid_argument("gated_delta_net: invalid shape for pooled ssm_states");
    }
    require_shape(source_state_slots, batch, 1, 1, 1, "source_state_slots");
    require_shape(destination_state_slots, batch, 1, 1, 1, "destination_state_slots");

    require_contiguous_nonnull(q, "q");
    require_contiguous_nonnull(k, "k");
    require_contiguous_nonnull(v, "v");
    require_contiguous_nonnull(g, "g");
    require_contiguous_nonnull(beta, "beta");
    require_contiguous_nonnull(ssm_states, "ssm_states");
    require_contiguous_nonnull(source_state_slots, "source_state_slots");
    require_contiguous_nonnull(destination_state_slots, "destination_state_slots");
    require_contiguous_nonnull(out, "out");

    require_scale(scale);
    return geometry;
}

// The multi-token batch form: same operand contract as the one-token update, but the token axis
// carries W >= 1 columns per row and `valid_columns` may end a row early. A row whose live prefix
// reaches the chunked route's minimum runs the chunked kernel; a shorter row runs the sequential
// one - each row executes the same arithmetic the single-lane form runs at that token count.
Geometry validate_batch(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                        const Tensor& beta, float scale, const Tensor& ssm_states,
                        const Tensor& valid_columns, const Tensor& source_state_slots,
                        const Tensor& destination_state_slots, const Tensor& out) {
    constexpr std::int32_t kMaximumBatch = 8;
    require_dtype(q, DType::BF16, "q must be BF16");
    require_dtype(k, DType::BF16, "k must be BF16");
    require_dtype(v, DType::BF16, "v must be BF16");
    require_dtype(out, DType::BF16, "out must be BF16");
    require_dtype(g, DType::FP32, "g must be FP32");
    require_dtype(beta, DType::FP32, "beta must be FP32");
    require_dtype(ssm_states, DType::FP32, "ssm_states must be FP32");
    if (valid_columns.data != nullptr) {
        require_dtype(valid_columns, DType::I32, "valid_columns must be I32");
    }
    require_dtype(source_state_slots, DType::I32, "source_state_slots must be I32");
    require_dtype(destination_state_slots, DType::I32, "destination_state_slots must be I32");

    const Geometry geometry  = require_geometry(q, v);
    const std::int32_t batch = q.ne[3];
    const std::int32_t width = geometry.tokens;
    if (batch <= 0 || batch > kMaximumBatch || width <= 0) {
        throw std::invalid_argument("gated_delta_net: batch requires B=1..8 and a positive W");
    }
    require_shape(q, detail::gated_delta_net::kStateDim, geometry.qk_heads, width, batch, "q");
    require_shape(k, detail::gated_delta_net::kStateDim, geometry.qk_heads, width, batch, "k");
    require_shape(v, detail::gated_delta_net::kStateDim, geometry.value_heads, width, batch, "v");
    require_shape(out, detail::gated_delta_net::kStateDim, geometry.value_heads, width, batch,
                  "out");
    require_shape(g, geometry.value_heads, width, batch, 1, "g");
    require_shape(beta, geometry.value_heads, width, batch, 1, "beta");
    if (ssm_states.ne[0] != detail::gated_delta_net::kStateDim ||
        ssm_states.ne[1] != detail::gated_delta_net::kStateDim ||
        ssm_states.ne[2] != geometry.value_heads || ssm_states.ne[3] <= 0) {
        throw std::invalid_argument("gated_delta_net: invalid shape for pooled ssm_states");
    }
    require_shape(source_state_slots, batch, 1, 1, 1, "source_state_slots");
    require_shape(destination_state_slots, batch, 1, 1, 1, "destination_state_slots");
    if (valid_columns.data != nullptr) {
        require_shape(valid_columns, batch, 1, 1, 1, "valid_columns");
    }

    require_contiguous_nonnull(q, "q");
    require_contiguous_nonnull(k, "k");
    require_contiguous_nonnull(v, "v");
    require_contiguous_nonnull(g, "g");
    require_contiguous_nonnull(beta, "beta");
    require_contiguous_nonnull(ssm_states, "ssm_states");
    if (valid_columns.data != nullptr) { require_contiguous_nonnull(valid_columns, "valid_columns"); }
    require_contiguous_nonnull(source_state_slots, "source_state_slots");
    require_contiguous_nonnull(destination_state_slots, "destination_state_slots");
    require_contiguous_nonnull(out, "out");

    require_scale(scale);
    return geometry;
}

void validate_chunked(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                      const Tensor& beta, float scale, const Tensor& ssm_state_in,
                      const Tensor& ssm_state_out, const Tensor& out) {
    // ssm_state_out carries the running-state contract validated by validate_recurrent;
    // ssm_state_in is an equally-shaped read view (may alias ssm_state_out for in-place).
    const Geometry geometry = validate_recurrent(q, k, v, g, beta, scale, ssm_state_out, out);
    require_dtype(ssm_state_in, DType::FP32, "ssm_state_in must be FP32");
    require_shape(ssm_state_in, detail::gated_delta_net::kStateDim,
                  detail::gated_delta_net::kStateDim, geometry.value_heads, 1, "ssm_state_in");
    require_contiguous_nonnull(ssm_state_in, "ssm_state_in");
}

} // namespace

std::size_t gated_delta_net_workspace_capacity_bytes(std::int32_t qk_heads,
                                                     std::int32_t value_heads,
                                                     std::int32_t min_tokens,
                                                     std::int32_t max_tokens) {
    if (!detail::gated_delta_net::are_head_counts_valid(qk_heads, value_heads) || min_tokens <= 0 ||
        max_tokens < min_tokens) {
        throw std::invalid_argument("gated_delta_net workspace: invalid profile or interval");
    }
    namespace chunked = detail::gated_delta_net::chunked;
    return max_tokens < chunked::kMinTokens
               ? 0
               : chunked::workspace_layout(qk_heads, value_heads, max_tokens).total_bytes;
}

std::size_t gated_delta_net_batch_workspace_capacity_bytes(std::int32_t qk_heads,
                                                          std::int32_t value_heads,
                                                          std::int32_t width, std::int32_t batch) {
    if (!detail::gated_delta_net::are_head_counts_valid(qk_heads, value_heads) || width <= 0 ||
        batch <= 0) {
        throw std::invalid_argument("gated_delta_net batch workspace: invalid rectangle");
    }
    namespace chunked = detail::gated_delta_net::chunked;
    return width < chunked::kMinTokens
               ? 0
               : chunked::workspace_layout(qk_heads, value_heads, width, batch).total_bytes;
}

void gated_delta_net(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                     const Tensor& beta, float scale, bool normalize_qk, WorkspaceArena& ws,
                     Tensor& ssm_state, Tensor& out, DeviceExecutionView execution) {
    gated_delta_net(q, k, v, g, beta, scale, normalize_qk, ws, ssm_state, ssm_state, out,
                    execution);
}

void gated_delta_net_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& g, const Tensor& beta, float scale,
                                  bool normalize_qk, Tensor& ssm_states,
                                  const Tensor& source_state_slots,
                                  const Tensor& destination_state_slots, Tensor& out,
                                  cudaStream_t stream) {
    validate_recurrent_batch_update(q, k, v, g, beta, scale, ssm_states, source_state_slots,
                                    destination_state_slots, out);

    detail::gated_delta_net::launch_recurrent_batch_update(
        q, k, v, g, beta, scale, normalize_qk, ssm_states, source_state_slots,
        destination_state_slots, Tensor{}, out, stream);
}

void gated_delta_net_batch(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                          const Tensor& beta, float scale, bool normalize_qk, WorkspaceArena& ws,
                          Tensor& ssm_states, const Tensor& valid_columns,
                          const Tensor& source_state_slots, const Tensor& destination_state_slots,
                          Tensor& out, DeviceExecutionView execution) {
    const Geometry geometry = validate_batch(q, k, v, g, beta, scale, ssm_states, valid_columns,
                                             source_state_slots, destination_state_slots, out);
    if (execution.multiprocessor_count <= 0) {
        throw std::invalid_argument("gated_delta_net: invalid SM count");
    }
    const std::int32_t width = geometry.tokens;
    const std::int32_t batch = q.ne[3];
    namespace chunked        = detail::gated_delta_net::chunked;
    // The physical width declares the rectangle; each row's live prefix comes out of the mask (or
    // is the whole width when the call is dense). Both kernels launch over every row and each
    // skips the other's rows inside the kernel, so no host read-back of the mask is needed and a
    // row always runs the same kernel its single-lane counterpart would.
    const chunked::Arguments args{
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data),
        static_cast<const float*>(ssm_states.data),
        static_cast<float*>(ssm_states.data),
        static_cast<__nv_bfloat16*>(out.data),
        geometry.qk_heads,
        geometry.value_heads,
        width,
        width,
        batch,
        valid_columns.data == nullptr
            ? nullptr
            : static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<const std::int32_t*>(source_state_slots.data),
        static_cast<const std::int32_t*>(destination_state_slots.data),
        static_cast<std::int64_t>(detail::gated_delta_net::kStateDim) *
            detail::gated_delta_net::kStateDim * geometry.value_heads,
        scale,
    };
    // The chunked packets and scan only exist when the physical width can reach the chunked
    // minimum; below it the capacity query reports zero and the sequential kernel owns every row.
    if (width >= chunked::kMinTokens) {
        auto scope        = ws.scope();
        const auto layout = chunked::workspace_layout(geometry.qk_heads, geometry.value_heads,
                                                      width, batch);
        const auto backing = ws.alloc_bytes(layout.total_bytes);
        auto* qk           = static_cast<chunked::QkChunk*>(layout.qk.bind(backing).data);
        auto* control = static_cast<chunked::ControlChunk*>(layout.control.bind(backing).data);
        chunked::launch_prepare(args, qk, control, normalize_qk, execution.stream);
        chunked::launch_recurrence(args, qk, control, execution);
    }
    detail::gated_delta_net::launch_recurrent_batch_update(
        q, k, v, g, beta, scale, normalize_qk, ssm_states, source_state_slots,
        destination_state_slots, valid_columns, out, execution.stream);
}

void gated_delta_net(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                     const Tensor& beta, float scale, bool normalize_qk, WorkspaceArena& ws,
                     const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                     DeviceExecutionView execution) {
    validate_chunked(q, k, v, g, beta, scale, ssm_state_in, ssm_state_out, out);
    if (execution.multiprocessor_count <= 0) {
        throw std::invalid_argument("gated_delta_net: invalid SM count");
    }
    namespace chunked = detail::gated_delta_net::chunked;
    if (q.ne[2] < chunked::kMinTokens) {
        detail::gated_delta_net::launch_recurrent_inout(q, k, v, g, beta, scale, normalize_qk,
                                                        ssm_state_in, ssm_state_out, out,
                                                        execution.stream);
        return;
    }
    auto scope         = ws.scope();
    const auto layout  = chunked::workspace_layout(q.ne[1], v.ne[1], q.ne[2]);
    const auto backing = ws.alloc_bytes(layout.total_bytes);
    auto* qk           = static_cast<chunked::QkChunk*>(layout.qk.bind(backing).data);
    auto* control      = static_cast<chunked::ControlChunk*>(layout.control.bind(backing).data);
    // The single-lane form is the batch form's B=1 dense case: width == tokens, no mask, and the
    // state views are the caller's own tensors, so every row offset in resolve_row_arguments is
    // zero and the kernel sequence is byte-for-byte the previous single-lane launch.
    const chunked::Arguments args{static_cast<const __nv_bfloat16*>(q.data),
                                  static_cast<const __nv_bfloat16*>(k.data),
                                  static_cast<const __nv_bfloat16*>(v.data),
                                  static_cast<const float*>(g.data),
                                  static_cast<const float*>(beta.data),
                                  static_cast<const float*>(ssm_state_in.data),
                                  static_cast<float*>(ssm_state_out.data),
                                  static_cast<__nv_bfloat16*>(out.data),
                                  q.ne[1],
                                  v.ne[1],
                                  q.ne[2],
                                  q.ne[2],
                                  1,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  0,
                                  scale};
    chunked::launch_prepare(args, qk, control, normalize_qk, execution.stream);
    chunked::launch_recurrence(args, qk, control, execution);
}

} // namespace ninfer::ops
