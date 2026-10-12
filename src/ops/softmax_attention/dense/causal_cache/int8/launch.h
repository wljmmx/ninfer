#pragma once

#include "ninfer/ops/softmax_attention.h"

namespace ninfer::ops::detail {

void int8_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& positions, const Tensor& valid, const Tensor& rows,
                              float scale, PagedKVBatchLayerView cache,
                              CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                              Tensor& out, DeviceExecutionView execution, Tensor* page_score = nullptr);

void int8_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                              const PagedKVLayerView& cache,
                              CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                              Tensor& out, DeviceExecutionView execution, Tensor* page_score = nullptr);

// rk4v4: rank-compressed 4-bit K + 4-bit V attention (same int8 kernel with PackedV=true).
void rk4v4_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                               const Tensor& positions, const Tensor& valid, const Tensor& rows,
                               float scale, PagedKVBatchLayerView cache,
                               CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                               Tensor& out, DeviceExecutionView execution, Tensor* page_score = nullptr);

void rk4v4_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                               const PagedKVLayerView& cache,
                               CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                               Tensor& out, DeviceExecutionView execution, Tensor* page_score = nullptr);

// rk8v4: 8-bit K (plain int8, no packing) + 4-bit V. Reuses the rk4v4 dispatch
// (PackedV=true, PackedK=false) since only the V plane differs from int8.
void rk8v4_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                               const Tensor& positions, const Tensor& valid, const Tensor& rows,
                               float scale, PagedKVBatchLayerView cache,
                               CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                               Tensor& out, DeviceExecutionView execution, Tensor* page_score = nullptr);

void rk8v4_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                               const PagedKVLayerView& cache,
                               CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                               Tensor& out, DeviceExecutionView execution, Tensor* page_score = nullptr);

// rk4v4-e8: E8 lattice key quantization + 4-bit V (PackedV=true, PackedK=true, E8Lattice=true).
void rk4v4e8_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& positions, const Tensor& valid, const Tensor& rows,
                                 float scale, PagedKVBatchLayerView cache,
                                 CausalAttentionExecutionEnvelope envelope,
                                 WorkspaceArena& workspace, Tensor& out,
                                 DeviceExecutionView execution, Tensor* page_score = nullptr);

void rk4v4e8_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                                 const PagedKVLayerView& cache,
                                 CausalAttentionExecutionEnvelope envelope,
                                 WorkspaceArena& workspace, Tensor& out,
                                 DeviceExecutionView execution, Tensor* page_score = nullptr);

// rk2v4-e8: E8 cylinder 2-bit K + 4-bit V (PackedV=true, E8Root=true).
void rk2v4e8_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& positions, const Tensor& valid, const Tensor& rows,
                                 float scale, PagedKVBatchLayerView cache,
                                 CausalAttentionExecutionEnvelope envelope,
                                 WorkspaceArena& workspace, Tensor& out,
                                 DeviceExecutionView execution, Tensor* page_score = nullptr);

void rk2v4e8_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                                 const PagedKVLayerView& cache,
                                 CausalAttentionExecutionEnvelope envelope,
                                 WorkspaceArena& workspace, Tensor& out,
                                 DeviceExecutionView execution, Tensor* page_score = nullptr);

// rk4v2-e8: 4-bit E8-lattice K + 2-bit E8-cylinder V
// (PackedK=true, E8Lattice=true, V8Root=true).
void rk4v2e8_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& positions, const Tensor& valid, const Tensor& rows,
                                 float scale, PagedKVBatchLayerView cache,
                                 CausalAttentionExecutionEnvelope envelope,
                                 WorkspaceArena& workspace, Tensor& out,
                                 DeviceExecutionView execution, Tensor* page_score = nullptr);

void rk4v2e8_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                                 const PagedKVLayerView& cache,
                                 CausalAttentionExecutionEnvelope envelope,
                                 WorkspaceArena& workspace, Tensor& out,
                                 DeviceExecutionView execution, Tensor* page_score = nullptr);

} // namespace ninfer::ops::detail
