#pragma once

#include "ninfer/ops/softmax_attention.h"

namespace ninfer::ops::detail {

void int8_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& positions, const Tensor& valid, const Tensor& rows,
                              float scale, PagedKVBatchLayerView cache,
                              CausalAttentionExecutionEnvelope envelope, MemoryArena& workspace,
                              Tensor& out, CudaExecutionView execution);

void int8_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                              const PagedKVLayerView& cache,
                              CausalAttentionExecutionEnvelope envelope, MemoryArena& workspace,
                              Tensor& out, CudaExecutionView execution);


// port (sm_89): rk8v4 = int8 K plane + packed int4 V plane (see the definition for the plan rule).
void rk8v4_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                               const Tensor& positions, const Tensor& valid, const Tensor& rows,
                               float scale, PagedKVBatchLayerView cache,
                               CausalAttentionExecutionEnvelope envelope, MemoryArena& workspace,
                               Tensor& out, CudaExecutionView execution);

void rk8v4_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                               const PagedKVLayerView& cache,
                               CausalAttentionExecutionEnvelope envelope, MemoryArena& workspace,
                               Tensor& out, CudaExecutionView execution);


// port (sm_89): rk4v4 = packed int4 K plane + packed int4 V plane.
void rk4v4_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                               const Tensor& positions, const Tensor& valid, const Tensor& rows,
                               float scale, PagedKVBatchLayerView cache,
                               CausalAttentionExecutionEnvelope envelope, MemoryArena& workspace,
                               Tensor& out, CudaExecutionView execution);

void rk4v4_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                               const PagedKVLayerView& cache,
                               CausalAttentionExecutionEnvelope envelope, MemoryArena& workspace,
                               Tensor& out, CudaExecutionView execution);


// port (sm_89): rk4v4-e8 = E8-lattice key carrier (96 B plane) + packed int4 value plane.
void rk4v4_e8_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& positions, const Tensor& valid,
                                  const Tensor& rows, float scale, PagedKVBatchLayerView cache,
                                  CausalAttentionExecutionEnvelope envelope, MemoryArena& workspace,
                                  Tensor& out, CudaExecutionView execution);

void rk4v4_e8_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                                  const PagedKVLayerView& cache,
                                  CausalAttentionExecutionEnvelope envelope,
                                  MemoryArena& workspace, Tensor& out, CudaExecutionView execution);


// port (sm_89): rk2v4-e8 = E8-cylinder key plane (64 B) + packed int4 value plane.
void rk2v4_e8_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& positions, const Tensor& valid,
                                  const Tensor& rows, float scale, PagedKVBatchLayerView cache,
                                  CausalAttentionExecutionEnvelope envelope, MemoryArena& workspace,
                                  Tensor& out, CudaExecutionView execution);

void rk2v4_e8_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                                  const PagedKVLayerView& cache,
                                  CausalAttentionExecutionEnvelope envelope,
                                  MemoryArena& workspace, Tensor& out, CudaExecutionView execution);

// port (sm_89): rk4v2-e8 = E8-lattice key plane (96 B) + E8-cylinder value plane (64 B).
void rk4v2_e8_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& positions, const Tensor& valid,
                                  const Tensor& rows, float scale, PagedKVBatchLayerView cache,
                                  CausalAttentionExecutionEnvelope envelope, MemoryArena& workspace,
                                  Tensor& out, CudaExecutionView execution);

void rk4v2_e8_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                                  const PagedKVLayerView& cache,
                                  CausalAttentionExecutionEnvelope envelope,
                                  MemoryArena& workspace, Tensor& out, CudaExecutionView execution);

} // namespace ninfer::ops::detail
