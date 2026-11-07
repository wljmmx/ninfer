#pragma once

#include "ninfer/ops/qsa.h"

namespace ninfer::ops::detail::qsa {

void launch_query_gate(const Tensor& x, const Weight& weight, Tensor& query, Tensor& gate,
                       cudaStream_t stream);
void launch_kv_index(const Tensor& x, const Weight& weight, Tensor& key, Tensor& value,
                     Tensor& index_query, Tensor& index_key, cudaStream_t stream);
void launch_joint(const Tensor& x, const Weight& weight, Tensor& query, Tensor& key, Tensor& gate,
                  Tensor& value, Tensor& index_query, Tensor& index_key, cudaStream_t stream);

void launch_prepare(const QsaProjectionView& projected, const QsaPrepareParameters& parameters,
                    const QsaBatchView& batch, const Tensor& rope_positions,
                    const QsaTailStateView& before, QsaCacheView cache,
                    QsaRawRecordWriteView records, Tensor& query, Tensor& index_query,
                    DeviceExecutionView execution);

void launch_tail_extract(const QsaTailStateView& before, const QsaRawRecordReadView& records,
                         const Tensor& accepted, QsaTailStateView after,
                         DeviceExecutionView execution);

} // namespace ninfer::ops::detail::qsa
