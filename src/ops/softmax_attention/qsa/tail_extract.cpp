#include "ninfer/ops/qsa.h"
#include "ops/softmax_attention/qsa/launch.h"
#include "ops/softmax_attention/qsa/validation.h"

namespace ninfer::ops {

void qsa_tail_extract(const QsaTailStateView& before, const QsaRawRecordReadView& records,
                      const Tensor& accepted, QsaTailStateView after,
                      DeviceExecutionView execution) {
    const detail::qsa::ViewValidation v{"qsa_tail_extract"};
    const int batch = accepted.ne[0], capacity = records.raw_keys.ne[1];
    v.require(batch >= 1 && batch <= 8, "batch extent");
    v.require(execution.multiprocessor_count > 0, "SM count");
    v.metadata(accepted, batch, "accepted columns");
    v.metadata(records.base_positions, batch, "record base positions");
    v.metadata(records.valid_columns, batch, "record valid columns");
    v.state(before, batch);
    v.state(after, batch);
    v.columns(records.raw_keys, DType::BF16, 128, capacity, batch, "record raw keys");
    v.columns(records.rope_positions, DType::I32, 3, capacity, batch, "record coordinates");
    // Only Device-selected slots are accessed. Overlapping pool bounds do not imply that
    // selected regions overlap; the caller guarantees their cross-row/plane relationships.
    detail::qsa::launch_tail_extract(before, records, accepted, after, execution);
}

} // namespace ninfer::ops
