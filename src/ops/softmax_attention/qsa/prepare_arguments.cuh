#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace ninfer::ops { struct E8TraceRecord; }

namespace ninfer::ops::detail::qsa {

template <class T>
struct PrepareColumns {
    T* data;
    std::int64_t column, outer;

    __device__ T* at(int feature, std::int64_t c, int row = 0) const {
        return data + feature + c * column + std::int64_t(row) * outer;
    }
};

template <class T>
struct PreparePage {
    T* data;
    std::int64_t row, head, page;

    __device__ T* at(int p, int h, int slot) const {
        return data + std::int64_t(p) * row + std::int64_t(h) * head + std::int64_t(slot) * page;
    }
};

// Defined by the E8 codec header (namespace ninfer::ops).
struct PrepareArguments {
    const __nv_bfloat16 *q, *k, *v, *iq, *ik, *wq, *wk, *wiq, *wik;
    __nv_bfloat16 *out_q, *out_iq;
    const std::int32_t *positions, *rope, *valid, *table_rows, *slots, *offsets, *tables;
    std::int64_t table_stride;
    PrepareColumns<const __nv_bfloat16> tail;
    PrepareColumns<const std::int32_t> first_rope;
    PrepareColumns<__nv_bfloat16> raw_records;
    PrepareColumns<std::int32_t> rope_records;
    PreparePage<__nv_bfloat16> index_keys;
    PreparePage<std::uint8_t> keys, values, key_scales, value_scales;
    int tokens, batch, axes, kv_heads;
    float epsilon;
    double frequency[32];
    // Default-off diagnostic (NINFER_QSA_E8_DUMP): print one line per E8 subspace with the values
    // its residual-axis decision was made from, for the row `e8_dump_row` (-1 = the executing row).
    int e8_dump     = 0;
    int e8_dump_row = -1;
    int e8_dump_plane = -1;
    int e8_dump_head  = -1;
    // Default-off input-side export (NINFER_QSA_V_SRC_TRACE): what prepare_kv actually read for one
    // (t, head) - the row index `m` it derived, the position `p` it wrote to, and the 256 projection
    // values it encoded.
    float* v_trace       = nullptr;
    int* v_trace_count   = nullptr;
    int v_trace_capacity = 0;
    int v_trace_t        = -1;
    int v_trace_head     = -1;
    // Default-off entry log (NINFER_QSA_PREP_ENTRY): every (t, head) this kernel reaches, unfiltered,
    // capped by NINFER_QSA_PREP_ENTRY_N. Answers whether a pass reaches prepare_kv at all.
    int* prep_entry_count = nullptr;
    int prep_entry_n      = 0;
    ninfer::ops::E8TraceRecord* trace_out = nullptr;
    int* trace_count         = nullptr;
    int trace_capacity       = 0;
};
} // namespace ninfer::ops::detail::qsa
