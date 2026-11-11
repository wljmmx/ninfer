// REPRODUCER, NOT YET IN THE SUITE. The QSA read side does not agree between a merged batch and one
// row per launch: with byte-identical per-row inputs, batch 8 and batch 1 differ from token 32 on -
// the first token of the second 32-query tile - for every row including row 0, and with equal lengths
// as well as mixed ones. Batch 1 against itself is exact, so this is not a race. The row's own table
// row, positions, live length and block selection are reproduced identically in both arms, so the
// divergence is in how the batched pass binds a tile that starts at token 32. This file stays out of
// tests.cmake until the Op is fixed; the binary exits non-zero today.
//
// The QSA read side under a merged batch: the same rows must produce the same bytes whether they
// are addressed as one (width x batch) rectangle or one row at a time. The per-row bindings - the
// row's own table row, positions, live length and selected blocks - are what keeps the rows
// block-diagonal without a mask, so this compares the Op against itself with only that geometry
// changed: every host array holds the same values in both arms, and only `batch` differs.
//
// Coverage: B=2 and B=8, mixed lengths including a one-token row, equal lengths, a row that crosses
// one 64-token index page, and a row past 2048 tokens (more than 512 index blocks).
#include "core/arena.h"
#include "core/device.h"

#include "ninfer/ops/qsa.h"
#include "ops/softmax_attention/qsa/storage.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_fp16.h>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kHeadDim = 256;

float sample(std::uint32_t seed) {
    seed ^= seed >> 16;
    seed *= 0x7feb352dU;
    seed ^= seed >> 15;
    seed *= 0x846ca68bU;
    seed ^= seed >> 16;
    return float(seed % 2000) / 1000.0F - 1.0F;
}

std::uint16_t f32_to_bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, 4);
    bits = (bits + 0x7FFFU + ((bits >> 16) & 1U)) >> 16;
    return std::uint16_t(bits);
}

float bf16_to_f32(std::uint16_t word) {
    const std::uint32_t bits = std::uint32_t(word) << 16;
    float value              = 0;
    std::memcpy(&value, &bits, 4);
    return value;
}

int max_visible_width(const std::vector<int>& lengths) {
    return lengths.empty() ? 0 : *std::max_element(lengths.begin(), lengths.end());
}

// Ordered integer key of a BF16 word, so the distance between two words counts representable steps.
int ordered_key(std::uint16_t word) {
    const int sign = (word >> 15) & 1;
    const int mag  = word & 0x7FFF;
    return sign ? -mag : mag;
}

struct Case {
    std::vector<int> lengths;   // one entry per row, the row's live columns
    int query_heads = 24;
};

// One launch's worth of host state, filled so that a row's data depends only on (row, t) - never on
// how many rows the pass happens to carry.
struct Rows {
    int tokens = 0;
    int batch  = 0;
};;

struct Harness {
    int T = 0, B = 0, q_heads = 0, kv_heads = 0, table_width = 0, table_stride = 0, pages = 0;
    int table_rows_total = 0;
    std::vector<int> lengths, positions, tables, ids, counts, rows;
    std::vector<std::uint16_t> qhost, ghost;
    std::vector<std::uint16_t> keys_host, values_host;
    DeviceBuffer q, gate, out, pos, len, row_dev, table_dev, ids_dev, counts_dev, keys, values;
    Tensor query, output_gate, output;
    PagedKVBatchLayerView cache;
    ops::QsaBatchView batch;
    ops::QsaBlockSelection selected;
    ops::QsaSparseGqaExecutionEnvelope envelope;

    // `row_ids[b]` names the logical row whose data row b carries, so a one-row arm reproduces the
    // merged arm's row exactly.
    void build(const std::vector<int>& lengths_in, const std::vector<int>& row_ids, int query_heads,
               int max_visible, int row_slots, int forced_width = 0) {
        B       = int(lengths_in.size());
        q_heads = query_heads;
        kv_heads = query_heads / 12;
        T       = forced_width > 0 ? forced_width
                                   : *std::max_element(lengths_in.begin(), lengths_in.end());
        table_width  = (max_visible + 63) / 64;
        table_stride = table_width;
        // A row's physical pages are addressed by its logical row id, so a one-row arm reads exactly
        // the pages the merged arm read for that row.
        const int pages_per_row = (max_visible_width(lengths_in) + 63) / 64;
        pages                   = 1 + std::max(1, row_slots) * pages_per_row + 1;
        std::vector<int> base_page(B, 0);
        for (int b = 0; b < B; ++b) base_page[b] = 1 + row_ids[b] * pages_per_row;

        lengths.resize(B);
        rows.assign(B, 0);
        // The table carries one row per logical row plus the unused row 0, so a one-row arm still
        // addresses the same table row the merged arm did for that row.
        const int table_rows = std::max(1, row_slots) + 1;
        table_rows_total     = table_rows;
        for (int b = 0; b < B; ++b) {
            lengths[b] = lengths_in[b];
            rows[b]    = row_ids[b] + 1;
        }
        positions.assign(std::size_t(T) * B, -1);
        counts.assign(std::size_t(T) * B, 0);
        ids.assign(std::size_t(T) * B * 512, -1);
        tables.assign(std::size_t(table_stride) * table_rows, -1);
        qhost.assign(std::size_t(T) * B * q_heads * kHeadDim, 0);
        ghost.assign(qhost.size(), 0);

        for (int b = 0; b < B; ++b) {
            const int n = lengths[b];
            const int row = row_ids[b];
            for (int g = 0; g < table_width; ++g) {
                tables[std::size_t(rows[b]) * table_stride + g] = base_page[b] + g;
            }
            for (int t = 0; t < n; ++t) {
                const int m = t + T * b;
                positions[m] = t;
                const int c  = std::min((t + 1) / 4, 512);
                counts[m]    = c;
                for (int i = 0; i < c; ++i) ids[std::size_t(m) * 512 + i] = i;
                for (int h = 0; h < q_heads; ++h)
                    for (int d = 0; d < kHeadDim; ++d) {
                        const std::size_t i = (std::size_t(m) * q_heads + h) * kHeadDim + d;
                        qhost[i] = f32_to_bf16(sample(std::uint32_t(row * 977 + t * 31 + h * 7 + d)));
                        ghost[i] = f32_to_bf16(sample(std::uint32_t(row * 613 + t * 53 + h * 11 + d)) *
                                               0.5F);
                    }
            }
        }

        keys_host.assign(std::size_t(pages) * kv_heads * 64 * kHeadDim, 0);
        values_host.assign(keys_host.size(), 0);
        for (int p = 0; p < pages; ++p)
            for (int h = 0; h < kv_heads; ++h)
                for (int slot = 0; slot < 64; ++slot)
                    for (int d = 0; d < kHeadDim; ++d) {
                        const std::size_t i =
                            ((std::size_t(p) * kv_heads + h) * 64 + slot) * kHeadDim + d;
                        keys_host[i]   = f32_to_bf16(sample(std::uint32_t(i + 17)));
                        values_host[i] =
                            __half_as_ushort(__float2half_rn(sample(std::uint32_t(i + 91)) * 0.75F));
                    }
    }

    void upload(DeviceExecutionView execution) {
        const auto key_bytes   = keys_host.size() * 2;
        const auto value_bytes = values_host.size() * 2;
        keys   = DeviceBuffer(key_bytes);
        values = DeviceBuffer(value_bytes);
        q      = DeviceBuffer(qhost.size() * 2);
        gate   = DeviceBuffer(ghost.size() * 2);
        out    = DeviceBuffer(qhost.size() * 2);
        pos    = DeviceBuffer(positions.size() * 4);
        len    = DeviceBuffer(lengths.size() * 4);
        row_dev = DeviceBuffer(rows.size() * 4);
        table_dev = DeviceBuffer(tables.size() * 4);
        ids_dev   = DeviceBuffer(ids.size() * 4);
        counts_dev = DeviceBuffer(counts.size() * 4);
        const auto to_device = [&](DeviceBuffer& buffer, const auto& host) {
            CUDA_CHECK(cudaMemcpyAsync(buffer.p, host.data(), host.size() * sizeof(host[0]),
                                       cudaMemcpyHostToDevice, execution.stream));
        };
        to_device(keys, keys_host);
        to_device(values, values_host);
        to_device(q, qhost);
        to_device(gate, ghost);
        to_device(pos, positions);
        to_device(len, lengths);
        to_device(row_dev, rows);
        to_device(table_dev, tables);
        to_device(ids_dev, ids);
        to_device(counts_dev, counts);
        CUDA_CHECK(cudaMemsetAsync(out.p, 0xa5, qhost.size() * 2, execution.stream));

        query       = Tensor(q.p, DType::BF16, {kHeadDim, q_heads, T, B});
        output_gate = Tensor(gate.p, DType::BF16, {kHeadDim, q_heads, T, B});
        output      = Tensor(out.p, DType::BF16, {kHeadDim, q_heads, T, B});
        Tensor k_view(keys.p, DType::BF16, {kHeadDim, 64, kv_heads, pages});
        Tensor v_view(values.p, DType::FP16, {kHeadDim, 64, kv_heads, pages});
        Tensor bt(table_dev.p, DType::I32, {table_width, table_rows_total});
        cache              = PagedKVBatchLayerView{k_view, v_view, {}, {}, bt};
        cache.head_dim     = kHeadDim;
        cache.num_kv_heads = kv_heads;
        cache.storage      = KvCacheStorage::BFloat16;
        batch   = {Tensor(pos.p, DType::I32, {T, B}), Tensor(len.p, DType::I32, {B}),
                   Tensor(row_dev.p, DType::I32, {B})};
        selected = {Tensor(ids_dev.p, DType::I32, {512, T, B}),
                    Tensor(counts_dev.p, DType::I32, {T, B})};
    }

    void launch(WorkspaceArena& arena, DeviceExecutionView execution) {
        ops::qsa_sparse_gqa(query, output_gate, cache, batch, selected, envelope, arena, output,
                            execution);
    }
};

struct ArmOut {
    std::vector<std::uint16_t> words;
    int tokens = 0;   // the arm's own rectangle width, which is the batch stride of its layout
};

// Returns the arm's raw output words in the Op's own layout, `[dim, head, token, batch]`.
ArmOut arm(DeviceContext& device, const Case& c,
           const std::vector<int>& row_ids, int batch, bool merged, int forced_width = 0) {
    Harness h;
    std::vector<int> lengths;
    if (merged) {
        lengths = c.lengths;
    } else {
        lengths.assign(1, c.lengths[row_ids[0]]);
    }
    h.envelope.max_visible_tokens = 8192;
    h.build(lengths, row_ids, c.query_heads, 8192, int(c.lengths.size()), forced_width);
    h.upload(device.execution_view());
    const auto bytes = ops::qsa_sparse_gqa_workspace_capacity_bytes(
        KvCacheStorage::BFloat16, c.query_heads, c.query_heads / 12, h.T, h.T, batch, h.envelope,
        device.execution_view());
    WorkspaceArena workspace(std::max<std::size_t>(bytes, 256));
    h.launch(workspace, device.execution_view());
    device.synchronize();
    CUDA_CHECK(cudaGetLastError());
    return ArmOut{from_device<std::uint16_t>(h.out.p, h.qhost.size()), h.T};
}

bool all_finite(const std::vector<std::uint16_t>& words) {
    for (auto w : words)
        if ((w & 0x7F80U) == 0x7F80U) return false;
    return true;
}

int run_case(DeviceContext& device, const Case& c, const std::string& name) {
    int failures = 0;
    const int rows = int(c.lengths.size());
    const int width = *std::max_element(c.lengths.begin(), c.lengths.end());
    std::vector<int> all_rows(rows);
    for (int b = 0; b < rows; ++b) all_rows[b] = b;
    const auto merged = arm(device, c, all_rows, rows, true);
    if (!all_finite(merged.words)) {
        std::fprintf(stderr, "%s: merged arm produced a non-finite value\n", name.c_str());
        ++failures;
    }
    // The Op's layout is [dim, head, token, batch], so one element is addressed by the same stride
    // arithmetic in either arm; only the rectangle width (the batch stride) and the batch extent
    // differ, and each arm reports its own width.
    const auto at = [&](const ArmOut& a, int d, int h, int t, int b) {
        return a.words[std::size_t(d) + std::size_t(kHeadDim) *
                                        (std::size_t(h) + std::size_t(c.query_heads) *
                                                              (std::size_t(t) +
                                                               std::size_t(a.tokens) * b))];
    };
    for (int b = 0; b < rows; ++b) {
        const auto alone = arm(device, c, {b}, 1, false, width);
        if (!all_finite(alone.words)) {
            std::fprintf(stderr, "%s: row %d alone produced a non-finite value\n", name.c_str(), b);
            ++failures;
        }
        const int n      = c.lengths[b];
        std::size_t compared = 0;
        int differing        = 0;
        int max_ulp          = 0;
        std::vector<int> first_hits;
        int first_t = -1, first_h = -1, first_d = -1;
        for (int t = 0; t < n; ++t)
            for (int h = 0; h < c.query_heads; ++h)
                for (int d = 0; d < kHeadDim; ++d) {
                    ++compared;
                    const auto got   = at(merged, d, h, t, b);
                    const auto alone_word = at(alone, d, h, t, 0);
                    if (got != alone_word) {
                        ++differing;
                        max_ulp = std::max(max_ulp, std::abs(ordered_key(got) - ordered_key(alone_word)));
                        if (first_t < 0) { first_t = t; first_h = h; first_d = d; }
                        if (first_hits.size() < 12 &&
                            (first_hits.empty() || first_hits.back() != t)) first_hits.push_back(t);
                    }
                }
        if (differing != 0) {
            std::fprintf(stderr,
                         "%s: row %d (len %d) differs from its single-row run in %d of %zu, max %d "
                         "BF16 steps; first at (t=%d,h=%d,d=%d), tokens",
                         name.c_str(), b, n, differing, compared, max_ulp, first_t, first_h, first_d);
            for (int t : first_hits) std::fprintf(stderr, " %d", t);
            std::fprintf(stderr, "\n");
            ++failures;
        }
        if (std::getenv("NINFER_QSA_BATCH_TRACE") != nullptr) {
            std::fprintf(stderr, "[batch] %s row=%d len=%d compared=%zu differing=%d\n", name.c_str(),
                         b, n, compared, differing);
        }
    }
    std::printf("%s: rows %d, width %d, heads %d\n", name.c_str(), rows, width, c.query_heads);
    return failures;
}

} // namespace

int main() {
    try {
        if (cuda_unavailable()) return 77;
        DeviceContext device;
        int failures = 0;
        // Mixed lengths in one rectangle, including a one-token row and a row that stops early.
        failures += run_case(device,
                             Case{{51, 17, 1, 8, 33, 64, 2, 7}, 24}, "mixed-8");
        failures += run_case(device, Case{{17, 51}, 24}, "mixed-2");
        // Equal lengths isolate "any batch past the first token tile" from "mixed lengths".
        failures += run_case(device, Case{{51, 51, 51, 51, 51, 51, 51, 51}, 24}, "equal-8");
        failures += run_case(device, Case{{51, 51}, 24}, "equal-2");
        // Crosses one 64-token index page: 130 tokens live in three pages of blocks.
        failures += run_case(device, Case{{130, 129}, 24}, "cross-page");
        // Past 2048 tokens on one row: more than 512 index blocks.
        failures += run_case(device, Case{{2049, 2048}, 12}, "past-512-blocks");
        if (failures == 0) std::printf("QSA batch: PASS\n");
        else std::printf("QSA batch: %d failures\n", failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "QSA batch: %s\n", error.what());
        return 1;
    }
}
