#pragma once

#include "options.h"
#include "ninfer/ops/ple.h"

#include <memory>
#include <numeric>

namespace ninfer::bench::ple {

inline constexpr int H = 2560, C = 4, D = H * C;

struct ConvFixture {
    const Options& options;
    int tokens, capacity;
    DeviceBuffer residual, gated, kernel, norm, history, records, updated, normalized;
    std::vector<int> counts;
    DeviceBuffer valid, slots, offsets;
    std::unique_ptr<SavedBuffer> saved;

    ConvFixture(const Options& o, int t)
        : options(o), tokens(t), capacity(t + o.offset),
          residual(make_bf16(std::size_t(D) * t * o.batch, 101, -1, 1)),
          gated(make_bf16(std::size_t(D) * t * o.batch, 103, -1, 1)),
          kernel(make_bf16(D * 4, 107, -.25f, .25f)), norm(make_bf16(D, 109)),
          history(make_bf16(std::size_t(D) * 9 * o.batch, 113, -1, 1)),
          records(make_bf16(std::size_t(D) * capacity * o.batch, 127, -1, 1)),
          updated(o.inplace ? DeviceBuffer{} : make_zeros(std::size_t(D) * t * o.batch * 2)),
          normalized(o.op == "conv-add-norm" ? make_zeros(std::size_t(D) * t * o.batch * 2)
                                             : DeviceBuffer{}),
          counts(extents(o.valid, t, o.batch)), valid(make_i32(counts)),
          offsets(make_i32(std::vector<int>(o.batch, o.offset))) {
        std::vector<int> indices(o.batch);
        std::iota(indices.begin(), indices.end(), 0);
        slots = make_i32(indices);
        if (o.inplace) saved = std::make_unique<SavedBuffer>(residual);
    }

    void restore(cudaStream_t stream) const {
        if (saved) saved->restore(stream);
    }

    void launch(DeviceExecutionView execution) {
        const int b = options.batch;
        Tensor h(residual.p, DType::BF16, {H, C, tokens, b}),
            u(gated.p, DType::BF16, {H, C, tokens, b});
        Tensor w(kernel.p, DType::BF16, {D, 4}),
            y(options.inplace ? residual.p : updated.p, DType::BF16, {H, C, tokens, b});
        const ops::PleBatchView batch{Tensor(valid.p, DType::I32, {b})};
        const ops::PleConvStateView before{Tensor(history.p, DType::BF16, {D, 9, b}),
                                           Tensor(slots.p, DType::I32, {b})};
        const ops::PleConvRecordWindowView current{Tensor(records.p, DType::BF16, {D, capacity, b}),
                                                   Tensor(offsets.p, DType::I32, {b})};
        if (options.op == "conv-add-norm") {
            Tensor weight(norm.p, DType::BF16, {H, C}),
                z(normalized.p, DType::BF16, {H, C, tokens, b});
            ops::ple_conv_add_norm(h, u, w, batch, before, current, weight, 1e-6f, y, z, execution);
        } else
            ops::ple_conv_add(h, u, w, batch, before, current, y, execution);
    }
};

struct ExtractFixture {
    const Options& options;
    int tokens;
    DeviceBuffer history, records, after, slots, valid, accepted;
    std::vector<int> counts, commits;
    std::unique_ptr<SavedBuffer> saved;

    ExtractFixture(const Options& o, int t)
        : options(o), tokens(t), history(make_bf16(std::size_t(D) * 9 * o.batch, 211)),
          records(make_bf16(std::size_t(D) * t * o.batch, 223)),
          after(o.inplace ? DeviceBuffer{} : make_zeros(std::size_t(D) * 9 * o.batch * 2)),
          counts(extents(o.valid, t, o.batch)),
          commits(o.accepted == "dense" ? counts : extents(o.accepted, t, o.batch)) {
        for (int i = 0; i < o.batch; ++i)
            if (commits[i] > counts[i])
                throw std::invalid_argument("accepted exceeds recorded prefix");
        std::vector<int> indices(o.batch);
        std::iota(indices.begin(), indices.end(), 0);
        slots    = make_i32(indices);
        valid    = make_i32(counts);
        accepted = make_i32(commits);
        if (o.inplace) saved = std::make_unique<SavedBuffer>(history);
    }

    void restore(cudaStream_t stream) const {
        if (saved) saved->restore(stream);
    }

    void launch(DeviceExecutionView execution) {
        const int b = options.batch;
        const Tensor indices(slots.p, DType::I32, {b}), a(accepted.p, DType::I32, {b});
        ops::ple_state_extract(
            {Tensor(history.p, DType::BF16, {D, 9, b}), indices},
            {Tensor(records.p, DType::BF16, {D, tokens, b}), Tensor(valid.p, DType::I32, {b})}, a,
            {Tensor(options.inplace ? history.p : after.p, DType::BF16, {D, 9, b}), indices},
            execution);
    }
};

} // namespace ninfer::bench::ple

