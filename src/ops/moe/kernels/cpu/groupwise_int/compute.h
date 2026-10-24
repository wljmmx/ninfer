#pragma once

#include "ops/moe/kernels/cpu/groupwise_int/work.h"
#include "ops/common/numeric/math.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail::moe_cpu::groupwise_int {

// Instantiated only in an ISA-specific C++ object. A decode feeds up to four columns;
// group scales are applied to vector partials, with one horizontal reduction per output.
template <class Simd, int Bits, int K, int Rows, int Columns, class Input>
inline void dot(const Matrix& matrix, const int (&rows)[Rows], const Input* const (&x)[Columns],
                float (&out)[Rows][Columns]) noexcept {
    using V = typename Simd::Float;
    V total[Rows][Columns];
    for (int r = 0; r < Rows; ++r) {
        for (int c = 0; c < Columns; ++c) { total[r][c] = Simd::zero(); }
    }
    for (int group = 0; group < K / 64; ++group) {
        V partial[Rows][Columns];
        for (int r = 0; r < Rows; ++r) {
            for (int c = 0; c < Columns; ++c) { partial[r][c] = Simd::zero(); }
        }
        for (int k = 0; k < 64; k += Simd::lanes) {
            V input[Columns];
            for (int c = 0; c < Columns; ++c) {
                if constexpr (std::is_same_v<Input, float>) {
                    input[c] = Simd::load(x[c] + group * 64 + k);
                } else {
                    input[c] = Simd::bf16(x[c] + group * 64 + k);
                }
            }
            for (int r = 0; r < Rows; ++r) {
                const auto* codes        = matrix.codes + rows[r] * (K / 2) + group * 32 + k / 2;
                const std::uint8_t* high = nullptr;
                if constexpr (Bits > 4) {
                    high = matrix.high + (rows[r] * (K / 8) + group * 8 + k / 8) * (Bits - 4);
                }
                const auto values = Simd::template decode<Bits>(codes, high);
                for (int c = 0; c < Columns; ++c) {
                    partial[r][c] = Simd::fma(values, input[c], partial[r][c]);
                }
            }
        }
        for (int r = 0; r < Rows; ++r) {
            const auto scale = Simd::scale(matrix.scales[rows[r] * (K / 64) + group]);
            for (int c = 0; c < Columns; ++c) {
                total[r][c] = Simd::fma(partial[r][c], scale, total[r][c]);
            }
        }
    }
    for (int r = 0; r < Rows; ++r) {
        for (int c = 0; c < Columns; ++c) { out[r][c] = Simd::sum(total[r][c]); }
    }
}

// port (sm_89): int8 dot product for 4-bit weights.
//
// The packed weights hold weight 2b in the low nibble of byte b and 2b+1 in the high nibble, and
// each nibble is two's complement - `decode_signed_q<4>` maps u to (u ^ 8) - 8. Flipping bit 3
// turns it into the offset-by-8 form b = u ^ 8, so w = b - 8 and maddubs' unsigned x signed
// product needs one correction per 64-value group: sum(b * x) = sum(w * x) + 8 * sum(x), with
// sum(x) accumulated over the quantised values while the row is quantised. The activations are
// stored even-first and odd-second, so each nibble half meets its own half of the row with no
// unpacking at all. One weight load, one xor, two masks, two maddubs, two madds and two adds
// cover 64 weights, against roughly a hundred instructions per eight weights on the float path.
// Out of line: this only runs when the path is enabled, and inlining it would let it change the
// layout of the vector loop that runs when it is not.
template <class Simd, int K, int Rows, int Columns>
[[gnu::noinline]] void dot_q4_q8(Work& work, const Matrix& matrix, const int (&rows)[Rows],
                                 const int* columns, float (&out)[Rows][Columns]) noexcept {
    static_assert(std::is_base_of_v<common::cpu::avx2::F32, Simd>,
                  "the int8 path is written against AVX2 registers");
    constexpr int kGroups = K / 64;
    const auto ones       = _mm256_set1_epi16(1);
    const auto nibble     = _mm256_set1_epi8(0x0F);
    // Bit 3 of each nibble is its sign, so flipping it in both nibbles at once turns the two's
    // complement nibble u into the offset-by-8 value u ^ 8 (0x88 is bit 3 of the low nibble and
    // bit 3 of the high one).
    const auto bias = _mm256_set1_epi8(static_cast<char>(0x88));
    // One vector accumulator per output and one reduction at the end: a horizontal sum per group
    // would cost more than the products it is summing.
    __m256 total[Rows][Columns];
    for (int r = 0; r < Rows; ++r) {
        for (int c = 0; c < Columns; ++c) { total[r][c] = _mm256_setzero_ps(); }
    }
    for (int group = 0; group < kGroups; ++group) {
        // The activation rows depend only on the group, so hoist them out of the row loop.
        __m256i even[Columns];
        __m256i odd[Columns];
        __m256i correction[Columns];
        float scale[Columns];
        for (int c = 0; c < Columns; ++c) {
            const auto* row = work.scratch.input_q8 + columns[c] * K + group * 64;
            even[c]       = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row));
            odd[c]        = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + 32));
            // One lane carries the whole offset: the group's 8 * sum(x) is subtracted once, not
            // once per lane, because the lanes are summed at the end of the group.
            correction[c] = _mm256_setr_epi32(
                8 * work.scratch.input_sum[columns[c] * kGroups + group], 0, 0, 0, 0, 0, 0, 0);
            scale[c] = work.scratch.input_scale[columns[c] * kGroups + group];
        }
        for (int r = 0; r < Rows; ++r) {
            const auto* codes  = matrix.codes + std::size_t(rows[r]) * (K / 2) + group * 32;
            const auto flipped = _mm256_xor_si256(
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes)), bias);
            const auto low  = _mm256_and_si256(flipped, nibble);
            const auto high = _mm256_and_si256(_mm256_srli_epi16(flipped, 4), nibble);
            // The weight scale is per row and group, so it folds into the same broadcast.
            const float weight = common::cpu::avx2::fp16_to_f32(
                matrix.scales[std::size_t(rows[r]) * kGroups + group]);
            for (int c = 0; c < Columns; ++c) {
                const auto pairs = _mm256_sub_epi32(
                    _mm256_add_epi32(
                        _mm256_madd_epi16(_mm256_maddubs_epi16(low, even[c]), ones),
                        _mm256_madd_epi16(_mm256_maddubs_epi16(high, odd[c]), ones)),
                    correction[c]);
                total[r][c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(pairs),
                                              _mm256_set1_ps(scale[c] * weight), total[r][c]);
            }
        }
    }
    for (int r = 0; r < Rows; ++r) {
        for (int c = 0; c < Columns; ++c) { out[r][c] = common::cpu::avx2::reduce_sum(total[r][c]); }
    }
}

// channel_block (the maximum unit width) lives in work.h.

template <class Simd, int Columns>
inline void gate_up(Work& work, const Group& group, int first, int channel, int begin,
                    unsigned bf16_columns) noexcept {
    constexpr int rows = Columns == 4 ? Simd::rows / 2 : Simd::rows;
    const std::conditional_t<Simd::bf16_gate, std::uint16_t, float>* input[Columns];
    for (int c = 0; c < Columns; ++c) {
        if constexpr (Simd::bf16_gate) {
            input[c] = work.args.input +
                       static_cast<std::size_t>(begin + group.tokens[first + c]) * hidden;
        } else {
            input[c] = work.scratch.input + group.tokens[first + c] * hidden;
        }
    }
    for (int i = channel; i < channel + work.block; i += rows / 2) {
        int indices[rows];
        for (int r = 0; r < rows / 2; ++r) {
            indices[2 * r]     = i + r;
            indices[2 * r + 1] = intermediate + i + r;
        }
        float values[rows][Columns];
        // The bf16 gate path only covers whole columns, so it is tried first and everything else
        // falls through to the int4 dot product.
        const auto run_default = [&]() noexcept {
            if constexpr (Simd::bf16_gate) {
                unsigned mask = 0;
                for (int c = 0; c < Columns; ++c) { mask |= 1U << group.tokens[first + c]; }
                if ((bf16_columns & mask) == mask) {
                    Simd::gate_dot(group.gate_up, indices, input, values);
                    return;
                }
            }
            dot<Simd, 4, hidden>(group.gate_up, indices, input, values);
        };
        if constexpr (std::is_base_of_v<common::cpu::avx2::F32, Simd>) {
            if (work.int8_path) {
                dot_q4_q8<Simd, hidden, rows, Columns>(work, group.gate_up, indices,
                                                       group.tokens + first, values);
            } else {
                run_default();
            }
        } else {
            run_default();
        }
        for (int r = 0; r < rows / 2; ++r) {
            for (int c = 0; c < Columns; ++c) {
                // The device kernels store this plane as BF16 and feed the down projection from
                // it, so round here too; otherwise the host fallback is more precise than the
                // path it is standing in for and the two diverge over the layer stack.
                work.scratch.activation[group.slots[first + c] * intermediate + i + r] =
                    common::bf16_round(common::silu_stable(values[2 * r][c]) *
                                       values[2 * r + 1][c]);
            }
        }
    }
}

template <class Simd, int DownBits, int Columns>
inline void down(Work& work, const Group& group, int first, int channel,
                 float (&accumulated)[max_token_block][channel_block]) noexcept {
    constexpr int rows = Columns == 4 ? Simd::rows / 2 : Simd::rows;
    const float* input[Columns];
    for (int c = 0; c < Columns; ++c) {
        input[c] = work.scratch.activation + group.slots[first + c] * intermediate;
    }
    for (int i = 0; i < work.block; i += rows) {
        int indices[rows];
        for (int r = 0; r < rows; ++r) { indices[r] = channel + i + r; }
        float values[rows][Columns];
        dot<Simd, DownBits, intermediate>(group.down, indices, input, values);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < Columns; ++c) {
                accumulated[group.tokens[first + c]][i + r] +=
                    group.coefficients[first + c] * values[r][c];
            }
        }
    }
}

template <class Simd, int DownBits, bool FourColumns>
[[gnu::always_inline]] inline void compute_chunk(Work& work, const CpuWorkerContext& worker,
                                                 int begin, int columns,
                                                 unsigned bf16_columns) noexcept {
    const int blocks = intermediate / work.block;
    const auto tasks     = worker.range(work.task_count * blocks);
    for (auto index = tasks.begin; index < tasks.end; ++index) {
        const auto& task  = work.scratch.tasks[index / blocks];
        const auto& group = work.scratch.groups[task.group_index];
        const int channel = (index % blocks) * work.block;
        if constexpr (FourColumns) {
            if (task.columns == 4) {
                gate_up<Simd, 4>(work, group, task.first_column, channel, begin, bf16_columns);
                continue;
            }
        }
        if (task.columns == 2) {
            gate_up<Simd, 2>(work, group, task.first_column, channel, begin, bf16_columns);
        } else {
            gate_up<Simd, 1>(work, group, task.first_column, channel, begin, bf16_columns);
        }
    }
    worker.barrier();
    const auto outputs = worker.range(hidden / work.block);
    for (auto block = outputs.begin; block < outputs.end; ++block) {
        const int channel = block * work.block;
        float accumulated[max_token_block][channel_block]{};
        if constexpr (FourColumns) {
            for (int index = 0; index < work.task_count; ++index) {
                const auto& task  = work.scratch.tasks[index];
                const auto& group = work.scratch.groups[task.group_index];
                if (task.columns == 4) {
                    down<Simd, DownBits, 4>(work, group, task.first_column, channel, accumulated);
                    continue;
                }
                if (task.columns == 2) {
                    down<Simd, DownBits, 2>(work, group, task.first_column, channel, accumulated);
                } else {
                    down<Simd, DownBits, 1>(work, group, task.first_column, channel, accumulated);
                }
            }
        } else {
            for (int g = 0; g < work.group_count; ++g) {
                const auto& group = work.scratch.groups[g];
                for (int first = 0; first < group.count; first += 2) {
                    if (first + 1 < group.count) {
                        down<Simd, DownBits, 2>(work, group, first, channel, accumulated);
                    } else {
                        down<Simd, DownBits, 1>(work, group, first, channel, accumulated);
                    }
                }
            }
        }
        for (int t = 0; t < columns; ++t) {
            auto* output =
                work.args.output + static_cast<std::size_t>(begin + t) * hidden + channel;
            for (int i = 0; i < work.block; ++i) { output[i] = accumulated[t][i]; }
        }
    }
}

// Keep the wide-column specializations outside the worker's live register range.
template <class Simd, int DownBits, bool FourColumns>
[[gnu::noinline]] void compute_wide_chunk(Work& work, const CpuWorkerContext& worker, int begin,
                                          int columns, unsigned bf16_columns) noexcept {
    compute_chunk<Simd, DownBits, FourColumns>(work, worker, begin, columns, bf16_columns);
}

// port (sm_89): quantise the input row to int8 in 64-value groups - the width the weight scales
// already use - so the q4 dot product can multiply 32 weights per instruction pair instead of
// decoding eight at a time. Decode is 44% of this kernel, which is what makes the rounding error
// (~0.1-0.5%, inside the split gate's 1% tolerance) worth paying. Out of line so it cannot change
// the layout of the vector loop that runs when the path is disabled.
template <class Simd>
[[gnu::noinline]] void quantise_input(Work& work, const CpuWorkerContext& worker,
                                      int columns) noexcept {
    constexpr int kGroups = hidden / 64;
    const auto range      = worker.range(kGroups * columns);
    for (auto g = range.begin; g < range.end; ++g) {
        const int column    = static_cast<int>(g) / kGroups;
        const int group     = static_cast<int>(g) % kGroups;
        const float* source = work.scratch.input + column * hidden + group * 64;
        float peak          = 0.0F;
        for (int i = 0; i < 64; ++i) { peak = std::max(peak, std::fabs(source[i])); }
        const float scale   = peak > 0.0F ? peak / 127.0F : 0.0F;
        const float inverse = scale > 0.0F ? 1.0F / scale : 0.0F;
        work.scratch.input_scale[column * kGroups + group] = scale;
        auto* out = work.scratch.input_q8 + column * hidden + group * 64;
        // The packed weights give weight 2b in the low nibble of byte b and 2b+1 in the high
        // one, so the activations are stored even-first and odd-second: the two nibble halves
        // then multiply straight against the two halves of the row, with no shuffle at all.
        int sum = 0;
        for (int i = 0; i < 32; ++i) {
            const int even = std::clamp(static_cast<int>(std::lrint(source[2 * i] * inverse)), -127, 127);
            const int odd  = std::clamp(static_cast<int>(std::lrint(source[2 * i + 1] * inverse)), -127, 127);
            out[i]      = static_cast<std::int8_t>(even);
            out[32 + i] = static_cast<std::int8_t>(odd);
            sum += even + odd;
        }
        work.scratch.input_sum[column * kGroups + group] = sum;
    }
}

template <class Simd, int DownBits>
void compute(void* argument, const CpuWorkerContext& worker) noexcept {
    auto& work = *static_cast<Work*>(argument);
    for (int begin = 0; begin < work.args.tokens;) {
        const int columns = std::min(work.plan.token_block, work.args.tokens - begin);
        // Caller-prepared metadata is published with run(); worker-prepared metadata
        // is published with the input barrier. Later chunks reuse joined scratch.
        if (begin != 0 || Simd::first_chunk == FirstChunkPreparation::Worker) {
            if (worker.rank() == 0) { prepare_chunk(work, begin, columns); }
        }
        if constexpr (!Simd::bf16_gate) {
            const auto input_range = worker.range(hidden * columns / Simd::lanes);
            for (auto v = input_range.begin; v < input_range.end; ++v) {
                const auto index = v * Simd::lanes;
                Simd::store(
                    work.scratch.input + index,
                    Simd::bf16(work.args.input + static_cast<std::size_t>(begin) * hidden + index));
            }
        }
        if (begin != 0 || !Simd::bf16_gate) { worker.barrier(); }
        if (work.int8_path) {
            quantise_input<Simd>(work, worker, columns);
            worker.barrier();
        }
        // port (sm_89): a split layer routes to a handful of experts, so at 16 channels per unit
        // the team gets 32 units per expert - one or two each, with a barrier on both sides.
        // Halving the unit doubles the units and lets the same team work; with eight experts the
        // wider unit wins instead. Sizes measured with bench/ops/moe_cpu_bench.
        const int units = work.task_count * (intermediate / channel_block);
        work.block = (work.workers > 1 && units < work.workers * 8) ? channel_block / 2
                                                                   : channel_block;
        unsigned bf16_columns = 0;
        if constexpr (Simd::bf16_gate) {
            bf16_columns = Simd::safe_columns(
                work.args.input + static_cast<std::size_t>(begin) * hidden, columns);
        }
        if constexpr (Simd::max_columns == 4) {
            if (work.has_four_columns) {
                compute_wide_chunk<Simd, DownBits, true>(work, worker, begin, columns,
                                                         bf16_columns);
            } else {
                compute_wide_chunk<Simd, DownBits, false>(work, worker, begin, columns,
                                                          bf16_columns);
            }
        } else {
            compute_chunk<Simd, DownBits, false>(work, worker, begin, columns, bf16_columns);
        }
        begin += columns;
        // Protect shared scratch before the next chunk; the final join is provided by run().
        if (begin < work.args.tokens) { worker.barrier(); }
    }
}

template <class Simd>
constexpr Kernel make_kernel() {
    return {.run =
                [](void* argument, const CpuWorkerContext& worker) noexcept {
                    if (static_cast<Work*>(argument)->plan.down_bits == 6) {
                        compute<Simd, 6>(argument, worker);
                    } else {
                        compute<Simd, 5>(argument, worker);
                    }
                },
            .first_chunk = Simd::first_chunk};
}

} // namespace ninfer::ops::detail::moe_cpu::groupwise_int
