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
// port (sm_89): five-bit codes through the 32-wide decoder. Same arithmetic as dot(), one pass per
// 32 weights instead of four passes of eight. NINFER_MOE_CPU_NODECODE puts the decode at 32% of this
// kernel, and this is the only cut available without changing the activation precision - int8
// activations for this projection were rejected at 12.3% PPL, so the dot stays in float.
template <class Simd, int K, int Rows, int Columns, class Input>
inline void dot_wide32(const Matrix& matrix, const int (&rows)[Rows],
                       const Input* const (&x)[Columns], float (&out)[Rows][Columns]) noexcept {
    static_assert(Simd::lanes == 8, "the 32-wide decoder feeds four 8-lane vectors");
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
        for (int k = 0; k < 64; k += 32) {
            for (int r = 0; r < Rows; ++r) {
                const auto* codes =
                    matrix.codes + std::size_t(rows[r]) * (K / 2) + group * 32 + k / 2;
                const auto* high = matrix.high + std::size_t(rows[r]) * (K / 8) + group * 8 + k / 8;
                V values[4];
                common::cpu::avx2::decode_signed_q32_q5(codes, high, values);
                for (int c = 0; c < Columns; ++c) {
                    for (int q = 0; q < 4; ++q) {
                        const int offset = group * 64 + k + q * Simd::lanes;
                        V in;
                        if constexpr (std::is_same_v<Input, float>) {
                            in = Simd::load(x[c] + offset);
                        } else {
                            in = Simd::bf16(x[c] + offset);
                        }
                        partial[r][c] = Simd::fma(values[q], in, partial[r][c]);
                    }
                }
            }
        }
        for (int r = 0; r < Rows; ++r) {
            const auto scale = Simd::scale(matrix.scales[std::size_t(rows[r]) * (K / 64) + group]);
            for (int c = 0; c < Columns; ++c) {
                total[r][c] = Simd::fma(partial[r][c], scale, total[r][c]);
            }
        }
    }
    for (int r = 0; r < Rows; ++r) {
        for (int c = 0; c < Columns; ++c) { out[r][c] = Simd::sum(total[r][c]); }
    }
}

template <class Simd, int Bits, int K, int Rows, int Columns, class Input>
inline void dot(const Matrix& matrix, const int (&rows)[Rows], const Input* const (&x)[Columns],
                float (&out)[Rows][Columns]) noexcept {
    if constexpr (Bits == 8) {
        // port (sm_89, P4-B): Q8_G32_FP16 rows — one plain int8 code per dimension and ONE
        // FP16 scale per 32-value group (Q8RowSplitStorage, ops/common/quant/integer.cuh:
        // 311-316), so both the group count and the plane strides differ from the packed
        // 4..6-bit family: codes stride K bytes per row (not K/2) and scales stride K/32 FP16
        // elements (not K/64). The accumulate order keeps the family's exact shape — group
        // partials scaled once per group, one horizontal reduction per output.
        static_assert(Bits == 8 && K % 32 == 0);
        using V = typename Simd::Float;
        constexpr int kGroups32 = K / 32;
        V total[Rows][Columns];
        for (int r = 0; r < Rows; ++r) {
            for (int c = 0; c < Columns; ++c) { total[r][c] = Simd::zero(); }
        }
        for (int group = 0; group < kGroups32; ++group) {
            V partial[Rows][Columns];
            for (int r = 0; r < Rows; ++r) {
                for (int c = 0; c < Columns; ++c) { partial[r][c] = Simd::zero(); }
            }
            for (int k = 0; k < 32; k += Simd::lanes) {
                V input[Columns];
                for (int c = 0; c < Columns; ++c) {
                    if constexpr (std::is_same_v<Input, float>) {
                        input[c] = Simd::load(x[c] + group * 32 + k);
                    } else {
                        input[c] = Simd::bf16(x[c] + group * 32 + k);
                    }
                }
                for (int r = 0; r < Rows; ++r) {
                    const auto* codes = matrix.codes + std::size_t(rows[r]) * K + group * 32 + k;
                    const auto values = Simd::template decode<8>(codes, nullptr);
                    for (int c = 0; c < Columns; ++c) {
                        partial[r][c] = Simd::fma(values, input[c], partial[r][c]);
                    }
                }
            }
            for (int r = 0; r < Rows; ++r) {
                const auto scale =
                    Simd::scale(matrix.scales[std::size_t(rows[r]) * kGroups32 + group]);
                for (int c = 0; c < Columns; ++c) {
                    total[r][c] = Simd::fma(partial[r][c], scale, total[r][c]);
                }
            }
        }
        for (int r = 0; r < Rows; ++r) {
            for (int c = 0; c < Columns; ++c) { out[r][c] = Simd::sum(total[r][c]); }
        }
        return;
    }
    if constexpr (Bits == 5 && std::is_base_of_v<common::cpu::avx2::F32, Simd>) {
        // Read once: getenv in the dispatch would run per expert, and this is an A/B knob.
        static const bool narrow_q5 = std::getenv("NINFER_MOE_CPU_Q5_NARROW") != nullptr;
        if (!narrow_q5) {
            dot_wide32<Simd, K, Rows, Columns, Input>(matrix, rows, x, out);
            return;
        }
    }
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
            const float weight = common::cpu::avx2::fp16_to_f32(
                matrix.scales[std::size_t(rows[r]) * kGroups + group]);
            const auto weight_vector = _mm256_set1_ps(weight);
            for (int c = 0; c < Columns; ++c) {
                const auto pairs = _mm256_sub_epi32(
                    _mm256_add_epi32(
                        _mm256_madd_epi16(_mm256_maddubs_epi16(low, even[c]), ones),
                        _mm256_madd_epi16(_mm256_maddubs_epi16(high, odd[c]), ones)),
                    correction[c]);
                // Scale chain in the device order - gate[0] += (float)acc_g[0] * sg_lo * a0
                // (grouped_gate_up_i8_tile.cuh): weight scale first, activation scale second,
                // one add at the end. Two roundings where a fused scale[c] * weight had one.
                const auto scaled = _mm256_mul_ps(_mm256_cvtepi32_ps(pairs), weight_vector);
                total[r][c] =
                    _mm256_add_ps(_mm256_mul_ps(scaled, _mm256_set1_ps(scale[c])), total[r][c]);
            }
        }
    }
    for (int r = 0; r < Rows; ++r) {
        for (int c = 0; c < Columns; ++c) { out[r][c] = common::cpu::avx2::reduce_sum(total[r][c]); }
    }
}

// port (sm_89): int16 dot product for 4-bit weights - the same tier as dot_q4_q8 with 258x
// finer activation codes. The activation plane stores natural-order int16 (two bytes a value),
// so the dot pays twice the bytes maddubs paid but keeps one instruction shape per 16 products.
//
// The packed weights hold weight 2b in the low nibble of byte b and 2b+1 in the high nibble,
// two's complement. unpack_u4 widens 16 code bytes into 16 lanes that each pack the pair
// w[2i] | w[2i+1] << 8; read as 32 bytes those are the group's nibbles in natural order, so
// widening each 128-bit half with cvtepu8_epi16 turns them into 16-bit channels 0..15, and
// (x ^ 8) - 8 restores the sign at two instructions per 16 values. madd then pairs adjacent
// weights and values with no shuffles at all, where the int8 path had to interleave its rows.
// Overflow: one group's partial sum is at most 64 * 8 * 32767 = 16777504 < 2^31, so the int32
// lanes of the four madds never wrap and the group total is exact.
// Out of line like dot_q4_q8: this only runs when the path is enabled, and inlining it would
// let it change the layout of the vector loop that runs when it is not.
template <class Simd, int K, int Rows, int Columns>
[[gnu::noinline]] void dot_q4_q16(Work& work, const Matrix& matrix, const int (&rows)[Rows],
                                  const int* columns, float (&out)[Rows][Columns]) noexcept {
    static_assert(std::is_base_of_v<common::cpu::avx2::F32, Simd>,
                  "the int16 path is written against AVX2 registers");
    constexpr int kGroups = K / 64;
    const auto eight      = _mm256_set1_epi16(8);
    // One vector accumulator per output and one reduction at the end: a horizontal sum per group
    // would cost more than the products it is summing.
    __m256 total[Rows][Columns];
    for (int r = 0; r < Rows; ++r) {
        for (int c = 0; c < Columns; ++c) { total[r][c] = _mm256_setzero_ps(); }
    }
    for (int group = 0; group < kGroups; ++group) {
        // The activation rows depend only on the group, so hoist them out of the row loop.
        __m256i activation[Columns][4];
        __m256 activation_scale[Columns];
        for (int c = 0; c < Columns; ++c) {
            const auto* row = work.scratch.input_q16 + columns[c] * K + group * 64;
            for (int q = 0; q < 4; ++q) {
                activation[c][q] =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + q * 16));
            }
            activation_scale[c] =
                _mm256_set1_ps(work.scratch.input_scale[columns[c] * kGroups + group]);
        }
        for (int r = 0; r < Rows; ++r) {
            const auto* codes = matrix.codes + std::size_t(rows[r]) * (K / 2) + group * 32;
            __m256i weights[4];
            for (int half = 0; half < 2; ++half) {
                const auto packed = common::cpu::avx2::unpack_u4(codes + half * 16);
                for (int part = 0; part < 2; ++part) {
                    const auto bytes = part == 0 ? _mm256_castsi256_si128(packed)
                                                 : _mm256_extracti128_si256(packed, 1);
                    const auto channels = _mm256_cvtepu8_epi16(bytes);
                    weights[2 * half + part] =
                        _mm256_sub_epi16(_mm256_xor_si256(channels, eight), eight);
                }
            }
            const auto weight_vector = _mm256_set1_ps(common::cpu::avx2::fp16_to_f32(
                matrix.scales[std::size_t(rows[r]) * kGroups + group]));
            for (int c = 0; c < Columns; ++c) {
                const auto acc = _mm256_add_epi32(
                    _mm256_add_epi32(_mm256_madd_epi16(weights[0], activation[c][0]),
                                     _mm256_madd_epi16(weights[1], activation[c][1])),
                    _mm256_add_epi32(_mm256_madd_epi16(weights[2], activation[c][2]),
                                     _mm256_madd_epi16(weights[3], activation[c][3])));
                // Scale chain in the device order - gate[0] += (float)acc_g[0] * sg_lo * a0
                // (grouped_gate_up_i8_tile.cuh): weight scale first, activation scale second,
                // one add at the end.
                const auto scaled = _mm256_mul_ps(_mm256_cvtepi32_ps(acc), weight_vector);
                total[r][c] =
                    _mm256_add_ps(_mm256_mul_ps(scaled, activation_scale[c]), total[r][c]);
            }
        }
    }
    for (int r = 0; r < Rows; ++r) {
        for (int c = 0; c < Columns; ++c) { out[r][c] = common::cpu::avx2::reduce_sum(total[r][c]); }
    }
}

// channel_block (the maximum unit width) lives in work.h.
//
// GateBits selects the gate/up plane's code width: 4 for the q4 family (q4_q5 / q4_q6), 8 for
// the Q8_G32_FP16 gate of q8_q8 (P4-B). The int8/int16 activation tiers and the bf16 gate dot
// are q4-gate-only paths (they decode packed nibbles), so GateBits == 8 always runs the float
// decode + FMA dot above.

template <class Simd, int GateBits, int Columns>
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
        // falls through to the bit-selected dot product.
        const auto run_default = [&]() noexcept {
            if constexpr (Simd::bf16_gate && GateBits == 4) {
                unsigned mask = 0;
                for (int c = 0; c < Columns; ++c) { mask |= 1U << group.tokens[first + c]; }
                if ((bf16_columns & mask) == mask) {
                    Simd::gate_dot(group.gate_up, indices, input, values);
                    return;
                }
            }
            dot<Simd, GateBits, hidden>(group.gate_up, indices, input, values);
        };
        if constexpr (GateBits == 4) {
            if constexpr (std::is_base_of_v<common::cpu::avx2::F32, Simd>) {
                if (work.int8_path) {
                    dot_q4_q8<Simd, hidden, rows, Columns>(work, group.gate_up, indices,
                                                           group.tokens + first, values);
                } else if (work.int16_path) {
                    dot_q4_q16<Simd, hidden, rows, Columns>(work, group.gate_up, indices,
                                                             group.tokens + first, values);
                } else {
                    run_default();
                }
            } else {
                run_default();
            }
        } else {
            run_default();
        }
        for (int r = 0; r < rows / 2; ++r) {
            for (int c = 0; c < Columns; ++c) {
                const float value =
                    common::silu_stable(values[2 * r][c]) * values[2 * r + 1][c];
                // The grouped device kernel stores this plane as BF16 and feeds the down
                // projection from it, so round in that regime or the host fallback is more
                // precise than the path it is standing in for and the two diverge over the
                // layer stack; the token kernel stores FP32 below grouped_first and the plane
                // must stay unrounded there. plane_round tracks which regime the launch is in.
                work.scratch.activation[group.slots[first + c] * intermediate + i + r] =
                    work.plane_round ? common::bf16_round(value) : value;
            }
        }
    }
}

template <class Simd, int DownBits, int Columns>
inline void down(Work& work, const Group& group, int first, int channel,
                 float (&contributions)[max_token_block * top_k][channel_block]) noexcept {
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
                // port (sm_89): the grouped device kernel stores the per-expert down values as
                // a BF16 plane and folds the coefficients at the reduction
                // (grouped_swiglu.cu reduce_assignments); the token kernel keeps FP32. The
                // merge order itself is decided by the caller. plane_round picks the regime so
                // the half-layers stay aligned.
                contributions[group.slots[first + c]][i + r] =
                    work.plane_round ? common::bf16_round(values[r][c]) : values[r][c];
            }
        }
    }
}

template <class Simd, int DownBits, int GateBits, bool FourColumns>
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
                gate_up<Simd, GateBits, 4>(work, group, task.first_column, channel, begin,
                                          bf16_columns);
                continue;
            }
        }
        if (task.columns == 2) {
            gate_up<Simd, GateBits, 2>(work, group, task.first_column, channel, begin,
                                      bf16_columns);
        } else {
            gate_up<Simd, GateBits, 1>(work, group, task.first_column, channel, begin,
                                      bf16_columns);
        }
    }
    worker.barrier();
    // port (sm_89): the merge runs per token in route order, not in expert-group order: the
    // token device kernel sums values[p] for p ascending (token_swiglu.cu) and the grouped
    // reducer walks route 0..top_k (grouped_swiglu.cu reduce_assignments). slot = t * top_k + j
    // is exactly that order, so the down pass collects per-slot values and the merge replays
    // each token by ascending j. Group order differs whenever two tokens share experts in
    // swapped route positions, and fp32 addition is not associative.
    bool present[max_token_block * top_k]       = {};
    float coefficients[max_token_block * top_k] = {};
    for (int g = 0; g < work.group_count; ++g) {
        const auto& group = work.scratch.groups[g];
        for (int member = 0; member < group.count; ++member) {
            present[group.slots[member]]       = true;
            coefficients[group.slots[member]] = group.coefficients[member];
        }
    }
    const auto outputs = worker.range(hidden / work.block);
    for (auto block = outputs.begin; block < outputs.end; ++block) {
        const int channel = block * work.block;
        float accumulated[max_token_block][channel_block]{};
        // Uninitialised on purpose: only slots marked present are written by the down pass
        // below and read by the merge, and every present slot is fully written for this block.
        float contributions[max_token_block * top_k][channel_block];
        if constexpr (FourColumns) {
            for (int index = 0; index < work.task_count; ++index) {
                const auto& task  = work.scratch.tasks[index];
                const auto& group = work.scratch.groups[task.group_index];
                if (task.columns == 4) {
                    down<Simd, DownBits, 4>(work, group, task.first_column, channel,
                                            contributions);
                    continue;
                }
                if (task.columns == 2) {
                    down<Simd, DownBits, 2>(work, group, task.first_column, channel,
                                            contributions);
                } else {
                    down<Simd, DownBits, 1>(work, group, task.first_column, channel,
                                            contributions);
                }
            }
        } else {
            for (int g = 0; g < work.group_count; ++g) {
                const auto& group = work.scratch.groups[g];
                for (int first = 0; first < group.count; first += 2) {
                    if (first + 1 < group.count) {
                        down<Simd, DownBits, 2>(work, group, first, channel, contributions);
                    } else {
                        down<Simd, DownBits, 1>(work, group, first, channel, contributions);
                    }
                }
            }
        }
        for (int t = 0; t < columns; ++t) {
            for (int j = 0; j < top_k; ++j) {
                const int slot = t * top_k + j;
                if (!present[slot]) { continue; }
                for (int i = 0; i < work.block; ++i) {
                    accumulated[t][i] += coefficients[slot] * contributions[slot][i];
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
template <class Simd, int DownBits, int GateBits, bool FourColumns>
[[gnu::noinline]] void compute_wide_chunk(Work& work, const CpuWorkerContext& worker, int begin,
                                          int columns, unsigned bf16_columns) noexcept {
    compute_chunk<Simd, DownBits, GateBits, FourColumns>(work, worker, begin, columns,
                                                         bf16_columns);
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

// port (sm_89): quantise the input row to int16 in the same 64-value groups as quantise_input,
// with the same degenerate semantics (an all-zero group keeps scale zero and codes zero). The
// int16 gate tier pays this once per row to buy 258x finer codes; the peak scan stays scalar so
// NaN and degenerate inputs behave exactly like the int8 tier. cvtepi32 rounds like lrint (both
// take MXCSR), and every x * (1/scale) product stays within half an ulp of 32767, so the
// saturating pack below can never clamp. Natural value order, one store per 16 values.
// Out of line so it cannot change the layout of the vector loop that runs when it is disabled.
template <class Simd>
[[gnu::noinline]] void quantise_input_i16(Work& work, const CpuWorkerContext& worker,
                                          int columns) noexcept {
    constexpr int kGroups = hidden / 64;
    const auto range      = worker.range(kGroups * columns);
    for (auto g = range.begin; g < range.end; ++g) {
        const int column    = static_cast<int>(g) / kGroups;
        const int group     = static_cast<int>(g) % kGroups;
        const float* source = work.scratch.input + column * hidden + group * 64;
        float peak          = 0.0F;
        for (int i = 0; i < 64; ++i) { peak = std::max(peak, std::fabs(source[i])); }
        const float scale   = peak > 0.0F ? peak / 32767.0F : 0.0F;
        const float inverse = scale > 0.0F ? 1.0F / scale : 0.0F;
        work.scratch.input_scale[column * kGroups + group] = scale;
        auto* out         = work.scratch.input_q16 + column * hidden + group * 64;
        const auto factor = _mm256_set1_ps(inverse);
        for (int i = 0; i < 64; i += 16) {
            const auto first  = _mm256_mul_ps(_mm256_loadu_ps(source + i), factor);
            const auto second = _mm256_mul_ps(_mm256_loadu_ps(source + i + 8), factor);
            // packs_epi32 joins two epi32 vectors as [a0..a3 b0..b3 | a4..a7 b4..b7]; the
            // permute restores natural order, 16 codes to one unaligned store.
            const auto packed = _mm256_permute4x64_epi64(
                _mm256_packs_epi32(_mm256_cvtps_epi32(first), _mm256_cvtps_epi32(second)), 0xD8);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), packed);
        }
    }
}

template <class Simd, int DownBits, int GateBits>
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
                const int index = v * Simd::lanes;
                Simd::store(
                    work.scratch.input + index,
                    Simd::bf16(work.args.input + static_cast<std::size_t>(begin) * hidden + index));
            }
        }
        if (begin != 0 || !Simd::bf16_gate) { worker.barrier(); }
        // port (sm_89, P4-B): the quantised activation tiers feed the q4 gate's nibble dot
        // only — a Q8 gate decodes float codes and never reads the plane.
        if constexpr (GateBits == 4) {
            if (work.int8_path) {
                quantise_input<Simd>(work, worker, columns);
                worker.barrier();
            } else if (work.int16_path) {
                // The int16 tier replaces the int8 quantisation pass instead of running beside it.
                quantise_input_i16<Simd>(work, worker, columns);
                worker.barrier();
            }
        }
        // port (sm_89): a split layer routes to a handful of experts, so at 16 channels per unit
        // the team gets 32 units per expert - one or two each, with a barrier on both sides.
        // Halving the unit doubles the units and lets the same team work; with eight experts the
        // wider unit wins instead. Sizes measured with bench/ops/moe_cpu_bench.
        const int units = work.task_count * (intermediate / channel_block);
        work.block = (work.workers > 1 && units < work.workers * 8) ? channel_block / 2
                                                                   : channel_block;
        unsigned bf16_columns = 0;
        if constexpr (Simd::bf16_gate && GateBits == 4) {
            bf16_columns = Simd::safe_columns(
                work.args.input + static_cast<std::size_t>(begin) * hidden, columns);
        }
        if constexpr (Simd::max_columns == 4) {
            if (work.has_four_columns) {
                compute_wide_chunk<Simd, DownBits, GateBits, true>(work, worker, begin, columns,
                                                                    bf16_columns);
            } else {
                compute_wide_chunk<Simd, DownBits, GateBits, false>(work, worker, begin, columns,
                                                                    bf16_columns);
            }
        } else {
            compute_chunk<Simd, DownBits, GateBits, false>(work, worker, begin, columns,
                                                            bf16_columns);
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
                    const Work& work = *static_cast<Work*>(argument);
                    // port (sm_89, P4-B): the (gate_bits, down_bits) pairs the shapes produce —
                    // (4, 5) / (4, 6) for the q4 family, (8, 8) for q8_q8.
                    if (work.plan.gate_bits == 8) {
                        compute<Simd, 8, 8>(argument, worker);
                    } else if (work.plan.down_bits == 6) {
                        compute<Simd, 6, 4>(argument, worker);
                    } else {
                        compute<Simd, 5, 4>(argument, worker);
                    }
                },
            .first_chunk = Simd::first_chunk};
}

} // namespace ninfer::ops::detail::moe_cpu::groupwise_int
