#pragma once

#include "ops/moe/kernels/cpu/invocation.h"

#include <cstdio>
#include <cstdlib>

namespace ninfer::ops::detail::moe_cpu::groupwise_int {

inline constexpr int hidden = 2048, intermediate = 512, experts = 256, top_k = 8;
inline constexpr int max_token_block = 8;
// One ownership unit is a run of output channels. 16 keeps a cache line of FP32 per unit, but it
// leaves only 32 units per expert, so a layer with one or two missing experts cannot fill a wide
// team; the launch halves it when there is not enough work (see `block` in Work).
inline constexpr int channel_block = 16;
// port (sm_89): the grouped CUDA path takes over from the token path at this many tokens
// (routed_swiglu/cuda/q4_q5/shapes/h2048_i512_e256_k8/dispatch.cpp, kGroupedFirst). The token
// path keeps its activation and value planes in FP32 while the grouped path stores BF16, and
// the CPU kernel must mirror whichever regime the launch is in.
inline constexpr int grouped_first = 47;

enum class FirstChunkPreparation { Caller, Worker };

struct Kernel {
    CpuWork run;
    FirstChunkPreparation first_chunk;
};

struct Plan {
    const Kernel* kernel;
    int token_block;
    int down_bits;
    int max_columns;
};

struct Matrix {
    const std::uint8_t* codes;
    const std::uint8_t* high;
    const std::uint16_t* scales;
};

struct MatrixBank {
    Matrix base;
    std::size_t code_stride;
    std::size_t high_stride;
    std::size_t scale_stride;
    // port (sm_89): per-expert offsets replace expert * stride when experts live in
    // separately committed segments.
    const std::uint64_t* offsets = nullptr;
    // port (sm_89): a fused bank keeps its three plane runs far apart inside one object, so each
    // plane carries its own table; when these are set the base pointers already point at the mapped
    // file and the entry is that plane's origin.
    const std::uint64_t* codes_offsets  = nullptr;
    const std::uint64_t* high_offsets   = nullptr;
    const std::uint64_t* scales_offsets = nullptr;

    [[nodiscard]] std::size_t offset_of(const std::uint64_t* table, int expert,
                                        std::size_t stride) const noexcept {
        return table     ? static_cast<std::size_t>(table[expert])
               : offsets ? static_cast<std::size_t>(offsets[expert])
                         : static_cast<std::size_t>(expert) * stride;
    }

    [[nodiscard]] Matrix matrix(int expert) const noexcept {
        return {base.codes ? base.codes + offset_of(codes_offsets, expert, code_stride) : nullptr,
                base.high ? base.high + offset_of(high_offsets, expert, high_stride) : nullptr,
                base.scales ? reinterpret_cast<const std::uint16_t*>(
                                  reinterpret_cast<const std::byte*>(base.scales) +
                                  offset_of(scales_offsets, expert, scale_stride))
                            : nullptr};
    }
};

struct Group {
    Matrix gate_up;
    Matrix down;
    int count;
    int tokens[max_token_block];
    int slots[max_token_block];
    float coefficients[max_token_block];
};

struct Task {
    int group_index;
    int first_column;
    int columns;
};

struct Scratch {
    Group* groups;
    Task* tasks;
    float* input;
    float* activation;
    // port (sm_89): the input row quantised to int8 alongside its per-64-group scale, so the q4 dot
    // product can run on 32 weights per instruction pair instead of decoding eight at a time.
    std::int8_t* input_q8;
    // port (sm_89): the same plane sized for int16 - the int16 gate tier uses it in place while
    // the int8 tier only reads the first half of the bytes. Both tiers share input_scale.
    std::int16_t* input_q16;
    float* input_scale;
    // Integer sum of each group's quantised values, so the offset-by-8 nibble product can undo its
    // bias once per group: sum(b * x) = sum(w * x) + 8 * sum(x).
    std::int32_t* input_sum;
};

struct Work {
    Invocation args;
    Plan plan;
    MatrixBank gate_up;
    MatrixBank down;
    Scratch scratch;
    int group_count       = 0;
    int task_count        = 0;
    // Channels per ownership unit and the team size. Chosen per chunk from the amount of work: a
    // split layer with one or two missing experts otherwise hands each worker a single unit and
    // pays the barrier for nothing.
    int block             = channel_block;
    int workers           = 1;
    // Decided once per launch, not inside the hot function: a getenv and a branch there cost 17%
    // with the path disabled, because it changed how the compiler laid out the vector loop.
    bool int8_path        = false;
    // port (sm_89): the int16 gate tier (NINFER_MOE_CPU_INT8=16) - 258x finer activation codes
    // at the same kernel shape. Same once-per-launch discipline as int8_path.
    bool int16_path       = false;
    bool has_four_columns = false;
    // port (sm_89): mirror of the device plane regime: the grouped path stores the activation
    // and value planes as BF16 from grouped_first tokens up, the token path stores FP32.
    bool plane_round      = false;
};

void prepare_chunk(Work&, int token_begin, int columns) noexcept;
extern const Kernel avx2;
extern const Kernel avx512;
extern const Kernel avx512_bf16;
std::size_t capacity(int tokens, int token_block);
void launch(const Invocation&, const Plan&, MemoryArena&, CpuExecutionView);

} // namespace ninfer::ops::detail::moe_cpu::groupwise_int
