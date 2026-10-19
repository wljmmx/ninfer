#pragma once

#include "ops/moe/kernels/cpu/invocation.h"

namespace ninfer::ops::detail::moe_cpu::groupwise_int {

inline constexpr int hidden = 2048, intermediate = 512, experts = 256, top_k = 8;
inline constexpr int max_token_block = 8;

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

    [[nodiscard]] std::size_t offset(int expert, std::size_t stride) const noexcept {
        return offsets ? static_cast<std::size_t>(offsets[expert])
                       : static_cast<std::size_t>(expert) * stride;
    }

    [[nodiscard]] Matrix matrix(int expert) const noexcept {
        return {base.codes ? base.codes + offset(expert, code_stride) : nullptr,
                base.high ? base.high + offset(expert, high_stride) : nullptr,
                base.scales ? reinterpret_cast<const std::uint16_t*>(
                                  reinterpret_cast<const std::byte*>(base.scales) +
                                  offset(expert, scale_stride))
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
};

struct Work {
    Invocation args;
    Plan plan;
    MatrixBank gate_up;
    MatrixBank down;
    Scratch scratch;
    int group_count       = 0;
    int task_count        = 0;
    bool has_four_columns = false;
};

void prepare_chunk(Work&, int token_begin, int columns) noexcept;
extern const Kernel avx2;
extern const Kernel avx512;
extern const Kernel avx512_bf16;
std::size_t capacity(int tokens, int token_block);
void launch(const Invocation&, const Plan&, MemoryArena&, CpuExecutionView);

} // namespace ninfer::ops::detail::moe_cpu::groupwise_int
