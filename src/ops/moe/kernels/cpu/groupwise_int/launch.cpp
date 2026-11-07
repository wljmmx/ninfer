#include "ops/moe/kernels/cpu/groupwise_int/work.h"
#include "core/layout.h"

#include <algorithm>
#include <array>
#include <cstring>

namespace ninfer::ops::detail::moe_cpu::groupwise_int {
namespace {
template <class Arena>
Scratch allocate(Arena& arena, int tokens, int token_block) {
    const int columns = std::min(token_block, tokens);
    // One weight scale covers 64 values, so one activation scale does too.
    const int groups = hidden / 64;
    // port (sm_89): the quantised activation plane is sized for int16 (two bytes per value) so
    // the int16 gate tier can use it in place; the int8 tier only touches the first half of the
    // bytes. Both plane sizes (hidden * columns * {1, 2}) are multiples of the arena's 256-byte
    // alignment, so the following allocations keep their alignment unchanged. The scales stay
    // fp32 per 64-value group and are shared by both tiers.
    auto* quantised = static_cast<std::int16_t*>(
        arena.alloc_bytes(sizeof(std::int16_t) * hidden * columns).data);
    return {static_cast<Group*>(arena.alloc_bytes(sizeof(Group) * top_k * columns).data),
            static_cast<Task*>(arena.alloc_bytes(sizeof(Task) * top_k * columns).data),
            static_cast<float*>(arena.alloc(DType::FP32, {hidden, columns}).data),
            static_cast<float*>(arena.alloc(DType::FP32, {intermediate, top_k * columns}).data),
            reinterpret_cast<std::int8_t*>(quantised),
            quantised,
            static_cast<float*>(arena.alloc_bytes(sizeof(float) * groups * columns).data),
            static_cast<std::int32_t*>(
                arena.alloc_bytes(sizeof(std::int32_t) * groups * columns).data)};
}

MatrixBank matrix_bank(const WeightBankView& bank) {
    const auto& layout = bank.layout;
    // port (sm_89): bases are the plane origins (already rebased by the host tier); the
    // per-expert table is added in matrix().
    const auto* data = bank.parent->data;
    return {{layout.codes.expert_bytes
                 ? reinterpret_cast<const std::uint8_t*>(data + layout.codes.offset)
                 : nullptr,
             layout.high.expert_bytes
                 ? reinterpret_cast<const std::uint8_t*>(data + layout.high.offset)
                 : nullptr,
             layout.scales.expert_bytes
                 ? reinterpret_cast<const std::uint16_t*>(data + layout.scales.offset)
                 : nullptr},
            layout.codes.expert_stride,
            layout.high.expert_stride,
            layout.scales.expert_stride,
            bank.expert_offsets,
            bank.codes_offsets,
            bank.high_offsets,
            bank.scales_offsets};
}
} // namespace

std::size_t capacity(int tokens, int token_block) {
    WorkspaceLayoutBuilder layout;
    (void)allocate(layout, tokens, token_block);
    return layout.peak_bytes(1);
}

void launch(const Invocation& args, const Plan& plan, MemoryArena& arena,
            CpuExecutionView execution) {
    Work work{.args    = args,
              .plan    = plan,
              .gate_up = matrix_bank(args.experts.gate_up),
              .down    = matrix_bank(args.experts.down),
              .scratch = allocate(arena, args.tokens, plan.token_block)};
    work.workers = static_cast<int>(execution.worker_count());
    // port (sm_89): the int16 row path is the default now. It keeps the int8 kernel's shape but
    // decodes the whole 64-value group in one pass (bias-flip + interleaved nibble planes +
    // signed widening, compute.h dot_q4_q16) and reads the activation plane through fused madd
    // memory operands, which took 17 instructions off each row's weight preparation and moved
    // the totals out of the stack. NINFER_MOE_CPU_INT8=8 selects the int8 tier, =0 keeps the
    // float reference for the gate that compares both.
    // 2026-11-07 A/B on the 4090 host (i9-14900K/AVX2, moe_cpu_bench misses=1, min of 400 with
    // a 96 MiB flush before every call, int8/int16 runs interleaved 5x in one session): int16
    // 0.0505 ms/expert vs int8 0.0491 at the shipped one-token split shape - 97.2% of the int8
    // speed, past the 97% bar - and 100.2% at the two-column batch shape with 98.1% at the
    // four-column one, both over the 95% bar, so the default flipped from int8 (which measured
    // 96.0%/94.1% on 2026-11-07 8d07afe3) to int16. The 258x finer codes come with the
    // package: 1.46e-5 vs 3.58e-3 relative against the float reference, still opt-out-able.
    // unset or "16" -> int16; "0" -> float; anything else ("8", "1", ...) -> int8.
    const char* int8_switch = std::getenv("NINFER_MOE_CPU_INT8");
    work.int16_path = int8_switch == nullptr || std::strcmp(int8_switch, "16") == 0;
    work.int8_path  = !work.int16_path && std::strcmp(int8_switch, "0") != 0;
    // port (sm_89): mirror the device plane regime - grouped stores BF16 planes from
    // grouped_first tokens up (see work.h), token keeps FP32.
    work.plane_round = args.tokens >= grouped_first;
    if (plan.kernel->first_chunk == FirstChunkPreparation::Caller) {
        prepare_chunk(work, 0, std::min(args.tokens, plan.token_block));
    }
    execution.run(plan.kernel->run, &work);
}

void prepare_chunk(Work& work, int token_begin, int columns) noexcept {
    std::array<int, experts> indexes;
    indexes.fill(-1);
    work.group_count = work.task_count = 0;
    work.has_four_columns              = false;
    for (int t = 0; t < columns; ++t) {
        const auto offset = static_cast<std::size_t>(token_begin + t) * top_k;
        for (int j = 0; j < top_k; ++j) {
            const int expert = work.args.ids[offset + j];
            if (!participates(work.args.subset, work.args.mask, expert)) { continue; }
            auto& index = indexes[expert];
            if (index < 0) {
                index         = work.group_count++;
                auto& group   = work.scratch.groups[index];
                group.gate_up = work.gate_up.matrix(expert);
                group.down    = work.down.matrix(expert);
                group.count   = 0;
            }
            auto& group                = work.scratch.groups[index];
            const int member           = group.count++;
            group.tokens[member]       = t;
            group.slots[member]        = t * top_k + j;
            group.coefficients[member] = work.args.coefficients[offset + j];
        }
    }
    for (int g = 0; g < work.group_count; ++g) {
        for (int first = 0; first < work.scratch.groups[g].count;) {
            const int remaining = work.scratch.groups[g].count - first;
            const int tile      = remaining >= 4 ? 4 : remaining >= 2 ? 2 : 1;
            const int count     = std::min(work.plan.max_columns, tile);

            work.scratch.tasks[work.task_count++] = {
                .group_index = g, .first_column = first, .columns = count};
            work.has_four_columns |= count == 4;
            first += count;
        }
    }
}

} // namespace ninfer::ops::detail::moe_cpu::groupwise_int
