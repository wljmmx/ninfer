// Stage M2 follow-up: the projection dispatch of an artifact that keeps one parent per matrix.
//
// qwen3_5's 27B/35B artifacts store query, key, the output gate and value in one fused parent (or
// as the Q4/Q5 pair), and the compiled AttnInputProj/GdnInputProj forms decode exactly those
// shapes. Qwen4Exp stores each projection in its own object - and, for GDN, q and k as two row
// blocks of one in_proj parent - so the fused geometry is absent and every projection has to run
// through Linear on its own. These checks read the prepared weights, never a device: no kernel runs
// and no CUDA context is created.

#include "core/weight_view.h"
#include "ninfer/ops/weight_input.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;

constexpr std::uint64_t kHidden = 2560;

int failures = 0;

void check(bool condition, const std::string& label) {
    if (condition) { return; }
    std::cerr << label << ": failed\n";
    ++failures;
}

// A resident parent of one encoded matrix. The checks here only follow addresses derivations, so a
// synthetic base pointer is enough.
struct Parent {
    WeightParent value;

    Parent(std::uint64_t rows, std::uint64_t columns, QType format, QuantLayout layout) {
        value.geometry = weight_geometry(format, layout, std::vector<std::uint64_t>{rows, columns});
        value.data     = reinterpret_cast<const std::byte*>(std::uintptr_t{0x100000});
    }
};

// The complete matrix a per-projection binding declares.
WeightView whole(const Parent& parent, std::uint64_t rows, std::uint64_t columns) {
    WeightView view;
    view.shape = {rows, columns};
    view.parts.push_back({&parent.value, 0, rows * columns});
    return view;
}

// A row block of a shared parent, as a fused parent splits into its projections.
WeightView rows_of(const Parent& parent, std::uint64_t columns, std::uint64_t first_row,
                   std::uint64_t rows) {
    WeightView view;
    view.shape = {rows, columns};
    const auto begin = first_row * columns;
    view.parts.push_back({&parent.value, begin, begin + rows * columns});
    return view;
}

const Weight& weight_of(const ops::SingleProjectionWeight& row) { return row.weight; }

// Qwen4Exp attention: q 6144, k 512, gate 6144, v 512 over hidden 2560, each its own parent.
void check_qwen4_exp_attention() {
    const Parent query(6144, kHidden, QType::Q4_G64_FP16, QuantLayout::RowSplit);
    const Parent key(512, kHidden, QType::Q4_G64_FP16, QuantLayout::RowSplit);
    const Parent gate(6144, kHidden, QType::Q4_G64_FP16, QuantLayout::RowSplit);
    const Parent value(512, kHidden, QType::Q4_G64_FP16, QuantLayout::RowSplit);
    const auto q = whole(query, 6144, kHidden);
    const auto k = whole(key, 512, kHidden);
    const auto g = whole(gate, 6144, kHidden);
    const auto v = whole(value, 512, kHidden);

    const auto prepared = ops::prepare_attn_input_proj_weights({q}, {k}, {g}, {v});
    const auto* separate = std::get_if<ops::SeparateProjectionWeights>(&prepared);
    check(separate != nullptr, "four attention parents dispatch per projection");
    if (separate == nullptr) { return; }

    const auto& rows = separate->rows;
    const auto kHidden32 = static_cast<std::int32_t>(kHidden);
    check(weight_of(rows[0]).n == 6144 && weight_of(rows[0]).k == kHidden32,
          "the query projection keeps its own [6144,2560]");
    check(weight_of(rows[1]).n == 512 && weight_of(rows[1]).k == kHidden32,
          "the key projection keeps its own [512,2560]");
    check(weight_of(rows[2]).n == 6144 && weight_of(rows[3]).n == 512,
          "the gate and value projections keep their own row counts");
    check(weight_of(rows[0]).qtype == QType::Q4_G64_FP16 &&
              weight_of(rows[0]).layout == QuantLayout::RowSplit,
          "each projection keeps the artifact's format");
}

// Qwen4Exp GDN: q and k are two blocks of one in_proj parent, while v and z are their own.
void check_qwen4_exp_gdn() {
    const Parent in_proj(4096, kHidden, QType::Q4_G64_FP16, QuantLayout::RowSplit);
    const Parent value(6144, kHidden, QType::Q4_G64_FP16, QuantLayout::RowSplit);
    const Parent z(6144, kHidden, QType::Q4_G64_FP16, QuantLayout::RowSplit);
    const auto q = rows_of(in_proj, kHidden, 0, 2048);
    const auto k = rows_of(in_proj, kHidden, 2048, 2048);
    const auto v = whole(value, 6144, kHidden);
    const auto zw = whole(z, 6144, kHidden);

    const auto prepared = ops::prepare_gdn_input_proj_weights({q}, {k}, {v}, {zw});
    const auto* separate = std::get_if<ops::SeparateProjectionWeights>(&prepared);
    check(separate != nullptr, "a shared q/k parent and two own parents dispatch per projection");
    if (separate == nullptr) { return; }

    const auto& rows = separate->rows;
    check(weight_of(rows[0]).n == 2048 && weight_of(rows[1]).n == 2048,
          "q and k are the two 2048-row blocks of in_proj");
    check(weight_of(rows[2]).n == 6144 && weight_of(rows[3]).n == 6144,
          "v and z keep their own 6144 rows");
    check(static_cast<const std::byte*>(weight_of(rows[0]).qdata) !=
                  static_cast<const std::byte*>(weight_of(rows[1]).qdata) &&
              rows[0].weight.qdata != nullptr,
          "the two q/k blocks start at different addresses");
}

// The output projection is an ordinary per-projection matrix: [2560,6144].
void check_qwen4_exp_output() {
    const Parent output(2560, 6144, QType::Q4_G64_FP16, QuantLayout::RowSplit);
    const auto prepared = ops::prepare_linear_weight({whole(output, 2560, 6144)});
    check(prepared.weight.n == 2560 && prepared.weight.k == 6144,
          "o_proj is one [2560,6144] Linear problem");
}

// The 35B fused attention parent keeps taking the compiled fused route: the four projections are
// consecutive rows of one [14336,5120] parent.
void check_fused_35b_stays_fused() {
    constexpr std::uint64_t kFusedHidden = 5120;
    constexpr std::uint64_t kQRows       = 6144;
    constexpr std::uint64_t kKvRows      = 1024;
    const Parent fused(2 * kQRows + 2 * kKvRows, kFusedHidden, QType::BF16,
                       QuantLayout::Contiguous);
    const auto q = rows_of(fused, kFusedHidden, 0, kQRows);
    const auto k = rows_of(fused, kFusedHidden, kQRows, kKvRows);
    const auto g = rows_of(fused, kFusedHidden, kQRows + kKvRows, kQRows);
    const auto v = rows_of(fused, kFusedHidden, 2 * kQRows + kKvRows, kKvRows);

    const auto prepared = ops::prepare_attn_input_proj_weights({q}, {k}, {g}, {v});
    const auto* single = std::get_if<ops::SingleProjectionWeight>(&prepared);
    check(single != nullptr, "the 35B fused parent still reports the fused path");
    if (single == nullptr) { return; }
    check(single->weight.n == 14336 &&
              single->weight.k == static_cast<std::int32_t>(kFusedHidden),
          "the fused route keeps the whole [14336,5120] parent");
}

// The 27B paired form is unchanged too: Q4 query/key and Q5 gate/value parents.
void check_paired_27b_stays_paired() {
    constexpr std::uint64_t kFusedHidden = 5120;
    constexpr std::uint64_t kQRows       = 6144;
    constexpr std::uint64_t kKvRows      = 1024;
    const Parent query_key(7168, kFusedHidden, QType::Q4_G64_FP16, QuantLayout::RowSplit);
    const Parent gate_value(7168, kFusedHidden, QType::Q5_G64_FP16, QuantLayout::RowSplit);
    const auto q = rows_of(query_key, kFusedHidden, 0, kQRows);
    const auto k = rows_of(query_key, kFusedHidden, kQRows, kKvRows);
    const auto g = rows_of(gate_value, kFusedHidden, 0, kQRows);
    const auto v = rows_of(gate_value, kFusedHidden, kQRows, kKvRows);

    const auto prepared = ops::prepare_attn_input_proj_weights({q}, {k}, {g}, {v});
    const auto* pair = std::get_if<ops::PairedProjectionWeights>(&prepared);
    check(pair != nullptr, "the 27B Q4/Q5 pair still reports the paired path");
    if (pair == nullptr) { return; }
    check(pair->first.qtype == QType::Q4_G64_FP16 && pair->second.qtype == QType::Q5_G64_FP16 &&
              pair->first.n == 7168 && pair->second.n == 7168,
          "the paired route keeps its two [7168,5120] parents");
}

} // namespace

int main() {
    try {
        check_qwen4_exp_attention();
        check_qwen4_exp_gdn();
        check_qwen4_exp_output();
        check_fused_35b_stays_fused();
        check_paired_27b_stays_paired();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    if (failures) { return 1; }
    std::cout << "OK per-projection dispatch: Qwen4Exp parents, the 35B fused route and the 27B "
                 "paired route\n";
    return 0;
}
