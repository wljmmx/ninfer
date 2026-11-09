// Stage M2 of the Qwen4Exp port: the routed expert banks of both artifact forms.
//
// qwen3_5 keeps gate and up adjacent inside one parent, so expert e owns rows [e*2I, (e+1)*2I) of
// that matrix. Qwen4Exp keeps three independent banks, so expert e owns rows [e*I, (e+1)*I) of
// gate, the same block of up, and rows [e*H, (e+1)*H) of down. The description is the only way a
// caller addresses one expert, so these checks are about which rows it names - no kernel runs and
// no device is touched.

#include "core/weight_view.h"
#include "models/qwen3_5/execution/parameters.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace qwen = ninfer::models::qwen3_5::execution;

// The artifact's own geometry.
constexpr std::uint32_t kHidden      = 2560;
constexpr std::uint32_t kExpertWidth = 640;
constexpr std::uint32_t kExperts     = 512;

constexpr std::string_view kParameter = "text/layers/3/moe/experts/0/gate";

int failures = 0;

void check(bool condition, const std::string& label) {
    if (condition) { return; }
    std::cerr << label << ": failed\n";
    ++failures;
}

// A pointer-free parent of `rows` encoded rows: every check here reads the description, never the
// bytes.
struct Parent {
    WeightParent value;

    Parent(std::uint64_t rows, std::uint64_t columns, QType format, QuantLayout layout) {
        value.geometry = weight_geometry(format, layout, std::vector<std::uint64_t>{rows, columns});
        value.data     = reinterpret_cast<const std::byte*>(std::uintptr_t{0x1000});
    }
};

// The row block `rows` experts deep into a bank, as a per-expert binding would declare it.
WeightView block_of(const Parent& bank, std::uint64_t rows, std::uint64_t columns,
                    std::uint64_t first_row) {
    WeightView view;
    view.shape        = {rows, columns};
    const auto begin  = first_row * columns;
    view.parts.push_back({&bank.value, begin, begin + rows * columns});
    return view;
}

std::string refusal(const auto& call) {
    try {
        call();
    } catch (const std::exception& error) { return error.what(); }
    return {};
}

// Qwen4Exp: three independent banks, one row block per expert.
void check_separate_banks() {
    const Parent gate(std::uint64_t(kExperts) * kExpertWidth, kHidden, QType::Q4_G64_FP16,
                      QuantLayout::RowSplit);
    const Parent up(std::uint64_t(kExperts) * kExpertWidth, kHidden, QType::Q4_G64_FP16,
                    QuantLayout::RowSplit);
    const Parent down(std::uint64_t(kExperts) * kHidden, kExpertWidth, QType::Q6_G64_FP16,
                      QuantLayout::RowSplit);
    const auto banks = qwen::ExpertBankDescription::from_bindings(
        block_of(gate, kExpertWidth, kHidden, 0), block_of(up, kExpertWidth, kHidden, 0),
        block_of(down, kHidden, kExpertWidth, 0), kExperts, kExpertWidth, std::string(kParameter));

    check(!banks.fused_form() && banks.whole_bank() && banks.count() == kExperts,
          "three separate banks are described as separate and whole");
    check(banks.rows(qwen::ExpertBankRole::Gate) == kExpertWidth &&
              banks.rows(qwen::ExpertBankRole::Up) == kExpertWidth &&
              banks.rows(qwen::ExpertBankRole::Down) == kHidden &&
              banks.rows(qwen::ExpertBankRole::GateUp) == 2 * kExpertWidth,
          "each bank's per-expert rows follow the form");

    const auto& gate_bank = banks.bank(qwen::ExpertBankRole::Gate);
    check(gate_bank.parent == &gate.value &&
              gate_bank.layout.region ==
                  WeightBankRegion{0, kExperts, kExpertWidth, kExpertWidth} &&
              gate_bank.layout.columns == kHidden,
          "the gate bank holds E blocks of I rows by H columns");
    const auto& up_bank = banks.bank(qwen::ExpertBankRole::Up);
    check(up_bank.parent == &up.value &&
              up_bank.layout.region ==
                  WeightBankRegion{0, kExperts, kExpertWidth, kExpertWidth},
          "the up bank is its own parent, not the tail of gate");
    const auto& down_bank = banks.bank(qwen::ExpertBankRole::Down);
    check(down_bank.parent == &down.value &&
              down_bank.layout.region == WeightBankRegion{0, kExperts, kHidden, kHidden} &&
              down_bank.layout.columns == kExpertWidth,
          "the down bank holds E blocks of H rows by I columns");

    const auto gate_first  = banks.block(qwen::ExpertBankRole::Gate, 0);
    const auto gate_last   = banks.block(qwen::ExpertBankRole::Gate, kExperts - 1);
    const auto up_last     = banks.block(qwen::ExpertBankRole::Up, kExperts - 1);
    const auto down_first  = banks.block(qwen::ExpertBankRole::Down, 0);
    const auto down_last   = banks.block(qwen::ExpertBankRole::Down, kExperts - 1);
    check(gate_first.row_begin == 0 && gate_first.count == 1 &&
              gate_first.rows == kExpertWidth && gate_first.row_stride == kExpertWidth,
          "expert 0 of gate is the bank's first row block");
    check(gate_last.row_begin == std::uint64_t(kExperts - 1) * kExpertWidth,
          "expert 511 of gate starts at 511*I");
    check(up_last.row_begin == std::uint64_t(kExperts - 1) * kExpertWidth,
          "expert 511 of up starts at 511*I");
    check(down_first.row_begin == 0 && down_first.rows == kHidden,
          "expert 0 of down is the bank's first row block");
    check(down_last.row_begin == std::uint64_t(kExperts - 1) * kHidden &&
              down_last.rows == kHidden && down_last.row_stride == kHidden,
          "expert 511 of down starts at 511*H");
}

// The qwen3_5 form must describe exactly what it described before: one fused parent, 2I rows per
// expert, gate first and up second.
void check_fused_bank_unchanged() {
    const Parent gate_up(std::uint64_t(kExperts) * 2 * kExpertWidth, kHidden, QType::Q4_G64_FP16,
                         QuantLayout::RowSplit);
    const Parent down(std::uint64_t(kExperts) * kHidden, kExpertWidth, QType::Q6_G64_FP16,
                      QuantLayout::RowSplit);
    const auto banks = qwen::ExpertBankDescription::from_bindings(
        block_of(gate_up, kExpertWidth, kHidden, 0),
        block_of(gate_up, kExpertWidth, kHidden, kExpertWidth),
        block_of(down, kHidden, kExpertWidth, 0), kExperts, kExpertWidth, std::string(kParameter));

    check(banks.fused_form() && banks.whole_bank(), "gate and up in one parent describe the fused form");
    const auto& fused = banks.bank(qwen::ExpertBankRole::GateUp);
    check(fused.parent == &gate_up.value &&
              fused.layout.region ==
                  WeightBankRegion{0, kExperts, 2 * kExpertWidth, 2 * kExpertWidth} &&
              fused.layout.columns == kHidden,
          "the fused gate_up bank holds E blocks of 2I rows");
    check(banks.block(qwen::ExpertBankRole::GateUp, 1).row_begin == 2 * std::uint64_t(kExpertWidth),
          "expert 1 of the fused bank starts at 2I");
    check(banks.block(qwen::ExpertBankRole::Gate, 1).row_begin == 2 * std::uint64_t(kExpertWidth),
          "expert 1's gate rows start the block");
    check(banks.block(qwen::ExpertBankRole::Up, 1).row_begin ==
              2 * std::uint64_t(kExpertWidth) + kExpertWidth,
          "expert 1's up rows follow its gate rows");
    check(banks.block(qwen::ExpertBankRole::Down, 1).row_begin == std::uint64_t(kHidden),
          "expert 1 of down starts at H");

    // A per-expert artifact (one object per expert) is the same form without a bank-wide parent.
    const Parent per_expert(2 * std::uint64_t(kExpertWidth), kHidden, QType::Q4_G64_FP16,
                            QuantLayout::RowSplit);
    const Parent per_expert_down(kHidden, kExpertWidth, QType::Q6_G64_FP16,
                                 QuantLayout::RowSplit);
    const auto single = qwen::ExpertBankDescription::from_bindings(
        block_of(per_expert, kExpertWidth, kHidden, 0),
        block_of(per_expert, kExpertWidth, kHidden, kExpertWidth),
        block_of(per_expert_down, kHidden, kExpertWidth, 0), kExperts, kExpertWidth,
        std::string(kParameter));
    check(single.fused_form() && !single.whole_bank(),
          "one object per expert describes the fused rows without a bank-wide parent");
}

// A lookup the description cannot answer names the parameter instead of returning another expert's
// rows.
void check_mismatched_lookups_are_refused() {
    const Parent gate(std::uint64_t(kExperts) * kExpertWidth, kHidden, QType::Q4_G64_FP16,
                      QuantLayout::RowSplit);
    const Parent up(std::uint64_t(kExperts) * kExpertWidth, kHidden, QType::Q4_G64_FP16,
                    QuantLayout::RowSplit);
    const Parent down(std::uint64_t(kExperts) * kHidden, kExpertWidth, QType::Q6_G64_FP16,
                      QuantLayout::RowSplit);
    const auto separate = qwen::ExpertBankDescription::from_bindings(
        block_of(gate, kExpertWidth, kHidden, 0), block_of(up, kExpertWidth, kHidden, 0),
        block_of(down, kHidden, kExpertWidth, 0), kExperts, kExpertWidth, std::string(kParameter));

    const auto fused_lookup = refusal([&] {
        (void)separate.bank(qwen::ExpertBankRole::GateUp);
    });
    check(fused_lookup.find("text/layers/3/moe/experts/0/gate") != std::string::npos &&
              fused_lookup.find("separate") != std::string::npos,
          "asking a separate bank for the fused gate_up is refused by parameter name");

    const auto outside = refusal([&] {
        (void)separate.block(qwen::ExpertBankRole::Gate, kExperts);
    });
    check(outside.find("text/layers/3/moe/experts/0/gate") != std::string::npos,
          "an expert outside the bank is refused by parameter name");

    // The dangerous case: gate and up in their own parents but each holding E*2I rows. Read as
    // separate banks, expert e would be handed the rows of experts 2e and 2e+1.
    const Parent wide_gate(std::uint64_t(kExperts) * 2 * kExpertWidth, kHidden, QType::Q4_G64_FP16,
                           QuantLayout::RowSplit);
    const Parent wide_up(std::uint64_t(kExperts) * 2 * kExpertWidth, kHidden, QType::Q4_G64_FP16,
                         QuantLayout::RowSplit);
    const auto fused_as_separate = refusal([&] {
        (void)qwen::ExpertBankDescription::from_bindings(
            block_of(wide_gate, kExpertWidth, kHidden, 0),
            block_of(wide_up, kExpertWidth, kHidden, 0),
            block_of(down, kHidden, kExpertWidth, 0), kExperts, kExpertWidth,
            std::string(kParameter));
    });
    check(fused_as_separate.find("text/layers/3/moe/experts/0/gate") != std::string::npos,
          "a gate bank holding 2I rows per expert is refused as a separate one by parameter name");

    // A bank whose rows match neither form is refused at description time.
    const Parent odd(std::uint64_t(kExperts) * kExpertWidth + 64, kHidden, QType::Q4_G64_FP16,
                     QuantLayout::RowSplit);
    const auto neither = refusal([&] {
        (void)qwen::ExpertBankDescription::from_bindings(
            block_of(odd, kExpertWidth, kHidden, 0), block_of(up, kExpertWidth, kHidden, 0),
            block_of(down, kHidden, kExpertWidth, 0), kExperts, kExpertWidth,
            std::string(kParameter));
    });
    check(neither.find("text/layers/3/moe/experts/0/gate") != std::string::npos,
          "a bank whose rows match neither form is refused by parameter name");
}

// The fused accessors of the routed parameters refuse the split description rather than returning
// the empty fused members.
void check_split_members_are_refused() {
    const Parent gate(std::uint64_t(kExperts) * kExpertWidth, kHidden, QType::Q4_G64_FP16,
                      QuantLayout::RowSplit);
    const Parent up(std::uint64_t(kExperts) * kExpertWidth, kHidden, QType::Q4_G64_FP16,
                    QuantLayout::RowSplit);
    const Parent down(std::uint64_t(kExperts) * kHidden, kExpertWidth, QType::Q6_G64_FP16,
                      QuantLayout::RowSplit);
    qwen::MoeSplitParameters split;
    split.parameter = std::string(kParameter);
    split.experts   = qwen::ExpertBankDescription::from_bindings(
        block_of(gate, kExpertWidth, kHidden, 0), block_of(up, kExpertWidth, kHidden, 0),
        block_of(down, kHidden, kExpertWidth, 0), kExperts, kExpertWidth, std::string(kParameter));

    qwen::RoutedMoeParameters moe;
    moe.split = split;
    check(refusal([&] { (void)moe.fused_control(); }).find(split.parameter) != std::string::npos,
          "the fused control rows are refused by parameter name");
    check(refusal([&] { (void)moe.fused_shared(); }).find(split.parameter) != std::string::npos,
          "the fused shared expert is refused by parameter name");
    check(!moe.stream && !moe.split->experts.fused_form(),
          "a split layer carries no stream and three banks");
}

} // namespace

int main() {
    try {
        check_separate_banks();
        check_fused_bank_unchanged();
        check_mismatched_lookups_are_refused();
        check_split_members_are_refused();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    if (failures) { return 1; }
    std::cout << "OK Qwen4Exp three-bank description, qwen3_5 fused bank and refused lookups\n";
    return 0;
}
