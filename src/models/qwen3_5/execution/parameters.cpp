#include "models/qwen3_5/execution/parameters.h"

#include "models/qwen3_5/execution/expert_stream.h"

#include "core/weight_view.h"

#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen3_5::execution {
namespace {
template <class Function>
auto with_context(const std::string& context, Function&& function) {
    try {
        return function();
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(context + ": " + error.what());
    }
}

[[noreturn]] void refuse(const std::string& parameter, const std::string& detail) {
    throw std::invalid_argument(parameter + ": " + detail);
}

// Rows of one parent that holds `columns`-wide matrices, refusing anything that is not a matrix
// bank this description can address.
std::uint64_t parent_rows(const WeightParent& parent, std::uint64_t columns,
                          const std::string& parameter, const char* role) {
    const auto& shape = parent.geometry.shape;
    if (shape.size() != 2 || !columns || shape[1] != columns) {
        refuse(parameter, std::string(role) + " bank is not a " + std::to_string(columns) +
                              "-column matrix");
    }
    return shape[0];
}

// One bank-wide view over `parent`, validated against the description it claims.
WeightBankView bank_view(const WeightParent& parent, const WeightBankRegion& region,
                         const std::string& parameter) {
    WeightBankView view;
    view.parent = &parent;
    try {
        view.layout = describe_weight_bank(parent.geometry, region);
    } catch (const std::invalid_argument& error) {
        refuse(parameter, std::string("expert bank rows do not match the description: ") +
                              error.what());
    }
    return view;
}

const WeightRegion& single_part(const WeightView& view, std::uint64_t columns,
                                const std::string& parameter, const char* role) {
    if (view.parts.size() != 1 || !view.parts[0].parent) {
        refuse(parameter, std::string(role) + " bank is not one contiguous row block");
    }
    const auto& part = view.parts[0];
    if (part.begin % columns || part.end % columns) {
        refuse(parameter, std::string(role) + " bank splits a row");
    }
    return part;
}

void require_shape(const WeightView& view, std::uint64_t rows, std::uint64_t columns,
                   const std::string& parameter, const char* role) {
    if (view.shape != std::vector<std::uint64_t>{rows, columns}) {
        refuse(parameter, std::string(role) + " bank is [" + std::to_string(rows) + "," +
                              std::to_string(columns) + "] per expert, not what the config declares");
    }
}

} // namespace

std::uint64_t ExpertBankDescription::rows(ExpertBankRole role) const {
    switch (role) {
    case ExpertBankRole::GateUp: return 2 * intermediate_;
    case ExpertBankRole::Gate:
    case ExpertBankRole::Up: return intermediate_;
    case ExpertBankRole::Down: return hidden_;
    }
    refuse(parameter_, "unknown expert bank role");
}

const WeightBankLayout& ExpertBankDescription::layout(ExpertBankRole role) const {
    const std::optional<WeightBankView>* view = nullptr;
    switch (role) {
    case ExpertBankRole::GateUp: view = &gate_up_; break;
    case ExpertBankRole::Gate: view = &gate_; break;
    case ExpertBankRole::Up: view = &up_; break;
    case ExpertBankRole::Down: view = &down_; break;
    }
    if (view == nullptr || !view->has_value()) {
        refuse(parameter_, fused_ ? "the artifact stores gate and up as one fused bank, not as " +
                                        std::string(role == ExpertBankRole::Down ? "down" : "gate/up") +
                                        " alone"
                                  : "the artifact stores gate, up and down as separate banks");
    }
    return (*view)->layout;
}

const WeightBankView& ExpertBankDescription::bank(ExpertBankRole role) const {
    (void)layout(role);
    if (!whole_bank_) {
        refuse(parameter_, "the artifact holds one expert per object, so this bank has no "
                           "bank-wide parent");
    }
    switch (role) {
    case ExpertBankRole::GateUp: return *gate_up_;
    case ExpertBankRole::Gate: return *gate_;
    case ExpertBankRole::Up: return *up_;
    case ExpertBankRole::Down: return *down_;
    }
    refuse(parameter_, "unknown expert bank role");
}

WeightBankRegion ExpertBankDescription::block(ExpertBankRole role, std::uint32_t expert) const {
    if (expert >= count_) {
        refuse(parameter_, "expert " + std::to_string(expert) + " is outside a bank of " +
                               std::to_string(count_));
    }
    const auto index = std::uint64_t(expert);
    // A fused parent holds gate and up back to back, so each role keeps its own half of the block.
    switch (role) {
    case ExpertBankRole::GateUp:
        if (!fused_) { (void)layout(role); }
        return {gate_up_begin_ + index * 2 * intermediate_, 1, 2 * intermediate_,
                2 * intermediate_};
    case ExpertBankRole::Gate:
        return {fused_ ? gate_up_begin_ + index * 2 * intermediate_
                       : gate_begin_ + index * intermediate_,
                1, intermediate_, intermediate_};
    case ExpertBankRole::Up:
        return {fused_ ? gate_up_begin_ + index * 2 * intermediate_ + intermediate_
                       : up_begin_ + index * intermediate_,
                1, intermediate_, intermediate_};
    case ExpertBankRole::Down:
        return {down_begin_ + index * hidden_, 1, hidden_, hidden_};
    }
    refuse(parameter_, "unknown expert bank role");
}

ExpertBankDescription ExpertBankDescription::fused(const WeightView& gate, const WeightView& up,
                                                   const WeightView& down, std::uint32_t count,
                                                   std::uint64_t intermediate,
                                                   std::string parameter) {
    if (!count || !intermediate) { refuse(parameter, "a fused expert bank needs E and I"); }
    const auto hidden = gate.shape.size() == 2 ? gate.shape[1] : 0;
    if (!hidden) { refuse(parameter, "fused gate bank is not a matrix"); }
    require_shape(gate, intermediate, hidden, parameter, "gate");
    require_shape(up, intermediate, hidden, parameter, "up");
    require_shape(down, hidden, intermediate, parameter, "down");
    const auto& gate_part = single_part(gate, hidden, parameter, "gate");
    const auto& up_part   = single_part(up, hidden, parameter, "up");
    const auto& down_part = single_part(down, intermediate, parameter, "down");
    if (gate_part.parent != up_part.parent) {
        refuse(parameter, "the fused form keeps gate and up in one parent");
    }
    const auto& parent   = *gate_part.parent;
    const auto gate_begin = gate_part.begin / hidden;
    const auto rows      = parent_rows(parent, hidden, parameter, "gate_up");
    const auto per       = 2 * intermediate;
    if (gate_begin >= rows) { refuse(parameter, "gate starts past its parent"); }
    const auto available = rows - gate_begin;
    const bool whole     = available == std::uint64_t(count) * per;
    if (!whole && available != per) {
        refuse(parameter, "gate_up holds " + std::to_string(available) + " rows from expert 0, "
                              "which is neither E*2I (" + std::to_string(std::uint64_t(count) * per) +
                              ") nor 2I (" + std::to_string(per) + ")");
    }
    // Expert 0's gate rows, then its up rows, immediately behind.
    if (gate_part.end != gate_part.begin + intermediate * hidden ||
        up_part.begin != gate_part.begin + intermediate * hidden ||
        up_part.end != gate_part.begin + per * hidden || up_part.begin % hidden) {
        refuse(parameter, "gate and up do not occupy the first expert's two halves");
    }
    const auto& down_parent = *down_part.parent;
    const auto down_begin   = down_part.begin / intermediate;
    const auto down_rows    = parent_rows(down_parent, intermediate, parameter, "down");
    if (down_begin >= down_rows) { refuse(parameter, "down starts past its parent"); }
    const auto down_available = down_rows - down_begin;
    const bool down_whole     = down_available >= std::uint64_t(count) * hidden;
    if (!down_whole && down_available != hidden) {
        refuse(parameter, "down holds " + std::to_string(down_available) +
                              " rows from expert 0, which is neither E*H (" +
                              std::to_string(std::uint64_t(count) * hidden) + ") nor H (" +
                              std::to_string(hidden) + ")");
    }
    if (down_part.end != down_part.begin + hidden * intermediate) {
        refuse(parameter, "down does not hold exactly one expert's rows");
    }
    ExpertBankDescription out;
    out.fused_         = true;
    out.whole_bank_    = whole && down_whole;
    out.count_         = count;
    out.intermediate_  = intermediate;
    out.hidden_        = hidden;
    out.parameter_     = std::move(parameter);
    out.gate_up_begin_ = gate_begin;
    out.gate_begin_    = gate_begin;
    out.up_begin_      = gate_begin + intermediate;
    out.down_begin_    = down_begin;
    if (out.whole_bank_) {
        out.gate_up_ =
            bank_view(parent, {gate_begin, count, per, per}, out.parameter_);
        out.down_ = bank_view(down_parent, {down_begin, count, hidden, hidden}, out.parameter_);
    }
    return out;
}

ExpertBankDescription ExpertBankDescription::separate(const WeightView& gate, const WeightView& up,
                                                      const WeightView& down, std::uint32_t count,
                                                      std::uint64_t intermediate,
                                                      std::string parameter) {
    if (!count || !intermediate) { refuse(parameter, "a separate expert bank needs E and I"); }
    const auto hidden = gate.shape.size() == 2 ? gate.shape[1] : 0;
    if (!hidden) { refuse(parameter, "separate gate bank is not a matrix"); }
    require_shape(gate, intermediate, hidden, parameter, "gate");
    require_shape(up, intermediate, hidden, parameter, "up");
    require_shape(down, hidden, intermediate, parameter, "down");
    const auto& gate_part = single_part(gate, hidden, parameter, "gate");
    const auto& up_part   = single_part(up, hidden, parameter, "up");
    const auto& down_part = single_part(down, intermediate, parameter, "down");
    if (gate_part.parent == up_part.parent) {
        refuse(parameter, "separate banks cannot share the gate and up parent of the fused form");
    }
    const auto& gate_parent = *gate_part.parent;
    const auto gate_begin   = gate_part.begin / hidden;
    const auto up_begin     = up_part.begin / hidden;
    const auto down_begin   = down_part.begin / intermediate;
    const auto gate_rows    = parent_rows(gate_parent, hidden, parameter, "gate");
    if (gate_begin >= gate_rows) { refuse(parameter, "gate starts past its parent"); }
    const auto available = gate_rows - gate_begin;
    const bool whole     = available == std::uint64_t(count) * intermediate;
    if (!whole && available != intermediate) {
        refuse(parameter, "gate holds " + std::to_string(available) + " rows from expert 0, "
                              "which is neither E*I (" +
                              std::to_string(std::uint64_t(count) * intermediate) + ") nor I (" +
                              std::to_string(intermediate) + ")");
    }
    const auto& up_parent = *up_part.parent;
    if (parent_rows(up_parent, hidden, parameter, "up") != gate_rows) {
        refuse(parameter, "gate and up banks hold a different number of rows");
    }
    for (const auto& part : {gate_part, up_part}) {
        if (part.end != part.begin + intermediate * hidden) {
            refuse(parameter, "a separate gate/up block does not hold exactly one expert's rows");
        }
    }
    const auto& down_parent = *down_part.parent;
    const auto down_rows    = parent_rows(down_parent, intermediate, parameter, "down");
    if (down_begin >= down_rows) { refuse(parameter, "down starts past its parent"); }
    const auto down_available = down_rows - down_begin;
    const bool down_whole     = down_available == std::uint64_t(count) * hidden;
    if (!down_whole && down_available != hidden) {
        refuse(parameter, "down holds " + std::to_string(down_available) +
                              " rows from expert 0, which is neither E*H (" +
                              std::to_string(std::uint64_t(count) * hidden) + ") nor H (" +
                              std::to_string(hidden) + ")");
    }
    if (down_part.end != down_part.begin + hidden * intermediate) {
        refuse(parameter, "down does not hold exactly one expert's rows");
    }
    ExpertBankDescription out;
    out.fused_         = false;
    out.whole_bank_    = whole && down_whole;
    out.count_         = count;
    out.intermediate_  = intermediate;
    out.hidden_        = hidden;
    out.parameter_     = std::move(parameter);
    out.gate_begin_    = gate_begin;
    out.up_begin_      = up_begin;
    out.down_begin_    = down_begin;
    out.gate_up_begin_ = gate_begin;
    if (out.whole_bank_) {
        out.gate_ =
            bank_view(gate_parent, {gate_begin, count, intermediate, intermediate}, out.parameter_);
        out.up_ =
            bank_view(up_parent, {up_begin, count, intermediate, intermediate}, out.parameter_);
        out.down_ = bank_view(down_parent, {down_begin, count, hidden, hidden}, out.parameter_);
    }
    return out;
}

ExpertBankDescription ExpertBankDescription::from_bindings(const WeightView& gate,
                                                           const WeightView& up,
                                                           const WeightView& down,
                                                           std::uint32_t count,
                                                           std::uint64_t intermediate,
                                                           std::string parameter) {
    if (!count || !intermediate || gate.shape.size() != 2 || up.shape != gate.shape ||
        down.shape.size() != 2) {
        refuse(parameter, "routed expert matrices are not the expected matrices of one bank");
    }
    const auto hidden = gate.shape[1];
    // The form is the artifact's: one parent holding 2I rows per expert is the fused bank, three
    // parents holding I, I and H rows are Qwen4Exp's independent banks. Anything else is refused
    // here rather than sliced with the wrong row count.
    const bool fused = gate.parts.size() == 1 && up.parts.size() == 1 &&
                       gate.parts[0].parent != nullptr && gate.parts[0].parent == up.parts[0].parent;
    return fused ? ExpertBankDescription::fused(gate, up, down, count, intermediate,
                                               std::move(parameter))
                 : ExpertBankDescription::separate(gate, up, down, count, intermediate,
                                                   std::move(parameter));
}

const Weight& RoutedMoeParameters::fused_control() const {
    if (split) {
        throw std::invalid_argument(
            split->parameter +
            ": the artifact keeps the router and the shared logit in their own parents; the routed "
            "kernel reads one fused [E+1,H] control matrix");
    }
    return control;
}

const ops::SwiGLUWeights& RoutedMoeParameters::fused_shared() const {
    if (split) {
        throw std::invalid_argument(
            split->parameter +
            ": the artifact keeps the shared expert's gate, up and down in their own parents; the "
            "routed kernel reads one fused gate_up [2I,H]");
    }
    return shared;
}

namespace {

class Prepare {
public:
    explicit Prepare(const Model& model) : model_(model) {}

    LinearParameters linear(WeightId id) const {
        return with_context(model_.weight(id).name,
                            [&] { return ops::prepare_linear_weight(model_.input(id)); });
    }

    LinearParameters linear(WeightUseId id) const {
        return with_context(model_.weight(id.parameter).name,
                            [&] { return ops::prepare_linear_weight(model_.input(id)); });
    }

    Tensor tensor(WeightId id) const {
        const auto& bound = model_.weight(id);
        return with_context(bound.name, [&] {
            const auto& view = bound.view;
            if (view.shape.size() > 4) {
                throw std::invalid_argument("direct parameter exceeds Tensor rank");
            }
            std::array<std::int32_t, 4> axes{1, 1, 1, 1};
            for (std::size_t i = 0; i < view.shape.size(); ++i) {
                const auto extent = view.shape[view.shape.size() - 1 - i];
                if (extent > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
                    throw std::invalid_argument("direct parameter exceeds Tensor extent");
                }
                axes[i] = static_cast<std::int32_t>(extent);
            }
            return weight_tensor(view, {axes[0], axes[1], axes[2], axes[3]});
        });
    }

    DenseParameters dense(const DenseWeights& w) const {
        return {with_context(model_.weight(w.gate).name,
                             [&] {
                                 return ops::prepare_linear_swiglu_weight(model_.input(w.gate),
                                                                          model_.input(w.up));
                             }),
                linear(w.down)};
    }

    // The fused [E+1,H] control rows, or nothing when the artifact keeps the router and the shared
    // logit in their own parents (Qwen4Exp). The shapes are always checked, so a caller that reads
    // the fused form never gets rows of the wrong width.
    std::optional<Weight> control_weight(const MoeWeights& moe) const {
        const auto& router_name = model_.weight(moe.router).name;
        const auto& router      = model_.weight(moe.router).view;
        const auto& score       = model_.weight(moe.shared_score).view;
        if (router.shape.size() != 2 || score.shape.size() != 2 ||
            router.shape.at(1) != score.shape.at(1) || score.shape.at(0) != 1) {
            throw std::invalid_argument(router_name +
                                        ": MoE control rows are not [E,H] followed by [1,H]");
        }
        const bool fused = router.parts.size() == 1 && score.parts.size() == 1 &&
                           router.parts[0].parent != nullptr &&
                           router.parts[0].parent == score.parts[0].parent &&
                           router.parts[0].end == score.parts[0].begin;
        if (!fused) { return std::nullopt; }
        const auto rows = router.shape.at(0) + 1;
        return native_weight({{rows, router.shape.at(1)},
                              {{router.parts[0].parent, router.parts[0].begin,
                                score.parts[0].end}}});
    }

    // The fused gate_up [2I,H] plus down [H,I], or nothing when the shared expert's three matrices
    // occupy their own parents (Qwen4Exp).
    std::optional<ops::SwiGLUWeights> shared_weights(const MoeWeights& moe) const {
        const auto& gate = model_.weight(moe.shared.gate).view;
        const auto& up   = model_.weight(moe.shared.up).view;
        const auto& down = model_.weight(moe.shared.down).view;
        if (gate.shape.size() != 2 || up.shape != gate.shape || down.shape.size() != 2 ||
            down.shape.at(0) != gate.shape.at(1) || down.shape.at(1) != gate.shape.at(0)) {
            throw std::invalid_argument(model_.weight(moe.shared.gate).name +
                                        ": shared expert matrices have inconsistent shapes");
        }
        const bool fused = gate.parts.size() == 1 && up.parts.size() == 1 &&
                           gate.parts[0].parent != nullptr &&
                           gate.parts[0].parent == up.parts[0].parent &&
                           gate.parts[0].end == up.parts[0].begin;
        if (!fused) { return std::nullopt; }
        const auto rows  = gate.shape.at(0) * 2;
        const auto fused_gate_up = native_weight({{rows, gate.shape.at(1)},
                                                  {{gate.parts[0].parent, gate.parts[0].begin,
                                                    up.parts[0].end}}});
        return ops::SwiGLUWeights{ops::SingleProjectionWeight{fused_gate_up,
                                                            ops::LinearPolicy::A16Only},
                                  ops::SingleProjectionWeight{native_weight(down),
                                                              ops::LinearPolicy::A16Only}};
    }

    RoutedMoeParameters routed_moe(const MoeWeights& moe) const {
        const auto count = static_cast<std::uint32_t>(moe.experts.size());
        if (count == 0) { throw std::invalid_argument("MoE has no experts"); }
        const auto top_k = static_cast<std::int32_t>(
            std::get<MoeConfig>(model_.config().text.ffn).num_experts_per_tok);
        if (top_k <= 0) { throw std::invalid_argument("MoE declares no top-k route"); }
        const auto& gate = model_.weight(moe.experts.front().gate).view;
        const auto& up   = model_.weight(moe.experts.front().up).view;
        const auto& down = model_.weight(moe.experts.front().down).view;
        if (gate.shape.size() != 2 || down.shape.size() != 2 || up.shape != gate.shape) {
            throw std::invalid_argument("MoE expert matrices must be 2D");
        }
        const auto intermediate = gate.shape.at(0);
        const auto hidden       = gate.shape.at(1);
        if (down.shape.at(1) != intermediate || down.shape.at(0) != hidden) {
            throw std::invalid_argument("MoE expert down differs in H or I");
        }
        // The banks are read off the artifact's own row blocks. qwen3_5 fuses gate|up into one
        // parent, Qwen4Exp keeps three; a bank that matches neither is refused here, which is what
        // stops a 2I-row slice from being taken out of an I-row bank.
        const auto banks = ExpertBankDescription::from_bindings(
            gate, up, down, count, intermediate, model_.weight(moe.experts.front().gate).name);
        const auto control = with_context(model_.weight(moe.router).name,
                                          [&] { return control_weight(moe); });
        const auto shared  = with_context(model_.weight(moe.shared.gate).name,
                                          [&] { return shared_weights(moe); });

        if (banks.fused_form() && control && shared) {
            const auto& gate_parent = *gate.parts.at(0).parent;
            const auto& down_parent = *down.parts.at(0).parent;
            // port (sm_89): a streamed bank has no bytes at load time, so its layout and profile
            // come from the artifact description instead of a bound parent.
            const bool resident = banks.whole_bank() && gate_parent.data != nullptr &&
                                  gate_parent.device.type() == DeviceType::CUDA;
            if (resident) {
                const auto& gate_up   = banks.bank(ExpertBankRole::GateUp);
                const auto& down_bank = banks.bank(ExpertBankRole::Down);
                RoutedMoeParameters out;
                out.control = *control;
                out.shared  = *shared;
                out.routed  = {device_expert_bank(gate_up, down_bank), {}};
                out.profile = {ops::routed_swiglu_profile(gate_up, down_bank, top_k,
                                                          ops::LinearPolicy::A16Only,
                                                          ops::LinearPolicy::A16Only),
                               ops::swiglu_profile(out.shared)};
                return out;
            }
            // port (sm_89): an artifact that keeps one object per expert gives this weight a parent
            // of exactly one expert's rows, so a bank-wide region would run past it. The bank keeps
            // the same per-expert shape either way; only where the row block starts differs.
            auto gate_up_layout =
                banks.whole_bank()
                    ? banks.layout(ExpertBankRole::GateUp)
                    : describe_weight_bank(gate_parent.geometry, {0, 1, 2 * intermediate,
                                                                 2 * intermediate});
            auto down_layout = banks.whole_bank()
                                   ? banks.layout(ExpertBankRole::Down)
                                   : describe_weight_bank(down_parent.geometry, {0, 1, hidden, hidden});
            if (!banks.whole_bank()) {
                // The bank still holds `count` experts; only its object layout is per expert.
                gate_up_layout.region.count = count;
                down_layout.region.count    = count;
            }
            RoutedMoeParameters streamed;
            streamed.control = *control;
            streamed.shared  = *shared;
            // port (sm_89): an artifact may hold one object per expert, which lets the loader copy a
            // whole expert with one range instead of six. Collect the handle per expert; a fused
            // artifact repeats the same value here and the stream stays on the single-object path.
            std::vector<std::uint64_t> gate_up_objects;
            std::vector<std::uint64_t> down_objects;
            gate_up_objects.reserve(count);
            down_objects.reserve(count);
            for (const auto& expert : moe.experts) {
                gate_up_objects.push_back(
                    model_.weight(expert.gate).view.parts.at(0).parent->source_object);
                down_objects.push_back(
                    model_.weight(expert.down).view.parts.at(0).parent->source_object);
            }
            streamed.stream = std::make_shared<ExpertStream>(
                count, gate_up_layout, down_layout, gate_parent.geometry, down_parent.geometry,
                std::move(gate_up_objects), std::move(down_objects));
            streamed.profile = {
                ops::routed_swiglu_profile(gate_up_layout, down_layout, top_k,
                                           ops::LinearPolicy::A16Only, ops::LinearPolicy::A16Only),
                ops::swiglu_profile(streamed.shared)};
            return streamed;
        }

        // Qwen4Exp keeps the router, the shared logit, the shared expert and the three expert banks
        // in their own parents. Describe each of them as the artifact stores it; the fused form the
        // routed kernel reads is refused by the accessors above, not approximated here.
        if (!banks.whole_bank()) {
            throw std::invalid_argument(
                banks.parameter() +
                ": one expert per object cannot be combined with separate router/shared parents");
        }
        MoeSplitParameters split;
        split.parameter    = banks.parameter();
        split.router       = with_context(model_.weight(moe.router).name, [&] {
            return native_weight(model_.weight(moe.router).view);
        });
        split.shared_score = with_context(model_.weight(moe.shared_score).name, [&] {
            return native_weight(model_.weight(moe.shared_score).view);
        });
        split.shared_gate  = linear(moe.shared.gate);
        split.shared_up    = linear(moe.shared.up);
        split.shared_down  = linear(moe.shared.down);
        split.experts      = banks;

        const auto& gate_layout =
            banks.layout(banks.fused_form() ? ExpertBankRole::GateUp : ExpertBankRole::Gate);
        const auto& down_layout = banks.layout(ExpertBankRole::Down);
        const auto& shared_gate = split.shared_gate.weight;
        const auto& shared_down = split.shared_down.weight;
        RoutedMoeParameters out;
        out.profile = {
            ops::RoutedSwiGLUProfile{
                {dimension(hidden), dimension(intermediate), gate_layout.format, down_layout.format,
                 gate_layout.layout, down_layout.layout, ops::LinearPolicy::A16Only,
                 ops::LinearPolicy::A16Only},
                dimension(count), top_k},
            ops::SwiGLUProfile{dimension(shared_gate.k), dimension(shared_gate.n), shared_gate.qtype,
                               shared_down.qtype, shared_gate.layout, shared_down.layout,
                               ops::LinearPolicy::A16Only, ops::LinearPolicy::A16Only}};
        out.split = std::move(split);
        return out;
    }

    FfnParameters ffn(const BlockWeights& w) const {
        if (const auto* d = std::get_if<DenseWeights>(&w.ffn)) { return dense(*d); }
        const auto& moe = std::get<MoeWeights>(w.ffn);
        // The route width is the config's: qwen3_5 routes 8, Qwen4Exp 10, and the routing Op
        // validates its own (H,E,K) profile.
        // port (sm_89): the shipped sparse MoE prefill kernel cannot index this artifact's expert
        // bank - it hardcodes 1024 rows per expert and stages no usable scales, so it writes a
        // zero routed term and the layer silently degenerates to the shared expert alone
        // (perplexity ~1.2e3 on ordinary text). The groupwise-int path is the only implementation
        // wired for this model.
        return routed_moe(moe);
    }

    BlockParameters block(const BlockWeights& w) const {
        BlockParameters out;
        out.input_norm          = tensor(w.input_norm);
        out.post_attention_norm = tensor(w.post_attention_norm);
        out.ffn                 = ffn(w);
        if (const auto* a = std::get_if<AttentionWeights>(&w.mixer)) {
            out.mixer = AttentionParameters{
                ops::prepare_attn_input_proj_weights(model_.input(a->query), model_.input(a->key),
                                                     model_.input(a->gate), model_.input(a->value)),
                tensor(a->query_norm), tensor(a->key_norm), linear(a->output)};
            out.projection_prefetch =
                prefetch(std::get<AttentionParameters>(out.mixer).projection, a->query);
        } else {
            const auto& g = std::get<GdnWeights>(w.mixer);
            out.mixer     = GdnParameters{
                ops::prepare_gdn_input_proj_weights(model_.input(g.query), model_.input(g.key),
                                                        model_.input(g.value), model_.input(g.z)),
                ops::prepare_gdn_control_proj_weights(model_.input(g.a_projection),
                                                          model_.input(g.b_projection)),
                tensor(g.a_log),
                tensor(g.dt_bias),
                tensor(g.convolution),
                tensor(g.norm),
                linear(g.output)};
            out.projection_prefetch =
                prefetch(std::get<GdnParameters>(out.mixer).projection, g.query);
        }
        return out;
    }

    ops::SparseMoeHints prefetch(const ops::ProjectionWeights& projection, WeightId query) const {
        const auto* single = std::get_if<LinearParameters>(&projection);
        const auto& weight =
            single ? single->weight : std::get<ops::PairedProjectionWeights>(projection).first;
        const auto& geometry = model_.weight(query).view.parts.front().parent->geometry;
        const auto row_bytes = geometry.layout == QuantLayout::Contiguous
                                   ? std::uint64_t(weight.k) * dtype_size(DType::BF16)
                                   : geometry.code_bytes_per_row;
        return {weight.qdata, static_cast<std::size_t>(row_bytes * weight.n)};
    }

    MtpParameters mtp(const MtpWeights& w) const {
        const auto& a = std::get<AttentionWeights>(w.layer.mixer);
        const std::array inputs{model_.input(a.query), model_.input(a.key), model_.input(a.gate),
                                model_.input(a.value)};
        MtpParameters out;
        out.input_projection    = linear(w.input_projection);
        out.embedding_norm      = tensor(w.embedding_norm);
        out.hidden_norm         = tensor(w.hidden_norm);
        out.input_norm          = tensor(w.layer.input_norm);
        out.post_attention_norm = tensor(w.layer.post_attention_norm);
        out.final_norm          = tensor(w.final_norm);
        out.projection.packed   = ops::prepare_linear_weight(inputs);
        if (model_.config().text.architecture == Architecture::Qwen3_5) {
            out.projection.rows = {linear(a.query), linear(a.key), linear(a.gate), linear(a.value)};
        }
        out.query_norm  = tensor(a.query_norm);
        out.key_norm    = tensor(a.key_norm);
        out.output      = linear(a.output);
        out.ffn         = ffn(w.layer);
        out.output_head = linear(w.output_head_use);
        return out;
    }

    NormParameters norm(const NormWeights& w) const { return {tensor(w.weight), tensor(w.bias)}; }

    std::optional<Tensor> joined_bias(const std::array<WeightId, 3>& ids) const {
        WeightView view;
        std::uint64_t count = 0;
        for (const auto id : ids) {
            const auto& input = model_.weight(id).view;
            count += weight_element_count(input.shape);
            for (const auto& part : input.parts) {
                if (!view.parts.empty() && (view.parts.back().parent != part.parent ||
                                            view.parts.back().end != part.begin)) {
                    return std::nullopt;
                }
                view.parts.push_back(part);
            }
        }
        if (count > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
            throw std::invalid_argument("Vision bias exceeds Tensor extent");
        }
        view.shape = {count};
        return weight_tensor(view, {static_cast<std::int32_t>(count)});
    }

    VisionParameters vision(const VisionWeights& w) const {
        VisionParameters out;
        out.patch_embedding      = linear(w.patch_embedding);
        out.patch_embedding_bias = tensor(w.patch_embedding_bias);
        out.position_embedding   = tensor(w.position_embedding);
        out.layers.reserve(w.layers.size());
        for (std::size_t i = 0; i < w.layers.size(); ++i) {
            out.layers.push_back(with_context("vision/layers/" + std::to_string(i), [&] {
                const auto& layer = w.layers[i];
                const std::array qkv{model_.input(layer.query), model_.input(layer.key),
                                     model_.input(layer.value)};
                const std::array ids{layer.query_bias, layer.key_bias, layer.value_bias};
                const auto bias = joined_bias(ids);
                if (!bias) {
                    throw std::invalid_argument(
                        "Vision QKV bias: this fixed call requires a contiguous bias bank");
                }
                return VisionBlockParameters{norm(layer.norm1),
                                             norm(layer.norm2),
                                             ops::prepare_linear_weight(qkv),
                                             *bias,
                                             linear(layer.output),
                                             linear(layer.fc1),
                                             linear(layer.fc2),
                                             tensor(layer.output_bias),
                                             tensor(layer.fc1_bias),
                                             tensor(layer.fc2_bias)};
            }));
        }
        out.merger_norm     = norm(w.merger_norm);
        out.merger_fc1      = linear(w.merger_fc1);
        out.merger_fc2      = linear(w.merger_fc2);
        out.merger_fc1_bias = tensor(w.merger_fc1_bias);
        out.merger_fc2_bias = tensor(w.merger_fc2_bias);
        return out;
    }

    DynamicConvParameters convolution(const DynamicConvWeights& w) const {
        return {tensor(w.base_kernel), linear(w.kernel_projection)};
    }

    DraftParameters draft(const DraftWeights& w) const {
        DraftParameters out;
        out.feature_projection = linear(w.feature_projection);
        out.context_norm       = tensor(w.context_norm);
        out.final_norm         = tensor(w.final_norm);
        out.output_head        = linear(w.output_head_use);
        out.layers.reserve(w.layers.size());
        for (std::size_t i = 0; i < w.layers.size(); ++i) {
            out.layers.push_back(with_context(
                std::string(model_.options().speculative_component()) + "/layers/" +
                    std::to_string(i),
                [&] {
                    const auto& layer = w.layers[i];
                    const auto& a     = layer.attention;
                    DraftBlockParameters result;
                    result.input_norm          = tensor(layer.input_norm);
                    result.post_attention_norm = tensor(layer.post_attention_norm);
                    result.query_key_value     = ops::prepare_attn_input_proj_weights(
                        model_.input(a.query), model_.input(a.key), model_.input(a.value));
                    result.context_key   = linear(a.context_key);
                    result.context_value = linear(a.context_value);
                    result.query_norm    = tensor(a.query_norm);
                    result.key_norm      = tensor(a.key_norm);
                    result.output        = linear(a.output);
                    result.mlp           = dense(layer.mlp);
                    if (layer.attention_conv) {
                        result.attention_conv = convolution(*layer.attention_conv);
                    }
                    if (layer.mlp_conv) { result.mlp_conv = convolution(*layer.mlp_conv); }
                    return result;
                }));
        }
        if (w.selector) {
            out.selector = SelectorParameters{linear(w.selector->hidden_projection),
                                              tensor(w.selector->predecessor_codebook),
                                              tensor(w.selector->successor_codebook)};
        }
        return out;
    }

private:
    const Model& model_;
};

} // namespace

Parameters::Parameters(const Model& source) : model(source) {
    const Prepare prepare(model);
    const auto& w        = model.weights();
    text.token_embedding = native_weight(model.weight(w.text.token_embedding).view);
    text.output_head     = prepare.linear(w.text.output_head_use);
    text.final_norm      = prepare.tensor(w.text.final_norm);
    text.layers.reserve(w.text.layers.size());
    for (std::size_t i = 0; i < w.text.layers.size(); ++i) {
        text.layers.push_back(with_context("text/layers/" + std::to_string(i),
                                           [&] { return prepare.block(w.text.layers[i]); }));
    }
    if (w.mtp) {
        mtp = with_context("mtp", [&] { return prepare.mtp(*w.mtp); });
    }
    if (w.vision) {
        vision = with_context("vision", [&] { return prepare.vision(*w.vision); });
    }
    if (w.draft) {
        draft = with_context(std::string(model.options().speculative_component()),
                             [&] { return prepare.draft(*w.draft); });
    }
    if (w.proposal) {
        proposal =
            ProposalParameters{prepare.linear(w.proposal->head), std::nullopt, w.proposal->rows};
        if (w.proposal->token_ids) { proposal->token_ids = prepare.tensor(*w.proposal->token_ids); }
    }
}

} // namespace ninfer::models::qwen3_5::execution
