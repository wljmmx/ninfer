#include "models/qwen3_5/load/bindings.h"

#include "artifact/views.h"

#include <algorithm>
#include <cmath>

namespace ninfer::models::qwen3_5::loading {
namespace {

// One artifact binding may hold every expert of a bank; a per-expert parameter is then the
// contiguous row block its index selects out of that logical element range.
artifact::Binding row_block(const std::string& parameter, const artifact::Binding& bank,
                            const artifact::Shape& shape, std::uint64_t first_row,
                            std::uint64_t rows) {
    if (shape.size() < 2) {
        throw artifact::ArtifactError(parameter + ": banked parameter needs two dimensions");
    }
    const auto width = shape.back();
    auto skip        = artifact::checked_mul(first_row, width, "bank row offset");
    auto remaining   = artifact::checked_mul(rows, width, "bank row count");
    artifact::Binding out;
    out.elements = remaining;
    for (const auto& part : bank.parts) {
        if (!remaining) { break; }
        const std::uint64_t length = part.end - part.begin;
        if (skip >= length) {
            skip -= length;
            continue;
        }
        const auto taken = std::min(length - skip, remaining);
        out.parts.push_back({part.object, part.begin + skip, part.begin + skip + taken});
        skip = 0;
        remaining -= taken;
    }
    if (remaining) { throw artifact::ArtifactError(parameter + ": bank is too short"); }
    return out;
}

} // namespace

void Bindings::alias_input(std::string logical, std::string artifact) {
    if (!input_aliases_.emplace(std::move(logical), std::move(artifact)).second) {
        throw artifact::ArtifactError("input alias declared twice for one logical input");
    }
}

const artifact::Use& Bindings::resolve_use(const std::string& name, std::string_view binding,
                                           std::string_view input) const {
    const auto& uses = binder.reader().directory().uses;
    const auto has   = [&](std::string_view candidate) {
        return uses.contains({std::string(binding), std::string(candidate)});
    };
    const bool canonical = has(input);
    const auto alias     = input_aliases_.find(input);
    if (alias != input_aliases_.end()) {
        const bool aliased = has(alias->second);
        if (canonical && aliased) {
            throw artifact::ArtifactError(name + ": Use is ambiguous, the artifact carries both " +
                                          std::string(binding) + "@" + std::string(input) + " and " +
                                          std::string(binding) + "@" + alias->second);
        }
        if (aliased) { return binder.use(binding, alias->second); }
    }
    if (!canonical) {
        if (alias == input_aliases_.end()) {
            throw artifact::ArtifactError(name + ": missing Use " + std::string(binding) + "@" +
                                          std::string(input));
        }
        throw artifact::ArtifactError(name + ": missing Use, the artifact carries neither " +
                                      std::string(binding) + "@" + std::string(input) + " nor " +
                                      std::string(binding) + "@" + alias->second);
    }
    return binder.use(binding, input);
}

WeightUseId Bindings::use(WeightId id, std::string_view input) const {
    const auto& parameter = at(id);
    // The declared Use keeps the logical spelling; the lookup accepts the artifact's alias too.
    (void)resolve_use(parameter.reference.name, parameter.binding_name, input);
    for (std::size_t i = 0; i < parameter.uses.size(); ++i) {
        if (parameter.uses[i].input == input) { return {id, i}; }
    }
    throw artifact::ArtifactError(parameter.reference.name + ": unresolved Use " +
                                  std::string(input));
}

WeightId Bindings::parameter(std::string name, artifact::Shape shape,
                             std::vector<std::string> inputs, std::optional<QType> exact_format,
                             artifact::Residency residency) {
    return bind(std::move(name), std::move(shape), std::move(inputs), exact_format, residency, {});
}

WeightId Bindings::direct(std::string name, artifact::Shape shape, QType format) {
    return bind(std::move(name), std::move(shape), {}, format, artifact::Residency::Device, {});
}

WeightId Bindings::aliased(std::string name, std::string binding, artifact::Shape shape,
                           std::vector<std::string> inputs, std::optional<QType> exact_format,
                           artifact::Residency residency) {
    return bind(std::move(name), std::move(shape), std::move(inputs), exact_format, residency,
                Alias{std::move(binding), 0, 0});
}

WeightId Bindings::aliased_direct(std::string name, std::string binding, artifact::Shape shape,
                                  QType format) {
    return bind(std::move(name), std::move(shape), {}, format, artifact::Residency::Device,
                Alias{std::move(binding), 0, 0});
}

WeightId Bindings::banked(std::string name, std::string binding, artifact::Shape shape,
                          std::uint64_t first_row, std::uint64_t rows,
                          std::vector<std::string> inputs, artifact::Residency residency) {
    return bind(std::move(name), std::move(shape), std::move(inputs), std::nullopt, residency,
                Alias{std::move(binding), first_row, rows});
}

WeightId Bindings::bind(std::string name, artifact::Shape shape, std::vector<std::string> inputs,
                        std::optional<QType> exact_format, artifact::Residency residency,
                        std::optional<Alias> alias) {
    if (parameters_.contains(name)) {
        throw artifact::ArtifactError(name + ": duplicate model parameter declaration");
    }
    PendingWeight pending;
    pending.binding_name = alias ? alias->binding : name;
    // A Use is keyed by the binding the artifact actually carries, not by the logical name.
    const std::string_view key = pending.binding_name;
    if (!alias) {
        pending.reference = binder.parameter(name, std::move(shape), residency, exact_format);
    } else {
        const auto& bindings = binder.reader().directory().bindings;
        const auto found     = bindings.find(alias->binding);
        if (found == bindings.end()) {
            throw artifact::ArtifactError(name + ": artifact has no " + alias->binding +
                                          " binding for this parameter");
        }
        if (binder.contains(name)) {
            throw artifact::ArtifactError(name + ": artifact carries both the canonical and the " +
                                          alias->binding + " binding");
        }
        const auto binding =
            alias->rows ? row_block(name, found->second, shape, alias->first_row, alias->rows)
                        : found->second;
        pending.reference =
            binder.binding(name, binding, std::move(shape), residency, exact_format);
    }
    for (const auto& input : inputs) {
        const auto& use = resolve_use(name, key, input);
        if (!use.activation_policy) {
            throw artifact::ArtifactError(name + "@" + input + ": missing activation policy");
        }
        WeightUse result;
        result.input = input;
        switch (*use.activation_policy) {
        case artifact::ActivationPolicy::A16Only:
            result.policy = ops::LinearPolicy::A16Only;
            break;
        case artifact::ActivationPolicy::AllowA8:
            result.policy = ops::LinearPolicy::AllowA8;
            break;
        case artifact::ActivationPolicy::AllowA4:
            result.policy = ops::LinearPolicy::AllowA4;
            break;
        }
        for (const auto& [role, binding] : use.auxiliaries) {
            if (role != "activation_input_divisor") {
                throw artifact::ArtifactError(name + "@" + input + ": unknown auxiliary " + role);
            }
            const auto value = binder.values(binding, QType::FP32).scalar_f32();
            if (!std::isfinite(value) || value <= 0) {
                throw artifact::ArtifactError(name + "@" + input +
                                              ": activation divisor must be positive finite FP32");
            }
            result.activation_input_divisor = value;
        }
        pending.uses.push_back(std::move(result));
    }
    for (const auto& part : pending.reference.binding.parts) {
        pending.source_objects.push_back(
            artifact::object_id(binder.reader().directory().object(part.object)));
    }
    const WeightId id{weights.size()};
    parameters_.emplace(std::move(name), id);
    weights.push_back(std::move(pending));
    return id;
}

std::vector<BoundWeight> resolve_weights(std::vector<PendingWeight>&& pending,
                                         const artifact::MaterializedArtifact& materialized) {
    std::vector<BoundWeight> out;
    out.reserve(pending.size());
    for (auto& item : pending) {
        auto view = artifact::bind_view(item.reference, materialized);
        out.push_back({std::move(item.reference.name), std::move(item.source_objects),
                       std::move(view), std::move(item.uses)});
    }
    return out;
}

} // namespace ninfer::models::qwen3_5::loading
