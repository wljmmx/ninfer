#include "models/qwen4_exp/load.h"

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "models/qwen3_5/load/bindings.h"

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

namespace ninfer::models::qwen4_exp::loading {

std::shared_ptr<const PleTableSource>
open_ple_table(const artifact::Reader& reader, artifact::Binder& binder,
               std::span<const qwen3_5::loading::PendingWeight> pending,
               const qwen3_5::TextWeights& weights, const qwen3_5::TextConfig& config) {
    if (!config.ple) throw artifact::ArtifactError("PLE table requested for a model without PLE");
    const auto& ple = *config.ple;
    if (ple.layer_ids.empty()) {
        throw artifact::ArtifactError("PLE declares no layer ids");
    }
    // Every PLE block gathers from the same n-gram bank, so its bound object ranges must be the
    // same object set; two different tables would otherwise be read as one table's row space.
    std::vector<artifact::ObjectHandle> shards;
    const auto collect = [&](const qwen3_5::PleWeights& w) {
        if (std::size_t(w.table.index) >= pending.size())
            throw artifact::ArtifactError("PLE table binding is not in the load plan");
        const auto& reference = pending[std::size_t(w.table.index)].reference;
        if (reference.binding.parts.empty())
            throw artifact::ArtifactError(reference.name + ": PLE table has no object parts");
        for (const auto& part : reference.binding.parts) {
            if (std::find(shards.begin(), shards.end(), part.object) == shards.end())
                shards.push_back(part.object);
        }
    };
    collect(weights.layers.at(ple.layer_ids.front()).ple.value());
    const auto first = shards;
    for (const auto layer : ple.layer_ids) {
        shards.clear();
        collect(weights.layers.at(layer).ple.value());
        if (shards != first) {
            throw artifact::ArtifactError(
                "PLE layer " + std::to_string(layer) +
                " reads a different table object set than layer " +
                std::to_string(ple.layer_ids.front()) +
                "; one n-gram stage cannot gather from two banks");
        }
    }
    shards = first;
    // The controls are a resource of the text component; the hash cannot be evaluated without
    // them, so a PLE artifact that omits them fails here, naming the resource.
    const auto controls = binder.resource("text", "ple_hash_controls");
    return std::make_shared<PleTableSource>(reader, shards, controls);
}

std::int32_t ple_eos_token(const artifact::Directory& directory) {
    const auto& config = directory.component("text").config;
    const auto found   = config.find("eos_token_id");
    if (found == config.end() || !found->is_number_integer()) return -1;
    const auto id = found->get<std::int64_t>();
    if (id < 0 || id > std::numeric_limits<std::int32_t>::max()) return -1;
    return static_cast<std::int32_t>(id);
}

} // namespace ninfer::models::qwen4_exp::loading
