#include "models/qwen3_5/load.h"
#include "artifact/reader.h"
#include "models/qwen4_exp/ple/table.h"
#include "models/qwen3_5/load/bindings.h"

#include <cstdio>
#include <cstdlib>
#include <utility>

namespace ninfer::models::qwen3_5 {

struct LoadPlan::Impl {
    Config config;
    LoadOptions options;
    ModelWeights weights;
    std::vector<loading::PendingWeight> pending;
    artifact::MaterializationPlan materialization;
    FrontendResources resources;
    InstanceInfo info;
    std::shared_ptr<const qwen4_exp::PleTableSource> ple_table;
    std::int32_t ple_eos = -1;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

LoadPlan::~LoadPlan()                              = default;
LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;

const Config& LoadPlan::config() const { return impl_->config; }

const ModelWeights& LoadPlan::weights() const { return impl_->weights; }

const FrontendResources& LoadPlan::resources() const { return impl_->resources; }

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    return impl_->materialization;
}

const artifact::ParameterReference& LoadPlan::parameter(WeightId id) const {
    return impl_->pending.at(id.index).reference;
}

std::span<const WeightUse> LoadPlan::uses(WeightId id) const {
    return impl_->pending.at(id.index).uses;
}

const std::shared_ptr<const qwen4_exp::PleTableSource>& LoadPlan::ple_table() const {
    return impl_->ple_table;
}

std::int32_t LoadPlan::ple_eos() const { return impl_->ple_eos; }

LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options) {
    auto out     = std::make_unique<LoadPlan::Impl>();
    out->options = options;
    out->config  = parse_config(reader.directory(), options);
    // The PLE-off arm. The n-gram embedding is a config feature, so the only honest way to measure
    // "the model without it" is to drop the layers the config names - then nothing binds, no table
    // is opened, no block carries a PLE parameter, and the text path is the one that shipped before
    // PLE existed. Set NINFER_PLE_OFF=1 to select that arm; it is not a tuning knob.
    const bool ple_off = [] {
        const char* value = std::getenv("NINFER_PLE_OFF");
        return value != nullptr && *value != '\0' && std::strtol(value, nullptr, 10) != 0;
    }();
    if (ple_off && out->config.text.ple) {
        std::fprintf(stderr, "[ple] NINFER_PLE_OFF: running without the n-gram layers\n");
        out->config.text.ple.reset();
    }
    artifact::Binder binder(reader);
    out->resources = loading::bind_resources(binder, out->config);
    loading::Bindings bindings(binder);
    bindings.stream_experts = options.stream_experts;
    bindings.expert_host_copy = options.expert_host_copy;
    const auto& text  = out->config.text;
    out->weights.text = loading::bind_text(bindings, text, options);
    if (out->config.vision) {
        out->weights.vision = loading::bind_vision(bindings, *out->config.vision, text);
    }
    if (out->config.mtp) {
        out->weights.mtp = loading::bind_mtp(bindings, text, out->weights.text);
    }
    if (out->config.draft) {
        out->weights.draft =
            loading::bind_draft(bindings, *out->config.draft, text, out->weights.text,
                                std::string(options.speculative_component()));
    }
    if (options.proposal_enabled()) {
        const auto& proposal = reader.directory().component("text").proposal;
        if (!proposal) {
            throw artifact::ArtifactError("selected proposal head is absent from artifact");
        }
        out->weights.proposal = loading::bind_proposal(bindings, *proposal, text, options,
                                                       out->resources.public_token_count);
        if (out->config.draft && out->config.draft->dflash2) {
            const auto domain =
                proposal->indexed ? proposal->rows : out->resources.public_token_count;
            if (out->config.draft->dflash2->selector_top_k > domain) {
                throw artifact::ArtifactError(
                    "proposal domain is smaller than DFlash2 selector_top_k");
            }
        }
        if (out->weights.mtp) { out->weights.mtp->output_head = out->weights.proposal->head; }
        if (out->weights.draft) { out->weights.draft->output_head = out->weights.proposal->head; }
    }
    out->weights.text.output_head_use =
        bindings.use(out->weights.text.output_head, "text/final_hidden");
    if (out->weights.mtp) {
        out->weights.mtp->output_head_use =
            bindings.use(out->weights.mtp->output_head, "mtp/final_hidden");
    }
    if (out->weights.draft) {
        out->weights.draft->output_head_use =
            bindings.use(out->weights.draft->output_head,
                         std::string(options.speculative_component()) + "/final_hidden");
    }
    out->pending         = std::move(bindings.weights);
    // The PLE table is opened here, while the artifact's object tables are still reachable. The
    // source owns its own read handles, so the plan - not the caller - decides its lifetime.
    if (text.ple) {
        out->ple_table = loading::open_ple_table(reader, binder, out->pending, out->weights.text, text);
        out->ple_eos   = loading::ple_eos_token(reader.directory());
    }
    out->materialization = std::move(binder).finish();
    out->info.name       = reader.directory().metadata.value(
        "name", std::string(architecture_name(text.architecture)));
    out->info.metadata_json   = reader.directory().metadata.dump();
    out->info.provenance_json = reader.directory().provenance.dump();
    out->info.artifact_id     = reader.artifact_id();
    return LoadPlan(std::move(out));
}

std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                         const StartupObserver* observer) {
    if (!plan.impl_) { throw artifact::ArtifactError("load plan was already consumed"); }
    auto data    = std::move(plan.impl_);
    auto backing = artifact::materialize(*data->materialization.source,
                                         std::move(data->materialization), device, observer);
    auto bound   = loading::resolve_weights(std::move(data->pending), backing);
    auto model = std::unique_ptr<Model>(new Model(
        std::move(data->config), data->options, std::move(data->weights), std::move(bound),
        std::move(data->resources), std::move(data->info), std::move(backing)));
    return model;
}

std::unique_ptr<Model> load_model(const std::filesystem::path& path, LoadOptions options,
                                  DeviceContext& device, const StartupObserver* observer) {
    artifact::Reader reader(path);
    return materialize_model(plan_load(reader, options), device, observer);
}

} // namespace ninfer::models::qwen3_5
