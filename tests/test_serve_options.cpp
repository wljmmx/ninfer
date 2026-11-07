#include "product/config/options.h"
#include "serve/translate.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninfer::serve;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ServeOptions parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return ninfer::product::parse_serve_options(static_cast<int>(argv.size()), argv.data());
}

} // namespace

int main() {
    int failures = 0;

    const ServeOptions defaults = parse({"ninfer-serve", "model.ninfer"});
    failures += check(defaults.launch.engine.context_cache.enabled, "prefix reuse is not enabled by default");
    failures +=
        check(!defaults.launch.generation.preserve_thinking, "thinking history is unexpectedly preserved by default");
    failures += check(!defaults.launch.engine.enable_vision, "Vision is not disabled by default");
    failures += check(defaults.launch.logging.request_jsonl.empty(),
                      "request JSONL logging is not disabled by default");
    failures += check(defaults.launch.engine.context_cost.preset_path.empty(),
                      "external context-cost presets are unexpectedly configured by default");
    failures += check(defaults.launch.logging.stats_interval_ms == 5000,
                      "periodic throughput interval default mismatch");
    failures += check(defaults.launch.engine.media_cache_bytes == ninfer::kDefaultMediaCacheBytes &&
                          defaults.launch.engine.media_live_bytes == ninfer::kDefaultMediaLiveBytes &&
                          defaults.launch.engine.media_preprocess_threads == 0,
                      "media preparation resource defaults mismatch");
    failures += check(defaults.launch.engine.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
                          defaults.launch.engine.kv_capacity.explicit_tokens == defaults.launch.engine.max_context,
                      "default KV capacity does not follow max context");
    failures += check(!defaults.launch.engine.context_cache.host_capacity_bytes.has_value(),
                      "default Host context capacity was resolved before Engine startup");
    failures += check(defaults.launch.engine.speculative.backend == ninfer::SpeculativeBackend::None,
                      "speculative decoding is not disabled by default");
    failures += check(defaults.response_store_max_records == ninfer::product::kDefaultResponseStoreRecords &&
                          defaults.response_store_max_bytes == ninfer::product::kDefaultResponseStoreBytes,
                      "Responses store defaults mismatch");
    failures += check(!defaults.model_id_override.has_value(),
                      "model id override is unexpectedly configured by default");
    failures += check(!defaults.launch.generation.thinking_budget,
                      "thinking budget is unexpectedly limited by default");
    failures += check(
        !defaults.launch.generation.sampling.temperature && !defaults.launch.generation.sampling.top_p &&
            !defaults.launch.generation.sampling.top_k && !defaults.launch.generation.sampling.presence_penalty &&
            !defaults.launch.generation.sampling.frequency_penalty,
        "server defaults unexpectedly override registered model sampling");
    failures += check(resolve_public_model_id(defaults, "artifact-model") == "artifact-model",
                      "artifact model id was not selected by default");

    const ServeOptions fp8 = parse({"ninfer-serve", "model.ninfer", "--kv-dtype", "fp8"});
    failures += check(fp8.launch.engine.kv_cache == ninfer::KvCacheStorage::Fp8E4M3Row256,
                      "--kv-dtype fp8 did not select row-scaled E4M3 KV");
    const ServeOptions nvfp4 = parse({"ninfer-serve", "model.ninfer", "--kv-dtype", "nvfp4"});
    failures += check(nvfp4.launch.engine.kv_cache == ninfer::KvCacheStorage::Nvfp4Group16,
                      "--kv-dtype nvfp4 did not select group-16 NVFP4 KV");
    const ServeOptions k8v4 = parse({"ninfer-serve", "model.ninfer", "--kv-dtype", "k8v4"});
    failures += check(k8v4.launch.engine.kv_cache == ninfer::KvCacheStorage::Fp8KeyNvfp4Value,
                      "--kv-dtype k8v4 did not select asymmetric K8V4 KV");
    const std::string kv_help = ninfer::product::startup_usage_text("ninfer-serve", ninfer::product::EntryPoint::Serve);
    failures += check(kv_help.find("nvfp4") != std::string::npos &&
                          kv_help.find("k8v4") != std::string::npos,
                      "serve help omits a production KV storage mode");

    const ServeOptions model_alias =
        parse({"ninfer-serve", "model.ninfer", "--model-id", "deployment-alias"});
    failures +=
        check(model_alias.model_id_override == "deployment-alias" &&
                  resolve_public_model_id(model_alias, "artifact-model") == "deployment-alias",
              "explicit model id did not override the artifact identity");

    const ServeOptions context_cost =
        parse({"ninfer-serve", "model.ninfer", "--context-cost-presets", "local-costs.json"});
    failures += check(context_cost.launch.engine.context_cost.preset_path == "local-costs.json",
                      "--context-cost-presets did not preserve its path");

    const ServeOptions thinking_budget =
        parse({"ninfer-serve", "model.ninfer", "--default-thinking-budget", "37"});
    failures += check(thinking_budget.launch.generation.thinking_budget == 37,
                      "--default-thinking-budget did not preserve its positive value");
    bool zero_thinking_budget_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--default-thinking-budget", "0"});
    } catch (const std::invalid_argument&) { zero_thinking_budget_rejected = true; }
    failures += check(zero_thinking_budget_rejected, "zero --default-thinking-budget was accepted");

    bool empty_model_id_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--model-id", ""});
    } catch (const std::invalid_argument&) { empty_model_id_rejected = true; }
    failures += check(empty_model_id_rejected, "empty --model-id was accepted");

    const ServeOptions dflash = parse({"ninfer-serve", "model.ninfer", "--spec", "dflash",
                                       "--draft-tokens", "15", "--lm-head-draft"});
    failures += check(dflash.launch.engine.speculative.backend == ninfer::SpeculativeBackend::DFlash,
                      "--spec dflash did not select DFlash");
    failures += check(dflash.launch.engine.speculative.draft_tokens == 15,
                      "--draft-tokens did not preserve the DFlash window");
    failures += check(dflash.launch.engine.speculative.proposal_head == ninfer::ProposalHead::Optimized,
                      "--lm-head-draft did not select the optimized proposal head");

    for (const auto k : {1U, 2U, 7U, 15U}) {
        const auto options = parse({"ninfer-serve", "model.ninfer", "--spec", "dflash2",
                                    "--draft-tokens", std::to_string(k), "--lm-head-draft"});
        failures += check(options.launch.engine.speculative.backend == ninfer::SpeculativeBackend::DFlash2 &&
                              options.launch.engine.speculative.draft_tokens == k &&
                              options.launch.engine.speculative.proposal_head == ninfer::ProposalHead::Optimized,
                          "serve options did not preserve DFlash2 configuration");
    }

    const ServeOptions dflash_vision = parse(
        {"ninfer-serve", "model.ninfer", "--spec", "dflash", "--draft-tokens", "15", "--vision"});
    failures += check(dflash_vision.launch.engine.enable_vision &&
                          dflash_vision.launch.engine.speculative.backend == ninfer::SpeculativeBackend::DFlash &&
                          dflash_vision.launch.engine.speculative.draft_tokens == 15,
                      "serve options did not preserve combined DFlash and Vision features");

    bool implicit_backend_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--draft-tokens", "3"});
    } catch (const std::invalid_argument&) { implicit_backend_rejected = true; }
    failures += check(implicit_backend_rejected, "--draft-tokens selected a backend implicitly");

    // v8 section 12, first phase: the controller-mode flag is the initial value for who the
    // runtime controller optimizes for; a typo must be refused rather than silently mean "auto".
    failures += check(defaults.launch.controller_mode == "auto",
                      "controller mode is not auto by default");
    const ServeOptions stable = parse({"ninfer-serve", "model.ninfer", "--controller-mode", "stable"});
    failures += check(stable.launch.controller_mode == "stable",
                      "--controller-mode stable did not parse");
    const ServeOptions performance =
        parse({"ninfer-serve", "model.ninfer", "--controller-mode", "performance"});
    failures += check(performance.launch.controller_mode == "performance",
                      "--controller-mode performance did not parse");
    bool bad_mode_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--controller-mode", "aggressive"});
    } catch (const std::invalid_argument&) { bad_mode_rejected = true; }
    failures += check(bad_mode_rejected, "an invalid --controller-mode was accepted");
    failures += check(
        ninfer::product::startup_usage_text("ninfer-serve", ninfer::product::EntryPoint::Serve).find("--controller-mode") != std::string::npos,
        "serve help omits --controller-mode");

    const ServeOptions configured = parse({"ninfer-serve",
                                           "model.ninfer",
                                           "--no-prefix-reuse",
                                           "--vision",
                                           "--max-concurrency",
                                           "4",
                                           "--max-pending-requests",
                                           "12",
                                           "--pending-timeout-ms",
                                           "2500",
                                           "--max-context",
                                           "4096",
                                           "--kv-capacity",
                                           "8192",
                                           "--log-stats-interval-ms",
                                           "0",
                                           "--preserve-thinking",
                                           "--media-cache-mib",
                                           "256",
                                           "--media-live-mib",
                                           "512",
                                           "--media-preprocess-threads",
                                           "6"});
    failures += check(!configured.launch.engine.context_cache.enabled,
                      "--no-prefix-reuse did not disable server prefix reuse");
    failures += check(!configured.launch.engine.context_cache.enabled &&
                          configured.launch.engine.context_cache.host_capacity_bytes ==
                              defaults.launch.engine.context_cache.host_capacity_bytes &&
                          configured.launch.engine.context_cache.device_state_slots ==
                              defaults.launch.engine.context_cache.device_state_slots,
                      "--no-prefix-reuse changed context capacities or retained cache enablement");
    failures += check(configured.launch.engine.enable_vision, "--vision did not enable Vision");
    failures += check(configured.launch.generation.preserve_thinking == true,
                      "--preserve-thinking did not reach serving options");
    failures +=
        check(configured.launch.engine.max_concurrency == 4, "--max-concurrency did not reach serving options");
    failures += check(configured.launch.engine.max_context == 4096 &&
                          configured.launch.engine.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
                          configured.launch.engine.kv_capacity.explicit_tokens == 8192,
                      "context and KV capacity options were not kept distinct");
    failures += check(configured.launch.engine.max_pending_requests == 12,
                      "--max-pending-requests did not reach serving options");
    failures += check(configured.launch.engine.pending_timeout_ms == 2500,
                      "--pending-timeout-ms did not reach serving options");
    failures += check(configured.launch.logging.stats_interval_ms == 0,
                      "--log-stats-interval-ms did not disable periodic reporting");
    failures += check(configured.launch.engine.media_cache_bytes == (256ULL << 20) &&
                          configured.launch.engine.media_live_bytes == (512ULL << 20) &&
                          configured.launch.engine.media_preprocess_threads == 6,
                      "media preparation limits did not reach serving options");

    const ServeOptions logging = parse({"ninfer-serve", "model.ninfer", "--log-level", "debug"});
    failures += check(logging.launch.logging.level == ninfer::product::LogLevel::Debug,
                      "log level did not reach serving options");

    const ServeOptions context_cache = parse(
        {"ninfer-serve", "model.ninfer", "--device-state-slots", "3", "--host-context-mib", "64"});
    failures += check(context_cache.launch.engine.context_cache.enabled &&
                          context_cache.launch.engine.context_cache.device_state_slots == 3 &&
                          context_cache.launch.engine.context_cache.host_capacity_bytes == (64ULL << 20),
                      "context-cache capacities did not reach serving options");

    const ServeOptions zero_host_context =
        parse({"ninfer-serve", "model.ninfer", "--host-context-mib", "0"});
    failures += check(zero_host_context.launch.engine.context_cache.enabled &&
                          zero_host_context.launch.engine.context_cache.host_capacity_bytes == 0,
                      "zero Host context capacity was not retained as an explicit budget");

    for (const auto& [mib, bytes] : std::vector<std::pair<std::string, std::size_t>>{
             {"146.822265625", 153954304},
             {"587.2890625", 615817216},
             {"186.822265625", 195897344},
             {"0.00000095367431640625", 1},
             {"0.000000953674316406250000", 1},
             {"000146.822265625000", 153954304},
             {"64.0000", 64ULL << 20},
             {"000.0000", 0},
         }) {
        const ServeOptions exact_host_context =
            parse({"ninfer-serve", "model.ninfer", "--host-context-mib", mib});
        failures += check(exact_host_context.launch.engine.context_cache.host_capacity_bytes == bytes,
                          ("Host context MiB did not preserve exact bytes: " + mib).c_str());
    }

    constexpr std::size_t bytes_per_mib  = 1ULL << 20;
    constexpr std::size_t max_host_bytes = std::numeric_limits<std::size_t>::max();
    std::string maximum_host_mib         = std::to_string(max_host_bytes / bytes_per_mib);
    std::size_t fractional_bytes         = max_host_bytes % bytes_per_mib;
    if (fractional_bytes != 0) { maximum_host_mib += '.'; }
    while (fractional_bytes != 0) {
        fractional_bytes *= 10;
        maximum_host_mib += static_cast<char>('0' + fractional_bytes / bytes_per_mib);
        fractional_bytes %= bytes_per_mib;
    }
    const ServeOptions maximum_host_context =
        parse({"ninfer-serve", "model.ninfer", "--host-context-mib", maximum_host_mib});
    failures += check(maximum_host_context.launch.engine.context_cache.host_capacity_bytes == max_host_bytes,
                      "maximum size_t Host context capacity was not accepted exactly");

    bool overflowing_host_context_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--host-context-mib",
                     std::to_string(std::numeric_limits<std::size_t>::max() / (1ULL << 20) + 1)});
    } catch (const std::invalid_argument&) { overflowing_host_context_rejected = true; }
    failures +=
        check(overflowing_host_context_rejected, "overflowing Host context capacity was accepted");

    for (const std::string mib : {"0.1", "0.0000001", "0.000000476837158203125", "-1", "-0", "+1",
                                  "nan", "inf", "1e3", "1.", ".5", "", " 1", "1 ", "1.2.3"}) {
        bool invalid_host_context_rejected = false;
        try {
            (void)parse({"ninfer-serve", "model.ninfer", "--host-context-mib", mib});
        } catch (const std::invalid_argument&) { invalid_host_context_rejected = true; }
        failures +=
            check(invalid_host_context_rejected,
                  ("invalid or fractional-byte Host context MiB was accepted: " + mib).c_str());
    }

    for (const std::string option : {"--max-request-mib", "--media-cache-mib", "--media-live-mib",
                                     "--response-store-max-mib"}) {
        bool fractional_mib_rejected = false;
        try {
            (void)parse({"ninfer-serve", "model.ninfer", option, "1.5"});
        } catch (const std::invalid_argument&) { fractional_mib_rejected = true; }
        failures += check(fractional_mib_rejected,
                          ("integer-only MiB option accepted a fraction: " + option).c_str());
    }

    for (const bool disable_first : {false, true}) {
        std::vector<std::string> arguments = {"ninfer-serve", "model.ninfer"};
        if (disable_first) { arguments.push_back("--no-prefix-reuse"); }
        arguments.insert(arguments.end(),
                         {"--device-state-slots", "3", "--host-context-mib", "64"});
        if (!disable_first) { arguments.push_back("--no-prefix-reuse"); }
        const ServeOptions disabled_cache = parse(std::move(arguments));
        failures +=
            check(!disabled_cache.launch.engine.context_cache.enabled &&
                      disabled_cache.launch.engine.context_cache.device_state_slots == 3 &&
                      disabled_cache.launch.engine.context_cache.host_capacity_bytes == (64ULL << 20),
                  "--no-prefix-reuse did not preserve independently configured capacities");
    }

    const ServeOptions response_store =
        parse({"ninfer-serve", "model.ninfer", "--response-store-max-records", "42",
               "--response-store-max-mib", "8"});
    failures += check(response_store.response_store_max_records == 42 &&
                          response_store.response_store_max_bytes == (8ULL << 20),
                      "Responses store limits did not reach serving options");

    const ServeOptions sampling =
        parse({"ninfer-serve", "model.ninfer", "--temperature", "0", "--top-p", "0.9", "--top-k",
               "20", "--min-p", "0.1", "--presence-penalty", "1.25", "--frequency-penalty", "-0.5",
               "--seed", "0"});
    failures += check(sampling.launch.generation.sampling.temperature == 0.0F &&
                          sampling.launch.generation.sampling.top_p == 0.9F &&
                          sampling.launch.generation.sampling.top_k == 20 &&
                          sampling.launch.generation.sampling.min_p == 0.1F &&
                          sampling.launch.generation.sampling.presence_penalty == 1.25F &&
                          sampling.launch.generation.sampling.frequency_penalty == -0.5F &&
                          sampling.launch.generation.sampling.seed == 0,
                      "server sampling flags did not preserve explicit values and zeros");
    bool oversized_top_k_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--top-k", "21"});
    } catch (const std::invalid_argument&) { oversized_top_k_rejected = true; }
    failures += check(oversized_top_k_rejected,
                      "server accepted top_k beyond the executable candidate domain");

    GenerationRequest request;
    request.max_tokens   = 1;
    const auto semantics = resolve_prompt_semantics(request, defaults);
    failures += check(!semantics.reasoning_effort && !semantics.enable_thinking &&
                          !semantics.reasoning_effort,
                      "omitted reasoning effort did not resolve to the template default");
    failures +=
        check(to_request_options(request, defaults, semantics, true).execution.allow_prefix_reuse,
              "resolved read-write cache policy did not reach Engine options");
    failures +=
        check(!to_request_options(request, defaults, semantics, false).execution.allow_prefix_reuse,
              "resolved disabled cache policy inherited external enablement");
    const ninfer::RequestOptions inherited_sampling =
        to_request_options(request, sampling, semantics, sampling.launch.engine.context_cache.enabled);
    failures += check(inherited_sampling.execution.sampling.temperature == 0.0F &&
                          inherited_sampling.execution.sampling.top_p == 0.9F &&
                          inherited_sampling.execution.sampling.seed == 0,
                      "server sampling overrides did not reach Engine options");
    request.sampling.temperature = 1.1;
    failures += check(to_request_options(request, sampling, semantics, sampling.launch.engine.context_cache.enabled)
                              .execution.sampling.temperature == 1.1F,
                      "request sampling override did not win over the server override");
    failures += check(
        to_request_options(request, thinking_budget, semantics, thinking_budget.launch.engine.context_cache.enabled)
                .execution.thinking.budget == 37,
        "thinking-enabled request did not inherit the server budget");
    request.enable_thinking = false;
    const auto non_thinking = resolve_prompt_semantics(request, thinking_budget);
    failures += check(!non_thinking.reasoning_effort,
                      "disabled thinking retained an effective reasoning effort");
    failures += check(!to_request_options(request, thinking_budget, non_thinking,
                                          thinking_budget.launch.engine.context_cache.enabled)
                           .execution.thinking.budget,
                      "non-thinking request inherited the server thinking budget");
    request.enable_thinking.reset();
    request.reasoning_effort   = RequestedReasoningEffort::Low;
    const auto explicit_effort = resolve_prompt_semantics(request, defaults);
    failures += check(explicit_effort.reasoning_effort == ninfer::ReasoningEffort::Low &&
                          explicit_effort.enable_thinking == true,
                      "explicit reasoning effort did not remain the effective effort");
    request.reasoning_effort.reset();
    failures += check(resolve_prompt_semantics(request, configured).preserve_thinking == true,
                      "server preserve-thinking default was not resolved");
    request.preserve_thinking = false;
    failures += check(resolve_prompt_semantics(request, configured).preserve_thinking == false,
                      "request preserve-thinking override did not win");

    failures +=
        check(ninfer::product::startup_usage_text("ninfer-serve", ninfer::product::EntryPoint::Serve).find("--no-prefix-reuse") != std::string::npos,
              "serve help omits --no-prefix-reuse");
    failures +=
        check(ninfer::product::startup_usage_text("ninfer-serve", ninfer::product::EntryPoint::Serve).find("--host-context-mib") != std::string::npos,
              "serve help omits context-cache capacities");
    failures += check(ninfer::product::startup_usage_text("ninfer-serve", ninfer::product::EntryPoint::Serve).find("device-state=max-concurrency") !=
                          std::string::npos,
                      "serve help omits context-cache defaults");
    failures +=
        check(ninfer::product::startup_usage_text("ninfer-serve", ninfer::product::EntryPoint::Serve).find("--preserve-thinking") != std::string::npos,
              "serve help omits --preserve-thinking");
    failures += check(ninfer::product::startup_usage_text("ninfer-serve", ninfer::product::EntryPoint::Serve).find("--default-thinking-budget") !=
                          std::string::npos,
                      "serve help omits --default-thinking-budget");
    failures += check(ninfer::product::startup_usage_text("ninfer-serve", ninfer::product::EntryPoint::Serve).find("--vision") != std::string::npos,
                      "serve help omits --vision");
    failures +=
        check(ninfer::product::startup_usage_text("ninfer-serve", ninfer::product::EntryPoint::Serve).find("--log-stats-interval-ms") != std::string::npos,
              "serve help omits --log-stats-interval-ms");
    failures += check(ninfer::product::startup_usage_text("ninfer-serve", ninfer::product::EntryPoint::Serve).find("--log-level") != std::string::npos,
                      "serve help omits the log-level control");
    failures += check(ninfer::product::startup_usage_text("ninfer-serve", ninfer::product::EntryPoint::Serve).find("--media-preprocess-threads") !=
                          std::string::npos,
                      "serve help omits media preparation controls");
    failures += check(ninfer::product::startup_usage_text("ninfer-serve", ninfer::product::EntryPoint::Serve).find("--kv-capacity") != std::string::npos,
                      "serve help omits --kv-capacity");
    failures += check(ninfer::product::startup_usage_text("ninfer-serve", ninfer::product::EntryPoint::Serve).find("--response-store-max-mib") !=
                          std::string::npos,
                      "serve help omits Responses store limits");
    failures +=
        check(ninfer::product::startup_usage_text("ninfer-serve", ninfer::product::EntryPoint::Serve).find("--context-cost-presets") != std::string::npos,
              "serve help omits external context-cost presets");
    failures += check(ninfer::product::startup_usage_text("ninfer-serve", ninfer::product::EntryPoint::Serve).find("metadata.name") != std::string::npos,
                      "serve help omits the artifact-derived model id default");

    const ServeOptions inherited =
        parse({"ninfer-serve", "model.ninfer", "--max-context", "16384"});
    failures += check(inherited.launch.engine.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
                          inherited.launch.engine.kv_capacity.explicit_tokens == 16384,
                      "omitted --kv-capacity did not follow --max-context");

    const ServeOptions automatic = parse({"ninfer-serve", "model.ninfer", "--kv-capacity", "auto"});
    failures += check(automatic.launch.engine.kv_capacity.mode == ninfer::KvCapacityMode::Automatic &&
                          automatic.launch.engine.kv_capacity.explicit_tokens == 0 &&
                          automatic.launch.engine.kv_capacity.automatic_headroom_bytes ==
                              ninfer::kDefaultKvSizingHeadroomBytes,
                      "--kv-capacity auto did not select automatic sizing");
    failures += check(automatic.launch.engine.startup_planning_headroom_bytes ==
                              ninfer::kDefaultStartupPlanningHeadroomBytes,
                      "the startup planning headroom default did not reach the engine");
    // T020 J3: the two headrooms are separate fields with separate names; the former spelling is
    // accepted for one release and still means KV sizing.
    failures +=
        check(ninfer::kDefaultKvCapacityHeadroomBytes == ninfer::kDefaultKvSizingHeadroomBytes,
              "the legacy KV headroom constant changed value");

    // An explicit arena below max_context would let admission accept a request the arena cannot
    // hold, so the server refuses it up front. `auto` has no such check because its floor is the
    // capacity curve's minimum, which is one full context by construction.
    bool kv_below_context_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--max-context", "16384", "--kv-capacity",
                     "8192"});
    } catch (const std::invalid_argument&) { kv_below_context_rejected = true; }
    failures += check(kv_below_context_rejected,
                      "explicit --kv-capacity below --max-context was accepted");

    const ServeOptions automatic_headroom =
        parse({"ninfer-serve", "model.ninfer", "--kv-capacity", "auto",
               "--kv-sizing-headroom-mib", "512"});
    failures +=
        check(automatic_headroom.launch.engine.kv_capacity.mode == ninfer::KvCapacityMode::Automatic &&
                  automatic_headroom.launch.engine.kv_capacity.automatic_headroom_bytes == (512ULL << 20),
              "--kv-sizing-headroom-mib did not reach the automatic policy");
    const ServeOptions legacy_headroom =
        parse({"ninfer-serve", "model.ninfer", "--kv-capacity", "auto",
               "--kv-capacity-headroom-mib", "768"});
    failures +=
        check(legacy_headroom.launch.engine.kv_capacity.automatic_headroom_bytes == (768ULL << 20) &&
                  legacy_headroom.launch.engine.startup_planning_headroom_bytes ==
                      ninfer::kDefaultStartupPlanningHeadroomBytes,
              "the legacy --kv-capacity-headroom-mib spelling no longer means KV sizing alone");
    const ServeOptions planning_headroom =
        parse({"ninfer-serve", "model.ninfer", "--startup-planning-headroom-mib", "2048"});
    failures +=
        check(planning_headroom.launch.engine.startup_planning_headroom_bytes == (2ULL << 30) &&
                  planning_headroom.launch.engine.kv_capacity.automatic_headroom_bytes == 0,
              "--startup-planning-headroom-mib did not stay separate from KV sizing");
    failures +=
        check(ninfer::product::startup_usage_text("ninfer-serve", ninfer::product::EntryPoint::Serve)
                      .find("--startup-planning-headroom-mib") != std::string::npos &&
                  ninfer::product::startup_usage_text("ninfer-serve", ninfer::product::EntryPoint::Serve)
                          .find("--kv-sizing-headroom-mib") != std::string::npos,
              "serve help omits one of the split headroom controls");

    bool headroom_with_explicit_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--kv-capacity", "16384",
                     "--kv-sizing-headroom-mib", "512"});
    } catch (const std::invalid_argument&) { headroom_with_explicit_rejected = true; }
    failures += check(headroom_with_explicit_rejected,
                      "KV sizing headroom was accepted without --kv-capacity auto");

    const ServeOptions logged = parse({"ninfer-serve", "model.ninfer", "--request-log-jsonl",
                                       "requests.jsonl", "--api-key", "do-not-log"});
    failures += check(logged.launch.logging.request_jsonl == "requests.jsonl",
                      "--request-log-jsonl did not preserve its path");
    failures +=
        check(ninfer::product::startup_usage_text("ninfer-serve", ninfer::product::EntryPoint::Serve).find("--request-log-jsonl") != std::string::npos,
              "serve help omits --request-log-jsonl");
    bool secret_present    = false;
    bool redaction_present = false;
    for (const std::string& argument : logged.launch.provenance.argv) {
        secret_present    = secret_present || argument == "do-not-log";
        redaction_present = redaction_present || argument == "<redacted>";
    }
    failures += check(!secret_present, "startup argv retained the API key");
    failures += check(redaction_present, "startup argv omitted the API-key redaction marker");

    // The serve entry point reads the same TOML schema as the CLI, and keeps every local
    // serve-only knob (KV sizing headroom, eviction contract) addressable from it.
    {
        const auto path = std::filesystem::temp_directory_path() / "ninfer-serve-options-test.toml";
        {
            std::ofstream file(path, std::ios::binary | std::ios::trunc);
            file << "model.path = \"model.ninfer\"\n"
                    "server.host = \"0.0.0.0\"\n"
                    "server.port = 9090\n"
                    "engine.max_concurrency = 3\n"
                    "engine.kv_capacity = \"auto\"\n"
                    "engine.startup_planning_headroom_mib = 1536\n"
                    "engine.kv_capacity_headroom_mib = 256\n"
                    "context_cache.evict = \"auto\"\n";
        }
        const ServeOptions from_file = parse({"ninfer-serve", "--config", path.string()});
        failures += check(from_file.host == "0.0.0.0" && from_file.port == 9090 &&
                              from_file.launch.engine.max_concurrency == 3 &&
                              from_file.launch.engine.kv_capacity.mode ==
                                  ninfer::KvCapacityMode::Automatic &&
                              from_file.launch.engine.kv_capacity.automatic_headroom_bytes ==
                                  (256ULL << 20) &&
                              from_file.launch.engine.startup_planning_headroom_bytes ==
                                  (1536ULL << 20) &&
                              from_file.launch.engine.evict_contract ==
                                  ninfer::EvictionContract::Auto,
                          "serve TOML configuration did not reach the local options");
        const ServeOptions overridden =
            parse({"ninfer-serve", "--config", path.string(), "--port", "9191"});
        failures += check(overridden.port == 9191 &&
                              overridden.launch.provenance.fields.at("server.port").kind ==
                                  ninfer::product::ConfigSourceKind::CommandLine,
                          "command line did not override the serve TOML layer");
        std::filesystem::remove(path);
    }
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
