#include "ninfer/engine.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

// Port (sm_89): the dense counterpart of test_engine_moe_real.cpp. The shipped 27B artifact carries
// a vision component (441 vision bindings), so the tower is in the file and the default arm exercises
// it; NINFER_TEST_VISION=0 keeps a text-only arm for the release that has no tower.
bool vision_requested() {
    const char* value = std::getenv("NINFER_TEST_VISION");
    return value == nullptr || *value == '\0' || (*value != '0' && *value != 'n' && *value != 'N');
}

// The artifact's text config declares architectures = ["Qwen3_5ForCausalLM"] and
// model_type = "qwen3_5_text" (single dense stack, 64 layers, no routed experts). Asserting the name
// the loader reports is the contract; the MoE fixture asserts its own name the same way.
constexpr const char* kDenseArchitecture = "Qwen3_5ForCausalLM";

ninfer::EngineOptions engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path             = artifact;
    options.max_context               = 4096;
    options.kv_capacity               = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk             = 1024;
    options.kv_cache                  = ninfer::KvCacheStorage::Int8Group64;
    options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens  = 3;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    options.enable_vision             = vision_requested();
    options.use_cuda_graph            = true;
    return options;
}

ninfer::RequestOptions greedy_options(std::uint32_t outputs, bool reuse) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.sampling.seed           = 0;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

std::vector<std::uint8_t> gradient_ppm() {
    std::vector<std::uint8_t> ppm;
    const std::string header = "P6\n64 64\n255\n";
    ppm.insert(ppm.end(), header.begin(), header.end());
    for (int index = 0; index < 64 * 64; ++index) {
        ppm.push_back(static_cast<std::uint8_t>(index & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 3) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 7) & 0xff));
    }
    return ppm;
}

int report_loaded_product(const ninfer::Engine& engine) {
    const ninfer::LoadSummary load     = engine.load_summary();
    const ninfer::MemorySummary memory = engine.memory_summary();

    // Print before asserting: a capacity-tight run has to be diagnosable from its own log.
    std::cerr << "[dense] architecture=" << load.architecture << " model=" << load.model_name
              << " artifact_bytes=" << load.artifact_bytes_read
              << " h2d_bytes=" << load.host_to_device_bytes
              << " device_objects=" << load.device_object_count
              << " host_objects=" << load.host_object_count
              << " kv_pages=" << load.kv_main_page_groups << "/" << load.kv_max_main_page_groups
              << " kv_resolved_tokens=" << load.kv_resolved_tokens
              << " kv_reservation_bytes=" << load.kv_runtime_reservation_bytes << std::endl;
    std::cerr << "[dense] weights=" << memory.weights.used_bytes << "/"
              << memory.weights.capacity_bytes << " sequence=" << memory.sequence.used_bytes << "/"
              << memory.sequence.capacity_bytes << " workspace=" << memory.workspace.used_bytes
              << "/" << memory.workspace.capacity_bytes
              << " kv_payload=" << memory.kv_payload_bytes
              << " kv_cache=" << static_cast<int>(memory.kv_cache)
              << " graph_allowance=" << memory.cuda_graph_allowance_bytes
              << " available_after_weights=" << memory.available_after_weights_bytes
              << " free_after_startup=" << memory.free_after_startup_bytes << std::endl;

    if (load.architecture != kDenseArchitecture || load.model_name.empty() ||
        load.host_to_device_bytes == 0 || load.artifact_bytes_read < load.host_to_device_bytes) {
        std::cerr << "27B dense Engine construction has an invalid load summary: architecture="
                  << load.architecture << " model=" << load.model_name
                  << " artifact_bytes=" << load.artifact_bytes_read
                  << " h2d_bytes=" << load.host_to_device_bytes << '\n';
        return 1;
    }
    // The construction contract this fixture asks for, read back from the engine's own report.
    if (memory.max_context != 4096 || memory.kv_cache != ninfer::KvCacheStorage::Int8Group64) {
        std::cerr << "27B dense Engine resolved a different cache contract: context="
                  << memory.max_context
                  << " kv_cache=" << static_cast<int>(memory.kv_cache) << '\n';
        return 1;
    }
    // The whole product is resident here (no routed-expert streaming), so the KV resolution has to
    // have produced pages for the requested context instead of quietly serving a shorter one.
    if (load.kv_main_page_groups == 0 || load.kv_resolved_tokens < engine.options().max_context) {
        std::cerr << "27B dense Engine did not resolve the requested 4096-token KV: pages="
                  << load.kv_main_page_groups << " resolved_tokens=" << load.kv_resolved_tokens
                  << " requested=" << engine.options().max_context << '\n';
        return 1;
    }

    // The vision workspace exists exactly when this construction planned for it, which follows the
    // request, not the artifact.
    if (memory.vision_workspace.has_value() != vision_requested()) {
        std::cerr << "27B dense Engine vision workspace does not match the requested mode: requested="
                  << (vision_requested() ? 1 : 0)
                  << " reported=" << (memory.vision_workspace.has_value() ? 1 : 0) << '\n';
        return 1;
    }
    if (memory.weights.capacity_bytes == 0 || memory.weights.used_bytes == 0 ||
        memory.weights.used_bytes > memory.weights.capacity_bytes ||
        memory.sequence.capacity_bytes == 0 || memory.sequence.used_bytes == 0 ||
        memory.sequence.used_bytes > memory.sequence.capacity_bytes ||
        memory.workspace.capacity_bytes == 0) {
        std::cerr << "27B dense Engine construction has an invalid memory summary\n";
        return 1;
    }
    // A dense 27B leaves far less room for graph storage than the streamed 35B; the allowance is the
    // plan's static reservation and is independent of that headroom. A zero allowance means startup
    // planned no capture at all, which is a different mode than the one this fixture asked for.
    if (memory.cuda_graph_allowance_bytes == 0) {
        std::cerr << "27B dense Engine planned no CUDA Graph allowance while capture was requested\n";
        return 1;
    }
    return 0;
}

int exercise_text_mtp_and_prefix(ninfer::Engine& engine) {
    // Real tokens from the artifact tokenizer: the prompt never depends on this fixture guessing
    // token ids for a vocabulary it did not load.
    const std::vector<ninfer::TokenId> prompt =
        engine.tokenize_text("The quick brown fox jumps over the lazy dog.");
    if (prompt.empty()) {
        std::cerr << "27B dense artifact tokenizer produced an empty prompt\n";
        return 1;
    }

    const ninfer::GenerationResult first =
        engine.generate(engine.prepare_tokens(prompt), greedy_options(5, true));
    std::cerr << "[dense] text outputs=" << first.generated_token_ids.size()
              << " finish=" << static_cast<int>(first.finish_reason)
              << " mtp_rounds=" << first.speculative.rounds
              << " accepted=" << first.speculative.accepted_tokens << std::endl;
    if (first.generated_token_ids.size() != 5 ||
        first.finish_reason != ninfer::FinishReason::OutputLimit ||
        first.speculative.backend != ninfer::SpeculativeBackend::Mtp ||
        first.speculative.rounds == 0) {
        std::cerr << "27B dense fixed prompt did not complete through MTP: outputs="
                  << first.generated_token_ids.size()
                  << " finish=" << static_cast<int>(first.finish_reason)
                  << " mtp_rounds=" << first.speculative.rounds << '\n';
        return 1;
    }

    // Second request: one token, so both ends of the 1..5 token band this fixture covers are
    // exercised on the same engine instance. Prefix reuse is not asserted here - the append arm
    // below pins that contract, and this arm is about the output budget.
    const ninfer::GenerationResult single =
        engine.generate(engine.prepare_tokens(prompt), greedy_options(1, false));
    std::cerr << "[dense] single_token outputs=" << single.generated_token_ids.size()
              << " finish=" << static_cast<int>(single.finish_reason) << std::endl;
    if (single.generated_token_ids.size() != 1 ||
        single.finish_reason != ninfer::FinishReason::OutputLimit) {
        std::cerr << "27B dense one-token request did not finish its output budget\n";
        return 1;
    }

    // Append/prefix reuse: the same contract the MoE fixture pins, on the dense stack.
    std::vector<ninfer::TokenId> continuation = prompt;
    continuation.insert(continuation.end(), first.generated_token_ids.begin(),
                        first.generated_token_ids.end());
    continuation.push_back(198);
    const ninfer::GenerationResult reused =
        engine.generate(engine.prepare_tokens(continuation), greedy_options(2, true));
    const std::uint32_t expected_reuse =
        static_cast<std::uint32_t>(prompt.size() + first.generated_token_ids.size() - 1);
    std::cerr << "[dense] append reused=" << reused.reused_prompt_tokens
              << " expected=" << expected_reuse
              << " outputs=" << reused.generated_token_ids.size() << std::endl;
    if (reused.reused_prompt_tokens != expected_reuse || reused.generated_token_ids.size() != 2) {
        std::cerr << "27B dense append/prefix reuse is incorrect: reused="
                  << reused.reused_prompt_tokens << " expected=" << expected_reuse
                  << " outputs=" << reused.generated_token_ids.size() << '\n';
        return 1;
    }
    return 0;
}

int exercise_vision(ninfer::Engine& engine) {
    engine.reset_memory_peaks();
    const ninfer::MemorySummary before = engine.memory_summary();
    if (!before.vision_workspace || before.vision_workspace->handoff_active_bytes != 0 ||
        before.vision_workspace->handoff_peak_bytes != 0) {
        std::cerr << "27B dense Vision handoff peak did not reset before Vision\n";
        return 1;
    }
    std::cerr << "[dense] vision before: general_capacity="
              << before.vision_workspace->general_capacity_bytes
              << " handoff_offset=" << before.vision_workspace->handoff_offset_bytes
              << " handoff_capacity=" << before.vision_workspace->handoff_capacity_bytes
              << " handoff_active=" << before.vision_workspace->handoff_active_bytes
              << " handoff_peak=" << before.vision_workspace->handoff_peak_bytes << std::endl;

    ninfer::MessagePart image;
    image.kind              = ninfer::MessagePartKind::Media;
    image.media.kind        = ninfer::MediaKind::Image;
    image.media.bytes       = gradient_ppm();
    image.media.media_type  = "image/x-portable-pixmap";
    image.media.source_name = "inline.ppm";

    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(std::move(image));
    message.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = "What is visible?", .media = {}});

    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;

    const ninfer::GenerationResult result =
        engine.generate(engine.prepare(std::move(input)), greedy_options(1, false));
    const ninfer::MemorySummary after = engine.memory_summary();
    if (after.vision_workspace) {
        std::cerr << "[dense] vision after: prompt_tokens="
                  << after.vision_workspace->aggregate_prompt_tokens
                  << " max_item_tokens=" << after.vision_workspace->max_item_tokens
                  << " general_capacity=" << after.vision_workspace->general_capacity_bytes
                  << " encode_peak=" << after.vision_workspace->encode_peak_bytes
                  << " handoff_offset=" << after.vision_workspace->handoff_offset_bytes
                  << " handoff_capacity=" << after.vision_workspace->handoff_capacity_bytes
                  << " handoff_active=" << after.vision_workspace->handoff_active_bytes
                  << " handoff_peak=" << after.vision_workspace->handoff_peak_bytes
                  << " workspace_logical_peak=" << after.workspace_logical_peak_bytes
                  << " workspace_capacity=" << after.workspace.capacity_bytes << std::endl;
    }
    if (!result.prompt.has_media || result.generated_token_ids.size() != 1 ||
        result.finish_reason != ninfer::FinishReason::OutputLimit) {
        std::cerr << "27B dense Vision request did not complete through the public Engine: media="
                  << (result.prompt.has_media ? 1 : 0)
                  << " outputs=" << result.generated_token_ids.size()
                  << " finish=" << static_cast<int>(result.finish_reason) << '\n';
        return 1;
    }
    if (!after.vision_workspace ||
        after.vision_workspace->handoff_offset_bytes !=
            before.vision_workspace->handoff_offset_bytes ||
        after.vision_workspace->handoff_capacity_bytes !=
            before.vision_workspace->handoff_capacity_bytes ||
        after.vision_workspace->handoff_active_bytes != 0 ||
        after.vision_workspace->handoff_peak_bytes == 0 ||
        after.vision_workspace->handoff_peak_bytes >
            after.vision_workspace->handoff_capacity_bytes ||
        after.workspace_logical_peak_bytes == 0 ||
        after.workspace_logical_peak_bytes > after.workspace.capacity_bytes) {
        std::cerr << "27B dense Vision request did not use the startup-frozen workspace handoff\n";
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }

    // port (sm_89): progress markers. Each step reports itself before running, and a construction
    // failure (memory budget, unsupported option) reports its message instead of terminating
    // silently.
    std::cerr << "[step] constructing engine" << std::endl;
    try {
        ninfer::Engine engine(engine_options(artifact));
        std::cerr << "[step] report_loaded_product" << std::endl;
        if (const int result = report_loaded_product(engine); result != 0) { return result; }
        std::cerr << "[step] exercise_text_mtp_and_prefix" << std::endl;
        if (const int result = exercise_text_mtp_and_prefix(engine); result != 0) { return result; }
        if (engine.memory_summary().vision_workspace.has_value()) {
            std::cerr << "[step] exercise_vision" << std::endl;
            if (const int result = exercise_vision(engine); result != 0) { return result; }
        } else {
            std::cerr << "[step] exercise_vision skipped: not requested by this fixture "
                         "(set NINFER_TEST_VISION=0 for a text-only arm)\n";
        }
    } catch (const std::exception& error) {
        const std::string message = error.what();
        // 27B dense is 19 GiB of resident weights on a 24 GiB card: a device that cannot hold the
        // reservation is a capacity condition, not a correctness failure - same treatment as the
        // MoE fixture's reservation check.
        if (message.find("reservation") != std::string::npos ||
            message.find("runtime capacity") != std::string::npos ||
            message.find("out of memory") != std::string::npos ||
            message.find("memory allocation") != std::string::npos) {
            std::cerr << "skip: " << message << std::endl;
            return 77;
        }
        std::cerr << "engine construction failed: " << message << std::endl;
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
