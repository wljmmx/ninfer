#include "ninfer/engine.h"

#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
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

// port (sm_89): the overrun arm of the KV residency contract.
//
// A prompt longer than the Device pool's resident page groups cannot be served at all: the
// full-attention layers read the whole history every step, so nothing can be streamed from the Host
// tier mid-step. The refusal must therefore arrive as an ERROR that names the two token counts, in
// bounded time, and the engine must still serve the next request - not be a request that runs
// forever. Measured by hand before this test existed: a 60,001-token prompt against a 59,648-token
// resident set was still outstanding after an hour. The capacity path answered the shortfall at the
// first bind (2 main + 1 backend page) by PAUSING the very lane that could never fit, and the request
// then sat in the paused set with nothing driving it back (`paused=1 resident=0 pending=0`, still
// true 240 s later), so neither the Program's own refusal nor any completion path was ever reached.
// port (sm_89): the swap arm. The residency counters are only interesting if something actually
// crosses the tier boundary, and a single short request never does: it touches one page group, which
// is resident, so fetched_pages stays 0 by construction. This arm builds the one legal pressure
// scenario the design has - a retained prefix that must give up its Device pages so a later request
// can be prefilled, and a third request that reuses that prefix, which has to FETCH it back:
//
//   R1  prefix A (4K tokens), reuse on  -> its checkpoint is retained, Device-resident
//   R2  filler B that fills the pool    -> the admission of B's last chunks runs out of Device pages,
//                                          so the retained checkpoint is DEMOTED to the Host tier
//   R3  prefix A again, reuse on        -> binding fetches those pages back (Host -> Device)
//
// The correctness check is the one the tier contract already uses: R3 must produce the same greedy
// tokens as R1, because the KV it reads is the same KV.
int exercise_256k_swap(ninfer::Engine& engine, std::uint32_t resident_pages,
                       const std::vector<ninfer::TokenId>& sample) {
    const std::uint64_t resident_tokens = static_cast<std::uint64_t>(resident_pages) * 64ULL;
    if (resident_tokens < 8192ULL) {
        std::cerr << "[256k] swap arm needs a pool of at least 8K resident tokens, has "
                  << resident_tokens << '\n';
        return 1;
    }
    const auto make_prompt = [&engine](std::uint32_t tokens, const char* seed) {
        std::string text = seed;
        while (text.size() <= static_cast<std::size_t>(tokens) * 8ULL) { text += seed; }
        std::vector<ninfer::TokenId> prompt = engine.tokenize_text(text);
        if (prompt.size() > tokens) { prompt.resize(tokens); }
        return prompt;
    };
    const std::uint32_t prefix_tokens = 4096;
    // Fill all but a few page groups: the last chunks of B are what force the demotion.
    const std::uint64_t filler_tokens =
        resident_tokens > 1024ULL ? resident_tokens - 512ULL : resident_tokens * 3ULL / 4ULL;
    const std::vector<ninfer::TokenId> prefix = make_prompt(prefix_tokens, "Alpha bravo charlie delta echo foxtrot. ");
    const std::vector<ninfer::TokenId> filler = make_prompt(
        static_cast<std::uint32_t>(filler_tokens), "Zulu yankee xray whiskey victor uniform tango. ");

    const auto counters = [&engine](const char* tag) {
        const ninfer::MemorySummary memory = engine.memory_summary();
        std::cerr << "[256k] swap " << tag << ": spilled_pages=" << memory.kv_spilled_pages
                  << " fetched_pages=" << memory.kv_fetched_pages
                  << " host_occupied=" << memory.host_context_occupied_bytes
                  << " host_capacity=" << memory.host_context_capacity_bytes << std::endl;
        return memory;
    };
    const auto clock = std::chrono::steady_clock::now();
    const auto elapsed = [&clock] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - clock).count();
    };

    (void)counters("before");
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare_tokens(prefix), greedy_options(2, true));
    std::cerr << "[256k] swap R1: prompt=" << first.prompt.prompt_tokens
              << " reused=" << first.reused_prompt_tokens
              << " outputs=" << first.generated_token_ids.size() << " t=" << elapsed() << " s"
              << std::endl;
    if (first.generated_token_ids.size() != 2) {
        std::cerr << "[256k] swap R1 did not generate two tokens\n";
        return 1;
    }
    const auto after_first = counters("R1");

    const ninfer::GenerationResult filling =
        engine.generate(engine.prepare_tokens(filler), greedy_options(1, false));
    std::cerr << "[256k] swap R2: prompt=" << filling.prompt.prompt_tokens
              << " prefill_s=" << filling.timings.prefill_seconds << " t=" << elapsed() << " s"
              << std::endl;
    const auto after_fill = counters("R2");

    const ninfer::GenerationResult reused =
        engine.generate(engine.prepare_tokens(prefix), greedy_options(2, true));
    std::cerr << "[256k] swap R3: prompt=" << reused.prompt.prompt_tokens
              << " reused=" << reused.reused_prompt_tokens
              << " computed=" << reused.computed_prefill_tokens
              << " ttft_s=" << reused.timings.first_token_seconds << " t=" << elapsed() << " s"
              << std::endl;
    const auto after_reuse = counters("R3");

    if (after_reuse.kv_fetched_pages <= after_fill.kv_fetched_pages) {
        std::cerr << "[256k] swap: no Host -> Device fetch was counted (" << after_fill.kv_fetched_pages
                  << " -> " << after_reuse.kv_fetched_pages << ")\n";
        return 1;
    }
    if (after_fill.host_context_occupied_bytes == 0 ||
        after_reuse.host_context_occupied_bytes == 0) {
        std::cerr << "[256k] swap: the Host KV tier never held the demoted pages\n";
        return 1;
    }
    if (reused.reused_prompt_tokens == 0) {
        std::cerr << "[256k] swap R3 did not reuse the retained prefix\n";
        return 1;
    }
    if (reused.generated_token_ids != first.generated_token_ids) {
        std::cerr << "[256k] swap R3 diverged from R1:";
        for (const auto token : first.generated_token_ids) { std::cerr << ' ' << token; }
        std::cerr << " vs";
        for (const auto token : reused.generated_token_ids) { std::cerr << ' ' << token; }
        std::cerr << '\n';
        return 1;
    }
    std::cerr << "[256k] swap: spilled=" << after_fill.kv_spilled_pages
              << " fetched=" << (after_reuse.kv_fetched_pages - after_fill.kv_fetched_pages)
              << " pages, host_occupied=" << after_reuse.host_context_occupied_bytes
              << " B, tokens identical, total=" << elapsed() << " s" << std::endl;
    (void)after_first;
    return 0;
}

// port (sm_89): the two-lane pool options. Shared by the lanes arm and the growth arm so both price
// the same engine: what differs between them is the load, not the configuration.
ninfer::EngineOptions two_lane_options(const char* artifact) {
    ninfer::EngineOptions options = engine_options(artifact);
    options.max_context                      = 262144;
    options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(262144);
    options.prefill_chunk                    = 1024;
    options.max_concurrency                  = 2;
    options.max_pending_requests             = 4;
    options.context_cache.device_state_slots = 2;
    // What these arms are about is the SHARED pool, so they buy room for the second lane by dropping
    // the parts the contract does not need: the 19 GiB of weights plus TWO device state images leave
    // too little for the tiered KV anchor with the vision tower, the capture allowance and the MTP
    // backend pool all resident as well (measured: "0 bytes available for runtime capacity"). The
    // MTP-3 residency path is the 256K arm's job; here a plain decode keeps the step growth at one
    // token and the accounting easy to read.
    options.speculative.backend       = ninfer::SpeculativeBackend::None;
    options.speculative.draft_tokens  = 0;
    options.speculative.proposal_head = ninfer::ProposalHead::Full;
    options.enable_vision             = false;
    options.use_cuda_graph            = false;
#if defined(_WIN32)
    (void)_putenv_s("NINFER_KV_HOST_RESIDENCY", "1");
#else
    (void)setenv("NINFER_KV_HOST_RESIDENCY", "1", 1);
#endif
    return options;
}

// port (sm_89): the GROWTH probe - two lanes admitted together at ~49.8% of the pool each, whose next
// steps then cross the edge. The aggregate admission rule prices `live + one step`, so the pair is
// legal when it is admitted and only collides a few hundred tokens later.
//
// What this asserts is termination with a defined outcome, because the failure it guards is an
// unterminating load: before the growth refusal existed, the round that could not reserve a page
// paused one of the two lanes, the other kept decode_ready, and nothing ever drove the paused lane
// back (measured `paused=1 running=1 decode_ready=1 preemptions=48`, still there at 120 s). Either
// answer is acceptable - both lanes complete when there is room, or ONE is refused with the growth
// numbers while the other finishes - but the engine has to stay serviceable afterwards.
constexpr unsigned kGrowthWatchdogSeconds  = 120;
constexpr std::uint32_t kGrowthOutputs     = 260;
constexpr std::uint64_t kGrowthLanePerMille = 498; // ~49.8% of the resident set per lane

int run_growth_probe(ninfer::Engine& engine, std::uint32_t resident_pages, unsigned watchdog_seconds) {
    const std::uint64_t resident_tokens = static_cast<std::uint64_t>(resident_pages) * 64ULL;
    const std::uint64_t near            = resident_tokens * kGrowthLanePerMille / 1000ULL;
    const auto prompt_of                = [&engine](std::uint64_t tokens, const char* seed) {
        std::string text = seed;
        while (text.size() <= tokens * 8ULL) { text += seed; }
        std::vector<ninfer::TokenId> prompt = engine.tokenize_text(text);
        if (prompt.size() > tokens) { prompt.resize(static_cast<std::size_t>(tokens)); }
        return prompt;
    };
    const std::vector<ninfer::TokenId> tight_a =
        prompt_of(near, "Sierra tango uniform victor whiskey xray yankee zulu. ");
    const std::vector<ninfer::TokenId> tight_b =
        prompt_of(near, "Quebec romeo sierra tango uniform victor whiskey xray. ");
    std::cerr << "[256k-growth] two lanes at " << tight_a.size() << "/" << tight_b.size()
              << " tokens of a " << resident_tokens << "-token pool" << std::endl;

    const auto started  = std::chrono::steady_clock::now();
    auto a              = engine.submit(engine.prepare_tokens(tight_a), greedy_options(kGrowthOutputs, false));
    auto b              = engine.submit(engine.prepare_tokens(tight_b), greedy_options(kGrowthOutputs, false));
    a.wait_for_admission();
    b.wait_for_admission();
    std::string outcome_a = "completed";
    std::string outcome_b = "completed";
    std::size_t tokens_a  = 0;
    std::size_t tokens_b  = 0;
    std::mutex done_mutex;
    std::condition_variable done_cv;
    int drained = 0;
    const auto drain = [&](ninfer::GenerationHandle& handle, std::string& outcome,
                           std::size_t& tokens) {
        try {
            const ninfer::GenerationResult result = handle.wait();
            tokens                                = result.generated_token_ids.size();
            outcome = "outputs=" + std::to_string(tokens);
        } catch (const std::exception& error) {
            outcome = std::string("error: ") + error.what();
        }
        {
            const std::lock_guard lock(done_mutex);
            ++drained;
        }
        done_cv.notify_all();
    };
    std::thread drain_a([&] { drain(a, outcome_a, tokens_a); });
    std::thread drain_b([&] { drain(b, outcome_b, tokens_b); });
    // The client-visible answer is the contract: both requests have to reach a defined outcome. The
    // engine's own counters are reported with it, but a lingering idle lane must not be able to hide
    // a request that never returns.
    bool terminated = false;
    {
        std::unique_lock lock(done_mutex);
        terminated = done_cv.wait_for(lock, std::chrono::seconds(watchdog_seconds),
                                      [&] { return drained == 2; });
    }
    const double seconds             = std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                          started).count();
    const ninfer::RuntimeStats stats = engine.runtime_stats();
    std::cerr << "[256k-growth] after " << seconds << " s: drained=" << drained
              << " running=" << stats.running_requests
              << " decode_ready=" << stats.decode_ready_requests
              << " paused=" << stats.paused_requests << " replaying=" << stats.replaying_requests
              << " waiting=" << stats.waiting_requests
              << " terminal_pending=" << stats.terminal_pending_requests
              << " preemptions=" << stats.preemptions
              << " occupied_pages=" << stats.device_main_kv_occupied_pages
              << " resident_pages=" << resident_pages
              << " outstanding=" << engine.outstanding_requests() << std::endl;
    if (!terminated) {
        // Keep the log; do not wait on the engine's destructor - the two requests still own lanes.
        std::cerr << "[256k-growth] the growth probe did not terminate within " << watchdog_seconds
                  << " s (the mutual-pause shape the aggregate admission does not reach)\n";
        std::cerr.flush();
        std::_Exit(3);
    }
    drain_a.join();
    drain_b.join();
    std::cerr << "[256k-growth] done: A[" << outcome_a << "] B[" << outcome_b << "]" << std::endl;

    const bool a_done = outcome_a.starts_with("outputs=");
    const bool b_done = outcome_b.starts_with("outputs=");
    if (!a_done && !b_done) {
        std::cerr << "[256k-growth] neither lane was served\n";
        return 1;
    }
    if (!a_done || !b_done) {
        const std::string& refused = a_done ? outcome_b : outcome_a;
        for (const char* field : {"growth_pages_needed=", "capacity_pages=", "live_lanes="}) {
            if (refused.find(field) == std::string::npos) {
                std::cerr << "[256k-growth] the refusal does not report " << field << ": " << refused
                          << '\n';
                return 1;
            }
        }
        std::cerr << "[256k-growth] one lane was refused at the shared edge, the other finished\n";
    }

    // The refusal (or the completed pair) must leave the engine able to serve a new request.
    const std::vector<ninfer::TokenId> sample =
        engine.tokenize_text("The quick brown fox jumps over the lazy dog.");
    const ninfer::GenerationResult after =
        engine.generate(engine.prepare_tokens(sample), greedy_options(1, false));
    std::cerr << "[256k-growth] after the probe, one lane: outputs="
              << after.generated_token_ids.size()
              << " finish=" << static_cast<int>(after.finish_reason) << std::endl;
    if (after.generated_token_ids.size() != 1 ||
        after.finish_reason != ninfer::FinishReason::OutputLimit) {
        std::cerr << "[256k-growth] the engine did not serve a normal request afterwards\n";
        return 1;
    }
    return 0;
}

int exercise_256k_growth(const char* artifact) {
    ninfer::EngineOptions options = two_lane_options(artifact);
    std::cerr << "[256k-growth] constructing a two-lane 262144-token engine" << std::endl;
    std::unique_ptr<ninfer::Engine> engine;
    std::string failure;
    for (int attempt = 1; attempt <= 3 && engine == nullptr; ++attempt) {
        try {
            engine = std::make_unique<ninfer::Engine>(options);
        } catch (const std::exception& error) {
            failure             = error.what();
            const bool capacity = failure.find("reservation") != std::string::npos ||
                                  failure.find("runtime capacity") != std::string::npos ||
                                  failure.find("out of memory") != std::string::npos ||
                                  failure.find("memory allocation") != std::string::npos;
            if (!capacity) {
                std::cerr << "[256k-growth] construction failed: " << failure << '\n';
                return 1;
            }
            std::cerr << "[256k-growth] attempt " << attempt << " found no Device room: " << failure
                      << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(10));
        }
    }
    if (engine == nullptr) {
        std::cerr << "[256k-growth] INFEASIBLE: " << failure << '\n';
        return 77;
    }
    const ninfer::LoadSummary load = engine->load_summary();
    std::cerr << "[256k-growth] logical_pages=" << load.kv_logical_page_groups
              << " resident_pages=" << load.kv_resident_page_groups << std::endl;
    if (load.kv_resident_page_groups == 0 ||
        load.kv_resident_page_groups >= load.kv_logical_page_groups) {
        std::cerr << "[256k-growth] no residency split to share\n";
        return 1;
    }
    return run_growth_probe(*engine, load.kv_resident_page_groups, kGrowthWatchdogSeconds);
}

// port (sm_89): the SHARED-pool arm (max_concurrency = 2). Under residency the Device pool holds
// only the resident part of max_context, and that pool is ONE physical page pool for every lane:
// two requests that each fit can still over-subscribe it together. Before the aggregate admission
// rule the second lane was admitted on its own length alone, both lanes then failed to reserve a
// page, each paused itself in reserve_resident_units(), and nothing ever drove either back - the
// load was measured non-terminating.
//
// This arm is the acceptance for the aggregate rule, in the order a client meets it:
//
//   R1, R2  two lanes that fit TOGETHER (each ~40% of the resident set), submitted before either
//           has finished, admitted, and both must complete;
//   R3      one request that is legal on its own (~70% of the resident set, under the single-request
//           bound) but does not fit beside R1+R2: it must be REFUSED at admission, in bounded time,
//           with the per-lane and the aggregate numbers, while R1 and R2 keep running untouched;
//   R4      a normal short request after the refusal: the engine still serves it.
//
// Correctness is pinned on the shape-independent property: the same two-lane batch run twice must
// produce the same greedy tokens (the 1-row and 2-row decode compositions are known to differ - see
// the merged-prefill fixture - so a cross-shape comparison would pin the composition, not the pool).
int exercise_256k_lanes(const char* artifact) {
    constexpr double kAggregateRefusalBudgetSeconds = 1.0;
    // Enough outputs for an engine-reported decode rate per lane, short enough that the arm stays a
    // contract test rather than a benchmark.
    constexpr std::uint32_t kPairOutputs = 32;
    ninfer::EngineOptions options        = two_lane_options(artifact);
    std::cerr << "[256k-lanes] constructing a two-lane 262144-token engine" << std::endl;
    std::unique_ptr<ninfer::Engine> engine;
    std::string failure;
    for (int attempt = 1; attempt <= 3 && engine == nullptr; ++attempt) {
        try {
            engine = std::make_unique<ninfer::Engine>(options);
        } catch (const std::exception& error) {
            failure             = error.what();
            const bool capacity = failure.find("reservation") != std::string::npos ||
                                  failure.find("runtime capacity") != std::string::npos ||
                                  failure.find("out of memory") != std::string::npos ||
                                  failure.find("memory allocation") != std::string::npos;
            if (!capacity) {
                std::cerr << "[256k-lanes] construction failed: " << failure << '\n';
                return 1;
            }
            std::cerr << "[256k-lanes] attempt " << attempt << " found no Device room: " << failure
                      << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(10));
        }
    }
    if (engine == nullptr) {
        std::cerr << "[256k-lanes] INFEASIBLE: " << failure << '\n';
        return 77;
    }

    const ninfer::LoadSummary load = engine->load_summary();
    std::cerr << "[256k-lanes] logical_pages=" << load.kv_logical_page_groups
              << " resident_pages=" << load.kv_resident_page_groups
              << " resolved_tokens=" << load.kv_resolved_tokens
              << " device_kv=" << load.kv_device_payload_bytes
              << " host_kv=" << load.kv_host_payload_bytes << std::endl;
    if (load.kv_resident_page_groups == 0 ||
        load.kv_resident_page_groups >= load.kv_logical_page_groups) {
        std::cerr << "[256k-lanes] no residency split to share: resident="
                  << load.kv_resident_page_groups << " logical=" << load.kv_logical_page_groups
                  << '\n';
        return 1;
    }
    // The resident set in tokens, page-exact: this is the pool every lane shares.
    const std::uint64_t resident_tokens =
        static_cast<std::uint64_t>(load.kv_resident_page_groups) * 64ULL;
    const auto prompt_of = [&engine](std::uint64_t tokens, const char* seed) {
        std::string text = seed;
        while (text.size() <= tokens * 8ULL) { text += seed; }
        std::vector<ninfer::TokenId> prompt = engine->tokenize_text(text);
        if (prompt.size() > tokens) { prompt.resize(static_cast<std::size_t>(tokens)); }
        return prompt;
    };
    // Two lanes at 40% each leave ~20% of the pool free, so the pair fits and the pool can still
    // absorb the one-step growth the admission rule charges for both lanes.
    const std::uint64_t lane_tokens    = resident_tokens * 4ULL / 10ULL;
    const std::uint64_t refused_tokens = resident_tokens * 7ULL / 10ULL;
    const std::vector<ninfer::TokenId> lane_a = prompt_of(lane_tokens, "Alpha bravo charlie delta echo foxtrot golf hotel. ");
    const std::vector<ninfer::TokenId> lane_b = prompt_of(lane_tokens, "India juliet kilo lima mike november oscar papa. ");
    const std::vector<ninfer::TokenId> over   = prompt_of(refused_tokens, "The quick brown fox jumps over the lazy dog. ");
    std::cerr << "[256k-lanes] resident_tokens=" << resident_tokens << " lane_tokens=" << lane_a.size()
              << "/" << lane_b.size() << " refused_prompt_tokens=" << over.size() << std::endl;

    const auto clock = std::chrono::steady_clock::now();
    const auto elapsed = [&clock] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - clock).count();
    };
    const auto pair = [&](const char* tag, std::vector<ninfer::TokenId>& first,
                          std::vector<ninfer::TokenId>& second) {
        auto a = engine->submit(engine->prepare_tokens(lane_a), greedy_options(kPairOutputs, false));
        auto b = engine->submit(engine->prepare_tokens(lane_b), greedy_options(kPairOutputs, false));
        a.wait_for_admission();
        b.wait_for_admission();
        const ninfer::GenerationResult ra = a.wait();
        const ninfer::GenerationResult rb = b.wait();
        std::cerr << "[256k-lanes] " << tag << ": A outputs=" << ra.generated_token_ids.size()
                  << " prompt=" << ra.prompt.prompt_tokens
                  << " prefill_s=" << ra.timings.prefill_seconds
                  << " decode_s=" << ra.timings.decode_seconds << " | B outputs="
                  << rb.generated_token_ids.size() << " prompt=" << rb.prompt.prompt_tokens
                  << " prefill_s=" << rb.timings.prefill_seconds
                  << " decode_s=" << rb.timings.decode_seconds << " t=" << elapsed() << " s"
                  << std::endl;
        if (ra.generated_token_ids.size() != kPairOutputs ||
            rb.generated_token_ids.size() != kPairOutputs ||
            ra.finish_reason != ninfer::FinishReason::OutputLimit ||
            rb.finish_reason != ninfer::FinishReason::OutputLimit) {
            std::cerr << "[256k-lanes] " << tag << ": a lane did not complete its token budget (A="
                      << ra.generated_token_ids.size() << " B=" << rb.generated_token_ids.size()
                      << ")\n";
            return false;
        }
        first  = ra.generated_token_ids;
        second = rb.generated_token_ids;
        return true;
    };

    // --- two lanes that fit together, with the shared-pool refusal in the middle -----------------
    std::vector<ninfer::TokenId> first_a, first_b, second_a, second_b;
    {
        auto a = engine->submit(engine->prepare_tokens(lane_a), greedy_options(kPairOutputs, false));
        auto b = engine->submit(engine->prepare_tokens(lane_b), greedy_options(kPairOutputs, false));
        a.wait_for_admission();
        b.wait_for_admission();
        // The refusal is delivered at the next scheduling boundary, so it is measured where the
        // engine is busy with SHORT rounds - both lanes decoding. Waiting for prefill would measure
        // the 1024-token chunk that happens to be in flight (~2 s observed) instead of the rule.
        const auto decode_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(300);
        while (engine->runtime_stats().decode_ready_requests < 2 &&
               std::chrono::steady_clock::now() < decode_deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        const ninfer::RuntimeStats busy = engine->runtime_stats();
        std::cerr << "[256k-lanes] both lanes admitted: prefilling=" << busy.prefilling_requests
                  << " decode_ready=" << busy.decode_ready_requests
                  << " device_main_kv_occupied_pages=" << busy.device_main_kv_occupied_pages
                  << " resident_pages=" << load.kv_resident_page_groups << std::endl;
        if (busy.decode_ready_requests < 2) {
            std::cerr << "[256k-lanes] the two lanes never reached a decode round together\n";
            return 1;
        }
        std::cerr << "[256k-lanes] both lanes decoding; submitting the over-subscribing request"
                  << std::endl;

        const auto refusal_started = std::chrono::steady_clock::now();
        bool refused               = false;
        std::string message;
        try {
            auto c = engine->submit(engine->prepare_tokens(over), greedy_options(1, false));
            (void)c.wait();
            std::cerr << "[256k-lanes] the over-subscribing request was SERVED" << std::endl;
        } catch (const std::exception& error) {
            refused = true;
            message = error.what();
        }
        const double refusal_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - refusal_started)
                .count();
        std::cerr << "[256k-lanes] aggregate refusal: refused=" << (refused ? 1 : 0)
                  << " seconds=" << refusal_seconds << " message=\"" << message << "\""
                  << std::endl;
        if (!refused) {
            std::cerr << "[256k-lanes] two lanes over-subscribed the resident pool and the second "
                         "request was not refused\n";
            return 1;
        }
        // Both the per-lane numbers and the aggregate must be in the message: the caller's remedy
        // depends on which lane holds what.
        for (const char* field : {"prompt_tokens=", "request_tokens=", "live_lanes=",
                                 "live_tokens=", "live_lane_tokens=", "aggregate_tokens=",
                                 "resident_tokens="}) {
            if (message.find(field) == std::string::npos) {
                std::cerr << "[256k-lanes] the refusal does not name " << field << ": " << message
                          << '\n';
                return 1;
            }
        }
        if (message.find("live_lanes=2") == std::string::npos) {
            std::cerr << "[256k-lanes] the refusal did not count both resident lanes: " << message
                      << '\n';
            return 1;
        }
        if (refusal_seconds > kAggregateRefusalBudgetSeconds) {
            std::cerr << "[256k-lanes] the aggregate refusal took " << refusal_seconds
                      << " s (budget " << kAggregateRefusalBudgetSeconds << " s)\n";
            return 1;
        }

        // The lane that was already resident must be untouched by the refusal: both complete.
        const ninfer::GenerationResult ra = a.wait();
        const ninfer::GenerationResult rb = b.wait();
        std::cerr << "[256k-lanes] after the refusal: A outputs="
                  << ra.generated_token_ids.size() << " prefill_s=" << ra.timings.prefill_seconds
                  << " decode_s=" << ra.timings.decode_seconds << " | B outputs="
                  << rb.generated_token_ids.size() << " prefill_s=" << rb.timings.prefill_seconds
                  << " decode_s=" << rb.timings.decode_seconds << std::endl;
        if (ra.generated_token_ids.size() != kPairOutputs ||
            rb.generated_token_ids.size() != kPairOutputs ||
            ra.finish_reason != ninfer::FinishReason::OutputLimit ||
            rb.finish_reason != ninfer::FinishReason::OutputLimit) {
            std::cerr << "[256k-lanes] the refusal disturbed the resident lanes (A="
                      << ra.generated_token_ids.size() << " B=" << rb.generated_token_ids.size()
                      << ")\n";
            return 1;
        }
        const double decode_a = ra.timings.decode_seconds > 0.0
                                    ? static_cast<double>(ra.generated_token_ids.size() - 1) /
                                          ra.timings.decode_seconds
                                    : 0.0;
        const double decode_b = rb.timings.decode_seconds > 0.0
                                    ? static_cast<double>(rb.generated_token_ids.size() - 1) /
                                          rb.timings.decode_seconds
                                    : 0.0;
        std::cerr << "[256k-lanes] two-lane decode: A=" << decode_a << " tok/s, B=" << decode_b
                  << " tok/s (engine-reported)" << std::endl;
        if (decode_a <= 0.0 || decode_b <= 0.0) {
            std::cerr << "[256k-lanes] no engine-reported decode rate for both lanes\n";
            return 1;
        }
        first_a = ra.generated_token_ids;
        first_b = rb.generated_token_ids;
        if (first_a == first_b) {
            std::cerr << "[256k-lanes] both lanes returned the same tokens for different prompts\n";
            return 1;
        }
    }

    // --- the same two-lane batch again: same shape, so the greedy tokens must be identical -------
    if (!pair("pair#2", second_a, second_b)) { return 1; }
    if (first_a != second_a || first_b != second_b) {
        std::cerr << "[256k-lanes] the shared pool changed the tokens between two identical "
                     "two-lane batches\n";
        return 1;
    }

    // --- a normal request after the refusal: the engine still serves it --------------------------
    const std::vector<ninfer::TokenId> sample =
        engine->tokenize_text("The quick brown fox jumps over the lazy dog.");
    const ninfer::GenerationResult after =
        engine->generate(engine->prepare_tokens(sample), greedy_options(1, false));
    std::cerr << "[256k-lanes] after the refusal, one lane: outputs="
              << after.generated_token_ids.size()
              << " finish=" << static_cast<int>(after.finish_reason) << std::endl;
    if (after.generated_token_ids.size() != 1 ||
        after.finish_reason != ninfer::FinishReason::OutputLimit) {
        std::cerr << "[256k-lanes] the engine did not serve a normal request after the refusal\n";
        return 1;
    }
    const ninfer::MemorySummary summary = engine->memory_summary();
    std::cerr << "[256k-lanes] done: resident_pages=" << summary.kv_resident_page_groups
              << " occupied_device_pages=" << engine->runtime_stats().device_main_kv_occupied_pages
              << " total=" << elapsed() << " s" << std::endl;

    // --- diagnostic: does the aggregate still hold once ADMITTED lanes grow? -------------------
    // The admission rule prices live + one step, so two lanes are admitted while their FUTURE steps
    // can still collide: with both at 49.8% of the pool there is room for ~190 tokens each before the
    // sum reaches the edge, which is a few seconds of decode instead of an hour. Opt-in here so the
    // default arm stays a contract test; the regression entry is the "256k-growth" arm, which runs
    // the same probe in a process of its own (see tests.cmake).
    if (const char* growth = std::getenv("NINFER_TEST_256K_LANES_GROWTH");
        growth != nullptr && *growth != '\0' && std::string(growth) != "0") {
        // The value is the watchdog in seconds ("1" keeps the default).
        const unsigned long requested = std::strtoul(growth, nullptr, 10);
        return run_growth_probe(*engine, load.kv_resident_page_groups,
                                requested > 1 ? static_cast<unsigned>(requested)
                                              : kGrowthWatchdogSeconds);
    }
    return 0;
}

int exercise_256k_overrun(ninfer::Engine& engine, std::uint32_t resident_pages,
                          std::uint32_t resolved_tokens,
                          const std::vector<ninfer::TokenId>& sample) {
    constexpr double kRefusalBudgetSeconds = 60.0;
    const std::uint64_t resident_tokens    = static_cast<std::uint64_t>(resident_pages) * 64ULL;
    if (resident_tokens == 0 || resolved_tokens <= resident_tokens) {
        std::cerr << "[256k] overrun arm needs a split plan: resident=" << resident_tokens
                  << " resolved=" << resolved_tokens << '\n';
        return 1;
    }
    // The prompt is one page group plus a little: enough to be refused, cheap enough that a refusal
    // that has to walk the whole prefill still finishes.
    const std::uint64_t wanted = resident_tokens + 353ULL;
    const std::uint32_t overrun =
        static_cast<std::uint32_t>(std::min<std::uint64_t>(wanted, resolved_tokens));
    std::string text;
    while (text.size() <= static_cast<std::size_t>(overrun) * 8ULL) {
        text += "The quick brown fox jumps over the lazy dog. ";
    }
    std::vector<ninfer::TokenId> prompt = engine.tokenize_text(text);
    if (prompt.size() > overrun) { prompt.resize(overrun); }
    if (prompt.size() <= resident_tokens) {
        std::cerr << "[256k] overrun arm could not build an over-limit prompt: tokens="
                  << prompt.size() << " resident_tokens=" << resident_tokens << '\n';
        return 1;
    }

    std::cerr << "[256k] overrun: prompt_tokens=" << prompt.size()
              << " resident_tokens=" << resident_tokens << std::endl;
    const auto started = std::chrono::steady_clock::now();
    bool refused       = false;
    std::string message;
    try {
        const ninfer::GenerationResult served =
            engine.generate(engine.prepare_tokens(prompt), greedy_options(1, false));
        std::cerr << "[256k] overrun was SERVED: outputs=" << served.generated_token_ids.size()
                  << " finish=" << static_cast<int>(served.finish_reason) << std::endl;
    } catch (const std::exception& error) {
        refused = true;
        message = error.what();
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::cerr << "[256k] overrun refusal: refused=" << (refused ? 1 : 0) << " seconds=" << seconds
              << " message=\"" << message << "\"" << std::endl;

    if (!refused) {
        std::cerr << "[256k] an over-limit prompt was not refused\n";
        return 1;
    }
    if (message.find("prompt_tokens=") == std::string::npos ||
        message.find("resident_tokens=") == std::string::npos) {
        std::cerr << "[256k] the refusal does not name the prompt/resident token counts: " << message
                  << '\n';
        return 1;
    }
    if (seconds > kRefusalBudgetSeconds) {
        std::cerr << "[256k] the refusal took " << seconds << " s (budget "
                  << kRefusalBudgetSeconds << " s)\n";
        return 1;
    }

    // The engine still serves. Same session, same Program, a request inside the resident set.
    const ninfer::GenerationResult after =
        engine.generate(engine.prepare_tokens(sample), greedy_options(1, false));
    std::cerr << "[256k] after the refusal: outputs=" << after.generated_token_ids.size()
              << " finish=" << static_cast<int>(after.finish_reason) << std::endl;
    if (after.generated_token_ids.size() != 1 ||
        after.finish_reason != ninfer::FinishReason::OutputLimit) {
        std::cerr << "[256k] the engine did not serve a normal request after the refusal: outputs="
                  << after.generated_token_ids.size()
                  << " finish=" << static_cast<int>(after.finish_reason) << '\n';
        return 1;
    }
    const ninfer::MemorySummary summary = engine.memory_summary();
    std::cerr << "[256k] after the refusal: resident_pages=" << summary.kv_resident_page_groups
              << " logical_pages=" << summary.kv_logical_page_groups
              << " host_occupied=" << summary.host_context_occupied_bytes << std::endl;
    return 0;
}

// port (sm_89): the 256K arm - "logical capacity above the Device residency, the rest on the Host".
//
// The artifact's Int8Group64 KV costs 35,936 B/token, so 262,144 tokens are ~9.4 GB against the
// ~4.0 GB this card has left after the 17 GiB of weights. Before the residency model existed the
// capacity resolution demanded the Device hold every logical page and the construction died with
// "minimum Engine runtime reservation requires ...". This arm is the acceptance for the split:
// construction succeeds, the Device pool is built at the resident page count, the Host tier is
// allocated for the payload that does not fit, and a one-token generation completes.
//
// The swap counters are asserted to be *reported*, not to be non-zero: a 9-token prompt only ever
// touches one page group, which is resident, so a healthy run fetches nothing and says so with the
// residency ratio. A run that needs more live pages than the pool holds is refused with the numbers
// (ProgramImpl::ensure_sequence_kv_mapped), never served from a stale block-table entry.
int exercise_256k_residency(const char* artifact) {
    constexpr std::uint32_t kTargetTokens = 262144;
    ninfer::EngineOptions options         = engine_options(artifact);
    options.max_context                   = kTargetTokens;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(kTargetTokens);
    options.prefill_chunk                 = 1024;

    std::cerr << "[step] constructing 262144-token engine (KV residency)" << std::endl;
    // The residency model is opt-in: it deliberately builds a Device pool smaller than max_context
    // and refuses requests past it with a diagnostic. That trade only makes sense for a plan that
    // would otherwise not exist at all, so no other arm of this fixture turns it on.
#if defined(_WIN32)
    (void)_putenv_s("NINFER_KV_HOST_RESIDENCY", "1");
#else
    (void)setenv("NINFER_KV_HOST_RESIDENCY", "1", 1);
#endif
    std::unique_ptr<ninfer::Engine> engine;
    std::string failure;
    for (int attempt = 1; attempt <= 3 && engine == nullptr; ++attempt) {
        try {
            engine = std::make_unique<ninfer::Engine>(options);
        } catch (const std::exception& error) {
            failure = error.what();
            const bool capacity = failure.find("reservation") != std::string::npos ||
                                  failure.find("runtime capacity") != std::string::npos ||
                                  failure.find("out of memory") != std::string::npos ||
                                  failure.find("memory allocation") != std::string::npos;
            if (!capacity) {
                std::cerr << "[256k] construction failed: " << failure << '\n';
                return 1;
            }
            std::cerr << "[256k] attempt " << attempt << " found no Device room: " << failure
                      << std::endl;
            // This fixture builds two 19 GiB engines in one process, and WDDM hands the first one's
            // Device memory back asynchronously: a second load can find 0 bytes free for a few
            // seconds after the first engine is destroyed. That is a transient of the harness, not
            // the plan, so it gets a retry before it is reported as infeasible.
            std::this_thread::sleep_for(std::chrono::seconds(10));
        }
    }
    if (engine == nullptr) {
        // The budget account, so an infeasible configuration says WHAT is missing rather than only
        // that it failed. The weight and per-token figures are the ones the 4096 arm measured on
        // this artifact and this card.
        constexpr std::uint64_t kWeightsBytes   = 18197517824ULL;
        constexpr std::uint64_t kBytesPerToken  = 35936ULL;
        constexpr std::uint64_t kKvPayloadBytes = kBytesPerToken * kTargetTokens;
        std::cerr << "[256k] INFEASIBLE: " << failure << '\n'
                  << "[256k] budget: weights=" << kWeightsBytes << " B, KV payload="
                  << kKvPayloadBytes << " B (" << kBytesPerToken << " B/token x " << kTargetTokens
                  << "), so the fully resident anchor alone is " << kKvPayloadBytes
                  << " B while the card has 24,564 MiB total; the plan needs a Device pool of at "
                     "least one page group plus the anchor's fixed part (workspace, graph "
                     "allowance, non-KV persistent layout)\n";
        // Capacity, not correctness: the same treatment the 4096 arm gives a device that cannot hold
        // its reservation - a diagnosed skip (the registration maps 77 to SKIP).
        return 77;
    }

    const ninfer::LoadSummary load     = engine->load_summary();
    const ninfer::MemorySummary memory = engine->memory_summary();
    std::cerr << "[256k] logical_pages=" << load.kv_logical_page_groups
              << " resident_pages=" << load.kv_resident_page_groups
              << " resolved_tokens=" << load.kv_resolved_tokens
              << " device_kv=" << load.kv_device_payload_bytes
              << " host_kv=" << load.kv_host_payload_bytes
              << " reservation=" << load.kv_runtime_reservation_bytes
              << " available_after_weights=" << load.kv_available_after_weights_bytes
              << std::endl;
    std::cerr << "[256k] program: kv_payload=" << memory.kv_payload_bytes
              << " kv_host_payload=" << memory.kv_host_payload_bytes
              << " resident_pages=" << memory.kv_resident_page_groups
              << " logical_pages=" << memory.kv_logical_page_groups
              << " host_context=" << memory.host_context_capacity_bytes
              << " pinned=" << (memory.host_context_pinned ? 1 : 0)
              << " free_after_startup=" << memory.free_after_startup_bytes << std::endl;

    const double residency_percent =
        100.0 * static_cast<double>(load.kv_resident_page_groups) /
        static_cast<double>(load.kv_logical_page_groups == 0 ? 1 : load.kv_logical_page_groups);
    if (load.kv_resolved_tokens < kTargetTokens || load.kv_logical_page_groups == 0 ||
        load.kv_resident_page_groups == 0 ||
        load.kv_resident_page_groups >= load.kv_logical_page_groups ||
        load.kv_device_payload_bytes == 0 || load.kv_host_payload_bytes == 0) {
        std::cerr << "[256k] the residency split was not resolved: resolved_tokens="
                  << load.kv_resolved_tokens << " logical=" << load.kv_logical_page_groups
                  << " resident=" << load.kv_resident_page_groups
                  << " device_kv=" << load.kv_device_payload_bytes
                  << " host_kv=" << load.kv_host_payload_bytes << '\n';
        return 1;
    }
    // Both tiers exist and agree with the resolution that priced them: the Device side is the
    // Program's own kv payload and the Host side is exactly the payload the resolution put there.
    if (memory.kv_payload_bytes == 0 || memory.kv_payload_bytes != load.kv_device_payload_bytes ||
        memory.kv_host_payload_bytes != load.kv_host_payload_bytes ||
        memory.kv_resident_page_groups != load.kv_resident_page_groups ||
        memory.host_context_capacity_bytes < memory.kv_host_payload_bytes) {
        std::cerr << "[256k] the Program's KV tiers do not match the resolution: device="
                  << memory.kv_payload_bytes << " expected=" << load.kv_device_payload_bytes
                  << " host=" << memory.kv_host_payload_bytes
                  << " expected_host=" << load.kv_host_payload_bytes
                  << " host_arena=" << memory.host_context_capacity_bytes << '\n';
        return 1;
    }

    const std::vector<ninfer::TokenId> prompt =
        engine->tokenize_text("The quick brown fox jumps over the lazy dog.");
    if (prompt.empty()) {
        std::cerr << "[256k] artifact tokenizer produced an empty prompt\n";
        return 1;
    }
    const ninfer::GenerationResult single =
        engine->generate(engine->prepare_tokens(prompt), greedy_options(1, false));
    const double prefill_rate = single.timings.prefill_seconds > 0.0
                                    ? static_cast<double>(single.computed_prefill_tokens) /
                                          single.timings.prefill_seconds
                                    : 0.0;
    std::cerr << "[256k] one token: outputs=" << single.generated_token_ids.size()
              << " finish=" << static_cast<int>(single.finish_reason)
              << " prompt_tokens=" << single.prompt.prompt_tokens
              << " prefill_s=" << single.timings.prefill_seconds
              << " prefill_tok_s=" << prefill_rate
              << " residency=" << residency_percent << "%" << std::endl;
    if (single.generated_token_ids.size() != 1 ||
        single.finish_reason != ninfer::FinishReason::OutputLimit) {
        std::cerr << "[256k] one-token generation did not complete: outputs="
                  << single.generated_token_ids.size()
                  << " finish=" << static_cast<int>(single.finish_reason) << '\n';
        return 1;
    }

    const ninfer::MemorySummary after = engine->memory_summary();
    std::cerr << "[256k] swap counters: fetched_pages=" << after.kv_fetched_pages
              << " fetched_bytes=" << after.kv_fetched_bytes
              << " spilled_pages=" << after.kv_spilled_pages
              << " spilled_bytes=" << after.kv_spilled_bytes
              << " host_occupied=" << after.host_context_occupied_bytes << std::endl;
    // The plan's own slack: the value in force, not the one the reader assumes. A run that exports
    // NINFER_KV_RESIDENCY_SLACK_MB overrides it (that is how the 512 MiB paging arm is reproduced).
    std::cerr << "[256k] residency slack=" << load.kv_residency_slack_bytes
              << " B (" << (load.kv_residency_slack_bytes >> 20) << " MiB) default="
              << ((std::getenv("NINFER_KV_RESIDENCY_SLACK_MB") == nullptr) ? 1 : 0) << std::endl;
    if (std::getenv("NINFER_KV_RESIDENCY_SLACK_MB") == nullptr &&
        load.kv_residency_slack_bytes != (2048ULL << 20)) {
        std::cerr << "[256k] the default residency slack is not 2 GiB: " << load.kv_residency_slack_bytes
                  << " B\n";
        return 1;
    }

    // port (sm_89): the overrun arm - a prompt longer than the resident set must come back as an
    // ERROR, in bounded time, and the engine must still serve afterwards. Measured by hand before the
    // fix: a 60,001-token prompt against a 59,648-token resident set never returned (the request was
    // still outstanding after an hour).
    if (const int result = exercise_256k_overrun(*engine, load.kv_resident_page_groups,
                                                 load.kv_resolved_tokens, prompt);
        result != 0) {
        return result;
    }

    // The swap arm is a pressure scenario, not a contract: it prefills almost the whole resident set
    // to force the retained prefix onto the Host tier and then reuses it. Opt-in because it costs
    // ~30 s of prefill that the residency contract itself does not need.
    if (const char* swap = std::getenv("NINFER_TEST_256K_SWAP"); swap != nullptr && *swap != '\0' &&
                                                                    std::string(swap) != "0") {
        if (const int result = exercise_256k_swap(*engine, load.kv_resident_page_groups, prompt);
            result != 0) {
            return result;
        }
    }

    // Optional: a decode long enough for a rate. The 1-token contract above is what the fixture
    // pins; this arm is the measurement (min-of-N comes from repeating the binary).
    if (const char* bench = std::getenv("NINFER_TEST_256K_BENCH"); bench != nullptr && *bench != '\0') {
        const auto outputs = static_cast<std::uint32_t>(std::strtoul(bench, nullptr, 10));
        // A prompt this arm can actually serve: everything above the resident set is refused by the
        // residency guard, so the measurement stays strictly inside the resident budget and says how
        // far that is.
        std::vector<ninfer::TokenId> bench_prompt = prompt;
        if (const char* want = std::getenv("NINFER_TEST_256K_PROMPT_TOKENS");
            want != nullptr && *want != '\0') {
            const std::uint64_t target = std::strtoull(want, nullptr, 10);
            const std::uint64_t resident_tokens =
                static_cast<std::uint64_t>(load.kv_resident_page_groups) * 64ULL;
            const std::uint64_t capped =
                std::min(target, resident_tokens > 256ULL ? resident_tokens - 256ULL : resident_tokens);
            std::string text;
            while (text.size() <= capped * 8ULL) {
                text += "The quick brown fox jumps over the lazy dog. ";
            }
            bench_prompt = engine->tokenize_text(text);
            if (bench_prompt.size() > capped) {
                bench_prompt.resize(static_cast<std::size_t>(capped));
            }
            std::cerr << "[256k] bench prompt: tokens=" << bench_prompt.size()
                      << " resident_tokens=" << resident_tokens << std::endl;
        }
        const ninfer::GenerationResult run =
            engine->generate(engine->prepare_tokens(bench_prompt), greedy_options(outputs, false));
        const double prefill_rate = run.timings.prefill_seconds > 0.0
                                        ? static_cast<double>(run.computed_prefill_tokens) /
                                              run.timings.prefill_seconds
                                        : 0.0;
        const double decode_rate =
            run.timings.decode_seconds > 0.0 && run.generated_token_ids.size() > 1
                ? static_cast<double>(run.generated_token_ids.size() - 1) /
                      run.timings.decode_seconds
                : 0.0;
        std::cerr << "[256k] bench: outputs=" << run.generated_token_ids.size()
                  << " prefill_tokens=" << run.computed_prefill_tokens
                  << " prefill_s=" << run.timings.prefill_seconds
                  << " prefill_tok_s=" << prefill_rate
                  << " decode_s=" << run.timings.decode_seconds
                  << " decode_tok_s=" << decode_rate << std::endl;
    }

    // The overrun arm (a prompt longer than the resident set) is NOT part of this fixture: measured
    // once by hand, a 60,001-token prompt against a 932-page (59,648-token) resident set never
    // returned - the request was still outstanding after an hour, so the refusal path throws
    // somewhere the request's completion path does not report back from. Until that is diagnosed the
    // arm is a manual, time-boxed experiment rather than something the suite runs.
    return 0;
}

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }

    // Which arm this process runs. Both arms build a 19 GiB engine and this card cannot hold two, so
    // the "both" mode destroys the first before the second - and on WDDM the second load can then
    // find the first engine's Device memory still charged to the process (measured: "0 bytes free"
    // for a full second after the destroy). Each arm therefore has its own registration and its own
    // process; "both" stays available for a one-shot local run.
    const char* arm    = std::getenv("NINFER_TEST_DENSE_ARM");
    const bool want_4096 = arm == nullptr || *arm == '\0' ||
                           (std::string(arm) != "256k" && std::string(arm) != "256k-lanes" &&
                            std::string(arm) != "256k-growth");
    const bool want_256k = arm != nullptr && (std::string(arm) == "256k" || std::string(arm) == "both");
    // port (sm_89): the two-lane shared-pool arm. Its own process for the same reason the 256K arm
    // has one: a 19 GiB engine cannot be built twice in one process on this card.
    const bool want_lanes = arm != nullptr && std::string(arm) == "256k-lanes";
    // port (sm_89): the growth arm - the same two-lane engine, loaded at ~49.8% of the pool per lane,
    // where the next steps cross the edge the aggregate admission does not reach.
    const bool want_growth = arm != nullptr && std::string(arm) == "256k-growth";

    // port (sm_89): progress markers. Each step reports itself before running, and a construction
    // failure (memory budget, unsupported option) reports its message instead of terminating
    // silently.
    if (want_4096) {
        std::cerr << "[step] constructing engine" << std::endl;
        try {
            ninfer::Engine engine(engine_options(artifact));
            std::cerr << "[step] report_loaded_product" << std::endl;
            if (const int result = report_loaded_product(engine); result != 0) { return result; }
            std::cerr << "[step] exercise_text_mtp_and_prefix" << std::endl;
            if (const int result = exercise_text_mtp_and_prefix(engine); result != 0) {
                return result;
            }
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
    }
    if (want_256k) {
        if (const int result = exercise_256k_residency(artifact); result != 0) { return result; }
    }
    if (want_lanes) {
        if (const int result = exercise_256k_lanes(artifact); result != 0) { return result; }
    }
    if (want_growth) {
        if (const int result = exercise_256k_growth(artifact); result != 0) { return result; }
    }
    std::cout << "ok\n";
    return 0;
}
