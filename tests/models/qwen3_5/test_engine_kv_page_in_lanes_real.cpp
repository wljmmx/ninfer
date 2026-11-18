// port (sm_89, V10-81): the resident-KV leg under a multi-lane batch, and the cross-row borrow.
//
// The claim under test: with NINFER_KV_PAGE_IN a Device arena smaller than the lanes' aggregate
// history serves FOUR lanes at once, every lane's tokens are the fully resident run's tokens, and a
// lane that runs out of its own non-addressed pages can take a slot from a sibling lane that holds
// pages no active row reads this step (NINFER_KV_PAGE_IN's cross-row borrow, sequence.cpp
// borrow_slot_from_another_row).
//
// Three arms, one binary, one artifact (the dense 27B - a routed model makes greedy continuations
// depend on which experts happen to be resident, see test_engine_kv_tier_real.cpp):
//
//   resident   four lanes, an arena of four whole contexts (no declared budget, no leg). The window
//              (NINFER_KV_EVICT_SINK / _WINDOW) masks the same pages in both arms because the mask,
//              not the replica's fate, is what attention reads - so this is the reference stream.
//   leg        the same four lanes, NINFER_KV_EVICT_CAPACITY declaring the arena, NINFER_KV_PAGE_IN=1.
//              Tokens must equal the reference lane for lane; spilled/fetched are reported.
//   probe      the leg plus NINFER_KV_PAGE_IN_PROBE: the round trip Device -> Host -> Device runs
//              inside a real multi-lane step, so a wrong slot surfaces as a token difference.
//
// Every knob is read from the environment (NINFER_LANES_*) so the same binary covers the tight-arena
// run and the borrow stress without a rebuild.

#include "ninfer/engine.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kLanes = 4;

std::uint32_t env_u32(const char* name, std::uint32_t fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') { return fallback; }
    const long parsed = std::strtol(value, nullptr, 10);
    return parsed < 0 ? 0U : static_cast<std::uint32_t>(parsed);
}

void set_env(const char* name, const std::string& value) {
#if defined(_WIN32)
    (void)_putenv_s(name, value.c_str());
#else
    (void)setenv(name, value.c_str(), 1);
#endif
}

const std::uint32_t kPromptTokens = env_u32("NINFER_LANES_PROMPT", 1500);
const std::uint32_t kOutputs      = env_u32("NINFER_LANES_OUTPUTS", 8);
const std::uint32_t kContext      = env_u32("NINFER_LANES_CONTEXT", 8192);
const std::uint32_t kArena        = env_u32("NINFER_LANES_ARENA", 0);
const std::uint32_t kHostMib      = env_u32("NINFER_LANES_HOST_MIB", 512);
const std::uint32_t kProbe        = env_u32("NINFER_LANES_PROBE", 0);
const std::uint32_t kWindow       = env_u32("NINFER_LANES_WINDOW", 0);
const std::uint32_t kSink         = env_u32("NINFER_LANES_SINK", 0);
const std::uint32_t kKeepScores   = env_u32("NINFER_LANES_KEEP_SCORES", 0);
const std::uint32_t kKeepRatio    = env_u32("NINFER_LANES_KEEP_RATIO", 0);
const std::uint32_t kDefer        = env_u32("NINFER_LANES_DEFER", 2);
const std::uint32_t kPrefixReuse  = env_u32("NINFER_LANES_PREFIX_REUSE", 0);
// V10-84: the per-unit growth a read-set price has to carry, so a smaller chunk is what makes a
// read-priced arena smaller than the lanes' histories. 1024 is the default every earlier arm ran.
const std::uint32_t kChunk = env_u32("NINFER_LANES_CHUNK", 1024);

ninfer::EngineOptions lane_options(const char* artifact, std::uint32_t capacity_tokens,
                                   std::uint64_t host_bytes) {
    ninfer::EngineOptions options;
    options.artifact_path                 = artifact;
    options.max_context                   = static_cast<std::int32_t>(kContext);
    options.kv_capacity                   = ninfer::KvCapacityPolicy::explicit_capacity(capacity_tokens);
    options.prefill_chunk                 = kChunk;
    options.kv_cache                      = ninfer::KvCacheStorage::BFloat16;
    options.speculative.backend           = ninfer::SpeculativeBackend::None;
    options.enable_vision                 = false;
    options.use_cuda_graph                = false;
    options.max_concurrency               = kLanes;
    options.max_pending_requests          = kLanes;
    options.context_cache.device_state_slots  = 2;
    options.context_cache.host_capacity_bytes = host_bytes;
    return options;
}

ninfer::RequestOptions greedy_options() {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = kOutputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.sampling.seed           = 0;
    // No prefix reuse in the default arm: every lane prefills its own prompt, so the two arms
    // compare two prefills of the same tokens instead of a prefill and a restored checkpoint.
    options.execution.allow_prefix_reuse      = kPrefixReuse != 0;
    options.stop.include_model_defaults       = false;
    return options;
}

// Four distinct raw-token prompts of the SAME length: the same length keeps the batch composition
// (and therefore the dense artifact's per-batch GEMM scheduling) identical between the arms, which is
// what makes a token difference the leg's fault rather than a composition window. With
// NINFER_LANES_SHARED_PREFIX every lane submits the SAME prompt, so the later lanes restore lane 0's
// prefix instead of prefilling it - the case where a lane's addressed pages are on the Host tier from
// the start.
std::vector<std::vector<ninfer::TokenId>> make_prompts() {
    const bool shared = env_u32("NINFER_LANES_SHARED_PREFIX", 0) != 0;
    std::vector<std::vector<ninfer::TokenId>> prompts(kLanes);
    for (std::uint32_t lane = 0; lane < kLanes; ++lane) {
        prompts[lane].reserve(kPromptTokens);
        const std::uint32_t salt = shared ? 0U : lane;
        for (std::uint32_t index = 0; index < kPromptTokens; ++index) {
            prompts[lane].push_back(static_cast<ninfer::TokenId>(
                400 + ((index * 7 + salt * 131 + (index * index) % 3) % 900)));
        }
    }
    return prompts;
}

void apply_switch_env() {
    set_env("NINFER_KV_EVICT_WINDOW", kWindow == 0 ? "" : std::to_string(kWindow));
    set_env("NINFER_KV_EVICT_SINK", kSink == 0 ? "" : std::to_string(kSink));
    set_env("NINFER_KV_PAGE_SCORE_ROW_SPAN", kKeepScores == 0 ? "" : "8");
    set_env("NINFER_KV_EVICT_KEEP_FROM_SCORES", kKeepScores == 0 ? "" : "24");
    set_env("NINFER_KV_EVICT_KEEP_RATIO", kKeepRatio == 0 ? "" : std::to_string(kKeepRatio));
    set_env("NINFER_KV_EVICT_DEFER", kDefer == 0 ? "" : (kDefer == 1 ? "1" : ""));
}

struct ArmResult {
    std::vector<std::vector<ninfer::TokenId>> tokens;
    ninfer::MemorySummary memory;
    double seconds = 0.0;
};

ArmResult run_arm(const char* artifact, const std::vector<std::vector<ninfer::TokenId>>& prompts,
                  std::uint32_t capacity_tokens, std::uint64_t host_bytes) {
    ninfer::Engine engine(lane_options(artifact, capacity_tokens, host_bytes));
    ArmResult result;
    result.tokens.resize(kLanes);
    std::vector<ninfer::GenerationHandle> handles;
    handles.reserve(kLanes);
    const auto started = std::chrono::steady_clock::now();
    for (std::uint32_t lane = 0; lane < kLanes; ++lane) {
        handles.push_back(engine.submit(engine.prepare_tokens(prompts[lane]), greedy_options()));
    }
    for (std::uint32_t lane = 0; lane < kLanes; ++lane) {
        const ninfer::GenerationResult generated = handles[lane].wait();
        result.tokens[lane]                       = generated.generated_token_ids;
        std::cerr << "[lanes] lane " << lane << " tokens:";
        for (const auto token : generated.generated_token_ids) { std::cerr << ' ' << token; }
        std::cerr << '\n';
    }
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    result.memory  = engine.memory_summary();
    return result;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    apply_switch_env();
    const char* const read_pricing = std::getenv("NINFER_KV_READ_PRICING");
    std::cerr << "[lanes] lanes=" << kLanes << " prompt=" << kPromptTokens << " outputs=" << kOutputs
              << " context=" << kContext << " arena=" << kArena << " host_mib=" << kHostMib
              << " window=" << kWindow << " sink=" << kSink << " probe=" << kProbe
              << " keep_scores=" << kKeepScores << " keep_ratio=" << kKeepRatio
              << " defer=" << kDefer << " prefix_reuse=" << kPrefixReuse << " chunk=" << kChunk
              << " read_pricing=" << (read_pricing == nullptr ? "on" : read_pricing) << '\n';
    const std::vector<std::vector<ninfer::TokenId>> prompts = make_prompts();
    try {
        // ---- reference: four whole contexts on the Device, no leg
        set_env("NINFER_KV_EVICT_CAPACITY", "");
        set_env("NINFER_KV_PAGE_IN", "");
        set_env("NINFER_KV_PAGE_IN_PROBE", "");
        set_env("NINFER_KV_HOST_RESIDENCY", "");
        const ArmResult resident =
            run_arm(artifact, prompts, kLanes * kContext, 0);
        std::cerr << "[lanes] resident: pages=" << resident.memory.kv_resident_page_groups << " of "
                  << resident.memory.kv_logical_page_groups
                  << " fetched=" << resident.memory.kv_fetched_pages
                  << " spilled=" << resident.memory.kv_spilled_pages
                  << " wall=" << resident.seconds << " s\n";

        // ---- the leg: the declared arena, four lanes at once
        set_env("NINFER_KV_EVICT_CAPACITY", kArena == 0 ? "" : std::to_string(kArena));
        set_env("NINFER_KV_PAGE_IN", "1");
        set_env("NINFER_KV_PAGE_IN_PROBE", kProbe == 0 ? "" : std::to_string(kProbe));
        const ArmResult leg = run_arm(artifact, prompts, kContext,
                                      static_cast<std::uint64_t>(kHostMib) << 20);
        std::cerr << "[lanes] leg: arena=" << leg.memory.kv_capacity_page_groups
                  << " resident=" << leg.memory.kv_resident_page_groups
                  << " logical=" << leg.memory.kv_logical_page_groups
                  << " host_payload=" << leg.memory.kv_host_payload_bytes
                  << " fetched=" << leg.memory.kv_fetched_pages
                  << " spilled=" << leg.memory.kv_spilled_pages
                  << " wall=" << leg.seconds << " s\n";

        bool equal = true;
        for (std::uint32_t lane = 0; lane < kLanes; ++lane) {
            if (leg.tokens[lane] != resident.tokens[lane]) {
                equal = false;
                std::cerr << "[lanes] lane " << lane << " diverged from the fully resident run\n";
            }
            if (leg.tokens[lane].size() != kOutputs) {
                equal = false;
                std::cerr << "[lanes] lane " << lane << " did not finish its output budget\n";
            }
        }
        std::cerr << "[lanes] summary lanes=" << kLanes << " arena="
                  << leg.memory.kv_capacity_page_groups << " resident="
                  << leg.memory.kv_resident_page_groups << " logical="
                  << leg.memory.kv_logical_page_groups
                  << " fetched=" << leg.memory.kv_fetched_pages
                  << " spilled=" << leg.memory.kv_spilled_pages
                  << " tokens_equal=" << (equal ? 1 : 0) << '\n';
        set_env("NINFER_KV_EVICT_CAPACITY", "");
        set_env("NINFER_KV_PAGE_IN", "");
        set_env("NINFER_KV_PAGE_IN_PROBE", "");
        if (!equal) { return 1; }
    } catch (const std::exception& error) {
        std::cerr << "kv page-in lanes real test failed: " << error.what() << std::endl;
        set_env("NINFER_KV_EVICT_CAPACITY", "");
        set_env("NINFER_KV_PAGE_IN", "");
        set_env("NINFER_KV_PAGE_IN_PROBE", "");
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
