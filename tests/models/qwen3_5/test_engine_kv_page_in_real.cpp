// port (sm_89, V10-79): the resident-KV leg, against the real artifact.
//
// The claim under test is narrow and falsifiable: with NINFER_KV_PAGE_IN a Device arena SMALLER
// than the logical context serves a request, and a page that was given back to the Host tier is
// copied into a Device slot and republished into the row before the layer that addresses it runs -
// so the tokens are the fully resident run's tokens, one for one.
//
// Three arms, one binary, one artifact:
//
//   resident   the arena is the whole logical capacity (no declared budget, no leg). This is the
//              reference: what the same prompt produces when every page is on the Device.
//   page-in    NINFER_KV_EVICT_CAPACITY declares 48 page groups for an 8192-token context (128),
//   and
//              NINFER_KV_PAGE_IN=1 makes the arena legal. NINFER_KV_PAGE_IN_PROBE then gives back
//              part of the set the pass just staged and stages it again, so the round trip Device
//              -> Host -> Device is executed inside a real step. The tokens must not move, and the
//              fetched-page counter must be non-zero (otherwise the probe proved nothing).
//   refusal    the same 48-page arena with a prompt longer than it: the request is refused at
//   admission
//              with the numbers, because one page group spans every layer's planes and the
//              full-attention layers address the whole history on every step. A refusal that names
//              the real limit is the correct answer there - a silent short read is not.
//
// A dense artifact is used on purpose: the experts of a routed model make greedy continuations
// depend on which experts happen to be resident (measured, test_engine_kv_tier_real.cpp), and this
// test is about KV placement, not about the pool.

#include "ninfer/engine.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kContextTokens = 8192;
constexpr std::uint32_t kDeclaredPages = 48; // 3072 tokens: strictly below the 128-page context
constexpr std::uint32_t kProbePages    = 4;
constexpr std::uint32_t kOutputs       = 8;

void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    (void)_putenv_s(name, value);
#else
    (void)setenv(name, value, 1);
#endif
}

ninfer::EngineOptions page_in_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path        = artifact;
    options.max_context          = kContextTokens;
    options.kv_capacity          = ninfer::KvCapacityPolicy::explicit_capacity(kContextTokens);
    options.prefill_chunk        = 1024;
    options.kv_cache             = ninfer::KvCacheStorage::BFloat16;
    options.speculative.backend  = ninfer::SpeculativeBackend::None;
    options.enable_vision        = false;
    options.use_cuda_graph       = false;
    options.max_concurrency      = 1;
    options.max_pending_requests = 1;
    options.context_cache.device_state_slots = 2;
    // A page that leaves the Device has to land somewhere: the leg needs a Host KV tier.
    options.context_cache.host_capacity_bytes = 1ULL << 30;
    return options;
}

ninfer::RequestOptions greedy_options(std::uint32_t outputs) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.sampling.seed           = 0;
    // No prefix reuse: every arm prefills its own prompt, so the comparison is between two prefills
    // of the same tokens and not between a prefill and a restored checkpoint.
    options.execution.allow_prefix_reuse = false;
    options.stop.include_model_defaults  = false;
    return options;
}

std::string fixture_text() {
    const std::string paragraph =
        "The engine keeps a page on the host tier, copies it back into the Device arena when the "
        "layer "
        "that reads it runs, and gives a page nobody is reading back to the pool to make room. ";
    std::string text;
    while (text.size() < 64 * 1024) { text += paragraph; }
    return text;
}

bool same_tokens(const ninfer::GenerationResult& left, const ninfer::GenerationResult& right) {
    return left.generated_token_ids == right.generated_token_ids;
}

void dump_tokens(const char* name, const ninfer::GenerationResult& result) {
    std::cerr << "[page-in] " << name << " tokens:";
    for (const auto token : result.generated_token_ids) { std::cerr << ' ' << token; }
    std::cerr << '\n';
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    // Env is process-wide and the Program reads these at construction, so each arm sets its own
    // state immediately before it builds an Engine. The leg switch itself is read per step (it is
    // what the probe toggles), which is why the two arms can live in one process at all.
    set_env("NINFER_KV_EVICT_CAPACITY", "");
    set_env("NINFER_KV_PAGE_IN", "");
    set_env("NINFER_KV_PAGE_IN_PROBE", "");
    set_env("NINFER_KV_HOST_RESIDENCY", "");

    try {
        // ---- resident reference
        // -------------------------------------------------------------------
        std::vector<ninfer::TokenId> prompt;
        ninfer::GenerationResult resident;
        {
            ninfer::Engine engine(page_in_options(artifact));
            const std::vector<ninfer::TokenId> stream = engine.tokenize_text(fixture_text());
            if (stream.size() < 4096) {
                std::cerr << "[page-in] fixture text tokenized to only " << stream.size()
                          << " tokens\n";
                return 1;
            }
            // ~2000 tokens: 32 page groups, which fits the declared 48-page arena with room for the
            // decode step's growth.
            prompt.assign(stream.begin(), stream.begin() + 2000);
            resident = engine.generate(engine.prepare_tokens(prompt), greedy_options(kOutputs));
            if (resident.generated_token_ids.size() != kOutputs) {
                std::cerr << "[page-in] the resident arm did not finish its output budget\n";
                return 1;
            }
            const ninfer::MemorySummary memory = engine.memory_summary();
            std::cerr << "[page-in] resident: pages=" << memory.kv_resident_page_groups << " of "
                      << memory.kv_logical_page_groups << " fetched=" << memory.kv_fetched_pages
                      << " spilled=" << memory.kv_spilled_pages << '\n';
            if (memory.kv_resident_page_groups < memory.kv_logical_page_groups) {
                std::cerr << "[page-in] the reference arm is not fully resident\n";
                return 1;
            }
        }

        // ---- the leg: a 48-page arena for a 128-page context
        // --------------------------------------
        {
            set_env("NINFER_KV_EVICT_CAPACITY", std::to_string(kDeclaredPages).c_str());
            set_env("NINFER_KV_PAGE_IN", "1");
            set_env("NINFER_KV_PAGE_IN_PROBE", std::to_string(kProbePages).c_str());
            ninfer::Engine engine(page_in_options(artifact));
            const ninfer::MemorySummary started = engine.memory_summary();
            std::cerr << "[page-in] arena: pool_pages=" << started.kv_capacity_page_groups
                      << " plan_resident=" << started.kv_resident_page_groups
                      << " logical=" << started.kv_logical_page_groups
                      << " host_payload=" << started.kv_host_payload_bytes << '\n';
            // The arena is the declared one. The live pool edge is what the admission bound and the
            // step guard read, so that is the number that has to be small.
            if (started.kv_capacity_page_groups == 0 ||
                started.kv_capacity_page_groups > kDeclaredPages) {
                std::cerr << "[page-in] the declared arena was not applied ("
                          << started.kv_capacity_page_groups << " page groups)\n";
                return 1;
            }
            const ninfer::GenerationResult staged =
                engine.generate(engine.prepare_tokens(prompt), greedy_options(kOutputs));
            const ninfer::MemorySummary memory = engine.memory_summary();
            std::cerr << "[page-in] leg: fetched=" << memory.kv_fetched_pages
                      << " spilled=" << memory.kv_spilled_pages << '\n';
            if (memory.kv_fetched_pages == 0) {
                std::cerr
                    << "[page-in] no page was fetched back from the Host tier: the probe did not "
                       "exercise the leg\n";
                return 1;
            }
            if (memory.kv_spilled_pages == 0) {
                std::cerr << "[page-in] no page was given back to the Host tier\n";
                return 1;
            }
            if (!same_tokens(staged, resident)) {
                std::cerr << "[page-in] the page-in arm diverged from the fully resident run\n";
                dump_tokens("resident", resident);
                dump_tokens("page-in", staged);
                return 1;
            }
            std::cerr << "[page-in] staged run: tokens identical to the resident run, fetched="
                      << memory.kv_fetched_pages << " spilled=" << memory.kv_spilled_pages << '\n';
        }

        // ---- the honest refusal: a prompt longer than the arena
        // ------------------------------------
        {
            ninfer::Engine engine(page_in_options(artifact));
            const std::vector<ninfer::TokenId> stream = engine.tokenize_text(fixture_text());
            std::vector<ninfer::TokenId> long_prompt(stream.begin(), stream.begin() + 3900);
            auto handle = engine.submit(engine.prepare_tokens(long_prompt), greedy_options(1));
            std::string message;
            try {
                (void)handle.wait();
            } catch (const std::exception& error) { message = error.what(); }
            std::cerr << "[page-in] refusal: " << message << '\n';
            for (const char* field : {"prompt_tokens=", "resident_tokens="}) {
                if (message.find(field) == std::string::npos) {
                    std::cerr << "[page-in] the refusal does not report " << field << '\n';
                    return 1;
                }
            }
            if (message.find("page-in leg is on") == std::string::npos) {
                // With the leg on the message must name the arena, not the missing capability.
                std::cerr << "[page-in] the leg-on refusal still blames the missing leg\n";
                return 1;
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "kv page-in real test failed: " << error.what() << std::endl;
        return 1;
    }
    set_env("NINFER_KV_EVICT_CAPACITY", "");
    set_env("NINFER_KV_PAGE_IN", "");
    set_env("NINFER_KV_PAGE_IN_PROBE", "");
    std::cout << "ok\n";
    return 0;
}
