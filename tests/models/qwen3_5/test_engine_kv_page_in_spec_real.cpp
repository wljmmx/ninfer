// port (sm_89, V10-80): the resident-KV leg extended to the speculative backend's KV.
//
// The claim: with NINFER_KV_PAGE_IN the MTP draft cache (`sequence.kv->backend`, consumed through
// io.backend_kv_table_row) is staged by the same rule as the text KV - a page the demotion gave
// back is copied into a free slot and its table entry rewritten before the draft/verify step that
// reads it - so a speculative run under a constrained arena produces the fully resident run's
// tokens, one for one.
//
// Why the probe is the honest instrument: nothing in this scenario makes the arena too small (the
// prompt is 32 of the 128 logical pages and the declared arena is 48), so no page would leave the
// Device on its own. NINFER_KV_PAGE_IN_PROBE gives four addressed pages back and stages the step
// again, once per address space, so the round trip Host -> Device -> Host -> Device is executed
// inside a real draft step for the text KV AND for the backend KV. Before V10-80 the backend space
// had no such pass at all, which is what the fetched-page count below discriminates: the text probe
// alone restores kProbePages pages, the two probes together restore twice that.
//
// Two arms, one binary, one artifact:
//
//   resident   the arena is the whole logical capacity, no leg. This is the reference token stream.
//   page-in    48 declared page groups for an 8192-token context (128) and NINFER_KV_PAGE_IN=1 with
//              the probe at 4 pages. Tokens must equal the reference, both spaces must report a
//              demotion and a re-fetch, and the speculative round must actually have drafted.

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
constexpr std::uint32_t kDeclaredPages = 48; // 3072 tokens: below the 128-page context
constexpr std::uint32_t kProbePages    = 4;
constexpr std::uint32_t kOutputs       = 8;

void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    (void)_putenv_s(name, value);
#else
    (void)setenv(name, value, 1);
#endif
}

ninfer::EngineOptions spec_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path             = artifact;
    options.max_context               = kContextTokens;
    options.kv_capacity               = ninfer::KvCapacityPolicy::explicit_capacity(kContextTokens);
    options.prefill_chunk             = 1024;
    options.kv_cache                  = ninfer::KvCacheStorage::BFloat16;
    // port (sm_89, V10-86): the same leg, shown on both drafters. The default stays MTP so the
    // V10-80 arm keeps its exact numbers; NINFER_SPEC_BACKEND=dflash2 swaps the drafter, because
    // the DFlash2 cache is a different draft window (its own page count per lane and its own page
    // store) and "the leg covers the speculative backend" has to be demonstrated on it too.
    const char* const backend         = std::getenv("NINFER_SPEC_BACKEND");
    options.speculative.backend       = backend != nullptr && std::strcmp(backend, "dflash2") == 0
                                            ? ninfer::SpeculativeBackend::DFlash2
                                            : ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens  = 3;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    options.enable_vision             = false;
    options.max_concurrency           = 1;
    options.max_pending_requests      = 1;
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
    // of the same tokens instead of between a prefill and a restored checkpoint.
    options.execution.allow_prefix_reuse = false;
    options.stop.include_model_defaults  = false;
    return options;
}

std::string fixture_text() {
    const std::string paragraph =
        "The draft block reads the cache it grew itself, page for page, and a page the demotion "
        "gave "
        "back has to be copied into the arena again before the block that reads it runs. ";
    std::string text;
    while (text.size() < 64 * 1024) { text += paragraph; }
    return text;
}

bool same_tokens(const ninfer::GenerationResult& left, const ninfer::GenerationResult& right) {
    return left.generated_token_ids == right.generated_token_ids;
}

void dump_tokens(const char* name, const ninfer::GenerationResult& result) {
    std::cerr << "[page-in-spec] " << name << " tokens:";
    for (const auto token : result.generated_token_ids) { std::cerr << ' ' << token; }
    std::cerr << '\n';
}

void dump_speed(const char* name, const ninfer::GenerationResult& result) {
    const std::uint32_t tokens = static_cast<std::uint32_t>(result.generated_token_ids.size());
    const double decode        = result.timings.decode_seconds;
    std::cerr << "[page-in-spec] " << name << " speed: decode_seconds=" << decode
              << " tok/s=" << (decode > 0.0 ? double(tokens) / decode : 0.0) << "\n";
}

void dump_speculative(const char* name, const ninfer::GenerationResult& result) {
    std::cerr << "[page-in-spec] " << name
              << " speculative backend=" << static_cast<int>(result.speculative.backend)
              << " rounds=" << result.speculative.rounds
              << " drafted=" << result.speculative.drafted_tokens
              << " accepted=" << result.speculative.accepted_tokens << '\n';
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    // Env is process-wide and the Program reads these at construction, so each arm sets its own
    // state immediately before it builds an Engine.
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
            ninfer::Engine engine(spec_options(artifact));
            const std::vector<ninfer::TokenId> stream = engine.tokenize_text(fixture_text());
            if (stream.size() < 2048) {
                std::cerr << "[page-in-spec] fixture text tokenized to only " << stream.size()
                          << " tokens\n";
                return 1;
            }
            prompt.assign(stream.begin(), stream.begin() + 2000);
            resident = engine.generate(engine.prepare_tokens(prompt), greedy_options(kOutputs));
            if (resident.generated_token_ids.size() != kOutputs) {
                std::cerr << "[page-in-spec] the resident arm did not finish its output budget\n";
                return 1;
            }
            const ninfer::MemorySummary memory = engine.memory_summary();
            std::cerr << "[page-in-spec] resident: pages=" << memory.kv_resident_page_groups
                      << " of " << memory.kv_logical_page_groups
                      << " fetched=" << memory.kv_fetched_pages
                      << " spilled=" << memory.kv_spilled_pages << '\n';
            dump_speculative("resident", resident);
            dump_tokens("resident", resident);
            dump_speed("resident", resident);
            if (memory.kv_resident_page_groups < memory.kv_logical_page_groups) {
                std::cerr << "[page-in-spec] the reference arm is not fully resident\n";
                return 1;
            }
            if (resident.speculative.rounds == 0 || resident.speculative.drafted_tokens == 0) {
                std::cerr << "[page-in-spec] the reference arm did not run the speculative path\n";
                return 1;
            }
        }

        // ---- the leg, on both address spaces
        // ------------------------------------------------------
        {
            set_env("NINFER_KV_EVICT_CAPACITY", std::to_string(kDeclaredPages).c_str());
            set_env("NINFER_KV_PAGE_IN", "1");
            set_env("NINFER_KV_PAGE_IN_PROBE", std::to_string(kProbePages).c_str());
            ninfer::Engine engine(spec_options(artifact));
            const ninfer::MemorySummary started = engine.memory_summary();
            std::cerr << "[page-in-spec] arena: pool_pages=" << started.kv_capacity_page_groups
                      << " plan_resident=" << started.kv_resident_page_groups
                      << " logical=" << started.kv_logical_page_groups
                      << " host_payload=" << started.kv_host_payload_bytes << '\n';
            if (started.kv_capacity_page_groups == 0 ||
                started.kv_capacity_page_groups > kDeclaredPages) {
                std::cerr << "[page-in-spec] the declared arena was not applied ("
                          << started.kv_capacity_page_groups << " page groups)\n";
                return 1;
            }
            const ninfer::GenerationResult staged =
                engine.generate(engine.prepare_tokens(prompt), greedy_options(kOutputs));
            const ninfer::MemorySummary memory = engine.memory_summary();
            std::cerr << "[page-in-spec] leg: fetched=" << memory.kv_fetched_pages
                      << " spilled=" << memory.kv_spilled_pages << '\n';
            dump_speculative("page-in", staged);
            dump_tokens("page-in", staged);
            dump_speed("page-in", staged);
            // The primary claim first: a demoted-and-not-restored backend page makes the draft read
            // the wrong bytes, so a token difference is the failure, not a side effect.
            if (!same_tokens(staged, resident)) {
                std::cerr
                    << "[page-in-spec] the page-in arm diverged from the fully resident run\n";
                dump_tokens("resident", resident);
                dump_tokens("page-in", staged);
                return 1;
            }
            // Each space's probe demotes kProbePages addressed pages and stages them again, so the
            // text leg alone accounts for kProbePages re-fetches. Anything at or below that number
            // means the speculative backend never entered the leg.
            //
            // port (sm_89, V10-86): that expectation holds for the MTP drafter and for DFlash's
            // paged "full" half, but not for DFlash2. The DFlash2 drafter keeps its whole draft
            // window in the cyclic local cache (draft_context.h:14) and its paged half is not
            // planned at all, so backend_kv_cache() returns null (sequence.cpp:604-607) and there
            // is no address space to demote from: NINFER_KV_PAGE_IN_TRACE prints
            // "backend space: cache=none bundle=0" for this arm while the MTP arm prints
            // "cache=paged bundle=1". Both arms must still produce the resident run's tokens, and
            // the text space must still round-trip, so the floor is one probe here and two there -
            // the difference is the structure, asserted rather than smoothed over.
            const char* const selected = std::getenv("NINFER_SPEC_BACKEND");
            const bool dflash2 = selected != nullptr && std::strcmp(selected, "dflash2") == 0;
            const std::uint32_t expected_fetches = dflash2 ? kProbePages : 2 * kProbePages;
            if (memory.kv_fetched_pages < expected_fetches ||
                memory.kv_spilled_pages < expected_fetches) {
                std::cerr << "[page-in-spec] the KV staging fell short: fetched="
                          << memory.kv_fetched_pages << " spilled=" << memory.kv_spilled_pages
                          << " expected at least " << expected_fetches
                          << (dflash2 ? " (text only: the DFlash2 drafter has no paged backend)"
                                      : " (text + MTP backend)")
                          << '\n';
                return 1;
            }
            if (staged.speculative.rounds == 0 || staged.speculative.drafted_tokens == 0) {
                std::cerr << "[page-in-spec] the leg arm did not run the speculative path\n";
                return 1;
            }
            std::cerr << "[page-in-spec] staged speculative run: tokens identical to the resident "
                         "run, fetched="
                      << memory.kv_fetched_pages << " spilled=" << memory.kv_spilled_pages << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "speculative kv page-in real test failed: " << error.what() << std::endl;
        return 1;
    }
    set_env("NINFER_KV_EVICT_CAPACITY", "");
    set_env("NINFER_KV_PAGE_IN", "");
    set_env("NINFER_KV_PAGE_IN_PROBE", "");
    std::cout << "ok\n";
    return 0;
}
