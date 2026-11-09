// port (sm_89): Q54's prefix index, against the real artifact.
//
// The tier used to be findable only by the whole prompt that stored it. This file exercises the
// index end to end: a store phase publishes blocks AND prefix notes; a fresh Engine (the tier
// directory is the only channel) answers
//
//   * an exact-prompt restore (the 224/397 shape, a chat prompt whose deepest capture is below
//     the prompt end) - the pre-index behaviour, kept byte-identical,
//   * a longer request that extends the stored prefix by 103 tokens (the ~500 tier) and one
//     that extends it by 3703 (the ~4100 tier) - both must restore the cached frontier, prefill
//     only the suffix, and still produce the cold run's tokens one for one,
//   * a whole-prompt payload (a plain prompt whose only capture is its end) - it must be
//     DECLINED rather than imported, because the zero-suffix path needs a tail hidden the tier
//     does not carry; the request then runs cold and matches the cold run,
//   * a directory whose notes were deleted, and one whose notes were corrupted - both must
//     fall back to exactly the pre-index miss behaviour: a cold prefill, identical output.
//
// "Cross-process" is modelled the way the tier actually persists it: every phase runs in a
// fresh Engine instance, so each reader re-opens the directory the way a new process would and
// holds no in-process state from the writer.

#include "ninfer/engine.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kPlainPrefixTokens = 397;
constexpr std::size_t kShortSuffixTokens = 103;   // 397 + 103 = 500
constexpr std::size_t kLongSuffixTokens  = 3703;  // 397 + 3703 = 4100

ninfer::EngineOptions tier_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                     = artifact;
    options.max_context                       = 4608;
    options.kv_capacity                       = ninfer::KvCapacityPolicy::explicit_capacity(4608);
    options.prefill_chunk                     = 1024;
    // BFloat16 keeps the page bytes on the plain path the fingerprint has to agree on; the
    // tier carries no draft/backend KV, so the speculative backend stays None.
    options.kv_cache                          = ninfer::KvCacheStorage::BFloat16;
    options.speculative.backend               = ninfer::SpeculativeBackend::None;
    options.enable_vision                     = false;
    options.use_cuda_graph                    = false;
    options.max_concurrency                   = 1;
    options.max_pending_requests              = 1;
    options.context_cache.device_state_slots  = 4;
    // The import lands the state image in the Host pool before promotion; without a Host arena
    // there is nowhere for the bytes to arrive.
    options.context_cache.host_capacity_bytes = 2ULL << 30;
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

// Ordinary natural text: the fixture needs a deterministic prompt whose next-token
// distributions are peaked, because the whole test compares greedy outputs across engines.
// Synthetic token soup leaves the model near flat ties, and a 1e-6 difference in the restored
// path (the store chunked [0,397), the cold run chunks [0,500) as one pass) is then enough to
// flip an argmax and produce a completely different - but equally "correct" - continuation.
std::string fixture_text() {
    const std::string paragraph =
        "The engine keeps a checkpoint at every prompt boundary, restores the longest cached "
        "prefix it can actually prove, and prefills only the suffix it has never seen before. ";
    std::string text;
    while (text.size() < 96 * 1024) { text += paragraph; }
    return text;
}

// The boundary fixture shape the dflash real test proved: a no-thinking seed whose follow-up
// renders the empty think block verbatim, and a preserved-thinking follow-up whose rewrite
// checkpoint (R) sits strictly below the prompt end. R is what the tier stores, so an
// exact-prompt restore prefills [R, L) instead of taking the (unavailable) zero-suffix path.
ninfer::PromptInput seed_conversation() {
    ninfer::PromptInput input;
    input.options.enable_thinking   = false;
    input.context_cache.session_key = "kvt-prefix-real";
    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = "Name one prime number.", .media = {}});
    input.messages.push_back(std::move(user));
    return input;
}

ninfer::PromptInput followup_conversation(const ninfer::GenerationResult& first) {
    ninfer::PromptInput input = seed_conversation();
    ninfer::ChatMessage assistant;
    assistant.role              = ninfer::ChatRole::Assistant;
    assistant.reasoning_content = first.reasoning;
    if (!first.content.empty()) {
        assistant.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = first.content, .media = {}});
    }
    input.messages.push_back(std::move(assistant));
    input.options.preserve_thinking = true;
    ninfer::ChatMessage next;
    next.role = ninfer::ChatRole::User;
    next.parts.push_back(ninfer::MessagePart{
        .kind  = ninfer::MessagePartKind::Text,
        .text  = "Now name a larger one and explain briefly.",
        .media = {}});
    input.messages.push_back(std::move(next));
    return input;
}

struct TierDirectoryShape {
    std::size_t blocks = 0;
    std::size_t notes  = 0;
};

TierDirectoryShape scan_tier_directory(const std::filesystem::path& directory) {
    TierDirectoryShape shape;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(directory, ec)) {
        if (!entry.is_regular_file(ec)) { continue; }
        const auto name = entry.path().filename().string();
        if (name.rfind("pfx-", 0) == 0) { ++shape.notes; } else { ++shape.blocks; }
    }
    return shape;
}

std::size_t count_sidecars_with_token_count(const std::filesystem::path& directory,
                                            std::size_t stored_prompt_tokens) {
    const auto suffix = "." + std::to_string(stored_prompt_tokens);
    std::size_t found = 0;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(directory, ec)) {
        if (!entry.is_regular_file(ec)) { continue; }
        const auto name = entry.path().filename().string();
        if (name.rfind("pfx-", 0) == 0 && name.size() >= suffix.size() &&
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
            ++found;
        }
    }
    return found;
}

void point_tier_at(const std::filesystem::path& directory) {
    const std::string value = directory.string();
    if (_putenv_s("NINFER_KV_TIER_DIR", value.c_str()) != 0) {
        throw std::runtime_error("could not set NINFER_KV_TIER_DIR for the fixture");
    }
}

bool same_tokens(const ninfer::GenerationResult& left, const ninfer::GenerationResult& right) {
    return left.generated_token_ids == right.generated_token_ids;
}

// Evidence helper for a failed comparison: prints both token lists and the first differing
// position so a divergence can be told apart from a total mismatch (wrong KV: divergence at
// token 0) and from late-decode numeric drift (divergence late in the output budget).
void dump_divergence(const char* name, const ninfer::GenerationResult& restored,
                     const ninfer::GenerationResult& cold) {
    std::cerr << "[divergence] " << name << " restored:";
    for (const auto token : restored.generated_token_ids) { std::cerr << ' ' << token; }
    std::cerr << "\n[divergence] " << name << " cold     :";
    for (const auto token : cold.generated_token_ids) { std::cerr << ' ' << token; }
    std::cerr << '\n';
    const auto count = std::min(restored.generated_token_ids.size(), cold.generated_token_ids.size());
    for (std::size_t index = 0; index < count; ++index) {
        if (restored.generated_token_ids[index] != cold.generated_token_ids[index]) {
            std::cerr << "[divergence] first difference at output token " << index << '\n';
            return;
        }
    }
    std::cerr << "[divergence] outputs identical up to the shorter list (lengths "
              << restored.generated_token_ids.size() << " vs " << cold.generated_token_ids.size()
              << ")\n";
}

void report_arm(const char* name, const ninfer::GenerationResult& restored,
                const ninfer::GenerationResult& cold) {
    std::cout << "[tier-real] " << name << ": prompt=" << restored.prompt.prompt_tokens
              << " reused=" << restored.reused_prompt_tokens
              << " computed=" << restored.computed_prefill_tokens
              << " ttft_hit_ms=" << restored.timings.first_token_seconds * 1000.0
              << " ttft_cold_ms=" << cold.timings.first_token_seconds * 1000.0
              << " prefill_hit_ms=" << restored.timings.prompt_wall_seconds * 1000.0
              << " prefill_cold_ms=" << cold.timings.prompt_wall_seconds * 1000.0 << '\n';
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    // The tier arms compare a restored run against a cold run generated in a different Engine.
    // With the auto-curved pool a few hundred experts stay unslotted and are served from the CPU
    // half, whose numerics differ from the device kernels; which experts land where depends on the
    // preceding traffic, so the comparison is a knife edge (measured 2026-11-09: the 500-tier arm
    // flipped its first token against an otherwise identical cold run). Pinning the pool to full
    // residency removes that variable: the tier contract is about KV restoration, not about which
    // half of a split layer happened to hold an expert.
    _putenv_s("NINFER_EXPERT_CACHE_MB", "19456");
    const auto guard = [](auto&& body) -> int {
        try {
            return body();
        } catch (const std::exception& error) {
            const std::string message = error.what();
            if (message.find("reservation") != std::string::npos ||
                message.find("runtime capacity") != std::string::npos) {
                std::cerr << "skip: " << message << std::endl;
                return 77;
            }
            std::cerr << "kv tier real test failed: " << message << std::endl;
            return 1;
        }
    };
    return guard([&]() -> int {
    const auto tier_root = std::filesystem::temp_directory_path() / "ninfer_kv_tier_real_test";
    const auto tier_dir  = tier_root / "tier";
    std::error_code ec;
    std::filesystem::remove_all(tier_root, ec);
    std::filesystem::create_directories(tier_dir, ec);
    point_tier_at(tier_dir);

    // The plain arm is one contiguous token stream from real text, so every arm shares the
    // exact same [0, 397) prefix and the suffixes continue natural prose. The store Engine
    // doubles as the tokenizer - no extra model load for the fixture.
    std::vector<ninfer::TokenId> plain_prefix;
    std::vector<ninfer::TokenId> short_prompt;
    std::vector<ninfer::TokenId> long_prompt;
    ninfer::GenerationResult seed_result;
    ninfer::GenerationResult stored_followup;
    ninfer::GenerationResult stored_plain;
    {
        ninfer::Engine store(tier_engine_options(artifact));
        const std::vector<ninfer::TokenId> stream = store.tokenize_text(fixture_text());
        if (stream.size() < kPlainPrefixTokens + kLongSuffixTokens) {
            std::cerr << "fixture text tokenized to only " << stream.size() << " token(s)\n";
            return 1;
        }
        plain_prefix.assign(stream.begin(), stream.begin() + kPlainPrefixTokens);
        short_prompt.assign(stream.begin(), stream.begin() + kPlainPrefixTokens + kShortSuffixTokens);
        long_prompt.assign(stream.begin(),
                           stream.begin() + kPlainPrefixTokens + kLongSuffixTokens);
        // Phase 1: one Engine stores both arms, and the plain prompt goes FIRST. The stored
        // KV bytes are the reference every restored run is compared against, and on the
        // streamed a3b artifact a greedy continuation is only reproducible while the expert
        // pool holds the same hot set (the dflash real test learned this the hard way) - so
        // the store's [0,397) must run at the same pool state as the reference prefill that
        // later cold-runs it, i.e. as the first request of a fresh engine.
        stored_plain = store.generate(store.prepare_tokens(plain_prefix), greedy_options(8, true));
        if (stored_plain.generated_token_ids.size() != 8) {
            std::cerr << "plain store prompt did not generate eight tokens\n";
            return 1;
        }
        seed_result = store.generate(store.prepare(seed_conversation()), greedy_options(8, true));
        if (seed_result.generated_token_ids.size() != 8) {
            std::cerr << "seed conversation did not produce its output budget\n";
            return 1;
        }
        stored_followup =
            store.generate(store.prepare(followup_conversation(seed_result)), greedy_options(8, true));
        if (stored_followup.generated_token_ids.size() != 8 ||
            stored_followup.reused_prompt_tokens == 0 ||
            stored_followup.prefix_reuse_path != ninfer::PrefixReusePath::Checkpoint) {
            std::cerr << "follow-up fixture did not reuse its assistant boundary in-process: reused="
                      << stored_followup.reused_prompt_tokens << '\n';
            return 1;
        }
    }
    const auto stored_shape = scan_tier_directory(tier_dir);
    if (stored_shape.blocks < 3 || stored_shape.notes < 3) {
        std::cerr << "store phase left " << stored_shape.blocks << " block(s) and "
                  << stored_shape.notes << " prefix note(s) on disk\n";
        return 1;
    }
    // The fallback directories need the STORE state, not the state later phases publish into
    // tier_dir (a restored run captures and stores its own prompt frontier), or the "longer
    // request finds nothing" scenario would be answered by its own phase-3 block.
    const auto stored_snapshot = tier_root / "tier-store-snapshot";
    std::filesystem::copy(tier_dir, stored_snapshot, std::filesystem::copy_options::recursive, ec);

    // Phase 2: a fresh Engine - the directory is the only channel. The exact-prompt restore
    // first (the pre-index behaviour, the 224/397 shape), then the whole-prompt decline.
    {
        ninfer::Engine reader(tier_engine_options(artifact));
        const ninfer::GenerationResult exact =
            reader.generate(reader.prepare(followup_conversation(seed_result)),
                            greedy_options(8, true));
        if (exact.reused_prompt_tokens == 0 ||
            exact.reused_prompt_tokens >= exact.prompt.prompt_tokens ||
            exact.prefix_reuse_path != ninfer::PrefixReusePath::Checkpoint ||
            exact.computed_prefill_tokens !=
                exact.prompt.prompt_tokens - exact.reused_prompt_tokens ||
            !same_tokens(exact, stored_followup)) {
            std::cerr << "exact-prompt tier restore lost the stored run: reused="
                      << exact.reused_prompt_tokens << " prompt=" << exact.prompt.prompt_tokens
                      << " computed=" << exact.computed_prefill_tokens << '\n';
            return 1;
        }
        const ninfer::GenerationResult cold_followup =
            reader.generate(reader.prepare(followup_conversation(seed_result)),
                            greedy_options(8, false));
        if (!same_tokens(cold_followup, stored_followup)) {
            std::cerr << "cold follow-up diverged from the stored run\n";
            return 1;
        }
        report_arm("exact-chat-followup", exact, cold_followup);

        const ninfer::GenerationResult cold_plain =
            reader.generate(reader.prepare_tokens(plain_prefix), greedy_options(8, false));
        if (cold_plain.generated_token_ids.size() != 8) {
            std::cerr << "cold plain oracle did not complete\n";
            return 1;
        }
        // A payload that covers the whole prompt has no suffix to prefill and no tail hidden to
        // sample from, so it must be declined: the request runs cold and matches the cold run
        // it just shared a pool state with.
        const ninfer::GenerationResult whole_prompt_plain =
            reader.generate(reader.prepare_tokens(plain_prefix), greedy_options(8, true));
        if (whole_prompt_plain.reused_prompt_tokens != 0 ||
            whole_prompt_plain.prefix_reuse_path != ninfer::PrefixReusePath::Root ||
            !same_tokens(whole_prompt_plain, cold_plain)) {
            std::cerr << "whole-prompt payload was imported instead of declined: reused="
                      << whole_prompt_plain.reused_prompt_tokens << '\n';
            return 1;
        }
        report_arm("whole-prompt-declined", whole_prompt_plain, cold_plain);
    }

    // Phase 3: the long-prefix arms. Each arm's cold reference runs in the SAME engine,
    // immediately before its restored run, and each PAIR runs first in its own fresh engine.
    // Two properties of the streamed a3b artifact shape this: a greedy continuation is only
    // token-reproducible while the expert pool holds the same hot set (the dflash real test
    // learned this), and the restored arm's [0,397) KV bytes were produced by the STORE
    // engine, so the comparison needs the reading engine's pool state to match a fresh
    // engine's - which heavy earlier traffic no longer guarantees. Running cold with reuse
    // off also means no in-process checkpoint is published, so the restored run can only
    // come from the tier. That is what these arms prove.
    {
        ninfer::Engine long_arm(tier_engine_options(artifact));
        const ninfer::GenerationResult cold_long =
            long_arm.generate(long_arm.prepare_tokens(long_prompt), greedy_options(8, false));
        const ninfer::GenerationResult long_hit =
            long_arm.generate(long_arm.prepare_tokens(long_prompt), greedy_options(8, true));
        if (long_hit.reused_prompt_tokens != kPlainPrefixTokens ||
            long_hit.prefix_reuse_path != ninfer::PrefixReusePath::Checkpoint ||
            long_hit.computed_prefill_tokens != kLongSuffixTokens ||
            !same_tokens(long_hit, cold_long)) {
            std::cerr << "4100-tier prefix restore missed or diverged: reused="
                      << long_hit.reused_prompt_tokens << " computed="
                      << long_hit.computed_prefill_tokens << '\n';
            dump_divergence("prefix-4100", long_hit, cold_long);
            return 1;
        }
        report_arm("prefix-4100", long_hit, cold_long);
    }
    {
        ninfer::Engine short_arm(tier_engine_options(artifact));
        const ninfer::GenerationResult cold_short =
            short_arm.generate(short_arm.prepare_tokens(short_prompt), greedy_options(8, false));
        const ninfer::GenerationResult short_hit =
            short_arm.generate(short_arm.prepare_tokens(short_prompt), greedy_options(8, true));
        if (short_hit.reused_prompt_tokens != kPlainPrefixTokens ||
            short_hit.prefix_reuse_path != ninfer::PrefixReusePath::Checkpoint ||
            short_hit.computed_prefill_tokens != kShortSuffixTokens ||
            !same_tokens(short_hit, cold_short)) {
            std::cerr << "500-tier prefix restore missed or diverged: reused="
                      << short_hit.reused_prompt_tokens << " computed="
                      << short_hit.computed_prefill_tokens << '\n';
            dump_divergence("prefix-500", short_hit, cold_short);
            return 1;
        }
        report_arm("prefix-500", short_hit, cold_short);

        // The restored runs had a suffix to prefill, so they crossed their own capture points
        // and published them: the index grows through reuse instead of freezing at the store.
        if (count_sidecars_with_token_count(tier_dir,
                                            kPlainPrefixTokens + kShortSuffixTokens) < 1 ||
            count_sidecars_with_token_count(tier_dir,
                                           kPlainPrefixTokens + kLongSuffixTokens) < 1) {
            std::cerr << "restored runs did not publish their own prompt-frontier notes\n";
            return 1;
        }
    }

    // Phase 4: the fallbacks. A copy of the stored directory with the notes deleted, and one
    // with every note corrupted, must both behave exactly like the pre-index tier: a longer
    // request finds nothing, runs cold, and still matches a cold run of the same prompt - the
    // reference generated in the same engine, immediately before, for the same pool-state
    // reason as phase 3.
    const auto no_index_dir  = tier_root / "tier-no-index";
    const auto bad_index_dir = tier_root / "tier-bad-index";
    std::filesystem::copy(stored_snapshot, no_index_dir, std::filesystem::copy_options::recursive, ec);
    std::filesystem::copy(stored_snapshot, bad_index_dir, std::filesystem::copy_options::recursive, ec);
    {
        std::error_code mutate_ec;
        for (const auto& entry : std::filesystem::directory_iterator(no_index_dir, mutate_ec)) {
            const auto name = entry.path().filename().string();
            if (name.rfind("pfx-", 0) == 0) { std::filesystem::remove(entry.path(), mutate_ec); }
        }
        for (const auto& entry : std::filesystem::directory_iterator(bad_index_dir, mutate_ec)) {
            const auto name = entry.path().filename().string();
            if (name.rfind("pfx-", 0) != 0) { continue; }
            std::FILE* file = std::fopen(entry.path().string().c_str(), "r+b");
            if (file == nullptr) { continue; }
            std::fseek(file, 20, SEEK_SET);  // inside the frontier field: the checksum must refuse
            const int flipped = 0x5a;
            std::fwrite(&flipped, 1, 1, file);
            std::fclose(file);
        }
    }
    const auto fallback = [&](const std::filesystem::path& directory, const char* name) -> int {
        point_tier_at(directory);
        ninfer::Engine engine(tier_engine_options(artifact));
        const ninfer::GenerationResult cold =
            engine.generate(engine.prepare_tokens(short_prompt), greedy_options(8, false));
        const ninfer::GenerationResult result =
            engine.generate(engine.prepare_tokens(short_prompt), greedy_options(8, true));
        if (result.reused_prompt_tokens != 0 ||
            result.prefix_reuse_path != ninfer::PrefixReusePath::Root ||
            result.computed_prefill_tokens != result.prompt.prompt_tokens ||
            !same_tokens(result, cold)) {
            std::cerr << name << " fallback changed the miss into something else: reused="
                      << result.reused_prompt_tokens << " computed=" << result.computed_prefill_tokens
                      << '\n';
            dump_divergence(name, result, cold);
            return 1;
        }
        report_arm(name, result, cold);
        return 0;
    };
    if (const int result = fallback(no_index_dir, "no-index-fallback"); result != 0) {
        return result;
    }
    if (const int result = fallback(bad_index_dir, "bad-index-fallback"); result != 0) {
        return result;
    }

    std::filesystem::remove_all(tier_root, ec);
    std::cout << "ok\n";
    return 0;
    });
}
