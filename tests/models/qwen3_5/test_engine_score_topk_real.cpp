#include "ninfer/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << message << '\n';
    ++failures;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }

    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.purpose       = ninfer::EnginePurpose::CausalScoring;
    options.max_context   = 2048;
    options.kv_cache      = ninfer::KvCacheStorage::Fp8E4M3Row256;
    ninfer::Engine engine(options);

    const std::string paragraph =
        "NInfer scores each target token from the preceding hidden state. "
        "Every evaluation window owns fresh state and a fresh KV address space.\n";
    std::string text;
    std::vector<ninfer::TokenId> tokens;
    while (tokens.size() < 513) {
        text += paragraph;
        tokens = engine.tokenize_text(text);
    }
    tokens.resize(513);
    constexpr std::uint32_t kRanks = 32;

    const std::vector<float> plain = engine.score_tokens(tokens, 1);
    check(plain.size() == 512, "score_tokens returned an unexpected size");

    const ninfer::ScoreTopK topk = engine.score_tokens_topk(tokens, 1, kRanks);
    check(topk.ids.size() == 512 * kRanks, "score_tokens_topk returned an unexpected id count");
    check(topk.logprobs.size() == 512 * kRanks,
          "score_tokens_topk returned an unexpected logprob count");
    check(topk.log_normalizer.size() == 512, "score_tokens_topk returned an unexpected normalizer");

    std::size_t argmax_matches = 0;
    for (std::size_t position = 0; position < 512; ++position) {
        const std::size_t base = position * kRanks;
        const float observed   = plain[position];
        check(std::isfinite(observed) && observed <= 0.0F,
              "score_tokens returned an invalid log probability");
        // The candidate list is ordered by decreasing logprob and never repeats a token, and every
        // candidate is a real vocabulary id with a real probability.
        std::set<std::uint32_t> seen;
        float previous = std::numeric_limits<float>::infinity();
        double covered = 0.0;
        for (std::uint32_t rank = 0; rank < kRanks; ++rank) {
            const float logprob    = topk.logprobs[base + rank];
            const std::uint32_t id = topk.ids[base + rank];
            check(std::isfinite(logprob) && logprob <= 0.0F, "a candidate logprob is invalid");
            check(logprob <= previous, "candidate logprobs are not non-increasing");
            check(id < 300000U, "a candidate id is outside any vocabulary");
            check(seen.insert(id).second, "a candidate list repeats a token");
            previous = logprob;
            covered += std::exp(static_cast<double>(logprob));
        }
        check(covered <= 1.0 + 1.0e-3, "candidate probabilities sum above one");
        check(std::isfinite(topk.log_normalizer[position]),
              "a column log-normalizer is not finite");
        // top-1 is the observed token's own logprob, bit for bit: both entries run the same
        // projection, the same maximum and the same reduction order.
        if (topk.ids[base] == static_cast<std::uint32_t>(tokens[position + 1])) {
            ++argmax_matches;
            if (topk.logprobs[base] != observed) {
                std::cerr << "position " << position << ": top-1 logprob " << topk.logprobs[base]
                          << " differs from score_tokens " << observed << '\n';
                ++failures;
            }
        }
    }
    check(argmax_matches > 0, "no position had the observed token as its argmax");

    // A suffix window re-derives state from the shorter prompt; the same relation must hold.
    const std::vector<float> suffix_plain = engine.score_tokens(tokens, 257);
    const ninfer::ScoreTopK suffix_topk   = engine.score_tokens_topk(tokens, 257, kRanks);
    check(suffix_plain.size() == 256 && suffix_topk.log_normalizer.size() == 256,
          "a suffix window returned an unexpected shape");
    std::size_t suffix_matches = 0;
    for (std::size_t position = 0; position < 256; ++position) {
        const std::size_t base = position * kRanks;
        if (suffix_topk.ids[base] == static_cast<std::uint32_t>(tokens[position + 257])) {
            ++suffix_matches;
            if (suffix_topk.logprobs[base] != suffix_plain[position]) {
                std::cerr << "suffix position " << position << ": top-1 does not match\n";
                ++failures;
            }
        }
    }
    check(suffix_matches > 0, "no suffix position had the observed token as its argmax");

    // k is validated at runtime, and the plain entry keeps its own verdicts.
    bool rejected = false;
    try {
        (void)engine.score_tokens_topk(tokens, 1, 0);
    } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "score_tokens_topk accepted k=0");
    rejected = false;
    try {
        (void)engine.score_tokens_topk(tokens, 1, ninfer::kScoreTopkCapacity + 1);
    } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "score_tokens_topk accepted k above the capacity");

    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "OK score_topk_real | " << argmax_matches << "/512 and " << suffix_matches
              << "/256 positions had the observed token as their argmax\n";
    return 0;
}
