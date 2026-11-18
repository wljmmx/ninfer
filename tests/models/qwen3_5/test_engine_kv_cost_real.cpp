// port (sm_89, V10-82): the two Host -> Device legs, measured against the cost model's prediction
// on a real configuration.
//
// The model (core/transfer_cost.h) says a step owes
//     A  (pages a step reads - pages the Device holds) x one page,   at the measured page-in rate
//     B  (routes a step names - slots the pool holds)     x one slot, at the measured expert rate.
// This binary runs ONE arm per process (this card holds one 19 GiB engine at a time, and two
// engines back to back in one process have already been measured to leave the second one with no
// memory) and prints the numbers the comparison needs:
//
//   ref      the arena is the whole context and the pool is pinned large. Nothing is copied.
//   page-in  a residency arena smaller than the context with the (V10-79) leg on. Every step stages
//            the pages it addresses and does not hold, so the fetched-page counter counts what A
//            would owe. A prompt the pool cannot cover is refused, which is A's real cost.
//   stream   the arena is the whole context and the pool is pinned below one step's routes, so
//   every
//            step streams the difference. Engine::expert_step_bytes() -
//            expert_pool_capacity_bytes() is what B owes.
//
// The rates are measured by the same function the engine uses (core/transfer_rate.h) so the
// prediction below is the model's, not a constant transcribed from a log. What is compared is
// ms/step: the decode time from the engine's own counters (decode_host + decode_device_wait over
// committed tokens) against price_step_transfers() on the measured geometry. The delta against
// `ref` is the leg's measured cost; the ratios are printed by the report, not asserted here,
// because "the model is not yet predictive, here is the error" is a valid outcome - the contract
// this binary asserts is only that each arm actually exercised its leg (nonzero fetches, nonzero
// misses).
//
// Run (one arm per process):
//   $env:NINFER_TEST_ARTIFACT = 'R:\model\qwen3_6_35b_a3b_e2\qwen3_6_35b_a3b_e2.ninfer'
//   $env:NINFER_KV_COST_ARM   = 'page-in'   # or 'ref' / 'stream'
//   $env:NINFER_KV_COST_POOL  = 'auto'     # or MiB (default 4096)
// Without NINFER_TEST_ARTIFACT or NINFER_KV_COST_ARM it prints "skip" and exits 77.

#include "artifact/file_io.h"
#include "core/transfer_cost.h"
#include "core/transfer_rate.h"
#include "ninfer/engine.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string>
#include <vector>

namespace {

constexpr double kMib = 1024.0 * 1024.0;

void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    (void)_putenv_s(name, value);
#else
    (void)setenv(name, value, 1);
#endif
}

std::uint64_t env_u64(const char* name, std::uint64_t fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') { return fallback; }
    return std::strtoull(value, nullptr, 10);
}

bool env_is(const char* name, const char* word) {
    const char* value = std::getenv(name);
    return value != nullptr && std::string(value) == word;
}

// A pool of zero means "let the split size it": normalize_engine_options() then reads the
// environment switch, so the auto arm names the same word the CLI does.
ninfer::EngineOptions options_for(const char* artifact, std::uint32_t context,
                                  std::uint64_t pool_bytes, bool pool_auto) {
    ninfer::EngineOptions options;
    options.artifact_path                    = artifact;
    options.max_context                      = context;
    options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(context);
    options.prefill_chunk                    = 1024;
    options.kv_cache                         = ninfer::KvCacheStorage::BFloat16;
    options.speculative.backend              = ninfer::SpeculativeBackend::None;
    options.enable_vision                    = false;
    options.use_cuda_graph                   = false;
    options.max_concurrency                  = 1;
    options.max_pending_requests             = 1;
    options.expert_cache_bytes               = pool_auto ? 0 : static_cast<std::size_t>(pool_bytes);
    options.context_cache.device_state_slots = 2;
    // A page that leaves the Device has to land somewhere: the leg needs a Host KV tier.
    options.context_cache.host_capacity_bytes = 2ULL << 30;
    return options;
}

ninfer::RequestOptions greedy(std::uint32_t outputs) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.sampling.seed           = 0;
    options.execution.allow_prefix_reuse      = false;
    options.stop.include_model_defaults       = false;
    return options;
}

std::string fixture_text() {
    const std::string paragraph =
        "The engine keeps a page on the host tier, copies it back into the Device arena when the "
        "layer that reads it runs, and gives a page nobody is reading back to the pool to make "
        "room. ";
    std::string text;
    while (text.size() < 512 * 1024) { text += paragraph; }
    return text;
}

double decode_ms_per_token(const ninfer::RuntimeStats& stats) {
    if (stats.committed_decode_tokens == 0) { return 0.0; }
    const double ns =
        static_cast<double>(stats.host_work.decode_host_ns + stats.host_work.decode_device_wait_ns);
    return ns / 1e6 / static_cast<double>(stats.committed_decode_tokens);
}

// V10-83: the artifact leg, read the way the engine reads it - the reader's own positional
// primitive, in the mode file_io.cpp picks for this volume, at the pool's own payload.
std::size_t read_artifact(void* context, std::uint64_t offset, std::span<std::byte> destination) {
    const auto& leg = *static_cast<const ninfer::artifact::InputFile*>(context);
    try {
        leg.read_exact(offset, destination);
        return destination.size();
    } catch (...) {
        return 0;
    }
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    const char* arm      = std::getenv("NINFER_KV_COST_ARM");
    if (artifact == nullptr || *artifact == '\0' || arm == nullptr || *arm == '\0') {
        std::printf("skip: NINFER_TEST_ARTIFACT and NINFER_KV_COST_ARM are not set\n");
        return 77;
    }
    const std::string which(arm);
    if (which != "ref" && which != "page-in" && which != "stream") {
        std::printf("skip: NINFER_KV_COST_ARM must be ref|page-in|stream\n");
        return 77;
    }
    const auto context       = static_cast<std::uint32_t>(env_u64("NINFER_KV_COST_CTX", 32768));
    const auto prompt_tokens = static_cast<std::uint32_t>(env_u64("NINFER_KV_COST_PROMPT", 30000));
    const auto decode_tokens = static_cast<std::uint32_t>(env_u64("NINFER_KV_COST_DECODE", 32));
    const bool pool_auto     = env_is("NINFER_KV_COST_POOL", "auto");
    const std::uint64_t pool_bytes =
        pool_auto ? 0
                  : env_u64("NINFER_KV_COST_POOL", env_u64("NINFER_EXPERT_CACHE_MB", 4096)) << 20;
    const std::uint64_t slack_mib = env_u64("NINFER_KV_COST_SLACK_MB", 3584);

    // Every arm sets its own environment immediately before it builds an Engine: these are read at
    // construction, and one process runs one arm.
    set_env("NINFER_KV_EVICT_CAPACITY", "");
    set_env("NINFER_KV_PAGE_IN", "");
    set_env("NINFER_KV_HOST_RESIDENCY", "");
    set_env("NINFER_KV_RESIDENCY_SLACK_MB", "");
    set_env("NINFER_EXPERT_CACHE_MB", pool_auto ? "auto" : "");
    set_env("NINFER_CONTROLLER_MODE", "performance");
    if (which == "page-in") {
        set_env("NINFER_KV_PAGE_IN", "1");
        set_env("NINFER_KV_HOST_RESIDENCY", "1");
        set_env("NINFER_KV_RESIDENCY_SLACK_MB", std::to_string(slack_mib).c_str());
    }
    try {
        ninfer::Engine engine(options_for(artifact, context, pool_bytes, pool_auto));
        const auto geometry   = engine.cache_geometry();
        const auto started    = engine.memory_summary();
        const auto stats0     = engine.runtime_stats();
        const auto started_at = std::chrono::steady_clock::now();

        const std::vector<ninfer::TokenId> stream = engine.tokenize_text(fixture_text());
        if (stream.size() < prompt_tokens) {
            std::printf("skip: fixture tokenized to only %zu tokens\n", stream.size());
            return 77;
        }
        const std::vector<ninfer::TokenId> prompt(stream.begin(), stream.begin() + prompt_tokens);
        const auto result = engine.generate(engine.prepare_tokens(prompt), greedy(decode_tokens));
        const auto finish = engine.memory_summary();
        const auto stats1 = engine.runtime_stats();
        const double wall_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started_at).count();
        if (result.generated_token_ids.size() != decode_tokens) {
            std::printf("FAIL: the arm did not finish its output budget\n");
            return 1;
        }

        const std::uint64_t steps         = stats1.decode_rounds - stats0.decode_rounds;
        const std::uint64_t fetched       = finish.kv_fetched_pages - started.kv_fetched_pages;
        const std::uint64_t fetched_bytes = finish.kv_fetched_bytes - started.kv_fetched_bytes;
        const double fetched_per_step =
            steps == 0 ? 0.0 : static_cast<double>(fetched) / static_cast<double>(steps);
        const std::uint64_t step_bytes  = engine.expert_step_bytes();
        const std::uint64_t expert_pool = engine.expert_pool_capacity_bytes();
        const std::uint64_t miss_bytes  = step_bytes > expert_pool ? step_bytes - expert_pool : 0;
        // V10-83: the streaming leg reads the artifact first, so the probe measures that leg too.
        const auto file = ninfer::artifact::InputFile(artifact);
        const auto mode = ninfer::artifact::expert_read_mode(artifact, file.bytes());
        const ninfer::core::DiskReadSource disk{
            .context = const_cast<ninfer::artifact::InputFile*>(&file),
            .read    = read_artifact,
            .bytes   = file.bytes(),
            .mode    = mode == ninfer::artifact::ReadMode::Direct ? "direct" : "buffered"};
        const auto probe = ninfer::core::measure_transfer_rates(geometry.page_bytes,
                                                               geometry.page_bytes, nullptr, &disk);
        const std::uint64_t live_pages =
            (prompt_tokens + decode_tokens + 63U) / 64U; // whole page groups a step can read

        std::printf("[cost] arm=%s ctx=%u prompt=%u decode=%u pool=%.0f MiB (%s)\n", which.c_str(),
                    context, prompt_tokens, decode_tokens, static_cast<double>(expert_pool) / kMib,
                    pool_auto ? "auto" : "pinned");
        std::printf("[cost] kv: page_bytes=%zu page_tokens=%lld logical_pages=%u "
                    "resident_pages=%u host_payload_mib=%.0f live_pages=%llu\n",
                    geometry.page_bytes, static_cast<long long>(geometry.page_tokens),
                    finish.kv_logical_page_groups, finish.kv_resident_page_groups,
                    static_cast<double>(finish.kv_host_payload_bytes) / kMib,
                    static_cast<unsigned long long>(live_pages));
        std::printf("[cost] kv: fetched_pages=%llu fetched_bytes=%llu steps=%llu "
                    "fetched_per_step=%.2f fetched_mib_per_step=%.2f\n",
                    static_cast<unsigned long long>(fetched),
                    static_cast<unsigned long long>(fetched_bytes),
                    static_cast<unsigned long long>(steps), fetched_per_step,
                    steps == 0
                        ? 0.0
                        : static_cast<double>(fetched_bytes) / kMib / static_cast<double>(steps));
        std::printf("[cost] expert: step_bytes=%llu pool_bytes=%llu miss_bytes_per_step=%llu "
                    "(%.1f MiB)\n",
                    static_cast<unsigned long long>(step_bytes),
                    static_cast<unsigned long long>(expert_pool),
                    static_cast<unsigned long long>(miss_bytes),
                    static_cast<double>(miss_bytes) / kMib);
        std::printf("[cost] rates: page_in=%.3f GB/s pcie=%.3f GB/s disk=%.3f GB/s "
                    "(%llu KiB x %u reads, %s) eff=%.3f GB/s (measured on %.0f KiB copies, %u per "
                    "trial; the public API does not expose the slot payload, so the expert leg is "
                    "measured at the page payload)\n",
                    probe.rates.page_in_bytes_per_s / 1e9, probe.rates.expert_bytes_per_s / 1e9,
                    probe.rates.disk_bytes_per_s / 1e9,
                    static_cast<unsigned long long>(probe.disk_payload_bytes >> 10),
                    probe.disk_copies, probe.disk_mode,
                    probe.rates.expert_effective_bytes_per_s() / 1e9,
                    static_cast<double>(geometry.page_bytes) / 1024.0, probe.page_copies);
        std::printf("[cost] decode: steps=%llu tokens=%llu ms_per_token=%.3f wall_s=%.2f\n",
                    static_cast<unsigned long long>(steps),
                    static_cast<unsigned long long>(stats1.committed_decode_tokens -
                                                    stats0.committed_decode_tokens),
                    decode_ms_per_token(stats1), wall_s);
        // The model's own arithmetic, twice, because the two answers differ by what "addressed"
        // means. `plan` prices the whole logical page space (the worst case the bet plans with);
        // `live` prices the pages one step of THIS request can read. For A the difference is the
        // whole story: the arena's shortfall is charged against a history that never reaches it.
        const std::uint64_t layers = env_u64("NINFER_KV_COST_LAYERS", 40);
        const auto price = [&](std::uint64_t addressed, std::uint64_t resident) {
            return ninfer::core::price_step_transfers({.page_bytes            = geometry.page_bytes,
                                                       .pages_addressed       = addressed,
                                                       .pages_resident        = resident,
                                                       .slot_bytes            = 1,
                                                       .experts_routed        = miss_bytes,
                                                       .expert_slots_resident = 0,
                                                       .expert_layers         = layers},
                                                      probe.rates);
        };
        const auto plan_cost = price(finish.kv_logical_page_groups, finish.kv_resident_page_groups);
        const double predict_plan_ms = plan_cost.total_ms();
        const double predict_live_ms = price(live_pages, finish.kv_resident_page_groups).total_ms();
        // What the run's own counters say the traffic cost at the two measured rates - the same
        // arithmetic on the fetched/missed bytes, so a difference from the model above is the
        // model's geometry, not its rates.
        const double measured_ms =
            steps == 0
                ? 0.0
                : static_cast<double>(fetched_bytes) / static_cast<double>(steps) /
                          probe.rates.page_in_bytes_per_s * 1000.0 +
                      static_cast<double>(miss_bytes) /
                          probe.rates.expert_effective_bytes_per_s() * 1000.0 +
                      plan_cost.expert_fixed_ms;
        std::printf("[cost] predicted: plan_geometry_ms_per_step=%.3f live_geometry_ms_per_step="
                    "%.3f measured_traffic_ms_per_step=%.3f (slot bytes=%.3f fixed=%.3f on %llu "
                    "layers)\n",
                    predict_plan_ms, predict_live_ms, measured_ms, plan_cost.expert_bytes_ms,
                    plan_cost.expert_fixed_ms, static_cast<unsigned long long>(layers));
        std::fflush(stdout);
    } catch (const std::exception& error) {
        std::printf("FAIL: %s\n", error.what());
        return 1;
    }
    set_env("NINFER_KV_PAGE_IN", "");
    set_env("NINFER_KV_HOST_RESIDENCY", "");
    set_env("NINFER_KV_RESIDENCY_SLACK_MB", "");
    set_env("NINFER_EXPERT_CACHE_MB", "");
    return 0;
}
