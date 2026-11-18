// port (sm_89): sequential-construction diagnostic. V10-73.
//
// The resize-under-load acceptance builds several engines back to back in one process, and on
// some runs the SECOND one is built degraded: a smaller expert pool, a clamped host tier, and a
// refused 2 GiB pinned context arena whose stale CUDA error then surfaces at the next kernel
// launch. This tool isolates that pattern - construct, touch, destroy, construct again with the
// same options - and prints the accounting around every step so the difference has a number
// instead of a guess.
//
// usage: diag-second-engine <model.ninfer> [rounds] [outputs]
#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#if defined(_WIN32)
#    include <windows.h>
#    include <psapi.h>
#endif

namespace {

// The three numbers the accounting is made of: what the process sees of Device memory, what the
// page file can still back, and how much of it is charged to this process.
struct Account {
    std::uint64_t device_free  = 0;
    std::uint64_t device_total = 0;
    std::uint64_t commit_left  = 0;
    std::uint64_t commit_used  = 0;
};

Account read_account() {
    Account account;
    if (cudaMemGetInfo(&account.device_free, &account.device_total) != cudaSuccess) {
        (void)cudaGetLastError();
    }
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status) != 0) { account.commit_left = status.ullAvailPageFile; }
    PROCESS_MEMORY_COUNTERS_EX counters{};
    if (GetProcessMemoryInfo(GetCurrentProcess(),
                             reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                             sizeof(counters)) != 0) {
        account.commit_used = counters.PrivateUsage;
    }
#endif
    return account;
}

void print_account(const char* what, const Account& account) {
    std::fprintf(
        stderr, "[probe] %-22s device_free=%llu MiB commit_left=%llu MiB process_commit=%llu MiB\n",
        what, static_cast<unsigned long long>(account.device_free >> 20),
        static_cast<unsigned long long>(account.commit_left >> 20),
        static_cast<unsigned long long>(account.commit_used >> 20));
    std::fflush(stderr);
}

ninfer::EngineOptions base_options(const char* artifact, std::uint64_t cache_bytes) {
    ninfer::EngineOptions options;
    options.artifact_path                     = artifact;
    options.max_context                       = 4096;
    options.kv_capacity                       = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk                     = 1024;
    options.kv_cache                          = ninfer::KvCacheStorage::BFloat16;
    options.speculative.backend               = ninfer::SpeculativeBackend::None;
    options.enable_vision                     = false;
    options.use_cuda_graph                    = false;
    options.max_concurrency                   = 4;
    options.max_pending_requests              = 8;
    options.expert_cache_bytes                = cache_bytes;
    options.context_cache.device_state_slots  = 4;
    options.context_cache.host_capacity_bytes = 2ULL << 30;
    return options;
}

ninfer::RequestOptions greedy_options(std::uint32_t outputs) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.sampling.seed           = 0;
    options.execution.allow_prefix_reuse      = false;
    options.stop.include_model_defaults       = false;
    return options;
}

// One construction, one short generation, one teardown - the pattern the acceptance runs.
struct Round {
    std::uint64_t pool_bytes = 0;
    bool generated           = false;
};

Round run_round(const char* artifact, std::uint64_t cache_bytes,
                const std::vector<ninfer::TokenId>& prompt, std::uint32_t outputs,
                const char* name) {
    Round round;
    print_account((std::string(name) + " before").c_str(), read_account());
    {
        ninfer::Engine engine(base_options(artifact, cache_bytes));
        round.pool_bytes = engine.expert_pool_capacity_bytes();
        print_account((std::string(name) + " built").c_str(), read_account());
        const ninfer::GenerationResult result =
            engine.generate(engine.prepare_tokens(prompt, false), greedy_options(outputs));
        round.generated = result.generated_token_ids.size() == outputs;
        std::fprintf(stderr, "[probe] %-22s pool=%llu MiB produced=%zu/%u tokens\n", name,
                     static_cast<unsigned long long>(round.pool_bytes >> 20),
                     result.generated_token_ids.size(), outputs);
        std::fflush(stderr);
    }
    print_account((std::string(name) + " destroyed").c_str(), read_account());
    return round;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: diag-second-engine <model.ninfer> [rounds] [outputs]\n");
        return 2;
    }
    const char* artifact = argv[1];
    const int rounds     = argc > 2 ? std::atoi(argv[2]) : 2;
    const auto outputs   = static_cast<std::uint32_t>(argc > 3 ? std::atoi(argv[3]) : 8);
    (void)_putenv_s("NINFER_EXPERT_DEBUG", "1");

    try {
        // The tier the acceptance uses: one decode step's routed activation, doubled. Read from a
        // tokenizer-shaped engine, exactly as the test's fixture block does.
        std::vector<ninfer::TokenId> prompt;
        std::uint64_t tier_full = 0;
        {
            ninfer::Engine tokenizer(base_options(artifact, 0));
            const std::string text =
                "The engine keeps its expert pool elastic: while requests run, a resize is "
                "refused; "
                "between them, the pool may hand whole segments back to the driver and take them "
                "again, and a running layer must never address an unmapped window.";
            prompt    = tokenizer.tokenize_text(text);
            tier_full = tokenizer.expert_step_bytes() * 2;
        }
        if (prompt.empty()) {
            std::fprintf(stderr, "the artifact tokenized the fixture to nothing\n");
            return 1;
        }
        print_account("tokenizer destroyed", read_account());
        // An arm that names a pool the Device can hold whole builds NO host tier, so the commit
        // this run leaves behind afterwards is what the rest of the teardown is responsible for.
        if (const char* pool_mb = std::getenv("NINFER_PROBE_POOL_MB");
            pool_mb != nullptr && *pool_mb) {
            tier_full = std::strtoull(pool_mb, nullptr, 10) << 20;
        }
        std::fprintf(stderr, "[probe] construction tier = %llu MiB\n",
                     static_cast<unsigned long long>(tier_full >> 20));

        std::uint64_t first_pool = 0;
        bool degraded            = false;
        for (int index = 0; index < rounds; ++index) {
            const std::string name = "engine-" + std::to_string(index);
            const Round round      = run_round(artifact, tier_full, prompt, outputs, name.c_str());
            if (!round.generated) {
                std::fprintf(stderr, "[probe] %s FAILED to produce its tokens\n", name.c_str());
            }
            if (index == 0) {
                first_pool = round.pool_bytes;
            } else if (round.pool_bytes < first_pool) {
                degraded = true;
                std::fprintf(stderr, "[probe] DEGRADED: pool %llu MiB against %llu MiB\n",
                             static_cast<unsigned long long>(round.pool_bytes >> 20),
                             static_cast<unsigned long long>(first_pool >> 20));
            }
        }
        std::printf("%s\n", degraded ? "degraded" : "ok");
        return degraded ? 1 : 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "diag-second-engine failed: %s\n", error.what());
        return 1;
    }
}
