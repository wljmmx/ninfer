// 13.5-5 / open #1 acceptance: the KV reallocation entry must never penetrate an engine that
// still has a request somewhere in its lifecycle. The 2026-11-05 acceptance measured one
// penetration under a live four-way batch on the e2 artifact - once refused, once through - and
// the batch survived it. This file is the instrument for that claim: four clients submit
// back-to-back while two hammer threads call Engine::resize_kv_pages() as fast as they can.
//
// The contract the fix has to satisfy:
//   * every attempt is either refused ("requires an idle engine" / "lost its idle window") or is
//     legal and safe. A shrink that lands while the engine has work is a penetration and fails the
//     test: `outstanding_` counts a request from submission (before it is even queued) until its
//     consumer releases it, so there is no window in which a live request is invisible.
//   * the workload itself is never damaged: every request completes its full budget with no
//     preemption and no replay.
//
// The attempts ask for a REAL shrink (edge minus eight page groups), not a no-op, because the
// guard is what is being measured; a penetration is detected by the call returning instead of
// throwing, and reported with the `[resize]` verdict lines the engine prints.

#include "ninfer/engine.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr std::uint32_t kLanes       = 4;
constexpr std::uint32_t kOutputs     = 16;
constexpr std::uint32_t kPromptTokens = 64;
constexpr std::uint32_t kQuietOutputs = 4;
constexpr int kAttempts              = 250;
constexpr int kBusyAttempts          = 100; // phase 1 target: saturated load
constexpr int kIdleAttempts          = 150; // phase 2 target: all-zero gauge snapshot
constexpr int kHammerThreads         = 2;

std::atomic<int> g_penetrations{0};
std::atomic<int> g_legal{0};
std::atomic<int> g_noop{0};
std::atomic<int> g_refused{0};
std::atomic<int> g_idle_attempts{0};
std::atomic<int> g_active_requests{0};
std::atomic<int> g_completed{0};
std::atomic<int> g_damaged{0};
std::atomic<std::uint32_t> g_initial_edge{0};
std::atomic<bool> g_stop{false};         // the hammer threads
std::atomic<bool> g_clients_stop{false}; // the saturated clients (phase 1 only)
std::atomic<std::uint32_t> g_edge{0};

ninfer::EngineOptions base_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                     = artifact;
    options.max_context                       = 4096;
    options.kv_capacity                       = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk                     = 1024;
    options.kv_cache                          = ninfer::KvCacheStorage::BFloat16;
    options.speculative.backend               = ninfer::SpeculativeBackend::None;
    options.enable_vision                     = false;
    options.use_cuda_graph                    = false;
    options.max_concurrency                   = kLanes;
    options.max_pending_requests              = 2 * kLanes;
    options.expert_cache_bytes                = 12288ULL << 20;
    options.context_cache.device_state_slots  = kLanes;
    options.context_cache.host_capacity_bytes = 2ULL << 30;
    return options;
}

ninfer::RequestOptions greedy_options(std::uint32_t outputs = kOutputs) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.sampling.seed           = 0;
    options.execution.allow_prefix_reuse      = false;
    options.stop.include_model_defaults       = false;
    return options;
}

std::vector<std::string> fixture_prompts() {
    return {
        "The engine keeps its expert pool elastic: while requests run, a resize is refused; "
        "between them, the pool may hand whole segments back to the driver and take them again, "
        "and a running layer must never address an unmapped window. A paged cache gives each "
        "sequence only the pages it has actually filled, so a reservation can shrink without "
        "invalidating anything a live request still owns.",
        "The observatory sits above the cloud line, where the air is thin and dry and the stars do "
        "not twinkle. Astronomers cool the sensor slowly over hours, because a sudden chill leaves "
        "strain in the glass and the strain becomes a false signal in every exposure. The schedule "
        "for the night is a long argument between the people who want deep fields and the people "
        "who want planets.",
        "The kitchen at the harbor opens before sunrise. The bakers let the dough rise overnight in "
        "the cold room, and in the morning it is soft and smells of wheat. The oven is stone, and "
        "the stone holds heat from yesterday's fire, so the first batches come out with a crust "
        "that the later ones never quite match.",
        "The canal was dug by hand over three generations, and the ledgers of the work are still "
        "kept in the archive by the water office. Each lock has a name, and the names follow the "
        "families that lived beside the bank when the cut first filled. Barges still crawl the "
        "length of it in summer, and the mules are gone but the towpath remains."};
}

std::uint64_t busy_gauges(const ninfer::RuntimeStats& stats) {
    return static_cast<std::uint64_t>(stats.running_requests) + stats.prefilling_requests +
           stats.decode_ready_requests + stats.waiting_requests + stats.paused_requests +
           stats.replaying_requests + stats.materializing_requests +
           stats.capture_pending_requests + stats.terminal_pending_requests;
}

void hammer(ninfer::Engine* engine) {
    while (!g_stop.load(std::memory_order_relaxed)) {
        // A real move every time: the edge bounces between the construction value and one step
        // below it, so an accepted attempt is always observable as a changed edge (a no-op passes
        // the same guards but says nothing about them). Both directions walk the same contract.
        std::uint32_t used = g_edge.load(std::memory_order_relaxed);
        if (used == 0) { break; }
        const std::uint32_t construction = g_initial_edge.load(std::memory_order_relaxed);
        const std::uint32_t target = used + 8 <= construction ? used + 8 : used - 8;
        // Count this attempt when the published snapshot says the engine is idle: in that state
        // only the live-state leg (any_inflight) can refuse it.
        if (busy_gauges(engine->runtime_stats()) == 0) { g_idle_attempts.fetch_add(1); }
        // Whether an accepted attempt illegally landed on a live request is decided by the engine
        // itself: every accept runs a verdict that requires outstanding_ == 0 under the same lock
        // that owns the mutation, and the engine counts any accept that breaks it
        // (Engine::resize_accepted_inflight, asserted at the end). Sampling counters out here
        // cannot tell a legal accept from a penetration - a request finishing inside the 80 ms
        // settle grace, or the next submission right after the return, looks identical.
        try {
            const std::uint32_t after = engine->resize_kv_pages(target);
            if (after != used && g_edge.compare_exchange_strong(used, after)) {
                g_legal.fetch_add(1);
                if (g_legal.load() <= 4) {
                    std::cerr << "[legal] idle-window KV resize applied: " << target << " -> "
                              << after << " page(s)\n";
                }
            } else {
                g_noop.fetch_add(1);
            }
        } catch (const std::invalid_argument& error) {
            // Target above the construction edge would be a test bug, not a contract refusal.
            std::cerr << "[FAIL] resize rejected as invalid: " << error.what() << '\n';
            std::exit(1);
        } catch (const std::logic_error&) {
            g_refused.fetch_add(1);
        }
        std::this_thread::yield();
    }
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    // The controller is pinned to stable here: the KV L1 dial would otherwise step the edge down
    // on its own 60 s clock, and this test's whole assertion is that the edge does not move while
    // requests are live. Stable also disables the pool dial; the lane-count actuator is the one
    // window surface this test does NOT need (admission is exercised by the saturated phase).
    _putenv_s("NINFER_CONTROLLER_MODE", "stable");
    _putenv_s("NINFER_CONTROLLER_SERIAL_OFF", "1");
    _putenv_s("NINFER_EXPERT_DEBUG", "0");
    try {
        ninfer::Engine engine(base_options(artifact));
        std::vector<std::vector<ninfer::TokenId>> prompts;
        for (const std::string& text : fixture_prompts()) {
            const std::vector<ninfer::TokenId> stream = engine.tokenize_text(text);
            if (stream.size() < kPromptTokens) {
                std::cerr << "a fixture paragraph tokenized to only " << stream.size() << '\n';
                return 1;
            }
            prompts.emplace_back(stream.begin(), stream.begin() + kPromptTokens);
        }

        const std::uint32_t edge = engine.memory_summary().kv_capacity_page_groups;
        if (edge < 32) {
            std::cout << "skip: the KV pool is too small to shrink under load (" << edge
                      << " page(s))\n";
            return 77;
        }
        g_edge.store(edge, std::memory_order_relaxed);
        g_initial_edge.store(edge, std::memory_order_relaxed);

        std::atomic<bool> clients_started{false};
        std::vector<std::thread> clients;
        for (std::uint32_t lane = 0; lane < kLanes; ++lane) {
            clients.emplace_back([&engine, &prompts, lane, &clients_started] {
                clients_started.store(true, std::memory_order_release);
                while (!g_clients_stop.load(std::memory_order_relaxed)) {
                    ninfer::GenerationHandle handle;
                    {
                        struct Active {
                            Active() { g_active_requests.fetch_add(1, std::memory_order_acq_rel); }
                            ~Active() { g_active_requests.fetch_sub(1, std::memory_order_acq_rel); }
                        } active;
                        ninfer::PreparedPrompt prepared = engine.prepare_tokens(prompts[lane]);
                        if (!prepared) {
                            std::cerr << "[FAIL] the workload prompt failed preparation\n";
                            ++g_damaged;
                            return;
                        }
                        handle = engine.submit(std::move(prepared), greedy_options());
                    }
                    const ninfer::GenerationResult result = handle.wait();
                    if (result.generated_token_ids.size() != kOutputs ||
                        result.finish_reason != ninfer::FinishReason::OutputLimit ||
                        result.scheduling.preemptions != 0 || result.scheduling.replayed_tokens != 0) {
                        std::cerr << "[FAIL] lane " << lane << " was damaged under the hammer: "
                                  << result.generated_token_ids.size() << " tokens, reason "
                                  << static_cast<int>(result.finish_reason) << ", preemptions "
                                  << result.scheduling.preemptions << ", replayed "
                                  << result.scheduling.replayed_tokens << '\n';
                        ++g_damaged;
                    }
                    g_completed.fetch_add(1);
                }
            });
        }
        while (!clients_started.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
        // Let the first batch actually occupy the engine before the hammer starts, otherwise every
        // attempt would land in the one window (an empty queue) where a refusal is trivial.
        std::this_thread::sleep_for(std::chrono::milliseconds{500});

        std::vector<std::thread> hammers;
        for (int index = 0; index < kHammerThreads; ++index) {
            hammers.emplace_back(hammer, &engine);
        }
        // ---- phase 1: saturated load ----
        const auto busy_deadline = std::chrono::steady_clock::now() + std::chrono::seconds{60};
        while (std::chrono::steady_clock::now() < busy_deadline &&
               g_refused.load() + g_penetrations.load() < kBusyAttempts) {
            std::this_thread::sleep_for(std::chrono::milliseconds{10});
        }
        const int busy_attempts = g_refused.load() + g_penetrations.load();
        std::cerr << "[handoff] phase 1 (saturated): attempts=" << busy_attempts << '\n';

        // ---- phase 2: the idle pre-queue window ----
        // With the busy clients parked and one quiet request at a time, the engine publishes an
        // all-zero gauge snapshot between requests, and the window this phase aims at is the one
        // inside submit(): the request counts (outstanding_) while it is in no container yet. Each
        // attempt is only *counted* when the nine published gauges are zero - so a refusal here is
        // the live-state leg answering, which is exactly the leg the 2026-11-05 penetration beat.
        g_clients_stop.store(true, std::memory_order_relaxed);
        for (auto& thread : clients) { thread.join(); }
        clients.clear();
        std::thread quiet([&engine, &prompts] {
            while (!g_stop.load(std::memory_order_relaxed)) {
                auto prepared = engine.prepare_tokens(prompts[0]);
                if (!prepared) {
                    ++g_damaged;
                    continue;
                }
                g_active_requests.fetch_add(1, std::memory_order_acq_rel);
                const ninfer::GenerationResult result =
                    engine.submit(std::move(prepared), greedy_options(kQuietOutputs)).wait();
                g_active_requests.fetch_sub(1, std::memory_order_acq_rel);
                if (result.generated_token_ids.size() != kQuietOutputs) { ++g_damaged; }
                std::this_thread::sleep_for(std::chrono::milliseconds{120});
            }
        });
        // The hammer threads keep running; this loop only observes the gauge state to count the
        // attempts that landed in an all-zero window, and to stop once enough of them have.
        const auto idle_deadline = std::chrono::steady_clock::now() + std::chrono::seconds{60};
        int idle_window_refusals = 0;
        while (std::chrono::steady_clock::now() < idle_deadline &&
               g_idle_attempts.load() < kIdleAttempts) {
            std::this_thread::sleep_for(std::chrono::milliseconds{5});
        }
        idle_window_refusals = g_idle_attempts.load();
        g_stop.store(true, std::memory_order_relaxed);
        for (auto& thread : hammers) { thread.join(); }
        quiet.join();

        const int attempts = g_refused.load() + g_penetrations.load() + g_legal.load();
        const std::uint32_t after = engine.memory_summary().kv_capacity_page_groups;
        const std::size_t engine_penetrations = engine.resize_accepted_inflight();
        std::cerr << "[handoff] attempts=" << attempts << " refused=" << g_refused.load()
                  << " engine_accepts_with_inflight=" << engine_penetrations
                  << " legal_idle_applies=" << g_legal.load()
                  << " noops=" << g_noop.load()
                  << " idle_gauge_window_attempts=" << idle_window_refusals
                  << " completed_requests=" << g_completed.load()
                  << " damaged=" << g_damaged.load() << " kv_edge=" << edge << " -> " << after
                  << '\n';
        if (attempts < kAttempts) {
            std::cerr << "[FAIL] only " << attempts << " of " << kAttempts
                      << " attempts were made\n";
            return 1;
        }
        if (idle_window_refusals < kIdleAttempts) {
            std::cerr << "[FAIL] only " << idle_window_refusals << " of " << kIdleAttempts
                      << " attempts landed in an all-zero-gauge window: the pre-queue window was "
                         "never exercised\n";
            return 1;
        }
        if (engine_penetrations != 0) {
            std::cerr << "[FAIL] " << engine_penetrations
                      << " KV reallocation(s) were accepted with a live request (engine counter)\n";
            return 1;
        }
        if (g_damaged.load() != 0 || g_completed.load() == 0) {
            std::cerr << "[FAIL] the workload did not survive the refused resizes\n";
            return 1;
        }
        if (after != edge) {
            // A legal idle-window apply during the quiet gaps is allowed by the contract, but the
            // edge must sit at one of the two bounce values and every move must be counted.
            if (g_legal.load() == 0 ||
                !(after == edge || after + 8 == edge || after == edge + 8)) {
                std::cerr << "[FAIL] the KV edge moved (" << edge << " -> " << after
                          << ") without a counted legal apply\n";
                return 1;
            }
            std::cerr << "[record] the edge moved only through " << g_legal.load()
                      << " counted idle-window applies (edge " << edge << " -> " << after << ")\n";
        }
        std::cout << "ok: " << attempts << " shrink attempts: " << g_refused.load()
                  << " refused, " << g_legal.load() << " legal idle-window applies, 0 penetrated, "
                  << g_completed.load() << " requests intact\n";
        return 0;
    } catch (const std::exception& error) {
        const std::string message = error.what();
        if (message.find("reservation") != std::string::npos ||
            message.find("runtime capacity") != std::string::npos) {
            std::cerr << "skip: " << message << '\n';
            return 77;
        }
        std::cerr << "handoff acceptance failed: " << message << '\n';
        return 1;
    }
}
