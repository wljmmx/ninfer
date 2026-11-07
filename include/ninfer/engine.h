#pragma once

#include "ninfer/types.h"

#include <chrono>
#include <memory>
#include <string_view>
#include <vector>

namespace ninfer {

class PreparedPrompt {
public:
    PreparedPrompt() noexcept;
    ~PreparedPrompt();

    PreparedPrompt(PreparedPrompt&&) noexcept;
    PreparedPrompt& operator=(PreparedPrompt&&) noexcept;

    PreparedPrompt(const PreparedPrompt&)            = delete;
    PreparedPrompt& operator=(const PreparedPrompt&) = delete;

    [[nodiscard]] const PromptSummary& summary() const noexcept;
    [[nodiscard]] const PromptPreparationStats& preparation_stats() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;

private:
    class Impl;
    explicit PreparedPrompt(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

class GenerationHandle {
public:
    GenerationHandle() noexcept;
    ~GenerationHandle();

    GenerationHandle(GenerationHandle&&) noexcept;
    GenerationHandle& operator=(GenerationHandle&&) noexcept;

    GenerationHandle(const GenerationHandle&)            = delete;
    GenerationHandle& operator=(const GenerationHandle&) = delete;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept;

    // A terminal result or request error is visible after logical scheduling membership is
    // released and the corresponding Engine snapshot is published.
    GenerationResult wait(OutputSink* sink = nullptr, const CancellationView& cancellation = {});

private:
    class Impl;
    explicit GenerationHandle(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

// The sizes an external cache has to be told before it can store anything. They come from the live
// KV page layout and the state pool, not from model constants, because every model differs in layer
// count, head width, plane count and state dtype.
struct CacheGeometry {
    std::size_t page_bytes        = 0;
    std::int64_t page_tokens      = 0;
    std::size_t state_image_bytes = 0;
    // The KV format, as the wire tag of KvCacheStorage, and how many vectors a page carries. Sizes
    // alone do not identify a format: a 4-bit layout at one head width and an 8-bit one at half of
    // it can both give 136 B per vector, and then a key made of sizes would collide.
    std::uint32_t kv_dtype_tag    = 0;
    std::uint32_t kv_planes       = 0;
};

class Engine {
public:
    explicit Engine(EngineOptions options);
    ~Engine();

    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;

    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;

    [[nodiscard]] PreparedPrompt prepare(PromptInput input,
                                         const PreparationControl& control = {}) const;

    // Raw token input is retained for repeatable correctness and performance measurement.
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity = true) const;

    // Artifact-tokenizer raw-text encoding. No chat template or implicit special token is added.
    [[nodiscard]] std::vector<TokenId> tokenize_text(std::string_view text) const;

    // Returns log p(tokens[i] | tokens[0..i)) for i in [first_target,tokens.size()).
    [[nodiscard]] std::vector<float> score_tokens(std::vector<TokenId> tokens,
                                                  std::uint32_t first_target);

    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
                                             const PreparationControl& control = {}) const;
    [[nodiscard]] ModelSamplingDefaults sampling_defaults() const;

    // Establishes queue membership synchronously with a fixed output consumer mode. Destroying an
    // unconsumed handle cancels its request; wait() owns result consumption and may run
    // independently from GPU execution. Streaming mode requires a non-null sink in wait() and
    // publishes one exact GenerationStart before output deltas; Aggregate mode requires a null
    // sink. Observation options request protocol-neutral publication facts without changing the
    // execution request.
    [[nodiscard]] GenerationHandle
    submit(PreparedPrompt prompt, RequestOptions options,
           OutputConsumerMode consumer_mode                       = OutputConsumerMode::Aggregate,
           GenerationObservationOptions observation               = {},
           std::chrono::steady_clock::time_point pending_deadline = {});

    GenerationResult generate(PreparedPrompt prompt, RequestOptions options,
                              OutputSink* sink                     = nullptr,
                              const CancellationView& cancellation = {});

    [[nodiscard]] const EngineOptions& options() const;
    [[nodiscard]] LoadSummary load_summary() const;
    [[nodiscard]] CacheGeometry cache_geometry() const;
    [[nodiscard]] MemorySummary memory_summary() const;
    [[nodiscard]] RuntimeStats runtime_stats() const;
    [[nodiscard]] MediaCacheSummary media_cache_summary() const;
    [[nodiscard]] bool is_available() const;

    // 13.4: runtime pool resize. Only legal while the engine is idle: an expert window a running
    // layer is addressing cannot be unmapped, and a captured graph froze the window offsets. The
    // call synchronizes the device after checking both, and returns the capacity actually reached
    // (whole windows only). Throws when the engine is not idle, when a graph pinned the pool, or
    // when the target cannot be met; it never silently shrinks below the current windows.
    // v8 13.5 (step 5): ACCEPTED under load on the RTX 4090 / qwen3.6-35b e2 artifact (four-way
    // batches, this commit's resize-under-load test): the idle double read refuses mid-batch, the
    // grow->shrink round trip at a request boundary leaves the pool reproducing the
    // never-resized run's tokens verbatim, and the frozen-graph pin refuses at construction.
    // Controller L1 wiring can proceed on these three; the KV entry's boundary hand-off race and
    // the post-KV-shrink page-placement divergence are the two open items it must not assume away.
    std::uint64_t resize_expert_pool(std::uint64_t target_bytes);

    // 13.4: the KV half of runtime resize. Same idle-window contract as the expert pool, and the
    // same refusal style: a retained checkpoint that owns pages above the new edge is an owner, not
    // slack, so the call throws instead of invalidating it. The physical driver reservation is not
    // returned (the VMM block stays mapped); the capacity, the free list and the address bound move.
    // Returns the capacity in pages (64 tokens each) actually in effect.
    std::uint32_t resize_kv_pages(std::uint32_t target_pages);

    void reset_memory_peaks() noexcept;

private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace ninfer
