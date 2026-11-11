#pragma once

#include "core/device.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace ninfer::models::qwen4_exp {
class PleTextStage;
} // namespace ninfer::models::qwen4_exp

namespace ninfer::models::qwen3_5::execution {
class Parameters;
} // namespace ninfer::models::qwen3_5::execution

namespace ninfer::models::qwen3_5::detail {

// Builds the PLE stage a model with `ple_layer_ids` needs: the n-gram table window, the Host
// preparation service and one layer runtime per block that carries PLE. Null when the config
// declares no PLE. It is a free function because the owner is the Program, not the pass: the
// nine-column history and each slot's two-token hash tail have to outlive a single TextContext.
[[nodiscard]] std::unique_ptr<qwen4_exp::PleTextStage>
make_ple_stage(const execution::Parameters& parameters, const DeviceContext& context,
               std::uint32_t prefill_chunk);

// The Program's PLE state, borrowed by every TextContext the Program builds.
//
// Why it lives here: a window's accepted columns advance a per-sequence nine-column convolution
// history, and the n-gram hash continues from the two tokens a slot ended with. Both are sequence
// state, so rebuilding them with a per-round pass would silently restart the convolution at every
// round - which is what a decode step used to do. One owner for the Program's lifetime makes the
// history continuous across rounds, across a speculative round's verify/accept split, and across
// the several TextContexts one round builds.
//
// The pool is indexed by the sequence's lane, not by its physical StateImage slot: a lane is the
// identity a resident sequence holds for its whole life, while a state slot moves when a checkpoint
// forks or when a demoted image is reloaded. `reset_lane` is therefore the "a new sequence starts
// from an all-+0 history and EOS hash tail" rule, applied when a lane is handed to another request.
class PleProgramState {
public:
    explicit PleProgramState(std::unique_ptr<qwen4_exp::PleTextStage> stage) noexcept;
    ~PleProgramState();
    PleProgramState(const PleProgramState&)            = delete;
    PleProgramState& operator=(const PleProgramState&) = delete;

    [[nodiscard]] bool present() const noexcept { return stage_ != nullptr; }

    [[nodiscard]] qwen4_exp::PleTextStage& stage() {
        return *stage_;
    }

    [[nodiscard]] const qwen4_exp::PleTextStage& stage() const { return *stage_; }

    // The pass-level half of a window. `deferred` marks a verify rectangle, whose acceptance the
    // verifier publishes after the pass: its window stays open until commit_verify() or the round's
    // failure path closes it.
    void window_opened(std::int32_t tokens, bool deferred) noexcept;
    void window_closed() noexcept;

    [[nodiscard]] bool window_open() const noexcept { return open_; }

    [[nodiscard]] bool window_deferred() const noexcept { return deferred_; }

    [[nodiscard]] std::int32_t window_tokens() const noexcept { return tokens_; }

    // Commits every PLE block by the verifier's I32 [batch] accepted-column counts and closes the
    // window. Returns false when no deferred window was waiting.
    bool commit_verify(const Tensor& accepted, std::size_t layers, CudaExecutionView execution);

    // Ends a window whose acceptance is its live columns (a prefill chunk, the merged prefill
    // rectangle, an ordinary decode step): every block advances by those columns, the hash tail
    // moves with them, and the window closes.
    void close_window(CudaExecutionView execution);

    // Closes a window a failed pass left open without committing it: the round produced nothing,
    // so none of its columns may enter the history - and the hash tail does not move either.
    void abort_window(CudaExecutionView execution) noexcept;

    // Forgets one lane's nine-column history and its hash tail.
    void reset_lane(std::int32_t lane, CudaExecutionView execution);

private:
    std::unique_ptr<qwen4_exp::PleTextStage> stage_;
    bool open_            = false;
    bool deferred_        = false;
    std::int32_t tokens_  = 0;
};

} // namespace ninfer::models::qwen3_5::detail
