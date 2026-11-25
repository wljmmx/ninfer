#pragma once

#include <atomic>

namespace ninfer::models::qwen3_5::execution {

// Whether the layer stack may be recorded into a CUDA graph.
//
// A capture records the work; it cannot execute a host fetch, and the expert split path's fetch
// waits on cudaStreamSynchronize (ffn.cpp), which a capture forbids. So the stack is capturable
// only when every expert of every bank is resident: then no layer can take the split path.
//
// The owner of this fact is the engine, which knows the pool (ExpertCache::all_experts_resident());
// the reader is text.cpp, which has no handle on the pool. Same shape as the policy controller:
// one process-wide answer, set once after the pool exists, read on the prefill path.
inline std::atomic<bool>& layer_stack_capturable_flag() noexcept {
    static std::atomic<bool> flag{false};
    return flag;
}

inline void set_layer_stack_capturable(bool on) noexcept { layer_stack_capturable_flag().store(on); }
inline bool layer_stack_capturable() noexcept { return layer_stack_capturable_flag().load(); }

} // namespace ninfer::models::qwen3_5::execution
