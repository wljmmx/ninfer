#pragma once

#include "core/device.h"
#include "models/qwen3_5/execution/parameters.h"

namespace ninfer::models::qwen3_5::execution {

[[nodiscard]] std::size_t ffn_workspace_bytes(const FfnParameters& parameters, std::int32_t first,
                                              std::int32_t last, bool mtp = false);
void ffn(const Tensor& hidden, const FfnParameters& parameters, Tensor& residual,
         const ops::SparseMoeHints& hints, WorkspaceArena& workspace, DeviceExecutionView execution,
         bool mtp = false,
         // port (sm_89): Q68b probe. The residual stream as it stood at the end of the previous
         // layer, or null. This layer's router applied to it is a *prediction* of this layer's
         // routes: z_(L-1) is what x_L would be if this block's attention added nothing. Diagnostic
         // only - the routes actually used are still computed from `hidden`.
         const Tensor* predictor_input = nullptr,
         // port (sm_89): Q68b, the form the plan describes. The NEXT layer's routed block: at the end
         // of this layer its router is scored against the residual just produced, and the answer is
         // handed to that layer's stream. Null on the last layer, and a no-op unless
         // NINFER_PREFETCH_PREDICT is set.
         const RoutedMoeParameters* predict_next = nullptr);

// Probe-only plumbing (NINFER_LAYER_PROBE=2, decode only): the layer loop records the residual
// stream after each layer and hands it back as the next layer's predictor input. Both are no-ops
// unless the switch is on, so a normal run allocates and copies nothing.
[[nodiscard]] bool router_probe_enabled();
void router_probe_save(const Tensor& residual);
[[nodiscard]] const Tensor* router_probe_input(Device device);

// port (sm_89), resize safety: destroys every captured per-bank layer graph. A captured exec
// bakes the offset-table addresses its kernels read; ExpertCache::resize moves the pool layout
// and hands freed segments back to the driver, so a replay against a pre-resize capture would
// read a table the pool no longer serves (expert_cache.cpp's own note calls the stale-table
// read an illegal address the first grouped kernel trips over). The engine's resize path calls
// this after a successful ExpertCache::resize; the next layer captures fresh, as it did the
// first time. Safe when no graph was captured at all (it is then a no-op).
void discard_layer_graph_captures() noexcept;

} // namespace ninfer::models::qwen3_5::execution
