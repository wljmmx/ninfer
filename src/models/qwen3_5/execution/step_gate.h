#pragma once

#include <cstdint>

namespace ninfer::models::qwen3_5::execution {

// port (sm_89): O1's step verdict, one mapped word for the whole model. A replay asks "did ANY layer
// of this step drop a route", not which layer, so the answer has to be readable where the step ends -
// the sampling synchronise in the decode batch, which already waits for every kernel. The kernels add
// to it exactly as they add to the per-bank counter; the host resets it at the start of a step and
// reads it once at the end. Null (and nothing counted) unless NINFER_O1_STEP_GATE is set.
//
// Its own header because the two readers sit far apart: ffn.cpp hands the pointer to the kernel, the
// decode batch reads the word, and neither should have to pull in the expert cache to do it.
std::uint32_t* step_miss_device();
// Only valid once the step's work has synchronised.
std::uint32_t step_miss_read();
void step_miss_reset();

} // namespace ninfer::models::qwen3_5::execution
