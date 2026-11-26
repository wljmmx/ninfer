#pragma once

#include "core/weight.h"
#include "ninfer/ops/linear.h"

namespace ninfer::ops::detail::ggml {

// One quantised Linear for a `ggml_block_v1` weight: y[N,T] = w[N,K] x[K,T], the weight read in its
// stored encoding. Shapes are validated by the caller (ops::linear); this only refuses a geometry the
// block layout cannot express.
void launch_linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

// The same product accumulated into an existing residual, y[N,T] += w[N,K] x[K,T]. The kernel owns no
// scratch, so this is the whole implementation of `linear_add` for these encodings.
void launch_linear_add(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream);

} // namespace ninfer::ops::detail::ggml
