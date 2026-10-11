#pragma once

#include "core/device.h"
#include "core/tensor.h"

#include <array>

namespace ninfer::ops {

// The scaled frequencies are evaluated in FP32 at the angle, so long-context accuracy degrades
// gradually rather than at a capacity boundary.
inline constexpr int kRopeMaximumPairs = 128;

/**
 * Optional frequency scaling. The default (yarn=false) reproduces the unscaled
 * theta^(-2*i/rotary_dim) frequencies exactly, so existing callers are unaffected.
 *
 * YaRN (Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models")
 * replaces the per-pair inverse frequency with a ramp blend of the extrapolated frequency and
 * the 1/factor interpolated one, and rescales the rotary sine/cosine by an attention factor.
 * The ramp window is derived from the original trained position capacity:
 *
 *   correction(n) = rotary_dim * ln(original_max_position_embeddings / (n * 2 * pi))
 *                            / (2 * ln(theta))
 *   low  = floor(correction(beta_fast)),  high = ceil(correction(beta_slow))
 *   ramp(i) = clamp((i - low) / (high - low), 0, 1)
 *   inv_frequency[i] = (extrapolated * (1 - ramp(i))) + (extrapolated / factor) * ramp(i)
 *
 * Pairs at or below low keep the trained extrapolation; pairs at or above high are fully
 * interpolated; the window between them is linear. inv_frequency is only read when yarn is set
 * and must hold rotary_dim/2 entries. attention_factor multiplies both sine and cosine.
 */
struct RopeScaling {
    // A plain array rather than std::array: this struct is passed by value into device code, and
    // std::array::operator[] is not a device function without --expt-relaxed-constexpr.
    bool yarn              = false;
    float attention_factor = 1.0f;
    float inv_frequency[kRopeMaximumPairs] = {};
};

/**
 * Applies split-half NeoX RoPE in place. For pair i in [0,rotary_dim/2), angle phi(i,t), and
 * each head:
 *
 *   ideal[i]              = x[i] * cos(phi) - x[i+R/2] * sin(phi)
 *   ideal[i+rotary_dim/2] = x[i+R/2] * cos(phi) + x[i] * sin(phi).
 *
 * Dimensions [rotary_dim,head_dim) are unchanged. Supported modes are:
 *
 * - Text 1-D: positions I32 [T], either head_dim=256 with even 0<rotary_dim<=256, or the
 *   DFlash full-head domain head_dim=rotary_dim=128; phi=positions[t]*theta^(-2*i/rotary_dim).
 * - Text MRoPE: positions I32 [T,3], head_dim=256, rotary_dim=64; pair i uses axis i%3 with
 *   the same frequency as Text 1-D.
 * - Vision 2-D: positions I32 [T,2], head_dim=rotary_dim=72; pairs 0..17 use axis 0 and pairs
 *   18..35 use axis 1, each with local frequency theta^(-2*(i%18)/36).
 *
 * positions is contiguous and theta is positive and finite. Q/K tensors are BF16
 * [head_dim,heads,T] with positive head counts, contiguous head features and heads, and an optional
 * padded token stride. The registered optimized domains are D256/R64 Text Q/K head geometries
 * 24/4 and 16/2, D128/R128 1-D Text geometry 32/8, plus Vision geometry 16/16. q and k must not
 * overlap one another or positions. The Op mutates only dimensions [0,rotary_dim) of the supplied
 * Q/K tensor storage. The oracle evaluates the rotated dimensions naively in FP64 from the
 * represented inputs. The updated BF16 values are promoted and compared directly with that result;
 * output storage rounding belongs to the Op's numerical criterion, not the oracle. Unrotated
 * dimensions remain bit-exact. Private kernel arithmetic is implementation-defined. The Op uses no
 * workspace or persistent state. `execution` supplies the stream and the selected device's
 * positive physical SM count; the count selects launch geometry, not the transformation.
 * scaling replaces the per-pair inverse frequency and the sine/cosine amplitude; the default is
 * the identity scaling.
 */
void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
          DeviceExecutionView execution, const RopeScaling& scaling = RopeScaling{});

// Single-tensor form with the same formula and storage contract. The head count comes directly
// from x; Q versus K role does not change the transformation.
void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
          DeviceExecutionView execution, const RopeScaling& scaling = RopeScaling{});

} // namespace ninfer::ops
