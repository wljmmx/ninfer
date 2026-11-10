#include "ninfer/ops/softmax_attention.h"

#include "core/layout.h"
#include "ops/softmax_attention/dense/packed/launch.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeads    = 16;
constexpr float kExpectedScale64 = 0.125f;
constexpr float kExpectedScale   = 0.11785113019775792073f;

void require_geometry(AttentionHeadGeometry geometry, const char* op) {
    if (!valid_attention_head_geometry(geometry) ||
        (geometry.head_dim != 64 && geometry.head_dim != 72) || geometry.query_heads != kHeads ||
        geometry.kv_heads != kHeads) {
        throw std::invalid_argument(std::string(op) + ": unsupported head geometry");
    }
}

float expected_scale(AttentionHeadGeometry geometry) {
    // T088 (upstream 6eb6056b): the noncausal profiles are D64 and D72, each with its own
    // 1/sqrt(D) scale. The text heads stay on 72; the vision heads run at 64.
    return geometry.head_dim == 64 ? kExpectedScale64 : kExpectedScale;
}

// The capacity query carries no scale, so it validates only the geometry it prices. Assuming a
// scale there rejected the D64 route before it could be sized.
void require_profile(AttentionHeadGeometry geometry, float scale, const char* op) {
    require_geometry(geometry, op);
    const float expected = expected_scale(geometry);
    if (!std::isfinite(scale) || std::abs(scale - expected) > 1.0e-7f) {
        throw std::invalid_argument(std::string(op) + ": scale must be 1/sqrt(head_dim)");
    }
}

std::int32_t scratch_tiles(std::int32_t tokens, std::int32_t segments) {
    if (segments == 1) { return 0; }
    constexpr std::int64_t tile_rows = detail::kPackedDescriptorRows;
    const std::int64_t tiles =
        (static_cast<std::int64_t>(tokens) + tile_rows - 1) / tile_rows + segments - 1LL;
    if (tiles > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error("packed_softmax_attention: descriptor tile count exceeds int32");
    }
    return static_cast<std::int32_t>(tiles);
}

template <class Allocator>
Tensor allocate_workspace(Allocator& allocator, std::int32_t tokens, std::int32_t segments) {
    const std::int32_t tiles = scratch_tiles(tokens, segments);
    return tiles == 0 ? Tensor{} : allocator.alloc(DType::I32, {4, tiles});
}

void require_qkv(const Tensor& tensor, std::int32_t tokens, std::int32_t head_dim, const char* op,
                 const char* name) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != head_dim || tensor.ne[1] != kHeads ||
        tensor.ne[2] != tokens || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid " + name + " shape");
    }
    constexpr std::int64_t elem = 2;
    if (tensor.nb[0] != elem || tensor.nb[1] != elem * head_dim ||
        tensor.nb[2] < elem * head_dim * kHeads || (tensor.nb[2] % elem) != 0) {
        throw std::invalid_argument(std::string(op) + ": invalid " + name + " strides");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name + " data must be non-null");
    }
}

std::int32_t validate_qkv(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& out,
                          AttentionHeadGeometry geometry, float scale, const char* op) {
    require_profile(geometry, scale, op);
    const std::int32_t tokens = q.ne[2];
    if (tokens <= 0) { throw std::invalid_argument(std::string(op) + ": T must be positive"); }
    require_qkv(q, tokens, geometry.head_dim, op, "q");
    require_qkv(k, tokens, geometry.head_dim, op, "k");
    require_qkv(v, tokens, geometry.head_dim, op, "v");
    require_qkv(out, tokens, geometry.head_dim, op, "out");
    if (!out.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": out must be contiguous");
    }
    return tokens;
}

} // namespace

std::size_t packed_softmax_attention_workspace_capacity_bytes(AttentionHeadGeometry geometry,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens,
                                                              std::int32_t min_segments,
                                                              std::int32_t max_segments) {
    require_geometry(geometry, "packed_softmax_attention workspace");
    if (min_tokens <= 0 || max_tokens < min_tokens || min_segments <= 0 ||
        max_segments < min_segments || min_segments > max_tokens) {
        throw std::invalid_argument(
            "packed_softmax_attention workspace: invalid execution envelope");
    }
    const std::int32_t segments = std::min(max_segments, max_tokens);
    WorkspaceLayoutBuilder layout;
    (void)allocate_workspace(layout, max_tokens, segments);
    return layout.peak_bytes(1);
}

void softmax_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                       AttentionHeadGeometry geometry, float scale, WorkspaceArena& workspace,
                       Tensor& out, cudaStream_t stream) {
    const std::int32_t tokens = validate_qkv(q, k, v, out, geometry, scale, "softmax_attention");
    auto scope                = workspace.scope();
    detail::packed_attention_uniform_launch(q, k, v, tokens, out, stream);
}

void packed_softmax_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              AttentionHeadGeometry geometry, float scale, const Tensor& cu_seqlens,
                              WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    const std::int32_t tokens =
        validate_qkv(q, k, v, out, geometry, scale, "packed_softmax_attention");
    const std::int32_t segments = cu_seqlens.ne[0] - 1;
    if (cu_seqlens.dtype != DType::I32 || segments <= 0 || segments > tokens ||
        cu_seqlens.ne[1] != 1 || cu_seqlens.ne[2] != 1 || cu_seqlens.ne[3] != 1 ||
        !cu_seqlens.is_contiguous() || cu_seqlens.data == nullptr) {
        throw std::invalid_argument(
            "packed_softmax_attention: cu_seqlens must be contiguous I32 [S+1]");
    }
    auto scratch_scope = workspace.scope();
    Tensor tiles       = allocate_workspace(workspace, tokens, segments);
    Tensor* tiles_ptr  = tiles.data == nullptr ? nullptr : &tiles;
    detail::packed_attention_launch(q, k, v, cu_seqlens, tiles_ptr, out, stream);
}

void packed_softmax_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              AttentionHeadGeometry geometry, float scale,
                              std::int32_t segment_length, Tensor& out, cudaStream_t stream) {
    const std::int32_t tokens =
        validate_qkv(q, k, v, out, geometry, scale, "packed_softmax_attention");
    if (segment_length <= 0 || tokens % segment_length != 0) {
        throw std::invalid_argument("packed_softmax_attention: invalid uniform segment length");
    }
    detail::packed_attention_uniform_launch(q, k, v, segment_length, out, stream);
}

} // namespace ninfer::ops
