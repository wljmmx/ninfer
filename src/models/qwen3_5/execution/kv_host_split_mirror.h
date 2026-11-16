#pragma once

// Split ownership, knives 2-3: the engine-side producer of the CPU partial's host view.
//
// WHY THIS EXISTS. `rk_kv_cpu_split_gate` (ops/.../int8/host_partial.h) has been able to run one
// split's partial on the CPU since P3 Phase C, but its first precondition - a view installed by
// `rk_host_partial_install` - had no caller anywhere in the tree, so every call fell back at
// kNoView. This header is that caller: it mirrors one layer's KV pages, the call's Q/positions and
// the row's block table into host memory, and installs the resulting view for the attention call.
//
// OWNERSHIP OF THE MIRRORED PAGES IS TEST-SHAPED, NOT POLICY. The engine's Host KV tier holds only
// demoted extents that no attention path reads back; there are no "host-resident split pages" yet.
// So this producer COPIES the pages it needs (D2H), which is exactly the cost the decision
// `decide_cpu_kv_split` prices and would avoid. Nothing here decides anything: the knob picks the
// split, the mirror feeds it, and the panel/counters record what happened. When a real host tier
// starts owning split pages, it replaces the copy with a view over bytes it already holds - the
// gate, the merge contract and the mirror's addressing stay as they are.
//
// THE APPEND LAG, STATED RATHER THAN ASSUMED. The production decode call fuses the current token's
// append into the same kernel that scores it, so a mirror taken at the call boundary cannot contain
// the newest token's K/V codes. The producer therefore reports how many leading positions its pages
// are authoritative for (`page_filled_keys`), and the gate refuses a split whose range reaches past
// it. A producer that mirrors *after* the append (a real tier) leaves it 0.
//
// COST. One mirror call copies the layer's referenced pages: 4 planes x referenced pages x page
// payload, plus the Q tile and the row's table. For rk4v4 D256 x 4 KV heads with a 4096-token
// context that is 64 x 69,632 B = 4.25 MiB per layer per call - the leg is a validation seam, not a
// win, and its numbers are reported as such.

#include "core/paged_kv_cache.h"
#include "core/paged_kv_storage.h"
#include "core/tensor.h"
#include "ops/softmax_attention/cpu/host_pages.h"
#include "ops/softmax_attention/dense/causal_cache/int8/host_partial.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

namespace cpu_partial = ninfer::ops::detail::softmax_cpu;

// True when the CPU-partial producer should run at all: the gate's own knob, or the mirror's
// one-shot consistency check. Read once per process; unset means the producer never runs, so the
// default path pays nothing but this call.
[[nodiscard]] inline bool kv_host_split_mirror_armed() {
    static const bool armed = [] {
        const char* partition = std::getenv("NINFER_KV_CPU_PARTITION");
        const char* check     = std::getenv("NINFER_KV_CPU_MIRROR_CHECK");
        const bool wants_partition =
            partition != nullptr && *partition != '\0' && std::strtol(partition, nullptr, 10) >= 0;
        const bool wants_check = check != nullptr && *check != '\0' && std::strtol(check, nullptr, 10) != 0;
        return wants_partition || wants_check;
    }();
    return armed;
}

[[nodiscard]] inline bool kv_host_split_mirror_check_enabled() {
    static const bool enabled = [] {
        const char* check = std::getenv("NINFER_KV_CPU_MIRROR_CHECK");
        return check != nullptr && *check != '\0' && std::strtol(check, nullptr, 10) != 0;
    }();
    return enabled;
}

// One layer's host mirror plus the view the gate reads. One instance lives in the execution
// context and is reused layer to layer; the planes and the table are rebuilt per call because the
// row's mapping and the newest page's bytes move with every token.
class KvHostSplitMirror {
public:
    KvHostSplitMirror() = default;
    KvHostSplitMirror(const KvHostSplitMirror&)            = delete;
    KvHostSplitMirror& operator=(const KvHostSplitMirror&) = delete;

    // Mirror one layer and (when `install`) publish the view on this thread. Returns false when the
    // layer's storage is not one the CPU partial covers; the caller then continues on the GPU.
    bool prepare(const Tensor& q, const Tensor& positions, const Tensor& kv_table_rows,
                 const PagedKVBatchLayerView& cache, int layer, cudaStream_t stream) {
        mirroring_ = false;
        if (q.dtype != DType::BF16 || positions.dtype != DType::I32 ||
            kv_table_rows.dtype != DType::I32 || cache.k_pages.data == nullptr) {
            return fail("the call's Q/positions/rows or the cache planes are not mirrored dtypes");
        }
        if (q.ne[0] != cpu_partial::kHeadDim || q.ne[1] <= 0 || q.ne[2] <= 0 || q.ne[3] != 1) {
            return fail("the call is not a single-row D256 attention tile");
        }
        if (q.ne[2] > ops::detail::rk_host_partial_detail::kStageWidth) {
            return fail("the token tile is wider than the CPU leg's pinned staging");
        }
        KVPageGeometry geometry;
        geometry.page_tokens = static_cast<std::uint32_t>(cache.k_pages.ne[1]);
        const Tensor* planes[4] = {&cache.k_pages, &cache.v_pages, &cache.k_scale_pages,
                                   &cache.v_scale_pages};
        for (const Tensor* plane : planes) {
            if (plane->data == nullptr || plane->ne[2] != cache.num_kv_heads) {
                return fail("a cache plane is not a PageMajor plane of this layer");
            }
            geometry.planes.push_back({plane->dtype, plane->ne[0], plane->ne[2],
                                       static_cast<std::size_t>(plane->nb[3])});
        }
        HostKVPageLayout layout;
        try {
            layout = plan_host_kv_page_layout(geometry);
        } catch (...) {
            return fail("the layer's plane geometry has no Host page layout");
        }
        cpu_partial::RkHostKvPlanes page_planes;
        if (!cpu_partial::plan_rk_host_kv_planes(layout, 0, page_planes)) {
            return fail("the layer's KV storage is not one the CPU partial decodes");
        }
        // The page family names the view's dtype through the ONE mapping in host_partial.h;
        // every family host_pages.h identifies has a value on that axis, and the E8 carriers
        // decode to the same int8/FP16-G64 form (cpu/e8_codec.h), so no carrier is refused here.
        // A pair no family names still fails rather than being guessed.
        ops::detail::RkHostKvDType dtype{};
        if (!ops::detail::rk_host_kv_dtype_of(page_planes.k_codec, page_planes.v_codec, dtype)) {
            return fail("the layer's (K, V) codec pair names no rk host dtype");
        }

        const auto logical_pages = static_cast<std::int32_t>(cache.block_tables.ne[0]);
        if (logical_pages <= 0) { return fail("the row has no logical pages"); }
        const std::int32_t physical_pages = cache.k_pages.ne[3];
        if (physical_pages <= 0) { return fail("the layer's pool has no physical pages"); }

        // --- the row's block table, host side -------------------------------------------------
        std::int32_t row = 0;
        if (cudaMemcpyAsync(&row, kv_table_rows.data, sizeof row, cudaMemcpyDeviceToHost, stream) !=
            cudaSuccess) {
            (void)cudaGetLastError();
            return fail("the row index could not be read back");
        }
        table_.assign(static_cast<std::size_t>(logical_pages), -1);
        const auto row_offset =
            static_cast<std::size_t>(row) * static_cast<std::size_t>(logical_pages) * sizeof(std::int32_t);
        if (row < 0 || row >= cache.block_tables.ne[1] ||
            cudaMemcpyAsync(table_.data(),
                            static_cast<const std::byte*>(cache.block_tables.data) + row_offset,
                            table_.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                            stream) != cudaSuccess) {
            (void)cudaGetLastError();
            return fail("the row's block table could not be read back");
        }
        // The page window is the whole physical pool with ids unchanged: the row's table is then
        // handed to the operator verbatim, and a page no entry names is simply masked out.
        const auto window = static_cast<std::uint32_t>(physical_pages);
        const auto mirror_layout =
            cpu_partial::rk_host_mirror_layout(layout, page_planes, window);

        // --- the layer's pages, D2H into the record array -------------------------------------
        if (!reserve_records(layout.page_stride * window)) {
            return fail("the record buffer is already sized for another geometry");
        }
        for (int channel = 0; channel < 4; ++channel) {
            const int index = channel;  // this layout holds one layer, so the plane index is the channel
            const Tensor& plane = *planes[static_cast<std::size_t>(channel)];
            const auto bytes    = layout.planes[static_cast<std::size_t>(index)].page_payload_bytes;
            for (std::int32_t page = 0; page < physical_pages; ++page) {
                if (!page_referenced(page)) { continue; }
                if (cudaMemcpyAsync(record_ + static_cast<std::size_t>(page) * layout.page_stride +
                                        layout.planes[static_cast<std::size_t>(index)].offset,
                                    static_cast<const std::byte*>(plane.data) +
                                        static_cast<std::size_t>(page) * plane.nb[3],
                                    bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
                    (void)cudaGetLastError();
                    return fail("a page's plane payload could not be read back");
                }
            }
        }

        // --- Q and positions ------------------------------------------------------------------
        const std::size_t q_terms =
            static_cast<std::size_t>(q.ne[2]) * q.ne[1] * cpu_partial::kHeadDim;  // tokens x heads x D
        q_.resize(q_terms);
        positions_.assign(static_cast<std::size_t>(q.ne[2]), 0);
        if (cudaMemcpyAsync(q_.data(), q.data, q_terms * sizeof(std::uint16_t),
                            cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
            cudaMemcpyAsync(positions_.data(), positions.data,
                            positions_.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                            stream) != cudaSuccess) {
            (void)cudaGetLastError();
            return fail("the Q tile or positions could not be read back");
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            (void)cudaGetLastError();
            return fail("the mirror's D2H batch did not complete");
        }

        // --- repack into the operator's plane blocks -------------------------------------------
        mirror_.resize(mirror_layout.total_bytes);
        cpu_partial::rk_host_mirror_pages(mirror_.data(), mirror_layout, record_,
                                                       layout, page_planes, 0, window);
        std::vector<std::int32_t> mirrored_table(table_.size(), -1);
        try {
            cpu_partial::rk_host_mirror_block_table(
                mirrored_table.data(), static_cast<int>(mirrored_table.size()), table_.data(),
                logical_pages, 0, window);
        } catch (...) {
            return fail("the row's table names a page outside the mirrored pool");
        }
        table_.swap(mirrored_table);

        view_.dtype       = dtype;
        view_.query_heads = q.ne[1];
        view_.pages = cpu_partial::rk_host_partial_pages(mirror_.data(), mirror_layout,
                                                                     table_.data(), cache.num_kv_heads);
        view_.q_bf16          = q_.data();
        view_.positions       = positions_.data();
        // The pages are authoritative for every position before this call's first token: the fused
        // append writes the tokens being scored, and those bytes do not exist on the host yet.
        view_.page_filled_keys = positions_.front();
        layer_                = layer;
        mirroring_            = true;
        if (kv_host_split_mirror_check_enabled()) { check_against_device(q, positions, cache); }
        return true;
    }

    [[nodiscard]] bool mirroring() const noexcept { return mirroring_; }
    [[nodiscard]] const ops::detail::RkHostPartialView& view() const noexcept { return view_; }

    // Installs the view for the call; passing nullptr clears it.
    static void install(const ops::detail::RkHostPartialView* view) {
        ops::detail::rk_host_partial_install(view);
    }

private:
    // One line per process on the first refusal, in the gate's own style: a producer that silently
    // declines would make "the knob is set but nothing engaged" look like a measured zero.
    [[nodiscard]] static bool fail(const char* why) {
        if (kv_host_split_mirror_check_enabled()) { return fail_verbose(why); }
        static bool reported = false;
        if (!reported) {
            reported = true;
            std::fprintf(stderr, "[kv-cpu] mirror: %s -> the split stays on the GPU\n", why);
            std::fflush(stderr);
        }
        return false;
    }

    // Diagnostic: the same refusal, printed every time up to a small cap. Used by the check knob.
    [[nodiscard]] static bool fail_verbose(const char* why) {
        static int printed = 0;
        if (printed < 40) {
            ++printed;
            std::fprintf(stderr, "[kv-cpu] mirror probe: %s\n", why);
            std::fflush(stderr);
        }
        return false;
    }

    [[nodiscard]] bool page_referenced(std::int32_t page) const {
        for (const std::int32_t entry : table_) {
            if (entry == page) { return true; }
        }
        return false;
    }

    bool reserve_records(std::size_t bytes) {
        if (record_bytes_ >= bytes) { return true; }
        // Pinned, deliberately never freed: the same reason host_partial.h's staging is not freed -
        // cudaFreeHost at teardown races the CUDA runtime on engine threads - and it is one layer's
        // page records.
        if (record_ != nullptr) { return false; }  // a growing geometry after the first mirror: refuse
        void* fresh = nullptr;
        if (cudaHostAlloc(&fresh, bytes, cudaHostAllocDefault) != cudaSuccess) {
            (void)cudaGetLastError();
            return false;
        }
        record_       = static_cast<std::byte*>(fresh);
        record_bytes_ = bytes;
        return true;
    }

    // One-shot assertion that the mirror really carries the device operands: it re-reads a few
    // elements through a SECOND path (element-wise D2H addressed by the operand's own formula) and
    // compares. A transposed Q copy, a wrong row or a stale position would show up here.
    void check_against_device(const Tensor& q, const Tensor& positions,
                              const PagedKVBatchLayerView& cache) {
        static bool reported = false;
        if (reported) { return; }
        reported = true;
        const int heads  = q.ne[1];
        const int tokens = q.ne[2];
        const auto* q_device = static_cast<const std::uint16_t*>(q.data);
        int q_checked = 0;
        int q_mismatch = 0;
        static constexpr int kSamples = 4;
        for (int sample = 0; sample < kSamples; ++sample) {
            const int head  = (sample * 7 + 1) % heads;
            const int token = (sample * 3) % tokens;
            const int dim   = (sample * 61 + 5) % cpu_partial::kHeadDim;
            // causal_q_index order: element (token, head, d) at d + 256 * (head + QHeads * token).
            const std::int64_t device_index = dim + std::int64_t{256} * (head + heads * token);
            const std::int64_t mirror_index = device_index;  // the mirror is a straight copy
            std::uint16_t device_value      = 0;
            if (cudaMemcpy(&device_value, q_device + device_index, sizeof device_value,
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
                (void)cudaGetLastError();
                continue;
            }
            ++q_checked;
            q_mismatch += q_[static_cast<std::size_t>(mirror_index)] == device_value ? 0 : 1;
        }
        int position_mismatch = 0;
        for (int token = 0; token < tokens; ++token) {
            std::int32_t device_position = 0;
            if (cudaMemcpy(&device_position,
                           static_cast<const std::int32_t*>(positions.data) + token,
                           sizeof device_position, cudaMemcpyDeviceToHost) != cudaSuccess) {
                (void)cudaGetLastError();
                continue;
            }
            position_mismatch += positions_[static_cast<std::size_t>(token)] == device_position ? 0 : 1;
        }
        std::int32_t row = 0;
        if (cudaMemcpy(&row, cache.block_tables.data, sizeof row, cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
            (void)cudaGetLastError();
            return;
        }
        int table_mismatch = 0;
        const auto* table_device = static_cast<const std::int32_t*>(cache.block_tables.data);
        for (std::int32_t logical = 0; logical < static_cast<std::int32_t>(table_.size()); ++logical) {
            std::int32_t device_entry = 0;
            const std::int64_t index =
                static_cast<std::int64_t>(row) * cache.block_tables.ne[0] + logical;
            if (cudaMemcpy(&device_entry, table_device + index, sizeof device_entry,
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
                (void)cudaGetLastError();
                break;
            }
            table_mismatch += table_[static_cast<std::size_t>(logical)] == device_entry ? 0 : 1;
        }
        std::fprintf(stderr,
                     "[kv-cpu] mirror check layer %d: q %d/%d table %d/%d positions %d/%d "
                     "(head=%d tokens=%d physical_pages=%d)\n",
                     layer_, q_checked - q_mismatch, q_checked,
                     static_cast<int>(table_.size()) - table_mismatch,
                     static_cast<int>(table_.size()),
                     tokens - position_mismatch, tokens, heads, tokens, cache.k_pages.ne[3]);
        std::fflush(stderr);
    }

    HostKVPageLayout layout_{};
    std::vector<std::byte> mirror_;
    std::vector<std::int32_t> table_;
    std::vector<std::uint16_t> q_;
    std::vector<std::int32_t> positions_;
    ops::detail::RkHostPartialView view_{};
    std::byte* record_        = nullptr;
    std::size_t record_bytes_ = 0;
    int layer_                = -1;
    bool mirroring_           = false;
};

} // namespace ninfer::models::qwen3_5::detail
