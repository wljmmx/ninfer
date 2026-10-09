#pragma once
#include "ops/softmax_attention/dense/causal_cache/int8/tile_io.cuh"

#include "ops/softmax_attention/dense/causal_cache/int8/schedule.cuh"
#include "ops/softmax_attention/common/causal_partition.h"
#include "ops/softmax_attention/common/causal_epilogue.cuh"
#include "ops/softmax_attention/common/causal_softmax.cuh"
#include "ops/kv_cache/rk4_codec.cuh"
#include "ops/kernel/e8_lattice.cuh"
#include "ops/kernel/e8_root_codec.cuh"

namespace ninfer::ops::detail {

// Native INT8 QK accumulates each G64 group in INT32, then applies Q/K scales in FP32.
// PV dequantizes represented V to FP16 and accumulates in FP32.
// When PackedV is true (rk8v4 / rk4v4 / rk4v4-e8 / rk2v4-e8), the value cache stores
// 4-bit codes (two per byte, symmetric [-7,7] with absmax/7 FP16 scales); the append
// path packs int4 and the V loader unpacks to int8 before the same FP16 PV MMA.
// When PackedK is true (rk4v4 / rk4v4-e8 / rk2v4-e8), the key cache also stores 4-bit
// codes; E8Lattice selects E8 nearest-lattice projection vs plain RTN for K quantization.
template <class Geometry, class Schedule, bool MultiBatch, bool Masked, class CacheInput,
          bool ParallelQueries = false, bool PackedV = false, bool PackedK = false,
          bool E8Lattice = false, bool E8Root = false, bool V8Root = false>
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocks) __global__
    void int8_kv_grouped_mma_kernel(
        const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos,
        typename Int8KvCacheView<CacheInput::writes_cache>::Code* cache_k_i8,
        typename Int8KvCacheView<CacheInput::writes_cache>::Code* cache_v_i8,
        typename Int8KvCacheView<CacheInput::writes_cache>::KeyScale* cache_k_scale,
        typename Int8KvCacheView<CacheInput::writes_cache>::ValueScale* cache_v_scale,
        const std::int32_t* block_tables, const std::int32_t* valid_columns,
        const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t full_width,
        std::int32_t logical_capacity, CausalKvPartition partition, float scale, float* partial_acc,
        float* partial_m, float* partial_l) {
    constexpr int TokenTile            = Schedule::kTokenTile;
    constexpr int WarpsPerCta          = Schedule::kWarps;
    constexpr int KeyBlock             = Schedule::kKeyRows;
    constexpr bool DynamicArena        = Schedule::kDynamicArena;
    constexpr int Wc                   = WarpsPerCta;
    constexpr int RowCount             = TokenTile * Geometry::GroupSize;
    constexpr int RowTiles             = (RowCount + 15) / 16;
    constexpr int Br                   = RowTiles * 16;
    constexpr int Bc                   = KeyBlock;
    constexpr int D                    = 256;
    constexpr int DB16                 = D / 2;
    constexpr int Threads              = Wc * 32;
    constexpr int Groups               = kKVCacheInt8Groups;
    constexpr int GroupKc              = kKVCacheInt8Group / 32;
    constexpr int QKKs                 = D / 32;
    constexpr int QKNt                 = Bc / 8;
    constexpr int ConsumerWarpsPerTile = Wc / RowTiles;
    constexpr int PVNtPerWarp          = D / (ConsumerWarpsPerTile * 8);
    constexpr int PVKs                 = Bc / 16;
    constexpr int ProducerThreads      = RowTiles * 32;
    constexpr int VLoaderThreads       = Threads - ProducerThreads;
    constexpr float Log2E              = kLog2E;
    constexpr unsigned FullMask        = 0xffffffffu;

    static_assert(TokenTile >= 1 && TokenTile * Geometry::GroupSize <= 64);
    static_assert(Bc == 32 || Bc == 64);
    static_assert(RowTiles >= 1 && RowTiles <= 4);
    static_assert(Wc % RowTiles == 0);
    static_assert(PVNtPerWarp == 2 || PVNtPerWarp == 4 || PVNtPerWarp == 8 || PVNtPerWarp == 16);
    static_assert(QKKs == Groups * GroupKc);

    // Keep Q in a compact dedicated tile so the producer can reload one
    // 64-dimension group at a time instead of carrying all eight fragments in
    // registers across the whole kernel. The main arena holds K i8, V i8, and
    // V FP16 during the key loop.
    __shared__ __align__(16) std::int8_t q_s[Br * D];
    __shared__ __align__(16) std::int8_t static_r_s[DynamicArena ? 16 : 4 * Bc * D];
    extern __shared__ __align__(16) std::int8_t dynamic_r_s[];
    std::int8_t* r_s     = DynamicArena ? dynamic_r_s : static_r_s;
    std::int8_t* q_i8    = q_s;
    float* q_scale_tmp   = reinterpret_cast<float*>(r_s);
    std::int8_t* k_i8    = r_s;
    __nv_bfloat16* q_b16 = reinterpret_cast<__nv_bfloat16*>(q_i8);
    __nv_bfloat16* k_b16 = reinterpret_cast<__nv_bfloat16*>(k_i8);
    std::int8_t* v_i8    = r_s + Bc * D;
    __half* v_f16        = reinterpret_cast<__half*>(r_s + 2 * Bc * D);
    __shared__ __align__(16) __half p_s[Br * Bc];
    __shared__ float alpha_s[Br];
    __shared__ __align__(16) __half k_scale_s[Bc * Groups];
    __shared__ __align__(16) __half v_scale_s[Bc * Groups];

    static_assert(!ParallelQueries || !CacheInput::writes_cache);
    const int kv_head      = ParallelQueries ? blockIdx.x % Geometry::KVHeads : blockIdx.x;
    const int column_begin = ParallelQueries ? (blockIdx.x / Geometry::KVHeads) * TokenTile : 0;
    const int tile_tokens = ParallelQueries ? min(TokenTile, full_width - column_begin) : TokenTile;
    const int partial_width = full_width;
    const int partial_begin = column_begin;
    const int split         = static_cast<int>(blockIdx.y);
    const int batch         = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count   = static_cast<int>(gridDim.y);
    const int tid           = static_cast<int>(threadIdx.x);
    const int warp          = tid >> 5;
    const int lane          = tid & 31;

    int valid_tokens = tile_tokens;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : min(remaining, tile_tokens);
    }
    std::int64_t column_base = column_begin;
    if constexpr (MultiBatch) { column_base += static_cast<std::int64_t>(batch) * full_width; }
    q += static_cast<std::int64_t>(256) * Geometry::QHeads * column_base;
    const int last_pos = pos[(MultiBatch ? batch * full_width : 0) + full_width - 1];
    pos += column_base;
    if constexpr (CacheInput::writes_cache) {
        input.k += static_cast<std::int64_t>(256) * Geometry::KVHeads * column_base;
        input.v += static_cast<std::int64_t>(256) * Geometry::KVHeads * column_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        partial_acc +=
            static_cast<std::int64_t>(batch) * 256 * Geometry::QHeads * partial_width * split_count;
        partial_m +=
            static_cast<std::int64_t>(batch) * Geometry::QHeads * partial_width * split_count;
        partial_l +=
            static_cast<std::int64_t>(batch) * Geometry::QHeads * partial_width * split_count;
    }

    if (valid_tokens == 0) return; // Merge writes exact zero for masked columns.
    if (pos[0] < 0 || last_pos < 0 || last_pos >= logical_capacity) return;
    const int window             = last_pos + 1;
    const int active_split_count = partition.active(window);
    if (split >= active_split_count) return;
    const int logical_tiles    = div_up(window, Bc);
    const int first_owned_tile = split * logical_tiles / active_split_count;
    const int end_owned_tile   = (split + 1) * logical_tiles / active_split_count;
    const int split_start      = first_owned_tile * Bc;
    const int split_end        = min(end_owned_tile * Bc, window);
    const int first_tile       = split_start;
    const int key_blocks       = div_up(split_end - first_tile, Bc);
    

    if constexpr (CacheInput::writes_cache) {
        
        // Decompose H256 as H4 over four independently transformed H64 groups. The existing
        // (token, group) warp schedule computes all H64 fragments in parallel; the FP32 main arena
        // is the exchange point for the final H4 stage. This retains the complete transform's
        // butterfly/rounding order while shortening the fused append critical path. V stays in
        // the original coordinates and retains the existing G64 codec.
        float* k_h64_s = reinterpret_cast<float*>(r_s);
        for (int pair = warp; pair < valid_tokens * Groups; pair += Wc) {
            const int token    = pair / Groups;
            const int grp      = pair - token * Groups;
            const int position = pos[token];
            if (position < split_start || position >= split_end) { continue; }
            const int d0            = grp * kKVCacheInt8Group + lane;
            const int d1            = d0 + 32;
            const std::int64_t src0 = causal_new_index<Geometry>(kv_head, d0, token);
            const std::int64_t src1 = causal_new_index<Geometry>(kv_head, d1, token);
            float k_h64[2] = {__bfloat162float(input.k[src0]), __bfloat162float(input.k[src1])};
            hadamard_d64_fragment_inplace(k_h64, lane);
            k_h64_s[token * D + d0] = k_h64[0];
            k_h64_s[token * D + d1] = k_h64[1];
        }
        __syncthreads();
        

        for (int pair = warp; pair < valid_tokens * Groups; pair += Wc) {
            const int token    = pair / Groups;
            const int grp      = pair - token * Groups;
            const int position = pos[token];
            
            if (position < split_start || position >= split_end) { continue; }
            const int physical_page = block_table[position >> kPagedKVPageShift];
            const int page_offset   = position & kPagedKVPageMask;
            const int d0            = grp * kKVCacheInt8Group + lane;
            const int d1            = d0 + 32;
            const std::int64_t src0 = causal_new_index<Geometry>(kv_head, d0, token);
            const std::int64_t src1 = causal_new_index<Geometry>(kv_head, d1, token);

            float k_out[2];
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int dh   = lane + half * 32;
                const float x0 = k_h64_s[token * D + dh];
                const float x1 = k_h64_s[token * D + kKVCacheInt8Group + dh];
                const float x2 = k_h64_s[token * D + 2 * kKVCacheInt8Group + dh];
                const float x3 = k_h64_s[token * D + 3 * kKVCacheInt8Group + dh];
                k_out[half]    = normalized_hadamard_d256_group_value_from_h64(x0, x1, x2, x3, grp);
            }

            float kv0    = k_out[0];
            float kv1    = k_out[1];
            const float vv0    = __bfloat162float(input.v[src0]);
            const float vv1    = __bfloat162float(input.v[src1]);
            float kamax        = fmaxf(fabsf(kv0), fabsf(kv1));
            float vamax        = fmaxf(fabsf(vv0), fabsf(vv1));
            kamax              = warp_max(kamax, FullMask);
            vamax              = warp_max(vamax, FullMask);
            
            const auto k_quant = kv_cache_int8_quant_params(kamax);
            // PackedV uses the 4-bit range [-7, 7]: scale = FP16-RNE(absmax/7).
            __half v_scale;
            float v_inv_scale;
            if constexpr (PackedV || V8Root) {
                v_scale     = rk4_absmax_to_scale_h(vamax);
                const float represented = __half2float(v_scale);
                v_inv_scale = represented > 0.0f ? 1.0f / represented : 0.0f;
            } else {
                v_scale     = kv_cache_int8_quant_params(vamax).scale;
                v_inv_scale = kv_cache_int8_quant_params(vamax).inverse_scale;
            }
            // K quantization: plain int8 (default), int4 RTN (PackedK, !E8Lattice),
            // or int4 E8 lattice projection (PackedK, E8Lattice). For E8Lattice the
            // 8D Hadamard-rotated vector is projected to the nearest E8 lattice point
            // before clamping to [-7, 7] and packing.
            if constexpr (E8Root) {
                // rk2v4-e8 K append: encode BOTH 32-dim halves of this G64 group.
                // Scale convention follows the reference cylinder codec (radius
                // table centered on absmax/7, NOT absmax/127 — a /127 scale saturates
                // rad_idx at 15 and collapses K magnitudes ~8x). Each 8-lane
                // subgroup encodes one 8D subspace: kv0 covers subspaces 0-3,
                // kv1 covers subspaces 4-7 of this group.
                __half k_scale_h = rk4_absmax_to_scale_h(kamax);
                const float ks   = __half2float(k_scale_h);
                uint8_t root0, rad0, root1, rad1;
                e8_encode_cylinder_8d_warp(kv0, ks, root0, rad0, lane);
                e8_encode_cylinder_8d_warp(kv1, ks, root1, rad1, lane);
                if ((lane & 7) == 0) {
                    const int s0 = lane >> 3;
                    const int s1 = 4 + (lane >> 3);
                    const std::int64_t ko0 = paged_kv_element_offset<64, Geometry::KVHeads>(
                        physical_page, kv_head, page_offset, (grp * 8 + s0) * 2);
                    *reinterpret_cast<std::uint16_t*>(&cache_k_i8[ko0]) =
                        static_cast<std::uint16_t>(root0 | (rad0 << 8));
                    const std::int64_t ko1 = paged_kv_element_offset<64, Geometry::KVHeads>(
                        physical_page, kv_head, page_offset, (grp * 8 + s1) * 2);
                    *reinterpret_cast<std::uint16_t*>(&cache_k_i8[ko1]) =
                        static_cast<std::uint16_t>(root1 | (rad1 << 8));
                }
                if (lane == 0) {
                    const std::int64_t so = kv_cache_int8_quant_scale_index<Geometry>(
                        physical_page, kv_head, grp, page_offset);
                    cache_k_scale[so] = k_scale_h;
                    cache_v_scale[so] = v_scale;
                }
            } else if constexpr (PackedK) {
                // int4 K: scale = absmax/7, clamp [-7, 7], pack two adjacent dims/byte.
                __half k_scale_h = rk4_absmax_to_scale_h(kamax);
                float k_inv = __half2float(k_scale_h);
                k_inv = k_inv > 0.0f ? 1.0f / k_inv : 0.0f;
                if constexpr (E8Lattice) {
                    // E8 projection in the SCALED space (units of k_scale),
                    // matching the reference codec: project the scaled 8D subspace
                    // to the nearest E8 lattice point, then round back onto the
                    // int4 grid below (the half-integral D8+0.5 coset collapses in
                    // that round — documented approximation of the reference).
                    float k0_mut = kv0 * k_inv, k1_mut = kv1 * k_inv;
                    e8_project_8d_warp(k0_mut, k1_mut, lane);
                    kv0 = k0_mut;
                    kv1 = k1_mut;
                    k_inv = 1.0f;
                }
                const float kv0_next = __shfl_xor_sync(FullMask, kv0, 1);
                const float kv1_next = __shfl_xor_sync(FullMask, kv1, 1);
                if ((lane & 1) == 0) {
                    const auto kc0 = rk4_quant_code(kv0, k_inv);
                    const auto kc1 = rk4_quant_code(kv0_next, k_inv);
                    const std::int64_t ko0 = rk4_v_code_index<Geometry>(physical_page, kv_head,
                                                                       d0 >> 1, page_offset);
                    cache_k_i8[ko0] = static_cast<std::int8_t>(rk4_pack(kc0, kc1));
                    const auto kc2 = rk4_quant_code(kv1, k_inv);
                    const auto kc3 = rk4_quant_code(kv1_next, k_inv);
                    const std::int64_t ko1 = rk4_v_code_index<Geometry>(physical_page, kv_head,
                                                                       d1 >> 1, page_offset);
                    cache_k_i8[ko1] = static_cast<std::int8_t>(rk4_pack(kc2, kc3));
                }
                if (lane == 0) {
                    const std::int64_t so = kv_cache_int8_quant_scale_index<Geometry>(
                        physical_page, kv_head, grp, page_offset);
                    cache_k_scale[so] = k_scale_h;
                    cache_v_scale[so] = v_scale;
                }
            } else {
                cache_k_i8[kv_cache_int8_quant_code_index<Geometry>(physical_page, kv_head, d0,
                                                                     page_offset)] =
                    kv_cache_int8_quant_code(kv0, k_quant.inverse_scale);
                cache_k_i8[kv_cache_int8_quant_code_index<Geometry>(physical_page, kv_head, d1,
                                                                     page_offset)] =
                    kv_cache_int8_quant_code(kv1, k_quant.inverse_scale);
                if (lane == 0) {
                    const std::int64_t so = kv_cache_int8_quant_scale_index<Geometry>(
                        physical_page, kv_head, grp, page_offset);
                    cache_k_scale[so] = k_quant.scale;
                    cache_v_scale[so] = v_scale;
                }
            }
            if constexpr (V8Root) {
                // EXPERIMENT rk4v2-e8 V append: 2-bit E8 cylinder codes (V is not
                // Hadamard-rotated). One (root, rad_axis) byte pair per 8D subspace;
                // lanes 0/8/16/24 (subgroup leaders) write the pair bytes.
                const float vs8 = __half2float(v_scale);
                uint8_t vr0, vra0, vr1, vra1;
                e8_encode_cylinder_8d_warp(vv0, vs8, vr0, vra0, lane);
                e8_encode_cylinder_8d_warp(vv1, vs8, vr1, vra1, lane);
                if ((lane & 7) == 0) {
                    const int s0 = lane >> 3;
                    const int s1 = 4 + (lane >> 3);
                    const std::int64_t vo0 = paged_kv_element_offset<64, Geometry::KVHeads>(
                        physical_page, kv_head, page_offset, (grp * 8 + s0) * 2);
                    *reinterpret_cast<std::uint16_t*>(&cache_v_i8[vo0]) =
                        static_cast<std::uint16_t>(vr0 | (vra0 << 8));
                    const std::int64_t vo1 = paged_kv_element_offset<64, Geometry::KVHeads>(
                        physical_page, kv_head, page_offset, (grp * 8 + s1) * 2);
                    *reinterpret_cast<std::uint16_t*>(&cache_v_i8[vo1]) =
                        static_cast<std::uint16_t>(vr1 | (vra1 << 8));
                }
            } else if constexpr (PackedV) {
                // rk V plane: 4-bit codes, adjacent dims (2k, 2k+1) packed per byte.
                // __shfl_xor_sync with FullMask requires ALL 32 lanes to participate —
                // it must be called OUTSIDE the even-lane guard to avoid a deadlock.

                // __shfl_xor_sync with FullMask requires ALL 32 lanes to participate —
                // it must be called OUTSIDE the even-lane guard to avoid a deadlock.
                const float vv0_next = __shfl_xor_sync(FullMask, vv0, 1);
                const float vv1_next = __shfl_xor_sync(FullMask, vv1, 1);
                if ((lane & 1) == 0) {
                    const std::int64_t vo0 = rk4_v_code_index<Geometry>(physical_page, kv_head,
                                                                       d0 >> 1, page_offset);
                    const auto c0 = rk4_quant_code(vv0, v_inv_scale);
                    const auto c1 = rk4_quant_code(vv0_next, v_inv_scale);
                    cache_v_i8[vo0] = static_cast<std::int8_t>(rk4_pack(c0, c1));
                    const std::int64_t vo1 = rk4_v_code_index<Geometry>(physical_page, kv_head,
                                                                       d1 >> 1, page_offset);
                    const auto c2 = rk4_quant_code(vv1, v_inv_scale);
                    const auto c3 = rk4_quant_code(vv1_next, v_inv_scale);
                    cache_v_i8[vo1] = static_cast<std::int8_t>(rk4_pack(c2, c3));
                }
            } else {
                cache_v_i8[kv_cache_int8_quant_code_index<Geometry>(physical_page, kv_head, d0,
                                                                    page_offset)] =
                    kv_cache_int8_quant_code(vv0, v_inv_scale);
                cache_v_i8[kv_cache_int8_quant_code_index<Geometry>(physical_page, kv_head, d1,
                                                                    page_offset)] =
                    kv_cache_int8_quant_code(vv1, v_inv_scale);
            }
        }
        __syncthreads();
    }

    for (int i = tid; i < Br * D; i += Threads) { q_i8[i] = 0; }
    for (int i = tid; i < RowCount * Groups; i += Threads) { q_scale_tmp[i] = 0.0f; }
    __syncthreads();

    for (int row = warp; row < RowCount; row += Wc) {
        int q_head = 0;
        int token  = 0;
        causal_row_to_qt<Geometry>(row, kv_head, q_head, token);
        float q_values[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            q_values[r] = token < valid_tokens
                              ? __bfloat162float(q[causal_q_index<Geometry>(q_head, d, token)])
                              : 0.0f;
        }
        normalized_hadamard_d256_inplace(q_values, lane);

#pragma unroll
        for (int grp = 0; grp < Groups; ++grp) {
            const int d0    = grp * kKVCacheInt8Group + lane;
            const int d1    = d0 + 32;
            const float x0  = q_values[2 * grp];
            const float x1  = q_values[2 * grp + 1];
            float amax      = fmaxf(fabsf(x0), fabsf(x1));
            amax            = warp_max(amax, FullMask);
            const float qs  = amax > 0.0f ? amax / 127.0f : 0.0f;
            const float inv = qs > 0.0f ? 1.0f / qs : 0.0f;
            causal_store_query_code(q_i8, row, d0, kv_cache_int8_quant_code(x0, inv));
            causal_store_query_code(q_i8, row, d1, kv_cache_int8_quant_code(x1, inv));
            if (lane == 0) { q_scale_tmp[row * Groups + grp] = qs; }
        }
    }
    __syncthreads();

    const int gid = lane >> 2;
    const int lid = lane & 3;

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    float q_scale_r0[Groups];
    float q_scale_r1[Groups];
    if (warp < RowTiles) {
        const int producer_row0 = warp * 16 + gid;
#pragma unroll
        for (int g = 0; g < Groups; ++g) {
            float qs0     = (lid == 0 && producer_row0 < tile_tokens * Geometry::GroupSize)
                                ? q_scale_tmp[producer_row0 * Groups + g]
                                : 0.0f;
            float qs1     = (lid == 0 && producer_row0 + 8 < RowCount)
                                ? q_scale_tmp[(producer_row0 + 8) * Groups + g]
                                : 0.0f;
            q_scale_r0[g] = __shfl_sync(FullMask, qs0, gid * 4);
            q_scale_r1[g] = __shfl_sync(FullMask, qs1, gid * 4);
        }
    }
    __syncthreads();

    float acc[PVNtPerWarp][4];
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }

    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F;
    float l0 = 0.0f, l1 = 0.0f;

    auto issue_kv_tile = [&](int tile_k0, int physical_page) {
        for (int key_l = tid; key_l < Bc; key_l += Threads) {
            const int key = tile_k0 + key_l;
            if (key >= split_start && key < split_end) {
                const std::int64_t off = kv_cache_int8_quant_scale_index<Geometry>(
                    physical_page, kv_head, 0, key & kPagedKVPageMask);
                ninfer::ops::cp_async<8>(&k_scale_s[key_l * Groups], &cache_k_scale[off]);
                ninfer::ops::cp_async<8>(&v_scale_s[key_l * Groups], &cache_v_scale[off]);
            } else {
                store_vec(&k_scale_s[key_l * Groups], make_int2(0, 0));
                store_vec(&v_scale_s[key_l * Groups], make_int2(0, 0));
            }
        }
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (D / 16); chunk += Threads) {
            const int key_l = chunk / (D / 16);
            const int dc    = chunk - key_l * (D / 16);
            const int d     = dc * 16;
            const int key   = tile_k0 + key_l;
            if (key >= split_start && key < split_end) {
                std::int8_t* dst = &k_i8[key_l * D + causal_swizzle(key_l, dc * 8) * 2];
                const std::int64_t off = kv_cache_int8_quant_code_index<Geometry>(
                    physical_page, kv_head, d, key & kPagedKVPageMask);
                if constexpr (E8Root) {
                    // rk2v4-e8 K read-back: this 16-dim chunk spans two consecutive
                    // 8D subspaces — 4 cache bytes (root+rad_axis each) starting at
                    // byte offset d/4 of the 64-byte per-token K plane. One uint32
                    // load (the offset is always 4-aligned: sub = (d/8)&7 is even
                    // because d is a multiple of 16); one thread decodes both
                    // subspaces and writes ALL 16 int8 codes into the swizzled smem
                    // slot (dst) that the ldmatrix B-fragment reads — the same slot
                    // the int8 cp_async path fills, so the write must not bypass the
                    // swizzle and must cover the full chunk.
                    const int grp_for_read = d / 64;
                    const int sub          = (d / 8) & 7;
                    const int byte_offset  = (grp_for_read * 8 + sub) * 2;
                    const std::int64_t koff = paged_kv_element_offset<64, Geometry::KVHeads>(
                        physical_page, kv_head, key & kPagedKVPageMask, byte_offset);
                    const std::uint32_t raw =
                        *reinterpret_cast<const std::uint32_t*>(&cache_k_i8[koff]);
                    const uint8_t root0 = static_cast<uint8_t>(raw);
                    const uint8_t rad0  = static_cast<uint8_t>(raw >> 8);
                    const uint8_t root1 = static_cast<uint8_t>(raw >> 16);
                    const uint8_t rad1  = static_cast<uint8_t>(raw >> 24);
                    __align__(8) int8_t dec0[8];
                    __align__(8) int8_t dec1[8];
                    e8_root_decode_8d_fast(root0, rad0, dec0);
                    e8_root_decode_8d_fast(root1, rad1, dec1);
                    *reinterpret_cast<uint64_t*>(dst) =
                        *reinterpret_cast<const uint64_t*>(dec0);
                    *reinterpret_cast<uint64_t*>(dst + 8) =
                        *reinterpret_cast<const uint64_t*>(dec1);
                } else if constexpr (PackedK) {
                    // rk K read-back: vectorized 4-byte loads + fast nibble unpack
                    // (the same rk4_unpack_lo/hi_fast path as the V loader) straight
                    // into the swizzled smem slot (dst) the ldmatrix B-fragment
                    // reads — the int8 cp_async path fills the same 16-byte chunk,
                    // so the unpack must not bypass the swizzle.
                    const std::int64_t koff = rk4_v_code_index<Geometry>(
                        physical_page, kv_head, d >> 1, key & kPagedKVPageMask);
#pragma unroll
                    for (int half4 = 0; half4 < 2; ++half4) {
                        const std::uint32_t raw = *reinterpret_cast<const std::uint32_t*>(
                            &cache_k_i8[koff + 4 * half4]);
                        const std::uint32_t lo = rk4_unpack_lo_fast(raw);
                        const std::uint32_t hi = rk4_unpack_hi_fast(raw);
                        const auto* lo8 = reinterpret_cast<const std::int8_t*>(&lo);
                        const auto* hi8 = reinterpret_cast<const std::int8_t*>(&hi);
#pragma unroll
                        for (int b = 0; b < 4; ++b) {
                            dst[8 * half4 + 2 * b]     = lo8[b];
                            dst[8 * half4 + 2 * b + 1] = hi8[b];
                        }
                    }
                } else {
                    ninfer::ops::cp_async<16>(dst, &cache_k_i8[off]);
                }
                if constexpr (V8Root) {
                    // EXPERIMENT rk4v2-e8 V read-back: this 16-dim chunk spans two
                    // consecutive 8D subspaces (4 cache bytes at byte offset d/4 of
                    // the 64-byte V plane). Decode straight into v_i8 at stride D
                    // (unswizzled: the V dequant reads &v_i8[key_l*D + d]).
                    const int grp_for_read = d / 64;
                    const int sub          = (d / 8) & 7;
                    const int byte_offset  = (grp_for_read * 8 + sub) * 2;
                    const std::int64_t voff = paged_kv_element_offset<64, Geometry::KVHeads>(
                        physical_page, kv_head, key & kPagedKVPageMask, byte_offset);
                    const std::uint32_t raw =
                        *reinterpret_cast<const std::uint32_t*>(&cache_v_i8[voff]);
                    const uint8_t vr0  = static_cast<uint8_t>(raw);
                    const uint8_t vra0 = static_cast<uint8_t>(raw >> 8);
                    const uint8_t vr1  = static_cast<uint8_t>(raw >> 16);
                    const uint8_t vra1 = static_cast<uint8_t>(raw >> 24);
                    __align__(8) int8_t vdec0[8];
                    __align__(8) int8_t vdec1[8];
                    e8_root_decode_8d_fast(vr0, vra0, vdec0);
                    e8_root_decode_8d_fast(vr1, vra1, vdec1);
                    *reinterpret_cast<uint64_t*>(&v_i8[key_l * D + d]) =
                        *reinterpret_cast<const uint64_t*>(vdec0);
                    *reinterpret_cast<uint64_t*>(&v_i8[key_l * D + d + 8]) =
                        *reinterpret_cast<const uint64_t*>(vdec1);
                } else if constexpr (PackedV) {
                    // rk V: 4-bit codes. cp_async 8 packed bytes (16 dims) into smem at

                    // the half-offset position (d>>1). The V cache has stride 128, so
                    // offsets are 8-byte aligned (guaranteed by paged_kv_element_offset
                    // with LeadingExtent=128 and dc*8 granularity).
                    const std::int64_t voff = rk4_v_code_index<Geometry>(
                        physical_page, kv_head, d >> 1, key & kPagedKVPageMask);
                    ninfer::ops::cp_async<8>(&v_i8[key_l * D + (d >> 1)], &cache_v_i8[voff]);
                } else {
                    ninfer::ops::cp_async<16>(&v_i8[key_l * D + d], &cache_v_i8[off]);
                }
            } else {
                std::int8_t* dst = &k_i8[key_l * D + causal_swizzle(key_l, dc * 8) * 2];
                store_vec(dst, make_int4(0, 0, 0, 0));
                rk4_zero_packed_v<PackedV>(&v_i8[key_l * D], d);
            }
        }
        ninfer::ops::cp_commit();
    };

    int physical_page = block_table[first_tile >> kPagedKVPageShift];
    
    issue_kv_tile(first_tile, physical_page);
    ninfer::ops::cp_wait<0>();
    __syncthreads();
    

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = first_tile + kb * Bc;

        // One warp per row tile produces P and alpha while the remaining warps
        // stream/dequant V.
        if (warp < RowTiles) {
            const int producer_row_base = warp * 16;
            __half* p_sw                = &p_s[producer_row_base * Bc];
            float score[QKNt][4];
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                score[nt][0] = 0.0f;
                score[nt][1] = 0.0f;
                score[nt][2] = 0.0f;
                score[nt][3] = 0.0f;
            }

#pragma unroll
            for (int g = 0; g < Groups; ++g) {
                unsigned af[GroupKc][4];
#pragma unroll
                for (int kk = 0; kk < GroupKc; ++kk) {
                    const int k    = g * GroupKc + kk;
                    const int acol = k * 16 + a_coloff;
                    ldmatrix_x4(
                        af[kk][0], af[kk][1], af[kk][2], af[kk][3],
                        smem_addr(&q_b16[(producer_row_base + a_rowoff) * DB16 +
                                         causal_swizzle(producer_row_base + a_rowoff, acol)]));
                }

#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#pragma unroll
                    for (int kk = 0; kk < GroupKc; ++kk) {
                        const int k    = g * GroupKc + kk;
                        const int brow = nt * 8 + b_rin;
                        const int bcol = k * 16 + b_koff;
                        unsigned bf[2];
                        ldmatrix_x2(bf[0], bf[1],
                                    smem_addr(&k_b16[brow * DB16 + causal_swizzle(brow, bcol)]));
                        mma_s8(c0, c1, c2, c3, af[kk][0], af[kk][1], af[kk][2], af[kk][3], bf[0],
                               bf[1]);
                    }
                    const int keya = nt * 8 + 2 * lid;
                    const int keyb = keya + 1;
                    float ka       = 0.0f;
                    float kb2      = 0.0f;
                    if (gid == 0) {
                        ka  = __half2float(k_scale_s[keya * Groups + g]);
                        kb2 = __half2float(k_scale_s[keyb * Groups + g]);
                    }
                    ka  = __shfl_sync(FullMask, ka, lid);
                    kb2 = __shfl_sync(FullMask, kb2, lid);
                    score[nt][0] += q_scale_r0[g] * ka * static_cast<float>(c0);
                    score[nt][1] += q_scale_r0[g] * kb2 * static_cast<float>(c1);
                    score[nt][2] += q_scale_r1[g] * ka * static_cast<float>(c2);
                    score[nt][3] += q_scale_r1[g] * kb2 * static_cast<float>(c3);
                }
            }

            const int row0 = producer_row_base + gid;
            const int row1 = row0 + 8;
            int q_head0 = 0, token0 = 0, q_head1 = 0, token1 = 0;
            causal_row_to_qt<Geometry>(row0, kv_head, q_head0, token0);
            causal_row_to_qt<Geometry>(row1, kv_head, q_head1, token1);
            const int qabs0 = (row0 < tile_tokens * Geometry::GroupSize) ? pos[token0] : -1;
            const int qabs1 = (row1 < tile_tokens * Geometry::GroupSize) ? pos[token1] : -1;
            float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int col0 = nt * 8 + 2 * lid;
                const int col1 = col0 + 1;
                const int key0 = k0 + col0;
                const int key1 = k0 + col1;
                score[nt][0]   = (row0 < tile_tokens * Geometry::GroupSize && key0 >= split_start &&
                                key0 < split_end && key0 <= qabs0)
                                     ? score[nt][0] * scale
                                     : -CUDART_INF_F;
                score[nt][1]   = (row0 < tile_tokens * Geometry::GroupSize && key1 >= split_start &&
                                key1 < split_end && key1 <= qabs0)
                                     ? score[nt][1] * scale
                                     : -CUDART_INF_F;
                score[nt][2]   = (row1 < tile_tokens * Geometry::GroupSize && key0 >= split_start &&
                                key0 < split_end && key0 <= qabs1)
                                     ? score[nt][2] * scale
                                     : -CUDART_INF_F;
                score[nt][3]   = (row1 < tile_tokens * Geometry::GroupSize && key1 >= split_start &&
                                key1 < split_end && key1 <= qabs1)
                                     ? score[nt][3] * scale
                                     : -CUDART_INF_F;
                bm0            = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1            = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
            bm0 = warp_max<4>(bm0, FullMask);
            bm1 = warp_max<4>(bm1, FullMask);

            const float nm0 = fmaxf(m0, bm0);
            const float nm1 = fmaxf(m1, bm1);
            const float alpha0 =
                (m0 == -CUDART_INF_F) ? 0.0f : causal_exp_difference(m0, nm0, Log2E);
            const float alpha1 =
                (m1 == -CUDART_INF_F) ? 0.0f : causal_exp_difference(m1, nm1, Log2E);

            float bl0 = 0.0f, bl1 = 0.0f;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int col0  = nt * 8 + 2 * lid;
                const int col1  = col0 + 1;
                const float p00 = (nm0 > -CUDART_INF_F && score[nt][0] > -CUDART_INF_F)
                                      ? causal_exp_difference(score[nt][0], nm0, Log2E)
                                      : 0.0f;
                const float p01 = (nm0 > -CUDART_INF_F && score[nt][1] > -CUDART_INF_F)
                                      ? causal_exp_difference(score[nt][1], nm0, Log2E)
                                      : 0.0f;
                const float p10 = (nm1 > -CUDART_INF_F && score[nt][2] > -CUDART_INF_F)
                                      ? causal_exp_difference(score[nt][2], nm1, Log2E)
                                      : 0.0f;
                const float p11 = (nm1 > -CUDART_INF_F && score[nt][3] > -CUDART_INF_F)
                                      ? causal_exp_difference(score[nt][3], nm1, Log2E)
                                      : 0.0f;
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                p_sw[gid * Bc + causal_probability_swizzle<Bc>(gid, col0)] = __float2half_rn(p00);
                p_sw[gid * Bc + causal_probability_swizzle<Bc>(gid, col1)] = __float2half_rn(p01);
                p_sw[(gid + 8) * Bc + causal_probability_swizzle<Bc>(gid + 8, col0)] =
                    __float2half_rn(p10);
                p_sw[(gid + 8) * Bc + causal_probability_swizzle<Bc>(gid + 8, col1)] =
                    __float2half_rn(p11);
            }
            bl0 = warp_sum<4>(bl0, FullMask);
            bl1 = warp_sum<4>(bl1, FullMask);

            l0 = l0 * alpha0 + bl0;
            l1 = l1 * alpha1 + bl1;
            m0 = nm0;
            m1 = nm1;
            if (lid == 0) {
                alpha_s[row0] = alpha0;
                alpha_s[row1] = alpha1;
            }
        } else {
            const int loader_tid = tid - ProducerThreads;
#pragma unroll 1
            for (int chunk = loader_tid; chunk < Bc * (D / 8); chunk += VLoaderThreads) {
                const int key_l = chunk / (D / 8);
                const int dc    = chunk - key_l * (D / 8);
                const int d     = dc * 8;
                const int key   = k0 + key_l;
                __half* dst     = &v_f16[key_l * D + causal_swizzle(key_l, d)];
                if (key >= split_start && key < split_end) {
                    const int grp = d >> 6;
                    float vs      = 0.0f;
                    if ((lane & 7) == 0) { vs = __half2float(v_scale_s[key_l * Groups + grp]); }
                    vs                = __shfl_sync(FullMask, vs, grp * 8);
                    if constexpr (PackedV) {
                        // rk V dequant: vectorized 4-byte load + fast unpack.
                        // Read 4 packed bytes (8 int4 values) as a single 32-bit load,
                        // then use vectorized nibble extraction + sign extension
                        // (3 ops vs 8 separate rk4_unpack calls = 16+ ops).
                        const std::uint32_t raw = *reinterpret_cast<const std::uint32_t*>(
                            &v_i8[key_l * D + (d >> 1)]);
                        // Vectorized unpack: extract low and high nibbles in parallel
                        const std::uint32_t lo = rk4_unpack_lo_fast(raw);
                        const std::uint32_t hi = rk4_unpack_hi_fast(raw);
                        const auto* lo8 = reinterpret_cast<const std::int8_t*>(&lo);
                        const auto* hi8 = reinterpret_cast<const std::int8_t*>(&hi);
                        // Interleave low/high nibbles and convert to f16 in one pass
                        int4 values;
                        values.x = pack_f16x2(static_cast<float>(lo8[0]) * vs,
                                              static_cast<float>(hi8[0]) * vs);
                        values.y = pack_f16x2(static_cast<float>(lo8[1]) * vs,
                                              static_cast<float>(hi8[1]) * vs);
                        values.z = pack_f16x2(static_cast<float>(lo8[2]) * vs,
                                              static_cast<float>(hi8[2]) * vs);
                        values.w = pack_f16x2(static_cast<float>(lo8[3]) * vs,
                                              static_cast<float>(hi8[3]) * vs);
                        store_vec(dst, values);
                    } else {
                        const int2 raw    = load_vec<int2>(&v_i8[key_l * D + d]);
                        const auto* codes = reinterpret_cast<const std::int8_t*>(&raw);
                        int4 values;
                        values.x = pack_f16x2(static_cast<float>(codes[0]) * vs,
                                              static_cast<float>(codes[1]) * vs);
                        values.y = pack_f16x2(static_cast<float>(codes[2]) * vs,
                                              static_cast<float>(codes[3]) * vs);
                        values.z = pack_f16x2(static_cast<float>(codes[4]) * vs,
                                              static_cast<float>(codes[5]) * vs);
                        values.w = pack_f16x2(static_cast<float>(codes[6]) * vs,
                                              static_cast<float>(codes[7]) * vs);
                        store_vec(dst, values);
                    }
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
        }
        __syncthreads();

        const bool has_next = kb + 1 < key_blocks;
        if (has_next) {
            const int next_k0 = k0 + Bc;
            if ((next_k0 & kPagedKVPageMask) == 0) {
                physical_page = block_table[next_k0 >> kPagedKVPageShift];
            }
            issue_kv_tile(next_k0, physical_page);
        }

        const int consumer_tile     = warp % RowTiles;
        const int consumer_slice    = warp / RowTiles;
        const int consumer_row_base = consumer_tile * 16;
        __half* p_consumer          = &p_s[consumer_row_base * Bc];
        const float alpha0          = alpha_s[consumer_row_base + gid];
        const float alpha1          = alpha_s[consumer_row_base + gid + 8];
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            const int global_n = consumer_slice * PVNtPerWarp + n;
#pragma unroll
            for (int k = 0; k < PVKs; ++k) {
                unsigned pf[4];
                const int pcol = k * 16 + a_coloff;
                ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                            smem_addr(&p_consumer[a_rowoff * Bc +
                                                  causal_probability_swizzle<Bc>(a_rowoff, pcol)]));
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = global_n * 8;
                ldmatrix_x2_t(vf[0], vf[1],
                              smem_addr(&v_f16[vrow * D + causal_swizzle(vrow, vcol)]));
                mma_f16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2], pf[3],
                        vf[0], vf[1]);
            }
        }
        if (has_next) { ninfer::ops::cp_wait<0>(); }
        __syncthreads();
    }
    

    if (warp < RowTiles && lid == 0) {
        const int row0 = warp * 16 + gid;
        const int row1 = row0 + 8;
        if (row0 < tile_tokens * Geometry::GroupSize) {
            int q_head = 0;
            int token  = 0;
            causal_row_to_qt<Geometry>(row0, kv_head, q_head, token);
            partial_m[causal_stat_index<Geometry>(q_head, partial_begin + token, split,
                                                  partial_width)] = m0;
            partial_l[causal_stat_index<Geometry>(q_head, partial_begin + token, split,
                                                  partial_width)] = l0;
        }
        if (row1 < tile_tokens * Geometry::GroupSize) {
            int q_head = 0;
            int token  = 0;
            causal_row_to_qt<Geometry>(row1, kv_head, q_head, token);
            partial_m[causal_stat_index<Geometry>(q_head, partial_begin + token, split,
                                                  partial_width)] = m1;
            partial_l[causal_stat_index<Geometry>(q_head, partial_begin + token, split,
                                                  partial_width)] = l1;
        }
    }

#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
        const int consumer_tile     = warp % RowTiles;
        const int consumer_slice    = warp / RowTiles;
        const int consumer_row_base = consumer_tile * 16;
        const int d0                = (consumer_slice * PVNtPerWarp + n) * 8 + 2 * lid;
        const int row0              = consumer_row_base + gid;
        const int row1              = row0 + 8;
        if (row0 < tile_tokens * Geometry::GroupSize) {
            int q_head = 0;
            int token  = 0;
            causal_row_to_qt<Geometry>(row0, kv_head, q_head, token);
            const std::int64_t dst = causal_partial_index<Geometry>(
                q_head, d0, partial_begin + token, split, partial_width);
            causal_store_partial_pair(&partial_acc[dst], acc[n][0], acc[n][1]);
        }
        if (row1 < tile_tokens * Geometry::GroupSize) {
            int q_head = 0;
            int token  = 0;
            causal_row_to_qt<Geometry>(row1, kv_head, q_head, token);
            const std::int64_t dst = causal_partial_index<Geometry>(
                q_head, d0, partial_begin + token, split, partial_width);
            causal_store_partial_pair(&partial_acc[dst], acc[n][2], acc[n][3]);
        }
    }
}
    } // namespace ninfer::ops::detail
