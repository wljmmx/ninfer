#pragma once

// P6 2c C3: this header can launch the INT8 KV attention that writes the page-score counters, so
// the scoring path is kept to a single translation unit. Only int8/launch.cu defines
// NINFER_INT8_KV_SCORING_OWNER; any new include from another TU fails to compile instead of silently
// adding a second scoring launch point that the counters cannot see (appendices CJ.1/C3).
#ifndef NINFER_INT8_KV_SCORING_OWNER
#error "int8 KV scoring kernels: include via int8/template_launch.cuh, from int8/launch.cu (P6 2c C3)"
#endif

#include "ops/softmax_attention/dense/causal_cache/int8/tile_io.cuh"

#include "ops/kernel/e8_root_codec.cuh"
#include "ops/kv_cache/rk4_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/int8/schedule.cuh"
#include "ops/softmax_attention/common/causal_epilogue.cuh"
#include "ops/softmax_attention/common/causal_softmax.cuh"

namespace ninfer::ops::detail {

// PackedV/PackedK/E8Root select the rk-family cache codecs for the read-back:
// int4 V (8 packed bytes per 16 dims, unpacked by the V-worker stage), int4 K
// (synchronous vectorized unpack into the swizzled K slot) or the 2-bit E8
// cylinder K (synchronous root decode). E8Lattice now DOES affect reads: the
// [EXPERIMENT E8+1bit] carrier stores an E8-cylinder + residual key inside the
// int4 plane, so the read-back must be selected per layout.
template <typename Geometry, typename Schedule, typename Metadata, bool PackedV = false,
          bool PackedK = false, bool E8Root = false, bool V8Root = false,
          bool E8Lattice = false>
__global__ __maxnreg__(Schedule::kMaxRegisters) void int8_kv_tiled_mma_kernel(
    const __nv_bfloat16* __restrict__ q, const std::int8_t* __restrict__ cache_k,
    const std::int8_t* __restrict__ cache_v, const __half* __restrict__ cache_k_scale,
    const __half* __restrict__ cache_v_scale, Metadata metadata,
    const std::int32_t* __restrict__ positions, float scale, __nv_bfloat16* __restrict__ out,
    unsigned* __restrict__ page_score, std::int32_t width) {
    // P6 2c hook site (appendix BL.2/BM.1 step 1.2). One CTA covers exactly one 64-token page
    // here, so the CTA-wide hit reduction feeds counters[k0 >> kPagedKVPageShift]. `page_score` is
    // the caller's already-resolved override; the metadata path is resolved below.
    constexpr int D             = 256;
    constexpr int Br            = Schedule::kQueryRows;
    constexpr int Bc            = Schedule::kKeyRows;
    constexpr int DB16          = 128;
    constexpr int Groups        = 4;
    constexpr int GroupKc       = kKVCacheInt8Group / 32;
    constexpr int QKNt          = Bc / 8;
    constexpr int PVNtPerWarp   = D / (Schedule::kDConsumers * 8);
    constexpr int PVKs          = Bc / 16;
    constexpr int ProducerWarps = Schedule::kRowTiles;
    constexpr int VWorkerWarps  = Schedule::kWarps - ProducerWarps;
    constexpr int WorkerThreads = VWorkerWarps * 32;
    constexpr float Log2E       = kLog2E;
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(GroupKc == 2);
    static_assert(PVNtPerWarp == 8);

    extern __shared__ __align__(16) unsigned char smem_raw[];
    std::int8_t* q_i8 = reinterpret_cast<std::int8_t*>(smem_raw);
    float* q_scale    = reinterpret_cast<float*>(q_i8 + Schedule::kQBytes);
    std::int8_t* k_i8 = reinterpret_cast<std::int8_t*>(reinterpret_cast<unsigned char*>(q_scale) +
                                                       Schedule::kQScaleBytes);
    std::int8_t* v_i8 = k_i8 + Schedule::kKBytes;
    __half* v_f16     = reinterpret_cast<__half*>(v_i8 + Schedule::kVBytes);
    __half* p_s =
        reinterpret_cast<__half*>(reinterpret_cast<unsigned char*>(v_f16) + Schedule::kVStageBytes);
    __half* k_scale_s =
        reinterpret_cast<__half*>(reinterpret_cast<unsigned char*>(p_s) + Schedule::kPBytes);
    __half* v_scale_s    = k_scale_s + Bc * Groups;
    float* alpha_s       = reinterpret_cast<float*>(v_scale_s + Bc * Groups);
    float* final_l_s     = alpha_s + Br;
    __nv_bfloat16* q_b16 = reinterpret_cast<__nv_bfloat16*>(q_i8);
    __nv_bfloat16* k_b16 = reinterpret_cast<__nv_bfloat16*>(k_i8);

    const int q_block = static_cast<int>(blockIdx.x);
    const int q_head  = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int q0      = q_block * Br;
    const int kv_head = q_head / Geometry::GroupSize;
    const int tokens  = metadata.valid_tokens(width);
    if (q_head >= Geometry::QHeads || q0 >= width) { return; }
    if (q0 >= tokens) {
        causal_zero_rows<Geometry>(out, q_head, q0, min(q0 + Br, width), tid, Schedule::kThreads);
        return;
    }
    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();
    // The relevance-counter row of this CTA, resolved exactly like block_table (appendix CA.2/CA.3).
    unsigned* const counters = page_score != nullptr ? page_score : metadata.counters();

    const int tile_rows     = min(Br, tokens - q0);
    const int max_query_abs = base_pos + q0 + tile_rows - 1;
    const int key_blocks    = max_query_abs / Bc + 1;

    // Quantize Q cooperatively. One full warp rotates and encodes one D256 row at a time.
    for (int row = warp; row < Br; row += Schedule::kWarps) {
        float q_values[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            q_values[r] = 0.0f;
            if (row < tile_rows) {
                q_values[r] = __bfloat162float(q[causal_q_index<Geometry>(q_head, d, q0 + row)]);
            }
        }
        normalized_hadamard_d256_inplace(q_values, lane);

#pragma unroll
        for (int grp = 0; grp < Groups; ++grp) {
            const int d0    = grp * kKVCacheInt8Group + lane;
            const int d1    = d0 + 32;
            const float x0  = q_values[2 * grp];
            const float x1  = q_values[2 * grp + 1];
            float absmax    = fmaxf(fabsf(x0), fabsf(x1));
            absmax          = warp_max(absmax, FullMask);
            const float qs  = absmax > 0.0f ? absmax / 127.0f : 0.0f;
            const float inv = qs > 0.0f ? 1.0f / qs : 0.0f;
            causal_store_query_code(q_i8, row, d0, kv_cache_int8_quant_code(x0, inv));
            causal_store_query_code(q_i8, row, d1, kv_cache_int8_quant_code(x1, inv));
            if (lane == 0) { q_scale[row * Groups + grp] = qs; }
        }
    }
    __syncthreads();

    auto issue_kv_tile = [&](int tile_k0) {
        // An absent page is addressed as page 0 (kept in bounds) and skipped by the score mask.
        const bool page_present = paged_kv_page_present(block_table, tile_k0);
        const int physical_page = paged_kv_page_for_address(block_table, tile_k0);
        for (int key_l = tid; key_l < Bc; key_l += Schedule::kThreads) {
            const int key = tile_k0 + key_l;
            __half* kd    = &k_scale_s[key_l * Groups];
            __half* vd    = &v_scale_s[key_l * Groups];
            if (key <= max_query_abs && page_present) {
                const std::int64_t off = kv_cache_int8_quant_scale_index<Geometry>(
                    physical_page, kv_head, 0, key & kPagedKVPageMask);
                ninfer::ops::cp_async<8>(kd, &cache_k_scale[off]);
                ninfer::ops::cp_async<8>(vd, &cache_v_scale[off]);
            } else {
                store_vec(kd, make_int2(0, 0));
                store_vec(vd, make_int2(0, 0));
            }
        }
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (D / 16); chunk += Schedule::kThreads) {
            const int key_l = chunk / (D / 16);
            const int dc    = chunk - key_l * (D / 16);
            const int d     = dc * 16;
            const int key   = tile_k0 + key_l;
            std::int8_t* kd = &k_i8[(key_l * DB16 + causal_swizzle(key_l, dc * 8)) * 2];
            // int4 V occupies half the smem extent; the V-worker pass unpacks it
            // into v_f16 after the barrier.
            std::int8_t* vd = &v_i8[key_l * D + (PackedV ? (d >> 1) : d)];
            if (key <= max_query_abs && page_present) {
                if constexpr (E8Root) {
                    // 2-bit cylinder K: 4 cache bytes cover this 16-dim chunk.
                    // Synchronous load + root decode straight into the swizzled
                    // slot — the tiny payload cannot be staged by cp_async
                    // without extra shared memory.
                    const int byte_offset = ((d / 64) * 8 + ((d / 8) & 7)) * 2;
                    const std::int64_t koff = paged_kv_element_offset<64, Geometry::KVHeads>(
                        physical_page, kv_head, key & kPagedKVPageMask, byte_offset);
                    const std::uint32_t raw =
                        load_ldg<std::uint32_t>(&cache_k[koff]);
                    const uint8_t root0 = static_cast<uint8_t>(raw);
                    const uint8_t rad0  = static_cast<uint8_t>(raw >> 8);
                    const uint8_t root1 = static_cast<uint8_t>(raw >> 16);
                    const uint8_t rad1  = static_cast<uint8_t>(raw >> 24);
                    __align__(8) int8_t dec0[8];
                    __align__(8) int8_t dec1[8];
                    e8_root_decode_8d_fast(root0, rad0, dec0);
                    e8_root_decode_8d_fast(root1, rad1, dec1);
                    *reinterpret_cast<uint64_t*>(kd) =
                        *reinterpret_cast<const uint64_t*>(dec0);
                    *reinterpret_cast<uint64_t*>(kd + 8) =
                        *reinterpret_cast<const uint64_t*>(dec1);
                } else if constexpr (PackedK && E8Lattice && kRkE8KAmplitudeOn) {
                    // [EXPERIMENT E8+1bit] carrier read-back: E8-cylinder decode on the
                    // 128-byte int4 plane plus the per-element +-1 residual signs.
                    // Register-lean on purpose (this kernel runs at a hard maxreg cap).
                    const int subidx = (d / 64) * 8 + ((d / 8) & 7);
                    const std::int64_t koff = paged_kv_element_offset<kRkE8KLatticeExtent, Geometry::KVHeads>(
                        physical_page, kv_head, key & kPagedKVPageMask, subidx * 2);
                    const std::uint32_t raw =
                        load_ldg<std::uint32_t>(&cache_k[koff]);
                    const std::uint16_t signs = *reinterpret_cast<const std::uint16_t*>(
                        &cache_k[koff + 64 - subidx]);
                    std::uint8_t th0 = 0, th1 = 0;
                    if constexpr (kRkE8KAmplitudeOn) {
                        th0 = static_cast<std::uint8_t>(cache_k[koff + 96 - subidx]);
                        th1 = static_cast<std::uint8_t>(cache_k[koff + 96 - subidx + 1]);
                    }
                    __align__(8) int8_t dec[8];
                    e8_root_decode_8d_fast(static_cast<uint8_t>(raw),
                                          static_cast<uint8_t>(raw >> 8), dec);
                    if constexpr (kRkE8KAmplitudeOn) {
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            int v = kRkE8KAmplitudeGrid * static_cast<int>(dec[i]) +
                                    (((signs >> i) & 1u) ? static_cast<int>(th0)
                                                         : -static_cast<int>(th0));
                            dec[i] = static_cast<std::int8_t>(v < -127 ? -127 : (v > 127 ? 127 : v));
                        }
                    } else if constexpr (kRkE8KResidualOn) {
rk4_apply_e8_residual(dec, static_cast<std::uint32_t>(signs));
                    }
                    *reinterpret_cast<uint64_t*>(kd) = *reinterpret_cast<const uint64_t*>(dec);
                    e8_root_decode_8d_fast(static_cast<uint8_t>(raw >> 16),
                                          static_cast<uint8_t>(raw >> 24), dec);
                    if constexpr (kRkE8KAmplitudeOn) {
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            int v = kRkE8KAmplitudeGrid * static_cast<int>(dec[i]) +
                                    (((signs >> (8 + i)) & 1u) ? static_cast<int>(th1)
                                                               : -static_cast<int>(th1));
                            dec[i] = static_cast<std::int8_t>(v < -127 ? -127 : (v > 127 ? 127 : v));
                        }
                    } else if constexpr (kRkE8KResidualOn) {
rk4_apply_e8_residual(dec, static_cast<std::uint32_t>(signs >> 8));
                    }
                    *reinterpret_cast<uint64_t*>(kd + 8) =
                        *reinterpret_cast<const uint64_t*>(dec);
                } else if constexpr (PackedK && E8Lattice) {
                    // [D5] cp.async stage (see grouped_mma.cuh): the 4-byte E8 code
                    // and a 4-byte-aligned sign window; decode_e8_tile() finishes them.
                    const int subidx = (d / 64) * 8 + ((d / 8) & 7);
                    const std::int64_t koff = paged_kv_element_offset<kRkE8KLatticeExtent, Geometry::KVHeads>(
                        physical_page, kv_head, key & kPagedKVPageMask, subidx * 2);
                    cp_async<4>(kd, &cache_k[koff]);
                    cp_async<4>(kd + 4, &cache_k[koff + 64 - subidx - (subidx & 3)]);
                } else if constexpr (PackedK) {
                    // int4 K: 8 packed bytes cover this 16-dim chunk; synchronous
                    // load + vectorized nibble unpack into the swizzled slot
                    // (the same proven path as the grouped kernel).
                    const std::int64_t koff = rk4_v_code_index<Geometry>(
                        physical_page, kv_head, d >> 1, key & kPagedKVPageMask);
#pragma unroll
                    for (int half4 = 0; half4 < 2; ++half4) {
                        const std::uint32_t raw = *reinterpret_cast<const std::uint32_t*>(
                            &cache_k[koff + 4 * half4]);
                        const std::uint32_t lo = rk4_unpack_lo_fast(raw);
                        const std::uint32_t hi = rk4_unpack_hi_fast(raw);
                        const auto* lo8        = reinterpret_cast<const std::int8_t*>(&lo);
                        const auto* hi8        = reinterpret_cast<const std::int8_t*>(&hi);
#pragma unroll
                        for (int b = 0; b < 4; ++b) {
                            kd[8 * half4 + 2 * b]     = lo8[b];
                            kd[8 * half4 + 2 * b + 1] = hi8[b];
                        }
                    }
                } else {
                    const std::int64_t off = kv_cache_int8_quant_code_index<Geometry>(
                        physical_page, kv_head, d, key & kPagedKVPageMask);
                    cp_async<16, Cache::cg>(kd, &cache_k[off]);
                }
                if constexpr (V8Root) {
                    // EXPERIMENT rk4v2-e8 V read-back: 4 cache bytes cover this 16-dim
                    // chunk (two 8D subspaces); synchronous root decode into vd.
                    const int byte_offset = ((d / 64) * 8 + ((d / 8) & 7)) * 2;
                    const std::int64_t voff = paged_kv_element_offset<64, Geometry::KVHeads>(
                        physical_page, kv_head, key & kPagedKVPageMask, byte_offset);
                    const std::uint32_t raw =
                        *reinterpret_cast<const std::uint32_t*>(&cache_v[voff]);
                    const uint8_t vr0  = static_cast<uint8_t>(raw);
                    const uint8_t vra0 = static_cast<uint8_t>(raw >> 8);
                    const uint8_t vr1  = static_cast<uint8_t>(raw >> 16);
                    const uint8_t vra1 = static_cast<uint8_t>(raw >> 24);
                    __align__(8) int8_t vdec0[8];
                    __align__(8) int8_t vdec1[8];
                    // V-plane E8-cylinder read-back uses the calibrated radius centre.
                    e8_root_decode_8d_fast_v(vr0, vra0, vdec0);
                    e8_root_decode_8d_fast_v(vr1, vra1, vdec1);
                    *reinterpret_cast<uint64_t*>(vd) = *reinterpret_cast<const uint64_t*>(vdec0);
                    *reinterpret_cast<uint64_t*>(vd + 8) = *reinterpret_cast<const uint64_t*>(vdec1);
                } else if constexpr (PackedV) {
                    const std::int64_t voff = rk4_v_code_index<Geometry>(
                        physical_page, kv_head, d >> 1, key & kPagedKVPageMask);
                    cp_async<8>(vd, &cache_v[voff]);
                } else {
                    const std::int64_t voff = kv_cache_int8_quant_code_index<Geometry>(
                        physical_page, kv_head, d, key & kPagedKVPageMask);
                    cp_async<16, Cache::cg>(vd, &cache_v[voff]);
                }
            } else {
                store_vec(kd, make_int4(0, 0, 0, 0));
                rk4_zero_packed_v<PackedV>(&v_i8[key_l * D], d);
            }
        }
        ninfer::ops::cp_commit();
    };

    // [D5] E8Lattice read-back decode phase (mirror of grouped_mma.cuh): the raw
    // 4-byte code and 4-byte-aligned sign window are staged by issue_kv_tile(), and
    // the 16 int8 codes are built here after cp_wait<0>() retires the async copy.
    auto decode_e8_chunk = [&](int tile_k0, int chunk) {
        {
            const int key_l = chunk / (D / 16);
            const int dc    = chunk - key_l * (D / 16);
            const int d     = dc * 16;
            const int key   = tile_k0 + key_l;
            if (key <= max_query_abs && paged_kv_page_present(block_table, tile_k0)) {
                std::int8_t* kd = &k_i8[(key_l * DB16 + causal_swizzle(key_l, dc * 8)) * 2];
                const int subidx = (d / 64) * 8 + ((d / 8) & 7);
                const std::uint32_t raw = *reinterpret_cast<const std::uint32_t*>(kd);
                const std::uint16_t signs = *reinterpret_cast<const std::uint16_t*>(
                    kd + 4 + (subidx & 3));
                __align__(8) int8_t dec[8];
                e8_root_decode_8d_fast(static_cast<uint8_t>(raw),
                                       static_cast<uint8_t>(raw >> 8), dec);
                if constexpr (kRkE8KResidualOn) {
rk4_apply_e8_residual(dec, static_cast<std::uint32_t>(signs));
                }
                *reinterpret_cast<uint64_t*>(kd) = *reinterpret_cast<const uint64_t*>(dec);
                e8_root_decode_8d_fast(static_cast<uint8_t>(raw >> 16),
                                       static_cast<uint8_t>(raw >> 24), dec);
                if constexpr (kRkE8KResidualOn) {
rk4_apply_e8_residual(dec, static_cast<std::uint32_t>(signs >> 8));
                }
                *reinterpret_cast<uint64_t*>(kd + 8) = *reinterpret_cast<const uint64_t*>(dec);
            }
        }
    };
    auto decode_e8_tile = [&](int tile_k0) {
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (D / 16); chunk += Schedule::kThreads) {
            decode_e8_chunk(tile_k0, chunk);
        }
    };

    issue_kv_tile(0);
    ninfer::ops::cp_wait<0>();
    if constexpr (PackedK && E8Lattice && !kRkE8KAmplitudeOn) { decode_e8_tile(0); }
    __syncthreads();

    const int gid      = lane >> 2;
    const int lid      = lane & 3;
    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    // Keep two group scales live; reload the other groups from the shared Q tile.
    float q_scale_r0[Groups - 2];
    float q_scale_r1[Groups - 2];
    if (warp < ProducerWarps) {
        const int scale_row0 = warp * 16 + gid;
        const int scale_row1 = scale_row0 + 8;
#pragma unroll
        for (int grp = 0; grp < Groups - 2; ++grp) {
            float qs0       = lid == 0 ? q_scale[scale_row0 * Groups + grp] : 0.0f;
            float qs1       = lid == 0 ? q_scale[scale_row1 * Groups + grp] : 0.0f;
            q_scale_r0[grp] = __shfl_sync(FullMask, qs0, gid * 4);
            q_scale_r1[grp] = __shfl_sync(FullMask, qs1, gid * 4);
        }
    }

    float acc[PVNtPerWarp][4];
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float running_m0     = -CUDART_INF_F;
    float running_m1     = -CUDART_INF_F;
    float running_l0     = 0.0f;
    float running_l1     = 0.0f;
    const float scale_l2 = scale * Log2E;
    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = kb * Bc;
        if (warp < ProducerWarps) {
            const int row_base = warp * 16;
            float score[QKNt][4];
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
            }

#pragma unroll
            for (int grp = 0; grp < Groups; ++grp) {
                float qs0;
                float qs1;
                if (grp < Groups - 2) {
                    qs0 = q_scale_r0[grp];
                    qs1 = q_scale_r1[grp];
                } else {
                    const int scale_row0 = row_base + gid;
                    const int scale_row1 = scale_row0 + 8;
                    qs0                  = lid == 0 ? q_scale[scale_row0 * Groups + grp] : 0.0f;
                    qs1                  = lid == 0 ? q_scale[scale_row1 * Groups + grp] : 0.0f;
                    qs0                  = __shfl_sync(FullMask, qs0, gid * 4);
                    qs1                  = __shfl_sync(FullMask, qs1, gid * 4);
                }

                unsigned af[GroupKc][4];
#pragma unroll
                for (int kk = 0; kk < GroupKc; ++kk) {
                    const int k    = grp * GroupKc + kk;
                    const int acol = k * 16 + a_coloff;
                    ldmatrix_x4(af[kk][0], af[kk][1], af[kk][2], af[kk][3],
                                smem_addr(&q_b16[(row_base + a_rowoff) * DB16 +
                                                 causal_swizzle(row_base + a_rowoff, acol)]));
                }

#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#pragma unroll
                    for (int kk = 0; kk < GroupKc; ++kk) {
                        const int k    = grp * GroupKc + kk;
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
                    float ks0      = 0.0f;
                    float ks1      = 0.0f;
                    if (gid == 0) {
                        ks0 = __half2float(k_scale_s[keya * Groups + grp]);
                        ks1 = __half2float(k_scale_s[keyb * Groups + grp]);
                    }
                    ks0          = __shfl_sync(FullMask, ks0, lid);
                    ks1          = __shfl_sync(FullMask, ks1, lid);
                    score[nt][0] = __fmaf_rn(qs0 * ks0, static_cast<float>(c0), score[nt][0]);
                    score[nt][1] = __fmaf_rn(qs0 * ks1, static_cast<float>(c1), score[nt][1]);
                    score[nt][2] = __fmaf_rn(qs1 * ks0, static_cast<float>(c2), score[nt][2]);
                    score[nt][3] = __fmaf_rn(qs1 * ks1, static_cast<float>(c3), score[nt][3]);
                }
            }

            const int row0             = row_base + gid;
            const int row1             = row0 + 8;
            const int qabs0            = row0 < tile_rows ? base_pos + q0 + row0 : -1;
            const int qabs1            = row1 < tile_rows ? base_pos + q0 + row1 : -1;
            // Bc == one 64-token page, so presence is uniform across this tile.
            const bool page_present =
                paged_kv_page_present(block_table, k0);
            const bool full_score_tile = page_present && q0 + Br <= tokens &&
                                         k0 + Bc - 1 <= base_pos + q0;
            float bm0                  = -CUDART_INF_F;
            float bm1                  = -CUDART_INF_F;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int key0 = k0 + nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                if (!full_score_tile) {
                    const bool vis0 = page_present && key0 <= qabs0;
                    const bool vis1 = page_present && key1 <= qabs0;
                    const bool vis2 = page_present && key0 <= qabs1;
                    const bool vis3 = page_present && key1 <= qabs1;
                    score[nt][0] = vis0 ? score[nt][0] : -CUDART_INF_F;
                    score[nt][1] = vis1 ? score[nt][1] : -CUDART_INF_F;
                    score[nt][2] = vis2 ? score[nt][2] : -CUDART_INF_F;
                    score[nt][3] = vis3 ? score[nt][3] : -CUDART_INF_F;
                }
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
            bm0 = warp_max<4>(bm0, FullMask);
            bm1 = warp_max<4>(bm1, FullMask);

            // P6 2c score hook, criterion (i) "record page" (appendices CC.5/CO.4). This path keeps
            // the running-best hit count because Tiled is the long-prefill shape and has no key
            // partitioning at all - one CTA sweeps every key block - so the count is already
            // deterministic here, and it is the statistic that actually carries the query signal: a
            // page's own peak logit saturates on whichever stream is most extreme and collapses the
            // profiles together (measured in CO.3). The counters are cleared per call (see
            // sequence.cpp), so nothing accumulates across calls.
            // A row's first page is not evidence - running_m starts at -inf and would record
            // unconditionally (appendix CD.4). bm0/bm1 are already segment-reduced, so exactly one
            // lane per segment reports, and nothing reads the counters yet.
            if (counters != nullptr && page_present && lid == 0 &&
                running_m0 != -CUDART_INF_F && running_m1 != -CUDART_INF_F) {
                const int records = (bm0 > running_m0 ? 1 : 0) + (bm1 > running_m1 ? 1 : 0);
                if (records > 0) {
                    atomicAdd(counters + (k0 >> kPagedKVPageShift),
                              static_cast<unsigned>(records));
                }
            }

            const float nm0        = fmaxf(running_m0, bm0);
            const float nm1        = fmaxf(running_m1, bm1);
            const float nm0_scaled = nm0 * scale_l2;
            const float nm1_scaled = nm1 * scale_l2;
            const float alpha0     = running_m0 == -CUDART_INF_F
                                         ? 0.0f
                                         : causal_exp_scaled(running_m0, nm0_scaled, scale_l2);
            const float alpha1     = running_m1 == -CUDART_INF_F
                                         ? 0.0f
                                         : causal_exp_scaled(running_m1, nm1_scaled, scale_l2);
            float bl0              = 0.0f;
            float bl1              = 0.0f;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int col0  = nt * 8 + 2 * lid;
                const int col1  = col0 + 1;
                const float p00 = score[nt][0] > -CUDART_INF_F
                                      ? causal_exp_scaled(score[nt][0], nm0_scaled, scale_l2)
                                      : 0.0f;
                const float p01 = score[nt][1] > -CUDART_INF_F
                                      ? causal_exp_scaled(score[nt][1], nm0_scaled, scale_l2)
                                      : 0.0f;
                const float p10 = score[nt][2] > -CUDART_INF_F
                                      ? causal_exp_scaled(score[nt][2], nm1_scaled, scale_l2)
                                      : 0.0f;
                const float p11 = score[nt][3] > -CUDART_INF_F
                                      ? causal_exp_scaled(score[nt][3], nm1_scaled, scale_l2)
                                      : 0.0f;
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                p_s[row0 * Bc + causal_probability_swizzle<Bc>(row0, col0)] = __float2half_rn(p00);
                p_s[row0 * Bc + causal_probability_swizzle<Bc>(row0, col1)] = __float2half_rn(p01);
                p_s[row1 * Bc + causal_probability_swizzle<Bc>(row1, col0)] = __float2half_rn(p10);
                p_s[row1 * Bc + causal_probability_swizzle<Bc>(row1, col1)] = __float2half_rn(p11);
            }
            bl0        = warp_sum<4>(bl0, FullMask);
            bl1        = warp_sum<4>(bl1, FullMask);
            running_l0 = __fmaf_rn(running_l0, alpha0, bl0);
            running_l1 = __fmaf_rn(running_l1, alpha1, bl1);
            running_m0 = nm0;
            running_m1 = nm1;
            if (lid == 0) {
                alpha_s[row0] = alpha0;
                alpha_s[row1] = alpha1;
            }
        } else if (warp < ProducerWarps + VWorkerWarps) {
            const int worker_tid = tid - ProducerWarps * 32;
#pragma unroll 1
            for (int chunk = worker_tid; chunk < Bc * (D / 8); chunk += WorkerThreads) {
                const int key_l = chunk / (D / 8);
                const int dc    = chunk - key_l * (D / 8);
                const int d     = dc * 8;
                const int key   = k0 + key_l;
                __half* dst     = &v_f16[key_l * D + causal_swizzle(key_l, d)];
                if (key <= max_query_abs && paged_kv_page_present(block_table, k0)) {
                    const int grp = d >> 6;
                    __half vs     = __float2half_rn(0.0f);
                    if ((lane & 7) == 0) { vs = v_scale_s[key_l * Groups + grp]; }
                    vs = __shfl_sync(FullMask, vs, grp * 8);
                    if constexpr (PackedV) {
                        // int4 V dequant: 4 packed bytes (8 dims) at half extent,
                        // vectorized nibble unpack + scale, straight into the
                        // swizzled v_f16 slot (same as the grouped kernel).
                        const std::uint32_t raw = *reinterpret_cast<const std::uint32_t*>(
                            &v_i8[key_l * D + (d >> 1)]);
                        const std::uint32_t lo = rk4_unpack_lo_fast(raw);
                        const std::uint32_t hi = rk4_unpack_hi_fast(raw);
                        const auto* lo8        = reinterpret_cast<const std::int8_t*>(&lo);
                        const auto* hi8        = reinterpret_cast<const std::int8_t*>(&hi);
                        const float vsf         = __half2float(vs);
                        int4 values;
                        values.x = pack_f16x2(static_cast<float>(lo8[0]) * vsf,
                                              static_cast<float>(hi8[0]) * vsf);
                        values.y = pack_f16x2(static_cast<float>(lo8[1]) * vsf,
                                              static_cast<float>(hi8[1]) * vsf);
                        values.z = pack_f16x2(static_cast<float>(lo8[2]) * vsf,
                                              static_cast<float>(hi8[2]) * vsf);
                        values.w = pack_f16x2(static_cast<float>(lo8[3]) * vsf,
                                              static_cast<float>(hi8[3]) * vsf);
                        store_vec(dst, values);
                    } else {
                        store_vec(dst, int8_kv_dequant_f16x8(&v_i8[key_l * D + d], vs));
                    }
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
        }
        __syncthreads();

        const bool has_next = kb + 1 < key_blocks;
        if (has_next) { issue_kv_tile((kb + 1) * Bc); }

        // Retire the next tile's async copies before PV so the decode slices below can
        // read them. The transfer had the whole QK phase to land.
        if (has_next) { ninfer::ops::cp_wait<0>(); }

        const int row_tile = warp % Schedule::kRowTiles;
        const int d_slice  = warp / Schedule::kRowTiles;
        const int row_base = row_tile * 16;
        const float alpha0 = alpha_s[row_base + gid];
        const float alpha1 = alpha_s[row_base + gid + 8];
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

#pragma unroll
        for (int k = 0; k < PVKs; ++k) {
            unsigned pf[4];
            const int pcol = k * 16 + a_coloff;
            ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                        smem_addr(&p_s[(row_base + a_rowoff) * Bc +
                                       causal_probability_swizzle<Bc>(row_base + a_rowoff, pcol)]));
#pragma unroll
            for (int n = 0; n < PVNtPerWarp; ++n) {
                const int global_n = d_slice * PVNtPerWarp + n;
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = global_n * 8;
                ldmatrix_x2_t(vf[0], vf[1],
                              smem_addr(&v_f16[vrow * D + causal_swizzle(vrow, vcol)]));
                mma_f16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2], pf[3],
                        vf[0], vf[1]);
            }
            // One slice of the next tile's E8 decode rides along with this PV stage. The
            // decode is latency-bound (unroll 1, no other work to fill it); issuing it
            // between mma groups lets the tensor-core work cover its stalls.
            if constexpr (PackedK && E8Lattice && !kRkE8KAmplitudeOn) {
                if (has_next) {
                    const int dchunk = tid + k * Schedule::kThreads;
                    if (dchunk < Bc * (D / 16)) { decode_e8_chunk((kb + 1) * Bc, dchunk); }
                }
            }
        }
        __syncthreads();
    }

    if (warp < ProducerWarps && lid == 0) {
        const int row0  = warp * 16 + gid;
        const int row1  = row0 + 8;
        final_l_s[row0] = running_l0;
        final_l_s[row1] = running_l1;
    }
    __syncthreads();

    const int row_tile = warp % Schedule::kRowTiles;
    const int d_slice  = warp / Schedule::kRowTiles;
    const int row_base = row_tile * 16;
    const int row0     = row_base + gid;
    const int row1     = row0 + 8;
    const float inv_l0 = final_l_s[row0] > 0.0f ? __frcp_rn(final_l_s[row0]) : 0.0f;
    const float inv_l1 = final_l_s[row1] > 0.0f ? __frcp_rn(final_l_s[row1]) : 0.0f;
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
        const int d0 = (d_slice * PVNtPerWarp + n) * 8 + 2 * lid;
        if (row0 < tile_rows) {
            causal_store_output_pair<Geometry>(out, q_head, d0, q0 + row0, acc[n][0] * inv_l0,
                                               acc[n][1] * inv_l0);
        }
        if (row1 < tile_rows) {
            causal_store_output_pair<Geometry>(out, q_head, d0, q0 + row1, acc[n][2] * inv_l1,
                                               acc[n][3] * inv_l1);
        }
    }
    causal_zero_rows<Geometry>(out, q_head, tokens, min(q0 + Br, width), tid, Schedule::kThreads);
}

} // namespace ninfer::ops::detail
