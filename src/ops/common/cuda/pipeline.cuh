#pragma once

#include "ops/common/cuda/memory.cuh"

#include <cuda.h>
#include <cuda_pipeline.h>

#include <cstdint>

namespace ninfer::ops::common {

// The cp.async global->shared copy instructions are an sm_80+ (Ampere) feature. Turing
// (sm_75, the rtx2080ti target) has no hardware async copy, so on sm_75 the same
// interface is provided over the __pipeline_* intrinsics: __pipeline_memcpy_async lowers
// to a plain ld.global/st.shared on every architecture, and commit/wait keep the
// identical group semantics the cp.async forms expose. Callers and pipeline depth are
// unchanged; the only difference is that the copy runs through the thread's load/store
// path instead of the async copy unit.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "cp_async supports 4, 8, or 16 bytes");
    __pipeline_memcpy_async(smem_dst, gmem_src, Bytes);
}

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async_zfill(void* smem_dst, const void* gmem_src,
                                               int src_bytes) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16,
                  "cp_async_zfill supports 4, 8, or 16 bytes");
    if (src_bytes <= 0) {
        // zfill semantics: a zero-length source leaves the whole destination zeroed.
        if constexpr (Bytes == 16) {
            *reinterpret_cast<unsigned long long*>(smem_dst) = 0ull;
            *reinterpret_cast<unsigned long long*>(
                static_cast<char*>(smem_dst) + 8) = 0ull;
        } else if constexpr (Bytes == 8) {
            *reinterpret_cast<unsigned long long*>(smem_dst) = 0ull;
        } else {
            *reinterpret_cast<unsigned int*>(smem_dst) = 0u;
        }
        return;
    }
    if (src_bytes < Bytes) {
        // Zero the tail the short copy leaves untouched, matching cp.async zfill.
        char* dst = static_cast<char*>(smem_dst);
        for (int i = src_bytes; i < Bytes; ++i) { dst[i] = 0; }
        if (src_bytes > Bytes) { src_bytes = Bytes; }
    }
    __pipeline_memcpy_async(smem_dst, gmem_src, src_bytes);
}

__device__ __forceinline__ void cp_commit() { __pipeline_commit(); }

template <int Groups>
__device__ __forceinline__ void cp_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "cp_wait group count must fit the PTX immediate");
    __pipeline_wait_prior(Groups);
}
#else  // __CUDA_ARCH__ == 750

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "cp_async supports 4, 8, or 16 bytes");
    if constexpr (Policy == Cache::cg) {
        static_assert(Bytes == 16, "cp.async.cg requires a 16-byte copy");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "n"(Bytes));
    }
}

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async_zfill(void* smem_dst, const void* gmem_src,
                                               int src_bytes) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16,
                  "cp_async_zfill supports 4, 8, or 16 bytes");
    if constexpr (Policy == Cache::cg) {
        static_assert(Bytes == 16, "cp.async.cg requires a 16-byte copy");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "r"(src_bytes));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "n"(Bytes), "r"(src_bytes));
    }
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }

template <int Groups>
__device__ __forceinline__ void cp_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "cp_wait group count must fit the PTX immediate");
    asm volatile("cp.async.wait_group %0;\n" : : "n"(Groups));
}

#endif  // __CUDA_ARCH__ == 750

__device__ __forceinline__ void cta_mbarrier_init(std::uint64_t* barrier, std::uint32_t arrivals) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :
                 : "r"(smem_addr(barrier)), "r"(arrivals)
                 : "memory");
}

__device__ __forceinline__ void cta_mbarrier_wait(std::uint64_t* barrier, std::uint32_t phase) {
    constexpr std::uint32_t kSuspendTicks = 0x989680;
    asm volatile("{\n"
                 ".reg .pred done;\n"
                 "wait_loop_%=: \n"
                 "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1, %2;\n"
                 "@done bra wait_done_%=;\n"
                 "bra wait_loop_%=;\n"
                 "wait_done_%=: \n"
                 "}\n"
                 :
                 : "r"(smem_addr(barrier)), "r"(phase), "r"(kSuspendTicks)
                 : "memory");
}

__device__ __forceinline__ void cta_mbarrier_arrive(std::uint64_t* barrier) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" : : "r"(smem_addr(barrier)) : "memory");
}

__device__ __forceinline__ void cta_mbarrier_arrive_expect_tx(std::uint64_t* barrier,
                                                              std::uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :
                 : "r"(smem_addr(barrier)), "r"(bytes)
                 : "memory");
}

__device__ __forceinline__ void cta_mbarrier_fence_init() {
    asm volatile("fence.mbarrier_init.release.cluster;" : : : "memory");
}

// The caller owns tensor-map geometry, shared layout, and mbarrier transaction accounting.
__device__ __forceinline__ void tma_load_2d(void* destination, const CUtensorMap* map, int c0,
                                            int c1, std::uint64_t* barrier) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3}], [%4];"
                 :
                 : "r"(smem_addr(destination)), "l"(map), "r"(c0), "r"(c1), "r"(smem_addr(barrier))
                 : "memory");
}

__device__ __forceinline__ void tma_load_3d(void* destination, const CUtensorMap* map, int c0,
                                            int c1, int c2, std::uint64_t* barrier) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3, %4}], [%5];"
                 :
                 : "r"(smem_addr(destination)), "l"(map), "r"(c0), "r"(c1), "r"(c2),
                   "r"(smem_addr(barrier))
                 : "memory");
}

__device__ __forceinline__ void tma_load_4d(void* destination, const CUtensorMap* map, int c0,
                                            int c1, int c2, int c3, std::uint64_t* barrier) {
    asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3, %4, %5}], [%6];"
                 :
                 : "r"(smem_addr(destination)), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(c3),
                   "r"(smem_addr(barrier))
                 : "memory");
}

template <int Bytes>
__device__ __forceinline__ void pipe_copy(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "pipe_copy supports 4, 8, or 16 bytes");
    __pipeline_memcpy_async(smem_dst, gmem_src, Bytes);
}

__device__ __forceinline__ void pipe_commit() { __pipeline_commit(); }

template <int Groups>
__device__ __forceinline__ void pipe_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "pipe_wait group count must fit the PTX immediate");
    __pipeline_wait_prior(Groups);
}

} // namespace ninfer::ops::common

// Callers that predate the family split resolve these names in ninfer::ops.
namespace ninfer::ops {
using common::cp_async;
using common::cp_async_zfill;
using common::cp_commit;
using common::cp_wait;
using common::pipe_commit;
using common::pipe_copy;
using common::pipe_wait;
} // namespace ninfer::ops
