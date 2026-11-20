#pragma once

#include "ops/common/cuda/memory.cuh"
#include "ops/common/cuda/pipeline.cuh"

#include <cuda_fp16.h>

namespace ninfer::ops::common {

__device__ __forceinline__ void ldmatrix_x2(unsigned& r0, unsigned& r1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                            unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_t(unsigned& r0, unsigned& r1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_t(unsigned& r0, unsigned& r1, unsigned& r2,
                                              unsigned& r3, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

// Convert one packed bf16 pair (a 32-bit register holding two bfloat16 values) to
// the same-packed fp16 pair. The value is exact as long as it lies in the fp16
// range; |x| > 65504 saturates to +/-inf and |x| < 2^-14 rounds to the fp16
// subnormal range, the same __float2half_rn rounding the hardware cvt would use.
// Turing (sm_75) has no bf16 tensor cores, so the rtx2080ti target runs the bf16
// MMA plane through this conversion on the fp16 mma below.
__device__ __forceinline__ unsigned cvt_bf16_to_fp16_bits(unsigned short bf16_bits) {
    // bf16 -> f32 -> fp16 through the same rounding the hardware cvt uses.
    const unsigned float_bits = static_cast<unsigned>(bf16_bits) << 16;
    const float as_float = __uint_as_float(float_bits);
    return static_cast<unsigned>(__half_as_ushort(__float2half_rn(as_float)));
}

__device__ __forceinline__ unsigned cvt_bf16x2_to_fp16x2(unsigned packed_bf16x2) {
    const unsigned lo = cvt_bf16_to_fp16_bits(static_cast<unsigned short>(packed_bf16x2 & 0xffffu));
    const unsigned hi = cvt_bf16_to_fp16_bits(static_cast<unsigned short>(packed_bf16x2 >> 16));
    return lo | (hi << 16);
}

// Turing (sm_75) FP16 tensor cores are m16n8k8 (the m16n8k16 FP16 shape is sm_80+).
// A m16n8k16 fragment splits cleanly into two m16n8k8 fragments along K: the A-block
// a0=[rows 0-7,k 0-7], a1=[rows 8-15,k 0-7] form the first k8 (with B={b0}), and
// a2=[rows 0-7,k 8-15], a3=[rows 8-15,k 8-15] form the second k8 (with B={b1}). The
// m16n8k8 A-fragment is 2 registers and B-fragment 1 register; the C/D-fragment is
// the same {c0..c3} for both, so the two k8 mmas accumulate into the same D. No
// operand regrouping is needed: the m16n8k16 registers are passed straight through.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
__device__ __forceinline__ void mma_f16_k8(float& c0, float& c1, float& c2, float& c3,
                                           unsigned a0, unsigned a1, unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(b0));
}

__device__ __forceinline__ void mma_f16_k16(float& c0, float& c1, float& c2, float& c3,
                                            unsigned a0, unsigned a1, unsigned a2,
                                            unsigned a3, unsigned b0, unsigned b1) {
    mma_f16_k8(c0, c1, c2, c3, a0, a1, b0); // k 0..7
    mma_f16_k8(c0, c1, c2, c3, a2, a3, b1); // k 8..15
}

__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3,
                                         unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                         unsigned b0, unsigned b1) {
    // Turing has no bf16 tensor cores: convert both operand fragments to fp16 and run
    // the k16 step as two m16n8k8 fp16 mmas.
    mma_f16_k16(c0, c1, c2, c3, cvt_bf16x2_to_fp16x2(a0), cvt_bf16x2_to_fp16x2(a1),
                cvt_bf16x2_to_fp16x2(a2), cvt_bf16x2_to_fp16x2(a3), cvt_bf16x2_to_fp16x2(b0),
                cvt_bf16x2_to_fp16x2(b1));
}

__device__ __forceinline__ void mma_f16(float& c0, float& c1, float& c2, float& c3,
                                        unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                        unsigned b0, unsigned b1) {
    mma_f16_k16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
}
#else
__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3,
                                         unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                         unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_f16(float& c0, float& c1, float& c2, float& c3,
                                        unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                        unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#endif

__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// On RTX 5090, block-scaled FP8 avoids the plain form's half-rate FP32 accumulation.
// UE8M0 0x7f is exactly 1: preserve the raw E4M3 dot product and keep the callers'
// existing row scales in FP32 after the MMA. No operand repacking is needed.
// The kind::mxf8f6f4 block-scaled form needs sm_100 or newer. Ada (sm_89) has FP8
// tensor cores with the plain m16n8k32 form only; since the scale factor below is
// exactly 1.0, the plain form produces bit-identical results.
// port (sm_89): this guard is the earlier implementation's, restored - upstream dropped it when it
// started building only for sm_120a, which left every FP8 route unable to compile on Ada.
// Turing (sm_75) has no FP8 tensor cores, so the m16n8k32 e4m3 mma cannot run on
// it. The A8 (FP8-activation) linear path is a performance optimization for large
// prefill batches; on sm_75 the fp8 dispatch remaps it to the A16 execution, which
// decodes the FP8 operands to BF16 and runs the fp16-MMA plane (see mma_bf16 above).
// The A8 kernels are still compiled for sm_75 (they live in the same translation
// units), so this function must remain a valid device symbol: it is a loud canary
// that traps if any A8 kernel is ever reached on sm_75, which the dispatch makes
// impossible.
__device__ __forceinline__ void mma_fp8_e4m3(float& c0, float& c1, float& c2, float& c3,
                                             unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
    (void)c0; (void)c1; (void)c2; (void)c3;
    (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
    asm volatile("trap;");
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    constexpr unsigned kUnitScale8          = 0x7Fu;
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
                 "m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
                 "{%10}, {%11,%12}, {%13}, {%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(kUnitScale8),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(kUnitScale8), "h"(kScaleBlockId),
                   "h"(kScaleThreadId));
#else
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_tf32_bits(float& c0, float& c1, float& c2, float& c3,
                                              unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                              unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_tf32(float& c0, float& c1, float& c2, float& c3, float a0,
                                         float a1, float a2, float a3, float b0, float b1) {
    mma_tf32_bits(c0, c1, c2, c3, __float_as_uint(a0), __float_as_uint(a1), __float_as_uint(a2),
                  __float_as_uint(a3), __float_as_uint(b0), __float_as_uint(b1));
}

__device__ __forceinline__ void mma_nvfp4_e4m3(float& c0, float& c1, float& c2, float& c3,
                                               unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                               unsigned b0, unsigned b1, unsigned sfa,
                                               unsigned sfb) {
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                 "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7}, "
                 "{%8,%9}, "
                 "{%0,%1,%2,%3}, "
                 "{%10}, "
                 "{%11,%12}, "
                 "{%13}, "
                 "{%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(sfb), "h"(kScaleBlockId),
                   "h"(kScaleThreadId));
}

} // namespace ninfer::ops::common

// Callers that predate the family split resolve these names in ninfer::ops.
namespace ninfer::ops {
using common::ldmatrix_x2;
using common::ldmatrix_x2_t;
using common::ldmatrix_x4;
using common::ldmatrix_x4_t;
using common::mma_bf16;
using common::mma_f16;
using common::mma_fp8_e4m3;
using common::mma_nvfp4_e4m3;
using common::mma_s8;
using common::mma_tf32;
using common::mma_tf32_bits;
} // namespace ninfer::ops
