// Regression guards for the int8 route the MoE gate_up would need, kept as a test so the two
// things that are easy to get wrong cannot silently rot:
//
//   1. the m16n8k32 s8 mma fragment layout (A 16x32, B 32x8, C 16x8)
//   2. the whole chain - int4 nibble decode, int8 activation, int8 mma, per-64-group scale -
//      against a double reference
//
// Both were verified before any product code was written against them. If either fails, an int8
// gate_up built on top is wrong, so this is the cheapest place to find out.
#include "ops/op_check.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned pack4(const std::int8_t* p) {
    return (unsigned)(std::uint8_t)p[0] | ((unsigned)(std::uint8_t)p[1] << 8) |
           ((unsigned)(std::uint8_t)p[2] << 16) | ((unsigned)(std::uint8_t)p[3] << 24);
}

// --- part 1: one mma, layout only ---------------------------------------------------------
__global__ void layout_probe(const std::int8_t* A, const std::int8_t* B, int* C) {
    const int t  = threadIdx.x;
    const int r  = t / 4;
    const int c4 = (t % 4) * 4;

    const unsigned a0 = pack4(&A[r * 32 + c4]);
    const unsigned a1 = pack4(&A[(r + 8) * 32 + c4]);
    const unsigned a2 = pack4(&A[r * 32 + c4 + 16]);
    const unsigned a3 = pack4(&A[(r + 8) * 32 + c4 + 16]);

    const int bc = t / 4, br = (t % 4) * 4;
    unsigned b0 = 0, b1 = 0;
    for (int i = 0; i < 4; ++i) {
        b0 |= (unsigned)(std::uint8_t)B[(br + i) * 8 + bc] << (8 * i);
        b1 |= (unsigned)(std::uint8_t)B[(br + 16 + i) * 8 + bc] << (8 * i);
    }

    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    mma_s8(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);

    const int cc             = (t % 4) * 2;
    C[r * 8 + cc]            = c0;
    C[r * 8 + cc + 1]        = c1;
    C[(r + 8) * 8 + cc]      = c2;
    C[(r + 8) * 8 + cc + 1]  = c3;
}

// --- part 2: the whole chain on one expert, one token -------------------------------------
constexpr int kRows = 1024;
constexpr int kK    = 2048;
constexpr int kG    = kK / 64;

// One nibble in [0,15] -> the signed value the quantiser meant: (n ^ 8) - 8, range [-8, 7].
__device__ __forceinline__ int nib_to_i8(const std::uint8_t* row_codes, int k) {
    const int byte = row_codes[k >> 1];
    const int n    = (k & 1) ? (byte >> 4) : (byte & 0x0f);
    return (n ^ 8) - 8;
}

__device__ __forceinline__ unsigned pack4_nib(const std::uint8_t* row_codes, int k) {
    unsigned w = 0;
    for (int i = 0; i < 4; ++i) {
        w |= (unsigned)(std::uint8_t)nib_to_i8(row_codes, k + i) << (8 * i);
    }
    return w;
}

__global__ void gemv_probe(const std::uint8_t* __restrict__ codes, const __half* __restrict__ wsc,
                           const std::int8_t* __restrict__ act, const float* __restrict__ asc,
                           float* __restrict__ out) {
    const int t    = threadIdx.x;
    const int row0 = blockIdx.x * 16;
    const int r    = t / 4;
    const int c4   = (t % 4) * 4;

    float accf[4] = {0.f, 0.f, 0.f, 0.f};
    for (int g = 0; g < kG; ++g) {
        int acc[4] = {0, 0, 0, 0};
        for (int sub = 0; sub < 2; ++sub) {
            const int k0     = g * 64 + sub * 32;
            const auto* c_lo = codes + (std::int64_t)(row0 + r) * (kK / 2);
            const auto* c_hi = codes + (std::int64_t)(row0 + r + 8) * (kK / 2);
            const unsigned a0 = pack4_nib(c_lo, k0 + c4);
            const unsigned a1 = pack4_nib(c_hi, k0 + c4);
            const unsigned a2 = pack4_nib(c_lo, k0 + c4 + 16);
            const unsigned a3 = pack4_nib(c_hi, k0 + c4 + 16);

            unsigned b0 = 0, b1 = 0;
            for (int i = 0; i < 4; ++i) {
                b0 |= (unsigned)(std::uint8_t)act[k0 + c4 + i] << (8 * i);
                b1 |= (unsigned)(std::uint8_t)act[k0 + c4 + 16 + i] << (8 * i);
            }
            mma_s8(acc[0], acc[1], acc[2], acc[3], a0, a1, a2, a3, b0, b1);
        }
        // c0/c1 belong to row r, c2/c3 to row r+8: the two rows carry different weight scales.
        const float s_lo = asc[g] * __half2float(wsc[(std::int64_t)(row0 + r) * kG + g]);
        const float s_hi = asc[g] * __half2float(wsc[(std::int64_t)(row0 + r + 8) * kG + g]);
        accf[0] += (float)acc[0] * s_lo;
        accf[1] += (float)acc[1] * s_lo;
        accf[2] += (float)acc[2] * s_hi;
        accf[3] += (float)acc[3] * s_hi;
    }
    if ((t % 4) == 0) {
        out[row0 + r]     = accf[0];
        out[row0 + r + 8] = accf[2];
    }
}

int check_layout() {
    std::int8_t A[16 * 32], B[32 * 8];
    for (int i = 0; i < 16 * 32; ++i) { A[i] = (std::int8_t)((i * 7) % 16 - 8); }
    for (int i = 0; i < 32 * 8; ++i) { B[i] = (std::int8_t)((i * 5) % 16 - 8); }

    int ref[16 * 8] = {};
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 8; ++j) {
            int sum = 0;
            for (int k = 0; k < 32; ++k) { sum += (int)A[i * 32 + k] * (int)B[k * 8 + j]; }
            ref[i * 8 + j] = sum;
        }
    }

    std::int8_t* dA = nullptr;
    std::int8_t* dB = nullptr;
    int* dC         = nullptr;
    if (cudaMalloc(&dA, sizeof(A)) != cudaSuccess || cudaMalloc(&dB, sizeof(B)) != cudaSuccess ||
        cudaMalloc(&dC, sizeof(ref)) != cudaSuccess) {
        std::printf("FAIL int8 mma layout: cudaMalloc\n");
        return 1;
    }
    cudaMemcpy(dA, A, sizeof(A), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(B), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, sizeof(ref));
    layout_probe<<<1, 32>>>(dA, dB, dC);
    cudaDeviceSynchronize();

    int out[16 * 8] = {};
    cudaMemcpy(out, dC, sizeof(out), cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    int bad = 0;
    for (int i = 0; i < 16 * 8; ++i) { bad += out[i] != ref[i] ? 1 : 0; }
    std::printf("int8 mma layout: %d / %d elements exact\n", 128 - bad, 128);
    return bad;
}

int check_gemv() {
    std::vector<std::uint8_t> codes((std::size_t)kRows * (kK / 2));
    std::vector<std::int8_t> act(kK);
    std::vector<__half> wsc((std::size_t)kRows * kG);
    std::vector<float> asc(kG);

    unsigned seed = 12345;
    auto next     = [&]() { seed = seed * 1664525u + 1013904223u; return seed >> 8; };
    for (auto& c : codes) { c = (std::uint8_t)(next() & 0xff); }
    for (int i = 0; i < kRows; ++i) {
        for (int g = 0; g < kG; ++g) {
            wsc[(std::size_t)i * kG + g] = __float2half(0.001f + 0.0005f * (float)(next() % 8));
        }
    }
    for (auto& a : act) { a = (std::int8_t)((int)(next() % 255) - 127); }
    for (auto& s : asc) { s = 0.01f + 0.002f * (float)(next() % 5); }

    std::vector<std::int8_t> w_i8((std::size_t)kRows * kK);
    for (int i = 0; i < kRows; ++i) {
        for (int k = 0; k < kK; ++k) {
            const int byte = codes[(std::size_t)i * (kK / 2) + (k >> 1)];
            const int n    = (k & 1) ? (byte >> 4) : (byte & 0x0f);
            w_i8[(std::size_t)i * kK + k] = (std::int8_t)((n ^ 8) - 8);
        }
    }

    std::vector<double> ref(kRows, 0.0);
    for (int i = 0; i < kRows; ++i) {
        double sum = 0.0;
        for (int g = 0; g < kG; ++g) {
            int acc = 0;
            for (int k = g * 64; k < (g + 1) * 64; ++k) {
                acc += (int)w_i8[(std::size_t)i * kK + k] * (int)act[k];
            }
            sum += (double)acc * (double)__half2float(wsc[(std::size_t)i * kG + g]) * (double)asc[g];
        }
        ref[i] = sum;
    }

    std::uint8_t* dCodes = nullptr;
    std::int8_t* dAct    = nullptr;
    __half* dWsc         = nullptr;
    float* dAsc          = nullptr;
    float* dOut          = nullptr;
    const bool alloc_ok =
        cudaMalloc(&dCodes, codes.size()) == cudaSuccess &&
        cudaMalloc(&dAct, act.size()) == cudaSuccess &&
        cudaMalloc(&dWsc, wsc.size() * sizeof(__half)) == cudaSuccess &&
        cudaMalloc(&dAsc, asc.size() * sizeof(float)) == cudaSuccess &&
        cudaMalloc(&dOut, kRows * sizeof(float)) == cudaSuccess;
    if (!alloc_ok) {
        std::printf("FAIL int8 gemv: cudaMalloc\n");
        return 1;
    }
    cudaMemcpy(dCodes, codes.data(), codes.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dAct, act.data(), act.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dWsc, wsc.data(), wsc.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dAsc, asc.data(), asc.size() * sizeof(float), cudaMemcpyHostToDevice);
    gemv_probe<<<kRows / 16, 32>>>(dCodes, dWsc, dAct, dAsc, dOut);
    cudaDeviceSynchronize();

    std::vector<float> got(kRows, 0.f);
    cudaMemcpy(got.data(), dOut, kRows * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dCodes);
    cudaFree(dAct);
    cudaFree(dWsc);
    cudaFree(dAsc);
    cudaFree(dOut);

    double worst_rel = 0.0;
    int bad          = 0;
    for (int i = 0; i < kRows; ++i) {
        const double a    = got[i], b = ref[i];
        const double absd = std::fabs(a - b);
        const double rel  = b != 0.0 ? absd / std::fabs(b) : absd;
        if (rel > worst_rel) { worst_rel = rel; }
        if (rel > 1e-5 && absd > 1e-4) { ++bad; }
    }
    std::printf("int8 gemv chain: %d / %d rows within 1e-5 (worst rel %.2e)\n", kRows - bad, kRows,
                worst_rel);
    return bad;
}

} // namespace

int main() {
    const int layout_bad = check_layout();
    const int gemv_bad   = check_gemv();
    const int failures   = layout_bad + gemv_bad;
    if (failures == 0) { std::printf("PASS moe int8 mma\n"); }
    else { std::printf("FAIL moe int8 mma: %d\n", failures); }
    return failures == 0 ? 0 : 1;
}
