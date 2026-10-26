// Regression guards for the int8 route the MoE gate_up uses, kept as a test so the things that are
// easy to get wrong cannot silently rot:
//
//   1. the m16n8k32 s8 mma fragment layout (A 16x32, B 32x8, C 16x8)
//   2. the whole chain - int4 nibble decode, int8 activation, int8 mma, per-64-group scale -
//      against a double reference
//   3. the activation quantiser's row-per-token layout
//   4. the prefill tile kernel: grouping, per-column activation scales, the swiglu product and the
//      [slot][row] store the down kernel reads
//
// (4) matters most: an earlier version of this kernel wrote two uncombined planes in a transposed
// layout and nothing in the tree could consume them. The reference here is the product's contract,
// not the kernel's own convention.
#include "ops/op_check.h"
#include "ops/moe/kernels/cuda/groupwise_int/expert.cuh"
#include "ops/moe/kernels/cuda/groupwise_int/grouped_gate_up_i8_tile.cuh"
#include "ops/moe/kernels/cuda/groupwise_int/quantize_a8.cuh"

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

    const int cc            = (t % 4) * 2;
    C[r * 8 + cc]           = c0;
    C[r * 8 + cc + 1]       = c1;
    C[(r + 8) * 8 + cc]     = c2;
    C[(r + 8) * 8 + cc + 1] = c3;
}

// --- part 2: the whole chain on one expert, one token -------------------------------------
constexpr int kRows = 1024;   // 2 * intermediate: gate rows then up rows
constexpr int kK    = 2048;
constexpr int kG    = kK / 64;
constexpr int kGate = kRows / 2;

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
            const int k0      = g * 64 + sub * 32;
            const auto* c_lo  = codes + (std::int64_t)(row0 + r) * (kK / 2);
            const auto* c_hi  = codes + (std::int64_t)(row0 + r + 8) * (kK / 2);
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
    const bool alloc_ok  = cudaMalloc(&dCodes, codes.size()) == cudaSuccess &&
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

int check_quantize() {
    // Three tokens, kK values each: bf16 in, int8 out, one float scale per 64, one row per token.
    constexpr int kTokens = 3;
    std::vector<__nv_bfloat16> x((std::size_t)kTokens * kK);
    unsigned seed = 777;
    auto next     = [&]() { seed = seed * 1664525u + 1013904223u; return seed >> 8; };
    for (auto& v : x) {
        const float f = ((float)(next() % 20001) / 10000.0f) - 1.0f;  // [-1, 1)
        v             = __float2bfloat16_rn(f);
    }

    std::int8_t* dQ    = nullptr;
    float* dS          = nullptr;
    __nv_bfloat16* dX  = nullptr;
    const bool ok      = cudaMalloc(&dX, x.size() * sizeof(__nv_bfloat16)) == cudaSuccess &&
                    cudaMalloc(&dQ, x.size()) == cudaSuccess &&
                    cudaMalloc(&dS, (std::size_t)kTokens * kG * sizeof(float)) == cudaSuccess;
    if (!ok) {
        std::printf("FAIL int8 quantize: cudaMalloc\n");
        return 1;
    }
    cudaMemcpy(dX, x.data(), x.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    namespace gi = ninfer::ops::detail::moe_cuda::groupwise_int;
    gi::launch_quantize_a8(dX, dQ, dS, kK, kTokens, nullptr);
    cudaDeviceSynchronize();

    std::vector<std::int8_t> got_q(x.size());
    std::vector<float> got_s((std::size_t)kTokens * kG);
    cudaMemcpy(got_q.data(), dQ, x.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(got_s.data(), dS, got_s.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dX);
    cudaFree(dQ);
    cudaFree(dS);

    int bad = 0;
    for (int tok = 0; tok < kTokens; ++tok) {
        for (int g = 0; g < kG; ++g) {
            const auto* row = &x[(std::size_t)tok * kK + g * 64];
            float maxv      = 0.f;
            for (int i = 0; i < 64; ++i) { maxv = fmaxf(maxv, std::fabs(__bfloat162float(row[i]))); }
            const float scale = maxv > 0.f ? maxv / 127.0f : 1.0f;
            const float got   = got_s[(std::size_t)tok * kG + g];
            if (std::fabs(scale - got) > 1e-7f * std::max(1.f, std::fabs(scale))) {
                if (bad < 3) {
                    std::printf("  scale[tok %d][%d] gpu=%.8f ref=%.8f\n", tok, g, got, scale);
                }
                ++bad;
            }
            const float inv = 1.0f / scale;
            for (int i = 0; i < 64; ++i) {
                // __float2int_rn is round-to-nearest-even; lrintf is the host equivalent.
                const int ref = (int)lrintf(__bfloat162float(row[i]) * inv);
                const int val = (int)got_q[(std::size_t)tok * kK + g * 64 + i];
                if (ref != val) {
                    if (bad < 3) {
                        std::printf("  q[tok %d][%d] gpu=%d ref=%d\n", tok, g * 64 + i, val, ref);
                    }
                    ++bad;
                }
            }
        }
    }
    const int total = (int)x.size();
    std::printf("int8 quantize: %d / %d values exact, %d tokens\n", total - bad, total, kTokens);
    return bad;
}

template <int Cols>
int check_tile(const char* label) {
    // Two experts, a full Cols-column job and a partial one each: the real gate_up shape, and the
    // tail job an expert leaves when its slot count is not a multiple of the tile width. Checks
    // grouping, the per-column activation scale, the swiglu product and the [slot][row] store, at
    // both tile widths the shape policies use (32 below 768 tokens, 64 at or above).
    constexpr int kExperts        = 2;
    constexpr int kTail           = 5;
    constexpr int kSlotsPerExpert = Cols + kTail;
    constexpr int kSlots          = kExperts * kSlotsPerExpert;
    constexpr int kTokens         = 20;

    std::vector<std::uint8_t> codes((std::size_t)kExperts * kRows * (kK / 2));
    std::vector<__half> wsc((std::size_t)kExperts * kRows * kG);
    std::vector<std::int8_t> act((std::size_t)kTokens * kK);
    std::vector<float> asc((std::size_t)kTokens * kG);

    unsigned seed = 999;
    auto next     = [&]() { seed = seed * 1664525u + 1013904223u; return seed >> 8; };
    for (auto& c : codes) { c = (std::uint8_t)(next() & 0xff); }
    for (auto& s : wsc) { s = __float2half(0.001f + 0.0005f * (float)(next() % 8)); }
    for (auto& a : act) { a = (std::int8_t)((int)(next() % 25) - 12); }
    for (auto& s : asc) { s = 0.01f + 0.002f * (float)(next() % 5); }

    std::vector<int> offsets(kExperts + 1, 0);
    for (int e = 0; e < kExperts; ++e) { offsets[e + 1] = offsets[e] + kSlotsPerExpert; }
    std::vector<int> order(kSlots);
    for (int s = 0; s < kSlots; ++s) { order[s] = (s * 7 + 3) % kTokens; }
    std::vector<int> job_experts{0, 0, 1, 1};
    std::vector<int> job_columns{0, Cols, 0, Cols};

    __nv_bfloat16* dOut = nullptr;
    std::int8_t* dAct   = nullptr;
    float* dAsc         = nullptr;
    std::uint8_t* dCodes = nullptr;
    __half* dWsc        = nullptr;
    int* dOff           = nullptr;
    int* dOrd           = nullptr;
    int* dJobE          = nullptr;
    int* dJobC          = nullptr;
    ninfer::ops::detail::moe_cuda::groupwise_int::GroupedInfo* dInfo = nullptr;
    const bool ok =
        cudaMalloc(&dOut, (std::size_t)kSlots * kGate * sizeof(__nv_bfloat16)) == cudaSuccess &&
        cudaMalloc(&dAct, act.size()) == cudaSuccess &&
        cudaMalloc(&dAsc, asc.size() * sizeof(float)) == cudaSuccess &&
        cudaMalloc(&dCodes, codes.size()) == cudaSuccess &&
        cudaMalloc(&dWsc, wsc.size() * sizeof(__half)) == cudaSuccess &&
        cudaMalloc(&dOff, offsets.size() * sizeof(int)) == cudaSuccess &&
        cudaMalloc(&dOrd, order.size() * sizeof(int)) == cudaSuccess &&
        cudaMalloc(&dJobE, job_experts.size() * sizeof(int)) == cudaSuccess &&
        cudaMalloc(&dJobC, job_columns.size() * sizeof(int)) == cudaSuccess &&
        cudaMalloc(&dInfo, sizeof(*dInfo)) == cudaSuccess;
    if (!ok) {
        std::printf("FAIL int8 tile: cudaMalloc\n");
        return 1;
    }
    const ninfer::ops::detail::moe_cuda::groupwise_int::GroupedInfo info{
        (int)job_experts.size(), 0, 0};
    cudaMemcpy(dAct, act.data(), act.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dAsc, asc.data(), asc.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dCodes, codes.data(), codes.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dWsc, wsc.data(), wsc.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dOff, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dOrd, order.data(), order.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dJobE, job_experts.data(), job_experts.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dJobC, job_columns.data(), job_columns.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dInfo, &info, sizeof(info), cudaMemcpyHostToDevice);

    namespace gi = ninfer::ops::detail::moe_cuda::groupwise_int;
    // PackedExpertMatrix is the bank layout the product uses for a per-expert artifact: a shared
    // base plus a row offset of expert * Rows.
    const gi::PackedExpertMatrix access{dCodes, nullptr,
                                        reinterpret_cast<const std::uint8_t*>(dWsc)};
    gi::grouped_gate_up_i8_tile<kK, kRows, Cols>
        <<<dim3(kGate / 16, (int)job_experts.size()), Cols * 4>>>(
            dAct, dAsc, access, dOff, dOrd, dJobE, dJobC, dInfo, kGate, dOut);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> got((std::size_t)kSlots * kGate);
    cudaMemcpy(got.data(), dOut, got.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    // Deterministic timing of the two forms on the same launch shape. The product-level rate is
    // useless here: it mixes prefill and decode calls and drifts by factors between sessions, while
    // this runs the identical grid in a loop with the clock already warm.
    {
        const auto time_form = [&](bool use_ldm, int iters) {
            for (int i = 0; i < 8; ++i) { // warm the clocks on this shape
                if (use_ldm) {
                    gi::grouped_gate_up_i8_tile_ldm<kK, kRows, Cols>
                        <<<dim3(kGate / 16, (int)job_experts.size()), Cols * 4>>>(
                            dAct, dAsc, access, dOff, dOrd, dJobE, dJobC, dInfo, kGate, dOut);
                } else {
                    gi::grouped_gate_up_i8_tile<kK, kRows, Cols>
                        <<<dim3(kGate / 16, (int)job_experts.size()), Cols * 4>>>(
                            dAct, dAsc, access, dOff, dOrd, dJobE, dJobC, dInfo, kGate, dOut);
                }
            }
            cudaDeviceSynchronize();
            cudaEvent_t a{}, b{};
            cudaEventCreate(&a);
            cudaEventCreate(&b);
            cudaEventRecord(a);
            for (int i = 0; i < iters; ++i) {
                if (use_ldm) {
                    gi::grouped_gate_up_i8_tile_ldm<kK, kRows, Cols>
                        <<<dim3(kGate / 16, (int)job_experts.size()), Cols * 4>>>(
                            dAct, dAsc, access, dOff, dOrd, dJobE, dJobC, dInfo, kGate, dOut);
                } else {
                    gi::grouped_gate_up_i8_tile<kK, kRows, Cols>
                        <<<dim3(kGate / 16, (int)job_experts.size()), Cols * 4>>>(
                            dAct, dAsc, access, dOff, dOrd, dJobE, dJobC, dInfo, kGate, dOut);
                }
            }
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            float ms = 0.0F;
            cudaEventElapsedTime(&ms, a, b);
            cudaEventDestroy(a);
            cudaEventDestroy(b);
            return ms / float(iters);
        };
        const float naive_ms = time_form(false, 200);
        const float ldm_ms   = time_form(true, 200);
        std::printf("int8 tile timing (%s): naive %.3f ms  ldmatrix %.3f ms  (%.2fx)\n", label,
                    naive_ms, ldm_ms, naive_ms / ldm_ms);
    }

    // The ldmatrix tile only changes how the fragments are assembled, so it has to land on the same
    // bytes; any difference is a staging or fragment-layout bug rather than arithmetic.
    __nv_bfloat16* dOut2 = nullptr;
    int ldm_bad          = 0;
    if (cudaMalloc(&dOut2, got.size() * sizeof(__nv_bfloat16)) != cudaSuccess) {
        std::printf("FAIL int8 tile ldm: cudaMalloc\n");
        return 1;
    }
    gi::grouped_gate_up_i8_tile_ldm<kK, kRows, Cols>
        <<<dim3(kGate / 16, (int)job_experts.size()), Cols * 4>>>(
            dAct, dAsc, access, dOff, dOrd, dJobE, dJobC, dInfo, kGate, dOut2);
    if (const auto err = cudaDeviceSynchronize(); err != cudaSuccess) {
        std::printf("FAIL int8 tile ldm (%s): %s\n", label, cudaGetErrorString(err));
        cudaFree(dOut2);
        return 1;
    }
    std::vector<__nv_bfloat16> got2(got.size());
    cudaMemcpy(got2.data(), dOut2, got.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaFree(dOut2);
    for (std::size_t i = 0; i < got2.size(); ++i) {
        if (__bfloat162float(got[i]) != __bfloat162float(got2[i])) { ++ldm_bad; }
    }
    std::printf("int8 tile ldmatrix vs naive (%s): %d / %d elements identical\n", label,
                (int)got2.size() - ldm_bad, (int)got2.size());

    cudaFree(dOut);
    cudaFree(dAct);
    cudaFree(dAsc);
    cudaFree(dCodes);
    cudaFree(dWsc);
    cudaFree(dOff);
    cudaFree(dOrd);
    cudaFree(dJobE);
    cudaFree(dJobC);
    cudaFree(dInfo);

    // Reference: decode both weight rows (gate at row, up at kGate + row), dot each against the
    // slot's own token, scale per group, then form the swiglu product.
    const auto w_at = [&](int e, int row, int k) {
        const int byte = codes[((std::size_t)e * kRows + row) * (kK / 2) + (k >> 1)];
        const int n    = (k & 1) ? (byte >> 4) : (byte & 0x0f);
        return (n ^ 8) - 8;
    };
    const auto ref_at = [&](int slot, int row) {
        const int e   = slot / kSlotsPerExpert;
        const int tok = order[slot];
        double gsum = 0.0, usum = 0.0;
        for (int g = 0; g < kG; ++g) {
            int ag = 0, au = 0;
            for (int k = g * 64; k < (g + 1) * 64; ++k) {
                const int a = (int)act[(std::size_t)tok * kK + k];
                ag += w_at(e, row, k) * a;
                au += w_at(e, kGate + row, k) * a;
            }
            const float sa = asc[(std::size_t)tok * kG + g];
            gsum += (double)ag * (double)__half2float(wsc[((std::size_t)e * kRows + row) * kG + g]) *
                    (double)sa;
            usum += (double)au *
                    (double)__half2float(wsc[((std::size_t)e * kRows + kGate + row) * kG + g]) *
                    (double)sa;
        }
        const double silu = gsum / (1.0 + std::exp(-gsum));
        return silu * usum;
    };

    double worst_rel = 0.0;
    int bad = 0, checked = 0;
    // Every slot at rows that land in different 16-row tiles, plus two slots row by row.
    const int probe_rows[] = {0, 8, 255, 256, 511};
    for (int slot = 0; slot < kSlots; ++slot) {
        for (int row : probe_rows) {
            const double ref = ref_at(slot, row);
            const double val = __bfloat162float(got[(std::size_t)slot * kGate + row]);
            const double absd = std::fabs(val - ref);
            const double rel  = ref != 0.0 ? absd / std::fabs(ref) : absd;
            if (rel > worst_rel) { worst_rel = rel; }
            if (rel > 2e-2 && absd > 1e-2) { ++bad; }
            ++checked;
        }
    }
    for (int slot : {0, 33}) {
        for (int row = 0; row < kGate; ++row) {
            const double ref = ref_at(slot, row);
            const double val = __bfloat162float(got[(std::size_t)slot * kGate + row]);
            const double absd = std::fabs(val - ref);
            const double rel  = ref != 0.0 ? absd / std::fabs(ref) : absd;
            if (rel > worst_rel) { worst_rel = rel; }
            if (rel > 2e-2 && absd > 1e-2) { ++bad; }
            ++checked;
        }
    }
    std::printf("int8 tile swiglu (%s): %d / %d values within bf16 rounding (worst rel %.2e)\n",
                label, checked - bad, checked, worst_rel);
    return bad + ldm_bad;
}

} // namespace

int main() {
    const int layout_bad  = check_layout();
    const int gemv_bad    = check_gemv();
    const int quant_bad   = check_quantize();
    const int grouped_bad = check_tile<32>("32 columns") + check_tile<64>("64 columns");
    const int failures    = layout_bad + gemv_bad + quant_bad + grouped_bad;
    if (failures == 0) { std::printf("PASS moe int8 mma\n"); }
    else { std::printf("FAIL moe int8 mma: %d\n", failures); }
    return failures == 0 ? 0 : 1;
}


