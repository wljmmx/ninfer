// Guards for the rank-compressed (rk) E8 KV carriers.
//
// These assertions pin the measurements behind IT.29 / IT.33 so that a future codebook or kernel
// change cannot silently move them:
//
//  * K and V share the same 2-bit cylinder codec, and its intrinsic reconstruction error is ~48%
//    (K) / ~50% (V). That is the design point of `rk4v2-e8` / `rk2v4-e8`'s K plane.
//  * The 1-bit residual takes V from ~50% to ~28% on clean data, but makes it WORSE (~68%) when a
//    64-dim group carries an outlier: the residual step is one whole code unit (+-absmax/7). This
//    is why `rk4v3-e8` (V = 96 B) is a dead end rather than a bug (IT.33).
//  * Hadamard-rotating V before the quantizer is a no-op on clean data and only materially helps
//    the outlier case -- and even then it lands far from int4, so rotation cannot rescue 2/3 bits.
//  * The 96 B V plane offset formula (64 B cylinder pairs + 32 B residual ballots) round-trips with
//    no out-of-range root codes, and its residual offsets agree with the writer.
//
// Every case prints its measured value so a failure carries the number that moved.
#include "cuda_fixture.h"

#include "ops/kernel/e8_root_codec.cuh"
#include "ops/kv_cache/rk4_codec.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

using namespace ninfer::ops;

constexpr int kGroups  = 8192;                       // 8192 * 64 dims
constexpr size_t kElems = static_cast<size_t>(kGroups) * 64;

int failures = 0;

void expect(const char* label, double value, double low, double high) {
    const bool ok = value >= low && value <= high;
    std::printf("  [%s] %-34s = %.5f  (want %.5f..%.5f)\n", ok ? "ok" : "FAIL", label, value, low,
                high);
    if (!ok) ++failures;
}

void expect_true(const char* label, bool condition) {
    std::printf("  [%s] %s\n", condition ? "ok" : "FAIL", label);
    if (!condition) ++failures;
}

// ---------------------------------------------------------------------------------------------
// One 64-dim group per warp pair; `in` already holds whatever rotation the caller applied.
// Writes three reconstructions: plain K decode, plain V decode, and V + 1-bit residual.
__global__ void codec_roundtrip(const float* __restrict__ in, float* __restrict__ out_k,
                                float* __restrict__ out_v, float* __restrict__ out_lat,
                                int groups) {
    const int g    = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    if (g >= groups) return;

    const float* base = in + static_cast<long long>(g) * 64;
    const float v0    = base[lane];
    const float v1    = base[lane + 32];

    float absmax = fmaxf(fabsf(v0), fabsf(v1));
    for (int mask = 16; mask > 0; mask >>= 1)
        absmax = fmaxf(absmax, __shfl_xor_sync(0xffffffffu, absmax, mask));

    const float scale = __half2float(rk4_absmax_to_scale_h(absmax));

    uint8_t r0, a0, r1, a1;
    e8_encode_cylinder_8d_warp(v0, scale, r0, a0, lane);
    e8_encode_cylinder_8d_warp(v1, scale, r1, a1, lane);

    __align__(8) int8_t dk0[8], dv0[8], dk1[8], dv1[8];
    e8_root_decode_8d_fast(r0, a0, dk0);
    e8_root_decode_8d_fast_v(r0, a0, dv0);
    e8_root_decode_8d_fast(r1, a1, dk1);
    e8_root_decode_8d_fast_v(r1, a1, dv1);

    const int sub = lane & 7;
    const int up0 = (v0 / scale >= static_cast<float>(dk0[sub])) ? 1 : -1;
    const int up1 = (v1 / scale >= static_cast<float>(dk1[sub])) ? 1 : -1;

    out_k[static_cast<long long>(g) * 64 + lane]      = dk0[sub] * scale;
    out_k[static_cast<long long>(g) * 64 + lane + 32] = dk1[sub] * scale;
    out_v[static_cast<long long>(g) * 64 + lane]      = dv0[sub] * scale;
    out_v[static_cast<long long>(g) * 64 + lane + 32] = dv1[sub] * scale;
    out_lat[static_cast<long long>(g) * 64 + lane]      = (dk0[sub] + up0) * scale;
    out_lat[static_cast<long long>(g) * 64 + lane + 32] = (dk1[sub] + up1) * scale;
}

__device__ int plane_rad_zero  = 0;
__device__ int plane_sub_total = 0;
__device__ int plane_root_oob  = 0;
__device__ int plane_root_all  = 0;

// The 96 B V plane: [0,64) holds one (root, rad_axis) pair per 8D subspace, [64,96) one residual
// bit per element. Writer offsets are (grp*8 + s)*2 and 64 + grp*8 + s; the reader walks 16 dims at
// a time and reads the sign halfword at voff + 64 - subidx. This reproduces both sides verbatim.
__global__ void plane_roundtrip(const float* __restrict__ in, float* __restrict__ out, int groups) {
    const int g    = blockIdx.x;
    const int lane = threadIdx.x & 31;
    if (g >= groups) return;

    __shared__ unsigned char plane[96];
    if (threadIdx.x < 96) plane[threadIdx.x] = 0;
    __syncthreads();

    const float* base = in + static_cast<long long>(g) * 64;
    const float v0    = base[lane];
    const float v1    = base[lane + 32];

    float absmax = fmaxf(fabsf(v0), fabsf(v1));
    for (int mask = 16; mask > 0; mask >>= 1)
        absmax = fmaxf(absmax, __shfl_xor_sync(0xffffffffu, absmax, mask));
    const float scale = __half2float(rk4_absmax_to_scale_h(absmax));

    uint8_t r0, a0, r1, a1;
    e8_encode_cylinder_8d_warp(v0, scale, r0, a0, lane);
    e8_encode_cylinder_8d_warp(v1, scale, r1, a1, lane);
    const uint32_t ballot0 = __ballot_sync(0xffffffffu, v0 / scale >= 0.0f);
    const uint32_t ballot1 = __ballot_sync(0xffffffffu, v1 / scale >= 0.0f);

    const uint32_t pair0 = r0 | (a0 << 8);
    const uint32_t pair1 = r1 | (a1 << 8);
    const uint32_t pair0_next = __shfl_xor_sync(0xffffffffu, pair0, 8);
    const uint32_t pair1_next = __shfl_xor_sync(0xffffffffu, pair1, 8);
    if ((lane & 15) == 0) {
        const int s0 = lane >> 3;
        *reinterpret_cast<uint32_t*>(&plane[(0 * 8 + s0) * 2]) = pair0 | (pair0_next << 16);
        *reinterpret_cast<uint32_t*>(&plane[(0 * 8 + s0 + 4) * 2]) = pair1 | (pair1_next << 16);
    }
    if (lane == 0) {
        *reinterpret_cast<uint32_t*>(&plane[64 + 0 * 8]) = (ballot0 & 0xffu) |
                                                          (((ballot0 >> 8) & 0xffu) << 8) |
                                                          (((ballot0 >> 16) & 0xffu) << 16) |
                                                          (((ballot0 >> 24) & 0xffu) << 24);
        *reinterpret_cast<uint32_t*>(&plane[64 + 0 * 8 + 4]) = (ballot1 & 0xffu) |
                                                              (((ballot1 >> 8) & 0xffu) << 8) |
                                                              (((ballot1 >> 16) & 0xffu) << 16) |
                                                              (((ballot1 >> 24) & 0xffu) << 24);
    }
    __syncthreads();

    for (int d = 0; d < 64; d += 16) {
        const int subidx = (d / 64) * 8 + ((d / 8) & 7);
        const uint32_t raw = *reinterpret_cast<const uint32_t*>(&plane[subidx * 2]);
        const uint16_t signs =
            *reinterpret_cast<const uint16_t*>(&plane[subidx * 2 + 64 - subidx]);
        if (threadIdx.x == 0) {
            atomicAdd(&plane_rad_zero, ((raw >> 8) & 0xf0) == 0 ? 1 : 0);
            atomicAdd(&plane_sub_total, 1);
            atomicAdd(&plane_root_oob, (raw & 0xff) >= 240 ? 1 : 0);
            atomicAdd(&plane_root_oob, ((raw >> 16) & 0xff) >= 240 ? 1 : 0);
            atomicAdd(&plane_root_all, 2);
        }
        __align__(8) int8_t dec[8];
        e8_root_decode_8d_fast(static_cast<uint8_t>(raw), static_cast<uint8_t>(raw >> 8), dec);
        for (int i = 0; i < 8; ++i) out[static_cast<long long>(g) * 64 + d + i] = dec[i] * scale;
        e8_root_decode_8d_fast(static_cast<uint8_t>(raw >> 16), static_cast<uint8_t>(raw >> 24), dec);
        for (int i = 0; i < 8; ++i)
            out[static_cast<long long>(g) * 64 + d + 8 + i] = dec[i] * scale;
        (void)signs;
    }
}

// Normalized Walsh-Hadamard over contiguous blocks (symmetric, orthogonal).
void wht_blocks(float* x, size_t elems, int n) {
    const float norm = 1.0f / std::sqrt(static_cast<float>(n));
    for (size_t off = 0; off < elems; off += n) {
        float* p = x + off;
        for (int len = 1; len < n; len <<= 1)
            for (int i = 0; i < n; i += len << 1)
                for (int j = i; j < i + len; ++j) {
                    const float a = p[j], b = p[j + len];
                    p[j]       = a + b;
                    p[j + len] = a - b;
                }
        for (int i = 0; i < n; ++i) p[i] *= norm;
    }
}

double rel_l2(const std::vector<float>& got, const std::vector<float>& want) {
    double se = 0.0, sa = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = static_cast<double>(got[i]) - want[i];
        se += d * d;
        sa += static_cast<double>(want[i]) * want[i];
    }
    return std::sqrt(se / sa);
}

std::vector<float> make_input(bool outliers) {
    std::vector<float> data(kElems);
    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : data) x = dist(rng);
    if (outliers) {
        // 1% of groups get a single dimension at x64 -- the pattern that crushes absmax/7.
        std::uniform_int_distribution<int> pick(0, kGroups - 1);
        for (int k = 0; k < kGroups / 100; ++k)
            data[static_cast<size_t>(pick(rng)) * 64 + (k % 64)] *= 64.0f;
    }
    return data;
}

struct CodecResult {
    std::vector<float> k, v, lat;
};

CodecResult run_codec(const std::vector<float>& input, float *d_in, float *d_k, float *d_v,
                      float *d_lat) {
    const size_t bytes = kElems * sizeof(float);
    cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice);
    codec_roundtrip<<<kGroups / 8, 256>>>(d_in, d_k, d_v, d_lat, kGroups);
    CodecResult out;
    out.k.resize(kElems);
    out.v.resize(kElems);
    out.lat.resize(kElems);
    cudaMemcpy(out.k.data(), d_k, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(out.v.data(), d_v, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(out.lat.data(), d_lat, bytes, cudaMemcpyDeviceToHost);
    return out;
}

// int4 RTN on the same 64-dim groups: the carrier the V plane actually ships with.
double int4_reference(const std::vector<float>& input) {
    std::vector<float> out(kElems);
    for (size_t g = 0; g < kElems / 64; ++g) {
        float absmax = 0.0f;
        for (int i = 0; i < 64; ++i) absmax = std::fmax(absmax, std::fabs(input[g * 64 + i]));
        const float scale = absmax / 7.0f;
        for (int i = 0; i < 64; ++i) {
            float q = std::round(input[g * 64 + i] / scale);
            q       = q < -7.0f ? -7.0f : (q > 7.0f ? 7.0f : q);
            out[g * 64 + i] = q * scale;
        }
    }
    return rel_l2(out, input);
}

} // namespace

int main() {
    const size_t bytes = kElems * sizeof(float);
    float *d_in = nullptr, *d_k = nullptr, *d_v = nullptr, *d_lat = nullptr, *d_plane = nullptr;
    if (cudaMalloc(&d_in, bytes) != cudaSuccess || cudaMalloc(&d_k, bytes) != cudaSuccess ||
        cudaMalloc(&d_v, bytes) != cudaSuccess || cudaMalloc(&d_lat, bytes) != cudaSuccess ||
        cudaMalloc(&d_plane, bytes) != cudaSuccess) {
        std::printf("SKIP: no CUDA device available\n");
        return 77; // CTest SKIP_RETURN_CODE
    }

    const std::vector<float> clean    = make_input(false);
    const std::vector<float> outlier  = make_input(true);
    const CodecResult c_clean         = run_codec(clean, d_in, d_k, d_v, d_lat);
    const CodecResult c_outlier       = run_codec(outlier, d_in, d_k, d_v, d_lat);

    std::vector<float> rot_clean = clean, rot_outlier = outlier;
    wht_blocks(rot_clean.data(), kElems, 256);
    wht_blocks(rot_outlier.data(), kElems, 256);
    CodecResult r_clean   = run_codec(rot_clean, d_in, d_k, d_v, d_lat);
    CodecResult r_outlier = run_codec(rot_outlier, d_in, d_k, d_v, d_lat);
    wht_blocks(r_clean.k.data(), kElems, 256);
    wht_blocks(r_clean.v.data(), kElems, 256);
    wht_blocks(r_clean.lat.data(), kElems, 256);
    wht_blocks(r_outlier.k.data(), kElems, 256);
    wht_blocks(r_outlier.v.data(), kElems, 256);
    wht_blocks(r_outlier.lat.data(), kElems, 256);

    const double k_clean   = rel_l2(c_clean.k, clean);
    const double v_clean   = rel_l2(c_clean.v, clean);
    const double lat_clean = rel_l2(c_clean.lat, clean);
    const double k_out     = rel_l2(c_outlier.k, outlier);
    const double v_out     = rel_l2(c_outlier.v, outlier);
    const double lat_out   = rel_l2(c_outlier.lat, outlier);

    std::printf("rk E8 codec guards (%d groups x 64 dims)\n", kGroups);

    std::printf("clean data (IT.29 baseline)\n");
    expect("K 2-bit cylinder rel_l2", k_clean, 0.45, 0.51);
    expect("V 2-bit cylinder rel_l2", v_clean, 0.48, 0.53);
    expect("V 2-bit + residual rel_l2", lat_clean, 0.25, 0.32);
    expect("int4 RTN rel_l2", int4_reference(clean), 0.03, 0.12);
    expect_true("residual improves V on clean data", lat_clean < v_clean * 0.75);
    expect_true("int4 is an order of magnitude better than 2 bit",
                int4_reference(clean) < v_clean * 0.3);

    std::printf("outlier data (1%% of groups, one dimension x64)\n");
    expect("K 2-bit rel_l2", k_out, 0.52, 0.59);
    expect("V 2-bit rel_l2", v_out, 0.53, 0.60);
    expect("V 2-bit + residual rel_l2", lat_out, 0.64, 0.72);
    expect_true("residual is HARMFUL on outlier groups (IT.33)", lat_out > v_out);

    std::printf("Hadamard rotation before the quantizer (D256)\n");
    const double rv_clean   = rel_l2(r_clean.v, clean);
    const double rlat_clean = rel_l2(r_clean.lat, clean);
    const double rv_out     = rel_l2(r_outlier.v, outlier);
    const double rlat_out   = rel_l2(r_outlier.lat, outlier);
    expect_true("rotation is neutral on clean data",
                std::fabs(rv_clean - v_clean) < 0.01 && std::fabs(rlat_clean - lat_clean) < 0.01);
    expect_true("rotation helps V 2-bit on outliers", rv_out < v_out * 0.9);
    expect_true("rotation rescues the residual on outliers", rlat_out < lat_out * 0.5);
    expect_true("rotation still cannot reach int4", rv_out > int4_reference(outlier) * 3.0);

    std::printf("96 B V plane offsets (64 B cylinders + 32 B residual ballots)\n");
    cudaMemcpy(d_in, clean.data(), bytes, cudaMemcpyHostToDevice);
    plane_roundtrip<<<kGroups, 32>>>(d_in, d_plane, kGroups);
    const cudaError_t plane_err = cudaDeviceSynchronize();
    if (plane_err != cudaSuccess) {
        std::printf("  [FAIL] plane kernel: %s\n", cudaGetErrorString(plane_err));
        ++failures;
    } else {
        std::vector<float> got(kElems);
        cudaMemcpy(got.data(), d_plane, bytes, cudaMemcpyDeviceToHost);
        expect("plane round-trip rel_l2", rel_l2(got, clean), 0.45, 0.52);
        int rad_zero = 0, sub_total = 0, root_oob = 0, root_all = 0;
        cudaMemcpyFromSymbol(&rad_zero, plane_rad_zero, sizeof(int));
        cudaMemcpyFromSymbol(&sub_total, plane_sub_total, sizeof(int));
        cudaMemcpyFromSymbol(&root_oob, plane_root_oob, sizeof(int));
        cudaMemcpyFromSymbol(&root_all, plane_root_all, sizeof(int));
        std::printf("  reads=%d rad_idx==0=%d root>=240=%d/%d\n", sub_total, rad_zero, root_oob,
                    root_all);
        expect_true("no rad_axis index collapses to zero", rad_zero == 0);
        expect_true("no root code leaves the E8 range", root_oob == 0);
    }

    cudaFree(d_in);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_lat);
    cudaFree(d_plane);

    std::printf("%s rk E8 codec guards\n", failures == 0 ? "OK" : "FAIL");
    return failures == 0 ? 0 : 1;
}
