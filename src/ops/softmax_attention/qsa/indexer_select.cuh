#pragma once

#include "ops/softmax_attention/qsa/indexer_arguments.cuh"
#include "ops/common/score_id_order.cuh"

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

namespace ninfer::ops::detail::qsa {

using IndexerScan   = cub::BlockScan<int, 256>;
using IndexerReduce = cub::BlockReduce<std::uint64_t, 256>;

struct IndexerRadixStorage {
    union {
        int histogram[8][256];
        IndexerScan::TempStorage scan;
        IndexerReduce::TempStorage reduce;
    } temp;

    std::uint64_t prefix, low, mask;
    int shift, remaining, selected_count;
};

struct IndexerKeyMin {
    __device__ std::uint64_t operator()(std::uint64_t a, std::uint64_t b) const {
        return min(a, b);
    }
};

struct IndexerKeyMax {
    __device__ std::uint64_t operator()(std::uint64_t a, std::uint64_t b) const {
        return max(a, b);
    }
};

struct IndexerStoreKeys {
    std::uint64_t* out;

    __device__ void operator()(int rank, std::uint64_t key) const { out[rank] = key; }
};

struct IndexerStoreIds {
    std::int32_t* out;

    __device__ void operator()(int rank, std::uint64_t key) const {
        out[rank] = id_from_order_key(key);
    }
};

// Full radix selection over a unique score/id total order. Prefix refinement stops only
// when exactly `selected` keys are >= the prefix. No approximate local candidate truncation.
template <int Items, class Store>
__device__ void indexer_select_topk(std::uint64_t (&keys)[Items], int count, int selected,
                                    IndexerRadixStorage& s, Store store) {
    const int tid = threadIdx.x;
    if (count <= selected) {
#pragma unroll
        for (int j = 0; j < Items; ++j)
            if (j * 256 + tid < count) store(j * 256 + tid, keys[j]);
        return;
    }
    std::uint64_t lo = ~std::uint64_t(0), hi = 0;
#pragma unroll
    for (int j = 0; j < Items; ++j) {
        if (keys[j]) lo = min(lo, keys[j]);
        hi = max(hi, keys[j]);
    }
    lo = IndexerReduce(s.temp.reduce).Reduce(lo, IndexerKeyMin{});
    if (tid == 0) s.low = lo;
    __syncthreads();
    hi = IndexerReduce(s.temp.reduce).Reduce(hi, IndexerKeyMax{});
    if (tid == 0) {
        s.shift     = ((63 - __clzll(s.low ^ hi)) / 8) * 8;
        s.mask      = s.shift == 56 ? 0 : (~std::uint64_t(0) << (s.shift + 8));
        s.prefix    = hi & s.mask;
        s.remaining = selected;
    }
    __syncthreads();
    for (int shift = s.shift; shift >= 0; shift -= 8) {
        for (int i = tid; i < 8 * 256; i += 256) (&s.temp.histogram[0][0])[i] = 0;
        __syncthreads();
#pragma unroll
        for (int j = 0; j < Items; ++j) {
            const auto key = keys[j];
            const int bin  = key && (key & s.mask) == s.prefix ? int((key >> shift) & 255) : 256;
            const unsigned peers = __match_any_sync(0xffffffffu, bin);
            if (bin < 256 && (tid % 32) == __ffs(peers) - 1)
                atomicAdd(&s.temp.histogram[tid / 32][bin], __popc(peers));
        }
        __syncthreads();
        int histogram = 0;
#pragma unroll
        for (int warp = 0; warp < 8; ++warp) histogram += s.temp.histogram[warp][tid];
        __syncthreads();
        int below, total;
        IndexerScan(s.temp.scan).ExclusiveSum(histogram, below, total);
        const int above     = total - below - histogram;
        const int remaining = s.remaining;
        __syncthreads();
        if (above < remaining && above + histogram >= remaining) {
            s.prefix |= std::uint64_t(tid) << shift;
            s.remaining      = remaining - above;
            s.selected_count = histogram;
        }
        __syncthreads();
        if (s.selected_count == s.remaining) break;
        if (tid == 0) s.mask |= std::uint64_t(255) << shift;
        __syncthreads();
    }
    const auto threshold = s.prefix;
    int kept             = 0;
#pragma unroll
    for (int j = 0; j < Items; ++j) kept += keys[j] >= threshold;
    int rank;
    IndexerScan(s.temp.scan).ExclusiveSum(kept, rank);
#pragma unroll
    for (int j = 0; j < Items; ++j)
        if (keys[j] >= threshold) store(rank++, keys[j]);
}

template <int Partition>
__global__
__launch_bounds__(256) void indexer_partition_kernel(const __grid_constant__ IndexerArguments a) {
    __shared__ IndexerJobs jobs;
    __shared__ IndexerRadixStorage radix;
    if (!jobs.build<1, Partition>(a)) return;
    for (int job = blockIdx.x; job < jobs.prefix[jobs.groups]; job += gridDim.x) {
        const int slot = jobs.group_for(job), part = job - jobs.prefix[slot];
        const int first = part * Partition;
        const int count = min(Partition, jobs.visible[slot] - first);
        std::uint64_t keys[Partition / 256];
#pragma unroll
        for (int j = 0; j < Partition / 256; ++j) {
            const int i = j * 256 + threadIdx.x;
            keys[j]     = i < count
                              ? score_id_order_key(a.scores[std::int64_t(slot) * a.columns + first + i],
                                                   first + i)
                              : 0;
        }
        // A discarded key already has `selected` better keys in its own partition, so it cannot
        // belong to the global selection. Only the final partition can emit fewer.
        auto* out = a.candidates + (std::int64_t(slot) * a.partitions + part) * a.selected;
        indexer_select_topk(keys, count, a.selected, radix, IndexerStoreKeys{out});
        __syncthreads();
    }
}

template <int Partition>
__global__
__launch_bounds__(256) void indexer_finish_kernel(const __grid_constant__ IndexerArguments a) {
    __shared__ IndexerRadixStorage radix;
    const int slot = blockIdx.x, tid = threadIdx.x, visible = a.visible(slot);
    const auto column = a.column(slot);
    auto* out         = a.ids + column * a.selected;
    if (tid == 0) a.counts[column] = min(visible, a.selected);
    if (visible <= a.selected) {
        for (int i = tid; i < visible; i += 256) out[i] = i;
        return;
    }
    const int parts = (visible + Partition - 1) / Partition;
    const int count =
        (parts - 1) * a.selected + min(a.selected, visible - (parts - 1) * Partition);
    const auto* input   = a.candidates + std::int64_t(slot) * a.partitions * a.selected;
    constexpr int Items = (kIndexerMaxBlocks / Partition) * kIndexerSelectedMax / 256;
    std::uint64_t keys[Items];
#pragma unroll
    for (int j = 0; j < Items; ++j) {
        const int i = j * 256 + tid;
        keys[j]     = i < count ? input[i] : 0;
    }
    indexer_select_topk(keys, count, a.selected, radix, IndexerStoreIds{out});
}
} // namespace ninfer::ops::detail::qsa
