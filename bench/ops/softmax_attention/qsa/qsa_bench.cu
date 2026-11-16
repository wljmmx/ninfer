// Public-Op benchmark for the QSA family: the block indexer/selector and the sparse grouped-query
// attention that consumes its selection.
//
// Harness rule (the same one the other Softmax Attention benches follow): fixture construction, the
// public workspace capacity queries, L2 conditioning, graph capture and synchronization stay
// outside the timed interval. An eager interval contains one public Op call; a captured graph
// contains that same one public call. `--entry chain` captures the two calls one model layer runs
// back to back (qsa_indexer, then qsa_sparse_gqa); `--entry indexer` and `--entry sparse` time the
// two stages separately. There are no private headers, launchers, tile/split controls, or
// kernel-name filters in this benchmark.
//
// Full-attention reference. The `full_*` columns are the analytic cost of a dense causal attention
// over the same visible set, not a second measurement. The dense public causal contract registers
// only [D256,Hq24,Hkv4] and [D256,Hq16,Hkv2], and QSA admits only [256,24,2] and [256,12,1], so no
// dense public profile shares the QSA geometry; re-running a different head pair would compare two
// different work models instead of two implementations. The reference is therefore derived from the
// same (shape, T, L, B) and printed next to the measured QSA row.
// TODO(v10-10): replace `full_bytes`/`full_flops` with a measured dense column once a public dense
// entry registers [256,24,2]; today ops::causal_softmax_attention rejects that query/KV head pair.
//
// Logical traffic model behind the printed GB/s:
//   indexer   : the four-head index queries it reads (128 BF16 words each), one 128-word index key
//               per scored four-token block, and the ids/counts it writes.
//   sparse    : the attended KV vectors (K BF16 + V FP16 = 1024 B per token per KV head), the query
//               read, the output-gate read and the BF16 output write.
//   chain     : both stages.
// `kv_share` divides the cache bytes the entry actually touches by the dense KV bytes of the same
// visible set. For the selector it mixes 128-wide index keys with 256-wide attention vectors, so it
// is a labelled share, not a like-for-like vector ratio.
//
// Only the default BFloat16-K/FP16-V cache profile is measured: the low-bit QSA profiles are opt-in
// behind NINFER_QSA_SPARSE_GQA_QUANT_KV and are not part of the default surface. Each query keeps
// min(S, floor((p+1)/4)) distinct four-token blocks plus its incomplete tail [4C, p+1), so a row
// exercises both the selected prefix and the tail suffix.

#include "ninfer/ops/qsa.h"

#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "ninfer_bench_common.h"

#include <cuda_profiler_api.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kHeadDim  = 256;  // QSA main-cache vector width
constexpr std::int32_t kIndexDim = 128;  // QSA index-key width
constexpr std::int32_t kIndexHeads = 4;
constexpr std::int32_t kIndexBlocksPerPage = 16;
constexpr std::int32_t kMaxSelectedBlocks = 512;  // the selector's documented selection bound
constexpr std::size_t kFlushBytes         = std::size_t{256} << 20;
// Nominal DRAM bandwidth is queried from the device (V10-11); see ninfer_bench_common.h. The
// printed TFLOP/s stays a plain useful-work rate: the selector's dot products and the attention's
// two contractions share no single dense Tensor Core reference, so no "% of peak" is implied.

enum class Entry : std::uint8_t { Indexer, Sparse, Chain, All };
enum class Execution : std::uint8_t { Eager, Graph, Both };
enum class CacheMode : std::uint8_t { Cold, Warm, Both };
enum class CacheState : std::uint8_t { Cold, Warm };
enum class GeometryChoice : std::uint8_t { H24Kv2, H12Kv1, All };

struct Geometry {
    const char* shape;
    std::int32_t query_heads;
    std::int32_t kv_heads;
};

constexpr Geometry kH24Kv2{"d256-h24-kv2", 24, 2};
constexpr Geometry kH12Kv1{"d256-h12-kv1", 12, 1};

struct Options {
    Entry entry             = Entry::All;
    GeometryChoice geometry = GeometryChoice::H24Kv2;
    std::vector<std::int32_t> tokens{1, 8, 64, 2048};
    std::vector<std::int32_t> kv_lengths{2048, 8192, 32768};
    std::int32_t selected = kMaxSelectedBlocks;
    std::int32_t batch    = 1;
    Execution execution   = Execution::Graph;
    CacheMode cache       = CacheMode::Cold;
    int warmup            = 3;
    int repeat            = 20;
    bool profile          = false;
    std::string csv_out;
};

// Semantic work of one measured row. Everything is a per-batch total unless named `per_query`.
struct Cost {
    double attended_keys    = 0.0;  // average keys the attention stage reads per query
    double scoped_blocks    = 0.0;  // average four-token blocks the selector scores per query
    double cache_bytes      = 0.0;  // KV/index bytes this entry actually reads
    double dense_kv_bytes   = 0.0;  // dense causal K+V bytes over the same visible set
    double qsa_bytes        = 0.0;
    double full_bytes       = 0.0;
    double qsa_flops        = 0.0;
    double full_flops       = 0.0;
};

struct Result {
    Entry entry;
    Geometry geometry;
    Execution execution;
    CacheState cache;
    std::int32_t tokens;
    std::int32_t kv_len;
    std::int32_t selected;
    std::int32_t batch;
    std::size_t workspace_bytes;
    std::size_t workspace_peak;
    Cost cost;
    bench::ColdTiming timing;
};

[[noreturn]] void usage(const char* message) {
    std::fprintf(stderr,
                 "error: %s\n"
                 "usage: ninfer_qsa_bench [--entry indexer|sparse|chain|all] "
                 "[--geometry d256-h24-kv2|d256-h12-kv1|all] "
                 "[--tokens 1,...] [--kv 2048,8192,32768] [--selected-blocks S] [--batch B] "
                 "[--execution eager|graph|both] [--cache cold|warm|both] "
                 "[--warmup N] [--repeat N] [--profile] [--csv-out PATH]\n",
                 message);
    std::exit(2);
}

std::int32_t parse_i32(std::string_view text, std::int32_t minimum, std::int32_t maximum,
                       const char* flag) {
    const std::string value(text);
    errno       = 0;
    char* end   = nullptr;
    long parsed = std::strtol(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0' || parsed < minimum ||
        parsed > maximum) {
        usage(flag);
    }
    return static_cast<std::int32_t>(parsed);
}

std::vector<std::int32_t> parse_list(const char* text, std::int32_t minimum, std::int32_t maximum,
                                     const char* flag) {
    std::vector<std::int32_t> result;
    std::string_view remaining(text);
    while (!remaining.empty()) {
        const std::size_t comma     = remaining.find(',');
        const std::string_view item = remaining.substr(0, comma);
        if (item.empty()) { usage(flag); }
        result.push_back(parse_i32(item, minimum, maximum, flag));
        if (comma == std::string_view::npos) { break; }
        remaining.remove_prefix(comma + 1);
    }
    if (result.empty()) { usage(flag); }
    return result;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](const char* flag) -> const char* {
            if (++index == argc) { usage(flag); }
            return argv[index];
        };
        if (argument == "--entry") {
            const std::string_view value(next("--entry requires a value"));
            if (value == "indexer")
                options.entry = Entry::Indexer;
            else if (value == "sparse")
                options.entry = Entry::Sparse;
            else if (value == "chain")
                options.entry = Entry::Chain;
            else if (value == "all")
                options.entry = Entry::All;
            else
                usage("--entry expects indexer, sparse, chain, or all");
        } else if (argument == "--geometry") {
            const std::string_view value(next("--geometry requires a value"));
            if (value == "d256-h24-kv2")
                options.geometry = GeometryChoice::H24Kv2;
            else if (value == "d256-h12-kv1")
                options.geometry = GeometryChoice::H12Kv1;
            else if (value == "all")
                options.geometry = GeometryChoice::All;
            else
                usage("--geometry expects d256-h24-kv2, d256-h12-kv1, or all");
        } else if (argument == "--tokens") {
            options.tokens = parse_list(next("--tokens requires a value"), 1, 262144, "--tokens");
        } else if (argument == "--kv") {
            options.kv_lengths =
                parse_list(next("--kv requires a value"), 1, 262144, "--kv");
        } else if (argument == "--selected-blocks") {
            options.selected = parse_i32(next("--selected-blocks requires a value"), 1,
                                         kMaxSelectedBlocks, "--selected-blocks");
        } else if (argument == "--batch") {
            options.batch = parse_i32(next("--batch requires a value"), 1, 8, "--batch");
        } else if (argument == "--execution") {
            const std::string_view value(next("--execution requires a value"));
            if (value == "eager")
                options.execution = Execution::Eager;
            else if (value == "graph")
                options.execution = Execution::Graph;
            else if (value == "both")
                options.execution = Execution::Both;
            else
                usage("--execution expects eager, graph, or both");
        } else if (argument == "--cache") {
            const std::string_view value(next("--cache requires a value"));
            if (value == "cold")
                options.cache = CacheMode::Cold;
            else if (value == "warm")
                options.cache = CacheMode::Warm;
            else if (value == "both")
                options.cache = CacheMode::Both;
            else
                usage("--cache expects cold, warm, or both");
        } else if (argument == "--warmup") {
            options.warmup = parse_i32(next("--warmup requires a value"), 0, 10000, "--warmup");
        } else if (argument == "--repeat") {
            options.repeat = parse_i32(next("--repeat requires a value"), 1, 10000, "--repeat");
        } else if (argument == "--profile") {
            options.profile = true;
        } else if (argument == "--csv-out") {
            options.csv_out = next("--csv-out requires a path");
        } else if (argument == "--help" || argument == "-h") {
            usage("help");
        } else {
            usage("unknown argument");
        }
    }
    // The public QSA envelope bounds every live p+1 by 262144 original tokens, so the last query of
    // a row (position L+T-1) is the tightest constraint on a shape.
    for (const std::int32_t tokens : options.tokens) {
        for (const std::int32_t kv_len : options.kv_lengths) {
            if (kv_len > 262144 - tokens) {
                usage("kv length + tokens exceeds the 262144 visible-token envelope");
            }
        }
    }
    if (options.profile &&
        (options.entry == Entry::All || options.geometry == GeometryChoice::All ||
         options.tokens.size() != 1 || options.kv_lengths.size() != 1 ||
         options.execution == Execution::Both || options.cache == CacheMode::Both)) {
        usage("--profile requires one entry, geometry, T, L, execution, and cache state");
    }
    return options;
}

DeviceBuffer make_fp16(std::size_t n, std::uint64_t seed, float low, float high) {
    DeviceBuffer result(n * sizeof(__half));
    CUDA_CHECK(bench::fixture::fill_values(static_cast<__half*>(result.p), n, seed, low, high));
    CUDA_CHECK(cudaDeviceSynchronize());
    return result;
}

// A coprime stride visits every block of [0,count) exactly once, so the fixture's selection is a
// distinct-id subset of the visible blocks with no host-side shuffle buffer.
std::int32_t permutation_step(std::int32_t count) {
    if (count <= 2) { return 1; }
    std::int32_t step = count / 3 + 1;
    while (step < count && std::gcd(step, count) != 1) { ++step; }
    return step < count ? step : 1;
}

class Case {
public:
    Case(const Geometry& geometry, std::int32_t tokens, std::int32_t kv_len, std::int32_t batch,
         std::int32_t selected, DeviceExecutionView execution)
        : geometry_(geometry), tokens_(tokens), kv_len_(kv_len), batch_(batch),
          selected_(selected), execution_(execution), visible_(kv_len + tokens),
          pages_((visible_ + kPagedKVPageSize - 1) / kPagedKVPageSize),
          physical_pages_(pages_ + 3),
          index_keys_(bench::make_bf16(std::size_t{kIndexDim} * kIndexBlocksPerPage *
                                           physical_pages_,
                                       401U)),
          keys_(bench::make_bf16(std::size_t{kHeadDim} * kPagedKVPageSize * geometry.kv_heads *
                                     physical_pages_,
                                 403U)),
          values_(make_fp16(std::size_t{kHeadDim} * kPagedKVPageSize * geometry.kv_heads *
                                physical_pages_,
                            409U, -1.F, 1.F)),
          table_(std::size_t(pages_) * batch_ * sizeof(std::int32_t)),
          query_(bench::make_bf16(std::size_t{kHeadDim} * geometry.query_heads * tokens_ * batch_,
                                  101U)),
          gate_(bench::make_bf16(std::size_t{kHeadDim} * geometry.query_heads * tokens_ * batch_,
                                 103U)),
          index_query_(bench::make_bf16(
              std::size_t{kIndexDim} * kIndexHeads * tokens_ * batch_, 107U)),
          out_(bench::make_zeros(std::size_t{kHeadDim} * geometry.query_heads * tokens_ * batch_ * 2)),
          positions_(std::size_t(tokens_) * batch_ * sizeof(std::int32_t)),
          valid_(std::size_t(batch_) * sizeof(std::int32_t)),
          rows_(std::size_t(batch_) * sizeof(std::int32_t)),
          block_ids_(std::size_t(selected_) * tokens_ * batch_ * sizeof(std::int32_t)),
          block_counts_(std::size_t(tokens_) * batch_ * sizeof(std::int32_t)),
          index_envelope_{static_cast<std::uint32_t>(visible_),
                          static_cast<std::uint32_t>(selected_)},
          sparse_envelope_{static_cast<std::uint32_t>(visible_)},
          index_capacity_(ops::qsa_indexer_workspace_capacity_bytes(tokens_, tokens_, batch_,
                                                                    index_envelope_, execution_)),
          sparse_capacity_(ops::qsa_sparse_gqa_workspace_capacity_bytes(
              KvCacheStorage::BFloat16, geometry.query_heads, geometry.kv_heads, tokens_, tokens_,
              batch_, sparse_envelope_, execution_)),
          index_workspace_(std::max<std::size_t>(index_capacity_, 256)),
          sparse_workspace_(std::max<std::size_t>(sparse_capacity_, 256)),
          index_keys_tensor_(index_keys_.p, DType::BF16,
                             {kIndexDim, kIndexBlocksPerPage, 1, physical_pages_}),
          keys_tensor_(keys_.p, DType::BF16,
                       {kHeadDim, kPagedKVPageSize, geometry.kv_heads, physical_pages_}),
          values_tensor_(values_.p, DType::FP16,
                         {kHeadDim, kPagedKVPageSize, geometry.kv_heads, physical_pages_}),
          table_tensor_(table_.p, DType::I32, {pages_, batch_}),
          query_tensor_(query_.p, DType::BF16,
                        {kHeadDim, geometry.query_heads, tokens_, batch_}),
          gate_tensor_(gate_.p, DType::BF16, {kHeadDim, geometry.query_heads, tokens_, batch_}),
          index_query_tensor_(index_query_.p, DType::BF16,
                              {kIndexDim, kIndexHeads, tokens_, batch_}),
          out_tensor_(out_.p, DType::BF16, {kHeadDim, geometry.query_heads, tokens_, batch_}),
          positions_tensor_(positions_.p, DType::I32, {tokens_, batch_}),
          valid_tensor_(valid_.p, DType::I32, {batch_}),
          rows_tensor_(rows_.p, DType::I32, {batch_}),
          block_ids_tensor_(block_ids_.p, DType::I32, {selected_, tokens_, batch_}),
          block_counts_tensor_(block_counts_.p, DType::I32, {tokens_, batch_}),
          index_cache_{index_keys_tensor_, table_tensor_},
          batch_view_{positions_tensor_, valid_tensor_, rows_tensor_},
          selection_{block_ids_tensor_, block_counts_tensor_} {
        main_cache_.k_pages      = keys_tensor_;
        main_cache_.v_pages      = values_tensor_;
        main_cache_.block_tables = table_tensor_;
        main_cache_.head_dim     = kHeadDim;
        main_cache_.num_kv_heads = geometry.kv_heads;
        main_cache_.storage      = KvCacheStorage::BFloat16;
        fill_host();
    }

    void launch(Entry entry, cudaStream_t stream) {
        const DeviceExecutionView view = execution_.on_stream(stream);
        if (entry != Entry::Sparse) {
            ops::qsa_indexer(index_query_tensor_, index_cache_, batch_view_, index_envelope_,
                             index_workspace_, selection_, view);
        }
        if (entry != Entry::Indexer) {
            ops::qsa_sparse_gqa(query_tensor_, gate_tensor_, main_cache_, batch_view_, selection_,
                                sparse_envelope_, sparse_workspace_, out_tensor_, view);
        }
    }

    // The public capacity query must bound the arena the Op actually uses; a mismatch means the
    // reported workspace column would be fiction, so it aborts instead of printing.
    void verify_workspace() const {
        if (index_workspace_.used() != 0 || sparse_workspace_.used() != 0 ||
            index_workspace_.peak_used() > index_capacity_ ||
            sparse_workspace_.peak_used() > sparse_capacity_) {
            throw std::runtime_error("QSA workspace capacity query does not bound the Op's use");
        }
    }

    [[nodiscard]] std::size_t workspace_bytes(Entry entry) const noexcept {
        if (entry == Entry::Indexer) { return index_capacity_; }
        if (entry == Entry::Sparse) { return sparse_capacity_; }
        return index_capacity_ + sparse_capacity_;
    }

    [[nodiscard]] std::size_t workspace_peak(Entry entry) const noexcept {
        if (entry == Entry::Indexer) { return index_workspace_.peak_used(); }
        if (entry == Entry::Sparse) { return sparse_workspace_.peak_used(); }
        return index_workspace_.peak_used() + sparse_workspace_.peak_used();
    }

    [[nodiscard]] const Geometry& geometry() const noexcept { return geometry_; }
    [[nodiscard]] std::int32_t tokens() const noexcept { return tokens_; }
    [[nodiscard]] std::int32_t kv_len() const noexcept { return kv_len_; }
    [[nodiscard]] std::int32_t selected() const noexcept { return selected_; }
    [[nodiscard]] std::int32_t batch() const noexcept { return batch_; }

    // Setup, never part of a timed interval: re-establish the fixture's own selection so a `sparse`
    // row never inherits the previous `indexer` row's device output and every entry measures the
    // same block set.
    void restore_selection() {
        block_ids_.copy_from_host(host_ids_.data(), block_ids_.bytes);
        block_counts_.copy_from_host(host_counts_.data(), block_counts_.bytes);
    }

    // Analytic work model; see the file header for what each column means and what it excludes.
    [[nodiscard]] Cost cost(Entry entry) const {
        Cost result;
        double sparse_keys = 0.0, dense_keys = 0.0, blocks = 0.0;
        double index_key_bytes = 0.0, index_flops = 0.0;
        for (std::int32_t token = 0; token < tokens_; ++token) {
            const std::int64_t position = kv_len_ + token;
            const std::int64_t scored   = (position + 1) / 4;
            const std::int64_t kept     = std::min<std::int64_t>(selected_, scored);
            sparse_keys += 4.0 * double(kept) + double(position + 1 - 4 * scored);
            dense_keys += double(position + 1);
            blocks += double(scored);
            index_key_bytes += double(scored) * kIndexDim * 2.0;
            index_flops += double(scored) * kIndexHeads * kIndexDim * 2.0;
        }
        const double rows = double(batch_);
        const double vectors_bytes = 2.0 * kHeadDim * 2.0;  // BF16 K + FP16 V per token, per KV head
        const double qsa_io_bytes =
            3.0 * double(kHeadDim) * double(geometry_.query_heads) * 2.0 * double(tokens_) * rows;
        const double selector_bytes =
            rows * (index_key_bytes + double(tokens_) *
                                          (double(kIndexHeads) * kIndexDim * 2.0 +
                                           double(selected_) * 4.0 + 4.0));
        const double attention_kv_bytes = sparse_keys * rows * geometry_.kv_heads * vectors_bytes;
        const double attention_flops =
            sparse_keys * rows * 4.0 * double(kHeadDim) * double(geometry_.query_heads);
        const double selector_flops = rows * index_flops;

        result.dense_kv_bytes = dense_keys * rows * geometry_.kv_heads * vectors_bytes;
        result.full_bytes     = result.dense_kv_bytes + qsa_io_bytes;
        result.full_flops =
            dense_keys * rows * 4.0 * double(kHeadDim) * double(geometry_.query_heads);
        result.attended_keys = entry == Entry::Indexer ? 0.0 : sparse_keys / double(tokens_);
        result.scoped_blocks = blocks / double(tokens_);
        switch (entry) {
        case Entry::Indexer:
            result.qsa_bytes   = selector_bytes;
            result.qsa_flops   = selector_flops;
            result.cache_bytes = rows * index_key_bytes;
            break;
        case Entry::Sparse:
            result.qsa_bytes   = attention_kv_bytes + qsa_io_bytes;
            result.qsa_flops   = attention_flops;
            result.cache_bytes = attention_kv_bytes;
            break;
        case Entry::Chain:
        case Entry::All:
            result.qsa_bytes   = selector_bytes + attention_kv_bytes + qsa_io_bytes;
            result.qsa_flops   = selector_flops + attention_flops;
            result.cache_bytes = rows * index_key_bytes + attention_kv_bytes;
            break;
        }
        return result;
    }

private:
    void fill_host() {
        std::vector<std::int32_t> positions(std::size_t(tokens_) * batch_, 0);
        std::vector<std::int32_t> valid(batch_, tokens_);
        std::vector<std::int32_t> rows(batch_, 0);
        std::vector<std::int32_t> table(std::size_t(pages_) * batch_, 0);
        host_ids_.assign(std::size_t(selected_) * tokens_ * batch_, 0);
        host_counts_.assign(std::size_t(tokens_) * batch_, 0);
        for (std::int32_t row = 0; row < batch_; ++row) {
            rows[std::size_t(row)] = row;
            // Reversed page order per row: every logical page maps to a different physical page than
            // its neighbours, so the benchmark never leans on an identity mapping.
            for (std::int32_t page = 0; page < pages_; ++page) {
                table[std::size_t(row) * pages_ + std::size_t(page)] =
                    (physical_pages_ - 1 - page + 3 * row) % physical_pages_;
            }
            for (std::int32_t token = 0; token < tokens_; ++token) {
                const std::size_t slot      = std::size_t(row) * tokens_ + std::size_t(token);
                const std::int32_t position = kv_len_ + token;
                positions[slot]             = position;
                const std::int32_t blocks   = (position + 1) / 4;
                const std::int32_t count    = std::min(selected_, blocks);
                host_counts_[slot]          = count;
                const std::int32_t step     = permutation_step(blocks);
                for (std::int32_t index = 0; index < count; ++index) {
                    host_ids_[slot * std::size_t(selected_) + std::size_t(index)] =
                        static_cast<std::int32_t>((std::int64_t(index) * step) % blocks);
                }
            }
        }
        positions_.copy_from_host(positions.data(), positions_.bytes);
        valid_.copy_from_host(valid.data(), valid_.bytes);
        rows_.copy_from_host(rows.data(), rows_.bytes);
        table_.copy_from_host(table.data(), table_.bytes);
        restore_selection();
    }

    Geometry geometry_;
    std::int32_t tokens_;
    std::int32_t kv_len_;
    std::int32_t batch_;
    std::int32_t selected_;
    DeviceExecutionView execution_;
    std::int32_t visible_;
    std::int32_t pages_;
    std::int32_t physical_pages_;
    // Fixture selection kept on the host so `restore_selection()` can re-upload it between entries.
    std::vector<std::int32_t> host_ids_;
    std::vector<std::int32_t> host_counts_;

    DeviceBuffer index_keys_;
    DeviceBuffer keys_;
    DeviceBuffer values_;
    DeviceBuffer table_;
    DeviceBuffer query_;
    DeviceBuffer gate_;
    DeviceBuffer index_query_;
    DeviceBuffer out_;
    DeviceBuffer positions_;
    DeviceBuffer valid_;
    DeviceBuffer rows_;
    DeviceBuffer block_ids_;
    DeviceBuffer block_counts_;

    ops::QsaIndexerExecutionEnvelope index_envelope_;
    ops::QsaSparseGqaExecutionEnvelope sparse_envelope_;
    std::size_t index_capacity_;
    std::size_t sparse_capacity_;
    WorkspaceArena index_workspace_;
    WorkspaceArena sparse_workspace_;

    Tensor index_keys_tensor_;
    Tensor keys_tensor_;
    Tensor values_tensor_;
    Tensor table_tensor_;
    Tensor query_tensor_;
    Tensor gate_tensor_;
    Tensor index_query_tensor_;
    Tensor out_tensor_;
    Tensor positions_tensor_;
    Tensor valid_tensor_;
    Tensor rows_tensor_;
    Tensor block_ids_tensor_;
    Tensor block_counts_tensor_;

    ops::QsaIndexCacheView index_cache_;
    PagedKVBatchLayerView main_cache_;
    ops::QsaBatchView batch_view_;
    ops::QsaBlockSelection selection_;
};

const char* entry_name(Entry entry) {
    switch (entry) {
    case Entry::Indexer: return "indexer";
    case Entry::Sparse: return "sparse";
    case Entry::Chain: return "chain";
    case Entry::All: break;
    }
    return "all";
}

const char* execution_name(Execution execution) {
    return execution == Execution::Eager ? "eager" : "graph";
}

const char* cache_name(CacheState cache) { return cache == CacheState::Cold ? "cold" : "warm"; }

std::vector<Geometry> selected_geometries(GeometryChoice choice) {
    if (choice == GeometryChoice::H24Kv2) { return {kH24Kv2}; }
    if (choice == GeometryChoice::H12Kv1) { return {kH12Kv1}; }
    return {kH24Kv2, kH12Kv1};
}

std::vector<Entry> selected_entries(Entry choice) {
    if (choice == Entry::All) { return {Entry::Indexer, Entry::Sparse, Entry::Chain}; }
    return {choice};
}

bench::ColdTiming measure(Case& data, Entry entry, Execution execution, CacheState cache,
                          bench::TimedGraph* graph, bench::L2FlushBuffer& flush,
                          cudaStream_t stream, int warmup, int repeat) {
    if (execution == Execution::Eager) {
        const auto launch = [&](cudaStream_t launch_stream) { data.launch(entry, launch_stream); };
        return cache == CacheState::Cold
                   ? bench::measure_cold_launch(launch, flush, stream, warmup, repeat)
                   : bench::measure_launch(launch, stream, warmup, repeat);
    }
    return cache == CacheState::Cold
               ? bench::measure_cold_graph(*graph, flush, stream, warmup, repeat)
               : bench::measure_graph(*graph, stream, warmup, repeat);
}

void report(const Result& result) {
    const double seconds = result.timing.median_us * 1.0e-6;
    const double gbps    = result.cost.qsa_bytes / seconds / 1.0e9;
    const double tflops  = result.cost.qsa_flops / seconds / 1.0e12;
    const double share   = result.cost.dense_kv_bytes > 0.0
                               ? result.cost.cache_bytes / result.cost.dense_kv_bytes * 100.0
                               : 0.0;
    std::printf("entry=%-7s shape=%-13s execution=%-5s cache=%-4s T=%5d L=%6d S=%3d B=%d "
                "attended=%7.1f blocks=%7.1f workspace=%8zu peak=%8zu "
                "median=%10.3f us min=%10.3f us p95=%10.3f us logical=%9.1f GB/s ",
                entry_name(result.entry), result.geometry.shape,
                execution_name(result.execution), cache_name(result.cache), result.tokens,
                result.kv_len, result.selected, result.batch, result.cost.attended_keys,
                result.cost.scoped_blocks, result.workspace_bytes, result.workspace_peak,
                result.timing.median_us, result.timing.min_us, result.timing.p95_us, gbps);
    const double peak = bench::device_peak_bw_gbs();
    if (peak > 0.0) {
        std::printf("(%5.1f%% of %.0f) ", gbps / peak * 100.0, peak);
    } else {
        std::printf("(roofline n/a) ");
    }
    std::printf("math=%7.3f TFLOP/s | full_ref= %.1f MB / %.2f GFLOP kv_share=%5.1f%%\n", tflops,
                result.cost.full_bytes / 1.0e6, result.cost.full_flops / 1.0e9, share);
}

void write_csv(const Options& options, const std::vector<Result>& results) {
    if (options.csv_out.empty()) { return; }
    const std::filesystem::path path(options.csv_out);
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream output(path);
    if (!output) { throw std::runtime_error("failed to open CSV output"); }
    output << "entry,shape,execution,cache,tokens,kv_len,selected_blocks,batch,workspace_bytes,"
              "workspace_peak_bytes,attended_keys,scoped_blocks,cache_bytes,dense_kv_bytes,"
              "logical_bytes,logical_flops,full_bytes,full_flops,gbs,tflops,median_us,min_us,p95_us\n";
    for (const Result& result : results) {
        const double seconds = result.timing.median_us * 1.0e-6;
        output << entry_name(result.entry) << ',' << result.geometry.shape << ','
               << execution_name(result.execution) << ',' << cache_name(result.cache) << ','
               << result.tokens << ',' << result.kv_len << ',' << result.selected << ','
               << result.batch << ',' << result.workspace_bytes << ',' << result.workspace_peak
               << ',' << result.cost.attended_keys << ',' << result.cost.scoped_blocks << ','
               << result.cost.cache_bytes << ',' << result.cost.dense_kv_bytes << ','
               << result.cost.qsa_bytes << ',' << result.cost.qsa_flops << ','
               << result.cost.full_bytes << ',' << result.cost.full_flops << ','
               << result.cost.qsa_bytes / seconds / 1.0e9 << ','
               << result.cost.qsa_flops / seconds / 1.0e12 << ',' << result.timing.median_us << ','
               << result.timing.min_us << ',' << result.timing.p95_us << '\n';
    }
}

void profile(Case& data, Entry entry, const Options& options, bench::L2FlushBuffer& flush,
             cudaStream_t stream) {
    const Execution execution = options.execution;
    const CacheState cache = options.cache == CacheMode::Cold ? CacheState::Cold : CacheState::Warm;
    bench::TimedGraph graph;
    if (execution == Execution::Graph) {
        data.launch(entry, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        graph.capture(stream,
                      [&](cudaStream_t launch_stream) { data.launch(entry, launch_stream); });
        for (int index = 0; index < options.warmup; ++index) { graph.launch(stream); }
    } else {
        for (int index = 0; index < options.warmup; ++index) { data.launch(entry, stream); }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (cache == CacheState::Cold) {
        bench::flush_l2(flush, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    std::printf("PROFILE entry=%s shape=%s dispatch=public execution=%s cache=%s T=%d L=%d S=%d "
                "B=%d\n",
                entry_name(entry), data.geometry().shape, execution_name(execution),
                cache_name(cache), data.tokens(), data.kv_len(), data.selected(), data.batch());
    std::fflush(stdout);
    CUDA_CHECK(cudaProfilerStart());
    if (execution == Execution::Graph)
        graph.launch(stream);
    else
        data.launch(entry, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaProfilerStop());
}

} // namespace

int main(int argc, char** argv) {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::printf("SKIP: no usable CUDA device\n");
            return 0;
        }
        const Options options = parse_options(argc, argv);
        DeviceContext device;
        const DeviceExecutionView execution = device.execution_view();
        const cudaStream_t stream           = execution.stream;
        bench::L2FlushBuffer flush(kFlushBytes);
        const std::vector<Geometry> geometries = selected_geometries(options.geometry);
        const std::vector<Entry> entries       = selected_entries(options.entry);

        if (options.profile) {
            Case data(geometries.front(), options.tokens.front(), options.kv_lengths.front(),
                      options.batch, options.selected, execution);
            profile(data, entries.front(), options, flush, stream);
            return 0;
        }

        std::vector<Result> results;
        for (const Geometry& geometry : geometries) {
            for (const std::int32_t kv_len : options.kv_lengths) {
                for (const std::int32_t tokens : options.tokens) {
                    Case data(geometry, tokens, kv_len, options.batch, options.selected,
                              execution);
                    for (const Entry entry : entries) {
                        data.restore_selection();
                        bench::TimedGraph graph;
                        if (options.execution != Execution::Eager) {
                            data.launch(entry, stream);
                            CUDA_CHECK(cudaStreamSynchronize(stream));
                            graph.capture(stream, [&](cudaStream_t launch_stream) {
                                data.launch(entry, launch_stream);
                            });
                        }
                        for (const Execution mode : {Execution::Eager, Execution::Graph}) {
                            if ((options.execution == Execution::Eager &&
                                 mode != Execution::Eager) ||
                                (options.execution == Execution::Graph &&
                                 mode != Execution::Graph)) {
                                continue;
                            }
                            for (const CacheState cache : {CacheState::Cold, CacheState::Warm}) {
                                if ((options.cache == CacheMode::Cold &&
                                     cache != CacheState::Cold) ||
                                    (options.cache == CacheMode::Warm &&
                                     cache != CacheState::Warm)) {
                                    continue;
                                }
                                data.verify_workspace();
                                const bench::ColdTiming timing =
                                    measure(data, entry, mode, cache, &graph, flush, stream,
                                            options.warmup, options.repeat);
                                data.verify_workspace();
                                Result result{entry,
                                              geometry,
                                              mode,
                                              cache,
                                              tokens,
                                              kv_len,
                                              options.selected,
                                              options.batch,
                                              data.workspace_bytes(entry),
                                              data.workspace_peak(entry),
                                              data.cost(entry),
                                              timing};
                                report(result);
                                results.push_back(result);
                            }
                        }
                    }
                }
            }
        }
        write_csv(options, results);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_qsa_bench: %s\n", error.what());
        return 1;
    }
}
