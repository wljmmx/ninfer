#include "models/qwen4_exp/ple/row_reader.h"

#include "models/qwen4_exp/ple/read_queue.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace ninfer::models::qwen4_exp::ple_detail {
namespace {

constexpr std::uint32_t none = std::numeric_limits<std::uint32_t>::max();

std::uint64_t mix(std::uint64_t word) {
    word ^= word >> 30;
    word *= 0xbf58476d1ce4e5b9ULL;
    word ^= word >> 27;
    word *= 0x94d049bb133111ebULL;
    return word ^ (word >> 31);
}

std::uint32_t capacity(std::uint64_t value) {
    if (!value || value >= none / 2) throw std::invalid_argument("PLE read capacity is too large");
    return static_cast<std::uint32_t>(value);
}

// Fixed index into separately owned records. Backshift deletion avoids accumulating tombstones.
class Index {
public:
    explicit Index(std::size_t size) : slots_(std::bit_ceil(size * 2), none) {}

    template <class Match>
    std::uint32_t find(std::uint64_t hash, Match match) const {
        for (auto p = hash & mask(); slots_[p] != none; p = (p + 1) & mask())
            if (match(slots_[p])) return slots_[p];
        return none;
    }

    void insert(std::uint64_t hash, std::uint32_t value) {
        auto p = hash & mask();
        while (slots_[p] != none) p = (p + 1) & mask();
        slots_[p] = value;
    }

    template <class Hash>
    void erase(std::uint64_t hash, std::uint32_t value, Hash hash_of) {
        auto gap = hash & mask();
        while (slots_[gap] != value) {
            if (slots_[gap] == none) throw std::logic_error("missing PLE index entry");
            gap = (gap + 1) & mask();
        }
        for (auto p = (gap + 1) & mask(); slots_[p] != none; p = (p + 1) & mask()) {
            const auto home = hash_of(slots_[p]) & mask();
            if (((p - home) & mask()) >= ((p - gap) & mask())) {
                slots_[gap] = slots_[p];
                gap         = p;
            }
        }
        slots_[gap] = none;
    }

    std::size_t bytes() const { return slots_.size() * sizeof(slots_[0]); }

private:
    std::size_t mask() const { return slots_.size() - 1; }

    std::vector<std::uint32_t> slots_;
};

struct CacheEntry {
    std::uint32_t row = none;
    std::array<std::byte, kPleMaxRowBytes> bytes;
};

struct alignas(128) CacheSet {
    std::array<CacheEntry, 4> ways;
    std::uint32_t next = 0;
};

} // namespace

struct RowReader::Impl {
    struct Target {
        std::uint32_t row = none, previous = none, next = none;
        bool active = false;
    };

    struct Row {
        std::uint32_t id = 0, head = none, parts = 0, active = 0;
        // The row's own source (object, encoding, per-part spans) so its decode never re-derives
        // the geometry, and `encoded` bytes of it were actually copied in.
        PleRowSource source;
        std::size_t encoded = 0;
        std::array<std::byte, kPleMaxRowBytes> bytes;
        std::exception_ptr error;
        bool ready = false, abandoned = false, indexed = false;
    };

    struct Fragment {
        std::uint32_t page = none, next = none, row = none, page_offset = 0;
        std::uint16_t row_offset = 0, bytes = 0;
    };

    struct Page {
        std::uint64_t offset = 0;
        std::uint32_t file = 0, head = none, previous = none, next = none;
        bool queued = false, required = false;
    };

    struct Owner {
        std::span<std::uint16_t> output;
        std::vector<std::uint32_t> targets;
        std::size_t pending = 0, unready = 0;
        std::exception_ptr error;
        bool required = false, bound = false;
    };

    const PleTableSource& table;
    std::uint32_t owner_rows, part_capacity, read_bytes = 4096;
    std::vector<Owner> owners;
    std::vector<Target> targets;
    std::vector<Row> rows;
    std::vector<Fragment> fragments;
    std::vector<Page> pages;
    std::vector<std::uint32_t> free_rows, free_pages, inflight, ready_rows;
    std::size_t ready_begin = 0, ready_end = 0;
    std::array<std::uint32_t, 2> queue_head{none, none}, queue_tail{none, none};
    Index row_index, page_index;
    std::vector<CacheSet> cache;
    std::vector<artifact::ReadSegment> segments;
    AlignedSlab slab;
    std::vector<std::span<std::byte>> buffers;
    std::vector<ReadCompletion> completions;
    PleReadStats stats;
    std::unique_ptr<ReadQueue> queue;

    Impl(const PleTableSource& table, std::uint32_t count, std::uint32_t per_owner,
         std::uint32_t depth, std::size_t cache_bytes)
        : table(table), owner_rows(per_owner),
          // Further segments start at a V3 file's 4 KiB-aligned payload prefix, so only the
          // first segment of one row can straddle a read page. A row is one span per encoding
          // part (a row_split q4 row is a code plane plus a scale plane), and each part can span
          // at most one page boundary in each of its segments.
          part_capacity(std::min<std::size_t>(
              kPleMaxRowBytes,
              std::size_t(table.row_parts()) * (table.max_row_segments() + 1))),
          owners(count), targets(capacity(std::uint64_t(count) * per_owner)), rows(targets.size()),
          fragments(capacity(rows.size() * part_capacity)), pages(fragments.size()),
          inflight(depth, none), ready_rows(rows.size()), row_index(rows.size()),
          page_index(pages.size()), cache(cache_bytes / sizeof(CacheSet)),
          segments(table.max_row_segments()), slab(std::size_t(depth) * read_bytes, read_bytes),
          completions(depth) {
        if (!depth || depth > 65535 || !table.file_count() || (cache_bytes && cache.empty()))
            throw std::invalid_argument("invalid PLE row reader resources");
        for (auto& owner : owners) owner.targets.reserve(per_owner);
        free_rows.resize(rows.size());
        free_pages.resize(pages.size());
        std::iota(free_rows.rbegin(), free_rows.rend(), 0);
        std::iota(free_pages.rbegin(), free_pages.rend(), 0);
        buffers.reserve(depth);
        for (std::uint32_t i = 0; i < depth; ++i)
            buffers.push_back(std::span<std::byte>(slab.data() + std::size_t(i) * read_bytes,
                                                   read_bytes));
        queue = std::make_unique<ReadQueue>(table.backing(),
                                            std::span<std::byte>(slab.data(), slab.size()), depth,
                                            read_bytes);
    }

    Owner& owner(std::uint32_t id) {
        if (id >= owners.size() || !owners[id].bound)
            throw std::invalid_argument("unbound PLE row owner");
        return owners[id];
    }

    std::uint64_t page_hash(std::uint32_t id) const {
        return mix(pages[id].offset ^ (std::uint64_t(pages[id].file) << 32));
    }

    void unlink_page(std::uint32_t id) {
        auto& p      = pages[id];
        const auto q = unsigned(p.required);
        if (p.previous == none)
            queue_head[q] = p.next;
        else
            pages[p.previous].next = p.next;
        if (p.next == none)
            queue_tail[q] = p.previous;
        else
            pages[p.next].previous = p.previous;
        p.previous = p.next = none;
        p.queued            = false;
    }

    void queue_page(std::uint32_t id) {
        auto& p      = pages[id];
        const auto q = unsigned(p.required);
        p.previous   = queue_tail[q];
        p.next       = none;
        if (p.previous == none)
            queue_head[q] = id;
        else
            pages[p.previous].next = id;
        queue_tail[q] = id;
        p.queued      = true;
    }

    void promote(std::uint32_t id) {
        auto& p = pages[id];
        if (p.required) return;
        const bool queued = p.queued;
        if (queued) unlink_page(id);
        p.required = true;
        if (queued) queue_page(id);
    }

    bool wanted(std::uint32_t page) const {
        for (auto f = pages[page].head; f != none; f = fragments[f].next)
            if (rows[fragments[f].row].active) return true;
        return false;
    }

    const CacheEntry* cached(std::uint32_t row) const {
        if (cache.empty()) return nullptr;
        for (const auto& entry : cache[mix(row) % cache.size()].ways)
            if (entry.row == row) return &entry;
        return nullptr;
    }

    void cache_row(const Row& row) {
        if (cache.empty()) return;
        auto& set = cache[mix(row.id) % cache.size()];
        for (auto& entry : set.ways)
            if (entry.row == row.id) return;
        auto& entry = set.ways[set.next++ % set.ways.size()];
        entry.row   = row.id;
        entry.bytes = row.bytes;
    }

    void unindex(std::uint32_t id) {
        auto& row = rows[id];
        if (!row.indexed) return;
        row_index.erase(mix(row.id), id, [&](auto i) { return mix(rows[i].id); });
        row.indexed = false;
    }

    void finish_page(std::uint32_t id, const std::byte* data, std::size_t bytes,
                     std::exception_ptr error = {}) {
        auto part = pages[id].head;
        while (part != none) {
            auto& f   = fragments[part];
            auto& row = rows[f.row];
            if (!data && !error) {
                row.abandoned = true;
                unindex(f.row);
            } else if (error || std::size_t(f.page_offset) + f.bytes > bytes) {
                if (!row.error)
                    row.error = error ? error
                                      : std::make_exception_ptr(std::runtime_error(
                                            "PLE read ended before required row bytes"));
            } else {
                std::memcpy(row.bytes.data() + f.row_offset, data + f.page_offset, f.bytes);
            }
            if (--row.parts == 0) ready_rows[ready_end++ % ready_rows.size()] = f.row;
            f.page = none;
            part   = f.next;
        }
        page_index.erase(page_hash(id), id, [&](auto i) { return page_hash(i); });
        pages[id] = {};
        free_pages.push_back(id);
    }

    void decode(std::uint32_t id) {
        auto& row = rows[id];
        std::array<std::uint16_t, kPleRowElements> decoded;
        if (!row.abandoned && !row.error) {
            try {
                decode_ple_row(row.source, std::span(row.bytes.data(), row.encoded), decoded);
                cache_row(row);
            } catch (...) { row.error = std::current_exception(); }
        }
        if (row.error || row.abandoned) unindex(id);
        for (auto t = row.head; t != none; t = targets[t].next) {
            auto& target = targets[t];
            auto& owner  = owners[t / owner_rows];
            --owner.unready;
            if (!target.active) continue;
            if (row.error)
                owner.error = row.error;
            else if (row.abandoned)
                throw std::logic_error("active PLE row was abandoned");
            else
                std::copy(decoded.begin(), decoded.end(),
                          owner.output.begin() + std::size_t(t % owner_rows) * kPleRowElements);
            --owner.pending;
            target.active = false;
        }
        row.active = 0;
        row.ready  = true;
    }

    void pump() {
        const auto n = queue->poll(completions);
        for (std::size_t i = 0; i < n; ++i) {
            const auto& c = completions[i];
            const auto id = inflight[c.buffer];
            if (id == none || c.tag != id) throw std::logic_error("unmatched PLE page completion");
            stats.read_bytes += c.bytes;
            finish_page(id, buffers[c.buffer].data(), c.bytes, c.error);
            inflight[c.buffer] = none;
        }
        std::size_t skipped = 0;
        for (std::uint32_t buffer = 0; buffer < buffers.size(); ++buffer) {
            if (inflight[buffer] != none) continue;
            while (queue_head[0] != none || queue_head[1] != none) {
                const auto id     = queue_head[1] != none ? queue_head[1] : queue_head[0];
                const auto p      = pages[id];
                const bool needed = wanted(id);
                unlink_page(id);
                if (!needed) {
                    finish_page(id, nullptr, 0);
                    if (++skipped == 256) break;
                    continue;
                }
                if (queue->try_enqueue({p.file, buffer, p.offset, id}) != ReadAdmission::Accepted)
                    throw std::logic_error("PLE read slot admission failed");
                inflight[buffer] = id;
                ++stats.reads;
                break;
            }
            if (skipped == 256) break;
        }
        queue->submit();
        for (std::size_t budget = 0; ready_begin != ready_end && budget < 128; ++budget)
            decode(ready_rows[ready_begin++ % ready_rows.size()]);
    }
};

RowReader::RowReader(const PleTableSource& table, std::uint32_t owners,
                     std::uint32_t rows_per_owner, std::uint32_t io_depth, std::size_t cache_bytes)
    : impl_(std::make_unique<Impl>(table, owners, rows_per_owner, io_depth, cache_bytes)) {}

RowReader::~RowReader() = default;

void RowReader::begin(std::uint32_t id, std::span<std::uint16_t> output, bool required) {
    auto& p = *impl_;
    if (id >= p.owners.size() || p.owners[id].bound || output.empty() ||
        output.size() % kPleRowElements || output.size() / kPleRowElements > p.owner_rows)
        throw std::invalid_argument("invalid PLE row output");
    auto& o    = p.owners[id];
    o.output   = output;
    o.required = required;
    o.bound    = true;
    o.error    = {};
}

void RowReader::request(std::uint32_t owner_id, std::uint32_t destination, std::uint32_t row_id) {
    auto& p     = *impl_;
    auto& owner = p.owner(owner_id);
    if (destination >= owner.output.size() / kPleRowElements || row_id >= p.table.rows())
        throw std::invalid_argument("invalid PLE row request");
    const auto target_id = owner_id * p.owner_rows + destination;
    auto& target         = p.targets[target_id];
    if (target.row != none) throw std::invalid_argument("PLE destination already requested");
    ++p.stats.rows;
    auto id = p.row_index.find(mix(row_id), [&](auto i) { return p.rows[i].id == row_id; });
    // A row this request creates still needs its pages planned; a row it joins only needs the pages
    // of the work already in flight promoted.
    const bool fresh = id == none;
    if (fresh) {
        if (p.free_rows.empty()) throw std::logic_error("PLE row capacity exhausted");
        id = p.free_rows.back();
        p.free_rows.pop_back();
        auto& row         = p.rows[id];
        const auto source = p.table.row(row_id);
        row.id            = row_id;
        row.source        = source;
        row.encoded       = source.encoded_bytes();
        row.indexed       = true;
        p.row_index.insert(mix(row_id), id);
        if (const auto* hit = p.cached(row_id)) {
            row.bytes = hit->bytes;
            row.ready = true;
            ++p.stats.cache_hits;
        }
    } else {
        ++p.stats.shared_rows;
    }
    auto& row = p.rows[id];
    target    = {id, none, row.head, !row.ready};
    if (row.head != none) p.targets[row.head].previous = target_id;
    row.head = target_id;
    owner.targets.push_back(target_id);
    if (row.ready) {
        decode_ple_row(row.source, std::span(row.bytes.data(), row.encoded),
                       owner.output.subspan(std::size_t(destination) * kPleRowElements,
                                            kPleRowElements));
        return;
    }
    // This target waits for the row. The row must be visible as active BEFORE its pages are queued:
    // the pump runs on the worker thread and retires a page whose rows are all inactive, which
    // would leave this row to be decoded from bytes that never arrived. A row_split row queues two
    // spans, so the window that one part left open is now wide enough to hit.
    ++row.active;
    ++owner.pending;
    ++owner.unready;
    if (!fresh) {
        if (owner.required)
            for (std::uint32_t f = 0; f < p.part_capacity; ++f) {
                const auto page = p.fragments[id * p.part_capacity + f].page;
                if (page != none) p.promote(page);
            }
        return;
    }
    // One row is one span per encoding part, and each span is planned for its own page boundary.
    // The fragment's destination offset runs across the parts, so the row's bytes land in the order
    // the decoder reads them: a row_split q4 row is its code plane followed by its scale plane.
    std::uint32_t part       = 0;
    std::uint32_t row_offset = 0;
    for (std::uint32_t source_part = 0; source_part < row.source.parts; ++source_part) {
        const auto count =
            artifact::map_file_range(row.source.object, row.source.part_offsets[source_part],
                                     row.source.part_bytes[source_part], p.segments);
        for (std::size_t s = 0; s < count; ++s) {
            const auto& segment  = p.segments[s];
            std::uint64_t copied = 0;
            while (copied < segment.bytes) {
                const auto position = segment.file_offset + copied;
                const auto offset   = position / p.read_bytes * p.read_bytes;
                const auto prefix   = std::uint32_t(position - offset);
                const auto bytes =
                    std::min<std::uint64_t>(p.read_bytes - prefix, segment.bytes - copied);
                const auto hash = mix(offset ^ (std::uint64_t(segment.file_index) << 32));
                auto page       = p.page_index.find(hash, [&](auto i) {
                    return p.pages[i].offset == offset && p.pages[i].file == segment.file_index;
                });
                if (page == none) {
                    if (p.free_pages.empty())
                        throw std::logic_error("PLE page capacity exhausted");
                    page = p.free_pages.back();
                    p.free_pages.pop_back();
                    p.pages[page].offset   = offset;
                    p.pages[page].file     = segment.file_index;
                    p.pages[page].required = owner.required;
                    p.page_index.insert(hash, page);
                    p.queue_page(page);
                } else {
                    ++p.stats.shared_pages;
                    if (owner.required) p.promote(page);
                }
                if (part == p.part_capacity)
                    throw std::logic_error("PLE row fragment capacity exceeded");
                const auto f   = id * p.part_capacity + part++;
                p.fragments[f] = {page,
                                  p.pages[page].head,
                                  id,
                                  prefix,
                                  std::uint16_t(row_offset + segment.destination_offset + copied),
                                  std::uint16_t(bytes)};
                p.pages[page].head = f;
                ++row.parts;
                copied += bytes;
            }
        }
        row_offset += row.source.part_bytes[source_part];
    }
}

void RowReader::cancel(std::uint32_t id, std::uint32_t begin, std::uint32_t end) {
    auto& p     = *impl_;
    auto& owner = p.owner(id);
    if (begin > end || end > owner.output.size() / kPleRowElements)
        throw std::invalid_argument("invalid PLE cancellation range");
    if (begin == end) return;
    for (auto t : owner.targets) {
        auto& target     = p.targets[t];
        const auto index = t % p.owner_rows;
        if (index >= begin && index < end && target.active) {
            target.active = false;
            --p.rows[target.row].active;
            --owner.pending;
            if (p.rows[target.row].active) continue;
            for (std::uint32_t f = 0; f < p.part_capacity; ++f) {
                const auto page = p.fragments[target.row * p.part_capacity + f].page;
                if (page == none || !p.pages[page].queued || p.wanted(page)) continue;
                p.unlink_page(page);
                p.finish_page(page, nullptr, 0);
            }
        }
    }
}

void RowReader::prioritize(std::uint32_t id) {
    auto& p        = *impl_;
    auto& owner    = p.owner(id);
    owner.required = true;
    for (auto t : owner.targets) {
        if (!p.targets[t].active) continue;
        for (std::uint32_t f = 0; f < p.part_capacity; ++f) {
            const auto page = p.fragments[p.targets[t].row * p.part_capacity + f].page;
            if (page != none) p.promote(page);
        }
    }
}

void RowReader::release(std::uint32_t id) {
    auto& p     = *impl_;
    auto& owner = p.owner(id);
    if (owner.pending || owner.unready) throw std::logic_error("PLE row owner still has writers");
    for (auto t : owner.targets) {
        const auto target = p.targets[t];
        auto& row         = p.rows[target.row];
        if (target.previous == none)
            row.head = target.next;
        else
            p.targets[target.previous].next = target.next;
        if (target.next != none) p.targets[target.next].previous = target.previous;
        if (row.head == none) {
            p.unindex(target.row);
            row = {};
            p.free_rows.push_back(target.row);
        }
        p.targets[t] = {};
    }
    owner.targets.clear();
    owner.output = {};
    owner.error  = {};
    owner.bound  = false;
}

void RowReader::pump() { impl_->pump(); }

bool RowReader::runnable() const noexcept {
    const auto& p = *impl_;
    return p.ready_begin != p.ready_end || ((p.queue_head[0] != none || p.queue_head[1] != none) &&
                                            p.queue->outstanding() < p.buffers.size());
}

void RowReader::wait() { impl_->queue->wait(); }

void RowReader::wake() { impl_->queue->wake(); }

std::size_t RowReader::pending(std::uint32_t owner) const { return impl_->owner(owner).pending; }

bool RowReader::drained(std::uint32_t owner) const { return impl_->owner(owner).unready == 0; }

std::exception_ptr RowReader::error(std::uint32_t owner) const { return impl_->owner(owner).error; }

PleReadStats RowReader::stats() const noexcept { return impl_->stats; }

std::size_t RowReader::memory_bytes() const noexcept {
    const auto& p = *impl_;
    return sizeof(Impl) + p.owners.size() * sizeof(Impl::Owner) +
           p.targets.size() * (sizeof(Impl::Target) + sizeof(std::uint32_t)) +
           p.rows.size() * (sizeof(Impl::Row) + 2 * sizeof(std::uint32_t)) +
           p.fragments.size() * sizeof(Impl::Fragment) +
           p.pages.size() * (sizeof(Impl::Page) + sizeof(std::uint32_t)) + p.row_index.bytes() +
           p.page_index.bytes() + p.cache.size() * sizeof(CacheSet) +
           p.segments.size() * sizeof(artifact::ReadSegment) + p.slab.size() +
           p.buffers.size() * (sizeof(std::span<std::byte>) + sizeof(ReadCompletion));
}

} // namespace ninfer::models::qwen4_exp::ple_detail
