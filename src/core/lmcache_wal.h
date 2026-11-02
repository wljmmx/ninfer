#pragma once

// 13.4 / Q64: write replay for the L3 cache.
//
// The async writer's queue lived only in memory: a restart dropped every checkpoint it had accepted
// but not published, which is indistinguishable from "the request never captured anything". This is
// a write-ahead log for those jobs.
//
// One file per job, written before the job is enqueued and deleted after the publish returns. A
// file that survives a crash is exactly the set that was accepted and not confirmed, so replay is
// the same `publish_job` call on the next start. Files are self-describing and validated before
// allocation, because a truncated record must be a miss, not a 2 GB allocation.
//
// Off by default: NINFER_LMCACHE_WAL_DIR names the directory. Without it nothing here is touched
// and the backend is byte-for-byte the old one.

#include "ninfer/types.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace ninfer::core::lmcache {

class CheckpointWriteAheadLog {
public:
    CheckpointWriteAheadLog() = default;

    // Returns false when the directory cannot be used. The caller then runs without the log: a
    // cache that cannot persist is still a cache, and refusing to serve because a directory is
    // missing would turn an optimisation into a startup dependency.
    bool open(const std::string& dir) {
        if (dir.empty()) { return false; }
        std::error_code error;
        std::filesystem::create_directories(dir, error);
        if (error) { return false; }
        std::lock_guard lock(mutex_);
        dir_ = std::filesystem::path(dir);
        std::string path = dir_.string();
        std::fprintf(stderr, "[lmcache] write-ahead log: %s\n", path.c_str());
        std::fflush(stderr);
        return true;
    }

    [[nodiscard]] bool enabled() const noexcept { return !dir_.empty(); }
    [[nodiscard]] std::filesystem::path directory() const { return dir_; }

    // Persists the job and returns its path (empty when the log is off or the write failed). A
    // failed write is reported and the job still runs: the in-memory queue is the primary path.
    std::filesystem::path append(std::span<const TokenId> tokens,
                                 const CachedCheckpoint& checkpoint) {
        if (!enabled() || tokens.empty() || checkpoint.kv_bytes.empty() ||
            checkpoint.state_shards.empty()) {
            return {};
        }
        std::lock_guard lock(mutex_);
        std::filesystem::path path;
        for (;;) {
            const auto sequence = next_sequence_.fetch_add(1, std::memory_order_relaxed);
            char name[64];
            std::snprintf(name, sizeof(name), "%020llu.wal",
                          static_cast<unsigned long long>(sequence));
            path = dir_ / name;
            if (!std::filesystem::exists(path)) { break; }
        }
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) { return {}; }
        const auto write = [&out](const void* data, std::size_t bytes) {
            out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
        };
        constexpr std::uint32_t kMagic   = 0x4C41574EU; // "NWAL"
        constexpr std::uint32_t kVersion = 1;
        const std::uint32_t token_count  = static_cast<std::uint32_t>(tokens.size());
        const std::uint64_t kv_bytes     = checkpoint.kv_bytes.size();
        const std::uint32_t shards       = static_cast<std::uint32_t>(checkpoint.state_shards.size());
        write(&kMagic, sizeof(kMagic));
        write(&kVersion, sizeof(kVersion));
        write(&checkpoint.frontier, sizeof(checkpoint.frontier));
        write(&checkpoint.valid_tokens, sizeof(checkpoint.valid_tokens));
        const std::uint64_t page_bytes = checkpoint.page_bytes;
        write(&page_bytes, sizeof(page_bytes));
        write(&token_count, sizeof(token_count));
        write(&kv_bytes, sizeof(kv_bytes));
        write(&shards, sizeof(shards));
        for (const auto& shard : checkpoint.state_shards) {
            const std::uint64_t size = shard.size();
            write(&size, sizeof(size));
        }
        write(tokens.data(), tokens.size() * sizeof(TokenId));
        write(checkpoint.kv_bytes.data(), checkpoint.kv_bytes.size());
        for (const auto& shard : checkpoint.state_shards) { write(shard.data(), shard.size()); }
        out.flush();
        if (!out) {
            out.close();
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
            return {};
        }
        return path;
    }

    void complete(const std::filesystem::path& path) {
        if (path.empty()) { return; }
        std::lock_guard lock(mutex_);
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
    }

    struct LoadedRecord {
        std::filesystem::path path;
        std::vector<TokenId> tokens;
        CachedCheckpoint checkpoint;
    };

    // Every confirmed-looking record on disk, oldest name first. The caller publishes and then
    // calls complete(); a record that fails validation is removed here so one bad file cannot stop
    // every later start.
    [[nodiscard]] std::vector<LoadedRecord> pending() {
        std::vector<LoadedRecord> records;
        if (!enabled()) { return records; }
        std::vector<std::filesystem::path> files;
        {
            std::lock_guard lock(mutex_);
            std::error_code error;
            for (const auto& entry : std::filesystem::directory_iterator(dir_, error)) {
                if (error) { break; }
                if (entry.is_regular_file() && entry.path().extension() == ".wal") {
                    files.push_back(entry.path());
                }
            }
        }
        std::sort(files.begin(), files.end());
        for (const auto& path : files) {
            auto record = read(path);
            if (!record) {
                std::fprintf(stderr, "[lmcache] wal: dropping unreadable record %s\n",
                             path.string().c_str());
                std::fflush(stderr);
                std::error_code ignored;
                std::filesystem::remove(path, ignored);
                continue;
            }
            record->path = path;
            records.push_back(std::move(*record));
        }
        return records;
    }

    [[nodiscard]] std::optional<LoadedRecord> read(const std::filesystem::path& path) const {
        std::ifstream in(path, std::ios::binary);
        if (!in) { return std::nullopt; }
        const auto load = [&in](void* data, std::size_t bytes) {
            in.read(static_cast<char*>(data), static_cast<std::streamsize>(bytes));
            return static_cast<std::size_t>(in.gcount()) == bytes;
        };
        constexpr std::uint32_t kMagic   = 0x4C41574EU;
        constexpr std::uint32_t kVersion = 1;
        std::uint32_t magic = 0, version = 0, token_count = 0, shards = 0;
        LoadedRecord record;
        std::uint64_t kv_bytes = 0, page_bytes = 0;
        if (!load(&magic, sizeof(magic)) || magic != kMagic) { return std::nullopt; }
        if (!load(&version, sizeof(version)) || version != kVersion) { return std::nullopt; }
        if (!load(&record.checkpoint.frontier, sizeof(record.checkpoint.frontier)) ||
            !load(&record.checkpoint.valid_tokens, sizeof(record.checkpoint.valid_tokens)) ||
            !load(&page_bytes, sizeof(page_bytes)) || !load(&token_count, sizeof(token_count)) ||
            !load(&kv_bytes, sizeof(kv_bytes)) || !load(&shards, sizeof(shards))) {
            return std::nullopt;
        }
        // Bounds before allocation. A corrupt header must not become a multi-gigabyte request.
        if (token_count == 0 || token_count > (1U << 24U) || kv_bytes == 0 ||
            kv_bytes > (2ULL << 30U) || shards == 0 || shards > 256) {
            return std::nullopt;
        }
        std::vector<std::uint64_t> sizes(shards, 0);
        std::uint64_t total_state = 0;
        for (auto& size : sizes) {
            if (!load(&size, sizeof(size)) || size > (2ULL << 30U) ||
                total_state > (4ULL << 30U) - size) {
                return std::nullopt;
            }
            total_state += size;
        }
        record.tokens.resize(token_count);
        if (!load(record.tokens.data(), record.tokens.size() * sizeof(TokenId))) {
            return std::nullopt;
        }
        record.checkpoint.page_bytes = static_cast<std::size_t>(page_bytes);
        record.checkpoint.kv_bytes.resize(static_cast<std::size_t>(kv_bytes));
        if (!load(record.checkpoint.kv_bytes.data(), record.checkpoint.kv_bytes.size())) {
            return std::nullopt;
        }
        for (const auto size : sizes) {
            std::string shard(static_cast<std::size_t>(size), '\0');
            if (!load(shard.data(), shard.size())) { return std::nullopt; }
            record.checkpoint.state_shards.push_back(std::move(shard));
        }
        return record;
    }

private:
    std::filesystem::path dir_;
    mutable std::mutex mutex_;
    std::atomic<std::uint64_t> next_sequence_{
        static_cast<std::uint64_t>(
            std::chrono::system_clock::now().time_since_epoch().count() & 0x7fffffffULL) *
        1024ULL};
};

} // namespace ninfer::core::lmcache
