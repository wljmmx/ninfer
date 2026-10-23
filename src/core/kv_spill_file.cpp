#include "core/kv_spill_file.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <system_error>

namespace ninfer {
namespace {

[[noreturn]] void fail(const std::string& what, const std::filesystem::path& path) {
    throw std::runtime_error(what + ": " + path.string());
}

} // namespace

KVSpillFile::KVSpillFile(std::filesystem::path path, std::size_t capacity_bytes,
                         std::size_t extent_bytes)
    : path_(std::move(path)), extent_bytes_(extent_bytes) {
    if (extent_bytes == 0) { throw std::invalid_argument("KV spill extent must be positive"); }
    if (capacity_bytes < extent_bytes) {
        throw std::invalid_argument("KV spill file holds less than one extent");
    }
    if (path_.has_parent_path()) {
        std::error_code error;
        std::filesystem::create_directories(path_.parent_path(), error);
        if (error) { fail("KV spill directory", path_); }
    }
    const auto extents = capacity_bytes / extent_bytes;
    if (extents > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("KV spill file holds too many extents");
    }
    slots_.resize(extents);
    // Preallocate: a spill at capacity must not turn into an out-of-space failure halfway through
    // a decode step.
    file_.open(path_, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
    if (!file_) { fail("KV spill file open", path_); }
    file_.seekp(0, std::ios::beg);
    const std::vector<char> zero(1 << 20, 0);
    auto remaining = static_cast<std::size_t>(extents) * extent_bytes;
    while (remaining != 0) {
        const auto chunk = std::min(remaining, zero.size());
        file_.write(zero.data(), static_cast<std::streamsize>(chunk));
        remaining -= chunk;
    }
    file_.flush();
    if (!file_) { fail("KV spill file preallocate", path_); }
}

KVSpillFile::~KVSpillFile() {
    if (!file_.is_open()) { return; }
    file_.close();
    std::error_code error;
    // The file is a scratch tier, not a durable store: leaving it behind only wastes disk.
    std::filesystem::remove(path_, error);
}

std::uint32_t KVSpillFile::least_recently_used() const noexcept {
    std::uint32_t victim = 0;
    std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
    for (std::uint32_t slot = 0; slot < slots_.size(); ++slot) {
        if (!slots_[slot].occupied) { return slot; }
        if (slots_[slot].use_epoch < oldest) {
            oldest = slots_[slot].use_epoch;
            victim = slot;
        }
    }
    return victim;
}

std::uint32_t KVSpillFile::write(std::span<const std::byte> bytes, std::uint64_t use_epoch) {
    if (bytes.size() > extent_bytes_) {
        throw std::invalid_argument("KV spill image exceeds one extent");
    }
    const auto slot = least_recently_used();
    seek(slot, 0);
    file_.write(reinterpret_cast<const char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
    if (!file_) { fail("KV spill write", path_); }
    spilled_bytes_ += bytes.size();
    if (!slots_[slot].occupied) {
        slots_[slot].occupied = true;
        ++resident_;
    }
    slots_[slot].use_epoch = use_epoch;
    return slot;
}

void KVSpillFile::read(std::uint32_t slot, std::span<std::byte> out) {
    if (slot >= slots_.size() || !slots_[slot].occupied) {
        throw std::invalid_argument("KV spill read from an empty slot");
    }
    if (out.size() > extent_bytes_) {
        throw std::invalid_argument("KV spill read exceeds one extent");
    }
    seek(slot, 0);
    file_.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()));
    if (!file_) { fail("KV spill read", path_); }
    restored_bytes_ += out.size();
}

void KVSpillFile::release(std::uint32_t slot) noexcept {
    if (slot >= slots_.size() || !slots_[slot].occupied) { return; }
    slots_[slot].occupied = false;
    slots_[slot].use_epoch = 0;
    --resident_;
}

void KVSpillFile::seek(std::uint32_t slot, std::streamoff offset) {
    const auto position = static_cast<std::streamoff>(slot) *
                              static_cast<std::streamoff>(extent_bytes_) +
                          offset;
    file_.clear();
    file_.seekg(position, std::ios::beg);
    file_.seekp(position, std::ios::beg);
}

} // namespace ninfer
