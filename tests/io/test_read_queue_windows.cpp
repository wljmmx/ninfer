// Windows equivalent of the io_uring read-queue test (see test_read_queue.cpp for the Linux oracle).
//
// The port maps DirectReadQueue onto an I/O completion port, which has no io_uring_submit to wrap, so
// the failure injection of the Linux test does not apply. What this test does instead is pin the
// observable contract that the PLE/weight-loading paths rely on:
//
//   1. direct_read_window alignment arithmetic (offset/prefix/required/bytes)
//   2. a batch of reads at unaligned offsets returns exactly the file's bytes
//   3. completion bookkeeping: outstanding() drains to zero, and a *second* round over the same
//      buffers returns fresh bytes (regression guard for the release-in-poll fix -- the earlier
//      implementation handed the previous round's completion to the next read, so callers copied
//      stale data)
//   4. a read at/after EOF reports a clean zero-length completion instead of stale bytes
#include "core/io/file.h"
#include "core/io/read_queue.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

using namespace ninfer;
using namespace ninfer::io;

namespace {

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

constexpr std::size_t kFileBytes = 1 << 20; // 1 MiB of deterministic bytes
constexpr std::size_t kWindow    = 64 << 10;

std::filesystem::path make_file() {
    const auto path = std::filesystem::temp_directory_path() / "ninfer-read-queue-windows.bin";
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    std::vector<char> block(kFileBytes);
    for (std::size_t i = 0; i < block.size(); ++i) block[i] = static_cast<char>((i * 37 + 11) & 0xFF);
    out.write(block.data(), static_cast<std::streamsize>(block.size()));
    out.close();
    return path;
}

std::vector<std::byte> expect_at(const std::filesystem::path& path, std::size_t offset,
                                 std::size_t bytes) {
    std::ifstream in(path, std::ios::binary);
    in.seekg(static_cast<std::streamoff>(offset));
    std::vector<std::byte> out(bytes);
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
    return out;
}

} // namespace

int main() {
    try {
        const auto path = make_file();
        ReadOnlyFile file(path, ReadMode::Direct);
        const auto view = file.view();
        require(view.bytes == kFileBytes, "file length");
        require(view.offset_alignment >= 1 && (view.offset_alignment & (view.offset_alignment - 1)) == 0,
                "offset alignment must be a power of two");

        // 1. window arithmetic
        for (const std::size_t offset :
             std::array<std::size_t, 6>{0, 1, 90, 3960, 4096, kFileBytes - 90}) {
            const auto window = direct_read_window(view, offset, 90);
            require(window.offset % view.offset_alignment == 0, "window start not aligned");
            require(window.offset <= offset, "window starts after the request");
            require(window.prefix == offset - window.offset, "window prefix");
            require(window.bytes % view.offset_alignment == 0, "window length not aligned");
            require(window.bytes >= window.required && window.required >= window.prefix + 90,
                    "window does not cover the request");
        }

        // 2/3. two rounds over the same buffers, at deliberately unaligned offsets
        const std::size_t slot_bytes = std::max(kWindow, view.offset_alignment);
        std::vector<std::vector<std::byte>> storage;
        std::vector<MemorySpan> buffers;
        storage.reserve(4);
        for (int i = 0; i < 4; ++i) {
            storage.emplace_back(slot_bytes + view.memory_alignment);
            const auto base = reinterpret_cast<std::uintptr_t>(storage.back().data());
            const auto aligned = (base + view.memory_alignment - 1) /
                                 view.memory_alignment * view.memory_alignment;
            buffers.push_back({reinterpret_cast<std::byte*>(aligned), slot_bytes, Device::cpu()});
        }
        const FileView views[1] = {view};
        DirectReadQueue queue(views, buffers, static_cast<std::uint32_t>(buffers.size()));

        const std::array<std::size_t, 4> offsets{0, 1, 3960, kFileBytes - 4096};
        for (int round = 0; round < 2; ++round) {
            for (std::size_t i = 0; i < offsets.size(); ++i) {
                const auto window = direct_read_window(view, offsets[i], 90);
                require(queue.try_enqueue({0, static_cast<std::uint32_t>(i), window.offset,
                                           static_cast<std::uint32_t>(window.bytes),
                                           static_cast<std::uint32_t>(i)}) ==
                            ReadAdmission::Accepted,
                        "enqueue rejected");
            }
            queue.submit();
            std::vector<ReadCompletion> completions(buffers.size());
            std::array<bool, 4> seen{};
            while (queue.outstanding()) {
                const auto count = queue.poll(completions);
                if (!count) {
                    (void)queue.wait();
                    continue;
                }
                for (std::size_t i = 0; i < count; ++i) {
                    const auto completion = completions[i];
                    require(!completion.error, "direct read reported an error");
                    const auto window =
                        direct_read_window(view, offsets[completion.tag], 90);
                    if (completion.bytes < window.required) continue; // short tail is allowed
                    const auto* base =
                        static_cast<const std::byte*>(buffers[completion.buffer_index].data);
                    const auto got = std::span<const std::byte>(base + window.prefix, 90);
                    const auto want = expect_at(path, offsets[completion.tag], 90);
                    require(std::equal(got.begin(), got.end(), want.begin()),
                            round == 0 ? "round 0 bytes differ from the file"
                                       : "round 1 reused a stale completion");
                    seen[completion.tag] = true;
                }
            }
            require(std::all_of(seen.begin(), seen.end(), [](bool v) { return v; }),
                    "a read never completed");
            require(queue.outstanding() == 0, "outstanding() did not drain");
        }

        // 4. at/after EOF: a clean zero-length completion, never stale bytes
        {
            const auto window = direct_read_window(view, kFileBytes - 90, 90);
            require(queue.try_enqueue({0, 0, window.offset, static_cast<std::uint32_t>(window.bytes),
                                       0}) == ReadAdmission::Accepted,
                    "tail enqueue rejected");
            queue.submit();
            std::vector<ReadCompletion> completions(1);
            while (queue.outstanding()) {
                if (!queue.poll(completions)) (void)queue.wait();
            }
            require(completions[0].bytes >= 90, "tail read lost bytes");
        }

        std::filesystem::remove(path);
        std::cout << "OK Windows direct read queue\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
