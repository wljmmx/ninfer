#include "core/kv_tier.h"
#include "core/prefetch.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <thread>
#include <vector>

int main() {
    int failures = 0;
    const auto expect = [&](bool condition, const char* what) {
        std::printf("%s %s\n", condition ? "ok  " : "FAIL", what);
        if (!condition) { ++failures; }
    };

    std::vector<ninfer::TokenId> a{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<ninfer::TokenId> b{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<ninfer::TokenId> c{1, 2, 3, 4};
    const auto ka = ninfer::core::kv_block_key(a);
    const auto kb = ninfer::core::kv_block_key(b);
    const auto kc = ninfer::core::kv_block_key(c);
    expect(ka == kb, "same prefix hashes to the same key");
    expect(!(ka == kc), "a shorter prefix is a different key");

    const auto dir = std::filesystem::temp_directory_path() / "ninfer_kv_seam_test";
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    {
        ninfer::core::DiskKvTier tier(dir);
        expect(tier.lookup(ka).state == ninfer::core::KvTierState::miss, "unknown prefix misses");
        std::vector<std::byte> payload(4096);
        for (std::size_t i = 0; i < payload.size(); ++i) { payload[i] = static_cast<std::byte>(i & 0xFF); }
        expect(tier.store(ka, payload), "store writes the block");
        const auto hit = tier.lookup(ka);
        expect(hit.state == ninfer::core::KvTierState::resident && hit.bytes == payload.size(),
               "stored block is resident with the right size");
        expect(tier.prefetch(ka), "prefetch accepts a resident block");
        std::vector<std::byte> out(payload.size());
        expect(tier.retrieve(ka, out) && out == payload, "retrieve returns the stored bytes");
        expect(!tier.prefetch(kc), "prefetch refuses a block the tier does not hold");
    }
    {
        ninfer::core::DiskKvTier reopened(dir);
        std::vector<std::byte> out(4096);
        expect(reopened.lookup(ka).state == ninfer::core::KvTierState::resident,
               "directory index survives a restart");
        expect(reopened.retrieve(ka, out), "restarted tier still serves the block");
    }
    std::filesystem::remove_all(dir, ec);

    {
        ninfer::core::Prefetcher prefetch(1024, 0.10);
        std::atomic<int> ran{0};
        const auto job = [&] { ++ran; return true; };
        expect(prefetch.enqueue(job, 768, 10, 0), "job inside the budget is accepted");
        expect(!prefetch.enqueue(job, 768, 10, 0), "job over the window budget is rejected");
        expect(!prefetch.enqueue(job, 64, 0, 0), "job already past its deadline is rejected");
        prefetch.advance(1);
        expect(prefetch.enqueue(job, 768, 10, 1), "the next window has budget again");
        for (int i = 0; i < 32; ++i) {
            prefetch.enqueue(job, 32, 100, 1);
            prefetch.advance(2);
            prefetch.note_used(32);
        }
        expect(prefetch.gate_open(), "a useful prefetcher keeps its gate open");

        ninfer::core::Prefetcher useless(1024, 0.10);
        for (int i = 0; i < 40; ++i) {
            useless.enqueue(job, 32, 100, 0);
            useless.advance(1);
        }
        expect(!useless.gate_open(), "a prefetcher whose bytes are never used closes its gate");
        expect(!useless.enqueue(job, 32, 100, 1), "a closed gate refuses new jobs");
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::printf("%s\n", failures == 0 ? "ALL OK" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
