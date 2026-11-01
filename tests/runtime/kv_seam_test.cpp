#include "core/kv_tier.h"
#include "core/prefetch.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <future>
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
        expect(prefetch.enabled(), "a prefetcher runs until something turns it off");

        // The queue measures, it does not judge (2026-11-01): a written usefulness threshold here
        // was a second decision point next to PolicyController, and two decision points disagree.
        ninfer::core::Prefetcher useless(1024, 0.10, /*max_pending=*/256);
        for (int i = 0; i < 40; ++i) {
            useless.enqueue(job, 32, 100, 0);
            useless.advance(1);
        }
        expect(useless.stats().usefulness() == 0.0,
               "a prefetcher whose bytes are never used reports zero usefulness");
        expect(useless.enabled(), "and stays enabled: the decision is the controller's");
        useless.set_enabled(false);
        expect(!useless.enqueue(job, 32, 100, 1), "a queue turned off refuses new jobs");
        expect(useless.stats().dropped_gate == 1, "the refusal is counted as a gate drop");
        useless.set_enabled(true);
        expect(useless.enqueue(job, 32, 100, 1), "and takes work again when it is turned back on");
    }

    {
        // Backpressure (D2's second gate). The byte budget bounds what one window may ask for; it
        // says nothing about how far the producer may run ahead of a worker stuck in a slow read.
        // The cap is therefore on running + waiting, and it is proved here rather than by hoping a
        // disk is slow at the right moment: one long job fills the cap, the next offer must be
        // refused and counted as backlog, and once the job finishes the count returns to zero.
        ninfer::core::Prefetcher blocked(1 << 20, 0.10, /*max_pending=*/1);
        std::promise<void> release;
        const std::shared_future<void> gate = release.get_future().share();
        std::atomic<int> entered{0};
        const auto long_job = [&] {
            ++entered;
            gate.wait();
            return true;
        };
        expect(blocked.enqueue(long_job, 64, 100, 0), "a job inside the cap is accepted");
        for (int i = 0; i < 500 && entered.load() == 0; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        expect(entered.load() == 1, "the worker picked the long job up");
        expect(!blocked.enqueue(long_job, 64, 100, 0),
               "a further job is refused while the cap is full");
        expect(blocked.stats().dropped_backlog == 1,
               "the refusal is counted as backlog, not as budget");
        expect(blocked.pending() == 1, "pending counts the job being run");
        release.set_value();
        for (int i = 0; i < 500 && blocked.pending() != 0; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        expect(blocked.pending() == 0, "pending returns to zero when the job finishes");
        blocked.stop();
        expect(blocked.stats().dropped_backlog == 1, "finishing a job is not a drop");
        expect(!blocked.enqueue(long_job, 64, 100, 0), "a stopped prefetcher refuses new jobs");
        expect(blocked.stats().dropped_stopped == 1, "the refusal after stop is counted as stopped");
        blocked.stop(); // idempotent: the second stop must not join the same thread twice
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::printf("%s\n", failures == 0 ? "ALL OK" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
