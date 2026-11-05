#include "core/kv_tier.h"
#include "core/prefetch.h"

#include <algorithm>
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
    {
        // Q54: the prefix index sidecar. One note per stored block, rewritten with each store,
        // surviving restarts, and skipped - not fatal - when the file is short or corrupted.
        const auto notes_dir =
            std::filesystem::temp_directory_path() / "ninfer_kv_seam_prefix_test";
        std::filesystem::remove_all(notes_dir, ec);
        {
            ninfer::core::DiskKvTier tier(notes_dir);
            ninfer::core::KvTierPrefixNote note;
            note.block_hash   = 0x0123456789abcdefULL;
            note.block_tokens = 397;
            note.frontier     = 388;
            note.digest_a     = 0x1111111111111111ULL;
            note.digest_b     = 0x2222222222222222ULL;
            note.identity_tag = 0x00020000U;
            for (std::size_t i = 0; i < note.fingerprint.size(); ++i) {
                note.fingerprint[i] = static_cast<std::uint8_t>(i & 0xff);
            }
            expect(tier.prefix_notes().empty(), "a fresh directory carries no prefix notes");
            expect(tier.note_prefix(note), "the tier accepts a prefix note");
            expect(tier.note_prefix(note), "restating the same note is idempotent");
            const auto notes = tier.prefix_notes();
            expect(notes.size() == 1, "one note is live after two identical stores");
            expect(notes[0].block_hash == note.block_hash && notes[0].frontier == note.frontier &&
                       notes[0].digest_a == note.digest_a && notes[0].digest_b == note.digest_b &&
                       notes[0].identity_tag == note.identity_tag &&
                       notes[0].fingerprint == note.fingerprint,
                   "the live note round-trips every field");
            ninfer::core::KvTierPrefixNote longer = note;
            longer.frontier = 397;
            longer.digest_a = 0x3333333333333333ULL;
            longer.digest_b = 0x4444444444444444ULL;
            expect(tier.note_prefix(longer), "a restated block replaces its own note");
            expect(tier.prefix_notes().size() == 1 &&
                       tier.prefix_notes().front().frontier == 397,
                   "the sidecar of one block holds the latest store");
            ninfer::core::KvTierPrefixNote other = note;
            other.block_hash   = 0x0fedcba987654321ULL;
            other.block_tokens = 215;
            other.frontier     = 215;
            expect(tier.note_prefix(other), "a second block gets its own note");
            expect(tier.prefix_notes().size() == 2, "notes accumulate across blocks");
        }
        {
            ninfer::core::DiskKvTier reopened(notes_dir);
            const auto notes = reopened.prefix_notes();
            expect(notes.size() == 2, "prefix notes survive a restart");
            expect(std::any_of(notes.begin(), notes.end(), [](const auto& entry) {
                       return entry.frontier == 397 && entry.block_hash == 0x0123456789abcdefULL;
                   }) &&
                       std::any_of(notes.begin(), notes.end(), [](const auto& entry) {
                           return entry.frontier == 215 &&
                                  entry.block_hash == 0x0fedcba987654321ULL;
                       }),
                   "restarted notes keep their digests and blocks");
        }
        {
            // A torn sidecar and a checksum failure must each fall back to "no entry", and must
            // not take the healthy entry with them.
            const auto sidecar = notes_dir / "pfx-0123456789abcdef.397";
            expect(std::filesystem::file_size(sidecar, ec) > 8, "the sidecar exists on disk");
            std::FILE* file = std::fopen(sidecar.string().c_str(), "r+b");
            expect(file != nullptr, "the sidecar is writable for the corruption fixture");
            if (file != nullptr) {
                std::fseek(file, 16, SEEK_SET);  // inside digest_a: breaks the checksum
                const int flipped = 'X';
                std::fwrite(&flipped, 1, 1, file);
                std::fclose(file);
            }
            const auto torn = notes_dir / "pfx-0fedcba987654321.215";
            std::filesystem::resize_file(torn, 7, ec);  // shorter than the header
            ninfer::core::DiskKvTier damaged(notes_dir);
            const auto notes = damaged.prefix_notes();
            expect(notes.empty(), "a bad index reads as no index, never as a wrong entry");
            // The pre-index constructor parsed every file as a block; a pfx- name would have
            // been indexed as hash 0 with the tokens after the dot. Miss here proves the
            // sidecar is recognized as an index file and not misread as a block.
            expect(damaged.lookup(ninfer::core::KvBlockKey{0, 397}).state ==
                       ninfer::core::KvTierState::miss,
                   "a pfx sidecar is not misread as a block");
        }
        std::filesystem::remove_all(notes_dir, ec);
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
