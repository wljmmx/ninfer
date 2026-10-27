// Does the fingerprint actually refuse the mismatches that matter?
//
// Built from the numbers measured on the live 35B run, so the positive case is a real configuration
// rather than a made-up one. The negative cases are the ones that would otherwise fail silently:
// a dtype change, a layer-size change that leaves the total plausible, and speculation being turned
// on (which adds draft-local regions to the image).

#include "core/checkpoint_fingerprint.h"

#include <cstdio>
#include <string>

namespace {

namespace fp = ninfer::core::fingerprint;

int failures = 0;

void expect(bool condition, const char* what) {
    std::printf("%s %s\n", condition ? "ok  " : "FAIL", what);
    if (!condition) {
        ++failures;
    }
}

fp::CheckpointFingerprint real_one() {
    fp::CheckpointFingerprint print;
    print.model_name_hash    = fp::hash_name("qwen3.6-35b-a3b");
    print.kv_page_bytes      = 1310720;  // measured: one engine page, 9 pages = 11,796,480 B
    print.kv_tokens_per_page = 64;       // kPagedKVPageSize
    print.kv_dtype           = fp::KVDtype::BF16;
    print.kv_planes          = 2;
    print.layers             = 30;       // linear-attention layers
    print.hidden             = 2048;
    print.kv_heads           = 32;
    print.head_dim           = 128;
    print.state_image_bytes  = 64393216;  // measured
    print.state_layer_bytes  = 2097152;   // 32 x 128 x 128 x 4
    print.state_conv_bytes   = 1474560;   // 8192 x 3 x 2 x 30
    print.state_layers       = 30;
    print.features.dflash_local = false;  // this run had no speculative draft
    return print;
}

void rejects(const char* what, const fp::CheckpointFingerprint& want, const char* expected_field) {
    const fp::CheckpointFingerprint have = real_one();
    std::string difference;
    const bool matched = fp::matches(have, want, difference);
    if (matched) {
        std::printf("FAIL %s was accepted\n", what);
        ++failures;
        return;
    }
    const bool named = difference == expected_field;
    std::printf("%s %s (reported: %s)\n", named ? "ok  " : "FAIL", what, difference.c_str());
    if (!named) {
        ++failures;
    }
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    // --- encode / decode ---------------------------------------------------
    const auto real = real_one();
    const std::string encoded = fp::encode(real);
    expect(encoded.size() == fp::kEncodedBytes, "encoding is a fixed size, so it can be a key");
    expect(encoded[0] == 1 && encoded[1] == 0, "version leads the encoding");

    fp::CheckpointFingerprint decoded;
    expect(fp::decode(encoded, decoded), "a fresh encoding decodes");
    std::string difference;
    expect(fp::matches(decoded, real, difference), "the decoded fingerprint matches the original");

    // Determinism matters because two processes have to agree: encode twice, same bytes.
    expect(fp::encode(real_one()) == fp::encode(real_one()), "encoding is deterministic");

    // A broken input is refused rather than half-read.
    std::string truncated = encoded.substr(0, encoded.size() - 1);
    fp::CheckpointFingerprint ignored;
    expect(!fp::decode(truncated, ignored), "a short buffer is refused");
    std::string wrong_version = encoded;
    wrong_version[0] = 9;
    expect(!fp::decode(wrong_version, ignored), "an unknown version is refused");

    // --- the mismatches that would otherwise corrupt a restore -------------
    std::printf("\n");
    {
        auto want = real_one();
        want.kv_dtype = fp::KVDtype::INT8;
        rejects("a KV dtype change", want, "kv_dtype");
    }
    {
        auto want = real_one();
        // bf16 recurrent state: half the layer, and the total stays a plausible-looking number.
        want.state_layer_bytes = 1048576;
        rejects("a layer size change with a plausible total", want, "state_layer_bytes");
    }
    {
        auto want = real_one();
        want.features.dflash_local = true;
        rejects("speculative decoding turned on", want, "features.dflash_local");
    }
    {
        auto want = real_one();
        want.kv_tokens_per_page = 128;
        rejects("a page granularity change", want, "kv_tokens_per_page");
    }
    {
        auto want = real_one();
        want.model_name_hash = fp::hash_name("qwen3.8-27b");
        rejects("a different model", want, "model_name");
    }

    // --- the positive case ------------------------------------------------
    std::printf("\n");
    expect(fp::matches(real_one(), real_one(), difference), "an identical fingerprint matches");

    std::printf("\n%d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
