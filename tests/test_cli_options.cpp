#include "product/config/options.h"

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::product::CliOptions;

CliOptions parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return ninfer::product::parse_cli_options(static_cast<int>(argv.size()), argv.data());
}

bool rejects(const std::function<void()>& operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

std::filesystem::path write_config(const std::string& text) {
    const auto path = std::filesystem::temp_directory_path() / "ninfer-cli-options-test.toml";
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file << text;
    file.close();
    return path;
}

} // namespace

int main() {
    int failures      = 0;
    const auto choice = parse({"ninfer", "model.ninfer", "--prompt", "hello", "--choice", "",
                               "--choice", "a|b", "--choice", "你好"});
    failures += check(choice.constraint.kind == ninfer::product::ConstraintKind::Choice &&
                          choice.constraint.choices == std::vector<std::string>{"", "a|b", "你好"},
                      "choice flags changed literal text or order");
    const auto regex = parse({"ninfer", "model.ninfer", "--prompt", "hello", "--regex", ""});
    failures += check(regex.constraint.kind == ninfer::product::ConstraintKind::Regex &&
                          regex.constraint.regex.empty(),
                      "empty regex must be distinct from no regex");
    failures += check(rejects([] {
                          (void)parse({"ninfer", "model.ninfer", "--prompt", "hello", "--regex", "",
                                       "--choice", "yes"});
                      }),
                      "regex and choice were accepted together");
    failures += check(rejects([] {
                          (void)parse({"ninfer", "model.ninfer", "--prompt", "hello", "--regex", "a",
                                       "--json-object"});
                      }),
                      "regex and JSON were accepted together");
    const auto grammar =
        parse({"ninfer", "model.ninfer", "--prompt", "hello", "--grammar-file", "answer.gbnf"});
    failures += check(grammar.constraint.kind == ninfer::product::ConstraintKind::Grammar &&
                          grammar.constraint.path == "answer.gbnf",
                      "grammar file argument was lost");
    failures += check(parse({"ninfer", "model.ninfer", "--prompt", "hello", "--json-object"})
                              .constraint.kind == ninfer::product::ConstraintKind::JsonObject,
                      "JSON object CLI option lost");
    const auto schema =
        parse({"ninfer", "model.ninfer", "--prompt", "hello", "--json-schema-file", "answer.json"});
    failures += check(schema.constraint.kind == ninfer::product::ConstraintKind::JsonSchema &&
                          schema.constraint.path == "answer.json",
                      "schema file CLI option lost");
    failures += check(rejects([] {
                          (void)parse({"ninfer", "model.ninfer", "--prompt", "hello",
                                       "--json-object", "--grammar-file", "a.gbnf"});
                      }),
                      "conflicting CLI constraints accepted");
    const CliOptions configured =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--thinking-budget", "37"});
    failures += check(configured.launch.generation.thinking_budget == 37,
                      "--thinking-budget did not preserve its positive value");
    const std::string help =
        ninfer::product::startup_usage_text("ninfer-cli", ninfer::product::EntryPoint::Cli);
    failures += check(help.find("--thinking-budget") != std::string::npos,
                      "CLI help omits --thinking-budget");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--thinking-budget", "0"});
                      }),
                      "zero --thinking-budget was accepted");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--thinking-budget", "8", "--no-thinking"});
                      }),
                      "--thinking-budget was accepted with --no-thinking");
    const CliOptions with_effort =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--thinking-budget", "8",
               "--reasoning-effort", "medium"});
    failures += check(with_effort.launch.generation.thinking_budget == 8 &&
                          with_effort.reasoning_effort,
                      "thinking budget did not coexist with reasoning effort");
    const CliOptions dflash_vision =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--vision", "--spec", "dflash",
               "--draft-tokens", "7"});
    failures += check(dflash_vision.launch.engine.enable_vision &&
                          dflash_vision.launch.engine.speculative.backend ==
                              ninfer::SpeculativeBackend::DFlash &&
                          dflash_vision.launch.engine.speculative.draft_tokens == 7,
                      "CLI did not preserve the combined DFlash and Vision startup features");
    for (const auto k : {1U, 2U, 7U, 15U}) {
        const auto dflash2 = parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec",
                                    "dflash2", "--draft-tokens", std::to_string(k)});
        failures += check(dflash2.launch.engine.speculative.backend ==
                                  ninfer::SpeculativeBackend::DFlash2 &&
                              dflash2.launch.engine.speculative.draft_tokens == k,
                          "CLI did not preserve the DFlash2 draft count");
    }
    for (const auto k : {0U, 16U}) {
        failures +=
            check(rejects([&] {
                      (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec",
                                   "dflash2", "--draft-tokens", std::to_string(k)});
                  }),
                  "CLI accepted an unsupported DFlash2 draft count");
    }
    const CliOptions nvfp4 =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--kv-dtype", "nvfp4"});
    failures += check(nvfp4.launch.engine.kv_cache == ninfer::KvCacheStorage::Nvfp4Group16,
                      "--kv-dtype nvfp4 did not select group-16 NVFP4 KV");
    const CliOptions k8v4 =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--kv-dtype", "k8v4"});
    failures += check(k8v4.launch.engine.kv_cache == ninfer::KvCacheStorage::Fp8KeyNvfp4Value,
                      "--kv-dtype k8v4 did not select asymmetric K8V4 KV");
    failures +=
        check(help.find("nvfp4") != std::string::npos && help.find("k8v4") != std::string::npos,
              "CLI help omits a production KV storage mode");
    const CliOptions logging =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--log-level", "debug"});
    failures += check(logging.launch.logging.level == ninfer::product::LogLevel::Debug,
                      "CLI log level was not parsed");
    failures += check(help.find("--log-level") != std::string::npos,
                      "CLI help omits the log-level control");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--log-level", "verbose"});
                      }),
                      "CLI accepted an unknown log level");
    failures +=
        check(rejects([] {
                  (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--top-k", "21"});
              }),
              "CLI accepted top_k beyond the executable candidate domain");
    // v8 section 12, first phase: the CLI accepts the controller-mode flag (the single-stream CLI
    // only feels it in the freeze/prefetch criteria weights); a typo is refused, not defaulted.
    const CliOptions stable_mode =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--controller-mode", "stable"});
    failures += check(stable_mode.launch.controller_mode == "stable",
                      "--controller-mode stable did not parse on the CLI");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--controller-mode", "aggressive"});
                      }),
                      "CLI accepted an invalid --controller-mode");
    failures += check(help.find("--controller-mode") != std::string::npos,
                      "CLI help omits --controller-mode");
    // The unified startup configuration keeps the historical spelling and the canonical one.
    const CliOptions canonical =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--max-tokens", "12"});
    failures += check(canonical.launch.generation.max_tokens == 12,
                      "canonical --max-tokens spelling was not accepted");
    // T050 / T020 J3: the startup planning headroom is its own startup field, configurable on both
    // entry points and never folded into the KV sizing headroom.
    const CliOptions planning_headroom =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
               "--startup-planning-headroom-mib", "3072"});
    failures += check(planning_headroom.launch.engine.startup_planning_headroom_bytes == (3ULL << 30),
                      "--startup-planning-headroom-mib did not reach the CLI engine options");
    failures += check(help.find("--startup-planning-headroom-mib") != std::string::npos,
                      "CLI help omits --startup-planning-headroom-mib");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--startup-planning-headroom-mib", "1.5x"});
                      }),
                      "CLI accepted a malformed startup planning headroom");

    // TOML file: the same field catalog, layered under argv.
    const auto config_path = write_config(
        "model.path = \"from-file.ninfer\"\n"
        "cli.prompt = \"file prompt\"\n"
        "generation.max_tokens = 24\n"
        "generation.sampling.temperature = 0.5\n"
        "engine.kv_dtype = \"int8\"\n"
        "engine.startup_planning_headroom_mib = 1280\n");
    const CliOptions from_file = parse({"ninfer-cli", "--config", config_path.string()});
    failures += check(from_file.launch.engine.artifact_path ==
                              config_path.parent_path() / "from-file.ninfer" &&
                          from_file.prompt == "file prompt" &&
                          from_file.launch.generation.max_tokens == 24 &&
                          from_file.launch.generation.sampling.temperature &&
                          *from_file.launch.generation.sampling.temperature == 0.5F &&
                          from_file.launch.engine.startup_planning_headroom_bytes ==
                              (1280ULL << 20) &&
                          from_file.launch.engine.kv_cache == ninfer::KvCacheStorage::Int8Group64,
                      "TOML startup file did not resolve through the shared schema");
    failures += check(from_file.launch.provenance.config_path ==
                              std::filesystem::absolute(config_path).lexically_normal() &&
                          from_file.launch.provenance.fields.contains("generation.max_tokens"),
                      "TOML provenance was not recorded");
    const CliOptions overridden =
        parse({"ninfer-cli", "--config", config_path.string(), "--max-tokens", "7"});
    failures += check(overridden.launch.generation.max_tokens == 7 &&
                          overridden.launch.provenance.fields.at("generation.max_tokens").kind ==
                              ninfer::product::ConfigSourceKind::CommandLine,
                      "command line did not override the TOML layer");
    failures += check(rejects([&] {
                          (void)parse({"ninfer-cli", "--config", config_path.string(),
                                       "--kv-dtype", "bogus"});
                      }),
                      "invalid command-line value was accepted beside a config file");
    failures += check(rejects([] {
                          const auto bad =
                              write_config("engine.max_context = 0\ncli.prompt = \"x\"\n");
                          (void)parse({"ninfer-cli", "--config", bad.string()});
                      }),
                      "out-of-range TOML value was accepted");
    failures += check(rejects([] {
                          const auto bad =
                              write_config("engine.unknown_knob = 1\ncli.prompt = \"x\"\n");
                          (void)parse({"ninfer-cli", "--config", bad.string()});
                      }),
                      "unknown TOML field was accepted");
    std::filesystem::remove(config_path);
    return failures == 0 ? 0 : 1;
}
