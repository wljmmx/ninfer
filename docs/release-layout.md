# Release layout and audit

Date: 2026-11-08. Build: `git describe` (see `VERSION` in the release folder), `Release`, sm_89.

This page is the checklist the Windows release folder is assembled from, and the audit that says
what is *inside* it, what is deliberately left out, and what is still missing from the product.

## Assemble

```powershell
cd H:\TeleAgentWorks\infer-4090-port\infer
cmake --build build-win -j 4                       # Release, all targets
powershell -File tools\release\package_windows_release.ps1 -Root R:\infer
```

`package_windows_release.ps1` performs every step below and rewrites `SHA256SUMS.txt`; it refuses
to run when the build products are older than the newest source change, so a stale `ninfer.exe`
cannot be shipped silently.

## Layout

| Path | Content | Why it is shipped |
|---|---|---|
| `*.exe` | `ninfer`, `ninfer-serve`, `ninfer-perplexity`, `ninfer_bench`, `diag-moe`, `diag-cpu-partial` | the product (generation, serving, gates, benchmark harness, two diagnostics) |
| `*.dll` | FFmpeg 9, libcurl, libzmq, libjpeg-turbo | runtime dependencies of the above |
| `symbols\*.pdb` | matching symbols | crash diagnosis without a rebuild |
| `docs\` | public docs + `maintainer\` registers + `performance\` baselines + `zh-CN\` notes | what the product does, why it does it that way, and what it measured |
| `templates\`, `tools\chat_templates\` | chat templates | serving needs them; the converter references them by path |
| `ninfer.example.toml` | commented configuration | the only complete option reference |
| `scripts\` | 3 serve profiles + 2 front ends | start the product without reading the docs first |
| `tools\console\` | local operations console | mode/freeze control + engine/LMCache/GPU readings on one origin |
| `dashboard\` | process-level dashboard (+ `node_modules`) | history, request tables, per-request detail |
| `tools\convert`, `tools\artifact`, `tools\freq_corpus` | converter stack | turning checkpoints into `.ninfer` is part of the product |
| `tools\upgrade_ninfer_v2_to_v3.py` | v2 artifact upgrade | one-way, standard-library migration |
| `LICENSE`, `README.md`, `README.zh-CN.md` | project licence, English introduction, Chinese introduction | legal + orientation in both languages |

## Deliberately not shipped

| Item | Reason |
|---|---|
| C++ sources, CMake, vcpkg manifest | development material; the repository is the source of truth |
| `tests/` | same |
| `bench\*.exe` (one per kernel, 60+ binaries, ~20 GB) | single-kernel micro-benchmarks; `ninfer_bench.exe` covers the user-facing cases |
| `dashboard\data\` | the dashboard's SQLite history is created on first run |
| `dashboard\node_modules` in `SHA256SUMS.txt` | hashed file-by-file it would be 4k lines of noise; the folder ships as-is from the tested build |
| `.obj`, `.err`, `*.log` | build/run outputs, not sources (two such files had been committed to the repository root and were removed) |

## Audit result (what was missing before this pass)

| Gap | Status |
|---|---|
| Local console lived outside the repository (`H:\TeleAgentWorks\tools`) | **fixed**: now `tools\console\`, shipped, and its page path resolves relative to the script |
| Dashboard was built but not shipped | **fixed**: `dashboard\` with `dist`, config, README and `node_modules` |
| Converter stack not shipped | **fixed**: `tools\{convert,artifact,freq_corpus,chat_templates}` |
| v2→v3 upgrade tool not shipped | **fixed** |
| `LICENSE` / `README.md` missing from the release | **fixed** |
| Maintainer registers not shipped | **fixed**: `docs\maintainer\` |
| Two run logs committed at the repository root (`ppl_e2.err`, `repack.err`) | **fixed**: removed |
| Dashboard `engine_url` pointed at the example port 8080 | **fixed**: release ships `config.local.json` pointing at 11438 |
| Qwen4Exp runtime (loading/program/vision/MTP) | **still open**: the artifact is not complete (`R:\model\qwen4exp\UD-Q4_K_XL` holds only shard 1 of 4); v9 T192–T198 |
| Joint context sizing (`plan_context_capacity` not called by `resolve_kv_capacity`) | **still open**: the StateImage stride does not exist before the Program is built |
| Prefix-checkpoint retention selection | **still open**: a policy change that needs its own gate |

## Verification the folder must pass before it is called a release

1. `ninfer.exe --help` and `ninfer-perplexity.exe --help` run from a foreign working directory
   (proves the DLLs are beside the executables, not found through `PATH`).
2. `ninfer-serve.exe` from the folder serves `/health`, `/control/state`, a mode write, and one
   real completion — recorded in `SMOKE.txt`.
3. Console and dashboard serve their pages, the console's engine source answers `ok=true`, a
   proxied control write lands, and the dashboard reports its instance — recorded in
   `SMOKE-UI.txt`.
4. `python -m tools.convert --help`, `python tools\upgrade_ninfer_v2_to_v3.py --help`,
   `python tools\console\lmcache_console.py --help` all run from the folder.
5. The numerical gates were run on the same build tree (see `RELEASE.md`).

## Two traps this page exists to prevent

* **BOM in files a program parses.** Windows PowerShell 5.1 writes a BOM for `-Encoding UTF8`, and
  Node's `JSON.parse` rejects it (`Unexpected token '﻿'`). The packager writes `config.local.json`
  and `VERSION` through a BOM-free helper — the first packaging run had shipped a BOM'd
  `config.local.json` and the dashboard refused to start.
* **Stale binaries.** `ninfer.exe` older than the newest `src/`/`include/` file stops the script;
  a release built from a tree that no longer matches its own sources is worse than a late one.

