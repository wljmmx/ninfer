# Release layout and audit

Date: 2026-11-08. Build: `git describe` (see `VERSION` in the release folder), `Release`, sm_89.

This page is the checklist the Windows release folder is assembled from, and the audit that says
what is *inside* it, what is deliberately left out, and what is still missing from the product.

## Assemble

```powershell
cd H:\TeleAgentWorks\ninfer-4090-port\infer
cmake --build build-win -j 4                       # Release, all targets
powershell -File tools\release\package_windows_release.ps1 -Root R:\infer
powershell -File tools\release\package_windows_release.ps1 -Root <qa dir> -WithDiagnostics
```

`package_windows_release.ps1` performs every step below and rewrites `SHA256SUMS.txt`; it refuses
to run when the build products are older than the newest source change, so a stale `ninfer.exe`
cannot be shipped silently.

Two profiles, and a folder holds one of them:

| Profile | Contents | For |
|---|---|---|
| **product** (default) | `ninfer`, `ninfer-serve`, `ninfer-perplexity` plus the rest of the table | what an operator is given |
| `-WithDiagnostics` | adds `diag-moe`, `diag-cpu-partial`, `diag-second-engine`, `ninfer_bench`, and `symbols\*.pdb` when the build emitted them | validation and crash analysis, never a customer folder |

Packaging a folder that holds QA-only files as the product profile deletes them, so the two cannot
be mixed by accident.

## Layout

| Path | Content | Why it is shipped |
|---|---|---|
| `ninfer.exe`, `ninfer-serve.exe`, `ninfer-perplexity.exe` | generation, serving, perplexity measurement | the product: the three programs an operator runs |
| `*.dll` | FFmpeg 9, libcurl, libzmq, libjpeg-turbo | runtime dependencies of the above |
| `diag-moe.exe`, `diag-cpu-partial.exe`, `diag-second-engine.exe`, `ninfer_bench.exe` | three diagnostics, one benchmark harness | **QA profile only** (`-WithDiagnostics`): they measure the product, they are not used with it |
| `symbols\*.pdb` | matching symbols | **QA profile only, and only if the build emitted them**: the current Release configuration produces no `.pdb`, so the folder is absent, and a product folder never creates it |
| `docs\` | public docs + `maintainer\` registers + `performance\` baselines + `zh-CN\` notes | what the product does, why it does it that way, and what it measured |
| `templates\`, `tools\chat_templates\` | chat templates | serving needs them; the converter references them by path |
| `ninfer.example.toml` | commented configuration | the only complete option reference |
| `scripts\` | 4 serve profiles + 2 front ends | start the product without reading the docs first |
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
| `bench\*.exe` (one per kernel, 60+ binaries, ~20 GB) | single-kernel micro-benchmarks; they are development material, and `ninfer_bench.exe` itself ships only in the QA profile |
| `dashboard\data\` | the dashboard's SQLite history is created on first run |
| `dashboard\node_modules` in `SHA256SUMS.txt` | hashed file-by-file it would be 4k lines of noise; the folder ships as-is from the tested build |
| `.obj`, `.err`, `*.log` | build/run outputs, not sources (two such files had been committed to the repository root and were removed) |
| `diag-commit.exe` | a leftover binary in `build-win\apps` whose target no longer exists in `apps\CMakeLists.txt`; shipping it would ship code the tree can no longer rebuild |

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
| Qwen4Exp runtime (loading/program/vision/MTP) | **fixed**: the artifact is complete (`R:\model\qwen4exp\UD-Q4_K_XL` holds shards 1-4 of 4), converted to `qwen3_8_flash_next_v5_e2.ninfer`, carded as `model-cards\Qwen3.8-Flash-Next-NInfer`, served by `scripts\serve_flashnext_2k.bat` (port 11439, bf16), and gated by `tools\release\gate_qsa.ps1` |
| QSA low-bit KV storages (`int8`, `fp8`, `nvfp4`, `k8v4`, five `rk*`) | **open, by admission not by code**: all ten enumerators are implemented in `qsa_storage_supported` and the Op sweeps pass, but `qsa_kv_storage_admitted` admits bf16 only; every low-bit route stays behind `NINFER_QSA_SPARSE_GQA_QUANT_KV` because the engine will not run a numeric route that has no recorded model-level PPL. `tools\release\gate_qsa.ps1 -KvDtype <storage>` is how one is qualified: it compares against that storage's own recorded value (`-Record` prints a fresh one; an unrecorded storage refuses to pass rather than borrowing bf16's number). bf16 is 15.533755882784398; `rk4v4-e8` has one recorded value, the other eight have none |
| Joint context sizing (`plan_context_capacity` not called by `resolve_kv_capacity`) | **fixed** (V10-01): `SequencePlanner::state_image_bytes()` exposes the stride pre-Program, and the automatic path sizes page groups, StateImage slots and workspace through `resolve_context_capacity`; a union that does not fit refuses the load by name |
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

