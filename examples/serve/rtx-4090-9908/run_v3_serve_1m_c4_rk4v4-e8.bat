@echo off
rem =====================================================================
rem  NInfer v3 (RTX 4090 build) - Qwen3.8-27B v3 artifact - port 9908
rem  Context: 1M (1048576 tokens) | Concurrency: 4 | KV: rk4v4-e8
rem  Speculative: dflash2 (draft 7) + lm-head draft
rem
rem  Optimisations this script turns ON by default:
rem    --kv-evict auto            eviction contract, decided at startup from the
rem                               resolved arena / host budget / this KV dtype
rem                               (the startup log prints the decision and, when
rem                               it declines, which condition failed)
rem    --kv-idle-reclaim-ms 60000 idle shared-prefix demotion: bytes move to the
rem                               pinned Host arena and are fetched back on reuse
rem                               (measured output-neutral; +30..80 ms first fetch)
rem    --wddm-evictable-budget    size the arena against total VRAM - 512 MB
rem                               instead of currently-free VRAM
rem  The draft's trained range is 262,144 tokens and this row is LONGER. The engine
rem  no longer refuses that (appendix FS: the old hard guard is gone). Instead a batch
rem  whose draft window would cross 262,144 tokens is served by the ORDINARY round, per
rem  batch (decode.cpp decode_raw) - the draft itself is never run out of range.
rem  Consequence: decode keeps the draft's acceleration while positions are <256K and
rem  falls back to plain decode past that. Use '--spec mtp --draft-tokens 3' instead if
rem  you want a draft that never stands down over a 1M range.

rem
rem  --kv-capacity auto keeps the 1 GiB sizing headroom and lets the engine take
rem  as much Device KV as the card can spare, which is what a 4-lane profile
rem  needs; pin an explicit value instead if this machine shares the card.
rem =====================================================================
title ninfer-serve v3 1M C=4 rk4v4-e8 (9908)
cd /d %~dp0

if not exist ".\ninfer-serve.exe" (
  echo.
  echo This launcher has to sit next to ninfer-serve.exe: it ships inside the release
  echo folder, which also holds the executable and, one level up, the model artifact.
  echo Copy it back there ^(or run it from there^) and start it again.
  echo.
  pause
  exit /b 1
)

rem Override the artifact path with NINFER_MODEL if it lives somewhere else.
if not defined NINFER_MODEL set "NINFER_MODEL=..\models\qwen3_8_27b_v3.ninfer"

.\ninfer-serve.exe %NINFER_MODEL% ^
  --host 0.0.0.0 --port 9908 ^
  --max-context 1048576 --kv-capacity auto ^
  --max-concurrency 4 --max-pending-requests 16 ^
  --prefill-chunk 1024 --kv-dtype rk4v4-e8 ^
  --spec dflash2 --draft-tokens 7 --lm-head-draft ^
  --kv-evict auto --kv-idle-reclaim-ms 60000 ^
  --wddm-evictable-budget

pause
