@echo off
rem =====================================================================
rem  NInfer v3 (RTX 4090 build) - Qwen3.8-27B v3 artifact - port 9908
rem  Context: 128K (131072 tokens) | Concurrency: 8 | KV: rk4v4
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
rem  The draft's trained range is 262,144 tokens and this row stays inside it,
rem  so the draft runs on every generated round.
rem

rem
rem  --kv-capacity auto keeps the 1 GiB sizing headroom and lets the engine take
rem  as much Device KV as the card can spare, which is what a 8-lane profile
rem  needs; pin an explicit value instead if this machine shares the card.
rem =====================================================================
title ninfer-serve v3 128K C=8 rk4v4 (9908)
cd /d %~dp0

if not exist ".\ninfer-serve.exe" (
  echo.
  echo This launcher has to sit next to ninfer-serve.exe: it ships inside the release folder,
  echo which also holds the executable and, one level up, the model artifact.
  echo Copy it back there ^(or run it from there^) and start it again.
  echo.
  pause
  exit /b 1
)

rem Override the artifact path with NINFER_MODEL if it lives somewhere else.
if not defined NINFER_MODEL set "NINFER_MODEL=..\models\qwen3_8_27b_v3.ninfer"

.\ninfer-serve.exe %NINFER_MODEL% ^
  --host 0.0.0.0 --port 9908 ^
  --max-context 131072 --kv-capacity auto ^
  --max-concurrency 8 --max-pending-requests 16 ^
  --prefill-chunk 1024 --kv-dtype rk4v4 ^
  --spec dflash2 --draft-tokens 7 --lm-head-draft ^
  --kv-evict auto --kv-idle-reclaim-ms 60000 ^
  --wddm-evictable-budget

pause
