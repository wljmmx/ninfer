@echo off
rem =====================================================================
rem  NInfer v3 (RTX 4090 build) - Qwen3.8-27B v3 artifact - port 9908
rem  Context: 1M (1048576 tokens) | Concurrency: 4 | KV: int8
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
rem  MEASURED: this row does NOT start on a 24 GB card AS WRITTEN. int8 needs ~33 KiB
rem  per token and the capacity policy in force here requires the WHOLE logical payload
rem  (~34.5 GiB) to be Device-resident, so startup fails after ~8 s with
rem    FATAL server failed during startup | minimum Engine runtime reservation requires
rem    36979631360 bytes in addition to 1073741824 bytes of automatic headroom, but only
rem    25219825664 bytes are available
rem  The same failure happens with MTP and with no speculation at all, and 1M x rk4v4
rem  without speculation starts, so it is the KV dtype, not dflash2.
rem  Two verified ways to run this profile anyway are at the end of this file:
rem    A) 512K - starts in 78 s, arena 558,720 tokens, full device residency,
rem    B) 1M with a DECLARED RESIDENT SET (NINFER_KV_EVICT_CAPACITY) plus the eviction
rem       contract - starts in 9.4 s and served a 130,084-token request in 64.5 s by
rem       demoting the masked pages to the Host arena; retention 2,000/16,384 pages
rem       = 12%, and quality at 12% retention is UNMEASURED (the lowest retention
rem       measured at <2% perplexity loss is 58%).
rem
rem  --kv-capacity auto keeps the 1 GiB sizing headroom and lets the engine take
rem  as much Device KV as the card can spare, which is what a 4-lane profile
rem  needs; pin an explicit value instead if this machine shares the card.
rem =====================================================================
title ninfer-serve v3 1M C=4 int8 (9908)
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
  --prefill-chunk 1024 --kv-dtype int8 ^
  --spec dflash2 --draft-tokens 7 --lm-head-draft ^
  --kv-evict auto --kv-idle-reclaim-ms 60000 ^
  --wddm-evictable-budget

rem ---------------------------------------------------------------------
rem  VERIFIED ALTERNATIVE A for this card: the same profile at 512K (dflash2 draft 7,
rem  lm-head draft, same optimisations). Measured: starts in 78 s and serves.
rem  To use it, delete the 'rem ' prefix of each line below.
rem
rem .\ninfer-serve.exe ..\models\qwen3_8_27b_v3.ninfer ^
rem   --host 0.0.0.0 --port 9908 ^
rem   --max-context 524288 --kv-capacity auto ^
rem   --max-concurrency 4 --max-pending-requests 16 ^
rem   --prefill-chunk 1024 --kv-dtype int8 ^
rem   --spec dflash2 --draft-tokens 7 --lm-head-draft ^
rem   --kv-evict auto --kv-idle-reclaim-ms 60000 ^
rem   --wddm-evictable-budget
rem 
rem ---------------------------------------------------------------------
rem  VERIFIED ALTERNATIVE B: keep 1M and let the contract serve the overflow instead of
rem  refusing - declare the Device resident set (pages) and turn the contract on. The
rem  logical capacity stays 1,048,576 tokens; masked pages demote to the Host arena.
rem  MEASURED: starts in 9.4 s, short request 0.9 s, a 130,084-token request 64.5 s.
rem  Smaller NINFER_KV_EVICT_CAPACITY = more overflow into the contract = lower quality.
rem  12% retention is unmeasured; 58% is the lowest point measured at <2% loss.
rem  To use it, delete the 'rem ' prefix of each line below.
rem
rem set NINFER_KV_EVICT_CAPACITY=2000
rem .\ninfer-serve.exe ..\models\qwen3_8_27b_v3.ninfer ^
rem   --host 0.0.0.0 --port 9908 ^
rem   --max-context 1048576 --kv-capacity 1048576 ^
rem   --max-concurrency 4 --max-pending-requests 16 ^
rem   --prefill-chunk 1024 --kv-dtype int8 ^
rem   --spec dflash2 --draft-tokens 7 --lm-head-draft ^
rem   --kv-evict on --kv-evict-window 4 --kv-evict-sink 1 ^
rem   --wddm-evictable-budget
rem ---------------------------------------------------------------------

pause
