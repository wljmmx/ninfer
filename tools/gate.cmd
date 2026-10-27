@echo off
rem ============================================================================
rem  P0 gate: one command, three checks, non-zero exit on the first failure.
rem
rem     gate.cmd            seam tests + MoE perplexity (Track B, 0.5% tolerance)
rem     gate.cmd quick      seam tests only
rem     gate.cmd record     run the MoE PPL and store it as the bit-exact golden
rem
rem  Why the PPL check is numeric: ninfer-perplexity prints with {:.6g}, so the
rem  gate value 7.776858 shows up as "7.77686". Matching the string can never
rem  work, and rounding to the printed digits would hide a real 0.0005 drift.
rem  The run therefore writes report.json (--output) and the gate reads the
rem  exact double from it.
rem
rem  A missing model is a FAIL, not a skip: a gate that silently drops its
rem  numeric invariant reports green for work it never checked. Use "quick"
rem  when only the seam test is meant to run.
rem ============================================================================
setlocal enabledelayedexpansion

for %%I in ("%~dp0..") do set "REPO=%%~fI"
if not exist "%REPO%\src\core\kv_tier.h" (
  echo FAIL: %REPO% is not the ninfer tree & exit /b 1
)
set "BUILD=%REPO%\build-win"
if not exist "%BUILD%\CMakeCache.txt" (
  echo FAIL: no build tree at %BUILD% & exit /b 1
)

rem ctest lives next to the cmake that configured the tree
set "CMAKE_EXE="
for /f "tokens=2 delims==" %%C in ('findstr /B /C:"CMAKE_COMMAND:INTERNAL=" "%BUILD%\CMakeCache.txt"') do set "CMAKE_EXE=%%C"
set "CTEST=ctest"
if defined CMAKE_EXE (
  for %%P in ("!CMAKE_EXE!") do if exist "%%~dpPctest.exe" set "CTEST=%%~dpPctest.exe"
)

set "START=%TIME%"
echo [1/3] seam test: KV tier protocol + prefetch gate
"%CTEST%" --test-dir "%BUILD%" -R "ninfer_kv_seam_test" --output-on-failure
if errorlevel 1 (echo FAIL: seam test & exit /b 1)
echo   PASS

if /i "%~1"=="quick" goto :done

if defined NINFER_GATE_MODEL (set "MODEL=%NINFER_GATE_MODEL%") else (set "MODEL=D:\infer-rtx4090\models\qwen3_6_35b_a3b_e2.ninfer")
set "CORPUS=%REPO%\.temp\kv\moe_corpus_varied.txt"
if not exist "%MODEL%" (
  echo FAIL: model not visible: %MODEL%
  echo       point NINFER_GATE_MODEL at the e2 artifact, or run: gate.cmd quick
  exit /b 1
)
if not exist "%CORPUS%" (
  echo FAIL: corpus not visible: %CORPUS% & exit /b 1
)

echo [2/3] perplexity, MoE streaming pool 12288 MiB (Track B, expect 7.776858 +-0.5%%)
set "OUTD=%TEMP%\ninfer_gate_ppl_%RANDOM%%RANDOM%"
set "NINFER_EXPERT_CACHE_MB=12288"
"%BUILD%\apps\ninfer-perplexity.exe" "%MODEL%" --text "%CORPUS%" --context 512 --stride 256 ^
  --output "%OUTD%" --log-level warning
if errorlevel 1 (echo FAIL: perplexity run & exit /b 1)

if /i "%~1"=="record" (
  powershell -NoProfile -Command ^
    "$v = [double](Get-Content -Raw '%OUTD%\report.json' | ConvertFrom-Json).overall.perplexity; $v.ToString('R') | Set-Content -NoNewline '%REPO%\.temp\kv\gate_golden_ppl.txt'; 'recorded golden PPL {0}' -f $v"
  if errorlevel 1 (echo FAIL: cannot record golden & exit /b 1)
  echo   PASS
  goto :done
)

powershell -NoProfile -Command ^
  "$v = [double](Get-Content -Raw '%OUTD%\report.json' | ConvertFrom-Json).overall.perplexity; $want = 7.776858; $d = [math]::Abs($v - $want) / $want; 'MoE PPL  = {0:F6}   (gate {1:F6}, delta {2:P3})' -f $v, $want, $d; if ($d -gt 0.005) { 'FAIL: MoE PPL drifted beyond 0.5%'; exit 1 }"
if errorlevel 1 (echo FAIL: MoE PPL gate & exit /b 1)
echo   PASS

echo [3/3] resident golden (Track A, bit exact) - only when a golden file exists
set "GOLDEN=%REPO%\.temp\kv\gate_golden_ppl.txt"
if not exist "%GOLDEN%" (
  echo   SKIP: no golden yet. Record one with: gate.cmd record
  goto :done
)
powershell -NoProfile -Command ^
  "$v = ([double](Get-Content -Raw '%OUTD%\report.json' | ConvertFrom-Json).overall.perplexity).ToString('R'); $g = (Get-Content -Raw '%GOLDEN%').Trim(); 'PPL now {0} / golden {1}' -f $v, $g; if ($v -ne $g) { 'FAIL: bit-exact drift'; exit 1 }"
if errorlevel 1 (echo FAIL: Track A gate & exit /b 1)
echo   PASS

:done
echo GATE OK  (elapsed %START% -^> %TIME%)
exit /b 0
