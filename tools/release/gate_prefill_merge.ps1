# The merged-prefill equality gate: a merged pass and the per-lane path must produce the same
# tokens, and the merge must actually happen.
#
#   .\tools\release\gate_prefill_merge.ps1          # both scenarios, PASS/FAIL each
#   .\tools\release\gate_prefill_merge.ps1 -Scenario merged-prefill
#
# Why this is a gate and not a note. Two env switches carry the merged prefill
# (NINFER_PREFILL_BATCH_EXEC runs the pass, NINFER_PREFILL_ADMIT_FIRST holds admission so a batch can
# form). Both are experiments until something proves that turning them on cannot change an answer,
# because the pass executes model layers through a rectangle instead of one lane's chunk: a kernel
# that routes differently at 512 columns would change tokens, not just throughput. The scenarios
# below are that proof - each one builds a reference engine with the merge off, runs the same
# prompts, and compares the generated token ids.
#
# Scenarios (see tests/models/qwen3_5/test_engine_prefix_real.cpp):
#   merged-prefill            two 256-token prompts, three phases (fresh pair, reservation-backed
#                             settled tail, default slot budget). Phases 1 and 3 assert the FULL
#                             generated sequence against the per-lane reference and require a
#                             [prefill-merged] rows=2 pass to have run; phase 2 asserts completion
#                             and the merge (its 9-column tail routes differently by construction).
#   merged-prefill-production four lanes at the production chunk width. Every load asserts the BEGIN
#                             token - the prefill's own sample, which is what the merge owns - and
#                             reports the rest of the sequence; the 256-token loads measure the whole
#                             sequence equal as well, the 3*1024+256 load's decode runs at a different
#                             batch size (four lanes together against a one-lane oracle) and is
#                             reported, not asserted.
#
# Status on 2026-11-24 (RTX 4090, the 27B dense artifact): both scenarios PASS. The four-lane red
# rows this gate first reported were the ORACLE, not the merge: the per-lane reference ran four lanes
# whose chunks executed in different compositions, and with four IDENTICAL prompts it answered
# 68, 330, 330, 198 - the same prompt, four answers. Measured with NINFER_MERGE_PROBE, the merged
# pass hands every row the same (lane, kv_row, base, count, state src/dst) the single-lane turn uses,
# so the bookkeeping was never in question either.
#
# The fix is in the scenario: the reference is now ONE lane running one prompt at a time - the only
# composition the merge claims to reproduce. With that oracle the merged arm is bit-exact on every
# 256-token load (begin AND the whole sequence), and on the 3*1024+256 load its begin is bit-exact.
#
# Why the bisect rows stay: four-256-roomy (roomier pool) and four-256-chunk512 (chunk budget equal
# to the rectangle) were both red under the old oracle and are now exact evidence that neither the
# pool nor the column budget is involved. A prompt that exactly fills one chunk is not a row:
# measured, the merge never fires for it. Promote on this gate, not on a tolerance: a merge that
# changes tokens is not a performance option.
param(
    [string]$Model = "R:\model\qwen3_8_27b_v3\qwen3_8_27b_v3.ninfer",
    [string[]]$Scenario = @("merged-prefill", "merged-prefill-production")
)

$ErrorActionPreference = 'Continue'   # a native program writing to stderr is not a failure
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$exe = Join-Path $repo 'build-win\tests\ninfer_qwen3_5_prefix_real_test.exe'

if (-not (Test-Path $exe)) {
    Write-Output "missing $exe - build ninfer_qwen3_5_prefix_real_test first"
    exit 3
}

$failed = 0
foreach ($name in $Scenario) {
    $env:NINFER_TEST_ARTIFACT = $Model
    $env:NINFER_PREFIX_REAL_SCENARIO = $name
    $started = Get-Date
    $output = & $exe 2>&1 | Out-String
    $seconds = [math]::Round(((Get-Date) - $started).TotalSeconds, 1)
    if ($LASTEXITCODE -eq 77) {
        Write-Output ("   {0,-26} SKIP   (NINFER_TEST_ARTIFACT is not set)" -f $name)
        continue
    }
    $ok = ($LASTEXITCODE -eq 0)
    if (-not $ok) {
        $failed++
        $reason = ($output -split "`n" | Where-Object { $_ -match 'differs|never|failed|mismatch' } |
                   Select-Object -First 3) -join ' '
        Write-Output ("   {0,-26} FAIL   {1} s  {2}" -f $name, $seconds, $reason.Trim())
    } else {
        Write-Output ("   {0,-26} PASS   {1} s" -f $name, $seconds)
    }
}

Write-Output ""
if ($failed -ne 0) {
    Write-Output "prefill merge gate: FAIL ($failed of $($Scenario.Count) scenario(s))"
    exit 1
}
Write-Output "prefill merge gate: PASS"
exit 0
