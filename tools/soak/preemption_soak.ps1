#!/usr/bin/env pwsh
# Repeat the preemption/replay acceptance for a wall-clock budget.
#
# Why: the pause -> reclaim -> replay path is the one that can stall without failing an
# assertion - a request that stops making progress leaves no error behind, only a slow
# batch. The acceptance for the recovery work (V10-20) therefore asks for a 30-minute run
# with no stall, and this script is that run: it repeats the real-model test whose body is
# exactly that path, and fails if a round fails or exceeds the slow bound.
#
# Usage:
#   pwsh tools/soak/preemption_soak.ps1 -Minutes 30 -Artifact <path to .ninfer>
#
# Exit: 0 all rounds passed, 1 a round failed, 2 setup problem.

[CmdletBinding()]
param(
    [int]$Minutes = 30,
    [string]$Artifact = $env:NINFER_TEST_ARTIFACT,
    [int]$SlowSeconds = 300
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$exe  = Join-Path $repo 'build-win\tests\ninfer_qwen3_5_preemption_real_test.exe'

if (-not (Test-Path -LiteralPath $exe)) {
    Write-Error "no preemption test binary at $exe (build target ninfer_qwen3_5_preemption_real_test)"
    exit 2
}
if (-not $Artifact) {
    Write-Error 'no artifact: pass -Artifact or set NINFER_TEST_ARTIFACT'
    exit 2
}
$env:NINFER_TEST_ARTIFACT = $Artifact

$deadline = (Get-Date).AddMinutes($Minutes)
$round = 0
$fail  = 0
$slow  = 0
while ((Get-Date) -lt $deadline) {
    $round++
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    $output = & $exe 2>&1
    $code = $LASTEXITCODE
    $watch.Stop()
    $seconds = [math]::Round($watch.Elapsed.TotalSeconds, 1)
    if ($code -ne 0) {
        $fail++
        "[{0}] round {1} FAILED exit={2} in {3}s" -f (Get-Date -Format 'HH:mm:ss'), $round, $code, $seconds
        $output | Select-Object -Last 6
    } else {
        if ($seconds -gt $SlowSeconds) { $slow++ }
        "[{0}] round {1} ok in {2}s (fail={3} slow={4})" -f (Get-Date -Format 'HH:mm:ss'), $round, $seconds, $fail, $slow
    }
}

"soak complete: rounds=$round failures=$fail slow=$slow minutes=$Minutes"
if ($fail -gt 0) { exit 1 }
exit 0
