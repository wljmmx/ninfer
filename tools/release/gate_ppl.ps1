# The two bit-exact numeric gates of the port, as the regression harness runs them.
#
#   .\tools\release\gate_ppl.ps1                     # measure against the recorded golden
#   .\tools\release\gate_ppl.ps1 -Record             # print the measured pair for a re-baseline
#
# Both gates run ninfer-perplexity on the 27B artifact with the same context/stride the release
# gate uses (4096/2048), and both are exact: the PPL must repeat to the printed 6 significant
# digits and the eviction-score dump must hash to the same 16 hex digits.
#
# Golden, as recorded on 2026-11-04 (tools/../.temp/kv/regress.py):
#   default PPL        7.441868
#   selector-off dump  636e38de95046781
# The `e7ceb89dff4eb40b` in the v9 quick-reference is the older value, superseded by T042/T002.
#
# Measured 2026-11-11 on CUDA 13.5.37 + MSVC 14.52 with the E8 top-2 tie-break fix in tree:
#   PPL 7.445538, dump 61bfd6e5efee6395 (4547 bytes against 4548)
# Attribution is measured, not assumed: the same two binaries with `--kv-dtype bf16`, which never
# enters the E8 encoder, agree bit for bit (7.226977 both), so the toolchain is neutral and the
# move comes from the tie-break fix changing codes at exact ties. Re-baselining is a separate,
# explicit decision; this script never rewrites the expected values by itself.
param(
    [string]$Model = "R:\model\qwen3_8_27b_v3\qwen3_8_27b_v3.ninfer",
    [string]$PplCorpus = "D:\ninfer-rtx4090\ppl_corpus.txt",
    [string]$DumpText = "D:\ninfer-rtx4090\n4_q1.txt",
    [string]$KvDtype = "rk4v4-e8",
    [double]$ExpectPpl = 7.441868,
    [string]$ExpectDump = "636e38de95046781",
    [switch]$Record
)

$ErrorActionPreference = 'Continue'   # a native program writing to stderr is not a failure
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$exe = Join-Path $repo 'build-win\apps\ninfer-perplexity.exe'
if (-not (Test-Path $exe)) { throw "no perplexity binary at $exe (build target ninfer-perplexity)" }
foreach ($p in @($Model, $PplCorpus, $DumpText)) {
    if (-not (Test-Path $p)) { throw "missing input: $p" }
}

# The gate runs with the eviction switches clean except the dump one it is asking for.
$switches = @('NINFER_KV_EVICT_WINDOW', 'NINFER_KV_PAGE_SCORE_ROW_SPAN', 'NINFER_KV_EVICT_SCORES_DUMP')
$saved = @{}
foreach ($s in $switches) { $saved[$s] = [Environment]::GetEnvironmentVariable($s) }

function Invoke-Ppl {
    $log = [System.IO.Path]::GetTempFileName()
    & $exe $Model --text $PplCorpus --context 4096 --stride 2048 --kv-dtype $KvDtype *> $log
    $line = Select-String -Path $log -Pattern '^overall\s+\d+\s+[\d.]+\s+([\d.]+)' | Select-Object -Last 1
    $value = if ($line) { [double]$line.Matches[0].Groups[1].Value } else { $null }
    Remove-Item $log -ErrorAction SilentlyContinue
    return $value
}

function Invoke-Dump {
    $out = Join-Path ([System.IO.Path]::GetTempPath()) ("ninfer_gate_dump_{0}.tsv" -f $PID)
    Remove-Item $out -ErrorAction SilentlyContinue
    $env:NINFER_KV_EVICT_WINDOW = '65535'
    $env:NINFER_KV_PAGE_SCORE_ROW_SPAN = '8'
    $env:NINFER_KV_EVICT_SCORES_DUMP = $out
    try {
        & $exe $Model --text $DumpText --context 4096 --stride 2048 --kv-dtype $KvDtype *> $null
    } finally {
        foreach ($s in $switches) {
            [Environment]::SetEnvironmentVariable($s, $saved[$s])
        }
    }
    if (-not (Test-Path $out)) { return @{ sha = ''; bytes = 0 } }
    $bytes = (Get-Item $out).Length
    $sha = (Get-FileHash $out -Algorithm SHA256).Hash.ToLower().Substring(0, 16)
    Remove-Item $out -ErrorAction SilentlyContinue
    return @{ sha = $sha; bytes = $bytes }
}

$ppl = Invoke-Ppl
$dump = Invoke-Dump

if ($Record) {
    "measured PPL   {0}" -f $ppl
    "measured dump  {0}  ({1} bytes)" -f $dump.sha, $dump.bytes
    exit 0
}

$ok = $true
if ([math]::Abs($ppl - $ExpectPpl) -lt 5e-7) {
    "PPL   {0}            PASS (expect {1})" -f $ppl, $ExpectPpl
} else {
    "PPL   {0}            FAIL (expect {1})" -f $ppl, $ExpectPpl
    $ok = $false
}
if ($dump.sha -eq $ExpectDump) {
    "dump  {0}  PASS" -f $dump.sha
} else {
    "dump  {0}  FAIL (expect {1}, {2} bytes)" -f $dump.sha, $ExpectDump, $dump.bytes
    $ok = $false
}
if (-not $ok) { exit 1 }
