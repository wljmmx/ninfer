# The QSA gate: perplexity on the one artifact here that runs indexed_attention layers.
#
#   .\tools\release\gate_qsa.ps1              # measure against the recorded golden
#   .\tools\release\gate_qsa.ps1 -Record      # print the measured value for a re-baseline
#
# Why this exists. qsa_prepare / qsa_indexer / qsa_sparse_gqa are only reachable from a layer whose
# text layer_types entry is "indexed_attention" (models/text_core.cpp counts exactly those into
# TextConfig::indexed_attention_layers, and both startup.cpp and model_instance.cpp gate the QSA
# path on that count being non-zero). The two gates the port already had are blind to it:
#
#   27B     qwen3_8_27b_v3.ninfer        16 full_attention + 48 linear_attention, 0 indexed
#   35B-A3B qwen3_6_35b_a3b_e2.ninfer    10 full_attention + 30 linear_attention, 0 indexed
#
# so tools/release/gate_ppl.ps1 (27B PPL + dump) and tools/gate.cmd (MoE PPL) cannot see a QSA
# kernel change: they pass whether or not qsa_sparse_gqa is correct. That is how the V11-07 slice
# regression (0b8ba8b8) shipped with "27B gate unchanged" as its evidence.
#
# qwen3_8_flash_next_v5_e2.ninfer is 12 indexed_attention layers and is the artifact that runs the
# whole QSA route on bf16. This gate measures it, and it refuses to pass an artifact that declares
# no indexed layer, so it cannot silently degrade into a gate for work it never runs.
#
# Route notes: qsa_sparse_gqa admits bf16 unless NINFER_QSA_SPARSE_GQA_QUANT_KV=1, and the low-bit
# QSA profiles are the ones covered by the Op sweeps (tests/.../test_sparse_gqa.cpp --kv-cache ...),
# not by this artifact. The expert cache size only moves throughput, but it is pinned here so two
# runs of this gate are comparable.
param(
    [string]$Model = "R:\model\qwen3_8_flash_next_v5\qwen3_8_flash_next_v5_e2.ninfer",
    [string]$PplCorpus = "",
    [ValidateSet('bf16', 'int8', 'fp8', 'nvfp4', 'k8v4', 'rk8v4', 'rk4v4', 'rk4v4-e8', 'rk2v4-e8', 'rk4v2-e8')]
    [string]$KvDtype = 'bf16',
    [int]$Context = 512,
    [int]$Stride = 256,
    [int]$ExpertCacheMb = 12288,
    # One recorded value per KV storage. bf16 is the qualified route; the low-bit ones are implemented
    # in qsa_storage_supported and pass the Op sweeps, but are only admitted by the runtime once a
    # model-level value exists here, which is what -Record fills in.
    [hashtable]$Goldens = @{ bf16 = 15.533755882784398; 'rk4v4-e8' = 15.3738122181841 },
    [double]$ExpectPpl = 0.0,
    [double]$Tolerance = 0.001,
    [switch]$Record
)

$ErrorActionPreference = 'Continue'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$exe = Join-Path $repo 'build-win\apps\ninfer-perplexity.exe'
if (-not (Test-Path $exe)) { throw "no perplexity binary at $exe (build target ninfer-perplexity)" }
if (-not (Test-Path $Model)) { throw "missing model: $Model" }

# The corpus ships with the script: a gate that reads its text from .temp is not reproducible on a
# machine that never had that file. 1,358 bytes is one 512-token window plus its tail, which is one
# expert sweep of a 124 GB artifact - about as small as this gate can be and still be a model run.
if ($PplCorpus -eq '') { $PplCorpus = Join-Path $PSScriptRoot 'qsa_gate_corpus.txt' }
if (-not (Test-Path $PplCorpus)) { throw "missing corpus: $PplCorpus" }

# The artifact's own header, not its name: count the indexed layers before measuring anything.
$stream = [System.IO.File]::OpenRead($Model)
try {
    $buffer = New-Object byte[] (8MB)
    $read = $stream.Read($buffer, 0, $buffer.Length)
    $header = [System.Text.Encoding]::UTF8.GetString($buffer, 0, $read)
} finally { $stream.Dispose() }
$match = [regex]::Match($header, '"layer_types":\[(.*?)\]')
if (-not $match.Success) { throw "$Model has no text layer_types in its first 8 MiB" }
$types = $match.Groups[1].Value
$indexed = ([regex]::Matches($types, '"indexed_attention"')).Count
if ($indexed -eq 0) {
    throw ("$Model declares no indexed_attention layer, so it never runs qsa_sparse_gqa: " +
           "this gate would pass without touching QSA. Point -Model at an artifact that does.")
}
"indexed layers {0}   ({1})" -f $indexed, (Split-Path -Leaf $Model)

$out = Join-Path ([System.IO.Path]::GetTempPath()) ("ninfer_gate_qsa_{0}" -f $PID)
if (Test-Path $out) { Remove-Item -Recurse -Force $out }
$env:NINFER_EXPERT_CACHE_MB = "$ExpertCacheMb"
# The low-bit QSA routes are behind this switch; bf16 needs it off so an ambient value cannot move the
# qualified route.
$env:NINFER_QSA_SPARSE_GQA_QUANT_KV = if ($KvDtype -eq 'bf16') { '0' } else { '1' }
$log = [System.IO.Path]::GetTempFileName()
try {
    & $exe $Model --text $PplCorpus --context $Context --stride $Stride --kv-dtype $KvDtype `
        --output $out --log-level warning *> $log
    if ($LASTEXITCODE -ne 0) {
        Get-Content $log -Tail 20
        throw "perplexity run failed (exit $LASTEXITCODE)"
    }
    $report = Join-Path $out 'report.json'
    if (-not (Test-Path $report)) { throw "no report.json at $report" }
    $ppl = [double](Get-Content -Raw $report | ConvertFrom-Json).overall.perplexity
} finally {
    Remove-Item $log -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $out -ErrorAction SilentlyContinue
}

if ($Record) {
    "measured PPL   {0:R}   storage {1}" -f $ppl, $KvDtype
    exit 0
}
# A -ExpectPpl override wins; otherwise the storage's recorded value, and an unrecorded storage
# refuses to pass rather than comparing against bf16's number.
$expect = if ($ExpectPpl -gt 0) { $ExpectPpl } elseif ($Goldens.ContainsKey($KvDtype)) { [double]$Goldens[$KvDtype] } else { 0.0 }
if ($expect -le 0) {
    throw ("no golden recorded for KV storage '{0}': run with -Record once, then add it to `$Goldens" -f $KvDtype)
}

$delta = [math]::Abs($ppl - $expect) / $expect
if ($delta -le $Tolerance) {
    "QSA PPL  {0:F6}   PASS (storage {1}, expect {2:F6}, delta {3:P3} <= {4:P3})" -f $ppl, $KvDtype, $expect, $delta, $Tolerance
    exit 0
}
"QSA PPL  {0:F6}   FAIL (storage {1}, expect {2:F6}, delta {3:P3} > {4:P3})" -f $ppl, $KvDtype, $expect, $delta, $Tolerance
exit 1
