# P1-2: perplexity at a long context for each KV representation, against the bf16 baseline.
# The shipped ppl_corpus is ~11k tokens, too short for a 32k window, so the corpus here is the
# upstream maintainer docs concatenated in name order: real prose, no repetition.
param(
    [int]$Context = 32768,
    [int]$Stride  = 16384,
    [string[]]$Dtypes = @('bf16', 'fp8', 'rk4v4-e8'),
    [int]$CorpusBytes = 220000,
    [string]$Model = '',
    [string]$Corpus = '',
    [switch]$QsaQuantKv,
    [switch]$QsaForceFull,
    [string]$Tag = ''
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$outDir = Join-Path $env:TEMP 'ninfer-p12'
New-Item -ItemType Directory -Path $outDir -Force | Out-Null
$corpus = if ($Corpus) { $Corpus } else { Join-Path $outDir "long_corpus_$Context.txt" }
$docs = Join-Path (Split-Path $repo -Parent) 'infer-upstream\docs'

if (-not (Test-Path $corpus)) {
    if (-not (Test-Path $docs)) { throw "missing docs: $docs" }
    $sb = New-Object System.Text.StringBuilder
    foreach ($f in (Get-ChildItem $docs -Recurse -Filter '*.md' | Sort-Object Name)) {
        if ($sb.Length -ge $CorpusBytes) { break }
        [void]$sb.AppendLine((Get-Content -LiteralPath $f.FullName -Raw))
    }
    Set-Content -LiteralPath $corpus -Value $sb.ToString() -Encoding utf8
}
Write-Output "corpus=$corpus bytes=$((Get-Item $corpus).Length)"

$ppl = Join-Path $repo 'build-win\apps\ninfer-perplexity.exe'
$model = if ($Model) { $Model } else { 'R:\model\qwen3_8_27b_v3\qwen3_8_27b_v3.ninfer' }
if (-not (Test-Path $model)) { $model = 'R:\model\qwen3_8_27b_v3\qwen3_8_27b_v3_gguf.ninfer' }
if ($QsaQuantKv) {
    # Qwen4Exp's indexed layers only admit a low-bit KV when this gate is open
    # (model_instance.cpp: qsa_kv_storage_admitted); otherwise the run is downgraded to bf16.
    $env:NINFER_QSA_SPARSE_GQA_QUANT_KV = '1'
    Write-Output 'NINFER_QSA_SPARSE_GQA_QUANT_KV=1'
}
if ($QsaForceFull) {
    # The dense reference arm of the same indexed layers (text.cpp: indexed && !qsa_dense_arm()).
    $env:NINFER_QSA_FORCE_FULL = '1'
    Write-Output 'NINFER_QSA_FORCE_FULL=1'
}
Write-Output "model=$model context=$Context stride=$Stride"

# The gate's own report format: "overall <count> <something> <ppl>".
$results = @()
foreach ($dt in $Dtypes) {
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $logPath = Join-Path $outDir "ppl-$Tag$dt.log"
    # Tee so a long run's progress is readable while it runs; the parse below reads the same file.
    & $ppl $model --text $corpus --context $Context --stride $Stride --kv-dtype $dt 2>&1 |
        Tee-Object -FilePath $logPath | Out-Null
    $code = $LASTEXITCODE
    $text = Get-Content -LiteralPath $logPath -Raw
    $m = [regex]::Match($text, 'overall\s+\d+\s+[\d.]+\s+([\d.]+)')
    $val = if ($m.Success) { [double]$m.Groups[1].Value } else { $null }
    $results += [pscustomobject]@{ Dtype = $dt; Exit = $code; Ppl = $val; Secs = [int]$sw.Elapsed.TotalSeconds }
    Write-Output ("dtype={0} exit={1} ppl={2} {3}s" -f $dt, $code, $val, [int]$sw.Elapsed.TotalSeconds)
}

$base = ($results | Where-Object { $_.Dtype -eq 'bf16' }).Ppl
$rows = foreach ($r in $results) {
    $delta = if ($base -and $r.Ppl) { ($r.Ppl - $base) / $base * 100.0 } else { $null }
    [pscustomobject]@{
        Dtype    = $r.Dtype
        Exit     = $r.Exit
        Ppl      = $r.Ppl
        DeltaPct = if ($null -ne $delta) { [Math]::Round($delta, 3) } else { $null }
        Secs     = $r.Secs
    }
}
$rows | Format-Table -AutoSize | Out-String -Width 120 | Write-Output
Write-Output "done"
