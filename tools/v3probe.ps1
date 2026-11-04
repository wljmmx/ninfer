# Chunk width comparison with a direct curl (no Start-Job: the job's curl never ran and made a healthy
# server look hung). Concurrency uses one curl process per lane.
param(
    [int]$Chunk   = 2048,
    [int]$Lanes   = 1,
    [int]$Port    = 18400,
    [string]$Tag  = '',
    [int]$CacheMiB = 19456,
    [int]$KvHeadroomMiB = 0
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$exe  = Join-Path $repo 'build-win\apps\ninfer-serve.exe'
$out  = 'H:\TeleAgentWorks\.temp\prefill-ab'
$model = (Get-ChildItem 'D:\*\models\qwen3_6_35b_a3b_e2.ninfer' | Select-Object -First 1).FullName
if (-not $Tag) { $Tag = "c$Chunk-l$Lanes" }
$log = Join-Path $out "v3-$Tag.log"
if (Test-Path $log) { Remove-Item $log -Force }

$env:NINFER_EXPERT_CACHE_MB = "$CacheMiB"
$env:NINFER_EXPERT_EPOCH    = '64'
Remove-Item Env:NINFER_PREFILL_BATCH_EXEC  -ErrorAction SilentlyContinue
Remove-Item Env:NINFER_PREFILL_ADMIT_FIRST -ErrorAction SilentlyContinue
Remove-Item Env:NINFER_PREFILL_BATCH       -ErrorAction SilentlyContinue

$proc = Start-Process -FilePath $exe -ArgumentList @(
    $model, '--port', "$Port", '--max-concurrency', "$Lanes", '--kv-capacity', 'auto',
    '--host-context-mib', '2048', '--prefill-chunk', "$Chunk",
        '--greedy', '--default-max-tokens', '8', '--kv-capacity-headroom-mib', "$KvHeadroomMiB"
) -RedirectStandardOutput (Join-Path $out "v3-$Tag.out.log") -RedirectStandardError $log -PassThru

try {
    $deadline = (Get-Date).AddSeconds(240)
    $ready = $false
    while ((Get-Date) -lt $deadline) {
        if ($proc.HasExited) { throw "died at startup" }
        $t = ''
        if (Test-Path $log) { $t = Get-Content -LiteralPath $log -Raw -ErrorAction SilentlyContinue }
        if ($t -match 'listening on http') { $ready = $true; break }
        Start-Sleep -Milliseconds 500
    }
    if (-not $ready) { throw 'no listening line within 240 s' }
    Start-Sleep -Seconds 2

    $prompts = @(Get-ChildItem 'H:\TeleAgentWorks\.temp' -Filter 'wd*.json' | Sort-Object Name | Select-Object -First $Lanes -ExpandProperty FullName)
    $url = "http://127.0.0.1:$Port/v1/chat/completions"
    $procs = @()
    for ($i = 0; $i -lt $prompts.Count; $i++) {
        $procs += Start-Process -FilePath 'curl.exe' -ArgumentList @(
            '-s', '-o', (Join-Path $out "v3-$Tag-$i.json"), '--max-time', '300',
            '-X', 'POST', $url, '-H', 'Content-Type: application/json',
            '--data-binary', "@$($prompts[$i].Replace('\','/'))"
        ) -PassThru -NoNewWindow
    }
    $sw = [Diagnostics.Stopwatch]::StartNew()
    foreach ($cp in $procs) { $cp.WaitForExit(300000) | Out-Null }
    $sw.Stop()
    "wall=$([math]::Round($sw.Elapsed.TotalSeconds,2))s"
    Start-Sleep -Seconds 2
} finally {
    if ($proc -and -not $proc.HasExited) { & taskkill.exe /PID $proc.Id /T /F 2>$null | Out-Null }
    Start-Sleep -Seconds 3
}

"--- $Tag ---"
Select-String -LiteralPath $log -Pattern 'req#.*done|req#.*failed|WARN  memory' | ForEach-Object { $_.Line.Trim() }




