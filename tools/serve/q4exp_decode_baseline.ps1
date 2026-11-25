# Qwen4Exp decode baseline: the number the IQ3_XXS target (85-97 tok/s, strata) is measured against.
# One short prompt, a long enough output to reach steady decoding, read back from the request line.
param(
    [string]$Model = 'R:\model\qwen3_8_flash_next_v5\qwen3_8_flash_next_v5_e2.ninfer',
    [string]$KvDtype = 'fp8',
    [int]$MaxContext = 262144,
    [int]$MaxTokens = 256,
    [int]$Port = 11438,
    [int]$Concurrency = 1,
    [switch]$QsaForceFull
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$outDir = Join-Path $env:TEMP 'ninfer-q4exp-decode'
New-Item -ItemType Directory -Path $outDir -Force | Out-Null
$stamp = "q4dec-$KvDtype-c$Concurrency-$([DateTime]::Now.ToString('HHmmss'))"
$err = Join-Path $outDir "$stamp.err"
Remove-Item $err -ErrorAction SilentlyContinue
if ($QsaForceFull) { $env:NINFER_QSA_FORCE_FULL = '1' }

$serve = Join-Path $repo 'build-win\apps\ninfer-serve.exe'
$serveArgs = @(
    $Model, '--host', '127.0.0.1', '--port', "$Port",
    '--max-context', "$MaxContext", '--max-concurrency', "$Concurrency",
    '--kv-dtype', $KvDtype, '--controller-mode', 'auto', '--kv-capacity', 'auto',
    '--host-context-mib', '16384', '--kv-residency-slack-mib', 'auto', '--kv-evict', 'auto',
    '--wddm-evictable-budget', '--top-p', '0.7'
)
$proc = Start-Process -FilePath $serve -ArgumentList $serveArgs `
    -RedirectStandardOutput (Join-Path $outDir "$stamp.log") -RedirectStandardError $err `
    -PassThru -WindowStyle Hidden

$ready = $false
for ($i = 0; $i -lt 200; $i++) {
    Start-Sleep -Seconds 2
    if ($proc.HasExited) { break }
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2 -UseBasicParsing
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch { }
}
if (-not $ready) { Write-Output 'FAIL serve not ready'; if (Test-Path $err) { Get-Content $err -Tail 15 }; exit 1 }

$uri = "http://127.0.0.1:$Port/v1/chat/completions"
$body = @{
    model = 'qwen3.8-flash-next'
    messages = @(@{ role='user'; content='Count from one to two hundred and fifty, one number per line.' })
    temperature = 0; max_tokens = $MaxTokens
} | ConvertTo-Json -Depth 8 -Compress

$sw = [Diagnostics.Stopwatch]::StartNew()
try {
    $r = Invoke-WebRequest -Uri $uri -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 1800 -UseBasicParsing
    Write-Output ("status={0} wall={1:N1}s" -f [int]$r.StatusCode, $sw.Elapsed.TotalSeconds)
} catch {
    $code = 0; if ($_.Exception.Response) { $code = [int]$_.Exception.Response.StatusCode }
    Write-Output ("status={0} wall={1:N1}s" -f $code, $sw.Elapsed.TotalSeconds)
}

Write-Output '=== request line (decode tok/s is the measurement) ==='
Get-Content -LiteralPath $err | Select-String -Pattern 'req#\d+ done' | ForEach-Object { $_.Line } | Write-Output
Write-Output '=== capacity and plan ==='
Get-Content -LiteralPath $err | Select-String -Pattern 'capacity \||kv-plan\]|\[kv\]' | Select-Object -First 6 | ForEach-Object { $_.Line } | Write-Output

if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue | ForEach-Object {
    $owner = Get-CimInstance Win32_Process -Filter ("ProcessId = " + $_.OwningProcess) -ErrorAction SilentlyContinue
    if ($owner -and $owner.CommandLine -match 'ninfer-serve\.exe') { Stop-Process -Id $owner.ProcessId -Force }
}
Write-Output "done $stamp"
