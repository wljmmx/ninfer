# P0-7: does a live decode lane slow a prefill lane? Two Device StateImages give each lane its own,
# so this measures the scheduling order (control -> decode -> prefill), not admission.
param(
    [int]$Port = 11438,
    [int]$DecodeTokens = 1024,
    [int]$PrefillRepeat = 1500
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$outDir = Join-Path $env:TEMP 'ninfer-p07'
New-Item -ItemType Directory -Path $outDir -Force | Out-Null
$stamp = "p07-$([DateTime]::Now.ToString('HHmmss'))"
$err = Join-Path $outDir "$stamp.err"
Remove-Item $err -ErrorAction SilentlyContinue

$serve = Join-Path $repo 'build-win\apps\ninfer-serve.exe'
$model = 'R:\model\qwen3_8_27b_v3\qwen3_8_27b_v3_gguf.ninfer'
$serveArgs = @(
    $model, '--host', '127.0.0.1', '--port', $Port,
    '--max-context', '262144', '--max-concurrency', '4', '--prefill-chunk', '1024',
    '--kv-dtype', 'rk4v4-e8', '--spec', 'mtp', '--draft-tokens', '7', '--lm-head-draft',
    '--max-pending-requests', '16', '--controller-mode', 'auto', '--kv-capacity', 'auto',
    '--host-context-mib', '16384', '--kv-residency-slack-mib', 'auto', '--kv-evict', 'auto',
    '--wddm-evictable-budget', '--preserve-thinking', '--default-max-tokens', '32768', '--top-p', '0.7'
)
$proc = Start-Process -FilePath $serve -ArgumentList $serveArgs -RedirectStandardOutput (Join-Path $outDir "$stamp.log") `
                      -RedirectStandardError $err -PassThru -WindowStyle Hidden

$ready = $false
for ($i = 0; $i -lt 150; $i++) {
    Start-Sleep -Seconds 2
    if ($proc.HasExited) { break }
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2 -UseBasicParsing
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch { }
}
if (-not $ready) { Write-Output 'FAIL serve not ready'; if (Test-Path $err) { Get-Content $err -Tail 15 }; exit 1 }

$uri = "http://127.0.0.1:$Port/v1/chat/completions"

# Lane A: a short prompt that then decodes for a long time.
$decodeBody = @{
    model = 'qwen3.8-27b'
    messages = @(@{ role='user'; content='Write out a long enumeration of integers, one per line, slowly.' })
    temperature = 0; max_tokens = $DecodeTokens
} | ConvertTo-Json -Depth 8 -Compress
$decodeJob = Start-Job -ArgumentList $uri,$decodeBody -ScriptBlock {
    param($u,$b)
    $sw = [Diagnostics.Stopwatch]::StartNew()
    try {
        $r = Invoke-WebRequest -Uri $u -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 900 -UseBasicParsing
        [pscustomobject]@{ Kind='decode'; Status=[int]$r.StatusCode; Ms=[int]$sw.Elapsed.TotalMilliseconds }
    } catch {
        $c=0; if ($_.Exception.Response) { $c=[int]$_.Exception.Response.StatusCode }
        [pscustomobject]@{ Kind='decode'; Status=$c; Ms=[int]$sw.Elapsed.TotalMilliseconds }
    }
}

Start-Sleep -Milliseconds 1500   # let the decode lane enter steady decoding

# Lane B: the 23k prefill, while the decode lane runs.
$filler = 'The harbour committee recorded the following ledger entries for the northern quay. ' * $PrefillRepeat
$prefillBody = @{
    model = 'qwen3.8-27b'
    messages = @(@{ role='user'; content=$filler })
    temperature = 0; max_tokens = 8
} | ConvertTo-Json -Depth 8 -Compress
$sw = [Diagnostics.Stopwatch]::StartNew()
try {
    $r = Invoke-WebRequest -Uri $uri -Method Post -ContentType 'application/json' -Body $prefillBody -TimeoutSec 900 -UseBasicParsing
    $prefill = [pscustomobject]@{ Kind='prefill'; Status=[int]$r.StatusCode; Ms=[int]$sw.Elapsed.TotalMilliseconds }
} catch {
    $c=0; if ($_.Exception.Response) { $c=[int]$_.Exception.Response.StatusCode }
    $prefill = [pscustomobject]@{ Kind='prefill'; Status=$c; Ms=[int]$sw.Elapsed.TotalMilliseconds }
}

$decoders = $decodeJob | Wait-Job -Timeout 900 | Receive-Job
$decodeJob | Remove-Job -Force
@($prefill) + @($decoders) | Format-Table -AutoSize | Out-String -Width 120 | Write-Output

Write-Output "=== request detail ==="
Get-Content -LiteralPath $err | Select-String -Pattern 'req#\d+ done' | ForEach-Object { $_.Line } | Write-Output
Write-Output "=== throughput (prefill column is the measurement) ==="
Get-Content -LiteralPath $err | Select-String -Pattern 'throughput \|' | ForEach-Object { $_.Line } | Write-Output

if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue | ForEach-Object {
    $owner = Get-CimInstance Win32_Process -Filter ("ProcessId = " + $_.OwningProcess) -ErrorAction SilentlyContinue
    if ($owner -and $owner.CommandLine -match 'ninfer-serve\.exe') { Stop-Process -Id $owner.ProcessId -Force }
}
Write-Output "done $stamp"
