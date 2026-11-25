# P1-1: one session, three prompt lengths, GPU sampled throughout.
# Every prompt carries a distinct opening so no request inherits another's prefix; lengths differ by
# 8x so the attention term grows while the linear term stays fixed.
param(
    [int]$Port = 11438,
    [int]$Rounds = 2
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$outDir = Join-Path $env:TEMP 'ninfer-p11'
New-Item -ItemType Directory -Path $outDir -Force | Out-Null
$stamp = "p11-$([DateTime]::Now.ToString('HHmmss'))"
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
$repeats = @(267, 1067, 2134)   # ~4.1k / ~16.4k / ~32.8k prompt tokens
$sw = [Diagnostics.Stopwatch]::StartNew()
$rows = @()
foreach ($round in 1..$Rounds) {
    foreach ($rep in $repeats) {
        $filler = "Survey round $round length $rep. " + ('The harbour committee recorded the following ledger entries for the northern quay. ' * $rep)
        $body = @{
            model = 'qwen3.8-27b'
            messages = @(
                @{ role='system'; content='You are a careful assistant. Answer with one short sentence.' },
                @{ role='user';   content=$filler })
            temperature = 0; max_tokens = 16
        }
        $payload = $body | ConvertTo-Json -Depth 8 -Compress
        $w = [Diagnostics.Stopwatch]::StartNew()
        try {
            $resp = Invoke-WebRequest -Uri $uri -Method Post -ContentType 'application/json' -Body $payload -TimeoutSec 900 -UseBasicParsing
            $rows += [pscustomobject]@{ Round=$round; Repeat=$rep; Status=[int]$resp.StatusCode; Ms=[int]$w.Elapsed.TotalMilliseconds }
        } catch {
            $code = 0; if ($_.Exception.Response) { $code = [int]$_.Exception.Response.StatusCode }
            $rows += [pscustomobject]@{ Round=$round; Repeat=$rep; Status=$code; Ms=[int]$w.Elapsed.TotalMilliseconds }
        }
    }
}
$rows | Format-Table -AutoSize | Out-String -Width 120 | Write-Output
Write-Output ("session elapsed {0:N1} s" -f $sw.Elapsed.TotalSeconds)

Write-Output "=== request detail (prompt tokens / TTFT / prefill / phases) ==="
Get-Content -LiteralPath $err | Select-String -Pattern 'req#\d+ done' | ForEach-Object { $_.Line } | Write-Output

Write-Output "=== throughput ==="
Get-Content -LiteralPath $err | Select-String -Pattern 'throughput \|' | ForEach-Object { $_.Line } | Write-Output

if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue | ForEach-Object {
    $owner = Get-CimInstance Win32_Process -Filter ("ProcessId = " + $_.OwningProcess) -ErrorAction SilentlyContinue
    if ($owner -and $owner.CommandLine -match 'ninfer-serve\.exe') { Stop-Process -Id $_.OwningProcess -Force }
}
Write-Output "done $stamp"
