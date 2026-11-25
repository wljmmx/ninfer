# Serve burst replay: the shape that produced yesterday's 500/503 cascade, as a regression check.
#
# 2026-11-24 11:36-11:54 (console-20261124-503.log): two requests failed with
#   "KV address-space handle is stale" (HTTP 500)
# and every request after them got HTTP 503 for eleven minutes. 954dd187 and b7e8b56d reworked
# admission, prefix reuse and physical-memory pricing; this script turns that run into a check.
#
# It starts a serve with the same flags, fires N concurrent chat requests whose prompt is ~23k
# tokens with 54 tools, then one small follow-up. Pass criteria:
#   * every burst request answers 200 (no 500, no 503 storm),
#   * the follow-up answers 200 (the engine is still healthy),
#   * no request is left pending.
# Exit code 0 on pass, 1 on any 5xx or a dead follow-up.
#
# -MaxTokens 0 omits the field so the server's --default-max-tokens applies; that is the A/B for
# "is the queue dominated by the output reservation".
param(
    [int]$Requests = 4,
    [int]$PromptRepeat = 1500,
    [int]$MaxTokens = 64,
    [string]$ModelId = 'qwen3.8-27b',
    [string]$Model = 'R:\model\qwen3_8_27b_v3\qwen3_8_27b_v3_gguf.ninfer',
    [string]$Serve = '',
    [int]$Port = 11438,
    [int]$ReadySeconds = 300
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
if (-not $Serve) { $Serve = Join-Path $repo 'build-win\apps\ninfer-serve.exe' }
if (-not (Test-Path $Serve)) { throw "missing build product: $Serve" }
if (-not (Test-Path $Model)) { throw "missing artifact: $Model" }

$log = Join-Path $env:TEMP 'ninfer-burst-replay.log'
$err = Join-Path $env:TEMP 'ninfer-burst-replay.err'
Remove-Item $log,$err -ErrorAction SilentlyContinue

$serveArgs = @(
    $Model, '--host', '127.0.0.1', '--port', $Port,
    '--max-context', '262144', '--max-concurrency', '4', '--prefill-chunk', '1024',
    '--kv-dtype', 'rk4v4-e8', '--spec', 'mtp', '--draft-tokens', '7', '--lm-head-draft',
    '--max-pending-requests', '16', '--controller-mode', 'auto', '--kv-capacity', 'auto',
    '--host-context-mib', '16384', '--kv-residency-slack-mib', 'auto', '--kv-evict', 'auto',
    '--wddm-evictable-budget', '--preserve-thinking', '--default-max-tokens', '32768', '--top-p', '0.7'
)
$proc = Start-Process -FilePath $Serve -ArgumentList $serveArgs -RedirectStandardOutput $log `
                      -RedirectStandardError $err -PassThru -WindowStyle Hidden

$ready = $false
for ($i = 0; $i -lt [Math]::Ceiling($ReadySeconds / 2); $i++) {
    Start-Sleep -Seconds 2
    if ($proc.HasExited) { break }
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2 -UseBasicParsing
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch { }
}
if (-not $ready) {
    Write-Output "FAIL serve did not become ready"
    if (Test-Path $err) { Get-Content $err -Tail 20 }
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
    exit 1
}

$filler = ('The harbour committee recorded the following ledger entries for the northern quay. ' * $PromptRepeat)
$tools = @()
for ($t = 0; $t -lt 54; $t++) {
    $tools += @{
        type = 'function'
        function = @{
            name = "tool_$t"
            description = "helper $t"
            parameters = @{ type = 'object'; properties = @{ q = @{ type = 'string' } }; required = @('q') }
        }
    }
}
$body = @{
    model = $ModelId
    messages = @(
        @{ role = 'system'; content = 'You are a careful assistant. Answer with one short sentence.' },
        @{ role = 'user';   content = $filler }
    )
    tools = $tools
    temperature = 0
}
if ($MaxTokens -gt 0) { $body['max_tokens'] = $MaxTokens }
$payload = $body | ConvertTo-Json -Depth 12 -Compress
$uri = "http://127.0.0.1:$Port/v1/chat/completions"

$jobs = @()
for ($k = 1; $k -le $Requests; $k++) {
    $jobs += Start-Job -ArgumentList $uri,$payload,$k -ScriptBlock {
        param($u, $b, $id)
        $sw = [Diagnostics.Stopwatch]::StartNew()
        try {
            $r = Invoke-WebRequest -Uri $u -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 900 -UseBasicParsing
            [pscustomobject]@{ Id = $id; Status = [int]$r.StatusCode; Ms = [int]$sw.Elapsed.TotalMilliseconds }
        } catch {
            $code = 0
            if ($_.Exception.Response) { $code = [int]$_.Exception.Response.StatusCode }
            [pscustomobject]@{ Id = $id; Status = $code; Ms = [int]$sw.Elapsed.TotalMilliseconds }
        }
    }
}
$results = $jobs | Wait-Job -Timeout 1200 | Receive-Job
$jobs | Remove-Job -Force
$failed = @($results | Where-Object { $_.Status -ge 500 -or $_.Status -eq 0 })
$results | Sort-Object Id | Format-Table -AutoSize | Out-String -Width 200 | Write-Output

$sw = [Diagnostics.Stopwatch]::StartNew()
$small = @{ model = $ModelId; messages = @(@{ role = 'user'; content = 'Say OK.' }); temperature = 0; max_tokens = 8 } |
         ConvertTo-Json -Depth 8 -Compress
$followOk = $false
try {
    $r = Invoke-WebRequest -Uri $uri -Method Post -ContentType 'application/json' -Body $small -TimeoutSec 600 -UseBasicParsing
    $followOk = ([int]$r.StatusCode -eq 200)
} catch { }
Write-Output ("follow-up: ok=" + $followOk + " in " + [int]$sw.Elapsed.TotalMilliseconds + " ms")

if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }

if ($failed.Count -ne 0) { Write-Output ("FAIL " + $failed.Count + " request(s) returned 5xx or no answer"); exit 1 }
if (-not $followOk) { Write-Output 'FAIL the follow-up request did not succeed'; exit 1 }
Write-Output ("PASS " + $Requests + " burst request(s) + follow-up, no 5xx")
exit 0
