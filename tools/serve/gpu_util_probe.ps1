# What the GPU actually does during prefill: start a serve with a given number of physical
# Device StateImages, fire the burst, and sample nvidia-smi every 200 ms for the whole window.
param(
    [int]$Images = 2,
    [switch]$UniquePrompts,
    [int]$Requests = 4,
    [int]$PromptRepeat = 1500,
    [int]$Port = 11438
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$outDir = Join-Path $env:TEMP 'ninfer-gpu-probe'
New-Item -ItemType Directory -Path $outDir -Force | Out-Null
$tag = "gpu-u$($Images)$(if ($UniquePrompts) { '-uniq' } else { '-same' })"
$csv = Join-Path $outDir "$tag.csv"
Remove-Item $csv -ErrorAction SilentlyContinue

$env:NINFER_STATE_IMAGE_DEVICE_SLOTS = "$Images"
$serve = Join-Path $repo 'build-win\apps\ninfer-serve.exe'
$model = 'R:\model\qwen3_8_27b_v3\qwen3_8_27b_v3_gguf.ninfer'
$log = Join-Path $outDir "$tag.log"
$err = Join-Path $outDir "$tag.err"
Remove-Item $log,$err -ErrorAction SilentlyContinue

$serveArgs = @(
    $model, '--host', '127.0.0.1', '--port', $Port,
    '--max-context', '262144', '--max-concurrency', '4', '--prefill-chunk', '1024',
    '--kv-dtype', 'rk4v4-e8', '--spec', 'mtp', '--draft-tokens', '7', '--lm-head-draft',
    '--max-pending-requests', '16', '--controller-mode', 'auto', '--kv-capacity', 'auto',
    '--host-context-mib', '16384', '--kv-residency-slack-mib', 'auto', '--kv-evict', 'auto',
    '--wddm-evictable-budget', '--preserve-thinking', '--default-max-tokens', '32768', '--top-p', '0.7'
)
$proc = Start-Process -FilePath $serve -ArgumentList $serveArgs -RedirectStandardOutput $log `
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

$filler = ('The harbour committee recorded the following ledger entries for the northern quay. ' * $PromptRepeat)
$tools = @()
for ($t = 0; $t -lt 54; $t++) {
    $tools += @{ type='function'; function=@{ name="tool_$t"; description="helper $t";
        parameters=@{ type='object'; properties=@{ q=@{ type='string' } }; required=@('q') } } }
}
$payloads = @()
for ($k = 1; $k -le $Requests; $k++) {
    $userContent = if ($UniquePrompts) { "Request $k of the northern quay survey. $filler" } else { $filler }
    $body = @{
        model = 'qwen3.8-27b'
        messages = @(
            @{ role='system'; content='You are a careful assistant. Answer with one short sentence.' },
            @{ role='user';   content=$userContent })
        tools = $tools; temperature = 0; max_tokens = 64
    }
    $payloads += ($body | ConvertTo-Json -Depth 12 -Compress)
}

# Sampler: 200 ms cadence, one line per sample. `-f` streams, no terminal output.
$sampler = Start-Process -FilePath 'nvidia-smi.exe' `
    -ArgumentList @('--query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,clocks.sm,memory.used',
                    '--format=csv,noheader,nounits', '-lms', '200', '-f', $csv) `
    -PassThru -WindowStyle Hidden
Start-Sleep -Milliseconds 600

$uri = "http://127.0.0.1:$Port/v1/chat/completions"
$jobs = @()
for ($k = 1; $k -le $Requests; $k++) {
    $jobs += Start-Job -ArgumentList $uri,$payloads[$k-1],$k -ScriptBlock {
        param($u,$b,$id)
        $sw = [Diagnostics.Stopwatch]::StartNew()
        try {
            $r = Invoke-WebRequest -Uri $u -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 900 -UseBasicParsing
            [pscustomobject]@{ Id=$id; Status=[int]$r.StatusCode; Ms=[int]$sw.Elapsed.TotalMilliseconds }
        } catch {
            $code = 0; if ($_.Exception.Response) { $code = [int]$_.Exception.Response.StatusCode }
            [pscustomobject]@{ Id=$id; Status=$code; Ms=[int]$sw.Elapsed.TotalMilliseconds }
        }
    }
}
$results = $jobs | Wait-Job -Timeout 1200 | Receive-Job
$jobs | Remove-Job -Force
Start-Sleep -Milliseconds 400
if (-not $sampler.HasExited) { Stop-Process -Id $sampler.Id -Force }
Start-Sleep -Milliseconds 300

$results | Sort-Object Id | Format-Table -AutoSize | Out-String -Width 120 | Write-Output

Write-Output ("=== GPU util over the request window ({0} samples) ===" -f $tag)
$rows = Get-Content -LiteralPath $csv | ForEach-Object {
    $p = $_ -split ','
    if ($p.Count -ge 6) {
        [pscustomobject]@{ gpu=[double]$p[1].Trim(); mem=[double]$p[2].Trim(); power=[double]$p[3].Trim(); sm=[double]$p[4].Trim(); used=[double]$p[5].Trim() }
    }
}
if ($rows.Count -gt 0) {
    $g = $rows.gpu | Sort-Object
    $p50 = $g[[int]($g.Count * 0.5)]
    $p10 = $g[[int]($g.Count * 0.1)]
    $p90 = $g[[Math]::Min($g.Count - 1, [int]($g.Count * 0.9))]
    "samples=$($g.Count) gpu_min=$($g[0]) p10=$p10 p50=$p50 p90=$p90 max=$($g[-1]) mean=$([Math]::Round(($g | Measure-Object -Average).Average,1))"
    "util>=90%: $([Math]::Round(100.0 * @($g | Where-Object { $_ -ge 90 }).Count / $g.Count,1))%  util>=50%: $([Math]::Round(100.0 * @($g | Where-Object { $_ -ge 50 }).Count / $g.Count,1))%"
    "mem_bw_util_mean=$([Math]::Round(($rows.mem | Measure-Object -Average).Average,1))%  sm_clock_mean=$([Math]::Round(($rows.sm | Measure-Object -Average).Average,0))MHz  power_mean=$([Math]::Round(($rows.power | Measure-Object -Average).Average,0))W"
}

Write-Output "=== throughput lines ==="
if (Test-Path $err) { Select-String -LiteralPath $err -Pattern 'throughput \|' | ForEach-Object { $_.Line } | Write-Output }

if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue | ForEach-Object {
    $owner = Get-CimInstance Win32_Process -Filter ("ProcessId = " + $_.OwningProcess) -ErrorAction SilentlyContinue
    if ($owner -and $owner.CommandLine -match 'ninfer-serve\.exe') { Stop-Process -Id $_.OwningProcess -Force }
}
Write-Output "done $tag"


