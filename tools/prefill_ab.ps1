<#
.SYNOPSIS
  Strict A/B for the merged-prefill switch: same server arguments, same prompts, and a hard assertion
  that the merged pass actually fired in the "on" arm.

.DESCRIPTION
  Byte-identical output is the only acceptance criterion this feature has, and it is only meaningful
  under conditions that earlier ad-hoc runs got wrong:

    1. The tier must be fully resident (cpu = 0). At a partial tier some routes are served by the CPU
       half in int8, which differs from GPU bf16 by ~0.4-0.5% PPL, so tokens diverge for reasons that
       have nothing to do with the change under test.
    2. The two arms must be launched with identical arguments - host context, KV capacity, chunk width,
       concurrency, sampling. Anything that changes the KV page count or the state-slot count makes the
       comparison meaningless.
    3. "Same output" proves nothing if the feature never ran. The merged pass is gated twice: it needs
       NINFER_PREFILL_BATCH_EXEC, and it needs the admit gate (NINFER_PREFILL_ADMIT_FIRST) to hold a
       head until every queued lane is bound - without the hold, two lanes never have a chunk in the
       same pass. The first version of this script enabled exec but explicitly deleted admit-first, so
       the "on" arm was the "off" arm plus a flag that could not fire, and it reported GATE OK on a
       no-op. This version enables both and fails when the log has no [prefill-merged] line.

  Why the shape of the prompt matters: advance_prefill_batch() refuses any pass whose widest row maps
  above the 16-column class, because gdn_input_proj_conv_snapshot only accepts width <= 16 when
  batch > 1. With the default 1024-token chunk a 3,260-token prompt ends on a 188-column tail, so no
  merged pass is even representable. To exercise the merged path today the prompts must end inside the
  16-column tail (L = k*PrefillChunk + m, m in 1..16). Use -AllowNoMerge to accept an arm that provably
  cannot merge, e.g. a production-shape regression run at chunk 1024.

  Both arms run sequentially (a 19.4 GiB pool does not fit twice on one card), each twice, so the
  script also shows whether the engine is deterministic before it compares across arms. A second round
  that hits the checkpoint cache is reported as CACHED and excluded from the determinism verdict
  instead of being counted as a free pass.

.EXAMPLE
  powershell -File tools\prefill_ab.ps1 -Prompts @('.temp\p16a.json','.temp\p16b.json')
  powershell -File tools\prefill_ab.ps1 -PrefillChunk 1024 -AllowNoMerge
#>
[CmdletBinding()]
param(
    [int]$Port            = 18300,
    [int]$Concurrency     = 2,
    [int]$CacheMiB        = 19456,
    [int]$HostMiB         = 2048,
    [int]$PrefillChunk    = 1024,
    [string]$Model        = '',
    [string[]]$Prompts    = @(),
    [string]$OutDir       = 'H:\TeleAgentWorks\.temp\prefill-ab',
    [int]$MaxTokens       = 64,
    [int]$StartupSeconds  = 240,
    [int]$RequestSeconds  = 1800,
    [switch]$AllowNoMerge
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$exe  = Join-Path $repo 'build-win\apps\ninfer-serve.exe'
if (-not (Test-Path $exe)) { throw "missing $exe" }

if (-not $Model) {
    $found = Get-ChildItem 'D:\*\models\qwen3_6_35b_a3b_e2.ninfer' -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $found) { throw 'model not found under D:\ models; pass -Model' }
    $Model = $found.FullName
}
if (-not (Test-Path $Model)) { throw "missing model $Model" }

if ($Prompts.Count -eq 0) {
    $Prompts = Get-ChildItem 'H:\TeleAgentWorks\.temp\wd*.json' -ErrorAction SilentlyContinue |
        Sort-Object Name | Select-Object -First $Concurrency -ExpandProperty FullName
}
if ($Prompts.Count -eq 0) { throw 'no prompt files: pass -Prompts or put wd*.json under .temp' }
if ($Prompts.Count -gt $Concurrency) { throw "more prompts than concurrency: they would queue and not overlap" }

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

function Test-PortFree([int]$Port) {
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
        $listener.Start()
        $listener.Stop()
        return $true
    } catch {
        return $false
    }
}

function Start-Arm([string]$Arm) {
    $errLog = Join-Path $OutDir "$Arm.log"
    $outLog = Join-Path $OutDir "$Arm.out.log"
    if (Test-Path $errLog) { Remove-Item $errLog -Force }
    if (Test-Path $outLog) { Remove-Item $outLog -Force }

    if (-not (Test-PortFree $Port)) { throw "port $Port is already in use; a stale server would be measured" }

    $env:NINFER_EXPERT_CACHE_MB = "$CacheMiB"
    $env:NINFER_EXPERT_EPOCH    = '64'
    # Report-only: prints [prefill-batch] / [prefill-rows] / [prefill-call]; changes no behaviour.
    $env:NINFER_PREFILL_BATCH   = 'all'
    if ($Arm -eq 'on') {
        # The feature is the pair, not the exec flag alone: exec decides whether a merged pass is
        # attempted, admit-first decides whether two lanes ever have a chunk in the same pass.
        $env:NINFER_PREFILL_BATCH_EXEC  = '1'
        $env:NINFER_PREFILL_ADMIT_FIRST = '1'
    } else {
        Remove-Item Env:NINFER_PREFILL_BATCH_EXEC  -ErrorAction SilentlyContinue
        Remove-Item Env:NINFER_PREFILL_ADMIT_FIRST -ErrorAction SilentlyContinue
    }

    $proc = Start-Process -FilePath $exe -ArgumentList @(
        $Model, '--port', "$Port", '--max-concurrency', "$Concurrency",
        '--kv-capacity', 'auto', '--host-context-mib', "$HostMiB", '--prefill-chunk', "$PrefillChunk",
        '--greedy', '--default-max-tokens', "$MaxTokens"
    ) -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru

    $started  = Get-Date
    $deadline = $started.AddSeconds($StartupSeconds)
    while ((Get-Date) -lt $deadline) {
        if ($proc.HasExited) {
            $tail = Get-Content -LiteralPath $errLog -Tail 4 -ErrorAction SilentlyContinue
            throw "server for arm '$Arm' exited during startup: $($tail -join ' | ')"
        }
        $client = $null
        try {
            $client = [System.Net.Sockets.TcpClient]::new()
            if ($client.ConnectAsync('127.0.0.1', $Port).Wait(300)) {
                "arm ${Arm}: listening after $([int]((Get-Date) - $started).TotalSeconds) s"
                return $proc
            }
        } catch {
            # not up yet
        } finally {
            if ($client) { $client.Dispose() }
        }
        Start-Sleep -Milliseconds 700
    }
    throw "server for arm '$Arm' did not listen within $StartupSeconds s; see $errLog"
}

function Stop-Arm($proc) {
    if ($proc -and -not $proc.HasExited) {
        & taskkill.exe /PID $proc.Id /T /F 2>$null | Out-Null
    }
    for ($i = 0; $i -lt 20; $i++) {
        if (Test-PortFree $Port) { break }
        Start-Sleep -Milliseconds 500
    }
    Get-Process ninfer-serve -ErrorAction SilentlyContinue | ForEach-Object {
        & taskkill.exe /PID $_.Id /T /F 2>$null | Out-Null
    }
    Start-Sleep -Seconds 2
}

function Invoke-Round([string]$Arm, [int]$Round) {
    $jobs  = @()
    $paths = @()
    for ($i = 0; $i -lt $Prompts.Count; $i++) {
        $body = Get-Content -LiteralPath $Prompts[$i] -Raw
        $path = Join-Path $OutDir "$Arm`-r$Round-$i.json"
        $reqPath = "$path.req"
        [System.IO.File]::WriteAllText($reqPath, $body, [System.Text.UTF8Encoding]::new($false))
        $paths += $path
        # --data-binary @file: a compressed JSON body carries dozens of double quotes and passing it as a
        # native argument mangles them; the file form is the only reliable one.
        $jobs += Start-Job -ScriptBlock {
            param($port, $reqPath, $path)
            $r = curl.exe -s -X POST "http://127.0.0.1:$port/v1/chat/completions" `
                -H 'Content-Type: application/json' --data-binary "@$reqPath"
            [System.IO.File]::WriteAllText($path, $r, [System.Text.UTF8Encoding]::new($false))
        } -ArgumentList $Port, $reqPath, $path
    }

    # Wait-Job with -Timeout applies the timeout per job in a pipeline, so N jobs could block N*timeout.
    # One deadline for the whole round instead.
    $deadline = (Get-Date).AddSeconds($RequestSeconds)
    while ((Get-Date) -lt $deadline) {
        $running = @($jobs | Where-Object { $_.State -eq 'Running' })
        if ($running.Count -eq 0) { break }
        Start-Sleep -Milliseconds 250
    }
    $stuck = @($jobs | Where-Object { $_.State -eq 'Running' })
    $jobs | Remove-Job -Force
    if ($stuck.Count -gt 0) { throw "round $Round of arm '$Arm' timed out after $RequestSeconds s" }

    $missing = @($paths | Where-Object { -not (Test-Path $_) })
    if ($missing.Count -gt 0) { throw "round $Round of arm '$Arm' produced no response for: $($missing -join ', ')" }
    return $paths
}

function Get-Response([string]$path) {
    $raw = Get-Content -LiteralPath $path -Raw
    if ($raw -match '"error"') { throw "request failed in $path : $($raw.Substring(0, [Math]::Min(200, $raw.Length)))" }
    $j = $raw | ConvertFrom-Json
    $m = $j.choices[0].message
    [pscustomobject]@{
        Text   = "$($m.content)|$($m.reasoning_content)"
        Prompt = [int]$j.usage.prompt_tokens
        Cached = [int]$j.usage.prompt_tokens_details.cached_tokens
        OutTok = [int]$j.usage.completion_tokens
    }
}

function Get-MergeStats([string]$errLog) {
    if (-not (Test-Path $errLog)) { return [pscustomobject]@{ Merged = 0; Lines = @() } }
    $hits  = @(Select-String -LiteralPath $errLog -SimpleMatch '[prefill-merged]')
    $lines = @($hits | ForEach-Object { $_.Line.Trim() })
    [pscustomobject]@{ Merged = $hits.Count; Lines = $lines }
}

$results = @{}
$stats   = @{}
foreach ($arm in 'off', 'on') {
    $proc = Start-Arm $arm
    try {
        foreach ($round in 1, 2) {
            $paths = Invoke-Round $arm $round
            for ($i = 0; $i -lt $Prompts.Count; $i++) {
                $results["$arm-r$round-$i"] = Get-Response $paths[$i]
            }
        }
    } finally {
        Stop-Arm $proc
    }
    $stats[$arm] = Get-MergeStats (Join-Path $OutDir "$arm.log")
    "arm ${arm}: $($results["$arm-r1-0"].Prompt) prompt tok | merged passes in log: $($stats[$arm].Merged)"
    $stats[$arm].Lines | Select-Object -First 6 | ForEach-Object { "    $_" }
}

$bad = 0
if ($stats['on'].Merged -eq 0 -and -not $AllowNoMerge) {
    "FAIL: the 'on' arm never printed [prefill-merged]; the comparison would be a no-op"
    "      (check PrefillChunk and prompt tail: L mod chunk must land in 1..16)"
    $bad++
}

for ($i = 0; $i -lt $Prompts.Count; $i++) {
    $o1 = $results["off-r1-$i"]
    $o2 = $results["off-r2-$i"]
    $n1 = $results["on-r1-$i"]
    $n2 = $results["on-r2-$i"]

    if ($o2.Cached -gt 0 -or $n2.Cached -gt 0) {
        $det = 'CACHED'
    } elseif ($o1.Text -ceq $o2.Text -and $n1.Text -ceq $n2.Text) {
        $det = 'yes'
    } else {
        $det = 'NO'
    }
    $same = ($o1.Text -ceq $n1.Text)

    $flag = if ($det -eq 'NO') { 'NONDETERMINISTIC' } elseif (-not $same) { 'MISMATCH' } else { 'same' }
    if ($flag -ne 'same') { $bad++ }
    "prompt $i : prompt=$($o1.Prompt) tok cached(r2)=$($o2.Cached)/$($n2.Cached) det=$det off/on_same=$same -> $flag"
    if (-not $same) {
        "    off: $($o1.Text -replace "`n", '\n')"
        "    on : $($n1.Text -replace "`n", '\n')"
    }
}

if ($bad -ne 0) {
    "FAIL: $bad issue(s); see above"
    exit 1
}
"GATE OK: $($Prompts.Count) prompt(s) deterministic and byte-identical; merged passes fired $($stats['on'].Merged)x"
exit 0

