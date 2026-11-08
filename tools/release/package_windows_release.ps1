# Assemble the Windows release folder from a completed Release build.
#
#   powershell -File tools\release\package_windows_release.ps1 -Root R:\infer
#
# Everything the folder must contain is decided here; docs\release-layout.md explains why each
# item is in the list (and what is deliberately left out). The script refuses to package a build
# that is older than the newest source file, because a stale executable is worse than no release.
param(
    [Parameter(Mandatory = $true)][string]$Root,
    [string]$Repo = '',
    [string]$Build = ''
)

$ErrorActionPreference = 'Stop'
if (-not $Repo) {
    # $PSScriptRoot is not always populated when the script is started with -File from another
    # shell; the command path always is.
    $here = Split-Path -Parent $MyInvocation.MyCommand.Path
    $Repo = (Resolve-Path (Join-Path $here '..\..')).Path
}
if (-not $Build) { $Build = Join-Path $Repo 'build-win' }
$apps = Join-Path $Build 'apps'
$exes = @('ninfer.exe', 'ninfer-serve.exe', 'ninfer-perplexity.exe', 'diag-moe.exe', 'diag-cpu-partial.exe')
$bench = Join-Path $Build 'bench\ninfer_bench.exe'

foreach ($needed in (@($exes | ForEach-Object { Join-Path $apps $_ }) + $bench)) {
    if (-not (Test-Path $needed)) { throw "missing build product: $needed (run the Release build first)" }
}

# Staleness: the newest source change may not be newer than ninfer.exe.
$newestSource = Get-ChildItem (Join-Path $Repo 'src'), (Join-Path $Repo 'include') -Recurse -File |
    Where-Object { $_.Extension -in '.h', '.hpp', '.cpp', '.cu', '.cuh' } |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
$binary = Get-Item (Join-Path $apps 'ninfer.exe')
if ($newestSource -and $newestSource.LastWriteTime -gt $binary.LastWriteTime) {
    throw "ninfer.exe is older than $($newestSource.FullName); rebuild before packaging"
}

foreach ($dir in @($Root, "$Root\docs", "$Root\templates", "$Root\symbols", "$Root\scripts")) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
}

foreach ($exe in $exes) { Copy-Item (Join-Path $apps $exe) $Root -Force }
Copy-Item $bench $Root -Force
Get-ChildItem $apps -Filter '*.dll' | ForEach-Object { Copy-Item $_.FullName $Root -Force }
Get-ChildItem $apps -Filter '*.pdb' | ForEach-Object { Copy-Item $_.FullName "$Root\symbols" -Force }

Copy-Item (Join-Path $Repo 'LICENSE') $Root -Force
Copy-Item (Join-Path $Repo 'README.md') $Root -Force
Copy-Item (Join-Path $Repo 'ninfer.example.toml') $Root -Force
Copy-Item (Join-Path $Repo 'docs\*.md') "$Root\docs" -Force
Copy-Item (Join-Path $Repo 'docs\maintainer') "$Root\docs" -Recurse -Force
Copy-Item (Join-Path $Repo 'tools\chat_templates\*.jinja') "$Root\templates" -Force
Copy-Item (Join-Path $Repo 'tools\chat_templates\LICENSE') "$Root\templates" -Force

$tools = Join-Path $Root 'tools'
New-Item -ItemType Directory -Path $tools -Force | Out-Null
foreach ($item in @('convert', 'artifact', 'freq_corpus', 'chat_templates', 'console', 'README.md',
                    'upgrade_ninfer_v2_to_v3.py', 'generate_jinja_unicode.py')) {
    $src = Join-Path $Repo "tools\$item"
    if (-not (Test-Path $src)) { continue }
    if (Test-Path $src -PathType Container) { Copy-Item $src $tools -Recurse -Force }
    else { Copy-Item $src $tools -Force }
}

$consoleBat = @'
@echo off
rem Local console: serves the page and proxies the engine/LMCache/GPU sources.
rem Open http://127.0.0.1:8099/ after it starts. Optional argument = engine base URL.
title ninfer console
if not "%~1"=="" (
  python "{ROOT}\tools\console\lmcache_console.py" --port 8099 --engine %~1
) else (
  python "{ROOT}\tools\console\lmcache_console.py" --port 8099 --engine http://127.0.0.1:11438
)
pause
'@
Set-Content (Join-Path $Root 'scripts\console.bat') ($consoleBat.Replace('{ROOT}', $Root)) -Encoding ASCII

# ---- dashboard: built dist + config + node_modules (offline run on this machine) ------------
$dash = Join-Path $Root 'dashboard'
New-Item -ItemType Directory -Path $dash -Force | Out-Null
foreach ($item in @('dist', 'node_modules')) {
    $src = Join-Path $Repo "dashboard\$item"
    if (Test-Path $src) { Copy-Item $src $dash -Recurse -Force }
}
foreach ($item in @('package.json', 'config.example.json', 'README.md')) {
    Copy-Item (Join-Path $Repo "dashboard\$item") $dash -Force
}
@'
{
  "engine_url": "http://127.0.0.1:11438",
  "host": "127.0.0.1",
  "port": 8899,
  "database": "./data/monitor.sqlite",
  "sampling_interval_ms": 1000,
  "fine_history_hours": 24,
  "history_days": 30,
  "default_theme": "paper"
}
'@ | Set-Content (Join-Path $dash 'config.local.json') -Encoding UTF8
New-Item -ItemType Directory -Path (Join-Path $dash 'data') -Force | Out-Null

@'
@echo off
rem Process-level dashboard (read-only, keeps its own SQLite history). Node 22+ required.
title ninfer dashboard
cd /d %~dp0..\dashboard
node dist\server\main.js --config config.local.json
pause
'@ | Set-Content (Join-Path $Root 'scripts\dashboard.bat') -Encoding ASCII

# ---- serve profiles -------------------------------------------------------------------------
function Write-Launcher([string]$name, [string]$title, [string]$model, [string]$flags) {
    $body = @"
@echo off
rem $title
title $title
cd /d %~dp0
if not defined NINFER_MODEL set "NINFER_MODEL=$model"
..\ninfer-serve.exe %NINFER_MODEL% $flags
pause
"@
    Set-Content (Join-Path $Root "scripts\$name") $body -Encoding ASCII
}
Write-Launcher 'serve_35b_e2_2k.bat' 'ninfer-serve 35B e2 2K (gate profile)' `
    (Join-Path $Root '..\model\qwen3_6_35b_a3b_e2\qwen3_6_35b_a3b_e2.ninfer') `
    '--host 127.0.0.1 --port 11438 --max-context 2048 --max-concurrency 1 --max-pending-requests 16 --prefill-chunk 1024'
Write-Launcher 'serve_27b_v3_32k.bat' 'ninfer-serve 27B v3 32K (rk4v4-e8 MTP-5)' `
    (Join-Path $Root '..\model\qwen3_8_27b_v3\qwen3_8_27b_v3.ninfer') `
    '--host 0.0.0.0 --port 21438 --max-context 32768 --kv-capacity 32768 --max-concurrency 1 --max-pending-requests 16 --prefill-chunk 1024 --kv-dtype rk4v4-e8 --spec mtp --draft-tokens 5 --lm-head-draft --wddm-evictable-budget'
Write-Launcher 'serve_27b_v3_128k_c8.bat' 'ninfer-serve 27B v3 128K C=8 rk4v4-e8 (9908)' `
    (Join-Path $Root '..\model\qwen3_8_27b_v3\qwen3_8_27b_v3.ninfer') `
    '--host 0.0.0.0 --port 9908 --max-context 131072 --kv-capacity auto --max-concurrency 8 --max-pending-requests 16 --prefill-chunk 1024 --kv-dtype rk4v4-e8 --spec dflash2 --draft-tokens 7 --lm-head-draft --kv-evict auto --kv-idle-reclaim-ms 60000 --wddm-evictable-budget'


$describe = & git -C $Repo describe --tags --always
$commit = & git -C $Repo rev-parse HEAD
@"
NInfer release for RTX 4090 (sm_89), Windows 11 x64
version:  $describe
commit:   $commit
built:    $((Get-Item (Join-Path $apps 'ninfer.exe')).LastWriteTime.ToString('yyyy-MM-dd HH:mm'))
toolchain CUDA 13.4 / MSVC / CMake Release / sm_89
"@ | Set-Content (Join-Path $Root 'VERSION') -Encoding UTF8

Get-ChildItem $Root -Recurse -File |
    Where-Object { $_.FullName -notlike '*\dashboard\node_modules\*' -and $_.Name -ne 'SHA256SUMS.txt' } |
    Sort-Object FullName | ForEach-Object {
        $rel = $_.FullName.Substring($Root.Length + 1)
        "$((Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower())  $rel"
    } | Set-Content (Join-Path $Root 'SHA256SUMS.txt') -Encoding ASCII

Write-Output "packaged $Root ($((Get-ChildItem $Root -Recurse -File).Count) files)"
