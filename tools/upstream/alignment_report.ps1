<#
    alignment_report.ps1 - incremental upstream alignment report.

    Reads docs/maintainer/upstream-dispositions.tsv (the per-commit disposition
    register), finds the newest upstream/master commit it contains (the pinned
    baseline), and lists every upstream commit after that baseline with its
    subject and whether it already has a disposition row.

    Exit codes:
      0  no upstream commit after the baseline lacks a disposition row
      1  at least one upstream commit after the baseline has no row
      2  the upstream ref / register / repository could not be used

    Requires git on PATH. PowerShell 5.1 compatible, no external modules.

    Register format
      docs/maintainer/upstream-dispositions.tsv, UTF-8 (no BOM), one row per
      upstream commit, header first:
          commit <TAB> date <TAB> subject <TAB> class <TAB> evidence <TAB> note

      class    absorbed     content already local (merge/port/parity record)
               na-arch      not applicable on sm_89 (SM100-only kernel paths)
               na-hardware  needs hardware this port does not have (sm_120a)
               dead         judged dead, no absorption planned
               todo         pending absorption, gap identified
               needs-review rule table could not place it, evidence is empty

      Rows are generated mechanically from the upstream history:
          git log upstream/master --format="%h%x09%ad%x09%s" --date=short
      and the rule table (prefix/path -> class -> evidence) documented in
      docs/maintainer/upstream-alignment.md section 7.2.
#>

[CmdletBinding()]
param(
    [string]$RegisterPath,
    [string]$Remote = 'upstream',
    [string]$Branch = 'master'
)

$ErrorActionPreference = 'Stop'

$scriptDir = $PSScriptRoot
if (-not $scriptDir) { $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path }
$repoRoot = Split-Path -Parent (Split-Path -Parent $scriptDir)

if (-not $RegisterPath) {
    $RegisterPath = Join-Path $repoRoot 'docs\maintainer\upstream-dispositions.tsv'
}

function Fail([string]$message) {
    Write-Host "alignment_report: $message" -ForegroundColor Red
    exit 2
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Fail 'git was not found on PATH.' }
if (-not (Test-Path -LiteralPath $repoRoot)) { Fail "repository root not found: $repoRoot" }

$toplevel = & git -C $repoRoot rev-parse --show-toplevel 2>$null
if ($LASTEXITCODE -ne 0 -or -not $toplevel) { Fail "not a git work tree: $repoRoot" }

$remoteUrl = & git -C $repoRoot remote 2>$null
if ($LASTEXITCODE -ne 0 -or ($remoteUrl -notcontains $Remote)) {
    Fail "the '$Remote' remote is not configured in $repoRoot (git remote add $Remote <url>)."
}

$upstreamRef = "refs/remotes/$Remote/$Branch"
$tipSha = & git -C $repoRoot rev-parse --verify --quiet "$upstreamRef^{commit}" 2>$null
if ($LASTEXITCODE -ne 0 -or -not $tipSha) {
    Fail "the ref '$Remote/$Branch' does not exist locally; run: git -C `"$repoRoot`" fetch $Remote"
}
$tipSha = $tipSha.Trim()

if (-not (Test-Path -LiteralPath $RegisterPath)) { Fail "disposition register not found: $RegisterPath" }

function Get-ShortSubject([string]$sha) {
    $line = (& git -C $repoRoot log -1 --format='%h%x09%ad%x09%s' --date=short $sha) -join ''
    return $line
}

# ---- register ----
$registerHashes = @{}
$rowCount = 0
$classCount = @{}
$lineNo = 0
foreach ($line in (Get-Content -LiteralPath $RegisterPath -Encoding UTF8)) {
    $lineNo++
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    $f = $line -split "`t"
    if ($lineNo -eq 1 -and $f[0] -eq 'commit') { continue }
    if ($f.Count -lt 6) {
        Write-Host "alignment_report: warning: line $lineNo has $($f.Count) fields, expected 6; skipped." -ForegroundColor Yellow
        continue
    }
    $short = $f[0].Trim().ToLowerInvariant()
    if ($short.Length -gt 8) { $short = $short.Substring(0, 8) }
    $registerHashes[$short] = $true
    $rowCount++
    $class = $f[3].Trim()
    if (-not $classCount.ContainsKey($class)) { $classCount[$class] = 0 }
    $classCount[$class]++
}

if ($rowCount -eq 0) { Fail "the register contains no disposition rows: $RegisterPath" }

# ---- pinned baseline: newest upstream commit that has a row ----
$history = @(& git -C $repoRoot rev-list "$Remote/$Branch")
if ($LASTEXITCODE -ne 0 -or $history.Count -eq 0) { Fail "could not enumerate $Remote/$Branch." }

$baselineSha = $null
foreach ($sha in $history) {
    $short = $sha.Substring(0, 8)
    if ($registerHashes.ContainsKey($short)) { $baselineSha = $sha.Trim(); break }
}
if (-not $baselineSha) {
    Fail "no commit of $Remote/$Branch appears in the register; the register belongs to another history."
}

$baseCount = (& git -C $repoRoot rev-list --count $baselineSha).Trim()
$totalCount = (& git -C $repoRoot rev-list --count "$Remote/$Branch").Trim()
$afterCount = (& git -C $repoRoot rev-list --count "$baselineSha..$Remote/$Branch").Trim()

# ---- report ----
Write-Host ''
Write-Host '== upstream alignment report ==' -ForegroundColor Cyan
Write-Host "register : $RegisterPath"
Write-Host "           $rowCount row(s), $($classCount.Keys.Count) class(es)"
$tipParts = (Get-ShortSubject $tipSha) -split "`t"
$baseParts = (Get-ShortSubject $baselineSha) -split "`t"
Write-Host ("upstream : {0}/{1} @ {2}  {3}  {4}" -f $Remote, $Branch, $tipParts[0], $tipParts[1], $tipParts[2])
Write-Host ("baseline : {0}  {1}  {2}" -f $baseParts[0], $baseParts[1], $baseParts[2])
Write-Host "           pinned at upstream commit $baseCount of $totalCount; $afterCount commit(s) after it"
Write-Host ''

$missing = @()
if ([int]$afterCount -eq 0) {
    Write-Host 'nothing to triage: the register already covers the upstream tip.' -ForegroundColor Green
} else {
    Write-Host "commits after the baseline ($afterCount):" -ForegroundColor Cyan
    $newShas = @(& git -C $repoRoot rev-list --reverse "$baselineSha..$Remote/$Branch")
    foreach ($sha in $newShas) {
        $short = $sha.Substring(0, 8)
        $subject = (& git -C $repoRoot log -1 --format='%h%x09%ad%x09%s' --date=short $short) -join ''
        $parts = $subject -split "`t"
        if ($registerHashes.ContainsKey($short)) {
            $mark = 'has row'
        } else {
            $mark = 'NO ROW '
            $missing += $short
        }
        Write-Host ("  {0}  {1}  {2}  {3}" -f $parts[0], $parts[1], $mark, $parts[2])
    }
    Write-Host ''
}

Write-Host 'classes over the whole register:' -ForegroundColor Cyan
foreach ($key in ($classCount.Keys | Sort-Object { - $classCount[$_] })) {
    Write-Host ("  {0,-14} {1}" -f $key, $classCount[$key])
}
Write-Host ''

if ($missing.Count -gt 0) {
    Write-Host "needs triage: $($missing.Count) upstream commit(s) have no disposition row:" -ForegroundColor Yellow
    Write-Host ("  " + ($missing -join ' '))
    exit 1
}

Write-Host 'ok: every upstream commit up to the tip has a disposition row.' -ForegroundColor Green
exit 0
