<#
.SYNOPSIS
    Move the local branch's upstream ancestry forward after a triage pass.

.DESCRIPTION
    The local tree is a Windows/sm_89 port, so taking upstream's tree is wrong; what must stay in
    step is the *review* baseline. This script records that baseline as real ancestry:

      1. fetch the upstream remote;
      2. run the disposition report, which refuses to pass while any upstream commit after the
         current baseline has no row in docs/maintainer/upstream-dispositions.tsv;
      3. refuse while the register still has `todo` rows (override with -AllowTodo, and say why);
      4. merge upstream with `-s ours --no-ff`: the tree does not change, but upstream becomes an
         ancestor, so `rev-list --count master..upstream/master` returns 0 and the next fetch shows
         only genuinely new commits;
      5. tag the merge as `upstream-triaged/<sha>` so the round is reproducible.

    After this, the only work a new upstream release creates is the new commits themselves - not a
    re-verification of the hundreds already triaged.

.PARAMETER Remote
    Upstream remote name (default: upstream).

.PARAMETER Branch
    Upstream branch (default: master).

.PARAMETER Trunk
    Local branch that receives the ancestry (default: current branch).

.PARAMETER AllowTodo
    Proceed although the register still has todo rows. The merge message records the count, so the
    ancestry never silently claims content absorption.

.PARAMETER DryRun
    Print what would happen and stop before the merge.
#>
[CmdletBinding()]
param(
    [string]$Remote  = 'upstream',
    [string]$Branch  = 'master',
    [string]$Trunk   = '',
    [switch]$AllowTodo,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$register = Join-Path $repoRoot 'docs\maintainer\upstream-dispositions.tsv'
$report   = Join-Path $repoRoot 'tools\upstream\alignment_report.ps1'

function Invoke-Git {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$GitArgs)
    & git -C $repoRoot @GitArgs
    if ($LASTEXITCODE -ne 0) { throw ("git " + ($GitArgs -join ' ') + " failed ($LASTEXITCODE)") }
}

function Read-RegisterCounts {
    $rows = Import-Csv -LiteralPath $register -Delimiter "`t"
    $groups = $rows | Group-Object class
    $counts = [ordered]@{}
    foreach ($g in ($groups | Sort-Object Name)) { $counts[$g.Name] = $g.Count }
    return [pscustomobject]@{ Total = $rows.Count; Counts = $counts }
}

if (-not (Test-Path -LiteralPath $register)) { throw "register not found: $register" }
if (-not (Test-Path -LiteralPath $report))   { throw "alignment report not found: $report" }

if (-not $Trunk) {
    $Trunk = (& git -C $repoRoot rev-parse --abbrev-ref HEAD).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'cannot read the current branch' }
}

$dirty = & git -C $repoRoot status --porcelain=v1 | Where-Object { $_ -notmatch '^\?\?' }
if ($dirty) {
    throw ("working tree is not clean; commit or stash first:`n" + ($dirty -join "`n"))
}

Write-Host "repo    : $repoRoot"
Write-Host "trunk   : $Trunk"
Write-Host "upstream: $Remote/$Branch"

Invoke-Git fetch $Remote --prune | Out-Null
$tip    = (Invoke-Git rev-parse "$Remote/$Branch").Trim()
$behind = [int]((Invoke-Git rev-list --count "$Trunk..$Remote/$Branch").Trim())
Write-Host ("upstream tip {0}; {1} commit(s) not yet in {2}" -f $tip.Substring(0, 8), $behind, $Trunk)

if ($behind -eq 0) {
    Write-Host "already at the upstream tip; nothing to record."
    return
}

# The report is the gate: exit 1 means at least one upstream commit after the baseline has no row.
# Run it through a child PowerShell so an execution policy on this host cannot turn the gate into a
# silent pass.
& powershell -NoProfile -ExecutionPolicy Bypass -File $report
if ($LASTEXITCODE -ne 0) {
    throw "alignment report failed ($LASTEXITCODE): add the missing rows above, then re-run"
}

$snapshot = Read-RegisterCounts
$todo = 0
if ($snapshot.Counts.Contains('todo')) { $todo = [int]$snapshot.Counts['todo'] }
if ($todo -gt 0 -and -not $AllowTodo) {
    throw ("register still has $todo todo row(s); close them or pass -AllowTodo (the merge message will record the count)")
}

$countsText = ($snapshot.Counts.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join ', '
$message = @(
    "chore(upstream): triage upstream through $($tip.Substring(0,8)) and record it as ancestry",
    "",
    "The tree is unchanged (-s ours): this commit records that every upstream commit up to",
    "$tip has a disposition row in docs/maintainer/upstream-dispositions.tsv.",
    "",
    "register: $($snapshot.Total) rows ($countsText).",
    "todo rows remaining: $todo$(if ($todo -gt 0) { ' (content, not ancestry: see the register)' }).",
    "",
    "Next fetch shows only new upstream commits, not the backlog."
) -join "`n"

Write-Host ""
Write-Host "register: $($snapshot.Total) rows ($countsText)"
Write-Host "todo    : $todo"
if ($DryRun) {
    Write-Host "dry run: would merge -s ours $Remote/$Branch into $Trunk and tag upstream-triaged/$($tip.Substring(0,8))"
    return
}

Invoke-Git merge -s ours --no-ff -m $message "$Remote/$Branch" | Out-Null

$tagName = "upstream-triaged/$($tip.Substring(0, 8))"
$existing = & git -C $repoRoot tag --list $tagName
if (-not $existing) {
    Invoke-Git tag -a $tagName -m $message HEAD | Out-Null
    Write-Host "tagged  : $tagName"
} else {
    Write-Host "tag     : $tagName already exists"
}

$after = [int]((Invoke-Git rev-list --count "$Trunk..$Remote/$Branch").Trim())
$mergeBase = (Invoke-Git merge-base $Trunk "$Remote/$Branch").Trim()
Write-Host ""
Write-Host ("done: behind={0}  merge-base={1}" -f $after, $mergeBase.Substring(0, 8))
if ($after -ne 0) { throw "ancestry did not reach the upstream tip (behind=$after)" }
