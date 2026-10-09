<#
.SYNOPSIS
    Delete build objects older than a commit so the next build recompiles them.

.DESCRIPTION
    Ninja only recompiles what its dependency graph says is stale. When a build aborts
    part-way (a compile error, an interrupted merge, a cancelled run) the dependency files
    for the affected targets can be left unwritten, so afterwards a plain "build succeeded"
    is not evidence: whole translation units may still be compiled against pre-merge
    headers. That is exactly how a latent signature mismatch in the sm_89 port stayed
    hidden until a merge invalidated an object file.

    Run this before trusting a build after a merge or a failed build, then build again.
    Objects are identified by mtime versus the reference commit's committer time, and
    CMake's own compiler-probe objects are never touched.

.PARAMETER BuildDir
    The CMake build directory (e.g. <repo>\build-win).

.PARAMETER Reference
    Git revision whose commit time is the cutoff (default: HEAD). Pass the merge commit
    (or its SHA) after merging so everything older than the merge is recompiled.

.PARAMETER Repo
    Repository root. Defaults to two levels above this script.

.PARAMETER Delete
    Actually delete the stale objects. Without it the script only reports, because the
    cutoff is easy to misapply: right after a merge every source file is freshly written,
    so everything looks stale and a full rebuild is the honest answer. Use -Delete for the
    targeted case (a build that aborted at a known point).

.EXAMPLE
    pwsh tools/port/reset_stale_objects.ps1 -BuildDir H:\port\ninfer\build-win -Reference cc3938c9
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$BuildDir,
    [string]$Reference = 'HEAD',
    [string]$Repo,
    [switch]$Delete
)

$ErrorActionPreference = 'Stop'

if (-not $Repo) {
    $Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
}
if (-not (Test-Path -LiteralPath $BuildDir)) { throw "build dir not found: $BuildDir" }

$iso = (& git -C $Repo log -1 --format=%cI $Reference 2>$null)
if (-not $iso) { throw "cannot resolve reference '$Reference' in $Repo" }
$cutoff = [DateTimeOffset]::Parse($iso).LocalDateTime
Write-Host ("reference {0} committed {1}" -f $Reference, $iso)

$all = Get-ChildItem -LiteralPath $BuildDir -Recurse -Filter '*.obj' -File -ErrorAction SilentlyContinue
$probe = '(?i)CompilerId|ShowIncludes'
$stale = $all | Where-Object { $_.LastWriteTime -lt $cutoff -and $_.FullName -notmatch $probe }

Write-Host ("objects: {0} total, {1} older than the reference" -f $all.Count, $stale.Count)

if ($stale.Count -eq 0) {
    Write-Host 'nothing to do: every object postdates the reference'
    exit 0
}

if (-not $Delete) {
    $stale | Select-Object -First 40 -ExpandProperty FullName | ForEach-Object { $_.Replace($BuildDir, '<build>') }
    Write-Host ("... {0} of {1} listed; re-run with -Delete to remove them" -f [Math]::Min(40, $stale.Count), $stale.Count)
    exit 0
}

$stale | Remove-Item -Force
Write-Host ("deleted {0} stale object(s); now rebuild before trusting the result" -f $stale.Count)
