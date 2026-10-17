<#
.SYNOPSIS
Runs deadline-aware weekly management for an FPL entry.
.DESCRIPTION
Defaults to an authenticated, zero-hit transfer plan (at most one transfer) and
lineup submission. Authentication must first be captured with connect_fpl.py.
Use -Preview for public read-only squad recommendations.
#>
[CmdletBinding()]
param(
    [ValidateRange(1, 2147483647)]
    [int]$EntryId = 8346787,
    [ValidateRange(0, 5)]
    [int]$MaxTransfers = 1,
    [ValidateRange(1, 168)]
    [int]$WithinHours = 36,
    [string]$AuthFile = (Join-Path $PSScriptRoot 'data\private\fpl-auth.json'),
    [switch]$Preview,
    [switch]$NoLlm
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Create .venv and install requirements.txt before running weekly management.'
}
$logs = Join-Path $PSScriptRoot 'data\logs'
[void](New-Item -ItemType Directory -Path $logs -Force)
$log = Join-Path $logs ("weekly-{0}-{1}.log" -f $EntryId, (Get-Date -Format 'yyyyMMddTHHmmss'))
Start-Transcript -Path $log | Out-Null
Push-Location $PSScriptRoot
try {
    $arguments = @(
        'fpl_manager.py', '--entry-id', "$EntryId", '--max-transfers', "$MaxTransfers",
        '--output', "reports\team-$EntryId.txt"
    )
    if (-not $Preview) {
        $arguments += @('--apply', '--refresh-auth', '--auth-file', $AuthFile, '--within-hours', "$WithinHours")
    }
    if ($NoLlm) { $arguments += '--no-llm' }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "FPL management failed with exit code $LASTEXITCODE. See $log"
    }
} finally {
    Pop-Location
    Stop-Transcript | Out-Null
}
