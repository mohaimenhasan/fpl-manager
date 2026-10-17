<#
.SYNOPSIS
Runs deadline-aware weekly management for an FPL entry.
.DESCRIPTION
Defaults to an authenticated, zero-hit transfer plan (at most one transfer),
deterministic tactical facts, and lineup submission. Authentication must first
be captured with connect_fpl.py. Use -UseLlm to opt into local LLM reasoning.
Use -Preview for public read-only squad recommendations.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 2147483647)]
    [int]$EntryId,
    [ValidateRange(0, 5)]
    [int]$MaxTransfers = 1,
    [ValidateRange(1, 168)]
    [int]$WithinHours = 36,
    [string]$AuthFile = (Join-Path $PSScriptRoot 'data\private\fpl-auth.json'),
    [ValidateSet('msedge', 'chromium')]
    [string]$BrowserChannel = 'msedge',
    [switch]$Preview,
    [switch]$NoLlm,
    [switch]$UseLlm
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
if ($NoLlm -and $UseLlm) {
    throw 'Choose either -UseLlm or -NoLlm, not both.'
}
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
        $arguments += @(
            '--apply', '--refresh-auth', '--auth-file', $AuthFile,
            '--auth-browser-channel', $BrowserChannel, '--within-hours', "$WithinHours"
        )
    }
    if (-not $UseLlm) { $arguments += '--no-llm' }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "FPL management failed with exit code $LASTEXITCODE. See $log"
    }
} finally {
    Pop-Location
    Stop-Transcript | Out-Null
}
