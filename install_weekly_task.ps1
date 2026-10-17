<#
.SYNOPSIS
Registers a local daily deadline check that manages each FPL gameweek once.
.DESCRIPTION
The task runs at 09:00 local time while you are logged in. The runner only acts
within 36 hours of the next FPL deadline and uses per-gameweek receipts to avoid
duplicate submissions. Keep this checkout and its .venv at their current paths.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 2147483647)]
    [int]$EntryId,
    [ValidateRange(0, 5)]
    [int]$MaxTransfers = 1,
    [ValidateSet('msedge', 'chromium')]
    [string]$BrowserChannel = 'msedge',
    [switch]$NoLlm,
    [switch]$UseLlm,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
if ($NoLlm -and $UseLlm) {
    throw 'Choose either -UseLlm or -NoLlm, not both.'
}
$name = "FPL Manager $EntryId"
$existing = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
if ($existing -and -not $Force) {
    throw "Task '$name' already exists. Inspect it first, then use -Force to replace it."
}
$pwsh = (Get-Command pwsh -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
$runner = Join-Path $PSScriptRoot 'run_weekly.ps1'
$arguments = "-NoProfile -NonInteractive -File `"$runner`" -EntryId $EntryId -MaxTransfers $MaxTransfers -BrowserChannel $BrowserChannel"
if ($UseLlm) { $arguments += ' -UseLlm' } else { $arguments += ' -NoLlm' }
$action = New-ScheduledTaskAction -Execute $pwsh -Argument $arguments -WorkingDirectory $PSScriptRoot
$trigger = New-ScheduledTaskTrigger -Daily -At '09:00'
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $identity -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1)
$parameters = @{
    TaskName = $name
    Action = $action
    Trigger = $trigger
    Principal = $principal
    Settings = $settings
    Description = 'Authenticated FPL management; no hits, at most configured transfers, once per gameweek within 36 hours of deadline.'
}
if ($Force) { $parameters.Force = $true }
Register-ScheduledTask @parameters | Select-Object TaskName, State
