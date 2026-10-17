<#
.SYNOPSIS
Creates a local Ollama model from an explicitly selected exported GGUF and smoke-tests generation.
.DESCRIPTION
Requires PowerShell 7+ and a running local Ollama server. Run export under Linux/WSL2,
then pass a GGUF path accessible to native Windows Ollama. The Modelfile is a
template; a uniquely named resolved copy is created beside it and always removed.
.EXAMPLE
.\4_deploy_ollama.ps1 -GgufPath ".\models\gguf\chosen-model.Q4_K_M.gguf" -ModelName "fpl-llama3"
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$GgufPath,

    [ValidatePattern('^[a-zA-Z0-9][a-zA-Z0-9._:/-]*$')]
    [string]$ModelName = 'fpl-llama3',

    [string]$ModelfilePath = (Join-Path $PSScriptRoot 'Modelfile'),

    [ValidateRange(1, 3600)]
    [int]$SmokeTimeoutSeconds = 180
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$temporaryModelfile = $null
$smokeProcess = $null

try {
    $ollama = Get-Command ollama -CommandType Application -ErrorAction Stop | Select-Object -First 1
    $gguf = (Resolve-Path -LiteralPath $GgufPath -ErrorAction Stop).ProviderPath
    $templatePath = (Resolve-Path -LiteralPath $ModelfilePath -ErrorAction Stop).ProviderPath
    if (-not (Test-Path -LiteralPath $gguf -PathType Leaf) -or
        [System.IO.Path]::GetExtension($gguf) -ine '.gguf') {
        throw 'GgufPath must point to an existing .gguf file.'
    }
    $stream = [System.IO.File]::OpenRead($gguf)
    try {
        $magic = [byte[]]::new(4)
        if ($stream.Read($magic, 0, 4) -ne 4 -or [System.Text.Encoding]::ASCII.GetString($magic) -ne 'GGUF') {
            throw 'The selected file does not have a GGUF header.'
        }
    } finally {
        $stream.Dispose()
    }
    if ($gguf.Contains('"') -or $gguf.Contains("`r") -or $gguf.Contains("`n")) {
        throw 'The GGUF path cannot contain quotes or newlines.'
    }
    $template = [System.IO.File]::ReadAllText($templatePath)
    if ([regex]::Matches($template, '\{\{GGUF_PATH\}\}').Count -ne 1) {
        throw 'Modelfile template must contain exactly one {{GGUF_PATH}} placeholder.'
    }
    # Use an absolute, quoted path so spaces and exporter-specific filenames are safe.
    $resolvedTemplate = $template.Replace('{{GGUF_PATH}}', '"' + $gguf.Replace('\', '/') + '"')
    $workDirectory = Split-Path -Parent $templatePath
    $token = [guid]::NewGuid().ToString('N')
    $temporaryModelfile = Join-Path $workDirectory ".Modelfile.deploy-$token"
    [System.IO.File]::WriteAllText(
        $temporaryModelfile, $resolvedTemplate, [System.Text.UTF8Encoding]::new($false)
    )
    & $ollama.Source create $ModelName -f $temporaryModelfile
    if ($LASTEXITCODE -ne 0) {
        throw "ollama create failed with native exit code $LASTEXITCODE."
    }

    # Drain both pipes concurrently, enforce a deadline, and inspect the native exit code.
    $info = [System.Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $ollama.Source
    $info.UseShellExecute = $false
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    foreach ($argument in @('run', $ModelName, '--format', 'json',
        'Return only JSON: {"players":[{"player_id":1,"points":2.0}]}')) {
        [void]$info.ArgumentList.Add($argument)
    }
    $smokeProcess = [System.Diagnostics.Process]::new()
    $smokeProcess.StartInfo = $info
    if (-not $smokeProcess.Start()) { throw 'Could not start ollama smoke generation.' }
    $outputTask = $smokeProcess.StandardOutput.ReadToEndAsync()
    $errorTask = $smokeProcess.StandardError.ReadToEndAsync()
    if (-not $smokeProcess.WaitForExit($SmokeTimeoutSeconds * 1000)) {
        $smokeProcess.Kill($true)
        $smokeProcess.WaitForExit()
        throw "ollama smoke generation timed out after $SmokeTimeoutSeconds seconds."
    }
    $stdout = $outputTask.GetAwaiter().GetResult()
    $stderr = $errorTask.GetAwaiter().GetResult()
    if ($smokeProcess.ExitCode -ne 0) {
        throw "ollama run failed with native exit code $($smokeProcess.ExitCode): $stderr"
    }
    if ([string]::IsNullOrWhiteSpace($stdout)) { throw 'ollama run returned empty output.' }
    $answer = $stdout | ConvertFrom-Json -ErrorAction Stop
    if (-not $answer.PSObject.Properties['players'] -or @($answer.players).Count -eq 0) {
        throw 'Smoke generation did not return a nonempty JSON players list.'
    }
    Write-Host "Created and smoke-tested Ollama model '$ModelName'."
    Write-Output $stdout
} catch {
    Write-Error "Ollama deployment failed: $($_.Exception.Message)" -ErrorAction Continue
    exit 1
} finally {
    if ($null -ne $smokeProcess) { $smokeProcess.Dispose() }
    foreach ($path in @($temporaryModelfile)) {
        if ($path -and (Test-Path -LiteralPath $path)) {
            Remove-Item -LiteralPath $path -Force
        }
    }
}
