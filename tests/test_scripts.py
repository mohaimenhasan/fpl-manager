import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which("pwsh"), "PowerShell 7 is required")
class WeeklyScriptTests(unittest.TestCase):
    def run_powershell(self, script, command):
        environment = dict(os.environ)
        environment["FPL_TEST_SCRIPT"] = str(Path(__file__).resolve().parents[1] / script)
        return subprocess.run(
            ["pwsh", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True, text=True, timeout=30, env=environment,
        )

    def test_entry_is_required_and_browser_choice_is_exposed(self):
        command = """
            $command = Get-Command -Name $env:FPL_TEST_SCRIPT
            $required = $command.Parameters['EntryId'].Attributes |
                Where-Object { $_ -is [System.Management.Automation.ParameterAttribute] }
            $browser = $command.Parameters['BrowserChannel'].Attributes |
                Where-Object { $_ -is [System.Management.Automation.ValidateSetAttribute] }
            @{
                required = [bool]$required.Mandatory
                browsers = @($browser.ValidValues)
                useLlm = $command.Parameters.ContainsKey('UseLlm')
            } | ConvertTo-Json -Compress
        """
        for script in ("run_weekly.ps1", "install_weekly_task.ps1"):
            with self.subTest(script=script):
                result = self.run_powershell(script, command)
                self.assertEqual(result.returncode, 0, result.stderr)
                metadata = json.loads(result.stdout)
                self.assertTrue(metadata["required"])
                self.assertEqual(set(metadata["browsers"]), {"msedge", "chromium"})
                self.assertTrue(metadata["useLlm"])

    def test_conflicting_reasoning_flags_fail_before_any_side_effect(self):
        command = """
            try {
                & $env:FPL_TEST_SCRIPT -EntryId 123 -NoLlm -UseLlm
                exit 2
            } catch {
                if ($_.Exception.Message -ne 'Choose either -UseLlm or -NoLlm, not both.') {
                    Write-Error $_
                    exit 1
                }
            }
            exit 0
        """
        for script in ("run_weekly.ps1", "install_weekly_task.ps1"):
            with self.subTest(script=script):
                result = self.run_powershell(script, command)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
