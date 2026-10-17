# Autonomous FPL manager

Live official FPL data -> fixture-aware heuristic xPts -> PuLP squad and
starting-XI optimization -> local Ollama tactical briefing.

By default the optimizer builds a **fresh squad**. Existing-team mode supports
public squad previews and authenticated no-hit transfers/lineup management.
Writes require explicit `--apply`; no chips are automatically activated.
No production data is mocked or bundled into the algorithm.

## Run on Windows

Python 3.11+ and PowerShell 7 are recommended. The live runtime also supports
Python 3.14. Install Ollama and make the chosen reasoning model available first.
The default is `llama3.1:latest`; model downloads are explicit:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull llama3.1
python fpl_manager.py
```

Ollama must be running locally. To use a different installed tactical model:

```powershell
python fpl_manager.py --ollama-model llama3.1:latest
```

The directive is printed and saved to `reports\latest-directive.txt`.
Pre-deadline official API responses and predictions are saved to
`data\snapshots`. Model files, training data, virtual environments, and snapshots
are excluded from Git; back up snapshots, because they cannot be reconstructed
without leaking future information.

Official HTTP failures, stale deadlines, infeasible optimization, and malformed
LLM responses fail explicitly. There is no hidden offline-data fallback.
If you intentionally want to run without an LLM:

```powershell
python fpl_manager.py --no-llm
```

This mode labels the briefing as deterministic. A snapshot is preserved even if
the later local LLM call fails. Never treat an old saved directive as a fresh run.

## What is optimized

- Only `status == "a"` players with a null or at-least-75% playing chance qualify.
- Prices are displayed in millions; budget constraints use exact integer tenths.
- The squad maximizes the sum of all 15 players' xPts under the budget, positional
  quotas (2/5/5/3), and three-per-club limit. A separate ILP chooses the optimal
  legal XI **within that squad**. This is not a joint captain-adjusted XI objective.
- The captain is the highest-xPts starter, breaking ties by API baseline, form,
  then player ID. The vice is the best remaining starter with no shared fixture
  IDs, including double-gameweek matches. Outfield substitutes descend by xPts;
  the reserve goalkeeper is last.

For each opponent, normalize its venue-specific overall strength and its
positional strength by the corresponding league mean. Goalkeepers/defenders use
opponent attack; midfielders/forwards use opponent defence. The modifier is the
average of the inverse normalized ratings, clipped to `[0.5, 1.5]`. Missing or
zero attack/defence ratings fall back to positive official overall ratings, then
official general strength. If no positive rating exists, stop rather than invent
a rating.

For `n` fixtures:

```text
xPts = max(0, (0.75 * ep_next + 0.25 * form * n) * mean_fixture_modifier)
```

Missing/null `ep_next` is zero. Blank gameweeks score zero. API `ep_next` is
treated as gameweek-level, so it is not multiplied again in doubles.
These weights are explicit modeling choices, **not calibrated probabilities or
a proven predictive model**. There is no explicit minutes/rotation model beyond
the required availability cutoff; season minutes and availability flags inform
the briefing. Ownership below 10% defines the reported differentials.
The local LLM explains locked decisions; it cannot alter optimization results.
Its prose is advisory and can still contain errors.

## Weekly learning without future leakage

The optional training pipeline learns **individual points forecasts**, not
tactical prose. Keep it separate from the tactical reasoning model. Training
does not silently replace the optimizer's heuristic or demonstrate improved
accuracy just because a training job completes.

1. Before each deadline, run `fpl_manager.py` to archive the official inputs.
2. After FPL marks that gameweek finished and data checked, run preparation.
3. Validate the dataset, then explicitly train in a compatible GPU environment.
4. Export a GGUF, deploy it to Ollama, and run snapshot-based forecast inference.

```powershell
.\.venv\Scripts\python.exe 1_prepare_data.py
.\.venv\Scripts\python.exe 2_train_lora.py --validate-only
```

Preparation joins the latest valid pre-deadline snapshot for each gameweek to
official `event/{gameweek}/live` realized points. It waits for all fixtures to
finish and requires official `finished` and `data_checked` flags. Labels are
individual points before captain multipliers and automatic substitutions.
Post-deadline snapshots are rejected. Current season API results are never
joined to another season's player IDs. Pending weeks produce an explicit
nonzero exit and leave an existing dataset untouched.

**A new checkout cannot immediately produce honest training examples from a
future gameweek.** Collect pre-deadline snapshots and wait for results. Do not
fabricate historical snapshots from today's form, availability, or season totals.
Each completed week produces one 15-player supervised example; useful training
and reliable validation require many more weeks. The newest entire gameweeks
are held out chronologically, never randomly split across players.

### Explicit GPU training and export

Use a **separate Linux/WSL2 environment**, preferably Python 3.10-3.12, with a
supported NVIDIA GPU and the appropriate PyTorch/CUDA installation. The native
Windows runtime environment is not a CUDA training setup. Install the matching
PyTorch build first following the official PyTorch instructions, then install
`requirements-training.txt` with that environment's Python. Unsloth and GGUF
conversion also need compiler/CMake/Git prerequisites and sufficient RAM/disk.
Accept the chosen gated Llama model's license and authenticate locally with
Hugging Face if necessary. Never commit access tokens.

Run these Python commands **inside that Linux/WSL2 environment** at the copied
repository root (with its real accumulated dataset):

```text
python 2_train_lora.py --model meta-llama/Meta-Llama-3-8B-Instruct
python 3_export_gguf.py
```

No GPU training is started automatically. `--backend transformers` offers an
alternative to Unsloth; `--no-load-in-4bit` allows nonquantized execution but an
8B CPU model requires considerable RAM and is very slow.
Use `--resume-from-checkpoint` only for resuming the same interrupted training
configuration. For weekly updates, retrain against the accumulated dataset or
use the documented backend's adapter-loading support; prevent repeated
overfitting to a single new gameweek. Inspect chronological held-out performance
before replacing a deployed model.

Copy the actual GGUF printed by export to Windows. Deployment accepts its real
path, checks the GGUF header and Ollama exit codes, then generates a JSON smoke
response. It does not guess the exporter's filename:

```powershell
.\4_deploy_ollama.ps1 -GgufPath $YourExportedGgufPath -ModelName fpl-llama3
.\.venv\Scripts\python.exe 5_forecast.py --model fpl-llama3
```

`5_forecast.py` uses the same archived input representation as training. Add
`--labels data\training.jsonl` to compare model MAE with heuristic MAE when that
exact season/gameweek and player set has official labels. Forecast reports are
saved in `data\forecasts`. They do not silently feed back into live optimization.

`Modelfile` is a deployment template, not a directly runnable fixed model path.
Run `--help` on the Python tools and `Get-Help .\4_deploy_ollama.ps1 -Full` for
the available options. Training/export dependencies are deliberately not
installed in the live runtime environment.

## Connecting an existing FPL team

Your public Team ID is in the Points-page URL, for example
`https://fantasy.premierleague.com/entry/123456/event/7`. Public
`entry/{id}/`, `entry/{id}/history/`, `entry/{id}/transfers/`, and
`entry/{id}/event/{gameweek}/picks/` API routes provide read-only information
without a password; picks generally describe a gameweek already past its
deadline. Run a public preview for the configured team:

```powershell
.\run_weekly.ps1 -EntryId 8346787 -Preview -NoLlm
```

Public previews do not assume the last-deadline squad or bank is current and
cannot infer exact selling prices/free transfers. Unavailable players already
owned are retained with zero xPts, rather than disappearing from the squad.
Only eligible players can be transfer acquisitions.

### Enable real transfers for Team 8346787

```powershell
.\.venv\Scripts\python.exe connect_fpl.py --entry-id 8346787
.\run_weekly.ps1 -EntryId 8346787
.\install_weekly_task.ps1 -EntryId 8346787
```

For an exact authenticated preview without submitting anything:

```powershell
.\.venv\Scripts\python.exe fpl_manager.py --entry-id 8346787 --authenticated --no-llm --output reports\team-8346787.txt
```

The connection tool opens a separate local Edge browser. Sign into FPL there
and visit **My Team**. The tool captures that browser's authenticated official
My Team request, not your password, and stores the session under
`data\private`. Never share this directory, cookies, browser profile, or
authorization headers. If Edge is unavailable, install Playwright Chromium
with `python -m playwright install chromium` and connect using
`--browser-channel chromium`.

The runner uses the **authenticated current squad, selling prices, bank, and
remaining free transfers**. It defaults to at most **one transfer, zero points
hits, and no chips**; set `-MaxTransfers` explicitly to change the cap. It
submits transfers and the optimized XI/captain/vice/bench only with `--apply`
(the weekly runner supplies it unless `-Preview` is given). A free transfer
will be rolled if no squad-xPts improvement is possible. This remains the
specified total-squad objective, not a multiweek transfer-value model.

Before writing, the client verifies account ownership, deadline, fresh prices
and that the account still matches the plan. It rereads the account after
submission to confirm the saved result. Mutation requests are never retried.
Persisted per-gameweek execution receipts prevent repeated automatic transfers.
An interrupted or ambiguous submission fails closed and requires manual
reconciliation in FPL before retrying; do not delete a pending receipt without
first checking whether the transfers already happened.

The Windows task checks daily at **09:00 local time** while you are logged in;
it acts only within **36 hours of the next deadline**, once per gameweek. This
catches changing/midweek deadlines rather than assuming Saturday fixtures.
Keep this checkout, its virtual environment, Ollama, and the PC available.
Use `install_weekly_task.ps1 -NoLlm -Force` for an existing task to use
deterministic tactical facts instead of blocking account actions on local LLM
generation. Squad optimization and transfer safeguards are unchanged.
Before a due run, the runner refreshes the API token headlessly through the
saved local browser profile. This lets the official frontend renew credentials
when its browser login is still valid; it cannot bypass an expired login or MFA.
If renewal needs interaction, the job stops explicitly: rerun `connect_fpl.py`
locally to sign in again. Inspect logs in `data\logs` and receipts in
`data\executions`. Scheduling is local Windows Task Scheduler, not a hosted
always-on service.

Inspect or remove the task using Windows Task Scheduler under
`FPL Manager 8346787`. To update its settings after moving the checkout, rerun
`install_weekly_task.ps1 -Force` from the new location. Do not delete this
worktree while its task still points here.

A local private-state export can also be used with `--team-state` for a
non-submitting precise transfer preview; `--team-state` never authorizes writes.
Fresh-squad directives should never be mistaken for account transfer plans.

## Behavior checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests use isolated synthetic inputs and mocked network responses only; the
production pipeline always ingests live official API data. They cover exact
squad/lineup optima against exhaustive small cases, constraints, filtering,
fixture modifiers, captaincy hedging, output shape, snapshot timing,
outcome-label isolation, chronological splitting, and optional-dependency help.
