#!/usr/bin/env python3
"""Run the trained FPL points-forecast prompt against local Ollama.

Uses only archived pre-deadline features, never labels, in the model prompt.
Optional realized labels score the result after inference. This predicts raw
individual player points, not tactical reasoning, and does NOT feed predictions
back into the live optimizer. Retrospective or training-set scores are not proof
of out-of-sample performance. No GPU training libraries are imported.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from uuid import uuid4

from training.data import load_examples
from training.prompts import FORECAST_INSTRUCTION, SYSTEM_PROMPT, forecast_input, user_prompt
from training.snapshots import finite_number, integer, utc_timestamp, validate_snapshot


def latest_snapshot(directory, explicit=None):
    """Choose the latest valid pre-deadline snapshot, not a filename sort."""
    if explicit is not None:
        return explicit, validate_snapshot(json.loads(explicit.read_text(encoding="utf-8")))
    candidates = []
    for path in sorted(directory.glob("*.json")):
        try:
            snapshot = validate_snapshot(json.loads(path.read_text(encoding="utf-8")))
            candidates.append((utc_timestamp(snapshot["fetched_at"]), str(path), path, snapshot))
        except (ValueError, KeyError, TypeError) as exc:
            print(f"Skipping invalid snapshot {path.name}: {exc}", file=sys.stderr)
    if not candidates:
        raise ValueError(f"no valid pre-deadline snapshots in {directory}")
    _, _, path, snapshot = max(candidates, key=lambda row: (row[0], row[1]))
    return path, snapshot


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON property {key!r}")
        result[key] = value
    return result


def validate_forecast(text, requested_ids):
    """Reject extra/missing IDs, duplicate IDs, nonfinite points and loose JSON."""
    if (
        len(requested_ids) != 15
        or not all(integer(pid) and pid > 0 for pid in requested_ids)
        or len(set(requested_ids)) != 15
    ):
        raise ValueError("forecast requests must contain exactly 15 distinct player IDs")
    if not isinstance(text, str):
        raise ValueError("Ollama response must contain JSON text")

    def reject_constant(value):
        raise ValueError(f"nonfinite JSON number {value}")

    result = json.loads(text, object_pairs_hook=_unique_object, parse_constant=reject_constant)
    if not isinstance(result, dict) or set(result) != {"players"} or not isinstance(result["players"], list):
        raise ValueError("forecast must be exactly a JSON object with a players list")
    if len(result["players"]) != 15:
        raise ValueError("forecast must contain exactly 15 players")
    predictions = {}
    for row in result["players"]:
        if not isinstance(row, dict) or set(row) != {"player_id", "points"}:
            raise ValueError("each forecast player must contain only player_id and points")
        pid = row["player_id"]
        if not integer(pid) or pid not in requested_ids or pid in predictions:
            raise ValueError("forecast has a duplicate, unexpected or invalid player_id")
        if not finite_number(row["points"]):
            raise ValueError("forecast points must be finite numeric values, not strings or booleans")
        predictions[pid] = row["points"]
    if set(predictions) != set(requested_ids):
        raise ValueError("forecast IDs must exactly match the requested squad")
    return {"players": [{"player_id": pid, "points": predictions[pid]} for pid in requested_ids]}


def request_forecast(session, snapshot, model, base_url, timeout):
    ids = snapshot["squad_ids"]
    schema = {
        "type": "object",
        "properties": {
            "players": {
                "type": "array", "minItems": 15, "maxItems": 15,
                "items": {
                    "type": "object",
                    "properties": {
                        "player_id": {"type": "integer", "enum": ids},
                        "points": {"type": "number"},
                    },
                    "required": ["player_id", "points"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["players"],
        "additionalProperties": False,
    }
    response = session.post(
        base_url.rstrip("/") + "/api/generate",
        json={
            "model": model, "stream": False, "system": SYSTEM_PROMPT,
            "prompt": user_prompt(FORECAST_INSTRUCTION, forecast_input(snapshot)),
            "format": schema,
            "options": {"temperature": 0.1, "num_ctx": 8192, "num_predict": 2048},
        },
        timeout=(min(10.0, timeout), timeout),
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise ValueError(f"Ollama generation failed: {payload['error']}")
    if payload.get("done") is not True or payload.get("done_reason") == "length":
        raise ValueError("Ollama generation was incomplete or reached its token limit")
    return validate_forecast(payload["response"], ids)


def evaluate(snapshot, forecast, examples):
    """Evaluate only the matching season/GW and the exact requested player set."""
    matching = [
        example for example in examples
        if example["input"]["season"] == snapshot["season"]
        and example["input"]["gameweek"] == snapshot["gameweek"]
    ]
    if not matching:
        return {"status": "unavailable", "reason": "No realized labels for this exact season/gameweek"}
    if len(matching) != 1:
        raise ValueError("evaluation requires one unique season/gameweek label record")
    targets = {row["player_id"]: row["points"] for row in matching[0]["output"]["players"]}
    ids = snapshot["squad_ids"]
    if set(targets) != set(ids):
        return {"status": "unavailable", "reason": "Label player IDs do not match the requested 15-player squad"}
    predictions = {row["player_id"]: row["points"] for row in forecast["players"]}
    baseline = {row["player_id"]: row["xpts"] for row in snapshot["predictions"]}
    model_mae = sum(abs(predictions[pid] - targets[pid]) for pid in ids) / len(ids)
    baseline_mae = sum(abs(baseline[pid] - targets[pid]) for pid in ids) / len(ids)
    if not all(math.isfinite(value) for value in (model_mae, baseline_mae)):
        raise ValueError("evaluation overflowed; points must be in a meaningful finite range")
    return {
        "status": "scored", "season": snapshot["season"], "gameweek": snapshot["gameweek"],
        "n_players": len(ids), "model_mae": model_mae, "baseline_mae": baseline_mae,
        "mae_improvement": baseline_mae - model_mae,
        "scope": "raw individual points, without captain multipliers or substitutions",
        "caution": "Only held-out forecasts generated before the deadline measure prospective performance",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshots", type=Path, default=Path("data") / "snapshots")
    parser.add_argument("--snapshot", type=Path, help="Explicit archived snapshot instead of latest valid snapshot")
    parser.add_argument("--model", default="fpl-llama3", help="Ollama name created by 4_deploy_ollama.ps1")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--timeout", type=float, default=180, help="Ollama read timeout in seconds")
    parser.add_argument("--output", type=Path, help="Report path; default data/forecasts/season-gw-timestamp.json")
    parser.add_argument("--labels", type=Path,
                        help="Optional accumulated training.jsonl; score matching season/GW only, never used in prompt")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a positive finite number")
    try:
        import requests

        path, snapshot = latest_snapshot(args.snapshots, args.snapshot)
        examples = load_examples(args.labels) if args.labels is not None else []
        with requests.Session() as session:
            session.headers.update({"Accept": "application/json"})
            prediction = request_forecast(session, snapshot, args.model, args.ollama_url, args.timeout)
        generated_at = datetime.now(timezone.utc)
        report = {
            "schema_version": 1, "task": "individual_fpl_points_forecast",
            "season": snapshot["season"], "gameweek": snapshot["gameweek"],
            "model": args.model, "snapshot": str(path),
            "snapshot_fetched_at": snapshot["fetched_at"], "deadline_time": snapshot["deadline_time"],
            "generated_at": generated_at.isoformat(),
            "generated_before_deadline": generated_at < utc_timestamp(snapshot["deadline_time"]),
            "optimizer_feedback": False,
            "note": "Standalone forecast only; does not change squad selection or train tactical reasoning",
            **prediction,
            "evaluation": evaluate(snapshot, prediction, examples) if args.labels is not None
            else {"status": "not_requested"},
        }
        output = args.output or (
            Path("data") / "forecasts"
            / f"{snapshot['season']}-gw{snapshot['gameweek']}-{generated_at.strftime('%Y%m%dT%H%M%S%fZ')}.json"
        )
        if output.resolve() == path.resolve() or (args.labels and output.resolve() == args.labels.resolve()):
            raise ValueError("--output must not overwrite the input snapshot or label dataset")
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = output.with_name(output.name + f".{uuid4().hex}.partial")
        serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        try:
            staging.write_text(serialized + "\n", encoding="utf-8")
            staging.replace(output)
        finally:
            staging.unlink(missing_ok=True)
        print(serialized)
        print(f"Forecast report saved: {output}", file=sys.stderr)
        return 0
    except (ImportError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Forecast failed: {exc}", file=sys.stderr)
        return 1
    except requests.RequestException as exc:
        print(f"Local Ollama request failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
