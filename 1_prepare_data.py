#!/usr/bin/env python3
"""Join archived pre-deadline features to official completed-gameweek labels."""

import argparse
import json
import math
from pathlib import Path
import sys

from training.data import load_examples
from training.prompts import FORECAST_INSTRUCTION, forecast_input
from training.snapshots import (
    finite_number, season_for_bootstrap, season_for_deadline, utc_timestamp, validate_snapshot,
)

API_ROOT = "https://fantasy.premierleague.com/api"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def fetch_json(session, endpoint, timeout):
    response = session.get(f"{API_ROOT}/{endpoint}/", timeout=(timeout, timeout))
    response.raise_for_status()
    return response.json()


def make_example(snapshot, live):
    elements = {player["id"]: player["stats"] for player in live["elements"]}
    labels = []
    for pid in snapshot["squad_ids"]:
        points = elements[pid]["total_points"]
        if not finite_number(points):
            raise ValueError(f"invalid realized points for player {pid}")
        labels.append({"player_id": pid, "points": points})
    return {
        "instruction": FORECAST_INSTRUCTION,
        "input": forecast_input(snapshot),
        "output": {"players": labels},
        "metadata": {
            "schema_version": 1,
            "season": snapshot["season"],
            "gameweek": snapshot["gameweek"],
            "snapshot_fetched_at": snapshot["fetched_at"],
            "label_type": "realized_individual_points",
            "label_source": f"{API_ROOT}/event/{snapshot['gameweek']}/live/",
            "outcome_summary": {
                "squad_raw_points": sum(row["points"] for row in labels),
                "starter_raw_points": sum(
                    row["points"] for row in labels if row["player_id"] in snapshot["starter_ids"]
                ),
            },
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, default=Path("data") / "snapshots")
    parser.add_argument("--output", type=Path, default=Path("data") / "training.jsonl")
    parser.add_argument("--timeout", type=float, default=30, help="Connect/read timeout in seconds")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a positive finite number")
    try:
        import requests

        existing = load_examples(args.output) if args.output.exists() else []
        paths = sorted(args.snapshots.glob("*.json"))
        if not paths:
            raise ValueError(f"no snapshot JSON files in {args.snapshots}")
        selected = {}
        for path in paths:
            try:
                snapshot = validate_snapshot(json.loads(path.read_text(encoding="utf-8")))
            except (ValueError, KeyError, TypeError) as exc:
                print(f"Rejecting {path.name}: {exc}", file=sys.stderr)
                continue
            key = (snapshot["season"], snapshot["gameweek"])
            if key not in selected or utc_timestamp(snapshot["fetched_at"]) > utc_timestamp(selected[key]["fetched_at"]):
                selected[key] = snapshot
        if not selected:
            raise ValueError("no valid pre-deadline snapshots")
        records = []
        with requests.Session() as session:
            session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
            current = fetch_json(session, "bootstrap-static", args.timeout)
            events = {event["id"]: event for event in current["events"]}
            if not events:
                raise ValueError("official bootstrap contains no events")
            current_season = season_for_bootstrap(current)
            fixtures = fetch_json(session, "fixtures", args.timeout)
            for (season, gw), snapshot in sorted(selected.items()):
                event = events.get(gw)
                if season != current_season or event is None:
                    print(f"Rejecting {season} GW{gw}: official API is not that season", file=sys.stderr)
                    continue
                if utc_timestamp(event["deadline_time"]) != utc_timestamp(snapshot["deadline_time"]):
                    print(f"Rejecting {season} GW{gw}: official deadline changed", file=sys.stderr)
                    continue
                gw_fixtures = [fixture for fixture in fixtures if fixture.get("event") == gw]
                if (
                    event.get("finished") is not True
                    or event.get("data_checked") is not True
                    or not gw_fixtures
                    or any(fixture.get("finished") is not True for fixture in gw_fixtures)
                ):
                    print(
                        f"Pending {season} GW{gw}: wait until all fixtures finish "
                        "and official FPL marks the event data checked",
                        file=sys.stderr,
                    )
                    continue
                live = fetch_json(session, f"event/{gw}/live", args.timeout)
                records.append(make_example(snapshot, live))
        if not records:
            raise ValueError(
                "no completed, checked gameweeks with matching pre-deadline snapshots; "
                "wait until the snapshotted gameweek is complete and officially checked. "
                "Any existing dataset has been left unchanged"
            )
        merged = {
            (record["input"]["season"], record["input"]["gameweek"]): record
            for record in existing
        }
        for record in records:
            merged[(record["input"]["season"], record["input"]["gameweek"])] = record
        ordered = sorted(merged.values(), key=lambda record: utc_timestamp(record["input"]["deadline_time"]))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # A failed HTTP request or validation leaves any previous dataset untouched.
        staging = args.output.with_name(args.output.name + ".partial")
        try:
            staging.write_text(
                "".join(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in ordered),
                encoding="utf-8",
            )
            load_examples(staging)
            staging.replace(args.output)
        finally:
            staging.unlink(missing_ok=True)
        print(
            f"Updated {len(records)} completed gameweeks; "
            f"preserved accumulated dataset of {len(ordered)} gameweeks in {args.output}"
        )
        return 0
    except (ImportError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Data preparation failed: {exc}", file=sys.stderr)
        return 1
    except requests.RequestException as exc:
        print(f"Official FPL request failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
