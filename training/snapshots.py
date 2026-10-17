"""Snapshot contract validation shared by data preparation and local inference."""

from datetime import datetime, timezone
import math

from training.prompts import forecast_input


def utc_timestamp(value):
    if not isinstance(value, str):
        raise ValueError("timestamps must be ISO-8601 strings with a UTC offset")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.utcoffset() is None:
        raise ValueError("timestamps must include a UTC offset")
    return result.astimezone(timezone.utc)


def season_for_deadline(value):
    deadline = utc_timestamp(value)
    start = deadline.year if deadline.month >= 7 else deadline.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def season_for_bootstrap(bootstrap):
    """Anchor the entire season to the first event, including delayed summer GWs."""
    events = bootstrap["events"]
    if not isinstance(events, list) or not events:
        raise ValueError("bootstrap must contain official events")
    first = min(events, key=lambda event: event["id"])
    return season_for_deadline(first["deadline_time"])


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_snapshot(snapshot):
    if (
        not isinstance(snapshot, dict)
        or not integer(snapshot.get("schema_version"))
        or snapshot["schema_version"] != 1
    ):
        raise ValueError("expected snapshot schema_version 1")
    if not integer(snapshot.get("gameweek")) or not 1 <= snapshot["gameweek"] <= 38:
        raise ValueError("gameweek must be an integer in 1..38")
    bootstrap = snapshot["bootstrap"]
    if not isinstance(bootstrap, dict) or not isinstance(snapshot["fixtures"], list):
        raise ValueError("bootstrap must be an object and fixtures must be a list")
    season = season_for_bootstrap(bootstrap)
    if snapshot.get("season") != season:
        raise ValueError(f"season must be {season!r}, inferred from the first official event")
    if utc_timestamp(snapshot["fetched_at"]) >= utc_timestamp(snapshot["deadline_time"]):
        raise ValueError("snapshot was fetched at or after the deadline")
    event = next(
        (event for event in bootstrap["events"] if event["id"] == snapshot["gameweek"]), None
    )
    if event is None or utc_timestamp(event["deadline_time"]) != utc_timestamp(snapshot["deadline_time"]):
        raise ValueError("snapshot deadline does not match its archived official event")
    elements = {player["id"]: player for player in bootstrap["elements"]}
    squad = snapshot["squad_ids"]
    starters = snapshot["starter_ids"]
    if (
        not isinstance(squad, list) or len(squad) != 15
        or not all(integer(pid) and pid in elements for pid in squad)
        or len(set(squad)) != 15
    ):
        raise ValueError("squad_ids must contain 15 distinct archived player IDs")
    if (
        not isinstance(starters, list) or len(starters) != 11
        or not all(integer(pid) and pid in squad for pid in starters)
        or len(set(starters)) != 11
    ):
        raise ValueError("starter_ids must contain 11 distinct squad IDs")
    captain, vice = snapshot["captain_id"], snapshot["vice_captain_id"]
    if (
        not integer(captain) or not integer(vice)
        or captain not in starters or vice not in starters or captain == vice
    ):
        raise ValueError("captain and vice-captain must be distinct starters")
    predictions = snapshot["predictions"]
    if not isinstance(predictions, list):
        raise ValueError("predictions must be a list")
    ids = []
    for prediction in predictions:
        pid = prediction["player_id"]
        if not integer(pid) or pid not in elements:
            raise ValueError("prediction has an unknown player_id")
        if not finite_number(prediction["xpts"]) or not finite_number(prediction["price"]):
            raise ValueError("prediction xpts and price must be finite numbers")
        ids.append(pid)
    if len(set(ids)) != len(ids) or not set(squad).issubset(ids):
        raise ValueError("predictions must uniquely cover every squad player")
    forecast_input(snapshot)
    return snapshot
