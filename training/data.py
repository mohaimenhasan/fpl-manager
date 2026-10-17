"""Strict JSONL validation and chronological splitting, without ML dependencies."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path


def _utc(value):
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.utcoffset() is None:
        raise ValueError("timestamps must include a UTC offset")
    return stamp.astimezone(timezone.utc)


def load_examples(path):
    examples = []
    seen = set()
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("each record must be an object")
                if not isinstance(item.get("instruction"), str) or not item["instruction"].strip():
                    raise ValueError("instruction must be a nonempty string")
                inputs, output, metadata = item["input"], item["output"], item["metadata"]
                if not isinstance(inputs, dict) or not isinstance(output, dict) or not isinstance(metadata, dict):
                    raise ValueError("input, output and metadata must be objects")
                if metadata.get("schema_version") != 1 or metadata.get("label_type") != "realized_individual_points":
                    raise ValueError("expected version 1 realized_individual_points labels")
                if _utc(inputs["as_of"]) >= _utc(inputs["deadline_time"]):
                    raise ValueError("input was recorded at or after the deadline")
                if _utc(metadata["snapshot_fetched_at"]) != _utc(inputs["as_of"]):
                    raise ValueError("metadata and input timestamps disagree")
                if inputs["season"] != metadata["season"] or inputs["gameweek"] != metadata["gameweek"]:
                    raise ValueError("metadata and input gameweeks disagree")
                if (
                    not isinstance(inputs["season"], str)
                    or not isinstance(inputs["gameweek"], int)
                    or isinstance(inputs["gameweek"], bool)
                    or not 1 <= inputs["gameweek"] <= 38
                ):
                    raise ValueError("invalid season or gameweek")
                feature_players, targets = inputs["players"], output["players"]
                if not isinstance(feature_players, list) or not feature_players or not isinstance(targets, list):
                    raise ValueError("input and output must contain nonempty player lists")
                requested = [player["player_id"] for player in feature_players]
                actual = [player["player_id"] for player in targets]
                if not all(isinstance(pid, int) and not isinstance(pid, bool) and pid > 0 for pid in requested + actual):
                    raise ValueError("player IDs must be positive integers")
                if len(set(requested)) != len(requested) or len(set(actual)) != len(actual) or set(requested) != set(actual):
                    raise ValueError("label IDs must uniquely match the requested player IDs")
                for player in targets:
                    points = player["points"]
                    if isinstance(points, bool) or not isinstance(points, (int, float)) or not math.isfinite(points):
                        raise ValueError("label points must be finite numbers")
                # Reject NaN in any feature, not just in labels.
                json.dumps(item, allow_nan=False)
                key = (inputs["season"], inputs["gameweek"])
                if key in seen:
                    raise ValueError("duplicate season/gameweek; retain only the latest pre-deadline snapshot")
                seen.add(key)
                examples.append(item)
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    if not examples:
        raise ValueError(f"{path}: no nonempty training examples")
    return sorted(examples, key=lambda item: _utc(item["input"]["deadline_time"]))


def chronological_split(examples, validation_fraction):
    """Keep newer, entire gameweeks out of training rather than randomly splitting."""
    count = int(len(examples) * validation_fraction)
    if validation_fraction > 0 and len(examples) > 1:
        count = max(1, count)
    count = min(count, len(examples) - 1)
    return (examples[:-count], examples[-count:]) if count else (examples, [])
