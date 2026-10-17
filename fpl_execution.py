"""Durable, fail-closed receipts for account mutations."""

from collections.abc import Callable
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4


def previous_execution(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("status") not in ("pending", "complete"):
        raise ValueError(f"Invalid execution receipt: {path}; reconcile the account manually")
    if record["status"] == "pending":
        raise RuntimeError(
            f"Unfinished submission recorded in {path}. Check the actual account before "
            "manually reconciling this receipt; automatic retries are disabled."
        )
    return record


def execute_once(
    path: Path, details: dict[str, Any], operation: Callable[[], None]
) -> None:
    """Persist intent before the first POST; ambiguity must never trigger a retry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        **details, "status": "pending",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError(f"Execution receipt already exists: {path}; refusing duplicate writes") from exc
    with handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    operation()
    record.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat())
    staging = path.with_name(f"{path.name}.{uuid4().hex}.partial")
    try:
        with staging.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)
