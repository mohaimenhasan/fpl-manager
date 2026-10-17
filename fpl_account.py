"""Read-only account snapshots and verified, no-hit transfer recommendations."""

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
from typing import Any

import pulp
import requests

from fpl_manager import (
    API_URL, QUOTAS, USER_AGENT, Player, engineer_players, fetch_json, next_event,
    rank, solve,
)


@dataclass(frozen=True)
class AccountState:
    entry_id: int
    name: str
    source_event: int
    squad_ids: tuple[int, ...]
    bank: int | None
    selling_prices: dict[int, int] | None
    free_transfers: int | None
    verified: bool
    warnings: tuple[str, ...]


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO UTC timestamp")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO UTC timestamp") from exc
    if result.tzinfo is None or result.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be an ISO UTC timestamp")
    return result


def _registry(bootstrap: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = bootstrap["elements"]
    result = {_integer(row["id"], "Player ID", 1): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate official player IDs")
    return result


def _squad_ids(picks: Any, registry: dict[int, dict[str, Any]]) -> tuple[int, ...]:
    if not isinstance(picks, list) or len(picks) != 15:
        raise ValueError("Account must contain exactly 15 picks")
    if any(not isinstance(pick, dict) for pick in picks):
        raise ValueError("Invalid account picks")
    ids = tuple(_integer(pick.get("element"), "Pick element", 1) for pick in picks)
    if len(set(ids)) != 15 or not set(ids).issubset(registry):
        raise ValueError("Account picks must be 15 unique official players")
    if Counter(registry[player_id]["element_type"] for player_id in ids) != QUOTAS:
        raise ValueError("Account picks do not satisfy position quotas")
    return ids


def _no_active_chip(team: dict[str, Any]) -> None:
    if team.get("active_chip"):
        raise ValueError("Active chip: capture private team state after the chip expires")
    chips = team.get("chips", [])
    if not isinstance(chips, list) or any(not isinstance(chip, dict) for chip in chips):
        raise ValueError("Invalid chip state")
    if any(
        chip.get("status_for_entry") == "active" or chip.get("is_pending")
        for chip in chips
    ):
        raise ValueError("Active chip: exact transfer planning is unavailable")


def load_account(
    bootstrap: dict[str, Any], entry_id: int, state_path: Path | None = None,
    private_payload: dict[str, Any] | None = None,
) -> AccountState:
    """Read public data or a recent local/in-memory private my-team wrapper.

    The local file is read only; this module never authenticates, persists owner
    details, or submits changes to FPL.
    """
    _integer(entry_id, "Entry ID", 1)
    now = datetime.now(timezone.utc)
    upcoming = next_event(bootstrap, now)
    registry = _registry(bootstrap)
    if state_path is not None and private_payload is not None:
        raise ValueError("Choose either a private state file or an in-memory payload")
    snapshot = private_payload
    if state_path is not None:
        with Path(state_path).open(encoding="utf-8") as handle:
            snapshot = json.load(handle)
        if snapshot is None:
            raise ValueError("Private state must be a snapshot wrapper")
    if snapshot is not None:
        if not isinstance(snapshot, dict):
            raise ValueError("Private state must be a snapshot wrapper")
        if _integer(snapshot.get("entry_id"), "Snapshot entry ID", 1) != entry_id:
            raise ValueError("Private state belongs to a different entry")
        if _integer(snapshot.get("gameweek"), "Snapshot gameweek", 1) != upcoming["id"]:
            raise ValueError("Private state does not match the upcoming gameweek")
        captured = _utc(snapshot.get("fetched_at"), "fetched_at")
        deadline = _utc(upcoming["deadline_time"], "Deadline")
        if not timedelta(0) <= now - captured <= timedelta(hours=24):
            raise ValueError("Private state must be captured within the last 24 hours")
        if captured >= deadline:
            raise ValueError("Private state must be captured before the next deadline")

    context = requests.Session() if private_payload is None else nullcontext()
    with context as session:
        if private_payload is not None:
            entry = {"name": snapshot.get("name", f"Entry {entry_id}")}
        else:
            session.headers["User-Agent"] = USER_AGENT
            entry = fetch_json(session, f"{API_URL}/entry/{entry_id}/")
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise ValueError("Public entry response lacks a team name")
        if snapshot is None:
            current = [event for event in bootstrap["events"] if event.get("is_current")]
            previous = [
                event for event in bootstrap["events"]
                if event["id"] < upcoming["id"]
                and _utc(event["deadline_time"], "Deadline") <= now
            ]
            events = current or previous
            if not events:
                raise ValueError("No published picks event; provide private team state")
            source_event = max(event["id"] for event in events)
            public = fetch_json(
                session, f"{API_URL}/entry/{entry_id}/event/{source_event}/picks/",
            )
            if not isinstance(public, dict):
                raise ValueError("Invalid public picks response")
            if public.get("active_chip") == "freehit":
                raise ValueError(
                    "Public Free Hit picks are temporary; provide private team state "
                    "after the chip expires to establish the permanent squad"
                )
            ids = _squad_ids(public.get("picks"), registry)
            bank = entry.get("last_deadline_bank")
            if type(bank) is not int or bank < 0:
                bank = None
            return AccountState(
                entry_id, entry["name"], source_event, ids, bank, None, None, False,
                (
                    f"Public GW{source_event} picks are a deadline snapshot and may "
                    f"miss transfers already made for GW{upcoming['id']}.",
                    "Public bank is last-deadline data, not a verified current balance. "
                    "Selling prices and free transfers are unknown; exact transfer "
                    "planning requires a recent private team-state file.",
                ),
            )

    team = snapshot.get("team")
    if not isinstance(team, dict):
        raise ValueError("Private state requires raw my-team JSON in 'team'")
    _no_active_chip(team)
    ids = _squad_ids(team.get("picks"), registry)
    prices = {}
    for pick in team["picks"]:
        player_id = pick["element"]
        price = _integer(pick.get("selling_price"), "Selling price")
        if price > _integer(registry[player_id]["now_cost"], "Current price", 1):
            raise ValueError("Selling price exceeds the current purchase price")
        prices[player_id] = price
    transfers = team.get("transfers")
    if not isinstance(transfers, dict):
        raise ValueError("Private state requires raw transfers data")
    if transfers.get("status") in ("unlimited", "wildcard", "freehit"):
        raise ValueError("Unlimited transfer modes are not supported")
    bank = _integer(transfers.get("bank"), "Bank")
    limit = _integer(transfers.get("limit"), "Finite transfer limit")
    made = _integer(transfers.get("made"), "Transfers made")
    if limit > 5:
        raise ValueError("Unlimited or invalid transfer limit; expected 0..5")
    free_transfers = max(0, limit - made)
    return AccountState(
        entry_id, entry["name"], upcoming["id"], ids, bank, prices,
        free_transfers, True, (),
    )


def project_owned_players(
    bootstrap: dict[str, Any], fixtures: list[dict[str, Any]], gw: int,
    state: AccountState,
) -> list[Player]:
    """Keep all owned players, including unavailable assets at zero projection."""
    registry = _registry(bootstrap)
    _squad_ids([{"element": player_id} for player_id in state.squad_ids], registry)
    eligible = {
        player.player_id: player for player in engineer_players(bootstrap, fixtures, gw)
    }
    missing = set(state.squad_ids) - eligible.keys()
    if missing:
        adjusted = {
            **bootstrap,
            "elements": [
                {**row, "status": "a", "chance_of_playing_next_round": None}
                if row["id"] in missing else row
                for row in bootstrap["elements"]
            ],
        }
        for player in engineer_players(adjusted, fixtures, gw):
            if player.player_id in missing:
                chance = registry[player.player_id].get("chance_of_playing_next_round")
                if chance is not None:
                    chance = float(chance)
                    if not math.isfinite(chance) or not 0 <= chance <= 100:
                        raise ValueError("Invalid owned player playing chance")
                eligible[player.player_id] = replace(player, xpts=0.0, chance=chance)
    return [eligible[player_id] for player_id in state.squad_ids]


def plan_transfers(
    players: list[Player], owned: list[Player], state: AccountState,
    max_transfers: int = 1,
) -> tuple[list[Player], list[tuple[Player, Player]], int]:
    """Maximize squad xpts, then minimize transfers, without hits or chip use."""
    if state.verified is not True:
        raise ValueError("Exact transfer planning requires verified private team state")
    bank = _integer(state.bank, "Verified bank")
    free = _integer(state.free_transfers, "Verified free transfers")
    if free > 5:
        raise ValueError("Free transfers must be in 0..5")
    _integer(max_transfers, "Maximum transfers")
    own = {player.player_id: player for player in owned}
    if len(own) != 15 or len(state.squad_ids) != 15 or set(own) != set(state.squad_ids):
        raise ValueError("Owned players must match the 15 verified account picks")
    if Counter(player.position for player in owned) != QUOTAS:
        raise ValueError("Owned players do not satisfy position quotas")
    prices = state.selling_prices
    if not isinstance(prices, dict) or set(prices) != set(own):
        raise ValueError("Verified selling prices must cover every owned player")
    for player_id, price in prices.items():
        if _integer(price, "Selling price") > own[player_id].cost:
            raise ValueError("Selling price exceeds current purchase price")
    if len({player.player_id for player in players}) != len(players):
        raise ValueError("Candidate player IDs must be unique")
    pool = {player.player_id: player for player in players}
    pool.update(own)
    for player in pool.values():
        _integer(player.player_id, "Player ID", 1)
        _integer(player.cost, "Purchase cost", 1)
        if player.position not in QUOTAS:
            raise ValueError("Invalid player position")
        if not math.isfinite(player.xpts) or player.xpts < 0:
            raise ValueError("Player projections must be finite and nonnegative")
    problem = pulp.LpProblem("FPL_account_transfers", pulp.LpMaximize)
    chosen = {
        player_id: pulp.LpVariable(f"account_{player_id}", cat="Binary")
        for player_id in pool
    }
    acquired = pulp.lpSum(chosen[player_id] for player_id in pool if player_id not in own)
    spending = pulp.lpSum(
        player.cost * chosen[player_id]
        for player_id, player in pool.items() if player_id not in own
    )
    proceeds = pulp.lpSum(
        prices[player_id] * (1 - chosen[player_id]) for player_id in own
    )
    score = pulp.lpSum(player.xpts * chosen[player_id] for player_id, player in pool.items())
    problem += score
    problem += pulp.lpSum(chosen.values()) == 15
    problem += acquired <= min(max_transfers, free)
    problem += spending <= bank + proceeds
    for position, count in QUOTAS.items():
        problem += pulp.lpSum(
            chosen[player_id] for player_id, player in pool.items()
            if player.position == position
        ) == count
    for team_id in {player.team_id for player in pool.values()}:
        problem += pulp.lpSum(
            chosen[player_id] for player_id, player in pool.items()
            if player.team_id == team_id
        ) <= 3
    solve(problem)
    optimum = pulp.value(score)
    if optimum is None:
        optimum = 0.0
    problem += score >= optimum - 1e-7
    problem.sense = pulp.LpMinimize
    problem.setObjective(acquired)
    solve(problem)
    squad = sorted(
        (player for player_id, player in pool.items() if chosen[player_id].value() > 0.5),
        key=lambda player: (player.position, tuple(-value for value in rank(player))),
    )
    selected = {player.player_id for player in squad}
    sold = sorted(
        (player for player_id, player in own.items() if player_id not in selected),
        key=lambda player: (player.position, player.player_id),
    )
    bought = sorted(
        (player for player in squad if player.player_id not in own),
        key=lambda player: (player.position, player.player_id),
    )
    bank_after = bank + sum(prices[player.player_id] for player in sold) - sum(
        player.cost for player in bought
    )
    if (
        len(squad) != 15 or len(selected) != 15
        or Counter(player.position for player in squad) != QUOTAS
        or max(Counter(player.team_id for player in squad).values()) > 3
        or len(bought) != len(sold) or len(bought) > min(max_transfers, free)
        or [player.position for player in bought] != [player.position for player in sold]
        or bank_after < 0 or sum(player.xpts for player in squad) < optimum - 1e-6
    ):
        raise RuntimeError("Transfer solution failed exact account constraint validation")
    return squad, list(zip(sold, bought)), bank_after
