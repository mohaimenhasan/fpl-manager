"""Live, fixture-aware FPL squad optimization and local tactical reasoning."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import pulp
import requests


API_URL = "https://fantasy.premierleague.com/api"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
)
POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
QUOTAS = {1: 2, 2: 5, 3: 5, 4: 3}


@dataclass(frozen=True)
class Player:
    player_id: int
    name: str
    team_id: int
    club: str
    position: int
    cost: int
    baseline: float
    form: float
    xpts: float
    ownership: float
    minutes: int
    chance: float | None
    fixture_ids: tuple[int, ...]
    opponents: tuple[str, ...]
    fixture_modifier: float

    @property
    def price(self) -> float:
        return self.cost / 10


@dataclass(frozen=True)
class Selection:
    squad: tuple[Player, ...]
    starters: tuple[Player, ...]
    bench: tuple[Player, ...]
    captain: Player
    vice_captain: Player

    @property
    def formation(self) -> str:
        counts = Counter(p.position for p in self.starters)
        return f"{counts[2]}-{counts[3]}-{counts[4]}"


def number(value: Any, field: str, default: float | None = None) -> float:
    if value is None and default is not None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric {field}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {field}: {value!r}")
    return result


def fetch_json(session: requests.Session, url: str) -> Any:
    response = session.get(url, timeout=(10, 45))
    response.raise_for_status()
    return response.json()


def fetch_live() -> tuple[dict[str, Any], list[dict[str, Any]], datetime]:
    with requests.Session() as session:
        session.headers["User-Agent"] = USER_AGENT
        bootstrap = fetch_json(session, f"{API_URL}/bootstrap-static/")
        fixtures = fetch_json(session, f"{API_URL}/fixtures/")
    if not isinstance(bootstrap, dict) or not all(
        isinstance(bootstrap.get(key), list) and bootstrap[key]
        for key in ("elements", "teams", "events")
    ):
        raise ValueError("Official bootstrap response lacks players, teams, or events")
    if not isinstance(fixtures, list) or not fixtures:
        raise ValueError("Official fixtures response is empty or invalid")
    return bootstrap, fixtures, datetime.now(timezone.utc)


def next_event(bootstrap: dict[str, Any], now: datetime) -> dict[str, Any]:
    events = [event for event in bootstrap["events"] if event.get("is_next")]
    if len(events) != 1:
        raise ValueError("Expected one upcoming gameweek; season may be over or not released")
    event = events[0]
    deadline = datetime.fromisoformat(event["deadline_time"].replace("Z", "+00:00"))
    if deadline <= now:
        raise ValueError("Upcoming gameweek deadline has passed; official data is stale")
    return event


def season_from_bootstrap(bootstrap: dict[str, Any]) -> str:
    first = min(bootstrap["events"], key=lambda event: event["id"])
    deadline = datetime.fromisoformat(first["deadline_time"].replace("Z", "+00:00"))
    year = deadline.year if deadline.month >= 7 else deadline.year - 1
    return f"{year}-{str(year + 1)[-2:]}"


def strength(team: dict[str, Any], kind: str, venue: str) -> float:
    for field in (f"strength_{kind}_{venue}", f"strength_overall_{venue}", "strength"):
        value = team.get(field)
        if value is not None:
            rating = number(value, field)
            if rating > 0:
                return rating
    raise ValueError(f"No positive {kind} strength for {team['name']} ({venue})")


def engineer_players(
    bootstrap: dict[str, Any], fixtures: list[dict[str, Any]], gameweek: int
) -> list[Player]:
    teams = {team["id"]: team for team in bootstrap["teams"]}
    means = {
        (kind, venue): sum(strength(t, kind, venue) for t in teams.values()) / len(teams)
        for kind in ("overall", "attack", "defence")
        for venue in ("home", "away")
    }
    by_team: dict[int, list[dict[str, Any]]] = {team_id: [] for team_id in teams}
    for fixture in fixtures:
        if fixture.get("event") == gameweek:
            if fixture.get("started") or fixture.get("finished"):
                raise ValueError("Upcoming gameweek contains an already-started fixture")
            for side in ("team_h", "team_a"):
                by_team[fixture[side]].append(fixture)

    frame = pd.DataFrame(bootstrap["elements"])
    required = {
        "id", "web_name", "team", "element_type", "now_cost", "status",
        "chance_of_playing_next_round", "form",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing player fields: {sorted(required - set(frame.columns))}")
    frame = frame[frame["status"].eq("a")].copy()
    chances = pd.to_numeric(frame["chance_of_playing_next_round"], errors="raise")
    frame = frame[chances.isna() | chances.ge(75)]
    players = []
    for row in frame.astype(object).where(pd.notna(frame), None).to_dict("records"):
        position = int(row["element_type"])
        if position not in POSITIONS:
            continue
        team_id = int(row["team"])
        team = teams[team_id]
        modifiers = []
        fixture_ids = []
        opponents = []
        for fixture in by_team[team_id]:
            is_home = fixture["team_h"] == team_id
            opponent = teams[fixture["team_a"] if is_home else fixture["team_h"]]
            opponent_venue = "away" if is_home else "home"
            kind = "attack" if position in (1, 2) else "defence"
            overall = means["overall", opponent_venue] / strength(
                opponent, "overall", opponent_venue
            )
            positional = means[kind, opponent_venue] / strength(
                opponent, kind, opponent_venue
            )
            modifiers.append(max(0.5, min(1.5, (overall + positional) / 2)))
            fixture_ids.append(int(fixture["id"]))
            opponents.append(f"{opponent['short_name']} ({'H' if is_home else 'A'})")
        baseline = number(row.get("ep_next"), "ep_next", 0.0)
        form = number(row["form"], "form", 0.0)
        modifier = sum(modifiers) / len(modifiers) if modifiers else 0.0
        # ep_next is already gameweek-level; only the per-match form is scaled for doubles.
        xpts = max(0.0, (0.75 * baseline + 0.25 * form * len(modifiers)) * modifier)
        chance = row["chance_of_playing_next_round"]
        chance = None if pd.isna(chance) else number(chance, "chance")
        cost_value = number(row["now_cost"], "now_cost")
        if cost_value <= 0 or not cost_value.is_integer():
            raise ValueError(f"Invalid price for player {row['id']}")
        players.append(Player(
            player_id=int(row["id"]), name=row["web_name"], team_id=team_id,
            club=team["short_name"], position=position, cost=int(cost_value),
            baseline=baseline, form=form, xpts=xpts,
            ownership=number(row.get("selected_by_percent"), "ownership", 0.0),
            minutes=int(number(row.get("minutes"), "minutes", 0.0)), chance=chance,
            fixture_ids=tuple(fixture_ids), opponents=tuple(opponents),
            fixture_modifier=modifier,
        ))
    if len({p.player_id for p in players}) != len(players):
        raise ValueError("Duplicate official player IDs")
    return players


def solve(problem: pulp.LpProblem) -> None:
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=120, gapRel=0)
    if not solver.available():
        raise RuntimeError("PuLP CBC solver unavailable; install a supported PuLP distribution")
    status = problem.solve(solver)
    if status != pulp.LpStatusOptimal or problem.sol_status != pulp.LpSolutionOptimal:
        raise RuntimeError(f"No proven optimal solution: {pulp.LpStatus[status]}")


def rank(player: Player) -> tuple[float, float, float, int]:
    return player.xpts, player.baseline, player.form, -player.player_id


def optimize(players: list[Player], budget: int = 1000) -> Selection:
    if budget <= 0:
        raise ValueError("Budget must be positive")
    if len({p.player_id for p in players}) != len(players):
        raise ValueError("Player IDs must be unique")
    squad_problem = pulp.LpProblem("FPL_squad", pulp.LpMaximize)
    chosen = {p.player_id: pulp.LpVariable(f"squad_{p.player_id}", cat="Binary") for p in players}
    squad_problem += pulp.lpSum(p.xpts * chosen[p.player_id] for p in players)
    squad_problem += pulp.lpSum(p.cost * chosen[p.player_id] for p in players) <= budget
    squad_problem += pulp.lpSum(chosen.values()) == 15
    for position, count in QUOTAS.items():
        squad_problem += pulp.lpSum(
            chosen[p.player_id] for p in players if p.position == position
        ) == count
    for team_id in {p.team_id for p in players}:
        squad_problem += pulp.lpSum(
            chosen[p.player_id] for p in players if p.team_id == team_id
        ) <= 3
    solve(squad_problem)
    squad = tuple(sorted(
        (p for p in players if chosen[p.player_id].value() > 0.5),
        key=lambda p: (p.position, -p.xpts, p.player_id),
    ))
    return select_lineup(squad, budget)


def select_lineup(squad: tuple[Player, ...], budget: int | None = None) -> Selection:
    """Choose a legal XI without replacing any of an existing squad's players."""
    squad = tuple(sorted(squad, key=lambda p: (p.position, -p.xpts, p.player_id)))
    if len(squad) != 15 or Counter(p.position for p in squad) != QUOTAS:
        raise ValueError("Lineup requires a complete 2/5/5/3 squad")
    lineup_problem = pulp.LpProblem("FPL_lineup", pulp.LpMaximize)
    starting = {p.player_id: pulp.LpVariable(f"start_{p.player_id}", cat="Binary") for p in squad}
    lineup_problem += pulp.lpSum(p.xpts * starting[p.player_id] for p in squad)
    lineup_problem += pulp.lpSum(starting.values()) == 11
    for position, minimum, maximum in ((1, 1, 1), (2, 3, 5), (3, 2, 5), (4, 1, 3)):
        count = pulp.lpSum(starting[p.player_id] for p in squad if p.position == position)
        lineup_problem += count >= minimum
        lineup_problem += count <= maximum
    solve(lineup_problem)
    starters = tuple(p for p in squad if starting[p.player_id].value() > 0.5)
    starter_ids = {p.player_id for p in starters}
    bench = tuple(sorted(
        (p for p in squad if p.player_id not in starter_ids),
        key=lambda p: (p.position == 1, -p.xpts, p.player_id),
    ))
    armbands = sorted(starters, key=rank, reverse=True)
    captain = armbands[0]
    vice_candidates = [
        p for p in armbands[1:]
        if p.fixture_ids and set(p.fixture_ids).isdisjoint(captain.fixture_ids)
    ]
    if not captain.fixture_ids or not vice_candidates:
        raise ValueError("Cannot allocate captain and vice-captain in independent fixtures")
    result = Selection(squad, starters, bench, captain, vice_candidates[0])
    validate_selection(result, sum(p.cost for p in squad) if budget is None else budget)
    return result


def validate_selection(selection: Selection, budget: int) -> None:
    squad = selection.squad
    ids = {p.player_id for p in squad}
    starter_ids = {p.player_id for p in selection.starters}
    bench_ids = {p.player_id for p in selection.bench}
    if len(squad) != 15 or len(ids) != 15:
        raise ValueError("Squad must contain 15 distinct players")
    if Counter(p.position for p in squad) != QUOTAS:
        raise ValueError("Invalid squad position quotas")
    if sum(p.cost for p in squad) > budget:
        raise ValueError("Squad exceeds budget")
    if max(Counter(p.team_id for p in squad).values()) > 3:
        raise ValueError("Squad exceeds club limit")
    if len(starter_ids) != 11 or len(bench_ids) != 4 or starter_ids & bench_ids or ids != starter_ids | bench_ids:
        raise ValueError("Lineup and bench must partition the squad")
    counts = Counter(p.position for p in selection.starters)
    if not (counts[1] == 1 and 3 <= counts[2] <= 5 and 2 <= counts[3] <= 5 and 1 <= counts[4] <= 3):
        raise ValueError("Invalid starting formation")
    if selection.bench[-1].position != 1 or any(p.position == 1 for p in selection.bench[:-1]):
        raise ValueError("Backup goalkeeper must be last on bench")
    if list(selection.bench[:-1]) != sorted(selection.bench[:-1], key=lambda p: (-p.xpts, p.player_id)):
        raise ValueError("Outfield bench is not ordered by descending xPts")
    if selection.captain.player_id not in starter_ids or selection.vice_captain.player_id not in starter_ids:
        raise ValueError("Armbands must be assigned to starters")
    if selection.captain.player_id == selection.vice_captain.player_id or not set(
        selection.captain.fixture_ids
    ).isdisjoint(selection.vice_captain.fixture_ids):
        raise ValueError("Captaincy must hedge distinct fixtures")


def player_description(player: Player) -> str:
    return (
        f"[{POSITIONS[player.position]}] {player.name} ({player.club}) - "
        f"\u00a3{player.price:.1f}m | xPts: {player.xpts:.1f}"
    )


def api_lineup(selection: Selection) -> list[dict[str, Any]]:
    # FPL's submission order puts the reserve GK at 12, unlike our display order.
    ordered = [*selection.starters, selection.bench[-1], *selection.bench[:-1]]
    return [
        {
            "element": player.player_id, "position": index,
            "multiplier": (2 if player == selection.captain else 1) if index <= 11 else 0,
            "is_captain": player == selection.captain,
            "is_vice_captain": player == selection.vice_captain,
        }
        for index, player in enumerate(ordered, 1)
    ]


def tactical_facts(selection: Selection) -> dict[str, str]:
    premium = sorted(selection.squad, key=lambda p: (-p.cost, -p.xpts))[:3]
    values = sorted(selection.squad, key=lambda p: (p.cost, -p.xpts))[:3]
    differentials = sorted(
        (p for p in selection.starters if p.ownership < 10), key=rank, reverse=True
    )
    risks = []
    for player in selection.squad:
        reasons = []
        if not player.fixture_ids:
            reasons.append("blank gameweek")
        if player.fixture_modifier < 0.9:
            reasons.append("tough fixture")
        if player.chance is not None and player.chance < 100:
            reasons.append(f"{player.chance:g}% playing chance")
        if player.minutes < 180:
            reasons.append("limited season minutes")
        if reasons:
            risks.append(f"{player.name}: {', '.join(reasons)}")
    return {
        "premium_core": "; ".join(f"{p.name} (\u00a3{p.price:.1f}m)" for p in premium),
        "value_enablers": "; ".join(
            f"{p.name} (\u00a3{p.price:.1f}m, {p.xpts:.1f} xPts)" for p in values
        ),
        "differentials": "; ".join(
            f"{p.name} ({p.ownership:.1f}% owned)" for p in differentials[:3]
        ) or "No starters below 10% ownership.",
        "risk_vectors": "; ".join(risks) or "No flags in available inputs; availability is not a minutes guarantee.",
    }


def llm_briefing(
    selection: Selection, model: str, url: str, facts: dict[str, str]
) -> dict[str, str]:
    payload = {
        "model": model, "stream": False,
        "format": {
            "type": "object",
            "properties": {key: {"type": "string"} for key in facts},
            "required": list(facts), "additionalProperties": False,
        },
        "options": {"temperature": 0, "num_predict": 400, "num_ctx": 4096},
        "messages": [
            {"role": "system", "content": (
                "You are an FPL tactical analyst. Use ONLY supplied facts; do not change the "
                "locked squad, lineup, armbands or invent injuries, match dates, player roles "
                "or historical performance. xPts is a heuristic, not calibrated probability. "
                "Return a JSON object with exactly four nonempty string fields: premium_core, "
                "value_enablers, differentials, risk_vectors. Each must be one sentence under 25 words. "
                "Explain price tradeoffs and fixture/rotation uncertainty, never promise returns."
            )},
            {"role": "user", "content": json.dumps({
                "formation": selection.formation,
                "captain": selection.captain.name,
                "vice_captain": selection.vice_captain.name,
                "facts": facts,
                "squad": [
                    {
                        "name": p.name, "club": p.club, "position": POSITIONS[p.position],
                        "price": p.price, "xpts": round(p.xpts, 2),
                        "fixtures": p.opponents,
                        "starting": p in selection.starters,
                    }
                    for p in selection.squad
                ],
            })},
        ],
    }
    response = requests.post(f"{url.rstrip('/')}/api/chat", json=payload, timeout=(10, 300))
    if not response.ok:
        raise requests.HTTPError(
            f"Ollama returned HTTP {response.status_code}: {response.text[:1000]}",
            response=response,
        )
    response.raise_for_status()
    body = response.json()
    if body.get("done_reason") == "length":
        raise ValueError("Local LLM briefing was truncated at the generation token limit")
    result = json.loads(body["message"]["content"])
    if not isinstance(result, dict) or set(result) != set(facts) or not all(
        isinstance(value, str) and value.strip() for value in result.values()
    ):
        raise ValueError("Local LLM returned an invalid tactical briefing")
    return result


def render_directive(
    selection: Selection, event: dict[str, Any], budget: int,
    briefing: dict[str, str], model: str | None, account_note: str | None = None,
) -> str:
    captain, vice = selection.captain, selection.vice_captain
    lines = [
        "=" * 61,
        f"FPL AUTONOMOUS SQUAD DIRECTIVE: GW {event['id']}",
        f"Budget Used: \u00a3{sum(p.cost for p in selection.squad) / 10:.1f}m / "
        f"\u00a3{budget / 10:.1f}m | Projected Squad xPts: {sum(p.xpts for p in selection.squad):.1f}",
        f"STARTING XI (Formation: {selection.formation})",
        *(player_description(p) for p in selection.starters),
        "CAPTAINCY",
        f"Captain (C):      {captain.name} ({captain.club}) - Highest starter xPts "
        f"({captain.xpts:.1f}); fixtures: {', '.join(captain.opponents)}.",
        f"Vice-Captain (VC): {vice.name} ({vice.club}) - Highest xPts "
        f"({vice.xpts:.1f}) among starters in matches independent of the captain; "
        f"fixtures: {', '.join(vice.opponents)}.",
        "BENCH HIERARCHY",
        *(player_description(p) for p in selection.bench),
        "TACTICAL BRIEFING",
        f"Premium Core: {briefing['premium_core']}",
        f"Value Enablers: {briefing['value_enablers']}",
        f"Differential Picks: {briefing['differentials']}",
        f"Risk Vectors: {briefing['risk_vectors']}",
        f"Reasoning: local Ollama ({model})" if model else "Reasoning: deterministic facts (--no-llm).",
        f"Deadline: {event['deadline_time']} | xPts is an uncalibrated heuristic.",
        account_note or "Fresh squad optimization only; no account transfers or team submissions are made.",
        "=" * 61,
    ]
    return "\n".join(lines)


def save_snapshot(
    directory: Path, bootstrap: dict[str, Any], fixtures: list[dict[str, Any]],
    event: dict[str, Any], fetched_at: datetime, players: list[Player], selection: Selection,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    season = season_from_bootstrap(bootstrap)
    path = directory / f"{season}-gw{event['id']}-{fetched_at.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    payload = {
        "schema_version": 1, "season": season, "fetched_at": fetched_at.isoformat(),
        "gameweek": event["id"], "deadline_time": event["deadline_time"],
        "bootstrap": bootstrap, "fixtures": fixtures,
        "squad_ids": [p.player_id for p in selection.squad],
        "starter_ids": [p.player_id for p in selection.starters],
        "captain_id": selection.captain.player_id,
        "vice_captain_id": selection.vice_captain.player_id,
        "predictions": [
            {"player_id": p.player_id, "xpts": p.xpts, "price": p.price} for p in players
        ],
    }
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, default=100.0, help="Fresh squad budget in millions")
    parser.add_argument("--ollama-model", default="llama3.1:latest")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--no-llm", action="store_true", help="Explicit deterministic briefing mode")
    parser.add_argument("--snapshot-dir", type=Path, default=Path("data") / "snapshots")
    parser.add_argument("--output", type=Path, default=Path("reports") / "latest-directive.txt")
    parser.add_argument("--entry-id", type=int, help="Manage an existing FPL entry rather than a fresh squad")
    parser.add_argument("--team-state", type=Path, help="Local private team-state export for exact transfer planning")
    parser.add_argument("--max-transfers", type=int, default=1, help="Transfer cap; never take points hits")
    parser.add_argument("--apply", action="store_true", help="Execute authenticated transfers and lineup changes")
    parser.add_argument("--authenticated", action="store_true", help="Use live private team state without submitting changes")
    parser.add_argument("--refresh-auth", action="store_true", help="Refresh authentication from the saved local browser session before account access")
    parser.add_argument("--auth-browser-channel", choices=("msedge", "chromium"), default="msedge")
    parser.add_argument("--auth-file", type=Path, default=Path("data") / "private" / "fpl-auth.json")
    parser.add_argument("--within-hours", type=float, help="Only run within this many hours of the next deadline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.entry_id is not None and args.entry_id <= 0:
        raise ValueError("Entry ID must be positive")
    if not 0 <= args.max_transfers <= 5:
        raise ValueError("Maximum transfers must be in 0..5")
    if (args.apply or args.authenticated or args.team_state) and not args.entry_id:
        raise ValueError("--apply, --authenticated and --team-state require --entry-id")
    if (args.apply or args.authenticated) and args.team_state:
        raise ValueError("Live authenticated mode cannot use an exported --team-state")
    if args.refresh_auth and not (args.apply or args.authenticated):
        raise ValueError("--refresh-auth requires --apply or --authenticated")
    if args.within_hours is not None and (
        not math.isfinite(args.within_hours) or args.within_hours <= 0
    ):
        raise ValueError("--within-hours must be positive and finite")
    budget_value = number(args.budget, "budget") * 10
    if budget_value <= 0 or not math.isclose(budget_value, round(budget_value)):
        raise ValueError("Budget must be positive and expressed in increments of 0.1 million")
    budget = round(budget_value)
    bootstrap, fixtures, fetched_at = fetch_live()
    event = next_event(bootstrap, fetched_at)
    deadline = datetime.fromisoformat(event["deadline_time"].replace("Z", "+00:00"))
    if args.within_hours is not None and (deadline - fetched_at).total_seconds() > args.within_hours * 3600:
        print(f"GW {event['id']} is not yet within {args.within_hours:g} hours of its deadline; no account changes.")
        return
    receipt = Path("data") / "executions" / (
        f"{season_from_bootstrap(bootstrap)}-gw{event['id']}-entry{args.entry_id}.json"
    )
    if args.apply:
        from fpl_execution import previous_execution

        if previous_execution(receipt) is not None:
            print(f"GW {event['id']} already applied for entry {args.entry_id}; no repeated account changes.")
            return
    players = engineer_players(bootstrap, fixtures, event["id"])
    client = None
    try:
        account_note = None
        transfers = []
        private_payload = None
        state = None
        bank_after = None
        if args.entry_id:
            from fpl_account import load_account, plan_transfers, project_owned_players

            if args.apply or args.authenticated:
                from fpl_actions import FPLClient

                if args.refresh_auth:
                    from connect_fpl import connect

                    connect(
                        args.entry_id, args.auth_file, 90, args.auth_browser_channel,
                        headless=True,
                    )
                client = FPLClient(entry_id=args.entry_id, auth_file=args.auth_file)
                private_payload = client.get_team_state(event["id"])
            state = load_account(
                bootstrap, args.entry_id, args.team_state, private_payload=private_payload
            )
            owned = project_owned_players(bootstrap, fixtures, event["id"], state)
            if state.verified:
                squad, transfers, bank_after = plan_transfers(players, owned, state, args.max_transfers)
            else:
                squad = owned
            selection = select_lineup(tuple(squad))
            budget = sum(p.cost for p in squad) + (bank_after if bank_after is not None else state.bank or 0)
            transfer_lines = [
                f"SELL {out.name} ({out.club}) -> BUY {incoming.name} ({incoming.club})"
                for out, incoming in transfers
            ]
            flags = [
                row["web_name"] for row in bootstrap["elements"]
                if row["id"] in state.squad_ids and (
                    row["status"] != "a" or (
                        row.get("chance_of_playing_next_round") is not None
                        and number(row["chance_of_playing_next_round"], "chance") < 75
                    )
                )
            ]
            account_note = (
                f"ENTRY {state.entry_id}: {state.name} | source GW {state.source_event}\n"
                + "\n".join(f"Warning: {warning}" for warning in state.warnings)
                + "\nTRANSFERS\n"
                + ("\n".join(transfer_lines) if transfers else "No transfers proposed.")
                + (f"\nVerified bank after plan: \u00a3{bank_after / 10:.1f}m | points hits: 0" if bank_after is not None else "")
                + (f"\nOwned availability exclusions (zero xPts): {', '.join(flags)}" if flags else "")
                + ("\nPlanned changes; submission is unconfirmed until the execution receipt is complete."
                   if args.apply else "\nPreview only; no account changes submitted.")
            )
            predictions = {p.player_id: p for p in players}
            predictions.update({p.player_id: p for p in owned})
            players = list(predictions.values())
        else:
            selection = optimize(players, budget)
        snapshot = save_snapshot(
            args.snapshot_dir, bootstrap, fixtures, event, fetched_at, players, selection
        )
        facts = tactical_facts(selection)
        model = None if args.no_llm else args.ollama_model
        briefing = facts if model is None else llm_briefing(selection, model, args.ollama_url, facts)
        if datetime.now(timezone.utc) >= deadline:
            raise ValueError("Gameweek deadline passed during this run; no new submission is permitted")
        directive = render_directive(selection, event, budget, briefing, model, account_note)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(directive + "\n", encoding="utf-8")
        if args.apply:
            from fpl_actions import get_fingerprint
            from fpl_execution import execute_once

            if client is None or private_payload is None or state is None or bank_after is None:
                raise RuntimeError("Authenticated account planning did not produce a valid executable state")
            payload = [
                {
                    "element_out": out.player_id, "element_in": incoming.player_id,
                    "purchase_price": incoming.cost,
                    "selling_price": state.selling_prices[out.player_id],
                }
                for out, incoming in transfers
            ]

            def submit() -> None:
                if payload:
                    client.execute_transfers(
                        event["id"], payload,
                        expected_squad={p.player_id for p in selection.squad},
                        expected_bank=bank_after,
                        source_fingerprint=get_fingerprint(private_payload["team"]),
                    )
                client.set_lineup(
                    api_lineup(selection), event["id"],
                    expected_squad={p.player_id for p in selection.squad},
                )

            execute_once(receipt, {
                "entry_id": args.entry_id, "gameweek": event["id"],
                "deadline_time": event["deadline_time"],
                "transfers": payload, "picks": api_lineup(selection),
            }, submit)
            directive += f"\nVERIFIED APPLIED: transfers and lineup confirmed. Receipt: {receipt}"
            args.output.write_text(directive + "\n", encoding="utf-8")
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        print(directive)
        print(f"Pre-deadline snapshot saved: {snapshot}", file=sys.stderr)
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
