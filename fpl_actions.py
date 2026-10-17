"""Fail-closed official FPL account actions using a locally captured session."""

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.cookies import CookieError, SimpleCookie
import json
import math
from pathlib import Path

import requests

ORIGIN = "https://fantasy.premierleague.com"
API_URL = ORIGIN + "/api/"
DEFAULT_AUTH_FILE = Path("data") / "private" / "fpl-auth.json"
AUTH_HEADERS = frozenset({
    "cookie", "authorization", "x-api-authorization", "x-csrftoken", "user-agent",
})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def owner_entry(payload: dict) -> int:
    player = payload.get("player")
    entry = player.get("entry") if isinstance(player, dict) else None
    if type(entry) is not int or entry <= 0:
        raise RuntimeError("Cannot verify FPL account ownership; reconnect locally.")
    return entry


def _integer(value, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RuntimeError(f"Invalid {label} in FPL state.")
    return value


def get_fingerprint(team: dict) -> dict:
    """Stable planning precondition, independent of lineup order."""
    if not isinstance(team, dict):
        raise RuntimeError("FPL returned an invalid private team.")
    picks = team.get("picks")
    if not isinstance(picks, list) or len(picks) != 15:
        raise RuntimeError("FPL private team must contain exactly 15 picks.")
    if any(not isinstance(p, dict) or not {"element", "selling_price"} <= p.keys() for p in picks):
        raise RuntimeError("FPL private picks are missing player IDs or selling prices.")
    roster = sorted(
        [{"element": _integer(p["element"], "element", 1),
          "selling_price": _integer(p["selling_price"], "selling price")} for p in picks],
        key=lambda p: p["element"],
    )
    if len({p["element"] for p in roster}) != 15:
        raise RuntimeError("FPL private team contains duplicate players.")
    transfers = team.get("transfers", {})
    if not isinstance(transfers, dict):
        raise RuntimeError("FPL private team has invalid transfer state.")
    return {
        "roster": roster,
        "bank": _integer(transfers.get("bank"), "bank"),
        "limit": _integer(transfers.get("limit"), "free transfer limit"),
        "made": _integer(transfers.get("made"), "transfers made"),
    }


@dataclass
class FPLClient:
    entry_id: int
    auth_file: Path = DEFAULT_AUTH_FILE
    timeout: int = 30
    session: requests.Session = field(init=False, repr=False)
    _verified: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        _integer(self.entry_id, "entry ID", 1)
        _integer(self.timeout, "request timeout", 1)
        try:
            auth = json.loads(Path(self.auth_file).read_text(encoding="utf-8"))
            if auth.get("schema_version") != 1 or auth.get("entry_id") != self.entry_id:
                raise ValueError()
            saved = auth["headers"]
            if not isinstance(saved, dict):
                raise ValueError()
            headers = {}
            for name, value in saved.items():
                if name.lower() not in AUTH_HEADERS:
                    continue
                if not isinstance(value, str) or "\r" in value or "\n" in value:
                    raise ValueError()
                headers[name.lower()] = value
            if not any(headers.get(key) for key in ("cookie", "authorization", "x-api-authorization")):
                raise ValueError()
            if not headers.get("x-csrftoken") and headers.get("cookie"):
                cookies = SimpleCookie()
                cookies.load(headers["cookie"])
                if "csrftoken" in cookies:
                    headers["x-csrftoken"] = cookies["csrftoken"].value
        except (OSError, ValueError, TypeError, KeyError, AttributeError, CookieError):
            raise RuntimeError(
                "Missing or invalid local FPL session; run connect_fpl.py for this entry."
            ) from None
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers.update(headers)
        self.session.headers.update({
            "Origin": ORIGIN, "Referer": ORIGIN + "/my-team",
            "Accept": "application/json",
        })

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def _request(self, method: str, endpoint: str, payload=None):
        # Only fixed relative endpoints are used; redirects must never forward credentials.
        if endpoint not in {"me/", "bootstrap-static/", "transfers/",
                            f"my-team/{self.entry_id}/"}:
            raise RuntimeError("Unsupported FPL endpoint.")
        try:
            response = self.session.request(
                method, API_URL + endpoint, json=payload,
                timeout=self.timeout, allow_redirects=False,
            )
        except requests.RequestException:
            suffix = " Outcome unknown; inspect your team before any retry." if method == "POST" else ""
            raise RuntimeError("FPL request failed." + suffix) from None
        if response.status_code in (401, 403):
            raise RuntimeError("FPL session expired or denied; rerun connect_fpl.py.")
        if not 200 <= response.status_code < 300:
            suffix = " Inspect your team before any retry." if method == "POST" else ""
            raise RuntimeError(f"FPL request returned HTTP {response.status_code}." + suffix)
        if method == "POST":
            return None
        try:
            result = response.json()
        except ValueError:
            raise RuntimeError("FPL returned invalid JSON.") from None
        if not isinstance(result, dict):
            raise RuntimeError("FPL returned an unexpected response.")
        return result

    def verify_owner(self):
        if not self._verified:
            if owner_entry(self._request("GET", "me/")) != self.entry_id:
                raise RuntimeError("Signed-in FPL account does not own the requested entry.")
            self._verified = True

    def _team(self) -> dict:
        self.verify_owner()
        team = self._request("GET", f"my-team/{self.entry_id}/")
        get_fingerprint(team)
        return team

    def get_team_state(self, event: int) -> dict:
        _integer(event, "gameweek", 1)
        return {"entry_id": self.entry_id, "gameweek": event,
                "fetched_at": utc_now(), "team": self._team()}

    def _bootstrap(self, event: int) -> dict:
        _integer(event, "gameweek", 1)
        bootstrap = self._request("GET", "bootstrap-static/")
        self._check_deadline(bootstrap, event)
        return bootstrap

    @staticmethod
    def _check_deadline(bootstrap: dict, event: int):
        try:
            future = [
                e for e in bootstrap["events"]
                if datetime.fromisoformat(e["deadline_time"].replace("Z", "+00:00"))
                > datetime.now(timezone.utc)
            ]
            target = min(future, key=lambda e: e["deadline_time"])
            if target["id"] != event:
                raise ValueError()
        except (KeyError, TypeError, ValueError):
            raise RuntimeError("Gameweek deadline has passed or target is not the next deadline.") from None

    @staticmethod
    def _no_active_chip(team: dict):
        if team.get("active_chip") or any(
            c.get("status_for_entry") == "active" or c.get("is_pending")
            for c in team.get("chips", [])
        ):
            raise RuntimeError("Active or pending chip detected; automatic actions are disabled.")

    def execute_transfers(self, event: int, transfers: list[dict],
                          expected_squad: set[int], expected_bank: int,
                          source_fingerprint: dict) -> dict:
        self.verify_owner()
        bootstrap = self._bootstrap(event)
        team = self._team()
        fingerprint = get_fingerprint(team)
        if fingerprint != source_fingerprint:
            raise RuntimeError("Private team changed since planning; regenerate the plan.")
        self._no_active_chip(team)
        if not transfers:
            raise RuntimeError("No transfers supplied.")
        if len(transfers) > max(0, fingerprint["limit"] - fingerprint["made"]):
            raise RuntimeError("Transfers would incur a points hit; automatic hits are forbidden.")
        required = {"element_in", "element_out", "purchase_price", "selling_price"}
        if any(set(t) != required for t in transfers):
            raise RuntimeError("Transfer payload must contain only player IDs and exact prices.")
        for transfer in transfers:
            for key, value in transfer.items():
                _integer(value, key, 1 if key.startswith("element_") else 0)
        outgoing = {t["element_out"] for t in transfers}
        incoming = {t["element_in"] for t in transfers}
        roster = {p["element"]: p["selling_price"] for p in fingerprint["roster"]}
        if (len(outgoing) != len(transfers) or len(incoming) != len(transfers)
                or outgoing & incoming or not outgoing <= roster.keys()
                or incoming & roster.keys()):
            raise RuntimeError("Invalid or duplicate incoming/outgoing players.")
        elements = {p["id"]: p for p in bootstrap.get("elements", [])}
        for t in transfers:
            if (t["selling_price"] != roster[t["element_out"]]
                    or t["element_in"] not in elements
                    or t["purchase_price"] != elements[t["element_in"]]["now_cost"]):
                raise RuntimeError("Player prices changed; regenerate the transfer plan.")
            incoming_player = elements[t["element_in"]]
            chance = incoming_player.get("chance_of_playing_next_round")
            if incoming_player.get("status") != "a" or (
                chance is not None and (
                    type(chance) not in (int, float) or not math.isfinite(chance)
                    or not 75 <= chance <= 100
                )
            ):
                raise RuntimeError("Incoming player availability changed; regenerate the transfer plan.")
        squad = (set(roster) - outgoing) | incoming
        bank = fingerprint["bank"] + sum(t["selling_price"] - t["purchase_price"] for t in transfers)
        if squad != set(expected_squad) or bank != expected_bank or bank < 0:
            raise RuntimeError("Expected squad or bank does not match the transfer payload.")
        try:
            types = Counter(elements[p]["element_type"] for p in squad)
            clubs = Counter(elements[p]["team"] for p in squad)
            if types != Counter({1: 2, 2: 5, 3: 5, 4: 3}) or max(clubs.values()) > 3:
                raise ValueError()
        except (KeyError, ValueError):
            raise RuntimeError("Transfer squad violates FPL position or club limits.") from None
        self._check_deadline(bootstrap, event)
        self._request("POST", "transfers/", {
            "entry": self.entry_id, "event": event, "transfers": transfers,
            "wildcard": False, "freehit": False, "confirmed": True,
        })
        try:
            saved = self._team()
            actual = get_fingerprint(saved)
            if ({p["element"] for p in actual["roster"]} != squad or actual["bank"] != bank):
                raise RuntimeError()
        except (RuntimeError, KeyError, TypeError):
            raise RuntimeError(
                "Transfer submission outcome is unverified; inspect FPL manually. Do not retry automatically."
            ) from None
        return saved

    def set_lineup(self, picks: list[dict], event: int,
                   expected_squad: set[int] | None = None) -> dict:
        self.verify_owner()
        bootstrap = self._bootstrap(event)
        team = self._team()
        self._no_active_chip(team)
        roster = {p["element"] for p in team["picks"]}
        required = {"element", "position", "is_captain", "is_vice_captain", "multiplier"}
        if (len(picks) != 15 or any(set(p) != required for p in picks)
                or {p["position"] for p in picks} != set(range(1, 16))
                or {p["element"] for p in picks} != roster
                or (expected_squad is not None and set(expected_squad) != roster)):
            raise RuntimeError("Lineup must contain the current 15 players in unique positions.")
        for p in picks:
            _integer(p["element"], "element", 1)
            _integer(p["position"], "position", 1)
            _integer(p["multiplier"], "multiplier")
            if type(p["is_captain"]) is not bool or type(p["is_vice_captain"]) is not bool:
                raise RuntimeError("Captain flags must be booleans.")
            if (p["is_captain"] or p["is_vice_captain"]) and p["position"] > 11:
                raise RuntimeError("Captain and vice-captain must start.")
            multiplier = 2 if p["is_captain"] else (1 if p["position"] <= 11 else 0)
            if p["multiplier"] != multiplier or (p["is_captain"] and p["is_vice_captain"]):
                raise RuntimeError("Invalid lineup multiplier or captain flags.")
        if sum(p["is_captain"] for p in picks) != 1 or sum(p["is_vice_captain"] for p in picks) != 1:
            raise RuntimeError("Choose exactly one captain and a different vice-captain.")
        elements = {p["id"]: p for p in bootstrap.get("elements", [])}
        try:
            ordered = sorted(picks, key=lambda p: p["position"])
            types = Counter(elements[p["element"]]["element_type"] for p in ordered[:11])
            if (elements[ordered[0]["element"]]["element_type"] != 1
                    or elements[ordered[11]["element"]]["element_type"] != 1
                    or types[1] != 1 or not 3 <= types[2] <= 5
                    or not 2 <= types[3] <= 5 or not 1 <= types[4] <= 3):
                raise ValueError()
        except (KeyError, ValueError):
            raise RuntimeError("Invalid lineup formation; reserve goalkeeper must be position 12.") from None
        self._check_deadline(bootstrap, event)
        self._request("POST", f"my-team/{self.entry_id}/", {"picks": ordered})
        try:
            saved = self._team()
            signature = lambda ps: sorted(
                (p["element"], p["position"], p["is_captain"], p["is_vice_captain"])
                for p in ps
            )
            if signature(saved["picks"]) != signature(ordered):
                raise RuntimeError()
        except (RuntimeError, KeyError, TypeError):
            raise RuntimeError(
                "Lineup submission outcome is unverified; inspect FPL manually. Do not retry automatically."
            ) from None
        return saved
