"""Synthetic account snapshots only; production account reads remain live."""

from collections import Counter
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from itertools import combinations
import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, mock_open, patch

import requests

from fpl_account import (
    AccountState, load_account, plan_transfers, project_owned_players,
)
from fpl_manager import API_URL, QUOTAS, USER_AGENT, Player


NOW = datetime(2026, 10, 17, 12, tzinfo=timezone.utc)
POSITIONS = [1] * 2 + [2] * 5 + [3] * 5 + [4] * 3


def bootstrap():
    return {
        "events": [
            {"id": 6, "deadline_time": "2026-10-03T10:00:00Z", "finished": True},
            {"id": 7, "deadline_time": "2026-10-10T10:00:00Z", "is_current": True},
            {"id": 8, "deadline_time": "2026-10-18T10:00:00Z", "is_next": True},
        ],
        "teams": [
            {
                "id": i, "name": f"Club {i}", "short_name": f"C{i}", "strength": 3,
                **{
                    f"strength_{kind}_{venue}": 1000
                    for kind in ("overall", "attack", "defence")
                    for venue in ("home", "away")
                },
            }
            for i in range(1, 7)
        ],
        "elements": [
            {
                "id": i, "web_name": f"Player {i}", "team": (i - 1) % 5 + 1,
                "element_type": position, "now_cost": 50, "status": "a",
                "chance_of_playing_next_round": None, "ep_next": "5",
                "form": "4", "selected_by_percent": "10", "minutes": 600,
            }
            for i, position in enumerate(POSITIONS, 1)
        ],
    }


def snapshot():
    return {
        "entry_id": 8346787, "gameweek": 8,
        "fetched_at": "2026-10-17T11:00:00Z",
        "team": {
            "picks": [{"element": i, "selling_price": 45} for i in range(1, 16)],
            "transfers": {"bank": 10, "limit": 3, "made": 1},
            "chips": [{"name": "freehit", "status_for_entry": "available"}],
        },
    }


def owned_players():
    return [
        Player(
            player_id=i, name=f"Player {i}", team_id=(i - 1) % 5 + 1,
            club=f"C{(i - 1) % 5 + 1}", position=position,
            cost=50, baseline=5, form=4, xpts=5, ownership=10, minutes=600,
            chance=None, fixture_ids=(i,), opponents=("Opponent",),
            fixture_modifier=1,
        )
        for i, position in enumerate(POSITIONS, 1)
    ]


def state(**changes):
    original = AccountState(
        8346787, "kanes-cursor", 8, tuple(range(1, 16)), 10,
        {i: 45 for i in range(1, 16)}, 2, True, (),
    )
    return replace(original, **changes)


class AccountLoadingTests(unittest.TestCase):
    def setUp(self):
        self.data = bootstrap()
        self.public = {
            "picks": [{"element": i} for i in range(1, 16)],
            "active_chip": None,
        }
        self.session = MagicMock()
        self.session.headers = {}
        session_factory = self.enterContext(patch("fpl_account.requests.Session"))
        session_factory.return_value.__enter__.return_value = self.session
        self.fetch = self.enterContext(patch("fpl_account.fetch_json"))
        self.fetch.side_effect = [
            {"name": "kanes-cursor", "last_deadline_bank": 17, "player_first_name": "PII"},
            self.public,
        ]
        self.clock = self.enterContext(patch("fpl_account.datetime", wraps=datetime))
        self.clock.now.return_value = NOW

    def load_private(self, payload=None):
        payload = snapshot() if payload is None else payload
        self.fetch.side_effect = [{"name": "kanes-cursor"}]
        with patch("pathlib.Path.open", mock_open(read_data=json.dumps(payload))) as opened:
            result = load_account(self.data, 8346787, Path("private-state.json"))
        opened.assert_called_once_with(encoding="utf-8")
        return result

    def test_public_current_picks_are_retained_but_never_verified(self):
        result = load_account(self.data, 8346787)
        self.assertEqual(result.name, "kanes-cursor")
        self.assertEqual(result.source_event, 7)
        self.assertEqual(result.squad_ids, tuple(range(1, 16)))
        self.assertEqual(result.bank, 17)
        self.assertIsNone(result.selling_prices)
        self.assertIsNone(result.free_transfers)
        self.assertFalse(result.verified)
        self.assertIn("miss transfers already made for GW8", " ".join(result.warnings))
        self.assertIn("last-deadline", " ".join(result.warnings))
        self.assertNotIn("PII", repr(result))
        self.assertEqual(self.session.headers["User-Agent"], USER_AGENT)
        self.assertEqual(
            [call.args[1] for call in self.fetch.call_args_list],
            [f"{API_URL}/entry/8346787/", f"{API_URL}/entry/8346787/event/7/picks/"],
        )

    def test_public_previous_event_when_current_is_absent(self):
        self.data["events"][1].pop("is_current")
        self.assertEqual(load_account(self.data, 8346787).source_event, 7)

    def test_public_bank_unknown_is_not_guessed(self):
        self.fetch.side_effect = [{"name": "kanes-cursor"}, self.public]
        self.assertIsNone(load_account(self.data, 8346787).bank)

    def test_public_free_hit_fails_instead_of_claiming_permanent_squad(self):
        self.public["active_chip"] = "freehit"
        with self.assertRaisesRegex(ValueError, "temporary.*private"):
            load_account(self.data, 8346787)

    def test_no_published_event_requires_private_state(self):
        self.data["events"] = [self.data["events"][-1]]
        with self.assertRaisesRegex(ValueError, "private"):
            load_account(self.data, 8346787)

    def test_public_invalid_picks_and_position_quotas(self):
        cases = [
            self.public["picks"][:-1],
            self.public["picks"][:-1] + [{"element": 1}],
            self.public["picks"][:-1] + [{"element": 999}],
            self.public["picks"][:-1] + [{"element": True}],
        ]
        for picks in cases:
            with self.subTest(picks=picks):
                self.fetch.side_effect = [
                    {"name": "kanes-cursor"}, {"picks": picks},
                ]
                with self.assertRaises(ValueError):
                    load_account(self.data, 8346787)
        self.data["elements"][0]["element_type"] = 2
        self.fetch.side_effect = [{"name": "kanes-cursor"}, self.public]
        with self.assertRaisesRegex(ValueError, "quotas"):
            load_account(self.data, 8346787)

    def test_private_raw_state_and_transfer_arithmetic(self):
        result = self.load_private()
        self.assertTrue(result.verified)
        self.assertEqual(result.source_event, 8)
        self.assertEqual(result.bank, 10)
        self.assertEqual(result.free_transfers, 2)
        self.assertEqual(result.selling_prices, {i: 45 for i in range(1, 16)})
        self.assertEqual(result.warnings, ())
        self.assertEqual(self.fetch.call_count, 1)
        payload = snapshot()
        payload["team"]["transfers"].update(limit=1, made=3)
        self.assertEqual(self.load_private(payload).free_transfers, 0)

    def test_live_private_wrapper_uses_same_validation_without_local_file(self):
        payload = snapshot()
        original = deepcopy(payload)
        self.fetch.side_effect = [{"name": "kanes-cursor"}]
        with patch("pathlib.Path.open") as opened, patch("fpl_account.requests.Session") as session:
            result = load_account(self.data, 8346787, private_payload=payload)
        opened.assert_not_called()
        session.assert_not_called()
        self.assertTrue(result.verified)
        self.assertEqual(result.name, "Entry 8346787")
        self.assertEqual(result.free_transfers, 2)
        self.assertEqual(result.selling_prices, {i: 45 for i in range(1, 16)})
        self.assertEqual(payload, original)
        self.fetch.assert_not_called()
        self.session.post.assert_not_called()

    def test_live_private_wrapper_accepts_optional_team_name_and_positional_payload(self):
        payload = snapshot()
        payload["name"] = "kanes-cursor"
        result = load_account(self.data, 8346787, None, payload)
        self.assertEqual(result.name, "kanes-cursor")
        self.fetch.assert_not_called()

    def test_live_private_wrapper_rejects_stale_unlimited_or_mismatched_state(self):
        payloads = []
        for key, value in (
            ("entry_id", 1), ("gameweek", 7),
            ("fetched_at", "2026-10-15T11:00:00Z"),
        ):
            payload = snapshot()
            payload[key] = value
            payloads.append(payload)
        unlimited = snapshot()
        unlimited["team"]["transfers"]["limit"] = None
        payloads.append(unlimited)
        for payload in payloads:
            self.fetch.side_effect = [{"name": "kanes-cursor"}]
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                load_account(self.data, 8346787, private_payload=payload)

    def test_private_state_file_and_live_payload_are_mutually_exclusive(self):
        with patch("pathlib.Path.open") as opened:
            with self.assertRaisesRegex(ValueError, "either"):
                load_account(
                    self.data, 8346787, Path("private-state.json"),
                    private_payload=snapshot(),
                )
        opened.assert_not_called()
        self.fetch.assert_not_called()

    def test_null_private_state_file_never_falls_back_to_public(self):
        with patch("pathlib.Path.open", mock_open(read_data="null")):
            with self.assertRaisesRegex(ValueError, "wrapper"):
                load_account(self.data, 8346787, Path("private-state.json"))
        self.fetch.assert_not_called()

    def test_private_identity_and_gameweek_must_match(self):
        for key, value in (("entry_id", 1), ("entry_id", "8346787"), ("gameweek", 7)):
            with self.subTest(key=key, value=value):
                payload = snapshot()
                payload[key] = value
                with self.assertRaises(ValueError):
                    self.load_private(payload)

    def test_private_capture_age_and_utc_are_strict(self):
        for captured in (
            (NOW - timedelta(hours=24, seconds=1)).isoformat(),
            (NOW + timedelta(seconds=1)).isoformat(),
            "2026-10-17T11:00:00", "2026-10-17T12:00:00+01:00", "not-a-date", None,
        ):
            with self.subTest(captured=captured):
                payload = snapshot()
                payload["fetched_at"] = captured
                with self.assertRaises(ValueError):
                    self.load_private(payload)
        payload = snapshot()
        payload["fetched_at"] = (NOW - timedelta(hours=24)).isoformat()
        self.assertTrue(self.load_private(payload).verified)

    def test_passed_upcoming_deadline_is_not_accepted(self):
        self.data["events"][-1]["deadline_time"] = NOW.isoformat()
        with self.assertRaisesRegex(ValueError, "passed"):
            self.load_private()

    def test_private_prices_require_exact_nonnegative_integers(self):
        for price in (-1, 51, 45.0, "45", None, True):
            with self.subTest(price=price):
                payload = snapshot()
                payload["team"]["picks"][0]["selling_price"] = price
                with self.assertRaises(ValueError):
                    self.load_private(payload)
        payload = snapshot()
        payload["team"]["picks"][0]["selling_price"] = 0
        self.assertEqual(self.load_private(payload).selling_prices[1], 0)

    def test_private_unlimited_and_malformed_transfers_rejected(self):
        for field, value in (
            ("limit", None), ("limit", 999), ("limit", -1), ("limit", 3.0),
            ("made", -1), ("made", True), ("bank", -1), ("bank", "10"),
            ("status", "unlimited"),
        ):
            with self.subTest(field=field, value=value):
                payload = snapshot()
                payload["team"]["transfers"][field] = value
                with self.assertRaises(ValueError):
                    self.load_private(payload)
        for missing in ("team", "fetched_at", "gameweek"):
            payload = snapshot()
            del payload[missing]
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                self.load_private(payload)

    def test_private_active_chips_rejected_but_chip_list_optional(self):
        for change in (
            {"active_chip": "freehit"},
            {"chips": [{"name": "wildcard", "status_for_entry": "active"}]},
            {"chips": [{"name": "freehit", "is_pending": True}]},
        ):
            payload = snapshot()
            payload["team"].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "chip"):
                self.load_private(payload)
        payload = snapshot()
        del payload["team"]["chips"]
        self.assertTrue(self.load_private(payload).verified)

    def test_network_failure_has_timeout_browser_agent_and_no_fallback(self):
        with patch("fpl_account.fetch_json", wraps=__import__("fpl_manager").fetch_json):
            self.session.get.return_value.raise_for_status.side_effect = requests.HTTPError("403")
            with self.assertRaises(requests.HTTPError):
                load_account(self.data, 8346787)
        self.session.get.assert_called_once_with(
            f"{API_URL}/entry/8346787/", timeout=(10, 45),
        )
        self.assertEqual(self.session.headers["User-Agent"], USER_AGENT)
        self.session.post.assert_not_called()

    def test_invalid_entry_and_snapshot_shape(self):
        for entry_id in (0, -1, "8346787", True):
            with self.subTest(entry_id=entry_id), self.assertRaises(ValueError):
                load_account(self.data, entry_id)
        for payload in ([], None, {"picks": []}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.load_private(payload if payload is not None else [])


class OwnedProjectionTests(unittest.TestCase):
    def test_preserves_unavailable_owned_players_without_mutating_bootstrap(self):
        data = bootstrap()
        data["elements"][0].update(status="i", chance_of_playing_next_round=0)
        data["elements"][1].update(status="d", chance_of_playing_next_round=50)
        data["elements"][2].update(status="a", chance_of_playing_next_round=74)
        original = deepcopy(data)
        fixtures = [
            {"id": 1, "event": 8, "team_h": 1, "team_a": 2},
            {"id": 2, "event": 8, "team_h": 3, "team_a": 4},
            {"id": 3, "event": 8, "team_h": 5, "team_a": 6},
        ]
        result = project_owned_players(data, fixtures, 8, state())
        self.assertEqual([player.player_id for player in result], list(range(1, 16)))
        self.assertEqual([player.xpts for player in result[:3]], [0, 0, 0])
        self.assertEqual([player.chance for player in result[:3]], [0, 50, 74])
        self.assertEqual(result[0].baseline, 5)
        self.assertEqual(result[0].fixture_ids, (1,))
        self.assertGreater(result[3].xpts, 0)
        self.assertEqual(data, original)

    def test_unavailable_unknown_chance_remains_unknown(self):
        data = bootstrap()
        data["elements"][0]["status"] = "u"
        result = project_owned_players(data, [], 8, state())
        self.assertIsNone(result[0].chance)
        self.assertEqual(result[0].xpts, 0)

    def test_owned_player_must_be_in_current_registry(self):
        with self.assertRaises(ValueError):
            project_owned_players(bootstrap(), [], 8, state(squad_ids=tuple(range(2, 17))))


class TransferPlanningTests(unittest.TestCase):
    def setUp(self):
        self.owned = owned_players()
        self.candidate = replace(
            self.owned[0], player_id=16, name="New goalkeeper",
            team_id=6, club="C6", cost=55, xpts=10,
        )

    def test_exact_sale_price_finances_purchase_and_does_not_use_market_value(self):
        squad, pairs, bank_after = plan_transfers([self.candidate], self.owned, state())
        self.assertEqual(len(squad), 15)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][1], self.candidate)
        self.assertEqual(pairs[0][0].position, self.candidate.position)
        self.assertEqual(bank_after, 0)
        _, pairs, bank_after = plan_transfers(
            [replace(self.candidate, cost=56)], self.owned, state(),
        )
        self.assertEqual(pairs, [])
        self.assertEqual(bank_after, 10)

    def test_total_squad_projection_is_primary_objective(self):
        midfielder = replace(
            self.owned[7], player_id=17, team_id=6, club="C6", cost=45, xpts=20,
        )
        squad, pairs, bank_after = plan_transfers(
            [self.candidate, midfielder], self.owned, state(), max_transfers=1,
        )
        self.assertEqual([incoming.player_id for _, incoming in pairs], [17])
        self.assertEqual(sum(player.xpts for player in squad), 90)
        self.assertEqual(bank_after, 10)

    def test_secondary_objective_avoids_equal_projection_transfers(self):
        squad, pairs, bank_after = plan_transfers(
            [replace(self.candidate, xpts=5, cost=1)], self.owned, state(),
        )
        self.assertEqual({player.player_id for player in squad}, set(range(1, 16)))
        self.assertEqual(pairs, [])
        self.assertEqual(bank_after, 10)

    def test_very_small_real_improvement_is_not_replaced_by_transfer_penalty(self):
        _, pairs, _ = plan_transfers(
            [replace(self.candidate, xpts=5.0001)], self.owned, state(),
        )
        self.assertEqual(len(pairs), 1)

    def test_free_transfer_and_explicit_caps_never_create_hits(self):
        for free, maximum in ((0, 2), (2, 0), (1, 5)):
            with self.subTest(free=free, maximum=maximum):
                second = replace(
                    self.owned[7], player_id=17, team_id=6, cost=45, xpts=20,
                )
                _, pairs, _ = plan_transfers(
                    [self.candidate, second], self.owned,
                    state(free_transfers=free), maximum,
                )
                self.assertEqual(len(pairs), min(free, maximum, 2))

    def test_two_transfers_are_same_position_pairs_and_exact_budget(self):
        midfielder = replace(
            self.owned[7], player_id=17, team_id=6, club="C6", cost=45, xpts=20,
        )
        squad, pairs, bank_after = plan_transfers(
            [self.candidate, midfielder], self.owned, state(), 2,
        )
        self.assertEqual(len(pairs), 2)
        self.assertEqual(
            [(outgoing.position, incoming.position) for outgoing, incoming in pairs],
            [(1, 1), (3, 3)],
        )
        self.assertEqual(bank_after, 0)
        self.assertEqual(Counter(player.position for player in squad), QUOTAS)
        self.assertLessEqual(max(Counter(player.team_id for player in squad).values()), 3)

    def test_global_primary_optimum_matches_exhaustive_legal_transfers(self):
        new = [
            self.candidate,
            replace(self.owned[2], player_id=17, team_id=6, cost=35, xpts=4),
            replace(self.owned[7], player_id=18, team_id=6, cost=65, xpts=20),
            replace(self.owned[12], player_id=19, team_id=6, cost=50, xpts=12),
        ]
        owned_ids = set(range(1, 16))
        account = state(bank=5)
        feasible = []
        for group in combinations(self.owned + new, 15):
            group_ids = {player.player_id for player in group}
            incoming = [player for player in group if player.player_id not in owned_ids]
            outgoing = owned_ids - group_ids
            if (
                Counter(player.position for player in group) == QUOTAS
                and max(Counter(player.team_id for player in group).values()) <= 3
                and len(incoming) <= 2
                and sum(player.cost for player in incoming) <= 5 + 45 * len(outgoing)
            ):
                feasible.append((sum(player.xpts for player in group), len(incoming)))
        optimum = max(score for score, _ in feasible)
        minimum_transfers = min(count for score, count in feasible if score == optimum)
        squad, pairs, _ = plan_transfers(new, self.owned, account, 2)
        self.assertAlmostEqual(sum(player.xpts for player in squad), optimum)
        self.assertEqual(len(pairs), minimum_transfers)

    def test_club_limit_is_enforced(self):
        candidate = replace(self.candidate, position=2, team_id=1, cost=45, xpts=20)
        _, pairs, _ = plan_transfers([candidate], self.owned, state())
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0].team_id, 1)

    def test_unavailable_owned_retained_and_only_provided_candidates_acquired(self):
        owned = [replace(self.owned[0], xpts=0, chance=0), *self.owned[1:]]
        squad, pairs, _ = plan_transfers([], owned, state())
        self.assertEqual(pairs, [])
        self.assertEqual(next(player for player in squad if player.player_id == 1).xpts, 0)
        self.assertEqual({player.player_id for player in squad}, set(range(1, 16)))
        stale_candidate = replace(self.owned[0], xpts=99)
        _, pairs, _ = plan_transfers([stale_candidate], owned, state())
        self.assertEqual(pairs, [])

    def test_all_zero_projections_and_no_candidates_still_hold(self):
        squad, pairs, bank_after = plan_transfers(
            [], [replace(player, xpts=0) for player in self.owned], state(),
        )
        self.assertEqual(len(squad), 15)
        self.assertEqual(pairs, [])
        self.assertEqual(bank_after, 10)

    def test_unverified_and_incomplete_exact_state_rejected(self):
        for account in (
            state(verified=False), state(bank=None), state(selling_prices=None),
            state(free_transfers=None), state(free_transfers=6),
            state(bank=True), state(selling_prices={1: 45}),
            state(squad_ids=tuple(range(2, 17))),
            state(selling_prices={i: 51 for i in range(1, 16)}),
        ):
            with self.subTest(account=account), self.assertRaises(ValueError):
                plan_transfers([self.candidate], self.owned, account)

    def test_invalid_candidate_and_transfer_limits_rejected(self):
        for maximum in (-1, 1.5, True):
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                plan_transfers([self.candidate], self.owned, state(), maximum)
        for candidate in (
            replace(self.candidate, cost=0), replace(self.candidate, xpts=float("nan")),
            replace(self.candidate, xpts=-1), replace(self.candidate, position=9),
        ):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                plan_transfers([candidate], self.owned, state())
        with self.assertRaisesRegex(ValueError, "unique"):
            plan_transfers([self.candidate, self.candidate], self.owned, state())

    def test_infeasible_ownership_surfaces_solver_error(self):
        owned = [replace(player, team_id=1) for player in self.owned]
        with self.assertRaisesRegex(RuntimeError, "optimal"):
            plan_transfers([], owned, state(), 0)


if __name__ == "__main__":
    unittest.main()
