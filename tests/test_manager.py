"""Isolated synthetic inputs test constraints; production always fetches live data."""

from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from itertools import combinations
import json
import unittest
from unittest.mock import Mock, patch

import requests

from fpl_manager import (
    Player, QUOTAS, api_lineup, engineer_players, fetch_json, llm_briefing, next_event,
    optimize, render_directive, season_from_bootstrap, tactical_facts,
)


def candidates():
    positions = [1] * 3 + [2] * 6 + [3] * 6 + [4] * 4
    return [
        Player(
            player_id=i, name=f"Player {i}", team_id=i % 7,
            club=f"Club {i % 7}", position=position,
            cost=45 + i % 7 * 5, baseline=float(i % 11),
            form=float(i % 5), xpts=float((i * 7) % 17 + 1),
            ownership=float(i), minutes=500, chance=None,
            fixture_ids=(i % 7,), opponents=(f"Opponent {i % 7}",),
            fixture_modifier=1.0,
        )
        for i, position in enumerate(positions, 1)
    ]


def bootstrap():
    teams = [
        {
            "id": i, "name": f"Club {i}", "short_name": f"C{i}", "strength": 3,
            **{
                f"strength_{kind}_{venue}": 1000
                for kind in ("overall", "attack", "defence")
                for venue in ("home", "away")
            },
        }
        for i in (1, 2)
    ]
    players = [
        {
            "id": i, "web_name": f"Player {i}", "team": 1, "element_type": 3,
            "now_cost": 50, "status": status,
            "chance_of_playing_next_round": chance, "ep_next": baseline,
            "form": "4.0", "selected_by_percent": "5.0", "minutes": 300,
        }
        for i, (status, chance, baseline) in enumerate(
            (("a", None, "6.0"), ("a", 75, None), ("a", 74, "6.0"),
             ("d", 100, "6.0"), ("i", None, "6.0"), ("a", 100, "6.0")), 1
        )
    ]
    return {
        "teams": teams, "elements": players,
        "events": [
            {"id": 1, "deadline_time": "2026-08-21T18:00:00Z", "is_current": True},
            {"id": 2, "deadline_time": "2026-08-28T18:00:00Z", "is_next": True},
        ],
    }


def fixtures():
    return [{"id": 1, "event": 2, "team_h": 1, "team_a": 2, "started": False}]


class FeatureTests(unittest.TestCase):
    def test_availability_cutoff_prices_and_null_baseline(self):
        result = engineer_players(bootstrap(), fixtures(), 2)
        self.assertEqual([p.player_id for p in result], [1, 2, 6])
        self.assertEqual(result[0].price, 5.0)
        self.assertEqual(result[0].xpts, 5.5)
        self.assertEqual(result[1].baseline, 0)
        self.assertEqual(result[1].xpts, 1)

    def test_missing_baseline_blank_and_double_gameweek(self):
        data = bootstrap()
        for row in data["elements"]:
            del row["ep_next"]
        self.assertEqual(engineer_players(data, fixtures(), 2)[0].baseline, 0)
        self.assertTrue(all(p.xpts == 0 for p in engineer_players(data, fixtures(), 3)))
        double = fixtures() + [{"id": 2, "event": 2, "team_h": 2, "team_a": 1}]
        result = engineer_players(bootstrap(), double, 2)[0]
        self.assertEqual(result.fixture_ids, (1, 2))
        self.assertEqual(result.xpts, 6.5)

    def test_inverse_opponent_strength_and_official_fallback(self):
        data = bootstrap()
        data["teams"][1]["strength_defence_away"] = 2000
        self.assertLess(engineer_players(data, fixtures(), 2)[0].xpts, 5.5)
        data["teams"][1]["strength_defence_away"] = 0
        self.assertEqual(engineer_players(data, fixtures(), 2)[0].xpts, 5.5)

    def test_stale_deadline_and_missing_next_event(self):
        data = bootstrap()
        self.assertEqual(next_event(data, datetime(2026, 8, 22, tzinfo=timezone.utc))["id"], 2)
        with self.assertRaisesRegex(ValueError, "passed"):
            next_event(data, datetime(2026, 8, 29, tzinfo=timezone.utc))
        data["events"][1]["is_next"] = False
        with self.assertRaisesRegex(ValueError, "upcoming"):
            next_event(data, datetime(2026, 8, 22, tzinfo=timezone.utc))
        self.assertEqual(season_from_bootstrap(data), "2026-27")

    def test_network_errors_propagate_without_retry_or_fallback(self):
        session = Mock()
        session.get.return_value.raise_for_status.side_effect = requests.HTTPError("403")
        with self.assertRaises(requests.HTTPError):
            fetch_json(session, "https://example.invalid")
        session.get.assert_called_once()
        session.get.return_value.json.assert_not_called()


class OptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.players = candidates()
        cls.result = optimize(cls.players, 900)

    def test_squad_global_optimum_against_exhaustive_search(self):
        feasible = [
            group for group in combinations(self.players, 15)
            if Counter(p.position for p in group) == QUOTAS
            and sum(p.cost for p in group) <= 900
            and max(Counter(p.team_id for p in group).values()) <= 3
        ]
        self.assertTrue(feasible)
        self.assertAlmostEqual(
            sum(p.xpts for p in self.result.squad),
            max(sum(p.xpts for p in group) for group in feasible),
        )

    def test_lineup_global_optimum_and_bench_hierarchy(self):
        feasible = []
        for group in combinations(self.result.squad, 11):
            count = Counter(p.position for p in group)
            if count[1] == 1 and 3 <= count[2] <= 5 and count[3] >= 2 and count[4] >= 1:
                feasible.append(sum(p.xpts for p in group))
        self.assertEqual(sum(p.xpts for p in self.result.starters), max(feasible))
        self.assertEqual(self.result.bench[-1].position, 1)
        values = [p.xpts for p in self.result.bench[:-1]]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_captain_and_vice_are_best_independent_starters(self):
        self.assertEqual(self.result.captain.xpts, max(p.xpts for p in self.result.starters))
        independent = [
            p for p in self.result.starters
            if set(p.fixture_ids).isdisjoint(self.result.captain.fixture_ids)
        ]
        self.assertEqual(self.result.vice_captain.xpts, max(p.xpts for p in independent))

    def test_submission_order_matches_fpl_goalkeeper_slot(self):
        picks = api_lineup(self.result)
        self.assertEqual([p["position"] for p in picks], list(range(1, 16)))
        self.assertEqual(picks[11]["element"], self.result.bench[-1].player_id)
        self.assertEqual(picks[11]["multiplier"], 0)
        self.assertEqual(sum(p["is_captain"] for p in picks), 1)
        self.assertEqual(sum(p["is_vice_captain"] for p in picks), 1)
        self.assertEqual(next(p for p in picks if p["is_captain"])["multiplier"], 2)

    def test_infeasibility_and_no_independent_vice_fail_explicitly(self):
        with self.assertRaisesRegex(RuntimeError, "optimal"):
            optimize(self.players, 1)
        with self.assertRaisesRegex(ValueError, "independent"):
            optimize([replace(p, fixture_ids=(1,)) for p in self.players], 900)

    def test_output_shape_and_explicit_llm_schema_error(self):
        facts = tactical_facts(self.result)
        directive = render_directive(
            self.result, {"id": 2, "deadline_time": "2026-08-28T18:00:00Z"}, 900, facts, None
        )
        self.assertIn("FPL AUTONOMOUS SQUAD DIRECTIVE: GW 2", directive)
        self.assertIn("BENCH HIERARCHY", directive)
        self.assertEqual(sum(line.startswith("[") for line in directive.splitlines()), 15)
        self.assertIn("--no-llm", directive)
        with patch("fpl_manager.requests.post") as post:
            post.return_value.json.return_value = {"message": {"content": '{"wrong":"shape"}'}}
            with self.assertRaisesRegex(ValueError, "invalid tactical"):
                llm_briefing(self.result, "model", "http://localhost:11434", facts)

    def test_llm_structured_output_and_truncation(self):
        facts = tactical_facts(self.result)
        with patch("fpl_manager.requests.post") as post:
            post.return_value.json.return_value = {
                "done_reason": "stop", "message": {"content": json.dumps(facts)}
            }
            self.assertEqual(
                llm_briefing(self.result, "model", "http://localhost:11434", facts), facts
            )
            payload = post.call_args.kwargs["json"]
            self.assertEqual(set(payload["format"]["required"]), set(facts))
            self.assertFalse(payload["format"]["additionalProperties"])
            post.return_value.json.return_value["done_reason"] = "length"
            with self.assertRaisesRegex(ValueError, "truncated"):
                llm_briefing(self.result, "model", "http://localhost:11434", facts)


if __name__ == "__main__":
    unittest.main()
