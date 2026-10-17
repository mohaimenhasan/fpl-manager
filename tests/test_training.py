"""Offline training-pipeline checks; no model download, GPU or live API calls."""

import builtins
import copy
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import requests

from training.data import chronological_split, load_examples
from training.prompts import FORECAST_INSTRUCTION, SYSTEM_PROMPT, forecast_input, user_prompt


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREPARE = load_script("1_prepare_data")
TRAIN = load_script("2_train_lora")
EXPORT = load_script("3_export_gguf")
FORECAST = load_script("5_forecast")


def snapshot():
    return {
        "schema_version": 1,
        "season": "2026-27",
        "gameweek": 8,
        "fetched_at": "2026-10-17T12:00:00Z",
        "deadline_time": "2026-10-23T17:30:00Z",
        "bootstrap": {
            "events": [{"id": 8, "deadline_time": "2026-10-23T17:30:00Z"}],
            "elements": [
                {"id": pid, "web_name": f"Player {pid}", "team": 1,
                 "element_type": 2, "total_points": 10}
                for pid in range(1, 16)
            ],
            "teams": [{"id": 1, "name": "Club", "short_name": "CLB"}],
        },
        "fixtures": [{
            "id": 1, "event": 8, "team_h": 1, "team_a": 2,
            "team_h_score": 999, "stats": [{"identifier": "goals_scored"}],
        }],
        "squad_ids": list(range(1, 16)),
        "starter_ids": list(range(1, 12)),
        "captain_id": 1,
        "vice_captain_id": 2,
        "predictions": [
            {"player_id": pid, "xpts": 3.2, "price": 5.0}
            for pid in range(1, 16)
        ],
    }


def live_outcomes():
    return {"elements": [
        {"id": pid, "stats": {"total_points": pid}}
        for pid in range(1, 16)
    ]}


class SnapshotTests(unittest.TestCase):
    def test_valid_snapshot_and_season(self):
        self.assertIsNotNone(PREPARE.validate_snapshot(snapshot()))
        self.assertEqual(PREPARE.season_for_deadline("2027-05-30T12:00:00Z"), "2026-27")
        self.assertEqual(PREPARE.season_for_deadline("2026-08-01T12:00:00Z"), "2026-27")

    def test_deadline_boundary_and_naive_time_rejected(self):
        for stamp in (
            "2026-10-23T17:30:00Z", "2026-10-24T12:00:00Z", "2026-10-17T12:00:00",
        ):
            with self.subTest(stamp=stamp):
                item = snapshot()
                item["fetched_at"] = stamp
                with self.assertRaises(ValueError):
                    PREPARE.validate_snapshot(item)

    def test_invalid_contract_rejected(self):
        mutations = [
            lambda item: item.update(season="2025-26"),
            lambda item: item.update(squad_ids=[1] * 15),
            lambda item: item.update(starter_ids=[1] * 11),
            lambda item: item.update(captain_id=2),
            lambda item: item["predictions"].pop(),
            lambda item: item["predictions"][0].update(xpts=float("nan")),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                item = snapshot()
                mutation(item)
                with self.assertRaises(ValueError):
                    PREPARE.validate_snapshot(item)

    def test_realized_labels_never_replace_predeadline_features(self):
        item = PREPARE.make_example(snapshot(), live_outcomes())
        self.assertEqual(item["output"]["players"][0], {"player_id": 1, "points": 1})
        self.assertEqual(item["input"]["players"][0]["total_points"], 10)
        self.assertNotIn("team_h_score", item["input"]["fixtures"][0])
        self.assertNotIn("stats", item["input"]["fixtures"][0])
        self.assertEqual(item["metadata"]["label_type"], "realized_individual_points")
        self.assertEqual(item["metadata"]["outcome_summary"]["starter_raw_points"], 66)
        self.assertEqual(item["metadata"]["outcome_summary"]["squad_raw_points"], 120)

    def test_shared_inference_input_matches_training(self):
        archived = snapshot()
        record = PREPARE.make_example(archived, live_outcomes())
        self.assertEqual(record["input"], forecast_input(archived))
        self.assertEqual(record["instruction"], FORECAST_INSTRUCTION)
        prompt = user_prompt(record["instruction"], record["input"])
        self.assertIn("Future outcomes are unknown", SYSTEM_PROMPT)
        self.assertIn('"baseline_xpts": 3.2', prompt)
        self.assertNotIn("outcome_summary", prompt)
        self.assertNotIn('"label_type"', prompt)

    def test_requests_use_timeout_and_fail_fast(self):
        session = Mock()
        session.get.return_value.json.return_value = {"ok": True}
        self.assertEqual(PREPARE.fetch_json(session, "bootstrap-static", 12), {"ok": True})
        session.get.assert_called_once_with(
            "https://fantasy.premierleague.com/api/bootstrap-static/", timeout=(12, 12)
        )
        session.get.return_value.raise_for_status.side_effect = requests.HTTPError("503")
        with self.assertRaises(requests.HTTPError):
            PREPARE.fetch_json(session, "fixtures", 12)

    def run_preparation(
        self, *, finished=True, checked=True, fixture_finished=True, season="2026-27", existing=None
    ):
        current = {"events": [{
            "id": 8, "deadline_time": "2026-10-23T17:30:00Z",
            "finished": finished, "data_checked": checked,
        }]}
        archived = snapshot()
        if season != "2026-27":
            current["events"][0]["deadline_time"] = "2027-10-23T17:30:00Z"
        fixtures = [{"id": 1, "event": 8, "finished": fixture_finished}]
        session = Mock()
        session.__enter__ = Mock(return_value=session)
        session.__exit__ = Mock(return_value=False)
        stderr, stdout = io.StringIO(), io.StringIO()

        def read_dataset(path):
            if path.name == "training.jsonl":
                return existing
            with patch.object(Path, "open", return_value=io.StringIO(write.call_args.args[0])):
                return load_examples(path)

        with (
            patch("requests.Session", return_value=session),
            patch.object(Path, "glob", return_value=[Path("archived.json")]),
            patch.object(Path, "read_text", return_value=json.dumps(archived)),
            patch.object(Path, "exists", return_value=existing is not None),
            patch.object(PREPARE, "load_examples", side_effect=read_dataset),
            patch.object(PREPARE, "fetch_json", side_effect=[current, fixtures, live_outcomes()]) as fetch,
            patch.object(Path, "mkdir"),
            patch.object(Path, "write_text") as write,
            patch.object(Path, "replace") as replace,
            patch.object(Path, "unlink"),
            patch("sys.stderr", stderr),
            patch("sys.stdout", stdout),
        ):
            code = PREPARE.main([])
        return code, stderr.getvalue(), write, replace, fetch, session

    def test_completed_gameweek_written(self):
        code, _, write, replace, fetch, session = self.run_preparation()
        self.assertEqual(code, 0)
        self.assertEqual(fetch.call_count, 3)
        self.assertEqual(json.loads(write.call_args.args[0])["output"]["players"][0]["points"], 1)
        replace.assert_called_once()
        self.assertIn("Mozilla/5.0", session.headers.update.call_args.args[0]["User-Agent"])

    def test_pending_data_leaves_existing_dataset_untouched(self):
        for options in (
            {"finished": False}, {"checked": False}, {"fixture_finished": False},
        ):
            with self.subTest(options=options):
                code, message, write, replace, fetch, _ = self.run_preparation(**options)
                self.assertEqual(code, 1)
                self.assertIn("Pending 2026-27 GW8", message)
                self.assertIn("wait until", message)
                self.assertIn("left unchanged", message)
                write.assert_not_called()
                replace.assert_not_called()
                self.assertEqual(fetch.call_count, 2)

    def test_first_event_anchors_delayed_july_gameweeks(self):
                archived = snapshot()
                archived.update(
                    season="2019-20", gameweek=38,
                    fetched_at="2020-07-25T12:00:00Z", deadline_time="2020-07-26T14:00:00Z",
                )
                archived["bootstrap"]["events"] = [
                    {"id": 1, "deadline_time": "2019-08-09T18:00:00Z"},
                    {"id": 38, "deadline_time": archived["deadline_time"]},
                ]
                self.assertEqual(PREPARE.season_for_bootstrap(archived["bootstrap"]), "2019-20")
                PREPARE.validate_snapshot(archived)

    def test_rebuild_preserves_past_seasons_and_replaces_only_matching_key(self):
                old = PREPARE.make_example(snapshot(), live_outcomes())
                old["input"].update(
                    season="2025-26", as_of="2025-10-17T12:00:00Z",
                    deadline_time="2025-10-23T17:30:00Z",
                )
                old["metadata"].update(season="2025-26", snapshot_fetched_at=old["input"]["as_of"])
                previous_current = PREPARE.make_example(snapshot(), live_outcomes())
                previous_current["output"]["players"][0]["points"] = 999
                code, _, write, replace, _, _ = self.run_preparation(existing=[old, previous_current])
                self.assertEqual(code, 0)
                saved = [json.loads(line) for line in write.call_args.args[0].splitlines()]
                self.assertEqual(len(saved), 2)
                self.assertEqual(saved[0], old)
                self.assertEqual(saved[1]["output"]["players"][0]["points"], 1)
                replace.assert_called_once()

    def test_pending_week_preserves_accumulated_history(self):
                old = PREPARE.make_example(snapshot(), live_outcomes())
                code, _, write, replace, _, _ = self.run_preparation(finished=False, existing=[old])
                self.assertEqual(code, 1)
                write.assert_not_called()
                replace.assert_not_called()

    def test_invalid_existing_dataset_aborts_before_any_api_request_or_write(self):
                with (
                    patch.object(Path, "exists", return_value=True),
                    patch.object(PREPARE, "load_examples", side_effect=ValueError("invalid saved dataset")),
                    patch("requests.Session") as session,
                    patch.object(Path, "write_text") as write,
                    patch("sys.stderr", io.StringIO()),
                ):
                    self.assertEqual(PREPARE.main([]), 1)
                session.assert_not_called()
                write.assert_not_called()

    def test_old_season_cannot_be_joined_to_current_live_endpoint(self):
        code, message, write, _, fetch, _ = self.run_preparation(season="2027-28")
        self.assertEqual(code, 1)
        self.assertIn("official API is not that season", message)
        write.assert_not_called()
        self.assertEqual(fetch.call_count, 2)


class TrainingDataTests(unittest.TestCase):
    def load_text(self, text):
        with patch.object(Path, "open", return_value=io.StringIO(text)):
            return load_examples("in-memory.jsonl")

    def test_nonempty_example_and_matching_labels(self):
        record = PREPARE.make_example(snapshot(), live_outcomes())
        self.assertEqual(len(self.load_text(json.dumps(record) + "\n")), 1)

    def test_empty_duplicate_and_nonfinite_rejected(self):
        record = PREPARE.make_example(snapshot(), live_outcomes())
        with self.assertRaisesRegex(ValueError, "no nonempty"):
            self.load_text(" \n")
        with self.assertRaisesRegex(ValueError, "duplicate season/gameweek"):
            self.load_text((json.dumps(record) + "\n") * 2)
        bad = copy.deepcopy(record)
        bad["output"]["players"][0]["points"] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.load_text(json.dumps(bad))
        bad = copy.deepcopy(record)
        bad["output"]["players"].pop()
        with self.assertRaisesRegex(ValueError, "match"):
            self.load_text(json.dumps(bad))

    def test_chronological_split_holds_out_newest_entire_gameweeks(self):
        self.assertEqual(chronological_split(list(range(10)), 0.2), (list(range(8)), [8, 9]))
        self.assertEqual(chronological_split([1], 0.1), ([1], []))
        self.assertEqual(chronological_split([1, 2], 0), ([1, 2], []))

    def test_validate_only_needs_no_gpu_packages(self):
        record = PREPARE.make_example(snapshot(), live_outcomes())
        with (
            patch.object(Path, "open", return_value=io.StringIO(json.dumps(record))),
            patch("sys.stdout", io.StringIO()),
        ):
            self.assertEqual(TRAIN.main(["--validate-only"]), 0)

    def test_transformers_rejects_adapter_directory_before_loading_gpu_packages(self):
        record = PREPARE.make_example(snapshot(), live_outcomes())
        error = io.StringIO()
        with (
            patch.object(TRAIN, "load_examples", return_value=[record]),
            patch.object(Path, "is_file", return_value=True),
            patch("sys.stdout", io.StringIO()),
            patch("sys.stderr", error),
        ):
            self.assertEqual(TRAIN.main(["--backend", "transformers", "--model", "local-adapter"]), 1)
        self.assertIn("requires a base model", error.getvalue())


class ForecastTests(unittest.TestCase):
    def forecast(self):
        return {"players": [{"player_id": pid, "points": pid} for pid in range(1, 16)]}

    def test_forecast_exact_ids_and_order(self):
        result = self.forecast()
        result["players"].reverse()
        validated = FORECAST.validate_forecast(json.dumps(result), list(range(1, 16)))
        self.assertEqual(validated, self.forecast())

    def test_latest_snapshot_uses_valid_timestamps_not_filenames(self):
        old = snapshot()
        recent = snapshot()
        recent["fetched_at"] = "2026-10-18T12:00:00Z"
        invalid = snapshot()
        invalid["fetched_at"] = invalid["deadline_time"]
        files = {
            "zzz-old.json": old, "aaa-new.json": recent, "zzzz-invalid.json": invalid,
        }
        with (
            patch.object(Path, "glob", return_value=[Path(name) for name in files]),
            patch.object(Path, "read_text", autospec=True,
                         side_effect=lambda path, **kwargs: json.dumps(files[path.name])),
            patch("sys.stderr", io.StringIO()),
        ):
            path, chosen = FORECAST.latest_snapshot(Path("snapshots"))
        self.assertEqual(path.name, "aaa-new.json")
        self.assertEqual(chosen["fetched_at"], recent["fetched_at"])

    def test_forecast_rejects_duplicate_unknown_missing_or_noninteger_ids(self):
        mutations = [
            lambda item: item["players"].pop(),
            lambda item: item["players"].append({"player_id": 16, "points": 1}),
            lambda item: item["players"][0].update(player_id=2),
            lambda item: item["players"][0].update(player_id=999),
            lambda item: item["players"][0].update(player_id=True),
            lambda item: item["players"][0].update(player_id=1.0),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                forecast = self.forecast()
                mutation(forecast)
                with self.assertRaises(ValueError):
                    FORECAST.validate_forecast(json.dumps(forecast), list(range(1, 16)))

    def test_forecast_rejects_nonfinite_strings_booleans_and_extra_properties(self):
        for points in (float("nan"), float("inf"), -float("inf"), "4", True, None):
            with self.subTest(points=points):
                forecast = self.forecast()
                forecast["players"][0]["points"] = points
                with self.assertRaises(ValueError):
                    FORECAST.validate_forecast(json.dumps(forecast), list(range(1, 16)))
        for forecast in (
            {**self.forecast(), "explanation": "extra text"},
            {"players": [{**row, "note": "extra"} for row in self.forecast()["players"]]},
        ):
            with self.assertRaises(ValueError):
                FORECAST.validate_forecast(json.dumps(forecast), list(range(1, 16)))
        with self.assertRaises(ValueError):
            FORECAST.validate_forecast('{"players":[],"players":[]}', list(range(1, 16)))
        with self.assertRaises(ValueError):
            FORECAST.validate_forecast("```json\n" + json.dumps(self.forecast()) + "\n```", list(range(1, 16)))

    def test_matching_labels_score_mae_against_archived_baseline(self):
        archived = snapshot()
        record = PREPARE.make_example(archived, live_outcomes())
        metrics = FORECAST.evaluate(archived, self.forecast(), [record])
        self.assertEqual(metrics["status"], "scored")
        self.assertEqual(metrics["n_players"], 15)
        self.assertEqual(metrics["model_mae"], 0)
        expected_baseline = sum(abs(3.2 - pid) for pid in range(1, 16)) / 15
        self.assertAlmostEqual(metrics["baseline_mae"], expected_baseline)
        self.assertAlmostEqual(metrics["mae_improvement"], expected_baseline)

    def test_scoring_never_crosses_season_or_gameweek(self):
        for change in ({"season": "2025-26"}, {"gameweek": 7}):
            with self.subTest(change=change):
                labels = PREPARE.make_example(snapshot(), live_outcomes())
                labels["input"].update(change)
                evaluation = FORECAST.evaluate(snapshot(), self.forecast(), [labels])
                self.assertEqual(evaluation["status"], "unavailable")
                self.assertNotIn("model_mae", evaluation)

    def test_scoring_rejects_incomplete_player_overlap(self):
        labels = PREPARE.make_example(snapshot(), live_outcomes())
        labels["output"]["players"][0]["player_id"] = 999
        evaluation = FORECAST.evaluate(snapshot(), self.forecast(), [labels])
        self.assertEqual(evaluation["status"], "unavailable")
        self.assertIn("player IDs", evaluation["reason"])

    def test_ollama_uses_shared_prompt_and_structured_schema_not_outcomes(self):
        session = Mock()
        session.post.return_value.json.return_value = {
            "done": True, "response": json.dumps(self.forecast()),
        }
        result = FORECAST.request_forecast(session, snapshot(), "fpl-llama3", "http://localhost:11434/", 90)
        self.assertEqual(result, self.forecast())
        args, kwargs = session.post.call_args
        self.assertEqual(args[0], "http://localhost:11434/api/generate")
        self.assertEqual(kwargs["timeout"], (10, 90))
        self.assertEqual(kwargs["json"]["system"], SYSTEM_PROMPT)
        self.assertIn(FORECAST_INSTRUCTION, kwargs["json"]["prompt"])
        self.assertNotIn("outcome_summary", kwargs["json"]["prompt"])
        self.assertEqual(kwargs["json"]["format"]["properties"]["players"]["minItems"], 15)
        self.assertFalse(kwargs["json"]["stream"])

    def test_truncated_generation_fails_instead_of_persisting_partial_json(self):
        session = Mock()
        session.post.return_value.json.return_value = {
            "done": True, "done_reason": "length", "response": json.dumps(self.forecast()),
        }
        with self.assertRaisesRegex(ValueError, "incomplete"):
            FORECAST.request_forecast(session, snapshot(), "fpl-llama3", "http://localhost:11434", 90)

    def test_forecast_main_persists_standalone_report(self):
        session = Mock()
        session.__enter__ = Mock(return_value=session)
        session.__exit__ = Mock(return_value=False)
        with (
            patch.object(FORECAST, "latest_snapshot", return_value=(Path("archived.json"), snapshot())),
            patch.object(FORECAST, "request_forecast", return_value=self.forecast()),
            patch("requests.Session", return_value=session),
            patch.object(Path, "mkdir"),
            patch.object(Path, "write_text") as write,
            patch.object(Path, "replace") as replace,
            patch.object(Path, "unlink"),
            patch("sys.stdout", io.StringIO()),
            patch("sys.stderr", io.StringIO()),
        ):
            self.assertEqual(FORECAST.main([]), 0)
        report = json.loads(write.call_args.args[0])
        self.assertEqual(report["players"], self.forecast()["players"])
        self.assertFalse(report["optimizer_feedback"])
        self.assertEqual(report["evaluation"]["status"], "not_requested")
        replace.assert_called_once()

    def test_failed_generation_does_not_persist_report(self):
        session = Mock()
        session.__enter__ = Mock(return_value=session)
        session.__exit__ = Mock(return_value=False)
        with (
            patch.object(FORECAST, "latest_snapshot", return_value=(Path("archived.json"), snapshot())),
            patch.object(FORECAST, "request_forecast", side_effect=requests.ReadTimeout("offline")),
            patch("requests.Session", return_value=session),
            patch.object(Path, "write_text") as write,
            patch.object(Path, "replace") as replace,
            patch("sys.stderr", io.StringIO()),
        ):
            self.assertEqual(FORECAST.main([]), 1)
        write.assert_not_called()
        replace.assert_not_called()


class OptionalDependencyTests(unittest.TestCase):
    def test_help_does_not_import_gpu_or_training_libraries(self):
        real_import = builtins.__import__
        forbidden = {"torch", "unsloth", "transformers", "trl", "peft", "datasets", "accelerate", "bitsandbytes"}

        def guard(name, *args, **kwargs):
            if name.split(".", 1)[0] in forbidden:
                self.fail(f"--help imported optional library {name}")
            return real_import(name, *args, **kwargs)

        for script in (PREPARE, TRAIN, EXPORT, FORECAST):
            with self.subTest(script=script.__name__):
                with patch("builtins.__import__", side_effect=guard), patch("sys.stdout", io.StringIO()):
                    with self.assertRaises(SystemExit) as exit_context:
                        script.main(["--help"])
                    self.assertEqual(exit_context.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
