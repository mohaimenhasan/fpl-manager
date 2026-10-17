import copy
from datetime import datetime, timedelta, timezone
import json
from io import StringIO
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import MagicMock, Mock, patch

import requests

from connect_fpl import connect, main
from fpl_actions import API_URL, FPLClient, get_fingerprint


ENTRY = 8346787


def bootstrap():
    positions = [1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 1, 2, 2, 3]
    elements = [
        {"id": i, "now_cost": 50, "element_type": pos, "team": (i - 1) // 3 + 1,
         "status": "a", "chance_of_playing_next_round": None}
        for i, pos in enumerate(positions, 1)
    ]
    elements.append({"id": 16, "now_cost": 55, "element_type": 3, "team": 6,
                     "status": "a", "chance_of_playing_next_round": None})
    return {
        "events": [{"id": 9, "deadline_time":
                    (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()}],
        "elements": elements,
    }


def team():
    return {
        "picks": [
            {"element": i, "position": i, "selling_price": 50, "purchase_price": 50,
             "is_captain": i == 9, "is_vice_captain": i == 10,
             "multiplier": 2 if i == 9 else (1 if i <= 11 else 0)}
            for i in range(1, 16)
        ],
        "transfers": {"bank": 10, "limit": 2, "made": 0},
        "chips": [],
    }


def response(payload=None, status=200):
    result = Mock(status_code=status, status=status)
    result.json.return_value = payload
    return result


class ActionsTests(unittest.TestCase):
    def setUp(self):
        self.auth = {"schema_version": 1, "entry_id": ENTRY,
                     "headers": {"cookie": "csrftoken=unit-test-placeholder",
                                 "origin": "https://untrusted.invalid"}}
        self.read = patch.object(Path, "read_text", return_value=json.dumps(self.auth))
        self.read.start()
        self.addCleanup(self.read.stop)
        self.session_patch = patch("fpl_actions.requests.Session")
        self.factory = self.session_patch.start()
        self.addCleanup(self.session_patch.stop)
        self.session = self.factory.return_value
        self.session.headers = {}
        self.client = FPLClient(ENTRY)
        self.before = team()
        self.after = copy.deepcopy(self.before)
        self.after["picks"][-1]["element"] = 16
        self.after["transfers"].update(bank=5, made=1)
        self.transfer = [{"element_in": 16, "element_out": 15,
                          "purchase_price": 55, "selling_price": 50}]

    def sequence(self, *items):
        self.session.request.side_effect = list(items)

    def execute(self):
        return self.client.execute_transfers(
            9, self.transfer, (set(range(1, 16)) - {15}) | {16},
            5, get_fingerprint(self.before),
        )

    def posts(self):
        return [c for c in self.session.request.call_args_list if c.args[0] == "POST"]

    def write_sequence(self, final=None):
        self.sequence(response({"player": {"entry": ENTRY}}), response(bootstrap()),
                      response(self.before), response(), response(final or self.after))

    def test_transfer_verified_and_official_only(self):
        self.write_sequence()
        self.assertEqual(self.execute(), self.after)
        self.assertEqual(len(self.posts()), 1)
        payload = self.posts()[0].kwargs["json"]
        self.assertFalse(payload["wildcard"])
        self.assertFalse(payload["freehit"])
        self.assertTrue(payload["confirmed"])
        for call in self.session.request.call_args_list:
            self.assertTrue(call.args[1].startswith(API_URL))
            self.assertFalse(call.kwargs["allow_redirects"])
        self.assertEqual(self.session.headers["x-csrftoken"], "unit-test-placeholder")
        self.assertEqual(self.session.headers["Origin"], API_URL.removesuffix("/api/"))
        self.assertFalse(self.session.trust_env)

    def test_wrong_owner_never_reads_or_writes_team(self):
        self.sequence(response({"player": {"entry": 1}}))
        with self.assertRaisesRegex(RuntimeError, "does not own"):
            self.execute()
        self.assertEqual(self.session.request.call_count, 1)

    def test_unknown_owner_fails_closed(self):
        self.sequence(response({"entry": ENTRY}))
        with self.assertRaisesRegex(RuntimeError, "ownership"):
            self.execute()
        self.assertFalse(self.posts())

    def test_expired_auth_is_sanitized(self):
        self.sequence(response({"private": "not-for-logs"}, 403))
        with self.assertRaisesRegex(RuntimeError, "rerun connect_fpl") as error:
            self.execute()
        self.assertNotIn("not-for-logs", str(error.exception))

    def test_deadline_passed(self):
        data = bootstrap()
        data["events"][0]["deadline_time"] = "2000-01-01T00:00:00Z"
        self.sequence(response({"player": {"entry": ENTRY}}), response(data))
        with self.assertRaisesRegex(RuntimeError, "deadline"):
            self.execute()
        self.assertFalse(self.posts())

    def test_wrong_event(self):
        data = bootstrap()
        data["events"][0]["id"] = 10
        self.sequence(response({"player": {"entry": ENTRY}}), response(data))
        with self.assertRaisesRegex(RuntimeError, "deadline"):
            self.execute()
        self.assertFalse(self.posts())

    def test_stale_team_fails_before_post(self):
        fresh = copy.deepcopy(self.before)
        fresh["transfers"]["bank"] = 20
        self.sequence(response({"player": {"entry": ENTRY}}), response(bootstrap()), response(fresh))
        with self.assertRaisesRegex(RuntimeError, "changed since planning"):
            self.execute()
        self.assertFalse(self.posts())

    def test_changed_price_fails_before_post(self):
        data = bootstrap()
        data["elements"][-1]["now_cost"] = 56
        self.sequence(response({"player": {"entry": ENTRY}}), response(data), response(self.before))
        with self.assertRaisesRegex(RuntimeError, "prices changed"):
            self.execute()
        self.assertFalse(self.posts())

    def test_changed_availability_fails_before_post(self):
        for status, chance in (("i", None), ("a", 74), ("a", float("nan")), ("a", 101)):
            with self.subTest(status=status, chance=chance):
                self.client._verified = False
                self.session.request.reset_mock()
                data = bootstrap()
                data["elements"][-1].update(status=status, chance_of_playing_next_round=chance)
                self.sequence(response({"player": {"entry": ENTRY}}), response(data), response(self.before))
                with self.assertRaisesRegex(RuntimeError, "availability changed"):
                    self.execute()
                self.assertFalse(self.posts())

    def test_points_hits_forbidden(self):
        self.before["transfers"]["made"] = 2
        self.write_sequence()
        with self.assertRaisesRegex(RuntimeError, "points hit"):
            self.execute()
        self.assertFalse(self.posts())

    def test_active_chip_forbidden(self):
        self.before["chips"] = [{"name": "wildcard", "status_for_entry": "active"}]
        self.write_sequence()
        with self.assertRaisesRegex(RuntimeError, "chip"):
            self.execute()
        self.assertFalse(self.posts())

    def test_duplicate_transfer_forbidden(self):
        self.transfer *= 2
        self.write_sequence()
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            self.execute()
        self.assertFalse(self.posts())

    def test_post_failure_no_retry(self):
        self.sequence(response({"player": {"entry": ENTRY}}), response(bootstrap()),
                      response(self.before), response({"private": "hidden"}, 500))
        with self.assertRaisesRegex(RuntimeError, "HTTP 500"):
            self.execute()
        self.assertEqual(len(self.posts()), 1)

    def test_authenticated_redirect_fails_without_following(self):
        for status in (301, 302, 303, 307, 308):
            with self.subTest(status=status):
                self.session.request.reset_mock()
                self.sequence(response(status=status))
                with self.assertRaisesRegex(RuntimeError, f"HTTP {status}"):
                    self.client.get_team_state(9)
                self.assertEqual(self.session.request.call_count, 1)
                self.assertFalse(self.session.request.call_args.kwargs["allow_redirects"])

    def test_mutating_redirect_fails_without_retry(self):
        self.sequence(response({"player": {"entry": ENTRY}}), response(bootstrap()),
                      response(self.before), response(status=307))
        with self.assertRaisesRegex(RuntimeError, "HTTP 307"):
            self.execute()
        self.assertEqual(len(self.posts()), 1)
        self.assertFalse(self.posts()[0].kwargs["allow_redirects"])

    def test_post_timeout_no_retry(self):
        self.sequence(response({"player": {"entry": ENTRY}}), response(bootstrap()),
                      response(self.before), requests.Timeout("sensitive request details"))
        with self.assertRaisesRegex(RuntimeError, "Outcome unknown") as error:
            self.execute()
        self.assertNotIn("sensitive", str(error.exception))
        self.assertEqual(len(self.posts()), 1)

    def test_http_success_without_saved_transfers_not_success(self):
        self.write_sequence(final=self.before)
        with self.assertRaisesRegex(RuntimeError, "outcome is unverified"):
            self.execute()
        self.assertEqual(len(self.posts()), 1)

    def test_verification_read_failure_is_unknown_outcome(self):
        self.sequence(response({"player": {"entry": ENTRY}}), response(bootstrap()),
                      response(self.before), response(), response(status=403))
        with self.assertRaisesRegex(RuntimeError, "Do not retry"):
            self.execute()
        self.assertEqual(len(self.posts()), 1)

    def lineup(self):
        return [{k: p[k] for k in ("element", "position", "is_captain",
                                  "is_vice_captain", "multiplier")} for p in self.before["picks"]]

    def test_lineup_is_verified(self):
        self.sequence(response({"player": {"entry": ENTRY}}), response(bootstrap()),
                      response(self.before), response(), response(self.before))
        self.client.set_lineup(self.lineup(), 9, set(range(1, 16)))
        self.assertEqual(len(self.posts()), 1)
        self.assertEqual(set(self.posts()[0].kwargs["json"]), {"picks"})

    def test_reserve_keeper_must_be_position_12(self):
        picks = self.lineup()
        picks[11]["position"], picks[14]["position"] = 15, 12
        self.write_sequence()
        with self.assertRaisesRegex(RuntimeError, "reserve goalkeeper"):
            self.client.set_lineup(picks, 9)
        self.assertFalse(self.posts())

    def test_lineup_verification_rejects_wrong_captain(self):
        saved = copy.deepcopy(self.before)
        saved["picks"][8]["is_captain"] = False
        self.sequence(response({"player": {"entry": ENTRY}}), response(bootstrap()),
                      response(self.before), response(), response(saved))
        with self.assertRaisesRegex(RuntimeError, "outcome is unverified"):
            self.client.set_lineup(self.lineup(), 9)
        self.assertEqual(len(self.posts()), 1)

    def test_lineup_chip_payload_rejected(self):
        picks = self.lineup()
        picks[0]["chip"] = "bboost"
        self.write_sequence()
        with self.assertRaisesRegex(RuntimeError, "Lineup must"):
            self.client.set_lineup(picks, 9)
        self.assertFalse(self.posts())

    def test_team_state_wrapper(self):
        self.sequence(response({"player": {"entry": ENTRY}}), response(self.before))
        wrapper = self.client.get_team_state(9)
        self.assertEqual(wrapper["team"], self.before)
        self.assertEqual(wrapper["entry_id"], ENTRY)
        self.assertEqual(wrapper["gameweek"], 9)
        self.assertTrue(wrapper["fetched_at"].endswith("Z"))

    def test_fingerprint_independent_of_lineup(self):
        reversed_team = copy.deepcopy(self.before)
        reversed_team["picks"].reverse()
        self.assertEqual(get_fingerprint(reversed_team), get_fingerprint(self.before))

    def test_bad_auth_never_creates_network_session(self):
        self.factory.reset_mock()
        with patch.object(Path, "read_text", return_value="not JSON"):
            with self.assertRaisesRegex(RuntimeError, "local FPL session"):
                FPLClient(ENTRY)
        self.factory.assert_not_called()

    def test_current_fpl_api_authorization_header_is_preserved(self):
        auth = {
            "schema_version": 1, "entry_id": ENTRY,
            "headers": {"X-API-Authorization": "unit-test-placeholder"},
        }
        with patch.object(Path, "read_text", return_value=json.dumps(auth)):
            client = FPLClient(ENTRY)
        self.assertEqual(client.session.headers["x-api-authorization"], "unit-test-placeholder")

    def test_unknown_endpoint_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Unsupported"):
            self.client._request("GET", "https://untrusted.invalid/")
        self.session.request.assert_not_called()

    def test_context_manager_closes_even_on_error(self):
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with self.client as client:
                self.assertIs(client, self.client)
                raise RuntimeError("test failure")
        self.session.close.assert_called_once()

    def test_help_does_not_require_playwright(self):
        with patch("sys.stdout", new=StringIO()), self.assertRaises(SystemExit) as result:
            main(["--help"])
        self.assertEqual(result.exception.code, 0)


class ConnectorTests(unittest.TestCase):
    def browser(self, url=None, payload=None):
        module = ModuleType("playwright.sync_api")
        module.Error = type("BrowserError", (Exception,), {})
        module.sync_playwright = MagicMock()
        playwright = module.sync_playwright.return_value.__enter__.return_value
        context = playwright.chromium.launch_persistent_context.return_value
        page = Mock()
        context.pages = [page]
        listener = {}
        context.on.side_effect = lambda event, callback: listener.update(callback=callback)
        observed = Mock(url=url or API_URL + f"my-team/{ENTRY}/", status=200)
        observed.json.return_value = payload if payload is not None else team()
        observed.request.method = "GET"
        observed.request.all_headers.return_value = {
            "cookie": "unit-test-placeholder", "authorization": "unit-test-placeholder",
            "user-agent": "test-agent", "x-csrftoken": "unit-test-placeholder",
            "x-private-personal-detail": "must-not-save",
        }
        page.goto.side_effect = lambda *a, **k: listener["callback"](observed)
        context.request.get.return_value = response({"player": {"entry": ENTRY}})
        return module, context, observed, playwright

    def connect_with(self, module):
        return patch.dict(sys.modules, {"playwright": ModuleType("playwright"),
                                        "playwright.sync_api": module})

    def test_captures_only_allowlisted_headers_and_verifies_owner(self):
        module, context, _, playwright = self.browser()
        with self.connect_with(module), patch.object(Path, "mkdir"), \
                patch("connect_fpl.restrict_permissions"), patch("connect_fpl.save_auth") as save, \
                patch("sys.stdout", new=StringIO()) as stdout:
            connect(ENTRY, Path("test-auth.json"), 1, "chromium")
        saved_headers = save.call_args.args[2]
        self.assertNotIn("x-private-personal-detail", saved_headers)
        self.assertEqual(set(saved_headers),
                         {"cookie", "authorization", "x-csrftoken", "user-agent"})
        self.assertNotIn("unit-test-placeholder", stdout.getvalue())
        self.assertEqual(context.request.get.call_args.kwargs["max_redirects"], 0)
        context.close.assert_called_once()
        self.assertNotIn("channel", playwright.chromium.launch_persistent_context.call_args.kwargs)

    def test_browser_wrong_owner_never_saves(self):
        module, context, _, _ = self.browser()
        context.request.get.return_value = response({"player": {"entry": 1}})
        with self.connect_with(module), patch.object(Path, "mkdir"), \
                patch("connect_fpl.restrict_permissions"), patch("connect_fpl.save_auth") as save, \
                patch("sys.stdout", new=StringIO()):
            with self.assertRaisesRegex(RuntimeError, "does not own"):
                connect(ENTRY, Path("test-auth.json"), 1, "msedge")
        save.assert_not_called()
        context.close.assert_called_once()

    def test_current_fpl_api_header_alone_can_authenticate_capture(self):
        module, context, observed, _ = self.browser()
        observed.request.all_headers.return_value = {
            "x-api-authorization": "unit-test-placeholder", "user-agent": "test-agent",
        }
        with self.connect_with(module), patch.object(Path, "mkdir"), \
                patch("connect_fpl.restrict_permissions"), patch("connect_fpl.save_auth") as save, \
                patch("sys.stdout", new=StringIO()):
            connect(ENTRY, Path("test-auth.json"), 1, "msedge")
        self.assertEqual(save.call_args.args[2]["x-api-authorization"], "unit-test-placeholder")
        self.assertEqual(
            context.request.get.call_args.kwargs["headers"]["x-api-authorization"], "unit-test-placeholder"
        )

    def test_headless_refresh_uses_saved_profile_without_interactive_prompts(self):
        module, context, _, playwright = self.browser()
        with self.connect_with(module), patch.object(Path, "mkdir"), \
                patch("connect_fpl.restrict_permissions"), patch("connect_fpl.save_auth"), \
                patch("sys.stdout", new=StringIO()) as output:
            connect(ENTRY, Path("test-auth.json"), 1, "msedge", headless=True)
        self.assertTrue(playwright.chromium.launch_persistent_context.call_args.kwargs["headless"])
        self.assertIn("Refreshing", output.getvalue())
        self.assertNotIn("Sign in yourself", output.getvalue())
        context.close.assert_called_once()

    def test_callback_json_error_is_surfaced(self):
        module, context, observed, _ = self.browser()
        observed.json.side_effect = ValueError("private response must not appear")
        with self.connect_with(module), patch.object(Path, "mkdir"), \
                patch("connect_fpl.restrict_permissions"), patch("connect_fpl.save_auth") as save, \
                patch("sys.stdout", new=StringIO()):
            with self.assertRaisesRegex(RuntimeError, "Cannot read") as error:
                connect(ENTRY, Path("test-auth.json"), 1, "msedge")
        self.assertNotIn("private response", str(error.exception))
        save.assert_not_called()
        context.close.assert_called_once()

    def test_other_origin_is_ignored_and_times_out(self):
        module, context, observed, _ = self.browser(
            url=f"https://untrusted.invalid/api/my-team/{ENTRY}/",
        )
        with self.connect_with(module), patch.object(Path, "mkdir"), \
                patch("connect_fpl.restrict_permissions"), patch("connect_fpl.save_auth") as save, \
                patch("connect_fpl.time.monotonic", side_effect=[0, 2]), \
                patch("sys.stdout", new=StringIO()):
            with self.assertRaisesRegex(RuntimeError, "Timed out"):
                connect(ENTRY, Path("test-auth.json"), 1, "msedge")
        observed.request.all_headers.assert_not_called()
        save.assert_not_called()
        context.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
