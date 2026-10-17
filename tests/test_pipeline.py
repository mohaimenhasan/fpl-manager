from contextlib import chdir, ExitStack, redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fpl_account import AccountState
from fpl_manager import main, optimize
from test_manager import candidates


class PipelineTests(unittest.TestCase):
    def test_apply_wires_prices_fingerprint_lineup_and_once_only_receipt(self):
        owned = list(optimize(candidates(), 900).squad)
        outgoing = owned[0]
        incoming = replace(
            outgoing, player_id=999, name="New player",
            cost=outgoing.cost + 1, xpts=outgoing.xpts + 1,
        )
        squad = [p if p != outgoing else incoming for p in owned]
        now = datetime.now(timezone.utc)
        deadline = (now + timedelta(hours=12)).isoformat()
        bootstrap = {
            "events": [{"id": 1, "is_next": True, "deadline_time": deadline}],
            "elements": [
                {"id": p.player_id, "web_name": p.name, "status": "a",
                 "chance_of_playing_next_round": None}
                for p in [*owned, incoming]
            ],
        }
        state = AccountState(
            123, "Team", 1, tuple(p.player_id for p in owned), 20,
            {p.player_id: p.cost for p in owned}, 1, True, (),
        )
        raw = {
            "picks": [{"element": p.player_id, "selling_price": p.cost} for p in owned],
            "transfers": {"bank": 20, "limit": 1, "made": 0},
        }
        with TemporaryDirectory() as directory, chdir(directory), ExitStack() as stack:
            stack.enter_context(redirect_stdout(StringIO()))
            stack.enter_context(patch("sys.argv", [
                "fpl_manager.py", "--entry-id", "123", "--apply", "--no-llm",
                "--output", "directive.txt",
            ]))
            stack.enter_context(patch("fpl_manager.fetch_live", return_value=(bootstrap, [], now)))
            stack.enter_context(patch("fpl_manager.engineer_players", return_value=[*owned, incoming]))
            stack.enter_context(patch("fpl_manager.save_snapshot", return_value=Path("snapshot.json")))
            stack.enter_context(patch("fpl_account.load_account", return_value=state))
            stack.enter_context(patch("fpl_account.project_owned_players", return_value=owned))
            stack.enter_context(patch("fpl_account.plan_transfers", return_value=(
                squad, [(outgoing, incoming)], 19,
            )))
            factory = stack.enter_context(patch("fpl_actions.FPLClient"))
            client = factory.return_value
            client.get_team_state.return_value = {
                "entry_id": 123, "gameweek": 1, "fetched_at": now.isoformat(), "team": raw,
            }
            main()
            factory.assert_called_once_with(
                entry_id=123, auth_file=Path("data") / "private" / "fpl-auth.json"
            )
            transfer_call = client.execute_transfers.call_args
            self.assertEqual(transfer_call.args, (1, [{
                "element_out": outgoing.player_id, "element_in": 999,
                "purchase_price": incoming.cost, "selling_price": outgoing.cost,
            }]))
            self.assertEqual(transfer_call.kwargs["expected_bank"], 19)
            self.assertEqual(transfer_call.kwargs["source_fingerprint"]["bank"], 20)
            picks, event = client.set_lineup.call_args.args
            self.assertEqual(event, 1)
            self.assertEqual(len(picks), 15)
            self.assertEqual(client.set_lineup.call_args.kwargs["expected_squad"], {p.player_id for p in squad})
            client.close.assert_called_once()
            receipt = next((Path("data") / "executions").glob("*.json"))
            self.assertEqual(json.loads(receipt.read_text())["status"], "complete")
            self.assertIn("VERIFIED APPLIED", Path("directive.txt").read_text(encoding="utf-8"))
            main()
            factory.assert_called_once()
            client.execute_transfers.assert_called_once()
            with patch("sys.argv", [
                "fpl_manager.py", "--entry-id", "123", "--authenticated", "--no-llm",
                "--refresh-auth", "--output", "preview.txt",
            ]), patch("connect_fpl.connect") as connect:
                main()
                connect.assert_called_once_with(
                    123, Path("data") / "private" / "fpl-auth.json", 90, "msedge", headless=True
                )
            self.assertEqual(factory.call_count, 2)
            client.execute_transfers.assert_called_once()
            client.set_lineup.assert_called_once()
            self.assertIn("Preview only", Path("preview.txt").read_text(encoding="utf-8"))
            self.assertNotIn("VERIFIED APPLIED", Path("preview.txt").read_text(encoding="utf-8"))

    def test_outside_deadline_window_does_not_load_auth_or_write(self):
        now = datetime.now(timezone.utc)
        bootstrap = {"events": [{
            "id": 8, "is_next": True,
            "deadline_time": (now + timedelta(days=6)).isoformat(),
        }]}
        with patch("sys.argv", [
            "fpl_manager.py", "--entry-id", "123", "--apply", "--within-hours", "36"
        ]), patch("fpl_manager.fetch_live", return_value=(bootstrap, [], now)), patch(
            "fpl_actions.FPLClient"
        ) as client, redirect_stdout(StringIO()) as output:
            main()
            client.assert_not_called()
            self.assertIn("no account changes", output.getvalue())


if __name__ == "__main__":
    unittest.main()
