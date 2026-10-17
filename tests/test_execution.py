import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from fpl_execution import execute_once, previous_execution


class ReceiptTests(unittest.TestCase):
    def test_successful_submission_is_durable_and_not_repeated(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            operation = Mock()
            self.assertIsNone(previous_execution(path))
            execute_once(path, {"entry_id": 123, "gameweek": 8}, operation)
            self.assertEqual(previous_execution(path)["status"], "complete")
            operation.assert_called_once()
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                execute_once(path, {}, operation)
            operation.assert_called_once()

    def test_pending_intent_exists_before_mutation_and_survives_failure(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"

            def fail():
                self.assertEqual(json.loads(path.read_text())["status"], "pending")
                raise RuntimeError("Ambiguous network failure after POST")

            with self.assertRaisesRegex(RuntimeError, "Ambiguous"):
                execute_once(path, {}, fail)
            with self.assertRaisesRegex(RuntimeError, "automatic retries"):
                previous_execution(path)

    def test_invalid_receipt_fails_closed(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_text('{"status":"unknown"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid execution"):
                previous_execution(path)


if __name__ == "__main__":
    unittest.main()
