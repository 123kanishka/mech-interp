import tempfile
import unittest
from pathlib import Path

from sae_jlens.run_io import append_jsonl, atomic_json, read_jsonl


class RunIoTest(unittest.TestCase):
    def test_json_and_jsonl_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root / "value.json", {"status": "SUCCESS"})
            self.assertIn("SUCCESS", (root / "value.json").read_text())
            append_jsonl(root / "rows.jsonl", {"i": 1})
            append_jsonl(root / "rows.jsonl", {"i": 2})
            self.assertEqual(read_jsonl(root / "rows.jsonl"), [{"i": 1}, {"i": 2}])

    def test_corrupt_jsonl_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text('{"ok": 1}\nnot-json\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "rows.jsonl:2"):
                read_jsonl(path)
