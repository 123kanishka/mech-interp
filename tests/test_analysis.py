import unittest

import yaml

from sae_jlens.analysis import summarise_records
from sae_jlens.synthetic import generate_records


class AnalysisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("configs/sae_jlens.yaml", encoding="utf-8") as handle:
            cls.config = yaml.safe_load(handle)

    def test_bootstrap_uses_sequences(self):
        records = generate_records(self.config)
        summary = summarise_records(records, self.config)
        self.assertEqual(summary["bootstrap_unit"], "sequence")
        self.assertEqual(summary["n_sequences"], 12)
        js_rows = [
            row for row in summary["estimates"] if row["metric"] == "js_similarity"
        ]
        self.assertEqual(len(js_rows), 3)
        self.assertTrue(all(row["n_sequences"] == 12 for row in js_rows))
