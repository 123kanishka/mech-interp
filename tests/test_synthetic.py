import unittest

import yaml

from sae_jlens.synthetic import generate_records


class SyntheticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("configs/sae_jlens.yaml", encoding="utf-8") as handle:
            cls.config = yaml.safe_load(handle)

    def test_deterministic(self):
        self.assertEqual(generate_records(self.config), generate_records(self.config))

    def test_expected_shape_and_trend(self):
        records = generate_records(self.config)
        self.assertEqual(len(records), 12 * 8 * 3)
        means = {}
        for layer in self.config["sae"]["layers"]:
            values = [
                row["js_similarity"]
                for row in records
                if row["layer"] == layer
            ]
            means[layer] = sum(values) / len(values)
        self.assertGreater(means[27], means[3])


if __name__ == "__main__":
    unittest.main()
