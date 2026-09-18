"""Distributed merge identity checks, independent of CUDA or live SSH."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('vast_v3_campaign',ROOT/'scripts/vast_v3_campaign.py')
campaign=importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)
from jlens_safety.common import atomic_json
from jlens_safety.protocol_v3 import fitting_conditions
from sae_jlens.run_io import append_jsonl, read_jsonl
import yaml


class CampaignTests(unittest.TestCase):
    def test_merge_requires_identical_provenance_and_complete_fits(self):
        c=yaml.safe_load((ROOT/'configs/jlens_safety_learned.yaml').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a',Path(tmp)/'b'
            for p in (a,b):
                p.mkdir()
                atomic_json(p/'manifest.json',dict(config=c))
                atomic_json(p/'layer_selection.json',dict(layer=15))
                for name in ('prompts.jsonl','runtime_environment.json','directions.npz','directions.json'):
                    (p/name).write_text('same fixture')
            expected=fitting_conditions(c,15)
            for index,p in enumerate((a,b)):
                atomic_json(p/f'learned_coefficients_shard_{index}.json',
                            [dict(row,coefficients=[0.]*32) for row in expected[index::2]])
            campaign.checked_merge(a,b,'fit-shard')
            merged=json.loads((a/'learned_coefficients.json').read_text())
            self.assertEqual([r['name'] for r in merged],[r['name'] for r in expected])
            atomic_json(b/'learned_coefficients_shard_1.json',[])
            with self.assertRaisesRegex(ValueError,'coverage'):
                campaign.checked_merge(a,b,'fit-shard')
            (b/'directions.npz').write_text('changed')
            with self.assertRaisesRegex(ValueError,'identity'):
                campaign.checked_merge(a,b,'fit-shard')

    def test_validation_merge_deduplicates_shared_screening_and_rejects_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a',Path(tmp)/'b'
            for p in (a,b):
                p.mkdir()
                for name in ('manifest.json','prompts.jsonl','runtime_environment.json',
                             'directions.npz','directions.json','layer_selection.json','learned_coefficients.json'):
                    (p/name).write_text('same fixture')
                for name in ('validation_generations.jsonl','validation_judgements.jsonl'):
                    append_jsonl(p/name,dict(id='shared',value=1))
            append_jsonl(b/'validation_generations.jsonl',dict(id='worker',value=2))
            campaign.checked_merge(a,b,'validate-shard')
            self.assertEqual(len(read_jsonl(a/'validation_generations.jsonl')),2)
            (b/'validation_generations.jsonl').write_text(json.dumps(dict(id='shared',value=999))+'\n')
            with self.assertRaisesRegex(ValueError,'Conflicting'):
                campaign.checked_merge(a,b,'validate-shard')


if __name__=='__main__': unittest.main()
