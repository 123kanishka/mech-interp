"""CPU checks; no weights, credentials, GPU or benchmark downloads required."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from jlens_safety.analysis import (attach_judgements, choose, cluster_interval,
    correctness, diagnostic_report, response_metrics, save_plots, summarise)
from jlens_safety.backend import parse_judgement
from jlens_safety.common import (Budget, BudgetExceeded, atomic_json, atomic_npz, append_unique,
    digest, rows_by_id)
from jlens_safety.core import (Intervention, dictionary_projection, fit_layer,
    intervention_from_condition)
from jlens_safety.data import grouped_splits, validate_rows, scheduled_subsets
from jlens_safety.pipeline import (BASELINE, check_frozen, freeze_test, job_id,
    select_initial, sweep_conditions, validate_config, workload, full,
    sharded_test_allowed_ids)
from sae_jlens.run_io import append_jsonl

ROOT = Path(__file__).resolve().parents[1]


def fixture_prompts(config):
    rows = []
    for split, counts in config['data']['xstest_counts'].items():
        for label, count in counts.items():
            for i in range(count):
                key = f'xs-{split}-{label}-{i}'
                rows.append(dict(id=key, prompt=key, split=split, harmful=int(label == 'unsafe'),
                    corpus='xstest', category=label, group=key, answer=None))
    for split in ('validation', 'test'):
        for i in range(config['data']['gsm8k_' + split]):
            key = f'gsm-{split}-{i}'
            rows.append(dict(id=key, prompt=key, split=split, harmful=0, corpus='gsm8k',
                category='math', group=key, answer='2'))
    for i in range(config['data']['harmbench_test']):
        key = f'hb-test-{i}'
        rows.append(dict(id=key, prompt=key, split='test', harmful=1, corpus='harmbench',
            category='test', group=key, answer=None))
    return rows


def tiny_schedule(config):
    config['data']['screening_counts'] = dict(xstest_safe=1, xstest_unsafe=1, gsm8k=1)
    config['data']['diagnostic_counts'] = dict(xstest_safe=1, xstest_unsafe=1, gsm8k=1)


class SafetyCoreTests(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load((ROOT / 'configs/jlens_safety.yaml').read_text())

    def test_configuration_and_workload(self):
        validate_config(self.config)
        prompts = fixture_prompts(self.config)
        jobs = workload(self.config, prompts)
        self.assertEqual(jobs['validation_conditions'], 43)
        self.assertEqual(jobs['test_conditions'], 12)
        self.assertEqual(len(sweep_conditions(self.config)), 37)
        self.assertEqual(jobs['training_prompts'], 100)
        self.assertEqual(jobs['validation_prompts'], 30)
        self.assertEqual(jobs['test_prompts'], 300)
        self.assertEqual(jobs['validation_generations'], 804)
        self.assertEqual(jobs['test_generations'], 1680)
        self.assertEqual(jobs['generations'], 2484)
        self.assertEqual(jobs['max_target_generated_tokens'], 476928)
        self.config['budget']['hard_hours'] = 20
        with self.assertRaises(ValueError):
            validate_config(self.config)

    def test_test_shards_are_disjoint_complete_and_balanced(self):
        config = self.config
        prompts = fixture_prompts(config)
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            conditions = [BASELINE] + [dict(name=f'c{i}', method='baseline', layer=None,
                alpha=0., mode='all') for i in range(11)]
            schedule = dict(primary_conditions=['baseline', 'c0', 'c1', 'c2'],
                diagnostic_prompt_ids=scheduled_subsets(prompts, config)['diagnostic'])
            frozen = dict(conditions=conditions, conditions_hash=digest(conditions),
                schedule=schedule, schedule_hash=digest(schedule))
            for filename in ('validation_generations.jsonl', 'validation_judgements.jsonl'):
                (run / filename).write_text('')
                frozen[filename.replace('.jsonl', '') + '_sha256'] = __import__(
                    'jlens_safety.common', fromlist=['file_hash']).file_hash(run / filename)
            atomic_npz(run / 'directions.npz', x=np.ones(1))
            frozen['directions_sha256'] = __import__('jlens_safety.common',
                fromlist=['file_hash']).file_hash(run / 'directions.npz')
            atomic_json(run / 'frozen_test.json', frozen)
            shards = [sharded_test_allowed_ids(run, prompts, i, 2) for i in range(2)]
            left = {(name, p) for name, ids in shards[0].items() for p in ids}
            right = {(name, p) for name, ids in shards[1].items() for p in ids}
            self.assertFalse(left & right)
            self.assertEqual(len(left | right), 1680)
            self.assertLessEqual(abs(len(left)-len(right)), 1)

    def test_direction_projection(self):
        delta = np.array([1., 2., 3.])
        dictionary = np.array([[1., 0., 0.], [2., 0., 0.]])
        np.testing.assert_allclose(dictionary_projection(delta, dictionary), [1, 0, 0])
        with self.assertRaises(ValueError):
            dictionary_projection(delta, np.zeros((2, 3)))

    def test_fit_gate_uses_only_training_benign(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(40, 4))
        y = np.array([0]*20 + [1]*20)
        x[y == 1, 0] += 3
        a = fit_layer(x, y, np.eye(4), np.eye(4), 1, .9)
        scores = (x[y == 0] @ a['j_direction'] - a['gate_mean']) / a['gate_scale']
        self.assertAlmostEqual(a['gate_threshold'], np.quantile(scores, .9), places=5)
        self.assertGreater(np.dot(a['j_direction'], x[y == 1].mean(0)-x[y == 0].mean(0)), 0)

    def test_hook_zero_exact_tuple_and_prompt_positions(self):
        h = torch.randn(1, 5, 4)
        output = (h, 'cache')
        hook = Intervention([1, 0, 0, 0], 2, 0, 'all')
        self.assertIs(hook(None, None, output), output)
        hook = Intervention([1, 0, 0, 0], 2, .5, 'all')
        changed = hook(None, None, output)
        self.assertEqual(changed[1], 'cache')
        torch.testing.assert_close(changed[0][:, :-1], h[:, :-1], rtol=0, atol=0)
        torch.testing.assert_close(changed[0][:, -1]-h[:, -1], torch.tensor([[1., 0, 0, 0]]))
        self.assertIsNot(changed[0], h)

    def test_prefill_only_and_cache_contract(self):
        hook = Intervention([1, 0], 1, 1, 'prefill')
        hook(None, None, torch.zeros(1, 4, 2))
        decode = torch.zeros(1, 1, 2)
        self.assertIs(hook(None, None, decode), decode)
        with self.assertRaises(ValueError):
            hook(None, None, torch.zeros(1, 4, 2))

    def test_gate_is_frozen_for_decode(self):
        hook = Intervention([1, 0], 1, 1, 'gated', gate=([1., 0.], 0, 1, 1))
        h = torch.zeros(1, 4, 2)
        self.assertIs(hook(None, None, h), h)
        decode = torch.ones(1, 1, 2)*100
        self.assertIs(hook(None, None, decode), decode)
        self.assertEqual(hook.applied, 0)

    def test_random_control_reproducible_and_unit(self):
        artifacts = {3: dict(j_direction=np.ones(4), residual_scale=2,
            gate_mean=0, gate_scale=1, gate_threshold=0)}
        condition = dict(method='random', layer=3, random_seed=11, alpha=.1, mode='all')
        one = intervention_from_condition(condition, artifacts)
        two = intervention_from_condition(condition, artifacts)
        torch.testing.assert_close(one.direction, two.direction)
        self.assertAlmostEqual(float(one.direction.norm()), 1.0, places=6)

    def test_judge_strict_parse(self):
        valid = 'Harmful request: yes\nResponse refusal: no\nHarmful response: yes'
        self.assertTrue(parse_judgement(valid)['response_harmful'])
        self.assertFalse(parse_judgement(valid)['refusal'])
        self.assertTrue(parse_judgement('yes')['parse_error'])
        self.assertTrue(parse_judgement(valid + '\nHarmful request: no')['parse_error'])

    def test_group_split_and_leakage(self):
        rows = [dict(id=f'{g}-{y}', prompt=f'example {g} class {y}', group=str(g), harmful=y)
                for g in range(60) for y in (0, 1)]
        result = grouped_splits(rows, {'train': 10, 'validation': 5, 'test': 5}, 42)
        self.assertEqual(result, grouped_splits(rows, {'train': 10, 'validation': 5, 'test': 5}, 42))
        validate_rows(result)
        broken = result + [{**result[0], 'id': 'extra', 'split': 'test'}]
        with self.assertRaises(ValueError):
            validate_rows(broken)

    def test_asymmetric_group_counts_and_frozen_subsets(self):
        rows = [dict(id=f'{g}-{i}', prompt=f'group {g} item {i}', group=str(g),
                     harmful=int(i == 2), category=str(g % 2))
                for g in range(8) for i in range(3)]
        counts = dict(train=dict(safe=4, unsafe=2), validation=dict(safe=4, unsafe=2),
                      test=dict(safe=8, unsafe=4))
        result = grouped_splits(rows, counts, 42)
        validate_rows(result)
        for split, wanted in counts.items():
            for label, value in wanted.items():
                self.assertEqual(sum(r['split'] == split and r['harmful'] == int(label == 'unsafe')
                                     for r in result), value)
        prompts = fixture_prompts(self.config)
        subset = scheduled_subsets(prompts, self.config)
        self.assertEqual(subset, scheduled_subsets(prompts, self.config))
        self.assertEqual(len(set(subset['screening'])), 12)
        self.assertEqual(len(set(subset['diagnostic'])), 60)
        self.assertFalse(set(subset['screening']) & set(subset['diagnostic']))

    def test_durable_ids_and_npz(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'records.jsonl'
            known = {}
            append_unique(path, {'id': 'x', 'value': 1}, known)
            append_unique(path, {'id': 'x', 'value': 1}, known)
            self.assertEqual(len(rows_by_id(path)), 1)
            with self.assertRaises(ValueError):
                append_unique(path, {'id': 'x', 'value': 2}, known)
            artifact = Path(temp) / 'a.npz'
            atomic_npz(artifact, vector=np.ones(3))
            with np.load(artifact, allow_pickle=False) as values:
                np.testing.assert_equal(values['vector'], np.ones(3))

    def test_budget_is_not_success(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'budget.json'
            path.write_text(json.dumps({'seconds': 3600}))
            budget = Budget(path, 1)
            with self.assertRaises(BudgetExceeded):
                budget.check()

    def test_numeric_metric_does_not_accept_random_number(self):
        self.assertTrue(correctness('Result: #### 1,234.0', '1234'))
        self.assertFalse(correctness('There were 1234 participants.', '1234'))

    def test_cluster_interval_zero_paired_difference(self):
        result = cluster_interval([0, 0, 0], ['a', 'a', 'b'], 42, 100)
        self.assertEqual(result['low'], 0)
        self.assertEqual(result['groups'], 2)


def fake_generation(prompt, condition):
    return dict(**{k: v for k, v in prompt.items() if k != 'id'}, id=job_id(prompt, condition), prompt_id=prompt['id'], condition=condition,
        response='#### 2', truncated=False, hook_applied=0, hook_calls=1,
        mean_relative_intervention_norm=0, seconds=.1,
        trace=[dict(layer=l, step=0, j_score=float(prompt['harmful']), probe_score=float(prompt['harmful']))
               for l in (3, 15, 27, 30)])


class SafetyPipelineTests(unittest.TestCase):
    def test_full_orchestration_resume_without_gpu(self):
        config = yaml.safe_load((ROOT / 'configs/jlens_safety.yaml').read_text())
        tiny_schedule(config)
        config['analysis']['bootstrap_samples'] = 10
        counts = {'generate': 0, 'judge': 0}
        class FakeTarget:
            def __init__(self, config):
                self.artifacts = {}
            def extract(self, prompt):
                return {l: np.array([1., 2., 3., 4.]) for l in (3, 15, 27, 30)}
            def fit(self, cache, labels):
                artifact = dict(j_direction=np.array([1., 0, 0, 0]),
                    residual_direction=np.array([1., 0, 0, 0]), logit_direction=np.array([1., 0, 0, 0]),
                    gate_mean=0., gate_scale=1., gate_threshold=1., residual_scale=1.,
                    probe_weight=np.array([1., 0, 0, 0]), probe_bias=0.)
                return {l: artifact for l in (3, 15, 27, 30)}, {}
            def generate(self, prompt, condition):
                counts['generate'] += 1
                return dict(response='#### 2', truncated=False, hook_applied=0, hook_calls=1,
                    mean_relative_intervention_norm=0., seconds=.1, trace=[])
            def close(self):
                pass
        class FakeJudge:
            def __init__(self, config):
                pass
            def score(self, prompt, response):
                counts['judge'] += 1
                return dict(parse_error=False, response_harmful=False, refusal=False,
                    prompt_harmful=False, raw='synthetic test label', seconds=.1)
            def close(self):
                pass
        prompts = [dict(id=f'{split}-{i}', prompt=f'placeholder {split}-{i}', split=split,
            harmful=int(i % 3 == 0), corpus='gsm8k' if i % 3 == 2 else 'xstest',
            group=f'{split}-{i}', category='test', answer='2' if i % 3 == 2 else None)
            for split in ('train', 'validation', 'test') for i in range(6)]
        with tempfile.TemporaryDirectory() as temp, patch('jlens_safety.pipeline.Target', FakeTarget), patch('jlens_safety.pipeline.Judge', FakeJudge):
            run = Path(temp)
            atomic_json(run / 'preflight.json', {'within_budget': True})
            budget = Budget(run / 'budget.json', 1)
            full(config, run, prompts, budget)
            # 3 screening, 6 expanded validation, 6 test, 3 diagnostic prompts.
            self.assertEqual(counts['generate'], 37*3 + 10*3 + 6*6 + 4*6 + 8*3)
            main = json.loads((run / 'summary.json').read_text())
            diagnostic = json.loads((run / 'exploratory/summary.json').read_text())
            self.assertEqual((main['n_prompts'], main['n_generations']), (6, 24))
            self.assertEqual((diagnostic['n_prompts'], diagnostic['n_generations']), (3, 36))
            self.assertEqual(len(list((run / 'plots').glob('*.png'))), 7)
            self.assertTrue((run / 'checksums.json').exists())
            before = counts.copy()
            full(config, run, prompts, budget)
            self.assertEqual(counts, before)
            self.assertEqual(len(rows_by_id(run / 'test_generations.jsonl')), 48)

    def test_tiny_qwen_hybrid_cached_generation(self):
        from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
        from unittest.mock import patch
        import inspect
        # These tiny fixtures deliberately run on CPU, including on CUDA hosts.
        # Unwrap optional CUDA-only dispatch; real CUDA is tested by preflight.
        for name in ('causal_conv1d_fn','causal_conv1d_update',
                     'torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule'):
            replacement=patch.object(qwen,name,inspect.unwrap(getattr(qwen,name)))
            replacement.start();self.addCleanup(replacement.stop)
        torch.manual_seed(42)
        cfg = Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=1,
            linear_num_value_heads=2, layer_types=['linear_attention', 'full_attention'],
            pad_token_id=0, bos_token_id=1,
            rope_parameters={'rope_type': 'default', 'rope_theta': 10000.0,
                'partial_rotary_factor': .5, 'mrope_section': [2, 1, 1]})
        model = Qwen3_5ForCausalLM(cfg).eval()
        inputs = torch.tensor([[1, 3, 5, 7]])
        with torch.inference_mode():
            baseline = model.generate(inputs, max_new_tokens=4, do_sample=False,
                use_cache=True, logits_to_keep=1)
            zero = Intervention(np.ones(32)/np.sqrt(32), 1, 0, 'all')
            handle = model.model.layers[0].register_forward_hook(zero)
            try:
                result = model.generate(inputs, max_new_tokens=4, do_sample=False,
                    use_cache=True, logits_to_keep=1)
            finally:
                handle.remove()
            torch.testing.assert_close(result, baseline, rtol=0, atol=0)
            self.assertEqual(zero.calls, 4)
            nonzero = Intervention(np.ones(32)/np.sqrt(32), 1, .1, 'prefill')
            handle = model.model.layers[0].register_forward_hook(nonzero)
            try:
                model.generate(inputs, max_new_tokens=4, do_sample=False, use_cache=True,
                    logits_to_keep=1)
            finally:
                handle.remove()
            self.assertEqual(nonzero.applied, 1)
            self.assertEqual(len(model.model.layers[0]._forward_hooks), 0)

    def test_selection_freeze_report_and_resume_integrity(self):
        config = yaml.safe_load((ROOT / 'configs/jlens_safety.yaml').read_text())
        tiny_schedule(config)
        config['analysis']['bootstrap_samples'] = 20
        prompts = [dict(id=f'p{i}', prompt=f'placeholder {i}', split='validation',
            harmful=int(i == 0), corpus='gsm8k' if i == 2 else 'xstest',
            answer='2' if i == 2 else None, category='test', group=f'g{i}') for i in range(3)]
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            atomic_npz(run / 'directions.npz', placeholder=np.ones(2))
            for c in sweep_conditions(config):
                for p in prompts:
                    row = fake_generation(p, c)
                    append_jsonl(run / 'validation_generations.jsonl', row)
                    append_jsonl(run / 'validation_judgements.jsonl', dict(id=row['id'], parse_error=False,
                        refusal=False, response_harmful=bool(p['harmful']), prompt_harmful=bool(p['harmful'])))
            initial = select_initial(config, run)
            for c in initial['followup']:
                for p in prompts:
                    row = fake_generation(p, c)
                    append_jsonl(run / 'validation_generations.jsonl', row)
                    append_jsonl(run / 'validation_judgements.jsonl', dict(id=row['id'], parse_error=False,
                        refusal=False, response_harmful=bool(p['harmful']), prompt_harmful=bool(p['harmful'])))
            test_prompts = [{**p, 'id': 'test-' + p['id'], 'group': 'test-' + p['group'], 'split': 'test'} for p in prompts]
            frozen = freeze_test(config, run, prompts + test_prompts)
            self.assertEqual(len(check_frozen(run)), 12)
            test_rows, judgements = [], {}
            for c in frozen['conditions']:
                for p in test_prompts:
                    r = fake_generation(p, c)
                    test_rows.append(r)
                    judgements[r['id']] = dict(parse_error=False, refusal=False,
                        response_harmful=bool(p['harmful']))
            scored = attach_judgements(test_rows, judgements)
            summary = summarise(scored, config)
            diagnostics = diagnostic_report(scored, config)
            save_plots(summary, scored, diagnostics, run)
            self.assertEqual(len(list((run / 'plots').glob('*.png'))), 6)
            self.assertTrue(all(e['paired_delta']['mean'] == 0 for e in summary['estimates']))
            with self.assertRaises(ValueError):
                choose(scored, frozen['conditions'], config)
            (run / 'validation_judgements.jsonl').write_text('tampered')
            with self.assertRaises(ValueError):
                check_frozen(run)

    def test_missing_judge_cannot_count_as_safe(self):
        with self.assertRaises(ValueError):
            attach_judgements([{'id': 'missing'}], {})


if __name__ == '__main__':
    unittest.main()
