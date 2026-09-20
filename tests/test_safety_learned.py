"""Signed coefficients, causal position semantics, data isolation and reporting."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from jlens_safety.core import Intervention, shortlist_tokens
from jlens_safety.common import (
    atomic_json,
    atomic_npz,
    file_hash,
    rows_by_id,
    Budget,
)
from jlens_safety.data import validate_rows
from jlens_safety.data_v3 import (
    TYPES,
    wild_rows,
    link_request_groups,
    select_training,
    add_harmbench_attacks,
    capped_test_groups,
)
from jlens_safety.learned import (
    token_intervention,
    steering_mask,
    differentiable_hook,
    constrain_coefficients,
    fit_coefficients,
    paired_power_plan,
)
from jlens_safety.pipeline import job_id, sharded_test_allowed_ids
from jlens_safety.protocol_v3 import (
    validate_config,
    scalar_conditions,
    score_conditions,
    freeze,
    fitting_conditions,
    verify_extra_hashes,
    workload,
    shard_conditions,
    validation_shard_ids,
)
from jlens_safety.report_v3 import final_report
from sae_jlens.run_io import append_jsonl

ROOT = Path(__file__).resolve().parents[1]


def config():
    c = yaml.safe_load((ROOT / "configs/jlens_safety_learned.yaml").read_text())
    c["analysis"]["bootstrap_samples"] = 40
    return c


class LearnedCoreTests(unittest.TestCase):
    def test_capped_test_keeps_whole_groups_and_is_deterministic(self):
        c = config()
        c["data"]["test_caps"] = {"xstest/0": 2, "xstest/1": 2}
        rows = []
        for group in range(3):
            for harmful in (0, 1):
                rows.append(
                    dict(
                        id=f"{group}-{harmful}",
                        group=str(group),
                        split="test",
                        corpus="xstest",
                        harmful=harmful,
                    )
                )
        first = capped_test_groups(rows, c)
        self.assertEqual(first, capped_test_groups(list(reversed(rows)), c)[::-1])
        self.assertEqual(len(first), 4)
        self.assertEqual(len({r["group"] for r in first}), 2)

    def test_two_gpu_shards_cover_fits_and_validation_exactly_once(self):
        c = config()
        conditions = fitting_conditions(c, 15)
        a, b = [shard_conditions(conditions, i, 2) for i in range(2)]
        self.assertEqual((len(a), len(b)), (9, 9))
        self.assertFalse({r["name"] for r in a} & {r["name"] for r in b})
        self.assertEqual({r["name"] for r in a + b}, {r["name"] for r in conditions})
        rows = [dict(id=str(i), split="validation") for i in range(11)]
        rows.append(dict(id="never-test", split="test"))
        assignments = [validation_shard_ids(rows, conditions, i, 2) for i in range(2)]
        for condition in conditions:
            name = condition["name"]
            self.assertFalse(assignments[0][name] & assignments[1][name])
            self.assertEqual(
                assignments[0][name] | assignments[1][name], {str(i) for i in range(11)}
            )
        with self.assertRaises(ValueError):
            shard_conditions(conditions, 2, 2)

    def test_same_32_token_rule_uses_absolute_means_and_excludes_special_ids(self):
        safe = torch.zeros(50)
        unsafe = torch.arange(50, dtype=torch.float32)
        unsafe[0] = 9999
        unsafe[1] = -100
        ids, contrast = shortlist_tokens(safe, unsafe, [0], 32)
        self.assertEqual(len(ids), 32)
        self.assertNotIn(0, ids.tolist())
        self.assertEqual(ids[0].item(), 1)
        self.assertEqual(set(ids.tolist()), {1} | set(range(19, 50)))
        self.assertEqual(contrast[1].item(), 100)

    def test_signed_token_update_and_zero_identity(self):
        artifacts = {15: dict(j_dictionary=np.eye(32), residual_scale=10.0)}
        beta = [0.012, -0.007, 0.003] + [0.0] * 29
        c = dict(
            method="jlens_tokens",
            layer=15,
            seed=11,
            coefficients=beta,
            max_update_norm=0.1,
            positions="last",
        )
        h = torch.zeros(1, 4, 32)
        hook = token_intervention(c, artifacts)
        result = hook(None, None, h)
        torch.testing.assert_close(result[0, -1], torch.tensor(beta) * 10)
        self.assertEqual(result[:, :-1].count_nonzero(), 0)
        zero = token_intervention({**c, "coefficients": [0.0] * 32}, artifacts)
        torch.testing.assert_close(zero(None, None, h), h, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            token_intervention({**c, "coefficients": [1.0] * 32}, artifacts)

    def test_all_positions_and_cached_decode(self):
        h = torch.zeros(1, 4, 2)
        hook = Intervention([1, -1], 2, 0.5, "all", positions="all")
        changed = hook(None, None, (h, "cache"))
        torch.testing.assert_close(
            changed[0], torch.tensor([[[1.0, -1.0]]]).expand_as(h)
        )
        self.assertEqual(changed[1], "cache")
        hook(None, None, torch.zeros(1, 1, 2))
        self.assertEqual(hook.edited_positions, 5)
        with self.assertRaises(ValueError):
            hook(None, None, h)

    def test_total_norm_matches_last_position_prefill_energy(self):
        h = torch.zeros(1, 9, 2)
        last = Intervention([1, 0], 1, 0.1, "all")
        all_tokens = Intervention(
            [1, 0], 1, 0.1, "all", positions="all", norm_matching="total"
        )
        self.assertAlmostEqual(
            float(last(None, None, h).norm()),
            float(all_tokens(None, None, h).norm()),
            places=6,
        )
        self.assertAlmostEqual(
            last.squared_update_norm, all_tokens.squared_update_norm, places=6
        )

    def test_teacher_forcing_masks_match_inference_positions(self):
        self.assertEqual(
            steering_mask(7, 4, "last").flatten().tolist(), [0, 0, 0, 1, 1, 1, 1]
        )
        self.assertEqual(steering_mask(7, 4, "all").flatten().tolist(), [1] * 7)
        beta = torch.nn.Parameter(torch.tensor([0.02, -0.03]))
        hook = differentiable_hook(beta, torch.eye(2), 10, 4, "last")
        output = hook(None, None, torch.zeros(1, 7, 2))
        output.sum().backward()
        torch.testing.assert_close(beta.grad, torch.tensor([40.0, 40.0]))

    def test_norm_constraint_handles_collinear_cancellation(self):
        dictionary = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        beta = torch.nn.Parameter(torch.tensor([10.0, -10.0]))
        constrain_coefficients(beta, dictionary, 0.1)
        torch.testing.assert_close(beta, torch.tensor([10.0, -10.0]))
        beta.data.copy_(torch.tensor([2.0, 1.0]))
        constrain_coefficients(beta, dictionary, 0.1)
        self.assertAlmostEqual(
            float((beta @ dictionary).norm().detach()), 0.1, places=6
        )

    def test_fitting_resume_matches_uninterrupted_adam(self):
        c = config()
        c["learning"].update(epochs=1, accumulation_steps=1, save_every_steps=1)
        rows = [dict(id=str(i), split="train") for i in range(5)]
        condition = dict(layer=15, seed=11, method="jlens_tokens", positions="all")

        def target():
            model = torch.nn.Linear(2, 2)
            return SimpleNamespace(
                model=model, artifacts={15: dict(j_dictionary=np.eye(2))}
            )

        def objective(target, row, beta, dictionary, layer, positions, c):
            loss = ((beta - torch.tensor([0.03, -0.02])) ** 2).sum()
            return loss, dict(nll=float(loss.detach()), truncated=False)

        class Stop:
            def __init__(self):
                self.calls = 0

            def check(self):
                self.calls += 1
                if self.calls == 3:
                    raise RuntimeError("interrupted")

        with (
            tempfile.TemporaryDirectory() as temp,
            patch("jlens_safety.learned.coefficient_loss", side_effect=objective),
        ):
            root = Path(temp)
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                fit_coefficients(
                    target(), rows, condition, c, root / "resume.npz", Stop()
                )
            budget = Budget(root / "budget.json", 1)
            resumed = fit_coefficients(
                target(), rows, condition, c, root / "resume.npz", budget
            )
            clean_target = target()
            clean = fit_coefficients(
                clean_target, rows, condition, c, root / "clean.npz", budget
            )
            np.testing.assert_allclose(
                resumed["coefficients"], clean["coefficients"], rtol=0, atol=0
            )
            self.assertGreater(resumed["coefficients"][0], 0)
            self.assertLess(resumed["coefficients"][1], 0)
            self.assertTrue(
                all(not p.requires_grad for p in clean_target.model.parameters())
            )

    def test_fitting_rejects_validation_and_test(self):
        with self.assertRaisesRegex(ValueError, "training"):
            fit_coefficients(
                None,
                [dict(split="validation")],
                {},
                config(),
                Path("/tmp/unused"),
                None,
            )

    def test_zero_discordance_cannot_claim_high_power(self):
        self.assertIsNone(paired_power_plan([0] * 100)["required_groups"])
        self.assertGreater(paired_power_plan([0, 1, -1, 0] * 20)["required_groups"], 2)


class LearnedDataTests(unittest.TestCase):
    def test_malformed_training_rows_are_audited_but_test_never_dropped(self):
        bad = dict(
            data_type="adversarial_benign",
            vanilla="base",
            adversarial="",
            completion="answer",
        )
        audit = []
        self.assertEqual(wild_rows([bad], invalid_training_rows=audit), [])
        self.assertEqual(
            audit,
            [
                dict(
                    id="wj-train-0",
                    category="adversarial_benign",
                    reason="missing_prompt",
                )
            ],
        )
        with self.assertRaisesRegex(ValueError, "Missing"):
            wild_rows([bad], evaluation=True, invalid_training_rows=[])

    def test_remote_benchmark_data_is_never_executed(self):
        from jlens_safety.data_v3 import parse_published_templates
        from jlens_safety.official_judge import parse_official_template

        self.assertEqual(
            parse_published_templates(
                "raise RuntimeError('do not execute')\nJAILBREAKS = ['one','two']"
            ),
            ["one", "two"],
        )
        self.assertEqual(
            parse_official_template(
                "raise RuntimeError('do not execute')\nLLAMA2_CLS_PROMPT = {'prompt':'{behavior}: {generation}'}"
            ),
            "{behavior}: {generation}",
        )
        with self.assertRaises((ValueError, TypeError)):
            parse_published_templates("JAILBREAKS = __import__('os').system('false')")

    def test_published_jailbreak_unicode_pairs_are_losslessly_decoded(self):
        pair = "\ud835\udd4f"
        self.assertEqual(
            pair.encode("utf-16", "surrogatepass").decode("utf-16"), "\U0001d54f"
        )
        with self.assertRaises(UnicodeError):
            "\ud835".encode("utf-16", "surrogatepass").decode("utf-16")

    def test_group_allocation_excludes_test_and_preserves_variant_groups(self):
        c = config()
        c["data"].update(
            train_per_type=5, validation_per_type=5, validation_group_fraction=0.5
        )
        records = []
        for i in range(100):
            for kind in TYPES:
                base = f'base {i} {kind.endswith("_harmful")}'
                records.append(
                    dict(
                        data_type=kind,
                        vanilla=base,
                        adversarial=f"{kind} variant {i}",
                        completion="reference",
                    )
                )
        training = wild_rows(records)
        evaluation = wild_rows(
            [
                dict(
                    data_type="adversarial_harmful",
                    vanilla="base 0 True",
                    adversarial="held out",
                )
            ],
            True,
        )
        linked = link_request_groups(training + evaluation)
        chosen, excluded = select_training(linked, c)
        validate_rows(chosen)
        self.assertEqual(sum(r["split"] == "train" for r in chosen), 20)
        self.assertEqual(sum(r["split"] == "validation" for r in chosen), 20)
        self.assertGreater(excluded["overlap_with_test_group"], 0)
        test_group = next(r["group"] for r in chosen if r["split"] == "test")
        self.assertTrue(
            all(r["split"] == "test" for r in chosen if r["group"] == test_group)
        )

    def test_cross_corpus_base_overlap_is_linked(self):
        rows = [
            dict(group="a", prompt="Some request", base_prompt="Some request"),
            dict(group="b", prompt="Different framing", base_prompt=" SOME  REQUEST "),
        ]
        linked = link_request_groups(rows)
        self.assertEqual(linked[0]["group"], linked[1]["group"])

    def test_attack_suite_required_and_exact_coverage(self):
        c = config()
        c["data"]["harmbench_attack_mode"] = "file"
        with self.assertRaisesRegex(ValueError, "Supply"):
            add_harmbench_attacks([], c)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cases.json"
            path.write_text(json.dumps({"behavior": ["test case"]}))
            c["data"].update(
                harmbench_attack_file=str(path),
                harmbench_attack_source="published test fixture",
            )
            rows, sha = add_harmbench_attacks(
                [dict(id="hb-behavior", behavior_id="behavior", group="g")], c
            )
            self.assertEqual(rows[0]["group"], "g")
            self.assertEqual(sha, file_hash(path))
            with self.assertRaisesRegex(ValueError, "Missing"):
                add_harmbench_attacks([dict(behavior_id="absent")], c)


def fixture_rows():
    result = []
    for split in ("train", "validation", "test"):
        for i, kind in enumerate(TYPES):
            result.append(
                dict(
                    id=f"{split}-{i}",
                    prompt=f"{split} example {i}",
                    group=f"{split}-{i}",
                    split=split,
                    corpus="wildjailbreak",
                    category=kind,
                    harmful=int(kind.endswith("_harmful")),
                    completion="reference",
                    answer=None,
                )
            )
        if split != "train":
            result.append(
                dict(
                    id=f"{split}-math",
                    prompt=f"{split} math",
                    group=f"{split}-math",
                    split=split,
                    corpus="gsm8k",
                    category="math",
                    harmful=0,
                    answer="2",
                )
            )
    return result


def fake_generation(prompt, condition):
    return dict(
        **{k: v for k, v in prompt.items() if k != "id"},
        id=job_id(prompt, condition),
        prompt_id=prompt["id"],
        condition=condition,
        response="#### 2",
        generated_ids=[2],
        truncated=False,
        hook_applied=0,
        hook_calls=1,
        mean_relative_intervention_norm=0.0,
        seconds=0.1,
        trace=[
            dict(
                layer=layer,
                step=0,
                j_score=float(prompt["harmful"]),
                probe_score=float(prompt["harmful"]),
            )
            for layer in (3, 15, 27, 30)
        ],
    )


class LearnedPipelineTests(unittest.TestCase):
    def test_complete_v3_orchestration_and_resume_without_gpu(self):
        import jlens_safety.pipeline as legacy
        import jlens_safety.protocol_v3 as protocol

        c = config()
        c["analysis"].update(
            screening_per_type=1, screening_gsm8k=1, diagnostic_per_stratum=1
        )
        prompts = fixture_rows()
        by_text = {r["prompt"]: r for r in prompts}
        counts = {"generation": 0, "fit": 0}

        class FakeTarget:
            def __init__(self, c):
                self.artifacts = {}

            def generate(self, text, condition):
                counts["generation"] += 1
                return fake_generation(by_text[text], condition)

            def close(self):
                pass

        class FakeJudge:
            def __init__(self, c):
                pass

            def score(self, text, response):
                return dict(
                    parse_error=False,
                    response_harmful=bool(by_text[text]["harmful"]),
                    refusal=False,
                )

            def close(self):
                pass

        def fake_fit(c, run, rows, budget):
            counts["fit"] += 1
            atomic_npz(run / "directions.npz", **{"15__j_direction": np.ones(32)})
            atomic_json(
                run / "directions.json",
                dict(
                    sha256=file_hash(run / "directions.npz"),
                    metadata={
                        "15": dict(
                            token_ids=list(range(32)),
                            tokens=[str(i) for i in range(32)],
                        )
                    },
                ),
            )

        def fake_coefficients(target, rows, condition, c, checkpoint, budget):
            return dict(coefficients=[0.0] * 32, max_update_norm=0.1)

        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(protocol, "Target", FakeTarget),
            patch.object(legacy, "Target", FakeTarget),
            patch.object(legacy, "Judge", FakeJudge),
            patch.object(protocol, "fit", side_effect=fake_fit),
            patch.object(protocol, "fit_coefficients", side_effect=fake_coefficients),
            patch("jlens_safety.report_v3.plots"),
        ):
            run = Path(temp)
            budget = Budget(run / "budget.json", 1)
            atomic_json(
                run / "preflight.json",
                dict(
                    within_budget=True,
                    informative=True,
                    generation_limit_adequate=True,
                    backward_and_zero_hooks_passed=True,
                ),
            )
            protocol.full(c, run, prompts, budget)
            costs = workload(c, prompts)
            self.assertEqual(
                len(rows_by_id(run / "validation_generations.jsonl")),
                costs["validation_generations"],
            )
            self.assertEqual(
                len(rows_by_id(run / "test_generations.jsonl")),
                costs["test_generations"],
            )
            self.assertTrue((run / "SUCCESS").exists())
            review = json.loads((run / "manual_review_blinded.json").read_text())
            review[0]["notes"] = "Human annotation that must survive rerender"
            atomic_json(run / "manual_review_blinded.json", review)
            before = counts.copy()
            protocol.full(c, run, prompts, budget)
            self.assertEqual(before, counts)
            self.assertEqual(
                json.loads((run / "manual_review_blinded.json").read_text())[0][
                    "notes"
                ],
                review[0]["notes"],
            )

    def test_new_config_and_shortlist_contract(self):
        c = config()
        validate_config(c)
        self.assertEqual(c["data"]["train_per_type"] * 4, 8000)
        self.assertEqual(len(fitting_conditions(c, 15)), 18)
        c["jlens"]["dictionary_size"] = 16
        with self.assertRaises(ValueError):
            validate_config(c)

    def test_no_arbitrary_sign_wins_when_baseline_ties(self):
        c = config()
        conditions = scalar_conditions(c, [15])
        from jlens_safety.analysis import attach_judgements

        rows = []
        judgements = {}
        for condition in conditions:
            for p in fixture_rows():
                if p["split"] != "validation":
                    continue
                row = fake_generation(p, condition)
                rows.append(row)
                judgements[row["id"]] = dict(
                    parse_error=False, response_harmful=False, refusal=False
                )
        chosen, _ = score_conditions(attach_judgements(rows, judgements), conditions, c)
        self.assertEqual(chosen["condition"]["method"], "baseline")

    def test_freeze_disjoint_shards_and_complete_report(self):
        c = config()
        c["learning"]["seeds"] = [11]
        c["analysis"]["diagnostic_per_stratum"] = 1
        c["analysis"]["screening_per_type"] = 1
        c["analysis"]["screening_gsm8k"] = 1
        prompts = fixture_rows()
        scalars = scalar_conditions(c, [15])
        learned = [
            dict(**condition, coefficients=[0.0] * 32, max_update_norm=0.1)
            for condition in fitting_conditions(c, 15)
        ]
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            atomic_npz(run / "directions.npz", placeholder=np.ones(2))
            atomic_json(run / "learned_coefficients.json", learned)
            for condition in scalars + learned:
                for p in prompts:
                    if p["split"] != "validation":
                        continue
                    r = fake_generation(p, condition)
                    append_jsonl(run / "validation_generations.jsonl", r)
                    append_jsonl(
                        run / "validation_judgements.jsonl",
                        dict(
                            id=r["id"],
                            parse_error=False,
                            response_harmful=bool(p["harmful"]),
                            refusal=False,
                        ),
                    )
            freeze(c, run, prompts, scalars, learned)
            verify_extra_hashes(run)
            frozen = json.loads((run / "frozen_test.json").read_text())
            self.assertEqual(len(frozen["schedule"]["primary_conditions"]), 5)
            shards = [sharded_test_allowed_ids(run, prompts, i, 2) for i in range(2)]
            sets = [
                {(name, p) for name, ids in shard.items() for p in ids}
                for shard in shards
            ]
            self.assertFalse(sets[0] & sets[1])
            for condition in frozen["conditions"]:
                for p in prompts:
                    if (
                        p["split"] != "test"
                        or (condition["name"], p["id"]) not in sets[0] | sets[1]
                    ):
                        continue
                    r = fake_generation(p, condition)
                    append_jsonl(run / "test_generations.jsonl", r)
                    append_jsonl(
                        run / "test_judgements.jsonl",
                        dict(
                            id=r["id"],
                            parse_error=False,
                            response_harmful=bool(p["harmful"]),
                            refusal=False,
                        ),
                    )
            with patch("jlens_safety.report_v3.plots"):
                final_report(c, run, prompts)
            self.assertTrue((run / "SUCCESS").exists())
            comparisons = json.loads((run / "factorial_comparisons.json").read_text())
            self.assertTrue(all(r["mean"] == 0 for r in comparisons))
            self.assertFalse(
                json.loads((run / "claim_readiness.json").read_text())[
                    "publication_claims_ready"
                ]
            )
            atomic_json(run / "learned_coefficients.json", [])
            with self.assertRaisesRegex(ValueError, "changed"):
                verify_extra_hashes(run)

    def test_tiny_qwen_backward_reaches_beta_only_for_both_scopes(self):
        from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
        import inspect

        for name in (
            "causal_conv1d_fn",
            "causal_conv1d_update",
            "torch_chunk_gated_delta_rule",
            "torch_recurrent_gated_delta_rule",
        ):
            replacement = patch.object(qwen, name, inspect.unwrap(getattr(qwen, name)))
            replacement.start()
            self.addCleanup(replacement.stop)
        torch.manual_seed(1)
        cfg = Qwen3_5TextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_num_key_heads=1,
            linear_num_value_heads=2,
            layer_types=["linear_attention", "full_attention"],
            pad_token_id=0,
            bos_token_id=1,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
                "mrope_section": [2, 1, 1],
            },
        )
        model = Qwen3_5ForCausalLM(cfg).eval()
        model.requires_grad_(False)
        ids = torch.tensor([[1, 3, 5, 7, 9, 11]])
        for positions in ("last", "all"):
            beta = torch.nn.Parameter(torch.zeros(32))
            handle = model.model.layers[0].register_forward_hook(
                differentiable_hook(beta, torch.eye(32), 1, 4, positions)
            )
            try:
                logits = model(ids, use_cache=False, logits_to_keep=3).logits
                torch.nn.functional.cross_entropy(
                    logits.reshape(-1, 64), torch.tensor([9, 11, 13])
                ).backward()
            finally:
                handle.remove()
            self.assertIsNotNone(beta.grad)
            self.assertGreater(float(beta.grad.norm()), 0)
            self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
