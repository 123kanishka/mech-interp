"""Explicit staged experiment graph. Test conditions freeze before test generation."""

from __future__ import annotations

import json
import shutil
import time

import numpy as np

from .analysis import (
    attach_judgements,
    choose,
    diagnostic_report,
    save_plots,
    summarise,
    save_validation_plots,
    challenge_report,
)
from .backend import Judge, Target
from .data import scheduled_subsets
from .common import (
    append_unique,
    atomic_json,
    atomic_npz,
    digest,
    file_hash,
    rows_by_id,
)
from sae_jlens.run_io import read_jsonl


BASELINE = dict(name="baseline", method="baseline", layer=None, alpha=0.0, mode="all")


def sweep_conditions(config):
    result = [BASELINE]
    for method in ("jlens", "residual", "logit"):
        for layer in config["jlens"]["intervention_layers"]:
            for alpha in config["steering"]["alphas"]:
                result.append(
                    dict(
                        name=f"{method}_l{layer}_a{alpha}",
                        method=method,
                        layer=layer,
                        alpha=alpha,
                        mode="all",
                    )
                )
    return result


def validate_config(config):
    if config["model"]["name"] != "Qwen/Qwen3.5-4B":
        raise ValueError("This adapter is validated for the chosen Qwen3.5-4B only")
    layers = config["jlens"]["intervention_layers"]
    if len(layers) != 3 or len(set(layers)) != 3 or sorted(layers) != layers:
        raise ValueError("Specify three distinct ordered early/middle/late layers")
    if not set(layers) <= set(config["jlens"]["observation_layers"]):
        raise ValueError("All intervention layers must be observed")
    if config["model"]["enable_thinking"]:
        raise ValueError("Budget/protocol is registered for non-thinking mode")
    if (
        config["budget"]["hard_hours"] > 14
        or config["budget"]["target_hours"] > config["budget"]["hard_hours"]
    ):
        raise ValueError("Configured budget exceeds the approved 14-hour ceiling")
    if not 0 < config["steering"]["gate_benign_quantile"] < 1:
        raise ValueError("Gate quantile must be between zero and one")
    for split in ("train", "validation", "test"):
        if any(
            config["data"]["xstest_counts"][split][label] < 1
            for label in ("safe", "unsafe")
        ):
            raise ValueError("Each XSTest partition needs both classes")
    for key in ("harmbench_test", "gsm8k_validation", "gsm8k_test"):
        if config["data"][key] < 1:
            raise ValueError("All dataset slices must be nonempty")


def workload(config, prompts):
    nval = sum(r["split"] == "validation" for r in prompts)
    ntest = sum(r["split"] == "test" for r in prompts)
    ntrain = sum(r["split"] == "train" for r in prompts)
    validation_conditions = len(sweep_conditions(config)) + 2 * len(
        config["jlens"]["intervention_layers"]
    )
    test_conditions = 9 + len(config["steering"]["random_seeds"])
    subsets = scheduled_subsets(prompts, config)
    nscreen, ndiagnostic = len(subsets["screening"]), len(subsets["diagnostic"])
    shortlist = 1 + 3 * len(config["jlens"]["intervention_layers"])
    nvalidation = (
        len(sweep_conditions(config)) * nscreen
        + shortlist * (nval - nscreen)
        + 2 * len(config["jlens"]["intervention_layers"]) * nval
    )
    ntest_jobs = 4 * ntest + (test_conditions - 4) * ndiagnostic
    return dict(
        training_prompts=ntrain,
        validation_prompts=nval,
        test_prompts=ntest,
        validation_conditions=validation_conditions,
        test_conditions=test_conditions,
        screening_prompts=nscreen,
        diagnostic_prompts=ndiagnostic,
        primary_test_conditions=4,
        validation_generations=nvalidation,
        test_generations=ntest_jobs,
        generations=nvalidation + ntest_jobs,
        max_target_generated_tokens=(nvalidation + ntest_jobs)
        * config["model"]["max_new_tokens"],
    )


def preflight(config, run, prompts, budget, distributed_shards=1):
    """Checks gated access BEFORE target downloads and measures both model phases."""
    from huggingface_hub import hf_hub_url, get_hf_file_metadata
    import torch

    if shutil.disk_usage(run).free < config["budget"]["min_free_gib"] * 1024**3:
        raise RuntimeError(
            "Insufficient free disk for target + sequential judge caches"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Preflight requires the RTX 3090, not this CPU environment")
    c = config["judge"]
    # Metadata request does not accept licence terms or download model weights.
    get_hf_file_metadata(
        hf_hub_url(
            c["name"], "model-00001-of-00002.safetensors", revision=c["revision"]
        )
    )
    selected = []
    for corpus, harmful in [("xstest", 0), ("xstest", 1), ("gsm8k", 0)]:
        subset = [
            p
            for p in prompts
            if p["split"] == "validation"
            and p["corpus"] == corpus
            and p["harmful"] == harmful
        ]
        # Include long prompts so short-only throughput is not the estimator.
        selected.extend(
            sorted(subset, key=lambda r: len(r["prompt"]), reverse=True)[:2]
        )
    target = Target(config)
    results, extraction_times = [], []
    # FLA compiles/autotunes persistent Triton kernels on the first forward pass.
    # Warm them before timing so a one-time startup cost is not multiplied by all
    # 100 training extractions in the remaining-work estimate.
    warmup_started = time.monotonic()
    try:
        warmup_row = max(selected, key=lambda row: len(row["prompt"]))
        target.extract(warmup_row["prompt"])
        target.generate(warmup_row["prompt"], BASELINE, max_new_tokens=8)
        kernel_warmup_seconds = time.monotonic() - warmup_started
        torch.cuda.reset_peak_memory_stats()
        for row in selected:
            budget.check()
            t = time.monotonic()
            target.extract(row["prompt"])
            extraction_times.append(time.monotonic() - t)
            results.append((row, target.generate(row["prompt"], BASELINE)))
        # Actual model alpha=0 equivalence with hooks AND cached greedy generation.
        d = target.wrapper.d_model
        layer = config["jlens"]["intervention_layers"][0]
        target.artifacts = {
            layer: dict(
                j_direction=np.full(d, 1.0 / np.sqrt(d), dtype=np.float32),
                gate_mean=0.0,
                gate_scale=1.0,
                gate_threshold=0.0,
                residual_scale=1.0,
                probe_weight=np.zeros(d, dtype=np.float32),
                probe_bias=0.0,
            )
        }
        zero = dict(name="zero", method="jlens", layer=layer, alpha=0.0, mode="all")
        a = target.generate(selected[0]["prompt"], BASELINE, max_new_tokens=8)
        b = target.generate(selected[0]["prompt"], zero, max_new_tokens=8)
        if a["generated_ids"] != b["generated_ids"]:
            raise RuntimeError("Zero-strength hooked generation differs from baseline")
        tiny = target.generate(
            selected[0]["prompt"], {**zero, "alpha": 0.01}, max_new_tokens=8
        )
        if tiny["hook_applied"] == 0:
            raise RuntimeError("Nonzero steering hook did not execute")
        target_peak = torch.cuda.max_memory_allocated()
    finally:
        target.close()
    judge = Judge(config)
    judge_results = []
    torch.cuda.reset_peak_memory_stats()
    try:
        for row, generation in results:
            budget.check()
            result = judge.score(row["prompt"], generation["response"])
            if result["parse_error"]:
                raise RuntimeError(
                    f'WildGuard preflight parsing failed: {result["raw"]!r}'
                )
            judge_results.append(result)
        judge_peak = torch.cuda.max_memory_allocated()
    finally:
        judge.close()
    costs = workload(config, prompts)
    # Conservative full token-cap estimate; include measured prefill overhead per token.
    seconds_per_token = max(
        g["seconds"] / max(1, g["generated_tokens"]) for _, g in results
    )
    generation_seconds = seconds_per_token * config["model"]["max_new_tokens"]
    judge_seconds = max(j["seconds"] for j in judge_results)
    coordinator_jobs = costs["validation_generations"] + int(
        np.ceil(costs["test_generations"] / distributed_shards)
    )
    estimate = (
        config["budget"]["safety_factor"]
        * (
            coordinator_jobs * (generation_seconds + judge_seconds)
            + costs["training_prompts"] * max(extraction_times)
        )
        + 1800
    )
    report = dict(
        workload=costs,
        projected_remaining_hours=estimate / 3600,
        distributed_shards=distributed_shards,
        projected_coordinator_jobs=coordinator_jobs,
        measured_generation_seconds_per_token=seconds_per_token,
        judge_seconds=judge_seconds,
        target_peak_gib=target_peak / 1024**3,
        judge_peak_gib=judge_peak / 1024**3,
        kernel_warmup_seconds=kernel_warmup_seconds,
        zero_strength_equivalence=True,
        nonzero_hook_executed=True,
        gpu=torch.cuda.get_device_name(),
        budget_seconds_already_used=budget.seconds,
        within_budget=(
            budget.seconds + estimate <= config["budget"]["target_hours"] * 3600
        ),
        notes="Estimate, not a runtime guarantee. Full-token cap, 1.5x margin plus 30 minutes; subject to host load.",
    )
    atomic_json(run / "preflight.json", report)
    atomic_json(
        run / "preflight_examples.json",
        [
            dict(prompt=p, generation=g, judge=j)
            for (p, g), j in zip(results, judge_results)
        ],
    )
    if not report["within_budget"]:
        raise RuntimeError(
            "Projected distributed coordinator run exceeds target budget. Revise protocol before fitting/testing."
        )
    return report


def fit(config, run, prompts, budget):
    train = [p for p in prompts if p["split"] == "train"]
    target = Target(config)
    cache_root = run / "activations"
    cache_root.mkdir(exist_ok=True)
    try:
        cached = []
        for p in train:
            budget.check()
            path = cache_root / (p["id"] + ".npz")
            if not path.exists():
                atomic_npz(
                    path, **{str(k): v for k, v in target.extract(p["prompt"]).items()}
                )
            with np.load(path, allow_pickle=False) as content:
                cached.append({int(k): content[k] for k in content.files})
        x = {
            layer: np.stack([record[layer] for record in cached])
            for layer in config["jlens"]["observation_layers"]
        }
        artifacts, metadata = target.fit(x, [r["harmful"] for r in train])
        atomic_npz(
            run / "directions.npz",
            **{
                f"{layer}__{key}": np.asarray(value)
                for layer, artifact in artifacts.items()
                for key, value in artifact.items()
            },
        )
        atomic_json(
            run / "directions.json",
            dict(
                metadata=metadata,
                training_ids=[r["id"] for r in train],
                sha256=file_hash(run / "directions.npz"),
            ),
        )
    finally:
        target.close()


def load_artifacts(run):
    metadata = json.loads((run / "directions.json").read_text())
    if file_hash(run / "directions.npz") != metadata["sha256"]:
        raise ValueError("Direction artifact checksum mismatch")
    result = {}
    with np.load(run / "directions.npz", allow_pickle=False) as content:
        for key in content.files:
            layer, field = key.split("__")
            a = content[key]
            result.setdefault(int(layer), {})[field] = float(a) if a.ndim == 0 else a
    return result


def job_id(prompt, condition):
    return digest(dict(prompt_id=prompt["id"], condition=condition))[:24]


def generate_jobs(config, run, prompts, conditions, split, budget, allowed_ids=None):
    path = run / f"{split}_generations.jsonl"
    known = rows_by_id(path)
    jobs = [
        (p, c)
        for c in conditions
        for p in prompts
        if p["split"] == split
        and (allowed_ids is None or p["id"] in allowed_ids[c["name"]])
    ]
    if all(job_id(p, c) in known for p, c in jobs):
        return
    target = Target(config)
    target.artifacts = load_artifacts(run)
    try:
        for prompt, condition in jobs:
            key = job_id(prompt, condition)
            if key in known:
                continue
            budget.check()
            output = target.generate(prompt["prompt"], condition)
            row = {
                **prompt,
                **output,
                "id": key,
                "prompt_id": prompt["id"],
                "condition": condition,
            }
            append_unique(path, row, known)
            atomic_json(
                run / "progress.json",
                dict(
                    stage=f"{split}_generation",
                    complete=len(known),
                    requested=len(jobs),
                    active_seconds=budget.seconds,
                ),
            )
    finally:
        target.close()


def judge_jobs(config, run, split, budget):
    generations = read_jsonl(run / f"{split}_generations.jsonl")
    path = run / f"{split}_judgements.jsonl"
    known = rows_by_id(path)
    # Reuse exact prompt-response judge inputs, not just repeated response strings.
    cache = {r["input_hash"]: r for r in known.values()}
    if all(g["id"] in known for g in generations):
        return
    judge = Judge(config)
    try:
        for row in generations:
            if row["id"] in known:
                continue
            budget.check()
            key = digest(
                {
                    "prompt": row["prompt"],
                    "response": row["response"],
                    "judge": config["judge"],
                }
            )
            if key in cache:
                result = {
                    k: v for k, v in cache[key].items() if k not in ("id", "input_hash")
                }
            else:
                result = judge.score(row["prompt"], row["response"])
                if result["parse_error"]:
                    # Store failure separately so resume can retry without corrupting completed labels.
                    atomic_json(run / "judge_error.json", dict(id=row["id"], **result))
                    raise RuntimeError(
                        "Judge returned malformed labels; stopped without silently treating them as safe"
                    )
            scored = dict(**result, id=row["id"], input_hash=key)
            append_unique(path, scored, known)
            cache[key] = scored
    finally:
        judge.close()


def select_initial(
    config,
    run,
    selection_ids=None,
    candidate_conditions=None,
    filename="validation_selection.json",
):
    rows = attach_judgements(
        read_jsonl(run / "validation_generations.jsonl"),
        rows_by_id(run / "validation_judgements.jsonl"),
    )
    if selection_ids is not None:
        rows = [r for r in rows if r["prompt_id"] in selection_ids]
    choices, scorecards = {}, []
    for method in ("jlens", "residual", "logit"):
        for layer in config["jlens"]["intervention_layers"]:
            candidates = [
                c
                for c in (candidate_conditions or sweep_conditions(config))
                if c["method"] == method and c["layer"] == layer
            ]
            selected, scores = choose(rows, candidates, config)
            choices[f"{method}_{layer}"] = selected
            scorecards.extend(scores)
    followup = []
    for layer in config["jlens"]["intervention_layers"]:
        c = choices[f"jlens_{layer}"]["condition"]
        for mode in ("gated", "prefill"):
            followup.append({**c, "mode": mode, "name": c["name"] + "_" + mode})
    result = dict(choices=choices, scorecards=scorecards, followup=followup)
    atomic_json(run / filename, result)
    return result


def freeze_test(config, run, prompts):
    rows = attach_judgements(
        read_jsonl(run / "validation_generations.jsonl"),
        rows_by_id(run / "validation_judgements.jsonl"),
    )
    initial = json.loads((run / "validation_selection.json").read_text())
    choices = initial["choices"]
    layers = config["jlens"]["intervention_layers"]
    j_candidates = [choices[f"jlens_{layer}"]["condition"] for layer in layers]
    best, _ = choose(rows, j_candidates, config)
    layer = best["condition"]["layer"]
    conditions = [BASELINE]
    for position, layer in zip(("early", "middle", "late"), layers):
        conditions.append(
            {
                **choices[f"jlens_{layer}"]["condition"],
                "name": "jlens_" + position,
            }
        )
    for mode in ("gated", "prefill"):
        selected, _ = choose(
            rows, [c for c in initial["followup"] if c["mode"] == mode], config
        )
        conditions.append({**selected["condition"], "name": "jlens_" + mode})
    for method in ("residual", "logit"):
        selected, _ = choose(
            rows,
            [choices[f"{method}_{layer}"]["condition"] for layer in layers],
            config,
        )
        conditions.append({**selected["condition"], "name": method})
    for seed in config["steering"]["random_seeds"]:
        conditions.append(
            {
                **best["condition"],
                "name": f"random_{seed}",
                "method": "random",
                "random_seed": seed,
            }
        )
    residual = choices[f"residual_{layer}"]["condition"]
    conditions.append(
        dict(
            name="combined",
            method="combined",
            layer=layer,
            j_alpha=best["condition"]["alpha"],
            residual_alpha=residual["alpha"],
            alpha=abs(best["condition"]["alpha"]),
            mode="all",
        )
    )
    best_name = "jlens_" + dict(zip(layers, ("early", "middle", "late")))[layer]
    schedule = dict(
        primary_conditions=["baseline", best_name, "residual", "logit"],
        diagnostic_prompt_ids=scheduled_subsets(prompts, config)["diagnostic"],
    )
    result = dict(
        conditions=conditions,
        conditions_hash=digest(conditions),
        schedule=schedule,
        schedule_hash=digest(schedule),
        validation_generations_sha256=file_hash(run / "validation_generations.jsonl"),
        validation_judgements_sha256=file_hash(run / "validation_judgements.jsonl"),
        directions_sha256=file_hash(run / "directions.npz"),
        best_jlens_validation=best,
        note="Combined is norm-matched mixture, not AlphaSteer or a published SOTA reproduction. Infeasible validation choices remain flagged diagnostic tests, not deployable settings.",
    )
    atomic_json(run / "frozen_test.json", result)
    return result


def check_frozen(run):
    result = json.loads((run / "frozen_test.json").read_text())
    for file, key in [
        ("validation_generations.jsonl", "validation_generations_sha256"),
        ("validation_judgements.jsonl", "validation_judgements_sha256"),
        ("directions.npz", "directions_sha256"),
    ]:
        if file_hash(run / file) != result[key]:
            raise ValueError(
                "Frozen selection inputs changed; refuse contaminated test resume"
            )
    if digest(result["conditions"]) != result["conditions_hash"]:
        raise ValueError("Frozen condition hash mismatch")
    if digest(result["schedule"]) != result["schedule_hash"]:
        raise ValueError("Frozen test schedule changed")
    return result["conditions"]


def test_allowed_ids(run, prompts):
    schedule = json.loads((run / "frozen_test.json").read_text())["schedule"]
    all_ids = {p["id"] for p in prompts if p["split"] == "test"}
    diagnostic = set(schedule["diagnostic_prompt_ids"])
    if not diagnostic <= all_ids:
        raise ValueError("Diagnostic IDs outside held-out test")
    return {
        c["name"]: (
            all_ids if c["name"] in schedule["primary_conditions"] else diagnostic
        )
        for c in check_frozen(run)
    }


def sharded_test_allowed_ids(run, prompts, shard_index, shard_count):
    """Partition exact frozen test jobs deterministically."""
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Require 0 <= shard_index < shard_count")
    conditions = check_frozen(run)
    base = test_allowed_ids(run, prompts)
    test = [p for p in prompts if p["split"] == "test"]
    jobs = sorted(
        (job_id(p, c), c["name"], p["id"])
        for c in conditions
        for p in test
        if p["id"] in base[c["name"]]
    )
    selected = {c["name"]: set() for c in conditions}
    for position, (_, condition, prompt_id) in enumerate(jobs):
        if position % shard_count == shard_index:
            selected[condition].add(prompt_id)
    return selected


def final_report(config, run, prompts):
    conditions = check_frozen(run)
    generated = rows_by_id(run / "test_generations.jsonl")
    allowed = test_allowed_ids(run, prompts)
    expected = {
        job_id(p, c)
        for p in prompts
        if p["split"] == "test"
        for c in conditions
        if p["id"] in allowed[c["name"]]
    }
    if set(generated) != expected:
        raise ValueError("Missing or unexpected test jobs; cannot mark SUCCESS")
    rows = attach_judgements(
        list(generated.values()), rows_by_id(run / "test_judgements.jsonl")
    )
    schedule = json.loads((run / "frozen_test.json").read_text())["schedule"]
    primary_rows = [
        r for r in rows if r["condition"]["name"] in schedule["primary_conditions"]
    ]
    diagnostic_rows = [
        r for r in rows if r["prompt_id"] in schedule["diagnostic_prompt_ids"]
    ]
    summary = summarise(primary_rows, config)
    summary["scope"] = "Four primary conditions on the complete held-out test"
    diagnostics = diagnostic_report(rows, config)
    atomic_json(run / "summary.json", summary)
    atomic_json(run / "diagnostics.json", diagnostics)
    atomic_json(run / "challenge_outcomes.json", challenge_report(diagnostic_rows))
    save_plots(summary, primary_rows, diagnostics, run)
    exploratory = run / "exploratory"
    exploratory.mkdir(exist_ok=True)
    small_summary = summarise(diagnostic_rows, config)
    small_summary["scope"] = (
        "Exploratory: all conditions on the SAME predefined diagnostic subset"
    )
    atomic_json(exploratory / "summary.json", small_summary)
    save_plots(
        small_summary,
        diagnostic_rows,
        diagnostic_report(diagnostic_rows, config),
        exploratory,
    )
    validation = attach_judgements(
        read_jsonl(run / "validation_generations.jsonl"),
        rows_by_id(run / "validation_judgements.jsonl"),
    )
    screening_ids = set(scheduled_subsets(prompts, config)["screening"])
    save_validation_plots(
        [r for r in validation if r["prompt_id"] in screening_ids], config, run
    )
    # Diverse review sample, stratified by condition/corpus, never called representative evidence.
    review = []
    for condition in conditions:
        for corpus in sorted({r["corpus"] for r in rows}):
            subset = [
                r
                for r in rows
                if r["condition"]["name"] == condition["name"] and r["corpus"] == corpus
            ]
            review.extend(subset[:2])
    atomic_json(run / "manual_review_sample.json", review)
    atomic_json(
        run / "checksums.json",
        {
            str(p.relative_to(run)): file_hash(p)
            for p in run.rglob("*")
            if p.is_file()
            and p.name not in ("checksums.json", "budget.json", "state.json", ".lock")
        },
    )


def full(config, run, prompts, budget):
    report = json.loads((run / "preflight.json").read_text())
    if not report["within_budget"]:
        raise RuntimeError("Preflight did not approve the projected budget")
    if not (run / "directions.json").exists():
        budget.check()
        fit(config, run, prompts, budget)
    if not (run / "frozen_test.json").exists():
        subsets = scheduled_subsets(prompts, config)
        screening = [p for p in prompts if p["id"] in subsets["screening"]]
        generate_jobs(
            config, run, screening, sweep_conditions(config), "validation", budget
        )
        judge_jobs(config, run, "validation", budget)
        screened = select_initial(
            config,
            run,
            selection_ids=set(subsets["screening"]),
            filename="screening_selection.json",
        )
        shortlist = [BASELINE] + [v["condition"] for v in screened["choices"].values()]
        generate_jobs(config, run, prompts, shortlist, "validation", budget)
        judge_jobs(config, run, "validation", budget)
        initial = select_initial(config, run, candidate_conditions=shortlist)
        generate_jobs(config, run, prompts, initial["followup"], "validation", budget)
        judge_jobs(config, run, "validation", budget)
        freeze_test(config, run, prompts)
    conditions = check_frozen(run)
    generate_jobs(
        config,
        run,
        prompts,
        conditions,
        "test",
        budget,
        allowed_ids=test_allowed_ids(run, prompts),
    )
    judge_jobs(config, run, "test", budget)
    final_report(config, run, prompts)


def fit_and_validate(config, run, prompts, budget, allow_unestimated=False):
    """Sequential coordinator phase; freeze choices before held-out generation."""
    if allow_unestimated:
        atomic_json(
            run / "preflight_override.json",
            dict(
                authorized=True,
                reason="Runtime-estimate gate bypassed with --allow-unestimated-run",
                consequence="Remaining runtime was not validated by a completed preflight",
                target_hours=config["budget"]["target_hours"],
                hard_hours=config["budget"]["hard_hours"],
            ),
        )
    else:
        report = json.loads((run / "preflight.json").read_text())
        if not report["within_budget"]:
            raise RuntimeError("Preflight did not approve the projected budget")
    if not (run / "directions.json").exists():
        budget.check()
        fit(config, run, prompts, budget)
    if (run / "frozen_test.json").exists():
        check_frozen(run)
        return
    subsets = scheduled_subsets(prompts, config)
    screening = [p for p in prompts if p["id"] in subsets["screening"]]
    generate_jobs(
        config, run, screening, sweep_conditions(config), "validation", budget
    )
    judge_jobs(config, run, "validation", budget)
    screened = select_initial(
        config,
        run,
        selection_ids=set(subsets["screening"]),
        filename="screening_selection.json",
    )
    shortlist = [BASELINE] + [v["condition"] for v in screened["choices"].values()]
    generate_jobs(config, run, prompts, shortlist, "validation", budget)
    judge_jobs(config, run, "validation", budget)
    initial = select_initial(config, run, candidate_conditions=shortlist)
    generate_jobs(config, run, prompts, initial["followup"], "validation", budget)
    judge_jobs(config, run, "validation", budget)
    freeze_test(config, run, prompts)


def test_shard(config, run, prompts, budget, shard_index, shard_count):
    """Run one disjoint test shard; report only after verified merge."""
    allowed = sharded_test_allowed_ids(run, prompts, shard_index, shard_count)
    conditions = check_frozen(run)
    generate_jobs(config, run, prompts, conditions, "test", budget, allowed_ids=allowed)
    judge_jobs(config, run, "test", budget)
    expected = sum(len(ids) for ids in allowed.values())
    generated = rows_by_id(run / "test_generations.jsonl")
    judged = rows_by_id(run / "test_judgements.jsonl")
    if len(generated) != expected or set(generated) != set(judged):
        raise ValueError("Shard output incomplete; refuse SHARD_SUCCESS")
    atomic_json(
        run / f"shard_{shard_index}_of_{shard_count}.json",
        dict(
            shard_index=shard_index,
            shard_count=shard_count,
            jobs=expected,
            generation_sha256=file_hash(run / "test_generations.jsonl"),
            judgement_sha256=file_hash(run / "test_judgements.jsonl"),
        ),
    )
