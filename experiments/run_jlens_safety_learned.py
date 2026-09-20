#!/usr/bin/env python3
"""Run the schema-3 signed-token steering protocol."""
from __future__ import annotations
import argparse
import fcntl
import importlib.metadata
import json
from pathlib import Path
import sys
import traceback
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jlens_safety.common import (
    Budget,
    BudgetExceeded,
    atomic_json,
    digest,
    file_hash,
    source_hash,
)
from jlens_safety.data import validate_rows
from jlens_safety.data_v3 import prepare
from jlens_safety.protocol_v3 import (
    validate_config,
    workload,
    preflight,
    fit_and_validate,
    full,
    run_test_shard,
    screen_and_select,
    fit_shard,
    validate_shard,
    freeze_validated,
    inherit_development_pilot,
)
from jlens_safety.report_v3 import final_report
from sae_jlens.run_io import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/jlens_safety_learned.yaml"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=(
            "prepare",
            "inherit-pilot",
            "preflight",
            "screen",
            "fit-shard",
            "validate-shard",
            "freeze",
            "validate",
            "test-shard",
            "official-judge",
            "full",
            "report",
        ),
        required=True,
    )
    parser.add_argument("--pilot-source", type=Path)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    c = yaml.safe_load(args.config.read_text())
    validate_config(c)
    run = args.run_dir.resolve()
    root = (ROOT / c["output"]["root"]).resolve()
    if root != (ROOT / "results/safety_runs_v3").resolve() or root not in run.parents:
        parser.error(
            "Use a NEW subdirectory of results/safety_runs_v3; historical results are protected"
        )
    if args.stage in ("test-shard", "fit-shard", "validate-shard") and (
        args.shard_index is None or not 0 <= args.shard_index < args.shard_count
    ):
        parser.error("Sharded stages require 0 <= shard-index < shard-count")
    run.mkdir(parents=True, exist_ok=True)
    identity = dict(
        config=c, code_sha256=source_hash(ROOT), runner_sha256=file_hash(Path(__file__))
    )
    with (run / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = run / "manifest.json"
        if path.exists():
            if json.loads(path.read_text()) != identity:
                raise ValueError("Code/config changed; use a NEW run directory")
        else:
            if any(p.name != ".lock" for p in run.iterdir()):
                raise ValueError("Refusing to adopt an existing unmanifested run")
            atomic_json(path, identity)
        if (run / "SUCCESS").exists() and args.stage != "report":
            print("Run already complete:", run)
            return 0
        budget = Budget(run / "budget.json", c["budget"]["hard_hours"])
        try:
            if args.stage not in ("prepare", "report"):
                packages = {}
                for name in (
                    "torch",
                    "transformers",
                    "numpy",
                    "scikit-learn",
                    "huggingface-hub",
                    "flash-linear-attention",
                    "fla-core",
                    "causal-conv1d",
                    "accelerate",
                ):
                    try:
                        packages[name] = importlib.metadata.version(name)
                    except importlib.metadata.PackageNotFoundError:
                        packages[name] = "missing"
                runtime = run / "runtime_environment.json"
                if runtime.exists() and json.loads(runtime.read_text()) != packages:
                    raise ValueError(
                        "GPU dependency versions changed across stages or workers"
                    )
                atomic_json(runtime, packages)
            if not (run / "prompts.jsonl").exists():
                if args.stage != "prepare":
                    raise ValueError("Run prepare first to freeze data and splits")
                prepare(c, run)
            rows = validate_rows(read_jsonl(run / "prompts.jsonl"))
            if (
                digest(rows)
                != json.loads((run / "data_manifest.json").read_text())["rows_hash"]
            ):
                raise ValueError("Frozen dataset checksum changed")
            atomic_json(run / "workload.json", workload(c, rows))
            if args.stage == "inherit-pilot":
                if args.pilot_source is None:
                    parser.error("--pilot-source is required for inherit-pilot")
                inherit_development_pilot(
                    c, run, rows, args.pilot_source, args.shard_count
                )
            elif args.stage == "preflight":
                preflight(c, run, rows, budget, args.shard_count)
            elif args.stage == "screen":
                screen_and_select(c, run, rows, budget)
            elif args.stage == "fit-shard":
                fit_shard(c, run, rows, budget, args.shard_index, args.shard_count)
            elif args.stage == "validate-shard":
                validate_shard(c, run, rows, budget, args.shard_index, args.shard_count)
            elif args.stage == "freeze":
                freeze_validated(c, run, rows)
            elif args.stage == "validate":
                fit_and_validate(c, run, rows, budget)
            elif args.stage == "full":
                full(c, run, rows, budget)
            elif args.stage == "test-shard":
                run_test_shard(c, run, rows, budget, args.shard_index, args.shard_count)
            elif args.stage == "official-judge":
                from jlens_safety.official_judge import judge_harmbench

                judge_harmbench(c, run, budget)
            elif args.stage == "report":
                final_report(c, run, rows)
            atomic_json(
                run / "state.json",
                dict(
                    stage=args.stage,
                    status="COMPLETE",
                    scientific_run_complete=(run / "SUCCESS").exists(),
                ),
            )
            print(args.stage, "complete:", run)
            return 0
        except Exception as exc:
            status = "BUDGET_STOP" if isinstance(exc, BudgetExceeded) else "FAILED"
            atomic_json(
                run / "state.json",
                dict(stage=args.stage, status=status, error=str(exc)),
            )
            (run / "traceback.txt").write_text(traceback.format_exc())
            if args.stage == "report":
                (run / "SUCCESS").unlink(missing_ok=True)
            print(status + ": " + str(exc), file=sys.stderr)
            return 2 if status == "BUDGET_STOP" else 1
        finally:
            budget.save()


if __name__ == "__main__":
    raise SystemExit(main())
