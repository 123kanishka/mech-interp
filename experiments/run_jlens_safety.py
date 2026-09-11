#!/usr/bin/env python3
"""Separate J-lens safety suite. No implicit GPU rental, uploads, or paid APIs."""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from jlens_safety.common import Budget, BudgetExceeded, atomic_json, digest, source_hash
from jlens_safety.data import prepare, validate_rows
from jlens_safety.pipeline import (preflight, full, validate_config, workload, final_report,
    fit_and_validate, test_shard)
from sae_jlens.run_io import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/jlens_safety.yaml')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--stage', choices=('prepare', 'preflight', 'validate', 'test-shard',
        'full', 'report'), required=True)
    parser.add_argument('--shard-index', type=int)
    parser.add_argument('--shard-count', type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    validate_config(config)
    run = args.run_dir.resolve()
    root = (ROOT / config['output']['root']).resolve()
    if root not in run.parents:
        parser.error(f'Use a new subdirectory of {root}; old experiment output roots are protected')
    run.mkdir(parents=True, exist_ok=True)
    # OS releases this lock on crash; two workers cannot append into the same run.
    with (run / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = dict(config=config, code_sha256=source_hash(ROOT),
            packages={p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'numpy',
                'scikit-learn', 'huggingface-hub', 'pyarrow', 'pandas', 'matplotlib', 'pyyaml',
                'flash-linear-attention', 'fla-core')})
        manifest = run / 'manifest.json'
        if manifest.exists():
            original = json.loads(manifest.read_text())
            comparable = ('config', 'code_sha256') if args.stage == 'report' else identity.keys()
            if any(original[k] != identity[k] for k in comparable):
                raise ValueError('Config/code/dependencies changed. Use a NEW run directory; never mix results.')
            if args.stage == 'report':
                atomic_json(run / 'report_environment.json', identity)
        else:
            if any(p.name != '.lock' for p in run.iterdir()):
                raise ValueError('Refusing to adopt a nonempty unmanifested run directory')
            atomic_json(manifest, identity)
        if (run / 'SUCCESS').exists() and args.stage != 'report':
            print(f'Already complete: {run}')
            return 0
        budget = Budget(run / 'budget.json', config['budget']['hard_hours'])
        try:
            if args.stage != 'report':
                budget.check()
            if not (run / 'prompts.jsonl').exists():
                if args.stage != 'prepare':
                    raise ValueError('Run prepare first to freeze datasets and splits')
                prepare(config, run)
            prompts = validate_rows(read_jsonl(run / 'prompts.jsonl'))
            data_manifest = json.loads((run / 'data_manifest.json').read_text())
            if digest(prompts) != data_manifest['rows_hash']:
                raise ValueError('Benchmark prompt checksum changed')
            atomic_json(run / 'workload.json', workload(config, prompts))
            if args.stage == 'preflight':
                preflight(config, run, prompts, budget, args.shard_count or 1)
            elif args.stage == 'full':
                full(config, run, prompts, budget)
                (run / 'SUCCESS').touch()
            elif args.stage == 'validate':
                fit_and_validate(config, run, prompts, budget)
            elif args.stage == 'test-shard':
                if args.shard_index is None or args.shard_count is None:
                    parser.error('test-shard requires --shard-index and --shard-count')
                test_shard(config, run, prompts, budget, args.shard_index, args.shard_count)
            elif args.stage == 'report':
                final_report(config, run, prompts)
            atomic_json(run / 'state.json', dict(stage=args.stage, status='COMPLETE',
                scientific_run_complete=(run / 'SUCCESS').exists()))
            print(f'{args.stage} complete: {run}')
            return 0
        except Exception as exc:
            status = 'BUDGET_STOP' if isinstance(exc, BudgetExceeded) else 'FAILED'
            atomic_json(run / 'state.json', dict(stage=args.stage, status=status, error=str(exc)))
            (run / 'traceback.txt').write_text(traceback.format_exc())
            print(f'{status}: {exc}', file=sys.stderr)
            return 2 if status == 'BUDGET_STOP' else 1
        finally:
            budget.save()


if __name__ == '__main__':
    raise SystemExit(main())
