#!/usr/bin/env python3
"""Merge independently produced test shards with strict identity checks."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jlens_safety.common import append_unique, file_hash, rows_by_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinator", type=Path, required=True)
    parser.add_argument("--worker", type=Path, action="append", required=True)
    args = parser.parse_args()
    coordinator = args.coordinator.resolve()
    identity = json.loads((coordinator / "manifest.json").read_text())
    frozen = json.loads((coordinator / "frozen_test.json").read_text())
    prompts_hash = file_hash(coordinator / "prompts.jsonl")
    for worker in args.worker:
        worker = worker.resolve()
        if json.loads((worker / "manifest.json").read_text()) != identity:
            raise ValueError(f"Manifest mismatch: {worker}")
        if json.loads((worker / "frozen_test.json").read_text()) != frozen:
            raise ValueError(f"Frozen selection mismatch: {worker}")
        if file_hash(worker / "prompts.jsonl") != prompts_hash:
            raise ValueError(f"Prompt checksum mismatch: {worker}")
        if identity["config"].get("experiment", {}).get("schema_version") == 3:
            for filename in (
                "directions.npz",
                "learned_coefficients.json",
                "runtime_environment.json",
            ):
                if file_hash(worker / filename) != file_hash(coordinator / filename):
                    raise ValueError(
                        f"Worker artifact/runtime mismatch: {worker}/{filename}"
                    )
        for filename in (
            "test_generations.jsonl",
            "test_judgements.jsonl",
            "harmbench_judgements.jsonl",
        ):
            destination = coordinator / filename
            known = rows_by_id(destination)
            for row in rows_by_id(worker / filename).values():
                append_unique(destination, row, known)
    print(f"Merged verified shards into {coordinator}")


if __name__ == "__main__":
    main()
