#!/usr/bin/env python3
"""Stage-aware entry point for the SAE--J-Lens experiment."""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sae_jlens.analysis import save_plots, summarise_records  # noqa: E402
from sae_jlens.run_io import (  # noqa: E402
    append_jsonl,
    atomic_json,
    base_manifest,
    create_run_dir,
    read_jsonl,
)
from sae_jlens.synthetic import generate_records  # noqa: E402
from sae_jlens.real import collect_real_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("synthetic", "preflight", "full"), required=True
    )
    parser.add_argument(
        "--run-dir", type=Path, help="resume/write an explicit run directory"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_dir = args.run_dir or create_run_dir(
        Path(config["output"]["root"]),
        config["experiment"]["name"],
        args.stage,
        int(config["experiment"]["seed"]),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    for stale_marker in (run_dir / "SUCCESS", run_dir / "FAILED"):
        stale_marker.unlink(missing_ok=True)
    manifest = base_manifest(config, args.stage, sys.argv)
    atomic_json(run_dir / "manifest.json", manifest)
    records_path = run_dir / "records.jsonl"
    existing = read_jsonl(records_path)
    existing_keys = {
        (
            row.get("corpus", "synthetic"),
            int(row["sequence_id"]),
            int(row["position"]),
            int(row["layer"]),
            str(row["control"]),
        )
        for row in existing
    }

    def append_once(row: dict) -> None:
        key = (
            row.get("corpus", "synthetic"),
            int(row["sequence_id"]),
            int(row["position"]),
            int(row["layer"]),
            str(row["control"]),
        )
        if key not in existing_keys:
            append_jsonl(records_path, row)
            existing_keys.add(key)

    try:
        if args.stage == "synthetic":
            generated = generate_records(config)
            records = existing + generated
            for row in records:
                append_once(row)
        else:
            new_records = collect_real_records(
                args.stage,
                config,
                run_dir,
                append_once,
                existing_records=existing,
            )
            records = existing + new_records
        # Re-read the durable file so summaries never include suppressed
        # duplicates from a resumed invocation.
        records = read_jsonl(records_path)
        summary = summarise_records(records, config)
        atomic_json(run_dir / "summary.json", summary)
        manifest["plots"] = save_plots(summary, run_dir)
        manifest.update(
            status="SUCCESS",
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        atomic_json(run_dir / "manifest.json", manifest)
        (run_dir / "SUCCESS").touch()
        print(run_dir)
        return 0
    except Exception as exc:
        error = traceback.format_exc()
        (run_dir / "traceback.txt").write_text(error, encoding="utf-8")
        manifest.update(
            status="FAILED",
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            error=f"{type(exc).__name__}: {exc}",
        )
        atomic_json(run_dir / "manifest.json", manifest)
        (run_dir / "FAILED").touch()
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
