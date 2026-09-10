"""Aggregation and plotting from durable per-position records."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import bootstrap_mean_ci


METRICS = (
    "js_similarity",
    "top_k_overlap",
    "rbo",
    "q_target_nll",
    "q_target_rr",
    "reconstruction_cosine",
    "normalised_mse",
    "frequency_adjusted_js_similarity",
    "frequency_adjusted_q_target_nll",
    "frequency_adjusted_q_target_rr",
)


def summarise_records(records: list[dict], config: dict) -> dict[str, Any]:
    """Bootstrap sequence-level means, not correlated token positions."""
    per_sequence: dict[tuple, list[float]] = defaultdict(list)
    for row in records:
        for metric in METRICS:
            value = row.get(metric)
            if value is None or not np.isfinite(value):
                continue
            key = (
                row.get("corpus", "synthetic"),
                row.get("corpus_role", "synthetic"),
                str(row["control"]),
                int(row["layer"]),
                int(row["sequence_id"]),
                metric,
            )
            per_sequence[key].append(float(value))

    grouped: dict[tuple, list[float]] = defaultdict(list)
    for key, values in per_sequence.items():
        corpus, role, control, layer, _, metric = key
        grouped[(corpus, role, control, layer, metric)].append(float(np.mean(values)))

    analysis = config["analysis"]
    seed = int(config["experiment"]["seed"])
    rows = []
    for index, (key, values) in enumerate(sorted(grouped.items())):
        corpus, role, control, layer, metric = key
        mean, low, high = bootstrap_mean_ci(
            values,
            samples=int(analysis["bootstrap_samples"]),
            confidence=float(analysis["confidence_level"]),
            seed=seed + index,
        )
        rows.append(
            {
                "corpus": corpus,
                "corpus_role": role,
                "control": control,
                "layer": layer,
                "metric": metric,
                "n_sequences": len(values),
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
            }
        )
    token_counts: dict[tuple, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for row in records:
        key = (
            row.get("corpus", "synthetic"),
            str(row["control"]),
            int(row["layer"]),
        )
        for token_id in row.get("q_top_token_ids", []):
            token_counts[key][int(token_id)] += 1
    top_token_frequencies = []
    for (corpus, control, layer), counts in sorted(token_counts.items()):
        top_token_frequencies.append(
            {
                "corpus": corpus,
                "control": control,
                "layer": layer,
                "tokens": [
                    {"token_id": token_id, "top_k_occurrences": count}
                    for token_id, count in sorted(
                        counts.items(), key=lambda item: (-item[1], item[0])
                    )[:50]
                ],
            }
        )
    return {
        "n_records": len(records),
        "n_sequences": len(
            {(r.get("corpus", "synthetic"), r["sequence_id"]) for r in records}
        ),
        "bootstrap_unit": "sequence",
        "estimates": rows,
        "top_token_frequencies": top_token_frequencies,
    }


def save_plots(summary: dict[str, Any], run_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    outputs = []
    for metric, ylabel in (
        ("js_similarity", "Jensen-Shannon similarity"),
        ("q_target_nll", "Target-token NLL"),
        ("q_target_rr", "Target-token reciprocal rank"),
    ):
        rows = [row for row in summary["estimates"] if row["metric"] == metric]
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        series: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in rows:
            series[(row["corpus"], row["control"])].append(row)
        for (corpus, control), points in sorted(series.items()):
            points.sort(key=lambda item: item["layer"])
            x = [item["layer"] for item in points]
            y = [item["mean"] for item in points]
            low = [item["mean"] - item["ci_low"] for item in points]
            high = [item["ci_high"] - item["mean"] for item in points]
            ax.errorbar(x, y, yerr=[low, high], marker="o", capsize=3,
                        label=f"{corpus}: {control}")
        ax.set_xlabel("Residual-stream layer")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by layer")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = plots_dir / f"{metric}_by_layer.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(str(path))
    return outputs
