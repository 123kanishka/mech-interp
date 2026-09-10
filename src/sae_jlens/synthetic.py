"""Deterministic synthetic validation of the analysis path."""

from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import compare_distributions, softmax


def generate_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    seed = int(config["experiment"]["seed"])
    analysis = config["analysis"]
    rng = np.random.default_rng(seed)
    vocab = 512
    d_model = 64
    layers = config["sae"]["layers"]
    unembed = rng.normal(size=(d_model, vocab)) / np.sqrt(d_model)
    records: list[dict[str, Any]] = []

    for sequence_id in range(12):
        for position in range(8):
            target = int(rng.integers(vocab))
            base = rng.normal(size=d_model)
            for layer in layers:
                depth = (layer + 1) / (max(layers) + 1)
                original_logits = base @ unembed
                original_logits[target] += 2.0 * depth
                noise_scale = 0.8 * (1.0 - depth) + 0.08
                reconstruction = base + rng.normal(
                    scale=noise_scale, size=d_model
                )
                sae_logits = reconstruction @ unembed
                metrics = compare_distributions(
                    softmax(original_logits),
                    softmax(sae_logits),
                    target,
                    top_k=int(analysis["top_k"]),
                    rbo_p=float(analysis["rbo_p"]),
                )
                records.append(
                    {
                        "sequence_id": sequence_id,
                        "position": position,
                        "layer": layer,
                        "target_token_id": target,
                        "control": "sae_reconstruction",
                        **metrics,
                    }
                )
    return records
