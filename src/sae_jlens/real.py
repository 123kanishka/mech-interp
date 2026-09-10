"""Real-model collection for the GPU preflight and full experiment."""

from __future__ import annotations

import random
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .metrics import (
    compare_distributions,
    ranked_future_metrics,
    weighted_jaccard_similarity,
)
from .run_io import append_jsonl, atomic_json


def collect_real_records(
    stage: str,
    config: dict[str, Any],
    run_dir: Path,
    on_record: Callable[[dict[str, Any]], None],
    existing_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Collect records without retaining full-vocabulary tensors on disk."""
    import torch
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from jlens import JacobianLens, from_hf
    from jlens.hooks import ActivationRecorder
    from sae_lens import SAE
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("real stages require a CUDA GPU")
    free_bytes = shutil.disk_usage(run_dir).free
    required_free = int(config["output"]["min_free_gb_before_download"]) * 1024**3
    if free_bytes < required_free:
        raise RuntimeError(
            f"only {free_bytes / 1024**3:.1f} GiB free; require at least "
            f"{required_free / 1024**3:.0f} GiB before artifact downloads"
        )

    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False

    model_cfg = config["model"]
    dtype = getattr(torch, model_cfg["dtype"])
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"], dtype=dtype, low_cpu_mem_usage=True
    ).to(model_cfg["device"])
    model = from_hf(hf_model, tokenizer)

    lens_cfg = config["jlens"]
    lens = JacobianLens.from_pretrained(
        lens_cfg["repo"],
        filename=lens_cfg["filename"],
        revision=lens_cfg["revision"],
    )
    layers = [int(layer) for layer in config["sae"]["layers"]]
    _validate_lens(model, lens, layers)

    sae_root = Path(
        snapshot_download(
            config["sae"]["repo"],
            revision=config["sae"]["revision"],
            allow_patterns=[f"{path}/*" for path in config["sae"]["ids"].values()],
        )
    )
    saes = {
        layer: SAE.load_from_disk(
            sae_root / config["sae"]["ids"][str(layer)],
            device=model_cfg["device"],
            dtype=model_cfg["dtype"],
        ).eval()
        for layer in layers
    }
    _validate_saes(model, saes, layers)

    count = (
        int(config["dataset"]["preflight_sequences"])
        if stage == "preflight"
        else int(config["dataset"]["full_sequences_per_corpus"])
    )
    sequence_length = (
        int(config["dataset"]["preflight_sequence_length"])
        if stage == "preflight"
        else int(config["dataset"]["sequence_length"])
    )
    prompts_path = run_dir / "prompts.jsonl"
    if prompts_path.exists():
        import json

        prompt_rows = [json.loads(line) for line in prompts_path.read_text().splitlines() if line]
    else:
        prompt_rows: list[dict[str, Any]] = []
        for corpus_index, corpus in enumerate(config["dataset"]["corpora"]):
            ds = load_dataset(
                corpus["name"],
                corpus.get("subset"),
                split=corpus["split"],
                streaming=bool(corpus.get("streaming", False)),
            )
            ds = _shuffle_dataset(
                ds,
                seed=seed + corpus_index,
                streaming=bool(corpus.get("streaming", False)),
            )
            prompt_rows.extend(
                _take_tokenizable_texts(
                    ds, corpus, tokenizer, config, count, sequence_length
                )
            )
        for row in prompt_rows:
            append_jsonl(prompts_path, row)

    records: list[dict[str, Any]] = []
    feature_ids = {layer: set() for layer in layers}
    residual_norms = {layer: [] for layer in layers}
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    existing_records = existing_records or []
    for row in existing_records:
        if row.get("control") != "sae_reconstruction":
            continue
        layer = int(row["layer"])
        feature_ids[layer].update(int(value) for value in row.get("active_feature_ids", []))
        if "residual_norm" in row:
            residual_norms[layer].append(float(row["residual_norm"]))
    audit_cap = int(config["analysis"]["feature_audit_max_features_per_layer"])
    for layer in layers:
        feature_ids[layer] = set(sorted(feature_ids[layer])[:audit_cap])
    completed = _completed_sequence_ids(existing_records, config)
    for sequence_id, prompt_row in enumerate(prompt_rows):
        if sequence_id in completed:
            continue
        sequence_records = _score_sequence(
            sequence_id,
            prompt_row,
            tokenizer,
            model,
            lens,
            saes,
            layers,
            config,
            ActivationRecorder,
            sequence_length,
            feature_ids,
            residual_norms,
        )
        for record in sequence_records:
            records.append(record)
            on_record(record)
        if (sequence_id + 1) % int(config["output"]["checkpoint_every_sequences"]) == 0:
            atomic_json(
                run_dir / "progress.json",
                {
                    "sequences_complete": sequence_id + 1,
                    "records_complete": len(records),
                    "elapsed_seconds": time.monotonic() - started,
                },
            )

    feature_path = run_dir / "feature_tokenizability.jsonl"
    feature_path.unlink(missing_ok=True)
    feature_summary = _audit_feature_tokenizability(
        feature_ids, residual_norms, saes, lens, model, tokenizer, config, run_dir
    )
    atomic_json(run_dir / "feature_tokenizability_summary.json", feature_summary)
    atomic_json(
        run_dir / "resource_usage.json",
        {
            "elapsed_seconds": time.monotonic() - started,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "n_sequences": len(prompt_rows),
            "n_records": len(records),
        },
    )
    return records


def _shuffle_dataset(dataset, seed: int, streaming: bool):
    """Use only shuffle arguments supported by the selected dataset mode."""
    if streaming:
        return dataset.shuffle(seed=seed, buffer_size=10_000)
    return dataset.shuffle(seed=seed)


def _estimate_unigram(prompt_rows, tokenizer, config, vocab_size: int) -> np.ndarray:
    counts = np.ones(vocab_size, dtype=np.float64)
    max_length = int(config["dataset"]["sequence_length"])
    for row in prompt_rows:
        ids = tokenizer(
            row["text"], add_special_tokens=True, truncation=True, max_length=max_length
        )["input_ids"]
        valid = np.asarray([token for token in ids if 0 <= token < vocab_size])
        if valid.size:
            counts += np.bincount(valid, minlength=vocab_size)
    return counts / counts.sum()


def _completed_sequence_ids(records: list[dict], config: dict) -> set[int]:
    expected = (
        int(config["analysis"]["positions_per_sequence"])
        * len(config["sae"]["layers"])
        * 6
    )
    counts: dict[int, int] = {}
    for row in records:
        sequence_id = int(row["sequence_id"])
        counts[sequence_id] = counts.get(sequence_id, 0) + 1
    return {sequence_id for sequence_id, count in counts.items() if count >= expected}


def _take_tokenizable_texts(
    ds, corpus, tokenizer, config, count: int, max_tokens: int
) -> list[dict]:
    rows = []
    min_tokens = int(config["dataset"]["min_non_special_tokens"])
    for item in ds:
        text = str(item.get(corpus["text_field"], "")).strip()
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=True, truncation=True, max_length=max_tokens)[
            "input_ids"
        ]
        if len(ids) < min_tokens:
            continue
        rows.append(
            {
                "corpus": corpus["name"],
                "corpus_role": corpus["role"],
                "text": text,
                "token_count": len(ids),
            }
        )
        if len(rows) >= count:
            return rows
    raise RuntimeError(f"corpus {corpus['name']} yielded only {len(rows)}/{count} texts")


def _score_sequence(
    sequence_id,
    prompt_row,
    tokenizer,
    model,
    lens,
    saes,
    layers,
    config,
    activation_recorder_cls,
    sequence_length_limit,
    feature_ids,
    residual_norms,
) -> list[dict]:
    import torch

    input_ids = tokenizer(
        prompt_row["text"],
        return_tensors="pt",
        truncation=True,
        max_length=int(sequence_length_limit),
    ).input_ids.to(model.input_device)
    sequence_length = input_ids.shape[1]
    future_horizon = int(config["analysis"]["future_token_horizon"])
    candidate_positions = list(range(0, sequence_length - future_horizon))
    position_count = min(
        int(config["analysis"]["positions_per_sequence"]), len(candidate_positions)
    )
    rng = np.random.default_rng(
        int(config["experiment"]["seed"]) * 1_000_003 + sequence_id
    )
    positions = sorted(rng.choice(candidate_positions, position_count, replace=False).tolist())

    final_layer = model.n_layers - 1
    record_at = sorted(set(layers) | {final_layer})
    with torch.inference_mode(), activation_recorder_cls(model.layers, at=record_at) as recorder:
        model.forward(input_ids)
        activations = {
            layer: recorder.activations[layer][0, positions].float()
            for layer in record_at
        }
        # Mirror JacobianLens.apply: read the final residual block output and
        # apply the wrapper's final norm/unembedding exactly once.
        model_logits = model.unembed(activations[final_layer]).float().cpu()

    analysis = config["analysis"]
    records: list[dict] = []
    for layer in layers:
        residual = activations[layer]
        with torch.inference_mode():
            feature_acts = saes[layer].encode(residual.to(model_cfg_dtype(saes[layer])))
            reconstruction = saes[layer].decode(feature_acts).float()
            feature_distributions, feature_metadata = _feature_token_distributions(
                feature_acts,
                saes[layer].W_dec,
                model,
                int(analysis["feature_top_k"]),
                vocab_size=int(model_logits.shape[-1]),
            )
            jlens_probabilities = torch.softmax(
                model.unembed(lens.transport(residual, layer)).float(), dim=-1
            ).cpu()
            reconstruction_probabilities = torch.softmax(
                model.unembed(reconstruction).float(), dim=-1
            ).cpu()
            model_probabilities = torch.softmax(model_logits, dim=-1)

        reconstruction_cosine = torch.nn.functional.cosine_similarity(
            residual, reconstruction, dim=-1
        ).cpu()
        normalised_mse = (
            (residual - reconstruction).square().mean(dim=-1)
            / residual.square().mean(dim=-1).clamp_min(1e-12)
        ).cpu()
        active_counts = (feature_acts > float(config["sae"]["active_threshold"])).sum(-1).cpu()
        residual_norms[layer].extend(residual.norm(dim=-1).detach().cpu().tolist())
        audit_cap = int(config["analysis"]["feature_audit_max_features_per_layer"])
        if len(feature_ids[layer]) < audit_cap:
            active_indices = torch.nonzero(feature_acts > 0, as_tuple=False)[:, 1]
            feature_ids[layer].update(
                int(index) for index in active_indices.detach().cpu().tolist()
            )
            if len(feature_ids[layer]) > audit_cap:
                feature_ids[layer] = set(sorted(feature_ids[layer])[:audit_cap])

        for row_index, position in enumerate(positions):
            future_ids = input_ids[
                0, position + 1 : position + 1 + future_horizon
            ].detach().cpu().tolist()
            target = int(future_ids[0])
            jlens_distribution, jlens_ranked = _truncate_distribution(
                jlens_probabilities[row_index].numpy(), int(analysis["jlens_top_k"])
            )
            feature_info = feature_metadata[row_index]
            feature_tokens = feature_info["feature_token_ids"]
            feature_weights = feature_info["feature_weights"]
            shuffled_distribution = _sparse_token_distribution(
                list(reversed(feature_tokens)), feature_weights, len(jlens_distribution)
            )
            control_rng = np.random.default_rng(
                int(config["experiment"]["seed"]) * 1_000_003
                + sequence_id * 10_007
                + layer * 131
                + position
            )
            random_tokens = control_rng.choice(
                len(jlens_distribution), size=len(feature_tokens), replace=False
            ).tolist()
            random_distribution = _sparse_token_distribution(
                random_tokens, feature_weights, len(jlens_distribution)
            )
            row_distributions = {
                "sae_feature_tokens": feature_distributions[row_index],
                "jlens": jlens_distribution,
                "shuffled_feature_labels_control": shuffled_distribution,
                "random_feature_labels_control": random_distribution,
                "sae_reconstruction_control": reconstruction_probabilities[row_index].numpy(),
                "model_control": model_probabilities[row_index].numpy(),
            }
            ranked_tokens = {
                "sae_feature_tokens": feature_info["feature_token_ids"],
                "jlens": jlens_ranked,
                "shuffled_feature_labels_control": list(reversed(feature_tokens)),
                "random_feature_labels_control": random_tokens,
                "sae_reconstruction_control": np.argsort(
                    -row_distributions["sae_reconstruction_control"]
                )[: int(analysis["feature_top_k"])].tolist(),
                "model_control": np.argsort(-row_distributions["model_control"])[
                    : int(analysis["feature_top_k"])
                ].tolist(),
            }
            reference = jlens_distribution
            common = {
                "sequence_id": sequence_id,
                "corpus": prompt_row["corpus"],
                "corpus_role": prompt_row["corpus_role"],
                "position": position,
                "source_token_id": int(input_ids[0, position]),
                "target_token_id": target,
                "future_token_ids": future_ids,
                "future_tokens": [tokenizer.decode([token]) for token in future_ids],
                "layer": layer,
                "reconstruction_cosine": float(reconstruction_cosine[row_index]),
                "normalised_mse": float(normalised_mse[row_index]),
                "active_feature_count": int(active_counts[row_index]),
                "residual_norm": float(residual[row_index].norm().detach().cpu()),
            }
            save_top_k = int(analysis["save_top_tokens"])
            ref_top = np.argsort(-reference)[:save_top_k]
            for control, q_distribution in row_distributions.items():
                metrics = compare_distributions(
                    reference,
                    q_distribution,
                    target,
                    top_k=int(analysis["top_k"]),
                    rbo_p=float(analysis["rbo_p"]),
                )
                metrics.update(
                    {
                        "weighted_jaccard": weighted_jaccard_similarity(
                            reference, q_distribution
                        ),
                        "future_token_mass": float(
                            q_distribution[list(set(future_ids))].sum()
                        ),
                        "reference_top_token_ids": ref_top.tolist(),
                        "reference_top_probabilities": reference[ref_top].tolist(),
                    }
                )
                metrics.update(
                    ranked_future_metrics(
                        ranked_tokens[control],
                        future_ids,
                        int(analysis["feature_top_k"]),
                    )
                )
                q_top = np.asarray(ranked_tokens[control][:save_top_k], dtype=int)
                metrics["q_top_token_ids"] = q_top.tolist()
                metrics["q_top_tokens"] = [
                    tokenizer.decode([int(token)]) for token in q_top
                ]
                metrics["q_top_probabilities"] = q_distribution[q_top].tolist()
                row = {**common, "control": control, **metrics}
                if control == "sae_feature_tokens":
                    row.update(feature_info)
                records.append(row)
    return records


def _feature_token_distributions(feature_acts, decoder, model, k: int, vocab_size: int):
    """Map top fired SAE features independently to tokens, then weight their labels."""
    import torch

    distributions = []
    metadata = []
    for activations in feature_acts:
        positive = torch.clamp(activations.float(), min=0)
        active = torch.nonzero(positive > 0, as_tuple=False).flatten()
        if not active.numel():
            active = torch.topk(activations.float(), k=1).indices
            positive = torch.clamp(activations.float(), min=0) + 1e-12
        count = min(int(k), int(active.numel()))
        values, order = torch.topk(positive[active], k=count)
        feature_ids = active[order]
        weights = values / values.sum().clamp_min(1e-12)
        directions = decoder[feature_ids].float()
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        feature_logits = model.unembed(directions).float()
        token_ids = feature_logits.argmax(dim=-1)
        distribution = torch.zeros(vocab_size, device=token_ids.device, dtype=torch.float32)
        distribution.scatter_add_(0, token_ids, weights)
        distributions.append(distribution.cpu().numpy())
        metadata.append(
            {
                "active_feature_ids": feature_ids.detach().cpu().tolist(),
                "active_feature_values": values.detach().cpu().tolist(),
                "feature_weights": weights.detach().cpu().tolist(),
                "feature_token_ids": token_ids.detach().cpu().tolist(),
            }
        )
    return distributions, metadata


def _truncate_distribution(probabilities: np.ndarray, k: int):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    token_ids = np.argsort(-probabilities)[: int(k)]
    truncated = np.zeros_like(probabilities)
    truncated[token_ids] = probabilities[token_ids]
    truncated /= truncated.sum()
    return truncated, token_ids.tolist()


def _sparse_token_distribution(token_ids, weights, vocab_size: int) -> np.ndarray:
    distribution = np.zeros(int(vocab_size), dtype=np.float64)
    np.add.at(distribution, np.asarray(token_ids, dtype=int), np.asarray(weights))
    return distribution / distribution.sum()


def _audit_feature_tokenizability(
    feature_ids, residual_norms, saes, _lens, model, tokenizer, config, run_dir
) -> dict:
    """Audit a bounded feature sample using direct unembedding only."""
    import torch

    top_k = int(config["analysis"]["feature_label_top_k"])
    threshold = float(config["analysis"]["feature_label_min_concentration"])
    summary = {}
    for layer, identifiers in feature_ids.items():
        rows = []
        scale = float(np.median(residual_norms[layer])) if residual_norms[layer] else 1.0
        decoder = saes[layer].W_dec
        for feature_id in sorted(identifiers):
            with torch.inference_mode():
                direction = decoder[feature_id].float()
                direction = direction * scale / direction.norm().clamp_min(1e-8)
                # Independent feature label: ordinary unembedding only.
                logits = model.unembed(direction[None])[0].float()
                probabilities = torch.softmax(logits, dim=-1)
                values, indices = probabilities.topk(top_k)
            concentration = float(values.sum().cpu())
            row = {
                "layer": layer,
                "feature_id": feature_id,
                "top_k_probability_mass": concentration,
                "token_associated": concentration >= threshold,
                "top_token_ids": indices.cpu().tolist(),
                "top_tokens": [tokenizer.decode([int(i)]) for i in indices.cpu()],
            }
            rows.append(row)
            append_jsonl(run_dir / "feature_tokenizability.jsonl", row)
        non_tokenizable = sum(not row["token_associated"] for row in rows)
        summary[str(layer)] = {
            "sampled_features": len(rows),
            "non_tokenizable_features": non_tokenizable,
            "non_tokenizable_percent": (
                100.0 * non_tokenizable / len(rows) if rows else None
            ),
            "definition": f"top-{top_k} direct-unembedding probability mass < {threshold}",
        }
    return summary


def model_cfg_dtype(sae):
    """Return the parameter dtype without depending on SAE config internals."""
    return next(sae.parameters()).dtype


def _validate_lens(model, lens, layers: list[int]) -> None:
    if lens.d_model != model.d_model:
        raise ValueError(f"J-Lens width {lens.d_model} != model width {model.d_model}")
    missing = sorted(set(layers) - set(lens.source_layers))
    if missing:
        raise ValueError(f"J-Lens is missing required layers: {missing}")
    if lens.n_prompts < 1000:
        raise ValueError(f"expected 1000-prompt J-Lens, found {lens.n_prompts}")


def _validate_saes(model, saes: dict, layers: list[int]) -> None:
    for layer in layers:
        sae = saes[layer]
        d_in = int(getattr(sae.cfg, "d_in"))
        if d_in != model.d_model:
            raise ValueError(f"layer {layer} SAE width {d_in} != model width {model.d_model}")
        hook_name = str(getattr(sae.cfg, "hook_name", ""))
        if hook_name and not hook_name.endswith(f"layers.{layer}"):
            raise ValueError(f"layer {layer} SAE hook mismatch: {hook_name}")
