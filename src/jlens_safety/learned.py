"""Joint signed token coefficients; the language model remains frozen."""

from __future__ import annotations

import math
import time
import numpy as np
import torch
import torch.nn.functional as F

from .common import atomic_npz, digest
from .core import Intervention


def normalized_dictionary(artifacts, layer, method, seed):
    a = artifacts[layer]
    if method == "random_tokens":
        raw = np.random.default_rng(seed).normal(size=a["j_dictionary"].shape)
    elif method in ("jlens_tokens", "logit_tokens"):
        raw = a["j_dictionary" if method == "jlens_tokens" else "logit_dictionary"]
    else:
        raise ValueError("Unknown coefficient dictionary")
    raw = np.asarray(raw, dtype=np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    if not np.isfinite(raw).all() or np.any(norms < 1e-10):
        raise ValueError("Invalid token dictionary")
    return raw / norms


def token_intervention(condition, artifacts):
    dictionary = normalized_dictionary(
        artifacts, condition["layer"], condition["method"], condition["seed"]
    )
    beta = np.asarray(condition["coefficients"], dtype=np.float32)
    if beta.shape != (len(dictionary),) or not np.isfinite(beta).all():
        raise ValueError("Invalid signed coefficients")
    update = beta @ dictionary
    if np.linalg.norm(update) > condition["max_update_norm"] + 1e-5:
        raise ValueError("Coefficient artifact exceeds registered perturbation budget")
    return Intervention(
        update,
        artifacts[condition["layer"]]["residual_scale"],
        1.0,
        "all",
        positions=condition["positions"],
        norm_matching=condition.get("norm_matching", "per_position"),
    )


def steering_mask(length, prompt_length, positions, device=None):
    """Teacher forcing edits prompt end + every response-input token in last mode.

    This matches cached decoding: earlier prompt positions are untouched, while
    every newly processed response token is steered exactly once.
    """
    if positions not in ("last", "all") or not 1 <= prompt_length <= length:
        raise ValueError("Invalid position scope or prompt length")
    mask = torch.ones(length, device=device)
    if positions == "last":
        mask[: prompt_length - 1] = 0
    return mask[None, :, None]


def differentiable_hook(beta, dictionary, scale, prompt_length, positions):
    def hook(module, inputs, output):
        h = output if torch.is_tensor(output) else output[0]
        if h.ndim != 3 or h.shape[0] != 1:
            raise ValueError("Training expects a single unpadded sequence")
        mask = steering_mask(h.shape[1], prompt_length, positions, h.device)
        changed = (h.float() + mask * (beta @ dictionary) * scale).to(h.dtype)
        return changed if torch.is_tensor(output) else (changed, *output[1:])

    return hook


@torch.no_grad()
def constrain_coefficients(beta, dictionary, maximum):
    # Preserve relative signed weights while constraining actual vector norm.
    norm = (beta @ dictionary).norm()
    if not torch.isfinite(norm):
        raise ValueError("Nonfinite learned intervention")
    if norm > maximum:
        beta.mul_(maximum / norm)


def training_tensors(target, row, max_completion_tokens):
    prompt = target.encode(row["prompt"])["input_ids"]
    completion = target.tokenizer(row["completion"], add_special_tokens=False)[
        "input_ids"
    ]
    eos = target.tokenizer.eos_token_id
    if not completion:
        raise ValueError("Empty reference completion")
    if eos is not None and completion[-1] != eos:
        completion.append(eos)
    truncated = len(completion) > max_completion_tokens
    completion = completion[:max_completion_tokens]
    labels = torch.tensor([completion], device=prompt.device)
    # Last prompt token predicts first completion token. The last reference token
    # is a label only; it is not needed as an input.
    ids = torch.cat([prompt, labels[:, :-1]], dim=1)
    return ids, labels, prompt.shape[1], truncated


def coefficient_loss(target, row, beta, dictionary, layer, positions, config):
    c = config["learning"]
    ids, labels, prompt_length, truncated = training_tensors(
        target, row, c["max_completion_tokens"]
    )
    n = labels.shape[1]
    baseline = None
    if not row["harmful"] and c["benign_kl_weight"]:
        with torch.no_grad():
            baseline = (
                target.model(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    use_cache=False,
                    logits_to_keep=n,
                )
                .logits.float()
                .log_softmax(-1)
            )
    hook = target.layers[layer].register_forward_hook(
        differentiable_hook(
            beta,
            dictionary,
            target.artifacts[layer]["residual_scale"],
            prompt_length,
            positions,
        )
    )
    try:
        logits = target.model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
            logits_to_keep=n,
        ).logits.float()
        if logits.shape[1] != n:
            raise ValueError(
                "Model did not honor completion-only logits; check adapter"
            )
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        kl = logits.new_zeros(())
        if baseline is not None:
            kl = (
                F.kl_div(
                    logits.log_softmax(-1), baseline, log_target=True, reduction="sum"
                )
                / n
            )
        loss = (
            nll + c["benign_kl_weight"] * kl + c["coefficient_l2"] * beta.square().sum()
        )
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite coefficient loss")
        return loss, dict(
            nll=float(nll.detach()), kl=float(kl.detach()), truncated=truncated
        )
    finally:
        hook.remove()


def fit_coefficients(
    target, rows, condition, config, checkpoint, budget, max_examples=None
):
    """Small pickle-free Adam checkpoint; save only at accumulation boundaries.

    Failed partial accumulation is replayed after resume. Model weights are never
    optimizer parameters. Seeds control row order, not final-test sampling.
    """
    if any(r["split"] != "train" for r in rows):
        raise ValueError("Coefficient fitting may only see training rows")
    c = config["learning"]
    for parameter in target.model.parameters():
        parameter.requires_grad_(False)
    target.model.eval()
    layer, seed = condition["layer"], condition["seed"]
    device = next(target.model.parameters()).device
    dictionary = torch.as_tensor(
        normalized_dictionary(target.artifacts, layer, condition["method"], seed),
        device=device,
    )
    beta = torch.nn.Parameter(torch.zeros(len(dictionary), device=dictionary.device))
    optimizer = torch.optim.Adam([beta], lr=c["learning_rate"])
    schedule = []
    for epoch in range(c["epochs"]):
        order = np.random.default_rng(seed + epoch).permutation(len(rows))
        schedule.extend(int(i) for i in order)
    if max_examples is not None:
        schedule = schedule[:max_examples]
    identity = digest(
        dict(
            condition=condition,
            ids=[r["id"] for r in rows],
            learning=c,
            max_examples=max_examples,
        )
    )
    cursor, updates, truncated, seconds, nll_sum = 0, 0, 0, 0.0, 0.0
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            if str(saved["identity"]) != identity:
                raise ValueError("Training checkpoint identity changed")
            beta.data.copy_(torch.as_tensor(saved["beta"], device=beta.device))
            cursor, updates = int(saved["cursor"]), int(saved["updates"])
            truncated, seconds, nll_sum = (
                int(saved["truncated"]),
                float(saved["seconds"]),
                float(saved["nll_sum"]),
            )
            if updates:
                optimizer.state[beta] = dict(
                    step=torch.tensor(float(updates)),
                    exp_avg=torch.as_tensor(saved["exp_avg"], device=beta.device),
                    exp_avg_sq=torch.as_tensor(saved["exp_avg_sq"], device=beta.device),
                )
    started = time.monotonic()

    def save():
        state = optimizer.state.get(beta, {})
        zeros = torch.zeros_like(beta)
        atomic_npz(
            checkpoint,
            identity=np.asarray(identity),
            beta=beta.detach().cpu().numpy(),
            cursor=np.asarray(cursor),
            updates=np.asarray(updates),
            truncated=np.asarray(truncated),
            seconds=np.asarray(seconds + time.monotonic() - started),
            nll_sum=np.asarray(nll_sum),
            exp_avg=state.get("exp_avg", zeros).cpu().numpy(),
            exp_avg_sq=state.get("exp_avg_sq", zeros).cpu().numpy(),
        )

    while cursor < len(schedule):
        budget.check()
        batch = schedule[cursor : cursor + c["accumulation_steps"]]
        optimizer.zero_grad(set_to_none=True)
        for index in batch:
            loss, metrics = coefficient_loss(
                target,
                rows[index],
                beta,
                dictionary,
                layer,
                condition["positions"],
                config,
            )
            (loss / len(batch)).backward()
            truncated += int(metrics["truncated"])
            nll_sum += metrics["nll"]
        if beta.grad is None or not torch.isfinite(beta.grad).all():
            raise ValueError(
                "Missing/nonfinite beta gradient; GPU backward preflight failed"
            )
        torch.nn.utils.clip_grad_norm_([beta], 1.0)
        optimizer.step()
        constrain_coefficients(beta, dictionary, c["max_update_norm"])
        cursor += len(batch)
        updates += 1
        if updates % c["save_every_steps"] == 0 or cursor == len(schedule):
            save()
    return dict(
        coefficients=beta.detach().cpu().tolist(),
        max_update_norm=c["max_update_norm"],
        training_examples=cursor,
        optimizer_steps=updates,
        training_reference_truncated=truncated,
        training_mean_nll=nll_sum / max(1, cursor),
        training_seconds=seconds + time.monotonic() - started,
        update_norm=float((beta @ dictionary).norm().detach()),
        coefficient_norm=float(beta.norm().detach()),
    )


def paired_power_plan(differences, target_effect=0.05, power=0.8, alpha=0.05):
    """Normal-approximation planning on independent paired GROUP averages.

    Pilot only, never a post-hoc justification of test significance. Zero
    discordance is uninformative, not evidence that one observation suffices.
    """
    from scipy.stats import norm

    values = np.asarray(differences, dtype=float)
    if len(values) < 2 or np.var(values, ddof=1) == 0:
        return dict(required_groups=None, status="insufficient_pilot_discordance")
    n = math.ceil(
        (norm.ppf(1 - alpha / 2) + norm.ppf(power)) ** 2
        * float(np.var(values, ddof=1))
        / target_effect**2
    )
    return dict(
        required_groups=max(2, n),
        status="approximate_planning_only",
        target_absolute_effect=target_effect,
        alpha=alpha,
        power=power,
    )
