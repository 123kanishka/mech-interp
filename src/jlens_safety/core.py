"""Math and hooks shared by real inference and CPU correctness tests."""

from __future__ import annotations

import numpy as np
import torch


def shortlist_tokens(safe_mean, harmful_mean, special_ids, count=32):
    """Unchanged rule: absolute training mean J-logit contrast, no special IDs."""
    if safe_mean.ndim != 1 or safe_mean.shape != harmful_mean.shape:
        raise ValueError("Token means must be equal-length vocabulary vectors")
    contrast = (harmful_mean - safe_mean).abs()
    contrast[list(special_ids)] = -torch.inf
    if int(torch.isfinite(contrast).sum()) < count:
        raise ValueError("Fewer eligible vocabulary tokens than dictionary size")
    return contrast.topk(count).indices, contrast


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x)
    if not np.isfinite(norm) or norm < 1e-10:
        raise ValueError("Cannot steer with zero/nonfinite direction")
    return x / norm


def dictionary_projection(delta, dictionary):
    """Project contrast onto selected token covectors, not the whole J-space.

    Truncated SVD handles duplicate/nearly collinear token directions.
    Signed coefficients are allowed: this is a selected linear span, NOT the
    paper's sparse nonnegative decomposition of an entire activation.
    """
    dictionary = np.asarray(dictionary, dtype=np.float64)
    norms = np.linalg.norm(dictionary, axis=1, keepdims=True)
    if np.any(norms < 1e-12) or not np.isfinite(dictionary).all():
        raise ValueError("Invalid token dictionary")
    _, singular, vh = np.linalg.svd(dictionary / norms, full_matrices=False)
    basis = vh[singular > singular[0] * 1e-5]
    return unit(basis.T @ (basis @ delta))


def linear_probe(x, y, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(C=0.1, max_iter=1000, random_state=seed).fit(
        scaler.transform(x), y
    )
    weight = clf.coef_[0] / scaler.scale_
    bias = clf.intercept_[0] - np.dot(weight, scaler.mean_)
    return weight.astype(np.float32), float(bias)


def fit_layer(activations, labels, jlens_dictionary, logit_dictionary, seed, quantile):
    x, y = np.asarray(activations), np.asarray(labels)
    if set(y.tolist()) != {0, 1}:
        raise ValueError("Both training labels required")
    delta = x[y == 1].mean(0) - x[y == 0].mean(0)
    j_direction = dictionary_projection(delta, jlens_dictionary)
    raw_direction = unit(delta)
    logit_direction = dictionary_projection(delta, logit_dictionary)
    projection = x @ j_direction
    gate_mean = float(projection[y == 0].mean())
    gate_scale = max(float(projection[y == 0].std()), 1e-6)
    gate_threshold = float(
        np.quantile((projection[y == 0] - gate_mean) / gate_scale, quantile)
    )
    probe_weight, probe_bias = linear_probe(x, y, seed)
    return dict(
        j_direction=j_direction,
        residual_direction=raw_direction,
        j_dictionary=np.asarray(jlens_dictionary, dtype=np.float32),
        logit_dictionary=np.asarray(logit_dictionary, dtype=np.float32),
        training_delta=delta.astype(np.float32),
        logit_direction=logit_direction,
        gate_mean=gate_mean,
        gate_scale=gate_scale,
        gate_threshold=gate_threshold,
        residual_scale=float(np.median(np.linalg.norm(x, axis=1))),
        probe_weight=probe_weight,
        probe_bias=probe_bias,
    )


class Intervention:
    """One-sequence hook; gate is decided at prefill and frozen for decoding.

    Position scope is explicit; legacy default changes the last position only.
    The gate reads the unmodified residual, and alpha=0 returns output exactly.
    """

    def __init__(
        self,
        direction,
        scale,
        alpha,
        mode,
        gate=None,
        positions="last",
        norm_matching="per_position",
    ):
        if mode not in ("all", "prefill", "gated"):
            raise ValueError("Unknown intervention mode")
        self.direction = torch.as_tensor(direction, dtype=torch.float32)
        if positions not in ("last", "all") or norm_matching not in (
            "per_position",
            "total",
        ):
            raise ValueError("Unknown position scope or norm matching")
        self.positions, self.norm_matching = positions, norm_matching
        self.scale, self.alpha, self.mode = float(scale), float(alpha), mode
        self.gate = gate
        self.calls, self.applied, self.gate_on = 0, 0, None
        self.total_relative_norm = 0.0
        self.edited_positions, self.squared_update_norm = 0, 0.0

    def __call__(self, module, inputs, output):
        h = output if torch.is_tensor(output) else output[0]
        if h.ndim != 3 or h.shape[0] != 1:
            raise ValueError("Intervention expects one unpadded sequence [1,T,D]")
        first = self.calls == 0
        self.calls += 1
        if not first and h.shape[1] != 1:
            raise ValueError(
                "Decode requires use_cache=True; refusing repeated full-prefix steering"
            )
        if first:
            if self.gate is None:
                self.gate_on = True
            else:
                vector, mean, std, threshold = self.gate
                score = (
                    h[0, -1].float() @ torch.as_tensor(vector, device=h.device) - mean
                ) / std
                self.gate_on = bool(score.item() >= threshold)
        active = (self.mode != "prefill" or first) and (
            self.mode != "gated" or self.gate_on
        )
        if not active or self.alpha == 0:
            return output
        if self.direction.device != h.device:
            self.direction = self.direction.to(h.device)
        change = self.direction * (self.alpha * self.scale)
        count = h.shape[1] if self.positions == "all" else 1
        if self.norm_matching == "total":
            change = change / count**0.5
        selected = h if self.positions == "all" else h[:, -1:, :]
        self.total_relative_norm += float(
            (change.norm() / selected.float().norm(dim=-1).clamp_min(1e-9))
            .mean()
            .item()
        )
        self.edited_positions += count
        self.squared_update_norm += count * float(change.square().sum().item())
        changed = h.clone()
        if self.positions == "all":
            changed = (changed.float() + change).to(h.dtype)
        else:
            changed[:, -1, :] = (h[:, -1, :].float() + change).to(h.dtype)
        self.applied += 1
        return changed if torch.is_tensor(output) else (changed, *output[1:])


def intervention_from_condition(condition, artifacts):
    if condition["method"] == "baseline":
        return None
    layer = condition["layer"]
    a = artifacts[layer]
    method = condition["method"]
    if method == "random":
        direction = unit(
            np.random.default_rng(condition["random_seed"]).normal(
                size=len(a["j_direction"])
            )
        )
    elif method == "combined":
        # Norm-matched mixture of independently selected signed baseline and J direction.
        mixture = (
            a["j_direction"] * condition["j_alpha"]
            + a["residual_direction"] * condition["residual_alpha"]
        )
        if np.linalg.norm(mixture) < 1e-10:
            # Exact cancellation is a genuine no-op, not an arbitrary replacement direction.
            return Intervention(
                a["j_direction"], a["residual_scale"], 0.0, condition["mode"]
            )
        direction = unit(mixture)
    else:
        direction = a[
            {
                "jlens": "j_direction",
                "residual": "residual_direction",
                "logit": "logit_direction",
            }[method]
        ]
    gate = (a["j_direction"], a["gate_mean"], a["gate_scale"], a["gate_threshold"])
    return Intervention(
        direction,
        a["residual_scale"],
        condition["alpha"],
        condition["mode"],
        gate=gate if condition["mode"] == "gated" else None,
        positions=condition.get("positions", "last"),
        norm_matching=condition.get("norm_matching", "per_position"),
    )
