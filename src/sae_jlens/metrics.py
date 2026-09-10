"""Numerically stable, dependency-light vocabulary metrics."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

EPS = 1e-12


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def frequency_adjust_logits(
    logits: np.ndarray, unigram_probabilities: np.ndarray, alpha: float = 1.0
) -> np.ndarray:
    """Remove a configurable unigram-frequency prior from token logits."""
    probs = np.asarray(unigram_probabilities, dtype=np.float64)
    if np.any(probs < 0) or not np.isfinite(probs).all() or probs.sum() <= 0:
        raise ValueError("unigram probabilities must be finite and non-negative")
    probs = probs / probs.sum()
    return np.asarray(logits, dtype=np.float64) - alpha * np.log(probs + EPS)


def jensen_shannon_similarity(p: np.ndarray, q: np.ndarray) -> float:
    p = _normalise(p)
    q = _normalise(q)
    m = 0.5 * (p + q)
    divergence = 0.5 * (_kl(p, m) + _kl(q, m))
    return float(1.0 - divergence / math.log(2.0))


def top_k_overlap(p: np.ndarray, q: np.ndarray, k: int) -> float:
    k = min(int(k), p.size)
    if k <= 0:
        raise ValueError("k must be positive")
    a = set(np.argpartition(p, -k)[-k:].tolist())
    b = set(np.argpartition(q, -k)[-k:].tolist())
    return len(a & b) / k


def rank_biased_overlap(
    p: np.ndarray, q: np.ndarray, k: int, persistence: float
) -> float:
    """Finite-depth extrapolated RBO, bounded in [0, 1]."""
    if not 0.0 < persistence < 1.0:
        raise ValueError("persistence must lie in (0, 1)")
    k = min(int(k), p.size)
    rp = np.argsort(-p)[:k]
    rq = np.argsort(-q)[:k]
    left: set[int] = set()
    right: set[int] = set()
    weighted = 0.0
    agreement = 0.0
    for depth in range(1, k + 1):
        left.add(int(rp[depth - 1]))
        right.add(int(rq[depth - 1]))
        agreement = len(left & right) / depth
        weighted += agreement * (persistence ** (depth - 1))
    return float((1.0 - persistence) * weighted + agreement * persistence**k)


def reciprocal_rank(probabilities: np.ndarray, target: int) -> float:
    target_score = probabilities[int(target)]
    rank = 1 + int(np.sum(probabilities > target_score))
    return 1.0 / rank


def compare_distributions(
    p: np.ndarray,
    q: np.ndarray,
    target: int,
    *,
    top_k: int = 100,
    rbo_p: float = 0.95,
) -> dict[str, float]:
    p = _normalise(p)
    q = _normalise(q)
    target = int(target)
    if not 0 <= target < p.size:
        raise IndexError("target token is outside the vocabulary")
    return {
        "js_similarity": jensen_shannon_similarity(p, q),
        "top_k_overlap": top_k_overlap(p, q, top_k),
        "rbo": rank_biased_overlap(p, q, top_k, rbo_p),
        "p_target": float(p[target]),
        "q_target": float(q[target]),
        "p_target_nll": float(-math.log(p[target] + EPS)),
        "q_target_nll": float(-math.log(q[target] + EPS)),
        "p_target_rr": reciprocal_rank(p, target),
        "q_target_rr": reciprocal_rank(q, target),
    }


def reconstruction_quality(
    original: np.ndarray, reconstruction: np.ndarray
) -> dict[str, float]:
    original = np.asarray(original, dtype=np.float64)
    reconstruction = np.asarray(reconstruction, dtype=np.float64)
    if original.shape != reconstruction.shape or original.ndim != 1:
        raise ValueError("reconstruction vectors must be one-dimensional and matched")
    error = original - reconstruction
    denominator = float(np.mean(original**2))
    cosine_denominator = float(
        np.linalg.norm(original) * np.linalg.norm(reconstruction)
    )
    return {
        "normalised_mse": float(np.mean(error**2) / max(denominator, EPS)),
        "reconstruction_cosine": float(
            np.dot(original, reconstruction) / max(cosine_denominator, EPS)
        ),
    }


def bootstrap_mean_ci(
    values: Iterable[float], *, samples: int, confidence: float, seed: int
) -> tuple[float, float, float]:
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        values, size=(int(samples), values.size), replace=True
    ).mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return (
        float(values.mean()),
        float(np.quantile(draws, tail)),
        float(np.quantile(draws, 1.0 - tail)),
    )


def _normalise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("expected a non-empty one-dimensional distribution")
    if np.any(values < 0) or not np.isfinite(values).all() or values.sum() <= 0:
        raise ValueError("distribution must be finite, non-negative, and non-zero")
    return values / values.sum()


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], EPS))))
