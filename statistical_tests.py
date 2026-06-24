"""Metrics and uncertainty estimates for 4D walk-forward validation."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def random_hit_probability(
    *,
    universe_size: int = 10_000,
    winner_count: int = 23,
    top_k: int = 5,
) -> float:
    """Exact probability that K unique random candidates hit at least one winner."""
    if not 0 <= winner_count <= universe_size:
        raise ValueError("winner_count must be inside the universe")
    if not 0 <= top_k <= universe_size:
        raise ValueError("top_k must be inside the universe")
    if top_k == 0 or winner_count == 0:
        return 0.0
    if universe_size - winner_count < top_k:
        return 1.0
    miss = 1.0
    for offset in range(top_k):
        miss *= (universe_size - winner_count - offset) / (
            universe_size - offset
        )
    return 1.0 - miss


def wilson_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson binomial confidence interval (95% uses z=1.95996)."""
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("invalid successes/trials")
    z_values = {0.90: 1.6448536269514722, 0.95: 1.959963984540054}
    z = z_values.get(confidence)
    if z is None:
        raise ValueError("supported confidence values are 0.90 and 0.95")
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            rate * (1.0 - rate) / trials + z * z / (4 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def paired_bootstrap_difference(
    improved: Sequence[float],
    baseline: Sequence[float],
    *,
    iterations: int = 5_000,
    seed: int = 417,
) -> dict[str, float]:
    """Paired bootstrap interval and two-sided sign probability for mean lift."""
    new = np.asarray(improved, dtype=np.float64)
    old = np.asarray(baseline, dtype=np.float64)
    if new.shape != old.shape or new.ndim != 1 or new.size == 0:
        raise ValueError("paired arrays must be non-empty, one-dimensional, and aligned")
    differences = new - old
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(iterations, differences.size))
    samples = np.mean(differences[indices], axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    probability_nonpositive = float(np.mean(samples <= 0))
    probability_nonnegative = float(np.mean(samples >= 0))
    p_value = min(1.0, 2.0 * min(probability_nonpositive, probability_nonnegative))
    return {
        "mean_difference": float(np.mean(differences)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "p_value": float(p_value),
    }


def binary_log_loss(labels: Any, probabilities: Any) -> float:
    y = np.asarray(labels, dtype=np.float64)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1 - 1e-12)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def brier_score(labels: Any, probabilities: Any) -> float:
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    labels: Any,
    probabilities: Any,
    *,
    bins: int = 10,
) -> float:
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (p >= boundaries[index]) & (p <= boundaries[index + 1])
        else:
            mask = (p >= boundaries[index]) & (p < boundaries[index + 1])
        if not np.any(mask):
            continue
        result += np.mean(mask) * abs(float(np.mean(y[mask]) - np.mean(p[mask])))
    return float(result)
