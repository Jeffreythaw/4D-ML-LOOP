"""Mathematically correct helpers for affine models over finite digit rings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


SUPPORTED_MODULI = (5, 10)


def validate_modulus(modulus: int) -> int:
    """Return a supported modulus or raise a clear error."""
    value = int(modulus)
    if value not in SUPPORTED_MODULI:
        raise ValueError(f"modulus must be one of {SUPPORTED_MODULI}, got {value}")
    return value


def as_modular_array(values: Any, modulus: int, *, name: str = "values") -> np.ndarray:
    """Validate an integer array and normalize it into ``Z_modulus``."""
    modulus = validate_modulus(modulus)
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integers, got dtype={array.dtype}")
    return np.mod(array, modulus).astype(np.int16, copy=False)


def circular_distance(a: Any, b: Any, modulus: int) -> np.ndarray:
    """Compute ``d_m(a,b)=min((a-b) mod m, (b-a) mod m)`` element-wise."""
    modulus = validate_modulus(modulus)
    left = as_modular_array(a, modulus, name="a").astype(np.int32, copy=False)
    right = as_modular_array(b, modulus, name="b").astype(np.int32, copy=False)
    forward = np.mod(left - right, modulus)
    backward = np.mod(right - left, modulus)
    return np.minimum(forward, backward).astype(np.int16, copy=False)


def circular_squared_loss(
    actual: Any,
    predicted: Any,
    modulus: int,
    *,
    sample_weight: Any | None = None,
    reduction: str = "mean",
) -> float | np.ndarray:
    """Squared circular loss with ``mean``, ``sum``, or ``none`` reduction."""
    distances = circular_distance(actual, predicted, modulus).astype(np.float64)
    losses = distances * distances
    if reduction == "none":
        return losses
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")
    if sample_weight is None:
        return float(np.mean(losses) if reduction == "mean" else np.sum(losses))
    weights = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if losses.ndim == 1:
        row_loss = losses
    else:
        row_loss = np.mean(losses, axis=tuple(range(1, losses.ndim)))
    if weights.shape[0] != row_loss.shape[0]:
        raise ValueError("sample_weight length must match the number of rows")
    if np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("sample_weight must be non-negative with positive total")
    if reduction == "sum":
        return float(np.sum(row_loss * weights))
    return float(np.average(row_loss, weights=weights))


def affine_modular_transform(
    vectors: Any,
    matrix: Any,
    bias: Any,
    modulus: int,
) -> np.ndarray:
    """Apply ``(X M^T + b) mod m`` using integer arithmetic."""
    modulus = validate_modulus(modulus)
    x = as_modular_array(vectors, modulus, name="vectors")
    matrix_array = as_modular_array(matrix, modulus, name="matrix")
    bias_array = as_modular_array(bias, modulus, name="bias")
    if x.ndim != 2:
        raise ValueError(f"vectors must be 2-D, got shape={x.shape}")
    width = x.shape[1]
    if matrix_array.shape != (width, width):
        raise ValueError(
            f"matrix must have shape {(width, width)}, got {matrix_array.shape}"
        )
    if bias_array.shape != (width,):
        raise ValueError(f"bias must have shape {(width,)}, got {bias_array.shape}")
    result = (
        x.astype(np.int64, copy=False)
        @ matrix_array.astype(np.int64, copy=False).T
        + bias_array.astype(np.int64, copy=False)
    ) % modulus
    return result.astype(np.int16, copy=False)


@dataclass(frozen=True)
class LstsqModularBaseline:
    """Audit-only real least-squares fit followed by rounding and modulo."""

    matrix: np.ndarray
    bias: np.ndarray
    raw_coefficients: np.ndarray
    modulus: int

    def predict(self, vectors: Any) -> np.ndarray:
        return affine_modular_transform(vectors, self.matrix, self.bias, self.modulus)


def fit_lstsq_round_mod(
    source: Any,
    target: Any,
    modulus: int,
    *,
    sample_weight: Any | None = None,
) -> LstsqModularBaseline:
    """Reproduce the legacy ``round(lstsq) % m`` behavior for comparison."""
    modulus = validate_modulus(modulus)
    x = as_modular_array(source, modulus, name="source")
    y = as_modular_array(target, modulus, name="target")
    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
        raise ValueError("source and target must be equal-shape 2-D arrays")
    design = np.hstack(
        [x.astype(np.float64), np.ones((x.shape[0], 1), dtype=np.float64)]
    )
    target_float = y.astype(np.float64)
    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if weights.shape[0] != x.shape[0]:
            raise ValueError("sample_weight length must match source rows")
        root = np.sqrt(weights).reshape(-1, 1)
        design = design * root
        target_float = target_float * root
    coefficients, *_ = np.linalg.lstsq(design, target_float, rcond=None)
    matrix = np.mod(
        np.rint(coefficients[:-1, :].T).astype(np.int64), modulus
    ).astype(np.int16)
    bias = np.mod(
        np.rint(coefficients[-1, :]).astype(np.int64), modulus
    ).astype(np.int16)
    return LstsqModularBaseline(
        matrix=matrix,
        bias=bias,
        raw_coefficients=coefficients,
        modulus=modulus,
    )


def affine_model_complexity(matrix: Any, bias: Any) -> int:
    """Count active matrix and bias coefficients."""
    return int(np.count_nonzero(matrix) + np.count_nonzero(bias))
