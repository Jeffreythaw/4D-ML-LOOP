"""Integer coordinate-search optimizer for affine models over ``Z_m``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from math_modular_utils import (
    affine_model_complexity,
    affine_modular_transform,
    as_modular_array,
    circular_squared_loss,
    fit_lstsq_round_mod,
    validate_modulus,
)


@dataclass(frozen=True)
class CircularAffineModel:
    """A fitted integer affine model with chronological validation metadata."""

    matrix: np.ndarray
    bias: np.ndarray
    modulus: int
    training_loss: float
    validation_loss: float
    adjusted_score: float
    complexity: int
    sample_count: int
    validation_count: int
    seed: int

    def predict(self, vectors: Any) -> np.ndarray:
        return affine_modular_transform(vectors, self.matrix, self.bias, self.modulus)


class CircularAffineOptimizer:
    """Optimize squared circular distance with random restarts and coordinate descent."""

    def __init__(
        self,
        *,
        modulus: int = 10,
        min_samples: int = 20,
        validation_fraction: float = 0.2,
        complexity_penalty: float = 0.01,
        random_restarts: int = 4,
        max_passes: int = 8,
        seed: int = 417,
    ) -> None:
        self.modulus = validate_modulus(modulus)
        self.min_samples = int(min_samples)
        self.validation_fraction = float(validation_fraction)
        self.complexity_penalty = float(complexity_penalty)
        self.random_restarts = int(random_restarts)
        self.max_passes = int(max_passes)
        self.seed = int(seed)
        if self.min_samples < 2:
            raise ValueError("min_samples must be at least 2")
        if not 0.0 <= self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in [0, 0.5)")
        if self.complexity_penalty < 0:
            raise ValueError("complexity_penalty must be non-negative")
        if self.random_restarts < 0 or self.max_passes <= 0:
            raise ValueError("invalid restart/pass configuration")

    def _row_objective(
        self,
        design: np.ndarray,
        target: np.ndarray,
        coefficients: np.ndarray,
        weights: np.ndarray | None,
    ) -> float:
        prediction = np.mod(design @ coefficients.astype(np.int64), self.modulus)
        loss = float(
            circular_squared_loss(
                target,
                prediction,
                self.modulus,
                sample_weight=weights,
            )
        )
        complexity = int(np.count_nonzero(coefficients))
        return loss + self.complexity_penalty * complexity

    def _coordinate_descent(
        self,
        design: np.ndarray,
        target: np.ndarray,
        initial: np.ndarray,
        weights: np.ndarray | None,
    ) -> np.ndarray:
        coefficients = np.mod(initial, self.modulus).astype(np.int16)
        current = self._row_objective(design, target, coefficients, weights)
        # Bias first preserves useful structural starts (identity/legacy) before
        # allowing matrix coefficients to move. This materially reduces local
        # minima caused by a temporarily wrong circular offset.
        coordinate_order = [coefficients.shape[0] - 1, *range(coefficients.shape[0] - 1)]
        for _ in range(self.max_passes):
            changed = False
            for coordinate in coordinate_order:
                best_value = int(coefficients[coordinate])
                best_objective = current
                for value in range(self.modulus):
                    if value == int(coefficients[coordinate]):
                        continue
                    trial = coefficients.copy()
                    trial[coordinate] = value
                    objective = self._row_objective(design, target, trial, weights)
                    if objective < best_objective - 1e-12:
                        best_objective = objective
                        best_value = value
                if best_value != int(coefficients[coordinate]):
                    coefficients[coordinate] = best_value
                    current = best_objective
                    changed = True
            if not changed:
                break
        return coefficients

    def fit(
        self,
        source: Any,
        target: Any,
        *,
        sample_weight: Any | None = None,
    ) -> CircularAffineModel:
        """Fit using a chronological tail holdout and integer-only coefficients."""
        x = as_modular_array(source, self.modulus, name="source")
        y = as_modular_array(target, self.modulus, name="target")
        if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
            raise ValueError("source and target must be equal-shape 2-D arrays")
        if x.shape[0] < self.min_samples:
            raise ValueError(
                f"need at least {self.min_samples} samples, got {x.shape[0]}"
            )
        weights = None
        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
            if weights.shape[0] != x.shape[0]:
                raise ValueError("sample_weight length must match source rows")
        validation_count = (
            max(1, int(round(x.shape[0] * self.validation_fraction)))
            if self.validation_fraction > 0
            else 0
        )
        split = x.shape[0] - validation_count
        fit_x, fit_y = x[:split], y[:split]
        val_x, val_y = x[split:], y[split:]
        fit_weights = weights[:split] if weights is not None else None

        design = np.hstack(
            [fit_x.astype(np.int64), np.ones((fit_x.shape[0], 1), dtype=np.int64)]
        )
        legacy = fit_lstsq_round_mod(
            fit_x, fit_y, self.modulus, sample_weight=fit_weights
        )
        rng = np.random.default_rng(self.seed)
        width = x.shape[1]
        matrix = np.zeros((width, width), dtype=np.int16)
        bias = np.zeros(width, dtype=np.int16)

        for output_index in range(width):
            identity = np.zeros(width + 1, dtype=np.int16)
            identity[output_index] = 1
            legacy_row = np.concatenate(
                [legacy.matrix[output_index], legacy.bias[output_index : output_index + 1]]
            )
            starts = [
                np.zeros(width + 1, dtype=np.int16),
                identity,
                legacy_row.astype(np.int16),
            ]
            starts.extend(
                rng.integers(
                    0, self.modulus, size=width + 1, dtype=np.int16
                )
                for _ in range(self.random_restarts)
            )
            candidates = [
                self._coordinate_descent(
                    design,
                    fit_y[:, output_index],
                    start,
                    fit_weights,
                )
                for start in starts
            ]

            def selection_key(coefficients: np.ndarray) -> tuple[float, float, tuple[int, ...]]:
                if validation_count:
                    validation_design = np.hstack(
                        [
                            val_x.astype(np.int64),
                            np.ones((val_x.shape[0], 1), dtype=np.int64),
                        ]
                    )
                    validation_prediction = np.mod(
                        validation_design @ coefficients.astype(np.int64),
                        self.modulus,
                    )
                    loss = float(
                        circular_squared_loss(
                            val_y[:, output_index],
                            validation_prediction,
                            self.modulus,
                        )
                    )
                else:
                    loss = self._row_objective(
                        design,
                        fit_y[:, output_index],
                        coefficients,
                        fit_weights,
                    )
                adjusted = loss + self.complexity_penalty * int(
                    np.count_nonzero(coefficients)
                )
                return adjusted, loss, tuple(int(v) for v in coefficients)

            best = min(candidates, key=selection_key)
            matrix[output_index] = best[:-1]
            bias[output_index] = best[-1]

        fit_prediction = affine_modular_transform(
            fit_x, matrix, bias, self.modulus
        )
        training_loss = float(
            circular_squared_loss(
                fit_y,
                fit_prediction,
                self.modulus,
                sample_weight=fit_weights,
            )
        )
        if validation_count:
            validation_prediction = affine_modular_transform(
                val_x, matrix, bias, self.modulus
            )
            validation_loss = float(
                circular_squared_loss(val_y, validation_prediction, self.modulus)
            )
        else:
            validation_loss = training_loss
        complexity = affine_model_complexity(matrix, bias)
        adjusted_score = -validation_loss - self.complexity_penalty * complexity
        return CircularAffineModel(
            matrix=matrix,
            bias=bias,
            modulus=self.modulus,
            training_loss=training_loss,
            validation_loss=validation_loss,
            adjusted_score=adjusted_score,
            complexity=complexity,
            sample_count=x.shape[0],
            validation_count=validation_count,
            seed=self.seed,
        )


class RegularizedAdaptiveAffineTrainer:
    """Train an adaptive formula from a rolling, same-day-type historical window."""

    def __init__(
        self,
        *,
        modulus: int = 10,
        min_samples: int = 64,
        rolling_window: int = 512,
        decay: float = 0.995,
        complexity_penalty: float = 0.02,
        seed: int = 417,
    ) -> None:
        if rolling_window < min_samples:
            raise ValueError("rolling_window must be >= min_samples")
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        self.modulus = validate_modulus(modulus)
        self.min_samples = int(min_samples)
        self.rolling_window = int(rolling_window)
        self.decay = float(decay)
        self.optimizer = CircularAffineOptimizer(
            modulus=self.modulus,
            min_samples=self.min_samples,
            complexity_penalty=complexity_penalty,
            seed=seed,
        )

    def fit(
        self,
        source: Any,
        target: Any,
        *,
        day_types: Sequence[str] | None = None,
        active_day_type: str | None = None,
    ) -> CircularAffineModel:
        x = np.asarray(source)
        y = np.asarray(target)
        if day_types is not None:
            if len(day_types) != x.shape[0]:
                raise ValueError("day_types length must match source rows")
            if active_day_type is None:
                raise ValueError("active_day_type is required with day_types")
            mask = np.asarray([value == active_day_type for value in day_types])
            x, y = x[mask], y[mask]
        x = x[-self.rolling_window :]
        y = y[-self.rolling_window :]
        ages = np.arange(x.shape[0] - 1, -1, -1, dtype=np.float64)
        weights = np.power(self.decay, ages)
        return self.optimizer.fit(x, y, sample_weight=weights)
