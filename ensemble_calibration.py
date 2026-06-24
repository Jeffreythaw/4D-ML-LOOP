"""Small dependency-free calibration models for candidate hit probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass(frozen=True)
class CalibrationFitSummary:
    sample_count: int
    positive_count: int
    feature_count: int
    converged: bool
    iterations: int


class LogisticCalibrator:
    """L2-regularized logistic calibration fitted by deterministic IRLS."""

    def __init__(
        self,
        *,
        l2: float = 1.0,
        max_iter: int = 80,
        tolerance: float = 1e-7,
        min_samples: int = 200,
    ) -> None:
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.min_samples = int(min_samples)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coefficients_: np.ndarray | None = None
        self.constant_probability_: float | None = None
        self.summary_: CalibrationFitSummary | None = None

    def fit(
        self,
        features: Any,
        labels: Any,
        *,
        sample_weight: Any | None = None,
    ) -> "LogisticCalibrator":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
        if x.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("features must be 2-D and align with labels")
        if np.any((y != 0) & (y != 1)):
            raise ValueError("labels must be binary")
        weights = (
            np.ones(y.shape[0], dtype=np.float64)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        )
        if weights.shape[0] != y.shape[0]:
            raise ValueError("sample_weight length must match labels")
        weighted_rate = float(np.average(y, weights=weights))
        if x.shape[0] < self.min_samples or np.unique(y).size < 2:
            self.constant_probability_ = float(
                np.clip(weighted_rate, 1e-9, 1.0 - 1e-9)
            )
            self.summary_ = CalibrationFitSummary(
                sample_count=x.shape[0],
                positive_count=int(np.sum(y)),
                feature_count=x.shape[1],
                converged=True,
                iterations=0,
            )
            return self

        self.mean_ = np.average(x, axis=0, weights=weights)
        variance = np.average((x - self.mean_) ** 2, axis=0, weights=weights)
        self.scale_ = np.sqrt(np.maximum(variance, 1e-12))
        standardized = (x - self.mean_) / self.scale_
        design = np.hstack(
            [np.ones((x.shape[0], 1), dtype=np.float64), standardized]
        )
        coefficients = np.zeros(design.shape[1], dtype=np.float64)
        coefficients[0] = np.log(
            np.clip(weighted_rate, 1e-9, 1 - 1e-9)
            / np.clip(1 - weighted_rate, 1e-9, 1)
        )
        regularizer = np.eye(design.shape[1], dtype=np.float64) * self.l2
        regularizer[0, 0] = 0.0
        converged = False
        iteration = 0
        total_weight = float(np.sum(weights))

        def objective(candidate: np.ndarray) -> float:
            probability = np.clip(
                _sigmoid(design @ candidate), 1e-12, 1.0 - 1e-12
            )
            data_loss = -np.sum(
                weights
                * (y * np.log(probability) + (1.0 - y) * np.log(1.0 - probability))
            ) / total_weight
            penalty = self.l2 * float(np.sum(candidate[1:] ** 2)) / (
                2.0 * total_weight
            )
            return float(data_loss + penalty)

        for iteration in range(1, self.max_iter + 1):
            probability = _sigmoid(design @ coefficients)
            gradient = (
                design.T @ ((probability - y) * weights) / total_weight
                + regularizer @ coefficients / total_weight
            )
            curvature = weights * probability * (1.0 - probability)
            hessian = (
                design.T @ (design * curvature.reshape(-1, 1)) / total_weight
                + regularizer / total_weight
                + np.eye(design.shape[1]) * 1e-9
            )
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(hessian) @ gradient
            current_objective = objective(coefficients)
            step_scale = 1.0
            accepted = False
            candidate = coefficients
            for _ in range(24):
                candidate = coefficients - step_scale * step
                if objective(candidate) <= current_objective + 1e-12:
                    accepted = True
                    break
                step_scale *= 0.5
            if not accepted:
                break
            coefficients = candidate
            if float(np.max(np.abs(step_scale * step))) < self.tolerance:
                converged = True
                break
        self.coefficients_ = coefficients
        self.constant_probability_ = None
        self.summary_ = CalibrationFitSummary(
            sample_count=x.shape[0],
            positive_count=int(np.sum(y)),
            feature_count=x.shape[1],
            converged=converged,
            iterations=iteration,
        )
        return self

    def predict_proba(self, features: Any) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("features must be 2-D")
        if self.constant_probability_ is not None:
            return np.full(x.shape[0], self.constant_probability_, dtype=np.float64)
        if self.coefficients_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("calibrator has not been fitted")
        standardized = (x - self.mean_) / self.scale_
        design = np.hstack(
            [np.ones((x.shape[0], 1), dtype=np.float64), standardized]
        )
        return _sigmoid(design @ self.coefficients_)


class IsotonicCalibrator:
    """One-dimensional isotonic probability calibration using PAV."""

    def __init__(self) -> None:
        self.thresholds_: np.ndarray | None = None
        self.values_: np.ndarray | None = None

    def fit(
        self,
        scores: Any,
        labels: Any,
        *,
        sample_weight: Any | None = None,
    ) -> "IsotonicCalibrator":
        x = np.asarray(scores, dtype=np.float64).reshape(-1)
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
        weights = (
            np.ones(x.shape[0], dtype=np.float64)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        )
        if x.shape != y.shape or x.shape != weights.shape:
            raise ValueError("scores, labels, and weights must align")
        order = np.argsort(x, kind="mergesort")
        sorted_x, sorted_y, sorted_w = x[order], y[order], weights[order]
        blocks: list[list[float]] = []
        for score, label, weight in zip(sorted_x, sorted_y, sorted_w):
            blocks.append([score, score, weight, label * weight])
            while (
                len(blocks) >= 2
                and blocks[-2][3] / blocks[-2][2]
                > blocks[-1][3] / blocks[-1][2]
            ):
                right = blocks.pop()
                left = blocks.pop()
                blocks.append(
                    [
                        left[0],
                        right[1],
                        left[2] + right[2],
                        left[3] + right[3],
                    ]
                )
        self.thresholds_ = np.asarray([block[1] for block in blocks])
        self.values_ = np.asarray(
            [np.clip(block[3] / block[2], 0.0, 1.0) for block in blocks]
        )
        return self

    def predict_proba(self, scores: Any) -> np.ndarray:
        if self.thresholds_ is None or self.values_ is None:
            raise RuntimeError("calibrator has not been fitted")
        x = np.asarray(scores, dtype=np.float64).reshape(-1)
        indices = np.searchsorted(self.thresholds_, x, side="left")
        indices = np.clip(indices, 0, self.values_.shape[0] - 1)
        return self.values_[indices]
