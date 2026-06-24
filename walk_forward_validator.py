#!/usr/bin/env python3
"""Strict chronological validation for legacy and mathematically improved engines."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np

from circular_affine_optimizer import CircularAffineModel, CircularAffineOptimizer
from ensemble_calibration import LogisticCalibrator
from markov_smoothing import (
    MarkovObservation,
    SmoothedMarkovModel,
    select_alpha_walk_forward,
)
from math_modular_utils import (
    LstsqModularBaseline,
    circular_distance,
    fit_lstsq_round_mod,
)
from statistical_tests import (
    binary_log_loss,
    brier_score,
    expected_calibration_error,
    paired_bootstrap_difference,
    random_hit_probability,
    wilson_interval,
)


ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from scripts.run_full_history_engine_training import (  # noqa: E402
    ChronologicalTrainingDataset,
    TrainingDraw,
    TrainingPair,
    load_draw_history,
)


TOP_K_VALUES = (1, 3, 5, 10, 20)
METHODS = (
    "frequency",
    "recent_frequency",
    "markov_smoothed",
    "production_static",
    "legacy_lstsq",
    "improved_calibrated",
)
REPORT_PATH = ROOT / "reports/math_improvement_walk_forward_report.md"
RESULT_PATH = ROOT / "reports/math_improvement_walk_forward_results.json"


def number_to_digits(number: str) -> tuple[int, int, int, int]:
    value = str(number).zfill(4)
    return tuple(int(char) for char in value)  # type: ignore[return-value]


UNIVERSE_NUMBERS = tuple(f"{value:04d}" for value in range(10_000))
UNIVERSE_DIGITS = np.asarray(
    [number_to_digits(number) for number in UNIVERSE_NUMBERS], dtype=np.int16
)


@dataclass(frozen=True)
class TemporalFirewallTrace:
    source_draw_no: int
    target_draw_no: int
    max_training_target_draw_no: int
    target_seen_before_lock: bool
    candidate_hash: str


@dataclass
class MethodStep:
    ranking: tuple[str, ...]
    hit_flags: dict[int, float]
    recalls: dict[int, float]
    precisions: dict[int, float]
    reciprocal_rank: float


@dataclass
class StepOutput:
    source_draw_no: int
    target_draw_no: int
    rankings: dict[str, tuple[str, ...]]
    improved_probabilities: np.ndarray
    feature_matrix: np.ndarray
    firewall: TemporalFirewallTrace
    alpha: float
    modular_validation_loss: float
    legacy_validation_loss: float


def _paired_rows(pairs: Sequence[TrainingPair]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    source_rows: list[tuple[int, int, int, int]] = []
    target_rows: list[tuple[int, int, int, int]] = []
    day_types: list[str] = []
    for pair in pairs:
        count = min(len(pair.source_digits), len(pair.target_digits))
        source_rows.extend(pair.source_digits[:count])
        target_rows.extend(pair.target_digits[:count])
        day_types.extend([pair.source_day_type] * count)
    return (
        np.asarray(source_rows, dtype=np.int16),
        np.asarray(target_rows, dtype=np.int16),
        day_types,
    )


def _markov_observations(pairs: Sequence[TrainingPair]) -> list[MarkovObservation]:
    output: list[MarkovObservation] = []
    for pair in pairs:
        count = min(len(pair.source_winners_23), len(pair.target_winners_23))
        output.extend(
            MarkovObservation(
                source=pair.source_winners_23[index],
                target=pair.target_winners_23[index],
                day_type=pair.source_day_type,
            )
            for index in range(count)
        )
    return output


def _minimum_circular_score(
    candidate_digits: np.ndarray,
    predicted_digits: np.ndarray,
    modulus: int,
) -> np.ndarray:
    if predicted_digits.size == 0:
        return np.zeros(candidate_digits.shape[0], dtype=np.float64)
    distances = circular_distance(
        candidate_digits[:, None, :],
        predicted_digits[None, :, :],
        modulus,
    ).astype(np.float64)
    squared = np.sum(distances * distances, axis=2)
    minimum = np.min(squared, axis=1)
    scale = max(1.0, float(candidate_digits.shape[1]))
    return np.exp(-minimum / scale)


def _rank(scores: np.ndarray, limit: int = 20) -> tuple[str, ...]:
    indices = np.lexsort((np.arange(scores.shape[0]), -scores))
    return tuple(UNIVERSE_NUMBERS[int(index)] for index in indices[:limit])


def _rank_with_primary_and_fallback(
    primary: Iterable[str],
    fallback_scores: np.ndarray,
    limit: int = 20,
) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for number in primary:
        if number not in seen:
            output.append(number)
            seen.add(number)
        if len(output) >= limit:
            return tuple(output)
    for number in _rank(fallback_scores, limit=limit + len(output)):
        if number not in seen:
            output.append(number)
            seen.add(number)
        if len(output) >= limit:
            break
    return tuple(output)


def _frequency_array(draws: Sequence[TrainingDraw]) -> np.ndarray:
    counts = np.ones(10_000, dtype=np.float64)
    for draw in draws:
        for number in draw.winners:
            counts[int(number)] += 1.0
    return counts / np.sum(counts)


def _markov_probability_array(
    model: SmoothedMarkovModel,
    source_numbers: Sequence[str],
    day_type: str,
) -> np.ndarray:
    prior = np.asarray(
        [model.prior_probability(number) for number in UNIVERSE_NUMBERS],
        dtype=np.float64,
    )
    if not source_numbers:
        return prior
    result = np.zeros(10_000, dtype=np.float64)
    for source in source_numbers:
        counts = model._counts(source, day_type)
        support = sum(counts.values())
        local = model.alpha * prior
        if counts:
            local = local.copy()
            for target, count in counts.items():
                local[int(target)] += count
        result += local / (support + model.alpha)
    return result / len(source_numbers)


def _static_scores(source_digits: np.ndarray, day_type: str) -> np.ndarray:
    import jeffrey_quad_engine_v2_step2_matrix_core as production

    scores = np.zeros(10_000, dtype=np.float64)
    engines = (
        production.Engine1CrossPairLinear(),
        production.Engine2SetProjector(),
        production.Engine3Polynomial(),
    )
    for engine in engines:
        predicted = engine.predict_vectors(source_digits, day_type)
        for rank, row in enumerate(predicted, start=1):
            number = int("".join(str(int(value)) for value in row))
            scores[number] = max(scores[number], 1.0 / rank)
    return scores


def _candidate_features(
    *,
    source_draw: TrainingDraw,
    modular_model: CircularAffineModel,
    legacy_model: LstsqModularBaseline,
    frequency: np.ndarray,
    recent_frequency: np.ndarray,
    markov_probability: np.ndarray,
    static_scores: np.ndarray,
) -> np.ndarray:
    source_digits = np.asarray(
        [number_to_digits(number) for number in source_draw.winners],
        dtype=np.int16,
    )
    modular_score = _minimum_circular_score(
        UNIVERSE_DIGITS, modular_model.predict(source_digits), 10
    )
    legacy_score = _minimum_circular_score(
        UNIVERSE_DIGITS, legacy_model.predict(source_digits), 10
    )
    exact_modular = (modular_score >= 1.0 - 1e-12).astype(np.float64)
    exact_legacy = (legacy_score >= 1.0 - 1e-12).astype(np.float64)
    exact_static = (static_scores > 0).astype(np.float64)
    consensus = exact_modular + exact_legacy + exact_static
    digit_sum = np.sum(UNIVERSE_DIGITS, axis=1) / 36.0
    zero_count = np.sum(UNIVERSE_DIGITS == 0, axis=1) / 4.0
    unique_count = np.asarray(
        [len(set(row.tolist())) / 4.0 for row in UNIVERSE_DIGITS],
        dtype=np.float64,
    )
    day_types = ("Wednesday", "Saturday", "Sunday", "Special")
    day_features = np.column_stack(
        [
            np.full(10_000, float(source_draw.day_type == value))
            for value in day_types
        ]
    )
    adaptive_confidence = np.full(
        10_000,
        max(0.0, 1.0 - modular_model.validation_loss / 8.5),
        dtype=np.float64,
    )
    return np.column_stack(
        [
            modular_score,
            legacy_score,
            frequency,
            recent_frequency,
            markov_probability,
            static_scores,
            consensus,
            adaptive_confidence * modular_score,
            digit_sum,
            zero_count,
            unique_count,
            day_features,
        ]
    )


class StrictWalkForwardValidator:
    """Predict, lock, and only then reveal each target draw."""

    def __init__(
        self,
        dataset: ChronologicalTrainingDataset,
        *,
        start_draw_no: int,
        end_draw_no: int,
        train_draw_window: int = 128,
        recent_draw_window: int = 52,
        refit_interval: int = 5,
        negative_samples: int = 160,
        seed: int = 417,
    ) -> None:
        self.dataset = dataset
        self.start_draw_no = int(start_draw_no)
        self.end_draw_no = int(end_draw_no)
        self.train_draw_window = int(train_draw_window)
        self.recent_draw_window = int(recent_draw_window)
        self.refit_interval = int(refit_interval)
        self.negative_samples = int(negative_samples)
        self.seed = int(seed)
        self._by_no = {draw.draw_no: draw for draw in dataset.draws}
        self._calibration_x: list[np.ndarray] = []
        self._calibration_y: list[np.ndarray] = []
        self._calibration_w: list[np.ndarray] = []
        self._model_cache: dict[
            str,
            tuple[
                int,
                tuple[
                    CircularAffineModel,
                    LstsqModularBaseline,
                    SmoothedMarkovModel,
                    float,
                    float,
                ],
            ],
        ] = {}
        self._cached_calibrator: LogisticCalibrator | None = None
        self._calibrator_fit_step_count = 0

    def _training_pairs(self, source_draw_no: int, day_type: str) -> tuple[TrainingPair, ...]:
        eligible = self.dataset.pairs_until(source_draw_no)
        if self.train_draw_window > 0:
            eligible = eligible[-self.train_draw_window :]
        same_day = tuple(
            pair for pair in eligible if pair.source_day_type == day_type
        )
        source_rows, _, _ = _paired_rows(same_day)
        return same_day if source_rows.shape[0] >= 64 else eligible

    def _fit_models(
        self,
        source_draw_no: int,
        day_type: str,
    ) -> tuple[
        CircularAffineModel,
        LstsqModularBaseline,
        SmoothedMarkovModel,
        float,
        float,
    ]:
        cached = self._model_cache.get(day_type)
        if cached is not None and source_draw_no - cached[0] < self.refit_interval:
            return cached[1]
        pairs = self._training_pairs(source_draw_no, day_type)
        source, target, _ = _paired_rows(pairs)
        optimizer = CircularAffineOptimizer(
            modulus=10,
            min_samples=64,
            validation_fraction=0.2,
            complexity_penalty=0.01,
            random_restarts=2,
            max_passes=5,
            seed=self.seed + source_draw_no,
        )
        modular = optimizer.fit(source, target)
        split = source.shape[0] - modular.validation_count
        legacy = fit_lstsq_round_mod(source[:split], target[:split], 10)
        legacy_validation = float(
            np.mean(
                circular_distance(
                    target[split:],
                    legacy.predict(source[split:]),
                    10,
                ).astype(np.float64)
                ** 2
            )
            if modular.validation_count
            else np.mean(
                circular_distance(target, legacy.predict(source), 10).astype(
                    np.float64
                )
                ** 2
            )
        )
        observations = _markov_observations(pairs)
        alpha_sample = observations[-min(5_000, len(observations)) :]
        if len(alpha_sample) > 101:
            alpha, _ = select_alpha_walk_forward(
                alpha_sample,
                warmup=min(100, len(alpha_sample) - 1),
            )
        else:
            alpha = 5.0
        markov = SmoothedMarkovModel(alpha=alpha).fit(observations)
        fitted = (
            modular,
            legacy,
            markov,
            alpha,
            legacy_validation,
        )
        self._model_cache[day_type] = (source_draw_no, fitted)
        return fitted

    def predict_locked(self, source_draw_no: int) -> StepOutput:
        target_draw_no = source_draw_no + 1
        source_draw = self._by_no[source_draw_no]
        pairs = self.dataset.pairs_until(source_draw_no)
        max_training_target = max(pair.target_draw_no for pair in pairs)
        if max_training_target > source_draw_no:
            raise RuntimeError("temporal firewall violation: future training row")
        modular, legacy, markov, alpha, legacy_validation = self._fit_models(
            source_draw_no, source_draw.day_type
        )
        historical_draws = tuple(
            draw for draw in self.dataset.draws if draw.draw_no <= source_draw_no
        )
        frequency = _frequency_array(historical_draws)
        recent_frequency = _frequency_array(
            historical_draws[-self.recent_draw_window :]
        )
        markov_probability = _markov_probability_array(
            markov, source_draw.winners, source_draw.day_type
        )
        source_digits = np.asarray(
            [number_to_digits(number) for number in source_draw.winners],
            dtype=np.int16,
        )
        static_scores = _static_scores(source_digits, source_draw.day_type)
        features = _candidate_features(
            source_draw=source_draw,
            modular_model=modular,
            legacy_model=legacy,
            frequency=frequency,
            recent_frequency=recent_frequency,
            markov_probability=markov_probability,
            static_scores=static_scores,
        )
        fallback = (
            2.0 * features[:, 0]
            + 0.5 * features[:, 1]
            + 0.8 * features[:, 4] / max(float(np.max(features[:, 4])), 1e-15)
            + 0.3 * features[:, 5]
            + 0.2 * features[:, 6]
        )
        if self._calibration_x:
            if (
                self._cached_calibrator is None
                or len(self._calibration_x) - self._calibrator_fit_step_count
                >= self.refit_interval
            ):
                calibrator = LogisticCalibrator(l2=2.0, min_samples=200)
                calibrator.fit(
                    np.vstack(self._calibration_x),
                    np.concatenate(self._calibration_y),
                    sample_weight=np.concatenate(self._calibration_w),
                )
                self._cached_calibrator = calibrator
                self._calibrator_fit_step_count = len(self._calibration_x)
            probabilities = self._cached_calibrator.predict_proba(features)
        else:
            raw = np.exp(3.0 * (fallback - np.max(fallback)))
            probabilities = np.clip(raw * 23.0 / np.sum(raw), 1e-12, 1 - 1e-12)

        static_primary = [
            UNIVERSE_NUMBERS[index]
            for index in np.flatnonzero(static_scores > 0)[
                np.argsort(-static_scores[static_scores > 0], kind="mergesort")
            ]
        ]
        rankings = {
            "frequency": _rank(frequency),
            "recent_frequency": _rank(recent_frequency),
            "markov_smoothed": _rank(markov_probability),
            "production_static": _rank_with_primary_and_fallback(
                static_primary, frequency
            ),
            "legacy_lstsq": _rank_with_primary_and_fallback(
                (
                    "".join(str(int(value)) for value in row)
                    for row in legacy.predict(source_digits)
                ),
                frequency,
            ),
            "improved_calibrated": _rank(probabilities),
        }
        serialized = json.dumps(rankings, sort_keys=True).encode("utf-8")
        firewall = TemporalFirewallTrace(
            source_draw_no=source_draw_no,
            target_draw_no=target_draw_no,
            max_training_target_draw_no=max_training_target,
            target_seen_before_lock=False,
            candidate_hash=hashlib.sha256(serialized).hexdigest(),
        )
        return StepOutput(
            source_draw_no=source_draw_no,
            target_draw_no=target_draw_no,
            rankings=rankings,
            improved_probabilities=probabilities,
            feature_matrix=features,
            firewall=firewall,
            alpha=alpha,
            modular_validation_loss=modular.validation_loss,
            legacy_validation_loss=legacy_validation,
        )

    def _update_calibration_after_reveal(
        self,
        output: StepOutput,
        target_draw: TrainingDraw,
    ) -> None:
        positive_indices = np.asarray(
            sorted({int(number) for number in target_draw.winners}), dtype=np.int64
        )
        positive_set = set(int(value) for value in positive_indices)
        available = np.asarray(
            [index for index in range(10_000) if index not in positive_set],
            dtype=np.int64,
        )
        rng = np.random.default_rng(self.seed + target_draw.draw_no)
        negative_count = min(self.negative_samples, available.shape[0])
        negative_indices = rng.choice(
            available, size=negative_count, replace=False
        )
        indices = np.concatenate([positive_indices, negative_indices])
        labels = np.concatenate(
            [
                np.ones(positive_indices.shape[0], dtype=np.float64),
                np.zeros(negative_indices.shape[0], dtype=np.float64),
            ]
        )
        weights = np.concatenate(
            [
                np.ones(positive_indices.shape[0], dtype=np.float64),
                np.full(
                    negative_indices.shape[0],
                    (10_000 - positive_indices.shape[0])
                    / max(1, negative_indices.shape[0]),
                    dtype=np.float64,
                ),
            ]
        )
        self._calibration_x.append(output.feature_matrix[indices])
        self._calibration_y.append(labels)
        self._calibration_w.append(weights)

    def run(self) -> dict[str, Any]:
        step_records: list[dict[str, Any]] = []
        winner_counts: list[int] = []
        method_steps: dict[str, list[MethodStep]] = {
            method: [] for method in METHODS
        }
        probability_metrics: list[dict[str, float]] = []
        for source_draw_no in range(self.start_draw_no, self.end_draw_no + 1):
            if source_draw_no not in self._by_no or source_draw_no + 1 not in self._by_no:
                continue
            output = self.predict_locked(source_draw_no)
            target_draw = self._by_no[output.target_draw_no]
            winners = set(target_draw.winners)
            winner_counts.append(len(winners))
            for method, ranking in output.rankings.items():
                positions = [
                    index + 1
                    for index, number in enumerate(ranking)
                    if number in winners
                ]
                method_steps[method].append(
                    MethodStep(
                        ranking=ranking,
                        hit_flags={
                            k: float(any(position <= k for position in positions))
                            for k in TOP_K_VALUES
                        },
                        recalls={
                            k: sum(position <= k for position in positions)
                            / max(1, len(winners))
                            for k in TOP_K_VALUES
                        },
                        precisions={
                            k: sum(position <= k for position in positions) / k
                            for k in TOP_K_VALUES
                        },
                        reciprocal_rank=(
                            1.0 / min(positions) if positions else 0.0
                        ),
                    )
                )
            labels = np.zeros(10_000, dtype=np.float64)
            labels[[int(number) for number in winners]] = 1.0
            probability_metrics.append(
                {
                    "log_loss": binary_log_loss(
                        labels, output.improved_probabilities
                    ),
                    "brier": brier_score(
                        labels, output.improved_probabilities
                    ),
                    "ece": expected_calibration_error(
                        labels, output.improved_probabilities
                    ),
                }
            )
            step_records.append(
                {
                    "source_draw_no": output.source_draw_no,
                    "target_draw_no": output.target_draw_no,
                    "target_winner_count": len(winners),
                    "rankings": {
                        method: list(ranking)
                        for method, ranking in output.rankings.items()
                    },
                    "firewall": asdict(output.firewall),
                    "alpha": output.alpha,
                    "modular_validation_loss": output.modular_validation_loss,
                    "legacy_validation_loss": output.legacy_validation_loss,
                }
            )
            self._update_calibration_after_reveal(output, target_draw)

        if not step_records:
            raise RuntimeError("validation range produced no steps")
        summary: dict[str, Any] = {}
        for method, rows in method_steps.items():
            method_summary: dict[str, Any] = {
                "draw_count": len(rows),
                "mean_reciprocal_rank": float(
                    np.mean([row.reciprocal_rank for row in rows])
                ),
                "top_k": {},
            }
            for k in TOP_K_VALUES:
                flags = [row.hit_flags[k] for row in rows]
                successes = int(sum(flags))
                lower, upper = wilson_interval(successes, len(rows))
                random_probability = float(
                    np.mean(
                        [
                            random_hit_probability(
                                winner_count=winner_count,
                                top_k=k,
                            )
                            for winner_count in winner_counts
                        ]
                    )
                )
                method_summary["top_k"][str(k)] = {
                    "hit_rate": float(np.mean(flags)),
                    "hit_count": successes,
                    "recall": float(np.mean([row.recalls[k] for row in rows])),
                    "precision": float(
                        np.mean([row.precisions[k] for row in rows])
                    ),
                    "ci_95": [lower, upper],
                    "random_hit_probability": random_probability,
                    "lift_over_random": (
                        float(np.mean(flags)) / random_probability
                        if random_probability
                        else None
                    ),
                }
            summary[method] = method_summary

        significance = {}
        for baseline in (
            "legacy_lstsq",
            "production_static",
            "frequency",
            "markov_smoothed",
        ):
            significance[baseline] = {}
            for k in TOP_K_VALUES:
                significance[baseline][str(k)] = paired_bootstrap_difference(
                    [row.hit_flags[k] for row in method_steps["improved_calibrated"]],
                    [row.hit_flags[k] for row in method_steps[baseline]],
                    iterations=2_000,
                    seed=self.seed + k,
                )
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": {
                "start_draw_no": self.start_draw_no,
                "end_draw_no": self.end_draw_no,
                "train_draw_window": self.train_draw_window,
                "recent_draw_window": self.recent_draw_window,
                "refit_interval": self.refit_interval,
                "negative_samples": self.negative_samples,
                "seed": self.seed,
            },
            "temporal_firewall_status": (
                "PASS"
                if all(
                    not row["firewall"]["target_seen_before_lock"]
                    and row["firewall"]["max_training_target_draw_no"]
                    <= row["source_draw_no"]
                    for row in step_records
                )
                else "FAIL"
            ),
            "summary": summary,
            "probability_metrics": {
                key: float(np.mean([row[key] for row in probability_metrics]))
                for key in ("log_loss", "brier", "ece")
            },
            "significance_improved_vs": significance,
            "mean_modular_validation_loss": float(
                np.mean([row["modular_validation_loss"] for row in step_records])
            ),
            "mean_legacy_validation_loss": float(
                np.mean([row["legacy_validation_loss"] for row in step_records])
            ),
            "steps": step_records,
        }


def render_report(results: dict[str, Any]) -> str:
    legacy_loss = results["mean_legacy_validation_loss"]
    modular_loss = results["mean_modular_validation_loss"]
    loss_reduction = (
        (legacy_loss - modular_loss) / legacy_loss if legacy_loss else 0.0
    )
    significant_positive = any(
        row["ci_lower"] > 0
        for comparisons in results["significance_improved_vs"].values()
        for row in comparisons.values()
    )
    lines = [
        "# Mathematical Improvement Walk-Forward Report",
        "",
        f"- Generated: {results['generated_at_utc']}",
        f"- Temporal firewall: **{results['temporal_firewall_status']}**",
        f"- Draws tested: {len(results['steps'])}",
        (
            "- Mean circular validation loss: "
            f"legacy `{legacy_loss:.6f}`, modular `{modular_loss:.6f}` "
            f"({loss_reduction:.2%} reduction)"
        ),
        (
            "- Promotion decision: **eligible for shadow research only**"
            if significant_positive
            else "- Promotion decision: **keep production ranking unchanged**"
        ),
        "",
        "The `legacy_lstsq` baseline reproduces the risky real least-squares, "
        "round, and modulo approach offline. `production_static` reproduces the "
        "three current static formula families without database-side adaptive formulas.",
        "",
        "## Before/after metrics",
        "",
        "| Method | K | Hit rate | 95% CI | Precision@K | Recall@K | Lift vs random |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        for k in TOP_K_VALUES:
            row = results["summary"][method]["top_k"][str(k)]
            lines.append(
                f"| {method} | {k} | {row['hit_rate']:.4%} | "
                f"[{row['ci_95'][0]:.4%}, {row['ci_95'][1]:.4%}] | "
                f"{row['precision']:.4%} | {row['recall']:.4%} | "
                f"{row['lift_over_random']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Probability calibration",
            "",
            f"- Binary log-loss: `{results['probability_metrics']['log_loss']:.8f}`",
            f"- Brier score: `{results['probability_metrics']['brier']:.8f}`",
            f"- Expected calibration error: `{results['probability_metrics']['ece']:.8f}`",
            "",
            "## Statistical interpretation",
            "",
            "A result is treated as statistically meaningful only when the paired "
            "bootstrap 95% interval excludes zero. Lottery-like data may remain "
            "indistinguishable from randomness even when training loss improves.",
            "",
            (
                "At least one tested comparison has a positive interval excluding zero."
                if significant_positive
                else "No tested Top-K improvement has a paired 95% interval excluding zero."
            ),
            "",
            "| Baseline | K | Mean hit-rate difference | 95% bootstrap CI | p-value |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for baseline, by_k in results["significance_improved_vs"].items():
        for k in TOP_K_VALUES:
            row = by_k[str(k)]
            lines.append(
                f"| {baseline} | {k} | {row['mean_difference']:.4%} | "
                f"[{row['ci_lower']:.4%}, {row['ci_upper']:.4%}] | "
                f"{row['p_value']:.4f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-draw-no", type=int, default=4051)
    parser.add_argument("--end-draw-no", type=int, default=4100)
    parser.add_argument("--train-draw-window", type=int, default=128)
    parser.add_argument("--recent-draw-window", type=int, default=52)
    parser.add_argument("--refit-interval", type=int, default=5)
    parser.add_argument("--negative-samples", type=int, default=160)
    parser.add_argument("--seed", type=int, default=417)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--results", type=Path, default=RESULT_PATH)
    args = parser.parse_args()
    dataset = load_draw_history()
    validator = StrictWalkForwardValidator(
        dataset,
        start_draw_no=args.start_draw_no,
        end_draw_no=args.end_draw_no,
        train_draw_window=args.train_draw_window,
        recent_draw_window=args.recent_draw_window,
        refit_interval=args.refit_interval,
        negative_samples=args.negative_samples,
        seed=args.seed,
    )
    results = validator.run()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.report.write_text(render_report(results), encoding="utf-8")
    print(f"TEMPORAL_FIREWALL: {results['temporal_firewall_status']}")
    print(f"TEST_DRAWS: {len(results['steps'])}")
    print(f"REPORT: {args.report}")
    print(f"RESULTS: {args.results}")


if __name__ == "__main__":
    main()
