#!/usr/bin/env python3
"""Step 164 offline full-history engine training foundation.

This module is deliberately separate from live prediction serving. It reads
DrawHistory (and optionally PredictionLedger for Group D diagnostics), writes
local JSON artifacts, and never executes database write SQL.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

ARTIFACT_VERSION = "step164.v1"
PHASE1_MAX_DRAW_NO = 4050
ENGINE_GROUPS = ("A", "B", "C", "D")
TRAINING_MODES = (
    "phase1_base",
    "rolling_origin_phase2",
    "retrospective_full_history",
)
RETROSPECTIVE_LABEL = "RETROSPECTIVE_FULL_HISTORY_NOT_FOR_LIVE_PREDICTION"
REQUIRED_ARTIFACT_FIELDS = (
    "artifact_version",
    "engine_group",
    "engine_name",
    "training_mode",
    "worker_id",
    "draw_cutoff",
    "training_pair_count",
    "training_draw_range",
    "day_type",
    "model_type",
    "modulus",
    "formula_space",
    "matrix_m",
    "bias_b",
    "coefficients",
    "feature_stats",
    "score_semantics",
    "residual_summary",
    "rank_condition",
    "singular_values_summary",
    "created_at_local",
    "created_at_utc",
    "temporal_firewall_status",
    "not_for_live_prediction",
    "sha256_hash",
)

DRAW_HISTORY_SQL = """
    SELECT
        DrawNo,
        CONVERT(varchar(10), DrawDate, 120) AS DrawDateText,
        CASE
            WHEN DATENAME(WEEKDAY, DrawDate) IN ('Wednesday', 'Saturday', 'Sunday')
                THEN DATENAME(WEEKDAY, DrawDate)
            ELSE 'Special'
        END AS DayType,
        WinningNumbers
    FROM dbo.DrawHistory
    WHERE WinningNumbers IS NOT NULL
    ORDER BY DrawNo;
"""

LEDGER_READ_SQL = """
    SELECT
        EngineSource,
        Mode,
        SourceDrawNo,
        TargetDrawNo,
        RankNo,
        PredictedNumber,
        Score,
        VerificationStatus,
        HitCount
    FROM dbo.PredictionLedger
    WHERE TargetDrawNo <= ?
    ORDER BY SourceDrawNo, TargetDrawNo, EngineSource, Mode, RankNo;
"""

DRAW_SCHEMA_SQL = """
    SELECT
        COLUMN_NAME,
        DATA_TYPE,
        IS_NULLABLE,
        CHARACTER_MAXIMUM_LENGTH
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = ?
      AND TABLE_NAME = ?
    ORDER BY ORDINAL_POSITION;
"""


@dataclass(frozen=True)
class TrainingDraw:
    draw_no: int
    draw_date: str
    day_type: str
    winners: tuple[str, ...]


@dataclass(frozen=True)
class TrainingPair:
    source_draw_no: int
    target_draw_no: int
    source_date: str
    target_date: str
    source_day_type: str
    target_day_type: str
    source_digits: tuple[tuple[int, int, int, int], ...]
    target_digits: tuple[tuple[int, int, int, int], ...]
    source_winners_23: tuple[str, ...]
    target_winners_23: tuple[str, ...]


class ChronologicalTrainingDataset:
    """Immutable chronological dataset with cutoff-enforced pair access."""

    def __init__(
        self,
        draws: Sequence[TrainingDraw],
        *,
        draw_history_schema: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        ordered = sorted(draws, key=lambda draw: draw.draw_no)
        numbers = [draw.draw_no for draw in ordered]
        if not ordered:
            raise ValueError("DrawHistory dataset is empty")
        if len(numbers) != len(set(numbers)):
            raise ValueError("DrawHistory contains duplicate DrawNo values")
        self.draws = tuple(ordered)
        self._by_no = {draw.draw_no: draw for draw in ordered}
        self._pairs = tuple(
            self._make_pair(source, target)
            for source, target in zip(ordered, ordered[1:])
        )
        self.draw_history_schema = tuple(draw_history_schema or ())

    @staticmethod
    def _make_pair(source: TrainingDraw, target: TrainingDraw) -> TrainingPair:
        return TrainingPair(
            source_draw_no=source.draw_no,
            target_draw_no=target.draw_no,
            source_date=source.draw_date,
            target_date=target.draw_date,
            source_day_type=source.day_type,
            target_day_type=target.day_type,
            source_digits=tuple(digits(number) for number in source.winners),
            target_digits=tuple(digits(number) for number in target.winners),
            source_winners_23=source.winners,
            target_winners_23=target.winners,
        )

    @property
    def first_draw_no(self) -> int:
        return self.draws[0].draw_no

    @property
    def last_draw_no(self) -> int:
        return self.draws[-1].draw_no

    def pairs_until(self, source_draw_no: int) -> tuple[TrainingPair, ...]:
        cutoff = int(source_draw_no)
        pairs = tuple(pair for pair in self._pairs if pair.target_draw_no <= cutoff)
        if any(pair.target_draw_no > cutoff for pair in pairs):
            raise RuntimeError("Temporal firewall violation in pairs_until")
        return pairs

    def phase1_pairs(self) -> tuple[TrainingPair, ...]:
        return self.pairs_until(PHASE1_MAX_DRAW_NO)

    def phase2_pairs_until(self, source_draw_no: int) -> tuple[TrainingPair, ...]:
        if source_draw_no <= PHASE1_MAX_DRAW_NO:
            raise ValueError("Phase2 cutoff must be greater than 4050")
        return self.pairs_until(source_draw_no)

    def retrospective_pairs(self) -> tuple[TrainingPair, ...]:
        return self._pairs

    def metadata(self) -> dict[str, Any]:
        day_types = Counter(draw.day_type for draw in self.draws)
        winner_counts = Counter(len(draw.winners) for draw in self.draws)
        return {
            "draw_count": len(self.draws),
            "draw_no_range": [self.first_draw_no, self.last_draw_no],
            "draw_date_range": [self.draws[0].draw_date, self.draws[-1].draw_date],
            "day_type_distribution": dict(sorted(day_types.items())),
            "winner_count_distribution": {
                str(key): value for key, value in sorted(winner_counts.items())
            },
            "phase1_draw_count": sum(
                draw.draw_no <= PHASE1_MAX_DRAW_NO for draw in self.draws
            ),
            "phase2_draw_count": sum(
                draw.draw_no > PHASE1_MAX_DRAW_NO for draw in self.draws
            ),
            "consecutive_pair_count": len(self._pairs),
            "draw_history_schema": list(self.draw_history_schema),
            "engine_foundation_discovery": {
                "chronological_draw_cache_exists": True,
                "live_training_window_default": 64,
                "offline_full_history_window_zero_supported": True,
                "production_default_changed": False,
            },
        }


def digits(number: str) -> tuple[int, int, int, int]:
    value = str(number).strip().zfill(4)
    if len(value) != 4 or not value.isdigit():
        raise ValueError(f"Invalid 4D number: {number!r}")
    return tuple(int(char) for char in value)  # type: ignore[return-value]


def parse_winners(raw: object) -> tuple[str, ...]:
    output = []
    for value in str(raw or "").replace(" ", "").split(","):
        if not value:
            continue
        number = value.zfill(4)
        if len(number) == 4 and number.isdigit():
            output.append(number)
    return tuple(output)


def load_draw_history() -> ChronologicalTrainingDataset:
    from app.core.config import get_settings
    import pyodbc  # type: ignore

    connection = pyodbc.connect(
        get_settings().sql_connection_string(),
        timeout=120,
        autocommit=True,
    )
    cursor = connection.cursor()
    try:
        cursor.execute(DRAW_SCHEMA_SQL, "dbo", "DrawHistory")
        schema_rows = cursor.fetchall()
        cursor.execute(DRAW_HISTORY_SQL)
        rows = cursor.fetchall()
    finally:
        cursor.close()
        connection.close()
    draws = [
        TrainingDraw(
            draw_no=int(row.DrawNo),
            draw_date=str(row.DrawDateText),
            day_type=str(row.DayType),
            winners=parse_winners(row.WinningNumbers),
        )
        for row in rows
        if parse_winners(row.WinningNumbers)
    ]
    schema = [
        {
            "column_name": str(row.COLUMN_NAME),
            "data_type": str(row.DATA_TYPE),
            "nullable": str(row.IS_NULLABLE),
            "max_length": (
                int(row.CHARACTER_MAXIMUM_LENGTH)
                if row.CHARACTER_MAXIMUM_LENGTH is not None
                else None
            ),
        }
        for row in schema_rows
    ]
    return ChronologicalTrainingDataset(
        draws,
        draw_history_schema=schema,
    )


def load_ledger_read_only(cutoff: int) -> list[dict[str, Any]]:
    from app.core.config import get_settings
    import pyodbc  # type: ignore

    connection = pyodbc.connect(
        get_settings().sql_connection_string(),
        timeout=120,
        autocommit=True,
    )
    cursor = connection.cursor()
    try:
        cursor.execute(LEDGER_READ_SQL, int(cutoff))
        rows = cursor.fetchall()
    finally:
        cursor.close()
        connection.close()
    return [
        {
            "engine_source": str(row.EngineSource),
            "mode": str(row.Mode),
            "source_draw_no": int(row.SourceDrawNo),
            "target_draw_no": int(row.TargetDrawNo),
            "rank": int(row.RankNo),
            "number": str(row.PredictedNumber).zfill(4),
            "score": float(row.Score) if row.Score is not None else None,
            "verification_status": str(row.VerificationStatus),
            "hit_count": int(row.HitCount) if row.HitCount is not None else None,
        }
        for row in rows
    ]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def artifact_hash(artifact: dict[str, Any]) -> str:
    payload = {key: value for key, value in artifact.items() if key != "sha256_hash"}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def validate_artifact_schema(artifact: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"Artifact missing fields: {missing}")
    if artifact["formula_space"] == "BASE5" and artifact["modulus"] != 5:
        raise ValueError("BASE5 artifact must declare modulus=5")
    if artifact["formula_space"] == "BASE10" and artifact["modulus"] != 10:
        raise ValueError("BASE10 artifact must declare modulus=10")
    if artifact["training_mode"] == "retrospective_full_history":
        if not artifact["not_for_live_prediction"]:
            raise ValueError("Retrospective artifact must be not-for-live")
        if artifact.get("retrospective_label") != RETROSPECTIVE_LABEL:
            raise ValueError("Retrospective artifact label is missing")
    expected = artifact_hash(artifact)
    if artifact["sha256_hash"] != expected:
        raise ValueError("Artifact hash mismatch")


def make_artifact(
    *,
    engine_group: str,
    engine_name: str,
    training_mode: str,
    worker_id: int,
    draw_cutoff: int,
    pairs: Sequence[TrainingPair],
    day_type: str,
    model_type: str,
    modulus: int | None,
    formula_space: str,
    matrix_m: Any = None,
    bias_b: Any = None,
    coefficients: Any = None,
    feature_stats: dict[str, Any] | None = None,
    score_semantics: str,
    residual_summary: dict[str, Any] | None = None,
    rank_condition: dict[str, Any] | None = None,
    singular_values_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now_local = datetime.now().astimezone()
    now_utc = datetime.now(timezone.utc)
    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "engine_group": engine_group,
        "engine_name": engine_name,
        "training_mode": training_mode,
        "worker_id": int(worker_id),
        "draw_cutoff": int(draw_cutoff),
        "training_pair_count": len(pairs),
        "training_draw_range": (
            [pairs[0].source_draw_no, pairs[-1].target_draw_no] if pairs else None
        ),
        "day_type": day_type,
        "model_type": model_type,
        "modulus": modulus,
        "formula_space": formula_space,
        "matrix_m": matrix_m,
        "bias_b": bias_b,
        "coefficients": coefficients,
        "feature_stats": feature_stats or {},
        "score_semantics": score_semantics,
        "residual_summary": residual_summary or {},
        "rank_condition": rank_condition or {},
        "singular_values_summary": singular_values_summary or {},
        "created_at_local": now_local.isoformat(),
        "created_at_utc": now_utc.isoformat(),
        "temporal_firewall_status": (
            "PASS_TARGET_DRAW_LE_CUTOFF"
            if all(pair.target_draw_no <= draw_cutoff for pair in pairs)
            else "FAIL"
        ),
        "not_for_live_prediction": training_mode == "retrospective_full_history",
        "retrospective_label": (
            RETROSPECTIVE_LABEL
            if training_mode == "retrospective_full_history"
            else None
        ),
        "sha256_hash": "",
    }
    artifact["sha256_hash"] = artifact_hash(artifact)
    validate_artifact_schema(artifact)
    return artifact


class NormalEquations:
    def __init__(self, feature_count: int, output_count: int = 4) -> None:
        self.xtx = np.zeros((feature_count, feature_count), dtype=np.float64)
        self.xty = np.zeros((feature_count, output_count), dtype=np.float64)
        self.yty = np.zeros((output_count, output_count), dtype=np.float64)
        self.weight_sum = 0.0
        self.sample_count = 0

    def add(self, x: np.ndarray, y: np.ndarray, weight: float = 1.0) -> None:
        self.xtx += weight * np.outer(x, x)
        self.xty += weight * np.outer(x, y)
        self.yty += weight * np.outer(y, y)
        self.weight_sum += weight
        self.sample_count += 1

    def add_cross_product_block(
        self,
        source: np.ndarray,
        target: np.ndarray,
        weight: float = 1.0,
    ) -> None:
        design = np.hstack(
            [source.astype(np.float64), np.ones((source.shape[0], 1))]
        )
        target_float = target.astype(np.float64)
        ns, nt = source.shape[0], target.shape[0]
        self.xtx += weight * nt * (design.T @ design)
        self.xty += weight * np.outer(design.sum(axis=0), target_float.sum(axis=0))
        self.yty += weight * ns * (target_float.T @ target_float)
        self.weight_sum += weight * ns * nt
        self.sample_count += ns * nt

    def solve(self) -> dict[str, Any]:
        coefficients, _, rank, singular = np.linalg.lstsq(
            self.xtx, self.xty, rcond=None
        )
        residual_matrix = (
            self.yty
            - 2.0 * coefficients.T @ self.xty
            + coefficients.T @ self.xtx @ coefficients
        )
        residual_diagonal = np.maximum(np.diag(residual_matrix), 0.0)
        condition = (
            float(singular[0] / singular[-1])
            if len(singular) and singular[-1] > 0
            else None
        )
        return {
            "coefficients": coefficients,
            "rank": int(rank),
            "singular_values": singular,
            "condition_number": condition,
            "residual_sse_by_output": residual_diagonal,
            "residual_sse_total": float(residual_diagonal.sum()),
            "weighted_sample_mass": self.weight_sum,
            "sample_count": self.sample_count,
        }


def pair_arrays(pair: TrainingPair, *, modulus: int = 10) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(pair.source_digits, dtype=np.float64) % modulus
    target = np.asarray(pair.target_digits, dtype=np.float64) % modulus
    return source, target


def observation_count(pairs: Sequence[TrainingPair]) -> int:
    return sum(len(pair.source_digits) * len(pair.target_digits) for pair in pairs)


def fit_affine(
    pairs: Sequence[TrainingPair],
    *,
    modulus: int,
    day_type: str | None = None,
    weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    selected = [
        pair for pair in pairs if day_type is None or pair.source_day_type == day_type
    ]
    equations = NormalEquations(5, 4)
    for index, pair in enumerate(selected):
        source, target = pair_arrays(pair, modulus=modulus)
        equations.add_cross_product_block(
            source, target, 1.0 if weights is None else float(weights[index])
        )
    if equations.sample_count < 20:
        raise ValueError(f"Need at least 20 observations, got {equations.sample_count}")
    solved = equations.solve()
    raw = solved["coefficients"]
    matrix = np.mod(np.rint(raw[:4, :].T).astype(np.int64), modulus)
    bias = np.mod(np.rint(raw[4, :]).astype(np.int64), modulus)
    return {
        **solved,
        "matrix_m": matrix,
        "bias_b": bias,
        "selected_pair_count": len(selected),
    }


def iter_cross_observations(
    pairs: Sequence[TrainingPair],
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    for pair in pairs:
        source, target = pair_arrays(pair)
        for source_row in source:
            x = np.append(source_row, 1.0)
            for target_row in target:
                yield x, target_row


def tail_observations(
    pairs: Sequence[TrainingPair], limit: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    output: list[tuple[np.ndarray, np.ndarray]] = []
    for pair in reversed(pairs):
        source, target = pair_arrays(pair)
        for source_row in reversed(source):
            x = np.append(source_row, 1.0)
            for target_row in reversed(target):
                output.append((x, target_row))
                if len(output) >= limit:
                    output.reverse()
                    return output
    output.reverse()
    return output


def fit_wls(
    pairs: Sequence[TrainingPair],
    *,
    decay: float,
    window_size: int,
) -> dict[str, Any]:
    total = observation_count(pairs)
    if window_size:
        observations: Iterable[tuple[np.ndarray, np.ndarray]] = tail_observations(
            pairs, min(window_size, total)
        )
        count = min(window_size, total)
    else:
        observations = iter_cross_observations(pairs)
        count = total
    equations = NormalEquations(5, 4)
    for index, (x, y) in enumerate(observations):
        weight = decay ** (count - index - 1)
        equations.add(x, y, weight)
    solved = equations.solve()
    raw = solved["coefficients"]
    matrix = np.mod(np.rint(raw[:4, :].T).astype(np.int64), 10)
    bias = np.mod(np.rint(raw[4, :]).astype(np.int64), 10)
    return {
        **solved,
        "matrix_m": matrix,
        "bias_b": bias,
        "effective_observation_count": count,
        "oldest_retained_weight": decay ** max(0, count - 1),
    }


def delta_sequence(pairs: Sequence[TrainingPair]) -> np.ndarray:
    rows = []
    for pair in pairs:
        source, target = pair_arrays(pair)
        count = min(source.shape[0], target.shape[0])
        rows.extend(((target[:count] - source[:count]) % 10).tolist())
    return np.asarray(rows, dtype=np.float64)


def fit_delta_affine(pairs: Sequence[TrainingPair], *, cap: int = 0) -> dict[str, Any]:
    sequence = delta_sequence(pairs)
    if cap:
        sequence = sequence[-cap:]
    if sequence.shape[0] < 6:
        raise ValueError("Delta sequence is too short")
    equations = NormalEquations(5, 4)
    for source, target in zip(sequence[:-1], sequence[1:]):
        equations.add(np.append(source, 1.0), target)
    solved = equations.solve()
    raw = solved["coefficients"]
    return {
        **solved,
        "matrix_m": np.mod(np.rint(raw[:4, :].T).astype(np.int64), 10),
        "bias_b": np.mod(np.rint(raw[4, :]).astype(np.int64), 10),
        "delta_rows": int(sequence.shape[0]),
    }


def polynomial_features(values: np.ndarray) -> np.ndarray:
    a, b, c, d = (values[:, index] for index in range(4))
    return np.column_stack(
        [
            values,
            values * values,
            a * b,
            b * c,
            c * d,
            d * a,
            np.ones(values.shape[0]),
        ]
    )


def fit_polynomial(pairs: Sequence[TrainingPair]) -> dict[str, Any]:
    equations = NormalEquations(13, 4)
    for pair in pairs:
        source, target = pair_arrays(pair)
        features = polynomial_features(source)
        ns, nt = source.shape[0], target.shape[0]
        equations.xtx += nt * (features.T @ features)
        equations.xty += np.outer(features.sum(axis=0), target.sum(axis=0))
        equations.yty += ns * (target.T @ target)
        equations.weight_sum += ns * nt
        equations.sample_count += ns * nt
    return equations.solve()


def solve_payload(result: dict[str, Any]) -> dict[str, Any]:
    singular = result["singular_values"]
    return {
        "matrix_m": (
            result["matrix_m"].astype(int).tolist()
            if "matrix_m" in result
            else None
        ),
        "bias_b": (
            result["bias_b"].astype(int).tolist()
            if "bias_b" in result
            else None
        ),
        "coefficients": result["coefficients"].astype(float).tolist(),
        "feature_stats": {
            "sample_count": int(result["sample_count"]),
            "weighted_sample_mass": float(result["weighted_sample_mass"]),
            **{
                key: value
                for key, value in result.items()
                if key
                in {
                    "selected_pair_count",
                    "effective_observation_count",
                    "oldest_retained_weight",
                    "delta_rows",
                }
            },
        },
        "residual_summary": {
            "sse_total": float(result["residual_sse_total"]),
            "sse_by_output": result["residual_sse_by_output"].astype(float).tolist(),
            "mse_per_observation": (
                float(result["residual_sse_total"]) / result["sample_count"]
                if result["sample_count"]
                else None
            ),
        },
        "rank_condition": {
            "rank": int(result["rank"]),
            "full_rank": int(result["rank"]) == result["coefficients"].shape[0],
            "condition_number": result["condition_number"],
        },
        "singular_values_summary": {
            "count": len(singular),
            "maximum": float(singular[0]) if len(singular) else None,
            "minimum": float(singular[-1]) if len(singular) else None,
        },
    }


def fit_e2_set_projector_bias(
    pairs: Sequence[TrainingPair],
    *,
    day_type: str,
) -> dict[str, Any]:
    """
    Train E2_SET_PROJECTOR without destroying its intended semantics.

    The E2 structure stays fixed:
        y0 = x0 + x1 + x2 + b0
        y1 = x1 + x2 + x3 + b1
        y2 = x0 + x2 + x3 + b2
        y3 = x0 + x1 + x3 + b3

    Only b is learned from historical source-target 23x23 cross-product pairs.
    """
    matrix_m = np.array(
        [
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
        ],
        dtype=np.int16,
    )

    residual_counts = [Counter() for _ in range(4)]
    observation_count = 0

    for pair in pairs:
        for source_number in pair.source_winners_23:
            source_digits = np.array(digits(source_number), dtype=np.int16)
            projected = (matrix_m @ source_digits) % 10

            for target_number in pair.target_winners_23:
                target_digits = np.array(digits(target_number), dtype=np.int16)
                residual = (target_digits - projected) % 10
                for position in range(4):
                    residual_counts[position][int(residual[position])] += 1
                observation_count += 1

    bias_b = np.array(
        [
            sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]
            if counter
            else 0
            for counter in residual_counts
        ],
        dtype=np.int16,
    )

    sse_by_output = np.zeros(4, dtype=np.float64)
    exact_position_hits = np.zeros(4, dtype=np.int64)
    exact_vector_hits = 0

    for pair in pairs:
        for source_number in pair.source_winners_23:
            source_digits = np.array(digits(source_number), dtype=np.int16)
            predicted = (matrix_m @ source_digits + bias_b) % 10

            for target_number in pair.target_winners_23:
                target_digits = np.array(digits(target_number), dtype=np.int16)
                raw_error = (target_digits - predicted) % 10
                circular_error = np.minimum(raw_error, 10 - raw_error)
                sse_by_output += circular_error.astype(np.float64) ** 2
                exact_position_hits += (predicted == target_digits).astype(np.int64)
                exact_vector_hits += int(np.array_equal(predicted, target_digits))

    sse_total = float(sse_by_output.sum())
    position_hit_rates = [
        float(value) / observation_count if observation_count else None
        for value in exact_position_hits
    ]

    return {
        "matrix_m": matrix_m.astype(int).tolist(),
        "bias_b": bias_b.astype(int).tolist(),
        "feature_stats": {
            "relationship": "fixed E2 set-projector matrix with learned modal residual bias",
            "day_type_scope": day_type,
            "observation_count": observation_count,
            "source_target_cross_product": "23x23",
            "residual_modal_counts_by_position": [
                dict(sorted(counter.items())) for counter in residual_counts
            ],
            "exact_position_hits": exact_position_hits.astype(int).tolist(),
            "exact_position_hit_rates": position_hit_rates,
            "exact_vector_hits": int(exact_vector_hits),
            "production_replacement": False,
            "training_semantics": "bias-only learned correction; matrix remains E2_SET_PROJECTOR structure",
        },
        "residual_summary": {
            "sse_total": sse_total,
            "sse_by_output": sse_by_output.astype(float).tolist(),
            "mse_per_observation": (
                sse_total / observation_count if observation_count else None
            ),
            "error_metric": "circular_digit_distance_squared",
        },
        "rank_condition": {
            "rank": None,
            "full_rank": None,
            "condition_number": None,
            "note": "No matrix solve; fixed projector matrix with modal residual bias.",
        },
        "singular_values_summary": {
            "count": 0,
            "maximum": None,
            "minimum": None,
            "note": "Not applicable to modal residual bias training.",
        },
    }


def train_group_a(
    pairs: Sequence[TrainingPair],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    artifacts = []
    day_types = sorted({pair.source_day_type for pair in pairs})
    for day_type in ["ALL", *day_types]:
        selected_pairs = (
            list(pairs)
            if day_type == "ALL"
            else [pair for pair in pairs if pair.source_day_type == day_type]
        )
        if len(selected_pairs) < 20:
            continue
        result = fit_affine(
            selected_pairs,
            modulus=10,
        )
        solved = solve_payload(result)
        artifacts.append(
            make_artifact(
                **context,
                engine_group="A",
                engine_name=f"LEARNED_CROSS_PAIR_AFFINE__{day_type}",
                pairs=selected_pairs,
                day_type=day_type,
                model_type="AFFINE_LSTSQ_NORMAL_EQUATIONS",
                modulus=10,
                formula_space="BASE10",
                matrix_m=solved["matrix_m"],
                bias_b=solved["bias_b"],
                coefficients=solved["coefficients"],
                feature_stats={
                    **solved["feature_stats"],
                    "relationship": "23x23 source-target winner cross product",
                    "production_replacement": False,
                },
                score_semantics="Offline fitted transform; not a calibrated probability.",
                residual_summary=solved["residual_summary"],
                rank_condition=solved["rank_condition"],
                singular_values_summary=solved["singular_values_summary"],
            )
        )

    for day_type in ["ALL", *day_types]:
        selected_pairs = (
            list(pairs)
            if day_type == "ALL"
            else [pair for pair in pairs if pair.source_day_type == day_type]
        )
        if len(selected_pairs) < 20:
            continue
        e2 = fit_e2_set_projector_bias(selected_pairs, day_type=day_type)
        artifacts.append(
            make_artifact(
                **context,
                engine_group="A",
                engine_name=f"E2_SET_PROJECTOR_LEARNED_BIAS__{day_type}",
                pairs=selected_pairs,
                day_type=day_type,
                model_type="FIXED_SET_PROJECTOR_MODAL_RESIDUAL_BIAS",
                modulus=10,
                formula_space="BASE10",
                matrix_m=e2["matrix_m"],
                bias_b=e2["bias_b"],
                feature_stats=e2["feature_stats"],
                score_semantics=(
                    "E2 fixed set-projector matrix with learned modal residual bias; "
                    "not a calibrated probability."
                ),
                residual_summary=e2["residual_summary"],
                rank_condition=e2["rank_condition"],
                singular_values_summary=e2["singular_values_summary"],
            )
        )

    for cap, suffix in ((64, "CAP64"), (0, "FULL")):
        result = fit_delta_affine(pairs, cap=cap)
        solved = solve_payload(result)
        artifacts.append(
            make_artifact(
                **context,
                engine_group="A",
                engine_name=f"DELTA_ROTATION_AFFINE__{suffix}",
                pairs=pairs,
                day_type="ALL",
                model_type="DELTA_AFFINE_LSTSQ",
                modulus=10,
                formula_space="BASE10",
                matrix_m=solved["matrix_m"],
                bias_b=solved["bias_b"],
                coefficients=solved["coefficients"],
                feature_stats={**solved["feature_stats"], "cap": cap},
                score_semantics="Offline delta transform; not a probability.",
                residual_summary=solved["residual_summary"],
                rank_condition=solved["rank_condition"],
                singular_values_summary=solved["singular_values_summary"],
            )
        )

    polynomial = solve_payload(fit_polynomial(pairs))
    artifacts.append(
        make_artifact(
            **context,
            engine_group="A",
            engine_name="POLYNOMIAL_AFFINE_LEARNED__ALL",
            pairs=pairs,
            day_type="ALL",
            model_type="POLYNOMIAL_FEATURE_LSTSQ",
            modulus=10,
            formula_space="BASE10",
            coefficients=polynomial["coefficients"],
            feature_stats={
                **polynomial["feature_stats"],
                "features": "digits,squares,adjacent_products,bias",
                "existing_production_polynomial_is_hardcoded": True,
            },
            score_semantics="Offline polynomial regression residual scale.",
            residual_summary=polynomial["residual_summary"],
            rank_condition=polynomial["rank_condition"],
            singular_values_summary=polynomial["singular_values_summary"],
        )
    )
    return artifacts


def train_group_b(
    pairs: Sequence[TrainingPair],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    variants = [
        ("WLS_CAP64_DECAY_0.98", 0.98, 64),
        ("WLS_FULL_DECAY_0.98", 0.98, 0),
        ("WLS_FULL_DECAY_0.995", 0.995, 0),
        ("WLS_FULL_DECAY_0.999", 0.999, 0),
        *[
            (f"WLS_WINDOW_{window}_DECAY_0.98", 0.98, window)
            for window in (128, 256, 512, 1000, 4050)
        ],
    ]
    artifacts = []
    for engine_name, decay, window in variants:
        result = fit_wls(pairs, decay=decay, window_size=window)
        solved = solve_payload(result)
        artifacts.append(
            make_artifact(
                **context,
                engine_group="B",
                engine_name=engine_name,
                pairs=pairs,
                day_type="ALL",
                model_type="WEIGHTED_AFFINE_LSTSQ",
                modulus=10,
                formula_space="BASE10",
                matrix_m=solved["matrix_m"],
                bias_b=solved["bias_b"],
                coefficients=solved["coefficients"],
                feature_stats={
                    **solved["feature_stats"],
                    "decay": decay,
                    "window_size": window,
                    "window_unit": "cross_product_observations",
                    "training_window_size_zero_means_full_history": window == 0,
                    "production_default_changed": False,
                },
                score_semantics="Exponentially weighted regression fit; not probability.",
                residual_summary=solved["residual_summary"],
                rank_condition=solved["rank_condition"],
                singular_values_summary=solved["singular_values_summary"],
            )
        )
    return artifacts


def markov_transition_mass_profile(
    pairs: Sequence[TrainingPair],
) -> dict[str, Any]:
    """
    Build an artifact-trained E4_MARKOV_TRANSITION_MASS profile.

    This is an offline training artifact for audit/packaging. It does not replace
    the runtime causal cache. Live/replay usage must still apply source_draw_no
    cutoff before selecting transitions.
    """
    transition_counts: dict[str, Counter[str]] = defaultdict(Counter)
    day_transition_counts: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    last_seen_draw_no: dict[str, dict[str, int]] = defaultdict(dict)
    day_last_seen_draw_no: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    source_state_count = 0
    target_state_count = 0
    transition_observation_count = 0

    for pair in pairs:
        target_draw_no = int(pair.target_draw_no)
        day_type = pair.source_day_type

        for source_number in pair.source_winners_23:
            source_state = str(source_number).zfill(4)
            source_state_count += 1

            for target_number in pair.target_winners_23:
                target_state = str(target_number).zfill(4)
                target_state_count += 1
                transition_observation_count += 1

                transition_counts[source_state][target_state] += 1
                day_transition_counts[day_type][source_state][target_state] += 1
                last_seen_draw_no[source_state][target_state] = max(
                    last_seen_draw_no[source_state].get(target_state, -1),
                    target_draw_no,
                )
                day_last_seen_draw_no[day_type][source_state][target_state] = max(
                    day_last_seen_draw_no[day_type][source_state].get(
                        target_state,
                        -1,
                    ),
                    target_draw_no,
                )

    def summarize_bucket(
        counts_by_source: dict[str, Counter[str]],
        seen_by_source: dict[str, dict[str, int]],
        *,
        top_n: int = 25,
    ) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for source_state in sorted(counts_by_source):
            ranked = []
            total = sum(counts_by_source[source_state].values())
            for target_state, count in counts_by_source[source_state].items():
                ranked.append(
                    {
                        "target_state": target_state,
                        "transition_count": int(count),
                        "last_seen_draw_no": int(
                            seen_by_source.get(source_state, {}).get(target_state, -1)
                        ),
                        "empirical_probability": (
                            float(count) / total if total else None
                        ),
                    }
                )
            ranked.sort(
                key=lambda row: (
                    -row["transition_count"],
                    -row["last_seen_draw_no"],
                    row["target_state"],
                )
            )
            output[source_state] = {
                "total_transition_count": int(total),
                "unique_target_count": len(ranked),
                "top_targets": ranked[:top_n],
            }
        return output

    top_all = summarize_bucket(transition_counts, last_seen_draw_no)
    top_by_day = {
        day_type: summarize_bucket(
            dict(day_transition_counts[day_type]),
            dict(day_last_seen_draw_no[day_type]),
        )
        for day_type in sorted(day_transition_counts)
    }

    source_totals = [
        payload["total_transition_count"] for payload in top_all.values()
    ]
    unique_target_totals = [
        payload["unique_target_count"] for payload in top_all.values()
    ]

    return {
        "relationship": "source 4-digit winner state -> next draw target 4-digit winner state",
        "source_target_cross_product": "23x23",
        "transition_observation_count": int(transition_observation_count),
        "source_state_observation_count": int(source_state_count),
        "target_state_observation_count": int(target_state_count),
        "unique_source_state_count": len(top_all),
        "day_type_count": len(top_by_day),
        "ordering": "transition_count DESC, last_seen_draw_no DESC, target_state ASC",
        "top_n_per_source": 25,
        "source_total_summary": {
            "minimum": int(min(source_totals)) if source_totals else 0,
            "maximum": int(max(source_totals)) if source_totals else 0,
            "mean": (
                float(sum(source_totals)) / len(source_totals)
                if source_totals
                else None
            ),
        },
        "unique_target_summary": {
            "minimum": int(min(unique_target_totals)) if unique_target_totals else 0,
            "maximum": int(max(unique_target_totals)) if unique_target_totals else 0,
            "mean": (
                float(sum(unique_target_totals)) / len(unique_target_totals)
                if unique_target_totals
                else None
            ),
        },
        "transition_mass_top25_by_source": top_all,
        "transition_mass_top25_by_day_type_and_source": top_by_day,
        "temporal_firewall_note": (
            "Artifact is trained only within its cutoff. Runtime/replay must still "
            "apply source_draw_no cutoff before selecting transitions."
        ),
        "production_replacement": False,
    }


def structural_profiles(pairs: Sequence[TrainingPair]) -> dict[str, Any]:
    base5_transitions = [
        [[0 for _ in range(5)] for _ in range(5)] for _ in range(4)
    ]
    zero_counts: Counter[int] = Counter()
    repeated_counts: Counter[int] = Counter()
    digit_sum_bands: Counter[str] = Counter()
    first_pairs: Counter[str] = Counter()
    last_pairs: Counter[str] = Counter()
    adjacent_pairs = [Counter() for _ in range(3)]
    relations = Counter()
    target_count = 0

    for pair in pairs:
        for source_number in pair.source_winners_23:
            source_digits = digits(source_number)
            for target_number in pair.target_winners_23:
                target_digits = digits(target_number)
                for position in range(4):
                    base5_transitions[position][source_digits[position] % 5][
                        target_digits[position] % 5
                    ] += 1
                relations["reverse"] += int(target_number == source_number[::-1])
                mirror = "".join(str((value + 5) % 10) for value in source_digits)
                relations["full_mirror"] += int(target_number == mirror)
                relations["box"] += int(
                    sorted(target_number) == sorted(source_number)
                )
        for target_number in pair.target_winners_23:
            target_count += 1
            values = digits(target_number)
            zero_counts[target_number.count("0")] += 1
            repeated_counts[4 - len(set(target_number))] += 1
            total = sum(values)
            band = f"{(total // 5) * 5:02d}-{(total // 5) * 5 + 4:02d}"
            digit_sum_bands[band] += 1
            first_pairs[target_number[:2]] += 1
            last_pairs[target_number[2:]] += 1
            for position in range(3):
                adjacent_pairs[position][target_number[position : position + 2]] += 1
    return {
        "base5_transition_counts": base5_transitions,
        "base5_expansion_factor_per_vector": 16,
        "target_number_count": target_count,
        "zero_count_profile": dict(sorted(zero_counts.items())),
        "repeated_digit_profile": dict(sorted(repeated_counts.items())),
        "digit_sum_bands": dict(sorted(digit_sum_bands.items())),
        "first_pair_top50": first_pairs.most_common(50),
        "last_pair_top50": last_pairs.most_common(50),
        "adjacent_pair_top50": [
            counter.most_common(50) for counter in adjacent_pairs
        ],
        "relationship_counts": dict(relations),
    }


def train_group_c(
    pairs: Sequence[TrainingPair],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    mirror = solve_payload(fit_affine(pairs, modulus=5))
    profile = structural_profiles(pairs)
    return [
        make_artifact(
            **context,
            engine_group="C",
            engine_name="MIRROR_BASE5_AFFINE__ALL",
            pairs=pairs,
            day_type="ALL",
            model_type="BASE5_AFFINE_LSTSQ",
            modulus=5,
            formula_space="BASE5",
            matrix_m=mirror["matrix_m"],
            bias_b=mirror["bias_b"],
            coefficients=mirror["coefficients"],
            feature_stats={
                **mirror["feature_stats"],
                "semantic_validation": "PASS_BASE5_MODULUS_5",
                "base10_application_prohibited": True,
            },
            score_semantics="Base5 class projection; expanded variants require separate ranking.",
            residual_summary=mirror["residual_summary"],
            rank_condition=mirror["rank_condition"],
            singular_values_summary=mirror["singular_values_summary"],
        ),
        make_artifact(
            **context,
            engine_group="C",
            engine_name="BASE5_STRUCTURAL_PROFILE__ALL",
            pairs=pairs,
            day_type="ALL",
            model_type="STRUCTURAL_FREQUENCY_PROFILE",
            modulus=5,
            formula_space="BASE5",
            feature_stats=profile,
            score_semantics="Counts/frequencies only; no calibrated candidate score.",
        ),
        make_artifact(
            **context,
            engine_group="C",
            engine_name="E4_MARKOV_TRANSITION_MASS__ALL",
            pairs=pairs,
            day_type="ALL",
            model_type="CAUSAL_MARKOV_TRANSITION_MASS_PROFILE",
            modulus=10,
            formula_space="BASE10",
            feature_stats=markov_transition_mass_profile(pairs),
            score_semantics=(
                "Artifact-trained Markov transition counts. Scores are empirical "
                "transition frequencies, not calibrated probabilities."
            ),
        ),
    ]


def temporal_profiles(pairs: Sequence[TrainingPair]) -> dict[str, Any]:
    day_position = defaultdict(
        lambda: [[0 for _ in range(10)] for _ in range(4)]
    )
    exact = Counter()
    gaps = [defaultdict(list) for _ in range(4)]
    last_seen = [dict() for _ in range(4)]
    dormant_buckets = [Counter() for _ in range(4)]
    renewal_buckets = [Counter() for _ in range(4)]

    for sequence_index, pair in enumerate(pairs):
        current_digits = [set() for _ in range(4)]
        for number in pair.target_winners_23:
            exact[number] += 1
            values = digits(number)
            for position, value in enumerate(values):
                day_position[pair.target_day_type][position][value] += 1
                current_digits[position].add(value)
        for position in range(4):
            for value in range(10):
                previous = last_seen[position].get(value)
                if value in current_digits[position]:
                    if previous is not None:
                        gap = sequence_index - previous
                        gaps[position][value].append(gap)
                        renewal_buckets[position][bucket_gap(gap)] += 1
                    last_seen[position][value] = sequence_index
                elif previous is not None:
                    dormant_buckets[position][
                        bucket_gap(sequence_index - previous)
                    ] += 1

    return {
        "candidate_generation_features": {
            "day_type_position_digit_counts": {
                day: values for day, values in sorted(day_position.items())
            },
            "positional_gap_summary": [
                {
                    str(digit): summarize_numeric(values)
                    for digit, values in sorted(position.items())
                }
                for position in gaps
            ],
            "long_dormant_gap_buckets": [
                dict(sorted(counter.items())) for counter in dormant_buckets
            ],
        },
        "post_mortem_verification_features": {
            "miss_streak_renewal_buckets": [
                dict(sorted(counter.items())) for counter in renewal_buckets
            ]
        },
        "ranker_features": {
            "exact_frequency_top200": exact.most_common(200),
            "exact_frequency_unique_numbers": len(exact),
        },
    }


def bucket_gap(gap: int) -> str:
    if gap <= 5:
        return "1-5"
    if gap <= 10:
        return "6-10"
    if gap <= 20:
        return "11-20"
    if gap <= 50:
        return "21-50"
    return "51+"


def summarize_numeric(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "minimum": None, "maximum": None}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def ledger_profiles(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    engine_support = Counter()
    rank_support = Counter()
    verified = Counter()
    for row in rows:
        engine_support[row["engine_source"]] += 1
        rank_support[f"{row['engine_source']}::rank_{row['rank']}"] += 1
        if row["verification_status"] == "Verified":
            verified[row["engine_source"]] += int((row["hit_count"] or 0) > 0)
    return {
        "read_only_ledger_rows": len(rows),
        "underlying_engine_support_rows": dict(sorted(engine_support.items())),
        "rank_support_rows": dict(sorted(rank_support.items())),
        "verified_run_hit_row_counts": dict(sorted(verified.items())),
        "warning": (
            "HitCount is run-level and repeated per candidate; use only for post-mortem "
            "support diagnostics, never candidate-level causal training."
        ),
    }


def train_group_d(
    pairs: Sequence[TrainingPair],
    context: dict[str, Any],
    ledger_rows: Sequence[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    profiles = temporal_profiles(pairs)
    if ledger_rows is not None:
        profiles["ranker_features"]["ledger_support"] = ledger_profiles(ledger_rows)
    else:
        profiles["ranker_features"]["ledger_support"] = {
            "status": "NOT_LOADED",
            "reason": "Ledger read was unavailable or intentionally skipped.",
        }
    return [
        make_artifact(
            **context,
            engine_group="D",
            engine_name="TEMPORAL_FREQUENCY_RENEWAL_META_PROFILE__ALL",
            pairs=pairs,
            day_type="ALL",
            model_type="TEMPORAL_FREQUENCY_PROFILE",
            modulus=10,
            formula_space="BASE10",
            feature_stats=profiles,
            score_semantics=(
                "Separated generation, post-mortem, and ranker feature profiles; "
                "no live score calibration."
            ),
        )
    ]


def artifact_filename(artifact: dict[str, Any]) -> str:
    safe_engine = "".join(
        char if char.isalnum() or char in "_-" else "_"
        for char in artifact["engine_name"]
    )
    return (
        f"{safe_engine}__{artifact['training_mode']}__worker_"
        f"{artifact['worker_id']}__cutoff_{artifact['draw_cutoff']}.json"
    )


def write_artifact(artifact: dict[str, Any], output_dir: Path) -> Path:
    directory = (
        output_dir
        / artifact["training_mode"]
        / artifact["engine_group"]
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / artifact_filename(artifact)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return path


def assigned_groups(
    engine_group: str, worker_id: int, worker_count: int
) -> tuple[str, ...]:
    if engine_group != "all":
        return (engine_group,)
    return tuple(
        group
        for index, group in enumerate(ENGINE_GROUPS)
        if index % worker_count == worker_id - 1
    )


def resolve_cutoffs(
    dataset: ChronologicalTrainingDataset,
    *,
    training_mode: str,
    source_start: int | None,
    source_end: int | None,
    shard_sources: bool,
    worker_id: int,
    worker_count: int,
) -> list[int]:
    if training_mode == "phase1_base":
        return [min(PHASE1_MAX_DRAW_NO, dataset.last_draw_no)]
    if training_mode == "retrospective_full_history":
        return [dataset.last_draw_no]
    start = source_start or (PHASE1_MAX_DRAW_NO + 1)
    end = source_end or (dataset.last_draw_no - 1)
    cutoffs = [
        draw.draw_no
        for draw in dataset.draws
        if start <= draw.draw_no <= end
    ]
    if shard_sources:
        cutoffs = [
            cutoff
            for index, cutoff in enumerate(cutoffs)
            if index % worker_count == worker_id - 1
        ]
    return cutoffs


def pairs_for_mode(
    dataset: ChronologicalTrainingDataset,
    training_mode: str,
    cutoff: int,
) -> tuple[TrainingPair, ...]:
    if training_mode == "phase1_base":
        pairs = dataset.phase1_pairs()
    elif training_mode == "rolling_origin_phase2":
        pairs = dataset.phase2_pairs_until(cutoff)
    elif training_mode == "retrospective_full_history":
        pairs = dataset.retrospective_pairs()
    else:
        raise ValueError(f"Unknown training mode: {training_mode}")
    if any(pair.target_draw_no > cutoff for pair in pairs):
        raise RuntimeError("Temporal firewall violation before training")
    return pairs


def train_group(
    group: str,
    pairs: Sequence[TrainingPair],
    context: dict[str, Any],
    ledger_rows: Sequence[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if group == "A":
        return train_group_a(pairs, context)
    if group == "B":
        return train_group_b(pairs, context)
    if group == "C":
        return train_group_c(pairs, context)
    if group == "D":
        return train_group_d(pairs, context, ledger_rows)
    raise ValueError(f"Unknown engine group: {group}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--engine-group", choices=(*ENGINE_GROUPS, "all"), required=True)
    parser.add_argument("--training-mode", choices=TRAINING_MODES, required=True)
    parser.add_argument("--source-start", type=int)
    parser.add_argument("--source-end", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/full_history_training",
    )
    parser.add_argument("--no-sql-write", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def enforce_no_sql_write(args: argparse.Namespace) -> None:
    if not args.no_sql_write and os.getenv("J4D_NO_SQL_WRITE") != "1":
        raise RuntimeError(
            "Offline trainer requires --no-sql-write or J4D_NO_SQL_WRITE=1"
        )
    for sql in (DRAW_HISTORY_SQL, LEDGER_READ_SQL, DRAW_SCHEMA_SQL):
        upper = sql.upper()
        for forbidden in (
            "INSERT ",
            "UPDATE ",
            "DELETE ",
            "MERGE ",
            "CREATE ",
            "ALTER ",
            "DROP ",
            "TRUNCATE ",
        ):
            if forbidden in upper:
                raise RuntimeError(f"Forbidden SQL token in trainer: {forbidden}")


def main() -> int:
    args = parse_args()
    if not 1 <= args.worker_id <= args.worker_count:
        raise ValueError("worker-id must be in 1..worker-count")
    enforce_no_sql_write(args)

    dataset = load_draw_history()
    groups = assigned_groups(
        args.engine_group, args.worker_id, args.worker_count
    )
    cutoffs = resolve_cutoffs(
        dataset,
        training_mode=args.training_mode,
        source_start=args.source_start,
        source_end=args.source_end,
        shard_sources=args.engine_group == "all",
        worker_id=args.worker_id,
        worker_count=args.worker_count,
    )
    if not groups:
        print(f"Worker {args.worker_id}: no engine groups assigned")
        return 0
    if not cutoffs:
        print(f"Worker {args.worker_id}: no source cutoffs assigned")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    worker_rows = []
    ledger_cache: dict[int, list[dict[str, Any]]] = {}
    for cutoff in cutoffs:
        pairs = pairs_for_mode(dataset, args.training_mode, cutoff)
        if not pairs:
            raise RuntimeError(f"No training pairs for cutoff {cutoff}")
        for group in groups:
            ledger_rows = None
            if group == "D":
                if cutoff not in ledger_cache:
                    try:
                        ledger_cache[cutoff] = load_ledger_read_only(cutoff)
                    except Exception as exc:
                        if args.verbose:
                            print(
                                f"Group D ledger read limitation: {type(exc).__name__}: {exc}",
                                file=sys.stderr,
                            )
                        ledger_cache[cutoff] = []
                ledger_rows = ledger_cache[cutoff]

            context = {
                "training_mode": args.training_mode,
                "worker_id": args.worker_id,
                "draw_cutoff": cutoff,
            }
            artifacts = train_group(group, pairs, context, ledger_rows)
            for artifact in artifacts:
                path = write_artifact(artifact, args.output_dir)
                row = {
                    "row_type": "trained_artifact",
                    "path": str(path),
                    "engine_group": group,
                    "engine_name": artifact["engine_name"],
                    "training_mode": args.training_mode,
                    "worker_id": args.worker_id,
                    "draw_cutoff": cutoff,
                    "training_pair_count": artifact["training_pair_count"],
                    "sha256_hash": artifact["sha256_hash"],
                    "temporal_firewall_status": artifact[
                        "temporal_firewall_status"
                    ],
                    "not_for_live_prediction": artifact[
                        "not_for_live_prediction"
                    ],
                }
                worker_rows.append(row)
                if args.verbose:
                    print(
                        f"TRAINED group={group} engine={artifact['engine_name']} "
                        f"pairs={artifact['training_pair_count']} artifact={path}"
                    )

    rows_dir = args.output_dir / "worker_rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    rows_path = rows_dir / (
        f"worker_{args.worker_id}__group_{args.engine_group}__"
        f"mode_{args.training_mode}.jsonl"
    )
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in worker_rows)
    )
    metadata_path = rows_dir / (
        f"worker_{args.worker_id}__group_{args.engine_group}__"
        f"mode_{args.training_mode}__dataset.json"
    )
    metadata_path.write_text(
        json.dumps(
            {
                "dataset": dataset.metadata(),
                "worker_id": args.worker_id,
                "worker_count": args.worker_count,
                "groups": groups,
                "cutoffs": cutoffs,
                "no_sql_write": True,
                "artifacts_written": len(worker_rows),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print("STEP 164 — FULL 40-YEAR ENGINE TRAINING FOUNDATION")
    print(f"WorkerId: {args.worker_id}/{args.worker_count}")
    print(f"EngineGroups: {','.join(groups)}")
    print(f"TrainingMode: {args.training_mode}")
    print(f"Cutoffs: {len(cutoffs)}")
    print(f"ArtifactsWritten: {len(worker_rows)}")
    print("DBWritePerformed: NO")
    if args.training_mode == "retrospective_full_history":
        print(f"Label: {RETROSPECTIVE_LABEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
