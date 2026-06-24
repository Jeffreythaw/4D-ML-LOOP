"""Read-only runtime loader and ensemble for promoted 40-year engine knowledge."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PACK_VERSION = "j4d.live40.v1"
FULL_HISTORY_ENGINE_NAME = "E40_FULL_HISTORY_KNOWLEDGE"
E2_PACK_ENGINE_PREFIX = "E2_SET_PROJECTOR_LEARNED_BIAS__"
E4_PACK_ENGINE_NAME = "E4_MARKOV_TRANSITION_MASS__ALL"
DEFAULT_PACK_PATH = (
    Path(__file__).resolve().parent
    / "live_knowledge"
    / "full_history_engine_pack.json"
)


def _pack_hash(payload: dict[str, Any]) -> str:
    canonical = {
        key: value for key, value in payload.items() if key != "pack_sha256"
    }
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _numbers_from_vectors(vectors: np.ndarray) -> list[str]:
    return [
        "".join(str(int(value)) for value in row)
        for row in np.asarray(vectors, dtype=np.int16)
    ]


def _affine(
    vectors: np.ndarray,
    matrix: Any,
    bias: Any,
    modulus: int,
) -> np.ndarray:
    work = np.asarray(vectors, dtype=np.int64) % modulus
    matrix_array = np.asarray(matrix, dtype=np.int64) % modulus
    bias_array = np.asarray(bias, dtype=np.int64) % modulus
    if work.ndim != 2 or work.shape[1] != 4:
        raise ValueError("vectors must have shape (N, 4)")
    if matrix_array.shape != (4, 4) or bias_array.shape != (4,):
        raise ValueError("invalid affine model shape in live knowledge pack")
    return ((work @ matrix_array.T + bias_array) % modulus).astype(np.int16)


def _polynomial_features(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
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


def _expand_base5(vectors: np.ndarray) -> np.ndarray:
    rows: list[list[int]] = []
    for vector in np.asarray(vectors, dtype=np.int16):
        partial: list[list[int]] = [[]]
        for digit in vector:
            partial = [
                prefix + [int(digit) + offset]
                for prefix in partial
                for offset in (0, 5)
            ]
        rows.extend(partial)
    return np.asarray(rows, dtype=np.int16)


@dataclass(frozen=True)
class FullHistoryCandidate:
    number: str
    score: float
    support_count: int
    source_details: tuple[str, ...]


class FullHistoryKnowledgePack:
    """Validated in-memory representation of the promoted knowledge pack."""

    def __init__(self, payload: dict[str, Any], *, path: Path) -> None:
        self.payload = payload
        self.path = path
        self.models = tuple(payload["models"])
        self.models_by_engine_name = {
            str(model["engine_name"]): model for model in self.models
        }
        self.minimum_source_draw_no = int(
            payload["promotion_policy"]["minimum_source_draw_no"]
        )

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PACK_PATH) -> "FullHistoryKnowledgePack":
        resolved = Path(path).expanduser().resolve()
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if payload.get("pack_version") != PACK_VERSION:
            raise ValueError("unsupported full-history knowledge pack version")
        if payload.get("pack_sha256") != _pack_hash(payload):
            raise ValueError("full-history knowledge pack hash mismatch")
        if payload.get("source_artifact_count") != 26:
            raise ValueError("full-history knowledge pack must contain 26 artifacts")
        if payload.get("source_artifact_group_counts") != {
            "A": 13,
            "B": 9,
            "C": 3,
            "D": 1,
        }:
            raise ValueError("full-history knowledge pack group coverage is incomplete")
        required_models = {
            f"{E2_PACK_ENGINE_PREFIX}{scope}"
            for scope in ("ALL", "Saturday", "Special", "Sunday", "Wednesday")
        } | {E4_PACK_ENGINE_NAME}
        if not required_models.issubset(
            str(model.get("engine_name")) for model in payload["models"]
        ):
            raise ValueError("full-history knowledge pack is missing E2/E4 models")
        policy = payload.get("promotion_policy", {})
        if not policy.get("enabled") or not policy.get("target_blind"):
            raise ValueError("full-history knowledge pack is not live eligible")
        return cls(payload, path=resolved)

    def eligible_for(self, source_draw_no: int) -> bool:
        """Prevent retrospective artifacts from leaking into older replay draws."""
        return int(source_draw_no) >= self.minimum_source_draw_no

    def rank_e2_set_projector_candidates(
        self,
        *,
        source_vectors: np.ndarray,
        day_type: str,
        source_draw_no: int,
        top_n: int = 25,
    ) -> list[FullHistoryCandidate]:
        """Apply the eligible day-scoped learned E2 model, falling back to ALL."""
        if not self.eligible_for(source_draw_no):
            return []
        model = self.models_by_engine_name.get(
            f"{E2_PACK_ENGINE_PREFIX}{day_type}"
        ) or self.models_by_engine_name.get(f"{E2_PACK_ENGINE_PREFIX}ALL")
        if model is None:
            return []
        numbers = _numbers_from_vectors(
            _affine(
                source_vectors,
                model["matrix_m"],
                model["bias_b"],
                int(model["modulus"]),
            )
        )
        output = []
        seen = set()
        for rank, number in enumerate(numbers, start=1):
            if number in seen:
                continue
            seen.add(number)
            output.append(
                FullHistoryCandidate(
                    number=number,
                    score=1.0 / rank,
                    support_count=1,
                    source_details=(str(model["engine_name"]),),
                )
            )
            if len(output) >= top_n:
                break
        return output

    @staticmethod
    def _model_weight(model: dict[str, Any], day_type: str) -> float:
        name = str(model["engine_name"])
        if name.startswith("LEARNED_CROSS_PAIR_AFFINE"):
            return 1.15 if model["day_type"] == day_type else 0.65
        if name.startswith("DELTA_ROTATION"):
            return 0.75
        if name.startswith("POLYNOMIAL"):
            return 0.70
        if name.startswith("WLS_"):
            return 0.55
        if name.startswith("MIRROR_BASE5"):
            return 0.45
        return 0.0

    @staticmethod
    def _profile_maps(
        models: Iterable[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        structural: dict[str, Any] = {}
        temporal: dict[str, Any] = {}
        for model in models:
            if model["engine_name"] == "BASE5_STRUCTURAL_PROFILE__ALL":
                structural = dict(model["feature_stats"])
            elif (
                model["engine_name"]
                == "TEMPORAL_FREQUENCY_RENEWAL_META_PROFILE__ALL"
            ):
                temporal = dict(model["feature_stats"])
        return structural, temporal

    def rank_candidates(
        self,
        *,
        source_vectors: np.ndarray,
        day_type: str,
        source_draw_no: int,
        latest_delta_vectors: np.ndarray | None = None,
        top_n: int = 50,
    ) -> list[FullHistoryCandidate]:
        """Combine all A/B/C/D artifacts into one target-blind ranked vote list."""
        if not self.eligible_for(source_draw_no):
            return []
        source = np.asarray(source_vectors, dtype=np.int16)
        if source.ndim != 2 or source.shape[1] != 4:
            raise ValueError("source_vectors must have shape (N, 4)")

        contributions: dict[str, float] = {}
        support: dict[str, set[str]] = {}
        details: dict[str, list[str]] = {}

        def add(number: str, score: float, model_name: str) -> None:
            value = str(number).zfill(4)
            if len(value) != 4 or not value.isdigit():
                return
            contributions[value] = contributions.get(value, 0.0) + float(score)
            support.setdefault(value, set()).add(model_name)
            bucket = details.setdefault(value, [])
            if model_name not in bucket:
                bucket.append(model_name)

        for model in self.models:
            model_name = str(model["engine_name"])
            weight = self._model_weight(model, day_type)
            if weight <= 0:
                continue
            model_day_type = str(model.get("day_type") or "ALL")
            if model_day_type not in {"ALL", day_type}:
                continue
            predictions: list[str] = []
            if model_name.startswith("DELTA_ROTATION"):
                if latest_delta_vectors is None:
                    continue
                delta = _affine(
                    latest_delta_vectors,
                    model["matrix_m"],
                    model["bias_b"],
                    10,
                )
                if delta.shape[0] == 1 and source.shape[0] > 1:
                    delta = np.repeat(delta, source.shape[0], axis=0)
                count = min(source.shape[0], delta.shape[0])
                predictions = _numbers_from_vectors(
                    (source[:count].astype(np.int32) + delta[:count]) % 10
                )
            elif model_name.startswith("POLYNOMIAL"):
                coefficients = np.asarray(model["coefficients"], dtype=np.float64)
                if coefficients.shape != (13, 4):
                    raise ValueError("invalid polynomial coefficient shape")
                predicted = np.mod(
                    np.rint(_polynomial_features(source) @ coefficients),
                    10,
                ).astype(np.int16)
                predictions = _numbers_from_vectors(predicted)
            elif model_name.startswith("MIRROR_BASE5"):
                projected = _affine(
                    source % 5,
                    model["matrix_m"],
                    model["bias_b"],
                    5,
                )
                predictions = _numbers_from_vectors(_expand_base5(projected))
            elif model.get("matrix_m") is not None:
                predictions = _numbers_from_vectors(
                    _affine(
                        source,
                        model["matrix_m"],
                        model["bias_b"],
                        int(model["modulus"]),
                    )
                )
            seen: set[str] = set()
            for raw_rank, number in enumerate(predictions, start=1):
                if number in seen:
                    continue
                seen.add(number)
                add(
                    number,
                    weight / (1.0 + 0.015 * (raw_rank - 1)),
                    model_name,
                )

        structural, temporal = self._profile_maps(self.models)
        ranker = temporal.get("ranker_features", {})
        exact_frequency = ranker.get("exact_frequency_top200", [])
        if exact_frequency:
            maximum = max(int(item[1]) for item in exact_frequency)
            for rank, item in enumerate(exact_frequency, start=1):
                add(
                    str(item[0]),
                    0.35 * int(item[1]) / max(1, maximum)
                    / (1.0 + 0.01 * (rank - 1)),
                    "TEMPORAL_EXACT_FREQUENCY",
                )

        day_counts = (
            temporal.get("candidate_generation_features", {})
            .get("day_type_position_digit_counts", {})
            .get(day_type)
        )
        first_pairs = dict(structural.get("first_pair_top50", []))
        last_pairs = dict(structural.get("last_pair_top50", []))
        adjacent = [
            dict(items) for items in structural.get("adjacent_pair_top50", [])
        ]
        if contributions:
            max_first = max(first_pairs.values(), default=1)
            max_last = max(last_pairs.values(), default=1)
            max_adjacent = [
                max(values.values(), default=1) for values in adjacent
            ]
            for number in list(contributions):
                profile_score = 0.0
                if day_counts:
                    for position, char in enumerate(number):
                        row = day_counts[position]
                        profile_score += row[int(char)] / max(1, sum(row))
                    profile_score /= 4.0
                profile_score += 0.08 * first_pairs.get(number[:2], 0) / max_first
                profile_score += 0.08 * last_pairs.get(number[2:], 0) / max_last
                for position in range(min(3, len(adjacent))):
                    profile_score += (
                        0.04
                        * adjacent[position].get(number[position : position + 2], 0)
                        / max_adjacent[position]
                    )
                contributions[number] += profile_score
                if profile_score > 0:
                    support.setdefault(number, set()).add(
                        "STRUCTURAL_TEMPORAL_PROFILE"
                    )

        ranked = []
        for number, base_score in contributions.items():
            support_count = len(support.get(number, ()))
            consensus_bonus = 0.10 * math.log1p(max(0, support_count - 1))
            ranked.append(
                FullHistoryCandidate(
                    number=number,
                    score=float(base_score + consensus_bonus),
                    support_count=support_count,
                    source_details=tuple(details.get(number, ())),
                )
            )
        ranked.sort(
            key=lambda item: (
                item.score,
                item.support_count,
                item.number,
            ),
            reverse=True,
        )
        return ranked[: max(1, int(top_n))]


def load_default_pack() -> FullHistoryKnowledgePack | None:
    """Load the default pack unless explicitly disabled; fail closed to legacy."""
    enabled = os.getenv("J4D_FULL_HISTORY_KNOWLEDGE_ENABLED", "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return None
    path = os.getenv("J4D_FULL_HISTORY_KNOWLEDGE_PATH", "").strip()
    try:
        return FullHistoryKnowledgePack.load(path or DEFAULT_PACK_PATH)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
