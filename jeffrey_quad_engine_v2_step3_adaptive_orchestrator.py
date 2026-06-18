# jeffrey_quad_engine_v2_step3_adaptive_orchestrator.py
# ============================================================
# JEFFREY QUAD-ENGINE HYBRID V2
# STEP 3 ONLY — CANDIDATE DIVERSITY GUARD + ADAPTIVE FEEDBACK
#
# Requires approved Step 2 module:
#   jeffrey_quad_engine_v2_step2_matrix_core.py
#
# Scope:
#   1. Candidate pool aggregation from:
#        - E1 Cross-Pair Linear
#        - E2 Set Projector
#        - E3 Polynomial
#        - E4 SQL Markov transition mass
#        - Registered adaptive formulas from FormulaRegistry
#   2. Anti-monopoly Diversity Guard:
#        - EXACTLY Top 5 unique candidates
#        - At least 3 distinct sub-engines represented when available
#   3. Sequential adaptive loop:
#        - Predict Draw N+1 using Draw N only
#        - Verify only through dbo.SP_Verify_Predictions
#        - If HitCount == 0, load Draw N+1 only after verification
#        - Solve/register adaptive affine formula
#   4. Rolling in-memory hit metrics:
#        - 1M, 3M, 12M, 24M
#
# Explicitly NOT included:
#   - Final full backtest CSV writer
#   - Step 4 summary reporting
# ============================================================

from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent

STEP2_MODULE_NAME = "jeffrey_quad_engine_v2_step2_matrix_core"

ENV_SQL_CONN_KEY = "J4D_SQL_CONN_STR"

SIM_START_DRAW_NO = 4051
SIM_END_DRAW_NO = 5494

TOP_K = 5
MIN_DISTINCT_ENGINES = 3

MARKOV_TOP_N_PER_SOURCE = 25
ADAPTIVE_FORMULA_TOP_N = 20

WINDOW_DRAW_COUNTS = {
    "1M": 13,
    "3M": 39,
    "12M": 156,
    "24M": 312,
}


logger = logging.getLogger("jeffrey_quad_engine_v2.step3")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)


# ============================================================
# ENV + STEP 2 IMPORT
# ============================================================

def load_env_file() -> None:
    env_path = PROJECT_ROOT / ".env"

    if not env_path.exists():
        raise FileNotFoundError(f"Missing .env file: {env_path}")

    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        logger.warning("python-dotenv is not installed. Using shell environment only.")
        return

    loaded = load_dotenv(env_path, override=False)

    if not loaded:
        raise RuntimeError(f"Failed to load .env file: {env_path}")


def get_sql_connection_string_from_env() -> str:
    """
    Priority:
      1. DB_DRIVER + DB_SERVER + DB_DATABASE + DB_USERNAME + DB_PASSWORD/DB_PASS/DB_PWD
      2. J4D_SQL_CONN_STR fallback

    This prevents stale J4D_SQL_CONN_STR from overriding DB_*.
    """

    db_driver = os.getenv("DB_DRIVER", "").strip()
    db_server = os.getenv("DB_SERVER", "").strip()
    db_database = os.getenv("DB_DATABASE", "").strip()
    db_username = os.getenv("DB_USERNAME", "").strip()

    db_password = (
        os.getenv("DB_PASSWORD", "").strip()
        or os.getenv("DB_PASS", "").strip()
        or os.getenv("DB_PWD", "").strip()
    )

    if all([db_driver, db_server, db_database, db_username, db_password]):
        return (
            f"DRIVER={{{db_driver}}};"
            f"SERVER={db_server};"
            f"DATABASE={db_database};"
            f"UID={db_username};"
            f"PWD={db_password};"
            "TrustServerCertificate=yes;"
        )

    fallback = os.getenv(ENV_SQL_CONN_KEY, "").strip()

    if fallback:
        return fallback

    raise EnvironmentError(
        "Missing SQL Server configuration. Provide DB_* variables or J4D_SQL_CONN_STR."
    )


def import_step2_core():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    try:
        core = __import__(STEP2_MODULE_NAME)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Missing approved Step 2 module: {PROJECT_ROOT / (STEP2_MODULE_NAME + '.py')}. "
            "Create this file first from the Step 2 approved Matrix Core before running Step 3."
        ) from exc

    required = [
        "SqlServerGateway",
        "Phase2SequentialInputLayer",
        "MatrixComputationCore",
        "MatrixFormula",
        "affine_mod10_transform",
        "vectors_from_4d_strings",
        "strings_from_vectors",
        "ENGINE_1_NAME",
        "ENGINE_2_NAME",
        "ENGINE_3_NAME",
        "ENGINE_4_NAME",
        "VALID_DAY_TYPES",
    ]

    missing = [name for name in required if not hasattr(core, name)]

    if missing:
        raise AttributeError(f"Step 2 module missing required symbols: {missing}")

    return core


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass(frozen=True)
class CandidateVote:
    number: str
    engine_name: str
    internal_score: float
    raw_rank: int
    source_detail: str


@dataclass
class CandidateAggregate:
    number: str
    total_score: float = 0.0
    engine_scores: Dict[str, float] = field(default_factory=dict)
    votes: List[CandidateVote] = field(default_factory=list)

    @property
    def best_engine(self) -> str:
        if not self.engine_scores:
            raise RuntimeError(f"Candidate {self.number} has no engine scores")
        return max(self.engine_scores.items(), key=lambda kv: kv[1])[0]

    @property
    def best_engine_score(self) -> float:
        if not self.engine_scores:
            return 0.0
        return max(self.engine_scores.values())

    @property
    def engine_count(self) -> int:
        return len(self.engine_scores)


@dataclass(frozen=True)
class LockedPrediction:
    target_draw_no: int
    source_draw_no: int
    top5: Tuple[str, ...]
    engine_sources: Tuple[str, ...]
    candidate_scores: Tuple[Tuple[str, float, str], ...]
    engine_candidate_scores: Tuple[Tuple[str, int, str, float], ...] = ()


@dataclass(frozen=True)
class DrawStepResult:
    source_draw_no: int
    target_draw_no: int
    source_day_type: str
    top5: Tuple[str, ...]
    engine_sources: Tuple[str, ...]
    hit_count: int
    adaptive_triggered: bool
    adaptive_formula_id: Optional[int]
    rolling_metrics: Dict[str, Dict[str, float]]


@dataclass
class RollingWindowMetrics:
    windows: Dict[str, int]
    buffers: Dict[str, Deque[int]] = field(init=False)

    def __post_init__(self) -> None:
        self.buffers = {
            name: deque(maxlen=size)
            for name, size in self.windows.items()
        }

    def update(self, binary_hit: int) -> Dict[str, Dict[str, float]]:
        if binary_hit not in (0, 1):
            raise ValueError("binary_hit must be 0 or 1")

        snapshot: Dict[str, Dict[str, float]] = {}

        for name, buf in self.buffers.items():
            buf.append(binary_hit)

            count = len(buf)
            hits = sum(buf)
            hit_rate = float(hits / count) if count else 0.0

            snapshot[name] = {
                "draws": float(count),
                "hits": float(hits),
                "hit_rate": hit_rate,
            }

        return snapshot


# ============================================================
# FORMULA REGISTRY READER FOR ADAPTIVE FORMULAS
# ============================================================

class FormulaRegistryReader:
    """
    Reads active FormulaRegistry rows and reconstructs Step 2 MatrixFormula.
    """

    def __init__(self, gateway: Any, core: Any) -> None:
        self.gateway = gateway
        self.core = core

    def load_active_formulas(
        self,
        *,
        day_type: str,
        engine_name: Optional[str] = None,
        top_n: int = ADAPTIVE_FORMULA_TOP_N,
    ) -> List[Any]:
        if day_type not in self.core.VALID_DAY_TYPES:
            raise ValueError(f"Invalid DayType: {day_type}")

        clauses = ["DayType = ?", "IsActive = 1"]
        params: List[Any] = [day_type]

        if engine_name is not None:
            clauses.append("EngineName = ?")
            params.append(engine_name)

        where_sql = " AND ".join(clauses)

        sql = f"""
            SELECT TOP ({int(top_n)})
                EngineName,
                FormulaVersion,
                DayType,
                MatrixPayload,
                HistoricalConfidence,
                HitRateTop5,
                SampleSize,
                CreatedAt
            FROM dbo.FormulaRegistry
            WHERE {where_sql}
            ORDER BY
                HistoricalConfidence DESC,
                HitRateTop5 DESC,
                SampleSize DESC,
                CreatedAt DESC;
        """

        rows = self.gateway.conn.cursor().execute(sql, params).fetchall()

        formulas: List[Any] = []

        for row in rows:
            payload = json.loads(str(row.MatrixPayload))
            matrix_m = np.array(payload["matrix_m"], dtype=np.int16)
            bias_b = np.array(payload["bias_b"], dtype=np.int16)
            metadata = payload.get("metadata", {})

            formula = self.core.MatrixFormula(
                engine_name=str(row.EngineName),
                formula_version=str(row.FormulaVersion),
                day_type=str(row.DayType),
                matrix_m=matrix_m,
                bias_b=bias_b,
                metadata=metadata,
            )

            formula.validate()
            formulas.append(formula)

        return formulas


# ============================================================
# CANDIDATE POOL BUILDER
# ============================================================

class CandidatePoolBuilder:
    """
    Aggregates candidates from:
      - Static E1/E2/E3 matrix outputs
      - E4 Markov transition mass
      - Active adaptive E4 formulas from FormulaRegistry
    """

    def __init__(
        self,
        *,
        core: Any,
        gateway: Any,
        phase2_layer: Any,
        matrix_core: Any,
        formula_reader: FormulaRegistryReader,
    ) -> None:
        self.core = core
        self.gateway = gateway
        self.phase2_layer = phase2_layer
        self.matrix_core = matrix_core
        self.formula_reader = formula_reader

    @staticmethod
    def _score_by_rank(rank: int, total: int, *, base_weight: float) -> float:
        if total <= 0:
            return 0.0
        rank = max(1, rank)
        return base_weight * ((total - rank + 1) / total)

    def _votes_from_static_engine_outputs(
        self,
        engine_outputs: Dict[str, np.ndarray],
    ) -> List[CandidateVote]:
        votes: List[CandidateVote] = []

        engine_base_weight = {
            self.core.ENGINE_1_NAME: 1.00,
            self.core.ENGINE_2_NAME: 1.00,
            self.core.ENGINE_3_NAME: 1.00,
        }

        for engine_name, vectors in engine_outputs.items():
            numbers = self.core.strings_from_vectors(vectors)
            total = len(numbers)

            seen_local = set()

            for idx, number in enumerate(numbers, start=1):
                if number in seen_local:
                    continue

                seen_local.add(number)

                score = self._score_by_rank(
                    idx,
                    total,
                    base_weight=engine_base_weight.get(engine_name, 1.0),
                )

                votes.append(
                    CandidateVote(
                        number=number,
                        engine_name=engine_name,
                        internal_score=score,
                        raw_rank=idx,
                        source_detail=f"{engine_name}:matrix_rank={idx}",
                    )
                )

        return votes

    def _votes_from_markov(
        self,
        *,
        source_states: Sequence[str],
        day_type: str,
        top_n_per_source: int,
    ) -> List[CandidateVote]:
        votes: List[CandidateVote] = []

        markov_mass = self.phase2_layer.load_markov_input_mass(
            source_states=source_states,
            day_type=day_type,
            top_n_per_source=top_n_per_source,
        )

        for source_state, transitions in markov_mass.items():
            if not transitions:
                continue

            max_count = max(t.transition_count for t in transitions)

            for idx, t in enumerate(transitions, start=1):
                if max_count <= 0:
                    normalized_count_score = 0.0
                else:
                    normalized_count_score = float(t.transition_count / max_count)

                rank_score = self._score_by_rank(
                    idx,
                    len(transitions),
                    base_weight=1.15,
                )

                score = 0.70 * normalized_count_score + 0.30 * rank_score

                votes.append(
                    CandidateVote(
                        number=t.target_state,
                        engine_name="E4_MARKOV_TRANSITION_MASS",
                        internal_score=float(score),
                        raw_rank=idx,
                        source_detail=(
                            f"E4_MARKOV source={source_state} "
                            f"count={t.transition_count} rank={idx}"
                        ),
                    )
                )

        return votes

    def _votes_from_adaptive_formulas(
        self,
        *,
        src_vectors: np.ndarray,
        day_type: str,
    ) -> List[CandidateVote]:
        votes: List[CandidateVote] = []

        adaptive_formulas = self.formula_reader.load_active_formulas(
            day_type=day_type,
            engine_name=self.core.ENGINE_4_NAME,
            top_n=ADAPTIVE_FORMULA_TOP_N,
        )

        for formula_idx, formula in enumerate(adaptive_formulas, start=1):
            predicted_vectors = self.matrix_core.apply_formula(src_vectors, formula)
            numbers = self.core.strings_from_vectors(predicted_vectors)

            total = len(numbers)
            formula_weight = max(0.25, 1.00 / formula_idx)

            seen_local = set()

            for rank, number in enumerate(numbers, start=1):
                if number in seen_local:
                    continue

                seen_local.add(number)

                score = self._score_by_rank(
                    rank,
                    total,
                    base_weight=0.95 * formula_weight,
                )

                votes.append(
                    CandidateVote(
                        number=number,
                        engine_name=self.core.ENGINE_4_NAME,
                        internal_score=score,
                        raw_rank=rank,
                        source_detail=(
                            f"{self.core.ENGINE_4_NAME}:"
                            f"{formula.formula_version}:formula_rank={formula_idx}:output_rank={rank}"
                        ),
                    )
                )

        return votes


    def _build_wls_training_pairs(
        self,
        *,
        source_draw_no: int,
        day_type: str,
        max_pairs: int = 64,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build temporally safe WLS training pairs using only rows up to source_draw_no.

        Pair rule:
          source row K -> target row K+1
          target row K+1 must be <= source_draw_no
          source row day_type must match the active prediction day_type

        This prevents future leakage for current prediction N -> N+1.
        """
        records = self.gateway.load_draw_history(max_draw_no=source_draw_no)

        by_draw_no = {record.draw_no: record for record in records}
        src_numbers: List[str] = []
        tgt_numbers: List[str] = []

        for record in records:
            if record.day_type != day_type:
                continue

            target_record = by_draw_no.get(record.draw_no + 1)

            if target_record is None:
                continue

            if target_record.draw_no > source_draw_no:
                continue

            for src_number in record.winning_numbers:
                for tgt_number in target_record.winning_numbers:
                    src_numbers.append(src_number)
                    tgt_numbers.append(tgt_number)

        if not src_numbers or not tgt_numbers:
            raise RuntimeError(
                f"No WLS training pairs available for day_type={day_type} source_draw_no={source_draw_no}"
            )

        if len(src_numbers) > max_pairs:
            src_numbers = src_numbers[-max_pairs:]
            tgt_numbers = tgt_numbers[-max_pairs:]

        return (
            self.core.vectors_from_4d_strings(src_numbers, dtype=np.int16),
            self.core.vectors_from_4d_strings(tgt_numbers, dtype=np.int16),
        )

    def _votes_from_wls_decay(
        self,
        *,
        src_vectors: np.ndarray,
        day_type: str,
        source_draw_no: int,
    ) -> List[CandidateVote]:
        votes: List[CandidateVote] = []

        training_src, training_tgt = self._build_wls_training_pairs(
            source_draw_no=source_draw_no,
            day_type=day_type,
        )

        wls_engine = self.core.Engine1WeightedLeastSquaresDecay(decay=0.98)
        formula = wls_engine.fit_formula(
            src_vectors=training_src,
            tgt_vectors=training_tgt,
            day_type=day_type,
            source_draw_no=source_draw_no,
        )

        primary_vectors = self.core.affine_mod10_transform(
            src_vectors,
            formula.matrix_m,
            formula.bias_b,
            dtype=np.int16,
        )

        # If the current source draw collapses to fewer than Top-K unique WLS
        # candidates, enrich with recent temporally eligible historical source
        # vectors transformed by the same fitted formula. This does not read the
        # hidden target draw N+1.
        recent_training_src = training_src[-64:]
        enrichment_vectors = self.core.affine_mod10_transform(
            recent_training_src,
            formula.matrix_m,
            formula.bias_b,
            dtype=np.int16,
        )

        numbers = (
            self.core.strings_from_vectors(primary_vectors)
            + self.core.strings_from_vectors(enrichment_vectors)
        )

        # Deterministic temporal-safe fallback: if the fitted WLS transform
        # collapses to fewer than Top-K unique numbers, append recent historical
        # target-side training states. Every fallback target state is <=
        # source_draw_no by construction in _build_wls_training_pairs().
        numbers += list(reversed(self.core.strings_from_vectors(training_tgt[-64:])))

        total = len(numbers)
        seen_local = set()
        rank_no = 0

        for raw_rank, number in enumerate(numbers, start=1):
            if number in seen_local:
                continue

            seen_local.add(number)
            rank_no += 1

            score = self._score_by_rank(
                rank_no,
                max(5, total),
                base_weight=1.02,
            )

            votes.append(
                CandidateVote(
                    number=number,
                    engine_name=self.core.ENGINE_1_WLS_NAME,
                    internal_score=score,
                    raw_rank=rank_no,
                    source_detail=(
                        f"{self.core.ENGINE_1_WLS_NAME}:decay=0.98:"
                        f"source_draw_no={source_draw_no}:raw_rank={raw_rank}:rank={rank_no}"
                    ),
                )
            )

            if rank_no >= TOP_K:
                break

        return votes


    def _votes_from_mirror_base5(
        self,
        *,
        src_vectors: np.ndarray,
        day_type: str,
        source_draw_no: int,
    ) -> List[CandidateVote]:
        votes: List[CandidateVote] = []

        training_src, training_tgt = self._build_wls_training_pairs(
            source_draw_no=source_draw_no,
            day_type=day_type,
        )

        mirror_engine = self.core.Engine1MirrorBase5Lsts()
        expanded_vectors = mirror_engine.expand_predictions(
            src_vectors=src_vectors,
            training_src_vectors=training_src,
            training_tgt_vectors=training_tgt,
            day_type=day_type,
            source_draw_no=source_draw_no,
        )

        numbers = self.core.strings_from_vectors(expanded_vectors)

        # Rank expansion candidates by closeness to baseline E1 output first.
        # This uses only source draw N and the static baseline formula, no target.
        baseline_vectors = self.matrix_core.run_engine1(src_vectors, day_type)
        baseline_numbers = self.core.strings_from_vectors(baseline_vectors)
        baseline_set = set(baseline_numbers)

        def digit_distance(a: str, b: str) -> int:
            return sum(abs(int(x) - int(y)) for x, y in zip(a, b))

        ranked_numbers = []
        seen_local = set()

        for raw_rank, number in enumerate(numbers, start=1):
            if number in seen_local:
                continue

            seen_local.add(number)

            if baseline_numbers:
                nearest_distance = min(digit_distance(number, base) for base in baseline_numbers)
            else:
                nearest_distance = 99

            baseline_exact_bonus = 0 if number in baseline_set else 1

            ranked_numbers.append(
                (
                    baseline_exact_bonus,
                    nearest_distance,
                    raw_rank,
                    number,
                )
            )

        ranked_numbers.sort(key=lambda item: (item[0], item[1], item[2], item[3]))

        total = max(TOP_K, len(ranked_numbers))

        for rank_no, item in enumerate(ranked_numbers[:TOP_K], start=1):
            _, nearest_distance, raw_rank, number = item

            distance_penalty = min(0.35, nearest_distance / 40.0)
            score = max(
                0.01,
                self._score_by_rank(rank_no, total, base_weight=1.04) - distance_penalty,
            )

            votes.append(
                CandidateVote(
                    number=number,
                    engine_name=self.core.ENGINE_1_MIRROR_BASE5_NAME,
                    internal_score=float(score),
                    raw_rank=rank_no,
                    source_detail=(
                        f"{self.core.ENGINE_1_MIRROR_BASE5_NAME}:"
                        f"source_draw_no={source_draw_no}:raw_rank={raw_rank}:"
                        f"rank={rank_no}:distance={nearest_distance}"
                    ),
                )
            )

        return votes


    def _build_delta_rotation_training_sets(
        self,
        *,
        source_draw_no: int,
        day_type: str,
        max_deltas: int = 64,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build temporally safe delta rotation training sets.

        Returns:
          training_delta_src_vectors: delta_t
          training_delta_tgt_vectors: delta_t+1
          latest_delta_vectors: latest known delta <= source_draw_no

        Absolute transition delta rule:
          delta = (current - previous + 10) % 10

        Only uses records with target draw <= source_draw_no.
        """
        records = self.gateway.load_draw_history(max_draw_no=source_draw_no)

        by_draw_no = {record.draw_no: record for record in records}
        delta_vectors: List[np.ndarray] = []

        for record in records:
            if record.day_type != day_type:
                continue

            target_record = by_draw_no.get(record.draw_no + 1)

            if target_record is None:
                continue

            if target_record.draw_no > source_draw_no:
                continue

            source_vectors = self.core.vectors_from_4d_strings(
                record.winning_numbers,
                dtype=np.int16,
            )
            target_vectors = self.core.vectors_from_4d_strings(
                target_record.winning_numbers,
                dtype=np.int16,
            )

            pair_count = min(source_vectors.shape[0], target_vectors.shape[0])
            if pair_count <= 0:
                continue

            delta = self.core.delta_vectors_mod10(
                source_vectors[:pair_count],
                target_vectors[:pair_count],
                dtype=np.int16,
            )
            delta_vectors.extend([row.reshape(1, -1) for row in delta])

        if len(delta_vectors) < 2:
            raise RuntimeError(
                f"Need at least 2 delta rows for delta rotation training; "
                f"got {len(delta_vectors)} for day_type={day_type} source_draw_no={source_draw_no}"
            )

        delta_matrix = np.vstack(delta_vectors)

        if delta_matrix.shape[0] > max_deltas:
            delta_matrix = delta_matrix[-max_deltas:]

        if delta_matrix.shape[0] < 2:
            raise RuntimeError("Delta rotation training matrix is too small after windowing")

        training_delta_src = delta_matrix[:-1]
        training_delta_tgt = delta_matrix[1:]
        latest_delta = delta_matrix[-1:]

        return training_delta_src, training_delta_tgt, latest_delta

    def _votes_from_delta_rotation(
        self,
        *,
        src_vectors: np.ndarray,
        day_type: str,
        source_draw_no: int,
    ) -> List[CandidateVote]:
        votes: List[CandidateVote] = []

        training_delta_src, training_delta_tgt, latest_delta = self._build_delta_rotation_training_sets(
            source_draw_no=source_draw_no,
            day_type=day_type,
        )

        delta_engine = self.core.Engine1DeltaRotationLsts()
        predicted_vectors = delta_engine.predict_absolute_vectors(
            base_vectors=src_vectors,
            latest_delta_vectors=latest_delta,
            training_delta_src_vectors=training_delta_src,
            training_delta_tgt_vectors=training_delta_tgt,
            day_type=day_type,
            source_draw_no=source_draw_no,
        )

        # Enrich with recent temporally safe delta momentum projections if the
        # primary source draw collapses to fewer than Top-K unique candidates.
        recent_delta_inputs = training_delta_src[-64:]
        recent_base_vectors = np.repeat(src_vectors[:1], recent_delta_inputs.shape[0], axis=0)
        recent_projected = delta_engine.predict_absolute_vectors(
            base_vectors=recent_base_vectors,
            latest_delta_vectors=recent_delta_inputs,
            training_delta_src_vectors=training_delta_src,
            training_delta_tgt_vectors=training_delta_tgt,
            day_type=day_type,
            source_draw_no=source_draw_no,
        )

        numbers = (
            self.core.strings_from_vectors(predicted_vectors)
            + self.core.strings_from_vectors(recent_projected)
        )

        total = max(TOP_K, len(numbers))
        seen_local = set()
        rank_no = 0

        for raw_rank, number in enumerate(numbers, start=1):
            if number in seen_local:
                continue

            seen_local.add(number)
            rank_no += 1

            score = self._score_by_rank(
                rank_no,
                total,
                base_weight=1.03,
            )

            votes.append(
                CandidateVote(
                    number=number,
                    engine_name=self.core.ENGINE_1_DELTA_ROTATION_NAME,
                    internal_score=float(score),
                    raw_rank=rank_no,
                    source_detail=(
                        f"{self.core.ENGINE_1_DELTA_ROTATION_NAME}:"
                        f"source_draw_no={source_draw_no}:raw_rank={raw_rank}:rank={rank_no}"
                    ),
                )
            )

            if rank_no >= TOP_K:
                break

        return votes

    @staticmethod
    def aggregate_votes(votes: Sequence[CandidateVote]) -> Dict[str, CandidateAggregate]:
        aggregates: Dict[str, CandidateAggregate] = {}

        for vote in votes:
            agg = aggregates.get(vote.number)

            if agg is None:
                agg = CandidateAggregate(number=vote.number)
                aggregates[vote.number] = agg

            agg.votes.append(vote)

            existing_engine_score = agg.engine_scores.get(vote.engine_name, 0.0)
            agg.engine_scores[vote.engine_name] = max(existing_engine_score, vote.internal_score)

        for agg in aggregates.values():
            # Consensus reward: multiple engines agreeing should lift a candidate,
            # but not enough to allow one engine to monopolize the final Top 5.
            base = sum(agg.engine_scores.values())
            consensus_bonus = 0.12 * max(0, agg.engine_count - 1)
            agg.total_score = float(base + consensus_bonus)

        return aggregates

    def build_candidate_pool(
        self,
        *,
        src_vectors: np.ndarray,
        source_states: Sequence[str],
        day_type: str,
        source_draw_no: Optional[int] = None,
    ) -> Dict[str, CandidateAggregate]:
        static_outputs = self.matrix_core.run_all_static_engines(
            src_vectors,
            day_type,
        )

        votes: List[CandidateVote] = []
        votes.extend(self._votes_from_static_engine_outputs(static_outputs))

        if source_draw_no is not None:
            votes.extend(
                self._votes_from_wls_decay(
                    src_vectors=src_vectors,
                    day_type=day_type,
                    source_draw_no=source_draw_no,
                )
            )
            votes.extend(
                self._votes_from_mirror_base5(
                    src_vectors=src_vectors,
                    day_type=day_type,
                    source_draw_no=source_draw_no,
                )
            )
            votes.extend(
                self._votes_from_delta_rotation(
                    src_vectors=src_vectors,
                    day_type=day_type,
                    source_draw_no=source_draw_no,
                )
            )

        votes.extend(
            self._votes_from_markov(
                source_states=source_states,
                day_type=day_type,
                top_n_per_source=MARKOV_TOP_N_PER_SOURCE,
            )
        )
        votes.extend(
            self._votes_from_adaptive_formulas(
                src_vectors=src_vectors,
                day_type=day_type,
            )
        )

        if not votes:
            raise RuntimeError("Candidate pool is empty")

        return self.aggregate_votes(votes)


# ============================================================
# DIVERSITY GUARD RANKER
# ============================================================

class DiversityGuardRanker:
    """
    Anti-monopoly candidate selector.

    Final output:
      - EXACTLY Top 5 unique numbers
      - At least 3 distinct sub-engines represented when enough engines exist
      - Dominant engine lower-ranked overlaps are suppressed
    """

    def __init__(
        self,
        *,
        top_k: int = TOP_K,
        min_distinct_engines: int = MIN_DISTINCT_ENGINES,
        max_slots_per_engine_before_minimum_met: int = 2,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if min_distinct_engines <= 0:
            raise ValueError("min_distinct_engines must be positive")

        self.top_k = top_k
        self.min_distinct_engines = min_distinct_engines
        self.max_slots_per_engine_before_minimum_met = max_slots_per_engine_before_minimum_met

    @staticmethod
    def _engine_candidate_lists(
        aggregates: Dict[str, CandidateAggregate],
    ) -> Dict[str, List[CandidateAggregate]]:
        by_engine: Dict[str, List[CandidateAggregate]] = defaultdict(list)

        for agg in aggregates.values():
            for engine in agg.engine_scores:
                by_engine[engine].append(agg)

        for engine, items in by_engine.items():
            items.sort(
                key=lambda a: (
                    a.engine_scores.get(engine, 0.0),
                    a.total_score,
                    a.engine_count,
                    a.number,
                ),
                reverse=True,
            )

        return dict(by_engine)


    @staticmethod
    def build_engine_rank_snapshots(
        aggregates: Dict[str, CandidateAggregate],
        *,
        engine_names: Sequence[str],
        top_k: int = TOP_K,
    ) -> Tuple[Tuple[str, int, str, float], ...]:
        """
        Build engine-specific Top-K snapshots for ledger/audit use.

        Each row is:
          (engine_name, rank, number, engine_score)

        This does not change the final diversified locked Top5. It only records
        what each requested engine would have contributed independently.
        """
        rows: List[Tuple[str, int, str, float]] = []

        for engine_name in engine_names:
            ranked = [
                agg for agg in aggregates.values()
                if engine_name in agg.engine_scores
            ]

            ranked.sort(
                key=lambda a: (
                    a.engine_scores.get(engine_name, 0.0),
                    a.total_score,
                    a.engine_count,
                    a.number,
                ),
                reverse=True,
            )

            seen_numbers = set()
            rank_no = 0

            for agg in ranked:
                if agg.number in seen_numbers:
                    continue

                seen_numbers.add(agg.number)
                rank_no += 1

                rows.append(
                    (
                        engine_name,
                        int(rank_no),
                        agg.number,
                        float(agg.engine_scores.get(engine_name, 0.0)),
                    )
                )

                if rank_no >= top_k:
                    break

        return tuple(rows)

    def select_top5(
        self,
        aggregates: Dict[str, CandidateAggregate],
        *,
        target_draw_no: int,
        source_draw_no: int,
    ) -> LockedPrediction:
        if len(aggregates) < self.top_k:
            raise RuntimeError(
                f"Need at least {self.top_k} unique candidates, got {len(aggregates)}"
            )

        available_engines = sorted(
            {
                engine
                for agg in aggregates.values()
                for engine in agg.engine_scores
            }
        )

        required_engine_count = min(self.min_distinct_engines, len(available_engines))

        global_ranked = sorted(
            aggregates.values(),
            key=lambda a: (
                a.total_score,
                a.engine_count,
                a.best_engine_score,
                a.number,
            ),
            reverse=True,
        )

        by_engine = self._engine_candidate_lists(aggregates)

        selected: List[CandidateAggregate] = []
        selected_numbers = set()
        selected_engine_sources: List[str] = []
        engine_slot_counts: Counter[str] = Counter()

        # Pass 1: force representation from distinct engines.
        engine_order = sorted(
            available_engines,
            key=lambda e: (
                by_engine[e][0].engine_scores.get(e, 0.0) if by_engine[e] else 0.0,
                len(by_engine[e]),
                e,
            ),
            reverse=True,
        )

        for engine in engine_order:
            if len(set(selected_engine_sources)) >= required_engine_count:
                break

            for candidate in by_engine.get(engine, []):
                if candidate.number in selected_numbers:
                    continue

                selected.append(candidate)
                selected_numbers.add(candidate.number)
                selected_engine_sources.append(engine)
                engine_slot_counts[engine] += 1
                break

        # Pass 2: fill by global rank, suppressing early monopoly.
        for candidate in global_ranked:
            if len(selected) >= self.top_k:
                break

            if candidate.number in selected_numbers:
                continue

            engine = candidate.best_engine
            distinct_so_far = len(set(selected_engine_sources))

            if distinct_so_far < required_engine_count:
                if engine_slot_counts[engine] >= self.max_slots_per_engine_before_minimum_met:
                    continue

            selected.append(candidate)
            selected_numbers.add(candidate.number)
            selected_engine_sources.append(engine)
            engine_slot_counts[engine] += 1

        # Pass 3: hard fallback if still short.
        for candidate in global_ranked:
            if len(selected) >= self.top_k:
                break

            if candidate.number in selected_numbers:
                continue

            engine = candidate.best_engine
            selected.append(candidate)
            selected_numbers.add(candidate.number)
            selected_engine_sources.append(engine)
            engine_slot_counts[engine] += 1

        if len(selected) != self.top_k:
            raise RuntimeError(
                f"DiversityGuard failed to produce exactly {self.top_k}; got {len(selected)}"
            )

        represented = set(selected_engine_sources)

        if len(represented) < required_engine_count:
            raise RuntimeError(
                f"DiversityGuard failed distinct-engine constraint. "
                f"Required={required_engine_count}, got={len(represented)}, engines={represented}"
            )

        top5 = tuple(c.number for c in selected)
        sources = tuple(selected_engine_sources)
        score_snapshot = tuple(
            (c.number, float(c.total_score), selected_engine_sources[idx])
            for idx, c in enumerate(selected)
        )

        engine_candidate_scores = self.build_engine_rank_snapshots(
            aggregates,
            engine_names=(
                "E1_CROSS_PAIR_LINEAR",
                "E1_WLS_DECAY_0.98",
                "E1_MIRROR_BASE5_LSTS",
                "E1_DELTA_ROTATION_LSTS",
            ),
            top_k=self.top_k,
        )

        return LockedPrediction(
            target_draw_no=target_draw_no,
            source_draw_no=source_draw_no,
            top5=top5,
            engine_sources=sources,
            candidate_scores=score_snapshot,
            engine_candidate_scores=engine_candidate_scores,
        )


# ============================================================
# ADAPTIVE ORCHESTRATION LOOP
# ============================================================

class Step3AdaptiveOrchestrator:
    def __init__(
        self,
        *,
        core: Any,
        gateway: Any,
        start_draw_no: int = SIM_START_DRAW_NO,
        end_draw_no: int = SIM_END_DRAW_NO,
    ) -> None:
        if start_draw_no >= end_draw_no:
            raise ValueError("start_draw_no must be less than end_draw_no")

        self.core = core
        self.gateway = gateway
        self.start_draw_no = start_draw_no
        self.end_draw_no = end_draw_no

        self.phase2_layer = core.Phase2SequentialInputLayer(gateway)
        self.matrix_core = core.MatrixComputationCore(gateway)
        self.formula_reader = FormulaRegistryReader(gateway, core)
        self.pool_builder = CandidatePoolBuilder(
            core=core,
            gateway=gateway,
            phase2_layer=self.phase2_layer,
            matrix_core=self.matrix_core,
            formula_reader=self.formula_reader,
        )
        self.ranker = DiversityGuardRanker()
        self.metrics = RollingWindowMetrics(WINDOW_DRAW_COUNTS)

    def _load_target_for_solver_after_zero_hit(
        self,
        target_draw_no: int,
    ) -> Tuple[Any, np.ndarray]:
        record = self.gateway.load_phase2_draw(target_draw_no)

        if record is None:
            raise LookupError(f"Target DrawNo {target_draw_no} not found for adaptive solver")

        vectors = self.core.vectors_from_4d_strings(record.winning_numbers, dtype=np.int16)

        return record, vectors

    def predict_one_step_locked(self, source_draw_no: int) -> LockedPrediction:
        """
        Build and lock Top 5 candidates for source_draw_no -> target_draw_no.

        This method is intended for read-only API prediction serving. It does not
        call SP_Verify_Predictions, does not load hidden target winners, does not
        update rolling metrics, and does not register adaptive formulas.
        """
        target_draw_no = source_draw_no + 1

        source_record, source_vectors = self.phase2_layer.load_source_draw_vectors(
            source_draw_no
        )

        # Leakage-safe target existence check:
        # Only checks DrawNo existence. It must not load target WinningNumbers.
        # Prediction serving may target the next future draw, which will not yet
        # exist in dbo.DrawHistory. This read-only path therefore must not require
        # the target draw row to exist. Historical verification remains protected
        # by /api/verify and dbo.SP_Verify_Predictions.
        source_states = list(source_record.winning_numbers)

        candidate_pool = self.pool_builder.build_candidate_pool(
            src_vectors=source_vectors,
            source_states=source_states,
            day_type=source_record.day_type,
            source_draw_no=source_draw_no,
        )

        return self.ranker.select_top5(
            candidate_pool,
            target_draw_no=target_draw_no,
            source_draw_no=source_draw_no,
        )

    def run_one_step(self, source_draw_no: int) -> Optional[DrawStepResult]:
        target_draw_no = source_draw_no + 1

        source_record, source_vectors = self.phase2_layer.load_source_draw_vectors(
            source_draw_no
        )

        # Leakage-safe target existence check:
        # Do NOT call load_phase2_draw(target_draw_no) here because that would
        # load Draw N+1 WinningNumbers into Python memory before the Top 5 is
        # locked and before SP_Verify_Predictions returns HitCount.
        target_exists_row = self.gateway.conn.cursor().execute(
            "SELECT 1 AS ExistsFlag FROM dbo.DrawHistory WHERE DrawNo = ?;",
            (target_draw_no,),
        ).fetchone()

        if target_exists_row is None:
            logger.warning(
                "Skipping source_draw_no=%d because target_draw_no=%d does not exist",
                source_draw_no,
                target_draw_no,
            )
            return None

        source_states = list(source_record.winning_numbers)

        candidate_pool = self.pool_builder.build_candidate_pool(
            src_vectors=source_vectors,
            source_states=source_states,
            day_type=source_record.day_type,
            source_draw_no=source_draw_no,
        )

        locked = self.ranker.select_top5(
            candidate_pool,
            target_draw_no=target_draw_no,
            source_draw_no=source_draw_no,
        )

        hit_count = self.gateway.verify_predictions(
            target_draw_no=target_draw_no,
            predictions=locked.top5,
        )

        binary_hit = 1 if hit_count > 0 else 0
        rolling_snapshot = self.metrics.update(binary_hit)

        adaptive_triggered = False
        adaptive_formula_id: Optional[int] = None

        if hit_count == 0:
            adaptive_triggered = True

            # Firewall rule:
            # target winners are loaded only after SQL SP has returned HitCount == 0.
            target_record, target_vectors = self._load_target_for_solver_after_zero_hit(
                target_draw_no
            )

            adaptive_formula_id = self.matrix_core.solve_and_register_adaptive_formula(
                source_draw_no=source_draw_no,
                target_draw_no=target_draw_no,
                day_type=target_record.day_type,
                src_vectors=source_vectors,
                tgt_vectors=target_vectors,
                training_start_draw_no=source_draw_no,
                training_end_draw_no=target_draw_no,
            )

        result = DrawStepResult(
            source_draw_no=source_draw_no,
            target_draw_no=target_draw_no,
            source_day_type=source_record.day_type,
            top5=locked.top5,
            engine_sources=locked.engine_sources,
            hit_count=hit_count,
            adaptive_triggered=adaptive_triggered,
            adaptive_formula_id=adaptive_formula_id,
            rolling_metrics=rolling_snapshot,
        )

        logger.info(
            "DrawStep source=%d target=%d top5=%s engines=%s hit=%d adaptive=%s formula_id=%s "
            "roll_1M=%.4f roll_3M=%.4f roll_12M=%.4f roll_24M=%.4f",
            source_draw_no,
            target_draw_no,
            list(locked.top5),
            list(locked.engine_sources),
            hit_count,
            adaptive_triggered,
            adaptive_formula_id,
            rolling_snapshot["1M"]["hit_rate"],
            rolling_snapshot["3M"]["hit_rate"],
            rolling_snapshot["12M"]["hit_rate"],
            rolling_snapshot["24M"]["hit_rate"],
        )

        return result

    def run_loop(self) -> List[DrawStepResult]:
        results: List[DrawStepResult] = []

        logger.info(
            "STEP 3 adaptive loop starting. source_draw_range=%d..%d target_range=%d..%d",
            self.start_draw_no,
            self.end_draw_no - 1,
            self.start_draw_no + 1,
            self.end_draw_no,
        )

        for source_draw_no in range(self.start_draw_no, self.end_draw_no):
            result = self.run_one_step(source_draw_no)

            if result is not None:
                results.append(result)

        logger.info("STEP 3 adaptive loop completed. steps=%d", len(results))

        return results


# ============================================================
# AUDIT ENTRYPOINT
# ============================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


def print_result_summary(results: Sequence[DrawStepResult]) -> None:
    print_header("STEP 3 IN-MEMORY RUN SUMMARY")

    total = len(results)
    hit_steps = sum(1 for r in results if r.hit_count > 0)
    adaptive_count = sum(1 for r in results if r.adaptive_triggered)

    print(f"TOTAL_STEPS: {total}")
    print(f"STEPS_WITH_HIT: {hit_steps}")
    print(f"ADAPTIVE_TRIGGER_COUNT: {adaptive_count}")
    print(f"OVERALL_BINARY_HIT_RATE: {(hit_steps / total) if total else 0.0:.8f}")

    if results:
        last = results[-1]
        print(f"LAST_SOURCE_DRAW_NO: {last.source_draw_no}")
        print(f"LAST_TARGET_DRAW_NO: {last.target_draw_no}")
        print(f"LAST_TOP5: {list(last.top5)}")
        print(f"LAST_ENGINE_SOURCES: {list(last.engine_sources)}")
        print(f"LAST_ROLLING_METRICS: {last.rolling_metrics}")


def main() -> int:
    print_header("STEP 3 ADAPTIVE ORCHESTRATOR — START")

    load_env_file()
    conn_str = get_sql_connection_string_from_env()
    core = import_step2_core()

    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DB_SERVER: {os.getenv('DB_SERVER', '<not set>')}")
    print(f"DB_DATABASE: {os.getenv('DB_DATABASE', '<not set>')}")
    print("STEP2_IMPORT: OK")
    print("STEP3_SCOPE: DiversityGuard + AdaptiveFeedback only")

    with core.SqlServerGateway(conn_str) as gateway:
        orchestrator = Step3AdaptiveOrchestrator(
            core=core,
            gateway=gateway,
            start_draw_no=SIM_START_DRAW_NO,
            end_draw_no=SIM_END_DRAW_NO,
        )

        results = orchestrator.run_loop()

    print_result_summary(results)

    print_header("STEP 3 ADAPTIVE ORCHESTRATOR — COMPLETED")
    print("RESULT: COMPLETED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print_header("STEP 3 ADAPTIVE ORCHESTRATOR — FAILED")
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise
