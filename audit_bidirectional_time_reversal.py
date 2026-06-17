# audit_bidirectional_time_reversal.py
from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
STEP2_MODULE_NAME = "jeffrey_quad_engine_v2_step2_matrix_core"

TOP_K = 5
MIN_DISTINCT_ENGINES = 3
MARKOV_TOP_N_PER_SOURCE = 25
ADAPTIVE_FORMULA_TOP_N = 20

logger = logging.getLogger("j4d_v2.time_reversal")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)


def load_env_file() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        raise FileNotFoundError(f"Missing .env file: {env_path}")

    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.warning("python-dotenv not installed; using shell environment only.")
        return

    load_dotenv(env_path, override=False)


def get_sql_connection_string_from_env() -> str:
    driver = os.getenv("DB_DRIVER", "").strip()
    server = os.getenv("DB_SERVER", "").strip()
    database = os.getenv("DB_DATABASE", "").strip()
    username = os.getenv("DB_USERNAME", "").strip()
    password = (
        os.getenv("DB_PASSWORD", "").strip()
        or os.getenv("DB_PASS", "").strip()
        or os.getenv("DB_PWD", "").strip()
    )

    if all([driver, server, database, username, password]):
        return (
            f"DRIVER={{{driver}}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"UID={username};"
            f"PWD={password};"
            "TrustServerCertificate=yes;"
        )

    fallback = os.getenv("J4D_SQL_CONN_STR", "").strip()
    if fallback:
        return fallback

    raise EnvironmentError("Missing SQL Server connection settings in .env")


def import_step2_core():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    core = __import__(STEP2_MODULE_NAME)

    required = [
        "SqlServerGateway",
        "MatrixFormula",
        "Engine1CrossPairLinear",
        "Engine2SetProjector",
        "Engine3Polynomial",
        "affine_mod10_transform",
        "solve_4_unknowns_affine",
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


@dataclass(frozen=True)
class CandidateVote:
    number: str
    engine_name: str
    score: float
    rank: int
    detail: str


@dataclass
class CandidateAggregate:
    number: str
    engine_scores: Dict[str, float] = field(default_factory=dict)
    votes: List[CandidateVote] = field(default_factory=list)
    total_score: float = 0.0

    @property
    def best_engine(self) -> str:
        return max(self.engine_scores.items(), key=lambda kv: kv[1])[0]

    @property
    def best_engine_score(self) -> float:
        return max(self.engine_scores.values())

    @property
    def engine_count(self) -> int:
        return len(self.engine_scores)


@dataclass(frozen=True)
class LockedTop5:
    top5: Tuple[str, ...]
    engines: Tuple[str, ...]


@dataclass
class RunSummary:
    direction: str
    total_checked: int
    hit_steps: int
    adaptive_triggers: int
    formula_types: Counter[str]

    @property
    def hit_rate_pct(self) -> float:
        return (self.hit_steps / self.total_checked * 100.0) if self.total_checked else 0.0

    @property
    def top3_formula_types(self) -> str:
        items = self.formula_types.most_common(3)
        return "; ".join(f"{name}={count}" for name, count in items) if items else "None"


def reset_formula_registry(gateway: Any) -> None:
    gateway.conn.cursor().execute("DELETE FROM dbo.FormulaRegistry;")
    gateway.commit()


def load_ordered_draw_numbers(gateway: Any) -> List[int]:
    rows = gateway.conn.cursor().execute(
        """
        SELECT DrawNo
        FROM dbo.DrawHistory
        ORDER BY DrawNo ASC;
        """
    ).fetchall()

    draw_numbers = [int(r.DrawNo) for r in rows]

    if len(draw_numbers) < 2:
        raise RuntimeError("dbo.DrawHistory must contain at least 2 draws")

    return draw_numbers


def load_draw_record(gateway: Any, draw_no: int) -> Any:
    rows = gateway.load_draw_history(min_draw_no=draw_no, max_draw_no=draw_no)
    if not rows:
        raise LookupError(f"DrawNo {draw_no} not found")
    return rows[0]


def score_by_rank(rank: int, total: int, weight: float) -> float:
    if total <= 0:
        return 0.0
    return float(weight * ((total - rank + 1) / total))


def add_vote(pool: Dict[str, CandidateAggregate], vote: CandidateVote) -> None:
    agg = pool.get(vote.number)
    if agg is None:
        agg = CandidateAggregate(number=vote.number)
        pool[vote.number] = agg

    agg.votes.append(vote)
    agg.engine_scores[vote.engine_name] = max(
        agg.engine_scores.get(vote.engine_name, 0.0),
        vote.score,
    )


def finalize_pool(pool: Dict[str, CandidateAggregate]) -> None:
    for agg in pool.values():
        consensus_bonus = 0.12 * max(0, agg.engine_count - 1)
        agg.total_score = float(sum(agg.engine_scores.values()) + consensus_bonus)


def add_static_engine_votes(
    *,
    core: Any,
    pool: Dict[str, CandidateAggregate],
    source_vectors: np.ndarray,
    day_type: str,
) -> None:
    engines = [
        (core.ENGINE_1_NAME, core.Engine1CrossPairLinear()),
        (core.ENGINE_2_NAME, core.Engine2SetProjector()),
        (core.ENGINE_3_NAME, core.Engine3Polynomial()),
    ]

    for engine_name, engine in engines:
        vectors = engine.predict_vectors(source_vectors, day_type)
        numbers = core.strings_from_vectors(vectors)
        seen = set()

        for rank, number in enumerate(numbers, start=1):
            if number in seen:
                continue
            seen.add(number)

            add_vote(
                pool,
                CandidateVote(
                    number=number,
                    engine_name=engine_name,
                    score=score_by_rank(rank, len(numbers), 1.0),
                    rank=rank,
                    detail=f"{engine_name}:rank={rank}",
                ),
            )


def add_markov_votes(
    *,
    gateway: Any,
    pool: Dict[str, CandidateAggregate],
    direction: str,
    source_numbers: Sequence[str],
    day_type: str,
) -> None:
    cursor = gateway.conn.cursor()

    for source_number in source_numbers:
        if direction == "Forward":
            rows = cursor.execute(
                f"""
                SELECT TOP ({MARKOV_TOP_N_PER_SOURCE})
                    SourceState,
                    TargetState,
                    TransitionCount,
                    LastSeenDrawNo
                FROM dbo.vw_MarkovTransitionMass
                WHERE SourceState = ? AND DayType = ?
                ORDER BY TransitionCount DESC, LastSeenDrawNo DESC, TargetState ASC;
                """,
                (source_number, day_type),
            ).fetchall()
        elif direction == "Reverse":
            rows = cursor.execute(
                f"""
                SELECT TOP ({MARKOV_TOP_N_PER_SOURCE})
                    SourceState,
                    TargetState,
                    TransitionCount,
                    LastSeenDrawNo
                FROM dbo.vw_MarkovTransitionMass
                WHERE TargetState = ? AND DayType = ?
                ORDER BY TransitionCount DESC, LastSeenDrawNo DESC, SourceState ASC;
                """,
                (source_number, day_type),
            ).fetchall()
        else:
            raise ValueError(f"Invalid direction: {direction}")

        if not rows:
            continue

        max_count = max(int(r.TransitionCount) for r in rows)

        for rank, row in enumerate(rows, start=1):
            count_score = int(row.TransitionCount) / max_count if max_count else 0.0
            rank_score = score_by_rank(rank, len(rows), 1.15)
            final_score = 0.70 * count_score + 0.30 * rank_score

            if direction == "Forward":
                number = str(row.TargetState)
                engine_name = "E4_MARKOV_FORWARD"
            else:
                number = str(row.SourceState)
                engine_name = "E4_MARKOV_REVERSE"

            add_vote(
                pool,
                CandidateVote(
                    number=number,
                    engine_name=engine_name,
                    score=float(final_score),
                    rank=rank,
                    detail=f"{engine_name}:from={source_number}:count={int(row.TransitionCount)}",
                ),
            )


def load_active_adaptive_formulas(
    *,
    gateway: Any,
    core: Any,
    day_type: str,
) -> List[Any]:
    rows = gateway.conn.cursor().execute(
        f"""
        SELECT TOP ({ADAPTIVE_FORMULA_TOP_N})
            EngineName,
            FormulaVersion,
            DayType,
            MatrixPayload,
            HistoricalConfidence,
            HitRateTop5,
            SampleSize,
            CreatedAt
        FROM dbo.FormulaRegistry
        WHERE DayType = ?
          AND EngineName = ?
          AND IsActive = 1
        ORDER BY
            HistoricalConfidence DESC,
            HitRateTop5 DESC,
            SampleSize DESC,
            CreatedAt DESC;
        """,
        (day_type, core.ENGINE_4_NAME),
    ).fetchall()

    formulas = []

    for row in rows:
        payload = json.loads(str(row.MatrixPayload))
        formula = core.MatrixFormula(
            engine_name=str(row.EngineName),
            formula_version=str(row.FormulaVersion),
            day_type=str(row.DayType),
            matrix_m=np.array(payload["matrix_m"], dtype=np.int16),
            bias_b=np.array(payload["bias_b"], dtype=np.int16),
            metadata=payload.get("metadata", {}),
        )
        formula.validate()
        formulas.append(formula)

    return formulas


def add_adaptive_formula_votes(
    *,
    gateway: Any,
    core: Any,
    pool: Dict[str, CandidateAggregate],
    source_vectors: np.ndarray,
    day_type: str,
) -> None:
    formulas = load_active_adaptive_formulas(
        gateway=gateway,
        core=core,
        day_type=day_type,
    )

    for formula_index, formula in enumerate(formulas, start=1):
        vectors = core.affine_mod10_transform(
            source_vectors,
            formula.matrix_m,
            formula.bias_b,
            dtype=np.int16,
        )
        numbers = core.strings_from_vectors(vectors)
        seen = set()
        weight = max(0.25, 0.95 / formula_index)

        for rank, number in enumerate(numbers, start=1):
            if number in seen:
                continue
            seen.add(number)

            add_vote(
                pool,
                CandidateVote(
                    number=number,
                    engine_name=core.ENGINE_4_NAME,
                    score=score_by_rank(rank, len(numbers), weight),
                    rank=rank,
                    detail=f"{formula.formula_version}:rank={rank}",
                ),
            )


def build_candidate_pool(
    *,
    gateway: Any,
    core: Any,
    direction: str,
    source_record: Any,
    source_vectors: np.ndarray,
) -> Dict[str, CandidateAggregate]:
    pool: Dict[str, CandidateAggregate] = {}

    add_static_engine_votes(
        core=core,
        pool=pool,
        source_vectors=source_vectors,
        day_type=source_record.day_type,
    )

    add_markov_votes(
        gateway=gateway,
        pool=pool,
        direction=direction,
        source_numbers=source_record.winning_numbers,
        day_type=source_record.day_type,
    )

    add_adaptive_formula_votes(
        gateway=gateway,
        core=core,
        pool=pool,
        source_vectors=source_vectors,
        day_type=source_record.day_type,
    )

    if len(pool) < TOP_K:
        raise RuntimeError(f"Candidate pool too small: {len(pool)}")

    finalize_pool(pool)
    return pool


class DiversityGuardRanker:
    def __init__(self, top_k: int = TOP_K, min_engines: int = MIN_DISTINCT_ENGINES) -> None:
        self.top_k = top_k
        self.min_engines = min_engines

    @staticmethod
    def engine_lists(pool: Dict[str, CandidateAggregate]) -> Dict[str, List[CandidateAggregate]]:
        grouped: Dict[str, List[CandidateAggregate]] = defaultdict(list)

        for agg in pool.values():
            for engine in agg.engine_scores:
                grouped[engine].append(agg)

        for engine, items in grouped.items():
            items.sort(
                key=lambda a: (
                    a.engine_scores.get(engine, 0.0),
                    a.total_score,
                    a.engine_count,
                    a.number,
                ),
                reverse=True,
            )

        return dict(grouped)

    def lock_top5(self, pool: Dict[str, CandidateAggregate]) -> LockedTop5:
        engines = sorted({e for agg in pool.values() for e in agg.engine_scores})
        required = min(self.min_engines, len(engines))

        by_engine = self.engine_lists(pool)
        global_ranked = sorted(
            pool.values(),
            key=lambda a: (
                a.total_score,
                a.engine_count,
                a.best_engine_score,
                a.number,
            ),
            reverse=True,
        )

        selected: List[CandidateAggregate] = []
        selected_numbers = set()
        selected_engines: List[str] = []
        engine_counts: Counter[str] = Counter()

        engine_order = sorted(
            engines,
            key=lambda e: (
                by_engine[e][0].engine_scores.get(e, 0.0) if by_engine.get(e) else 0.0,
                len(by_engine.get(e, [])),
                e,
            ),
            reverse=True,
        )

        for engine in engine_order:
            if len(set(selected_engines)) >= required:
                break

            for candidate in by_engine.get(engine, []):
                if candidate.number in selected_numbers:
                    continue
                selected.append(candidate)
                selected_numbers.add(candidate.number)
                selected_engines.append(engine)
                engine_counts[engine] += 1
                break

        for candidate in global_ranked:
            if len(selected) >= self.top_k:
                break
            if candidate.number in selected_numbers:
                continue

            engine = candidate.best_engine
            if len(set(selected_engines)) < required and engine_counts[engine] >= 2:
                continue

            selected.append(candidate)
            selected_numbers.add(candidate.number)
            selected_engines.append(engine)
            engine_counts[engine] += 1

        for candidate in global_ranked:
            if len(selected) >= self.top_k:
                break
            if candidate.number in selected_numbers:
                continue

            engine = candidate.best_engine
            selected.append(candidate)
            selected_numbers.add(candidate.number)
            selected_engines.append(engine)
            engine_counts[engine] += 1

        if len(selected) != self.top_k:
            raise RuntimeError(f"DiversityGuard failed: selected {len(selected)}")

        if len(set(selected_engines)) < required:
            raise RuntimeError(
                f"DiversityGuard distinct-engine failure: required={required}, got={len(set(selected_engines))}"
            )

        return LockedTop5(
            top5=tuple(c.number for c in selected),
            engines=tuple(selected_engines),
        )


def register_adaptive_formula(
    *,
    gateway: Any,
    core: Any,
    direction: str,
    source_draw_no: int,
    target_draw_no: int,
    source_vectors: np.ndarray,
    target_vectors: np.ndarray,
    target_day_type: str,
) -> int:
    formula_type = (
        "E4_AFFINE_LSTSQ_FORWARD"
        if direction == "Forward"
        else "E4_AFFINE_LSTSQ_REVERSE"
    )

    formula_version = (
        f"F{source_draw_no}T{target_draw_no}"
        if direction == "Forward"
        else f"R{source_draw_no}T{target_draw_no}"
    )

    formula = core.solve_4_unknowns_affine(
        src_vectors=source_vectors,
        tgt_vectors=target_vectors,
        day_type=target_day_type,
        formula_version=formula_version,
        source_draw_no=source_draw_no,
        target_draw_no=target_draw_no,
    )

    formula.metadata["direction"] = direction
    formula.metadata["formula_type"] = formula_type
    formula.metadata["firewall_boundary"] = (
        "Target winners loaded only after SP_Verify_Predictions returned HitCount == 0"
    )

    formula_id = gateway.register_formula(
        formula,
        training_start_draw_no=min(source_draw_no, target_draw_no),
        training_end_draw_no=max(source_draw_no, target_draw_no),
        historical_confidence=0.0,
        hit_rate_top5=0.0,
        sample_size=int(min(source_vectors.shape[0], target_vectors.shape[0])),
        is_active=True,
    )

    gateway.commit()
    return int(formula_id)


def run_direction(
    *,
    gateway: Any,
    core: Any,
    direction: str,
    ordered_draw_numbers: Sequence[int],
) -> RunSummary:
    reset_formula_registry(gateway)

    if direction == "Forward":
        pairs = list(zip(ordered_draw_numbers[:-1], ordered_draw_numbers[1:]))
    elif direction == "Reverse":
        reversed_numbers = list(reversed(ordered_draw_numbers))
        pairs = list(zip(reversed_numbers[:-1], reversed_numbers[1:]))
    else:
        raise ValueError(f"Invalid direction: {direction}")

    ranker = DiversityGuardRanker()
    formula_counter: Counter[str] = Counter()

    total_checked = 0
    hit_steps = 0
    adaptive_triggers = 0

    logger.info(
        "%s run starting. pairs=%d first_pair=%s last_pair=%s",
        direction,
        len(pairs),
        pairs[0],
        pairs[-1],
    )

    for idx, (source_draw_no, target_draw_no) in enumerate(pairs, start=1):
        source_record = load_draw_record(gateway, source_draw_no)
        source_vectors = core.vectors_from_4d_strings(
            source_record.winning_numbers,
            dtype=np.int16,
        )

        pool = build_candidate_pool(
            gateway=gateway,
            core=core,
            direction=direction,
            source_record=source_record,
            source_vectors=source_vectors,
        )

        locked = ranker.lock_top5(pool)

        hit_count = gateway.verify_predictions(
            target_draw_no=target_draw_no,
            predictions=locked.top5,
        )

        total_checked += 1

        if hit_count > 0:
            hit_steps += 1
        else:
            adaptive_triggers += 1

            target_record = load_draw_record(gateway, target_draw_no)
            target_vectors = core.vectors_from_4d_strings(
                target_record.winning_numbers,
                dtype=np.int16,
            )

            register_adaptive_formula(
                gateway=gateway,
                core=core,
                direction=direction,
                source_draw_no=source_draw_no,
                target_draw_no=target_draw_no,
                source_vectors=source_vectors,
                target_vectors=target_vectors,
                target_day_type=target_record.day_type,
            )

            formula_type = (
                "E4_AFFINE_LSTSQ_FORWARD"
                if direction == "Forward"
                else "E4_AFFINE_LSTSQ_REVERSE"
            )
            formula_counter[formula_type] += 1

        if idx == 1 or idx % 100 == 0 or idx == len(pairs):
            rate = hit_steps / total_checked if total_checked else 0.0
            logger.info(
                "%s progress %d/%d source=%d target=%d hit=%d running_hit_rate=%.6f adaptive=%d top5=%s engines=%s",
                direction,
                idx,
                len(pairs),
                source_draw_no,
                target_draw_no,
                hit_count,
                rate,
                adaptive_triggers,
                list(locked.top5),
                list(locked.engines),
            )

    return RunSummary(
        direction=direction,
        total_checked=total_checked,
        hit_steps=hit_steps,
        adaptive_triggers=adaptive_triggers,
        formula_types=formula_counter,
    )


def print_header(title: str) -> None:
    print("\n" + "=" * 84)
    print(title)
    print("=" * 84)


def print_comparison(rows: Sequence[RunSummary]) -> None:
    print_header("BIDIRECTIONAL TIME-REVERSAL VALIDATION RESULT")

    print(
        "| Direction | Total Draws Checked | Overall Honest Binary Hit Rate % | "
        "Total Adaptive Triggers Fired | Top 3 Most Frequently Generated Formula Types |"
    )
    print("|---|---:|---:|---:|---|")

    for row in rows:
        print(
            f"| {row.direction} "
            f"| {row.total_checked:,} "
            f"| {row.hit_rate_pct:.6f}% "
            f"| {row.adaptive_triggers:,} "
            f"| {row.top3_formula_types} |"
        )


def main() -> int:
    print_header("TASK B — BIDIRECTIONAL TIME-REVERSAL EXPERIMENT")

    load_env_file()
    conn_str = get_sql_connection_string_from_env()
    core = import_step2_core()

    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DB_SERVER: {os.getenv('DB_SERVER', '<not set>')}")
    print(f"DB_DATABASE: {os.getenv('DB_DATABASE', '<not set>')}")
    print("STEP2_IMPORT: OK")
    print("FIREWALL_RULE: Target winners load only after HitCount == 0")

    with core.SqlServerGateway(conn_str) as gateway:
        draw_numbers = load_ordered_draw_numbers(gateway)

        print(f"TOTAL_DRAWS_AVAILABLE: {len(draw_numbers):,}")
        print(f"FIRST_DRAW_NO: {draw_numbers[0]}")
        print(f"LATEST_DRAW_NO: {draw_numbers[-1]}")

        forward = run_direction(
            gateway=gateway,
            core=core,
            direction="Forward",
            ordered_draw_numbers=draw_numbers,
        )

        reverse = run_direction(
            gateway=gateway,
            core=core,
            direction="Reverse",
            ordered_draw_numbers=draw_numbers,
        )

    print_comparison([forward, reverse])

    print_header("TASK B — COMPLETED")
    print("RESULT: COMPLETED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print_header("TASK B — FAILED")
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise
