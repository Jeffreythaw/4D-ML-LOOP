from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pyodbc
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

load_dotenv(BACKEND_ROOT / ".env")

from app.core.config import get_settings
from app.core.db import record_predictions_to_ledger
from app.core.ml_adapter import run_existing_engine_prediction
from app.schemas.prediction import PredictionCandidate, PredictionRequest


MODE = "Weighted_Grand_Loop"
META_ENGINE = "E2_WEIGHTED_META_ENSEMBLE_RANKER"
UNDERLYING_MODE = "Engine_Grand_Loop"
TOP_K = 5
DEFAULT_START_SOURCE_DRAW_NO = 4050

ENGINES = (
    "E1_CROSS_PAIR_LINEAR",
    "E1_WLS_DECAY_0.98",
    "E1_MIRROR_BASE5_LSTS",
    "E1_DELTA_ROTATION_LSTS",
)


@dataclass(frozen=True)
class EngineWeight:
    engine_source: str
    weight: float
    draws_checked: int
    draws_with_hit: int
    raw_hits: int


@dataclass(frozen=True)
class WeightedMetaSummary:
    source_start: int
    source_end: int
    draws_checked: int
    draws_with_hit: int
    raw_hits: int
    ledger_rows: int

    @property
    def hit_rate(self) -> float:
        return (self.draws_with_hit / self.draws_checked * 100.0) if self.draws_checked else 0.0


def get_conn():
    settings = get_settings()
    return pyodbc.connect(settings.sql_connection_string(), timeout=60)


def fetch_latest_draw_no() -> int:
    with get_conn() as conn:
        row = conn.cursor().execute(
            "SELECT MAX(DrawNo) AS LatestDrawNo FROM dbo.DrawHistory;"
        ).fetchone()

    if row is None or row.LatestDrawNo is None:
        raise RuntimeError("dbo.DrawHistory has no rows")

    return int(row.LatestDrawNo)


def call_sql_firewall_verify(
    cursor,
    *,
    target_draw_no: int,
    predictions: List[str],
) -> int:
    if len(predictions) != TOP_K:
        raise RuntimeError(f"Expected {TOP_K} predictions, got {len(predictions)}")

    row = cursor.execute(
        """
        EXEC dbo.SP_Verify_Predictions
            @TargetDrawNo = ?,
            @Top5Predictions = ?;
        """,
        int(target_draw_no),
        ",".join(predictions),
    ).fetchone()

    if row is None:
        raise RuntimeError("SP_Verify_Predictions returned no row")

    return int(row[0])


def fetch_verified_meta_group(
    cursor,
    *,
    source_draw_no: int,
    target_draw_no: int,
) -> tuple[bool, int]:
    row = cursor.execute(
        """
        SELECT
            COUNT(*) AS LedgerRows,
            MAX(ISNULL(HitCount, 0)) AS GroupHitCount
        FROM dbo.PredictionLedger
        WHERE Mode = ?
          AND EngineSource = ?
          AND SourceDrawNo = ?
          AND TargetDrawNo = ?
          AND VerificationStatus = 'Verified';
        """,
        MODE,
        META_ENGINE,
        int(source_draw_no),
        int(target_draw_no),
    ).fetchone()

    ledger_rows = int(row.LedgerRows or 0)
    hit_count = int(row.GroupHitCount or 0)

    return ledger_rows == TOP_K, hit_count


def update_weighted_meta_verification(
    cursor,
    *,
    source_draw_no: int,
    target_draw_no: int,
    hit_count: int,
) -> int:
    cursor.execute(
        """
        UPDATE dbo.PredictionLedger
        SET
            VerificationStatus = 'Verified',
            HitCount = ?,
            VerifiedAt = SYSUTCDATETIME()
        WHERE Mode = ?
          AND SourceDrawNo = ?
          AND TargetDrawNo = ?
          AND EngineSource = ?;
        """,
        int(hit_count),
        MODE,
        int(source_draw_no),
        int(target_draw_no),
        META_ENGINE,
    )

    return int(cursor.rowcount)


def extract_underlying_candidates(
    ledger_predictions: Iterable[PredictionCandidate],
) -> Dict[str, List[PredictionCandidate]]:
    grouped: Dict[str, List[PredictionCandidate]] = defaultdict(list)

    for item in ledger_predictions:
        source = str(item.source)
        if source in ENGINES:
            grouped[source].append(item)

    result: Dict[str, List[PredictionCandidate]] = {}

    for engine in ENGINES:
        items = sorted(grouped.get(engine, []), key=lambda item: int(item.rank))
        if len(items) != TOP_K:
            raise RuntimeError(f"{engine} expected {TOP_K} rows, got {len(items)}")
        result[engine] = items

    return result


def fetch_engine_cumulative_stats(
    cursor,
    *,
    engine_source: str,
    source_draw_no: int,
) -> tuple[int, int, int]:
    """
    Temporal firewall:
      For source draw N predicting N+1, only use previously resolved targets <= N.
      Equivalent ledger predicate: TargetDrawNo <= source_draw_no.
    """
    row = cursor.execute(
        """
        WITH EngineGroups AS (
            SELECT
                SourceDrawNo,
                TargetDrawNo,
                MAX(ISNULL(HitCount, 0)) AS GroupHitCount,
                COUNT(*) AS LedgerRows
            FROM dbo.PredictionLedger
            WHERE Mode = ?
              AND EngineSource = ?
              AND TargetDrawNo <= ?
              AND VerificationStatus = 'Verified'
            GROUP BY SourceDrawNo, TargetDrawNo
        )
        SELECT
            COUNT(*) AS DrawsChecked,
            SUM(CASE WHEN GroupHitCount > 0 THEN 1 ELSE 0 END) AS DrawsWithHit,
            SUM(GroupHitCount) AS RawHits
        FROM EngineGroups
        WHERE LedgerRows = 5;
        """,
        UNDERLYING_MODE,
        engine_source,
        int(source_draw_no),
    ).fetchone()

    return (
        int(row.DrawsChecked or 0),
        int(row.DrawsWithHit or 0),
        int(row.RawHits or 0),
    )


def compute_engine_weights(
    cursor,
    *,
    source_draw_no: int,
) -> Dict[str, EngineWeight]:
    raw_stats: Dict[str, tuple[int, int, int]] = {}
    raw_rates: Dict[str, float] = {}

    for engine in ENGINES:
        draws_checked, draws_with_hit, raw_hits = fetch_engine_cumulative_stats(
            cursor,
            engine_source=engine,
            source_draw_no=source_draw_no,
        )

        raw_stats[engine] = (draws_checked, draws_with_hit, raw_hits)
        raw_rates[engine] = (draws_with_hit / draws_checked) if draws_checked else 0.0

    mean_rate = sum(raw_rates.values()) / len(ENGINES)

    weights: Dict[str, EngineWeight] = {}

    for engine in ENGINES:
        draws_checked, draws_with_hit, raw_hits = raw_stats[engine]
        rate = raw_rates[engine]

        if draws_checked <= 0 or mean_rate <= 0:
            engine_weight = 1.0
        else:
            # Relative cumulative accuracy. If engine is 25% better than mean,
            # its Borda contribution becomes 1.25x. Clamp to prevent overfitting
            # in sparse early windows.
            engine_weight = rate / mean_rate
            engine_weight = max(0.70, min(engine_weight, 1.80))

        weights[engine] = EngineWeight(
            engine_source=engine,
            weight=float(engine_weight),
            draws_checked=draws_checked,
            draws_with_hit=draws_with_hit,
            raw_hits=raw_hits,
        )

    return weights


def build_weighted_meta_candidates(
    *,
    grouped: Dict[str, List[PredictionCandidate]],
    weights: Dict[str, EngineWeight],
) -> List[PredictionCandidate]:
    points_by_number: Dict[str, float] = {}
    best_rank_by_number: Dict[str, int] = {}
    score_by_number: Dict[str, float] = {}
    first_seen_by_number: Dict[str, int] = {}

    seen_idx = 0

    for engine in ENGINES:
        engine_weight = weights[engine].weight

        for item in grouped[engine]:
            rank_no = int(item.rank)
            number = str(item.number).zfill(4)
            source_score = float(item.score or 0.0)

            if rank_no < 1 or rank_no > TOP_K:
                continue

            borda_points = float(TOP_K - rank_no + 1)
            weighted_points = borda_points * engine_weight

            points_by_number[number] = points_by_number.get(number, 0.0) + weighted_points
            score_by_number[number] = score_by_number.get(number, 0.0) + source_score
            best_rank_by_number[number] = min(best_rank_by_number.get(number, TOP_K + 1), rank_no)
            first_seen_by_number.setdefault(number, seen_idx)

            seen_idx += 1

    ranked_numbers = sorted(
        points_by_number,
        key=lambda number: (
            -points_by_number[number],
            best_rank_by_number[number],
            -score_by_number[number],
            first_seen_by_number[number],
            number,
        ),
    )

    selected = ranked_numbers[:TOP_K]

    if len(selected) != TOP_K:
        raise RuntimeError(f"{META_ENGINE} failed to produce {TOP_K}; got {len(selected)}")

    return [
        PredictionCandidate(
            rank=rank_no,
            number=number,
            score=float(points_by_number[number]),
            source=META_ENGINE,
        )
        for rank_no, number in enumerate(selected, start=1)
    ]


def fetch_summary(source_start: int, source_end: int) -> WeightedMetaSummary:
    with get_conn() as conn:
        row = conn.cursor().execute(
            """
            WITH MetaGroups AS (
                SELECT
                    SourceDrawNo,
                    TargetDrawNo,
                    MAX(ISNULL(HitCount, 0)) AS GroupHitCount,
                    COUNT(*) AS LedgerRows
                FROM dbo.PredictionLedger
                WHERE Mode = ?
                  AND EngineSource = ?
                  AND SourceDrawNo BETWEEN ? AND ?
                  AND TargetDrawNo BETWEEN ? AND ?
                  AND VerificationStatus = 'Verified'
                GROUP BY SourceDrawNo, TargetDrawNo
            )
            SELECT
                COUNT(*) AS DrawsChecked,
                SUM(CASE WHEN GroupHitCount > 0 THEN 1 ELSE 0 END) AS DrawsWithHit,
                SUM(GroupHitCount) AS RawHits,
                SUM(LedgerRows) AS LedgerRows
            FROM MetaGroups
            WHERE LedgerRows = 5;
            """,
            MODE,
            META_ENGINE,
            int(source_start),
            int(source_end),
            int(source_start + 1),
            int(source_end + 1),
        ).fetchone()

    return WeightedMetaSummary(
        source_start=int(source_start),
        source_end=int(source_end),
        draws_checked=int(row.DrawsChecked or 0),
        draws_with_hit=int(row.DrawsWithHit or 0),
        raw_hits=int(row.RawHits or 0),
        ledger_rows=int(row.LedgerRows or 0),
    )


def print_summary(summary: WeightedMetaSummary) -> None:
    expected_rows = summary.draws_checked * TOP_K

    print("=" * 96)
    print("STEP 140 — WEIGHTED GRAND LOOP SUMMARY")
    print("=" * 96)
    print(f"Mode: {MODE}")
    print(f"EngineSource: {META_ENGINE}")
    print(f"Source Draw Range: {summary.source_start}..{summary.source_end}")
    print(f"Target Draw Range: {summary.source_start + 1}..{summary.source_end + 1}")
    print("-" * 96)
    print(f"Draws Checked:          {summary.draws_checked}")
    print(f"Draws With >=1 Hit:     {summary.draws_with_hit}")
    print(f"Raw Hits:               {summary.raw_hits}")
    print(f"Hit Rate:               {summary.hit_rate:.6f}%")
    print(f"Verified Ledger Rows:   {summary.ledger_rows}")
    print(f"Expected Ledger Rows:   {expected_rows}")
    print("=" * 96)

    if summary.ledger_rows != expected_rows:
        raise RuntimeError(
            f"Ledger row mismatch: expected {expected_rows}, got {summary.ledger_rows}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run weighted meta ensemble grand master loop."
    )
    parser.add_argument("--start-source", type=int, default=DEFAULT_START_SOURCE_DRAW_NO)
    parser.add_argument("--end-source", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    latest_draw_no = fetch_latest_draw_no()
    source_start = int(args.start_source)
    source_end = int(args.end_source) if args.end_source is not None else latest_draw_no - 1

    if source_end >= latest_draw_no:
        raise ValueError(f"source_end must be <= {latest_draw_no - 1}, got {source_end}")

    if source_start > source_end:
        raise ValueError(f"Invalid source range: {source_start}..{source_end}")

    total = source_end - source_start + 1

    print("=" * 96)
    print("STEP 140 — WEIGHTED GRAND MASTER LOOP")
    print("=" * 96)
    print(f"Latest historical DrawNo: {latest_draw_no}")
    print(f"Mode: {MODE}")
    print(f"EngineSource: {META_ENGINE}")
    print(f"Source Draw Range: {source_start}..{source_end}")
    print(f"Target Draw Range: {source_start + 1}..{source_end + 1}")
    print(f"Total draw steps: {total}")
    print("=" * 96)

    processed_count = 0
    skipped_count = 0

    with get_conn() as verify_conn:
        cursor = verify_conn.cursor()

        for index, source_draw_no in enumerate(range(source_start, source_end + 1), start=1):
            target_draw_no = source_draw_no + 1

            already_verified, existing_hit_count = fetch_verified_meta_group(
                cursor,
                source_draw_no=source_draw_no,
                target_draw_no=target_draw_no,
            )

            if already_verified:
                skipped_count += 1

                if index == 1 or index == total or index % int(args.progress_every) == 0:
                    print(
                        f"[{index:>5}/{total:<5}] "
                        f"{source_draw_no}->{target_draw_no} "
                        f"SKIP already_verified hits={existing_hit_count}"
                    )

                continue

            result = run_existing_engine_prediction(
                PredictionRequest(draw_number=source_draw_no, mode="Historical")
            )

            grouped = extract_underlying_candidates(result.ledger_predictions)
            weights = compute_engine_weights(cursor, source_draw_no=source_draw_no)
            meta_candidates = build_weighted_meta_candidates(grouped=grouped, weights=weights)

            record_predictions_to_ledger(
                mode=MODE,
                source_draw_no=source_draw_no,
                target_draw_no=target_draw_no,
                day_type=result.day_type,
                predictions=meta_candidates,
            )

            top5 = [str(item.number).zfill(4) for item in meta_candidates]
            hit_count = call_sql_firewall_verify(
                cursor,
                target_draw_no=target_draw_no,
                predictions=top5,
            )

            affected = update_weighted_meta_verification(
                cursor,
                source_draw_no=source_draw_no,
                target_draw_no=target_draw_no,
                hit_count=hit_count,
            )

            if affected != TOP_K:
                raise RuntimeError(
                    f"Verification update affected {affected} rows for "
                    f"{source_draw_no}->{target_draw_no}; expected {TOP_K}"
                )

            verify_conn.commit()
            processed_count += 1

            if index == 1 or index == total or index % int(args.progress_every) == 0:
                weight_text = " ".join(
                    f"{engine}={weights[engine].weight:.3f}"
                    for engine in ENGINES
                )
                print(
                    f"[{index:>5}/{total:<5}] "
                    f"{source_draw_no}->{target_draw_no} "
                    f"day={result.day_type} "
                    f"hits={hit_count} "
                    f"top5={','.join(top5)} "
                    f"weights: {weight_text}"
                )

    print("-" * 96)
    print(f"Processed this run: {processed_count}")
    print(f"Skipped verified:    {skipped_count}")
    print("-" * 96)

    print_summary(fetch_summary(source_start, source_end))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
