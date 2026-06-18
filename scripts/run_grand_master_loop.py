from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

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


MODE = "Grand_Loop"
META_ENGINE = "E2_META_ENSEMBLE_RANKER"
TOP_K = 5
DEFAULT_START_SOURCE_DRAW_NO = 4050


@dataclass(frozen=True)
class GrandLoopSummary:
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
    return pyodbc.connect(settings.sql_connection_string(), timeout=30)


def fetch_latest_draw_no() -> int:
    with get_conn() as conn:
        row = conn.cursor().execute(
            "SELECT MAX(DrawNo) AS LatestDrawNo FROM dbo.DrawHistory;"
        ).fetchone()

    if row is None or row.LatestDrawNo is None:
        raise RuntimeError("dbo.DrawHistory has no rows")

    return int(row.LatestDrawNo)


def call_sql_firewall_verify(target_draw_no: int, predictions: List[str]) -> int:
    if len(predictions) != TOP_K:
        raise RuntimeError(f"Expected {TOP_K} predictions, got {len(predictions)}")

    with get_conn() as conn:
        row = conn.cursor().execute(
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


def update_meta_verification(
    *,
    source_draw_no: int,
    target_draw_no: int,
    hit_count: int,
) -> int:
    with get_conn() as conn:
        cursor = conn.cursor()
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
        affected = int(cursor.rowcount)
        conn.commit()

    return affected


def extract_meta_candidates(ledger_predictions: Iterable[PredictionCandidate]) -> List[PredictionCandidate]:
    meta = [
        item
        for item in ledger_predictions
        if str(item.source) == META_ENGINE
    ]

    meta = sorted(meta, key=lambda item: int(item.rank))

    if len(meta) != TOP_K:
        raise RuntimeError(f"{META_ENGINE} expected {TOP_K} rows, got {len(meta)}")

    return meta



def fetch_verified_meta_group(
    *,
    source_draw_no: int,
    target_draw_no: int,
) -> tuple[bool, int]:
    """
    Return whether the Grand_Loop meta group is already fully verified.

    A complete verified group means:
      - Mode = Grand_Loop
      - EngineSource = E2_META_ENSEMBLE_RANKER
      - Source/Target match
      - VerificationStatus = Verified
      - exactly TOP_K ledger rows
    """
    with get_conn() as conn:
        row = conn.cursor().execute(
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


def fetch_summary(source_start: int, source_end: int) -> GrandLoopSummary:
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
            FROM MetaGroups;
            """,
            MODE,
            META_ENGINE,
            int(source_start),
            int(source_end),
            int(source_start + 1),
            int(source_end + 1),
        ).fetchone()

    return GrandLoopSummary(
        source_start=int(source_start),
        source_end=int(source_end),
        draws_checked=int(row.DrawsChecked or 0),
        draws_with_hit=int(row.DrawsWithHit or 0),
        raw_hits=int(row.RawHits or 0),
        ledger_rows=int(row.LedgerRows or 0),
    )


def print_summary(summary: GrandLoopSummary) -> None:
    expected_rows = summary.draws_checked * TOP_K

    print("=" * 88)
    print("STEP 137 — GRAND MASTER LOOP SUMMARY")
    print("=" * 88)
    print(f"Mode: {MODE}")
    print(f"EngineSource: {META_ENGINE}")
    print(f"Source Draw Range: {summary.source_start}..{summary.source_end}")
    print(f"Target Draw Range: {summary.source_start + 1}..{summary.source_end + 1}")
    print("-" * 88)
    print(f"Draws Checked:          {summary.draws_checked}")
    print(f"Draws With >=1 Hit:     {summary.draws_with_hit}")
    print(f"Raw Hits:               {summary.raw_hits}")
    print(f"Hit Rate:               {summary.hit_rate:.4f}%")
    print(f"Verified Ledger Rows:   {summary.ledger_rows}")
    print(f"Expected Ledger Rows:   {expected_rows}")
    print("=" * 88)

    if summary.ledger_rows != expected_rows:
        raise RuntimeError(
            f"Ledger row mismatch: expected {expected_rows}, got {summary.ledger_rows}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E2_META_ENSEMBLE_RANKER Grand_Loop historical backtest."
    )
    parser.add_argument(
        "--start-source",
        type=int,
        default=DEFAULT_START_SOURCE_DRAW_NO,
        help="First source/base DrawNo N. Target is N+1.",
    )
    parser.add_argument(
        "--end-source",
        type=int,
        default=None,
        help="Last source/base DrawNo N. Defaults to latest historical draw - 1.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N source draws.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    latest_draw_no = fetch_latest_draw_no()
    source_start = int(args.start_source)
    source_end = int(args.end_source) if args.end_source is not None else latest_draw_no - 1

    if source_start < 1:
        raise ValueError("source_start must be >= 1")

    if source_end >= latest_draw_no:
        raise ValueError(
            f"source_end must be <= latest_draw_no - 1 ({latest_draw_no - 1}); got {source_end}"
        )

    if source_start > source_end:
        raise ValueError(f"Invalid source range: {source_start}..{source_end}")

    total = source_end - source_start + 1

    print("=" * 88)
    print("STEP 137 — GRAND MASTER LOOP")
    print("=" * 88)
    print(f"Latest historical DrawNo: {latest_draw_no}")
    print(f"Mode: {MODE}")
    print(f"EngineSource: {META_ENGINE}")
    print(f"Source Draw Range: {source_start}..{source_end}")
    print(f"Target Draw Range: {source_start + 1}..{source_end + 1}")
    print(f"Total draw steps: {total}")
    print("=" * 88)

    processed_count = 0
    skipped_count = 0

    for index, source_draw_no in enumerate(range(source_start, source_end + 1), start=1):
        target_draw_no = source_draw_no + 1

        already_verified, existing_hit_count = fetch_verified_meta_group(
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

        meta_candidates = extract_meta_candidates(result.ledger_predictions)

        record_predictions_to_ledger(
            mode=MODE,
            source_draw_no=source_draw_no,
            target_draw_no=target_draw_no,
            day_type=result.day_type,
            predictions=meta_candidates,
        )

        top5 = [str(item.number).zfill(4) for item in meta_candidates]
        hit_count = call_sql_firewall_verify(target_draw_no, top5)
        affected = update_meta_verification(
            source_draw_no=source_draw_no,
            target_draw_no=target_draw_no,
            hit_count=hit_count,
        )

        if affected != TOP_K:
            raise RuntimeError(
                f"Verification update affected {affected} rows for "
                f"{source_draw_no}->{target_draw_no}; expected {TOP_K}"
            )

        processed_count += 1

        if index == 1 or index == total or index % int(args.progress_every) == 0:
            print(
                f"[{index:>5}/{total:<5}] "
                f"{source_draw_no}->{target_draw_no} "
                f"day={result.day_type} "
                f"hits={hit_count} "
                f"top5={','.join(top5)}"
            )

    print("-" * 88)
    print(f"Processed this run: {processed_count}")
    print(f"Skipped verified:    {skipped_count}")
    print("-" * 88)

    summary = fetch_summary(source_start, source_end)
    print_summary(summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
