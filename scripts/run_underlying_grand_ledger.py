from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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


MODE = "Engine_Grand_Loop"
TOP_K = 5
DEFAULT_START_SOURCE_DRAW_NO = 4050

ENGINES = (
    "E1_CROSS_PAIR_LINEAR",
    "E1_WLS_DECAY_0.98",
    "E1_MIRROR_BASE5_LSTS",
    "E1_DELTA_ROTATION_LSTS",
)


@dataclass(frozen=True)
class EngineGroupSummary:
    engine_source: str
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


def fetch_verified_engine_group(
    cursor,
    *,
    source_draw_no: int,
    target_draw_no: int,
    engine_source: str,
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
        engine_source,
        int(source_draw_no),
        int(target_draw_no),
    ).fetchone()

    ledger_rows = int(row.LedgerRows or 0)
    hit_count = int(row.GroupHitCount or 0)

    return ledger_rows == TOP_K, hit_count


def update_engine_verification(
    cursor,
    *,
    source_draw_no: int,
    target_draw_no: int,
    engine_source: str,
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
        engine_source,
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


def fetch_summary(source_start: int, source_end: int) -> Dict[str, EngineGroupSummary]:
    placeholders = ",".join(["?"] * len(ENGINES))

    query = f"""
        WITH EngineGroups AS (
            SELECT
                EngineSource,
                SourceDrawNo,
                TargetDrawNo,
                MAX(ISNULL(HitCount, 0)) AS GroupHitCount,
                COUNT(*) AS LedgerRows
            FROM dbo.PredictionLedger
            WHERE Mode = ?
              AND EngineSource IN ({placeholders})
              AND SourceDrawNo BETWEEN ? AND ?
              AND TargetDrawNo BETWEEN ? AND ?
              AND VerificationStatus = 'Verified'
            GROUP BY EngineSource, SourceDrawNo, TargetDrawNo
        )
        SELECT
            EngineSource,
            COUNT(*) AS DrawsChecked,
            SUM(CASE WHEN GroupHitCount > 0 THEN 1 ELSE 0 END) AS DrawsWithHit,
            SUM(GroupHitCount) AS RawHits,
            SUM(LedgerRows) AS LedgerRows
        FROM EngineGroups
        GROUP BY EngineSource;
    """

    params = [
        MODE,
        *ENGINES,
        int(source_start),
        int(source_end),
        int(source_start + 1),
        int(source_end + 1),
    ]

    summary = {
        engine: EngineGroupSummary(
            engine_source=engine,
            draws_checked=0,
            draws_with_hit=0,
            raw_hits=0,
            ledger_rows=0,
        )
        for engine in ENGINES
    }

    with get_conn() as conn:
        rows = conn.cursor().execute(query, params).fetchall()

    for row in rows:
        engine = str(row.EngineSource)
        summary[engine] = EngineGroupSummary(
            engine_source=engine,
            draws_checked=int(row.DrawsChecked or 0),
            draws_with_hit=int(row.DrawsWithHit or 0),
            raw_hits=int(row.RawHits or 0),
            ledger_rows=int(row.LedgerRows or 0),
        )

    return summary


def print_summary(
    *,
    source_start: int,
    source_end: int,
    summary: Dict[str, EngineGroupSummary],
) -> None:
    print("=" * 96)
    print("STEP 139 — UNDERLYING ENGINE GRAND LEDGER SUMMARY")
    print("=" * 96)
    print(f"Mode: {MODE}")
    print(f"Source Draw Range: {source_start}..{source_end}")
    print(f"Target Draw Range: {source_start + 1}..{source_end + 1}")
    print("-" * 96)
    print(f"{'Engine':<30} {'DrawsHit':>10} {'HitRate%':>10} {'RawHits':>8} {'Rows':>8}")
    print("-" * 96)

    total_rows = 0
    total_draws = source_end - source_start + 1
    expected_rows = total_draws * len(ENGINES) * TOP_K

    for engine in ENGINES:
        s = summary[engine]
        total_rows += s.ledger_rows
        print(
            f"{engine:<30} "
            f"{s.draws_with_hit:>4}/{s.draws_checked:<5} "
            f"{s.hit_rate:>9.4f}% "
            f"{s.raw_hits:>8} "
            f"{s.ledger_rows:>8}"
        )

    print("-" * 96)
    print(f"{'TOTAL VERIFIED LEDGER ROWS':<76} {total_rows:>8}")
    print(f"{'EXPECTED VERIFIED LEDGER ROWS':<76} {expected_rows:>8}")
    print("=" * 96)

    if total_rows != expected_rows:
        raise RuntimeError(f"Ledger row mismatch: expected {expected_rows}, got {total_rows}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 4 underlying engines across historical timeline into Engine_Grand_Loop ledger."
    )
    parser.add_argument(
        "--start-source",
        type=int,
        default=DEFAULT_START_SOURCE_DRAW_NO,
    )
    parser.add_argument(
        "--end-source",
        type=int,
        default=None,
        help="Defaults to latest DrawNo - 1.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
    )
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
    print("STEP 139 — UNDERLYING ENGINE GRAND LEDGER")
    print("=" * 96)
    print(f"Latest historical DrawNo: {latest_draw_no}")
    print(f"Mode: {MODE}")
    print("Engines:", ", ".join(ENGINES))
    print(f"Source Draw Range: {source_start}..{source_end}")
    print(f"Target Draw Range: {source_start + 1}..{source_end + 1}")
    print(f"Total draw steps: {total}")
    print("=" * 96)

    processed_groups = 0
    skipped_groups = 0

    with get_conn() as verify_conn:
        verify_cursor = verify_conn.cursor()

        for index, source_draw_no in enumerate(range(source_start, source_end + 1), start=1):
            target_draw_no = source_draw_no + 1

            already_verified: Dict[str, tuple[bool, int]] = {}
            for engine in ENGINES:
                already_verified[engine] = fetch_verified_engine_group(
                    verify_cursor,
                    source_draw_no=source_draw_no,
                    target_draw_no=target_draw_no,
                    engine_source=engine,
                )

            if all(done for done, _hit in already_verified.values()):
                skipped_groups += len(ENGINES)

                if index == 1 or index == total or index % int(args.progress_every) == 0:
                    hits_text = " ".join(
                        f"{engine}={hit}"
                        for engine, (_done, hit) in already_verified.items()
                    )
                    print(
                        f"[{index:>5}/{total:<5}] "
                        f"{source_draw_no}->{target_draw_no} "
                        f"SKIP all_verified {hits_text}"
                    )

                continue

            result = run_existing_engine_prediction(
                PredictionRequest(draw_number=source_draw_no, mode="Historical")
            )

            grouped = extract_underlying_candidates(result.ledger_predictions)

            all_candidates: List[PredictionCandidate] = []
            for engine in ENGINES:
                all_candidates.extend(grouped[engine])

            record_predictions_to_ledger(
                mode=MODE,
                source_draw_no=source_draw_no,
                target_draw_no=target_draw_no,
                day_type=result.day_type,
                predictions=all_candidates,
            )

            hit_parts = []

            for engine in ENGINES:
                done, existing_hit = already_verified[engine]

                if done:
                    skipped_groups += 1
                    hit_parts.append(f"{engine}=SKIP:{existing_hit}")
                    continue

                top5 = [str(item.number).zfill(4) for item in grouped[engine]]
                hit_count = call_sql_firewall_verify(
                    verify_cursor,
                    target_draw_no=target_draw_no,
                    predictions=top5,
                )

                affected = update_engine_verification(
                    verify_cursor,
                    source_draw_no=source_draw_no,
                    target_draw_no=target_draw_no,
                    engine_source=engine,
                    hit_count=hit_count,
                )

                if affected != TOP_K:
                    raise RuntimeError(
                        f"{engine} verification update affected {affected} rows for "
                        f"{source_draw_no}->{target_draw_no}; expected {TOP_K}"
                    )

                verify_conn.commit()
                processed_groups += 1
                hit_parts.append(f"{engine}={hit_count}")

            if index == 1 or index == total or index % int(args.progress_every) == 0:
                print(
                    f"[{index:>5}/{total:<5}] "
                    f"{source_draw_no}->{target_draw_no} "
                    f"day={result.day_type} "
                    + " ".join(hit_parts)
                )

    print("-" * 96)
    print(f"Processed engine groups this run: {processed_groups}")
    print(f"Skipped verified engine groups:   {skipped_groups}")
    print("-" * 96)

    print_summary(
        source_start=source_start,
        source_end=source_end,
        summary=fetch_summary(source_start, source_end),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
