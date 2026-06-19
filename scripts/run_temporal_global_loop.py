from __future__ import annotations

import argparse
import inspect
import itertools
import sys
from collections import Counter, defaultdict
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


MODE = "Temporal_Global_Loop"
ENGINE_SOURCE = "E1_TEMPORAL_CONTEXT_MATCH"
UNDERLYING_ENGINES = (
    "E1_CROSS_PAIR_LINEAR",
    "E1_WLS_DECAY_0.98",
    "E1_MIRROR_BASE5_LSTS",
    "E1_DELTA_ROTATION_LSTS",
)
TOP_K = 5
DEFAULT_START_SOURCE_DRAW_NO = 4050
TEMPORAL_COLD_START_MIN_MATCHES = 5
TEMPORAL_COLD_START_FALLBACK_ENGINE = "E1_DELTA_ROTATION_LSTS"

TIER_A_WEIGHT = 4.0
TIER_B_WEIGHT = 3.0
TIER_C_WEIGHT = 2.0
TIER_D_WEIGHT = 1.0
TIER_E_WEIGHT = 2.5
MIN_MONTH_DAY_SAMPLES = 5

TARGET_METADATA_SQL = """
    SELECT
        DrawNo,
        DrawDate,
        DAY(DrawDate) AS DayOfMonth,
        MONTH(DrawDate) AS MonthNo,
        YEAR(DrawDate) AS YearNo,
        DATEPART(weekday, DrawDate) AS WeekdayNo,
        DATEPART(week, DrawDate) AS WeekOfYear,
        CASE
            WHEN DATENAME(weekday, DrawDate) IN ('Wednesday', 'Saturday', 'Sunday')
                THEN DATENAME(weekday, DrawDate)
            ELSE 'Special'
        END AS DayType
    FROM dbo.DrawHistory
    WHERE DrawNo = ?;
"""

HISTORICAL_CLUSTER_SQL = """
    SELECT
        DrawNo,
        WinningNumbers,
        CASE
            WHEN MONTH(DrawDate) = ?
             AND DATEPART(weekday, DrawDate) = ?
                THEN 1 ELSE 0
        END AS MatchTierA,
        CASE
            WHEN DATEPART(week, DrawDate) = ?
                THEN 1 ELSE 0
        END AS MatchTierB,
        CASE
            WHEN DAY(DrawDate) = ?
                THEN 1 ELSE 0
        END AS MatchTierC,
        CASE
            WHEN MONTH(DrawDate) = ?
             AND DATEPART(week, DrawDate) = ?
                THEN 1 ELSE 0
        END AS MatchTierD,
        CASE
            WHEN MONTH(DrawDate) = ?
             AND DAY(DrawDate) = ?
                THEN 1 ELSE 0
        END AS MatchTierE
    FROM dbo.DrawHistory
    WHERE DrawNo <= ?
      AND (
            (MONTH(DrawDate) = ? AND DATEPART(weekday, DrawDate) = ?)
         OR DATEPART(week, DrawDate) = ?
         OR DAY(DrawDate) = ?
         OR (MONTH(DrawDate) = ? AND DATEPART(week, DrawDate) = ?)
         OR (MONTH(DrawDate) = ? AND DAY(DrawDate) = ?)
      )
    ORDER BY DrawNo;
"""


@dataclass(frozen=True)
class TemporalMetadata:
    draw_date: object
    day_of_month: int
    month: int
    year: int
    weekday: int
    week_of_year: int
    day_type: str


@dataclass(frozen=True)
class TemporalSummary:
    source_start: int
    source_end: int
    draws_checked: int
    draws_with_hit: int
    raw_hits: int
    ledger_rows: int

    @property
    def hit_rate(self) -> float:
        return (self.draws_with_hit / self.draws_checked * 100.0) if self.draws_checked else 0.0

    @property
    def expected_rows(self) -> int:
        return (self.source_end - self.source_start + 1) * TOP_K


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


def fetch_target_temporal_metadata(cursor, *, target_draw_no: int) -> TemporalMetadata:
    """
    Metadata-only target read. No target winner column is selected or loaded.
    """
    row = cursor.execute(
        TARGET_METADATA_SQL,
        int(target_draw_no),
    ).fetchone()

    if row is None:
        raise RuntimeError(f"Target DrawNo {target_draw_no} not found in dbo.DrawHistory")

    return TemporalMetadata(
        draw_date=row.DrawDate,
        day_of_month=int(row.DayOfMonth),
        month=int(row.MonthNo),
        year=int(row.YearNo),
        weekday=int(row.WeekdayNo),
        week_of_year=int(row.WeekOfYear),
        day_type=str(row.DayType),
    )


def fetch_historical_temporal_cluster(
    cursor,
    *,
    source_draw_no: int,
    target_metadata: TemporalMetadata,
) -> list[tuple[str, float]]:
    """
    Return historical winners and their matching-tier weight.

    Temporal firewall invariant: the winner-bearing query is constrained by
    DrawNo <= source_draw_no. It never reads target DrawNo N+1 winners.
    """
    rows = cursor.execute(
        HISTORICAL_CLUSTER_SQL,
        target_metadata.month,
        target_metadata.weekday,
        target_metadata.week_of_year,
        target_metadata.day_of_month,
        target_metadata.month,
        target_metadata.week_of_year,
        target_metadata.month,
        target_metadata.day_of_month,
        int(source_draw_no),
        target_metadata.month,
        target_metadata.weekday,
        target_metadata.week_of_year,
        target_metadata.day_of_month,
        target_metadata.month,
        target_metadata.week_of_year,
        target_metadata.month,
        target_metadata.day_of_month,
    ).fetchall()

    observations: list[tuple[str, float]] = []
    month_day_samples = sum(int(row.MatchTierE) for row in rows)
    use_month_day_tier = month_day_samples >= MIN_MONTH_DAY_SAMPLES

    for row in rows:
        tier_weight = (
            int(row.MatchTierA) * TIER_A_WEIGHT
            + int(row.MatchTierB) * TIER_B_WEIGHT
            + int(row.MatchTierC) * TIER_C_WEIGHT
            + int(row.MatchTierD) * TIER_D_WEIGHT
            + int(row.MatchTierE) * TIER_E_WEIGHT * int(use_month_day_tier)
        )
        if tier_weight <= 0:
            continue

        for raw_number in str(row.WinningNumbers or "").split(","):
            number = raw_number.strip()
            if len(number) == 4 and number.isdigit():
                observations.append((number, tier_weight))

    if not observations:
        raise RuntimeError(
            f"No historical temporal observations found at or before DrawNo {source_draw_no}"
        )

    return observations


def extract_underlying_candidates(
    ledger_predictions: Iterable[PredictionCandidate],
) -> Dict[str, List[PredictionCandidate]]:
    grouped: Dict[str, List[PredictionCandidate]] = defaultdict(list)

    for item in ledger_predictions:
        source = str(item.source)
        if source in UNDERLYING_ENGINES:
            grouped[source].append(item)

    result: Dict[str, List[PredictionCandidate]] = {}
    for engine in UNDERLYING_ENGINES:
        items = sorted(grouped.get(engine, []), key=lambda item: int(item.rank))
        if len(items) != TOP_K:
            raise RuntimeError(f"{engine} expected {TOP_K} rows, got {len(items)}")
        result[engine] = items

    return result


def build_temporal_candidates(
    *,
    observations: list[tuple[str, float]],
    grouped: Dict[str, List[PredictionCandidate]],
) -> List[PredictionCandidate]:
    exact_frequency: Counter[str] = Counter()
    position_frequency = [Counter() for _ in range(4)]
    digit_sum_frequency: Counter[int] = Counter()
    observation_weight_total = 0.0
    first_seen_by_number: dict[str, int] = {}
    first_seen_index = 0

    for number, weight in observations:
        exact_frequency[number] += weight
        observation_weight_total += weight
        digit_sum_frequency[sum(int(digit) for digit in number)] += weight
        if number not in first_seen_by_number:
            first_seen_by_number[number] = first_seen_index
            first_seen_index += 1
        for position, digit in enumerate(number):
            position_frequency[position][digit] += weight

    candidate_numbers = set(exact_frequency)
    top_position_digits: list[list[str]] = []
    for frequencies in position_frequency:
        ranked_digits = sorted(frequencies, key=lambda digit: (-frequencies[digit], digit))
        top_position_digits.append(ranked_digits[:3])

    for digits in itertools.product(*top_position_digits):
        number = "".join(digits)
        candidate_numbers.add(number)
        if number not in first_seen_by_number:
            first_seen_by_number[number] = first_seen_index
            first_seen_index += 1

    borda_by_number: Counter[str] = Counter()
    best_engine_rank: dict[str, int] = {}
    for engine in UNDERLYING_ENGINES:
        for item in grouped[engine]:
            rank_no = int(item.rank)
            number = str(item.number).zfill(4)
            borda_by_number[number] += TOP_K - rank_no + 1
            best_engine_rank[number] = min(best_engine_rank.get(number, TOP_K + 1), rank_no)
            candidate_numbers.add(number)
            if number not in first_seen_by_number:
                first_seen_by_number[number] = first_seen_index
                first_seen_index += 1

    def position_score(number: str) -> float:
        return sum(
            position_frequency[position][digit] / observation_weight_total
            for position, digit in enumerate(number)
        )

    scores: dict[str, float] = {}
    positions: dict[str, float] = {}
    for number in candidate_numbers:
        exact_score = float(exact_frequency[number])
        pos_score = position_score(number)
        digit_sum_score = (
            float(digit_sum_frequency[sum(int(digit) for digit in number)])
            / observation_weight_total
            * 12.0
        )
        borda_score = float(borda_by_number[number]) * 8.0

        temporal_score = exact_score * 10.0 + pos_score * 20.0 + digit_sum_score
        if borda_by_number[number] and exact_score > 0:
            temporal_score *= 1.50

        scores[number] = temporal_score + borda_score
        positions[number] = pos_score

    ranked_numbers = sorted(
        candidate_numbers,
        key=lambda number: (
            -scores[number],
            -float(exact_frequency[number]),
            best_engine_rank.get(number, TOP_K + 1),
            first_seen_by_number[number],
            number,
        ),
    )

    selected = ranked_numbers[:TOP_K]

    if len(observations) < TEMPORAL_COLD_START_MIN_MATCHES:
        for item in grouped.get(TEMPORAL_COLD_START_FALLBACK_ENGINE, []):
            number = str(item.number).zfill(4)
            if number not in selected:
                selected.append(number)
            if len(selected) == TOP_K:
                break

    if len(selected) < TOP_K:
        for value in range(10000):
            number = f"{value:04d}"
            if number not in selected:
                selected.append(number)
            if len(selected) == TOP_K:
                break

    if len(selected) != TOP_K or len(set(selected)) != TOP_K:
        raise RuntimeError(f"{ENGINE_SOURCE} failed to produce {TOP_K} unique candidates")

    return [
        PredictionCandidate(
            rank=rank_no,
            number=number,
            score=float(scores.get(number, 0.0)),
            source=ENGINE_SOURCE,
        )
        for rank_no, number in enumerate(selected, start=1)
    ]


def fetch_verified_group(
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
        ENGINE_SOURCE,
        int(source_draw_no),
        int(target_draw_no),
    ).fetchone()

    return int(row.LedgerRows or 0) == TOP_K, int(row.GroupHitCount or 0)


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


def update_temporal_verification(
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
          AND EngineSource = ?
          AND SourceDrawNo = ?
          AND TargetDrawNo = ?;
        """,
        int(hit_count),
        MODE,
        ENGINE_SOURCE,
        int(source_draw_no),
        int(target_draw_no),
    )
    return int(cursor.rowcount)


def fetch_summary(source_start: int, source_end: int) -> TemporalSummary:
    with get_conn() as conn:
        row = conn.cursor().execute(
            """
            WITH TemporalGroups AS (
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
            FROM TemporalGroups
            WHERE LedgerRows = ?;
            """,
            MODE,
            ENGINE_SOURCE,
            int(source_start),
            int(source_end),
            int(source_start + 1),
            int(source_end + 1),
            TOP_K,
        ).fetchone()

    return TemporalSummary(
        source_start=int(source_start),
        source_end=int(source_end),
        draws_checked=int(row.DrawsChecked or 0),
        draws_with_hit=int(row.DrawsWithHit or 0),
        raw_hits=int(row.RawHits or 0),
        ledger_rows=int(row.LedgerRows or 0),
    )


def print_summary(summary: TemporalSummary) -> None:
    print("=" * 96)
    print("STEP 144 — TEMPORAL GLOBAL LOOP SUMMARY")
    print("=" * 96)
    print(f"Mode: {MODE}")
    print(f"EngineSource: {ENGINE_SOURCE}")
    print(f"Source Draw Range: {summary.source_start}..{summary.source_end}")
    print(f"Target Draw Range: {summary.source_start + 1}..{summary.source_end + 1}")
    print("-" * 96)
    print(f"Draws checked:          {summary.draws_checked}")
    print(f"Draws with >=1 hit:     {summary.draws_with_hit}")
    print(f"Raw hits:               {summary.raw_hits}")
    print(f"Hit rate:               {summary.hit_rate:.6f}%")
    print(f"Verified ledger rows:   {summary.ledger_rows}")
    print(f"Expected rows:          {summary.expected_rows}")
    print("=" * 96)

    expected_draws = summary.source_end - summary.source_start + 1
    if summary.draws_checked != expected_draws:
        raise RuntimeError(
            f"Draw count mismatch: expected {expected_draws}, got {summary.draws_checked}"
        )
    if summary.ledger_rows != summary.expected_rows:
        raise RuntimeError(
            f"Ledger row mismatch: expected {summary.expected_rows}, got {summary.ledger_rows}"
        )


def print_audit_commands(source_start: int, source_end: int) -> None:
    print("Audit commands:")
    print(
        "EXEC dbo.SP_Summarize_EngineLedger "
        f"@Mode='{MODE}', @EngineSource='{ENGINE_SOURCE}', "
        f"@SourceStart={source_start}, @SourceEnd={source_end};"
    )
    print(
        "EXEC dbo.SP_Audit_PredictionLedgerIntegrity "
        f"@Mode='{MODE}', @EngineSource='{ENGINE_SOURCE}', "
        f"@SourceStart={source_start}, @SourceEnd={source_end};"
    )


def run_preflight_audits() -> None:
    target_sql = " ".join(TARGET_METADATA_SQL.lower().split())
    historical_sql = " ".join(HISTORICAL_CLUSTER_SQL.lower().split())
    main_source = inspect.getsource(main)

    forbidden_target_columns = (
        "winningnumbers",
        "firstprize",
        "secondprize",
        "thirdprize",
        "starter",
        "consolation",
    )
    sql_firewall_ok = (
        "where drawno <= ?" in historical_sql
        and "winningnumbers" in historical_sql
        and "where drawno = ?" in target_sql
        and "drawdate" in target_sql
        and all(column not in target_sql for column in forbidden_target_columns)
    )

    lock_position = main_source.find("record_predictions_to_ledger(")
    verify_position = main_source.find("call_sql_firewall_verify(")
    update_position = main_source.find("update_temporal_verification(")
    leakage_ok = (
        sql_firewall_ok
        and lock_position >= 0
        and verify_position > lock_position
        and update_position > verify_position
    )

    required_draw_columns = {"DrawNo", "DrawDate", "WinningNumbers"}
    required_ledger_columns = {
        "Mode",
        "SourceDrawNo",
        "TargetDrawNo",
        "DayType",
        "RankNo",
        "PredictedNumber",
        "EngineSource",
        "Score",
        "VerificationStatus",
        "HitCount",
        "VerifiedAt",
    }
    required_modes = {
        "Current",
        "Historical",
        "Grand_Loop",
        "Engine_Grand_Loop",
        "Weighted_Grand_Loop",
        "Temporal_Global_Loop",
    }

    with get_conn() as conn:
        cursor = conn.cursor()

        def fetch_columns(object_name: str) -> set[str]:
            return {
                str(row.name)
                for row in cursor.execute(
                    """
                    SELECT name
                    FROM sys.columns
                    WHERE object_id = OBJECT_ID(?);
                    """,
                    object_name,
                ).fetchall()
            }

        draw_columns = fetch_columns("dbo.DrawHistory")
        ledger_columns = fetch_columns("dbo.PredictionLedger")

        verify_parameters = [
            str(row.name)
            for row in cursor.execute(
                """
                SELECT name
                FROM sys.parameters
                WHERE object_id = OBJECT_ID('dbo.SP_Verify_Predictions')
                ORDER BY parameter_id;
                """
            ).fetchall()
        ]
        summary_parameters = [
            str(row.name)
            for row in cursor.execute(
                """
                SELECT name
                FROM sys.parameters
                WHERE object_id = OBJECT_ID('dbo.SP_Summarize_EngineLedger')
                ORDER BY parameter_id;
                """
            ).fetchall()
        ]
        integrity_parameters = [
            str(row.name)
            for row in cursor.execute(
                """
                SELECT name
                FROM sys.parameters
                WHERE object_id = OBJECT_ID('dbo.SP_Audit_PredictionLedgerIntegrity')
                ORDER BY parameter_id;
                """
            ).fetchall()
        ]
        constraint_row = cursor.execute(
            """
            SELECT definition
            FROM sys.check_constraints
            WHERE name = 'CK_PredictionLedger_Mode'
              AND parent_object_id = OBJECT_ID('dbo.PredictionLedger');
            """
        ).fetchone()
        index_row = cursor.execute(
            """
            SELECT name
            FROM sys.indexes
            WHERE name = 'IX_DrawHistory_TemporalContext'
              AND object_id = OBJECT_ID('dbo.DrawHistory');
            """
        ).fetchone()

    constraint_definition = str(constraint_row.definition) if constraint_row else ""
    schema_ok = (
        required_draw_columns <= draw_columns
        and required_ledger_columns <= ledger_columns
        and verify_parameters == ["@TargetDrawNo", "@Top5Predictions"]
        and summary_parameters == ["@Mode", "@EngineSource", "@SourceStart", "@SourceEnd"]
        and integrity_parameters == ["@Mode", "@EngineSource", "@SourceStart", "@SourceEnd"]
        and all(mode in constraint_definition for mode in required_modes)
        and index_row is not None
    )

    print("=" * 96)
    print("STEP 144 — PREFLIGHT AUDIT")
    print("=" * 96)
    print(f"SQL_FIREWALL_AUDIT: {'PASS' if sql_firewall_ok else 'FAIL'}")
    print(f"LEAKAGE_AUDIT: {'PASS' if leakage_ok else 'FAIL'}")
    print(f"SCHEMA_COMPATIBILITY_AUDIT: {'PASS' if schema_ok else 'FAIL'}")
    print("=" * 96)

    if not sql_firewall_ok:
        raise RuntimeError("SQL firewall audit failed; loop execution stopped")
    if not leakage_ok:
        raise RuntimeError("Leakage audit failed; loop execution stopped")
    if not schema_ok:
        raise RuntimeError("Schema compatibility audit failed; loop execution stopped")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run temporal context historical simulation.")
    parser.add_argument("--start-source", type=int, default=DEFAULT_START_SOURCE_DRAW_NO)
    parser.add_argument("--end-source", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_preflight_audits()

    latest_draw_no = fetch_latest_draw_no()
    source_start = int(args.start_source)
    source_end = int(args.end_source) if args.end_source is not None else latest_draw_no - 1
    progress_every = int(args.progress_every)

    if source_start < 1:
        raise ValueError("source_start must be >= 1")
    if source_end >= latest_draw_no:
        raise ValueError(f"source_end must be <= {latest_draw_no - 1}, got {source_end}")
    if source_start > source_end:
        raise ValueError(f"Invalid source range: {source_start}..{source_end}")
    if progress_every < 1:
        raise ValueError("progress_every must be >= 1")

    total = source_end - source_start + 1

    print("=" * 96)
    print("STEP 144 — TEMPORAL GLOBAL LOOP")
    print("=" * 96)
    print(f"Latest historical DrawNo: {latest_draw_no}")
    print(f"Mode: {MODE}")
    print(f"EngineSource: {ENGINE_SOURCE}")
    print(f"Underlying engines: {', '.join(UNDERLYING_ENGINES)}")
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
            already_verified, existing_hit_count = fetch_verified_group(
                cursor,
                source_draw_no=source_draw_no,
                target_draw_no=target_draw_no,
            )

            if already_verified:
                skipped_count += 1
                if index == 1 or index == total or index % progress_every == 0:
                    print(
                        f"[{index:>5}/{total:<5}] {source_draw_no}->{target_draw_no} "
                        f"SKIP already_verified hits={existing_hit_count}"
                    )
                continue

            result = run_existing_engine_prediction(
                PredictionRequest(draw_number=source_draw_no, mode="Historical")
            )
            grouped = extract_underlying_candidates(result.ledger_predictions)

            target_metadata = fetch_target_temporal_metadata(
                cursor,
                target_draw_no=target_draw_no,
            )
            observations = fetch_historical_temporal_cluster(
                cursor,
                source_draw_no=source_draw_no,
                target_metadata=target_metadata,
            )
            candidates = build_temporal_candidates(
                observations=observations,
                grouped=grouped,
            )

            record_predictions_to_ledger(
                mode=MODE,
                source_draw_no=source_draw_no,
                target_draw_no=target_draw_no,
                day_type=target_metadata.day_type,
                predictions=candidates,
            )

            top5 = [str(item.number).zfill(4) for item in candidates]
            hit_count = call_sql_firewall_verify(
                cursor,
                target_draw_no=target_draw_no,
                predictions=top5,
            )
            affected = update_temporal_verification(
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

            if index == 1 or index == total or index % progress_every == 0:
                print(
                    f"[{index:>5}/{total:<5}] {source_draw_no}->{target_draw_no} "
                    f"day={target_metadata.day_type} "
                    f"cluster_numbers={len(observations)} "
                    f"hits={hit_count} top5={','.join(top5)}"
                )

    print("-" * 96)
    print(f"Processed this run: {processed_count}")
    print(f"Skipped verified:    {skipped_count}")
    print("-" * 96)

    print_summary(fetch_summary(source_start, source_end))
    print_audit_commands(source_start, source_end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
