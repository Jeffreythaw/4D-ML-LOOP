from __future__ import annotations

import itertools
from datetime import date, datetime, timedelta
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from app.core.config import get_settings
from app.schemas.prediction import PredictionCandidate


ENGINE_SOURCE = "E1_TEMPORAL_CONTEXT_MATCH"
FALLBACK_ENGINE_SOURCE = "E1_DELTA_ROTATION_LSTS"

REQUIRED_UNDERLYING_ENGINES = (
    "E1_CROSS_PAIR_LINEAR",
    "E1_WLS_DECAY_0.98",
    "E1_MIRROR_BASE5_LSTS",
    "E1_DELTA_ROTATION_LSTS",
)
OPTIONAL_UNDERLYING_ENGINES = (
    "E40_FULL_HISTORY_KNOWLEDGE",
)
UNDERLYING_ENGINES = REQUIRED_UNDERLYING_ENGINES + OPTIONAL_UNDERLYING_ENGINES

TOP_K = 5
TEMPORAL_COLD_START_MIN_MATCHES = 5

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

SOURCE_METADATA_SQL = """
    SELECT
        DrawNo,
        DrawDate,
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
    day_of_month: int
    month: int
    year: int
    weekday: int
    week_of_year: int
    day_type: str


@dataclass(frozen=True)
class TemporalContextResult:
    source_draw_number: int
    target_draw_number: int
    day_type: str
    predictions: list[PredictionCandidate]
    observations_count: int


def run_temporal_context_prediction(
    *,
    source_draw_no: int,
    target_draw_no: int,
    underlying_candidates: Iterable[PredictionCandidate],
) -> TemporalContextResult:
    """
    Production temporal-context master engine for live Current mode.

    Firewall invariant:
      - Target query reads only DrawDate/datepart metadata.
      - Historical winner-bearing query is constrained to DrawNo <= source_draw_no.
      - No target winner fields are selected or loaded.
    """
    try:
        import pyodbc  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pyodbc is not installed in this backend environment.") from exc

    settings = get_settings()

    with pyodbc.connect(settings.sql_connection_string(), timeout=15) as connection:
        cursor = connection.cursor()
        metadata = fetch_target_temporal_metadata(cursor, source_draw_no=source_draw_no, target_draw_no=target_draw_no)
        observations = fetch_historical_temporal_cluster(
            cursor,
            source_draw_no=source_draw_no,
            target_metadata=metadata,
        )

    grouped = extract_underlying_candidates(underlying_candidates)
    predictions = build_temporal_candidates(observations=observations, grouped=grouped)

    return TemporalContextResult(
        source_draw_number=int(source_draw_no),
        target_draw_number=int(target_draw_no),
        day_type=metadata.day_type,
        predictions=predictions,
        observations_count=len(observations),
    )


def fetch_target_temporal_metadata(
    cursor,
    *,
    source_draw_no: int,
    target_draw_no: int,
) -> TemporalMetadata:
    """
    Metadata-only target read.

    If the target draw already exists historically, read only DrawDate/datepart
    metadata. If it does not exist yet for live Current mode, infer the next
    scheduled draw date from the source DrawDate. No target winner fields are
    selected or loaded.
    """
    row = cursor.execute(TARGET_METADATA_SQL, int(target_draw_no)).fetchone()

    if row is not None:
        return TemporalMetadata(
            day_of_month=int(row.DayOfMonth),
            month=int(row.MonthNo),
            year=int(row.YearNo),
            weekday=int(row.WeekdayNo),
            week_of_year=int(row.WeekOfYear),
            day_type=str(row.DayType),
        )

    source_row = cursor.execute(SOURCE_METADATA_SQL, int(source_draw_no)).fetchone()
    if source_row is None:
        raise RuntimeError(f"Source DrawNo {source_draw_no} not found in dbo.DrawHistory")

    source_date = _coerce_date(source_row.DrawDate)
    inferred_date = _infer_next_draw_date(source_date)
    day_type = _day_type_from_date(inferred_date)

    return TemporalMetadata(
        day_of_month=int(inferred_date.day),
        month=int(inferred_date.month),
        year=int(inferred_date.year),
        weekday=_sqlserver_weekday_number(inferred_date),
        week_of_year=int(inferred_date.strftime("%U")) + 1,
        day_type=day_type,
    )


def _coerce_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)).date()


def _infer_next_draw_date(source_date: date) -> date:
    # Python weekday: Monday=0 ... Sunday=6.
    # Draw schedule: Wednesday=2, Saturday=5, Sunday=6.
    draw_weekdays = (2, 5, 6)
    for offset in range(1, 8):
        candidate = source_date + timedelta(days=offset)
        if candidate.weekday() in draw_weekdays:
            return candidate
    raise RuntimeError(f"Unable to infer next draw date after {source_date}")


def _day_type_from_date(value: date) -> str:
    names = {
        2: "Wednesday",
        5: "Saturday",
        6: "Sunday",
    }
    return names.get(value.weekday(), "Special")


def _sqlserver_weekday_number(value: date) -> int:
    # SQL Server DATEPART(weekday) default us_english DATEFIRST=7:
    # Sunday=1, Monday=2, ... Saturday=7.
    return ((value.weekday() + 1) % 7) + 1


def fetch_historical_temporal_cluster(
    cursor,
    *,
    source_draw_no: int,
    target_metadata: TemporalMetadata,
) -> list[tuple[str, float]]:
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
) -> dict[str, list[PredictionCandidate]]:
    grouped: dict[str, list[PredictionCandidate]] = defaultdict(list)

    for item in ledger_predictions:
        source = str(item.source)
        if source in UNDERLYING_ENGINES:
            grouped[source].append(item)

    result: dict[str, list[PredictionCandidate]] = {}

    for engine in REQUIRED_UNDERLYING_ENGINES:
        items = sorted(grouped.get(engine, []), key=lambda item: int(item.rank))
        if len(items) != TOP_K:
            raise RuntimeError(f"{engine} expected {TOP_K} rows, got {len(items)}")
        result[engine] = items
    for engine in OPTIONAL_UNDERLYING_ENGINES:
        items = sorted(grouped.get(engine, []), key=lambda item: int(item.rank))
        if items and len(items) != TOP_K:
            raise RuntimeError(f"{engine} expected {TOP_K} rows when present, got {len(items)}")
        if items:
            result[engine] = items

    return result


def build_temporal_candidates(
    *,
    observations: list[tuple[str, float]],
    grouped: dict[str, list[PredictionCandidate]],
) -> list[PredictionCandidate]:
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
        for item in grouped.get(engine, []):
            rank_no = int(item.rank)
            number = str(item.number).zfill(4)
            borda_by_number[number] += TOP_K - rank_no + 1
            best_engine_rank[number] = min(best_engine_rank.get(number, TOP_K + 1), rank_no)
            candidate_numbers.add(number)

            if number not in first_seen_by_number:
                first_seen_by_number[number] = first_seen_index
                first_seen_index += 1

    if observation_weight_total <= 0:
        raise RuntimeError("Temporal observation weight total is zero")

    def position_score(number: str) -> float:
        return sum(
            position_frequency[position][digit] / observation_weight_total
            for position, digit in enumerate(number)
        )

    scores: dict[str, float] = {}

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
        for item in grouped.get(FALLBACK_ENGINE_SOURCE, []):
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
