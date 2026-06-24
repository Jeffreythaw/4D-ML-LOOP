from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
REPORT_PATH = PROJECT_ROOT / "reports" / "patches" / "e5_draw_5498_segment_attribution_prototype.txt"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))
load_dotenv(BACKEND_ROOT / ".env")

from app.core.config import get_settings  # noqa: E402
from j4d_e5.analyzer import analyze_completed_draw_segments  # noqa: E402
from j4d_e5.provenance import normalize_4d  # noqa: E402


TARGET_DRAW_NO = 5498
PREDICTED_TOP5 = ("4445", "4640", "9917", "9373", "5335")

# Post-result fallback only. These values are for completed Draw 5498 segment
# analysis when SQL access is unavailable; never use this script for prediction.
MANUAL_COMPLETED_DRAW_5498_ACTUALS = (
    "9954",
    "2614",
    "6272",
    "0324",
    "0327",
    "1364",
    "1835",
    "3726",
    "3800",
    "5816",
    "6608",
    "6989",
    "9564",
    "0062",
    "0219",
    "0445",
    "4693",
    "6118",
    "6424",
    "7552",
    "8286",
    "8663",
    "8916",
)

HIGH_CONFIDENCE_CLASSES = {
    "EXACT4_MATCH",
    "LAST3_MATCH",
    "PREFIX3_MATCH",
    "SAME_POSITION_3",
}
MEDIUM_CONFIDENCE_CLASSES = {
    "PREFIX2_MATCH",
    "SUFFIX2_MATCH",
    "MIDDLE2_MATCH",
    "SAME_POSITION_2",
    "PAIR_13_MATCH",
    "PAIR_14_MATCH",
    "PAIR_24_MATCH",
}
LOW_CONFIDENCE_CLASSES = {
    "DIGIT_BAG_3_MATCH",
    "DIGIT_BAG_2_MATCH",
}


def _row_value(row: Any, column_names: tuple[str, ...]) -> Any | None:
    for name in column_names:
        if hasattr(row, name):
            value = getattr(row, name)
            if value is not None:
                return value
    return None


def _numbers_from_rows(rows: list[Any]) -> tuple[str, ...]:
    candidates: list[str] = []
    candidate_columns = (
        "NumberText",
        "PrizeNumber",
        "WinningNumber",
        "WinningNo",
        "ResultNumber",
        "ResultNo",
        "Number",
        "FourDNumber",
        "Value",
    )
    for row in rows:
        value = _row_value(row, candidate_columns)
        if value is None:
            continue
        try:
            number = normalize_4d(value, field_name="actual_number")
        except ValueError:
            continue
        if number not in candidates:
            candidates.append(number)
    return tuple(candidates)


def load_actuals_from_sql() -> tuple[tuple[str, ...] | None, str]:
    try:
        import pyodbc  # type: ignore
    except ImportError:
        return None, "pyodbc_not_installed"

    queries = (
        """
        SELECT TOP (23)
            [NumberText],
            [PrizeType],
            [PrizeRank]
        FROM [LS].[dbo].[J4D_Result]
        WHERE [DrawNo] = ?
        ORDER BY
            CASE [PrizeType]
                WHEN '1st' THEN 1
                WHEN '2nd' THEN 2
                WHEN '3rd' THEN 3
                WHEN 'Starter' THEN 4
                WHEN 'Consolation' THEN 5
                ELSE 9
            END,
            [PrizeRank];
        """,
        """
        SELECT TOP (23)
            [NumberText],
            [PrizeType],
            [PrizeRank]
        FROM dbo.J4D_Result
        WHERE [DrawNo] = ?
        ORDER BY
            CASE [PrizeType]
                WHEN '1st' THEN 1
                WHEN '2nd' THEN 2
                WHEN '3rd' THEN 3
                WHEN 'Starter' THEN 4
                WHEN 'Consolation' THEN 5
                ELSE 9
            END,
            [PrizeRank];
        """,
    )

    try:
        with pyodbc.connect(get_settings().sql_connection_string(), timeout=30) as connection:
            cursor = connection.cursor()
            for query in queries:
                try:
                    rows = list(cursor.execute(query, TARGET_DRAW_NO).fetchall())
                except Exception:
                    continue
                numbers = _numbers_from_rows(rows)
                if numbers:
                    return numbers, "sql_j4d_result"
    except Exception as exc:
        return None, f"sql_unavailable:{type(exc).__name__}"

    return None, "sql_no_rows"


def _row_tier(classes: set[str]) -> str | None:
    if classes & HIGH_CONFIDENCE_CLASSES:
        return "high"
    if classes & MEDIUM_CONFIDENCE_CLASSES:
        return "medium"
    if classes & LOW_CONFIDENCE_CLASSES:
        return "low"
    return None


def build_report(actuals: tuple[str, ...], actual_source: str) -> str:
    result = analyze_completed_draw_segments(
        target_draw_no=TARGET_DRAW_NO,
        predicted_candidates=PREDICTED_TOP5,
        actual_numbers=actuals,
        provenance_rows=None,
        no_write=True,
    )

    pair_classes: dict[tuple[str, str], set[str]] = {}
    for row in result.attribution_rows:
        if row.is_exact:
            continue
        pair_classes.setdefault((row.candidate_number, row.actual_number), set()).add(row.segment_class)

    high_rows: list[tuple[str, str, list[str]]] = []
    medium_rows: list[tuple[str, str, list[str]]] = []
    low_rows: list[tuple[str, str, list[str]]] = []

    for (candidate, actual), classes in sorted(pair_classes.items()):
        tier = _row_tier(classes)
        item = (candidate, actual, sorted(classes))
        if tier == "high":
            high_rows.append(item)
        elif tier == "medium":
            medium_rows.append(item)
        elif tier == "low":
            low_rows.append(item)

    lines = [
        "E5 Draw 5498 Segment Attribution Prototype",
        f"TargetDrawNo: {TARGET_DRAW_NO}",
        f"PredictedTop5: {', '.join(PREDICTED_TOP5)}",
        f"ActualSource: {actual_source}",
        f"ActualCount: {len(actuals)}",
        f"ExactHitCount: {result.exact_hit_count}",
        f"ProvenanceAvailable: {result.provenance_available}",
        "ProvenanceStatus: PROVENANCE_MISSING_NEEDS_IMPLEMENTATION",
        "UseForFuturePrediction: false",
        "ActualNumbersRedacted: true",
        "",
        "HighConfidenceSegmentRows:",
    ]

    if high_rows:
        for candidate, actual, classes in high_rows:
            lines.append(f"{candidate} -> {actual}: {', '.join(classes)}")
    else:
        lines.append("(none)")

    lines.extend(["", "MediumConfidenceSegmentRows:"])
    if medium_rows:
        for candidate, actual, classes in medium_rows:
            lines.append(f"{candidate} -> {actual}: {', '.join(classes)}")
    else:
        lines.append("(none)")

    lines.extend(["", "LowConfidenceDigitBagRows:"])
    if low_rows:
        for candidate, actual, classes in low_rows:
            lines.append(f"{candidate} -> {actual}: {', '.join(classes)}")
    else:
        lines.append("(none)")

    return "\n".join(lines) + "\n"


def main() -> int:
    sql_actuals, source = load_actuals_from_sql()
    actuals = sql_actuals or MANUAL_COMPLETED_DRAW_5498_ACTUALS
    actual_source = source if sql_actuals else f"manual_completed_draw_fallback:{source}"
    report = build_report(actuals, actual_source)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"WROTE {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
