from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pyodbc
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

load_dotenv(BACKEND_ROOT / ".env")

from app.core.config import get_settings


ENGINE = "E1_TEMPORAL_CONTEXT_MATCH"
THRESHOLDS = [10, 30, 50, 100]
SOURCE_START = 5450
SOURCE_END = 5496
RECENT_SUPPORT_WINDOW = 365

REPORT_PATH = PROJECT_ROOT / "reports" / "step_150c_temporal_bucket_audit.txt"
JSONL_PATH = PROJECT_ROOT / "reports" / "step_150c_temporal_bucket_audit_rows.jsonl"

MODE_PRIORITY = {
    "Current": 0,
    "Temporal_Global_Loop": 1,
    "Historical": 2,
    "Engine_Grand_Loop": 3,
    "Grand_Loop": 4,
    "Weighted_Grand_Loop": 5,
}


@dataclass(frozen=True)
class DrawRecord:
    draw_no: int
    draw_date: str | None
    day_type: str
    sql_weekday: str | None
    month: int | None
    year: int | None
    winners: list[str]


@dataclass(frozen=True)
class PredictionCase:
    source: int
    target: int
    target_day_type: str
    target_weekday: str | None
    target_month: int | None
    target_year: int | None
    baseline_mode: str
    baseline_top5: list[str]
    actual: list[str]


def get_conn():
    settings = get_settings()
    return pyodbc.connect(settings.sql_connection_string(), timeout=120)


def z4(value: str | int) -> str:
    return str(value).strip().zfill(4)


def parse_winners(value: str | None) -> list[str]:
    if not value:
        return []
    return [z4(part) for part in str(value).replace(" ", "").split(",") if part.strip()]


def digits(number: str) -> list[int]:
    return [int(ch) for ch in z4(number)]


def digit_sum(number: str) -> int:
    return sum(digits(number))


def circular_digit_distance(a: int, b: int) -> int:
    diff = abs(a - b)
    return min(diff, 10 - diff)


def circular_distance(a: str, b: str) -> int:
    return sum(circular_digit_distance(x, y) for x, y in zip(digits(a), digits(b)))


def hamming_distance(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(z4(a), z4(b)))


def mirror_signature(number: str) -> str:
    return "".join(str(int(ch) % 5) for ch in z4(number))


def box_overlap(a: str, b: str) -> int:
    ca = Counter(z4(a))
    cb = Counter(z4(b))
    return sum((ca & cb).values())


def runtime_bucket(day_type: str) -> str:
    normalized = str(day_type or "Unknown").strip()
    if normalized in {"Saturday", "Sunday"}:
        return "WEEKEND_SPACE"
    if normalized in {"Wednesday", "Special"}:
        return "MIDWEEK_SPECIAL_SPACE"
    return f"OTHER_SPACE::{normalized}"


def hit_count(predicted: Iterable[str], actual: Iterable[str]) -> int:
    return len(set(predicted).intersection(set(actual)))


def best_distance_metrics(predicted: list[str], actual: list[str]) -> dict:
    if not predicted or not actual:
        return {
            "min_circular_distance": None,
            "min_hamming_distance": None,
            "max_box_overlap": None,
            "mirror_exact_count": 0,
            "hamming_le1_count": 0,
        }

    min_circ = 999
    min_hamming = 999
    max_box = 0
    mirror_exact = 0
    hamming_le1 = 0

    for p in predicted:
        for a in actual:
            circ = circular_distance(p, a)
            ham = hamming_distance(p, a)
            box = box_overlap(p, a)

            min_circ = min(min_circ, circ)
            min_hamming = min(min_hamming, ham)
            max_box = max(max_box, box)

            if mirror_signature(p) == mirror_signature(a) and p != a:
                mirror_exact += 1

            if ham <= 1 and p != a:
                hamming_le1 += 1

    return {
        "min_circular_distance": min_circ,
        "min_hamming_distance": min_hamming,
        "max_box_overlap": max_box,
        "mirror_exact_count": mirror_exact,
        "hamming_le1_count": hamming_le1,
    }


def fetch_draws(cursor) -> dict[int, DrawRecord]:
    rows = cursor.execute("""
        SELECT
            DrawNo,
            CONVERT(varchar(10), DrawDate, 120) AS DrawDateText,
            DayType,
            DATENAME(WEEKDAY, DrawDate) AS SqlWeekday,
            DATEPART(MONTH, DrawDate) AS DrawMonth,
            DATEPART(YEAR, DrawDate) AS DrawYear,
            WinningNumbers
        FROM dbo.DrawHistory
        WHERE WinningNumbers IS NOT NULL
        ORDER BY DrawNo;
    """).fetchall()

    draws: dict[int, DrawRecord] = {}

    for row in rows:
        draws[int(row.DrawNo)] = DrawRecord(
            draw_no=int(row.DrawNo),
            draw_date=str(row.DrawDateText) if row.DrawDateText else None,
            day_type=str(row.DayType) if row.DayType else "Unknown",
            sql_weekday=str(row.SqlWeekday) if row.SqlWeekday else None,
            month=int(row.DrawMonth) if row.DrawMonth else None,
            year=int(row.DrawYear) if row.DrawYear else None,
            winners=parse_winners(row.WinningNumbers),
        )

    return draws


def fetch_baseline_predictions(cursor) -> dict[tuple[int, int], tuple[str, list[str]]]:
    rows = cursor.execute("""
        SELECT
            Mode,
            SourceDrawNo,
            TargetDrawNo,
            RankNo,
            PredictedNumber
        FROM dbo.PredictionLedger
        WHERE EngineSource = ?
          AND RankNo BETWEEN 1 AND 5
        ORDER BY SourceDrawNo, TargetDrawNo, Mode, RankNo;
    """, ENGINE).fetchall()

    grouped: dict[tuple[int, int, str], list[tuple[int, str]]] = defaultdict(list)

    for row in rows:
        grouped[(int(row.SourceDrawNo), int(row.TargetDrawNo), str(row.Mode))].append(
            (int(row.RankNo), z4(row.PredictedNumber))
        )

    selected: dict[tuple[int, int], tuple[str, list[str]]] = {}

    for (source, target, mode), items in grouped.items():
        ranked = [number for _, number in sorted(items)]
        if len(ranked) != 5:
            continue

        pair = (source, target)
        if pair not in selected:
            selected[pair] = (mode, ranked)
            continue

        old_mode, _ = selected[pair]
        if MODE_PRIORITY.get(mode, 99) < MODE_PRIORITY.get(old_mode, 99):
            selected[pair] = (mode, ranked)

    return selected


def build_cases(
    draws: dict[int, DrawRecord],
    baseline_predictions: dict[tuple[int, int], tuple[str, list[str]]],
) -> list[PredictionCase]:
    cases: list[PredictionCase] = []

    for source in range(SOURCE_START, SOURCE_END + 1):
        target = source + 1
        target_draw = draws.get(target)
        baseline = baseline_predictions.get((source, target))

        if not target_draw or not target_draw.winners or not baseline:
            continue

        mode, top5 = baseline

        cases.append(
            PredictionCase(
                source=source,
                target=target,
                target_day_type=target_draw.day_type,
                target_weekday=target_draw.sql_weekday,
                target_month=target_draw.month,
                target_year=target_draw.year,
                baseline_mode=mode,
                baseline_top5=top5,
                actual=target_draw.winners,
            )
        )

    return cases


def historical_support_exact(
    draws: dict[int, DrawRecord],
    source: int,
    day_type: str,
    recent_window: int | None = None,
) -> int:
    min_draw = -10**9 if recent_window is None else source - recent_window + 1
    return sum(
        1
        for draw in draws.values()
        if min_draw <= draw.draw_no <= source and draw.day_type == day_type and draw.winners
    )


def historical_support_bucket(
    draws: dict[int, DrawRecord],
    source: int,
    bucket: str,
    recent_window: int | None = None,
) -> int:
    min_draw = -10**9 if recent_window is None else source - recent_window + 1
    return sum(
        1
        for draw in draws.values()
        if min_draw <= draw.draw_no <= source and runtime_bucket(draw.day_type) == bucket and draw.winners
    )


def score_draw_contribution(source: int, target_case: PredictionCase, hist_draw: DrawRecord) -> float:
    distance = max(1, source - hist_draw.draw_no)
    recency = 1.0 / (distance ** 0.35)

    month_bonus = 1.20 if target_case.target_month and hist_draw.month == target_case.target_month else 1.0
    weekday_bonus = 1.10 if target_case.target_weekday and hist_draw.sql_weekday == target_case.target_weekday else 1.0

    return recency * month_bonus * weekday_bonus


def generate_bucket_candidates(
    draws: dict[int, DrawRecord],
    case: PredictionCase,
    threshold: int,
    mode: str,
    limit: int = 100,
) -> tuple[list[str], dict]:
    """
    mode:
      EXACT = exact DayType only
      NORMALIZED = support guard; if exact support < threshold, fallback to runtime bucket
    """

    exact_support = historical_support_exact(draws, case.source, case.target_day_type, recent_window=None)
    recent_exact_support = historical_support_exact(
        draws,
        case.source,
        case.target_day_type,
        recent_window=RECENT_SUPPORT_WINDOW,
    )
    target_bucket = runtime_bucket(case.target_day_type)
    bucket_support = historical_support_bucket(draws, case.source, target_bucket, recent_window=None)
    recent_bucket_support = historical_support_bucket(
        draws,
        case.source,
        target_bucket,
        recent_window=RECENT_SUPPORT_WINDOW,
    )

    if mode == "EXACT":
        selected_scope = "EXACT_DAYTYPE"
        selected_value = case.target_day_type
    elif mode == "NORMALIZED":
        if recent_exact_support < threshold:
            selected_scope = "RUNTIME_BUCKET"
            selected_value = target_bucket
        else:
            selected_scope = "EXACT_DAYTYPE"
            selected_value = case.target_day_type
    else:
        raise ValueError(f"Unknown mode: {mode}")

    scores: Counter[str] = Counter()

    for draw in draws.values():
        if draw.draw_no > case.source:
            continue
        if not draw.winners:
            continue

        if selected_scope == "EXACT_DAYTYPE":
            if draw.day_type != selected_value:
                continue
        else:
            if runtime_bucket(draw.day_type) != selected_value:
                continue

        contribution = score_draw_contribution(case.source, case, draw)

        for rank, number in enumerate(draw.winners, start=1):
            rank_weight = 1.0 / (rank ** 0.15)
            sum_alignment = 1.05 if abs(digit_sum(number) - 18) <= 5 else 1.0
            scores[number] += contribution * rank_weight * sum_alignment

    ranked = [number for number, _ in scores.most_common(limit)]

    metadata = {
        "exact_support": exact_support,
        "recent_exact_support": recent_exact_support,
        "bucket": target_bucket,
        "bucket_support": bucket_support,
        "recent_bucket_support": recent_bucket_support,
        "selected_scope": selected_scope,
        "selected_value": selected_value,
        "candidate_count": len(scores),
    }

    return ranked, metadata


def summarize_rows(rows: list[dict], label: str) -> list[str]:
    lines = []
    total = len(rows)

    baseline_hit_draws = sum(1 for r in rows if r["baseline_hit_count"] > 0)
    exact_hit_draws = sum(1 for r in rows if r[f"{label}_top5_hit_count"] > 0)
    top100_coverage = sum(1 for r in rows if r[f"{label}_top100_hit_count"] > 0)

    baseline_raw_hits = sum(r["baseline_hit_count"] for r in rows)
    exact_raw_hits = sum(r[f"{label}_top5_hit_count"] for r in rows)
    top100_raw_hits = sum(r[f"{label}_top100_hit_count"] for r in rows)

    lines.append(f"{label}:")
    lines.append(f"  RowsChecked: {total}")
    lines.append(f"  BaselineTop5HitDraws: {baseline_hit_draws}")
    lines.append(f"  {label}Top5HitDraws: {exact_hit_draws}")
    lines.append(f"  {label}Top100CoverageDraws: {top100_coverage}")
    lines.append(f"  BaselineRawHits: {baseline_raw_hits}")
    lines.append(f"  {label}Top5RawHits: {exact_raw_hits}")
    lines.append(f"  {label}Top100RawHits: {top100_raw_hits}")

    if total:
        lines.append(f"  BaselineTop5HitRate: {baseline_hit_draws / total * 100:.4f}%")
        lines.append(f"  {label}Top5HitRate: {exact_hit_draws / total * 100:.4f}%")
        lines.append(f"  {label}Top100CoverageRate: {top100_coverage / total * 100:.4f}%")

    return lines


def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with get_conn() as conn:
        cursor = conn.cursor()
        draws = fetch_draws(cursor)
        baseline_predictions = fetch_baseline_predictions(cursor)

    cases = build_cases(draws, baseline_predictions)

    rows: list[dict] = []

    for case in cases:
        baseline_metrics = best_distance_metrics(case.baseline_top5, case.actual)

        for threshold in THRESHOLDS:
            exact_pool, exact_meta = generate_bucket_candidates(
                draws=draws,
                case=case,
                threshold=threshold,
                mode="EXACT",
                limit=100,
            )
            normalized_pool, normalized_meta = generate_bucket_candidates(
                draws=draws,
                case=case,
                threshold=threshold,
                mode="NORMALIZED",
                limit=100,
            )

            exact_top5 = exact_pool[:5]
            normalized_top5 = normalized_pool[:5]

            exact_top5_metrics = best_distance_metrics(exact_top5, case.actual)
            normalized_top5_metrics = best_distance_metrics(normalized_top5, case.actual)

            row = {
                "source": case.source,
                "target": case.target,
                "target_day_type": case.target_day_type,
                "target_weekday": case.target_weekday,
                "target_month": case.target_month,
                "threshold": threshold,
                "baseline_mode": case.baseline_mode,
                "baseline_top5": case.baseline_top5,
                "baseline_hit_count": hit_count(case.baseline_top5, case.actual),
                "baseline_min_circular_distance": baseline_metrics["min_circular_distance"],
                "baseline_min_hamming_distance": baseline_metrics["min_hamming_distance"],
                "baseline_max_box_overlap": baseline_metrics["max_box_overlap"],
                "exact_support": exact_meta["exact_support"],
                "recent_exact_support": exact_meta["recent_exact_support"],
                "bucket": exact_meta["bucket"],
                "bucket_support": exact_meta["bucket_support"],
                "recent_bucket_support": exact_meta["recent_bucket_support"],
                "exact_selected_scope": exact_meta["selected_scope"],
                "exact_candidate_count": exact_meta["candidate_count"],
                "exact_top5": exact_top5,
                "exact_top5_hit_count": hit_count(exact_top5, case.actual),
                "exact_top10_hit_count": hit_count(exact_pool[:10], case.actual),
                "exact_top25_hit_count": hit_count(exact_pool[:25], case.actual),
                "exact_top50_hit_count": hit_count(exact_pool[:50], case.actual),
                "exact_top100_hit_count": hit_count(exact_pool[:100], case.actual),
                "exact_min_circular_distance": exact_top5_metrics["min_circular_distance"],
                "exact_min_hamming_distance": exact_top5_metrics["min_hamming_distance"],
                "exact_max_box_overlap": exact_top5_metrics["max_box_overlap"],
                "normalized_selected_scope": normalized_meta["selected_scope"],
                "normalized_selected_value": normalized_meta["selected_value"],
                "normalized_candidate_count": normalized_meta["candidate_count"],
                "normalized_top5": normalized_top5,
                "normalized_top5_hit_count": hit_count(normalized_top5, case.actual),
                "normalized_top10_hit_count": hit_count(normalized_pool[:10], case.actual),
                "normalized_top25_hit_count": hit_count(normalized_pool[:25], case.actual),
                "normalized_top50_hit_count": hit_count(normalized_pool[:50], case.actual),
                "normalized_top100_hit_count": hit_count(normalized_pool[:100], case.actual),
                "normalized_min_circular_distance": normalized_top5_metrics["min_circular_distance"],
                "normalized_min_hamming_distance": normalized_top5_metrics["min_hamming_distance"],
                "normalized_max_box_overlap": normalized_top5_metrics["max_box_overlap"],
            }
            rows.append(row)

    with JSONL_PATH.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    lines: list[str] = []
    lines.append("=" * 112)
    lines.append("STEP 150C — TEMPORAL BUCKET NORMALIZATION & SUPPORT GUARD AUDIT")
    lines.append("=" * 112)
    lines.append("Mode: REPORT ONLY")
    lines.append("ProductionMathChanged: NO")
    lines.append("SQLSchemaChanged: NO")
    lines.append("FrontendChanged: NO")
    lines.append(f"RecentWindow: {SOURCE_START}->{SOURCE_END + 1}")
    lines.append(f"RecentSupportWindow: {RECENT_SUPPORT_WINDOW} draws")
    lines.append(f"CasesLoaded: {len(cases)}")
    lines.append("RuntimeMapping:")
    lines.append("  Saturday/Sunday   -> WEEKEND_SPACE")
    lines.append("  Wednesday/Special -> MIDWEEK_SPECIAL_SPACE")
    lines.append("")

    lines.append("DAYTYPE SUPPORT IN FULL DRAW HISTORY")
    lines.append("-" * 112)
    support = Counter(draw.day_type for draw in draws.values() if SOURCE_START <= draw.draw_no <= SOURCE_END + 1)
    for day, count in support.most_common():
        lines.append(f"  {day:<12} Rows={count}")
    lines.append("")

    for threshold in THRESHOLDS:
        subset = [r for r in rows if r["threshold"] == threshold]
        lines.append("=" * 112)
        lines.append(f"THRESHOLD = {threshold}")
        lines.append("=" * 112)

        lines.extend(summarize_rows(subset, "exact"))
        lines.append("")
        lines.extend(summarize_rows(subset, "normalized"))
        lines.append("")

        fallback_count = sum(1 for r in subset if r["normalized_selected_scope"] == "RUNTIME_BUCKET")
        lines.append(f"  NormalizedFallbackCount: {fallback_count} / {len(subset)}")
        lines.append("")

        by_day = defaultdict(list)
        for r in subset:
            by_day[r["target_day_type"]].append(r)

        lines.append("  DAYTYPE BREAKDOWN")
        for day_type, items in sorted(by_day.items()):
            n = len(items)
            b = sum(1 for r in items if r["baseline_hit_count"] > 0)
            e = sum(1 for r in items if r["exact_top5_hit_count"] > 0)
            norm = sum(1 for r in items if r["normalized_top5_hit_count"] > 0)
            norm100 = sum(1 for r in items if r["normalized_top100_hit_count"] > 0)
            fb = sum(1 for r in items if r["normalized_selected_scope"] == "RUNTIME_BUCKET")
            avg_support = sum(int(r["exact_support"]) for r in items) / n if n else 0
            avg_recent_support = sum(int(r["recent_exact_support"]) for r in items) / n if n else 0
            lines.append(
                f"    {day_type:<12} Rows={n:<3} "
                f"AvgExactSupport={avg_support:7.2f} "
                f"AvgRecentSupport={avg_recent_support:7.2f} "
                f"Fallbacks={fb:<3} "
                f"BaselineHitDraws={b:<2} ExactHitDraws={e:<2} "
                f"NormHitDraws={norm:<2} NormTop100Coverage={norm100:<2}"
            )
        lines.append("")

        lines.append("  LIVE 5495->5496 AND 5496->5497")
        for live_source in [5495, 5496]:
            live = [r for r in subset if r["source"] == live_source]
            if not live:
                lines.append(f"    {live_source}->{live_source + 1}: NOT FOUND")
                continue
            r = live[0]
            lines.append(
                f"    {r['source']}->{r['target']} "
                f"DayType={r['target_day_type']} "
                f"ExactSupport={r['exact_support']} RecentExactSupport={r['recent_exact_support']} "
                f"Bucket={r['bucket']} BucketSupport={r['bucket_support']} RecentBucketSupport={r['recent_bucket_support']} "
                f"Selected={r['normalized_selected_scope']}:{r['normalized_selected_value']}"
            )
            lines.append(
                f"      BaselineTop5={','.join(r['baseline_top5'])} "
                f"Hit={r['baseline_hit_count']} "
                f"MinCirc={r['baseline_min_circular_distance']} "
                f"MinHam={r['baseline_min_hamming_distance']} "
                f"MaxBox={r['baseline_max_box_overlap']}"
            )
            lines.append(
                f"      ExactTop5={','.join(r['exact_top5'])} "
                f"Hit={r['exact_top5_hit_count']} "
                f"Top100Hit={r['exact_top100_hit_count']} "
                f"MinCirc={r['exact_min_circular_distance']} "
                f"MinHam={r['exact_min_hamming_distance']} "
                f"MaxBox={r['exact_max_box_overlap']}"
            )
            lines.append(
                f"      NormalizedTop5={','.join(r['normalized_top5'])} "
                f"Hit={r['normalized_top5_hit_count']} "
                f"Top100Hit={r['normalized_top100_hit_count']} "
                f"MinCirc={r['normalized_min_circular_distance']} "
                f"MinHam={r['normalized_min_hamming_distance']} "
                f"MaxBox={r['normalized_max_box_overlap']}"
            )
        lines.append("")

    lines.append("=" * 112)
    lines.append("FINAL STEP 150C CONCLUSION")
    lines.append("=" * 112)
    lines.append("ProductionMathChanged: NO")
    lines.append("ProductionSwitchRecommendedNow: NO")
    lines.append("InterpretationRules:")
    lines.append("  If normalized Top5 beats baseline and improves live near-miss distance, continue to hardened prototype.")
    lines.append("  If normalized Top100 remains random-like or live Top100 remains zero, bucket normalization alone is insufficient.")
    lines.append("  SQL DayType values must not be mass-updated; derived runtime buckets should remain runtime-only.")
    lines.append("NextStep: Inspect Step 150C output before deciding Step 150D.")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print("")
    print(f"REPORT_WRITTEN: {REPORT_PATH}")
    print(f"JSONL_WRITTEN: {JSONL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
