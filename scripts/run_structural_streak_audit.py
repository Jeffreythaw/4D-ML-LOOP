from __future__ import annotations

import json
import math
import statistics
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
REPORT_PATH = PROJECT_ROOT / "reports" / "step_150d_structural_streak_audit.txt"
JSONL_PATH = PROJECT_ROOT / "reports" / "step_150d_structural_streak_audit_analogs.jsonl"

LIVE_SOURCES = [5495, 5496, 5497]
LIVE_CURRENT_SOURCE = 5497
ANALOG_TOP_KS = [10, 25, 50, 100]

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
    winning_numbers: list[str]


@dataclass(frozen=True)
class EngineOutcome:
    mode: str
    source_draw_no: int
    target_draw_no: int
    hit_count: int
    is_hit: bool
    target_day_type: str | None
    target_weekday: str | None
    target_month: int | None
    target_year: int | None


@dataclass(frozen=True)
class StreakOutcome:
    outcome: EngineOutcome
    pre_miss_streak: int
    post_miss_streak: int


@dataclass(frozen=True)
class StructuralProfile:
    source_draw_no: int
    day_type: str
    sql_weekday: str | None
    month: int | None
    year: int | None
    digit_sum_hist: dict[int, float]
    digit_hist: dict[int, float]
    mirror_hist: dict[int, float]
    first_pair_hist: dict[str, float]
    last_pair_hist: dict[str, float]
    repeated_digit_rate: float
    odd_digit_rate: float
    high_digit_rate: float
    mean_digit_sum: float
    pair_recurrence_3: float
    mirror_signature_counter: Counter[str]


@dataclass(frozen=True)
class AnalogMatch:
    live_source: int
    analog_source: int
    analog_target: int
    mode: str
    pre_miss_streak: int
    abs_streak_delta: int
    is_hit_next: bool
    hit_count_next: int
    distance_total: float
    distance_digit_sum: float
    distance_mirror: float
    distance_pair: float
    distance_shape: float
    month_match: bool
    daytype_match: bool
    runtime_bucket_match: bool


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


def mirror_signature(number: str) -> str:
    return "".join(str(int(ch) % 5) for ch in z4(number))


def runtime_bucket(day_type: str | None) -> str:
    normalized = str(day_type or "Unknown").strip()
    if normalized in {"Saturday", "Sunday"}:
        return "WEEKEND_SPACE"
    if normalized in {"Wednesday", "Special"}:
        return "MIDWEEK_SPECIAL_SPACE"
    return f"OTHER_SPACE::{normalized}"


def normalize_counter(counter: Counter, keys: Iterable) -> dict:
    total = sum(counter.values())
    if total <= 0:
        return {key: 0.0 for key in keys}
    return {key: float(counter.get(key, 0)) / float(total) for key in keys}


def l1_distance(a: dict, b: dict, keys: Iterable) -> float:
    return sum(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in keys)


def jaccard_distance_from_hists(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    min_sum = sum(min(float(a.get(k, 0.0)), float(b.get(k, 0.0))) for k in keys)
    max_sum = sum(max(float(a.get(k, 0.0)), float(b.get(k, 0.0))) for k in keys)
    if max_sum <= 0:
        return 1.0
    return 1.0 - (min_sum / max_sum)


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
            winning_numbers=parse_winners(row.WinningNumbers),
        )

    return draws


def fetch_engine_outcomes(cursor) -> dict[tuple[int, int], EngineOutcome]:
    rows = cursor.execute("""
        SELECT
            p.Mode,
            p.SourceDrawNo,
            p.TargetDrawNo,
            MAX(ISNULL(p.HitCount, 0)) AS HitCount,
            MAX(d.DayType) AS TargetDayType,
            MAX(DATENAME(WEEKDAY, d.DrawDate)) AS TargetWeekday,
            MAX(DATEPART(MONTH, d.DrawDate)) AS TargetMonth,
            MAX(DATEPART(YEAR, d.DrawDate)) AS TargetYear
        FROM dbo.PredictionLedger p
        LEFT JOIN dbo.DrawHistory d
            ON d.DrawNo = p.TargetDrawNo
        WHERE p.EngineSource = ?
          AND p.VerificationStatus = 'Verified'
        GROUP BY
            p.Mode,
            p.SourceDrawNo,
            p.TargetDrawNo
        ORDER BY
            p.SourceDrawNo,
            p.TargetDrawNo,
            p.Mode;
    """, ENGINE).fetchall()

    selected: dict[tuple[int, int], EngineOutcome] = {}

    for row in rows:
        mode = str(row.Mode)
        source = int(row.SourceDrawNo)
        target = int(row.TargetDrawNo)
        hit_count = int(row.HitCount or 0)

        candidate = EngineOutcome(
            mode=mode,
            source_draw_no=source,
            target_draw_no=target,
            hit_count=hit_count,
            is_hit=hit_count > 0,
            target_day_type=str(row.TargetDayType) if row.TargetDayType else None,
            target_weekday=str(row.TargetWeekday) if row.TargetWeekday else None,
            target_month=int(row.TargetMonth) if row.TargetMonth else None,
            target_year=int(row.TargetYear) if row.TargetYear else None,
        )

        key = (source, target)

        if key not in selected:
            selected[key] = candidate
            continue

        old = selected[key]
        if MODE_PRIORITY.get(candidate.mode, 99) < MODE_PRIORITY.get(old.mode, 99):
            selected[key] = candidate

    return selected


def build_streak_outcomes(outcomes: dict[tuple[int, int], EngineOutcome]) -> list[StreakOutcome]:
    ordered = sorted(outcomes.values(), key=lambda o: (o.source_draw_no, o.target_draw_no))

    streaks: list[StreakOutcome] = []
    current_miss_streak = 0

    for outcome in ordered:
        pre = current_miss_streak

        if outcome.is_hit:
            current_miss_streak = 0
        else:
            current_miss_streak += 1

        streaks.append(
            StreakOutcome(
                outcome=outcome,
                pre_miss_streak=pre,
                post_miss_streak=current_miss_streak,
            )
        )

    return streaks


def build_profile(draws: dict[int, DrawRecord], source_draw_no: int) -> StructuralProfile | None:
    draw = draws.get(source_draw_no)
    if not draw or not draw.winning_numbers:
        return None

    digit_sum_counter: Counter[int] = Counter()
    digit_counter: Counter[int] = Counter()
    mirror_counter: Counter[int] = Counter()
    first_pair_counter: Counter[str] = Counter()
    last_pair_counter: Counter[str] = Counter()
    mirror_signature_counter: Counter[str] = Counter()

    repeated_count = 0
    odd_count = 0
    high_count = 0
    total_digits = 0
    sums = []

    for number in draw.winning_numbers:
        number = z4(number)
        ds = digit_sum(number)
        sums.append(ds)
        digit_sum_counter[ds] += 1
        first_pair_counter[number[:2]] += 1
        last_pair_counter[number[2:]] += 1
        mirror_signature_counter[mirror_signature(number)] += 1

        if len(set(number)) < 4:
            repeated_count += 1

        for d in digits(number):
            digit_counter[d] += 1
            mirror_counter[d % 5] += 1
            odd_count += int(d % 2 == 1)
            high_count += int(d >= 5)
            total_digits += 1

    current_pairs = set(first_pair_counter) | set(last_pair_counter)
    previous_pairs = set()

    for back in [1, 2, 3]:
        prev = draws.get(source_draw_no - back)
        if not prev:
            continue
        for number in prev.winning_numbers:
            number = z4(number)
            previous_pairs.add(number[:2])
            previous_pairs.add(number[2:])

    pair_recurrence_3 = 0.0
    if current_pairs:
        pair_recurrence_3 = len(current_pairs & previous_pairs) / len(current_pairs)

    return StructuralProfile(
        source_draw_no=source_draw_no,
        day_type=draw.day_type,
        sql_weekday=draw.sql_weekday,
        month=draw.month,
        year=draw.year,
        digit_sum_hist=normalize_counter(digit_sum_counter, range(37)),
        digit_hist=normalize_counter(digit_counter, range(10)),
        mirror_hist=normalize_counter(mirror_counter, range(5)),
        first_pair_hist=normalize_counter(first_pair_counter, [f"{i:02d}" for i in range(100)]),
        last_pair_hist=normalize_counter(last_pair_counter, [f"{i:02d}" for i in range(100)]),
        repeated_digit_rate=repeated_count / max(1, len(draw.winning_numbers)),
        odd_digit_rate=odd_count / max(1, total_digits),
        high_digit_rate=high_count / max(1, total_digits),
        mean_digit_sum=statistics.mean(sums) if sums else 0.0,
        pair_recurrence_3=pair_recurrence_3,
        mirror_signature_counter=mirror_signature_counter,
    )


def structural_distance(a: StructuralProfile, b: StructuralProfile) -> tuple[float, float, float, float, float]:
    d_sum = l1_distance(a.digit_sum_hist, b.digit_sum_hist, range(37))
    d_mirror = l1_distance(a.mirror_hist, b.mirror_hist, range(5))

    first_pair_dist = jaccard_distance_from_hists(a.first_pair_hist, b.first_pair_hist)
    last_pair_dist = jaccard_distance_from_hists(a.last_pair_hist, b.last_pair_hist)
    d_pair = (first_pair_dist + last_pair_dist) / 2.0

    d_shape = (
        abs(a.repeated_digit_rate - b.repeated_digit_rate)
        + abs(a.odd_digit_rate - b.odd_digit_rate)
        + abs(a.high_digit_rate - b.high_digit_rate)
        + abs(a.mean_digit_sum - b.mean_digit_sum) / 36.0
        + abs(a.pair_recurrence_3 - b.pair_recurrence_3)
    ) / 5.0

    total = (
        0.25 * d_sum
        + 0.30 * d_mirror
        + 0.25 * d_pair
        + 0.20 * d_shape
    )

    return total, d_sum, d_mirror, d_pair, d_shape


def engine_current_pre_miss_streak(streaks: list[StreakOutcome], before_source: int) -> int:
    eligible = [s for s in streaks if s.outcome.source_draw_no < before_source]
    if not eligible:
        return 0
    latest = max(eligible, key=lambda s: s.outcome.source_draw_no)
    return latest.post_miss_streak


def find_analogs(
    draws: dict[int, DrawRecord],
    streaks: list[StreakOutcome],
    live_profile: StructuralProfile,
    live_pre_miss_streak: int,
) -> list[AnalogMatch]:
    analogs: list[AnalogMatch] = []

    for streak in streaks:
        source = streak.outcome.source_draw_no

        if source >= live_profile.source_draw_no:
            continue

        profile = build_profile(draws, source)
        if not profile:
            continue

        total, d_sum, d_mirror, d_pair, d_shape = structural_distance(live_profile, profile)

        abs_streak_delta = abs(streak.pre_miss_streak - live_pre_miss_streak)

        month_match = bool(live_profile.month and profile.month == live_profile.month)
        daytype_match = profile.day_type == live_profile.day_type
        runtime_bucket_match = runtime_bucket(profile.day_type) == runtime_bucket(live_profile.day_type)

        streak_penalty = abs_streak_delta * 0.25
        month_bonus = -0.03 if month_match else 0.0
        bucket_bonus = -0.04 if runtime_bucket_match else 0.0

        weighted_total = total + streak_penalty + month_bonus + bucket_bonus

        analogs.append(
            AnalogMatch(
                live_source=live_profile.source_draw_no,
                analog_source=source,
                analog_target=streak.outcome.target_draw_no,
                mode=streak.outcome.mode,
                pre_miss_streak=streak.pre_miss_streak,
                abs_streak_delta=abs_streak_delta,
                is_hit_next=streak.outcome.is_hit,
                hit_count_next=streak.outcome.hit_count,
                distance_total=weighted_total,
                distance_digit_sum=d_sum,
                distance_mirror=d_mirror,
                distance_pair=d_pair,
                distance_shape=d_shape,
                month_match=month_match,
                daytype_match=daytype_match,
                runtime_bucket_match=runtime_bucket_match,
            )
        )

    analogs.sort(
        key=lambda a: (
            a.abs_streak_delta,
            a.distance_total,
            a.distance_mirror,
            a.distance_pair,
            a.analog_source,
        )
    )

    return analogs


def summarize_cluster(analogs: list[AnalogMatch], max_streak_delta: int, top_k: int) -> dict:
    filtered = [a for a in analogs if a.abs_streak_delta <= max_streak_delta][:top_k]
    samples = len(filtered)
    hits = sum(1 for a in filtered if a.is_hit_next)
    raw_hits = sum(a.hit_count_next for a in filtered)

    return {
        "samples": samples,
        "hits": hits,
        "raw_hits": raw_hits,
        "hit_rate": hits / samples * 100.0 if samples else 0.0,
        "raw_hit_rate": raw_hits / samples if samples else 0.0,
    }


def target_repair_features(draws: dict[int, DrawRecord], analogs: list[AnalogMatch], max_streak_delta: int, top_k: int) -> dict:
    filtered = [a for a in analogs if a.abs_streak_delta <= max_streak_delta][:top_k]
    hit_filtered = [a for a in filtered if a.is_hit_next]

    digit_sum_delta_values = []
    pair_overlap_values = []
    mirror_l1_values = []
    top_source_to_target_mirror_pairs: Counter[str] = Counter()

    for analog in hit_filtered:
        source_profile = build_profile(draws, analog.analog_source)
        target_profile = build_profile(draws, analog.analog_target)

        source_draw = draws.get(analog.analog_source)
        target_draw = draws.get(analog.analog_target)

        if not source_profile or not target_profile or not source_draw or not target_draw:
            continue

        digit_sum_delta_values.append(target_profile.mean_digit_sum - source_profile.mean_digit_sum)

        source_pairs = set()
        target_pairs = set()

        for number in source_draw.winning_numbers:
            number = z4(number)
            source_pairs.add(number[:2])
            source_pairs.add(number[2:])

        for number in target_draw.winning_numbers:
            number = z4(number)
            target_pairs.add(number[:2])
            target_pairs.add(number[2:])

        if target_pairs:
            pair_overlap_values.append(len(source_pairs & target_pairs) / len(target_pairs))

        mirror_l1_values.append(l1_distance(source_profile.mirror_hist, target_profile.mirror_hist, range(5)))

        for source_sig, source_count in source_profile.mirror_signature_counter.most_common(5):
            for target_sig, target_count in target_profile.mirror_signature_counter.most_common(5):
                top_source_to_target_mirror_pairs[f"{source_sig}->{target_sig}"] += min(source_count, target_count)

    return {
        "hit_samples": len(hit_filtered),
        "avg_digit_sum_delta": statistics.mean(digit_sum_delta_values) if digit_sum_delta_values else None,
        "avg_pair_overlap": statistics.mean(pair_overlap_values) if pair_overlap_values else None,
        "avg_mirror_l1": statistics.mean(mirror_l1_values) if mirror_l1_values else None,
        "top_mirror_transitions": top_source_to_target_mirror_pairs.most_common(10),
    }


def fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "NULL"
    return f"{value:.{digits}f}"


def analog_to_json(analog: AnalogMatch) -> dict:
    return {
        "live_source": analog.live_source,
        "analog_source": analog.analog_source,
        "analog_target": analog.analog_target,
        "mode": analog.mode,
        "pre_miss_streak": analog.pre_miss_streak,
        "abs_streak_delta": analog.abs_streak_delta,
        "is_hit_next": analog.is_hit_next,
        "hit_count_next": analog.hit_count_next,
        "distance_total": analog.distance_total,
        "distance_digit_sum": analog.distance_digit_sum,
        "distance_mirror": analog.distance_mirror,
        "distance_pair": analog.distance_pair,
        "distance_shape": analog.distance_shape,
        "month_match": analog.month_match,
        "daytype_match": analog.daytype_match,
        "runtime_bucket_match": analog.runtime_bucket_match,
    }


def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with get_conn() as conn:
        cursor = conn.cursor()
        draws = fetch_draws(cursor)
        outcomes = fetch_engine_outcomes(cursor)

    streaks = build_streak_outcomes(outcomes)
    streak_by_source = {s.outcome.source_draw_no: s for s in streaks}

    lines: list[str] = []
    all_json_rows: list[dict] = []

    lines.append("=" * 118)
    lines.append("STEP 150D — STRUCTURAL & STREAK REGIME ANALOG AUDIT")
    lines.append("=" * 118)
    lines.append("Mode: REPORT ONLY")
    lines.append("ProductionMathChanged: NO")
    lines.append("SQLSchemaChanged: NO")
    lines.append("PrimaryDriver: Engine Pre-Draw Miss-Streak Renewal")
    lines.append("StructuralFeatures: DigitSumCloud, MirrorClassFootprint, PairRecurrence, ShapeMetrics")
    lines.append("")

    lines.append("ENGINE LEDGER STATE")
    lines.append("-" * 118)
    lines.append(f"Engine: {ENGINE}")
    lines.append(f"VerifiedOutcomePairs: {len(streaks)}")
    if streaks:
        lines.append(f"OutcomeSourceRange: {min(s.outcome.source_draw_no for s in streaks)}..{max(s.outcome.source_draw_no for s in streaks)}")
    lines.append("")

    miss_streak_counter = Counter(s.pre_miss_streak for s in streaks)
    hit_by_streak = defaultdict(lambda: {"samples": 0, "hits": 0, "raw_hits": 0})

    for s in streaks:
        bucket = hit_by_streak[s.pre_miss_streak]
        bucket["samples"] += 1
        bucket["hits"] += int(s.outcome.is_hit)
        bucket["raw_hits"] += s.outcome.hit_count

    lines.append("GLOBAL RENEWAL TABLE BY PRE-DRAW MISS STREAK")
    lines.append("-" * 118)
    for streak_len in sorted(hit_by_streak):
        item = hit_by_streak[streak_len]
        if item["samples"] < 3 and streak_len not in {0, 1, 2, 3, 4, 5}:
            continue
        rate = item["hits"] / item["samples"] * 100.0 if item["samples"] else 0.0
        lines.append(
            f"PreMissStreak={streak_len:<4} "
            f"Samples={item['samples']:<5} "
            f"NextHitDraws={item['hits']:<4} "
            f"RawHits={item['raw_hits']:<4} "
            f"NextHitRate={rate:7.3f}%"
        )
    lines.append("")

    current_pre_miss = engine_current_pre_miss_streak(streaks, before_source=LIVE_CURRENT_SOURCE + 1)
    lines.append("CURRENT LIVE STREAK STATE")
    lines.append("-" * 118)
    lines.append(f"CurrentLiveSourceForNextPrediction: {LIVE_CURRENT_SOURCE}")
    lines.append(f"ComputedCurrentFullLedgerPreMissStreakBeforeNextDraw: {current_pre_miss}")
    lines.append("LiveDeploymentLocalMissStreak: 3")
    lines.append("Note: Full-ledger streak is the primary macro feature for this audit; live-local streak is reported only for context.")
    lines.append("")

    for live_source in LIVE_SOURCES:
        live_profile = build_profile(draws, live_source)

        if not live_profile:
            lines.append(f"LIVE SOURCE {live_source}: profile unavailable")
            continue

        if live_source in streak_by_source:
            live_pre = streak_by_source[live_source].pre_miss_streak
            live_outcome_known = True
        else:
            live_pre = current_pre_miss
            live_outcome_known = False

        if live_source == LIVE_CURRENT_SOURCE:
            live_pre = current_pre_miss

        analogs = find_analogs(
            draws=draws,
            streaks=streaks,
            live_profile=live_profile,
            live_pre_miss_streak=live_pre,
        )

        for analog in analogs[:200]:
            all_json_rows.append(analog_to_json(analog))

        lines.append("=" * 118)
        lines.append(f"LIVE STRUCTURAL ANALOG PROFILE — SOURCE {live_source}")
        lines.append("=" * 118)
        lines.append(
            f"SourceDayType={live_profile.day_type} "
            f"SqlWeekday={live_profile.sql_weekday} "
            f"Month={live_profile.month} "
            f"RuntimeBucket={runtime_bucket(live_profile.day_type)}"
        )
        lines.append(
            f"PreDrawMissStreak={live_pre} "
            f"OutcomeKnownInLedger={live_outcome_known} "
            f"MeanDigitSum={live_profile.mean_digit_sum:.4f} "
            f"RepeatedDigitRate={live_profile.repeated_digit_rate:.4f} "
            f"OddDigitRate={live_profile.odd_digit_rate:.4f} "
            f"HighDigitRate={live_profile.high_digit_rate:.4f} "
            f"PairRecurrence3={live_profile.pair_recurrence_3:.4f}"
        )
        lines.append("")

        lines.append("ANALOG CLUSTER RENEWAL PROBABILITIES")
        lines.append("-" * 118)
        for delta in [0, 1, 2, 3]:
            for top_k in ANALOG_TOP_KS:
                summary = summarize_cluster(analogs, max_streak_delta=delta, top_k=top_k)
                if summary["samples"] == 0:
                    continue
                lines.append(
                    f"StreakDelta<= {delta} | TopK={top_k:<3} "
                    f"Samples={summary['samples']:<3} "
                    f"NextHitDraws={summary['hits']:<3} "
                    f"RawHits={summary['raw_hits']:<3} "
                    f"NextHitRate={summary['hit_rate']:7.3f}% "
                    f"RawHitsPerDraw={summary['raw_hit_rate']:.4f}"
                )
            lines.append("")

        lines.append("TOP 25 STRUCTURAL + STREAK ANALOGS")
        lines.append("-" * 118)
        for idx, analog in enumerate(analogs[:25], start=1):
            lines.append(
                f"{idx:02d}. "
                f"{analog.analog_source}->{analog.analog_target} "
                f"Mode={analog.mode:<22} "
                f"PreMiss={analog.pre_miss_streak:<3} "
                f"StreakDelta={analog.abs_streak_delta:<2} "
                f"NextHit={int(analog.is_hit_next)} "
                f"HitCount={analog.hit_count_next:<2} "
                f"Dist={analog.distance_total:.5f} "
                f"DigitSumD={analog.distance_digit_sum:.5f} "
                f"MirrorD={analog.distance_mirror:.5f} "
                f"PairD={analog.distance_pair:.5f} "
                f"ShapeD={analog.distance_shape:.5f} "
                f"MonthMatch={int(analog.month_match)} "
                f"BucketMatch={int(analog.runtime_bucket_match)}"
            )
        lines.append("")

        lines.append("POST-STREAK REPAIR PATTERNS AMONG HIT ANALOGS")
        lines.append("-" * 118)
        for delta in [0, 1, 2]:
            repair = target_repair_features(draws, analogs, max_streak_delta=delta, top_k=100)
            lines.append(
                f"StreakDelta<= {delta} | "
                f"HitAnalogSamples={repair['hit_samples']} "
                f"AvgDigitSumDelta={fmt_float(repair['avg_digit_sum_delta'])} "
                f"AvgPairOverlap={fmt_float(repair['avg_pair_overlap'])} "
                f"AvgMirrorL1={fmt_float(repair['avg_mirror_l1'])}"
            )
            if repair["top_mirror_transitions"]:
                transition_text = ", ".join(f"{k}:{v}" for k, v in repair["top_mirror_transitions"][:5])
                lines.append(f"  TopMirrorTransitions: {transition_text}")
        lines.append("")

    lines.append("=" * 118)
    lines.append("FINAL STEP 150D CONCLUSION")
    lines.append("=" * 118)
    lines.append("ProductionMathChanged: NO")
    lines.append("ProductionSwitchRecommendedNow: NO")
    lines.append("DecisionRules:")
    lines.append("  If current full-ledger PreMissStreak analog clusters show elevated renewal probability, use streak as primary macro weight.")
    lines.append("  If structural analog hit rate does not exceed the global same-streak hit rate, do not promote.")
    lines.append("  If repair patterns are stable among hit analogs, Step 150E may build a report-only streak-weighted repair prototype.")
    lines.append("NextStep: Inspect analog renewal probabilities and repair patterns before designing Step 150E.")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    with JSONL_PATH.open("w", encoding="utf-8") as fh:
        for row in all_json_rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    print("\n".join(lines))
    print("")
    print(f"REPORT_WRITTEN: {REPORT_PATH}")
    print(f"JSONL_WRITTEN: {JSONL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
