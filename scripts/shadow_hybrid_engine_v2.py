from __future__ import annotations

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


ENGINE_NAME = "E_SHADOW_HYBRID_GUARD_V2_BASELINE_PROTECTED"
BASELINE_ENGINE = "E1_TEMPORAL_CONTEXT_MATCH"
TOP_K = 5
MIN_HIGH_RISK_HIT_SAMPLE = 5
MAX_REPAIR_VARIANTS = 10
MIN_REPLACEMENT_SUPPORTS = 4
LOCKED_150F_BASELINE_5497 = ("7006", "4723", "3193", "9098", "2698")
LOCKED_150F_POLICY_CANDIDATES_5497 = (
    "7006", "4723", "3193", "9098", "2698",
    "6065", "6515", "7776", "0374", "1111", "1389", "2071",
    "2689", "2869", "2896", "2968", "2986", "3509", "8059", "9106", "2086",
)
MODE_PRIORITY = {
    "Current": 0,
    "Temporal_Global_Loop": 1,
    "Historical": 2,
    "Engine_Grand_Loop": 3,
    "Grand_Loop": 4,
    "Weighted_Grand_Loop": 5,
}


@dataclass(frozen=True)
class Draw:
    draw_no: int
    draw_date: str | None
    month: int | None
    day_type: str
    winners: tuple[str, ...]


@dataclass(frozen=True)
class LedgerPair:
    mode: str
    source: int
    target: int
    predictions: tuple[str, ...]
    ledger_hit_count: int


@dataclass(frozen=True)
class Feature:
    number: str
    digit_sum: int
    mirror: str
    box: str
    repeated_digits: int
    odd_ratio: float
    high_ratio: float
    pair_1: bool
    pair_3: bool
    pair_5: bool


@dataclass(frozen=True)
class ReplacementCandidate:
    number: str
    score: float
    supports: tuple[str, ...]
    reason: str
    history_max_draw_no: int
    source_kind: str


class V2HistoricalState:
    def __init__(self) -> None:
        self.high_risk_hit_sums: list[int] = []
        self.hit_mirrors: Counter[str] = Counter()
        self.hit_boxes: Counter[str] = Counter()
        self.near_miss_mirrors: Counter[str] = Counter()
        self.near_miss_boxes: Counter[str] = Counter()
        self.candidate_last_seen: dict[str, int] = {}
        self.candidate_appearances: Counter[str] = Counter()
        self.candidate_hits: Counter[str] = Counter()
        self.bucket_draws: Counter[str] = Counter()
        self.bucket_hit_draws: Counter[str] = Counter()
        self.observed_max_target = 0

    def observe(
        self,
        bucket: str,
        pair: LedgerPair,
        actuals: tuple[str, ...],
    ) -> None:
        actual_set = set(actuals)
        draw_hit = False
        for number in pair.predictions:
            self.candidate_last_seen[number] = pair.source
            self.candidate_appearances[number] += 1
            if number in actual_set:
                draw_hit = True
                self.candidate_hits[number] += 1
                self.hit_mirrors[mirror_signature(number)] += 1
                self.hit_boxes[box_signature(number)] += 1
                if bucket in {"HIGH_RISK", "EXTREME_RISK"}:
                    self.high_risk_hit_sums.append(digit_sum(number))
            else:
                for actual in actuals:
                    if hamming_distance(number, actual) <= 1:
                        self.near_miss_mirrors[mirror_signature(number)] += 1
                    if box_overlap(number, actual) >= 3:
                        self.near_miss_boxes[box_signature(number)] += 1
        self.bucket_draws[bucket] += 1
        self.bucket_hit_draws[bucket] += int(draw_hit)
        self.observed_max_target = max(self.observed_max_target, pair.target)

    def bucket_hit_rate(self, bucket: str) -> float:
        draws = self.bucket_draws[bucket]
        return self.bucket_hit_draws[bucket] / draws if draws else 0.0


def get_conn():
    return pyodbc.connect(get_settings().sql_connection_string(), timeout=120)


def z4(value: str | int) -> str:
    return str(value).strip().zfill(4)


def parse_numbers(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(z4(part) for part in str(value).replace(" ", "").split(",") if part)


def digits(number: str) -> tuple[int, ...]:
    return tuple(int(ch) for ch in z4(number))


def digit_sum(number: str) -> int:
    return sum(digits(number))


def mirror_signature(number: str) -> str:
    return "".join(str(value % 5) for value in digits(number))


def box_signature(number: str) -> str:
    return "".join(sorted(z4(number)))


def repeated_digits(number: str) -> int:
    normalized = z4(number)
    return len(normalized) - len(set(normalized))


def hamming_distance(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(z4(left), z4(right)))


def box_overlap(left: str, right: str) -> int:
    return sum((Counter(z4(left)) & Counter(z4(right))).values())


def risk_bucket(streak: int) -> str:
    if streak <= 5:
        return "LOW_RISK"
    if streak <= 15:
        return "MEDIUM_RISK"
    if streak <= 40:
        return "HIGH_RISK"
    return "EXTREME_RISK"


def runtime_bucket(day_type: str | None) -> str:
    normalized = str(day_type or "Unknown").strip()
    if normalized in {"Saturday", "Sunday"}:
        return "WEEKEND_SPACE"
    if normalized in {"Wednesday", "Special"}:
        return "MIDWEEK_SPECIAL_SPACE"
    return f"OTHER_SPACE::{normalized}"


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def fetch_draws(cursor) -> dict[int, Draw]:
    rows = cursor.execute(
        """
        SELECT
            DrawNo,
            CONVERT(varchar(10), DrawDate, 120) AS DrawDateText,
            DATEPART(MONTH, DrawDate) AS DrawMonth,
            DayType,
            WinningNumbers
        FROM dbo.DrawHistory
        WHERE WinningNumbers IS NOT NULL
        ORDER BY DrawNo;
        """
    ).fetchall()
    return {
        int(row.DrawNo): Draw(
            draw_no=int(row.DrawNo),
            draw_date=str(row.DrawDateText) if row.DrawDateText else None,
            month=int(row.DrawMonth) if row.DrawMonth else None,
            day_type=str(row.DayType or "Unknown"),
            winners=parse_numbers(row.WinningNumbers),
        )
        for row in rows
    }


def fetch_ledger_pairs(cursor) -> list[LedgerPair]:
    rows = cursor.execute(
        """
        SELECT
            Mode,
            SourceDrawNo,
            TargetDrawNo,
            RankNo,
            PredictedNumber,
            ISNULL(HitCount, 0) AS HitCount
        FROM dbo.PredictionLedger
        WHERE EngineSource = ?
          AND RankNo BETWEEN 1 AND 5
        ORDER BY SourceDrawNo, TargetDrawNo, Mode, RankNo;
        """,
        BASELINE_ENGINE,
    ).fetchall()
    grouped: dict[tuple[str, int, int], list] = defaultdict(list)
    for row in rows:
        grouped[(str(row.Mode), int(row.SourceDrawNo), int(row.TargetDrawNo))].append(row)
    selected: dict[tuple[int, int], LedgerPair] = {}
    for (mode, source, target), items in grouped.items():
        ranked = sorted(items, key=lambda item: int(item.RankNo))
        predictions = tuple(z4(item.PredictedNumber) for item in ranked)
        if len(predictions) != TOP_K:
            continue
        candidate = LedgerPair(
            mode=mode,
            source=source,
            target=target,
            predictions=predictions,
            ledger_hit_count=max(int(item.HitCount or 0) for item in ranked),
        )
        old = selected.get((source, target))
        if old is None or MODE_PRIORITY.get(mode, 99) < MODE_PRIORITY.get(old.mode, 99):
            selected[(source, target)] = candidate
    return sorted(selected.values(), key=lambda item: (item.source, item.target))


def pair_cloud(draws: dict[int, Draw], source: int, depth: int) -> set[str]:
    output: set[str] = set()
    for draw_no in sorted(no for no in draws if no <= source)[-depth:]:
        for number in draws[draw_no].winners:
            output.update((number[:2], number[2:]))
    return output


def features(numbers: Iterable[str], draws: dict[int, Draw], source: int) -> dict[str, Feature]:
    clouds = {depth: pair_cloud(draws, source, depth) for depth in (1, 3, 5)}
    output = {}
    for raw in numbers:
        number = z4(raw)
        first, last = number[:2], number[2:]
        values = digits(number)
        output[number] = Feature(
            number=number,
            digit_sum=sum(values),
            mirror=mirror_signature(number),
            box=box_signature(number),
            repeated_digits=repeated_digits(number),
            odd_ratio=sum(value % 2 for value in values) / 4,
            high_ratio=sum(value >= 5 for value in values) / 4,
            pair_1=first in clouds[1] or last in clouds[1],
            pair_3=first in clouds[3] or last in clouds[3],
            pair_5=first in clouds[5] or last in clouds[5],
        )
    return output


def recent_structure_ranges(draws: dict[int, Draw], source: int) -> dict[str, tuple[float, float]]:
    numbers = []
    for draw_no in sorted(no for no in draws if no <= source)[-5:]:
        numbers.extend(draws[draw_no].winners)
    if not numbers:
        return {"sum": (10, 26), "odd": (0, 1), "high": (0, 1)}
    sums = [digit_sum(number) for number in numbers]
    odds = [sum(value % 2 for value in digits(number)) / 4 for number in numbers]
    highs = [sum(value >= 5 for value in digits(number)) / 4 for number in numbers]
    return {
        "sum": (max(6.0, percentile(sums, 0.10) - 2), min(30.0, percentile(sums, 0.90) + 2)),
        "odd": (max(0.0, percentile(odds, 0.10) - 0.25), min(1.0, percentile(odds, 0.90) + 0.25)),
        "high": (max(0.0, percentile(highs, 0.10) - 0.25), min(1.0, percentile(highs, 0.90) + 0.25)),
    }


def controlled_repairs(baseline: tuple[str, ...], source: int) -> dict[str, str]:
    repairs: dict[str, str] = {}
    for base in baseline:
        for position in range(4):
            chars = list(base)
            chars[position] = str((int(chars[position]) + 5) % 10)
            repairs.setdefault("".join(chars), f"single_mirror_flip_of_{base}")
            if len(repairs) >= MAX_REPAIR_VARIANTS:
                return repairs
        reversed_number = base[::-1]
        if reversed_number != base:
            repairs.setdefault(reversed_number, f"bounded_reverse_of_{base}")
        if len(repairs) >= MAX_REPAIR_VARIANTS:
            return repairs
    return repairs


def support_set(
    number: str,
    feature: Feature,
    state: V2HistoricalState,
    ranges: dict[str, tuple[float, float]],
    repair_reason: str | None,
) -> tuple[str, ...]:
    supports = []
    if len(state.high_risk_hit_sums) >= MIN_HIGH_RISK_HIT_SAMPLE:
        lower = math.floor(percentile(state.high_risk_hit_sums, 0.20))
        upper = math.ceil(percentile(state.high_risk_hit_sums, 0.80))
        if lower <= feature.digit_sum <= upper:
            supports.append("RISK_SUM_BAND_SUPPORT")
    if feature.pair_1 or (feature.pair_3 and feature.pair_5):
        supports.append("PAIR_RECURRENCE_SUPPORT")
    mirror_support = (
        state.hit_mirrors[feature.mirror] + state.near_miss_mirrors[feature.mirror]
    )
    box_support = state.hit_boxes[feature.box] + state.near_miss_boxes[feature.box]
    if mirror_support >= 2 or box_support >= 2:
        supports.append("MIRROR_BOX_SUPPORT")
    if repair_reason is not None:
        supports.append("BASELINE_REPAIR_SUPPORT")
    sum_ok = ranges["sum"][0] <= feature.digit_sum <= ranges["sum"][1]
    odd_ok = ranges["odd"][0] <= feature.odd_ratio <= ranges["odd"][1]
    high_ok = ranges["high"][0] <= feature.high_ratio <= ranges["high"][1]
    if sum_ok and odd_ok and high_ok and feature.repeated_digits <= 1:
        supports.append("RECENT_STRUCTURE_SUPPORT")
    return tuple(supports)


def baseline_quality(
    number: str,
    rank: int,
    feature: Feature,
    state: V2HistoricalState,
    ranges: dict[str, tuple[float, float]],
) -> float:
    supports = support_set(number, feature, state, ranges, None)
    score = 40.0 + (6 - rank) * 3.0 + len(supports) * 5.0
    score -= max(0, feature.repeated_digits - 1) * 8.0
    if not ranges["sum"][0] <= feature.digit_sum <= ranges["sum"][1]:
        score -= 8.0
    appearances = state.candidate_appearances[number]
    if appearances:
        score += 20.0 * state.candidate_hits[number] / appearances
    return score


def generate_with_context(
    source_draw_no: int,
    draws: dict[int, Draw],
    pairs_by_source: dict[int, LedgerPair],
    state: V2HistoricalState,
    pre_miss_streak: int,
) -> dict:
    if source_draw_no not in draws:
        raise ValueError(f"Source draw {source_draw_no} is unavailable")
    pair = pairs_by_source.get(source_draw_no)
    warnings: list[str] = []
    if pair is not None:
        baseline = pair.predictions
        baseline_source = "stored PredictionLedger E1 Top5"
    elif source_draw_no == 5497:
        baseline = LOCKED_150F_BASELINE_5497
        baseline_source = "locked Step 150F baseline"
        warnings.append("BASELINE_FROM_LOCKED_STEP_150F")
    else:
        raise RuntimeError("Baseline E1 Top5 unavailable; V2 refuses historical-winner fallback")

    bucket = risk_bucket(pre_miss_streak)
    ranges = recent_structure_ranges(draws, source_draw_no)
    baseline_features = features(baseline, draws, source_draw_no)
    baseline_scores = {
        number: baseline_quality(number, rank, baseline_features[number], state, ranges)
        for rank, number in enumerate(baseline, start=1)
    }
    weakest = min(
        enumerate(baseline, start=1),
        key=lambda item: (baseline_scores[item[1]], -item[0]),
    )
    weakest_rank, weakest_number = weakest

    replacement_attempts = 0
    eligible: list[ReplacementCandidate] = []
    repairs = controlled_repairs(baseline, source_draw_no)
    pool_sources: dict[str, tuple[int, str]] = {
        number: (draw_no, "PRIOR_E1_CANDIDATE")
        for number, draw_no in state.candidate_last_seen.items()
        if draw_no <= source_draw_no
        and source_draw_no - draw_no <= 365
        and state.candidate_appearances[number] >= 2
        and number not in baseline
    }
    if source_draw_no == 5497:
        for number in LOCKED_150F_POLICY_CANDIDATES_5497:
            if number not in baseline:
                pool_sources[number] = (source_draw_no, "LOCKED_STEP_150F_POLICY")
    for number, reason in repairs.items():
        if number not in baseline:
            pool_sources[number] = (source_draw_no, f"CONTROLLED_REPAIR::{reason}")

    if bucket in {"HIGH_RISK", "EXTREME_RISK"}:
        pool_features = features(pool_sources, draws, source_draw_no)
        for number, (history_max, source_kind) in pool_sources.items():
            replacement_attempts += 1
            repair_reason = (
                source_kind.split("::", 1)[1]
                if source_kind.startswith("CONTROLLED_REPAIR::")
                else None
            )
            candidate_supports = support_set(
                number,
                pool_features[number],
                state,
                ranges,
                repair_reason,
            )
            if len(candidate_supports) < MIN_REPLACEMENT_SUPPORTS:
                continue
            appearances = state.candidate_appearances[number]
            historical_rate = (
                state.candidate_hits[number] / appearances if appearances else 0.0
            )
            if source_kind == "PRIOR_E1_CANDIDATE" and (
                appearances < 5 or state.candidate_hits[number] < 1
            ):
                continue
            if source_kind == "LOCKED_STEP_150F_POLICY" and appearances < 2:
                continue
            score = (
                38.0
                + len(candidate_supports) * 7.0
                + historical_rate * 25.0
                + (3.0 if source_kind == "PRIOR_E1_CANDIDATE" else 0.0)
                + (2.0 if source_kind == "LOCKED_STEP_150F_POLICY" else 0.0)
            )
            eligible.append(
                ReplacementCandidate(
                    number=number,
                    score=score,
                    supports=candidate_supports,
                    reason=f"{source_kind}; historical appearances={appearances}; hit_rate={historical_rate:.5f}",
                    history_max_draw_no=history_max,
                    source_kind=source_kind,
                )
            )
    else:
        warnings.append("NO_REPLACEMENT_BELOW_HIGH_RISK")

    best = max(eligible, key=lambda item: (item.score, item.number), default=None)
    replacement_used = False
    removed = None
    added = None
    replacement_reason = "No candidate passed the strict multi-signal and score guards."
    output = list(baseline)

    if best is not None:
        required_supports = 4 if weakest_rank == 1 else 3
        severe_rank1_penalty = baseline_scores[weakest_number] < 45.0
        margin_ok = best.score >= baseline_scores[weakest_number] + 15.0
        rank1_ok = weakest_rank != 1 or (
            severe_rank1_penalty and len(best.supports) >= required_supports
        )
        bucket_rate = state.bucket_hit_rate(bucket)
        baseline_advantage_guard = bucket_rate >= 0.0215
        if margin_ok and rank1_ok and not baseline_advantage_guard:
            output[weakest_rank - 1] = best.number
            replacement_used = True
            removed = weakest_number
            added = best.number
            replacement_reason = (
                f"Replaced weakest baseline rank {weakest_rank}; replacement score "
                f"{best.score:.3f} vs baseline score {baseline_scores[weakest_number]:.3f}."
            )
        else:
            if not margin_ok:
                warnings.append("NO_REPLACEMENT_SCORE_MARGIN")
            if not rank1_ok:
                warnings.append("RANK1_PROTECTION")
            if baseline_advantage_guard:
                warnings.append("BASELINE_RISK_BUCKET_ADVANTAGE_GUARD")
    else:
        warnings.append("NO_REPLACEMENT_PASSED_4_SUPPORTS")

    candidate_details = []
    for rank, number in enumerate(output, start=1):
        if replacement_used and number == added:
            candidate_details.append(
                {
                    "rank": rank,
                    "number": number,
                    "family": "V2_REPLACEMENT",
                    "score": round(best.score, 6),
                    "reason": best.reason,
                    "supports": list(best.supports),
                    "history_max_draw_no": best.history_max_draw_no,
                }
            )
        else:
            candidate_details.append(
                {
                    "rank": rank,
                    "number": number,
                    "family": "BASELINE_E1",
                    "score": round(baseline_scores[number], 6),
                    "reason": f"{baseline_source}; protected baseline rank {baseline.index(number) + 1}",
                    "supports": list(
                        support_set(
                            number,
                            baseline_features[number],
                            state,
                            ranges,
                            None,
                        )
                    ),
                    "history_max_draw_no": source_draw_no,
                }
            )

    temporal_ok = all(
        item["history_max_draw_no"] <= source_draw_no for item in candidate_details
    )
    if len(output) != TOP_K or len(set(output)) != TOP_K:
        warnings.append("INVALID_OR_DUPLICATE_TOP5")
    if not temporal_ok:
        warnings.append("TEMPORAL_FIREWALL_VIOLATION")

    return {
        "engine_name": ENGINE_NAME,
        "source_draw_no": source_draw_no,
        "target_draw_no": source_draw_no + 1,
        "full_ledger_pre_miss_streak": pre_miss_streak,
        "risk_bucket": bucket,
        "baseline_top5": list(baseline),
        "top5": output,
        "baseline_kept_count": sum(number in baseline for number in output),
        "replacement_used": replacement_used,
        "replacement_attempts": replacement_attempts,
        "replacement_detail": {
            "removed": removed,
            "added": added,
            "supports": list(best.supports) if replacement_used and best else [],
            "reason": replacement_reason,
            "source_kind": best.source_kind if replacement_used and best else None,
        },
        "candidate_details": candidate_details,
        "warnings": sorted(set(warnings)),
        "production_safe": True,
        "temporal_firewall_ok": temporal_ok,
    }


def replay_state_before_source(
    source_draw_no: int,
    draws: dict[int, Draw],
    pairs: list[LedgerPair],
) -> tuple[V2HistoricalState, int]:
    state = V2HistoricalState()
    streak = 0
    for pair in pairs:
        if pair.source >= source_draw_no:
            break
        target = draws.get(pair.target)
        if target is None:
            continue
        state.observe(risk_bucket(streak), pair, target.winners)
        streak = 0 if pair.ledger_hit_count > 0 else streak + 1
    return state, streak


def generate_shadow_hybrid_v2_top5(source_draw_no: int) -> dict:
    with get_conn() as connection:
        cursor = connection.cursor()
        draws = fetch_draws(cursor)
        pairs = fetch_ledger_pairs(cursor)
    state, streak = replay_state_before_source(source_draw_no, draws, pairs)
    return generate_with_context(
        source_draw_no,
        draws,
        {pair.source: pair for pair in pairs},
        state,
        streak,
    )
