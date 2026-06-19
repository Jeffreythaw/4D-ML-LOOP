from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
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
from app.core.ml_adapter import run_existing_engine_prediction
from app.schemas.prediction import PredictionRequest


ENGINE = "E1_TEMPORAL_CONTEXT_MATCH"
REPORT_PATH = PROJECT_ROOT / "reports" / "step_150e_streak_risk_gated_pruning_audit.txt"
JSONL_PATH = PROJECT_ROOT / "reports" / "step_150e_streak_risk_gated_pruning_rows.jsonl"

POLICIES = (
    "BASELINE",
    "RISK_SUPPRESS_REPEATS",
    "RISK_SUM_BAND",
    "RISK_MIRROR_PRIOR",
    "RISK_PAIR_RECURRENCE",
    "HYBRID_RISK_GATE",
)

MODE_PRIORITY = {
    "Current": 0,
    "Temporal_Global_Loop": 1,
    "Historical": 2,
    "Engine_Grand_Loop": 3,
    "Grand_Loop": 4,
    "Weighted_Grand_Loop": 5,
}

HIGH_RISK_MIN_STREAK = 16
PRIOR_MIN_HIT_ROWS = 5
REPLACEMENT_HISTORY_DRAWS = 12
PAIR_HISTORY_DRAWS = 3
RANDOM_NUMBER_SPACE = 10_000
TOP_K = 5
LIVE_SOURCES = (5494, 5495, 5496)
NEXT_SOURCE = 5497


@dataclass(frozen=True)
class DrawRecord:
    draw_no: int
    day_type: str
    winning_numbers: tuple[str, ...]


@dataclass(frozen=True)
class LedgerPair:
    mode: str
    source_draw_no: int
    target_draw_no: int
    verification_status: str
    ledger_hit_count: int
    predictions: tuple[str, ...]


@dataclass(frozen=True)
class EvaluatedPair:
    ledger: LedgerPair
    pre_miss_streak: int
    post_miss_streak: int
    risk_bucket: str
    actuals: tuple[str, ...]
    baseline_raw_hits: int
    random_draw_hit_probability: float


@dataclass(frozen=True)
class CandidateFeatures:
    number: str
    digit_sum: int
    repeated_digit_count: int
    odd_ratio: float
    high_digit_ratio: float
    mirror_signature: str
    first_pair: str
    last_pair: str
    box_signature: str
    pair_recurrence: bool
    recency_score: float


@dataclass(frozen=True)
class PolicyResult:
    policy: str
    source_draw_no: int
    target_draw_no: int
    pre_miss_streak: int
    risk_bucket: str
    selected_numbers: tuple[str, ...]
    raw_hits: int | None
    draws_with_hit: int | None
    verified: bool
    limitation: str | None
    candidate_history_max_draw_no: int
    prior_training_max_target_draw_no: int


def get_conn():
    settings = get_settings()
    return pyodbc.connect(settings.sql_connection_string(), timeout=120)


def z4(value: str | int) -> str:
    return str(value).strip().zfill(4)


def parse_numbers(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(z4(part) for part in str(value).replace(" ", "").split(",") if part.strip())


def risk_bucket(pre_miss_streak: int) -> str:
    if pre_miss_streak <= 5:
        return "LOW_RISK"
    if pre_miss_streak <= 15:
        return "MEDIUM_RISK"
    if pre_miss_streak <= 40:
        return "HIGH_RISK"
    return "EXTREME_RISK"


def digit_values(number: str) -> tuple[int, ...]:
    return tuple(int(ch) for ch in z4(number))


def digit_sum(number: str) -> int:
    return sum(digit_values(number))


def repeated_digit_count(number: str) -> int:
    normalized = z4(number)
    return len(normalized) - len(set(normalized))


def odd_ratio(number: str) -> float:
    return sum(value % 2 == 1 for value in digit_values(number)) / 4.0


def high_digit_ratio(number: str) -> float:
    return sum(value >= 5 for value in digit_values(number)) / 4.0


def mirror_signature(number: str) -> str:
    return "".join(str(value % 5) for value in digit_values(number))


def box_signature(number: str) -> str:
    return "".join(sorted(z4(number)))


def hamming_distance(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(z4(left), z4(right)))


def box_distance(left: str, right: str) -> int:
    overlap = sum((Counter(z4(left)) & Counter(z4(right))).values())
    return 4 - overlap


def mirror_hamming_distance(left: str, right: str) -> int:
    return hamming_distance(mirror_signature(left), mirror_signature(right))


def percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower]) * (1.0 - weight) + float(ordered[upper]) * weight


def fetch_draws(cursor) -> dict[int, DrawRecord]:
    rows = cursor.execute(
        """
        SELECT
            DrawNo,
            DayType,
            WinningNumbers
        FROM dbo.DrawHistory
        WHERE WinningNumbers IS NOT NULL
        ORDER BY DrawNo;
        """
    ).fetchall()

    return {
        int(row.DrawNo): DrawRecord(
            draw_no=int(row.DrawNo),
            day_type=str(row.DayType or "Unknown"),
            winning_numbers=parse_numbers(row.WinningNumbers),
        )
        for row in rows
    }


def fetch_e1_ledger_pairs(cursor) -> list[LedgerPair]:
    rows = cursor.execute(
        """
        SELECT
            Mode,
            SourceDrawNo,
            TargetDrawNo,
            RankNo,
            PredictedNumber,
            VerificationStatus,
            ISNULL(HitCount, 0) AS HitCount
        FROM dbo.PredictionLedger
        WHERE EngineSource = ?
          AND VerificationStatus = 'Verified'
        ORDER BY
            SourceDrawNo,
            TargetDrawNo,
            Mode,
            RankNo;
        """,
        ENGINE,
    ).fetchall()

    grouped: dict[tuple[str, int, int], list] = defaultdict(list)
    for row in rows:
        grouped[(str(row.Mode), int(row.SourceDrawNo), int(row.TargetDrawNo))].append(row)

    selected: dict[tuple[int, int], LedgerPair] = {}

    for (mode, source, target), items in grouped.items():
        ranked = sorted(items, key=lambda item: int(item.RankNo))
        predictions = tuple(z4(item.PredictedNumber) for item in ranked[:TOP_K])
        if len(predictions) != TOP_K:
            continue

        candidate = LedgerPair(
            mode=mode,
            source_draw_no=source,
            target_draw_no=target,
            verification_status=str(ranked[0].VerificationStatus),
            ledger_hit_count=max(int(item.HitCount or 0) for item in ranked),
            predictions=predictions,
        )

        key = (source, target)
        old = selected.get(key)
        if old is None or MODE_PRIORITY.get(mode, 99) < MODE_PRIORITY.get(old.mode, 99):
            selected[key] = candidate

    return sorted(selected.values(), key=lambda item: (item.source_draw_no, item.target_draw_no))


def evaluate_pairs(
    ledger_pairs: list[LedgerPair],
    draws: dict[int, DrawRecord],
) -> tuple[list[EvaluatedPair], int]:
    current_streak = 0
    evaluated: list[EvaluatedPair] = []

    for ledger in ledger_pairs:
        target_draw = draws.get(ledger.target_draw_no)
        if target_draw is None:
            continue

        actual_set = set(target_draw.winning_numbers)
        baseline_raw_hits = sum(number in actual_set for number in ledger.predictions)
        random_probability = 1.0 - (
            (RANDOM_NUMBER_SPACE - len(actual_set)) / RANDOM_NUMBER_SPACE
        ) ** TOP_K
        pre_streak = current_streak
        is_hit = ledger.ledger_hit_count > 0
        current_streak = 0 if is_hit else current_streak + 1

        evaluated.append(
            EvaluatedPair(
                ledger=ledger,
                pre_miss_streak=pre_streak,
                post_miss_streak=current_streak,
                risk_bucket=risk_bucket(pre_streak),
                actuals=target_draw.winning_numbers,
                baseline_raw_hits=baseline_raw_hits,
                random_draw_hit_probability=random_probability,
            )
        )

    return evaluated, current_streak


def recent_pair_set(draws: dict[int, DrawRecord], source_draw_no: int) -> set[str]:
    pairs: set[str] = set()
    for draw_no in range(max(min(draws), source_draw_no - PAIR_HISTORY_DRAWS + 1), source_draw_no + 1):
        draw = draws.get(draw_no)
        if draw is None:
            continue
        for number in draw.winning_numbers:
            pairs.add(number[:2])
            pairs.add(number[2:])
    return pairs


def replacement_pool(
    draws: dict[int, DrawRecord],
    source_draw_no: int,
    baseline: Iterable[str],
) -> tuple[set[str], Counter[str]]:
    candidates = {z4(number) for number in baseline}
    recency: Counter[str] = Counter()
    first_draw = max(min(draws), source_draw_no - REPLACEMENT_HISTORY_DRAWS + 1)

    for draw_no in range(first_draw, source_draw_no + 1):
        draw = draws.get(draw_no)
        if draw is None:
            continue
        weight = 1.0 / (1.0 + source_draw_no - draw_no)
        for number in draw.winning_numbers:
            normalized = z4(number)
            candidates.add(normalized)
            recency[normalized] += weight

    return candidates, recency


def candidate_features(
    number: str,
    pair_set: set[str],
    recency: Counter[str],
) -> CandidateFeatures:
    normalized = z4(number)
    first_pair = normalized[:2]
    last_pair = normalized[2:]
    return CandidateFeatures(
        number=normalized,
        digit_sum=digit_sum(normalized),
        repeated_digit_count=repeated_digit_count(normalized),
        odd_ratio=odd_ratio(normalized),
        high_digit_ratio=high_digit_ratio(normalized),
        mirror_signature=mirror_signature(normalized),
        first_pair=first_pair,
        last_pair=last_pair,
        box_signature=box_signature(normalized),
        pair_recurrence=first_pair in pair_set or last_pair in pair_set,
        recency_score=float(recency.get(normalized, 0.0)),
    )


def fill_to_top5(
    retained: list[str],
    ranked_candidates: Iterable[str],
    baseline: tuple[str, ...],
) -> tuple[str, ...]:
    selected: list[str] = []
    for number in [*retained, *ranked_candidates, *baseline]:
        normalized = z4(number)
        if normalized not in selected:
            selected.append(normalized)
        if len(selected) == TOP_K:
            break
    return tuple(selected)


def apply_diversity_guard(ranked: list[CandidateFeatures]) -> tuple[str, ...]:
    selected: list[CandidateFeatures] = []
    box_counts: Counter[str] = Counter()
    mirror_counts: Counter[str] = Counter()

    for candidate in ranked:
        if box_counts[candidate.box_signature] >= 1:
            continue
        if mirror_counts[candidate.mirror_signature] >= 2:
            continue
        selected.append(candidate)
        box_counts[candidate.box_signature] += 1
        mirror_counts[candidate.mirror_signature] += 1
        if len(selected) == TOP_K:
            break

    if len(selected) < TOP_K:
        already = {candidate.number for candidate in selected}
        for candidate in ranked:
            if candidate.number in already:
                continue
            selected.append(candidate)
            already.add(candidate.number)
            if len(selected) == TOP_K:
                break

    return tuple(candidate.number for candidate in selected)


def generate_policy_sets(
    baseline: tuple[str, ...],
    pre_miss_streak: int,
    draws: dict[int, DrawRecord],
    source_draw_no: int,
    prior_high_risk_hit_features: list[CandidateFeatures],
) -> tuple[dict[str, tuple[str, ...]], dict[str, str | None]]:
    selected = {policy: baseline for policy in POLICIES}
    limitations: dict[str, str | None] = {policy: None for policy in POLICIES}

    if pre_miss_streak < HIGH_RISK_MIN_STREAK:
        for policy in POLICIES[1:]:
            limitations[policy] = "NO_OP_BELOW_HIGH_RISK"
        return selected, limitations

    pair_set = recent_pair_set(draws, source_draw_no)
    pool, recency = replacement_pool(draws, source_draw_no, baseline)
    features = {
        number: candidate_features(number, pair_set, recency)
        for number in pool
    }

    suppress_retained = [
        number for number in baseline if features[number].repeated_digit_count <= 1
    ]
    suppress_ranked = sorted(
        features.values(),
        key=lambda item: (
            item.repeated_digit_count,
            -item.recency_score,
            abs(item.digit_sum - 18),
            item.number,
        ),
    )
    selected["RISK_SUPPRESS_REPEATS"] = fill_to_top5(
        suppress_retained,
        (item.number for item in suppress_ranked),
        baseline,
    )

    prior_sums = [item.digit_sum for item in prior_high_risk_hit_features]
    prior_mirrors = Counter(item.mirror_signature for item in prior_high_risk_hit_features)
    enough_prior_hits = len(prior_high_risk_hit_features) >= PRIOR_MIN_HIT_ROWS

    if enough_prior_hits:
        lower_sum = math.floor(percentile(prior_sums, 0.20))
        upper_sum = math.ceil(percentile(prior_sums, 0.80))
        median_sum = statistics.median(prior_sums)

        sum_retained = [
            number
            for number in baseline
            if lower_sum <= features[number].digit_sum <= upper_sum
        ]
        sum_ranked = sorted(
            features.values(),
            key=lambda item: (
                not (lower_sum <= item.digit_sum <= upper_sum),
                abs(item.digit_sum - median_sum),
                -item.recency_score,
                item.number,
            ),
        )
        selected["RISK_SUM_BAND"] = fill_to_top5(
            sum_retained,
            (item.number for item in sum_ranked),
            baseline,
        )

        mirror_retained = [
            number for number in baseline if features[number].mirror_signature in prior_mirrors
        ]
        mirror_ranked = sorted(
            features.values(),
            key=lambda item: (
                -prior_mirrors[item.mirror_signature],
                -item.recency_score,
                item.repeated_digit_count,
                item.number,
            ),
        )
        selected["RISK_MIRROR_PRIOR"] = fill_to_top5(
            mirror_retained,
            (item.number for item in mirror_ranked),
            baseline,
        )
    else:
        limitations["RISK_SUM_BAND"] = (
            f"LOW_SAMPLE_NO_OP_PRIOR_HIGH_RISK_HIT_ROWS={len(prior_high_risk_hit_features)}"
        )
        limitations["RISK_MIRROR_PRIOR"] = (
            f"LOW_SAMPLE_NO_OP_PRIOR_HIGH_RISK_HIT_ROWS={len(prior_high_risk_hit_features)}"
        )

    if pair_set:
        pair_retained = [number for number in baseline if features[number].pair_recurrence]
        pair_ranked = sorted(
            features.values(),
            key=lambda item: (
                not item.pair_recurrence,
                -item.recency_score,
                item.repeated_digit_count,
                item.number,
            ),
        )
        selected["RISK_PAIR_RECURRENCE"] = fill_to_top5(
            pair_retained,
            (item.number for item in pair_ranked),
            baseline,
        )
    else:
        limitations["RISK_PAIR_RECURRENCE"] = "NO_RECENT_SOURCE_PAIRS_NO_OP"

    max_recency = max((item.recency_score for item in features.values()), default=1.0)
    baseline_set = set(baseline)
    lower_sum = math.floor(percentile(prior_sums, 0.20)) if enough_prior_hits else None
    upper_sum = math.ceil(percentile(prior_sums, 0.80)) if enough_prior_hits else None

    def hybrid_score(item: CandidateFeatures) -> float:
        score = 0.0
        score += 1.0 if item.repeated_digit_count <= 1 else -1.0 * item.repeated_digit_count
        score += 1.25 if item.pair_recurrence else 0.0
        score += item.recency_score / max(1.0, max_recency)
        score += 0.35 if item.number in baseline_set else 0.0
        if enough_prior_hits and lower_sum is not None and upper_sum is not None:
            score += 1.5 if lower_sum <= item.digit_sum <= upper_sum else -0.25
            score += min(2.0, prior_mirrors[item.mirror_signature] * 0.25)
        return score

    hybrid_ranked = sorted(
        features.values(),
        key=lambda item: (
            -hybrid_score(item),
            -item.recency_score,
            item.number,
        ),
    )
    selected["HYBRID_RISK_GATE"] = apply_diversity_guard(hybrid_ranked)

    if not enough_prior_hits:
        limitations["HYBRID_RISK_GATE"] = (
            "PARTIAL_PRIORS_ONLY_REPEAT_PAIR_RECENCY;"
            f"PRIOR_HIGH_RISK_HIT_ROWS={len(prior_high_risk_hit_features)}"
        )

    return selected, limitations


def candidate_near_miss_features(
    features: CandidateFeatures,
    actuals: tuple[str, ...],
) -> dict[str, int | None]:
    if not actuals:
        return {
            "min_hamming": None,
            "min_box_distance": None,
            "min_mirror_hamming": None,
        }
    return {
        "min_hamming": min(hamming_distance(features.number, actual) for actual in actuals),
        "min_box_distance": min(box_distance(features.number, actual) for actual in actuals),
        "min_mirror_hamming": min(
            mirror_hamming_distance(features.number, actual) for actual in actuals
        ),
    }


def run_online_backtest(
    evaluated_pairs: list[EvaluatedPair],
    draws: dict[int, DrawRecord],
) -> tuple[list[PolicyResult], list[dict], list[CandidateFeatures]]:
    policy_results: list[PolicyResult] = []
    candidate_rows: list[dict] = []
    prior_high_risk_hit_features: list[CandidateFeatures] = []

    for evaluated in evaluated_pairs:
        source = evaluated.ledger.source_draw_no
        target = evaluated.ledger.target_draw_no
        baseline = evaluated.ledger.predictions
        pair_set = recent_pair_set(draws, source)
        _, recency = replacement_pool(draws, source, baseline)

        policy_sets, limitations = generate_policy_sets(
            baseline=baseline,
            pre_miss_streak=evaluated.pre_miss_streak,
            draws=draws,
            source_draw_no=source,
            prior_high_risk_hit_features=prior_high_risk_hit_features,
        )

        actual_set = set(evaluated.actuals)
        for policy, selected_numbers in policy_sets.items():
            raw_hits = sum(number in actual_set for number in selected_numbers)
            policy_results.append(
                PolicyResult(
                    policy=policy,
                    source_draw_no=source,
                    target_draw_no=target,
                    pre_miss_streak=evaluated.pre_miss_streak,
                    risk_bucket=evaluated.risk_bucket,
                    selected_numbers=selected_numbers,
                    raw_hits=raw_hits,
                    draws_with_hit=int(raw_hits > 0),
                    verified=True,
                    limitation=limitations[policy],
                    candidate_history_max_draw_no=source,
                    prior_training_max_target_draw_no=source,
                )
            )

        for rank, number in enumerate(baseline, start=1):
            features = candidate_features(number, pair_set, recency)
            is_actual_hit = number in actual_set
            row = {
                "record_type": "baseline_candidate",
                "mode": evaluated.ledger.mode,
                "source_draw_no": source,
                "target_draw_no": target,
                "rank_no": rank,
                "pre_miss_streak": evaluated.pre_miss_streak,
                "risk_bucket": evaluated.risk_bucket,
                "number": number,
                "is_actual_hit": is_actual_hit,
                "ledger_draw_hit_count": evaluated.ledger.ledger_hit_count,
                **asdict(features),
                **candidate_near_miss_features(features, evaluated.actuals),
            }
            candidate_rows.append(row)
            if evaluated.pre_miss_streak >= HIGH_RISK_MIN_STREAK and is_actual_hit:
                prior_high_risk_hit_features.append(features)

    return policy_results, candidate_rows, prior_high_risk_hit_features


def summarize_baseline_by_bucket(
    evaluated_pairs: list[EvaluatedPair],
) -> list[dict]:
    by_bucket: dict[str, list[EvaluatedPair]] = defaultdict(list)
    for item in evaluated_pairs:
        by_bucket[item.risk_bucket].append(item)

    order = ("LOW_RISK", "MEDIUM_RISK", "HIGH_RISK", "EXTREME_RISK")
    summaries = []
    for bucket in order:
        items = by_bucket.get(bucket, [])
        draw_count = len(items)
        rows = draw_count * TOP_K
        draws_with_hit = sum(item.baseline_raw_hits > 0 for item in items)
        raw_hits = sum(item.baseline_raw_hits for item in items)
        hit_rate = draws_with_hit / draw_count if draw_count else 0.0
        random_expectation = (
            statistics.mean(item.random_draw_hit_probability for item in items)
            if items
            else 0.0
        )
        enrichment = hit_rate / random_expectation if random_expectation else 0.0
        summaries.append(
            {
                "risk_bucket": bucket,
                "rows": rows,
                "draws": draw_count,
                "draws_with_hit": draws_with_hit,
                "raw_hits": raw_hits,
                "hit_rate": hit_rate,
                "random_expectation": random_expectation,
                "enrichment_vs_random": enrichment,
            }
        )
    return summaries


def summarize_policy_results(policy_results: list[PolicyResult]) -> list[dict]:
    summaries = []
    baseline_high = [
        item
        for item in policy_results
        if item.policy == "BASELINE"
        and item.risk_bucket in {"HIGH_RISK", "EXTREME_RISK"}
    ]
    baseline_high_hit_rate = (
        sum(item.draws_with_hit or 0 for item in baseline_high) / len(baseline_high)
        if baseline_high
        else 0.0
    )

    for policy in POLICIES:
        all_items = [item for item in policy_results if item.policy == policy]
        high_items = [
            item
            for item in all_items
            if item.risk_bucket in {"HIGH_RISK", "EXTREME_RISK"}
        ]
        all_hit_rate = (
            sum(item.draws_with_hit or 0 for item in all_items) / len(all_items)
            if all_items
            else 0.0
        )
        high_hit_rate = (
            sum(item.draws_with_hit or 0 for item in high_items) / len(high_items)
            if high_items
            else 0.0
        )
        changed_draws = sum(
            item.selected_numbers
            != next(
                base.selected_numbers
                for base in policy_results
                if base.policy == "BASELINE"
                and base.source_draw_no == item.source_draw_no
                and base.target_draw_no == item.target_draw_no
            )
            for item in high_items
        )
        summaries.append(
            {
                "policy": policy,
                "all_draws": len(all_items),
                "all_draws_with_hit": sum(item.draws_with_hit or 0 for item in all_items),
                "all_raw_hits": sum(item.raw_hits or 0 for item in all_items),
                "all_hit_rate": all_hit_rate,
                "high_risk_draws": len(high_items),
                "high_risk_draws_with_hit": sum(
                    item.draws_with_hit or 0 for item in high_items
                ),
                "high_risk_raw_hits": sum(item.raw_hits or 0 for item in high_items),
                "high_risk_hit_rate": high_hit_rate,
                "delta_vs_baseline_high_risk": high_hit_rate - baseline_high_hit_rate,
                "changed_high_risk_draws": changed_draws,
                "limited_draws": sum(bool(item.limitation) for item in high_items),
            }
        )
    return summaries


def summarize_high_risk_features(candidate_rows: list[dict]) -> dict:
    high_rows = [
        row
        for row in candidate_rows
        if int(row["pre_miss_streak"]) >= HIGH_RISK_MIN_STREAK
    ]
    hit_rows = [row for row in high_rows if row["is_actual_hit"]]
    miss_rows = [row for row in high_rows if not row["is_actual_hit"]]

    def cohort(rows: list[dict]) -> dict:
        if not rows:
            return {
                "rows": 0,
                "avg_digit_sum": None,
                "avg_repeated_digit_count": None,
                "repeated_digit_rate": None,
                "avg_odd_ratio": None,
                "avg_high_digit_ratio": None,
                "pair_recurrence_rate": None,
                "avg_min_hamming": None,
                "avg_min_box_distance": None,
                "avg_min_mirror_hamming": None,
                "mirror_distribution": {},
                "top_first_pairs": [],
                "top_last_pairs": [],
            }

        return {
            "rows": len(rows),
            "avg_digit_sum": statistics.mean(row["digit_sum"] for row in rows),
            "avg_repeated_digit_count": statistics.mean(
                row["repeated_digit_count"] for row in rows
            ),
            "repeated_digit_rate": statistics.mean(
                row["repeated_digit_count"] > 0 for row in rows
            ),
            "avg_odd_ratio": statistics.mean(row["odd_ratio"] for row in rows),
            "avg_high_digit_ratio": statistics.mean(row["high_digit_ratio"] for row in rows),
            "pair_recurrence_rate": statistics.mean(row["pair_recurrence"] for row in rows),
            "avg_min_hamming": statistics.mean(row["min_hamming"] for row in rows),
            "avg_min_box_distance": statistics.mean(
                row["min_box_distance"] for row in rows
            ),
            "avg_min_mirror_hamming": statistics.mean(
                row["min_mirror_hamming"] for row in rows
            ),
            "mirror_distribution": dict(
                Counter(row["mirror_signature"] for row in rows).most_common(10)
            ),
            "top_first_pairs": Counter(row["first_pair"] for row in rows).most_common(10),
            "top_last_pairs": Counter(row["last_pair"] for row in rows).most_common(10),
        }

    return {
        "all_high_risk_rows": len(high_rows),
        "hit_rows": cohort(hit_rows),
        "miss_rows": cohort(miss_rows),
        "low_sample_warning": len(hit_rows) < 20,
    }


def generate_next_candidate_sets(
    current_pre_miss_streak: int,
    draws: dict[int, DrawRecord],
    prior_high_risk_hit_features: list[CandidateFeatures],
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]], dict[str, str | None], str]:
    target_draw_no = NEXT_SOURCE + 1
    try:
        result = run_existing_engine_prediction(
            PredictionRequest(draw_number=NEXT_SOURCE, mode="Current"),
            allow_fallback=False,
        )
        baseline = tuple(candidate.number for candidate in result.predictions[:TOP_K])
        target_draw_no = int(result.target_draw_number)
        source_label = "READ_ONLY_PRODUCTION_ADAPTER"
    except Exception as exc:
        baseline = ()
        source_label = f"UNAVAILABLE:{type(exc).__name__}"

    if len(baseline) != TOP_K:
        return baseline, {policy: baseline for policy in POLICIES}, {
            policy: "NO_BASELINE_CANDIDATE_SET" for policy in POLICIES
        }, source_label

    policy_sets, limitations = generate_policy_sets(
        baseline=baseline,
        pre_miss_streak=current_pre_miss_streak,
        draws=draws,
        source_draw_no=NEXT_SOURCE,
        prior_high_risk_hit_features=prior_high_risk_hit_features,
    )
    return baseline, policy_sets, limitations, f"{source_label}:{NEXT_SOURCE}->{target_draw_no}"


def format_pct(value: float) -> str:
    return f"{value * 100.0:8.3f}%"


def format_optional(value: float | None, decimals: int = 4) -> str:
    if value is None:
        return "NULL"
    return f"{value:.{decimals}f}"


def build_report(
    evaluated_pairs: list[EvaluatedPair],
    current_streak: int,
    baseline_summary: list[dict],
    feature_summary: dict,
    policy_summary: list[dict],
    policy_results: list[PolicyResult],
    next_baseline: tuple[str, ...],
    next_policy_sets: dict[str, tuple[str, ...]],
    next_limitations: dict[str, str | None],
    next_source_label: str,
    ledger_mismatch_count: int,
    next_target_available: bool,
) -> str:
    lines: list[str] = []
    width = 124
    current_bucket = risk_bucket(current_streak)

    lines.append("=" * width)
    lines.append("STEP 150E — STREAK RISK-GATED CANDIDATE PRUNING AUDIT — REPORT ONLY")
    lines.append("=" * width)
    lines.append("ProductionMathChanged: NO")
    lines.append("SQLSchemaChanged: NO")
    lines.append("FrontendChanged: NO")
    lines.append("ProductionPredictBehaviorChanged: NO")
    lines.append(f"Engine: {ENGINE}")
    lines.append(
        "TemporalFirewall: Candidate generation uses only earlier verified E1 rows and DrawHistory DrawNo <= source_draw_no."
    )
    lines.append(
        "EvaluationOnly: Target winners are applied only after each baseline/prototype candidate set is locked."
    )
    lines.append(
        "ReplacementPool: Prior winning numbers from the latest 12 draws ending at source_draw_no; no target rows."
    )
    lines.append(f"VerifiedOutcomePairs: {len(evaluated_pairs)}")
    lines.append(f"LedgerVsActualHitMismatchPairs: {ledger_mismatch_count}")
    lines.append("")

    lines.append("CURRENT FULL-LEDGER MISS STREAK STATE")
    lines.append("-" * width)
    lines.append(f"CurrentSourceForNextPrediction: {NEXT_SOURCE}")
    lines.append(f"CurrentTargetForNextPrediction: {NEXT_SOURCE + 1}")
    lines.append(f"CurrentFullLedgerPreMissStreak: {current_streak}")
    lines.append(f"CurrentRiskBucket: {current_bucket}")
    lines.append("LiveDeploymentLocalMissStreak: 3")
    lines.append(
        "Note: Full-ledger streak is the primary macro feature; live-local streak is context only."
    )
    lines.append("")

    lines.append("RISK BUCKET DEFINITIONS")
    lines.append("-" * width)
    lines.append("LOW_RISK: 0-5")
    lines.append("MEDIUM_RISK: 6-15")
    lines.append("HIGH_RISK: 16-40")
    lines.append("EXTREME_RISK: 41+")
    lines.append("")

    lines.append("CANDIDATE FEATURE DEFINITIONS")
    lines.append("-" * width)
    lines.append("DigitSum: Sum of four digits.")
    lines.append("RepeatedDigitCount: 4 minus the number of unique digits.")
    lines.append("OddRatio: Odd digits divided by 4. HighDigitRatio: Digits >= 5 divided by 4.")
    lines.append("MirrorSignature: Each digit reduced modulo 5. BoxSignature: Digits sorted ascending.")
    lines.append(
        "PairRecurrence: Candidate first or last pair appears in source/current-known winner history from the latest 3 draws."
    )
    lines.append(
        "NearMiss fields (Hamming, box, mirror) are calculated only after candidate lock against verified target actuals."
    )
    lines.append("")

    lines.append("BASELINE E1 TOP5 HIT RATE BY STREAK RISK BUCKET")
    lines.append("-" * width)
    lines.append(
        "Bucket         Rows   Draws  DrawsWithHit  RawHits  HitRate   RandomTop5  EnrichmentVsRandom"
    )
    for item in baseline_summary:
        lines.append(
            f"{item['risk_bucket']:<14} "
            f"{item['rows']:>6} "
            f"{item['draws']:>7} "
            f"{item['draws_with_hit']:>13} "
            f"{item['raw_hits']:>8} "
            f"{format_pct(item['hit_rate'])} "
            f"{format_pct(item['random_expectation'])} "
            f"{item['enrichment_vs_random']:>18.4f}"
        )
    lines.append("")

    lines.append("HIGH/EXTREME-RISK CANDIDATE SHAPE: SUCCESSFUL HIT ROWS VS MISS ROWS")
    lines.append("-" * width)
    if feature_summary["low_sample_warning"]:
        lines.append(
            "LOW_SAMPLE_WARNING: Fewer than 20 successful high-risk candidate rows exist; pattern estimates are sparse/noisy."
        )
    for label, key in (("HIT_ROWS", "hit_rows"), ("MISS_ROWS", "miss_rows")):
        cohort = feature_summary[key]
        lines.append(
            f"{label}: Rows={cohort['rows']} "
            f"AvgDigitSum={format_optional(cohort['avg_digit_sum'])} "
            f"AvgRepeatedDigits={format_optional(cohort['avg_repeated_digit_count'])} "
            f"RepeatedDigitRate={format_optional(cohort['repeated_digit_rate'])} "
            f"OddRatio={format_optional(cohort['avg_odd_ratio'])} "
            f"HighDigitRatio={format_optional(cohort['avg_high_digit_ratio'])} "
            f"PairRecurrenceRate={format_optional(cohort['pair_recurrence_rate'])}"
        )
        lines.append(
            f"  NearMissAfterLock: AvgHamming={format_optional(cohort['avg_min_hamming'])} "
            f"AvgBoxDistance={format_optional(cohort['avg_min_box_distance'])} "
            f"AvgMirrorHamming={format_optional(cohort['avg_min_mirror_hamming'])}"
        )
        lines.append(f"  MirrorDistributionTop10: {cohort['mirror_distribution']}")
        lines.append(f"  TopFirstPairs: {cohort['top_first_pairs']}")
        lines.append(f"  TopLastPairs: {cohort['top_last_pairs']}")
    lines.append("")

    lines.append("REPORT-ONLY PRUNING PROTOTYPE COMPARISON")
    lines.append("-" * width)
    lines.append(
        "Policy                       AllDraws  AllHitRate  AllRawHits  HighRiskDraws  HighRiskHitRate  HighRiskRawHits  DeltaVsBase  Changed  Limited"
    )
    for item in policy_summary:
        lines.append(
            f"{item['policy']:<28} "
            f"{item['all_draws']:>8} "
            f"{format_pct(item['all_hit_rate'])} "
            f"{item['all_raw_hits']:>11} "
            f"{item['high_risk_draws']:>14} "
            f"{format_pct(item['high_risk_hit_rate'])} "
            f"{item['high_risk_raw_hits']:>15} "
            f"{item['delta_vs_baseline_high_risk'] * 100.0:>10.3f}pp "
            f"{item['changed_high_risk_draws']:>8} "
            f"{item['limited_draws']:>8}"
        )
    lines.append("")
    lines.append(
        "PrototypeLimitation: Production E1 persists only Top5. Replacements here come from a leakage-safe recent historical winner pool,"
    )
    lines.append(
        "not from a persisted deeper E1 candidate pool. Any apparent gain must be treated as exploratory and non-production."
    )
    lines.append("")

    lines.append("RECENT LIVE WINDOW")
    lines.append("-" * width)
    baseline_index = {
        (item.source_draw_no, item.target_draw_no): item
        for item in policy_results
        if item.policy == "BASELINE"
    }
    policy_index = {
        (item.policy, item.source_draw_no, item.target_draw_no): item
        for item in policy_results
    }

    for source in LIVE_SOURCES:
        target = source + 1
        baseline_item = baseline_index.get((source, target))
        if baseline_item is None:
            lines.append(f"{source}->{target}: NO_VERIFIED_E1_LEDGER_PAIR")
            continue
        lines.append(
            f"{source}->{target} PreMiss={baseline_item.pre_miss_streak} Bucket={baseline_item.risk_bucket} "
            f"BASELINE={list(baseline_item.selected_numbers)} RawHits={baseline_item.raw_hits}"
        )
        for policy in POLICIES[1:]:
            item = policy_index[(policy, source, target)]
            lines.append(
                f"  {policy}: Candidates={list(item.selected_numbers)} RawHits={item.raw_hits} "
                f"Limitation={item.limitation or 'NONE'}"
            )

    lines.append(
        f"{NEXT_SOURCE}->{NEXT_SOURCE + 1} PreMiss={current_streak} Bucket={current_bucket} "
        f"VerificationStatus=UNVERIFIED CandidateSource={next_source_label}"
    )
    lines.append(
        f"  TargetAvailableInDrawHistory: {'YES' if next_target_available else 'NO'}"
    )
    lines.append(f"  BASELINE: Candidates={list(next_baseline)}")
    for policy in POLICIES[1:]:
        lines.append(
            f"  {policy}: Candidates={list(next_policy_sets.get(policy, ()))} "
            f"Limitation={next_limitations.get(policy) or 'NONE'}"
        )
    lines.append("")

    baseline_policy = next(
        item for item in policy_summary if item["policy"] == "BASELINE"
    )
    best_policy = max(
        (item for item in policy_summary if item["policy"] != "BASELINE"),
        key=lambda item: (
            item["high_risk_hit_rate"],
            item["high_risk_raw_hits"],
            -item["limited_draws"],
        ),
    )
    promote = (
        best_policy["high_risk_hit_rate"] > baseline_policy["high_risk_hit_rate"]
        and best_policy["high_risk_draws"] >= 100
        and not feature_summary["low_sample_warning"]
    )

    lines.append("FINAL RECOMMENDATION")
    lines.append("-" * width)
    lines.append(f"Recommendation: {'PROMOTE' if promote else 'DO NOT PROMOTE'}")
    if promote:
        lines.append(
            f"Reason: {best_policy['policy']} exceeds baseline high-risk hit rate with adequate evaluated coverage."
        )
        lines.append(
            "NextStep: Reproduce with a persisted deep candidate pool and a locked holdout before any production proposal."
        )
    else:
        reasons = []
        if best_policy["high_risk_hit_rate"] <= baseline_policy["high_risk_hit_rate"]:
            reasons.append("no prototype exceeds baseline high-risk draw hit rate")
        else:
            reasons.append(
                f"best apparent prototype {best_policy['policy']} improves high-risk hit rate by "
                f"{best_policy['delta_vs_baseline_high_risk'] * 100.0:.3f}pp but does not clear evidence guards"
            )
        if feature_summary["low_sample_warning"]:
            reasons.append("successful high-risk candidate rows are low-sample")
        reasons.append("replacement candidates are reconstructed from historical winners, not a persisted deep E1 pool")
        lines.append(f"Reason: {'; '.join(reasons)}.")
        lines.append(
            "NextStep: Persist a leakage-safe deeper E1 candidate pool, then rerun the same fixed gates on a locked forward holdout."
        )

    lines.append("")
    lines.append(f"REPORT_WRITTEN: {REPORT_PATH}")
    lines.append(f"JSONL_WRITTEN: {JSONL_PATH}")
    return "\n".join(lines)


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with get_conn() as connection:
        cursor = connection.cursor()
        draws = fetch_draws(cursor)
        ledger_pairs = fetch_e1_ledger_pairs(cursor)

    evaluated_pairs, current_streak = evaluate_pairs(ledger_pairs, draws)
    ledger_mismatch_count = sum(
        (item.ledger.ledger_hit_count > 0) != (item.baseline_raw_hits > 0)
        for item in evaluated_pairs
    )

    policy_results, candidate_rows, prior_high_risk_hit_features = run_online_backtest(
        evaluated_pairs,
        draws,
    )
    baseline_summary = summarize_baseline_by_bucket(evaluated_pairs)
    feature_summary = summarize_high_risk_features(candidate_rows)
    policy_summary = summarize_policy_results(policy_results)
    next_baseline, next_policy_sets, next_limitations, next_source_label = (
        generate_next_candidate_sets(
            current_pre_miss_streak=current_streak,
            draws=draws,
            prior_high_risk_hit_features=prior_high_risk_hit_features,
        )
    )
    next_target_available = (NEXT_SOURCE + 1) in draws

    report = build_report(
        evaluated_pairs=evaluated_pairs,
        current_streak=current_streak,
        baseline_summary=baseline_summary,
        feature_summary=feature_summary,
        policy_summary=policy_summary,
        policy_results=policy_results,
        next_baseline=next_baseline,
        next_policy_sets=next_policy_sets,
        next_limitations=next_limitations,
        next_source_label=next_source_label,
        ledger_mismatch_count=ledger_mismatch_count,
        next_target_available=next_target_available,
    )

    json_rows: list[dict] = []
    json_rows.extend(candidate_rows)
    json_rows.extend(
        {
            "record_type": "policy_result",
            **asdict(item),
        }
        for item in policy_results
    )
    json_rows.extend(
        {
            "record_type": "next_unverified_policy",
            "policy": policy,
            "source_draw_no": NEXT_SOURCE,
            "target_draw_no": NEXT_SOURCE + 1,
            "pre_miss_streak": current_streak,
            "risk_bucket": risk_bucket(current_streak),
            "selected_numbers": list(numbers),
            "limitation": next_limitations.get(policy),
            "candidate_source": next_source_label,
            "target_available_in_drawhistory": next_target_available,
            "candidate_history_max_draw_no": NEXT_SOURCE,
            "prior_training_max_target_draw_no": NEXT_SOURCE,
        }
        for policy, numbers in next_policy_sets.items()
    )

    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    with JSONL_PATH.open("w", encoding="utf-8") as handle:
        for row in json_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    print(report)


if __name__ == "__main__":
    main()
