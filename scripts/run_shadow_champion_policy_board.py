from __future__ import annotations

import itertools
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import pyodbc
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))
load_dotenv(BACKEND_ROOT / ".env")

from app.core.config import get_settings


ENGINE = "E1_TEMPORAL_CONTEXT_MATCH"
REPORT_PATH = PROJECT_ROOT / "reports" / "step_150f_shadow_champion_policy_board.txt"
JSONL_PATH = PROJECT_ROOT / "reports" / "step_150f_shadow_champion_policy_board_rows.jsonl"

TOP_K = 5
NUMBER_SPACE = 10_000
REPLACEMENT_HISTORY_DRAWS = 12
MIN_PRIOR_HIGH_RISK_HITS = 5
POLICIES = (
    "BASELINE_E1",
    "RISK_SUM_BAND_SHADOW",
    "PAIR_RECURRENCE_SHADOW",
    "MIRROR_BOX_REPAIR_SHADOW",
    "STREAK_RISK_SUPPRESSION_SHADOW",
    "HYBRID_GUARD_SHADOW",
)
MODE_PRIORITY = {
    "Current": 0,
    "Temporal_Global_Loop": 1,
    "Historical": 2,
    "Engine_Grand_Loop": 3,
    "Grand_Loop": 4,
    "Weighted_Grand_Loop": 5,
}
LIVE_SOURCES = (5494, 5495, 5496)


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
    sum_band: str
    mirror: str
    box: str
    first_pair: str
    last_pair: str
    repeated_digits: int
    odd_ratio: float
    high_ratio: float
    pair_1: bool
    pair_3: bool
    pair_5: bool
    recency: float


@dataclass(frozen=True)
class Candidate:
    number: str
    family: str
    score: float
    source: str


@dataclass(frozen=True)
class PolicySelection:
    numbers: tuple[str, ...]
    families: tuple[str, ...]
    sources: tuple[str, ...]
    limitation: str | None
    replacements: int
    could_not_replace: bool
    low_sample: bool


@dataclass(frozen=True)
class BoardRow:
    source: int
    target: int
    mode: str
    target_month: int | None
    target_day_type: str | None
    runtime_bucket: str
    pre_miss_streak: int
    risk_bucket: str
    actuals: tuple[str, ...]
    actual_prize_count: int
    policy: str
    numbers: tuple[str, ...]
    families: tuple[str, ...]
    sources: tuple[str, ...]
    verified: bool
    raw_hits: int | None
    draws_with_hit: int | None
    random_hit_probability: float | None
    expected_raw_hits: float | None
    limitation: str | None
    low_sample: bool
    replacements: int
    could_not_replace: bool
    candidate_history_max_draw_no: int
    prior_training_max_target_draw_no: int
    min_circular_distance: int | None
    min_hamming_distance: int | None
    max_box_overlap: int | None
    mirror_exact_hidden_count: int | None


class OnlineState:
    def __init__(self) -> None:
        self.high_risk_hit_features: list[Feature] = []
        self.risk_shape_trials: dict[str, Counter[tuple]] = defaultdict(Counter)
        self.risk_shape_hits: dict[str, Counter[tuple]] = defaultdict(Counter)
        self.repair_success: Counter[tuple[str, str]] = Counter()
        self.observed_max_target = 0

    @staticmethod
    def shape_keys(feature: Feature) -> tuple[tuple, ...]:
        return (
            ("sum_band", feature.sum_band),
            ("repeat", feature.repeated_digits),
            ("odd", round(feature.odd_ratio, 2)),
            ("high", round(feature.high_ratio, 2)),
            ("mirror", feature.mirror),
        )

    def suppression_score(self, bucket: str, feature: Feature) -> float:
        scores = []
        for key in self.shape_keys(feature):
            trials = self.risk_shape_trials[bucket][key]
            hits = self.risk_shape_hits[bucket][key]
            scores.append((hits + 0.5) / (trials + 25.0))
        return statistics.mean(scores) if scores else 0.0

    def observe_baseline(
        self,
        bucket: str,
        baseline: tuple[str, ...],
        features: dict[str, Feature],
        actuals: tuple[str, ...],
        target: int,
    ) -> None:
        actual_set = set(actuals)
        for number in baseline:
            feature = features[number]
            for key in self.shape_keys(feature):
                self.risk_shape_trials[bucket][key] += 1
                if number in actual_set:
                    self.risk_shape_hits[bucket][key] += 1
            if bucket in {"HIGH_RISK", "EXTREME_RISK"} and number in actual_set:
                self.high_risk_hit_features.append(feature)

            if number not in actual_set:
                for actual in actuals:
                    if feature.mirror == mirror_signature(actual):
                        self.repair_success[("MIRROR_REPAIR", feature.mirror)] += 1
                    if feature.box == box_signature(actual):
                        self.repair_success[("BOX_REPAIR", feature.box)] += 1
        self.observed_max_target = max(self.observed_max_target, target)


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


def sum_band(value: int) -> str:
    if value <= 9:
        return "00_09"
    if value <= 14:
        return "10_14"
    if value <= 19:
        return "15_19"
    if value <= 24:
        return "20_24"
    return "25_36"


def mirror_signature(number: str) -> str:
    return "".join(str(value % 5) for value in digits(number))


def box_signature(number: str) -> str:
    return "".join(sorted(z4(number)))


def repeated_digits(number: str) -> int:
    normalized = z4(number)
    return len(normalized) - len(set(normalized))


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
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def hamming(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(z4(left), z4(right)))


def circular_distance(left: str, right: str) -> int:
    delta = abs(int(z4(left)) - int(z4(right)))
    return min(delta, NUMBER_SPACE - delta)


def box_overlap(left: str, right: str) -> int:
    return sum((Counter(z4(left)) & Counter(z4(right))).values())


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


def fetch_ledger(cursor, verified_only: bool) -> list[LedgerPair]:
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
          AND RankNo BETWEEN 1 AND 5
        ORDER BY SourceDrawNo, TargetDrawNo, Mode, RankNo;
        """,
        ENGINE,
    ).fetchall()
    grouped: dict[tuple[str, int, int], list] = defaultdict(list)
    for row in rows:
        if verified_only and str(row.VerificationStatus) != "Verified":
            continue
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
    pairs: set[str] = set()
    eligible = sorted(no for no in draws if no <= source)[-depth:]
    for draw_no in eligible:
        for number in draws[draw_no].winners:
            pairs.update((number[:2], number[2:]))
    return pairs


def source_profile(draws: dict[int, Draw], source: int) -> dict:
    draw = draws.get(source)
    winners = draw.winners if draw else ()
    clouds = {depth: pair_cloud(draws, source, depth) for depth in (1, 3, 5)}
    sums = Counter(digit_sum(number) for number in winners)
    mirrors = Counter(mirror_signature(number) for number in winners)
    return {
        "digit_sum_distribution": dict(sorted(sums.items())),
        "mirror_class_distribution": dict(mirrors),
        "first_pair_cloud": sorted({number[:2] for number in winners}),
        "last_pair_cloud": sorted({number[2:] for number in winners}),
        "repeated_digit_rate": (
            statistics.mean(repeated_digits(number) > 0 for number in winners)
            if winners
            else 0.0
        ),
        "odd_even_ratio": (
            statistics.mean(sum(value % 2 for value in digits(number)) / 4 for number in winners)
            if winners
            else 0.0
        ),
        "high_low_ratio": (
            statistics.mean(sum(value >= 5 for value in digits(number)) / 4 for number in winners)
            if winners
            else 0.0
        ),
        "pair_recurrence_1": len(clouds[1]),
        "pair_recurrence_3": len(clouds[3]),
        "pair_recurrence_5": len(clouds[5]),
    }


def historical_pool(
    draws: dict[int, Draw], source: int, baseline: tuple[str, ...]
) -> tuple[dict[str, str], Counter[str]]:
    origins = {number: "BASELINE_E1" for number in baseline}
    recency: Counter[str] = Counter()
    eligible = sorted(no for no in draws if no <= source)[-REPLACEMENT_HISTORY_DRAWS:]
    for draw_no in eligible:
        weight = 1.0 / (1 + source - draw_no)
        for number in draws[draw_no].winners:
            origins.setdefault(number, f"HISTORICAL_WINNER_DRAW_{draw_no}")
            recency[number] += weight
    return origins, recency


def feature_map(
    numbers: Iterable[str], draws: dict[int, Draw], source: int, recency: Counter[str]
) -> dict[str, Feature]:
    cloud_1 = pair_cloud(draws, source, 1)
    cloud_3 = pair_cloud(draws, source, 3)
    cloud_5 = pair_cloud(draws, source, 5)
    output = {}
    for raw in numbers:
        number = z4(raw)
        first, last = number[:2], number[2:]
        values = digits(number)
        total = digit_sum(number)
        output[number] = Feature(
            number=number,
            digit_sum=total,
            sum_band=sum_band(total),
            mirror=mirror_signature(number),
            box=box_signature(number),
            first_pair=first,
            last_pair=last,
            repeated_digits=repeated_digits(number),
            odd_ratio=sum(value % 2 for value in values) / 4,
            high_ratio=sum(value >= 5 for value in values) / 4,
            pair_1=first in cloud_1 or last in cloud_1,
            pair_3=first in cloud_3 or last in cloud_3,
            pair_5=first in cloud_5 or last in cloud_5,
            recency=float(recency.get(number, 0.0)),
        )
    return output


def make_selection(
    ranked: Iterable[Candidate],
    baseline: tuple[str, ...],
    limitation: str | None = None,
    low_sample: bool = False,
    diversity: bool = False,
) -> PolicySelection:
    candidates = list(ranked)
    baseline_candidates = [
        Candidate(number, "BASELINE_E1", -float(rank), f"stored_rank_{rank}")
        for rank, number in enumerate(baseline, start=1)
    ]
    candidates.extend(baseline_candidates)

    selected: list[Candidate] = []
    seen: set[str] = set()
    family_counts: Counter[str] = Counter()
    if diversity:
        available_families = {item.family for item in candidates}
        for family in sorted(available_families):
            family_items = sorted(
                (item for item in candidates if item.family == family),
                key=lambda item: (-item.score, item.number),
            )
            if family_items and len(selected) < 3:
                item = family_items[0]
                if item.number not in seen:
                    selected.append(item)
                    seen.add(item.number)
                    family_counts[item.family] += 1

    for item in sorted(candidates, key=lambda candidate: (-candidate.score, candidate.number)):
        if item.number in seen:
            continue
        if diversity and family_counts[item.family] >= 2:
            continue
        selected.append(item)
        seen.add(item.number)
        family_counts[item.family] += 1
        if len(selected) == TOP_K:
            break

    if len(selected) < TOP_K:
        for item in baseline_candidates:
            if item.number not in seen:
                selected.append(item)
                seen.add(item.number)
            if len(selected) == TOP_K:
                break

    selected = selected[:TOP_K]
    replacements = sum(item.number not in baseline for item in selected)
    could_not_replace = len(selected) < TOP_K or (
        limitation is not None and "NO_SAFE_REPLACEMENT" in limitation
    )
    if diversity and len({item.family for item in selected}) < 3:
        detail = f"DIVERSITY_UNAVAILABLE_DISTINCT_FAMILIES={len({item.family for item in selected})}"
        limitation = f"{limitation};{detail}" if limitation else detail

    return PolicySelection(
        numbers=tuple(item.number for item in selected),
        families=tuple(item.family for item in selected),
        sources=tuple(item.source for item in selected),
        limitation=limitation,
        replacements=replacements,
        could_not_replace=could_not_replace,
        low_sample=low_sample,
    )


def baseline_selection(baseline: tuple[str, ...]) -> PolicySelection:
    return PolicySelection(
        numbers=baseline,
        families=tuple("BASELINE_E1" for _ in baseline),
        sources=tuple(f"stored_rank_{rank}" for rank in range(1, len(baseline) + 1)),
        limitation=None,
        replacements=0,
        could_not_replace=False,
        low_sample=False,
    )


def generate_policy_board(
    baseline: tuple[str, ...],
    source: int,
    streak: int,
    draws: dict[int, Draw],
    state: OnlineState,
) -> dict[str, PolicySelection]:
    board = {"BASELINE_E1": baseline_selection(baseline)}
    bucket = risk_bucket(streak)
    if bucket not in {"HIGH_RISK", "EXTREME_RISK"}:
        for policy in POLICIES[1:]:
            selection = baseline_selection(baseline)
            board[policy] = PolicySelection(
                **{
                    **asdict(selection),
                    "limitation": "NO_OP_BELOW_HIGH_RISK",
                }
            )
        return board

    origins, recency = historical_pool(draws, source, baseline)
    features = feature_map(origins, draws, source, recency)
    prior_hits = state.high_risk_hit_features
    low_sample = len(prior_hits) < MIN_PRIOR_HIGH_RISK_HITS

    if low_sample:
        board["RISK_SUM_BAND_SHADOW"] = PolicySelection(
            **{
                **asdict(baseline_selection(baseline)),
                "limitation": f"LOW_SAMPLE_NO_OP_PRIOR_HIGH_RISK_HITS={len(prior_hits)}",
                "low_sample": True,
            }
        )
    else:
        prior_sums = [item.digit_sum for item in prior_hits]
        low = math.floor(percentile(prior_sums, 0.20))
        high = math.ceil(percentile(prior_sums, 0.80))
        median = statistics.median(prior_sums)
        ranked = [
            Candidate(
                number=number,
                family="SUM_BAND_BASELINE" if number in baseline else "SUM_BAND_HISTORICAL",
                score=(
                    8.0 * (low <= feature.digit_sum <= high)
                    - abs(feature.digit_sum - median) / 10
                    + feature.recency
                    + (2.0 if number in baseline else 0.0)
                ),
                source=origins[number],
            )
            for number, feature in features.items()
        ]
        board["RISK_SUM_BAND_SHADOW"] = make_selection(
            ranked,
            baseline,
            limitation=f"EMPIRICAL_HIGH_RISK_SUM_BAND={low}-{high};PRIOR_HITS={len(prior_hits)}",
        )

    pair_ranked = [
        Candidate(
            number=number,
            family=(
                "PAIR_RECURRENCE_BASELINE" if number in baseline else "PAIR_RECURRENCE_HISTORICAL"
            ),
            score=(
                5.0 * feature.pair_1
                + 3.0 * feature.pair_3
                + 1.5 * feature.pair_5
                + feature.recency
                + (1.0 if number in baseline else 0.0)
            ),
            source=origins[number],
        )
        for number, feature in features.items()
        if feature.pair_5
    ]
    board["PAIR_RECURRENCE_SHADOW"] = make_selection(
        pair_ranked,
        baseline,
        limitation=None if pair_ranked else "NO_SAFE_REPLACEMENT_PAIR_CLOUD_EMPTY",
    )

    repair_pool: dict[str, Candidate] = {}
    for base_rank, base in enumerate(baseline, start=1):
        base_feature = features[base]
        mirror_variants = []
        for position in range(4):
            chars = list(base)
            chars[position] = str((int(chars[position]) + 5) % 10)
            mirror_variants.append("".join(chars))
        for candidate in mirror_variants[:4]:
            score = (
                3.0
                + state.repair_success[("MIRROR_REPAIR", base_feature.mirror)]
                + 1.0 / base_rank
            )
            old = repair_pool.get(candidate)
            item = Candidate(candidate, "MIRROR_REPAIR", score, f"single_mirror_flip_of_{base}")
            if old is None or item.score > old.score:
                repair_pool[candidate] = item

        permutations = sorted(set("".join(item) for item in itertools.permutations(base)))
        for candidate in permutations:
            if candidate == base:
                continue
            score = (
                2.0
                + state.repair_success[("BOX_REPAIR", base_feature.box)]
                + 1.0 / base_rank
            )
            old = repair_pool.get(candidate)
            item = Candidate(candidate, "BOX_REPAIR", score, f"capped_box_perm_of_{base}")
            if old is None or item.score > old.score:
                repair_pool[candidate] = item
    repair_ranked = sorted(
        repair_pool.values(), key=lambda item: (-item.score, item.family, item.number)
    )[:30]
    board["MIRROR_BOX_REPAIR_SHADOW"] = make_selection(
        repair_ranked,
        baseline,
        limitation="STRICT_REPAIR_CAP=30;BASELINE_DERIVED_ONLY",
    )

    suppression_ranked = [
        Candidate(
            number=number,
            family=(
                "RISK_SHAPE_BASELINE" if number in baseline else "RISK_SHAPE_HISTORICAL"
            ),
            score=(
                state.suppression_score(bucket, feature) * 1000
                + feature.recency
                + (0.75 if number in baseline else 0.0)
            ),
            source=origins[number],
        )
        for number, feature in features.items()
    ]
    risk_trials = sum(state.risk_shape_trials[bucket].values())
    suppression_low_sample = risk_trials < 100
    board["STREAK_RISK_SUPPRESSION_SHADOW"] = make_selection(
        suppression_ranked,
        baseline,
        limitation=(
            f"LOW_SAMPLE_RISK_SHAPE_TRIALS={risk_trials}" if suppression_low_sample else None
        ),
        low_sample=suppression_low_sample,
    )

    hybrid_candidates: list[Candidate] = []
    for policy in (
        "RISK_SUM_BAND_SHADOW",
        "PAIR_RECURRENCE_SHADOW",
        "MIRROR_BOX_REPAIR_SHADOW",
        "STREAK_RISK_SUPPRESSION_SHADOW",
    ):
        selection = board[policy]
        for rank, (number, family, source_label) in enumerate(
            zip(selection.numbers, selection.families, selection.sources), start=1
        ):
            hybrid_candidates.append(
                Candidate(number, f"HYBRID::{family}", 10.0 - rank, source_label)
            )
    board["HYBRID_GUARD_SHADOW"] = make_selection(
        hybrid_candidates,
        baseline,
        limitation="MAX_2_PER_FAMILY;MIN_3_DISTINCT_IF_AVAILABLE",
        low_sample=low_sample or suppression_low_sample,
        diversity=True,
    )
    return board


def near_miss(numbers: tuple[str, ...], actuals: tuple[str, ...]) -> dict[str, int | None]:
    if not actuals:
        return {
            "min_circular_distance": None,
            "min_hamming_distance": None,
            "max_box_overlap": None,
            "mirror_exact_hidden_count": None,
        }
    pairs = [(candidate, actual) for candidate in numbers for actual in actuals]
    return {
        "min_circular_distance": min(circular_distance(a, b) for a, b in pairs),
        "min_hamming_distance": min(hamming(a, b) for a, b in pairs),
        "max_box_overlap": max(box_overlap(a, b) for a, b in pairs),
        "mirror_exact_hidden_count": sum(
            mirror_signature(a) == mirror_signature(b) and a != b for a, b in pairs
        ),
    }


def make_board_row(
    pair: LedgerPair,
    draw: Draw | None,
    streak: int,
    policy: str,
    selection: PolicySelection,
    prior_max_target: int,
) -> BoardRow:
    actuals = draw.winners if draw else ()
    verified = draw is not None
    actual_set = set(actuals)
    raw_hits = sum(number in actual_set for number in selection.numbers) if verified else None
    prize_count = len(actual_set)
    random_probability = (
        1 - ((NUMBER_SPACE - prize_count) / NUMBER_SPACE) ** TOP_K if verified else None
    )
    expected_raw = TOP_K * prize_count / NUMBER_SPACE if verified else None
    distances = near_miss(selection.numbers, actuals)
    return BoardRow(
        source=pair.source,
        target=pair.target,
        mode=pair.mode,
        target_month=draw.month if draw else None,
        target_day_type=draw.day_type if draw else None,
        runtime_bucket=runtime_bucket(draw.day_type if draw else None),
        pre_miss_streak=streak,
        risk_bucket=risk_bucket(streak),
        actuals=actuals,
        actual_prize_count=prize_count,
        policy=policy,
        numbers=selection.numbers,
        families=selection.families,
        sources=selection.sources,
        verified=verified,
        raw_hits=raw_hits,
        draws_with_hit=int(bool(raw_hits)) if verified else None,
        random_hit_probability=random_probability,
        expected_raw_hits=expected_raw,
        limitation=selection.limitation,
        low_sample=selection.low_sample,
        replacements=selection.replacements,
        could_not_replace=selection.could_not_replace,
        candidate_history_max_draw_no=pair.source,
        prior_training_max_target_draw_no=prior_max_target,
        **distances,
    )


def run_backtest(
    pairs: list[LedgerPair], draws: dict[int, Draw]
) -> tuple[list[BoardRow], int, int]:
    rows: list[BoardRow] = []
    state = OnlineState()
    streak = 0
    mismatch_count = 0
    for pair in pairs:
        target_draw = draws.get(pair.target)
        if target_draw is None:
            continue
        board = generate_policy_board(pair.predictions, pair.source, streak, draws, state)
        prior_max = state.observed_max_target
        pair_rows = [
            make_board_row(pair, target_draw, streak, policy, board[policy], prior_max)
            for policy in POLICIES
        ]
        rows.extend(pair_rows)

        baseline_row = pair_rows[0]
        mismatch_count += (pair.ledger_hit_count > 0) != bool(baseline_row.raw_hits)
        baseline_features = feature_map(
            pair.predictions, draws, pair.source, Counter()
        )
        state.observe_baseline(
            risk_bucket(streak),
            pair.predictions,
            baseline_features,
            target_draw.winners,
            pair.target,
        )
        streak = 0 if pair.ledger_hit_count > 0 else streak + 1
    return rows, streak, mismatch_count


def load_live_baseline(
    latest_source: int, all_ledger: list[LedgerPair]
) -> tuple[LedgerPair, str]:
    stored = [
        pair
        for pair in all_ledger
        if pair.source == latest_source and pair.target == latest_source + 1
    ]
    if stored:
        return stored[0], "STORED_PREDICTION_LEDGER"

    from app.core.ml_adapter import run_existing_engine_prediction
    from app.schemas.prediction import PredictionRequest

    result = run_existing_engine_prediction(
        PredictionRequest(draw_number=latest_source, mode="Current"),
        allow_fallback=False,
    )
    predictions = tuple(item.number for item in result.predictions[:TOP_K])
    if len(predictions) != TOP_K:
        raise RuntimeError("Read-only production adapter did not return exactly five candidates")
    return (
        LedgerPair(
            mode="Current",
            source=latest_source,
            target=int(result.target_draw_number),
            predictions=predictions,
            ledger_hit_count=0,
        ),
        "READ_ONLY_PRODUCTION_ADAPTER",
    )


def replay_state(
    pairs: list[LedgerPair], draws: dict[int, Draw]
) -> tuple[OnlineState, int]:
    state = OnlineState()
    streak = 0
    for pair in pairs:
        target = draws.get(pair.target)
        if target is None:
            continue
        features = feature_map(pair.predictions, draws, pair.source, Counter())
        state.observe_baseline(
            risk_bucket(streak), pair.predictions, features, target.winners, pair.target
        )
        streak = 0 if pair.ledger_hit_count > 0 else streak + 1
    return state, streak


def summarize(rows: list[BoardRow], predicate: Callable[[BoardRow], bool]) -> list[dict]:
    selected = [row for row in rows if row.verified and predicate(row)]
    baseline = [row for row in selected if row.policy == "BASELINE_E1"]
    baseline_rate = (
        sum(row.draws_with_hit or 0 for row in baseline) / len(baseline) if baseline else 0.0
    )
    output = []
    for policy in POLICIES:
        items = [row for row in selected if row.policy == policy]
        count = len(items)
        draws_with_hit = sum(row.draws_with_hit or 0 for row in items)
        raw_hits = sum(row.raw_hits or 0 for row in items)
        hit_rate = draws_with_hit / count if count else 0.0
        random_rate = (
            statistics.mean(row.random_hit_probability or 0.0 for row in items)
            if items
            else 0.0
        )
        output.append(
            {
                "policy": policy,
                "rows": count,
                "draws_with_hit": draws_with_hit,
                "raw_hits": raw_hits,
                "hit_rate": hit_rate,
                "raw_hits_per_draw": raw_hits / count if count else 0.0,
                "random_hit_rate": random_rate,
                "enrichment": hit_rate / random_rate if random_rate else 0.0,
                "delta_vs_baseline": hit_rate - baseline_rate,
                "low_sample": sum(row.low_sample for row in items),
                "replacements": sum(row.replacements for row in items),
                "could_not_replace": sum(row.could_not_replace for row in items),
            }
        )
    return output


def table(lines: list[str], title: str, summary: list[dict], width: int) -> None:
    lines.extend(("", title, "-" * width))
    lines.append(
        "Policy                              Rows  HitDraws RawHits HitRate Raw/Draw Random  Enrich DeltaBase LowSmpl Repl NoSafe"
    )
    for item in summary:
        lines.append(
            f"{item['policy']:<35} {item['rows']:>5} {item['draws_with_hit']:>9} "
            f"{item['raw_hits']:>7} {item['hit_rate'] * 100:>6.3f}% "
            f"{item['raw_hits_per_draw']:>8.4f} {item['random_hit_rate'] * 100:>6.3f}% "
            f"{item['enrichment']:>7.3f} {item['delta_vs_baseline'] * 100:>+8.3f}pp "
            f"{item['low_sample']:>7} {item['replacements']:>4} {item['could_not_replace']:>6}"
        )


def promotion_decision(summaries: dict[str, list[dict]]) -> tuple[str, str]:
    by_window = {
        window: {item["policy"]: item for item in summary}
        for window, summary in summaries.items()
    }
    candidates = []
    for policy in POLICIES[1:]:
        full = by_window["FULL_VERIFIED"][policy]
        high = by_window["HIGH_RISK_ONLY"][policy]
        recent_90 = by_window["RECENT_90"][policy]
        recent_47 = by_window["RECENT_47"][policy]
        broad_gain = full["rows"] >= 100 and full["delta_vs_baseline"] > 0
        risk_gain = high["rows"] >= 30 and high["delta_vs_baseline"] > 0
        recent_safe = (
            recent_90["delta_vs_baseline"] >= 0
            and recent_47["delta_vs_baseline"] >= 0
        )
        replacement_guard = policy not in {
            "RISK_SUM_BAND_SHADOW",
            "PAIR_RECURRENCE_SHADOW",
            "STREAK_RISK_SUPPRESSION_SHADOW",
        } or full["replacements"] <= full["rows"]
        if (broad_gain or risk_gain) and recent_safe and replacement_guard:
            candidates.append(policy)
    if candidates:
        return (
            "DO NOT PROMOTE",
            "Future shadow prototype candidate(s): "
            + ", ".join(candidates)
            + ". Production promotion remains blocked because the observed edge is small and reconstructed replacement sources remain shadow-only.",
        )
    return (
        "DO NOT PROMOTE",
        "No shadow policy clears the combined performance, recent-window, sample-size, and replacement-source guards.",
    )


def build_report(
    rows: list[BoardRow],
    current_row_set: list[BoardRow],
    current_source_label: str,
    mismatch_count: int,
    source_profiles: dict[int, dict],
) -> str:
    width = 140
    verified_baseline = [row for row in rows if row.policy == "BASELINE_E1"]
    ordered_sources = sorted({row.source for row in verified_baseline})
    recent_365 = set(ordered_sources[-365:])
    recent_90 = set(ordered_sources[-90:])
    recent_47 = set(ordered_sources[-47:])
    filters: dict[str, Callable[[BoardRow], bool]] = {
        "FULL_VERIFIED": lambda row: True,
        "PHASE_2_SOURCE_GE_4050": lambda row: row.source >= 4050,
        "RECENT_365": lambda row: row.source in recent_365,
        "RECENT_90": lambda row: row.source in recent_90,
        "RECENT_47": lambda row: row.source in recent_47,
        "HIGH_RISK_ONLY": lambda row: row.risk_bucket
        in {"HIGH_RISK", "EXTREME_RISK"},
        "HIGH_RISK_MONTH_JUNE": lambda row: row.risk_bucket
        in {"HIGH_RISK", "EXTREME_RISK"}
        and row.target_month == 6,
        "HIGH_RISK_WEEKEND_SPACE": lambda row: row.risk_bucket
        in {"HIGH_RISK", "EXTREME_RISK"}
        and row.runtime_bucket == "WEEKEND_SPACE",
        "HIGH_RISK_MIDWEEK_SPECIAL_SPACE": lambda row: row.risk_bucket
        in {"HIGH_RISK", "EXTREME_RISK"}
        and row.runtime_bucket == "MIDWEEK_SPECIAL_SPACE",
    }
    summaries = {name: summarize(rows, predicate) for name, predicate in filters.items()}
    current = current_row_set[0]
    decision, decision_reason = promotion_decision(summaries)

    lines = [
        "=" * width,
        "STEP 150F — SHADOW CHAMPION POLICY BOARD — REPORT ONLY",
        "=" * width,
        "ProductionMathChanged: NO",
        "APIChanged: NO",
        "FrontendChanged: NO",
        "SQLSchemaChanged: NO",
        "DeploymentChanged: NO",
        f"Engine: {ENGINE}",
        "TemporalFirewall: Every policy uses DrawHistory DrawNo <= SourceDrawNo and policy priors with TargetDrawNo <= SourceDrawNo.",
        "CandidateLock: Target winners are read only after all policy Top5 sets for that source are fixed.",
        "SQLAuthority: Stored PredictionLedger E1 Top5 and verification status drive the historical ledger and miss streak.",
        "ReplacementWarning: Historical winner candidates are leakage-safe but shadow-only; production does not persist a deeper E1 pool.",
        f"VerifiedOutcomePairs: {len(verified_baseline)}",
        f"LedgerVsActualHitMismatchPairs: {mismatch_count}",
        "",
        "CURRENT LIVE MACRO STATE",
        "-" * width,
        f"CurrentSourceDrawNo: {current.source}",
        f"CurrentTargetDrawNo: {current.target}",
        f"CurrentFullLedgerPreMissStreak: {current.pre_miss_streak}",
        f"CurrentRiskBucket: {current.risk_bucket}",
        f"CurrentTargetStatus: {'VERIFIED' if current.verified else 'UNVERIFIED'}",
        f"CurrentBaselineSource: {current_source_label}",
        "RiskUse: suppression/caution only; no streak recovery boost.",
        "",
        "SOURCE STRUCTURAL FEATURE CONTRACT",
        "-" * width,
        "Features: digit-sum distribution, mirror-class distribution, first/last pair cloud, repeated-digit rate, odd/even ratio,",
        "high/low ratio, and pair recurrence over prior 1/3/5 source-known draws.",
        f"CurrentSourceProfile: {json.dumps(source_profiles[current.source], sort_keys=True)}",
    ]

    title_map = {
        "FULL_VERIFIED": "POLICY PERFORMANCE SUMMARY — FULL VERIFIED RANGE",
        "PHASE_2_SOURCE_GE_4050": "POLICY PERFORMANCE SUMMARY — PHASE 2 (SOURCE >= 4050)",
        "RECENT_365": "POLICY PERFORMANCE SUMMARY — RECENT 365",
        "RECENT_90": "POLICY PERFORMANCE SUMMARY — RECENT 90",
        "RECENT_47": "POLICY PERFORMANCE SUMMARY — RECENT 47",
        "HIGH_RISK_ONLY": "POLICY PERFORMANCE SUMMARY — HIGH_RISK / EXTREME_RISK ONLY",
        "HIGH_RISK_MONTH_JUNE": "POLICY PERFORMANCE SUMMARY — HIGH_RISK + MONTH JUNE",
        "HIGH_RISK_WEEKEND_SPACE": "POLICY PERFORMANCE SUMMARY — HIGH_RISK + WEEKEND_SPACE",
        "HIGH_RISK_MIDWEEK_SPECIAL_SPACE": "POLICY PERFORMANCE SUMMARY — HIGH_RISK + MIDWEEK_SPECIAL_SPACE",
    }
    for name in filters:
        table(lines, title_map[name], summaries[name], width)

    lines.extend(("", "EXACT LIVE WINDOW — 5494→5497", "-" * width))
    index = {(row.policy, row.source, row.target): row for row in rows}
    for source in LIVE_SOURCES:
        baseline = index.get(("BASELINE_E1", source, source + 1))
        if baseline is None:
            lines.append(f"{source}->{source + 1}: NO_VERIFIED_E1_PAIR")
            continue
        lines.append(
            f"{source}->{source + 1} Risk={baseline.risk_bucket} PreMiss={baseline.pre_miss_streak} "
            f"ActualPrizeCount={baseline.actual_prize_count} Baseline={list(baseline.numbers)}"
        )
        for policy in POLICIES:
            row = index[(policy, source, source + 1)]
            lines.append(
                f"  {policy}: Top5={list(row.numbers)} Families={list(row.families)} Hits={row.raw_hits} "
                f"MinCircular={row.min_circular_distance} MinHamming={row.min_hamming_distance} "
                f"MaxBoxOverlap={row.max_box_overlap} MirrorExactHidden={row.mirror_exact_hidden_count} "
                f"Limitation={row.limitation or 'NONE'}"
            )

    lines.extend(
        (
            "",
            f"CURRENT UNVERIFIED {current.source}→{current.target} CANDIDATE BOARD",
            "-" * width,
            f"Status: {'VERIFIED' if current.verified else 'UNVERIFIED'}",
            "No target verification is performed when the target is absent from DrawHistory.",
        )
    )
    for row in current_row_set:
        lines.append(
            f"{row.policy}: Top5={list(row.numbers)} Families={list(row.families)} "
            f"Sources={list(row.sources)} Limitation={row.limitation or 'NONE'}"
        )

    lines.extend(
        (
            "",
            "PROMOTION GUARD REVIEW",
            "-" * width,
            "Minimum broad claim rows: 100. Minimum risk-bucket claim rows: 30.",
            "Recent-90 and recent-47 may not underperform baseline.",
            "Historical-winner replacement dependence remains shadow-only.",
            "Temporal firewall and candidate-lock checks: PASS by construction; candidate/prior maxima are emitted per JSONL row.",
            "",
            "FINAL PROMOTION DECISION",
            "-" * width,
            f"Decision: {decision}",
            f"Reason: {decision_reason}",
            "Production action: NONE.",
            "",
            f"REPORT_WRITTEN: {REPORT_PATH}",
            f"JSONL_WRITTEN: {JSONL_PATH}",
        )
    )
    return "\n".join(lines)


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_conn() as connection:
        cursor = connection.cursor()
        draws = fetch_draws(cursor)
        verified_pairs = fetch_ledger(cursor, verified_only=True)
        all_pairs = fetch_ledger(cursor, verified_only=False)

    rows, current_streak, mismatch_count = run_backtest(verified_pairs, draws)
    state, replay_streak = replay_state(verified_pairs, draws)
    if replay_streak != current_streak:
        raise RuntimeError("Streak replay mismatch")

    latest_source = max(draws)
    live_pair, live_source_label = load_live_baseline(latest_source, all_pairs)
    live_board = generate_policy_board(
        live_pair.predictions, live_pair.source, current_streak, draws, state
    )
    live_draw = draws.get(live_pair.target)
    live_rows = [
        make_board_row(
            live_pair,
            live_draw,
            current_streak,
            policy,
            live_board[policy],
            state.observed_max_target,
        )
        for policy in POLICIES
    ]

    profile_sources = set(LIVE_SOURCES) | {latest_source}
    profiles = {source: source_profile(draws, source) for source in profile_sources}
    report = build_report(
        rows,
        live_rows,
        live_source_label,
        mismatch_count,
        profiles,
    )

    json_rows = []
    for row in [*rows, *live_rows]:
        payload = {"record_type": "policy_board_row", **asdict(row)}
        payload["source_profile"] = profiles.get(row.source) or source_profile(draws, row.source)
        json_rows.append(payload)

    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    with JSONL_PATH.open("w", encoding="utf-8") as handle:
        for row in json_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(report)


if __name__ == "__main__":
    main()
