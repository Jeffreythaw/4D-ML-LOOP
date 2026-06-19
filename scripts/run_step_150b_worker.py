from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
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

MODE_PRIORITY = {
    "Current": 0,
    "Temporal_Global_Loop": 1,
    "Historical": 2,
    "Engine_Grand_Loop": 3,
    "Grand_Loop": 4,
    "Weighted_Grand_Loop": 5,
}

REPORT_DIR = PROJECT_ROOT / "reports" / "step_150b_workers"


@dataclass
class DrawRecord:
    draw_no: int
    draw_date: str | None
    day_type: str
    winners: list[str]


@dataclass
class PredictionCase:
    source: int
    target: int
    mode: str
    engine: str
    predicted: list[str]
    actual: list[str]
    day_type: str
    month: int | None
    year: int | None


@dataclass
class Candidate:
    number: str
    family: str
    score: float
    reason: str


@dataclass
class ResidualState:
    delta_counter: Counter[tuple[int, int, int, int]] = field(default_factory=Counter)
    mirror_sig_to_actual: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    digit_sum_by_day: dict[str, Counter[int]] = field(default_factory=lambda: defaultdict(Counter))
    first_pair_by_day: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    last_pair_by_day: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    actual_number_counter: Counter[str] = field(default_factory=Counter)
    actual_mirror_counter: Counter[str] = field(default_factory=Counter)
    box_pattern_counter: Counter[str] = field(default_factory=Counter)
    observed_cases: int = 0
    observed_hits: int = 0
    observed_misses: int = 0
    mirror_hidden_hits: int = 0
    hamming_near_misses: int = 0
    box_near_misses: int = 0


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


def circular_digit_delta(src: int, dst: int) -> int:
    return (dst - src) % 10


def circular_digit_distance(a: int, b: int) -> int:
    diff = abs(a - b)
    return min(diff, 10 - diff)


def circular_distance(a: str, b: str) -> int:
    return sum(circular_digit_distance(x, y) for x, y in zip(digits(a), digits(b)))


def hamming_distance(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(z4(a), z4(b)))


def mirror_signature(number: str) -> str:
    return "".join(str(int(ch) % 5) for ch in z4(number))


def box_signature(number: str) -> str:
    return "".join(sorted(z4(number)))


def box_overlap(a: str, b: str) -> int:
    ca = Counter(z4(a))
    cb = Counter(z4(b))
    return sum((ca & cb).values())


def pair_match_score(a: str, b: str) -> int:
    a = z4(a)
    b = z4(b)
    score = 0
    if a[:2] == b[:2]:
        score += 2
    if a[2:] == b[2:]:
        score += 2
    if a[:2] == b[2:]:
        score += 1
    if a[2:] == b[:2]:
        score += 1
    return score


def nearest_actual(pred: str, actuals: list[str]) -> str | None:
    if not actuals:
        return None
    return min(
        actuals,
        key=lambda actual: (
            circular_distance(pred, actual),
            hamming_distance(pred, actual),
            4 - box_overlap(pred, actual),
            abs(digit_sum(pred) - digit_sum(actual)),
            actual,
        ),
    )


def delta_tuple(pred: str, actual: str) -> tuple[int, int, int, int]:
    return tuple(circular_digit_delta(p, a) for p, a in zip(digits(pred), digits(actual)))


def apply_delta(number: str, delta: tuple[int, int, int, int]) -> str:
    return "".join(str((d + shift) % 10) for d, shift in zip(digits(number), delta))


def mirror_variants(number: str) -> Iterable[str]:
    groups = []
    for ch in z4(number):
        d = int(ch)
        mirror = (d + 5) % 10
        groups.append([str(d), str(mirror)])
    seen = set()
    for combo in itertools.product(*groups):
        candidate = "".join(combo)
        if candidate not in seen:
            seen.add(candidate)
            yield candidate


def box_permutations(number: str) -> Iterable[str]:
    seen = set()
    for combo in itertools.permutations(z4(number), 4):
        candidate = "".join(combo)
        if candidate not in seen:
            seen.add(candidate)
            yield candidate


def fetch_draws(cursor) -> dict[int, DrawRecord]:
    rows = cursor.execute("""
        SELECT
            DrawNo,
            CONVERT(varchar(10), DrawDate, 120) AS DrawDateText,
            DayType,
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
            winners=parse_winners(row.WinningNumbers),
        )
    return draws


def fetch_predictions(cursor) -> dict[tuple[int, int], PredictionCase]:
    rows = cursor.execute("""
        SELECT
            Mode,
            EngineSource,
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
        key = (int(row.SourceDrawNo), int(row.TargetDrawNo), str(row.Mode))
        grouped[key].append((int(row.RankNo), z4(row.PredictedNumber)))

    by_pair: dict[tuple[int, int], tuple[str, list[str]]] = {}

    for (source, target, mode), items in grouped.items():
        ranked = [num for _, num in sorted(items)]
        if len(ranked) != 5:
            continue

        pair = (source, target)
        if pair not in by_pair:
            by_pair[pair] = (mode, ranked)
        else:
            old_mode, _ = by_pair[pair]
            if MODE_PRIORITY.get(mode, 99) < MODE_PRIORITY.get(old_mode, 99):
                by_pair[pair] = (mode, ranked)

    output: dict[tuple[int, int], PredictionCase] = {}
    return output, by_pair


def build_cases(draws: dict[int, DrawRecord], pair_predictions: dict[tuple[int, int], tuple[str, list[str]]]) -> list[PredictionCase]:
    cases: list[PredictionCase] = []

    for (source, target), (mode, predicted) in sorted(pair_predictions.items()):
        target_draw = draws.get(target)
        if not target_draw or not target_draw.winners:
            continue

        month = None
        year = None
        if target_draw.draw_date:
            try:
                year = int(target_draw.draw_date[:4])
                month = int(target_draw.draw_date[5:7])
            except Exception:
                pass

        cases.append(
            PredictionCase(
                source=source,
                target=target,
                mode=mode,
                engine=ENGINE,
                predicted=predicted,
                actual=target_draw.winners,
                day_type=target_draw.day_type,
                month=month,
                year=year,
            )
        )

    return cases


def update_state(state: ResidualState, case: PredictionCase) -> None:
    actual_set = set(case.actual)
    hit = bool(actual_set.intersection(case.predicted))

    state.observed_cases += 1
    state.observed_hits += int(hit)
    state.observed_misses += int(not hit)

    for actual in case.actual:
        state.actual_number_counter[actual] += 1
        state.actual_mirror_counter[mirror_signature(actual)] += 1
        state.box_pattern_counter[box_signature(actual)] += 1
        state.mirror_sig_to_actual[mirror_signature(actual)][actual] += 1
        state.digit_sum_by_day[case.day_type][digit_sum(actual)] += 1
        state.digit_sum_by_day["ALL"][digit_sum(actual)] += 1
        state.first_pair_by_day[case.day_type][actual[:2]] += 1
        state.first_pair_by_day["ALL"][actual[:2]] += 1
        state.last_pair_by_day[case.day_type][actual[2:]] += 1
        state.last_pair_by_day["ALL"][actual[2:]] += 1

    for pred in case.predicted:
        nearest = nearest_actual(pred, case.actual)
        if not nearest:
            continue

        state.delta_counter[delta_tuple(pred, nearest)] += 1

        if mirror_signature(pred) == mirror_signature(nearest) and pred != nearest:
            state.mirror_hidden_hits += 1

        if hamming_distance(pred, nearest) <= 1 and pred != nearest:
            state.hamming_near_misses += 1

        if box_overlap(pred, nearest) >= 3 and pred != nearest:
            state.box_near_misses += 1


def add_candidate(pool: dict[str, Candidate], number: str, family: str, score: float, reason: str) -> None:
    number = z4(number)
    existing = pool.get(number)
    if existing is None or score > existing.score:
        pool[number] = Candidate(number=number, family=family, score=score, reason=reason)


def score_candidate(number: str, state: ResidualState, day_type: str) -> float:
    total_actual = max(1, sum(state.actual_number_counter.values()))
    mirror_total = max(1, sum(state.actual_mirror_counter.values()))
    sum_total_day = max(1, sum(state.digit_sum_by_day[day_type].values()))
    first_total = max(1, sum(state.first_pair_by_day[day_type].values()))
    last_total = max(1, sum(state.last_pair_by_day[day_type].values()))

    n = z4(number)
    score = 0.0

    score += 30.0 * (state.actual_number_counter[n] / total_actual)
    score += 18.0 * (state.actual_mirror_counter[mirror_signature(n)] / mirror_total)
    score += 16.0 * (state.digit_sum_by_day[day_type][digit_sum(n)] / sum_total_day)
    score += 12.0 * (state.first_pair_by_day[day_type][n[:2]] / first_total)
    score += 12.0 * (state.last_pair_by_day[day_type][n[2:]] / last_total)

    score += 8.0 * (state.digit_sum_by_day["ALL"][digit_sum(n)] / max(1, sum(state.digit_sum_by_day["ALL"].values())))
    score += 6.0 * (state.first_pair_by_day["ALL"][n[:2]] / max(1, sum(state.first_pair_by_day["ALL"].values())))
    score += 6.0 * (state.last_pair_by_day["ALL"][n[2:]] / max(1, sum(state.last_pair_by_day["ALL"].values())))

    return score


def generate_reconstructed_pool(case: PredictionCase, state: ResidualState, limit: int = 150) -> list[Candidate]:
    pool: dict[str, Candidate] = {}

    for idx, pred in enumerate(case.predicted, start=1):
        base_score = 1000.0 - idx
        add_candidate(pool, pred, "TEMPORAL_BASE", base_score, f"baseline_rank_{idx}")

        for mv in mirror_variants(pred):
            if mv == pred:
                continue
            score = 450.0 + score_candidate(mv, state, case.day_type)
            add_candidate(pool, mv, "MIRROR_EXPANSION", score, f"mirror_of_{pred}")

        for perm in box_permutations(pred):
            if perm == pred:
                continue
            score = 330.0 + score_candidate(perm, state, case.day_type)
            add_candidate(pool, perm, "BOX_REPAIR", score, f"box_perm_of_{pred}")

        for delta, count in state.delta_counter.most_common(25):
            repaired = apply_delta(pred, delta)
            score = 300.0 + (count * 5.0) + score_candidate(repaired, state, case.day_type)
            add_candidate(pool, repaired, "RESIDUAL_DELTA", score, f"delta_{delta}_from_{pred}")

        pred_sig = mirror_signature(pred)
        for actual_like, count in state.mirror_sig_to_actual[pred_sig].most_common(20):
            score = 280.0 + (count * 6.0) + score_candidate(actual_like, state, case.day_type)
            add_candidate(pool, actual_like, "MIRROR_PRIOR_ACTUAL", score, f"prior_actual_same_mirror_{pred_sig}")

        common_last_pairs = state.last_pair_by_day[case.day_type].most_common(15)
        common_first_pairs = state.first_pair_by_day[case.day_type].most_common(15)

        for last_pair, count in common_last_pairs:
            candidate = pred[:2] + last_pair
            score = 260.0 + (count * 3.0) + score_candidate(candidate, state, case.day_type)
            add_candidate(pool, candidate, "PAIR_REPAIR", score, f"keep_first_pair_{pred[:2]}")

        for first_pair, count in common_first_pairs:
            candidate = first_pair + pred[2:]
            score = 250.0 + (count * 3.0) + score_candidate(candidate, state, case.day_type)
            add_candidate(pool, candidate, "PAIR_REPAIR", score, f"keep_last_pair_{pred[2:]}")

    ranked = sorted(pool.values(), key=lambda c: (-c.score, c.family, c.number))
    return ranked[:limit]


def select_diverse_top5(pool: list[Candidate]) -> list[Candidate]:
    selected: list[Candidate] = []
    used_numbers: set[str] = set()
    used_families: set[str] = set()

    family_order = [
        "TEMPORAL_BASE",
        "MIRROR_EXPANSION",
        "BOX_REPAIR",
        "RESIDUAL_DELTA",
        "PAIR_REPAIR",
        "MIRROR_PRIOR_ACTUAL",
    ]

    for family in family_order:
        if len(selected) >= 5:
            break
        candidates = [c for c in pool if c.family == family and c.number not in used_numbers]
        if not candidates:
            continue
        pick = candidates[0]
        selected.append(pick)
        used_numbers.add(pick.number)
        used_families.add(pick.family)

    for candidate in pool:
        if len(selected) >= 5:
            break
        if candidate.number in used_numbers:
            continue
        selected.append(candidate)
        used_numbers.add(candidate.number)
        used_families.add(candidate.family)

    return selected[:5]


def hit_count(predicted: list[str], actual: list[str]) -> int:
    return len(set(predicted).intersection(set(actual)))


def depth_hit(pool: list[Candidate], actual: list[str], depth: int) -> int:
    candidate_set = {candidate.number for candidate in pool[:depth]}
    return len(candidate_set.intersection(set(actual)))


def process_range(cases: list[PredictionCase], source_start: int, source_end: int, worker_id: str) -> list[dict]:
    state = ResidualState()
    rows: list[dict] = []

    sorted_cases = sorted(cases, key=lambda c: c.source)

    for case in sorted_cases:
        if case.target <= source_start:
            update_state(state, case)

    for case in sorted_cases:
        if case.source < source_start or case.source > source_end:
            continue

        if state.observed_cases < 10:
            update_state(state, case)
            continue

        reconstructed_pool = generate_reconstructed_pool(case, state, limit=150)
        proto_top5 = select_diverse_top5(reconstructed_pool)

        baseline_numbers = case.predicted
        proto_numbers = [candidate.number for candidate in proto_top5]

        baseline_hits = hit_count(baseline_numbers, case.actual)
        proto_hits = hit_count(proto_numbers, case.actual)

        row = {
            "worker": worker_id,
            "source": case.source,
            "target": case.target,
            "mode": case.mode,
            "day_type": case.day_type,
            "month": case.month,
            "year": case.year,
            "baseline_top5": baseline_numbers,
            "baseline_hit_count": baseline_hits,
            "prototype_top5": proto_numbers,
            "prototype_families": [candidate.family for candidate in proto_top5],
            "prototype_hit_count": proto_hits,
            "pool_hit_top10": depth_hit(reconstructed_pool, case.actual, 10),
            "pool_hit_top25": depth_hit(reconstructed_pool, case.actual, 25),
            "pool_hit_top50": depth_hit(reconstructed_pool, case.actual, 50),
            "pool_hit_top100": depth_hit(reconstructed_pool, case.actual, 100),
            "state_observed_cases_before_prediction": state.observed_cases,
            "state_observed_hits_before_prediction": state.observed_hits,
            "state_mirror_hidden_hits": state.mirror_hidden_hits,
            "state_hamming_near_misses": state.hamming_near_misses,
            "state_box_near_misses": state.box_near_misses,
        }
        rows.append(row)

        update_state(state, case)

    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-start", type=int, required=True)
    parser.add_argument("--source-end", type=int, required=True)
    parser.add_argument("--worker-id", type=str, required=True)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / f"step150b_worker_{args.worker_id}_{args.source_start}_{args.source_end}.jsonl"

    with get_conn() as conn:
        cursor = conn.cursor()
        draws = fetch_draws(cursor)
        _, pair_predictions = fetch_predictions(cursor)

    cases = build_cases(draws, pair_predictions)
    rows = process_range(cases, args.source_start, args.source_end, args.worker_id)

    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    baseline_draws_with_hit = sum(1 for row in rows if row["baseline_hit_count"] > 0)
    proto_draws_with_hit = sum(1 for row in rows if row["prototype_hit_count"] > 0)

    print("=" * 100)
    print("STEP 150B WORKER COMPLETE")
    print("=" * 100)
    print(f"Worker: {args.worker_id}")
    print(f"Range: {args.source_start}..{args.source_end}")
    print(f"RowsWritten: {len(rows)}")
    print(f"BaselineDrawsWithHit: {baseline_draws_with_hit}")
    print(f"PrototypeDrawsWithHit: {proto_draws_with_hit}")
    print(f"Output: {output_path}")
    print("ProductionMathChanged: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
