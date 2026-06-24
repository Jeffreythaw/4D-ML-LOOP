#!/usr/bin/env python3
"""Step 165: explainable 40-year causal column learning, offline only."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from scripts.run_full_history_engine_training import (  # noqa: E402
    PHASE1_MAX_DRAW_NO,
    RETROSPECTIVE_LABEL,
    ChronologicalTrainingDataset,
    TrainingDraw,
    TrainingPair,
    load_draw_history,
)


POSITIONS = ("Pos_1", "Pos_2", "Pos_3", "Pos_4")
DIGITS = tuple(range(10))
GAP_BUCKETS = (
    (3, "1-3"),
    (7, "4-7"),
    (14, "8-14"),
    (30, "15-30"),
    (60, "31-60"),
    (120, "61-120"),
    (365, "121-365"),
    (10**9, "366+"),
)
STREAK_BUCKETS = (
    (0, "0"),
    (2, "1-2"),
    (5, "3-5"),
    (10, "6-10"),
    (20, "11-20"),
    (30, "21-30"),
    (50, "31-50"),
    (10**9, "51+"),
)
RANDOM_TOP5_23 = 1.0 - ((10000 - 23) / 10000) ** 5

REPORT = ROOT / "reports/step_165_causal_40year_column_learning_report.txt"
MATRICES = ROOT / "reports/step_165_causal_40year_column_learning_matrices.json"
ROWS = ROOT / "reports/step_165_causal_40year_column_learning_rows.jsonl"


def digits(number: str) -> tuple[int, int, int, int]:
    value = str(number).zfill(4)
    if len(value) != 4 or not value.isdigit():
        raise ValueError(f"Invalid 4D number: {number!r}")
    return tuple(int(char) for char in value)  # type: ignore[return-value]


def bucket(value: int, definitions: Sequence[tuple[int, str]]) -> str:
    for maximum, label in definitions:
        if value <= maximum:
            return label
    raise RuntimeError("Bucket definitions are incomplete")


def entropy(probabilities: Sequence[float]) -> float:
    return -sum(value * math.log2(value) for value in probabilities if value > 0)


def normalized(counts: Sequence[int], alpha: float = 1.0) -> list[float]:
    denominator = sum(counts) + alpha * len(counts)
    return [(count + alpha) / denominator for count in counts]


def sha256_payload(payload: dict[str, Any]) -> str:
    data = {key: value for key, value in payload.items() if key != "sha256_hash"}
    encoded = json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def artifact_envelope(
    *,
    name: str,
    training_mode: str,
    dataset: ChronologicalTrainingDataset,
    pair_count: int,
    payload: Any,
) -> dict[str, Any]:
    artifact = {
        "artifact_name": name,
        "artifact_version": "step165.v1",
        "training_mode": training_mode,
        "draw_range": [dataset.first_draw_no, dataset.last_draw_no],
        "pair_count": pair_count,
        "temporal_firewall_status": "PASS",
        "not_for_live_prediction": training_mode == "retrospective_full_history",
        "retrospective_label": (
            RETROSPECTIVE_LABEL
            if training_mode == "retrospective_full_history"
            else None
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
        "sha256_hash": "",
    }
    artifact["sha256_hash"] = sha256_payload(artifact)
    return artifact


@dataclass(frozen=True)
class LockedCandidates:
    source_draw_no: int
    target_draw_no: int
    candidates: tuple[str, ...]
    candidate_hash: str
    target_seen_before_lock: bool
    ranked_pool: tuple[dict[str, Any], ...]
    column_top_digits: tuple[tuple[int, ...], ...]


class CausalColumnLearner:
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.marginal = [[0 for _ in DIGITS] for _ in POSITIONS]
        self.conditional = [
            [[0 for _ in DIGITS] for _ in DIGITS] for _ in POSITIONS
        ]
        self.daytype = defaultdict(
            lambda: [
                [[0 for _ in DIGITS] for _ in DIGITS] for _ in POSITIONS
            ]
        )
        self.weekday = defaultdict(
            lambda: [
                [[0 for _ in DIGITS] for _ in DIGITS] for _ in POSITIONS
            ]
        )
        self.month = defaultdict(
            lambda: [
                [[0 for _ in DIGITS] for _ in DIGITS] for _ in POSITIONS
            ]
        )
        self.adjacent = [Counter() for _ in range(3)]
        self.zero_counts = Counter()
        self.repeat_counts = Counter()
        self.digit_sum_bands = Counter()
        self.first_pairs = Counter()
        self.last_pairs = Counter()
        self.gap_samples = [
            {digit: Counter() for digit in DIGITS} for _ in POSITIONS
        ]
        self.streak_samples = [
            {digit: Counter() for digit in DIGITS} for _ in POSITIONS
        ]
        self.streak_hits = [
            {digit: Counter() for digit in DIGITS} for _ in POSITIONS
        ]
        self.last_seen = [dict() for _ in POSITIONS]
        self.absence_streak = [
            {digit: 0 for digit in DIGITS} for _ in POSITIONS
        ]
        self.target_draw_index = 0
        self.pair_count = 0

    def update(self, pair: TrainingPair) -> None:
        source_date = date.fromisoformat(pair.source_date)
        weekday = source_date.strftime("%A")
        month = source_date.month
        source_values = pair.source_digits
        target_values = pair.target_digits
        target_sets = [
            {value[position] for value in target_values} for position in range(4)
        ]

        for position in range(4):
            source_frequency = Counter(
                value[position] for value in source_values
            )
            target_frequency = Counter(
                value[position] for value in target_values
            )
            for target_digit, count in target_frequency.items():
                self.marginal[position][target_digit] += count
            for source_digit, source_count in source_frequency.items():
                for target_digit, target_count in target_frequency.items():
                    increment = source_count * target_count
                    self.conditional[position][source_digit][target_digit] += increment
                    self.daytype[pair.source_day_type][position][source_digit][
                        target_digit
                    ] += increment
                    self.weekday[weekday][position][source_digit][target_digit] += increment
                    self.month[month][position][source_digit][target_digit] += increment

            for digit in DIGITS:
                current_gap = (
                    self.target_draw_index
                    - self.last_seen[position].get(digit, self.target_draw_index)
                )
                gap_label = bucket(max(1, current_gap), GAP_BUCKETS)
                streak_label = bucket(
                    self.absence_streak[position][digit], STREAK_BUCKETS
                )
                self.gap_samples[position][digit][gap_label] += 1
                self.streak_samples[position][digit][streak_label] += 1
                if digit in target_sets[position]:
                    self.streak_hits[position][digit][streak_label] += 1
                    self.last_seen[position][digit] = self.target_draw_index
                    self.absence_streak[position][digit] = 0
                else:
                    self.absence_streak[position][digit] += 1

        for number in pair.target_winners_23:
            values = digits(number)
            for position in range(3):
                self.adjacent[position][number[position : position + 2]] += 1
            self.zero_counts[number.count("0")] += 1
            self.repeat_counts[4 - len(set(number))] += 1
            total = sum(values)
            self.digit_sum_bands[f"{(total // 5) * 5:02d}-{(total // 5) * 5 + 4:02d}"] += 1
            self.first_pairs[number[:2]] += 1
            self.last_pairs[number[2:]] += 1
        self.target_draw_index += 1
        self.pair_count += 1

    def fit(self, pairs: Iterable[TrainingPair]) -> "CausalColumnLearner":
        for pair in pairs:
            self.update(pair)
        return self

    def source_conditioned_distribution(
        self,
        source_winners: Sequence[str],
        position: int,
        *,
        day_type: str | None = None,
        weekday: str | None = None,
        month: int | None = None,
    ) -> dict[str, list[float]]:
        source_frequency = Counter(
            digits(number)[position] for number in source_winners
        )

        def mixture(matrix: Sequence[Sequence[int]]) -> list[float]:
            output = [0.0 for _ in DIGITS]
            total = sum(source_frequency.values())
            for source_digit, count in source_frequency.items():
                row = normalized(matrix[source_digit], self.alpha)
                for target_digit in DIGITS:
                    output[target_digit] += count / total * row[target_digit]
            return output

        result = {
            "global": mixture(self.conditional[position]),
            "marginal": normalized(self.marginal[position], self.alpha),
        }
        if day_type is not None:
            result["day_type"] = mixture(self.daytype[day_type][position])
        if weekday is not None:
            result["weekday"] = mixture(self.weekday[weekday][position])
        if month is not None:
            result["month"] = mixture(self.month[month][position])
        return result

    def column_summary(self) -> list[dict[str, Any]]:
        output = []
        for position in range(4):
            marginal = normalized(self.marginal[position], self.alpha)
            conditional_entropy = 0.0
            support_total = sum(sum(row) for row in self.conditional[position])
            top = []
            for source_digit, row in enumerate(self.conditional[position]):
                support = sum(row)
                probabilities = normalized(row, self.alpha)
                if support_total:
                    conditional_entropy += support / support_total * entropy(probabilities)
                for target_digit, count in enumerate(row):
                    probability = probabilities[target_digit]
                    lift = probability / marginal[target_digit]
                    top.append(
                        {
                            "source_digit": source_digit,
                            "target_digit": target_digit,
                            "count": count,
                            "probability": probability,
                            "lift": lift,
                            "support": support,
                        }
                    )
            marginal_entropy = entropy(marginal)
            output.append(
                {
                    "position": POSITIONS[position],
                    "marginal_distribution": marginal,
                    "marginal_entropy": marginal_entropy,
                    "conditional_entropy": conditional_entropy,
                    "information_gain": marginal_entropy - conditional_entropy,
                    "top_transitions": sorted(
                        top,
                        key=lambda item: (
                            -item["lift"],
                            -item["count"],
                            item["source_digit"],
                            item["target_digit"],
                        ),
                    )[:25],
                }
            )
        return output

    def gap_profiles(self) -> list[dict[str, Any]]:
        return [
            {
                "position": POSITIONS[position],
                "digits": {
                    str(digit): dict(self.gap_samples[position][digit])
                    for digit in DIGITS
                },
            }
            for position in range(4)
        ]

    def renewal_profiles(self) -> list[dict[str, Any]]:
        output = []
        for position in range(4):
            digit_rows = {}
            for digit in DIGITS:
                rows = {}
                baseline = (
                    sum(self.streak_hits[position][digit].values())
                    / max(1, sum(self.streak_samples[position][digit].values()))
                )
                for label, samples in self.streak_samples[position][digit].items():
                    hits = self.streak_hits[position][digit][label]
                    rate = hits / samples if samples else 0.0
                    rows[label] = {
                        "sample_count": samples,
                        "hit_count": hits,
                        "hit_frequency": rate,
                        "lift_vs_digit_baseline": rate / baseline if baseline else None,
                    }
                digit_rows[str(digit)] = rows
            output.append(
                {"position": POSITIONS[position], "digits": digit_rows}
            )
        return output

    def current_gap(self, position: int, digit: int) -> int:
        previous = self.last_seen[position].get(digit)
        return self.target_draw_index - previous if previous is not None else 366

    def renewal_rate(self, position: int, digit: int) -> float:
        label = bucket(self.absence_streak[position][digit], STREAK_BUCKETS)
        samples = self.streak_samples[position][digit][label]
        hits = self.streak_hits[position][digit][label]
        return (hits + 1) / (samples + 2)


def candidate_components(
    learner: CausalColumnLearner,
    source: TrainingDraw,
    values: tuple[int, int, int, int],
    distributions: Sequence[dict[str, list[float]]],
) -> dict[str, float]:
    column_probability = math.prod(
        distributions[position]["global"][values[position]]
        for position in range(4)
    )
    day_probability = math.prod(
        distributions[position]["day_type"][values[position]]
        for position in range(4)
    )
    marginal_probability = math.prod(
        distributions[position]["marginal"][values[position]]
        for position in range(4)
    )
    day_lift = math.log1p(day_probability / max(marginal_probability, 1e-15))
    gap_support = statistics.fmean(
        1.0 / (1.0 + learner.current_gap(position, values[position]))
        for position in range(4)
    )
    streak_support = statistics.fmean(
        learner.renewal_rate(position, values[position])
        for position in range(4)
    )
    adjacent = statistics.fmean(
        (learner.adjacent[position][f"{values[position]}{values[position + 1]}"] + 1)
        / (sum(learner.adjacent[position].values()) + 100)
        for position in range(3)
    )
    number = "".join(str(value) for value in values)
    zeros = number.count("0")
    repeats = 4 - len(set(number))
    structure_total = max(1, sum(learner.zero_counts.values()))
    structural = (
        (learner.zero_counts[zeros] + 1) / (structure_total + 5)
        + (learner.repeat_counts[repeats] + 1) / (structure_total + 5)
    ) / 2.0
    return {
        "column_probability_product": column_probability,
        "daytype_conditioned_lift": day_lift,
        "gap_recurrence_support": gap_support,
        "miss_streak_renewal_support": streak_support,
        "adjacent_pair_alignment": adjacent,
        "zero_repeated_structure": structural,
    }


def total_score(components: dict[str, float]) -> float:
    return (
        math.log(max(components["column_probability_product"], 1e-300))
        + 0.40 * components["daytype_conditioned_lift"]
        + 2.0 * components["gap_recurrence_support"]
        + 1.0 * components["miss_streak_renewal_support"]
        + 2.0 * components["adjacent_pair_alignment"]
        + 0.5 * components["zero_repeated_structure"]
    )


def lock_candidates(
    learner: CausalColumnLearner,
    source: TrainingDraw,
    target_draw_no: int,
    *,
    top_digits: int = 5,
    broad_pool_size: int = 625,
    top_k: int = 5,
) -> LockedCandidates:
    source_date = date.fromisoformat(source.draw_date)
    distributions = [
        learner.source_conditioned_distribution(
            source.winners,
            position,
            day_type=source.day_type,
            weekday=source_date.strftime("%A"),
            month=source_date.month,
        )
        for position in range(4)
    ]
    top_by_position = [
        tuple(
            sorted(
                DIGITS,
                key=lambda digit: (
                    -distributions[position]["global"][digit],
                    -distributions[position]["day_type"][digit],
                    digit,
                ),
            )[:top_digits]
        )
        for position in range(4)
    ]
    ranked = []
    for values in itertools.product(*top_by_position):
        components = candidate_components(learner, source, values, distributions)
        ranked.append(
            {
                "number": "".join(str(value) for value in values),
                "score": total_score(components),
                "components": components,
            }
        )
    ranked.sort(key=lambda item: (-item["score"], item["number"]))
    ranked = ranked[:broad_pool_size]
    locked = tuple(item["number"] for item in ranked[:top_k])
    candidate_hash = hashlib.sha256(
        json.dumps(
            {
                "source_draw_no": source.draw_no,
                "target_draw_no": target_draw_no,
                "locked": locked,
                "scores": [
                    [item["number"], round(item["score"], 15)]
                    for item in ranked[:top_k]
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return LockedCandidates(
        source_draw_no=source.draw_no,
        target_draw_no=target_draw_no,
        candidates=locked,
        candidate_hash=candidate_hash,
        target_seen_before_lock=False,
        ranked_pool=tuple(ranked),
        column_top_digits=tuple(top_by_position),
    )


def verify_after_lock(
    locked: LockedCandidates, target: TrainingDraw
) -> dict[str, Any]:
    if locked.target_seen_before_lock:
        raise RuntimeError("Temporal firewall violation: target seen before lock")
    if target.draw_no != locked.target_draw_no:
        raise ValueError("Verifier target mismatch")
    pool_numbers = {
        item["number"]: (rank, item)
        for rank, item in enumerate(locked.ranked_pool, start=1)
    }
    locked_set = set(locked.candidates)
    actual = set(target.winners)
    matched = sorted(locked_set & actual)
    generated_actual = sorted(actual & pool_numbers.keys())
    never_generated = sorted(actual - pool_numbers.keys())
    generated_but_dropped = sorted(
        number for number in generated_actual if number not in locked_set
    )
    dropped_details = []
    cutoff_score = (
        locked.ranked_pool[len(locked.candidates) - 1]["score"]
        if locked.ranked_pool
        else None
    )
    for number in generated_but_dropped:
        rank, item = pool_numbers[number]
        weak = min(item["components"], key=item["components"].get)
        dropped_details.append(
            {
                "number": number,
                "rank_before_cutoff": rank,
                "score_before_drop": item["score"],
                "cutoff_score": cutoff_score,
                "weak_component": weak,
                "drop_reason": f"TOP5_CUTOFF_WEAK_{weak.upper()}",
            }
        )
    return {
        "hit_count": len(matched),
        "matched_numbers": matched,
        "candidate_ranks": {
            number: locked.candidates.index(number) + 1 for number in matched
        },
        "never_generated": never_generated,
        "generated_but_dropped": generated_but_dropped,
        "generated_but_dropped_details": dropped_details,
        "target_seen_before_lock": False,
        "target_seen_after_lock": True,
        "target_winners_post_lock": list(target.winners),
    }


def random_baseline() -> float:
    return RANDOM_TOP5_23


def evaluate_rolling_origin(
    dataset: ChronologicalTrainingDataset,
    *,
    start_draw: int = PHASE1_MAX_DRAW_NO + 1,
    top_digits: int = 5,
    broad_pool_size: int = 625,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_no = {draw.draw_no: draw for draw in dataset.draws}
    learner = CausalColumnLearner().fit(dataset.phase1_pairs())
    events = []
    miss_streak = 0
    conditional = defaultdict(lambda: Counter(samples=0, hits=0))

    for source in dataset.draws:
        if source.draw_no < start_draw:
            continue
        target = by_no.get(source.draw_no + 1)
        if target is None:
            continue
        locked = lock_candidates(
            learner,
            source,
            target.draw_no,
            top_digits=top_digits,
            broad_pool_size=broad_pool_size,
        )
        verification = verify_after_lock(locked, target)
        hit = verification["hit_count"] > 0
        streak_label = bucket(miss_streak, STREAK_BUCKETS)
        contexts = {
            "miss_streak": streak_label,
            "day_type": source.day_type,
            "weekday": date.fromisoformat(source.draw_date).strftime("%A"),
            "month": str(date.fromisoformat(source.draw_date).month),
        }
        for dimension, value in contexts.items():
            key = f"{dimension}::{value}"
            conditional[key]["samples"] += 1
            conditional[key]["hits"] += int(hit)
        events.append(
            {
                "source_draw_no": source.draw_no,
                "target_draw_no": target.draw_no,
                "locked_top5": list(locked.candidates),
                "candidate_hash": locked.candidate_hash,
                "column_top_digits": [list(values) for values in locked.column_top_digits],
                "target_seen_before_lock": False,
                "verification": verification,
                "source_day_type": source.day_type,
                "source_weekday": contexts["weekday"],
                "source_month": int(contexts["month"]),
                "pre_miss_streak_bucket": streak_label,
                "training_pair_count_before_lock": learner.pair_count,
            }
        )
        miss_streak = 0 if hit else miss_streak + 1
        learner.update(
            ChronologicalTrainingDataset._make_pair(source, target)
        )

    draws = len(events)
    hit_draws = sum(event["verification"]["hit_count"] > 0 for event in events)
    actual_slots = sum(
        len(event["verification"]["target_winners_post_lock"])
        for event in events
    )
    never_generated_count = sum(
        len(event["verification"]["never_generated"]) for event in events
    )
    dropped_count = sum(
        len(event["verification"]["generated_but_dropped"]) for event in events
    )
    reasons = Counter(
        detail["drop_reason"]
        for event in events
        for detail in event["verification"]["generated_but_dropped_details"]
    )
    conditional_summary = {}
    for key, values in sorted(conditional.items()):
        rate = values["hits"] / values["samples"] if values["samples"] else 0.0
        conditional_summary[key] = {
            **values,
            "hit_rate": rate,
            "lift_vs_random": rate / RANDOM_TOP5_23 if RANDOM_TOP5_23 else None,
        }
    summary = {
        "draws_evaluated": draws,
        "draws_with_hit": hit_draws,
        "raw_hits": sum(event["verification"]["hit_count"] for event in events),
        "top5_hit_rate": hit_draws / draws if draws else None,
        "random_top5_23_baseline": RANDOM_TOP5_23,
        "enrichment_vs_random": (
            (hit_draws / draws) / RANDOM_TOP5_23 if draws else None
        ),
        "actual_winner_slots": actual_slots,
        "never_generated_count": never_generated_count,
        "never_generated_rate": (
            never_generated_count / actual_slots if actual_slots else None
        ),
        "generated_but_dropped_count": dropped_count,
        "generated_but_dropped_rate": (
            dropped_count / actual_slots if actual_slots else None
        ),
        "present_in_top5_count": sum(
            event["verification"]["hit_count"] for event in events
        ),
        "common_drop_reasons": reasons.most_common(20),
        "conditional_hit_patterns": conditional_summary,
    }
    return events, summary


def render_report(
    dataset: ChronologicalTrainingDataset,
    columns: Sequence[dict[str, Any]],
    gaps: Any,
    renewals: Any,
    verification: dict[str, Any],
    artifacts: Sequence[Path],
) -> str:
    lines = [
        "STEP 165 — 40-YEAR CAUSAL COLUMN LEARNING ENGINE",
        "ProductionPredictionChanged: NO",
        "DBWritePerformed: NO",
        "PredictionLedgerWritePerformed: NO",
        "DeepCandidateLedgerWritePerformed: NO",
        "LivePredictionGenerated: NO",
        f"RetrospectiveLabel: {RETROSPECTIVE_LABEL}",
        "",
        "DATASET SUMMARY",
        json.dumps(dataset.metadata(), sort_keys=True),
        "",
        "COLUMN-BY-COLUMN LEARNING SUMMARY",
        "Position  EntropyMarginal  EntropyConditional  InformationGain",
    ]
    for column in columns:
        lines.append(
            f"{column['position']:<8} {column['marginal_entropy']:<16.8f} "
            f"{column['conditional_entropy']:<19.8f} "
            f"{column['information_gain']:.8f}"
        )
        lines.append(f"{column['position']} TopTransitions:")
        for item in column["top_transitions"][:10]:
            lines.append(
                f"  {item['source_digit']}->{item['target_digit']} "
                f"count={item['count']} p={item['probability']:.6f} "
                f"lift={item['lift']:.6f} support={item['support']}"
            )
    lines.extend(
        [
            "",
            "GAP RECURRENCE SUMMARY",
        ]
    )
    for position in gaps:
        aggregate = Counter()
        for values in position["digits"].values():
            aggregate.update(values)
        lines.append(f"{position['position']}: {dict(aggregate)}")
    lines.extend(
        [
            "",
            "MISS-STREAK RENEWAL SUMMARY",
        ]
    )
    for position in renewals:
        samples = Counter()
        hits = Counter()
        for values in position["digits"].values():
            for label, item in values.items():
                samples[label] += item["sample_count"]
                hits[label] += item["hit_count"]
        summary = {
            label: {
                "samples": samples[label],
                "hits": hits[label],
                "hit_rate": hits[label] / samples[label]
                if samples[label]
                else None,
            }
            for _, label in STREAK_BUCKETS
        }
        lines.append(f"{position['position']}: {summary}")
    lines.extend(
        [
            "",
            "CAUSAL ALIGNMENT OFFLINE TOP5 VERIFICATION",
        ]
    )
    for key, value in verification.items():
        if key != "conditional_hit_patterns":
            lines.append(f"{key}: {value}")
    lines.extend(
        [
            "",
            "RANDOM BASELINE COMPARISON",
            f"RandomTop5Against23: {RANDOM_TOP5_23:.8%}",
            f"ObservedTop5HitRate: {verification['top5_hit_rate']}",
            f"EnrichmentVsRandom: {verification['enrichment_vs_random']}",
            "",
            "DROP / FILTER AUDIT",
            f"never_generated: {verification['never_generated_count']} ({verification['never_generated_rate']})",
            f"generated_but_dropped: {verification['generated_but_dropped_count']} ({verification['generated_but_dropped_rate']})",
            f"present_in_top5: {verification['present_in_top5_count']}",
            f"CommonDropReasons: {verification['common_drop_reasons']}",
            "",
            "BLIND-SPOT SUMMARY",
            (
                "Never-generated indicates column candidate coverage weakness. "
                "Generated-but-dropped indicates ranking/component-weight weakness."
            ),
            "",
            "ARTIFACTS",
            *[f"- {path}" for path in artifacts],
            "",
            "FINAL INTERPRETATION",
            "40-year retrospective causal learning artifacts were created.",
            "Do not claim predictive success unless rolling-origin hit rate exceeds baseline consistently.",
            "Next step: refine explainable causal features and sequential adaptive corrections.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-mode",
        choices=("phase1_base", "rolling_origin", "retrospective_full_history"),
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/causal_40year_learning",
    )
    parser.add_argument("--no-sql-write", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def enforce_no_write(args: argparse.Namespace) -> None:
    if not args.no_sql_write and os.getenv("J4D_NO_SQL_WRITE") != "1":
        raise RuntimeError("Require --no-sql-write or J4D_NO_SQL_WRITE=1")


def main() -> int:
    args = parse_args()
    enforce_no_write(args)
    dataset = load_draw_history()
    if args.training_mode == "phase1_base":
        learning_pairs = dataset.phase1_pairs()
    elif args.training_mode == "retrospective_full_history":
        learning_pairs = dataset.retrospective_pairs()
    else:
        learning_pairs = dataset.phase1_pairs()

    learner = CausalColumnLearner().fit(learning_pairs)
    columns = learner.column_summary()
    gaps = learner.gap_profiles()
    renewals = learner.renewal_profiles()
    events, verification = evaluate_rolling_origin(dataset)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_payloads = {
        "column_transition_matrices.json": {
            "marginal": learner.marginal,
            "conditional": learner.conditional,
            "column_summary": columns,
        },
        "daytype_conditioned_column_matrices.json": {
            key: value for key, value in sorted(learner.daytype.items())
        },
        "gap_recurrence_profiles.json": gaps,
        "miss_streak_renewal_profiles.json": renewals,
        "causal_alignment_score_components.json": {
            "components": [
                "column_probability_product",
                "daytype_conditioned_lift",
                "gap_recurrence_support",
                "miss_streak_renewal_support",
                "adjacent_pair_alignment",
                "zero_repeated_structure",
            ],
            "rolling_origin_summary": verification,
        },
        "drop_filter_audit_summary.json": {
            "summary": verification,
            "events": [
                {
                    "source_draw_no": event["source_draw_no"],
                    "target_draw_no": event["target_draw_no"],
                    "never_generated": event["verification"]["never_generated"],
                    "generated_but_dropped": event["verification"][
                        "generated_but_dropped"
                    ],
                    "details": event["verification"][
                        "generated_but_dropped_details"
                    ],
                }
                for event in events
            ],
        },
        "retrospective_full_history_learning_manifest.json": {
            "dataset": dataset.metadata(),
            "column_information": columns,
            "rolling_origin_verification": verification,
            "retrospective_label": RETROSPECTIVE_LABEL,
        },
    }
    artifact_paths = []
    for filename, payload in artifact_payloads.items():
        artifact = artifact_envelope(
            name=filename.removesuffix(".json"),
            training_mode=args.training_mode,
            dataset=dataset,
            pair_count=len(learning_pairs),
            payload=payload,
        )
        path = args.output_dir / filename
        path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        artifact_paths.append(path)

    report_payload = {
        "metadata": {
            "step": 165,
            "production_prediction_changed": False,
            "db_write_performed": False,
            "live_prediction_generated": False,
            "training_mode": args.training_mode,
        },
        "dataset": dataset.metadata(),
        "cause_effect_pair_schema": {
            "source_draw_no": "cause DrawNo",
            "target_draw_no": "next chronological effect DrawNo",
            "source_date": "cause date",
            "target_date": "effect date",
            "source_day_type": "cause DayType",
            "target_day_type": "effect DayType",
            "source_winners_23": "all available source winners",
            "target_winners_23": "post-lock effect winners",
            "source_main_digits": (
                "digits of first stored source winner; no separate main-prize "
                "column exists in DrawHistory"
            ),
            "source_positional_digits": list(POSITIONS),
            "target_all_23_positional_digits": list(POSITIONS),
        },
        "column_learning": columns,
        "gap_recurrence": gaps,
        "miss_streak_renewal": renewals,
        "offline_verification": verification,
        "events": events,
        "artifacts": [str(path) for path in artifact_paths],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        render_report(
            dataset, columns, gaps, renewals, verification, artifact_paths
        )
    )
    MATRICES.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n")
    row_items = [
        {"row_type": "column_summary", **column} for column in columns
    ]
    row_items.extend({"row_type": "verification_event", **event} for event in events)
    ROWS.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in row_items)
    )
    print("STEP 165 — 40-YEAR CAUSAL COLUMN LEARNING ENGINE")
    print(f"Draws: {len(dataset.draws)}")
    print(f"PairsLearned: {len(learning_pairs)}")
    print(f"RollingDrawsEvaluated: {verification['draws_evaluated']}")
    print(f"Top5HitRate: {verification['top5_hit_rate']}")
    print(f"RandomBaseline: {RANDOM_TOP5_23}")
    print(f"ArtifactsWritten: {len(artifact_paths)}")
    print("DBWritePerformed: NO")
    print("LivePredictionGenerated: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
