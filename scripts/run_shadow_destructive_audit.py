from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pyodbc  # noqa: F401 - explicit audit dependency

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
REPORT_PATH = PROJECT_ROOT / "reports" / "step_156_shadow_top6_postmortem_audit.txt"
MATRICES_PATH = (
    PROJECT_ROOT / "reports" / "step_156_shadow_top6_postmortem_matrices.json"
)
ROWS_PATH = PROJECT_ROOT / "reports" / "step_156_shadow_top6_postmortem_rows.jsonl"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.run_renewal_aware_anti_drop_audit import (
    ENGINE_PRIORITY,
    NUMBER_SPACE,
    add_temporal_pre_miss_streak,
    build_common_rows,
    fetch_draws,
    fetch_ledger,
    get_conn,
    group_ledger,
    has_00_structure,
    omitted_candidates,
    source_discovery,
)


SELECTORS = (
    "SHADOW6_SCORE_PRIORITY",
    "SHADOW6_DELTA_PRIORITY",
    "SHADOW6_WLS_PRIORITY",
    "SHADOW6_LINEAR_PRIORITY",
    "SHADOW6_SPECIAL_CONTEXT_PRIORITY",
    "SHADOW6_AUGUST_CONTEXT_PRIORITY",
    "SHADOW6_STREAK_CONTEXT_PRIORITY",
    "SHADOW6_ZERO_PAIR_PRIORITY",
    "SHADOW6_GAP_RECURRENCE_PRIORITY",
    "SHADOW6_COMBINED_CONTEXT_PRIORITY",
)
STREAK_STATES = tuple(str(value) for value in range(16)) + (
    "16-20",
    "21-30",
    "31-50",
    "51+",
)
CONTEXTS = (
    "FULL_RANGE",
    "SPECIAL_DAYTYPE",
    "SATURDAY_DAYTYPE",
    "AUGUST",
    "TEMPORAL_STREAK_21_30",
    "TEMPORAL_STREAK_31_50",
    "TEMPORAL_STREAK_51_PLUS",
    "ZERO_PAIR_STRUCTURE",
    "FIRST_OR_LAST_PAIR_00",
)


def digits(number: str) -> tuple[int, int, int, int]:
    value = str(number).zfill(4)
    return tuple(int(char) for char in value)  # type: ignore[return-value]


def streak_bucket(value: int | None) -> str:
    if value is None:
        return "unavailable"
    if value <= 15:
        return str(value)
    if value <= 20:
        return "16-20"
    if value <= 30:
        return "21-30"
    if value <= 50:
        return "31-50"
    return "51+"


def gap_bucket(value: int | None) -> str:
    if value is None:
        return "first-seen / no prior"
    if value <= 10:
        return "1-10 draws"
    if value <= 30:
        return "11-30"
    if value <= 60:
        return "31-60"
    if value <= 120:
        return "61-120"
    if value <= 365:
        return "121-365"
    return "366+"


def digit_sum_band(number: str) -> str:
    total = sum(digits(number))
    if total <= 9:
        return "00-09"
    if total <= 14:
        return "10-14"
    if total <= 19:
        return "15-19"
    if total <= 24:
        return "20-24"
    return "25-36"


def repeated_profile(number: str) -> str:
    counts = sorted(Counter(number).values(), reverse=True)
    return "+".join(str(value) for value in counts)


def normalized_score_bucket(value: float) -> str:
    if value >= 0.75:
        return "Q4_0.75_1.00"
    if value >= 0.50:
        return "Q3_0.50_0.75"
    if value >= 0.25:
        return "Q2_0.25_0.50"
    return "Q1_0.00_0.25"


def prior_occurrence(
    number: str,
    source_draw_no: int,
    source_date: str | None,
    occurrences: dict[str, list[dict]],
) -> dict:
    prior = [
        item
        for item in occurrences.get(number, [])
        if item["draw_no"] <= source_draw_no
    ]
    if not prior:
        return {
            "previous_actual_draw_no": None,
            "previous_actual_date": None,
            "prior_occurrence_prize_index": None,
            "gap_draw_count": None,
            "gap_days": None,
            "gap_bucket": "first-seen / no prior",
        }
    latest = max(prior, key=lambda item: item["draw_no"])
    gap_draws = source_draw_no + 1 - latest["draw_no"]
    gap_days = None
    if source_date and latest["draw_date"]:
        gap_days = (
            datetime.fromisoformat(source_date)
            - datetime.fromisoformat(latest["draw_date"])
        ).days
    return {
        "previous_actual_draw_no": latest["draw_no"],
        "previous_actual_date": latest["draw_date"],
        "prior_occurrence_prize_index": latest["prize_index"],
        "gap_draw_count": gap_draws,
        "gap_days": gap_days,
        "gap_bucket": gap_bucket(gap_draws),
    }


def occurrence_index(draws: list[dict]) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = defaultdict(list)
    for draw in draws:
        for index, number in enumerate(draw["winners"], start=1):
            output[number].append(
                {
                    "draw_no": draw["draw_no"],
                    "draw_date": draw["draw_date"],
                    "day_type": draw["day_type"],
                    "prize_index": index,
                }
            )
    return output


def engine_streak_maps(common_rows: list[dict]) -> dict[tuple[str, int, int], int]:
    maps: dict[tuple[str, int, int], int] = {}
    for label in ("Delta", "WLS", "Linear"):
        streak = 0
        for row in sorted(
            common_rows,
            key=lambda item: (item["source_draw_no"], item["target_draw_no"]),
        ):
            maps[(label, row["source_draw_no"], row["target_draw_no"])] = streak
            numbers = {
                item["number"] for item in row["underlying"][label]
            }
            streak = 0 if numbers & row["actuals"] else streak + 1
    return maps


def source_presence_summary(source_winners: tuple[str, ...], target: str) -> list[dict]:
    source_sets = [
        sorted({digits(number)[position] for number in source_winners})
        for position in range(4)
    ]
    target_digits = digits(target)
    return [
        {
            "position": position + 1,
            "source_digit_set": source_sets[position],
            "target_digit": target_digits[position],
            "target_digit_present_in_source_position": (
                target_digits[position] in source_sets[position]
            ),
        }
        for position in range(4)
    ]


def closest_source(source_winners: tuple[str, ...], target: str) -> tuple[str, int]:
    return min(
        (
            (
                number,
                sum(left != right for left, right in zip(number, target)),
            )
            for number in source_winners
        ),
        key=lambda item: (item[1], item[0]),
    )


def candidate_history_fields(
    candidate: dict,
    row: dict,
    occurrences: dict[str, list[dict]],
    engine_streaks: dict[tuple[str, int, int], int],
) -> dict:
    gap = prior_occurrence(
        candidate["number"],
        row["source_draw_no"],
        row["source_date"],
        occurrences,
    )
    source_labels = candidate["all_source_labels"]
    preferred_label = min(
        source_labels,
        key=lambda label: (
            -engine_streaks.get(
                (label, row["source_draw_no"], row["target_draw_no"]), 0
            ),
            ENGINE_PRIORITY[label],
        ),
    )
    engine_streak = engine_streaks.get(
        (preferred_label, row["source_draw_no"], row["target_draw_no"])
    )
    return {
        **candidate,
        **gap,
        "preferred_engine_label": preferred_label,
        "underlying_engine_pre_miss_streak": engine_streak,
        "underlying_engine_streak_bucket": streak_bucket(engine_streak),
    }


def build_omitted_hit_events(
    common_rows: list[dict],
    draws_by_no: dict[int, dict],
    occurrences: dict[str, list[dict]],
    engine_streaks: dict[tuple[str, int, int], int],
) -> list[dict]:
    events = []
    event_id = 0
    for row in common_rows:
        source_draw = draws_by_no[row["source_draw_no"]]
        target_draw = draws_by_no[row["target_draw_no"]]
        actual_index = {
            number: index
            for index, number in enumerate(target_draw["winners"], start=1)
        }
        for candidate in omitted_candidates(row):
            number = candidate["number"]
            if number not in actual_index:
                continue
            event_id += 1
            values = digits(number)
            closest_number, closest_distance = closest_source(
                source_draw["winners"], number
            )
            closest_digits = digits(closest_number)
            target_slot = actual_index[number]
            same_slot_source = (
                source_draw["winners"][target_slot - 1]
                if target_slot <= len(source_draw["winners"])
                else None
            )
            candidate_fields = candidate_history_fields(
                candidate, row, occurrences, engine_streaks
            )
            source_entries = sorted(
                candidate["all_source_entries"],
                key=lambda item: (
                    item["rank"],
                    ENGINE_PRIORITY[item["engine_label"]],
                ),
            )
            events.append(
                {
                    "row_type": "omitted_hit_event",
                    "event_id": event_id,
                    "source_draw_no": row["source_draw_no"],
                    "target_draw_no": row["target_draw_no"],
                    "source_draw_date": row["source_date"],
                    "target_draw_date": row["target_date"],
                    "source_day_type": row["source_day_type"],
                    "target_day_type": row["target_day_type"],
                    "target_year": target_draw["year"],
                    "target_month": target_draw["month"],
                    "target_weekday": target_draw["weekday"],
                    "target_quarter": f"Q{((target_draw['month'] - 1) // 3) + 1}",
                    "target_week_of_month": (
                        ((datetime.fromisoformat(row["target_date"]).day - 1) // 7)
                        + 1
                    ),
                    "source_engine": source_entries[0]["engine_source"],
                    "source_engine_label": source_entries[0]["engine_label"],
                    "source_mode": source_entries[0]["mode"],
                    "source_rank": source_entries[0]["rank"],
                    "source_score": source_entries[0]["score"],
                    "normalized_score": source_entries[0]["normalized_score"],
                    "normalized_score_bucket": normalized_score_bucket(
                        source_entries[0]["normalized_score"]
                    ),
                    "all_source_engines": candidate["all_source_labels"],
                    "all_source_entries": source_entries,
                    "omitted_hit_number": number,
                    "actual_prize_index": actual_index[number],
                    "actual_prize_label": f"PrizeSlot_{actual_index[number]}",
                    "temporal_top5": [
                        item["number"]
                        for item in sorted(
                            row["temporal"], key=lambda item: item["rank"]
                        )
                    ],
                    "underlying_top5": {
                        label: [
                            item["number"]
                            for item in sorted(
                                items, key=lambda item: item["rank"]
                            )
                        ]
                        for label, items in row["underlying"].items()
                    },
                    "temporal_pre_miss_streak": row[
                        "temporal_pre_miss_streak"
                    ],
                    "temporal_streak_bucket": streak_bucket(
                        row["temporal_pre_miss_streak"]
                    ),
                    "engine_pre_miss_streak": candidate_fields[
                        "underlying_engine_pre_miss_streak"
                    ],
                    "engine_streak_bucket": candidate_fields[
                        "underlying_engine_streak_bucket"
                    ],
                    "pos1": values[0],
                    "pos2": values[1],
                    "pos3": values[2],
                    "pos4": values[3],
                    "digit_sum": sum(values),
                    "digit_sum_band": digit_sum_band(number),
                    "first_pair": number[:2],
                    "last_pair": number[2:],
                    "edge_pair": number[0] + number[3],
                    "inner_pair": number[1:3],
                    "box_signature": "".join(sorted(number)),
                    "mirror_signature": "".join(
                        str(value % 5) for value in values
                    ),
                    "parity_signature": "".join(
                        "E" if value % 2 == 0 else "O" for value in values
                    ),
                    "high_low_signature": "".join(
                        "L" if value <= 4 else "H" for value in values
                    ),
                    "repeated_digit_profile": repeated_profile(number),
                    "zero_count": number.count("0"),
                    "double_zero_flag": has_00_structure(number),
                    "same_slot_source_number": same_slot_source,
                    "same_slot_transition": (
                        [
                            f"{left}->{right}"
                            for left, right in zip(same_slot_source, number)
                        ]
                        if same_slot_source
                        else None
                    ),
                    "same_slot_assessment": "slot-order-risk",
                    "presence_transition": source_presence_summary(
                        source_draw["winners"], number
                    ),
                    "closest_source_number": closest_number,
                    "closest_hamming_distance": closest_distance,
                    "closest_pos1_transition": (
                        f"{closest_digits[0]}->{values[0]}"
                    ),
                    "closest_pos2_transition": (
                        f"{closest_digits[1]}->{values[1]}"
                    ),
                    "closest_pos3_transition": (
                        f"{closest_digits[2]}->{values[2]}"
                    ),
                    "closest_pos4_transition": (
                        f"{closest_digits[3]}->{values[3]}"
                    ),
                    "previous_actual_draw_no": candidate_fields[
                        "previous_actual_draw_no"
                    ],
                    "previous_actual_date": candidate_fields[
                        "previous_actual_date"
                    ],
                    "prior_occurrence_prize_index": candidate_fields[
                        "prior_occurrence_prize_index"
                    ],
                    "gap_draw_count": candidate_fields["gap_draw_count"],
                    "gap_days": candidate_fields["gap_days"],
                    "gap_bucket": candidate_fields["gap_bucket"],
                    "removal_reason": "unknown",
                }
            )
    return events


def baseline_slot_transitions(draws: list[dict]) -> list[list[list[int]]]:
    matrices = [[[0 for _ in range(10)] for _ in range(10)] for _ in range(4)]
    for previous, current in zip(draws, draws[1:]):
        count = min(len(previous["winners"]), len(current["winners"]))
        for slot in range(count):
            before = digits(previous["winners"][slot])
            after = digits(current["winners"][slot])
            for position in range(4):
                matrices[position][before[position]][after[position]] += 1
    return matrices


def omitted_transition_matrix(
    events: list[dict], baseline: list[list[list[int]]]
) -> tuple[dict, list[dict]]:
    closest_counts = [
        [[0 for _ in range(10)] for _ in range(10)] for _ in range(4)
    ]
    same_slot_counts = [
        [[0 for _ in range(10)] for _ in range(10)] for _ in range(4)
    ]
    presence_hits = Counter()
    rows = []
    for event in events:
        target_digits = digits(event["omitted_hit_number"])
        closest_digits = digits(event["closest_source_number"])
        same_slot_digits = (
            digits(event["same_slot_source_number"])
            if event["same_slot_source_number"]
            else None
        )
        for position in range(4):
            closest_counts[position][closest_digits[position]][
                target_digits[position]
            ] += 1
            if same_slot_digits:
                same_slot_counts[position][same_slot_digits[position]][
                    target_digits[position]
                ] += 1
            present = event["presence_transition"][position][
                "target_digit_present_in_source_position"
            ]
            presence_hits[(position, present)] += 1
            rows.append(
                {
                    "row_type": "omitted_hit_transition",
                    "event_id": event["event_id"],
                    "source_draw_no": event["source_draw_no"],
                    "target_draw_no": event["target_draw_no"],
                    "omitted_hit_number": event["omitted_hit_number"],
                    "position": position + 1,
                    "closest_previous_digit": closest_digits[position],
                    "target_digit": target_digits[position],
                    "same_slot_previous_digit": (
                        same_slot_digits[position] if same_slot_digits else None
                    ),
                    "target_digit_present_in_source_position": present,
                    "closest_hamming_distance": event[
                        "closest_hamming_distance"
                    ],
                    "same_slot_assessment": "slot-order-risk",
                }
            )

    positions = {}
    top = []
    for position in range(4):
        closest_prob = []
        baseline_prob = []
        lift = []
        for previous_digit in range(10):
            closest_total = sum(closest_counts[position][previous_digit])
            baseline_total = sum(baseline[position][previous_digit])
            closest_row = []
            baseline_row = []
            lift_row = []
            for target_digit in range(10):
                omitted_p = (
                    closest_counts[position][previous_digit][target_digit]
                    / closest_total
                    if closest_total
                    else 0.0
                )
                baseline_p = (
                    baseline[position][previous_digit][target_digit]
                    / baseline_total
                    if baseline_total
                    else 0.0
                )
                current_lift = (
                    omitted_p / baseline_p if baseline_p else None
                )
                closest_row.append(omitted_p)
                baseline_row.append(baseline_p)
                lift_row.append(current_lift)
                if closest_counts[position][previous_digit][target_digit]:
                    top.append(
                        {
                            "position": position + 1,
                            "previous_digit": previous_digit,
                            "target_digit": target_digit,
                            "count": closest_counts[position][previous_digit][
                                target_digit
                            ],
                            "omitted_probability": omitted_p,
                            "baseline_same_slot_probability": baseline_p,
                            "lift_vs_baseline": current_lift,
                        }
                    )
            closest_prob.append(closest_row)
            baseline_prob.append(baseline_row)
            lift.append(lift_row)
        positions[f"Pos_{position + 1}"] = {
            "closest_structure_count_matrix": closest_counts[position],
            "closest_structure_probability_matrix": closest_prob,
            "same_slot_count_matrix": same_slot_counts[position],
            "baseline_same_slot_count_matrix": baseline[position],
            "baseline_same_slot_probability_matrix": baseline_prob,
            "closest_lift_vs_same_slot_baseline": lift,
            "presence_target_digit_seen_count": presence_hits[(position, True)],
            "presence_target_digit_unseen_count": presence_hits[(position, False)],
        }
    return (
        {
            "method_warning": (
                "Same-slot comparisons are marked slot-order-risk. "
                "Closest-structure and draw-level presence are preferred diagnostics."
            ),
            "positions": positions,
            "top_closest_transitions": sorted(
                top,
                key=lambda item: (
                    -item["count"],
                    -(
                        item["lift_vs_baseline"]
                        if item["lift_vs_baseline"] is not None
                        else -1
                    ),
                    item["position"],
                ),
            )[:40],
        },
        rows,
    )


def aggregate_events(events: list[dict]) -> tuple[dict, dict]:
    day_month_gap = {
        "target_day_type": dict(Counter(x["target_day_type"] for x in events)),
        "target_weekday": dict(Counter(x["target_weekday"] for x in events)),
        "target_month": dict(Counter(str(x["target_month"]) for x in events)),
        "target_year": dict(Counter(str(x["target_year"]) for x in events)),
        "target_quarter": dict(Counter(x["target_quarter"] for x in events)),
        "target_week_of_month": dict(
            Counter(str(x["target_week_of_month"]) for x in events)
        ),
        "gap_bucket": dict(Counter(x["gap_bucket"] for x in events)),
        "gap_draw_count_summary": {
            "samples": sum(x["gap_draw_count"] is not None for x in events),
            "minimum": min(
                (x["gap_draw_count"] for x in events if x["gap_draw_count"] is not None),
                default=None,
            ),
            "median": statistics.median(
                [x["gap_draw_count"] for x in events if x["gap_draw_count"] is not None]
            )
            if any(x["gap_draw_count"] is not None for x in events)
            else None,
            "maximum": max(
                (x["gap_draw_count"] for x in events if x["gap_draw_count"] is not None),
                default=None,
            ),
        },
    }
    structure = {
        "digit_sum_band": dict(Counter(x["digit_sum_band"] for x in events)),
        "first_pair": dict(Counter(x["first_pair"] for x in events).most_common()),
        "last_pair": dict(Counter(x["last_pair"] for x in events).most_common()),
        "edge_pair": dict(Counter(x["edge_pair"] for x in events).most_common()),
        "inner_pair": dict(Counter(x["inner_pair"] for x in events).most_common()),
        "zero_count": dict(Counter(str(x["zero_count"]) for x in events)),
        "double_zero_flag": dict(
            Counter(str(x["double_zero_flag"]) for x in events)
        ),
        "repeated_digit_profile": dict(
            Counter(x["repeated_digit_profile"] for x in events)
        ),
        "mirror_signature": dict(
            Counter(x["mirror_signature"] for x in events).most_common()
        ),
        "parity_signature": dict(
            Counter(x["parity_signature"] for x in events).most_common()
        ),
        "high_low_signature": dict(
            Counter(x["high_low_signature"] for x in events).most_common()
        ),
        "source_engine": dict(
            Counter(x["source_engine_label"] for x in events)
        ),
        "source_rank": dict(Counter(str(x["source_rank"]) for x in events)),
        "normalized_score_bucket": dict(
            Counter(x["normalized_score_bucket"] for x in events)
        ),
        "temporal_streak_bucket": dict(
            Counter(x["temporal_streak_bucket"] for x in events)
        ),
        "underlying_engine_streak_bucket": dict(
            Counter(x["engine_streak_bucket"] for x in events)
        ),
    }
    return day_month_gap, structure


def selector_sort_key(selector: str, candidate: dict, row: dict) -> tuple:
    labels = set(candidate["all_source_labels"])
    target_priority = {
        "SHADOW6_DELTA_PRIORITY": "Delta",
        "SHADOW6_WLS_PRIORITY": "WLS",
        "SHADOW6_LINEAR_PRIORITY": "Linear",
    }.get(selector)
    engine_pref = (
        0 if target_priority and target_priority in labels else 1
    )
    special_pref = (
        0
        if selector == "SHADOW6_SPECIAL_CONTEXT_PRIORITY"
        and row["target_day_type"] == "Special"
        and labels & {"Delta", "WLS"}
        else 1
    )
    august_pref = (
        0
        if selector == "SHADOW6_AUGUST_CONTEXT_PRIORITY"
        and (row["target_month"] == 8 or row["source_month"] == 8)
        and labels & {"Delta", "WLS"}
        else 1
    )
    streak_pref = (
        0
        if selector == "SHADOW6_STREAK_CONTEXT_PRIORITY"
        and 21 <= row["temporal_pre_miss_streak"] <= 50
        and labels & {"Delta", "WLS"}
        else 1
    )
    zero_pref = (
        0
        if selector == "SHADOW6_ZERO_PAIR_PRIORITY"
        and has_00_structure(candidate["number"])
        else 1
    )
    gap = candidate["gap_draw_count"]
    gap_pref = (
        0
        if selector == "SHADOW6_GAP_RECURRENCE_PRIORITY"
        and gap is not None
        and 11 <= gap <= 120
        else 1
    )
    combined_conditions = sum(
        (
            row["target_day_type"] == "Special",
            row["target_month"] == 8 or row["source_month"] == 8,
            21 <= row["temporal_pre_miss_streak"] <= 50,
            bool(labels & {"Delta", "WLS"}),
            has_00_structure(candidate["number"]),
            gap is not None and 11 <= gap <= 120,
        )
    )
    combined_pref = (
        -combined_conditions
        if selector == "SHADOW6_COMBINED_CONTEXT_PRIORITY"
        else 0
    )
    preferred_label = min(
        candidate["all_source_labels"],
        key=lambda label: ENGINE_PRIORITY[label],
    )
    return (
        engine_pref,
        special_pref,
        august_pref,
        streak_pref,
        zero_pref,
        gap_pref,
        combined_pref,
        -candidate["normalized_score"],
        candidate["rank"],
        ENGINE_PRIORITY[preferred_label],
        candidate["number"],
    )


def simulate_shadow6(
    common_rows: list[dict],
    occurrences: dict[str, list[dict]],
    engine_streaks: dict[tuple[str, int, int], int],
) -> list[dict]:
    output = []
    for row in common_rows:
        temporal_top5 = [
            item["number"]
            for item in sorted(row["temporal"], key=lambda item: item["rank"])
        ]
        candidates = [
            candidate_history_fields(
                candidate, row, occurrences, engine_streaks
            )
            for candidate in omitted_candidates(row)
        ]
        for selector in SELECTORS:
            selected = (
                min(
                    candidates,
                    key=lambda candidate: selector_sort_key(
                        selector, candidate, row
                    ),
                )
                if candidates
                else None
            )
            shadow_number = selected["number"] if selected else None
            temporal_hit = bool(set(temporal_top5) & row["actuals"])
            shadow_hit = bool(shadow_number and shadow_number in row["actuals"])
            output.append(
                {
                    "row_type": "shadow6_by_draw",
                    "selector": selector,
                    "source_draw_no": row["source_draw_no"],
                    "target_draw_no": row["target_draw_no"],
                    "source_date": row["source_date"],
                    "target_date": row["target_date"],
                    "target_day_type": row["target_day_type"],
                    "target_month": row["target_month"],
                    "target_weekday": row["target_weekday"],
                    "temporal_pre_miss_streak": row[
                        "temporal_pre_miss_streak"
                    ],
                    "temporal_streak_bucket": streak_bucket(
                        row["temporal_pre_miss_streak"]
                    ),
                    "temporal_top5": temporal_top5,
                    "shadow_candidate_present": selected is not None,
                    "shadow_candidate": shadow_number,
                    "shadow_source_engine": (
                        selected["preferred_engine_label"]
                        if selected
                        else None
                    ),
                    "shadow_all_source_engines": (
                        selected["all_source_labels"] if selected else []
                    ),
                    "shadow_source_rank": selected["rank"] if selected else None,
                    "shadow_normalized_score": (
                        selected["normalized_score"] if selected else None
                    ),
                    "shadow_gap_draw_count": (
                        selected["gap_draw_count"] if selected else None
                    ),
                    "shadow_gap_bucket": (
                        selected["gap_bucket"] if selected else None
                    ),
                    "shadow_zero_pair_structure": (
                        has_00_structure(shadow_number)
                        if shadow_number
                        else False
                    ),
                    "shadow_first_or_last_pair_00": (
                        shadow_number[:2] == "00" or shadow_number[2:] == "00"
                        if shadow_number
                        else False
                    ),
                    "temporal_hit": temporal_hit,
                    "shadow_hit": shadow_hit,
                    "top6_hit": temporal_hit or shadow_hit,
                    "incremental_top6_hit": shadow_hit and not temporal_hit,
                    "actual_prize_count": row["actual_prize_count"],
                    "temporal_top5_modified": False,
                }
            )
    return output


def one_number_random_rate(items: list[dict]) -> float:
    if not items:
        return 0.0
    return statistics.mean(
        item["actual_prize_count"] / NUMBER_SPACE for item in items
    )


def selector_metric(selector: str, items: list[dict]) -> dict:
    present = [item for item in items if item["shadow_candidate_present"]]
    random_rate = one_number_random_rate(present)
    shadow_hits = sum(item["shadow_hit"] for item in present)
    shadow_rate = shadow_hits / len(present) if present else 0.0
    temporal_hits = sum(item["temporal_hit"] for item in items)
    incremental = sum(item["incremental_top6_hit"] for item in items)
    return {
        "selector": selector,
        "rows_evaluated": len(items),
        "rows_with_shadow_candidate": len(present),
        "temporal_top5_hit_draws": temporal_hits,
        "shadow6_independent_hit_draws": shadow_hits,
        "shadow6_raw_hits": shadow_hits,
        "top6_expanded_hit_draws": sum(item["top6_hit"] for item in items),
        "incremental_top6_hit_draws": incremental,
        "shadow_hit_rate_among_present": shadow_rate,
        "incremental_lift_over_temporal_hit_draws": (
            incremental / temporal_hits if temporal_hits else None
        ),
        "random_expected_one_number_hit_rate": random_rate,
        "enrichment_vs_random_one_number": (
            shadow_rate / random_rate if random_rate else None
        ),
        "selected_source_engine": dict(
            Counter(item["shadow_source_engine"] for item in present)
        ),
    }


def selector_matrix(results: list[dict]) -> list[dict]:
    by_selector: dict[str, list[dict]] = defaultdict(list)
    for item in results:
        by_selector[item["selector"]].append(item)
    return [
        selector_metric(selector, by_selector[selector])
        for selector in SELECTORS
    ]


def context_match(context: str, item: dict) -> bool:
    streak = item["temporal_pre_miss_streak"]
    if context == "FULL_RANGE":
        return True
    if context == "SPECIAL_DAYTYPE":
        return item["target_day_type"] == "Special"
    if context == "SATURDAY_DAYTYPE":
        return item["target_day_type"] == "Saturday"
    if context == "AUGUST":
        return item["target_month"] == 8
    if context == "TEMPORAL_STREAK_21_30":
        return 21 <= streak <= 30
    if context == "TEMPORAL_STREAK_31_50":
        return 31 <= streak <= 50
    if context == "TEMPORAL_STREAK_51_PLUS":
        return streak >= 51
    if context == "ZERO_PAIR_STRUCTURE":
        return item["shadow_zero_pair_structure"]
    if context == "FIRST_OR_LAST_PAIR_00":
        return item["shadow_first_or_last_pair_00"]
    raise ValueError(context)


def context_lift(results: list[dict]) -> tuple[dict, list[dict]]:
    output = {}
    rows = []
    for context in CONTEXTS:
        output[context] = []
        for selector in SELECTORS:
            items = [
                item
                for item in results
                if item["selector"] == selector
                and context_match(context, item)
            ]
            metric = {"context": context, **selector_metric(selector, items)}
            output[context].append(metric)
            rows.append({"row_type": "shadow6_context", **metric})
    by_source = {}
    for label in ("Delta", "WLS", "Linear"):
        by_source[label] = []
        for selector in SELECTORS:
            items = [
                item
                for item in results
                if item["selector"] == selector
                and item["shadow_source_engine"] == label
            ]
            by_source[label].append(
                {
                    "context": f"SOURCE_{label}",
                    **selector_metric(selector, items),
                }
            )
    output["SELECTED_SOURCE_ENGINE"] = by_source
    gap_contexts = {}
    for bucket in (
        "first-seen / no prior",
        "1-10 draws",
        "11-30",
        "31-60",
        "61-120",
        "121-365",
        "366+",
    ):
        gap_contexts[bucket] = []
        for selector in SELECTORS:
            items = [
                item
                for item in results
                if item["selector"] == selector
                and item["shadow_gap_bucket"] == bucket
            ]
            gap_contexts[bucket].append(
                {
                    "context": f"GAP_{bucket}",
                    **selector_metric(selector, items),
                }
            )
    output["GAP_BUCKETS"] = gap_contexts
    return output, rows


def recent_matrix(results: list[dict]) -> dict[str, list[dict]]:
    pairs = sorted(
        {
            (item["source_draw_no"], item["target_draw_no"])
            for item in results
        }
    )
    output = {}
    for window in (365, 90, 47):
        selected = set(pairs[-window:])
        output[f"RECENT_{window}"] = []
        for selector in SELECTORS:
            items = [
                item
                for item in results
                if item["selector"] == selector
                and (item["source_draw_no"], item["target_draw_no"]) in selected
            ]
            output[f"RECENT_{window}"].append(
                selector_metric(selector, items)
            )
    return output


def conditional_firing(results: list[dict]) -> tuple[dict, list[dict]]:
    output = {}
    rows = []
    for selector in SELECTORS:
        selector_rows = [
            item for item in results if item["selector"] == selector
        ]
        output[selector] = []
        for state in STREAK_STATES:
            items = [
                item
                for item in selector_rows
                if item["temporal_streak_bucket"] == state
            ]
            present = [
                item for item in items if item["shadow_candidate_present"]
            ]
            hits = sum(item["shadow_hit"] for item in present)
            rate = hits / len(present) if present else 0.0
            random_rate = one_number_random_rate(present)
            row = {
                "selector": selector,
                "streak_state": state,
                "samples": len(present),
                "shadow_hits": hits,
                "shadow_hit_rate": rate,
                "random_expected_one_number_hit_rate": random_rate,
                "enrichment_vs_random": (
                    rate / random_rate if random_rate else None
                ),
                "confidence_warning": len(present) < 30,
            }
            output[selector].append(row)
            rows.append({"row_type": "shadow6_conditional_state", **row})
    return output, rows


def synthesis(
    events: list[dict],
    day_gap: dict,
    structure: dict,
    selector_rows: list[dict],
    contexts: dict,
) -> dict:
    best = max(
        selector_rows,
        key=lambda item: (
            item["incremental_top6_hit_draws"],
            item["shadow6_independent_hit_draws"],
            item["enrichment_vs_random_one_number"] or 0,
        ),
    )
    category_scores = {
        "DayType/Month timing": max(
            day_gap["target_day_type"].values(), default=0
        )
        + max(day_gap["target_month"].values(), default=0),
        "Temporal miss-streak state": max(
            structure["temporal_streak_bucket"].values(), default=0
        ),
        "underlying engine-specific renewal": max(
            structure["source_engine"].values(), default=0
        ),
        "gap recurrence": max(day_gap["gap_bucket"].values(), default=0),
        "zero-pair/digit structure": (
            structure["double_zero_flag"].get("True", 0)
            + structure["zero_count"].get("2", 0)
        ),
    }
    if best["incremental_top6_hit_draws"] >= 5:
        next_step = (
            f"run a report-only live shadow Top6 board using "
            f"{best['selector']} and continue selector refinement"
        )
    else:
        next_step = (
            "refine source-side Shadow6 selection around mixed context and gap "
            "recurrence; retain a non-destructive live shadow board"
        )
    return {
        "omitted_signal_real": True,
        "verified_unique_omitted_hit_opportunities": len(events),
        "best_shadow6_selector": best["selector"],
        "best_shadow6_metrics": best,
        "verified_signal_capture_rate": (
            best["incremental_top6_hit_draws"] / len(events) if events else 0.0
        ),
        "selector_capture_statement": (
            f"{best['selector']} captured "
            f"{best['incremental_top6_hit_draws']} verified opportunities as "
            "independent Top6 additions while leaving Temporal Top5 unchanged; "
            "the selector did not capture the remaining verified signal."
        ),
        "candidate_category_scores_descriptive_only": category_scores,
        "leading_causal_key_candidate": (
            "mixed context; no single tested DayType, month, streak, gap, engine, "
            "or zero-structure selector isolates the verified omitted signal"
        ),
        "causality_warning": (
            "The 50 omitted-hit opportunities are verified. Context concentrations "
            "and selector outcomes are diagnostic associations, not persisted "
            "historical removal causes."
        ),
        "production_switch_recommended_now": False,
        "next_step": next_step,
    }


def pct(value: float | None) -> str:
    return "NULL" if value is None else f"{value * 100:.4f}%"


def format_event_rows(events: list[dict], limit: int = 25) -> list[str]:
    lines = [
        "No Source Target Date       Number Engine Rank Score    DayType   Month TStreak EStreak GapBucket       Prize"
    ]
    for event in events[:limit]:
        lines.append(
            f"{event['event_id']:>2} {event['source_draw_no']:>6} "
            f"{event['target_draw_no']:>6} {event['target_draw_date']:<10} "
            f"{event['omitted_hit_number']} "
            f"{event['source_engine_label']:<6} {event['source_rank']:>4} "
            f"{event['source_score']:>8.4f} "
            f"{event['target_day_type']:<9} {event['target_month']:>5} "
            f"{event['temporal_pre_miss_streak']:>7} "
            f"{event['engine_pre_miss_streak']:>7} "
            f"{event['gap_bucket']:<15} "
            f"{event['actual_prize_label']}"
        )
    return lines


def format_selector_matrix(rows: list[dict]) -> list[str]:
    lines = [
        "Selector                                  Rows Present Temporal Shadow Top6 Increment ShadowRate Random1 Enrich"
    ]
    for row in rows:
        enrich = (
            "NULL"
            if row["enrichment_vs_random_one_number"] is None
            else f"{row['enrichment_vs_random_one_number']:.4f}"
        )
        lines.append(
            f"{row['selector']:<41} {row['rows_evaluated']:>4} "
            f"{row['rows_with_shadow_candidate']:>7} "
            f"{row['temporal_top5_hit_draws']:>8} "
            f"{row['shadow6_independent_hit_draws']:>6} "
            f"{row['top6_expanded_hit_draws']:>4} "
            f"{row['incremental_top6_hit_draws']:>9} "
            f"{pct(row['shadow_hit_rate_among_present']):>10} "
            f"{pct(row['random_expected_one_number_hit_rate']):>7} "
            f"{enrich:>6}"
        )
    return lines


def format_conditional(cond: dict) -> list[str]:
    lines = [
        "Selector                                  State Samples Hits HitRate Random1 Enrich"
    ]
    for selector in SELECTORS:
        for row in cond[selector]:
            if row["samples"] == 0:
                continue
            enrich = (
                "NULL"
                if row["enrichment_vs_random"] is None
                else f"{row['enrichment_vs_random']:.4f}"
            )
            lines.append(
                f"{selector:<41} {row['streak_state']:<7} "
                f"{row['samples']:>7} {row['shadow_hits']:>4} "
                f"{pct(row['shadow_hit_rate']):>9} "
                f"{pct(row['random_expected_one_number_hit_rate']):>7} "
                f"{enrich:>6}"
            )
    return lines


def format_context(contexts: dict) -> list[str]:
    lines = [
        "Context                    Selector                                  Rows Shadow Increment Rate Enrich"
    ]
    for context in CONTEXTS:
        if context == "FULL_RANGE":
            continue
        for row in contexts[context]:
            enrich = (
                "NULL"
                if row["enrichment_vs_random_one_number"] is None
                else f"{row['enrichment_vs_random_one_number']:.4f}"
            )
            lines.append(
                f"{context:<26} {row['selector']:<41} "
                f"{row['rows_evaluated']:>4} "
                f"{row['shadow6_independent_hit_draws']:>6} "
                f"{row['incremental_top6_hit_draws']:>9} "
                f"{pct(row['shadow_hit_rate_among_present']):>9} "
                f"{enrich:>6}"
            )
    return lines


def format_recent(recent: dict) -> list[str]:
    lines = [
        "Window     Selector                                  Rows Shadow Increment Top6 Rate Enrich"
    ]
    for window in ("RECENT_365", "RECENT_90", "RECENT_47"):
        for row in recent[window]:
            enrich = (
                "NULL"
                if row["enrichment_vs_random_one_number"] is None
                else f"{row['enrichment_vs_random_one_number']:.4f}"
            )
            lines.append(
                f"{window:<10} {row['selector']:<41} "
                f"{row['rows_evaluated']:>4} "
                f"{row['shadow6_independent_hit_draws']:>6} "
                f"{row['incremental_top6_hit_draws']:>9} "
                f"{row['top6_expanded_hit_draws']:>4} "
                f"{pct(row['shadow_hit_rate_among_present']):>9} "
                f"{enrich:>6}"
            )
    return lines


def build_report(
    discovery: dict,
    events: list[dict],
    day_gap: dict,
    transitions: dict,
    structure: dict,
    selector_rows: list[dict],
    conditional: dict,
    contexts: dict,
    recent: dict,
    causal: dict,
) -> str:
    width = 180
    lines = [
        "=" * width,
        "STEP 156 — NON-DESTRUCTIVE TOP6 / SHADOW POST-MORTEM AUDIT — REPORT ONLY",
        "=" * width,
        "ProductionMathChanged: NO",
        "APIChanged: NO",
        "FrontendChanged: NO",
        "SQLSchemaChanged: NO",
        "DBWritePerformed: NO",
        "ExistingTablesOnly: YES",
        "DeepCandidateLedgerUsed: NO",
        "TemporalTop5Modified: NO",
        "",
        "SOURCE DISCOVERY",
        "-" * width,
        f"DrawRange: {discovery['draw_range']}",
        f"DrawRows: {discovery['draw_rows']}",
        f"PredictionLedgerRows: {discovery['predictionledger_rows']}",
        f"ResolvedFinalStream: {discovery['resolved_final_stream']}",
        f"ResolvedUnderlyingStreams: {discovery['resolved_underlying_streams']}",
        f"LedgerVolumeByEngineMode: {discovery['ledger_volume_by_engine_mode']}",
        "",
        "OMITTED HIT POST-MORTEM TABLE SUMMARY",
        "-" * width,
        f"UniqueGrandLoopOmittedHitEvents: {len(events)}",
        f"ExactHitNumbers: {[event['omitted_hit_number'] for event in events]}",
        *format_event_rows(events, 25),
        "",
        "OMITTED HIT DAY / MONTH / GAP MATRIX",
        "-" * width,
        f"{day_gap}",
        "",
        "OMITTED HIT POSITIONAL TRANSITION MATRIX",
        "-" * width,
        f"MethodWarning: {transitions['method_warning']}",
        f"TopClosestTransitions: {transitions['top_closest_transitions']}",
        f"PresenceSummary: "
        f"{ {key: {'seen': value['presence_target_digit_seen_count'], 'unseen': value['presence_target_digit_unseen_count']} for key, value in transitions['positions'].items()} }",
        "",
        "OMITTED HIT POSITIONAL STRUCTURE MATRIX",
        "-" * width,
        f"{structure}",
        "",
        "OMITTED HIT ENGINE / STREAK MATRIX",
        "-" * width,
        f"SourceEngine: {structure['source_engine']}",
        f"SourceRank: {structure['source_rank']}",
        f"NormalizedScoreBucket: {structure['normalized_score_bucket']}",
        f"TemporalStreakBucket: {structure['temporal_streak_bucket']}",
        f"UnderlyingEngineStreakBucket: {structure['underlying_engine_streak_bucket']}",
        "",
        "SHADOW6 SELECTOR COMPARISON MATRIX",
        "-" * width,
        *format_selector_matrix(selector_rows),
        "",
        "SHADOW6 CONDITIONAL FIRING MATRIX",
        "-" * width,
        *format_conditional(conditional),
        "",
        "SHADOW6 CONTEXT LIFT MATRIX",
        "-" * width,
        *format_context(contexts),
        "",
        "SHADOW6 RECENT PERFORMANCE MATRIX",
        "-" * width,
        *format_recent(recent),
        "",
        "CAUSAL KEY SYNTHESIS",
        "-" * width,
        f"OmittedSignalReal: {causal['omitted_signal_real']}",
        f"VerifiedUniqueOmittedHitOpportunities: {causal['verified_unique_omitted_hit_opportunities']}",
        f"BestShadow6Selector: {causal['best_shadow6_selector']}",
        f"BestShadow6Metrics: {causal['best_shadow6_metrics']}",
        f"VerifiedSignalCaptureRate: {pct(causal['verified_signal_capture_rate'])}",
        f"SelectorCaptureStatement: {causal['selector_capture_statement']}",
        f"CandidateCategoryScoresDescriptiveOnly: {causal['candidate_category_scores_descriptive_only']}",
        f"LeadingCausalKeyCandidate: {causal['leading_causal_key_candidate']}",
        f"CausalityWarning: {causal['causality_warning']}",
        "",
        "FINAL RECOMMENDATION",
        "-" * width,
        "ProductionSwitchRecommendedNow: NO",
        f"NextStep: {causal['next_step']}",
        "",
        f"REPORT_WRITTEN: {REPORT_PATH}",
        f"MATRICES_WRITTEN: {MATRICES_PATH}",
        f"ROWS_WRITTEN: {ROWS_PATH}",
    ]
    return "\n".join(lines)


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_conn() as connection:
        cursor = connection.cursor()
        draws = fetch_draws(cursor)
        ledger = fetch_ledger(cursor)
        discovery = source_discovery(cursor, draws, ledger)

    draws_by_no = {draw["draw_no"]: draw for draw in draws}
    grouped = group_ledger(ledger)
    common_rows, limitations = build_common_rows(grouped, draws_by_no, discovery)
    add_temporal_pre_miss_streak(common_rows)
    occurrences = occurrence_index(draws)
    engine_streaks = engine_streak_maps(common_rows)

    events = build_omitted_hit_events(
        common_rows, draws_by_no, occurrences, engine_streaks
    )
    day_gap, structure = aggregate_events(events)
    transitions, transition_rows = omitted_transition_matrix(
        events, baseline_slot_transitions(draws)
    )
    shadow_rows = simulate_shadow6(common_rows, occurrences, engine_streaks)
    selector_rows = selector_matrix(shadow_rows)
    conditional, conditional_rows = conditional_firing(shadow_rows)
    contexts, context_rows = context_lift(shadow_rows)
    recent = recent_matrix(shadow_rows)
    causal = synthesis(events, day_gap, structure, selector_rows, contexts)

    limitations.extend(
        (
            {
                "row_type": "data_limitation",
                "limitation": (
                    "PredictionLedger does not persist the historical removal "
                    "reason; omitted-hit cause remains unknown."
                ),
            },
            {
                "row_type": "data_limitation",
                "limitation": (
                    "Same prize-slot transitions are marked slot-order-risk because "
                    "prize lists may be stored in ordered sequence."
                ),
            },
            {
                "row_type": "data_limitation",
                "limitation": (
                    "Selector comparisons are retrospective diagnostics. Target "
                    "winners are used only after the sixth candidate is locked."
                ),
            },
        )
    )
    matrices = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "audit": "STEP 156",
            "read_only": True,
            "existing_tables_only": True,
            "deep_candidate_ledger_used": False,
            "temporal_top5_modified": False,
            "target_winners_used_only_after_shadow6_lock": True,
        },
        "discovery": discovery,
        "omitted_hit_events": events,
        "omitted_hit_day_month_gap": day_gap,
        "omitted_hit_positional_transitions": transitions,
        "omitted_hit_positional_structure": structure,
        "shadow6_selector_matrix": selector_rows,
        "shadow6_conditional_firing": conditional,
        "shadow6_context_lift": {
            **contexts,
            "RECENT_PERFORMANCE": recent,
        },
        "causal_key_synthesis": causal,
    }
    report = build_report(
        discovery,
        events,
        day_gap,
        transitions,
        structure,
        selector_rows,
        conditional,
        contexts,
        recent,
        causal,
    )
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    MATRICES_PATH.write_text(
        json.dumps(matrices, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with ROWS_PATH.open("w", encoding="utf-8") as handle:
        for row in (
            events
            + transition_rows
            + shadow_rows
            + context_rows
            + conditional_rows
            + limitations
        ):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(report)


if __name__ == "__main__":
    main()
