from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import pyodbc
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
REPORT_PATH = (
    PROJECT_ROOT / "reports" / "step_154_positional_conditional_renewal_audit.txt"
)
MATRICES_PATH = (
    PROJECT_ROOT / "reports" / "step_154_positional_conditional_renewal_matrices.json"
)
ROWS_PATH = (
    PROJECT_ROOT / "reports" / "step_154_positional_conditional_renewal_rows.jsonl"
)

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))
load_dotenv(BACKEND_ROOT / ".env")

from app.core.config import get_settings


NUMBER_SPACE = 10_000
POSITIONS = ("Pos_1", "Pos_2", "Pos_3", "Pos_4")
ENGINE_ALIASES = {
    "Temporal": ("E1_TEMPORAL_CONTEXT_MATCH",),
    "Delta": ("E1_DELTA", "E1_DELTA_ROTATION_LSTS"),
    "WLS": ("E1_WLS", "E1_WLS_DECAY_0.98"),
    "Linear": ("E1_LINEAR", "E1_CROSS_PAIR_LINEAR"),
}
MODE_PRIORITY = {
    "Current": 0,
    "Temporal_Global_Loop": 1,
    "Historical": 2,
    "Engine_Grand_Loop": 3,
    "Grand_Loop": 4,
    "Weighted_Grand_Loop": 5,
}
STREAK_STATES = tuple(str(value) for value in range(16)) + (
    "16-20",
    "21-30",
    "31-50",
    "51+",
)


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


def sum_band(number: str) -> str:
    total = digit_sum(number)
    if total <= 9:
        return "00_09"
    if total <= 14:
        return "10_14"
    if total <= 19:
        return "15_19"
    if total <= 24:
        return "20_24"
    return "25_36"


def table_columns(cursor, table_name: str) -> list[dict]:
    rows = cursor.execute(
        """
        SELECT
            c.column_id,
            c.name AS ColumnName,
            t.name AS DataType,
            c.max_length,
            c.is_nullable
        FROM sys.columns c
        JOIN sys.types t ON t.user_type_id = c.user_type_id
        WHERE c.object_id = OBJECT_ID(?)
        ORDER BY c.column_id;
        """,
        table_name,
    ).fetchall()
    return [
        {
            "column_id": int(row.column_id),
            "name": str(row.ColumnName),
            "data_type": str(row.DataType),
            "max_length": int(row.max_length),
            "nullable": bool(row.is_nullable),
        }
        for row in rows
    ]


def fetch_draws(cursor) -> list[dict]:
    rows = cursor.execute(
        """
        SELECT
            DrawNo,
            CONVERT(varchar(10), DrawDate, 120) AS DrawDateText,
            DATEPART(YEAR, DrawDate) AS DrawYear,
            DATEPART(MONTH, DrawDate) AS DrawMonth,
            DATENAME(WEEKDAY, DrawDate) AS WeekdayName,
            DayType,
            WinningNumbers
        FROM dbo.DrawHistory
        WHERE WinningNumbers IS NOT NULL
        ORDER BY DrawNo;
        """
    ).fetchall()
    return [
        {
            "draw_no": int(row.DrawNo),
            "draw_date": str(row.DrawDateText) if row.DrawDateText else None,
            "year": int(row.DrawYear) if row.DrawYear else None,
            "month": int(row.DrawMonth) if row.DrawMonth else None,
            "weekday": str(row.WeekdayName) if row.WeekdayName else None,
            "day_type": str(row.DayType or "Unknown"),
            "winners": parse_numbers(row.WinningNumbers),
        }
        for row in rows
    ]


def fetch_ledger(cursor) -> list[dict]:
    rows = cursor.execute(
        """
        SELECT
            Mode,
            EngineSource,
            SourceDrawNo,
            TargetDrawNo,
            RankNo,
            PredictedNumber,
            Score,
            VerificationStatus,
            HitCount,
            CONVERT(varchar(33), CreatedAt, 126) AS CreatedAtText,
            CONVERT(varchar(33), VerifiedAt, 126) AS VerifiedAtText
        FROM dbo.PredictionLedger
        ORDER BY SourceDrawNo, TargetDrawNo, EngineSource, Mode, RankNo;
        """
    ).fetchall()
    return [
        {
            "mode": str(row.Mode),
            "engine_source": str(row.EngineSource or "UNKNOWN"),
            "source_draw_no": int(row.SourceDrawNo),
            "target_draw_no": int(row.TargetDrawNo),
            "rank": int(row.RankNo),
            "number": z4(row.PredictedNumber),
            "score": float(row.Score) if row.Score is not None else None,
            "verification_status": str(row.VerificationStatus),
            "ledger_hit_count": int(row.HitCount) if row.HitCount is not None else None,
            "created_at": str(row.CreatedAtText) if row.CreatedAtText else None,
            "verified_at": str(row.VerifiedAtText) if row.VerifiedAtText else None,
        }
        for row in rows
    ]


def source_discovery(cursor, draws: list[dict], ledger: list[dict]) -> dict:
    engine_modes = Counter(
        (row["engine_source"], row["mode"]) for row in ledger
    )
    engines = sorted({row["engine_source"] for row in ledger})
    aliases = {
        label: [
            engine
            for engine in engines
            if engine in preferred
            or label.lower() in engine.lower()
        ]
        for label, preferred in ENGINE_ALIASES.items()
    }
    ledger_columns = table_columns(cursor, "dbo.PredictionLedger")
    winner_columns = [
        item["name"]
        for item in table_columns(cursor, "dbo.DrawHistory")
        if any(token in item["name"].lower() for token in ("winner", "prize", "number"))
    ]
    return {
        "drawhistory_columns": table_columns(cursor, "dbo.DrawHistory"),
        "draw_range": [draws[0]["draw_no"], draws[-1]["draw_no"]] if draws else None,
        "draw_rows": len(draws),
        "winner_prize_columns": winner_columns,
        "draw_date_available": any(draw["draw_date"] for draw in draws),
        "day_type_values": sorted({draw["day_type"] for draw in draws}),
        "predictionledger_columns": ledger_columns,
        "predictionledger_rows": len(ledger),
        "distinct_engine_sources": engines,
        "distinct_modes": sorted({row["mode"] for row in ledger}),
        "rank_columns": [
            item["name"] for item in ledger_columns if "rank" in item["name"].lower()
        ],
        "score_columns": [
            item["name"]
            for item in ledger_columns
            if any(token in item["name"].lower() for token in ("score", "confidence"))
        ],
        "hitcount_available": any(item["name"] == "HitCount" for item in ledger_columns),
        "verification_status_available": any(
            item["name"] == "VerificationStatus" for item in ledger_columns
        ),
        "engine_alias_availability": aliases,
        "ledger_volume_by_engine_mode": {
            f"{engine}::{mode}": count
            for (engine, mode), count in sorted(engine_modes.items())
        },
        "uses_deep_candidate_ledger": False,
    }


def frequency_payload(counter: Counter[int]) -> dict[str, dict]:
    total = sum(counter.values())
    return {
        str(digit): {
            "count": int(counter[digit]),
            "percentage": counter[digit] / total if total else 0.0,
        }
        for digit in range(10)
    }


def positional_frequency(draws: list[dict]) -> tuple[dict, list[dict]]:
    dimensions: dict[str, Callable[[dict], str]] = {
        "full_history": lambda draw: "ALL",
        "decade": lambda draw: (
            f"{(draw['year'] // 10) * 10}s" if draw["year"] is not None else "Unknown"
        ),
        "year": lambda draw: str(draw["year"]),
        "month": lambda draw: str(draw["month"]),
        "weekday": lambda draw: str(draw["weekday"]),
        "day_type": lambda draw: draw["day_type"],
    }
    counters: dict[str, dict[str, list[Counter[int]]]] = {
        dimension: defaultdict(lambda: [Counter() for _ in range(4)])
        for dimension in dimensions
    }
    for draw in draws:
        for number in draw["winners"]:
            values = digits(number)
            for dimension, key_fn in dimensions.items():
                key = key_fn(draw)
                for position, value in enumerate(values):
                    counters[dimension][key][position][value] += 1

    output = {}
    rows = []
    for dimension, contexts in counters.items():
        output[dimension] = {}
        for context, position_counters in sorted(contexts.items()):
            output[dimension][context] = {}
            for position, counter in enumerate(position_counters):
                position_name = POSITIONS[position]
                output[dimension][context][position_name] = frequency_payload(counter)
                for digit in range(10):
                    item = output[dimension][context][position_name][str(digit)]
                    rows.append(
                        {
                            "row_type": "positional_frequency",
                            "dimension": dimension,
                            "context": context,
                            "position": position_name,
                            "digit": digit,
                            **item,
                        }
                    )
    return output, rows


def positional_transitions(draws: list[dict]) -> tuple[dict, list[dict]]:
    matrices = [[[0 for _ in range(10)] for _ in range(10)] for _ in range(4)]
    transition_pairs = 0
    for previous, current in zip(draws, draws[1:]):
        slot_count = min(len(previous["winners"]), len(current["winners"]))
        for slot in range(slot_count):
            before = digits(previous["winners"][slot])
            after = digits(current["winners"][slot])
            for position in range(4):
                matrices[position][before[position]][after[position]] += 1
        transition_pairs += slot_count

    output = {
        "method": "same prize-list slot across consecutive draws",
        "consecutive_draw_pairs": max(0, len(draws) - 1),
        "prize_slot_pairs": transition_pairs,
        "positions": {},
    }
    rows = []
    for position, counts in enumerate(matrices):
        probabilities = []
        for source_digit in range(10):
            total = sum(counts[source_digit])
            probabilities.append(
                [
                    counts[source_digit][target_digit] / total if total else 0.0
                    for target_digit in range(10)
                ]
            )
            for target_digit in range(10):
                rows.append(
                    {
                        "row_type": "positional_transition",
                        "position": POSITIONS[position],
                        "previous_digit": source_digit,
                        "next_digit": target_digit,
                        "count": counts[source_digit][target_digit],
                        "probability": probabilities[source_digit][target_digit],
                    }
                )
        top = sorted(
            (
                {
                    "previous_digit": source_digit,
                    "next_digit": target_digit,
                    "count": counts[source_digit][target_digit],
                    "probability": probabilities[source_digit][target_digit],
                    "lift_vs_uniform": probabilities[source_digit][target_digit] / 0.1,
                }
                for source_digit in range(10)
                for target_digit in range(10)
            ),
            key=lambda item: (-item["probability"], -item["count"], item["previous_digit"], item["next_digit"]),
        )[:15]
        output["positions"][POSITIONS[position]] = {
            "count_matrix": counts,
            "probability_matrix": probabilities,
            "top_transitions": top,
        }
    pos1_peak = output["positions"]["Pos_1"]["top_transitions"][0]
    output["ordering_artifact_assessment"] = {
        "suspected": pos1_peak["lift_vs_uniform"] >= 3.0,
        "evidence": (
            "Same prize-list slots show extreme thousands-digit persistence while "
            "unconditional positional frequencies remain uniform. Starter and "
            "consolation lists are commonly stored in ordered numeric sequence, so "
            "this is not treated as a predictive draw-to-draw transition."
        ),
        "strongest_same_slot_transition": pos1_peak,
    }
    return output, rows


def positional_alignment(draws: list[dict], frequency: dict) -> dict:
    high_low = Counter()
    parity = Counter()
    mirror_profiles = Counter()
    repeated_patterns = Counter()
    first_pairs = Counter()
    last_pairs = Counter()
    edge_pairs = Counter()
    inner_pairs = Counter()
    position_sums = [0, 0, 0, 0]
    total_numbers = 0
    for draw in draws:
        for number in draw["winners"]:
            values = digits(number)
            high_low["".join("H" if value >= 5 else "L" for value in values)] += 1
            parity["".join("O" if value % 2 else "E" for value in values)] += 1
            mirror_profiles[mirror_signature(number)] += 1
            repeated_patterns[f"unique_{len(set(number))}"] += 1
            first_pairs[number[:2]] += 1
            last_pairs[number[2:]] += 1
            edge_pairs[number[0] + number[3]] += 1
            inner_pairs[number[1:3]] += 1
            for position, value in enumerate(values):
                position_sums[position] += value
            total_numbers += 1

    global_freq = frequency["full_history"]["ALL"]
    positional_strength = []
    for position_name in POSITIONS:
        percentages = [
            global_freq[position_name][str(digit)]["percentage"] for digit in range(10)
        ]
        max_deviation = max(abs(value - 0.1) for value in percentages)
        classification = (
            "strong trend"
            if max_deviation >= 0.01
            else "weak trend"
            if max_deviation >= 0.003
            else "random-like"
        )
        positional_strength.append(
            {
                "position": position_name,
                "max_absolute_deviation_from_uniform": max_deviation,
                "classification": classification,
            }
        )

    context_shifts = []
    for dimension in ("month", "weekday", "day_type", "year", "decade"):
        for context, positions in frequency[dimension].items():
            for position_name in POSITIONS:
                local = [
                    positions[position_name][str(digit)]["percentage"]
                    for digit in range(10)
                ]
                global_values = [
                    global_freq[position_name][str(digit)]["percentage"]
                    for digit in range(10)
                ]
                total_variation = 0.5 * sum(
                    abs(left - right) for left, right in zip(local, global_values)
                )
                sample_count = sum(
                    positions[position_name][str(digit)]["count"] for digit in range(10)
                )
                context_shifts.append(
                    {
                        "dimension": dimension,
                        "context": context,
                        "position": position_name,
                        "sample_count": sample_count,
                        "total_variation_from_global": total_variation,
                        "classification": (
                            "data-limited"
                            if sample_count < 300
                            else "strong trend"
                            if total_variation >= 0.08
                            else "weak trend"
                            if total_variation >= 0.03
                            else "random-like"
                        ),
                    }
                )

    return {
        "total_winning_numbers": total_numbers,
        "high_low_profiles_top20": high_low.most_common(20),
        "odd_even_profiles_top20": parity.most_common(20),
        "mirror_profiles_top20": mirror_profiles.most_common(20),
        "repeated_position_patterns": dict(repeated_patterns),
        "first_pairs_top20": first_pairs.most_common(20),
        "last_pairs_top20": last_pairs.most_common(20),
        "edge_pairs_top20": edge_pairs.most_common(20),
        "inner_pairs_top20": inner_pairs.most_common(20),
        "mean_digit_contribution_by_position": {
            POSITIONS[position]: position_sums[position] / total_numbers
            if total_numbers
            else 0.0
            for position in range(4)
        },
        "positional_strength": positional_strength,
        "largest_context_shifts": sorted(
            context_shifts,
            key=lambda item: (
                -item["total_variation_from_global"],
                -item["sample_count"],
                item["dimension"],
                item["context"],
            ),
        )[:30],
    }


def engine_label(engine_source: str) -> str | None:
    for label, aliases in ENGINE_ALIASES.items():
        if engine_source in aliases or label.lower() in engine_source.lower():
            return label
    return None


def build_engine_outcomes(
    ledger: list[dict],
    draws_by_no: dict[int, dict],
) -> list[dict]:
    grouped: dict[tuple[str, str, int, int], list[dict]] = defaultdict(list)
    for row in ledger:
        if engine_label(row["engine_source"]) is None:
            continue
        grouped[
            (
                row["engine_source"],
                row["mode"],
                row["source_draw_no"],
                row["target_draw_no"],
            )
        ].append(row)

    outcomes = []
    for (engine, mode, source, target), items in sorted(grouped.items()):
        target_draw = draws_by_no.get(target)
        ranked = sorted(items, key=lambda item: item["rank"])
        top5 = [item["number"] for item in ranked if item["rank"] <= 5]
        if target_draw is None or len(top5) != 5:
            continue
        actuals = set(target_draw["winners"])
        raw_hits = len(set(top5) & actuals)
        ledger_group_hit = max(
            (item["ledger_hit_count"] or 0 for item in ranked), default=0
        )
        locked_before_verification = all(
            item["verified_at"] is None
            or item["created_at"] is None
            or item["created_at"] <= item["verified_at"]
            for item in ranked
        )
        outcomes.append(
            {
                "engine_label": engine_label(engine),
                "engine_source": engine,
                "mode": mode,
                "source_draw_no": source,
                "target_draw_no": target,
                "top5": top5,
                "raw_hits": raw_hits,
                "outcome_hit": raw_hits > 0,
                "ledger_group_hit_count": ledger_group_hit,
                "ledger_actual_mismatch": (ledger_group_hit > 0) != (raw_hits > 0),
                "actual_prize_count": len(actuals),
                "target_date": target_draw["draw_date"],
                "year": target_draw["year"],
                "month": target_draw["month"],
                "weekday": target_draw["weekday"],
                "day_type": target_draw["day_type"],
                "locked_before_verification": locked_before_verification,
            }
        )
    return outcomes


def streak_state(streak: int) -> str:
    if streak <= 15:
        return str(streak)
    if streak <= 20:
        return "16-20"
    if streak <= 30:
        return "21-30"
    if streak <= 50:
        return "31-50"
    return "51+"


def classify_state(samples: int, hit_rate: float, random_rate: float) -> str:
    if samples < 30:
        return "LOW_SAMPLE_SPIKE" if hit_rate > random_rate else "data-limited"
    if random_rate and hit_rate >= random_rate * 1.25:
        return "RENEWAL_SIGNAL"
    if random_rate and hit_rate <= random_rate * 0.80:
        return "STREAK_RISK"
    return "NO_RENEWAL_EDGE"


def conditional_renewal(outcomes: list[dict]) -> tuple[dict, list[dict], dict]:
    by_stream: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for outcome in outcomes:
        by_stream[(outcome["engine_source"], outcome["mode"])].append(outcome)

    stream_rows: dict[str, list[dict]] = {}
    outcome_streak_map: dict[tuple[str, str, int, int], int] = {}
    json_rows = []
    summaries = {}
    for (engine, mode), items in sorted(by_stream.items()):
        current_streak = 0
        enriched = []
        for item in sorted(
            items, key=lambda value: (value["source_draw_no"], value["target_draw_no"])
        ):
            enriched_item = {**item, "pre_miss_streak": current_streak}
            enriched.append(enriched_item)
            outcome_streak_map[
                (engine, mode, item["source_draw_no"], item["target_draw_no"])
            ] = current_streak
            current_streak = 0 if item["outcome_hit"] else current_streak + 1

        state_groups: dict[str, list[dict]] = defaultdict(list)
        for item in enriched:
            state_groups[streak_state(item["pre_miss_streak"])].append(item)
        rows = []
        for state in STREAK_STATES:
            state_items = state_groups.get(state, [])
            samples = len(state_items)
            hit_draws = sum(item["outcome_hit"] for item in state_items)
            raw_hits = sum(item["raw_hits"] for item in state_items)
            random_rate = (
                statistics.mean(
                    1
                    - (
                        (NUMBER_SPACE - item["actual_prize_count"]) / NUMBER_SPACE
                    )
                    ** 5
                    for item in state_items
                )
                if state_items
                else 0.0
            )
            hit_rate = hit_draws / samples if samples else 0.0
            row = {
                "engine_label": engine_label(engine),
                "engine_source": engine,
                "mode": mode,
                "streak_state": state,
                "samples": samples,
                "hit_draws": hit_draws,
                "miss_draws": samples - hit_draws,
                "raw_hits": raw_hits,
                "empirical_hit_rate": hit_rate,
                "random_expected_top5_hit_rate": random_rate,
                "enrichment_vs_random": hit_rate / random_rate
                if random_rate
                else None,
                "confidence_warning": samples < 30,
                "interpretation": classify_state(samples, hit_rate, random_rate)
                if samples
                else "data-limited",
            }
            rows.append(row)
            json_rows.append({"row_type": "engine_streak_state", **row})
        key = f"{engine}::{mode}"
        stream_rows[key] = rows
        sufficient = [row for row in rows if row["samples"] >= 30]
        renewal_states = [
            row["streak_state"]
            for row in sufficient
            if row["interpretation"] == "RENEWAL_SIGNAL"
        ]
        risk_states = [
            row["streak_state"]
            for row in sufficient
            if row["interpretation"] == "STREAK_RISK"
        ]
        summaries[key] = {
            "engine_label": engine_label(engine),
            "engine_source": engine,
            "mode": mode,
            "outcome_rows": len(enriched),
            "hit_draws": sum(item["outcome_hit"] for item in enriched),
            "raw_hits": sum(item["raw_hits"] for item in enriched),
            "maximum_pre_miss_streak": max(
                (item["pre_miss_streak"] for item in enriched), default=0
            ),
            "renewal_signal_states": renewal_states,
            "streak_risk_states": risk_states,
            "low_sample_spikes": [
                row["streak_state"]
                for row in rows
                if row["interpretation"] == "LOW_SAMPLE_SPIKE"
            ],
            "ledger_actual_mismatch_count": sum(
                item["ledger_actual_mismatch"] for item in enriched
            ),
        }
    return (
        {"streams": stream_rows, "strength_summary": summaries},
        json_rows,
        outcome_streak_map,
    )


def selected_final_by_pair(
    outcomes: list[dict],
) -> dict[tuple[int, int], dict]:
    output = {}
    for item in outcomes:
        if item["engine_label"] != "Temporal":
            continue
        key = (item["source_draw_no"], item["target_draw_no"])
        old = output.get(key)
        if old is None or MODE_PRIORITY.get(item["mode"], 99) < MODE_PRIORITY.get(
            old["mode"], 99
        ):
            output[key] = item
    return output


def dropped_winners(
    ledger: list[dict],
    draws_by_no: dict[int, dict],
    outcomes: list[dict],
    streak_map: dict[tuple[str, str, int, int], int],
) -> tuple[dict, list[dict]]:
    grouped: dict[tuple[str, str, int, int], list[dict]] = defaultdict(list)
    for row in ledger:
        grouped[
            (
                row["engine_source"],
                row["mode"],
                row["source_draw_no"],
                row["target_draw_no"],
            )
        ].append(row)
    final_pairs = selected_final_by_pair(outcomes)
    events = []
    for (source, target), final in sorted(final_pairs.items()):
        target_draw = draws_by_no.get(target)
        if target_draw is None:
            continue
        actuals = set(target_draw["winners"])
        final_set = set(final["top5"])
        for (engine, mode, group_source, group_target), items in grouped.items():
            label = engine_label(engine)
            if label not in {"Delta", "WLS", "Linear"}:
                continue
            if (group_source, group_target) != (source, target):
                continue
            ranked = sorted(items, key=lambda item: item["rank"])
            if len([item for item in ranked if item["rank"] <= 5]) != 5:
                continue
            locked_before_verification = all(
                item["verified_at"] is None
                or item["created_at"] is None
                or item["created_at"] <= item["verified_at"]
                for item in ranked
            )
            if not locked_before_verification:
                continue
            for item in ranked:
                if item["rank"] > 5:
                    continue
                number = item["number"]
                if number in final_set or number not in actuals:
                    continue
                values = digits(number)
                pre_streak = streak_map.get(
                    (
                        final["engine_source"],
                        final["mode"],
                        source,
                        target,
                    )
                )
                events.append(
                    {
                        "row_type": "dropped_winner",
                        "source_draw_no": source,
                        "target_draw_no": target,
                        "target_date": target_draw["draw_date"],
                        "year": target_draw["year"],
                        "month": target_draw["month"],
                        "weekday": target_draw["weekday"],
                        "day_type": target_draw["day_type"],
                        "source_engine": engine,
                        "source_engine_label": label,
                        "source_mode": mode,
                        "source_rank": item["rank"],
                        "source_score": item["score"],
                        "dropped_number": number,
                        "final_engine": final["engine_source"],
                        "final_mode": final["mode"],
                        "final_top5": final["top5"],
                        "actual_prize_match": True,
                        "pos_1": values[0],
                        "pos_2": values[1],
                        "pos_3": values[2],
                        "pos_4": values[3],
                        "positional_pattern": "".join(str(value) for value in values),
                        "digit_sum": sum(values),
                        "digit_sum_band": sum_band(number),
                        "mirror_signature": mirror_signature(number),
                        "first_pair": number[:2],
                        "last_pair": number[2:],
                        "box_signature": box_signature(number),
                        "final_engine_pre_miss_streak": pre_streak,
                        "removal_reason": "unknown",
                    }
                )

    dimensions: dict[str, Callable[[dict], str]] = {
        "source_engine": lambda row: row["source_engine"],
        "source_engine_mode": lambda row: f"{row['source_engine']}::{row['source_mode']}",
        "final_engine_mode": lambda row: f"{row['final_engine']}::{row['final_mode']}",
        "year": lambda row: str(row["year"]),
        "month": lambda row: str(row["month"]),
        "weekday": lambda row: str(row["weekday"]),
        "day_type": lambda row: row["day_type"],
        "digit_sum_band": lambda row: row["digit_sum_band"],
        "mirror_signature": lambda row: row["mirror_signature"],
        "first_pair": lambda row: row["first_pair"],
        "last_pair": lambda row: row["last_pair"],
        "positional_pattern": lambda row: row["positional_pattern"],
        "final_pre_miss_streak": lambda row: str(
            row["final_engine_pre_miss_streak"]
        ),
        "removal_reason": lambda row: row["removal_reason"],
    }
    matrix = {
        "total_events": len(events),
        "unique_source_target_pairs": len(
            {(row["source_draw_no"], row["target_draw_no"]) for row in events}
        ),
        "aggregates": {
            dimension: dict(Counter(key_fn(row) for row in events).most_common())
            for dimension, key_fn in dimensions.items()
        },
        "most_damaging_sources": Counter(
            row["source_engine"] for row in events
        ).most_common(),
        "removal_cause_limitation": (
            "Exact guard/reranker cause is not persisted in PredictionLedger; "
            "all events are marked unknown."
        ),
    }
    return matrix, events


def top_transition_findings(transitions: dict) -> list[dict]:
    output = []
    for position, payload in transitions["positions"].items():
        for item in payload["top_transitions"][:5]:
            output.append({"position": position, **item})
    return sorted(
        output,
        key=lambda item: (-item["lift_vs_uniform"], -item["count"], item["position"]),
    )


def primary_stream_key(
    renewal: dict,
    label: str,
) -> str | None:
    candidates = [
        (key, summary)
        for key, summary in renewal["strength_summary"].items()
        if summary["engine_label"] == label
    ]
    if not candidates:
        return None
    preferred_mode = {
        "Temporal": ("Temporal_Global_Loop", "Current", "Historical"),
        "Delta": ("Engine_Grand_Loop", "Historical"),
        "WLS": ("Engine_Grand_Loop", "Historical"),
        "Linear": ("Engine_Grand_Loop", "Historical"),
    }[label]
    return min(
        candidates,
        key=lambda item: (
            preferred_mode.index(item[1]["mode"])
            if item[1]["mode"] in preferred_mode
            else 99,
            -item[1]["outcome_rows"],
            item[0],
        ),
    )[0]


def cause_effect_synthesis(
    alignment: dict,
    transitions: dict,
    renewal: dict,
    dropped: dict,
) -> dict:
    strongest_position = max(
        alignment["positional_strength"],
        key=lambda item: item["max_absolute_deviation_from_uniform"],
    )
    top_transition = top_transition_findings(transitions)[0]
    engine_findings = {}
    real_renewal = False
    for label in ("Temporal", "Delta", "WLS", "Linear"):
        key = primary_stream_key(renewal, label)
        if key is None:
            engine_findings[label] = {"status": "unavailable"}
            continue
        summary = renewal["strength_summary"][key]
        real_renewal = real_renewal or bool(summary["renewal_signal_states"])
        engine_findings[label] = {
            "primary_stream": key,
            "renewal_signal_states": summary["renewal_signal_states"],
            "streak_risk_states": summary["streak_risk_states"],
            "low_sample_spikes": summary["low_sample_spikes"],
        }

    dropped_total = dropped["total_events"]
    positional_is_strong = (
        strongest_position["classification"] == "strong trend"
        and top_transition["count"] >= 500
        and top_transition["lift_vs_uniform"] >= 1.20
    )
    if dropped_total >= 30 and real_renewal:
        key_type = "mixed: miss-streak renewal associations and guard/meta over-suppression"
        next_step = "build renewal-aware suppression layer and reduce final meta over-filtering"
    elif dropped_total >= 30 and not real_renewal:
        key_type = "guard/meta over-suppression"
        next_step = "reduce final meta over-filtering"
    elif positional_is_strong and real_renewal:
        key_type = "mixed"
        next_step = "build positional-column transition engine with renewal-aware suppression"
    elif positional_is_strong:
        key_type = "positional-column transition logic"
        next_step = "build positional-column transition engine"
    elif real_renewal:
        key_type = "miss-streak renewal logic"
        next_step = "build renewal-aware suppression layer"
    else:
        key_type = "mixed data-limited signal"
        next_step = "gather deeper candidate pools first"

    return {
        "strongest_position": strongest_position,
        "strongest_transition": top_transition,
        "transition_ordering_artifact": transitions[
            "ordering_artifact_assessment"
        ],
        "largest_context_shifts": alignment["largest_context_shifts"][:10],
        "engine_renewal_findings": engine_findings,
        "dropped_winner_total": dropped_total,
        "largest_dropped_source": (
            dropped["most_damaging_sources"][0]
            if dropped["most_damaging_sources"]
            else None
        ),
        "key_diagnosis": key_type,
        "recommended_next_step": next_step,
        "causality_warning": (
            "Associations are descriptive. No removal rule is claimed causal "
            "because rule metadata is not persisted."
        ),
    }


def pct(value: float | None) -> str:
    return "NULL" if value is None else f"{value * 100:.4f}%"


def format_frequency_table(frequency: dict) -> list[str]:
    lines = [
        "Position  " + " ".join(f"D{digit:>1} Count      Pct" for digit in range(10))
    ]
    full = frequency["full_history"]["ALL"]
    for position in POSITIONS:
        cells = []
        for digit in range(10):
            item = full[position][str(digit)]
            cells.append(f"{item['count']:>7} {item['percentage'] * 100:>7.3f}%")
        lines.append(f"{position:<9} " + " ".join(cells))
    return lines


def format_transition_table(transitions: dict) -> list[str]:
    lines = [
        "Position Prev Next Count Probability LiftVsUniform"
    ]
    for finding in top_transition_findings(transitions)[:20]:
        lines.append(
            f"{finding['position']:<8} {finding['previous_digit']:>4} "
            f"{finding['next_digit']:>4} {finding['count']:>7} "
            f"{finding['probability'] * 100:>10.4f}% "
            f"{finding['lift_vs_uniform']:>13.4f}"
        )
    return lines


def format_streak_table(rows: list[dict]) -> list[str]:
    lines = [
        "State   Samples Hit Miss RawHits HitRate RandomTop5 Enrich Interpretation"
    ]
    for row in rows:
        enrichment = (
            "NULL"
            if row["enrichment_vs_random"] is None
            else f"{row['enrichment_vs_random']:.4f}"
        )
        lines.append(
            f"{row['streak_state']:<7} {row['samples']:>7} "
            f"{row['hit_draws']:>3} {row['miss_draws']:>4} "
            f"{row['raw_hits']:>7} {pct(row['empirical_hit_rate']):>9} "
            f"{pct(row['random_expected_top5_hit_rate']):>10} "
            f"{enrichment:>7} {row['interpretation']}"
        )
    return lines


def build_report(
    discovery: dict,
    frequency: dict,
    transitions: dict,
    alignment: dict,
    renewal: dict,
    dropped: dict,
    synthesis: dict,
) -> str:
    width = 172
    lines = [
        "=" * width,
        "STEP 154 — POSITIONAL COLUMN & CONDITIONAL RENEWAL STREAK AUDIT — REPORT ONLY",
        "=" * width,
        "ProductionMathChanged: NO",
        "APIChanged: NO",
        "FrontendChanged: NO",
        "SQLSchemaChanged: NO",
        "DBWritePerformed: NO",
        "UsesExistingTablesOnly: YES",
        "DeepCandidateLedgerUsed: NO",
        "",
        "SOURCE DISCOVERY SUMMARY",
        "-" * width,
        f"DrawHistoryColumns: {[item['name'] for item in discovery['drawhistory_columns']]}",
        f"DrawRange: {discovery['draw_range']}",
        f"DrawRows: {discovery['draw_rows']}",
        f"WinnerPrizeColumns: {discovery['winner_prize_columns']}",
        f"DayTypeValues: {discovery['day_type_values']}",
        f"PredictionLedgerColumns: {[item['name'] for item in discovery['predictionledger_columns']]}",
        f"PredictionLedgerRows: {discovery['predictionledger_rows']}",
        f"EngineAliases: {discovery['engine_alias_availability']}",
        f"LedgerVolumeByEngineMode: {discovery['ledger_volume_by_engine_mode']}",
        "",
        "POSITIONAL FREQUENCY MATRIX — FULL HISTORY",
        "-" * width,
        *format_frequency_table(frequency),
        "",
        "POSITIONAL TRANSITION MATRIX — TOP TRANSITIONS",
        "-" * width,
        f"TransitionMethod: {transitions['method']}",
        f"ConsecutiveDrawPairs: {transitions['consecutive_draw_pairs']}",
        f"PrizeSlotPairs: {transitions['prize_slot_pairs']}",
        f"OrderingArtifactAssessment: {transitions['ordering_artifact_assessment']}",
        *format_transition_table(transitions),
        "",
        "TOP POSITIONAL TRENDS",
        "-" * width,
        f"PositionalStrength: {alignment['positional_strength']}",
        f"MeanDigitContribution: {alignment['mean_digit_contribution_by_position']}",
        f"LargestContextShifts: {alignment['largest_context_shifts'][:15]}",
        "",
        "COLUMN ALIGNMENT FINDINGS",
        "-" * width,
        f"HighLowProfilesTop20: {alignment['high_low_profiles_top20']}",
        f"OddEvenProfilesTop20: {alignment['odd_even_profiles_top20']}",
        f"RepeatedPositionPatterns: {alignment['repeated_position_patterns']}",
        f"FirstPairsTop20: {alignment['first_pairs_top20']}",
        f"LastPairsTop20: {alignment['last_pairs_top20']}",
        f"EdgePairsTop20: {alignment['edge_pairs_top20']}",
        f"InnerPairsTop20: {alignment['inner_pairs_top20']}",
    ]
    for label in ("Temporal", "Delta", "WLS", "Linear"):
        key = primary_stream_key(renewal, label)
        lines.extend(
            (
                "",
                f"CONDITIONAL MISS-TO-HIT MATRIX — {label.upper()}",
                "-" * width,
            )
        )
        if key is None:
            lines.append("DATA_LIMITATION: engine stream unavailable")
        else:
            lines.append(f"PrimaryStream: {key}")
            lines.extend(format_streak_table(renewal["streams"][key]))

    lines.extend(
        (
            "",
            "ENGINE RENEWAL STRENGTH SUMMARY",
            "-" * width,
            f"{renewal['strength_summary']}",
            "",
            "STREAK STATES WITH REAL SIGNAL",
            "-" * width,
            f"{ {key: value['renewal_signal_states'] for key, value in renewal['strength_summary'].items()} }",
            "",
            "STREAK STATES TO SUPPRESS",
            "-" * width,
            f"{ {key: value['streak_risk_states'] for key, value in renewal['strength_summary'].items()} }",
            "",
            "DROPPED WINNER MATRIX",
            "-" * width,
            f"TotalDroppedWinnerEvents: {dropped['total_events']}",
            f"UniqueSourceTargetPairs: {dropped['unique_source_target_pairs']}",
            f"BySourceEngine: {dropped['aggregates']['source_engine']}",
            f"BySourceEngineMode: {dropped['aggregates']['source_engine_mode']}",
            f"ByFinalEngineMode: {dropped['aggregates']['final_engine_mode']}",
            f"RemovalReasons: {dropped['aggregates']['removal_reason']}",
            "",
            "DROPPED WINNER PATTERN BY DATE CONTEXT",
            "-" * width,
            f"ByYear: {dropped['aggregates']['year']}",
            f"ByMonth: {dropped['aggregates']['month']}",
            f"ByWeekday: {dropped['aggregates']['weekday']}",
            f"ByDayType: {dropped['aggregates']['day_type']}",
            f"ByFinalPreMissStreak: {dropped['aggregates']['final_pre_miss_streak']}",
            "",
            "DROPPED WINNER PATTERN BY POSITIONAL STRUCTURE",
            "-" * width,
            f"ByDigitSumBand: {dropped['aggregates']['digit_sum_band']}",
            f"TopMirrorSignatures: {list(dropped['aggregates']['mirror_signature'].items())[:20]}",
            f"TopFirstPairs: {list(dropped['aggregates']['first_pair'].items())[:20]}",
            f"TopLastPairs: {list(dropped['aggregates']['last_pair'].items())[:20]}",
            f"TopPositionalPatterns: {list(dropped['aggregates']['positional_pattern'].items())[:20]}",
            "RemovalCause: unknown unless persisted metadata proves otherwise.",
            "",
            "MOST DAMAGING FILTER / META OMISSION SOURCES",
            "-" * width,
            f"MostDamagingSources: {dropped['most_damaging_sources']}",
            f"RemovalCauseLimitation: {dropped['removal_cause_limitation']}",
            "",
            "CAUSE-EFFECT SYNTHESIS",
            "-" * width,
            f"StrongestPosition: {synthesis['strongest_position']}",
            f"StrongestTransition: {synthesis['strongest_transition']}",
            f"TransitionOrderingArtifact: {synthesis['transition_ordering_artifact']}",
            f"EngineRenewalFindings: {synthesis['engine_renewal_findings']}",
            f"LargestDroppedSource: {synthesis['largest_dropped_source']}",
            f"KeyDiagnosis: {synthesis['key_diagnosis']}",
            f"CausalityWarning: {synthesis['causality_warning']}",
            "",
            "FINAL RECOMMENDATION",
            "-" * width,
            "ProductionSwitchRecommendedNow: NO",
            f"NextStep: {synthesis['recommended_next_step']}",
            "",
            f"REPORT_WRITTEN: {REPORT_PATH}",
            f"MATRICES_WRITTEN: {MATRICES_PATH}",
            f"ROWS_WRITTEN: {ROWS_PATH}",
        )
    )
    return "\n".join(lines)


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_conn() as connection:
        cursor = connection.cursor()
        draws = fetch_draws(cursor)
        ledger = fetch_ledger(cursor)
        discovery = source_discovery(cursor, draws, ledger)
    draws_by_no = {draw["draw_no"]: draw for draw in draws}

    frequency, frequency_rows = positional_frequency(draws)
    transitions, transition_rows = positional_transitions(draws)
    alignment = positional_alignment(draws, frequency)
    outcomes = build_engine_outcomes(ledger, draws_by_no)
    renewal, renewal_rows, streak_map = conditional_renewal(outcomes)
    dropped, dropped_rows = dropped_winners(
        ledger, draws_by_no, outcomes, streak_map
    )
    synthesis = cause_effect_synthesis(
        alignment, transitions, renewal, dropped
    )

    matrices = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "audit": "STEP 154",
            "read_only": True,
            "uses_existing_tables_only": True,
            "deep_candidate_ledger_used": False,
        },
        "discovery": discovery,
        "positional_frequency": frequency,
        "positional_transitions": transitions,
        "positional_alignment": alignment,
        "conditional_renewal": renewal,
        "dropped_winner_matrix": dropped,
        "cause_effect_synthesis": synthesis,
    }
    MATRICES_PATH.write_text(
        json.dumps(matrices, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    limitations = [
        {
            "row_type": "data_limitation",
            "limitation": (
                "Transition matrices assume WinningNumbers order preserves the same "
                "prize-list slot across consecutive draws."
            ),
        },
        {
            "row_type": "data_limitation",
            "limitation": (
                "PredictionLedger does not persist exact final removal rules; "
                "dropped-winner removal_reason is unknown."
            ),
        },
        {
            "row_type": "data_limitation",
            "limitation": (
                "Conditional engine outcomes use persisted Top5 rows and target winners "
                "only for post-lock verification."
            ),
        },
    ]
    with ROWS_PATH.open("w", encoding="utf-8") as handle:
        for row in [
            *frequency_rows,
            *transition_rows,
            *renewal_rows,
            *dropped_rows,
            *limitations,
        ]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    report = build_report(
        discovery,
        frequency,
        transitions,
        alignment,
        renewal,
        dropped,
        synthesis,
    )
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
