from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable

import pyodbc
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
REPORT_PATH = PROJECT_ROOT / "reports" / "step_155_renewal_aware_anti_drop_audit.txt"
MATRICES_PATH = (
    PROJECT_ROOT / "reports" / "step_155_renewal_aware_anti_drop_matrices.json"
)
ROWS_PATH = PROJECT_ROOT / "reports" / "step_155_renewal_aware_anti_drop_rows.jsonl"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))
load_dotenv(BACKEND_ROOT / ".env")

from app.core.config import get_settings


NUMBER_SPACE = 10_000
FINAL_STREAM = ("E1_TEMPORAL_CONTEXT_MATCH", "Temporal_Global_Loop")
UNDERLYING_STREAMS = {
    "Delta": ("E1_DELTA_ROTATION_LSTS", "Engine_Grand_Loop"),
    "WLS": ("E1_WLS_DECAY_0.98", "Engine_Grand_Loop"),
    "Linear": ("E1_CROSS_PAIR_LINEAR", "Engine_Grand_Loop"),
}
ENGINE_PRIORITY = {"Delta": 0, "WLS": 1, "Linear": 2}
POLICIES = (
    "NO_RESCUE_BASELINE",
    "ANY_UNDERLYING_SCORE_RESCUE",
    "SPECIAL_DAYTYPE_RESCUE",
    "AUGUST_CONTEXT_RESCUE",
    "TEMPORAL_RISK_STREAK_RESCUE_21_30",
    "TEMPORAL_RISK_STREAK_RESCUE_21_50",
    "DELTA_WLS_PRIORITY_RESCUE",
    "PAIR_00_CONTEXT_RESCUE",
    "COMBINED_SAFE_RESCUE_V1",
    "COMBINED_STRICT_RESCUE_V2",
)
CONTEXTS = (
    "FULL_RANGE",
    "SPECIAL_DAYTYPE",
    "SATURDAY_DAYTYPE",
    "AUGUST",
    "TEMPORAL_STREAK_21_30",
    "TEMPORAL_STREAK_31_50",
    "TEMPORAL_STREAK_51_PLUS",
)


def get_conn():
    return pyodbc.connect(get_settings().sql_connection_string(), timeout=120)


def z4(value: str | int) -> str:
    return str(value).strip().zfill(4)


def parse_numbers(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(z4(part) for part in str(value).replace(" ", "").split(",") if part)


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
            "verification_status": str(row.VerificationStatus or ""),
            "hit_count": int(row.HitCount) if row.HitCount is not None else None,
            "created_at": str(row.CreatedAtText) if row.CreatedAtText else None,
            "verified_at": str(row.VerifiedAtText) if row.VerifiedAtText else None,
        }
        for row in rows
    ]


def resolve_stream(
    available: Counter[tuple[str, str]],
    preferred: tuple[str, str],
    label: str,
    preferred_mode: str,
) -> tuple[str, str] | None:
    if available[preferred] > 0:
        return preferred
    candidates = [
        stream
        for stream, count in available.items()
        if count > 0
        and label.lower() in stream[0].lower()
        and stream[1] == preferred_mode
    ]
    if candidates:
        return max(candidates, key=lambda stream: (available[stream], stream))
    return None


def source_discovery(cursor, draws: list[dict], ledger: list[dict]) -> dict:
    volumes = Counter((row["engine_source"], row["mode"]) for row in ledger)
    resolved_final = resolve_stream(
        volumes, FINAL_STREAM, "Temporal", "Temporal_Global_Loop"
    )
    resolved_underlying = {
        label: resolve_stream(volumes, preferred, label, "Engine_Grand_Loop")
        for label, preferred in UNDERLYING_STREAMS.items()
    }
    return {
        "drawhistory_columns": table_columns(cursor, "dbo.DrawHistory"),
        "predictionledger_columns": table_columns(cursor, "dbo.PredictionLedger"),
        "draw_range": [draws[0]["draw_no"], draws[-1]["draw_no"]] if draws else None,
        "draw_rows": len(draws),
        "predictionledger_rows": len(ledger),
        "rank_column": "RankNo",
        "score_column": "Score",
        "hitcount_column": "HitCount",
        "verification_status_column": "VerificationStatus",
        "ledger_volume_by_engine_mode": {
            f"{engine}::{mode}": count
            for (engine, mode), count in sorted(volumes.items())
        },
        "resolved_final_stream": list(resolved_final) if resolved_final else None,
        "resolved_underlying_streams": {
            label: list(stream) if stream else None
            for label, stream in resolved_underlying.items()
        },
        "deep_candidate_ledger_used": False,
    }


def group_ledger(
    ledger: list[dict],
) -> dict[tuple[str, str, int, int], list[dict]]:
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
    return grouped


def ranked_top5(items: list[dict]) -> list[dict] | None:
    top = sorted(
        (item for item in items if item["rank"] <= 5),
        key=lambda item: (item["rank"], item["number"]),
    )
    if len(top) != 5 or len({item["number"] for item in top}) != 5:
        return None
    return top


def normalized_entries(items: list[dict], label: str) -> list[dict]:
    scores = [item["score"] for item in items if item["score"] is not None]
    low = min(scores) if scores else None
    high = max(scores) if scores else None
    output = []
    for item in items:
        if item["score"] is None or low is None or high is None:
            normalized_score = (6 - item["rank"]) / 5
        elif high == low:
            normalized_score = 1.0
        else:
            normalized_score = (item["score"] - low) / (high - low)
        output.append(
            {
                **item,
                "engine_label": label,
                "normalized_score": normalized_score,
                "rank_strength": (6 - item["rank"]) / 5,
            }
        )
    return output


def build_common_rows(
    grouped: dict[tuple[str, str, int, int], list[dict]],
    draws_by_no: dict[int, dict],
    discovery: dict,
) -> tuple[list[dict], list[dict]]:
    final_stream = tuple(discovery["resolved_final_stream"] or ())
    underlying_streams = {
        label: tuple(stream)
        for label, stream in discovery["resolved_underlying_streams"].items()
        if stream
    }
    limitations = []
    if len(final_stream) != 2:
        limitations.append(
            {
                "row_type": "data_limitation",
                "limitation": "Required Temporal final stream was unavailable.",
            }
        )
        return [], limitations
    if set(underlying_streams) != set(UNDERLYING_STREAMS):
        limitations.append(
            {
                "row_type": "data_limitation",
                "limitation": "One or more required underlying streams were unavailable.",
                "resolved_underlying_streams": discovery[
                    "resolved_underlying_streams"
                ],
            }
        )
        return [], limitations

    final_pair_keys = {
        (source, target)
        for engine, mode, source, target in grouped
        if (engine, mode) == final_stream
    }
    common_rows = []
    for source, target in sorted(final_pair_keys):
        target_draw = draws_by_no.get(target)
        source_draw = draws_by_no.get(source)
        if target_draw is None or source_draw is None:
            continue
        final_items = ranked_top5(
            grouped.get((final_stream[0], final_stream[1], source, target), [])
        )
        if final_items is None:
            continue
        underlying = {}
        complete = True
        for label, stream in underlying_streams.items():
            items = ranked_top5(
                grouped.get((stream[0], stream[1], source, target), [])
            )
            if items is None:
                complete = False
                break
            underlying[label] = normalized_entries(items, label)
        if not complete:
            continue
        locked_before_verification = all(
            item["verified_at"] is None
            or item["created_at"] is None
            or item["created_at"] <= item["verified_at"]
            for item in final_items
            for _ in (0,)
        ) and all(
            item["verified_at"] is None
            or item["created_at"] is None
            or item["created_at"] <= item["verified_at"]
            for items in underlying.values()
            for item in items
        )
        if not locked_before_verification:
            continue
        common_rows.append(
            {
                "source_draw_no": source,
                "target_draw_no": target,
                "source_date": source_draw["draw_date"],
                "source_month": source_draw["month"],
                "source_day_type": source_draw["day_type"],
                "target_date": target_draw["draw_date"],
                "target_month": target_draw["month"],
                "target_day_type": target_draw["day_type"],
                "target_weekday": target_draw["weekday"],
                "actuals": set(target_draw["winners"]),
                "actual_prize_count": len(set(target_draw["winners"])),
                "temporal": final_items,
                "underlying": underlying,
            }
        )
    return common_rows, limitations


def add_temporal_pre_miss_streak(rows: list[dict]) -> None:
    streak = 0
    for row in sorted(
        rows, key=lambda item: (item["source_draw_no"], item["target_draw_no"])
    ):
        row["temporal_pre_miss_streak"] = streak
        temporal_numbers = {item["number"] for item in row["temporal"]}
        hit = bool(temporal_numbers & row["actuals"])
        streak = 0 if hit else streak + 1


def has_00_structure(number: str) -> bool:
    return number[:2] == "00" or number[2:] == "00" or "00" in number


def omitted_candidates(row: dict) -> list[dict]:
    temporal_numbers = {item["number"] for item in row["temporal"]}
    by_number: dict[str, list[dict]] = defaultdict(list)
    for items in row["underlying"].values():
        for item in items:
            if item["number"] not in temporal_numbers:
                by_number[item["number"]].append(item)
    candidates = []
    for number, entries in by_number.items():
        entries = sorted(
            entries,
            key=lambda item: (
                -item["normalized_score"],
                item["rank"],
                ENGINE_PRIORITY[item["engine_label"]],
                item["number"],
            ),
        )
        representative = entries[0]
        candidates.append(
            {
                "number": number,
                "engine_label": representative["engine_label"],
                "engine_source": representative["engine_source"],
                "mode": representative["mode"],
                "rank": representative["rank"],
                "score": representative["score"],
                "normalized_score": representative["normalized_score"],
                "rank_strength": representative["rank_strength"],
                "all_source_labels": sorted(
                    {entry["engine_label"] for entry in entries}
                ),
                "all_source_entries": [
                    {
                        "engine_label": entry["engine_label"],
                        "engine_source": entry["engine_source"],
                        "mode": entry["mode"],
                        "rank": entry["rank"],
                        "score": entry["score"],
                        "normalized_score": entry["normalized_score"],
                    }
                    for entry in entries
                ],
            }
        )
    return candidates


def source_conditions(row: dict, candidate: dict) -> dict[str, bool]:
    streak = row["temporal_pre_miss_streak"]
    return {
        "special_day_type": row["target_day_type"] == "Special",
        "august_context": row["target_month"] == 8 or row["source_month"] == 8,
        "temporal_streak_21_50": 21 <= streak <= 50,
        "delta_or_wls": bool(
            set(candidate["all_source_labels"]) & {"Delta", "WLS"}
        ),
        "pair_00_or_double_zero": has_00_structure(candidate["number"]),
        "strong_engine_rank": min(
            entry["rank"] for entry in candidate["all_source_entries"]
        )
        <= 2,
    }


def policy_gate(policy: str, row: dict) -> bool:
    streak = row["temporal_pre_miss_streak"]
    if policy == "NO_RESCUE_BASELINE":
        return False
    if policy in {
        "ANY_UNDERLYING_SCORE_RESCUE",
        "DELTA_WLS_PRIORITY_RESCUE",
        "PAIR_00_CONTEXT_RESCUE",
        "COMBINED_SAFE_RESCUE_V1",
        "COMBINED_STRICT_RESCUE_V2",
    }:
        return True
    if policy == "SPECIAL_DAYTYPE_RESCUE":
        return row["target_day_type"] == "Special"
    if policy == "AUGUST_CONTEXT_RESCUE":
        return row["target_month"] == 8 or row["source_month"] == 8
    if policy == "TEMPORAL_RISK_STREAK_RESCUE_21_30":
        return 21 <= streak <= 30
    if policy == "TEMPORAL_RISK_STREAK_RESCUE_21_50":
        return 21 <= streak <= 50
    raise ValueError(f"Unknown policy: {policy}")


def qualifying_candidates(policy: str, row: dict) -> list[dict]:
    candidates = omitted_candidates(row)
    if policy == "PAIR_00_CONTEXT_RESCUE":
        candidates = [
            candidate
            for candidate in candidates
            if source_conditions(row, candidate)["pair_00_or_double_zero"]
        ]
    elif policy == "COMBINED_SAFE_RESCUE_V1":
        candidates = [
            candidate
            for candidate in candidates
            if sum(source_conditions(row, candidate).values()) >= 2
        ]
    elif policy == "COMBINED_STRICT_RESCUE_V2":
        candidates = [
            candidate
            for candidate in candidates
            if sum(source_conditions(row, candidate).values()) >= 3
        ]
    return candidates


def candidate_sort_key(policy: str, candidate: dict) -> tuple:
    source_labels = set(candidate["all_source_labels"])
    delta_wls_priority = (
        0 if source_labels & {"Delta", "WLS"} else 1
    )
    preferred_label = min(
        candidate["all_source_labels"], key=lambda label: ENGINE_PRIORITY[label]
    )
    if policy == "DELTA_WLS_PRIORITY_RESCUE":
        return (
            delta_wls_priority,
            -candidate["normalized_score"],
            candidate["rank"],
            ENGINE_PRIORITY[preferred_label],
            candidate["number"],
        )
    return (
        -candidate["normalized_score"],
        candidate["rank"],
        ENGINE_PRIORITY[preferred_label],
        candidate["number"],
    )


def lock_simulated_top5(policy: str, row: dict) -> dict:
    baseline = [
        item["number"]
        for item in sorted(row["temporal"], key=lambda item: item["rank"])
    ]
    gate = policy_gate(policy, row)
    if not gate:
        return {
            "policy": policy,
            "gate_passed": False,
            "rescue_used": False,
            "rescue": None,
            "removed_number": None,
            "locked_top5": baseline,
            "temporal_candidates_kept": 5,
        }
    candidates = qualifying_candidates(policy, row)
    if not candidates:
        return {
            "policy": policy,
            "gate_passed": True,
            "rescue_used": False,
            "rescue": None,
            "removed_number": None,
            "locked_top5": baseline,
            "temporal_candidates_kept": 5,
        }
    selected = sorted(
        candidates, key=lambda candidate: candidate_sort_key(policy, candidate)
    )[0]
    removed = baseline[-1]
    locked = baseline[:4] + [selected["number"]]
    return {
        "policy": policy,
        "gate_passed": True,
        "rescue_used": True,
        "rescue": selected,
        "removed_number": removed,
        "locked_top5": locked,
        "temporal_candidates_kept": 4,
    }


def verify_locked_simulation(row: dict, locked: dict) -> dict:
    actuals = row["actuals"]
    baseline_top5 = [
        item["number"]
        for item in sorted(row["temporal"], key=lambda item: item["rank"])
    ]
    baseline_raw_hits = len(set(baseline_top5) & actuals)
    simulated_raw_hits = len(set(locked["locked_top5"]) & actuals)
    rescue_number = locked["rescue"]["number"] if locked["rescue"] else None
    rescue_hit = bool(rescue_number and rescue_number in actuals)
    removed_hit = bool(
        locked["removed_number"] and locked["removed_number"] in actuals
    )
    harmful = removed_hit and not rescue_hit
    beneficial = rescue_hit and not removed_hit
    swap_both_hit = rescue_hit and removed_hit
    return {
        "row_type": "policy_by_draw",
        "policy": locked["policy"],
        "source_draw_no": row["source_draw_no"],
        "target_draw_no": row["target_draw_no"],
        "source_date": row["source_date"],
        "target_date": row["target_date"],
        "target_month": row["target_month"],
        "target_day_type": row["target_day_type"],
        "target_weekday": row["target_weekday"],
        "temporal_pre_miss_streak": row["temporal_pre_miss_streak"],
        "baseline_top5": baseline_top5,
        "simulated_top5": locked["locked_top5"],
        "baseline_hit": baseline_raw_hits > 0,
        "baseline_raw_hits": baseline_raw_hits,
        "simulated_hit": simulated_raw_hits > 0,
        "simulated_raw_hits": simulated_raw_hits,
        "gate_passed": locked["gate_passed"],
        "rescue_used": locked["rescue_used"],
        "rescue_number": rescue_number,
        "rescue_source_engine": (
            locked["rescue"]["engine_label"] if locked["rescue"] else None
        ),
        "rescue_all_source_engines": (
            locked["rescue"]["all_source_labels"] if locked["rescue"] else []
        ),
        "rescue_source_rank": (
            locked["rescue"]["rank"] if locked["rescue"] else None
        ),
        "rescue_source_score": (
            locked["rescue"]["score"] if locked["rescue"] else None
        ),
        "rescue_hit": rescue_hit,
        "removed_number": locked["removed_number"],
        "removed_temporal_rank": 5 if locked["removed_number"] else None,
        "removed_hit": removed_hit,
        "harmful_replacement": harmful,
        "beneficial_replacement": beneficial,
        "swap_both_hit": swap_both_hit,
        "net_raw_hit_gain": simulated_raw_hits - baseline_raw_hits,
        "temporal_candidates_kept": locked["temporal_candidates_kept"],
        "actual_prize_count": row["actual_prize_count"],
    }


def simulate_policies(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    results = []
    rescue_rows = []
    for row in rows:
        for policy in POLICIES:
            locked = lock_simulated_top5(policy, row)
            verified = verify_locked_simulation(row, locked)
            results.append(verified)
            if verified["rescue_used"]:
                rescue_rows.append(
                    {
                        "row_type": "rescued_candidate",
                        **{
                            key: value
                            for key, value in verified.items()
                            if key != "row_type"
                        },
                    }
                )
    return results, rescue_rows


def random_expected_rate(items: list[dict]) -> float:
    if not items:
        return 0.0
    return statistics.mean(
        1 - ((NUMBER_SPACE - item["actual_prize_count"]) / NUMBER_SPACE) ** 5
        for item in items
    )


def metric_row(policy: str, items: list[dict]) -> dict:
    rows = len(items)
    hit_draws = sum(item["simulated_hit"] for item in items)
    raw_hits = sum(item["simulated_raw_hits"] for item in items)
    baseline_hit_draws = sum(item["baseline_hit"] for item in items)
    baseline_raw_hits = sum(item["baseline_raw_hits"] for item in items)
    used = sum(item["rescue_used"] for item in items)
    rescue_hits = sum(item["rescue_hit"] for item in items)
    baseline_hits_lost = sum(item["removed_hit"] for item in items)
    harmful = sum(item["harmful_replacement"] for item in items)
    beneficial = sum(item["beneficial_replacement"] for item in items)
    random_rate = random_expected_rate(items)
    hit_rate = hit_draws / rows if rows else 0.0
    return {
        "policy": policy,
        "rows_evaluated": rows,
        "baseline_hit_draws": baseline_hit_draws,
        "hit_draws": hit_draws,
        "delta_hit_draws_vs_temporal": hit_draws - baseline_hit_draws,
        "baseline_raw_hits": baseline_raw_hits,
        "raw_hits": raw_hits,
        "delta_raw_hits_vs_temporal": raw_hits - baseline_raw_hits,
        "hit_rate": hit_rate,
        "rescues_attempted": sum(item["gate_passed"] for item in items),
        "rescues_used": used,
        "rescue_candidates_that_hit": rescue_hits,
        "baseline_hits_lost_due_to_replacement": baseline_hits_lost,
        "harmful_replacements": harmful,
        "beneficial_replacements": beneficial,
        "net_hit_gain": raw_hits - baseline_raw_hits,
        "replacement_harm_rate": harmful / used if used else 0.0,
        "replacement_benefit_rate": beneficial / used if used else 0.0,
        "average_temporal_candidates_kept": (
            statistics.mean(item["temporal_candidates_kept"] for item in items)
            if items
            else 0.0
        ),
        "random_expected_top5_hit_rate": random_rate,
        "enrichment_vs_random": hit_rate / random_rate if random_rate else None,
    }


def policy_matrix(results: list[dict]) -> list[dict]:
    by_policy: dict[str, list[dict]] = defaultdict(list)
    for result in results:
        by_policy[result["policy"]].append(result)
    return [metric_row(policy, by_policy[policy]) for policy in POLICIES]


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
        window_items = [
            item
            for item in results
            if (item["source_draw_no"], item["target_draw_no"]) in selected
        ]
        output[f"RECENT_{window}"] = policy_matrix(window_items)
    return output


def context_match(name: str, item: dict) -> bool:
    streak = item["temporal_pre_miss_streak"]
    if name == "FULL_RANGE":
        return True
    if name == "SPECIAL_DAYTYPE":
        return item["target_day_type"] == "Special"
    if name == "SATURDAY_DAYTYPE":
        return item["target_day_type"] == "Saturday"
    if name == "AUGUST":
        return item["target_month"] == 8
    if name == "TEMPORAL_STREAK_21_30":
        return 21 <= streak <= 30
    if name == "TEMPORAL_STREAK_31_50":
        return 31 <= streak <= 50
    if name == "TEMPORAL_STREAK_51_PLUS":
        return streak >= 51
    raise ValueError(name)


def context_breakdowns(results: list[dict]) -> dict[str, list[dict]]:
    output = {}
    for context in CONTEXTS:
        output[context] = policy_matrix(
            [item for item in results if context_match(context, item)]
        )
    by_source = {}
    for label in ("Delta", "WLS", "Linear"):
        by_source[label] = policy_matrix(
            [
                item
                for item in results
                if item["rescue_source_engine"] == label
            ]
        )
    output["RESCUED_CANDIDATE_SOURCE"] = by_source
    return output


def baseline_metrics(rows: list[dict]) -> dict:
    temporal_hit_draws = 0
    temporal_raw_hits = 0
    omitted = Counter()
    omitted_unique = set()
    for row in rows:
        actuals = row["actuals"]
        temporal_set = {item["number"] for item in row["temporal"]}
        baseline_hits = temporal_set & actuals
        temporal_hit_draws += bool(baseline_hits)
        temporal_raw_hits += len(baseline_hits)
        for label, items in row["underlying"].items():
            for item in items:
                if item["number"] in actuals and item["number"] not in temporal_set:
                    omitted[label] += 1
                    omitted_unique.add(
                        (
                            row["source_draw_no"],
                            row["target_draw_no"],
                            item["number"],
                        )
                    )
    random_rate = statistics.mean(
        1 - ((NUMBER_SPACE - row["actual_prize_count"]) / NUMBER_SPACE) ** 5
        for row in rows
    )
    return {
        "rows": len(rows),
        "temporal_hit_draws": temporal_hit_draws,
        "temporal_raw_hits": temporal_raw_hits,
        "temporal_hit_rate": temporal_hit_draws / len(rows) if rows else 0.0,
        "delta_raw_omitted_hit_opportunities": omitted["Delta"],
        "wls_raw_omitted_hit_opportunities": omitted["WLS"],
        "linear_raw_omitted_hit_opportunities": omitted["Linear"],
        "total_raw_omitted_hit_opportunities": sum(omitted.values()),
        "unique_omitted_target_number_opportunities": len(omitted_unique),
        "random_expected_top5_hit_rate": random_rate,
        "temporal_enrichment_vs_random": (
            (temporal_hit_draws / len(rows)) / random_rate
            if rows and random_rate
            else None
        ),
    }


def build_step154_events(
    grouped: dict[tuple[str, str, int, int], list[dict]],
    draws_by_no: dict[int, dict],
    discovery: dict,
) -> list[dict]:
    final_stream = tuple(discovery["resolved_final_stream"] or ())
    if len(final_stream) != 2:
        return []
    final_pairs: dict[tuple[int, int], list[dict]] = {}
    for (engine, mode, source, target), items in grouped.items():
        if (engine, mode) != final_stream:
            continue
        top = ranked_top5(items)
        if top:
            final_pairs[(source, target)] = top
    events = []
    for (source, target), final_items in sorted(final_pairs.items()):
        draw = draws_by_no.get(target)
        if draw is None:
            continue
        actuals = set(draw["winners"])
        final_set = {item["number"] for item in final_items}
        for (engine, mode, group_source, group_target), items in grouped.items():
            if (group_source, group_target) != (source, target):
                continue
            label = next(
                (
                    candidate_label
                    for candidate_label in ("Delta", "WLS", "Linear")
                    if candidate_label.lower() in engine.lower()
                    or engine == UNDERLYING_STREAMS[candidate_label][0]
                ),
                None,
            )
            if label is None:
                continue
            top = ranked_top5(items)
            if top is None:
                continue
            for item in top:
                if item["number"] in actuals and item["number"] not in final_set:
                    events.append(
                        {
                            "source_draw_no": source,
                            "target_draw_no": target,
                            "number": item["number"],
                            "source_engine_label": label,
                            "source_engine": engine,
                            "source_mode": mode,
                            "source_rank": item["rank"],
                            "target_month": draw["month"],
                            "target_day_type": draw["day_type"],
                        }
                    )
    return events


def false_negative_recovery(
    step154_events: list[dict],
    results: list[dict],
    common_rows: list[dict],
) -> tuple[list[dict], list[dict]]:
    by_policy_pair = {
        (
            item["policy"],
            item["source_draw_no"],
            item["target_draw_no"],
        ): item
        for item in results
    }
    pre_streak_by_pair = {
        (item["source_draw_no"], item["target_draw_no"]): item[
            "temporal_pre_miss_streak"
        ]
        for item in results
        if item["policy"] == "NO_RESCUE_BASELINE"
    }
    eligible_numbers_by_pair = {
        (row["source_draw_no"], row["target_draw_no"]): {
            candidate["number"] for candidate in omitted_candidates(row)
        }
        for row in common_rows
    }
    matrix = []
    rows = []
    for policy in POLICIES:
        recovered = []
        missed = []
        for event in step154_events:
            result = by_policy_pair.get(
                (policy, event["source_draw_no"], event["target_draw_no"])
            )
            was_recovered = bool(
                result
                and result["rescue_used"]
                and result["rescue_number"] == event["number"]
            )
            enriched = {
                **event,
                "policy": policy,
                "recovered": was_recovered,
                "eligible_via_current_policy_pool": event["number"]
                in eligible_numbers_by_pair.get(
                    (event["source_draw_no"], event["target_draw_no"]), set()
                ),
                "temporal_pre_miss_streak": pre_streak_by_pair.get(
                    (event["source_draw_no"], event["target_draw_no"])
                ),
            }
            rows.append({"row_type": "false_negative_recovery", **enriched})
            (recovered if was_recovered else missed).append(enriched)
        policy_results = [
            item for item in results if item["policy"] == policy
        ]
        matrix.append(
            {
                "policy": policy,
                "step154_dropped_events": len(step154_events),
                "eligible_via_current_policy_pool": sum(
                    item["eligible_via_current_policy_pool"]
                    for item in recovered + missed
                ),
                "outside_current_policy_pool": sum(
                    not item["eligible_via_current_policy_pool"]
                    for item in recovered + missed
                ),
                "recovered_events": len(recovered),
                "still_missed_events": len(missed),
                "baseline_temporal_hits_lost": sum(
                    item["removed_hit"] for item in policy_results
                ),
                "net_recovery_after_lost_hits": len(recovered)
                - sum(item["removed_hit"] for item in policy_results),
                "recovery_by_source_engine": dict(
                    Counter(item["source_engine_label"] for item in recovered)
                ),
                "recovery_by_day_type": dict(
                    Counter(item["target_day_type"] for item in recovered)
                ),
                "recovery_by_month": dict(
                    Counter(str(item["target_month"]) for item in recovered)
                ),
                "recovery_by_streak_bucket": dict(
                    Counter(
                        streak_bucket(item["temporal_pre_miss_streak"])
                        for item in recovered
                    )
                ),
                "recovery_by_digit_structure": dict(
                    Counter(
                        "00_PAIR_OR_DOUBLE_ZERO"
                        if has_00_structure(item["number"])
                        else "OTHER"
                        for item in recovered
                    )
                ),
            }
        )
    return matrix, rows


def streak_bucket(value: int | None) -> str:
    if value is None:
        return "unavailable"
    if value <= 20:
        return "0-20"
    if value <= 30:
        return "21-30"
    if value <= 50:
        return "31-50"
    return "51+"


def harm_matrix(results: list[dict]) -> tuple[list[dict], list[dict]]:
    matrix = []
    harm_rows = []
    for policy in POLICIES:
        items = [item for item in results if item["policy"] == policy]
        harmful = [item for item in items if item["harmful_replacement"]]
        removed_hits = [item for item in items if item["removed_hit"]]
        matrix.append(
            {
                "policy": policy,
                "baseline_hits_lost": len(removed_hits),
                "harmful_replacements": len(harmful),
                "swap_both_hit": sum(item["swap_both_hit"] for item in items),
                "harm_by_candidate_source": dict(
                    Counter(item["rescue_source_engine"] for item in harmful)
                ),
                "harm_by_day_type": dict(
                    Counter(item["target_day_type"] for item in harmful)
                ),
                "harm_by_month": dict(
                    Counter(str(item["target_month"]) for item in harmful)
                ),
            }
        )
        for item in harmful:
            harm_rows.append(
                {
                    "row_type": "harm_event",
                    **{
                        key: value
                        for key, value in item.items()
                        if key != "row_type"
                    },
                }
            )
    return matrix, harm_rows


def classify_policies(
    global_matrix: list[dict],
    recent: dict[str, list[dict]],
    recovery: list[dict],
) -> dict:
    recent_lookup = {
        window: {row["policy"]: row for row in rows}
        for window, rows in recent.items()
    }
    recovery_lookup = {row["policy"]: row for row in recovery}
    output = {}
    for row in global_matrix:
        policy = row["policy"]
        if policy == "NO_RESCUE_BASELINE":
            output[policy] = {
                "decision": "REJECT_NO_EDGE",
                "reason": "Reference baseline; no rescue rule is applied.",
            }
            continue
        recent90 = recent_lookup["RECENT_90"][policy]
        recent47 = recent_lookup["RECENT_47"][policy]
        recovered = recovery_lookup[policy]["recovered_events"]
        if (
            row["baseline_hits_lost_due_to_replacement"] > 0
            or recent90["delta_hit_draws_vs_temporal"] < 0
            or recent47["delta_hit_draws_vs_temporal"] < 0
        ):
            decision = "REJECT_HARMFUL"
            reason = (
                "The policy removes at least one baseline hit or reduces a required "
                "recent-window hit count."
            )
        elif (
            row["delta_hit_draws_vs_temporal"] > 0
            and recovered >= 5
            and row["rescues_used"] >= 30
        ):
            decision = "CANDIDATE_FOR_SHADOW_V3"
            reason = (
                "Positive full-range result, no recent regression, no baseline hit "
                "removed, and sufficient rescue volume."
            )
        elif row["delta_hit_draws_vs_temporal"] > 0 or recovered > 0:
            decision = "WATCH_LOW_SAMPLE"
            reason = (
                "Some recovery is present, but sample volume or net evidence is "
                "insufficient for shadow candidacy."
            )
        else:
            decision = "REJECT_NO_EDGE"
            reason = "No measurable anti-drop benefit."
        output[policy] = {"decision": decision, "reason": reason}
    return output


def final_diagnosis(
    global_matrix: list[dict],
    decisions: dict,
) -> dict:
    eligible = [
        row
        for row in global_matrix
        if decisions[row["policy"]]["decision"] == "CANDIDATE_FOR_SHADOW_V3"
    ]
    if eligible:
        best = max(
            eligible,
            key=lambda row: (
                row["delta_hit_draws_vs_temporal"],
                row["net_hit_gain"],
                -row["rescues_used"],
            ),
        )
        next_step = (
            f"shadow-test {best['policy']} without changing production or Current mode"
        )
    else:
        non_baseline = [
            row for row in global_matrix if row["policy"] != "NO_RESCUE_BASELINE"
        ]
        best = max(
            non_baseline,
            key=lambda row: (
                row["delta_hit_draws_vs_temporal"],
                row["net_hit_gain"],
                -row["baseline_hits_lost_due_to_replacement"],
            ),
        )
        next_step = (
            "reject direct Top5 replacement; evaluate a non-destructive shadow Top6 "
            "or abstention design using the same source-side signals"
        )
    return {
        "best_policy_by_net_result": best["policy"],
        "best_policy_metrics": best,
        "best_policy_decision": decisions[best["policy"]],
        "policy_decisions": decisions,
        "production_switch_recommended_now": False,
        "next_step": next_step,
        "causality_warning": (
            "This is a retrospective source-side what-if audit. Target winners were "
            "used only after each simulated Top5 was locked."
        ),
    }


def pct(value: float | None) -> str:
    return "NULL" if value is None else f"{value * 100:.4f}%"


def format_policy_matrix(rows: list[dict]) -> list[str]:
    lines = [
        "Policy                                  Rows Hits Raw HitRate DeltaHit DeltaRaw Attempt Used RescueHit Lost Net HarmRate BenefitRate Kept Random Enrich Decision"
    ]
    for row in rows:
        enrichment = (
            "NULL"
            if row["enrichment_vs_random"] is None
            else f"{row['enrichment_vs_random']:.4f}"
        )
        lines.append(
            f"{row['policy']:<39} {row['rows_evaluated']:>4} "
            f"{row['hit_draws']:>4} {row['raw_hits']:>3} "
            f"{pct(row['hit_rate']):>9} "
            f"{row['delta_hit_draws_vs_temporal']:>8} "
            f"{row['delta_raw_hits_vs_temporal']:>8} "
            f"{row['rescues_attempted']:>7} "
            f"{row['rescues_used']:>4} "
            f"{row['rescue_candidates_that_hit']:>9} "
            f"{row['baseline_hits_lost_due_to_replacement']:>4} "
            f"{row['net_hit_gain']:>3} "
            f"{pct(row['replacement_harm_rate']):>8} "
            f"{pct(row['replacement_benefit_rate']):>11} "
            f"{row['average_temporal_candidates_kept']:>4.2f} "
            f"{pct(row['random_expected_top5_hit_rate']):>8} "
            f"{enrichment:>6} "
            f"{row.get('decision', 'UNCLASSIFIED')}"
        )
    return lines


def format_recent_matrix(recent: dict[str, list[dict]]) -> list[str]:
    lines = [
        "Window     Policy                                  Hits DeltaHit Raw DeltaRaw Used Lost Harm"
    ]
    for window in ("RECENT_365", "RECENT_90", "RECENT_47"):
        for row in recent[window]:
            lines.append(
                f"{window:<10} {row['policy']:<39} "
                f"{row['hit_draws']:>4} "
                f"{row['delta_hit_draws_vs_temporal']:>8} "
                f"{row['raw_hits']:>3} "
                f"{row['delta_raw_hits_vs_temporal']:>8} "
                f"{row['rescues_used']:>4} "
                f"{row['baseline_hits_lost_due_to_replacement']:>4} "
                f"{row['harmful_replacements']:>4}"
            )
    return lines


def format_context_matrix(contexts: dict[str, list[dict]]) -> list[str]:
    lines = [
        "Context                    Policy                                  Rows Hits DeltaHit Raw DeltaRaw Used Lost Harm"
    ]
    for context in CONTEXTS[1:]:
        for row in contexts[context]:
            lines.append(
                f"{context:<26} {row['policy']:<39} "
                f"{row['rows_evaluated']:>4} {row['hit_draws']:>4} "
                f"{row['delta_hit_draws_vs_temporal']:>8} "
                f"{row['raw_hits']:>3} "
                f"{row['delta_raw_hits_vs_temporal']:>8} "
                f"{row['rescues_used']:>4} "
                f"{row['baseline_hits_lost_due_to_replacement']:>4} "
                f"{row['harmful_replacements']:>4}"
            )
    return lines


def build_report(
    discovery: dict,
    baseline: dict,
    global_matrix: list[dict],
    recent: dict,
    contexts: dict,
    recovery: list[dict],
    harm: list[dict],
    diagnosis: dict,
) -> str:
    width = 176
    lines = [
        "=" * width,
        "STEP 155 — RENEWAL-AWARE ANTI-DROP META LAYER AUDIT — REPORT ONLY",
        "=" * width,
        "ProductionMathChanged: NO",
        "APIChanged: NO",
        "FrontendChanged: NO",
        "SQLSchemaChanged: NO",
        "DBWritePerformed: NO",
        "ExistingTablesOnly: YES",
        "DeepCandidateLedgerUsed: NO",
        "",
        "SOURCE DISCOVERY",
        "-" * width,
        f"DrawRange: {discovery['draw_range']}",
        f"DrawRows: {discovery['draw_rows']}",
        f"PredictionLedgerRows: {discovery['predictionledger_rows']}",
        f"ResolvedFinalStream: {discovery['resolved_final_stream']}",
        f"ResolvedUnderlyingStreams: {discovery['resolved_underlying_streams']}",
        f"LedgerVolumeByEngineMode: {discovery['ledger_volume_by_engine_mode']}",
        f"RankColumn: {discovery['rank_column']}",
        f"ScoreColumn: {discovery['score_column']}",
        f"HitCountColumn: {discovery['hitcount_column']}",
        f"VerificationStatusColumn: {discovery['verification_status_column']}",
        "",
        "BASELINE TEMPORAL METRICS",
        "-" * width,
        f"{baseline}",
        "",
        "POLICY COMPARISON GLOBAL MATRIX",
        "-" * width,
        *format_policy_matrix(global_matrix),
        "",
        "RECENT 365 / 90 / 47 MATRIX",
        "-" * width,
        *format_recent_matrix(recent),
        "",
        "CONTEXT BREAKDOWN MATRIX",
        "-" * width,
        *format_context_matrix(contexts),
        "",
        "FALSE NEGATIVE RECOVERY MATRIX",
        "-" * width,
    ]
    for row in recovery:
        lines.append(
            f"{row['policy']}: eligible={row['eligible_via_current_policy_pool']} "
            f"outside_pool={row['outside_current_policy_pool']} "
            f"recovered={row['recovered_events']} "
            f"still_missed={row['still_missed_events']} "
            f"baseline_hits_lost={row['baseline_temporal_hits_lost']} "
            f"net_recovery={row['net_recovery_after_lost_hits']} "
            f"by_source={row['recovery_by_source_engine']} "
            f"by_daytype={row['recovery_by_day_type']} "
            f"by_month={row['recovery_by_month']} "
            f"by_streak={row['recovery_by_streak_bucket']} "
            f"by_structure={row['recovery_by_digit_structure']}"
        )
    lines.extend(
        (
            "",
            "HARM MATRIX",
            "-" * width,
        )
    )
    for row in harm:
        lines.append(
            f"{row['policy']}: baseline_hits_lost={row['baseline_hits_lost']} "
            f"harmful_replacements={row['harmful_replacements']} "
            f"swap_both_hit={row['swap_both_hit']} "
            f"harm_by_source={row['harm_by_candidate_source']} "
            f"harm_by_daytype={row['harm_by_day_type']} "
            f"harm_by_month={row['harm_by_month']}"
        )
    lines.extend(
        (
            "",
            "BEST POLICY DISCUSSION",
            "-" * width,
            f"BestPolicyByNetResult: {diagnosis['best_policy_by_net_result']}",
            f"BestPolicyMetrics: {diagnosis['best_policy_metrics']}",
            f"BestPolicyDecision: {diagnosis['best_policy_decision']}",
            f"AllPolicyDecisions: {diagnosis['policy_decisions']}",
            f"CausalityWarning: {diagnosis['causality_warning']}",
            "",
            "FINAL RECOMMENDATION",
            "-" * width,
            "ProductionSwitchRecommendedNow: NO",
            f"NextStep: {diagnosis['next_step']}",
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
    grouped = group_ledger(ledger)
    common_rows, limitations = build_common_rows(grouped, draws_by_no, discovery)
    limitations.extend(
        (
            {
                "row_type": "data_limitation",
                "limitation": (
                    "Raw scores use different scales across engines. Candidate "
                    "selection therefore uses deterministic within-stream score "
                    "normalization, rank, engine priority, and number tie-breaking."
                ),
            },
            {
                "row_type": "data_limitation",
                "limitation": (
                    "The 52 Step 154 dropped-event rows contain 50 unique current "
                    "Grand Loop target-number opportunities; repeated source/mode "
                    "attribution is retained in the recovery matrix."
                ),
            },
            {
                "row_type": "data_limitation",
                "limitation": (
                    "PredictionLedger does not persist an exact removal rule. This "
                    "audit tests prescribed source-side policies and does not infer "
                    "the historical removal cause."
                ),
            },
        )
    )
    add_temporal_pre_miss_streak(common_rows)
    baseline = baseline_metrics(common_rows)
    results, rescue_rows = simulate_policies(common_rows)
    global_matrix = policy_matrix(results)
    recent = recent_matrix(results)
    contexts = context_breakdowns(results)
    step154_events = build_step154_events(grouped, draws_by_no, discovery)
    recovery, recovery_rows = false_negative_recovery(
        step154_events, results, common_rows
    )
    harm, harm_rows = harm_matrix(results)
    decisions = classify_policies(global_matrix, recent, recovery)
    for row in global_matrix:
        row["decision"] = decisions[row["policy"]]["decision"]
    diagnosis = final_diagnosis(global_matrix, decisions)

    matrices = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "audit": "STEP 155",
            "read_only": True,
            "existing_tables_only": True,
            "deep_candidate_ledger_used": False,
            "target_winners_used_only_after_simulated_top5_lock": True,
            "maximum_replacements_per_draw": 1,
            "temporal_rank_1_replaced": False,
        },
        "discovery": discovery,
        "baseline_metrics": baseline,
        "policy_global_matrix": global_matrix,
        "policy_recent_matrix": recent,
        "context_breakdowns": contexts,
        "false_negative_recovery": recovery,
        "harm_matrix": harm,
        "final_diagnosis": diagnosis,
    }
    report = build_report(
        discovery,
        baseline,
        global_matrix,
        recent,
        contexts,
        recovery,
        harm,
        diagnosis,
    )
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    MATRICES_PATH.write_text(
        json.dumps(matrices, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with ROWS_PATH.open("w", encoding="utf-8") as handle:
        for row in results + rescue_rows + recovery_rows + harm_rows + limitations:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    print(report)


if __name__ == "__main__":
    main()
