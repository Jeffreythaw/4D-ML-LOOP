from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import pyodbc
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
TMP_DIR = PROJECT_ROOT / "reports" / "tmp_step152"
REPORT_PATH = PROJECT_ROOT / "reports" / "step_152_global_ml_diagnostics_report.txt"
MATRICES_PATH = PROJECT_ROOT / "reports" / "step_152_global_ml_diagnostics_matrices.json"
ROWS_PATH = PROJECT_ROOT / "reports" / "step_152_global_ml_diagnostics_rows.jsonl"
DISCOVERY_PATH = TMP_DIR / "step152_discovery.json"
PLAN_PATH = TMP_DIR / "step152_worker_plan.json"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))
load_dotenv(BACKEND_ROOT / ".env")

from app.core.config import get_settings


TOP_CUTOFFS = (5, 10, 15, 20)
NUMBER_SPACE = 10_000
TARGET_ENGINES = {
    "LINEAR": ("E1_LINEAR", "E1_CROSS_PAIR_LINEAR"),
    "WLS": ("E1_WLS", "E1_WLS_DECAY_0.98"),
    "DELTA": ("E1_DELTA", "E1_DELTA_ROTATION_LSTS"),
    "TEMPORAL": ("E1_TEMPORAL_CONTEXT_MATCH",),
}
FINAL_ENGINE = "E1_TEMPORAL_CONTEXT_MATCH"
MODE_PRIORITY = {
    "Current": 0,
    "Temporal_Global_Loop": 1,
    "Historical": 2,
    "Engine_Grand_Loop": 3,
    "Grand_Loop": 4,
    "Weighted_Grand_Loop": 5,
}


def get_conn():
    return pyodbc.connect(get_settings().sql_connection_string(), timeout=120)


def z4(value: str | int) -> str:
    return str(value).strip().zfill(4)


def parse_numbers(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(z4(part) for part in str(value).replace(" ", "").split(",") if part)


def digit_sum(number: str) -> int:
    return sum(int(ch) for ch in z4(number))


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


def mirror_signature(number: str) -> str:
    return "".join(str(int(ch) % 5) for ch in z4(number))


def box_signature(number: str) -> str:
    return "".join(sorted(z4(number)))


def risk_bucket(streak: int | None) -> str:
    if streak is None:
        return "UNAVAILABLE"
    if streak <= 5:
        return "LOW_RISK"
    if streak <= 15:
        return "MEDIUM_RISK"
    if streak <= 40:
        return "HIGH_RISK"
    return "EXTREME_RISK"


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


def distinct_values(cursor, column: str) -> list[str]:
    allowed = {"EngineSource", "Mode", "VerificationStatus"}
    if column not in allowed:
        raise ValueError(f"Unsupported discovery column: {column}")
    query = {
        "EngineSource": "SELECT DISTINCT EngineSource AS Value FROM dbo.PredictionLedger ORDER BY EngineSource;",
        "Mode": "SELECT DISTINCT Mode AS Value FROM dbo.PredictionLedger ORDER BY Mode;",
        "VerificationStatus": "SELECT DISTINCT VerificationStatus AS Value FROM dbo.PredictionLedger ORDER BY VerificationStatus;",
    }[column]
    return [
        str(row.Value)
        for row in cursor.execute(query).fetchall()
        if row.Value is not None
    ]


def repo_search() -> dict[str, dict]:
    terms = (
        "E1_LINEAR",
        "E1_WLS",
        "E1_DELTA",
        "E1_TEMPORAL_CONTEXT_MATCH",
        "Top20",
        "Top100",
        "candidate",
        "rank",
    )
    output = {}
    for term in terms:
        process = subprocess.run(
            [
                "rg",
                "-n",
                "--glob",
                "*.py",
                "--glob",
                "*.sql",
                "--glob",
                "!**/__pycache__/**",
                term,
                str(PROJECT_ROOT),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        matches = [line for line in process.stdout.splitlines() if line][:40]
        output[term] = {
            "match_count_capped": len(matches),
            "matches": matches,
        }
    return output


def discover() -> dict:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    with get_conn() as connection:
        cursor = connection.cursor()
        draw_columns = table_columns(cursor, "dbo.DrawHistory")
        ledger_columns = table_columns(cursor, "dbo.PredictionLedger")
        engines = distinct_values(cursor, "EngineSource")
        modes = distinct_values(cursor, "Mode")
        statuses = distinct_values(cursor, "VerificationStatus")
        depth_rows = cursor.execute(
            """
            SELECT
                EngineSource,
                Mode,
                MIN(RankNo) AS MinRank,
                MAX(RankNo) AS MaxRank,
                COUNT(*) AS LedgerRows,
                COUNT(DISTINCT SourceDrawNo) AS SourceDraws
            FROM dbo.PredictionLedger
            GROUP BY EngineSource, Mode
            ORDER BY EngineSource, Mode;
            """
        ).fetchall()
        depth_matrix = [
            {
                "engine_source": str(row.EngineSource),
                "mode": str(row.Mode),
                "min_rank": int(row.MinRank),
                "max_rank": int(row.MaxRank),
                "ledger_rows": int(row.LedgerRows),
                "source_draws": int(row.SourceDraws),
                "data_source_type": "PERSISTED_LEDGER",
            }
            for row in depth_rows
        ]

    score_columns = [
        item["name"]
        for item in ledger_columns
        if any(
            token in item["name"].lower()
            for token in ("rank", "position", "confidence", "score")
        )
    ]
    engine_availability = {}
    for label, aliases in TARGET_ENGINES.items():
        present = [engine for engine in engines if engine in aliases]
        engine_availability[label] = {
            "aliases_checked": list(aliases),
            "persisted_engine_sources": present,
            "status": "PERSISTED_LEDGER" if present else "UNAVAILABLE",
        }
    max_depth = max((row["max_rank"] for row in depth_matrix), default=0)
    discovery = {
        "generated_at": datetime.now().isoformat(),
        "drawhistory_columns": draw_columns,
        "predictionledger_columns": ledger_columns,
        "distinct_engine_sources": engines,
        "distinct_modes": modes,
        "distinct_verification_statuses": statuses,
        "rank_score_columns": score_columns,
        "candidate_depth_matrix": depth_matrix,
        "maximum_persisted_rank": max_depth,
        "top10_persisted": max_depth >= 10,
        "top15_persisted": max_depth >= 15,
        "top20_persisted": max_depth >= 20,
        "depth_reconstruction": {
            "status": "UNAVAILABLE",
            "reason": "No globally persisted deterministic deeper rank snapshot was found; Step 150B reconstruction is separate experimental output and is not treated as an engine ledger.",
        },
        "engine_availability": engine_availability,
        "repo_search": repo_search(),
        "data_limitations": [
            "Top10/Top15/Top20 are unavailable for engine/mode groups whose persisted MaxRank is below the cutoff.",
            "Independent engine aliases are mapped only when matching persisted EngineSource values exist.",
            "Step 150B reconstructed Top100 is not mixed with persisted engine rankings.",
        ],
    }
    DISCOVERY_PATH.write_text(
        json.dumps(discovery, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(discovery, indent=2, sort_keys=True))
    print(f"DISCOVERY_WRITTEN: {DISCOVERY_PATH}")
    return discovery


def eligible_sources() -> list[int]:
    with get_conn() as connection:
        rows = connection.cursor().execute(
            """
            SELECT DISTINCT p.SourceDrawNo
            FROM dbo.PredictionLedger p
            JOIN dbo.DrawHistory d ON d.DrawNo = p.TargetDrawNo
            WHERE d.WinningNumbers IS NOT NULL
              AND p.RankNo >= 1
            ORDER BY p.SourceDrawNo;
            """
        ).fetchall()
    return [int(row.SourceDrawNo) for row in rows]


def plan_workers() -> dict:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    sources = eligible_sources()
    if not sources:
        raise RuntimeError("No eligible source draws found")
    chunk_size = math.ceil(len(sources) / 4)
    workers = []
    for index in range(4):
        chunk = sources[index * chunk_size : (index + 1) * chunk_size]
        if not chunk:
            continue
        worker = f"T{index + 1}"
        command = (
            f'PYTHONPATH="$PWD/backend:$PWD" .venv/bin/python '
            f"scripts/run_global_ml_diagnostics.py --worker {worker} "
            f"--source-min {min(chunk)} --source-max {max(chunk)}"
        )
        workers.append(
            {
                "worker": worker,
                "source_min": min(chunk),
                "source_max": max(chunk),
                "eligible_source_count": len(chunk),
                "command": command,
            }
        )
    plan = {
        "eligible_source_min": min(sources),
        "eligible_source_max": max(sources),
        "eligible_source_count": len(sources),
        "workers": workers,
    }
    PLAN_PATH.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(plan, indent=2, sort_keys=True))
    print("EXACT WORKER COMMANDS")
    for worker in workers:
        print(worker["command"])
    print(f"WORKER_PLAN_WRITTEN: {PLAN_PATH}")
    return plan


def fetch_draws(cursor, source_min: int, source_max: int) -> dict[int, dict]:
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
        WHERE DrawNo BETWEEN ? AND ?
           OR DrawNo BETWEEN ? AND ?
        ORDER BY DrawNo;
        """,
        source_min,
        source_max,
        source_min + 1,
        source_max + 1,
    ).fetchall()
    return {
        int(row.DrawNo): {
            "draw_no": int(row.DrawNo),
            "draw_date": str(row.DrawDateText) if row.DrawDateText else None,
            "year": int(row.DrawYear) if row.DrawYear else None,
            "month": int(row.DrawMonth) if row.DrawMonth else None,
            "weekday": str(row.WeekdayName) if row.WeekdayName else None,
            "day_type": str(row.DayType or "Unknown"),
            "winners": parse_numbers(row.WinningNumbers),
        }
        for row in rows
    }


def fetch_ledger_rows(cursor, source_min: int, source_max: int) -> list[dict]:
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
            ISNULL(HitCount, 0) AS HitCount
        FROM dbo.PredictionLedger
        WHERE SourceDrawNo BETWEEN ? AND ?
        ORDER BY SourceDrawNo, TargetDrawNo, EngineSource, Mode, RankNo;
        """,
        source_min,
        source_max,
    ).fetchall()
    return [
        {
            "mode": str(row.Mode),
            "engine_source": str(row.EngineSource),
            "source": int(row.SourceDrawNo),
            "target": int(row.TargetDrawNo),
            "rank": int(row.RankNo),
            "number": z4(row.PredictedNumber),
            "score": float(row.Score) if row.Score is not None else None,
            "verification_status": str(row.VerificationStatus),
            "hit_count": int(row.HitCount or 0),
        }
        for row in rows
    ]


def build_global_risk_map(cursor) -> dict[int, tuple[int, str]]:
    rows = cursor.execute(
        """
        SELECT
            p.Mode,
            p.SourceDrawNo,
            p.TargetDrawNo,
            p.RankNo,
            p.PredictedNumber,
            d.WinningNumbers
        FROM dbo.PredictionLedger p
        JOIN dbo.DrawHistory d ON d.DrawNo = p.TargetDrawNo
        WHERE p.EngineSource = ?
          AND p.RankNo BETWEEN 1 AND 5
          AND d.WinningNumbers IS NOT NULL
        ORDER BY p.SourceDrawNo, p.TargetDrawNo, p.Mode, p.RankNo;
        """,
        FINAL_ENGINE,
    ).fetchall()
    grouped: dict[tuple[int, int, str], list] = defaultdict(list)
    actuals: dict[tuple[int, int], tuple[str, ...]] = {}
    for row in rows:
        key = (int(row.SourceDrawNo), int(row.TargetDrawNo), str(row.Mode))
        grouped[key].append(row)
        actuals[(int(row.SourceDrawNo), int(row.TargetDrawNo))] = parse_numbers(
            row.WinningNumbers
        )
    selected: dict[tuple[int, int], tuple[str, tuple[str, ...]]] = {}
    for (source, target, mode), items in grouped.items():
        candidates = tuple(
            z4(item.PredictedNumber)
            for item in sorted(items, key=lambda item: int(item.RankNo))[:5]
        )
        if len(candidates) != 5:
            continue
        old = selected.get((source, target))
        if old is None or MODE_PRIORITY.get(mode, 99) < MODE_PRIORITY.get(old[0], 99):
            selected[(source, target)] = (mode, candidates)
    streak = 0
    output = {}
    for (source, target), (_, candidates) in sorted(selected.items()):
        output[source] = (streak, risk_bucket(streak))
        hit = bool(set(candidates) & set(actuals[(source, target)]))
        streak = 0 if hit else streak + 1
    return output


def engine_label(engine_source: str) -> str | None:
    for label, aliases in TARGET_ENGINES.items():
        if engine_source in aliases:
            return label
    return None


def cutoff_candidates(items: list[dict], cutoff: int) -> list[str] | None:
    ranked = sorted(items, key=lambda item: item["rank"])
    if not ranked or max(item["rank"] for item in ranked) < cutoff:
        return None
    return [item["number"] for item in ranked if item["rank"] <= cutoff]


def hit_metrics(candidates: list[str] | None, actuals: tuple[str, ...]) -> tuple[bool | None, int | None]:
    if candidates is None:
        return None, None
    raw_hits = len(set(candidates) & set(actuals))
    return raw_hits > 0, raw_hits


def process_worker(worker: str, source_min: int, source_max: int) -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = TMP_DIR / f"step152_worker_{worker}_rows.jsonl"
    summary_path = TMP_DIR / f"step152_worker_{worker}_summary.json"
    with get_conn() as connection:
        cursor = connection.cursor()
        draws = fetch_draws(cursor, source_min, source_max)
        ledger = fetch_ledger_rows(cursor, source_min, source_max)
        risk_map = build_global_risk_map(cursor)

    grouped: dict[tuple[int, int, str, str], list[dict]] = defaultdict(list)
    for item in ledger:
        if item["target"] in draws and draws[item["target"]]["winners"]:
            grouped[
                (
                    item["source"],
                    item["target"],
                    item["engine_source"],
                    item["mode"],
                )
            ].append(item)

    by_pair: dict[tuple[int, int], list[tuple[str, str, list[dict]]]] = defaultdict(list)
    for (source, target, engine, mode), items in grouped.items():
        by_pair[(source, target)].append((engine, mode, items))

    output_rows: list[dict] = []
    processed_sources = 0
    for pair_index, ((source, target), groups) in enumerate(sorted(by_pair.items()), start=1):
        target_draw = draws[target]
        actuals = target_draw["winners"]
        pre_streak, bucket = risk_map.get(source, (None, "UNAVAILABLE"))
        for engine, mode, items in sorted(groups):
            available_depth = max(item["rank"] for item in items)
            candidates_by_cutoff = {
                cutoff: cutoff_candidates(items, cutoff) for cutoff in TOP_CUTOFFS
            }
            metrics = {
                cutoff: hit_metrics(candidates_by_cutoff[cutoff], actuals)
                for cutoff in TOP_CUTOFFS
            }
            limitations = [
                f"TOP{cutoff}_UNAVAILABLE_PERSISTED_MAX_RANK={available_depth}"
                for cutoff in TOP_CUTOFFS
                if candidates_by_cutoff[cutoff] is None
            ]
            output_rows.append(
                {
                    "record_type": "recall_depth",
                    "worker": worker,
                    "source_draw_no": source,
                    "target_draw_no": target,
                    "target_date": target_draw["draw_date"],
                    "year": target_draw["year"],
                    "month": target_draw["month"],
                    "weekday": target_draw["weekday"],
                    "day_type": target_draw["day_type"],
                    "risk_bucket": bucket,
                    "full_ledger_pre_miss_streak": pre_streak,
                    "engine_source": engine,
                    "engine_label": engine_label(engine),
                    "mode": mode,
                    "available_candidate_depth": available_depth,
                    "top5_candidates": candidates_by_cutoff[5],
                    "top10_candidates": candidates_by_cutoff[10],
                    "top15_candidates": candidates_by_cutoff[15],
                    "top20_candidates": candidates_by_cutoff[20],
                    "hit_top5": metrics[5][0],
                    "hit_top10": metrics[10][0],
                    "hit_top15": metrics[15][0],
                    "hit_top20": metrics[20][0],
                    "raw_hits_top5": metrics[5][1],
                    "raw_hits_top10": metrics[10][1],
                    "raw_hits_top15": metrics[15][1],
                    "raw_hits_top20": metrics[20][1],
                    "actual_prize_count": len(set(actuals)),
                    "data_source_type": "PERSISTED_LEDGER",
                    "limitations": limitations,
                }
            )

        label_groups: dict[str, tuple[str, str, list[dict]]] = {}
        for engine, mode, items in groups:
            label = engine_label(engine)
            if label is None:
                continue
            old = label_groups.get(label)
            if old is None or MODE_PRIORITY.get(mode, 99) < MODE_PRIORITY.get(old[1], 99):
                label_groups[label] = (engine, mode, items)

        if len(label_groups) >= 2:
            common_cutoff = min(max(item["rank"] for item in value[2]) for value in label_groups.values())
            common_cutoff = max(cutoff for cutoff in TOP_CUTOFFS if cutoff <= common_cutoff)
            number_engines: dict[str, set[str]] = defaultdict(set)
            for label, (_, _, items) in label_groups.items():
                for item in items:
                    if item["rank"] <= common_cutoff:
                        number_engines[item["number"]].add(label)
            for number, labels in sorted(number_engines.items()):
                output_rows.append(
                    {
                        "record_type": "consensus_candidate",
                        "worker": worker,
                        "source_draw_no": source,
                        "target_draw_no": target,
                        "target_date": target_draw["draw_date"],
                        "year": target_draw["year"],
                        "month": target_draw["month"],
                        "weekday": target_draw["weekday"],
                        "day_type": target_draw["day_type"],
                        "risk_bucket": bucket,
                        "common_cutoff": common_cutoff,
                        "number": number,
                        "engines": sorted(labels),
                        "consensus_level": len(labels),
                        "is_hit": number in set(actuals),
                        "actual_prize_count": len(set(actuals)),
                    }
                )

        final_options = [
            value
            for label, value in label_groups.items()
            if label == "TEMPORAL"
        ]
        final_group = final_options[0] if final_options else None
        if final_group is not None:
            final_engine, final_mode, final_items = final_group
            final_top5 = [
                item["number"]
                for item in sorted(final_items, key=lambda item: item["rank"])
                if item["rank"] <= 5
            ]
            final_set = set(final_top5)
            for source_label in ("LINEAR", "WLS", "DELTA"):
                underlying = label_groups.get(source_label)
                if underlying is None:
                    continue
                source_engine, _, source_items = underlying
                max_rank = min(20, max(item["rank"] for item in source_items))
                for item in source_items:
                    if item["rank"] > max_rank:
                        continue
                    if item["number"] in final_set or item["number"] not in set(actuals):
                        continue
                    output_rows.append(
                        {
                            "record_type": "dropped_hit",
                            "worker": worker,
                            "source_draw_no": source,
                            "target_draw_no": target,
                            "target_date": target_draw["draw_date"],
                            "year": target_draw["year"],
                            "month": target_draw["month"],
                            "weekday": target_draw["weekday"],
                            "day_type": target_draw["day_type"],
                            "risk_bucket": bucket,
                            "source_engine": source_engine,
                            "source_engine_label": source_label,
                            "source_rank": item["rank"],
                            "source_score": item["score"],
                            "dropped_number": item["number"],
                            "final_engine": final_engine,
                            "final_mode": final_mode,
                            "final_top5": final_top5,
                            "target_actual_match": True,
                            "digit_sum": digit_sum(item["number"]),
                            "digit_sum_band": sum_band(item["number"]),
                            "mirror_signature": mirror_signature(item["number"]),
                            "first_pair": item["number"][:2],
                            "last_pair": item["number"][2:],
                            "box_signature": box_signature(item["number"]),
                            "guard_filter_reason": None,
                            "removed_by": "unknown",
                            "data_source_type": "PERSISTED_LEDGER",
                            "limitations": (
                                []
                                if max_rank >= 20
                                else [f"UNDERLYING_TOP20_UNAVAILABLE_MAX_RANK={max_rank}"]
                            ),
                        }
                    )

        if pair_index % 100 == 0:
            print(
                f"{worker}: processed {pair_index} source/target pairs "
                f"through source {source}",
                flush=True,
            )
        processed_sources = pair_index

    with rows_path.open("w", encoding="utf-8") as handle:
        for row in sorted(
            output_rows,
            key=lambda item: (
                item["source_draw_no"],
                item["target_draw_no"],
                item["record_type"],
                item.get("engine_source", ""),
                item.get("mode", ""),
                item.get("number", item.get("dropped_number", "")),
            ),
        ):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "worker": worker,
        "source_min": source_min,
        "source_max": source_max,
        "pairs_processed": processed_sources,
        "rows_written": len(output_rows),
        "record_counts": dict(Counter(row["record_type"] for row in output_rows)),
        "completed": True,
        "rows_path": str(rows_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def load_worker_outputs() -> tuple[list[dict], list[dict]]:
    if not PLAN_PATH.exists():
        raise RuntimeError("Worker plan is missing; run --plan-workers first")
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    rows = []
    summaries = []
    for worker in plan["workers"]:
        worker_id = worker["worker"]
        rows_path = TMP_DIR / f"step152_worker_{worker_id}_rows.jsonl"
        summary_path = TMP_DIR / f"step152_worker_{worker_id}_summary.json"
        if not rows_path.exists() or not summary_path.exists():
            raise RuntimeError(f"Worker {worker_id} output is incomplete")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not summary.get("completed"):
            raise RuntimeError(f"Worker {worker_id} did not complete")
        summaries.append(summary)
        with rows_path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    rows.sort(
        key=lambda item: (
            item["source_draw_no"],
            item["target_draw_no"],
            item["record_type"],
            item.get("engine_source", ""),
            item.get("mode", ""),
            item.get("number", item.get("dropped_number", "")),
        )
    )
    return rows, summaries


def random_rate(cutoff: int, prize_count: float) -> float:
    return 1.0 - ((NUMBER_SPACE - prize_count) / NUMBER_SPACE) ** cutoff


def summarize_recall(items: list[dict]) -> dict:
    output = {}
    for cutoff in TOP_CUTOFFS:
        available = [row for row in items if row[f"hit_top{cutoff}"] is not None]
        count = len(available)
        hit_draws = sum(bool(row[f"hit_top{cutoff}"]) for row in available)
        raw_hits = sum(int(row[f"raw_hits_top{cutoff}"] or 0) for row in available)
        expected = (
            sum(random_rate(cutoff, row["actual_prize_count"]) for row in available)
            / count
            if count
            else None
        )
        hit_rate = hit_draws / count if count else None
        output[f"top{cutoff}"] = {
            "rows": count,
            "draws_with_hit": hit_draws,
            "raw_hits": raw_hits,
            "hit_rate": hit_rate,
            "raw_hits_per_draw": raw_hits / count if count else None,
            "random_expected_hit_rate": expected,
            "enrichment_vs_random": (
                hit_rate / expected if hit_rate is not None and expected else None
            ),
        }
    for upper, lower in ((10, 5), (15, 10), (20, 15)):
        upper_rate = output[f"top{upper}"]["hit_rate"]
        lower_rate = output[f"top{lower}"]["hit_rate"]
        output[f"marginal_lift_top{upper}_minus_top{lower}"] = (
            upper_rate - lower_rate
            if upper_rate is not None and lower_rate is not None
            else None
        )
    return output


def grouped_matrix(
    rows: list[dict],
    key_fn: Callable[[dict], str],
) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)
    return {key: summarize_recall(items) for key, items in sorted(grouped.items())}


def consensus_matrix(rows: list[dict]) -> dict:
    output = {}
    levels: dict[str, Callable[[int], bool]] = {
        "exactly_1": lambda value: value == 1,
        "exactly_2": lambda value: value == 2,
        "exactly_3": lambda value: value == 3,
        "exactly_4": lambda value: value == 4,
        "2_plus": lambda value: value >= 2,
        "3_plus": lambda value: value >= 3,
    }
    for name, predicate in levels.items():
        selected = [row for row in rows if predicate(row["consensus_level"])]
        draws = {(row["source_draw_no"], row["target_draw_no"]) for row in selected}
        hit_draws = {
            (row["source_draw_no"], row["target_draw_no"])
            for row in selected
            if row["is_hit"]
        }
        exact_hits = sum(row["is_hit"] for row in selected)
        random_candidate_probability = (
            sum(row["actual_prize_count"] / NUMBER_SPACE for row in selected)
            / len(selected)
            if selected
            else None
        )
        candidate_probability = exact_hits / len(selected) if selected else None
        output[name] = {
            "candidate_instances": len(selected),
            "unique_candidate_numbers": len({row["number"] for row in selected}),
            "draw_instances": len(draws),
            "exact_hits": exact_hits,
            "hit_probability_per_candidate": candidate_probability,
            "draw_hit_rate_if_consensus_exists": (
                len(hit_draws) / len(draws) if draws else None
            ),
            "random_candidate_hit_probability": random_candidate_probability,
            "enrichment_vs_random": (
                candidate_probability / random_candidate_probability
                if candidate_probability is not None and random_candidate_probability
                else None
            ),
        }
    combinations: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if len(row["engines"]) >= 2:
            combinations["+".join(row["engines"])].append(row)
    output["engine_combinations"] = {
        key: {
            "candidate_instances": len(items),
            "exact_hits": sum(item["is_hit"] for item in items),
            "hit_probability": (
                sum(item["is_hit"] for item in items) / len(items) if items else None
            ),
        }
        for key, items in sorted(combinations.items())
    }
    return output


def dropped_matrix(rows: list[dict]) -> dict:
    dimensions = {
        "year": lambda row: str(row["year"]),
        "month": lambda row: str(row["month"]),
        "day_type": lambda row: row["day_type"],
        "weekday": lambda row: row["weekday"],
        "risk_bucket": lambda row: row["risk_bucket"],
        "source_engine": lambda row: row["source_engine"],
        "rank_bucket": lambda row: (
            "01_05"
            if row["source_rank"] <= 5
            else "06_10"
            if row["source_rank"] <= 10
            else "11_15"
            if row["source_rank"] <= 15
            else "16_20"
        ),
        "digit_sum_band": lambda row: row["digit_sum_band"],
        "mirror_signature": lambda row: row["mirror_signature"],
        "first_pair": lambda row: row["first_pair"],
        "last_pair": lambda row: row["last_pair"],
        "removed_by": lambda row: row["removed_by"],
    }
    output = {"total_dropped_hits": len(rows)}
    for name, key_fn in dimensions.items():
        output[name] = dict(Counter(key_fn(row) for row in rows).most_common())
    return output


def format_rate(value: float | None) -> str:
    return "UNAVAILABLE" if value is None else f"{value * 100:.4f}%"


def strength_summary(
    recall_global: dict,
    consensus: dict,
    dropped: dict,
    discovery: dict,
) -> dict:
    persisted_depth = discovery.get("maximum_persisted_rank", 0)
    top5_rates = [
        item["top5"]["enrichment_vs_random"]
        for item in recall_global.values()
        if item["top5"]["enrichment_vs_random"] is not None
    ]
    average_enrichment = sum(top5_rates) / len(top5_rates) if top5_rates else None
    candidate_strength = (
        "unavailable"
        if average_enrichment is None
        else "strong"
        if average_enrichment >= 1.5
        else "weak"
        if average_enrichment >= 1.05
        else "random-like"
    )
    consensus_enrichment = consensus.get("2_plus", {}).get("enrichment_vs_random")
    consensus_strength = (
        "unavailable"
        if consensus_enrichment is None
        else "helpful"
        if consensus_enrichment > 1.1
        else "harmful"
        if consensus_enrichment < 0.9
        else "neutral"
    )
    dropped_count = dropped["total_dropped_hits"]
    reranking_strength = (
        "unavailable"
        if not recall_global
        else "harmful"
        if dropped_count > 0
        else "neutral"
    )
    guard_strength = "unknown" if dropped_count == 0 else "over-suppressing"
    if persisted_depth < 10:
        bottleneck = "Data sparsity"
        next_step = "Build deeper candidate persistence"
    elif candidate_strength == "random-like":
        bottleneck = "Candidate generation"
        next_step = "Stop expansion due to random-like behavior"
    elif reranking_strength == "harmful":
        bottleneck = "Reranking"
        next_step = "Fix reranker"
    elif consensus_strength == "helpful":
        bottleneck = "Over-filtering"
        next_step = "Build consensus engine"
    else:
        bottleneck = "Engine unavailability"
        next_step = "Build deeper candidate persistence"
    return {
        "candidate_generation_strength": candidate_strength,
        "reranking_strength": reranking_strength,
        "consensus_strength": consensus_strength,
        "guard_layer_strength": guard_strength,
        "main_bottleneck": bottleneck,
        "recommended_next_engineering_step": next_step,
        "average_top5_enrichment": average_enrichment,
        "persisted_max_depth": persisted_depth,
    }


def merge() -> None:
    if not DISCOVERY_PATH.exists():
        discover()
    discovery_data = json.loads(DISCOVERY_PATH.read_text(encoding="utf-8"))
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    rows, summaries = load_worker_outputs()
    recall_rows = [row for row in rows if row["record_type"] == "recall_depth"]
    consensus_rows = [row for row in rows if row["record_type"] == "consensus_candidate"]
    dropped_rows = [row for row in rows if row["record_type"] == "dropped_hit"]

    global_by_engine_mode = grouped_matrix(
        recall_rows, lambda row: f"{row['engine_source']}::{row['mode']}"
    )
    by_year = grouped_matrix(recall_rows, lambda row: str(row["year"]))
    by_decade = grouped_matrix(
        recall_rows,
        lambda row: (
            f"{(row['year'] // 10) * 10}s" if row["year"] is not None else "Unknown"
        ),
    )
    by_day_type = grouped_matrix(recall_rows, lambda row: row["day_type"])
    by_risk = grouped_matrix(recall_rows, lambda row: row["risk_bucket"])

    source_order = sorted({row["source_draw_no"] for row in recall_rows})
    windows = {
        "FULL_VERIFIED": set(source_order),
        "RECENT_365": set(source_order[-365:]),
        "RECENT_90": set(source_order[-90:]),
        "RECENT_47": set(source_order[-47:]),
    }
    recent_windows = {
        name: summarize_recall(
            [row for row in recall_rows if row["source_draw_no"] in sources]
        )
        for name, sources in windows.items()
    }
    consensus = consensus_matrix(consensus_rows)
    consensus_by_day_type = {
        key: consensus_matrix(items)
        for key, items in sorted(
            _group(consensus_rows, lambda row: row["day_type"]).items()
        )
    }
    consensus_by_month = {
        key: consensus_matrix(items)
        for key, items in sorted(
            _group(consensus_rows, lambda row: str(row["month"])).items()
        )
    }
    consensus_by_year = {
        key: consensus_matrix(items)
        for key, items in sorted(
            _group(consensus_rows, lambda row: str(row["year"])).items()
        )
    }
    consensus_by_risk = {
        key: consensus_matrix(items)
        for key, items in sorted(
            _group(consensus_rows, lambda row: row["risk_bucket"]).items()
        )
    }
    dropped = dropped_matrix(dropped_rows)
    strength = strength_summary(
        global_by_engine_mode, consensus, dropped, discovery_data
    )

    matrices = {
        "discovery": discovery_data,
        "worker_plan": plan,
        "worker_summaries": summaries,
        "multi_tier_recall": {
            "global_by_engine_mode": global_by_engine_mode,
            "by_year": by_year,
            "by_decade": by_decade,
            "by_day_type": by_day_type,
            "by_risk_bucket": by_risk,
            "recent_windows": recent_windows,
        },
        "consensus": {
            "global": consensus,
            "by_day_type": consensus_by_day_type,
            "by_month": consensus_by_month,
            "by_year": consensus_by_year,
            "by_risk_bucket": consensus_by_risk,
        },
        "dropped_signals": dropped,
        "true_matrix_strength": strength,
    }
    MATRICES_PATH.write_text(
        json.dumps(matrices, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with ROWS_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    lines = [
        "=" * 156,
        "STEP 152 — GLOBAL ML FORENSIC & PRECISION DIAGNOSTICS — REPORT ONLY",
        "=" * 156,
        "ParallelWorkers: 4",
        "ProductionMathChanged: NO",
        "APIChanged: NO",
        "FrontendChanged: NO",
        "SQLSchemaChanged: NO",
        "DeploymentChanged: NO",
        "DBWritePerformed: NO",
        "",
        "SOURCE DISCOVERY",
        "-" * 156,
        f"DrawHistoryColumns: {[item['name'] for item in discovery_data['drawhistory_columns']]}",
        f"PredictionLedgerColumns: {[item['name'] for item in discovery_data['predictionledger_columns']]}",
        f"EngineSources: {discovery_data['distinct_engine_sources']}",
        f"Modes: {discovery_data['distinct_modes']}",
        f"RankScoreColumns: {discovery_data['rank_score_columns']}",
        f"MaximumPersistedRank: {discovery_data['maximum_persisted_rank']}",
        f"EngineAvailability: {discovery_data['engine_availability']}",
        f"DATA_LIMITATION: {discovery_data['data_limitations']}",
        "",
        "WORKER PLAN AND COMPLETION",
        "-" * 156,
        f"WorkerPlan: {plan['workers']}",
        f"WorkerSummaries: {summaries}",
        "",
        "MULTI-TIER RECALL DEPTH MATRIX",
        "-" * 156,
        "Engine/Mode                                              Top5 Rows HitRate Enrich   Top10 HitRate   Top15 HitRate   Top20 HitRate",
    ]
    for key, item in global_by_engine_mode.items():
        lines.append(
            f"{key:<56} {item['top5']['rows']:>9} "
            f"{format_rate(item['top5']['hit_rate']):>11} "
            f"{str(round(item['top5']['enrichment_vs_random'], 4)) if item['top5']['enrichment_vs_random'] is not None else 'UNAVAILABLE':>8} "
            f"{format_rate(item['top10']['hit_rate']):>15} "
            f"{format_rate(item['top15']['hit_rate']):>15} "
            f"{format_rate(item['top20']['hit_rate']):>15}"
        )
    lines.extend(
        (
            "",
            f"RecentWindows: {recent_windows}",
            f"ByYear: {by_year}",
            f"ByDecade: {by_decade}",
            f"ByDayType: {by_day_type}",
            f"ByRiskBucket: {by_risk}",
            "",
            "CONSENSUS MATRIX",
            "-" * 156,
            f"Availability: {discovery_data['engine_availability']}",
            f"GlobalConsensus: {consensus}",
            f"ConsensusByDayType: {consensus_by_day_type}",
            f"ConsensusByMonth: {consensus_by_month}",
            f"ConsensusByYear: {consensus_by_year}",
            f"ConsensusByRiskBucket: {consensus_by_risk}",
            "",
            "DROPPED SIGNAL MATRIX",
            "-" * 156,
            f"DroppedSignalSummary: {dropped}",
            "Guard/filter reasons are marked unknown unless persisted metadata proves the cause.",
            "",
            "YEAR / MONTH / DAYTYPE PATTERN TABLES",
            "-" * 156,
            f"RecallByYear: {by_year}",
            f"ConsensusByMonth: {consensus_by_month}",
            f"RecallByDayType: {by_day_type}",
            f"DroppedByMonth: {dropped['month']}",
            f"DroppedByDayType: {dropped['day_type']}",
            "",
            "TRUE MATRIX STRENGTH CONCLUSION",
            "-" * 156,
            f"CandidateGenerationStrength: {strength['candidate_generation_strength']}",
            f"RerankingStrength: {strength['reranking_strength']}",
            f"ConsensusStrength: {strength['consensus_strength']}",
            f"GuardLayerStrength: {strength['guard_layer_strength']}",
            f"MainBottleneck: {strength['main_bottleneck']}",
            f"RecommendedNextEngineeringStep: {strength['recommended_next_engineering_step']}",
            "",
            "FINAL RECOMMENDATION",
            "-" * 156,
            "ProductionSwitchRecommendedNow: NO",
            f"NextStep: {strength['recommended_next_engineering_step']}",
            "",
            f"REPORT_WRITTEN: {REPORT_PATH}",
            f"MATRICES_WRITTEN: {MATRICES_PATH}",
            f"ROWS_WRITTEN: {ROWS_PATH}",
        )
    )
    report = "\n".join(lines)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    print(report)


def _group(rows: Iterable[dict], key_fn: Callable[[dict], str]) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        output[key_fn(row)].append(row)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--discover", action="store_true")
    modes.add_argument("--plan-workers", action="store_true")
    modes.add_argument("--worker")
    modes.add_argument("--merge", action="store_true")
    parser.add_argument("--source-min", type=int)
    parser.add_argument("--source-max", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.discover:
        discover()
    elif args.plan_workers:
        plan_workers()
    elif args.worker:
        if args.worker not in {"T1", "T2", "T3", "T4"}:
            raise ValueError("Worker must be T1, T2, T3, or T4")
        if args.source_min is None or args.source_max is None:
            raise ValueError("Worker mode requires --source-min and --source-max")
        process_worker(args.worker, args.source_min, args.source_max)
    elif args.merge:
        merge()


if __name__ == "__main__":
    main()
