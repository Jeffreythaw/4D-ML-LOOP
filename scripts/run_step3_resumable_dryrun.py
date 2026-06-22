#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jeffrey_quad_engine_v2_step2_matrix_core as core
import jeffrey_quad_engine_v2_step3_adaptive_orchestrator as step3


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume-safe Step3 dry-run runner with local JSONL checkpointing."
    )
    parser.add_argument("--start-draw-no", type=int, default=step3.SIM_START_DRAW_NO)
    parser.add_argument("--end-draw-no", type=int, default=step3.SIM_END_DRAW_NO)
    parser.add_argument("--training-window-size", type=int, default=0)
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=PROJECT_ROOT / "reports" / "step3_resumable_dryrun_4051_5494.jsonl",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N newly processed draws.",
    )
    return parser.parse_args(argv)


def load_completed(path: Path) -> Set[int]:
    completed: Set[int] = set()
    if not path.exists:
        return completed
    if not path.exists():
        return completed

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc

            if row.get("status") == "ok":
                completed.add(int(row["source_draw_no"]))

    return completed


def load_binary_hit_history(path: Path) -> List[int]:
    history: List[int] = []
    if not path.exists():
        return history

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc

            if row.get("status") != "ok":
                continue

            hit_count = int(row.get("hit_count", 0))
            history.append(1 if hit_count > 0 else 0)

    return history


def hydrate_rolling_metrics(metrics: Any, binary_history: Sequence[int]) -> None:
    for binary_hit in binary_history:
        metrics.update(1 if int(binary_hit) > 0 else 0)


def result_to_row(result: Any) -> Dict[str, Any]:
    return {
        "status": "ok",
        "source_draw_no": int(result.source_draw_no),
        "target_draw_no": int(result.target_draw_no),
        "source_day_type": str(result.source_day_type),
        "top5": list(result.top5),
        "engine_sources": list(result.engine_sources),
        "hit_count": int(result.hit_count),
        "adaptive_triggered": bool(result.adaptive_triggered),
        "adaptive_formula_id": (
            None if result.adaptive_formula_id is None else int(result.adaptive_formula_id)
        ),
        "rolling_metrics": result.rolling_metrics,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.start_draw_no < step3.SIM_START_DRAW_NO:
        raise ValueError(f"start_draw_no must be >= {step3.SIM_START_DRAW_NO}")
    if args.end_draw_no > step3.SIM_END_DRAW_NO:
        raise ValueError(f"end_draw_no must be <= {step3.SIM_END_DRAW_NO}")
    if args.start_draw_no >= args.end_draw_no:
        raise ValueError("start_draw_no must be less than end_draw_no")
    if args.training_window_size < 0:
        raise ValueError("training_window_size must be >= 0")
    if args.progress_every <= 0:
        raise ValueError("progress_every must be positive")

    step3.load_env_file()
    conn_str = core.get_sql_connection_string_from_env()

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(args.output_jsonl)
    binary_history = load_binary_hit_history(args.output_jsonl)

    total_requested = args.end_draw_no - args.start_draw_no
    to_process = [
        draw_no
        for draw_no in range(args.start_draw_no, args.end_draw_no)
        if draw_no not in completed
    ]

    print("=" * 96)
    print("STEP3 RESUMABLE DRY-RUN START")
    print("=" * 96)
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DB_SERVER: {os.getenv('DB_SERVER', '<not set>')}")
    print(f"DB_DATABASE: {os.getenv('DB_DATABASE', '<not set>')}")
    print(f"RANGE: {args.start_draw_no}..{args.end_draw_no - 1}")
    print(f"TOTAL_REQUESTED: {total_requested}")
    print(f"ALREADY_COMPLETED: {len(completed)}")
    print(f"TO_PROCESS: {len(to_process)}")
    print(f"TRAINING_WINDOW_SIZE: {args.training_window_size}")
    print(f"OUTPUT_JSONL: {args.output_jsonl}")
    print("-" * 96)

    newly_processed = 0
    skipped = total_requested - len(to_process)

    with args.output_jsonl.open("a", encoding="utf-8") as out:
        for source_draw_no in to_process:
            gateway = core.SqlServerGateway(conn_str, autocommit=False, timeout_seconds=60)
            gateway.connect()
            try:
                orchestrator = step3.Step3AdaptiveOrchestrator(
                    core=core,
                    gateway=gateway,
                    start_draw_no=source_draw_no,
                    end_draw_no=source_draw_no + 1,
                    training_window_size=args.training_window_size,
                )
                hydrate_rolling_metrics(orchestrator.metrics, binary_history)

                result = orchestrator.run_one_step(source_draw_no)

                if result is None:
                    row = {
                        "status": "skipped",
                        "source_draw_no": int(source_draw_no),
                    }
                else:
                    row = result_to_row(result)

                out.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
                out.flush()

                if row.get("status") == "ok":
                    binary_history.append(1 if int(row.get("hit_count", 0)) > 0 else 0)

                gateway.rollback()

                newly_processed += 1
                if newly_processed == 1 or newly_processed % args.progress_every == 0:
                    hit = row.get("hit_count", "NA")
                    adaptive = row.get("adaptive_triggered", "NA")
                    print(
                        f"[{newly_processed}/{len(to_process)}] "
                        f"source={source_draw_no} status={row['status']} "
                        f"hit={hit} adaptive={adaptive}"
                    )

            except Exception as exc:
                gateway.rollback()
                error_row = {
                    "status": "error",
                    "source_draw_no": int(source_draw_no),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                out.write(json.dumps(error_row, separators=(",", ":"), sort_keys=True) + "\n")
                out.flush()
                print(f"ERROR_AT_SOURCE_DRAW_NO: {source_draw_no}")
                print(f"{type(exc).__name__}: {exc}")
                raise
            finally:
                gateway.close()

    print("-" * 96)
    print("STEP3 RESUMABLE DRY-RUN SUMMARY")
    print(f"TOTAL_REQUESTED: {total_requested}")
    print(f"SKIPPED_ALREADY_COMPLETED: {skipped}")
    print(f"NEWLY_PROCESSED: {newly_processed}")
    print(f"OUTPUT_JSONL: {args.output_jsonl}")
    print("RESULT: COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
