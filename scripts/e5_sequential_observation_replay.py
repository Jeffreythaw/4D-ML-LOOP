from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))
load_dotenv(BACKEND_ROOT / ".env")

from j4d_e5.memory import load_memory, update_memory, write_memory  # noqa: E402
from j4d_e5.provenance import normalize_4d  # noqa: E402
from j4d_e5.segments import compare_segments  # noqa: E402
from jeffrey_quad_engine_v2_step3_adaptive_orchestrator import (  # noqa: E402
    Step3AdaptiveOrchestrator,
    get_sql_connection_string_from_env,
    import_step2_core,
    load_env_file,
)


HIGH_CLASSES = {
    "EXACT4_MATCH",
    "LAST3_MATCH",
    "PREFIX3_MATCH",
    "SAME_POSITION_3",
}
MEDIUM_CLASSES = {
    "PREFIX2_MATCH",
    "SUFFIX2_MATCH",
    "MIDDLE2_MATCH",
    "SAME_POSITION_2",
    "PAIR_13_MATCH",
    "PAIR_14_MATCH",
    "PAIR_24_MATCH",
}


@dataclass(frozen=True)
class MemoryAttributionRow:
    target_draw_no: int
    candidate_number: str
    actual_number: str
    segment_class: str
    is_exact: bool
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_draw_no": self.target_draw_no,
            "candidate_number": self.candidate_number,
            "actual_number": self.actual_number,
            "segment_class": self.segment_class,
            "is_exact": self.is_exact,
            "provenance": self.provenance,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequential E5 completed-draw observation replay. Read-only prediction, post-prediction actual loading.",
    )
    parser.add_argument("--start-target-draw", type=int, required=True)
    parser.add_argument("--end-target-draw", type=int, required=True)
    parser.add_argument("--memory-path", default="reports/patches/e5_segment_memory_replay_observation.json")
    parser.add_argument("--report-path", default="reports/patches/e5_sequential_observation_replay_report.jsonl")
    parser.add_argument("--training-window-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--write-memory",
        action="store_true",
        help="Persist memory JSON. Default is no-write/report-only.",
    )
    return parser.parse_args()


def _engine_family(engine_name: str | None) -> str | None:
    if not engine_name:
        return None
    for prefix in ("E40_", "E1_", "E2_", "E3_", "E4_"):
        if engine_name.startswith(prefix):
            return prefix[:-1]
    return engine_name


def _formula_id(vote: dict[str, Any]) -> str | None:
    return (
        vote.get("formula_version")
        or vote.get("formula_id")
        or vote.get("method_name")
        or vote.get("engine_name")
    )


def _tier(classes: set[str]) -> str | None:
    if classes & HIGH_CLASSES:
        return "HIGH"
    if classes & MEDIUM_CLASSES:
        return "MEDIUM"
    return None


def _rows_for_locked_prediction(
    *,
    locked: Any,
    actuals: tuple[str, ...],
    target_draw_no: int,
    day_type: str | None,
) -> tuple[MemoryAttributionRow, ...]:
    rows: list[MemoryAttributionRow] = []

    for candidate_row in locked.candidate_provenance:
        candidate = normalize_4d(candidate_row["candidate_number"], field_name="candidate_number")
        votes = tuple(candidate_row.get("votes", ()))

        for actual in actuals:
            classes = compare_segments(candidate, actual)
            tier = _tier(classes)
            if tier is None:
                continue

            selected_classes = sorted((classes & HIGH_CLASSES) | (classes & MEDIUM_CLASSES))
            if not selected_classes:
                continue

            for vote in votes:
                provenance = dict(vote)
                provenance["engine_family"] = _engine_family(provenance.get("engine_name"))
                provenance["formula_id"] = _formula_id(provenance)
                provenance["day_type"] = day_type
                provenance["memory_tier"] = tier
                provenance["selected_engine_source"] = candidate_row.get("selected_engine_source")
                provenance["final_rank"] = candidate_row.get("final_rank")

                for segment_class in selected_classes:
                    rows.append(
                        MemoryAttributionRow(
                            target_draw_no=target_draw_no,
                            candidate_number=candidate,
                            actual_number=actual,
                            segment_class=segment_class,
                            is_exact=segment_class == "EXACT4_MATCH",
                            provenance=provenance,
                        )
                    )

    return tuple(rows)


def main() -> int:
    args = parse_args()
    load_env_file()

    if args.end_target_draw < args.start_target_draw:
        raise ValueError("--end-target-draw must be >= --start-target-draw")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when supplied")

    core = import_step2_core()
    conn_str = get_sql_connection_string_from_env()
    gateway = core.SqlServerGateway(conn_str, autocommit=False, no_sql_write=True)
    gateway.connect()

    memory_path = PROJECT_ROOT / args.memory_path
    report_path = PROJECT_ROOT / args.report_path
    memory = load_memory(memory_path)

    report_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    total_rows = 0

    try:
        with report_path.open("w", encoding="utf-8") as handle:
            for target_draw_no in range(args.start_target_draw, args.end_target_draw + 1):
                if args.limit is not None and processed >= args.limit:
                    break

                source_draw_no = target_draw_no - 1

                # FIREWALL: prediction is locked before loading target actuals.
                orchestrator = Step3AdaptiveOrchestrator(
                    core=core,
                    gateway=gateway,
                    start_draw_no=source_draw_no,
                    end_draw_no=target_draw_no,
                    training_window_size=args.training_window_size,
                )
                locked = orchestrator.predict_one_step_locked(source_draw_no)

                # POST-PREDICTION ONLY: completed actuals loaded after locked Top5.
                actual_record = gateway.load_phase2_draw(target_draw_no)
                if actual_record is None:
                    skipped += 1
                    handle.write(
                        json.dumps(
                            {
                                "target_draw_no": target_draw_no,
                                "source_draw_no": source_draw_no,
                                "status": "SKIPPED_TARGET_DRAW_NOT_FOUND",
                                "production_ranking_changed": False,
                                "sql_write": False,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    continue

                actuals = tuple(normalize_4d(v, field_name="actual") for v in actual_record.winning_numbers)
                actual_set = set(actuals)
                rows = _rows_for_locked_prediction(
                    locked=locked,
                    actuals=actuals,
                    target_draw_no=target_draw_no,
                    day_type=getattr(actual_record, "day_type", None),
                )
                memory = update_memory(memory, rows, target_draw_no=target_draw_no)

                payload = {
                    "target_draw_no": target_draw_no,
                    "source_draw_no": source_draw_no,
                    "status": "OBSERVED",
                    "top5": list(locked.top5),
                    "exact_hit_count": sum(1 for n in locked.top5 if n in actual_set),
                    "actual_count": len(actuals),
                    "day_type": getattr(actual_record, "day_type", None),
                    "attribution_rows": len(rows),
                    "memory_entry_count": len(memory.get("entries", {})),
                    "production_ranking_changed": False,
                    "sql_write": False,
                    "memory_write": bool(args.write_memory),
                    "firewall_order": "predict_locked_before_actual_load",
                }
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
                processed += 1
                total_rows += len(rows)

        write_memory(memory_path, memory, no_write=not args.write_memory)

    finally:
        gateway.rollback()
        gateway.close()

    print(f"WROTE {report_path}")
    if args.write_memory:
        print(f"WROTE {memory_path}")
    else:
        print(f"MEMORY_NOT_WRITTEN {memory_path}")
    print(f"processed={processed}")
    print(f"skipped={skipped}")
    print(f"attribution_rows={total_rows}")
    print("E5_SEQUENTIAL_OBSERVATION_REPLAY_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
