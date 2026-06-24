from __future__ import annotations

import argparse
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
        description="Observe one completed draw with E5 segment attribution. Observation-only; does not alter production ranking.",
    )
    parser.add_argument("--source-draw", type=int, required=True)
    parser.add_argument("--target-draw", type=int, required=True)
    parser.add_argument(
        "--actuals",
        required=True,
        help="Comma-separated completed actual 4D results. Must be supplied after draw completion.",
    )
    parser.add_argument(
        "--memory-path",
        default="reports/patches/e5_segment_memory_observation.json",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional report path. Defaults to reports/patches/e5_draw_<target>_observation_report.txt",
    )
    parser.add_argument(
        "--write-memory",
        action="store_true",
        help="Persist observation memory JSON. Default is report-only/no memory write.",
    )
    parser.add_argument("--day-type", default=None)
    parser.add_argument("--training-window-size", type=int, default=64)
    return parser.parse_args()


def _parse_actuals(raw: str) -> tuple[str, ...]:
    values = tuple(normalize_4d(part.strip(), field_name="actual") for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("--actuals must contain at least one completed actual number")
    if len(values) != len(set(values)):
        raise ValueError("--actuals contains duplicate numbers")
    return values


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


def _build_locked_prediction(*, source_draw_no: int, target_draw_no: int, training_window_size: int) -> Any:
    core = import_step2_core()
    conn_str = get_sql_connection_string_from_env()
    gateway = core.SqlServerGateway(conn_str, autocommit=False, no_sql_write=True)
    gateway.connect()
    try:
        orchestrator = Step3AdaptiveOrchestrator(
            core=core,
            gateway=gateway,
            start_draw_no=source_draw_no,
            end_draw_no=target_draw_no,
            training_window_size=training_window_size,
        )
        return orchestrator.predict_one_step_locked(source_draw_no)
    finally:
        gateway.rollback()
        gateway.close()


def main() -> int:
    args = parse_args()
    load_env_file()

    if args.target_draw <= args.source_draw:
        raise ValueError("--target-draw must be greater than --source-draw")

    # FIREWALL CLARITY: lock prediction before parsing completed actuals.
    locked = _build_locked_prediction(
        source_draw_no=args.source_draw,
        target_draw_no=args.target_draw,
        training_window_size=args.training_window_size,
    )

    # POST-PREDICTION ONLY: explicit completed actuals are parsed after locked Top5.
    actuals = _parse_actuals(args.actuals)
    actual_set = set(actuals)

    report_path = Path(args.report_path) if args.report_path else (
        PROJECT_ROOT / "reports" / "patches" / f"e5_draw_{args.target_draw}_observation_report.txt"
    )
    memory_path = PROJECT_ROOT / args.memory_path

    rows: list[MemoryAttributionRow] = []
    report_lines: list[str] = [
        f"E5 Draw {args.target_draw} Completed Observation Report",
        f"SourceDrawNo: {args.source_draw}",
        f"TargetDrawNo: {args.target_draw}",
        f"PredictedTop5: {', '.join(locked.top5)}",
        "ActualSource: explicit_cli_completed_draw_input",
        f"ActualCount: {len(actuals)}",
        f"ExactHitCount: {sum(1 for n in locked.top5 if n in actual_set)}",
        f"WriteMemory: {bool(args.write_memory)}",
        "UseForFuturePrediction: false",
        "MemoryMode: observation_only",
        "",
        "LearnedHighMediumRows:",
    ]

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
                provenance["day_type"] = args.day_type
                provenance["memory_tier"] = tier
                provenance["selected_engine_source"] = candidate_row.get("selected_engine_source")
                provenance["final_rank"] = candidate_row.get("final_rank")

                for segment_class in selected_classes:
                    rows.append(
                        MemoryAttributionRow(
                            target_draw_no=args.target_draw,
                            candidate_number=candidate,
                            actual_number=actual,
                            segment_class=segment_class,
                            is_exact=segment_class == "EXACT4_MATCH",
                            provenance=provenance,
                        )
                    )

            report_lines.append(
                f"{tier}: {candidate} -> {actual}: {', '.join(selected_classes)} "
                f"| votes={len(votes)} | selected={candidate_row.get('selected_engine_source')}"
            )

    memory = load_memory(memory_path)
    memory = update_memory(memory, rows, target_draw_no=args.target_draw)
    write_memory(memory_path, memory, no_write=not args.write_memory)

    report_lines.extend(
        [
            "",
            f"AttributionRowsGenerated: {len(rows)}",
            f"MemoryEntryCountAfterObservation: {len(memory.get('entries', {}))}",
            f"MemoryPath: {memory_path}",
            f"ReportPath: {report_path}",
            "SQLWrite: false",
            "ProductionRankingChanged: false",
            "FinalDecision: E5_COMPLETED_DRAW_OBSERVATION_READY",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"WROTE {report_path}")
    if args.write_memory:
        print(f"WROTE {memory_path}")
    else:
        print(f"MEMORY_NOT_WRITTEN {memory_path}")
    print("E5_COMPLETED_DRAW_OBSERVATION_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
