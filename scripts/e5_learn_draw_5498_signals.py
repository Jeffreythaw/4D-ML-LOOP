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
MEMORY_PATH = PROJECT_ROOT / "reports" / "patches" / "e5_segment_memory_observation.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "patches" / "e5_draw_5498_memory_learning_report.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Legacy 5498 E5 observation script. Defaults to no memory write.",
    )
    parser.add_argument(
        "--write-memory",
        action="store_true",
        help="Persist memory JSON. Default is report-only/no memory write.",
    )
    return parser.parse_args()


sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))
load_dotenv(BACKEND_ROOT / ".env")

from j4d_e5.memory import load_memory, update_memory, write_memory  # noqa: E402
from j4d_e5.segments import compare_segments  # noqa: E402
from jeffrey_quad_engine_v2_step3_adaptive_orchestrator import (  # noqa: E402
    Step3AdaptiveOrchestrator,
    get_sql_connection_string_from_env,
    import_step2_core,
    load_env_file,
)


TARGET_DRAW_NO = 5498
SOURCE_DRAW_NO = 5497

# Completed draw actuals only. Never used for prediction.
ACTUAL_5498 = (
    "9954", "2614", "6272",
    "0324", "0327", "1364", "1835", "3726", "3800", "5816", "6608", "6989", "9564",
    "0062", "0219", "0445", "4693", "6118", "6424", "7552", "8286", "8663", "8916",
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
IGNORED_LOW_CLASSES = {
    "DIGIT_BAG_3_MATCH",
    "DIGIT_BAG_2_MATCH",
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


def _engine_family(engine_name: str | None) -> str | None:
    if not engine_name:
        return None
    if engine_name.startswith("E1_"):
        return "E1"
    if engine_name.startswith("E2_"):
        return "E2"
    if engine_name.startswith("E3_"):
        return "E3"
    if engine_name.startswith("E4_"):
        return "E4"
    if engine_name.startswith("E40_"):
        return "E40"
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


def _completed_actuals_from_sql(core: Any) -> tuple[str, ...] | None:
    # Keep this intentionally optional. If DB shape differs, use constant above.
    try:
        conn_str = get_sql_connection_string_from_env()
        gateway = core.SqlServerGateway(conn_str, autocommit=False, no_sql_write=True)
        gateway.connect()
        try:
            record = gateway.load_phase2_draw(TARGET_DRAW_NO)
            if record and tuple(record.winning_numbers):
                return tuple(str(x).zfill(4) for x in record.winning_numbers)
        finally:
            gateway.rollback()
            gateway.close()
    except Exception:
        return None
    return None


def _build_locked_prediction(core: Any) -> Any:
    conn_str = get_sql_connection_string_from_env()
    gateway = core.SqlServerGateway(conn_str, autocommit=False, no_sql_write=True)
    gateway.connect()
    try:
        orchestrator = Step3AdaptiveOrchestrator(
            core=core,
            gateway=gateway,
            start_draw_no=SOURCE_DRAW_NO,
            end_draw_no=TARGET_DRAW_NO,
            training_window_size=64,
        )
        return orchestrator.predict_one_step_locked(SOURCE_DRAW_NO)
    finally:
        gateway.rollback()
        gateway.close()


def main() -> int:
    args = parse_args()
    load_env_file()
    core = import_step2_core()

    # FIREWALL CLARITY: lock prediction before loading completed actuals.
    locked = _build_locked_prediction(core)

    # POST-PREDICTION ONLY: completed actuals are loaded after locked Top5.
    actuals = _completed_actuals_from_sql(core) or ACTUAL_5498
    actual_source = "sql_drawhistory_or_fallback_completed_draw"

    rows: list[MemoryAttributionRow] = []
    report_lines: list[str] = [
        "E5 Draw 5498 Memory Learning Report",
        f"SourceDrawNo: {SOURCE_DRAW_NO}",
        f"TargetDrawNo: {TARGET_DRAW_NO}",
        f"PredictedTop5: {', '.join(locked.top5)}",
        f"ActualSource: {actual_source}",
        f"ActualCount: {len(actuals)}",
        f"ExactHitCount: {sum(1 for n in locked.top5 if n in set(actuals))}",
        "UseForFuturePrediction: false",
        f"WriteMemory: {bool(args.write_memory)}",
        "MemoryMode: observation_only",
        "",
        "LearnedHighMediumRows:",
    ]

    for candidate_row in locked.candidate_provenance:
        candidate = str(candidate_row["candidate_number"]).zfill(4)
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
                provenance["day_type"] = "Wednesday"
                provenance["memory_tier"] = tier
                provenance["selected_engine_source"] = candidate_row.get("selected_engine_source")
                provenance["final_rank"] = candidate_row.get("final_rank")

                for segment_class in selected_classes:
                    row = MemoryAttributionRow(
                        target_draw_no=TARGET_DRAW_NO,
                        candidate_number=candidate,
                        actual_number=str(actual).zfill(4),
                        segment_class=segment_class,
                        is_exact=segment_class == "EXACT4_MATCH",
                        provenance=provenance,
                    )
                    rows.append(row)

            report_lines.append(
                f"{tier}: {candidate} -> {actual}: {', '.join(selected_classes)} "
                f"| votes={len(votes)} | selected={candidate_row.get('selected_engine_source')}"
            )

    memory = load_memory(MEMORY_PATH)
    memory = update_memory(memory, rows, target_draw_no=TARGET_DRAW_NO)
    # This is observation file under reports/patches, not SQL or production registry.
    write_memory(MEMORY_PATH, memory, no_write=not args.write_memory)

    report_lines.extend(
        [
            "",
            f"AttributionRowsGenerated: {len(rows)}",
            f"MemoryEntryCount: {len(memory.get('entries', {}))}",
            f"MemoryPath: {MEMORY_PATH}",
            "FinalDecision: E5_MEMORY_LEARNS_5498_SEGMENT_SIGNALS_OBSERVATION_ONLY",
        ]
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"WROTE {REPORT_PATH}")
    if args.write_memory:
        print(f"WROTE {MEMORY_PATH}")
    else:
        print(f"MEMORY_NOT_WRITTEN {MEMORY_PATH}")
    print("E5_MEMORY_LEARNS_5498_SEGMENT_SIGNALS_OBSERVATION_ONLY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
