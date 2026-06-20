from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyodbc
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
SCHEMA_PATH = PROJECT_ROOT / "sql" / "step_153_deep_candidate_persistence_schema.sql"
REPORT_PATH = PROJECT_ROOT / "reports" / "step_153_deep_candidate_persistence_report.txt"
PLAN_PATH = PROJECT_ROOT / "reports" / "step_153_deep_candidate_persistence_plan.json"
PREVIEW_PATH = (
    PROJECT_ROOT / "reports" / "step_153_deep_candidate_persistence_preview_rows.jsonl"
)

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))
load_dotenv(BACKEND_ROOT / ".env")

from app.core.config import get_settings


PROPOSED_TABLE = "dbo.DeepCandidateLedger"
ENGINE_VERSION = "PERSISTED_LEDGER_V1"
DESIRED_DEPTH = 50
LIVE_SAMPLE_SOURCES = (5494, 5495, 5496, 5497)
REQUIRED_COLUMNS = (
    "DeepCandidateId",
    "EngineSource",
    "EngineVersion",
    "Mode",
    "SourceDrawNo",
    "TargetDrawNo",
    "CandidateRank",
    "CandidateNumber",
    "CandidateScore",
    "CandidateFamily",
    "GenerationMethod",
    "FeatureJson",
    "CandidateBatchHash",
    "CandidateRowHash",
    "TemporalCutoffDrawNo",
    "TargetAvailableAtGeneration",
    "VerificationStatus",
    "HitCount",
    "CreatedAtUtc",
)


def get_conn():
    return pyodbc.connect(get_settings().sql_connection_string(), timeout=120)


def z4(value: str | int) -> str:
    return str(value).strip().zfill(4)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


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


def current_discovery(cursor) -> dict:
    table_row = cursor.execute(
        """
        SELECT CASE
            WHEN OBJECT_ID(?, 'U') IS NULL THEN 0
            ELSE 1
        END AS TableExists;
        """,
        PROPOSED_TABLE,
    ).fetchone()
    depth_row = cursor.execute(
        """
        SELECT ISNULL(MAX(RankNo), 0) AS MaxRank
        FROM dbo.PredictionLedger;
        """
    ).fetchone()
    engines = [
        str(row.EngineSource)
        for row in cursor.execute(
            """
            SELECT DISTINCT EngineSource
            FROM dbo.PredictionLedger
            WHERE EngineSource IS NOT NULL
            ORDER BY EngineSource;
            """
        ).fetchall()
    ]
    modes = [
        str(row.Mode)
        for row in cursor.execute(
            """
            SELECT DISTINCT Mode
            FROM dbo.PredictionLedger
            ORDER BY Mode;
            """
        ).fetchall()
    ]
    return {
        "table_exists": bool(table_row and int(table_row.TableExists)),
        "deep_table_columns": table_columns(cursor, PROPOSED_TABLE),
        "predictionledger_columns": table_columns(cursor, "dbo.PredictionLedger"),
        "current_max_persisted_depth": int(depth_row.MaxRank or 0),
        "available_engines": engines,
        "available_modes": modes,
    }


def evenly_spaced(values: list[int], count: int) -> list[int]:
    if len(values) <= count:
        return values
    indexes = {
        round(index * (len(values) - 1) / (count - 1))
        for index in range(count)
    }
    return [values[index] for index in sorted(indexes)]


def select_preview_sources(cursor) -> list[int]:
    historical = [
        int(row.SourceDrawNo)
        for row in cursor.execute(
            """
            SELECT DISTINCT p.SourceDrawNo
            FROM dbo.PredictionLedger p
            WHERE p.SourceDrawNo >= 4050
            ORDER BY p.SourceDrawNo;
            """
        ).fetchall()
    ]
    draw_sources = {
        int(row.DrawNo)
        for row in cursor.execute(
            """
            SELECT DrawNo
            FROM dbo.DrawHistory
            WHERE DrawNo IN (?, ?, ?, ?);
            """,
            *LIVE_SAMPLE_SOURCES,
        ).fetchall()
    }
    selected = set(evenly_spaced(historical, 20))
    selected.update(draw_sources)
    return sorted(selected)


def fetch_preview_ledger(cursor, sources: list[int]) -> list[dict]:
    if not sources:
        return []
    source_min = min(sources)
    source_max = max(sources)
    source_set = set(sources)
    rows = cursor.execute(
        """
        SELECT
            EngineSource,
            Mode,
            SourceDrawNo,
            TargetDrawNo,
            RankNo,
            PredictedNumber,
            Score
        FROM dbo.PredictionLedger
        WHERE SourceDrawNo BETWEEN ? AND ?
        ORDER BY SourceDrawNo, TargetDrawNo, EngineSource, Mode, RankNo;
        """,
        source_min,
        source_max,
    ).fetchall()
    return [
        {
            "engine_source": str(row.EngineSource or "UNKNOWN"),
            "mode": str(row.Mode),
            "source_draw_no": int(row.SourceDrawNo),
            "target_draw_no": int(row.TargetDrawNo),
            "candidate_rank": int(row.RankNo),
            "candidate_number": z4(row.PredictedNumber),
            "candidate_score": float(row.Score) if row.Score is not None else None,
        }
        for row in rows
        if int(row.SourceDrawNo) in source_set
    ]


def target_availability(cursor, sources: list[int]) -> dict[int, bool]:
    if not sources:
        return {}
    targets = {source + 1 for source in sources}
    rows = cursor.execute(
        """
        SELECT DrawNo
        FROM dbo.DrawHistory
        WHERE DrawNo BETWEEN ? AND ?;
        """,
        min(targets),
        max(targets),
    ).fetchall()
    available = {int(row.DrawNo) for row in rows}
    return {source: source + 1 in available for source in sources}


def base_preview_row(
    *,
    row_type: str,
    engine_source: str | None,
    mode: str | None,
    source_draw_no: int | None,
    target_draw_no: int | None,
    candidate_rank: int | None,
    candidate_number: str | None,
    candidate_score: float | None,
    candidate_family: str | None,
    generation_method: str | None,
    feature_json: dict | None,
    batch_hash: str | None,
    target_available: bool | None,
    sql_ready: bool,
    limitation: str | None,
) -> dict:
    row_core = {
        "row_type": row_type,
        "engine_source": engine_source,
        "engine_version": ENGINE_VERSION if engine_source else None,
        "mode": mode,
        "source_draw_no": source_draw_no,
        "target_draw_no": target_draw_no,
        "candidate_rank": candidate_rank,
        "candidate_number": candidate_number,
        "candidate_score": candidate_score,
        "candidate_family": candidate_family,
        "generation_method": generation_method,
        "feature_json": feature_json,
        "candidate_batch_hash": batch_hash,
        "temporal_cutoff_draw_no": source_draw_no,
        "target_available_at_generation": target_available,
        "sql_ready": sql_ready,
        "limitation": limitation,
    }
    row_core["candidate_row_hash"] = sha256(row_core)
    return row_core


def build_preview_rows(
    discovery: dict,
    sources: list[int],
    ledger_rows: list[dict],
    target_available: dict[int, bool],
) -> list[dict]:
    rows = [
        base_preview_row(
            row_type="schema_check",
            engine_source=None,
            mode=None,
            source_draw_no=None,
            target_draw_no=None,
            candidate_rank=None,
            candidate_number=None,
            candidate_score=None,
            candidate_family=None,
            generation_method="READ_ONLY_SCHEMA_DISCOVERY",
            feature_json={
                "proposed_table": PROPOSED_TABLE,
                "table_exists": discovery["table_exists"],
                "current_max_persisted_depth": discovery[
                    "current_max_persisted_depth"
                ],
            },
            batch_hash=None,
            target_available=None,
            sql_ready=False,
            limitation="SCHEMA_DESIGN_ONLY_NO_DDL_EXECUTED",
        )
    ]
    grouped: dict[tuple[str, str, int, int], list[dict]] = defaultdict(list)
    for item in ledger_rows:
        grouped[
            (
                item["engine_source"],
                item["mode"],
                item["source_draw_no"],
                item["target_draw_no"],
            )
        ].append(item)

    for key, persisted in sorted(grouped.items()):
        engine_source, mode, source, target = key
        by_rank = {item["candidate_rank"]: item for item in persisted}
        batch_candidates = []
        for rank in range(1, DESIRED_DEPTH + 1):
            item = by_rank.get(rank)
            batch_candidates.append(
                {
                    "candidate_rank": rank,
                    "candidate_number": item["candidate_number"] if item else None,
                    "candidate_score": item["candidate_score"] if item else None,
                    "source_type": (
                        "PERSISTED_LEDGER_TOP5"
                        if item is not None and rank <= 5
                        else "UNAVAILABLE_DEPTH_PLACEHOLDER"
                    ),
                }
            )
        batch_payload = {
            "engine_source": engine_source,
            "engine_version": ENGINE_VERSION,
            "mode": mode,
            "source_draw_no": source,
            "target_draw_no": target,
            "temporal_cutoff_draw_no": source,
            "desired_depth": DESIRED_DEPTH,
            "candidates": batch_candidates,
        }
        batch_hash = sha256(batch_payload)
        for candidate in batch_candidates:
            rank = candidate["candidate_rank"]
            number = candidate["candidate_number"]
            source_type = candidate["source_type"]
            available = target_available.get(source, False)
            if number is not None:
                rows.append(
                    base_preview_row(
                        row_type="candidate_preview",
                        engine_source=engine_source,
                        mode=mode,
                        source_draw_no=source,
                        target_draw_no=target,
                        candidate_rank=rank,
                        candidate_number=number,
                        candidate_score=candidate["candidate_score"],
                        candidate_family=engine_source,
                        generation_method="COPIED_FROM_PREDICTION_LEDGER_LOCKED_RANK",
                        feature_json={
                            "candidate_source_type": source_type,
                            "persisted_rank_locked": True,
                            "desired_batch_depth": DESIRED_DEPTH,
                        },
                        batch_hash=batch_hash,
                        target_available=available,
                        sql_ready=True,
                        limitation=None,
                    )
                )
            else:
                rows.append(
                    base_preview_row(
                        row_type="depth_unavailable",
                        engine_source=engine_source,
                        mode=mode,
                        source_draw_no=source,
                        target_draw_no=target,
                        candidate_rank=rank,
                        candidate_number=None,
                        candidate_score=None,
                        candidate_family=None,
                        generation_method="UNAVAILABLE_DEPTH_PLACEHOLDER",
                        feature_json={
                            "candidate_source_type": source_type,
                            "desired_batch_depth": DESIRED_DEPTH,
                        },
                        batch_hash=batch_hash,
                        target_available=available,
                        sql_ready=False,
                        limitation=(
                            f"DEPTH_UNAVAILABLE_RANK_{rank};"
                            f"CURRENT_PERSISTED_MAX_DEPTH={discovery['current_max_persisted_depth']}"
                        ),
                    )
                )
    return rows


def build_report(plan: dict, discovery: dict, rows: list[dict]) -> str:
    width = 148
    counts = Counter(row["row_type"] for row in rows)
    source_counts = Counter(
        (
            row.get("feature_json") or {}
        ).get("candidate_source_type", "SCHEMA_CHECK")
        for row in rows
    )
    for source_type in (
        "PERSISTED_LEDGER_TOP5",
        "SAFE_RECONSTRUCTED_PREVIEW",
        "UNAVAILABLE_DEPTH_PLACEHOLDER",
    ):
        source_counts.setdefault(source_type, 0)
    lines = [
        "=" * width,
        "STEP 153 — DEEP CANDIDATE PERSISTENCE LAYER — DESIGN + DRY RUN ONLY",
        "=" * width,
        "ProductionMathChanged: NO",
        "APIChanged: NO",
        "FrontendChanged: NO",
        "SQLSchemaChangedLive: NO",
        "DeploymentChanged: NO",
        "DBWritePerformed: NO",
        "",
        "STEP 152 MOTIVATION",
        "-" * width,
        "MaximumPersistedCandidateDepth: 5",
        "Top10Top15Top20Status: UNAVAILABLE",
        "MainBottleneck: Data sparsity",
        "RecommendedResponse: design a hash-locked deeper candidate persistence layer.",
        "",
        "PROPOSED SCHEMA SUMMARY",
        "-" * width,
        f"ProposedTable: {PROPOSED_TABLE}",
        f"SchemaFile: {SCHEMA_PATH}",
        f"Columns: {list(REQUIRED_COLUMNS)}",
        "Constraints: numeric four-digit candidate; rank 1-50; temporal cutoff equals source; unique batch rank; unique batch candidate number.",
        "Indexes: Source/Target, Engine/Mode, CandidateBatchHash, VerificationStatus, CandidateNumber.",
        "Idempotency: table and indexes are guarded by existence checks.",
        "RollbackNote: commented manual DROP TABLE section is included.",
        "",
        "CURRENT DB DISCOVERY",
        "-" * width,
        f"DeepCandidateLedgerExists: {'YES' if discovery['table_exists'] else 'NO'}",
        f"PredictionLedgerCurrentMaxDepth: {discovery['current_max_persisted_depth']}",
        f"PredictionLedgerColumns: {[item['name'] for item in discovery['predictionledger_columns']]}",
        f"AvailableEngines: {discovery['available_engines']}",
        f"AvailableModes: {discovery['available_modes']}",
        "",
        "DRY-RUN PREVIEW SUMMARY",
        "-" * width,
        f"PreviewSourceDraws: {plan['preview_source_draws']}",
        f"PreviewSourceDrawCount: {len(plan['preview_source_draws'])}",
        f"PreviewRowsTotal: {len(rows)}",
        f"CandidatePreviewRows: {counts['candidate_preview']}",
        f"UnavailableDepthPlaceholders: {counts['depth_unavailable']}",
        f"SchemaCheckRows: {counts['schema_check']}",
        f"SQLReadyRows: {plan['sql_ready_row_count']}",
        f"CandidateSourceBreakdown: {dict(source_counts)}",
        "",
        "TEMPORAL FIREWALL CHECKS",
        "-" * width,
        "CandidateGenerationCutoff: TemporalCutoffDrawNo equals SourceDrawNo.",
        "TargetWinnersUsed: NO",
        "TargetAvailabilityUsage: metadata only; no WinningNumbers were queried.",
        "PersistedRankOrderTreatment: locked and copied without reranking.",
        "CandidateBatchHashGenerated: YES",
        "CandidateRowHashGenerated: YES",
        "HashEncoding: canonical JSON, sorted keys, compact separators, UTF-8, SHA256.",
        "",
        "LIMITATIONS",
        "-" * width,
        "DEPTH_UNAVAILABLE: Current persisted engine depth stops at rank 5.",
        "Ranks 6-50 are null preview placeholders and are not SQL-ready.",
        "SAFE_RECONSTRUCTED_PREVIEW count is zero because no globally locked, reproducible deeper engine ranking exists.",
        "This dry run does not claim that Top20 or Top50 candidates currently exist.",
        "",
        "FINAL RECOMMENDATION",
        "-" * width,
        "ExecuteMigrationNow: NO",
        "ProductionSwitchRecommendedNow: NO",
        "NextStep: Human review schema, then Step 153B migration execution if approved",
        "",
        f"REPORT_WRITTEN: {REPORT_PATH}",
        f"PLAN_WRITTEN: {PLAN_PATH}",
        f"PREVIEW_WRITTEN: {PREVIEW_PATH}",
    ]
    return "\n".join(lines)


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_conn() as connection:
        cursor = connection.cursor()
        discovery = current_discovery(cursor)
        sources = select_preview_sources(cursor)
        ledger_rows = fetch_preview_ledger(cursor, sources)
        target_available = target_availability(cursor, sources)

    preview_rows = build_preview_rows(
        discovery,
        sources,
        ledger_rows,
        target_available,
    )
    counts = Counter(row["row_type"] for row in preview_rows)
    source_counts = Counter(
        (
            row.get("feature_json") or {}
        ).get("candidate_source_type", "SCHEMA_CHECK")
        for row in preview_rows
    )
    for source_type in (
        "PERSISTED_LEDGER_TOP5",
        "SAFE_RECONSTRUCTED_PREVIEW",
        "UNAVAILABLE_DEPTH_PLACEHOLDER",
    ):
        source_counts.setdefault(source_type, 0)
    plan = {
        "schema_file": str(SCHEMA_PATH),
        "proposed_table": PROPOSED_TABLE,
        "table_exists": discovery["table_exists"],
        "current_max_persisted_depth": discovery["current_max_persisted_depth"],
        "preview_source_draws": sources,
        "candidate_counts_by_source_type": dict(source_counts),
        "preview_row_counts": dict(counts),
        "sql_ready_row_count": sum(row["sql_ready"] for row in preview_rows),
        "unavailable_depth_row_count": counts["depth_unavailable"],
        "migration_required": not discovery["table_exists"],
        "production_write_performed": False,
        "db_write_performed": False,
        "next_steps": [
            "Human review the idempotent migration SQL.",
            "Approve or revise engine-version and batch-lock conventions.",
            "If approved, execute migration separately in Step 153B.",
            "Update candidate-generating engines to emit locked Top20/Top50 before verification.",
        ],
    }

    with PREVIEW_PATH.open("w", encoding="utf-8") as handle:
        for row in preview_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    PLAN_PATH.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = build_report(plan, discovery, preview_rows)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
