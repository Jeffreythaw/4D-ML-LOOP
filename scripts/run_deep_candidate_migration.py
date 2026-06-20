from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pyodbc
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
SQL_PATH = PROJECT_ROOT / "sql" / "step_153_deep_candidate_persistence_schema.sql"
REPORT_PATH = PROJECT_ROOT / "reports" / "step_153b_deep_candidate_migration_report.txt"
SNAPSHOT_PATH = (
    PROJECT_ROOT / "reports" / "step_153b_deep_candidate_migration_schema_snapshot.json"
)
TABLE_NAME = "dbo.DeepCandidateLedger"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))
load_dotenv(BACKEND_ROOT / ".env")

from app.core.config import get_settings


EXPECTED_COLUMNS = {
    "DeepCandidateId": {"type": "bigint", "max_length": 8, "nullable": False},
    "EngineSource": {"type": "nvarchar", "max_length": 200, "nullable": False},
    "EngineVersion": {"type": "nvarchar", "max_length": 200, "nullable": False},
    "Mode": {"type": "nvarchar", "max_length": 200, "nullable": False},
    "SourceDrawNo": {"type": "int", "max_length": 4, "nullable": False},
    "TargetDrawNo": {"type": "int", "max_length": 4, "nullable": False},
    "CandidateRank": {"type": "int", "max_length": 4, "nullable": False},
    "CandidateNumber": {"type": "char", "max_length": 4, "nullable": False},
    "CandidateScore": {"type": "float", "max_length": 8, "nullable": True},
    "CandidateFamily": {"type": "nvarchar", "max_length": 200, "nullable": True},
    "GenerationMethod": {"type": "nvarchar", "max_length": 400, "nullable": True},
    "FeatureJson": {"type": "nvarchar", "max_length": -1, "nullable": True},
    "CandidateBatchHash": {"type": "char", "max_length": 64, "nullable": False},
    "CandidateRowHash": {"type": "char", "max_length": 64, "nullable": False},
    "TemporalCutoffDrawNo": {"type": "int", "max_length": 4, "nullable": False},
    "TargetAvailableAtGeneration": {"type": "bit", "max_length": 1, "nullable": False},
    "VerificationStatus": {"type": "nvarchar", "max_length": 80, "nullable": False},
    "HitCount": {"type": "int", "max_length": 4, "nullable": True},
    "CreatedAtUtc": {"type": "datetime2", "max_length": 8, "nullable": False},
}

REQUIRED_CONSTRAINTS = {
    "PK_DeepCandidateLedger",
    "CK_DeepCandidateLedger_CandidateNumber",
    "CK_DeepCandidateLedger_CandidateRank",
    "CK_DeepCandidateLedger_TemporalCutoff",
    "UQ_DeepCandidateLedger_BatchRank",
    "UQ_DeepCandidateLedger_BatchNumber",
}

REQUIRED_INDEXES = {
    "IX_DeepCandidateLedger_SourceTarget": ["SourceDrawNo", "TargetDrawNo"],
    "IX_DeepCandidateLedger_EngineMode": ["EngineSource", "Mode"],
    "IX_DeepCandidateLedger_BatchHash": ["CandidateBatchHash"],
    "IX_DeepCandidateLedger_VerificationStatus": ["VerificationStatus"],
    "IX_DeepCandidateLedger_CandidateNumber": ["CandidateNumber"],
}


def get_conn():
    return pyodbc.connect(
        get_settings().sql_connection_string(),
        timeout=120,
        autocommit=False,
    )


def strip_sql_comments(sql_text: str) -> str:
    without_blocks = re.sub(r"/\*.*?\*/", " ", sql_text, flags=re.DOTALL)
    return re.sub(r"--[^\r\n]*", " ", without_blocks)


def guard_sql(sql_text: str) -> dict:
    active_sql = strip_sql_comments(sql_text)
    upper = active_sql.upper()
    violations = []
    forbidden_patterns = {
        "DROP_TABLE": r"\bDROP\s+TABLE\b",
        "TRUNCATE": r"\bTRUNCATE\b",
        "DELETE": r"\bDELETE\b",
        "UPDATE": r"\bUPDATE\b",
        "INSERT": r"\bINSERT\b",
        "MERGE": r"\bMERGE\b",
    }
    for label, pattern in forbidden_patterns.items():
        if re.search(pattern, upper):
            violations.append(label)

    for match in re.finditer(r"\bALTER\s+TABLE\b[^;]*;", upper, flags=re.DOTALL):
        statement = match.group(0)
        if " ADD CONSTRAINT " not in statement:
            violations.append("UNAPPROVED_ALTER_TABLE")

    create_tables = re.findall(r"\bCREATE\s+TABLE\s+([A-Z0-9_.\[\]]+)", upper)
    unexpected_tables = [
        table
        for table in create_tables
        if table.replace("[", "").replace("]", "") != "DBO.DEEPCANDIDATELEDGER"
    ]
    if unexpected_tables:
        violations.append(f"UNEXPECTED_CREATE_TABLE:{unexpected_tables}")

    return {
        "status": "PASS" if not violations else "FAIL",
        "violations": violations,
        "active_sql_contains_create_table": bool(create_tables),
        "commented_rollback_ignored": "DROP TABLE" in sql_text.upper()
        and "DROP_TABLE" not in violations,
    }


def table_exists(cursor) -> bool:
    row = cursor.execute(
        """
        SELECT CASE
            WHEN OBJECT_ID(?, 'U') IS NULL THEN 0
            ELSE 1
        END AS TableExists;
        """,
        TABLE_NAME,
    ).fetchone()
    return bool(row and int(row.TableExists))


def fetch_columns(cursor) -> list[dict]:
    rows = cursor.execute(
        """
        SELECT
            c.column_id,
            c.name AS ColumnName,
            t.name AS DataType,
            c.max_length,
            c.is_nullable,
            c.is_identity
        FROM sys.columns c
        JOIN sys.types t ON t.user_type_id = c.user_type_id
        WHERE c.object_id = OBJECT_ID(?)
        ORDER BY c.column_id;
        """,
        TABLE_NAME,
    ).fetchall()
    return [
        {
            "column_id": int(row.column_id),
            "name": str(row.ColumnName),
            "data_type": str(row.DataType),
            "max_length": int(row.max_length),
            "nullable": bool(row.is_nullable),
            "identity": bool(row.is_identity),
        }
        for row in rows
    ]


def fetch_constraints(cursor) -> list[dict]:
    rows = cursor.execute(
        """
        SELECT
            kc.name AS ConstraintName,
            kc.type_desc AS ConstraintType,
            CAST(NULL AS nvarchar(max)) AS Definition
        FROM sys.key_constraints kc
        WHERE kc.parent_object_id = OBJECT_ID(?)

        UNION ALL

        SELECT
            cc.name AS ConstraintName,
            'CHECK_CONSTRAINT' AS ConstraintType,
            cc.definition AS Definition
        FROM sys.check_constraints cc
        WHERE cc.parent_object_id = OBJECT_ID(?)

        UNION ALL

        SELECT
            dc.name AS ConstraintName,
            'DEFAULT_CONSTRAINT' AS ConstraintType,
            dc.definition AS Definition
        FROM sys.default_constraints dc
        WHERE dc.parent_object_id = OBJECT_ID(?)

        ORDER BY ConstraintType, ConstraintName;
        """,
        TABLE_NAME,
        TABLE_NAME,
        TABLE_NAME,
    ).fetchall()
    return [
        {
            "name": str(row.ConstraintName),
            "type": str(row.ConstraintType),
            "definition": str(row.Definition) if row.Definition is not None else None,
        }
        for row in rows
    ]


def fetch_indexes(cursor) -> list[dict]:
    rows = cursor.execute(
        """
        SELECT
            i.name AS IndexName,
            i.is_unique,
            i.is_primary_key,
            i.is_unique_constraint,
            ic.key_ordinal,
            c.name AS ColumnName
        FROM sys.indexes i
        JOIN sys.index_columns ic
          ON ic.object_id = i.object_id
         AND ic.index_id = i.index_id
        JOIN sys.columns c
          ON c.object_id = ic.object_id
         AND c.column_id = ic.column_id
        WHERE i.object_id = OBJECT_ID(?)
          AND i.name IS NOT NULL
          AND ic.is_included_column = 0
        ORDER BY i.name, ic.key_ordinal;
        """,
        TABLE_NAME,
    ).fetchall()
    grouped: dict[str, dict] = {}
    for row in rows:
        name = str(row.IndexName)
        item = grouped.setdefault(
            name,
            {
                "name": name,
                "unique": bool(row.is_unique),
                "primary_key": bool(row.is_primary_key),
                "unique_constraint": bool(row.is_unique_constraint),
                "columns": [],
            },
        )
        item["columns"].append(str(row.ColumnName))
    return [grouped[name] for name in sorted(grouped)]


def row_count(cursor) -> int:
    row = cursor.execute(
        """
        SELECT COUNT_BIG(*) AS TotalRows
        FROM dbo.DeepCandidateLedger;
        """
    ).fetchone()
    return int(row.TotalRows)


def normalize_definition(value: str | None) -> str:
    return re.sub(r"[\s\[\]\(\)]", "", str(value or "").upper())


def verify_schema(cursor) -> dict:
    exists = table_exists(cursor)
    columns = fetch_columns(cursor) if exists else []
    constraints = fetch_constraints(cursor) if exists else []
    indexes = fetch_indexes(cursor) if exists else []
    count = row_count(cursor) if exists else None

    columns_by_name = {item["name"]: item for item in columns}
    column_failures = []
    for name, expected in EXPECTED_COLUMNS.items():
        actual = columns_by_name.get(name)
        if actual is None:
            column_failures.append(f"MISSING_COLUMN:{name}")
            continue
        for actual_key, expected_key in (
            ("data_type", "type"),
            ("max_length", "max_length"),
            ("nullable", "nullable"),
        ):
            if actual[actual_key] != expected[expected_key]:
                column_failures.append(
                    f"COLUMN_MISMATCH:{name}:{actual_key}:"
                    f"expected={expected[expected_key]}:actual={actual[actual_key]}"
                )
    identity_ok = bool(
        columns_by_name.get("DeepCandidateId", {}).get("identity")
    )
    if not identity_ok:
        column_failures.append("DEEPCANDIDATEID_NOT_IDENTITY")

    constraint_names = {item["name"] for item in constraints}
    constraint_failures = sorted(REQUIRED_CONSTRAINTS - constraint_names)
    constraint_by_name = {item["name"]: item for item in constraints}
    definition_expectations = {
        "CK_DeepCandidateLedger_CandidateRank": (
            "CANDIDATERANK>=1",
            "CANDIDATERANK<=50",
        ),
        "CK_DeepCandidateLedger_CandidateNumber": (
            "LENCANDIDATENUMBER=4",
            "NOTCANDIDATENUMBERLIKE",
            "0-9",
        ),
        "CK_DeepCandidateLedger_TemporalCutoff": (
            "TEMPORALCUTOFFDRAWNO=SOURCEDRAWNO",
        ),
    }
    for name, fragments in definition_expectations.items():
        normalized = normalize_definition(
            constraint_by_name.get(name, {}).get("definition")
        )
        if name in constraint_names and not all(
            fragment in normalized for fragment in fragments
        ):
            constraint_failures.append(f"DEFINITION_MISMATCH:{name}:{normalized}")

    indexes_by_name = {item["name"]: item for item in indexes}
    index_failures = []
    for name, expected_columns in REQUIRED_INDEXES.items():
        actual = indexes_by_name.get(name)
        if actual is None:
            index_failures.append(f"MISSING_INDEX:{name}")
        elif actual["columns"] != expected_columns:
            index_failures.append(
                f"INDEX_COLUMN_MISMATCH:{name}:"
                f"expected={expected_columns}:actual={actual['columns']}"
            )

    primary_key_ok = any(
        item["name"] == "PK_DeepCandidateLedger"
        and item["type"] == "PRIMARY_KEY_CONSTRAINT"
        for item in constraints
    )
    unique_rank_ok = any(
        item["name"] == "UQ_DeepCandidateLedger_BatchRank"
        and item["type"] == "UNIQUE_CONSTRAINT"
        for item in constraints
    )
    unique_number_ok = any(
        item["name"] == "UQ_DeepCandidateLedger_BatchNumber"
        and item["type"] == "UNIQUE_CONSTRAINT"
        for item in constraints
    )
    if not primary_key_ok:
        constraint_failures.append("PRIMARY_KEY_NOT_VERIFIED")
    if not unique_rank_ok:
        constraint_failures.append("UNIQUE_BATCH_RANK_NOT_VERIFIED")
    if not unique_number_ok:
        constraint_failures.append("UNIQUE_BATCH_NUMBER_NOT_VERIFIED")

    return {
        "table_exists": exists,
        "columns": columns,
        "constraints": constraints,
        "indexes": indexes,
        "row_count": count,
        "columns_status": "PASS" if not column_failures else "FAIL",
        "column_failures": column_failures,
        "constraints_status": "PASS" if not constraint_failures else "FAIL",
        "constraint_failures": sorted(set(constraint_failures)),
        "indexes_status": "PASS" if not index_failures else "FAIL",
        "index_failures": index_failures,
        "row_count_status": (
            "PASS"
            if count == 0
            else "UNEXPECTED_ROWS_PRESENT"
            if count is not None
            else "FAIL"
        ),
        "schema_valid": (
            exists
            and not column_failures
            and not constraint_failures
            and not index_failures
            and count == 0
        ),
    }


def build_snapshot(
    *,
    table_before: bool,
    table_after: bool,
    execution_status: str,
    transaction_status: str,
    guard: dict,
    verification: dict,
    db_write_performed: bool,
    error: str | None,
) -> dict:
    return {
        "table_exists_before": table_before,
        "table_exists_after": table_after,
        "migration_execution_status": execution_status,
        "transaction_status": transaction_status,
        "columns": verification.get("columns", []),
        "constraints": verification.get("constraints", []),
        "indexes": verification.get("indexes", []),
        "row_count_after": verification.get("row_count"),
        "columns_status": verification.get("columns_status", "FAIL"),
        "constraints_status": verification.get("constraints_status", "FAIL"),
        "indexes_status": verification.get("indexes_status", "FAIL"),
        "row_count_status": verification.get("row_count_status", "FAIL"),
        "verification_failures": {
            "columns": verification.get("column_failures", []),
            "constraints": verification.get("constraint_failures", []),
            "indexes": verification.get("index_failures", []),
        },
        "destructive_sql_guard_status": guard["status"],
        "destructive_sql_guard_details": guard,
        "db_write_performed": db_write_performed,
        "data_write_performed": False,
        "production_changed": False,
        "error": error,
    }


def build_report(snapshot: dict) -> str:
    width = 144
    ready = (
        snapshot["table_exists_after"]
        and snapshot["columns_status"] == "PASS"
        and snapshot["constraints_status"] == "PASS"
        and snapshot["indexes_status"] == "PASS"
        and snapshot["row_count_status"] == "PASS"
        and snapshot["destructive_sql_guard_status"] == "PASS"
    )
    lines = [
        "=" * width,
        "STEP 153B — DEEPCANDIDATELEDGER MIGRATION EXECUTION",
        "=" * width,
        "ProductionMathChanged: NO",
        "APIChanged: NO",
        "FrontendChanged: NO",
        "DeploymentChanged: NO",
        "PredictionRowsWritten: NO",
        "DataRowsInserted: NO",
        "",
        "MIGRATION STATUS",
        "-" * width,
        f"TableExistedBefore: {'YES' if snapshot['table_exists_before'] else 'NO'}",
        f"TableExistsAfter: {'YES' if snapshot['table_exists_after'] else 'NO'}",
        f"MigrationExecutionStatus: {snapshot['migration_execution_status']}",
        f"TransactionStatus: {snapshot['transaction_status']}",
        f"DBWritePerformed: {'YES' if snapshot['db_write_performed'] else 'NO'}",
        f"DestructiveSQLGuard: {snapshot['destructive_sql_guard_status']}",
        f"DestructiveSQLGuardDetails: {snapshot['destructive_sql_guard_details']}",
        "",
        "SCHEMA VERIFICATION",
        "-" * width,
        f"RequiredColumns: {snapshot['columns_status']}",
        f"RequiredConstraints: {snapshot['constraints_status']}",
        f"RequiredIndexes: {snapshot['indexes_status']}",
        f"RowCountAfter: {snapshot['row_count_after']}",
        f"RowCountVerification: {snapshot['row_count_status']}",
        f"VerificationFailures: {snapshot['verification_failures']}",
        "",
        "SAFETY VERIFICATION",
        "-" * width,
        "PredictionWritesPerformed: NO",
        "CandidateRowsInserted: NO",
        "ProductionPathChanges: NO",
        "APIChanges: NO",
        "FrontendChanges: NO",
        "DeploymentChanges: NO",
        f"DataWritePerformed: {'YES' if snapshot['data_write_performed'] else 'NO'}",
        f"Error: {snapshot['error'] or 'NONE'}",
        "",
        "FINAL RECOMMENDATION",
        "-" * width,
        f"MigrationReadyForCandidatePersistence: {'YES' if ready else 'NO'}",
        "NextStep: Step 153C generate forward Top50 candidate batches",
        "",
        f"REPORT_WRITTEN: {REPORT_PATH}",
        f"SNAPSHOT_WRITTEN: {SNAPSHOT_PATH}",
    ]
    return "\n".join(lines)


def write_outputs(snapshot: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = build_report(snapshot)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    print(report)


def main() -> None:
    sql_text = SQL_PATH.read_text(encoding="utf-8")
    guard = guard_sql(sql_text)
    if guard["status"] != "PASS":
        snapshot = build_snapshot(
            table_before=False,
            table_after=False,
            execution_status="REJECTED_BY_STATIC_GUARD",
            transaction_status="NOT_STARTED",
            guard=guard,
            verification={},
            db_write_performed=False,
            error=f"Static SQL guard violations: {guard['violations']}",
        )
        write_outputs(snapshot)
        raise RuntimeError(snapshot["error"])

    connection = get_conn()
    table_before = False
    db_write_performed = False
    execution_status = "NOT_STARTED"
    transaction_status = "NOT_STARTED"
    verification: dict = {}
    error: str | None = None
    try:
        cursor = connection.cursor()
        table_before = table_exists(cursor)
        if table_before:
            verification = verify_schema(cursor)
            if not verification["schema_valid"]:
                execution_status = "ALREADY_EXISTS_SCHEMA_MISMATCH"
                transaction_status = "ROLLED_BACK"
                connection.rollback()
                raise RuntimeError(
                    f"Existing {TABLE_NAME} does not match approved design: "
                    f"{verification}"
                )
            execution_status = "ALREADY_EXISTS_VERIFIED"
            transaction_status = "NO_SCHEMA_EXECUTION_REQUIRED"
            connection.rollback()
        else:
            execution_status = "EXECUTING"
            cursor.execute(sql_text)
            while cursor.nextset():
                pass
            db_write_performed = True
            verification = verify_schema(cursor)
            if not verification["schema_valid"]:
                execution_status = "POST_VERIFICATION_FAILED"
                transaction_status = "ROLLED_BACK"
                connection.rollback()
                db_write_performed = False
                raise RuntimeError(
                    f"Post-migration verification failed: {verification}"
                )
            connection.commit()
            execution_status = "EXECUTED"
            transaction_status = "COMMITTED"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        try:
            connection.rollback()
        except Exception:
            pass
        if transaction_status == "NOT_STARTED":
            transaction_status = "ROLLED_BACK"
        if execution_status in {"NOT_STARTED", "EXECUTING"}:
            execution_status = "FAILED"
    finally:
        connection.close()

    with get_conn() as check_connection:
        check_cursor = check_connection.cursor()
        table_after = table_exists(check_cursor)
        verification_after = verify_schema(check_cursor) if table_after else {}
        check_connection.rollback()

    snapshot = build_snapshot(
        table_before=table_before,
        table_after=table_after,
        execution_status=execution_status,
        transaction_status=transaction_status,
        guard=guard,
        verification=verification_after,
        db_write_performed=db_write_performed and execution_status == "EXECUTED",
        error=error,
    )
    write_outputs(snapshot)
    if error:
        raise RuntimeError(error)


if __name__ == "__main__":
    main()
