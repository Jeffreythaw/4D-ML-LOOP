# migrate_raw_to_sql.py
# ============================================================
# JEFFREY QUAD-ENGINE HYBRID V2
# RAW HISTORY -> SQL SERVER MIGRATION LOADER
#
# Purpose:
#   - Load .env configuration
#   - Read local historical 4D draw logs
#   - Normalize into dbo.DrawHistory:
#       DrawNo, DrawDate, DayType, WinningNumbers
#   - Bulk insert into SQL Server
#   - Verify Phase 1 and Phase 2 row counts
#
# Safety:
#   - Does NOT run Step 2/3 inference.
#   - Does NOT touch FormulaRegistry except trigger side-effects from DrawHistory insert.
#   - Uses dbo.DrawHistory as approved SQL layer target.
# ============================================================

from __future__ import annotations

import csv
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pandas as pd
except ImportError as exc:
    raise ImportError("pandas is required. Install with: pip install pandas") from exc


PROJECT_ROOT = Path(__file__).resolve().parent

ENV_SQL_CONN_KEY = "J4D_SQL_CONN_STR"
ENV_DATA_PATH_KEY = "DATA_PATH"

PHASE1_MAX_DRAW_NO = 4050
PHASE2_MIN_DRAW_NO = 4051

VALID_DAY_TYPES = {"Wednesday", "Saturday", "Sunday", "Special"}

TARGET_TABLE = "dbo.DrawHistory"


@dataclass(frozen=True)
class NormalizedDraw:
    draw_no: int
    draw_date: date
    day_type: str
    winning_numbers: str


def print_header(title: str) -> None:
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


def print_kv(key: str, value: Any) -> None:
    print(f"{key}: {value}")


def load_env() -> None:
    env_path = PROJECT_ROOT / ".env"

    if not env_path.exists():
        raise FileNotFoundError(f"Missing .env file: {env_path}")

    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        print(
            "[WARN] python-dotenv not installed. Existing shell env will be used only.",
            file=sys.stderr,
        )
        return

    loaded = load_dotenv(env_path, override=False)

    if not loaded:
        raise RuntimeError(f"Failed to load .env: {env_path}")


def get_sql_connection_string() -> str:
    """
    Priority:
      1. DB_DRIVER + DB_SERVER + DB_DATABASE + DB_USERNAME + DB_PASSWORD/DB_PASS/DB_PWD
      2. J4D_SQL_CONN_STR fallback

    This avoids stale J4D_SQL_CONN_STR silently overriding DB_* config.
    """

    db_driver = os.getenv("DB_DRIVER", "").strip()
    db_server = os.getenv("DB_SERVER", "").strip()
    db_database = os.getenv("DB_DATABASE", "").strip()
    db_username = os.getenv("DB_USERNAME", "").strip()
    db_password = (
        os.getenv("DB_PASSWORD", "").strip()
        or os.getenv("DB_PASS", "").strip()
        or os.getenv("DB_PWD", "").strip()
    )

    if all([db_driver, db_server, db_database, db_username, db_password]):
        return (
            f"DRIVER={{{db_driver}}};"
            f"SERVER={db_server};"
            f"DATABASE={db_database};"
            f"UID={db_username};"
            f"PWD={db_password};"
            "TrustServerCertificate=yes;"
        )

    fallback = os.getenv(ENV_SQL_CONN_KEY, "").strip()

    if fallback:
        return fallback

    missing = []
    if not db_driver:
        missing.append("DB_DRIVER")
    if not db_server:
        missing.append("DB_SERVER")
    if not db_database:
        missing.append("DB_DATABASE")
    if not db_username:
        missing.append("DB_USERNAME")
    if not db_password:
        missing.append("DB_PASSWORD or DB_PASS or DB_PWD")

    raise EnvironmentError(
        "SQL connection config missing. "
        f"Missing DB_* keys: {missing}. "
        f"Alternatively provide {ENV_SQL_CONN_KEY}."
    )


def connect_sql(conn_str: str):
    try:
        import pyodbc  # type: ignore
    except ImportError as exc:
        raise ImportError("pyodbc is required. Install with: pip install pyodbc") from exc

    conn = pyodbc.connect(conn_str, autocommit=False, timeout=30)
    return conn


def normalize_col_name(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def find_first_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    normalized = {normalize_col_name(c): c for c in columns}

    for cand in candidates:
        key = normalize_col_name(cand)
        if key in normalized:
            return normalized[key]

    for col in columns:
        n = normalize_col_name(col)
        for cand in candidates:
            c = normalize_col_name(cand)
            if c and c in n:
                return col

    return None


def parse_draw_no(value: Any) -> int:
    if pd.isna(value):
        raise ValueError("DrawNo is empty")

    text = str(value).strip()
    match = re.search(r"\d+", text)

    if not match:
        raise ValueError(f"Invalid DrawNo value: {value!r}")

    return int(match.group(0))


def parse_draw_date(value: Any) -> date:
    if pd.isna(value):
        raise ValueError("DrawDate is empty")

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    text = str(value).strip()

    if not text:
        raise ValueError("DrawDate is empty")

    parsed = pd.to_datetime(text, errors="coerce", dayfirst=False)

    if pd.isna(parsed):
        parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)

    if pd.isna(parsed):
        raise ValueError(f"Invalid DrawDate value: {value!r}")

    return parsed.date()


def compute_day_type(draw_date: date, explicit_day_type: Optional[Any] = None) -> str:
    if explicit_day_type is not None and not pd.isna(explicit_day_type):
        raw = str(explicit_day_type).strip()

        if raw:
            lowered = raw.lower()

            if "wed" in lowered or "wednesday" in lowered:
                return "Wednesday"
            if "sat" in lowered or "saturday" in lowered:
                return "Saturday"
            if "sun" in lowered or "sunday" in lowered:
                return "Sunday"
            if "special" in lowered or "sp" == lowered:
                return "Special"

    weekday = draw_date.weekday()

    if weekday == 2:
        return "Wednesday"
    if weekday == 5:
        return "Saturday"
    if weekday == 6:
        return "Sunday"

    return "Special"


def clean_4d(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    if re.fullmatch(r"\d+\.0", text):
        text = text.split(".", 1)[0]

    digits = re.sub(r"\D", "", text)

    if len(digits) == 0:
        return None

    if len(digits) < 4:
        digits = digits.zfill(4)

    if len(digits) == 4:
        return digits

    return None


def extract_winners_from_row(row: pd.Series, columns: Sequence[str]) -> Tuple[str, ...]:
    """
    Supports common repository formats:
      A) one consolidated column:
         WinningNumbers='1234,5678,9012'
      B) prize/candidate columns:
         FirstPrize, SecondPrize, ThirdPrize, Starter*, Consolation*, col1, etc.
      C) any 4-digit-looking fields except DrawNo/Date/Day columns.
    """

    consolidated_candidates = [
        "WinningNumbers",
        "WinningNumber",
        "Winners",
        "Winner",
        "Candidate",
        "Candidates",
        "Numbers",
        "Number",
    ]

    excluded_col_tokens = {
        "drawno",
        "drawnumber",
        "drawid",
        "drawdate",
        "date",
        "day",
        "daytype",
        "weekday",
    }

    winner_values: List[str] = []
    seen = set()

    consolidated_col = find_first_column(columns, consolidated_candidates)

    if consolidated_col is not None:
        raw = row.get(consolidated_col)

        if raw is not None and not pd.isna(raw):
            pieces = re.split(r"[,|;/\s]+", str(raw).strip())

            for piece in pieces:
                cleaned = clean_4d(piece)
                if cleaned and cleaned not in seen:
                    winner_values.append(cleaned)
                    seen.add(cleaned)

    priority_patterns = [
        "first",
        "second",
        "third",
        "starter",
        "consol",
        "consolation",
        "prize",
        "winner",
        "winning",
        "candidate",
        "col1",
        "col2",
        "col3",
        "number",
    ]

    for col in columns:
        ncol = normalize_col_name(col)

        if ncol in excluded_col_tokens:
            continue

        should_scan = any(p in ncol for p in priority_patterns)

        if not should_scan:
            continue

        cleaned = clean_4d(row.get(col))

        if cleaned and cleaned not in seen:
            winner_values.append(cleaned)
            seen.add(cleaned)

    if not winner_values:
        for col in columns:
            ncol = normalize_col_name(col)

            if ncol in excluded_col_tokens:
                continue

            cleaned = clean_4d(row.get(col))

            if cleaned and cleaned not in seen:
                winner_values.append(cleaned)
                seen.add(cleaned)

    if not winner_values:
        raise ValueError("No 4-digit winning/candidate number found in row")

    return tuple(winner_values)


def discover_data_files() -> List[Path]:
    """
    Strict source discovery.

    Uses only DATA_PATH from .env.
    Does NOT fallback-scan the repository, because research/backtest CSVs
    and .venv package CSVs are not authoritative draw-history sources.
    """

    env_data_path = os.getenv(ENV_DATA_PATH_KEY, "").strip()

    if not env_data_path:
        raise FileNotFoundError(
            f"{ENV_DATA_PATH_KEY} is not set in .env. "
            "Set it to the authoritative master draw-history CSV/XLSX path."
        )

    p = Path(env_data_path)

    if not p.is_absolute():
        p = PROJECT_ROOT / p

    if not p.exists():
        raise FileNotFoundError(
            f"DATA_PATH does not exist: {p}. "
            "Fix .env DATA_PATH to point to the real master history file."
        )

    if p.is_file():
        if p.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
            raise ValueError(f"DATA_PATH file must be CSV/XLSX/XLS, got: {p}")
        return [p]

    if p.is_dir():
        files = []
        files.extend(sorted(p.glob("*.csv")))
        files.extend(sorted(p.glob("*.xlsx")))
        files.extend(sorted(p.glob("*.xls")))

        if not files:
            raise FileNotFoundError(f"DATA_PATH directory has no CSV/XLSX/XLS files: {p}")

        return files

    raise FileNotFoundError(f"DATA_PATH is not a file or directory: {p}")


def read_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        try:
            return pd.read_csv(path, dtype=str, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(path, dtype=str, encoding="latin1")

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str)

    raise ValueError(f"Unsupported file extension: {path}")


def normalize_file(path: Path) -> List[NormalizedDraw]:
    df = read_dataframe(path)

    if df.empty:
        return []

    df.columns = [str(c).strip() for c in df.columns]
    columns = list(df.columns)

    draw_no_col = find_first_column(
        columns,
        ["DrawNo", "Draw No", "DrawNumber", "Draw Number", "Draw", "No"],
    )
    draw_date_col = find_first_column(
        columns,
        ["DrawDate", "Draw Date", "Date", "Draw_Date"],
    )
    day_type_col = find_first_column(
        columns,
        ["DayType", "Day Type", "Day", "Weekday", "DrawDay"],
    )

    if draw_no_col is None:
        raise ValueError(f"{path}: Could not detect DrawNo column. Columns={columns}")

    if draw_date_col is None:
        raise ValueError(f"{path}: Could not detect DrawDate column. Columns={columns}")

    normalized: List[NormalizedDraw] = []

    for idx, row in df.iterrows():
        try:
            draw_no = parse_draw_no(row[draw_no_col])
            draw_date = parse_draw_date(row[draw_date_col])
            day_type = compute_day_type(
                draw_date,
                row[day_type_col] if day_type_col is not None else None,
            )
            winners = extract_winners_from_row(row, columns)

            if day_type not in VALID_DAY_TYPES:
                raise ValueError(f"Invalid DayType {day_type!r}")

            normalized.append(
                NormalizedDraw(
                    draw_no=draw_no,
                    draw_date=draw_date,
                    day_type=day_type,
                    winning_numbers=",".join(winners),
                )
            )
        except Exception as exc:
            raise ValueError(f"{path}: row_index={idx} normalization failed: {exc}") from exc

    return normalized


def load_all_draws() -> List[NormalizedDraw]:
    files = discover_data_files()

    if not files:
        raise FileNotFoundError(
            "No local draw-history files found. "
            "Set DATA_PATH in .env or place CSV/XLSX under data/."
        )

    print_header("SOURCE FILE DISCOVERY")
    for p in files:
        print(f"FOUND: {p}")

    all_rows: Dict[int, NormalizedDraw] = {}

    for path in files:
        try:
            rows = normalize_file(path)
        except Exception as exc:
            print(f"[SKIP] {path}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        print(f"LOADED_FROM_FILE: {path} rows={len(rows)}")

        for row in rows:
            existing = all_rows.get(row.draw_no)

            if existing is None:
                all_rows[row.draw_no] = row
                continue

            if existing != row:
                raise ValueError(
                    f"Conflicting duplicate DrawNo={row.draw_no}. "
                    f"Existing={existing} New={row}"
                )

    if not all_rows:
        raise RuntimeError("No valid draw rows loaded from discovered files.")

    rows_sorted = [all_rows[k] for k in sorted(all_rows)]

    return rows_sorted


def verify_local_rows(rows: Sequence[NormalizedDraw]) -> None:
    if not rows:
        raise RuntimeError("No rows to migrate")

    draw_numbers = [r.draw_no for r in rows]

    print_header("LOCAL NORMALIZATION VERIFY")
    print_kv("LOCAL_TOTAL_ROWS", len(rows))
    print_kv("LOCAL_MIN_DRAWNO", min(draw_numbers))
    print_kv("LOCAL_MAX_DRAWNO", max(draw_numbers))
    print_kv("LOCAL_PHASE1_ROWS_<=4050", sum(1 for x in draw_numbers if x <= PHASE1_MAX_DRAW_NO))
    print_kv("LOCAL_PHASE2_ROWS_>=4051", sum(1 for x in draw_numbers if x >= PHASE2_MIN_DRAW_NO))
    print_kv("LOCAL_FIRST_ROW", rows[0])
    print_kv("LOCAL_LAST_ROW", rows[-1])

    bad = [r for r in rows if not re.fullmatch(r"\d{4}(,\d{4})*", r.winning_numbers)]

    if bad:
        raise ValueError(f"Found {len(bad)} rows with invalid WinningNumbers. First={bad[0]}")


def truncate_drawhistory(conn) -> None:
    """
    Uses DELETE rather than TRUNCATE because MarkovTransitions has FK references.
    Deletes child rows first so trigger-derived mass can be rebuilt cleanly.
    """

    cursor = conn.cursor()

    cursor.execute("DELETE FROM dbo.MarkovTransitions;")
    cursor.execute("DELETE FROM dbo.FormulaRegistry;")
    cursor.execute("DELETE FROM dbo.DrawHistory;")

    conn.commit()


def bulk_insert_drawhistory(conn, rows: Sequence[NormalizedDraw], batch_size: int = 1000) -> None:
    cursor = conn.cursor()
    cursor.fast_executemany = True

    sql = """
        INSERT INTO dbo.DrawHistory (
            DrawNo,
            DrawDate,
            DayType,
            WinningNumbers
        )
        VALUES (?, ?, ?, ?);
    """

    payload = [
        (
            int(r.draw_no),
            r.draw_date,
            r.day_type,
            r.winning_numbers,
        )
        for r in rows
    ]

    total = len(payload)

    for start in range(0, total, batch_size):
        batch = payload[start:start + batch_size]
        cursor.executemany(sql, batch)
        conn.commit()
        print(f"INSERTED_BATCH: {start + len(batch)}/{total}")


def verify_sql_counts(conn) -> None:
    cursor = conn.cursor()

    queries = {
        "SQL_TOTAL_ROWS": "SELECT COUNT(*) FROM dbo.DrawHistory;",
        "SQL_MIN_DRAWNO": "SELECT MIN(DrawNo) FROM dbo.DrawHistory;",
        "SQL_MAX_DRAWNO": "SELECT MAX(DrawNo) FROM dbo.DrawHistory;",
        "SQL_PHASE1_ROWS_<=4050": "SELECT COUNT(*) FROM dbo.DrawHistory WHERE DrawNo <= 4050;",
        "SQL_PHASE2_ROWS_>=4051": "SELECT COUNT(*) FROM dbo.DrawHistory WHERE DrawNo >= 4051;",
        "SQL_MARKOV_ROWS": "SELECT COUNT(*) FROM dbo.MarkovTransitions;",
    }

    print_header("SQL POST-MIGRATION VERIFY")

    for label, sql in queries.items():
        value = cursor.execute(sql).fetchval()
        print_kv(label, value)

    print_header("SQL SAMPLE ROWS")
    rows = cursor.execute(
        """
        SELECT TOP (5)
            DrawNo,
            DrawDate,
            DayType,
            WinningNumbers
        FROM dbo.DrawHistory
        ORDER BY DrawNo ASC;
        """
    ).fetchall()

    for row in rows:
        print(row)

    rows = cursor.execute(
        """
        SELECT TOP (5)
            DrawNo,
            DrawDate,
            DayType,
            WinningNumbers
        FROM dbo.DrawHistory
        ORDER BY DrawNo DESC;
        """
    ).fetchall()

    for row in rows:
        print(row)


def main() -> int:
    print_header("RAW -> SQL MIGRATION START")

    load_env()

    print_kv("PROJECT_ROOT", PROJECT_ROOT)
    print_kv("DATA_PATH", os.getenv(ENV_DATA_PATH_KEY, "<not set>"))
    print_kv("DB_SERVER", os.getenv("DB_SERVER", "<not set>"))
    print_kv("DB_DATABASE", os.getenv("DB_DATABASE", "<not set>"))
    print_kv("DB_USERNAME", os.getenv("DB_USERNAME", "<not set>"))

    rows = load_all_draws()
    verify_local_rows(rows)

    conn_str = get_sql_connection_string()
    conn = connect_sql(conn_str)

    try:
        print_header("CLEAR EXISTING SQL ROWS")
        truncate_drawhistory(conn)
        print("CLEARED: dbo.FormulaRegistry, dbo.MarkovTransitions, dbo.DrawHistory")

        print_header("BULK INSERT dbo.DrawHistory")
        bulk_insert_drawhistory(conn, rows, batch_size=1000)

        verify_sql_counts(conn)

        print_header("RAW -> SQL MIGRATION PASSED")
        print("RESULT: PASSED")
        return 0

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print_header("RAW -> SQL MIGRATION FAILED")
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise
PY
