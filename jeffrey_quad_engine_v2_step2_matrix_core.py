from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_KEY = "J4D_SQL_CONN_STR"
PHASE1_MAX_DRAW_NO = 4050
PHASE2_TRACE_DRAW_NO = 4051

MOD_BASE = 10
VECTOR_WIDTH = 4

VALID_DAY_TYPES = {"Wednesday", "Saturday", "Sunday", "Special"}

ENGINE_1_NAME = "E1_CROSS_PAIR_LINEAR"
ENGINE_1_WLS_NAME = "E1_WLS_DECAY_0.98"
ENGINE_1_MIRROR_BASE5_NAME = "E1_MIRROR_BASE5_LSTS"
ENGINE_1_DELTA_ROTATION_NAME = "E1_DELTA_ROTATION_LSTS"
ENGINE_2_NAME = "E2_SET_PROJECTOR"
ENGINE_3_NAME = "E3_POLYNOMIAL"
ENGINE_4_NAME = "E4_ADAPTIVE_4_UNKNOWNS_AFFINE"


logger = logging.getLogger("jeffrey_quad_engine_v2.step2")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)


@dataclass(frozen=True)
class DrawRecord:
    draw_no: int
    draw_date: date
    day_type: str
    winning_numbers: Tuple[str, ...]


@dataclass(frozen=True)
class MatrixFormula:
    engine_name: str
    formula_version: str
    day_type: str
    matrix_m: np.ndarray
    bias_b: np.ndarray
    metadata: Dict[str, Any]

    def validate(self) -> None:
        validate_day_type(self.day_type)

        if self.matrix_m.shape != (VECTOR_WIDTH, VECTOR_WIDTH):
            raise ValueError(
                f"matrix_m must have shape {(VECTOR_WIDTH, VECTOR_WIDTH)}, got {self.matrix_m.shape}"
            )

        if self.bias_b.shape != (VECTOR_WIDTH,):
            raise ValueError(
                f"bias_b must have shape {(VECTOR_WIDTH,)}, got {self.bias_b.shape}"
            )

        validate_int_matrix(self.matrix_m, "matrix_m")
        validate_int_matrix(self.bias_b, "bias_b")


@dataclass(frozen=True)
class MarkovTransition:
    source_state: str
    target_state: str
    day_type: str
    transition_count: int
    first_seen_draw_no: int
    last_seen_draw_no: int


def print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def print_kv(key: str, value: Any) -> None:
    print(f"{key}: {value}")


def load_env_file() -> None:
    env_path = PROJECT_ROOT / ".env"

    if not env_path.exists():
        raise FileNotFoundError(f"Missing .env file at project root: {env_path}")

    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        print(
            "[WARN] python-dotenv not installed. Using existing shell environment only.",
            file=sys.stderr,
        )
        return

    loaded = load_dotenv(dotenv_path=env_path, override=False)

    if not loaded:
        raise RuntimeError(f"Failed to load .env file: {env_path}")


def get_required_env(key: str) -> str:
    value = os.getenv(key)

    if value is None or not value.strip():
        raise EnvironmentError(
            f"Required environment variable '{key}' is missing or empty in {PROJECT_ROOT / '.env'}"
        )

    return value.strip()


def get_sql_connection_string_from_env() -> str:
    """
    Builds SQL Server connection string from .env component variables first.

    Priority:
      1. DB_DRIVER + DB_SERVER + DB_DATABASE + DB_USERNAME + DB_PASSWORD/DB_PASS/DB_PWD
      2. fallback to J4D_SQL_CONN_STR

    This prevents stale J4D_SQL_CONN_STR from silently overriding the actual DB_* configuration.
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

    has_component_config = all(
        [db_driver, db_server, db_database, db_username, db_password]
    )

    if has_component_config:
        return (
            f"DRIVER={{{db_driver}}};"
            f"SERVER={db_server};"
            f"DATABASE={db_database};"
            f"UID={db_username};"
            f"PWD={db_password};"
            "TrustServerCertificate=yes;"
        )

    fallback = os.getenv("J4D_SQL_CONN_STR", "").strip()

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
        "SQL connection configuration is incomplete. "
        f"Missing: {missing}. "
        "Either provide DB_* variables or J4D_SQL_CONN_STR."
    )


def validate_day_type(day_type: str) -> None:
    if day_type not in VALID_DAY_TYPES:
        raise ValueError(f"Invalid DayType '{day_type}'. Expected: {sorted(VALID_DAY_TYPES)}")


def validate_4d_string(value: str, field_name: str = "value") -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str, got {type(value).__name__}")

    if len(value) != 4 or not value.isdigit():
        raise ValueError(f"{field_name} must be a 4-digit string, got '{value}'")


def validate_int_matrix(arr: np.ndarray, name: str) -> None:
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be numpy.ndarray")

    if arr.dtype not in (np.int16, np.int32, np.int64):
        raise TypeError(f"{name} must be int16/int32/int64. Got {arr.dtype}")

    if np.any(arr < 0) or np.any(arr > 9):
        raise ValueError(f"{name} values must be modulo-10 digits [0..9]")


def ensure_mod_int_array(arr: Any, *, name: str, dtype: np.dtype = np.int16) -> np.ndarray:
    out = np.asarray(arr)

    if not np.issubdtype(out.dtype, np.integer):
        raise TypeError(f"{name} must be integer-valued before modulo. Got {out.dtype}")

    out = out.astype(dtype, copy=False)
    out = np.mod(out, MOD_BASE).astype(dtype, copy=False)
    return out


def parse_winning_numbers(raw: str) -> Tuple[str, ...]:
    if raw is None:
        raise ValueError("WinningNumbers cannot be NULL")

    parts = tuple(x.strip() for x in str(raw).split(",") if x.strip())

    if not parts:
        raise ValueError("WinningNumbers cannot be empty")

    for idx, item in enumerate(parts):
        validate_4d_string(item, f"WinningNumbers[{idx}]")

    return parts


def vector_from_4d(value: str, *, dtype: np.dtype = np.int16) -> np.ndarray:
    validate_4d_string(value)
    return np.fromiter((int(ch) for ch in value), dtype=dtype, count=VECTOR_WIDTH)


def vectors_from_4d_strings(values: Sequence[str], *, dtype: np.dtype = np.int16) -> np.ndarray:
    if not values:
        raise ValueError("values cannot be empty")

    matrix = np.vstack([vector_from_4d(v, dtype=dtype) for v in values])
    return ensure_mod_int_array(matrix, name="vectors_from_4d_strings", dtype=dtype)


def string_from_vector(vec: np.ndarray) -> str:
    vec = ensure_mod_int_array(vec, name="vec", dtype=np.int16)

    if vec.shape != (VECTOR_WIDTH,):
        raise ValueError(f"vec must have shape {(VECTOR_WIDTH,)}, got {vec.shape}")

    return "".join(str(int(x)) for x in vec)


def strings_from_vectors(vectors: np.ndarray) -> List[str]:
    vectors = ensure_mod_int_array(vectors, name="vectors", dtype=np.int16)

    if vectors.ndim != 2 or vectors.shape[1] != VECTOR_WIDTH:
        raise ValueError(f"vectors must have shape (N, {VECTOR_WIDTH}), got {vectors.shape}")

    return [string_from_vector(row) for row in vectors]


class SqlServerGateway:
    def __init__(self, connection_string: str, *, autocommit: bool = False, timeout_seconds: int = 30) -> None:
        if not connection_string or not connection_string.strip():
            raise ValueError("connection_string cannot be empty")

        self.connection_string = connection_string
        self.autocommit = autocommit
        self.timeout_seconds = timeout_seconds
        self._conn: Optional[Any] = None

    def __enter__(self) -> "SqlServerGateway":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.rollback()
        else:
            self.commit()
        self.close()

    @property
    def conn(self) -> Any:
        if self._conn is None:
            raise RuntimeError("SQL Server connection is not open")
        return self._conn

    def connect(self) -> None:
        if self._conn is not None:
            return

        try:
            import pyodbc  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "pyodbc is required for SQL Server connectivity. Install with: pip install pyodbc"
            ) from exc

        try:
            self._conn = pyodbc.connect(
                self.connection_string,
                autocommit=self.autocommit,
                timeout=self.timeout_seconds,
            )
        except Exception as exc:
            raise ConnectionError(
                "Failed to open SQL Server connection through pyodbc. "
                "Verify .env J4D_SQL_CONN_STR, ODBC driver, SQL Server, and credentials."
            ) from exc

        logger.info("SQL Server connection opened.")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.info("SQL Server connection closed.")

    def commit(self) -> None:
        if self._conn is not None and not self.autocommit:
            self._conn.commit()

    def rollback(self) -> None:
        if self._conn is not None and not self.autocommit:
            self._conn.rollback()

    def load_draw_history(
        self,
        *,
        min_draw_no: Optional[int] = None,
        max_draw_no: Optional[int] = None,
    ) -> List[DrawRecord]:
        clauses: List[str] = []
        params: List[Any] = []

        if min_draw_no is not None:
            clauses.append("DrawNo >= ?")
            params.append(int(min_draw_no))

        if max_draw_no is not None:
            clauses.append("DrawNo <= ?")
            params.append(int(max_draw_no))

        where_sql = ""
        if clauses:
            where_sql = "WHERE " + " AND ".join(clauses)

        sql = f"""
            SELECT
                DrawNo,
                DrawDate,
                DayType,
                WinningNumbers
            FROM dbo.DrawHistory
            {where_sql}
            ORDER BY DrawNo ASC;
        """

        rows = self.conn.cursor().execute(sql, params).fetchall()

        records: List[DrawRecord] = []

        for row in rows:
            day_type = str(row.DayType)
            validate_day_type(day_type)

            records.append(
                DrawRecord(
                    draw_no=int(row.DrawNo),
                    draw_date=row.DrawDate,
                    day_type=day_type,
                    winning_numbers=parse_winning_numbers(str(row.WinningNumbers)),
                )
            )

        return records

    def load_phase1_training_block(self) -> List[DrawRecord]:
        return self.load_draw_history(max_draw_no=PHASE1_MAX_DRAW_NO)

    def load_phase2_draw(self, draw_no: int) -> Optional[DrawRecord]:
        rows = self.load_draw_history(min_draw_no=draw_no, max_draw_no=draw_no)
        return rows[0] if rows else None

    def load_markov_transitions(
        self,
        *,
        source_state: Optional[str] = None,
        day_type: Optional[str] = None,
        top_n: Optional[int] = None,
    ) -> List[MarkovTransition]:
        clauses: List[str] = []
        params: List[Any] = []

        if source_state is not None:
            validate_4d_string(source_state, "source_state")
            clauses.append("SourceState = ?")
            params.append(source_state)

        if day_type is not None:
            validate_day_type(day_type)
            clauses.append("DayType = ?")
            params.append(day_type)

        top_sql = ""
        if top_n is not None:
            if top_n <= 0:
                raise ValueError("top_n must be positive")
            top_sql = f"TOP ({int(top_n)})"

        where_sql = ""
        if clauses:
            where_sql = "WHERE " + " AND ".join(clauses)

        sql = f"""
            SELECT {top_sql}
                SourceState,
                TargetState,
                DayType,
                TransitionCount,
                FirstSeenDrawNo,
                LastSeenDrawNo
            FROM dbo.vw_MarkovTransitionMass
            {where_sql}
            ORDER BY TransitionCount DESC, LastSeenDrawNo DESC;
        """

        rows = self.conn.cursor().execute(sql, params).fetchall()

        transitions: List[MarkovTransition] = []

        for row in rows:
            transitions.append(
                MarkovTransition(
                    source_state=str(row.SourceState),
                    target_state=str(row.TargetState),
                    day_type=str(row.DayType),
                    transition_count=int(row.TransitionCount),
                    first_seen_draw_no=int(row.FirstSeenDrawNo),
                    last_seen_draw_no=int(row.LastSeenDrawNo),
                )
            )

        return transitions

    def register_formula(
        self,
        formula: MatrixFormula,
        *,
        training_start_draw_no: int,
        training_end_draw_no: int,
        historical_confidence: float = 0.0,
        hit_rate_top5: float = 0.0,
        sample_size: int = 0,
        is_active: bool = True,
    ) -> int:
        formula.validate()

        payload = {
            "engine_name": formula.engine_name,
            "formula_version": formula.formula_version,
            "day_type": formula.day_type,
            "mod_base": MOD_BASE,
            "vector_width": VECTOR_WIDTH,
            "matrix_m": formula.matrix_m.astype(int).tolist(),
            "bias_b": formula.bias_b.astype(int).tolist(),
            "metadata": formula.metadata,
            "registered_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }

        sql = """
            INSERT INTO dbo.FormulaRegistry (
                EngineName,
                FormulaVersion,
                DayType,
                MatrixPayload,
                TrainingStartDrawNo,
                TrainingEndDrawNo,
                HistoricalConfidence,
                HitRateTop5,
                SampleSize,
                IsActive
            )
            OUTPUT INSERTED.FormulaId
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """

        formula_id = self.conn.cursor().execute(
            sql,
            (
                formula.engine_name,
                formula.formula_version,
                formula.day_type,
                json.dumps(payload, separators=(",", ":"), sort_keys=True),
                int(training_start_draw_no),
                int(training_end_draw_no),
                float(historical_confidence),
                float(hit_rate_top5),
                int(sample_size),
                1 if is_active else 0,
            ),
        ).fetchval()

        return int(formula_id)

    def verify_predictions(self, target_draw_no: int, predictions: Sequence[str]) -> int:
        if not predictions:
            raise ValueError("predictions cannot be empty")

        clean: List[str] = []
        seen = set()

        for value in predictions:
            validate_4d_string(value, "prediction")
            if value not in seen:
                clean.append(value)
                seen.add(value)
            if len(clean) == 5:
                break

        if not clean:
            raise ValueError("No valid predictions supplied")

        row = self.conn.cursor().execute(
            "EXEC dbo.SP_Verify_Predictions @TargetDrawNo = ?, @Top5Predictions = ?;",
            (int(target_draw_no), ",".join(clean)),
        ).fetchone()

        if row is None:
            raise RuntimeError("SP_Verify_Predictions returned no result")

        return int(row.HitCount)


def affine_mod10_transform(
    vectors: np.ndarray,
    matrix_m: np.ndarray,
    bias_b: np.ndarray,
    *,
    dtype: np.dtype = np.int16,
) -> np.ndarray:
    vectors = ensure_mod_int_array(vectors, name="vectors", dtype=dtype)
    matrix_m = ensure_mod_int_array(matrix_m, name="matrix_m", dtype=dtype)
    bias_b = ensure_mod_int_array(bias_b, name="bias_b", dtype=dtype)

    if vectors.ndim != 2 or vectors.shape[1] != VECTOR_WIDTH:
        raise ValueError(f"vectors must have shape (N, {VECTOR_WIDTH}), got {vectors.shape}")

    if matrix_m.shape != (VECTOR_WIDTH, VECTOR_WIDTH):
        raise ValueError(f"matrix_m must have shape {(VECTOR_WIDTH, VECTOR_WIDTH)}, got {matrix_m.shape}")

    if bias_b.shape != (VECTOR_WIDTH,):
        raise ValueError(f"bias_b must have shape {(VECTOR_WIDTH,)}, got {bias_b.shape}")

    result = (
        vectors.astype(np.int32, copy=False)
        @ matrix_m.astype(np.int32, copy=False).T
        + bias_b.astype(np.int32, copy=False)
    ) % MOD_BASE

    return result.astype(dtype, copy=False)


class Engine1CrossPairLinear:
    @staticmethod
    def default_formula(day_type: str) -> MatrixFormula:
        validate_day_type(day_type)

        matrix_m = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 0, 1],
                [0, 1, 1, 0],
            ],
            dtype=np.int16,
        )

        bias_map = {
            "Wednesday": [0, 2, 4, 6],
            "Saturday": [1, 3, 5, 7],
            "Sunday": [2, 4, 6, 8],
            "Special": [9, 7, 5, 3],
        }

        return MatrixFormula(
            engine_name=ENGINE_1_NAME,
            formula_version="E1_BASE_V1",
            day_type=day_type,
            matrix_m=matrix_m,
            bias_b=np.array(bias_map[day_type], dtype=np.int16),
            metadata={"description": "Cross-pair linear modulo-10 transform"},
        )

    def predict_vectors(self, src_vectors: np.ndarray, day_type: str) -> np.ndarray:
        formula = self.default_formula(day_type)
        return affine_mod10_transform(src_vectors, formula.matrix_m, formula.bias_b)





def delta_vectors_mod10(
    previous_vectors: np.ndarray,
    current_vectors: np.ndarray,
    *,
    dtype: np.dtype = np.int16,
) -> np.ndarray:
    """
    Compute digit-wise rotation/click delta:

        delta = (current - previous + 10) % 10
    """
    previous_vectors = ensure_mod_int_array(previous_vectors, name="previous_vectors", dtype=np.int16)
    current_vectors = ensure_mod_int_array(current_vectors, name="current_vectors", dtype=np.int16)

    if previous_vectors.shape != current_vectors.shape:
        raise ValueError(
            f"previous_vectors and current_vectors must have same shape, "
            f"got {previous_vectors.shape} and {current_vectors.shape}"
        )

    if previous_vectors.ndim != 2 or previous_vectors.shape[1] != VECTOR_WIDTH:
        raise ValueError(
            f"delta inputs must have shape (N, {VECTOR_WIDTH}), got {previous_vectors.shape}"
        )

    delta = (
        current_vectors.astype(np.int32, copy=False)
        - previous_vectors.astype(np.int32, copy=False)
        + MOD_BASE
    ) % MOD_BASE

    return delta.astype(dtype, copy=False)


def reconstruct_vectors_from_delta(
    base_vectors: np.ndarray,
    delta_vectors: np.ndarray,
    *,
    dtype: np.dtype = np.int16,
) -> np.ndarray:
    """
    Reconstruct absolute vectors:

        predicted = (base + delta) % 10
    """
    base_vectors = ensure_mod_int_array(base_vectors, name="base_vectors", dtype=np.int16)
    delta_vectors = ensure_mod_int_array(delta_vectors, name="delta_vectors", dtype=np.int16)

    if delta_vectors.ndim == 1:
        delta_vectors = delta_vectors.reshape(1, VECTOR_WIDTH)

    if base_vectors.ndim != 2 or base_vectors.shape[1] != VECTOR_WIDTH:
        raise ValueError(f"base_vectors must have shape (N, {VECTOR_WIDTH}), got {base_vectors.shape}")

    if delta_vectors.ndim != 2 or delta_vectors.shape[1] != VECTOR_WIDTH:
        raise ValueError(f"delta_vectors must have shape (N, {VECTOR_WIDTH}), got {delta_vectors.shape}")

    if delta_vectors.shape[0] == 1 and base_vectors.shape[0] > 1:
        delta_vectors = np.repeat(delta_vectors, base_vectors.shape[0], axis=0)

    if base_vectors.shape[0] != delta_vectors.shape[0]:
        raise ValueError(
            f"base_vectors and delta_vectors row counts must match or delta must have one row. "
            f"got {base_vectors.shape[0]} and {delta_vectors.shape[0]}"
        )

    reconstructed = (
        base_vectors.astype(np.int32, copy=False)
        + delta_vectors.astype(np.int32, copy=False)
    ) % MOD_BASE

    return reconstructed.astype(dtype, copy=False)


def solve_delta_rotation_lstsq_transition(
    delta_src_vectors: np.ndarray,
    delta_tgt_vectors: np.ndarray,
    *,
    day_type: str,
    formula_version: str = "E1_DELTA_ROTATION_LSTS_V1",
    source_draw_no: Optional[int] = None,
) -> MatrixFormula:
    """
    Solve affine transition law in digit-delta rotation space.

    The caller must provide only temporally eligible delta_t -> delta_t+1 pairs.
    """
    validate_day_type(day_type)

    delta_src_vectors = ensure_mod_int_array(delta_src_vectors, name="delta_src_vectors", dtype=np.int16)
    delta_tgt_vectors = ensure_mod_int_array(delta_tgt_vectors, name="delta_tgt_vectors", dtype=np.int16)

    pair_count = min(delta_src_vectors.shape[0], delta_tgt_vectors.shape[0])

    if pair_count < VECTOR_WIDTH + 1:
        raise ValueError(
            f"Delta rotation solver requires at least {VECTOR_WIDTH + 1} pairs, got {pair_count}"
        )

    src = delta_src_vectors[:pair_count].astype(np.float64, copy=False)
    tgt = delta_tgt_vectors[:pair_count].astype(np.float64, copy=False)

    design = np.hstack([src, np.ones((pair_count, 1), dtype=np.float64)])

    coeff, residuals, rank, singular_values = np.linalg.lstsq(
        design,
        tgt,
        rcond=None,
    )

    raw_m = coeff[:VECTOR_WIDTH, :].T
    raw_b = coeff[VECTOR_WIDTH, :]

    matrix_m = np.mod(np.rint(raw_m).astype(np.int32), MOD_BASE).astype(np.int16)
    bias_b = np.mod(np.rint(raw_b).astype(np.int32), MOD_BASE).astype(np.int16)

    formula = MatrixFormula(
        engine_name=ENGINE_1_DELTA_ROTATION_NAME,
        formula_version=formula_version,
        day_type=day_type,
        matrix_m=matrix_m,
        bias_b=bias_b,
        metadata={
            "solver": "delta_rotation_np.linalg.lstsq",
            "pair_count_used": int(pair_count),
            "rank": int(rank),
            "singular_values": singular_values.astype(float).tolist(),
            "residuals": residuals.astype(float).tolist(),
            "raw_matrix_m": raw_m.astype(float).tolist(),
            "raw_bias_b": raw_b.astype(float).tolist(),
            "source_draw_no": source_draw_no,
        },
    )

    formula.validate()
    return formula


class Engine1DeltaRotationLsts:
    engine_name = ENGINE_1_DELTA_ROTATION_NAME

    def fit_formula(
        self,
        *,
        delta_src_vectors: np.ndarray,
        delta_tgt_vectors: np.ndarray,
        day_type: str,
        source_draw_no: Optional[int] = None,
    ) -> MatrixFormula:
        return solve_delta_rotation_lstsq_transition(
            delta_src_vectors=delta_src_vectors,
            delta_tgt_vectors=delta_tgt_vectors,
            day_type=day_type,
            source_draw_no=source_draw_no,
        )

    def predict_delta_vectors(
        self,
        *,
        latest_delta_vectors: np.ndarray,
        training_delta_src_vectors: np.ndarray,
        training_delta_tgt_vectors: np.ndarray,
        day_type: str,
        source_draw_no: Optional[int] = None,
    ) -> np.ndarray:
        formula = self.fit_formula(
            delta_src_vectors=training_delta_src_vectors,
            delta_tgt_vectors=training_delta_tgt_vectors,
            day_type=day_type,
            source_draw_no=source_draw_no,
        )

        latest_delta_vectors = ensure_mod_int_array(
            latest_delta_vectors,
            name="latest_delta_vectors",
            dtype=np.int16,
        )

        return affine_mod10_transform(
            latest_delta_vectors,
            formula.matrix_m,
            formula.bias_b,
            dtype=np.int16,
        )

    def predict_absolute_vectors(
        self,
        *,
        base_vectors: np.ndarray,
        latest_delta_vectors: np.ndarray,
        training_delta_src_vectors: np.ndarray,
        training_delta_tgt_vectors: np.ndarray,
        day_type: str,
        source_draw_no: Optional[int] = None,
    ) -> np.ndarray:
        predicted_delta_vectors = self.predict_delta_vectors(
            latest_delta_vectors=latest_delta_vectors,
            training_delta_src_vectors=training_delta_src_vectors,
            training_delta_tgt_vectors=training_delta_tgt_vectors,
            day_type=day_type,
            source_draw_no=source_draw_no,
        )

        return reconstruct_vectors_from_delta(
            base_vectors=base_vectors,
            delta_vectors=predicted_delta_vectors,
            dtype=np.int16,
        )


def digits_to_base5_space(vectors: np.ndarray, *, dtype: np.dtype = np.int16) -> np.ndarray:
    """
    Collapse decimal digits 0..9 into mirror/shadow Base-5 classes.

    Mapping:
      0/5 -> 0
      1/6 -> 1
      2/7 -> 2
      3/8 -> 3
      4/9 -> 4
    """
    vectors = ensure_mod_int_array(vectors, name="vectors", dtype=np.int16)
    return np.mod(vectors, 5).astype(dtype, copy=False)


def expand_base5_vector_to_base10_vectors(base5_vector: np.ndarray) -> np.ndarray:
    """
    Expand one 4-digit Base-5 vector into 16 mirror/shadow Base-10 variants.

    Each Base-5 digit d expands to [d, d + 5].
    """
    base5_vector = np.asarray(base5_vector, dtype=np.int16)

    if base5_vector.shape != (VECTOR_WIDTH,):
        raise ValueError(f"base5_vector must have shape {(VECTOR_WIDTH,)}, got {base5_vector.shape}")

    if np.any(base5_vector < 0) or np.any(base5_vector > 4):
        raise ValueError("base5_vector values must be in [0..4]")

    variants: List[List[int]] = [[]]

    for digit in base5_vector.tolist():
        next_variants: List[List[int]] = []
        for prefix in variants:
            next_variants.append(prefix + [int(digit)])
            next_variants.append(prefix + [int(digit) + 5])
        variants = next_variants

    return ensure_mod_int_array(np.array(variants, dtype=np.int16), name="base5_expansion", dtype=np.int16)


def expand_base5_vectors_to_base10_vectors(base5_vectors: np.ndarray) -> np.ndarray:
    """
    Expand N Base-5 vectors into N*16 Base-10 candidate vectors.
    """
    base5_vectors = np.asarray(base5_vectors, dtype=np.int16)

    if base5_vectors.ndim != 2 or base5_vectors.shape[1] != VECTOR_WIDTH:
        raise ValueError(f"base5_vectors must have shape (N, {VECTOR_WIDTH}), got {base5_vectors.shape}")

    expanded = [expand_base5_vector_to_base10_vectors(row) for row in base5_vectors]
    return ensure_mod_int_array(np.vstack(expanded), name="expanded_base10_vectors", dtype=np.int16)


def solve_base5_lstsq_transition(
    src_vectors: np.ndarray,
    tgt_vectors: np.ndarray,
    *,
    day_type: str,
    formula_version: str = "E1_MIRROR_BASE5_LSTS_V1",
    source_draw_no: Optional[int] = None,
) -> MatrixFormula:
    """
    Solve an affine transition law in collapsed Base-5 mirror space.

    The caller must provide only temporally eligible pairs. The returned matrix
    and bias are Base-5-class coefficients stored in MatrixFormula for audit.
    """
    validate_day_type(day_type)

    src_base5 = digits_to_base5_space(src_vectors, dtype=np.int16)
    tgt_base5 = digits_to_base5_space(tgt_vectors, dtype=np.int16)

    pair_count = min(src_base5.shape[0], tgt_base5.shape[0])

    if pair_count < VECTOR_WIDTH + 1:
        raise ValueError(
            f"Base5 LSTS solver requires at least {VECTOR_WIDTH + 1} pairs, got {pair_count}"
        )

    src = src_base5[:pair_count].astype(np.float64, copy=False)
    tgt = tgt_base5[:pair_count].astype(np.float64, copy=False)

    design = np.hstack([src, np.ones((pair_count, 1), dtype=np.float64)])

    coeff, residuals, rank, singular_values = np.linalg.lstsq(
        design,
        tgt,
        rcond=None,
    )

    raw_m = coeff[:VECTOR_WIDTH, :].T
    raw_b = coeff[VECTOR_WIDTH, :]

    matrix_m = np.mod(np.rint(raw_m).astype(np.int32), 5).astype(np.int16)
    bias_b = np.mod(np.rint(raw_b).astype(np.int32), 5).astype(np.int16)

    formula = MatrixFormula(
        engine_name=ENGINE_1_MIRROR_BASE5_NAME,
        formula_version=formula_version,
        day_type=day_type,
        matrix_m=matrix_m,
        bias_b=bias_b,
        metadata={
            "solver": "base5_np.linalg.lstsq",
            "pair_count_used": int(pair_count),
            "rank": int(rank),
            "singular_values": singular_values.astype(float).tolist(),
            "residuals": residuals.astype(float).tolist(),
            "raw_matrix_m": raw_m.astype(float).tolist(),
            "raw_bias_b": raw_b.astype(float).tolist(),
            "source_draw_no": source_draw_no,
        },
    )

    # MatrixFormula.validate() enforces 0..9, so Base-5 coefficients are valid.
    formula.validate()
    return formula


class Engine1MirrorBase5Lsts:
    engine_name = ENGINE_1_MIRROR_BASE5_NAME

    def fit_formula(
        self,
        *,
        src_vectors: np.ndarray,
        tgt_vectors: np.ndarray,
        day_type: str,
        source_draw_no: Optional[int] = None,
    ) -> MatrixFormula:
        return solve_base5_lstsq_transition(
            src_vectors=src_vectors,
            tgt_vectors=tgt_vectors,
            day_type=day_type,
            source_draw_no=source_draw_no,
        )

    def predict_base5_vectors(
        self,
        *,
        src_vectors: np.ndarray,
        training_src_vectors: np.ndarray,
        training_tgt_vectors: np.ndarray,
        day_type: str,
        source_draw_no: Optional[int] = None,
    ) -> np.ndarray:
        formula = self.fit_formula(
            src_vectors=training_src_vectors,
            tgt_vectors=training_tgt_vectors,
            day_type=day_type,
            source_draw_no=source_draw_no,
        )

        src_base5 = digits_to_base5_space(src_vectors, dtype=np.int16)

        projected = (
            src_base5.astype(np.int32, copy=False)
            @ formula.matrix_m.astype(np.int32, copy=False).T
            + formula.bias_b.astype(np.int32, copy=False)
        ) % 5

        return projected.astype(np.int16, copy=False)

    def expand_predictions(
        self,
        *,
        src_vectors: np.ndarray,
        training_src_vectors: np.ndarray,
        training_tgt_vectors: np.ndarray,
        day_type: str,
        source_draw_no: Optional[int] = None,
    ) -> np.ndarray:
        base5_vectors = self.predict_base5_vectors(
            src_vectors=src_vectors,
            training_src_vectors=training_src_vectors,
            training_tgt_vectors=training_tgt_vectors,
            day_type=day_type,
            source_draw_no=source_draw_no,
        )
        return expand_base5_vectors_to_base10_vectors(base5_vectors)


def solve_weighted_affine_mod10(
    src_vectors: np.ndarray,
    tgt_vectors: np.ndarray,
    *,
    day_type: str,
    decay: float = 0.98,
    formula_version: str = "E1_WLS_DECAY_0.98_V1",
    source_draw_no: Optional[int] = None,
) -> MatrixFormula:
    """
    Solve an experimental weighted affine modulo-10 transform.

    The caller must provide only temporally eligible training pairs. This
    function applies exponential decay so the newest row has weight 1.0 and
    older rows receive decay ** age lower weight.
    """
    validate_day_type(day_type)

    if not 0.0 < decay <= 1.0:
        raise ValueError("decay must be in the range (0, 1].")

    src_vectors = ensure_mod_int_array(src_vectors, name="src_vectors", dtype=np.int16)
    tgt_vectors = ensure_mod_int_array(tgt_vectors, name="tgt_vectors", dtype=np.int16)

    if src_vectors.ndim != 2 or src_vectors.shape[1] != VECTOR_WIDTH:
        raise ValueError(f"src_vectors must have shape (N, {VECTOR_WIDTH}), got {src_vectors.shape}")

    if tgt_vectors.ndim != 2 or tgt_vectors.shape[1] != VECTOR_WIDTH:
        raise ValueError(f"tgt_vectors must have shape (N, {VECTOR_WIDTH}), got {tgt_vectors.shape}")

    pair_count = min(src_vectors.shape[0], tgt_vectors.shape[0])

    if pair_count < VECTOR_WIDTH + 1:
        raise ValueError(
            f"Weighted affine solver requires at least {VECTOR_WIDTH + 1} pairs, got {pair_count}"
        )

    src = src_vectors[:pair_count].astype(np.float64, copy=False)
    tgt = tgt_vectors[:pair_count].astype(np.float64, copy=False)

    design = np.hstack([src, np.ones((pair_count, 1), dtype=np.float64)])

    # Oldest row gets the smallest weight. Newest row gets weight 1.0.
    ages = np.arange(pair_count - 1, -1, -1, dtype=np.float64)
    weights = np.power(float(decay), ages)
    sqrt_weights = np.sqrt(weights).reshape(-1, 1)

    weighted_design = design * sqrt_weights
    weighted_target = tgt * sqrt_weights

    coeff, residuals, rank, singular_values = np.linalg.lstsq(
        weighted_design,
        weighted_target,
        rcond=None,
    )

    raw_m = coeff[:VECTOR_WIDTH, :].T
    raw_b = coeff[VECTOR_WIDTH, :]

    matrix_m = np.mod(np.rint(raw_m).astype(np.int32), MOD_BASE).astype(np.int16)
    bias_b = np.mod(np.rint(raw_b).astype(np.int32), MOD_BASE).astype(np.int16)

    formula = MatrixFormula(
        engine_name=ENGINE_1_WLS_NAME,
        formula_version=formula_version,
        day_type=day_type,
        matrix_m=matrix_m,
        bias_b=bias_b,
        metadata={
            "solver": "weighted_np.linalg.lstsq",
            "decay": float(decay),
            "pair_count_used": int(pair_count),
            "rank": int(rank),
            "singular_values": singular_values.astype(float).tolist(),
            "residuals": residuals.astype(float).tolist(),
            "raw_matrix_m": raw_m.astype(float).tolist(),
            "raw_bias_b": raw_b.astype(float).tolist(),
            "oldest_weight": float(weights[0]),
            "newest_weight": float(weights[-1]),
            "source_draw_no": source_draw_no,
        },
    )

    formula.validate()
    return formula


class Engine1WeightedLeastSquaresDecay:
    engine_name = ENGINE_1_WLS_NAME

    def __init__(self, *, decay: float = 0.98) -> None:
        self.decay = float(decay)

    def fit_formula(
        self,
        *,
        src_vectors: np.ndarray,
        tgt_vectors: np.ndarray,
        day_type: str,
        source_draw_no: Optional[int] = None,
    ) -> MatrixFormula:
        return solve_weighted_affine_mod10(
            src_vectors=src_vectors,
            tgt_vectors=tgt_vectors,
            day_type=day_type,
            decay=self.decay,
            source_draw_no=source_draw_no,
        )

    def predict_vectors(
        self,
        *,
        src_vectors: np.ndarray,
        training_src_vectors: np.ndarray,
        training_tgt_vectors: np.ndarray,
        day_type: str,
        source_draw_no: Optional[int] = None,
    ) -> np.ndarray:
        formula = self.fit_formula(
            src_vectors=training_src_vectors,
            tgt_vectors=training_tgt_vectors,
            day_type=day_type,
            source_draw_no=source_draw_no,
        )
        return affine_mod10_transform(src_vectors, formula.matrix_m, formula.bias_b)


class Engine2SetProjector:
    @staticmethod
    def default_formula(day_type: str) -> MatrixFormula:
        validate_day_type(day_type)

        matrix_m = np.array(
            [
                [1, 1, 1, 0],
                [0, 1, 1, 1],
                [1, 0, 1, 1],
                [1, 1, 0, 1],
            ],
            dtype=np.int16,
        )

        bias_map = {
            "Wednesday": [1, 0, 1, 0],
            "Saturday": [0, 1, 0, 1],
            "Sunday": [2, 2, 0, 0],
            "Special": [3, 0, 3, 0],
        }

        return MatrixFormula(
            engine_name=ENGINE_2_NAME,
            formula_version="E2_BASE_V1",
            day_type=day_type,
            matrix_m=matrix_m,
            bias_b=np.array(bias_map[day_type], dtype=np.int16),
            metadata={"description": "Set projector modulo-10 transform"},
        )

    def predict_vectors(self, src_vectors: np.ndarray, day_type: str) -> np.ndarray:
        formula = self.default_formula(day_type)
        return affine_mod10_transform(src_vectors, formula.matrix_m, formula.bias_b)


class Engine3Polynomial:
    @staticmethod
    def default_formula(day_type: str) -> MatrixFormula:
        validate_day_type(day_type)

        base_m = np.array(
            [
                [1, 0, 0, 1],
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 1],
            ],
            dtype=np.int16,
        )

        square_m = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
            ],
            dtype=np.int16,
        )

        adjacent_m = np.array(
            [
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 1],
                [1, 0, 0, 1],
            ],
            dtype=np.int16,
        )

        bias_map = {
            "Wednesday": [3, 1, 4, 1],
            "Saturday": [5, 9, 2, 6],
            "Sunday": [2, 7, 1, 8],
            "Special": [8, 1, 8, 2],
        }

        return MatrixFormula(
            engine_name=ENGINE_3_NAME,
            formula_version="E3_BASE_V1",
            day_type=day_type,
            matrix_m=base_m,
            bias_b=np.array(bias_map[day_type], dtype=np.int16),
            metadata={
                "description": "Polynomial modulo-10 transform",
                "square_m": square_m.astype(int).tolist(),
                "adjacent_m": adjacent_m.astype(int).tolist(),
            },
        )

    @staticmethod
    def polynomial_features(src_vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        src_vectors = ensure_mod_int_array(src_vectors, name="src_vectors", dtype=np.int16)

        if src_vectors.ndim != 2 or src_vectors.shape[1] != VECTOR_WIDTH:
            raise ValueError(f"src_vectors must have shape (N, {VECTOR_WIDTH}), got {src_vectors.shape}")

        work = src_vectors.astype(np.int32, copy=False)
        base = work % MOD_BASE
        squared = (work * work) % MOD_BASE

        a = work[:, 0]
        b = work[:, 1]
        c = work[:, 2]
        d = work[:, 3]

        adjacent = np.stack(
            [
                (a * b) % MOD_BASE,
                (b * c) % MOD_BASE,
                (c * d) % MOD_BASE,
                (d * a) % MOD_BASE,
            ],
            axis=1,
        )

        return (
            base.astype(np.int16, copy=False),
            squared.astype(np.int16, copy=False),
            adjacent.astype(np.int16, copy=False),
        )

    def predict_vectors(self, src_vectors: np.ndarray, day_type: str) -> np.ndarray:
        formula = self.default_formula(day_type)
        base, squared, adjacent = self.polynomial_features(src_vectors)

        square_m = np.array(formula.metadata["square_m"], dtype=np.int16)
        adjacent_m = np.array(formula.metadata["adjacent_m"], dtype=np.int16)

        zero = np.zeros(VECTOR_WIDTH, dtype=np.int16)

        base_out = affine_mod10_transform(base, formula.matrix_m, zero)
        square_out = affine_mod10_transform(squared, square_m, zero)
        adjacent_out = affine_mod10_transform(adjacent, adjacent_m, formula.bias_b)

        combined = (
            base_out.astype(np.int32)
            + square_out.astype(np.int32)
            + adjacent_out.astype(np.int32)
        ) % MOD_BASE

        return combined.astype(np.int16, copy=False)


def solve_4_unknowns_affine(
    src_vectors: np.ndarray,
    tgt_vectors: np.ndarray,
    *,
    day_type: str,
    formula_version: Optional[str] = None,
    source_draw_no: Optional[int] = None,
    target_draw_no: Optional[int] = None,
) -> MatrixFormula:
    validate_day_type(day_type)

    src_vectors = ensure_mod_int_array(src_vectors, name="src_vectors", dtype=np.int16)
    tgt_vectors = ensure_mod_int_array(tgt_vectors, name="tgt_vectors", dtype=np.int16)

    if src_vectors.ndim != 2 or src_vectors.shape[1] != VECTOR_WIDTH:
        raise ValueError(f"src_vectors must have shape (N, {VECTOR_WIDTH}), got {src_vectors.shape}")

    if tgt_vectors.ndim != 2 or tgt_vectors.shape[1] != VECTOR_WIDTH:
        raise ValueError(f"tgt_vectors must have shape (N, {VECTOR_WIDTH}), got {tgt_vectors.shape}")

    if src_vectors.shape[0] == 0 or tgt_vectors.shape[0] == 0:
        raise ValueError("src_vectors and tgt_vectors must contain at least one row")

    pair_count = min(src_vectors.shape[0], tgt_vectors.shape[0])
    src = src_vectors[:pair_count].astype(np.float64, copy=False)
    tgt = tgt_vectors[:pair_count].astype(np.float64, copy=False)

    design = np.hstack([src, np.ones((pair_count, 1), dtype=np.float64)])
    coeff, residuals, rank, singular_values = np.linalg.lstsq(design, tgt, rcond=None)

    raw_m = coeff[:VECTOR_WIDTH, :].T
    raw_b = coeff[VECTOR_WIDTH, :]

    matrix_m = np.mod(np.rint(raw_m).astype(np.int32), MOD_BASE).astype(np.int16)
    bias_b = np.mod(np.rint(raw_b).astype(np.int32), MOD_BASE).astype(np.int16)

    if formula_version is None:
        if source_draw_no is not None and target_draw_no is not None:
            formula_version = f"E4_FIX_{source_draw_no}_TO_{target_draw_no}"
        else:
            formula_version = "E4_FIX_UNSPECIFIED_TRANSITION"

    formula = MatrixFormula(
        engine_name=ENGINE_4_NAME,
        formula_version=formula_version,
        day_type=day_type,
        matrix_m=matrix_m,
        bias_b=bias_b,
        metadata={
            "solver": "np.linalg.lstsq",
            "pair_count_used": int(pair_count),
            "rank": int(rank),
            "singular_values": singular_values.astype(float).tolist(),
            "residuals": residuals.astype(float).tolist(),
            "raw_matrix_m": raw_m.astype(float).tolist(),
            "raw_bias_b": raw_b.astype(float).tolist(),
            "source_draw_no": source_draw_no,
            "target_draw_no": target_draw_no,
        },
    )

    formula.validate()
    return formula


class Phase1BaselineBuilder:
    def __init__(self, gateway: SqlServerGateway) -> None:
        self.gateway = gateway

    def build_baseline_formulas(self) -> List[MatrixFormula]:
        phase1_records = self.gateway.load_phase1_training_block()

        if not phase1_records:
            raise RuntimeError("No Phase 1 records loaded from dbo.DrawHistory")

        max_draw_no = max(r.draw_no for r in phase1_records)

        if max_draw_no > PHASE1_MAX_DRAW_NO:
            raise RuntimeError(
                f"Temporal firewall violation: loaded DrawNo {max_draw_no}, expected <= {PHASE1_MAX_DRAW_NO}"
            )

        observed_day_types = sorted({r.day_type for r in phase1_records})
        formulas: List[MatrixFormula] = []

        for day_type in observed_day_types:
            formulas.append(Engine1CrossPairLinear.default_formula(day_type))
            formulas.append(Engine2SetProjector.default_formula(day_type))
            formulas.append(Engine3Polynomial.default_formula(day_type))

        return formulas


class Phase2SequentialInputLayer:
    def __init__(self, gateway: SqlServerGateway) -> None:
        self.gateway = gateway

    def load_source_draw_vectors(self, draw_no: int) -> Tuple[DrawRecord, np.ndarray]:
        record = self.gateway.load_phase2_draw(draw_no)

        if record is None:
            raise LookupError(f"DrawNo {draw_no} not found in dbo.DrawHistory")

        return record, vectors_from_4d_strings(record.winning_numbers, dtype=np.int16)

    def load_markov_input_mass(
        self,
        *,
        source_states: Sequence[str],
        day_type: str,
        top_n_per_source: int = 5,
    ) -> Dict[str, List[MarkovTransition]]:
        validate_day_type(day_type)

        output: Dict[str, List[MarkovTransition]] = {}

        for source_state in source_states:
            validate_4d_string(source_state, "source_state")
            output[source_state] = self.gateway.load_markov_transitions(
                source_state=source_state,
                day_type=day_type,
                top_n=top_n_per_source,
            )

        return output


class MatrixComputationCore:
    def __init__(self, gateway: SqlServerGateway) -> None:
        self.gateway = gateway
        self.engine1 = Engine1CrossPairLinear()
        self.engine2 = Engine2SetProjector()
        self.engine3 = Engine3Polynomial()

    def run_engine1(self, src_vectors: np.ndarray, day_type: str) -> np.ndarray:
        return self.engine1.predict_vectors(src_vectors, day_type)

    def run_engine2(self, src_vectors: np.ndarray, day_type: str) -> np.ndarray:
        return self.engine2.predict_vectors(src_vectors, day_type)

    def run_engine3(self, src_vectors: np.ndarray, day_type: str) -> np.ndarray:
        return self.engine3.predict_vectors(src_vectors, day_type)

    def apply_formula(self, src_vectors: np.ndarray, formula: MatrixFormula) -> np.ndarray:
        formula.validate()
        return affine_mod10_transform(
            src_vectors,
            formula.matrix_m,
            formula.bias_b,
            dtype=np.int16,
        )

    def run_all_static_engines(self, src_vectors: np.ndarray, day_type: str) -> Dict[str, np.ndarray]:
        validate_day_type(day_type)
        src_vectors = ensure_mod_int_array(src_vectors, name="src_vectors", dtype=np.int16)

        return {
            ENGINE_1_NAME: self.run_engine1(src_vectors, day_type),
            ENGINE_2_NAME: self.run_engine2(src_vectors, day_type),
            ENGINE_3_NAME: self.run_engine3(src_vectors, day_type),
        }

    def solve_and_register_adaptive_formula(
        self,
        *,
        source_draw_no: int,
        target_draw_no: int,
        day_type: str,
        src_vectors: np.ndarray,
        tgt_vectors: np.ndarray,
        training_start_draw_no: int,
        training_end_draw_no: int,
    ) -> int:
        validate_day_type(day_type)

        formula = solve_4_unknowns_affine(
            src_vectors=src_vectors,
            tgt_vectors=tgt_vectors,
            day_type=day_type,
            formula_version=f"E4_FIX_{source_draw_no}_TO_{target_draw_no}",
            source_draw_no=source_draw_no,
            target_draw_no=target_draw_no,
        )

        formula_id = self.gateway.register_formula(
            formula,
            training_start_draw_no=training_start_draw_no,
            training_end_draw_no=training_end_draw_no,
            historical_confidence=0.0,
            hit_rate_top5=0.0,
            sample_size=int(min(src_vectors.shape[0], tgt_vectors.shape[0])),
            is_active=True,
        )

        self.gateway.commit()
        return int(formula_id)


def engine_outputs_to_strings(engine_outputs: Dict[str, np.ndarray]) -> Dict[str, List[str]]:
    return {engine_name: strings_from_vectors(vectors) for engine_name, vectors in engine_outputs.items()}


def run_database_free_smoke_test() -> None:
    print_header("DATABASE-FREE MATRIX SMOKE TEST")

    src = vectors_from_4d_strings(["1234", "5678", "9012"], dtype=np.int16)
    tgt = vectors_from_4d_strings(["2345", "6789", "0123"], dtype=np.int16)
    day_type = "Wednesday"

    outputs = {
        ENGINE_1_NAME: Engine1CrossPairLinear().predict_vectors(src, day_type),
        ENGINE_2_NAME: Engine2SetProjector().predict_vectors(src, day_type),
        ENGINE_3_NAME: Engine3Polynomial().predict_vectors(src, day_type),
    }

    for engine_name, values in engine_outputs_to_strings(outputs).items():
        print(f"{engine_name}: {values}")

    formula = solve_4_unknowns_affine(
        src_vectors=src,
        tgt_vectors=tgt,
        day_type=day_type,
        source_draw_no=100,
        target_draw_no=101,
    )

    solved = affine_mod10_transform(src, formula.matrix_m, formula.bias_b)

    print("E4_MATRIX_M:")
    print(formula.matrix_m)
    print(f"E4_BIAS_B: {formula.bias_b}")
    print(f"E4_SOLVED_OUTPUTS: {strings_from_vectors(solved)}")


def run_sql_integration_audit() -> None:
    print_header("STEP 2 SQL INTEGRATION AUDIT — START")

    load_env_file()
    conn_str = get_sql_connection_string_from_env()

    print_kv("PROJECT_ROOT", PROJECT_ROOT)
    print_kv("ENV_FILE", PROJECT_ROOT / ".env")
    print_kv("CONNECTION_SOURCE_PRIORITY", "DB_* variables first, J4D_SQL_CONN_STR fallback")
    print_kv("DB_SERVER", os.getenv("DB_SERVER", "").strip() or "<not set>")
    print_kv("DB_DATABASE", os.getenv("DB_DATABASE", "").strip() or "<not set>")
    print_kv("DB_USERNAME", os.getenv("DB_USERNAME", "").strip() or "<not set>")
    print_kv("CONNECTION_STRING_LOADED", "YES")
    print_kv("CONNECTION_STRING_LENGTH", len(conn_str))

    with SqlServerGateway(conn_str) as gw:
        print_header("SQL CONNECTION")
        print_kv("SQL_CONNECTION", "OPENED")

        print_header("PHASE 1 BASELINE LOAD")
        phase1_records = gw.load_phase1_training_block()

        if not phase1_records:
            raise RuntimeError("Phase 1 training block returned zero rows")

        print_kv("PHASE1_ROWS_LOADED", len(phase1_records))
        print_kv("PHASE1_FIRST_DRAW", phase1_records[0])
        print_kv("PHASE1_LAST_DRAW", phase1_records[-1])

        if phase1_records[-1].draw_no > PHASE1_MAX_DRAW_NO:
            raise RuntimeError(
                f"Temporal firewall violation: Phase 1 last DrawNo={phase1_records[-1].draw_no}"
            )

        print_kv("PHASE1_TEMPORAL_FIREWALL", "PASSED")

        print_header("PHASE 1 BASELINE FORMULA BUILD")
        baseline_builder = Phase1BaselineBuilder(gw)
        baseline_formulas = baseline_builder.build_baseline_formulas()

        print_kv("BASELINE_FORMULAS_BUILT", len(baseline_formulas))

        for idx, formula in enumerate(baseline_formulas, start=1):
            formula.validate()
            print(
                f"FORMULA_{idx}: "
                f"engine={formula.engine_name} "
                f"version={formula.formula_version} "
                f"day_type={formula.day_type} "
                f"M_shape={formula.matrix_m.shape} "
                f"B_shape={formula.bias_b.shape}"
            )

        print_header("PHASE 2 DRAW 4051 SOURCE INPUT TRACE")
        phase2 = Phase2SequentialInputLayer(gw)

        source_record, source_vectors = phase2.load_source_draw_vectors(PHASE2_TRACE_DRAW_NO)

        print_kv("SOURCE_DRAW_NO", source_record.draw_no)
        print_kv("SOURCE_DRAW_DATE", source_record.draw_date)
        print_kv("SOURCE_DAY_TYPE", source_record.day_type)
        print_kv("SOURCE_WINNING_NUMBERS", source_record.winning_numbers)
        print_kv("SOURCE_VECTOR_SHAPE", source_vectors.shape)
        print("SOURCE_VECTORS:")
        print(source_vectors)

        print_header("PHASE 2 STATIC MATRIX ENGINE TRACE")
        matrix_core = MatrixComputationCore(gw)
        engine_outputs = matrix_core.run_all_static_engines(source_vectors, source_record.day_type)
        engine_output_strings = engine_outputs_to_strings(engine_outputs)

        for engine_name, values in engine_output_strings.items():
            print(f"{engine_name}: {values}")

        print_header("PHASE 2 MARKOV MASS TRACE")
        markov_mass = phase2.load_markov_input_mass(
            source_states=list(source_record.winning_numbers),
            day_type=source_record.day_type,
            top_n_per_source=5,
        )

        print_kv("MARKOV_SOURCE_COUNT", len(markov_mass))

        for source_state, transitions in markov_mass.items():
            print(f"SOURCE_STATE={source_state} TRANSITION_ROWS={len(transitions)}")
            for row in transitions:
                print(
                    "  "
                    f"source={row.source_state} "
                    f"target={row.target_state} "
                    f"day_type={row.day_type} "
                    f"count={row.transition_count} "
                    f"first_seen={row.first_seen_draw_no} "
                    f"last_seen={row.last_seen_draw_no}"
                )

        print_header("OPTIONAL FIREWALL SP SMOKE TRACE")
        flat_candidates: List[str] = []
        seen = set()

        for values in engine_output_strings.values():
            for value in values:
                if value not in seen:
                    flat_candidates.append(value)
                    seen.add(value)
                if len(flat_candidates) == 5:
                    break
            if len(flat_candidates) == 5:
                break

        if len(flat_candidates) < 5:
            raise RuntimeError(f"Unable to produce 5 unique SP candidates: {flat_candidates}")

        target_draw_no = PHASE2_TRACE_DRAW_NO + 1
        hit_count = gw.verify_predictions(target_draw_no, flat_candidates)

        print_kv("SP_TARGET_DRAW_NO", target_draw_no)
        print_kv("SP_TOP5_INPUT", flat_candidates)
        print_kv("SP_RETURNED_HITCOUNT_ONLY", hit_count)

    print_header("STEP 2 SQL INTEGRATION AUDIT — PASSED")
    print_kv("RESULT", "PASSED")


def main() -> int:
    run_database_free_smoke_test()
    run_sql_integration_audit()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print_header("STEP 2 SQL INTEGRATION AUDIT — FAILED")
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise
