# audit_step2_sql_integration.py
# ============================================================
# JEFFREY QUAD-ENGINE HYBRID V2
# STEP 2 SQL INTEGRATION AUDIT RUNNER
#
# Scope:
#   - Loads .env automatically
#   - Reads J4D_SQL_CONN_STR from environment
#   - Connects through approved SqlServerGateway
#   - Runs Phase 1 baseline formula build
#   - Runs Phase 2 Draw 4051 integration trace
#
# Does NOT execute Step 3.
# Does NOT perform Diversity Guard.
# Does NOT perform full blind backtest.
# ============================================================

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent
STEP2_MODULE_NAME = "jeffrey_quad_engine_v2_step2_matrix_core"
ENV_KEY = "J4D_SQL_CONN_STR"
PHASE2_TRACE_DRAW_NO = 4051


def load_env_file() -> None:
    """
    Loads project-root .env if python-dotenv is installed.
    Falls back safely to existing process environment if not installed.
    """

    env_path = PROJECT_ROOT / ".env"

    if not env_path.exists():
        raise FileNotFoundError(
            f"Missing .env file at project root: {env_path}"
        )

    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        print(
            "[WARN] python-dotenv is not installed. "
            "Falling back to existing shell environment only.",
            file=sys.stderr,
        )
        return

    loaded = load_dotenv(dotenv_path=env_path, override=False)

    if not loaded:
        raise RuntimeError(
            f"Failed to load .env file from: {env_path}"
        )


def get_required_env(key: str) -> str:
    value = os.getenv(key)

    if value is None or not value.strip():
        raise EnvironmentError(
            f"Required environment variable '{key}' is missing or empty. "
            f"Confirm it exists in {PROJECT_ROOT / '.env'}."
        )

    return value.strip()


def import_step2_core():
    """
    Imports the approved Step 2 module from project root.
    """

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    try:
        return __import__(STEP2_MODULE_NAME)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Could not import '{STEP2_MODULE_NAME}'. "
            f"Expected file at: {PROJECT_ROOT / (STEP2_MODULE_NAME + '.py')}"
        ) from exc


def print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def print_kv(key: str, value) -> None:
    print(f"{key}: {value}")


def main() -> int:
    print_header("STEP 2 SQL INTEGRATION AUDIT — START")

    load_env_file()
    conn_str = get_required_env(ENV_KEY)

    print_kv("PROJECT_ROOT", PROJECT_ROOT)
    print_kv("ENV_FILE", PROJECT_ROOT / ".env")
    print_kv("CONNECTION_ENV_KEY", ENV_KEY)
    print_kv("CONNECTION_STRING_LOADED", "YES")
    print_kv("CONNECTION_STRING_LENGTH", len(conn_str))

    core = import_step2_core()

    required_symbols = [
        "SqlServerGateway",
        "Phase1BaselineBuilder",
        "Phase2SequentialInputLayer",
        "MatrixComputationCore",
        "engine_outputs_to_strings",
        "PHASE1_MAX_DRAW_NO",
    ]

    missing_symbols = [
        name for name in required_symbols
        if not hasattr(core, name)
    ]

    if missing_symbols:
        raise AttributeError(
            f"Step 2 module is missing required symbols: {missing_symbols}"
        )

    print_kv("STEP2_MODULE_IMPORT", "OK")

    with core.SqlServerGateway(conn_str) as gw:
        print_header("SQL CONNECTION")
        print_kv("SQL_CONNECTION", "OPENED")

        print_header("PHASE 1 BASELINE LOAD")
        phase1_records = gw.load_phase1_training_block()

        if not phase1_records:
            raise RuntimeError("Phase 1 training block returned zero rows.")

        phase1_count = len(phase1_records)
        phase1_first = phase1_records[0]
        phase1_last = phase1_records[-1]

        print_kv("PHASE1_ROWS_LOADED", phase1_count)
        print_kv("PHASE1_FIRST_DRAW", phase1_first)
        print_kv("PHASE1_LAST_DRAW", phase1_last)

        if phase1_last.draw_no > core.PHASE1_MAX_DRAW_NO:
            raise RuntimeError(
                f"Temporal firewall violation: Phase 1 loaded DrawNo "
                f"{phase1_last.draw_no}, expected <= {core.PHASE1_MAX_DRAW_NO}"
            )

        print_kv("PHASE1_TEMPORAL_FIREWALL", "PASSED")

        print_header("PHASE 1 BASELINE FORMULA BUILD")
        baseline_builder = core.Phase1BaselineBuilder(gw)
        baseline_formulas = baseline_builder.build_baseline_formulas()

        if not baseline_formulas:
            raise RuntimeError("No baseline formulas were built.")

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
        phase2 = core.Phase2SequentialInputLayer(gw)

        source_record, source_vectors = phase2.load_source_draw_vectors(
            PHASE2_TRACE_DRAW_NO
        )

        print_kv("SOURCE_DRAW_NO", source_record.draw_no)
        print_kv("SOURCE_DRAW_DATE", source_record.draw_date)
        print_kv("SOURCE_DAY_TYPE", source_record.day_type)
        print_kv("SOURCE_WINNING_NUMBERS", source_record.winning_numbers)
        print_kv("SOURCE_VECTOR_SHAPE", source_vectors.shape)
        print("SOURCE_VECTORS:")
        print(source_vectors)

        print_header("PHASE 2 STATIC MATRIX ENGINE TRACE")
        matrix_core = core.MatrixComputationCore(gw)

        engine_outputs = matrix_core.run_all_static_engines(
            source_vectors,
            source_record.day_type,
        )

        engine_output_strings = core.engine_outputs_to_strings(engine_outputs)

        for engine_name, values in engine_output_strings.items():
            print(f"{engine_name}: {values}")

        print_header("PHASE 2 MARKOV MASS TRACE")
        source_states = list(source_record.winning_numbers)

        markov_mass = phase2.load_markov_input_mass(
            source_states=source_states,
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
        flat_candidates = []
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
            raise RuntimeError(
                f"Unable to produce 5 unique SP smoke candidates. "
                f"Produced: {flat_candidates}"
            )

        target_draw_no = PHASE2_TRACE_DRAW_NO + 1

        hit_count = gw.verify_predictions(
            target_draw_no=target_draw_no,
            predictions=flat_candidates,
        )

        print_kv("SP_TARGET_DRAW_NO", target_draw_no)
        print_kv("SP_TOP5_INPUT", flat_candidates)
        print_kv("SP_RETURNED_HITCOUNT_ONLY", hit_count)

        print_header("STEP 2 SQL INTEGRATION AUDIT — PASSED")
        print_kv("RESULT", "PASSED")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print_header("STEP 2 SQL INTEGRATION AUDIT — FAILED")
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise
