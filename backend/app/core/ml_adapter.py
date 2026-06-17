from importlib import import_module
import sys
from pathlib import Path
from typing import Any

from app.schemas.prediction import PredictionCandidate, PredictionRequest


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class PredictionAdapterError(RuntimeError):
    pass


def run_existing_engine_prediction(request: PredictionRequest) -> list[PredictionCandidate]:
    """
    Thin adapter boundary for existing Step 2 / Step 3 NumPy engine scripts.

    This is not a replacement engine and must not reimplement the ML algorithm.
    Wire this function to approved public entry points in:
      - jeffrey_quad_engine_v2_step2_matrix_core.py
      - jeffrey_quad_engine_v2_step3_adaptive_orchestrator.py

    The adapter must never read hidden target winners. Verification belongs only
    in the SQL stored procedure called from app.core.db.
    """
    _load_existing_engine_modules()

    # Integration placeholder: replace only this boundary call once the existing
    # Step 3 script exposes a stable prediction function for a single draw.
    raise PredictionAdapterError(
        "Prediction adapter is ready, but no safe single-draw Step 3 entry point is configured yet."
    )


def _load_existing_engine_modules() -> tuple[Any, Any]:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    try:
        step2 = import_module("jeffrey_quad_engine_v2_step2_matrix_core")
        step3 = import_module("jeffrey_quad_engine_v2_step3_adaptive_orchestrator")
    except Exception as exc:
        raise PredictionAdapterError("Could not import existing Step 2 / Step 3 engine modules.") from exc

    return step2, step3
