from importlib import import_module
import sys
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.schemas.prediction import PredictionCandidate, PredictionRequest


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class PredictionAdapterError(RuntimeError):
    pass


def run_existing_engine_prediction(request: PredictionRequest) -> list[PredictionCandidate]:
    """
    Read-only adapter boundary for the existing Step 2 / Step 3 NumPy engine.

    request.draw_number is treated as source_draw_no. The engine predicts the
    locked Top 5 candidates for target_draw_no = source_draw_no + 1.

    This function must not verify predictions, must not load hidden target
    winners, and must not register adaptive formulas. Verification remains only
    in the SQL stored procedure path in app.core.db.
    """
    step2, step3 = _load_existing_engine_modules()
    settings = get_settings()

    try:
        with step2.SqlServerGateway(settings.sql_connection_string()) as gateway:
            orchestrator = step3.Step3AdaptiveOrchestrator(
                core=step2,
                gateway=gateway,
                start_draw_no=int(request.draw_number),
                end_draw_no=int(request.draw_number) + 1,
            )
            locked = orchestrator.predict_one_step_locked(int(request.draw_number))
    except Exception as exc:
        raise PredictionAdapterError("Existing engine prediction failed.") from exc

    score_by_number: dict[str, float | None] = {}
    for score_item in getattr(locked, "candidate_scores", ()) or ():
        try:
            number = str(score_item[0])
            score_by_number[number] = float(score_item[1])
        except (TypeError, ValueError, IndexError):
            continue

    candidates: list[PredictionCandidate] = []
    for idx, number in enumerate(locked.top5, start=1):
        source = None
        try:
            source = str(locked.engine_sources[idx - 1])
        except (IndexError, TypeError):
            source = None

        candidates.append(
            PredictionCandidate(
                rank=idx,
                number=str(number).zfill(4),
                score=score_by_number.get(str(number)),
                source=source,
            )
        )

    return candidates


def _load_existing_engine_modules() -> tuple[Any, Any]:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    try:
        step2 = import_module("jeffrey_quad_engine_v2_step2_matrix_core")
        step3 = import_module("jeffrey_quad_engine_v2_step3_adaptive_orchestrator")
    except Exception as exc:
        raise PredictionAdapterError("Could not import existing Step 2 / Step 3 engine modules.") from exc

    return step2, step3
