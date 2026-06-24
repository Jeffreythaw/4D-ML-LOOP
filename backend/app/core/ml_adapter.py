from dataclasses import dataclass
from importlib import import_module
import sys
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.memory import log_memory
from app.schemas.prediction import PredictionCandidate, PredictionRequest


PROJECT_ROOT = Path(__file__).resolve().parents[3]

FALLBACK_ENGINE_SOURCE = "E0_STABILIZED_BASELINE_FALLBACK"
FALLBACK_TOP_K = 5


class PredictionAdapterError(RuntimeError):
    pass


@dataclass(frozen=True)
class PredictionAdapterResult:
    source_draw_number: int
    target_draw_number: int
    day_type: str
    predictions: list[PredictionCandidate]
    ledger_predictions: list[PredictionCandidate]


def run_existing_engine_prediction(request: PredictionRequest, *, allow_fallback: bool = False) -> PredictionAdapterResult:
    """
    Read-only adapter boundary for the existing Step 2 / Step 3 NumPy engine.

    request.draw_number is treated as source_draw_no / base draw number. The
    engine predicts locked Top 5 candidates for target_draw_no = source_draw_no + 1.

    DayType is auto-detected from dbo.DrawHistory through the existing source
    draw loader. The UI does not need to provide DayType.

    This function must not verify predictions, must not load hidden target
    winners, and must not register adaptive formulas. Verification remains only
    in the SQL stored procedure path in app.core.db.
    """
    log_memory("prediction_request_start")
    step2, step3 = _load_existing_engine_modules()
    log_memory("prediction_engine_modules_loaded")
    settings = get_settings()
    source_draw_no = int(request.draw_number)

    try:
        with step2.SqlServerGateway(settings.sql_connection_string()) as gateway:
            log_memory("prediction_sql_gateway_open")
            orchestrator = step3.Step3AdaptiveOrchestrator(
                core=step2,
                gateway=gateway,
                start_draw_no=source_draw_no,
                end_draw_no=source_draw_no + 1,
            )
            log_memory("prediction_orchestrator_ready")
            locked = orchestrator.predict_one_step_locked(source_draw_no)
            log_memory("prediction_locked_top5_ready")

            source_record = gateway.load_phase2_draw(source_draw_no)
            if source_record is None:
                raise LookupError(f"Source DrawNo {source_draw_no} not found in dbo.DrawHistory")
            day_type = str(source_record.day_type)
    except Exception as exc:
        if allow_fallback:
            return _build_stabilized_fallback_result(
                source_draw_no=source_draw_no,
                reason=exc,
            )
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


    ledger_candidates: list[PredictionCandidate] = []
    for item in getattr(locked, "engine_candidate_scores", ()) or ():
        try:
            engine_name = str(item[0])
            rank_no = int(item[1])
            number = str(item[2]).zfill(4)
            score = float(item[3])
        except (TypeError, ValueError, IndexError):
            continue

        ledger_candidates.append(
            PredictionCandidate(
                rank=rank_no,
                number=number,
                score=score,
                source=engine_name,
            )
        )

    if request.mode == "Current":
        from app.core.temporal_context_engine import run_temporal_context_prediction

        log_memory("prediction_temporal_context_loaded")
        temporal_result = run_temporal_context_prediction(
            source_draw_no=source_draw_no,
            target_draw_no=int(locked.target_draw_no),
            underlying_candidates=ledger_candidates or candidates,
        )

        return PredictionAdapterResult(
            source_draw_number=temporal_result.source_draw_number,
            target_draw_number=temporal_result.target_draw_number,
            day_type=temporal_result.day_type,
            predictions=temporal_result.predictions,
            ledger_predictions=temporal_result.predictions,
        )

    return PredictionAdapterResult(
        source_draw_number=source_draw_no,
        target_draw_number=int(locked.target_draw_no),
        day_type=day_type,
        predictions=candidates,
        ledger_predictions=ledger_candidates or candidates,
    )


def _build_stabilized_fallback_result(
    *,
    source_draw_no: int,
    reason: Exception,
) -> PredictionAdapterResult:
    """
    Last-resort fail-safe for /api/predict stability.

    This fallback is deterministic, target-blind, and uses only the source draw
    number. It is not intended to improve accuracy; it prevents API crashes when
    a solver/module/database anomaly occurs before the normal engine can return.
    """
    seed = abs(int(source_draw_no)) % 10000
    offsets = (0, 137, 271, 409, 733, 997, 1229, 1531, 1877, 2213)

    numbers: list[str] = []
    for offset in offsets:
        candidate = f"{(seed + offset) % 10000:04d}"
        if candidate not in numbers:
            numbers.append(candidate)
        if len(numbers) == FALLBACK_TOP_K:
            break

    value = 0
    while len(numbers) < FALLBACK_TOP_K:
        candidate = f"{value:04d}"
        if candidate not in numbers:
            numbers.append(candidate)
        value += 1

    candidates = [
        PredictionCandidate(
            rank=rank_no,
            number=number,
            score=0.0,
            source=FALLBACK_ENGINE_SOURCE,
        )
        for rank_no, number in enumerate(numbers, start=1)
    ]

    return PredictionAdapterResult(
        source_draw_number=int(source_draw_no),
        target_draw_number=int(source_draw_no) + 1,
        day_type="Special",
        predictions=candidates,
        ledger_predictions=candidates,
    )


def _load_existing_engine_modules() -> tuple[Any, Any]:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    try:
        log_memory("prediction_engine_import_start")
        step2 = import_module("jeffrey_quad_engine_v2_step2_matrix_core")
        step3 = import_module("jeffrey_quad_engine_v2_step3_adaptive_orchestrator")
        log_memory("prediction_engine_import_done")
    except Exception as exc:
        raise PredictionAdapterError("Could not import existing Step 2 / Step 3 engine modules.") from exc

    return step2, step3
