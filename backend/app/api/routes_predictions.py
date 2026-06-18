from fastapi import APIRouter, HTTPException, status

from app.core.db import (
    VerificationError,
    get_latest_draw_metadata,
    record_predictions_to_ledger,
    update_ledger_after_verification,
    verify_predictions_with_sql,
)
from app.core.ml_adapter import PredictionAdapterError, run_existing_engine_prediction
from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    LatestDrawResponse,
    VerificationRequest,
    VerificationResponse,
)


router = APIRouter(tags=["predictions"])


@router.get("/latest-draw", response_model=LatestDrawResponse)
def latest_draw() -> LatestDrawResponse:
    try:
        return get_latest_draw_metadata()
    except VerificationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        adapter_result = run_existing_engine_prediction(request)
    except PredictionAdapterError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    predictions = adapter_result.predictions[:5]

    try:
        record_predictions_to_ledger(
            mode=request.mode,
            source_draw_no=adapter_result.source_draw_number,
            target_draw_no=adapter_result.target_draw_number,
            day_type=adapter_result.day_type,
            predictions=adapter_result.ledger_predictions,
        )
    except VerificationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    return PredictionResponse(
        draw_number=adapter_result.source_draw_number,
        target_draw_number=adapter_result.target_draw_number,
        day_type=adapter_result.day_type,
        predictions=predictions,
        verification_status="not_verified",
    )


@router.post("/verify", response_model=VerificationResponse)
def verify(request: VerificationRequest) -> VerificationResponse:
    try:
        verification = verify_predictions_with_sql(request)

        if request.source_draw_number is not None:
            update_ledger_after_verification(
                mode=request.mode,
                source_draw_no=request.source_draw_number,
                target_draw_no=request.draw_number,
                hit_count=verification.hit_count,
            )

        return verification
    except VerificationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
