from fastapi import APIRouter, HTTPException, status

from app.core.db import VerificationError, verify_predictions_with_sql
from app.core.ml_adapter import PredictionAdapterError, run_existing_engine_prediction
from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    VerificationRequest,
    VerificationResponse,
)


router = APIRouter(tags=["predictions"])


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        candidates = run_existing_engine_prediction(request)
    except PredictionAdapterError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return PredictionResponse(
        draw_number=request.draw_number,
        day_type=request.day_type,
        predictions=candidates[:5],
        verification_status="not_verified",
    )


@router.post("/verify", response_model=VerificationResponse)
def verify(request: VerificationRequest) -> VerificationResponse:
    try:
        return verify_predictions_with_sql(request)
    except VerificationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
