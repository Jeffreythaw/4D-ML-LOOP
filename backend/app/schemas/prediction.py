from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


DayType = Literal["Wednesday", "Saturday", "Sunday", "Special"]


class PredictionRequest(BaseModel):
    draw_number: int = Field(..., ge=1)
    day_type: DayType | None = None


class PredictionCandidate(BaseModel):
    rank: int = Field(..., ge=1, le=5)
    number: str = Field(..., min_length=4, max_length=4)
    score: float | None = None
    source: str | None = None

    @field_validator("number")
    @classmethod
    def validate_4d_number(cls, value: str) -> str:
        if not value.isdigit():
            raise ValueError("Prediction number must contain exactly four digits.")
        return value


class PredictionResponse(BaseModel):
    draw_number: int
    target_draw_number: int | None = None
    day_type: DayType | None = None
    predictions: list[PredictionCandidate]
    verification_status: str


class VerificationRequest(BaseModel):
    draw_number: int = Field(..., ge=1)
    day_type: DayType
    predictions: list[PredictionCandidate] = Field(..., min_length=1, max_length=5)


class VerificationResponse(BaseModel):
    draw_number: int
    day_type: DayType
    verification_status: str
    hit_count: int = Field(..., ge=0)
    details: dict[str, Any] = Field(default_factory=dict)
